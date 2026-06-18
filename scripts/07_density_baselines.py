import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils import (
    METRICS,
    SAE_FEATURES,
    get_logger,
    load_json,
    load_npz,
    log_exception,
    save_json,
    set_seed,
    setup_mlflow,
)

logger = get_logger("density_baselines")

TRAIN_SET = {
    "factorized_bernoulli": "combined_safe_train_plus_unsafe_select",
    "ocsvm": "safe_only",
    "gmm": "safe_only",
    "mahalanobis": "safe_only",
    "logistic_probe_supervised": "combined_safe_train_plus_unsafe_select",
}


def load_split_features():
    pc_split = load_json(METRICS / "pc_split.json")
    safe_split = load_json(METRICS / "safe_split.json")
    unsafe_split = load_json(METRICS / "unsafe_split.json")
    layer = pc_split["layer"]
    feat_map = np.array(pc_split["feature_index_map"], dtype=np.int64)

    safe = load_npz(SAE_FEATURES / f"layer_{layer}_safe.npz")["z_binary"]
    unsafe = load_npz(SAE_FEATURES / f"layer_{layer}_unsafe.npz")["z_binary"]

    if safe_split["n_safe"] != len(safe):
        raise RuntimeError("safe_split.json out of sync with saved safe features; "
                           "rerun 02_localize_features and 03_train_pc")
    if unsafe_split["n_unsafe"] != len(unsafe):
        raise RuntimeError("unsafe_split.json out of sync with saved unsafe features; "
                           "rerun 02_localize_features")

    train_idx = np.array(safe_split["train_idx"], dtype=np.int64)
    val_idx = np.array(safe_split["val_idx"], dtype=np.int64)
    select_idx = np.array(unsafe_split["select_idx"], dtype=np.int64)
    eval_idx = np.array(unsafe_split["eval_idx"], dtype=np.int64)

    X_safe = safe[:, feat_map].astype(np.float32)
    X_unsafe = unsafe[:, feat_map].astype(np.float32)

    return {
        "layer": layer,
        "train": X_safe[train_idx],
        "val_safe": X_safe[val_idx],
        "unsafe_select": X_unsafe[select_idx],
        "unsafe_eval": X_unsafe[eval_idx],
    }


def auroc_from_scores(score_safe, score_unsafe):
    labels = np.concatenate([np.zeros(len(score_safe)), np.ones(len(score_unsafe))])
    scores = np.concatenate([score_safe, score_unsafe])
    return float(roc_auc_score(labels, scores))


def run_factorized_bernoulli(train, val_safe, unsafe_eval, unsafe_select):
    X_train = np.vstack([train, unsafe_select]).astype(np.float64)
    n = len(X_train)
    p = (X_train.sum(axis=0) + 1.0) / (n + 2.0)
    log_p1 = np.log(p)
    log_p0 = np.log1p(-p)

    def nll(X):
        Xf = X.astype(np.float64)
        return -(Xf @ log_p1 + (1.0 - Xf) @ log_p0)

    return auroc_from_scores(nll(val_safe), nll(unsafe_eval))


def run_ocsvm(train, val_safe, unsafe_eval, unsafe_select):
    scaler = StandardScaler().fit(train)
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    clf.fit(scaler.transform(train))
    s_safe = -clf.decision_function(scaler.transform(val_safe))
    s_unsafe = -clf.decision_function(scaler.transform(unsafe_eval))
    return auroc_from_scores(s_safe, s_unsafe)


def run_gmm(train, val_safe, unsafe_eval, unsafe_select, n_components=8):
    scaler = StandardScaler().fit(train)
    gmm = GaussianMixture(n_components=n_components, covariance_type="diag",
                          reg_covar=1e-4, random_state=42)
    gmm.fit(scaler.transform(train))
    s_safe = -gmm.score_samples(scaler.transform(val_safe))
    s_unsafe = -gmm.score_samples(scaler.transform(unsafe_eval))
    return auroc_from_scores(s_safe, s_unsafe)


def run_mahalanobis(train, val_safe, unsafe_eval, unsafe_select):
    mu = train.mean(axis=0)
    cov = np.cov(train, rowvar=False) + 1e-4 * np.eye(train.shape[1])
    inv = np.linalg.pinv(cov)

    def dist(X):
        d = X - mu
        return np.einsum("ij,jk,ik->i", d, inv, d)

    return auroc_from_scores(dist(val_safe), dist(unsafe_eval))


def run_logistic_probe(train, val_safe, unsafe_eval, unsafe_select):
    X_tr = np.vstack([train, unsafe_select])
    y_tr = np.concatenate([np.zeros(len(train)), np.ones(len(unsafe_select))])
    scaler = StandardScaler().fit(X_tr)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(scaler.transform(X_tr), y_tr)
    s_safe = clf.predict_proba(scaler.transform(val_safe))[:, 1]
    s_unsafe = clf.predict_proba(scaler.transform(unsafe_eval))[:, 1]
    return auroc_from_scores(s_safe, s_unsafe)


def main():
    import mlflow
    set_seed()

    data = load_split_features()
    layer = data["layer"]
    train = data["train"]
    val_safe = data["val_safe"]
    unsafe_eval = data["unsafe_eval"]
    unsafe_select = data["unsafe_select"]
    logger.info(f"layer={layer} train={train.shape} val_safe={val_safe.shape} "
                f"unsafe_eval={unsafe_eval.shape} unsafe_select={unsafe_select.shape}")

    results = {
        "layer": int(layer),
        "_methodology": (
            "All methods scored on the same held-out split (S_val vs U_eval) over the same "
            "selected feature set. factorized_bernoulli is the no-tree ablation of the HCLT: "
            "product of independent Bernoulli marginals (Laplace-smoothed MLE), NLL score, "
            "trained on the same combined corpus (safe_train + unsafe_select) as the HCLT. "
            "pc_sae is the full HCLT (from evaluation_summary). ocsvm/gmm/mahalanobis train on "
            "safe only; logistic_probe_supervised is the supervised ceiling (sees unsafe), not "
            "anomaly detection. HCLT minus factorized_bernoulli isolates the detection value of "
            "the dependency structure."
        ),
    }

    with setup_mlflow("07_density_baselines"):
        mlflow.log_param("layer", layer)

        baselines = {
            "factorized_bernoulli": run_factorized_bernoulli,
            "ocsvm": run_ocsvm,
            "gmm": run_gmm,
            "mahalanobis": run_mahalanobis,
            "logistic_probe_supervised": run_logistic_probe,
        }
        for name, fn in baselines.items():
            try:
                auroc = fn(train, val_safe, unsafe_eval, unsafe_select)
                results[name] = {"auroc": auroc, "train_set": TRAIN_SET[name]}
                mlflow.log_metric(f"{name}_auroc", auroc)
                logger.info(f"{name} ({TRAIN_SET[name]}): AUROC={auroc:.4f}")
            except Exception:
                log_exception(logger, f"{name} failed")
                results[name] = {"auroc": None, "train_set": TRAIN_SET[name]}

        try:
            ev = load_json(METRICS / "evaluation_summary.json")
            results["pc_sae"] = {"auroc": ev.get("density_auroc"),
                                 "train_set": "combined_safe_train_plus_unsafe_select",
                                 "model": "HCLT_full_tree"}
        except Exception:
            results["pc_sae"] = {"auroc": None}

        pc_auroc = results.get("pc_sae", {}).get("auroc")
        fac_auroc = results.get("factorized_bernoulli", {}).get("auroc")
        if pc_auroc is not None and fac_auroc is not None:
            delta = float(pc_auroc - fac_auroc)
            results["tree_contribution"] = {
                "pc_sae_auroc": float(pc_auroc),
                "factorized_auroc": float(fac_auroc),
                "delta_auroc": delta,
                "interpretation": (
                    "dependency structure adds detection value over independent marginals"
                    if delta > 0.01 else
                    "tree adds negligible detection value over independent marginals; the "
                    "circuit's contribution is interpretability (named dependency structure), "
                    "not detection performance"
                ),
            }
            mlflow.log_metric("tree_contribution_delta_auroc", delta)
            logger.info(f"TREE CONTRIBUTION: HCLT={pc_auroc:.4f} "
                        f"factorized={fac_auroc:.4f} delta={delta:+.4f}")
        else:
            logger.warning("tree contribution delta unavailable "
                           "(pc_sae or factorized missing; rerun 04 then 07)")

        save_json(results, METRICS / "density_baselines.json")
        mlflow.log_artifact(str(METRICS / "density_baselines.json"))
        logger.info(f"density baselines complete: {results}")


if __name__ == "__main__":
    main()