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


def load_split_features():
    split = load_json(METRICS / "pc_split.json")
    unsafe_split = load_json(METRICS / "unsafe_split.json")
    layer = split["layer"]
    feat_map = np.array(split["feature_index_map"], dtype=np.int64)
    train_idx = np.array(split["train_idx"], dtype=np.int64)
    val_idx = np.array(split["val_idx"], dtype=np.int64)
    select_idx = np.array(unsafe_split["select_idx"], dtype=np.int64)
    eval_idx = np.array(unsafe_split["eval_idx"], dtype=np.int64)

    safe = load_npz(SAE_FEATURES / f"layer_{layer}_safe.npz")["z_binary"]
    unsafe = load_npz(SAE_FEATURES / f"layer_{layer}_unsafe.npz")["z_binary"]

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

    results = {"layer": int(layer)}

    with setup_mlflow("07_density_baselines"):
        mlflow.log_param("layer", layer)

        baselines = {
            "ocsvm": run_ocsvm,
            "gmm": run_gmm,
            "mahalanobis": run_mahalanobis,
            "logistic_probe_supervised": run_logistic_probe,
        }
        for name, fn in baselines.items():
            try:
                auroc = fn(train, val_safe, unsafe_eval, unsafe_select)
                results[name] = {"auroc": auroc}
                mlflow.log_metric(f"{name}_auroc", auroc)
                logger.info(f"{name}: AUROC={auroc:.4f}")
            except Exception:
                log_exception(logger, f"{name} failed")
                results[name] = {"auroc": None}

        try:
            ev = load_json(METRICS / "evaluation_summary.json")
            results["pc_sae"] = {"auroc": ev.get("density_auroc")}
        except Exception:
            results["pc_sae"] = {"auroc": None}

        save_json(results, METRICS / "density_baselines.json")
        mlflow.log_artifact(str(METRICS / "density_baselines.json"))
        logger.info(f"density baselines complete: {results}")


if __name__ == "__main__":
    main()