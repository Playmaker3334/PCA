import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.dependency_analysis import select_real_features, transition_entropy, detection_metrics
from src.monitor import SafetyMonitor, calibrate_threshold
from src.pc import HCLT
from src.sae import JumpReLUSAE
from src.utils import (
    METRICS,
    PC_BATCH_SIZE,
    PC_COLLAPSE_ENTROPY_THRESHOLD,
    PC_DIR,
    PC_LATENT_STATES,
    PC_LR,
    PC_NUM_EPOCHS,
    PC_REAL_FEATURE_PROBE_THRESHOLD,
    PC_TRANS_INIT_SCALE,
    SAE_FEATURES,
    SAFETY_ALPHA,
    SEED,
    Timer,
    get_logger,
    load_json,
    load_npz,
    log_exception,
    save_json,
    set_seed,
    setup_mlflow,
)

logger = get_logger("train_pc")


def load_features(layer):
    safe = load_npz(SAE_FEATURES / f"layer_{layer}_safe.npz")
    unsafe = load_npz(SAE_FEATURES / f"layer_{layer}_unsafe.npz")
    return safe["z_binary"], unsafe["z_binary"]


def split_train_val_indices(n, val_frac=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(val_frac * n)
    return idx[n_val:], idx[:n_val]


def main():
    import mlflow
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    decision = load_json(METRICS / "localization_decision.json")
    layer = decision["best_layer"]
    selected = decision["selected_feature_indices"]
    report = load_json(METRICS / f"localization_layer_{layer}.json")

    positions, feat_map_clean = select_real_features(
        selected, report["probe_top"], report["contrastive_top"],
        PC_REAL_FEATURE_PROBE_THRESHOLD,
    )
    logger.info(f"layer={layer} selected={len(selected)} "
                f"real_features(no padding)={len(feat_map_clean)}")

    z_safe, z_unsafe = load_features(layer)
    X_safe = z_safe[:, feat_map_clean].astype(np.float32)
    X_unsafe = z_unsafe[:, feat_map_clean].astype(np.float32)
    logger.info(f"X_safe={X_safe.shape} X_unsafe={X_unsafe.shape}")

    unsafe_split = load_json(METRICS / "unsafe_split.json")
    if unsafe_split["n_unsafe"] != len(X_unsafe):
        raise RuntimeError("unsafe_split.json out of sync with saved features; "
                           "rerun 02_localize_features")
    select_idx = np.array(unsafe_split["select_idx"], dtype=np.int64)
    eval_idx = np.array(unsafe_split["eval_idx"], dtype=np.int64)
    X_unsafe_select = X_unsafe[select_idx]
    X_unsafe_eval = X_unsafe[eval_idx]

    train_idx, val_idx = split_train_val_indices(len(X_safe), val_frac=0.1, seed=SEED)
    X_safe_train = X_safe[train_idx]
    X_safe_val = X_safe[val_idx]

    X_train_combined = np.vstack([X_safe_train, X_unsafe_select])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X_train_combined))
    n_inner_val = max(1, int(0.15 * len(X_train_combined)))
    inner_val = X_train_combined[perm[:n_inner_val]]
    inner_train = X_train_combined[perm[n_inner_val:]]
    logger.info(f"combined training: total={len(X_train_combined)} "
                f"(safe_train={len(X_safe_train)} + unsafe_select={len(X_unsafe_select)}) "
                f"inner_train={len(inner_train)} inner_val={len(inner_val)}")

    split_record = {
        "layer": int(layer),
        "seed": int(SEED),
        "val_frac": 0.1,
        "n_safe_total": int(len(X_safe)),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "feature_index_map": [int(x) for x in feat_map_clean],
        "padding_dropped": True,
        "n_features_after_padding_drop": int(len(feat_map_clean)),
        "training_distribution": "combined_safe_train_plus_unsafe_select",
    }
    save_json(split_record, METRICS / "pc_split.json")

    with setup_mlflow("03_train_pc"):
        mlflow.log_params({
            "layer": layer,
            "n_features": len(feat_map_clean),
            "K_latent_states": PC_LATENT_STATES,
            "epochs": PC_NUM_EPOCHS,
            "lr": PC_LR,
            "trans_init_scale": PC_TRANS_INIT_SCALE,
            "batch_size": PC_BATCH_SIZE,
            "training_distribution": "combined",
        })

        try:
            with Timer("HCLT structure", logger=logger):
                pc = HCLT.from_data(inner_train, num_states=PC_LATENT_STATES, root=0,
                                    device=device, trans_init_scale=PC_TRANS_INIT_SCALE)
            with Timer("HCLT training", logger=logger):
                history = pc.fit(
                    inner_train,
                    num_epochs=PC_NUM_EPOCHS,
                    batch_size=PC_BATCH_SIZE,
                    lr=PC_LR,
                    val_X=inner_val,
                    mlflow_log=True,
                )
        except Exception:
            log_exception(logger, "PC training failed")
            raise

        collapse = transition_entropy(pc, collapse_threshold=PC_COLLAPSE_ENTROPY_THRESHOLD)
        logger.info(f"transition entropy mean={collapse['mean']:.4f} "
                    f"near_uniform={collapse['n_near_uniform']}/{collapse['n_transitions']} "
                    f"collapsed={collapse['collapsed']}")
        if collapse["collapsed"]:
            logger.warning("PC COLLAPSED: transitions near-uniform, conditional dependencies "
                           "not modeled. Increase PC_TRANS_INIT_SCALE, lower PC_LATENT_STATES, "
                           "or raise PC_NUM_EPOCHS.")
        mlflow.log_metric("transition_entropy_mean", collapse["mean"])
        mlflow.log_metric("transition_n_near_uniform", collapse["n_near_uniform"])

        det = detection_metrics(pc, X_safe_val, X_unsafe_eval)
        logger.info(f"held-out detection AUROC={det['auroc']:.4f} "
                    f"nll safe={det['nll_safe_mean']:.2f} unsafe={det['nll_unsafe_mean']:.2f}")
        mlflow.log_metric("heldout_detection_auroc", det["auroc"])

        scores_safe = -pc.log_prob(X_safe_val).detach().cpu().numpy()
        scores_unsafe = -pc.log_prob(X_unsafe_select).detach().cpu().numpy()
        cal = calibrate_threshold(scores_safe, scores_unsafe)
        logger.info(f"calibration: tau={cal['threshold']:.3f} tpr={cal['tpr_at_threshold']:.3f}")
        mlflow.log_metric("threshold", cal["threshold"])

        pc.save(PC_DIR / f"hclt_layer_{layer}")

        refusal_sae = set(int(i) for i in decision.get("refusal_sae_indices", []))
        refusal_pc_features = [i for i, s in enumerate(feat_map_clean) if int(s) in refusal_sae]
        feature_descriptions = {int(k): v for k, v in decision.get("feature_descriptions", {}).items()}

        sae = JumpReLUSAE.from_gemma_scope(layer, device=device)
        monitor = SafetyMonitor(
            pc=pc,
            sae=sae,
            layer_idx=layer,
            refusal_features=refusal_pc_features,
            alpha=SAFETY_ALPHA,
            threshold=cal["threshold"],
            feature_index_map=[int(x) for x in feat_map_clean],
            feature_descriptions=feature_descriptions,
        )
        monitor.save(PC_DIR / "monitor")
        mlflow.log_artifact(str(PC_DIR / "monitor" / "config.json"))

        save_json({
            "layer": layer,
            "n_features": len(feat_map_clean),
            "feature_index_map": [int(x) for x in feat_map_clean],
            "hyperparameters": {
                "K_latent_states": PC_LATENT_STATES,
                "epochs": PC_NUM_EPOCHS,
                "lr": PC_LR,
                "trans_init_scale": PC_TRANS_INIT_SCALE,
                "batch_size": PC_BATCH_SIZE,
                "padding_dropped": True,
                "training_distribution": "combined_safe_plus_unsafe_select",
            },
            "history": history,
            "collapse_diagnostic": collapse,
            "heldout_detection": det,
            "calibration": cal,
            "n_refusal_features": len(refusal_pc_features),
            "rationale": (
                "Trained on the combined corpus because the dependency structure among "
                "safety concepts is only estimable from data that contains unsafe examples. "
                "Anti-collapse hyperparameters (strong transition init, smaller K, more epochs, "
                "padding dropped) prevent the circuit from degenerating to independent marginals."
            ),
        }, METRICS / "pc_training.json")
        mlflow.log_artifact(str(METRICS / "pc_training.json"))
        logger.info("training complete")


if __name__ == "__main__":
    main()
