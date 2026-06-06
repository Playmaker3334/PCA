import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.monitor import SafetyMonitor, calibrate_threshold
from src.pc import HCLT
from src.sae import JumpReLUSAE
from src.utils import (
    METRICS,
    PC_BATCH_SIZE,
    PC_DIR,
    PC_LATENT_STATES,
    PC_LR,
    PC_NUM_EPOCHS,
    PC_NUM_TOP_FEATURES,
    SAE_FEATURES,
    SAFETY_ALPHA,
    SEED,
    Timer,
    get_logger,
    log_exception,
    load_json,
    load_npz,
    save_json,
    set_seed,
    setup_mlflow,
)

logger = get_logger("train_pc")


def load_features(layer):
    safe = load_npz(SAE_FEATURES / f"layer_{layer}_safe.npz")
    unsafe = load_npz(SAE_FEATURES / f"layer_{layer}_unsafe.npz")
    return safe["z_binary"], unsafe["z_binary"]


def restrict_features(z, feature_indices, max_features=PC_NUM_TOP_FEATURES):
    selected = np.array(feature_indices[:max_features], dtype=np.int64)
    return z[:, selected].astype(np.float32), selected


def split_train_val_indices(n, val_frac=0.1, seed=SEED):
    """Devuelve indices (no arrays) para que el split sea reconstruible
    de forma identica en 04_evaluate.py y se evite leakage en density AUROC.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(val_frac * n)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def main():
    import mlflow
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    decision = load_json(METRICS / "localization_decision.json")
    layer = decision["best_layer"]
    feature_indices = decision["selected_feature_indices"]
    logger.info(f"layer={layer} n_features={len(feature_indices)}")

    z_safe, z_unsafe = load_features(layer)
    X_safe_pc, feat_map = restrict_features(z_safe, feature_indices)
    X_unsafe_pc, _ = restrict_features(z_unsafe, feature_indices)
    logger.info(f"X_safe={X_safe_pc.shape} X_unsafe={X_unsafe_pc.shape}")
    logger.info(f"density safe={X_safe_pc.mean():.4f} unsafe={X_unsafe_pc.mean():.4f}")

    # split por indices, persistido a disco para reuso sin leakage
    train_idx, val_idx = split_train_val_indices(len(X_safe_pc), val_frac=0.1, seed=SEED)
    train_X = X_safe_pc[train_idx]
    val_X = X_safe_pc[val_idx]
    logger.info(f"train={train_X.shape} val={val_X.shape}")

    split_record = {
        "layer": int(layer),
        "seed": int(SEED),
        "val_frac": 0.1,
        "n_safe_total": int(len(X_safe_pc)),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "feature_index_map": feat_map.tolist(),
    }
    save_json(split_record, METRICS / "pc_split.json")
    logger.info(f"saved split indices to {METRICS / 'pc_split.json'}")

    with setup_mlflow("03_train_pc"):
        mlflow.log_params({
            "layer": layer,
            "n_features": len(feature_indices),
            "K_latent_states": PC_LATENT_STATES,
            "epochs": PC_NUM_EPOCHS,
            "lr": PC_LR,
            "batch_size": PC_BATCH_SIZE,
            "n_train": len(train_X),
            "n_val": len(val_X),
            "n_unsafe": len(X_unsafe_pc),
        })

        try:
            with Timer("HCLT structure", logger=logger):
                pc = HCLT.from_data(train_X, num_states=PC_LATENT_STATES, root=0, device=device)
            with Timer("HCLT training", logger=logger):
                history = pc.fit(
                    train_X,
                    num_epochs=PC_NUM_EPOCHS,
                    batch_size=PC_BATCH_SIZE,
                    lr=PC_LR,
                    val_X=val_X,
                    mlflow_log=True,
                )
        except Exception:
            log_exception(logger, "PC training failed")
            raise

        train_nll = float(-pc.log_prob(train_X).detach().cpu().mean())
        val_nll = float(-pc.log_prob(val_X).detach().cpu().mean())
        unsafe_nll = float(-pc.log_prob(X_unsafe_pc).detach().cpu().mean())
        separation = unsafe_nll - val_nll
        logger.info(f"NLL: train={train_nll:.3f} val={val_nll:.3f} "
                    f"unsafe={unsafe_nll:.3f} sep={separation:.3f}")
        mlflow.log_metrics({
            "final_train_nll": train_nll,
            "final_val_nll": val_nll,
            "final_unsafe_nll": unsafe_nll,
            "separation": separation,
        })

        # calibracion sobre VAL (no visto en entrenamiento) vs unsafe
        scores_safe = -pc.log_prob(val_X).detach().cpu().numpy()
        scores_unsafe = -pc.log_prob(X_unsafe_pc).detach().cpu().numpy()
        cal = calibrate_threshold(scores_safe, scores_unsafe)
        logger.info(f"calibration: AUROC={cal['auroc']:.4f} tau={cal['threshold']:.3f} "
                    f"tpr={cal['tpr_at_threshold']:.3f}")
        mlflow.log_metrics({
            "calibration_auroc": cal["auroc"],
            "threshold": cal["threshold"],
            "tpr_at_threshold": cal["tpr_at_threshold"],
        })

        pc.save(PC_DIR / f"hclt_layer_{layer}")

        sae = JumpReLUSAE.from_gemma_scope(layer, device=device)
        refusal_pc_features = list(range(min(20, len(feature_indices))))
        monitor = SafetyMonitor(
            pc=pc,
            sae=sae,
            layer_idx=layer,
            refusal_features=refusal_pc_features,
            alpha=SAFETY_ALPHA,  # inerte: el score es NLL puro (B2)
            threshold=cal["threshold"],
            feature_index_map=feat_map.tolist(),
            feature_descriptions={},
        )
        monitor.save(PC_DIR / "monitor")
        mlflow.log_artifact(str(PC_DIR / "monitor" / "config.json"))

        save_json({
            "layer": layer,
            "n_features": len(feature_indices),
            "history": history,
            "train_nll": train_nll,
            "val_nll": val_nll,
            "unsafe_nll": unsafe_nll,
            "separation": separation,
            "calibration": cal,
            "score_definition": "nll_only",  # documenta la decision B2
        }, METRICS / "pc_training.json")
        mlflow.log_artifact(str(METRICS / "pc_training.json"))
        logger.info("training complete")


if __name__ == "__main__":
    main()