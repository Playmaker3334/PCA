import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.localization import (
    attribution_score,
    contrastive_difference,
    intersect_rankings,
    linear_probe,
    neuronpedia_keyword_search,
    save_localization_report,
)
from src.sae import JumpReLUSAE, topk_binarize
from src.dataset_fields import SAFE_CORPORA, UNSAFE_CORPORA
from src.utils import (
    ACTIVATIONS,
    CANDIDATE_LAYERS,
    METRICS,
    SAE_FEATURES,
    TOP_K_FEATURES,
    Timer,
    get_logger,
    log_exception,
    load_npz,
    save_json,
    save_npz,
    set_seed,
    setup_mlflow,
)

logger = get_logger("localize")

SAFE = [c for c in SAFE_CORPORA if c != "xstest"]
UNSAFE = list(UNSAFE_CORPORA)


def load_activations_for_layer(layer):
    safe_acts, unsafe_acts = [], []
    safe_prompts, unsafe_prompts = [], []
    for name in SAFE + UNSAFE:
        path = ACTIVATIONS / f"{name}.npz"
        if not path.exists():
            logger.warning(f"missing {path}")
            continue
        try:
            d = load_npz(path)
            key = f"layer_{layer}"
            if key not in d.files:
                continue
            acts = d[key]
            prompts = d["prompts"]
            if name in SAFE:
                safe_acts.append(acts)
                safe_prompts.extend(list(prompts))
            else:
                unsafe_acts.append(acts)
                unsafe_prompts.extend(list(prompts))
        except Exception:
            log_exception(logger, f"failed loading {name}")
    safe = np.vstack(safe_acts) if safe_acts else np.zeros((0, 1))
    unsafe = np.vstack(unsafe_acts) if unsafe_acts else np.zeros((0, 1))
    return safe, unsafe, safe_prompts, unsafe_prompts


def encode_batched(sae, activations, batch_size=64):
    if activations.shape[0] == 0:
        return np.zeros((0, sae.n_features), dtype=np.uint8), np.zeros((0, sae.n_features))
    z_bin_all, z_dense_all = [], []
    for i in range(0, len(activations), batch_size):
        batch = activations[i:i+batch_size]
        z = sae.encode_np(batch)
        z_dense_all.append(z)
        z_bin_all.append(topk_binarize(z, TOP_K_FEATURES))
    return np.vstack(z_bin_all), np.vstack(z_dense_all)


def localize_layer(layer):
    import torch
    logger.info(f"=== localizing layer {layer} ===")
    safe_acts, unsafe_acts, safe_p, unsafe_p = load_activations_for_layer(layer)
    if len(safe_acts) == 0 or len(unsafe_acts) == 0:
        logger.warning(f"insufficient data for layer {layer}")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = JumpReLUSAE.from_gemma_scope(layer, device=device)
    err_safe = sae.reconstruction_error(safe_acts[:200])
    err_unsafe = sae.reconstruction_error(unsafe_acts[:200])
    logger.info(f"reconstruction error: safe={err_safe:.3f} unsafe={err_unsafe:.3f}")

    with Timer(f"SAE encode layer {layer}", logger=logger):
        z_safe_bin, z_safe_dense = encode_batched(sae, safe_acts)
        z_unsafe_bin, z_unsafe_dense = encode_batched(sae, unsafe_acts)

    save_npz(SAE_FEATURES / f"layer_{layer}_safe.npz",
             z_binary=z_safe_bin, z_dense=z_safe_dense,
             prompts=np.array(safe_p, dtype=object))
    save_npz(SAE_FEATURES / f"layer_{layer}_unsafe.npz",
             z_binary=z_unsafe_bin, z_dense=z_unsafe_dense,
             prompts=np.array(unsafe_p, dtype=object))

    logger.info("running linear probe")
    probe = linear_probe(z_unsafe_bin.astype(np.float32), z_safe_bin.astype(np.float32))
    logger.info(f"probe AUROC={probe['auroc']:.4f} nonzero={probe['nonzero_coefs']}")

    logger.info("running contrastive difference")
    diff = contrastive_difference(z_unsafe_bin.astype(np.float32), z_safe_bin.astype(np.float32))

    logger.info("running attribution")
    labels = np.concatenate([np.ones(len(z_unsafe_bin)), np.zeros(len(z_safe_bin))])
    X_all = np.vstack([z_unsafe_bin, z_safe_bin]).astype(np.float32)
    attribution = attribution_score(X_all, labels)
    top_attribution = [(int(i), float(attribution[i]))
                       for i in np.argsort(-np.abs(attribution))[:50]]

    intersect = intersect_rankings([
        probe["top_features"][:50],
        diff["top_features"][:50],
        top_attribution[:50],
    ], top_k=128)

    logger.info("querying neuronpedia")
    np_results = neuronpedia_keyword_search(layer)

    report = {
        "layer": layer,
        "reconstruction_error_safe": err_safe,
        "reconstruction_error_unsafe": err_unsafe,
        "probe_auroc": probe["auroc"],
        "probe_nonzero": probe["nonzero_coefs"],
        "probe_top": probe["top_features"][:50],
        "contrastive_top": diff["top_features"][:50],
        "attribution_top": top_attribution,
        "intersected_top_features": intersect,
        "neuronpedia": np_results,
        "n_safe": int(len(z_safe_bin)),
        "n_unsafe": int(len(z_unsafe_bin)),
    }
    save_localization_report(report, layer)
    del sae
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()
    return report


def main():
    import mlflow
    set_seed()
    with setup_mlflow("02_localize_features"):
        reports = {}
        for layer in CANDIDATE_LAYERS:
            try:
                r = localize_layer(layer)
                if r is not None:
                    reports[layer] = r
                    mlflow.log_metric(f"probe_auroc_L{layer}", r["probe_auroc"])
                    mlflow.log_metric(f"nonzero_L{layer}", r["probe_nonzero"])
            except Exception:
                log_exception(logger, f"layer {layer} failed")

        if not reports:
            logger.error("no layers succeeded")
            return

        best_layer = max(reports.keys(), key=lambda L: reports[L]["probe_auroc"])
        best = reports[best_layer]
        selected = best["intersected_top_features"]
        decision = {
            "best_layer": int(best_layer),
            "best_auroc": float(best["probe_auroc"]),
            "selected_feature_indices": list(map(int, selected)),
            "summary_per_layer": {L: {"auroc": float(reports[L]["probe_auroc"]),
                                       "nonzero": int(reports[L]["probe_nonzero"]),
                                       "n_safe": int(reports[L]["n_safe"]),
                                       "n_unsafe": int(reports[L]["n_unsafe"])}
                                  for L in reports},
        }
        save_json(decision, METRICS / "localization_decision.json")
        mlflow.log_param("best_layer", best_layer)
        mlflow.log_metric("best_auroc", best["probe_auroc"])
        mlflow.log_artifact(str(METRICS / "localization_decision.json"))
        logger.info(f"selected layer {best_layer} AUROC={best['probe_auroc']:.4f} "
                    f"with {len(selected)} features")


if __name__ == "__main__":
    main()