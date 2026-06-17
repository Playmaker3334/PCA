import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.dependency_analysis import (
    build_dependency_tree,
    build_semantic_analysis,
    derive_safety_set,
    detection_metrics,
    fetch_descriptions,
    generate_figures,
    overfit_validation,
    select_real_features,
    transition_entropy,
)
from src.pc import HCLT
from src.utils import (
    FIGURES,
    METRICS,
    PC_COLLAPSE_ENTROPY_THRESHOLD,
    PC_DIR,
    PC_REAL_FEATURE_PROBE_THRESHOLD,
    SAE_FEATURES,
    get_logger,
    load_json,
    load_npz,
    log_exception,
    save_json,
    set_seed,
)

logger = get_logger("dependency_analysis_script")

SAFETY_BRANCH_EDGES = [
    (8947, 10656, "ethical_violations -> medical_warnings"),
    (10656, 8051, "medical_warnings -> social_exclusion"),
    (8051, 1347, "social_exclusion -> violence"),
    (8051, 15270, "social_exclusion -> legal_judicial"),
    (8051, 839, "social_exclusion -> damage"),
    (8947, 12916, "ethical_violations -> investigations"),
    (12916, 11202, "investigations -> health"),
    (11202, 9803, "health -> legal_matters"),
]


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    config = load_json(PC_DIR / "monitor" / "config.json")
    layer = config["layer_idx"]
    feat_map_clean = [int(x) for x in config["feature_index_map"]]

    decision = load_json(METRICS / "localization_decision.json")
    report = load_json(METRICS / f"localization_layer_{layer}.json")

    pc = HCLT.load(PC_DIR / "monitor" / "pc", device=device)
    logger.info(f"loaded monitor PC: {pc.num_vars} vars, K={pc.K}, layer={layer}")

    logger.info("fetching neuronpedia descriptions for all real features")
    descriptions = fetch_descriptions(layer, indices=feat_map_clean)
    if not descriptions:
        descriptions = {int(k): v for k, v in config.get("feature_descriptions", {}).items()}
        logger.warning("falling back to descriptions stored in monitor config")

    safety_set = derive_safety_set(feat_map_clean, descriptions,
                                   refusal_sae_indices=decision.get("refusal_sae_indices", []))
    logger.info(f"safety features identified: {len(safety_set)} of {len(feat_map_clean)}")

    z_safe = load_npz(SAE_FEATURES / f"layer_{layer}_safe.npz")["z_binary"]
    z_unsafe = load_npz(SAE_FEATURES / f"layer_{layer}_unsafe.npz")["z_binary"]
    X_safe = z_safe[:, feat_map_clean].astype(np.float32)
    X_unsafe = z_unsafe[:, feat_map_clean].astype(np.float32)
    X = np.vstack([X_safe, X_unsafe])
    n_safe = len(X_safe)
    logger.info(f"feature matrix for empirical lifts: X={X.shape} (safe={n_safe})")

    collapse = transition_entropy(pc, collapse_threshold=PC_COLLAPSE_ENTROPY_THRESHOLD)
    save_json(collapse, METRICS / "collapse_diagnostic.json")
    logger.info(f"collapse diagnostic: mean_entropy={collapse['mean']:.4f} "
                f"collapsed={collapse['collapsed']}")

    det = detection_metrics(pc, X_safe, X_unsafe)
    save_json(det, METRICS / "detection_metrics.json")
    logger.info(f"in-corpus detection AUROC={det['auroc']:.4f}")

    logger.info("building dependency tree with model and empirical lifts")
    tree = build_dependency_tree(pc, feat_map_clean, descriptions, safety_set, X=X)
    save_json(tree, METRICS / "dependency_structure.json")
    logger.info(f"tree: root=sae#{tree['root_sae_index']} nodes={tree['n_nodes']} "
                f"edges={tree['n_edges']} safety_to_safety_edges={tree['n_safety_to_safety_edges']} "
                f"levels={tree['levels']}")

    pc_idx = {int(feat_map_clean[i]): i for i in range(len(feat_map_clean))}
    safety_branch = []
    for parent_sae, child_sae, label in SAFETY_BRANCH_EDGES:
        if parent_sae in pc_idx and child_sae in pc_idx:
            edge = next((e for e in tree["edges"]
                         if e["parent_sae_index"] == parent_sae
                         and e["child_sae_index"] == child_sae), None)
            entry = {
                "edge": label,
                "parent_sae_index": parent_sae,
                "child_sae_index": child_sae,
                "in_tree": edge is not None,
            }
            if edge is not None:
                entry["model_lift"] = edge["model_lift"]
                entry["empirical_lift"] = edge.get("empirical_lift")
            safety_branch.append(entry)
    save_json({
        "description": "key safety-branch edges, model lift vs empirical lift; "
                       "model conservative vs empirical indicates regularization not memorization",
        "edges": safety_branch,
    }, METRICS / "dependency_lifts.json")

    logger.info("running held-out overfit validation (fresh circuit, clean split)")
    try:
        overfit = overfit_validation(X, n_safe, feat_map_clean, SAFETY_BRANCH_EDGES, device)
        save_json(overfit, METRICS / "overfit_validation.json")
        logger.info(f"overfit check: heldout_auroc={overfit['detection_auroc_heldout_test']:.4f} "
                    f"verdict={overfit['verdict'][:40]}")
    except Exception:
        log_exception(logger, "overfit validation failed")

    logger.info("building semantic analysis")
    semantic = build_semantic_analysis(
        feat_map_clean, descriptions, report["probe_top"], report["contrastive_top"],
        safety_set, tree,
    )
    save_json(semantic, METRICS / "semantic_analysis.json")

    logger.info("generating figures")
    try:
        figs = generate_figures(tree, descriptions, safety_set, FIGURES)
        logger.info(f"figures saved: {figs}")
    except Exception:
        log_exception(logger, "figure generation failed")

    save_json({
        "layer": int(layer),
        "n_features": int(len(feat_map_clean)),
        "n_safety_features": int(len(safety_set & set(feat_map_clean))),
        "root_concept": tree["root_description"],
        "root_sae_index": tree["root_sae_index"],
        "collapsed": collapse["collapsed"],
        "transition_entropy_mean": collapse["mean"],
        "in_corpus_detection_auroc": det["auroc"],
        "n_safety_to_safety_edges": tree["n_safety_to_safety_edges"],
        "outputs": {
            "dependency_structure": "metrics/dependency_structure.json",
            "dependency_lifts": "metrics/dependency_lifts.json",
            "collapse_diagnostic": "metrics/collapse_diagnostic.json",
            "detection_metrics": "metrics/detection_metrics.json",
            "overfit_validation": "metrics/overfit_validation.json",
            "semantic_analysis": "metrics/semantic_analysis.json",
            "figures": ["figures/concept_hierarchy.png", "figures/concept_hierarchy_pruned.png"],
        },
    }, METRICS / "dependency_analysis_summary.json")
    logger.info("dependency analysis complete")


if __name__ == "__main__":
    main()
