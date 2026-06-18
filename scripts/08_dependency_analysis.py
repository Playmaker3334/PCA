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
from src.pc import HCLT, pairwise_mutual_information
from src.utils import (
    FIGURES,
    METRICS,
    PC_COLLAPSE_ENTROPY_THRESHOLD,
    PC_DIR,
    PC_REAL_FEATURE_PROBE_THRESHOLD,
    SAE_FEATURES,
    SEED,
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

SAFETY_SET_MANUAL_OVERRIDE = []
N_PERMUTATIONS = 10000


def _tree_nodes_edges(tree):
    nodes = [int(n["sae_index"]) for n in tree["nodes"]]
    edges = [(int(e["parent_sae_index"]), int(e["child_sae_index"])) for e in tree["edges"]]
    return nodes, edges


def _rank_of(metric, target):
    ordered = sorted(metric.items(),
                     key=lambda kv: (-(kv[1] if kv[1] is not None else float("-inf")), kv[0]))
    for r, (idx, v) in enumerate(ordered, 1):
        if int(idx) == int(target):
            return r, (float(v) if v is not None else None)
    return None, None


def _top(metric, descriptions, safety_set, k=8):
    ordered = sorted(metric.items(),
                     key=lambda kv: (-(kv[1] if kv[1] is not None else float("-inf")), kv[0]))
    return [{"sae_index": int(idx),
             "value": (float(v) if v is not None else None),
             "is_safety": bool(int(idx) in safety_set),
             "description": descriptions.get(int(idx), "")}
            for idx, v in ordered[:k]]


def compute_centrality(tree, X, feat_map_clean, descriptions, safety_set):
    nodes, edges = _tree_nodes_edges(tree)
    pos = {n: i for i, n in enumerate(nodes)}
    pc_pos = {int(feat_map_clean[i]): i for i in range(len(feat_map_clean))}

    degree = {n: 0 for n in nodes}
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    degree = {int(n): int(d) for n, d in degree.items()}

    betweenness = {n: None for n in nodes}
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        bc = nx.betweenness_centrality(G)
        betweenness = {int(n): float(bc[n]) for n in nodes}
    except Exception:
        log_exception(logger, "betweenness unavailable")

    summed_mi = {n: None for n in nodes}
    try:
        M = pairwise_mutual_information(X)
        acc = {n: 0.0 for n in nodes}
        for u, v in edges:
            w = float(M[pc_pos[u], pc_pos[v]])
            acc[u] += w
            acc[v] += w
        summed_mi = {int(n): float(acc[n]) for n in nodes}
    except Exception:
        log_exception(logger, "summed MI unavailable")

    root = int(tree["root_sae_index"])
    named = sorted({a for a, _, _ in SAFETY_BRANCH_EDGES} | {b for _, b, _ in SAFETY_BRANCH_EDGES})
    named_centrality = []
    for idx in named:
        if idx in degree:
            rd, _ = _rank_of(degree, idx)
            rb, _ = _rank_of(betweenness, idx)
            rm, _ = _rank_of(summed_mi, idx)
            named_centrality.append({
                "sae_index": int(idx),
                "description": descriptions.get(int(idx), ""),
                "is_safety": bool(int(idx) in safety_set),
                "degree": degree.get(idx), "rank_by_degree": rd,
                "betweenness": betweenness.get(idx), "rank_by_betweenness": rb,
                "summed_mi": summed_mi.get(idx), "rank_by_summed_mi": rm,
            })

    rd_root, dv = _rank_of(degree, root)
    rb_root, bv = _rank_of(betweenness, root)
    rm_root, mv = _rank_of(summed_mi, root)

    return {
        "n_nodes": len(nodes),
        "root_sae_index": root,
        "root_is_fixed_by_convention": True,
        "note": ("Chow-Liu tree is undirected; the root and all edge directions are a rooting "
                 "convention (root=0=feat_map_clean[0], the top fused localization feature), NOT "
                 "a learned hierarchy. Degree and betweenness are rooting-invariant. summed_mi is "
                 "empirical MI over the full corpus (safe+unsafe) on incident tree edges."),
        "root_centrality": {
            "degree": dv, "rank_by_degree": rd_root,
            "betweenness": bv, "rank_by_betweenness": rb_root,
            "summed_mi": mv, "rank_by_summed_mi": rm_root,
        },
        "top_by_degree": _top(degree, descriptions, safety_set),
        "top_by_betweenness": _top(betweenness, descriptions, safety_set),
        "top_by_summed_mi": _top(summed_mi, descriptions, safety_set),
        "named_concept_centrality": named_centrality,
        "degree": degree,
        "betweenness": betweenness,
        "summed_mi": summed_mi,
    }


def safety_clustering_permutation(tree, safety_indices, n_permutations, seed):
    nodes, edges = _tree_nodes_edges(tree)
    pos = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    node_set = set(nodes)
    safety = set(int(i) for i in safety_indices) & node_set
    s = len(safety)
    observed = int(sum(1 for u, v in edges if u in safety and v in safety))
    expected = (s * (s - 1) / N) if N > 0 else 0.0

    if N == 0 or s == 0 or s == N or len(edges) == 0:
        return {
            "n_nodes": N, "n_safety": s, "n_edges": len(edges),
            "observed_safety_to_safety_edges": observed,
            "expected_analytic": float(expected),
            "p_value": None, "z_score": None, "significant_p05": None,
            "note": "degenerate (no safety features, all safety, or no edges)",
        }

    edge_idx = np.array([[pos[u], pos[v]] for u, v in edges], dtype=np.int64)
    rng = np.random.default_rng(seed)
    counts = np.empty(n_permutations, dtype=np.int64)
    for b in range(n_permutations):
        mask = np.zeros(N, dtype=bool)
        mask[rng.choice(N, size=s, replace=False)] = True
        both = mask[edge_idx[:, 0]] & mask[edge_idx[:, 1]]
        counts[b] = int(both.sum())

    perm_mean = float(counts.mean())
    perm_std = float(counts.std())
    p_value = float((1 + int(np.sum(counts >= observed))) / (n_permutations + 1))
    z = float((observed - perm_mean) / perm_std) if perm_std > 1e-9 else None
    return {
        "n_nodes": N, "n_safety": s, "n_edges": len(edges),
        "observed_safety_to_safety_edges": observed,
        "expected_analytic": float(expected),
        "permutation_mean": perm_mean,
        "permutation_std": perm_std,
        "z_score": z,
        "p_value": p_value,
        "n_permutations": int(n_permutations),
        "significant_p05": bool(p_value < 0.05),
        "interpretation": (
            "safety concepts cluster in the tree more than expected for s randomly placed "
            "nodes (one-sided permutation test); the connected safety subtree is not an "
            "artifact of node labeling"
            if p_value < 0.05 else
            "cannot reject random placement; the observed safety-to-safety edge count is "
            "consistent with chance given the tree topology and s"
        ),
    }


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
    logger.info(f"safety features identified (substring): {len(safety_set)} of {len(feat_map_clean)}")

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
    logger.info(f"in-corpus detection AUROC={det['auroc']:.4f} (DIAGNOSTIC ONLY, includes "
                f"training data; headline number is overfit_validation held-out)")

    logger.info("building dependency tree with model and empirical lifts")
    tree = build_dependency_tree(pc, feat_map_clean, descriptions, safety_set, X=X)
    save_json(tree, METRICS / "dependency_structure.json")
    logger.info(f"tree: root=sae#{tree['root_sae_index']} nodes={tree['n_nodes']} "
                f"edges={tree['n_edges']} safety_to_safety_edges={tree['n_safety_to_safety_edges']} "
                f"levels={tree['levels']}")

    logger.info("computing rooting-invariant centrality (degree, betweenness, summed MI)")
    centrality = compute_centrality(tree, X, feat_map_clean, descriptions, safety_set)
    save_json(centrality, METRICS / "centrality.json")
    rc = centrality["root_centrality"]
    logger.info(f"root sae#{centrality['root_sae_index']} centrality: "
                f"degree={rc['degree']} (rank {rc['rank_by_degree']}/{centrality['n_nodes']}) "
                f"summed_mi_rank={rc['rank_by_summed_mi']} -- if rank is high the root is a "
                f"leaf and the 'central concept' framing must be revised")

    logger.info("running safety-clustering permutation test")
    perm_substring = safety_clustering_permutation(tree, safety_set, N_PERMUTATIONS, SEED)
    clustering = {
        "safety_set_source_used": "manual_override" if SAFETY_SET_MANUAL_OVERRIDE else "substring_keywords",
        "n_safety_substring": int(perm_substring["n_safety"]),
        "permutation_substring": perm_substring,
        "caveat": ("substring-derived safety_set has false positives (e.g. 'harm' in "
                   "'pharmaceutical'); validate by hand and set SAFETY_SET_MANUAL_OVERRIDE, "
                   "then the expected count s(s-1)/N and the p-value recompute for the clean s"),
    }
    logger.info(f"clustering (substring s={perm_substring['n_safety']}): "
                f"observed={perm_substring['observed_safety_to_safety_edges']} "
                f"expected={perm_substring['expected_analytic']:.3f} "
                f"p={perm_substring['p_value']} sig={perm_substring['significant_p05']}")
    if SAFETY_SET_MANUAL_OVERRIDE:
        perm_manual = safety_clustering_permutation(
            tree, SAFETY_SET_MANUAL_OVERRIDE, N_PERMUTATIONS, SEED)
        clustering["n_safety_manual"] = int(perm_manual["n_safety"])
        clustering["permutation_manual"] = perm_manual
        logger.info(f"clustering (manual s={perm_manual['n_safety']}): "
                    f"observed={perm_manual['observed_safety_to_safety_edges']} "
                    f"expected={perm_manual['expected_analytic']:.3f} "
                    f"p={perm_manual['p_value']} sig={perm_manual['significant_p05']}")
    save_json(clustering, METRICS / "safety_clustering.json")

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
        "description": "key safety-branch edges, model lift vs empirical lift; in-sample, weak "
                       "evidence. Lead with overfit_validation.dependency_lift_generalization "
                       "(model lift on train vs empirical lift on held-out test).",
        "edges": safety_branch,
    }, METRICS / "dependency_lifts.json")

    logger.info("running held-out overfit validation (fresh circuit, clean split)")
    overfit_auroc = None
    try:
        overfit = overfit_validation(X, n_safe, feat_map_clean, SAFETY_BRANCH_EDGES, device)
        save_json(overfit, METRICS / "overfit_validation.json")
        overfit_auroc = overfit.get("detection_auroc_heldout_test")
        logger.info(f"overfit check: heldout_auroc={overfit_auroc:.4f} "
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

    most_central_degree = (centrality["top_by_degree"][0]
                           if centrality["top_by_degree"] else None)
    most_central_mi = (centrality["top_by_summed_mi"][0]
                       if centrality["top_by_summed_mi"] else None)

    save_json({
        "layer": int(layer),
        "n_features": int(len(feat_map_clean)),
        "n_safety_features": int(len(safety_set & set(feat_map_clean))),
        "headline_detection_auroc_heldout": overfit_auroc,
        "in_corpus_detection_auroc": det["auroc"],
        "in_corpus_auroc_is_diagnostic_only": True,
        "root_sae_index": tree["root_sae_index"],
        "root_concept": tree["root_description"],
        "root_is_fixed_by_convention": True,
        "most_central_by_degree": most_central_degree,
        "most_central_by_summed_mi": most_central_mi,
        "collapsed": collapse["collapsed"],
        "transition_entropy_mean": collapse["mean"],
        "n_safety_to_safety_edges": tree["n_safety_to_safety_edges"],
        "safety_clustering_observed_edges": perm_substring["observed_safety_to_safety_edges"],
        "safety_clustering_expected": perm_substring["expected_analytic"],
        "safety_clustering_p_value": perm_substring["p_value"],
        "safety_clustering_significant_p05": perm_substring["significant_p05"],
        "outputs": {
            "dependency_structure": "metrics/dependency_structure.json",
            "dependency_lifts": "metrics/dependency_lifts.json",
            "centrality": "metrics/centrality.json",
            "safety_clustering": "metrics/safety_clustering.json",
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