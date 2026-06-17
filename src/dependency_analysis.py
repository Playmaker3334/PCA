import gzip
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from .pc import HCLT
from .utils import (
    PC_COLLAPSE_ENTROPY_THRESHOLD,
    PC_LATENT_STATES,
    PC_LR,
    PC_NUM_EPOCHS,
    PC_TRANS_INIT_SCALE,
    SEED,
    get_logger,
    log_exception,
)

logger = get_logger("dependency_analysis")

SAFETY_KEYWORDS = [
    "ethical", "violation", "violence", "violent", "harm", "harmful",
    "danger", "dangerous", "weapon", "illegal", "unsafe", "warning",
    "caution", "medical", "contraindication", "judicial", "legal",
    "disclaimer", "damage", "investigation", "misinformation",
    "cybersecurity", "threat", "abuse", "compliance", "accountability",
    "refuse", "refusal", "unethical", "inappropriate",
]

KNOWN_SHORT_LABELS = {
    8947: "ethical\nviolations", 10656: "medical\nwarnings",
    8051: "social\nexclusion", 1347: "violence", 15270: "legal/\njudicial",
    839: "damage", 12916: "investigations", 11202: "health",
    9803: "legal\nmatters", 1262: "disclaimers", 14974: "cybersec\nthreats",
    15661: "misinfo", 11155: "compliance", 500: "political\naccountability",
}


def _safe_float(x, cap=1.0e6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return cap if v > 0 else -cap
    return v


def select_real_features(selected_feature_indices, probe_top, contrastive_top, probe_threshold):
    real = set()
    for pair in probe_top:
        idx, coef = int(pair[0]), float(pair[1])
        if abs(coef) > probe_threshold:
            real.add(idx)
    for pair in contrastive_top:
        real.add(int(pair[0]))
    positions = [i for i, sae in enumerate(selected_feature_indices) if int(sae) in real]
    feat_map_clean = [int(selected_feature_indices[i]) for i in positions]
    return positions, feat_map_clean


def transition_entropy(pc, collapse_threshold=PC_COLLAPSE_ENTROPY_THRESHOLD):
    ents = []
    for v in pc.order:
        key = f"t_{v}"
        if key in pc.logit_trans:
            log_trans = F.log_softmax(pc.logit_trans[key], dim=-1)
            tr = torch.exp(log_trans).detach().cpu().numpy()
            K = tr.shape[-1]
            row_ent = -(tr * np.log(tr + 1e-12)).sum(axis=-1).mean()
            ents.append(float(row_ent / np.log(K)))
    ents = np.array(ents, dtype=np.float64)
    if len(ents) == 0:
        return {"mean": None, "n_transitions": 0, "collapsed": None}
    n_uniform = int((ents > collapse_threshold).sum())
    return {
        "mean": _safe_float(ents.mean()),
        "median": _safe_float(np.median(ents)),
        "min": _safe_float(ents.min()),
        "max": _safe_float(ents.max()),
        "n_transitions": int(len(ents)),
        "n_near_uniform": n_uniform,
        "collapse_threshold": float(collapse_threshold),
        "collapsed": bool(ents.mean() > collapse_threshold),
        "interpretation": (
            "COLLAPSED: transitions are near-uniform, the circuit reduces to "
            "independent marginals and does not model conditional dependencies"
            if ents.mean() > collapse_threshold else
            "OK: transitions carry structure, the circuit models conditional dependencies"
        ),
        "per_transition_entropy": [_safe_float(e) for e in ents],
    }


def model_lift(pc, idx_a, idx_b):
    D = pc.num_vars
    dev = pc.device

    def lp(values):
        x = torch.zeros(1, D, device=dev)
        m = torch.zeros(1, D, device=dev)
        for i, val in values.items():
            x[0, i] = val
            m[0, i] = 1.0
        with torch.no_grad():
            return float(pc.log_prob(x, observed_mask=m).item())

    p1 = np.exp(lp({idx_a: 1.0, idx_b: 1.0}) - lp({idx_a: 1.0}))
    p0 = np.exp(lp({idx_a: 0.0, idx_b: 1.0}) - lp({idx_a: 0.0}))
    lift = p1 / p0 if p0 > 1e-9 else float("inf")
    return {
        "p_child_given_parent_active": _safe_float(p1),
        "p_child_given_parent_inactive": _safe_float(p0),
        "lift": _safe_float(lift),
        "parent_inactive_prob_zero": bool(p0 <= 1e-9),
    }


def empirical_lift(X, idx_a, idx_b):
    pa = float(X[:, idx_a].mean())
    pb = float(X[:, idx_b].mean())
    pab = float((X[:, idx_a] * X[:, idx_b]).mean())
    p1 = pab / pa if pa > 1e-9 else 0.0
    p0 = (pb - pab) / (1.0 - pa) if (1.0 - pa) > 1e-9 else 0.0
    lift = p1 / p0 if p0 > 1e-9 else float("inf")
    return {
        "p_child_given_parent_active": _safe_float(p1),
        "p_child_given_parent_inactive": _safe_float(p0),
        "lift": _safe_float(lift),
        "freq_parent": _safe_float(pa),
        "freq_child": _safe_float(pb),
        "parent_inactive_prob_zero": bool(p0 <= 1e-9),
    }


def is_safety_description(desc):
    d = (desc or "").lower()
    return any(k in d for k in SAFETY_KEYWORDS)


def derive_safety_set(feat_map_clean, descriptions, refusal_sae_indices=None):
    safety = set()
    for sae_idx in feat_map_clean:
        if is_safety_description(descriptions.get(int(sae_idx), "")):
            safety.add(int(sae_idx))
    for idx in (refusal_sae_indices or []):
        safety.add(int(idx))
    return safety


def auroc_pure(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def detection_metrics(pc, X_safe, X_unsafe):
    with torch.no_grad():
        nll_safe = -pc.log_prob(X_safe).detach().cpu().numpy()
        nll_unsafe = -pc.log_prob(X_unsafe).detach().cpu().numpy()
    scores = np.concatenate([nll_safe, nll_unsafe])
    labels = np.concatenate([np.zeros(len(nll_safe)), np.ones(len(nll_unsafe))])
    return {
        "auroc": auroc_pure(scores, labels),
        "nll_safe_mean": _safe_float(nll_safe.mean()),
        "nll_unsafe_mean": _safe_float(nll_unsafe.mean()),
        "separation": _safe_float(nll_unsafe.mean() - nll_safe.mean()),
        "n_safe": int(len(nll_safe)),
        "n_unsafe": int(len(nll_unsafe)),
    }


def build_dependency_tree(pc, feat_map_clean, descriptions, safety_set, X=None):
    pc_idx_to_sae = {i: int(feat_map_clean[i]) for i in range(len(feat_map_clean))}
    root_pc = pc.order[0]
    root_sae = pc_idx_to_sae[root_pc]

    nodes = []
    for pc_idx in range(pc.num_vars):
        sae_idx = pc_idx_to_sae[pc_idx]
        nodes.append({
            "pc_index": int(pc_idx),
            "sae_index": int(sae_idx),
            "description": descriptions.get(sae_idx, ""),
            "is_safety": bool(sae_idx in safety_set),
            "is_root": bool(pc_idx == root_pc),
            "parent_pc_index": int(pc.parents[pc_idx]),
            "parent_sae_index": (int(pc_idx_to_sae[pc.parents[pc_idx]])
                                 if pc.parents[pc_idx] != -1 else None),
            "n_children": int(sum(1 for v in pc.order if pc.parents[v] == pc_idx)),
        })

    edges = []
    for v in pc.order:
        parent = pc.parents[v]
        if parent == -1:
            continue
        child_sae = pc_idx_to_sae[v]
        parent_sae = pc_idx_to_sae[parent]
        ml = model_lift(pc, parent, v)
        edge = {
            "parent_sae_index": int(parent_sae),
            "child_sae_index": int(child_sae),
            "parent_description": descriptions.get(parent_sae, ""),
            "child_description": descriptions.get(child_sae, ""),
            "parent_is_safety": bool(parent_sae in safety_set),
            "child_is_safety": bool(child_sae in safety_set),
            "safety_to_safety": bool(parent_sae in safety_set and child_sae in safety_set),
            "model_lift": ml["lift"],
            "model_p_child_given_parent_active": ml["p_child_given_parent_active"],
            "model_p_child_given_parent_inactive": ml["p_child_given_parent_inactive"],
        }
        if X is not None:
            el = empirical_lift(X, parent, v)
            edge["empirical_lift"] = el["lift"]
            edge["empirical_p_child_given_parent_active"] = el["p_child_given_parent_active"]
            edge["empirical_p_child_given_parent_inactive"] = el["p_child_given_parent_inactive"]
            edge["freq_parent"] = el["freq_parent"]
            edge["freq_child"] = el["freq_child"]
        edges.append(edge)

    levels = {root_pc: 0}
    queue = [root_pc]
    children_map = defaultdict(list)
    for v in pc.order:
        if pc.parents[v] != -1:
            children_map[pc.parents[v]].append(v)
    while queue:
        u = queue.pop(0)
        for c in children_map[u]:
            if c not in levels:
                levels[c] = levels[u] + 1
                queue.append(c)
    level_counts = defaultdict(int)
    for lv in levels.values():
        level_counts[lv] += 1

    return {
        "root_sae_index": int(root_sae),
        "root_description": descriptions.get(root_sae, ""),
        "n_nodes": int(pc.num_vars),
        "n_edges": int(len(edges)),
        "n_safety_nodes": int(len(safety_set & set(feat_map_clean))),
        "n_safety_to_safety_edges": int(sum(1 for e in edges if e["safety_to_safety"])),
        "levels": {int(k): int(v) for k, v in sorted(level_counts.items())},
        "node_depth": {int(pc_idx_to_sae[k]): int(v) for k, v in levels.items()},
        "nodes": nodes,
        "edges": edges,
    }


def overfit_validation(X, n_safe, feat_map_clean, edges_to_check, device,
                       test_frac=0.25, seed=123,
                       num_states=PC_LATENT_STATES, lr=PC_LR,
                       num_epochs=PC_NUM_EPOCHS, trans_init_scale=PC_TRANS_INIT_SCALE):
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_test = int(test_frac * n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    X_train = X[train_idx]
    X_test = X[test_idx]

    is_safe = np.zeros(n, dtype=bool)
    is_safe[:n_safe] = True
    test_is_safe = is_safe[test_idx]

    inner = rng.permutation(len(X_train))
    n_val = int(0.15 * len(X_train))
    val_inner = X_train[inner[:n_val]]
    tr_inner = X_train[inner[n_val:]]

    pc_v = HCLT.from_data(tr_inner, num_states=num_states, root=0,
                          device=device, trans_init_scale=trans_init_scale)
    pc_v.fit(tr_inner, num_epochs=num_epochs, batch_size=128, lr=lr, val_X=val_inner)

    with torch.no_grad():
        nll_test = -pc_v.log_prob(X_test).detach().cpu().numpy()
    test_scores = nll_test
    test_labels = (~test_is_safe).astype(np.int64)
    test_auroc = auroc_pure(test_scores, test_labels)

    pc_idx = {int(feat_map_clean[i]): i for i in range(len(feat_map_clean))}
    lift_generalization = []
    for parent_sae, child_sae, label in edges_to_check:
        if parent_sae in pc_idx and child_sae in pc_idx:
            a, b = pc_idx[parent_sae], pc_idx[child_sae]
            ml = model_lift(pc_v, a, b)["lift"]
            el_test = empirical_lift(X_test, a, b)["lift"]
            lift_generalization.append({
                "edge": label,
                "parent_sae_index": int(parent_sae),
                "child_sae_index": int(child_sae),
                "model_lift_train": ml,
                "empirical_lift_heldout_test": el_test,
            })

    ent = transition_entropy(pc_v)
    return {
        "method": "clean train/test split, fresh circuit trained only on train, evaluated on held-out test",
        "test_frac": float(test_frac),
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "detection_auroc_heldout_test": test_auroc,
        "transition_entropy_validation_circuit": ent["mean"],
        "dependency_lift_generalization": lift_generalization,
        "verdict": (
            "NOT OVERFIT: detection generalizes to held-out test and dependency lifts "
            "persist when measured on data the circuit never saw during training"
            if (test_auroc is not None and test_auroc > 0.9) else
            "INCONCLUSIVE OR OVERFIT: held-out detection degraded, inspect lift generalization"
        ),
    }


def fetch_descriptions(layer, indices=None, width="16k", model="gemma-2-2b", timeout=60):
    try:
        import requests
    except ImportError:
        logger.warning("requests not available, cannot fetch neuronpedia descriptions")
        return {}
    bucket = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
    sae_id = f"{layer}-gemmascope-res-{width}"
    prefix = f"v1/{model}/{sae_id}/explanations"
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    try:
        lr = requests.get(f"{bucket}/?list-type=2&prefix={prefix}/", timeout=timeout)
        root = ET.fromstring(lr.text)
        keys = [c.find(f"{ns}Key").text for c in root.findall(f"{ns}Contents")
                if c.find(f"{ns}Key").text.endswith(".jsonl.gz")]
    except Exception:
        log_exception(logger, f"neuronpedia list failed for {prefix}")
        return {}
    want = set(int(i) for i in indices) if indices is not None else None
    out = {}
    for key in keys:
        try:
            br = requests.get(f"{bucket}/{key}", timeout=timeout)
            text = gzip.decompress(br.content).decode("utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                idx = rec.get("index")
                desc = rec.get("description", "")
                if idx is None:
                    continue
                idx = int(idx)
                if want is not None and idx not in want:
                    continue
                if desc:
                    out[idx] = desc
        except Exception:
            log_exception(logger, f"neuronpedia batch {key} failed")
    logger.info(f"fetched {len(out)} descriptions for layer {layer}")
    return out


def build_semantic_analysis(feat_map_clean, descriptions, probe_top, contrastive_top,
                            safety_set, tree):
    probe_map = {int(p[0]): float(p[1]) for p in probe_top}
    contrast_map = {int(p[0]): float(p[1]) for p in contrastive_top}
    depth = tree.get("node_depth", {})
    parent_of = {}
    children_of = defaultdict(list)
    for e in tree["edges"]:
        parent_of[e["child_sae_index"]] = e["parent_sae_index"]
        children_of[e["parent_sae_index"]].append(e["child_sae_index"])
    feats = []
    for sae_idx in feat_map_clean:
        sae_idx = int(sae_idx)
        role = "root" if sae_idx == tree["root_sae_index"] else (
            "internal" if children_of.get(sae_idx) else "leaf")
        feats.append({
            "sae_index": sae_idx,
            "description": descriptions.get(sae_idx, ""),
            "is_safety": bool(sae_idx in safety_set),
            "probe_coefficient": _safe_float(probe_map.get(sae_idx)),
            "contrastive_diff": _safe_float(contrast_map.get(sae_idx)),
            "tree_role": role,
            "tree_depth": int(depth.get(sae_idx, -1)),
            "tree_parent_sae_index": parent_of.get(sae_idx),
            "tree_children_sae_indices": children_of.get(sae_idx, []),
        })
    feats.sort(key=lambda f: (-(f["probe_coefficient"] or 0.0)))
    return {
        "n_features": len(feats),
        "n_safety_features": int(sum(1 for f in feats if f["is_safety"])),
        "features": feats,
    }


def _short_label(sae_idx, descriptions):
    if sae_idx in KNOWN_SHORT_LABELS:
        return KNOWN_SHORT_LABELS[sae_idx]
    d = descriptions.get(sae_idx, "")
    if d:
        words = d.split()[:2]
        return "\n".join(words) if words else f"f{sae_idx}"
    return f"f{sae_idx}"


def generate_figures(tree, descriptions, safety_set, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.lines import Line2D
    except ImportError:
        logger.warning("matplotlib/networkx not available, skipping figures")
        return []

    edge_lift = {}
    G_full = nx.DiGraph()
    for e in tree["edges"]:
        u, v = e["parent_sae_index"], e["child_sae_index"]
        G_full.add_edge(u, v)
        edge_lift[(u, v)] = e["model_lift"] if e["model_lift"] is not None else 1.0
    root = tree["root_sae_index"]

    def hierarchical_pos(G):
        levels = {root: 0} if root in G else {}
        queue = [root] if root in G else []
        while queue:
            u = queue.pop(0)
            for v in G.successors(u):
                if v not in levels:
                    levels[v] = levels[u] + 1
                    queue.append(v)
        for n in G.nodes():
            levels.setdefault(n, 1)
        by_level = defaultdict(list)
        for n, lv in levels.items():
            by_level[lv].append(n)
        max_w = max((len(v) for v in by_level.values()), default=1)
        pos = {}
        for lv, nodes in by_level.items():
            nodes_sorted = sorted(nodes, key=lambda n: (n not in safety_set, n))
            w = len(nodes_sorted)
            for i, n in enumerate(nodes_sorted):
                pos[n] = ((i - (w - 1) / 2) * (max_w / max(w, 1)), -lv * 2.2)
        return pos

    saved = []

    def draw(G, path, title):
        plt.figure(figsize=(20, 11))
        pos = hierarchical_pos(G)
        ncol = ["#d62728" if n in safety_set else "#dce3ea" for n in G.nodes()]
        nsz = [2600 if n in safety_set else 1000 for n in G.nodes()]
        lifts = np.array([np.clip(edge_lift.get(e, 1.0), 1, 50) for e in G.edges()])
        wd = 0.6 + 4.0 * (lifts / 50.0)
        ec = ["#d62728" if (u in safety_set and v in safety_set) else "#aab4be"
              for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=wd, edge_color=ec, arrows=True,
                               arrowsize=12, alpha=0.55)
        nx.draw_networkx_nodes(G, pos, node_color=ncol, node_size=nsz,
                               edgecolors="#2c3e50", linewidths=1.3, alpha=0.95)
        nx.draw_networkx_labels(G, pos,
                                labels={n: _short_label(n, descriptions) for n in G.nodes()},
                                font_size=7.5, font_weight="bold")
        plt.legend(handles=[
            Line2D([0], [0], color="#d62728", lw=4, label="safety to safety dependency"),
            Line2D([0], [0], color="#aab4be", lw=1.5, label="other dependency"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   markersize=13, label="safety concept", lw=0),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#dce3ea",
                   markersize=10, label="other concept", lw=0),
        ], loc="lower right", fontsize=10, framealpha=0.9)
        plt.title(title, fontsize=13, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=170, bbox_inches="tight")
        plt.close()
        saved.append(str(path))

    draw(G_full, out_dir / "concept_hierarchy.png",
         "Learned dependency hierarchy over SAE concepts (Gemma-2-2B layer 17)\n"
         "edge width proportional to conditional lift")

    keep = set(safety_set)
    for s in list(safety_set):
        if s in G_full:
            keep.update(G_full.predecessors(s))
            keep.update(G_full.successors(s))
    G_pruned = G_full.subgraph(keep).copy()
    draw(G_pruned, out_dir / "concept_hierarchy_pruned.png",
         "Safety-concept dependency subgraph (Gemma-2-2B layer 17)\n"
         "pruned to safety concepts and direct neighbors, edge width proportional to conditional lift")

    return saved
