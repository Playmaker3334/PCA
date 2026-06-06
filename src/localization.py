import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .utils import METRICS, get_logger, log_exception, save_json

logger = get_logger("localization")


def neuronpedia_keyword_search(layer, width="16k", model="gemma-2-2b",
                                keywords=None, top_per_keyword=10, timeout=20):
    if keywords is None:
        keywords = [
            "refusal", "refuse", "deny", "decline",
            "harmful", "harm", "danger", "dangerous", "unsafe",
            "violence", "violent", "weapon",
            "illegal", "unethical", "inappropriate",
            "warning", "caution",
            "instruction following", "compliance",
        ]
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed")
        return {}
    base = "https://www.neuronpedia.org/api/explanation/search"
    sae_id = f"{layer}-gemmascope-res-{width}"
    results = {}
    for kw in tqdm(keywords, desc="neuronpedia", leave=False):
        try:
            r = requests.post(
                base,
                json={"modelId": model, "saeId": sae_id, "query": kw, "limit": top_per_keyword},
                timeout=timeout,
            )
            if r.status_code == 200:
                data = r.json()
                feats = []
                for item in data.get("results", [])[:top_per_keyword]:
                    feat_idx = item.get("index") or item.get("feature")
                    desc = item.get("description") or item.get("explanation", "")
                    if feat_idx is not None:
                        feats.append({"index": int(feat_idx), "description": desc})
                results[kw] = feats
            else:
                logger.warning(f"neuronpedia {kw}: status {r.status_code}")
                results[kw] = []
        except Exception as e:
            log_exception(logger, f"neuronpedia query '{kw}' failed")
            results[kw] = []
    return results


def linear_probe(X_harmful, X_benign, C=0.1, test_size=0.2, seed=42):
    X = np.vstack([X_harmful, X_benign]).astype(np.float32)
    y = np.hstack([np.ones(len(X_harmful)), np.zeros(len(X_benign))])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(test_size * len(X))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auroc = float(roc_auc_score(y_test, probs))
    coefs = clf.coef_[0]
    nz = int((np.abs(coefs) > 1e-6).sum())
    order = np.argsort(-np.abs(coefs))
    top_features = order[:50]
    return {
        "auroc": auroc,
        "nonzero_coefs": nz,
        "top_features": [(int(i), float(coefs[i])) for i in top_features],
        "coef_vector": coefs,
    }


def contrastive_difference(X_harmful, X_benign, top_k=50):
    mu_h = X_harmful.mean(axis=0)
    mu_b = X_benign.mean(axis=0)
    combined = np.vstack([X_harmful, X_benign])
    std = combined.std(axis=0) + 1e-8
    diff = (mu_h - mu_b) / std
    order = np.argsort(-np.abs(diff))
    top = order[:top_k]
    return {
        "diff_vector": diff,
        "top_features": [(int(i), float(diff[i])) for i in top],
    }


def attribution_score(activations, labels, eps=1e-8):
    activations = activations.astype(np.float32)
    mask_h = labels == 1
    mask_b = labels == 0
    mu_h = activations[mask_h].mean(axis=0)
    mu_b = activations[mask_b].mean(axis=0)
    var_h = activations[mask_h].var(axis=0) + eps
    var_b = activations[mask_b].var(axis=0) + eps
    pooled_std = np.sqrt(0.5 * (var_h + var_b))
    cohen_d = (mu_h - mu_b) / (pooled_std + eps)
    return cohen_d


def intersect_rankings(rankings, top_k=64):
    counts = {}
    weights = {}
    for ranking in rankings:
        for rank, (idx, score) in enumerate(ranking):
            counts[idx] = counts.get(idx, 0) + 1
            weights[idx] = weights.get(idx, 0.0) + (len(ranking) - rank)
    order = sorted(weights.keys(), key=lambda i: (-counts[i], -weights[i]))
    return order[:top_k]


def save_localization_report(report, layer):
    path = METRICS / f"localization_layer_{layer}.json"
    serializable = {}
    for k, v in report.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, dict):
            sub = {}
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray):
                    sub[kk] = vv.tolist()
                else:
                    sub[kk] = vv
            serializable[k] = sub
        else:
            serializable[k] = v
    save_json(serializable, path)
    logger.info(f"saved report to {path}")
