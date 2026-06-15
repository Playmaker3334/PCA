import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .utils import METRICS, get_logger, log_exception, save_json

logger = get_logger("localization")


def neuronpedia_keyword_search(layer, width="16k", model="gemma-2-2b",
                                keywords=None, top_per_keyword=10, timeout=60):
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
        import gzip
        import json
        import xml.etree.ElementTree as ET
    except ImportError:
        logger.warning("requests not installed")
        return {}

    bucket = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
    sae_id = f"{layer}-gemmascope-res-{width}"
    prefix = f"v1/{model}/{sae_id}/explanations"
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"

    list_url = f"{bucket}/?list-type=2&prefix={prefix}/"
    try:
        lr = requests.get(list_url, timeout=timeout)
        root = ET.fromstring(lr.text)
        batch_keys = [c.find(f"{ns}Key").text
                      for c in root.findall(f"{ns}Contents")
                      if c.find(f"{ns}Key").text.endswith(".jsonl.gz")]
    except Exception:
        log_exception(logger, f"neuronpedia S3 list failed for {prefix}")
        return {}

    if not batch_keys:
        logger.warning(f"neuronpedia: no explanation batches at {prefix}")
        return {}

    all_features = {}
    for key in tqdm(batch_keys, desc="neuronpedia dl", leave=False):
        try:
            br = requests.get(f"{bucket}/{key}", timeout=timeout)
            text = gzip.decompress(br.content).decode("utf-8")
            for line in text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                idx = rec.get("index")
                desc = rec.get("description", "")
                if idx is not None and desc:
                    all_features[int(idx)] = desc
        except Exception:
            log_exception(logger, f"neuronpedia batch {key} failed")

    logger.info(f"neuronpedia layer {layer}: loaded {len(all_features)} "
                f"feature descriptions (sae_id={sae_id})")

    results = {}
    total_hits = 0
    for kw in keywords:
        kw_lower = kw.lower()
        matches = [{"index": idx, "description": desc}
                   for idx, desc in all_features.items()
                   if kw_lower in desc.lower()]
        matches = matches[:top_per_keyword]
        results[kw] = matches
        total_hits += len(matches)

    logger.info(f"neuronpedia layer {layer}: {total_hits} keyword hits "
                f"across {len(keywords)} keywords")
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


def rank_fusion(rankings, top_k=64):
    counts = {}
    weights = {}
    for ranking in rankings:
        for rank, (idx, score) in enumerate(ranking):
            counts[idx] = counts.get(idx, 0) + 1
            weights[idx] = weights.get(idx, 0.0) + (len(ranking) - rank)
    order = sorted(weights.keys(), key=lambda i: (-counts[i], -weights[i]))
    return order[:top_k]


intersect_rankings = rank_fusion


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