from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .utils import FPR_TARGET, SAFETY_ALPHA, get_logger, save_json

logger = get_logger("monitor")


def calibrate_threshold(scores_safe, scores_unsafe, fpr_target=FPR_TARGET):
    labels = np.concatenate([np.zeros(len(scores_safe)), np.ones(len(scores_unsafe))])
    s = np.concatenate([scores_safe, scores_unsafe])
    fpr, tpr, thresholds = roc_curve(labels, s)
    auroc = float(roc_auc_score(labels, s))
    target_idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    target_idx = max(0, target_idx)
    tau = float(thresholds[target_idx])
    if not np.isfinite(tau):
        tau = float(np.quantile(scores_safe, 1.0 - fpr_target))
        return {
            "threshold": tau,
            "fpr_at_threshold": float((scores_safe > tau).mean()),
            "tpr_at_threshold": float((scores_unsafe > tau).mean()),
            "auroc": auroc,
        }
    return {
        "threshold": tau,
        "fpr_at_threshold": float(fpr[target_idx]),
        "tpr_at_threshold": float(tpr[target_idx]),
        "auroc": auroc,
    }


def expected_calibration_error(scores, labels, n_bins=10):
    s = np.array(scores)
    y = np.array(labels)
    s_min, s_max = s.min(), s.max()
    if s_max - s_min < 1e-8:
        return 0.0
    s_norm = (s - s_min) / (s_max - s_min)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(s)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (s_norm >= bin_edges[i]) & (s_norm <= bin_edges[i + 1])
        else:
            mask = (s_norm >= bin_edges[i]) & (s_norm < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = s_norm[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)


class SafetyMonitor:
    def __init__(self, pc, sae, layer_idx, refusal_features=None,
                 alpha=SAFETY_ALPHA, threshold=None,
                 feature_index_map=None, feature_descriptions=None):
        self.pc = pc
        self.sae = sae
        self.layer_idx = layer_idx
        self.refusal_features = refusal_features or []
        self.alpha = alpha
        self.threshold = threshold
        self.feature_index_map = (np.array(feature_index_map, dtype=np.int64)
                                  if feature_index_map is not None else None)
        self.feature_descriptions = feature_descriptions or {}

    def score(self, z_binary):
        if z_binary.ndim == 1:
            z_binary = z_binary[None, :]
        nll = -self.pc.log_prob(z_binary).detach().cpu().numpy()
        return {"nll": nll, "score": nll}

    def predict(self, z_binary):
        s = self.score(z_binary)
        s["is_unsafe"] = s["score"] > self.threshold if self.threshold is not None else None
        return s

    def explain(self, z_binary, top_k=5):
        if z_binary.ndim == 1:
            z_binary = z_binary[None, :]
        contributions = self.pc.feature_contributions(z_binary)
        explanations = []
        for i in range(z_binary.shape[0]):
            order = np.argsort(-np.abs(contributions[i]))[:top_k]
            row = []
            for j in order:
                feat_idx = int(j)
                if self.feature_index_map is not None:
                    orig = int(self.feature_index_map[feat_idx])
                else:
                    orig = feat_idx
                row.append({
                    "pc_index": feat_idx,
                    "sae_index": orig,
                    "contribution": float(contributions[i, j]),
                    "active": bool(z_binary[i, j] > 0.5),
                    "description": self.feature_descriptions.get(orig, ""),
                })
            explanations.append(row)
        return explanations

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.pc.save(path / "pc")
        config = {
            "layer_idx": int(self.layer_idx),
            "refusal_features": [int(f) for f in self.refusal_features],
            "alpha": float(self.alpha),
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "feature_index_map": ([int(x) for x in self.feature_index_map]
                                   if self.feature_index_map is not None else None),
            "feature_descriptions": {str(k): v for k, v in self.feature_descriptions.items()},
        }
        save_json(config, path / "config.json")
        logger.info(f"saved monitor to {path}")