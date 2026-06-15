import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.eval import baseline_refusal_direction, baseline_score_with_direction
from src.model import GemmaResidualExtractor, format_prompt, is_refusal
from src.dataset_fields import DATASET_FIELDS, SAFE_CORPORA, UNSAFE_CORPORA
from src.length_matching import apply_length_matching
from src.utils import (
    ACTIVATIONS,
    METRICS,
    RAW,
    SEED,
    get_logger,
    load_json,
    load_npz,
    log_exception,
    save_json,
    set_seed,
    setup_mlflow,
)

logger = get_logger("baselines")

SAFE_FILES = [c for c in SAFE_CORPORA if c != "xstest"]
UNSAFE_FILES = list(UNSAFE_CORPORA)
MATCH_TOL = 15
MATCH_K = 3


def load_layer_acts(layer):
    safe_acts, unsafe_acts = [], []
    safe_prompts, unsafe_prompts = [], []
    for n in SAFE_FILES:
        p = ACTIVATIONS / f"{n}.npz"
        if p.exists():
            d = load_npz(p)
            if f"layer_{layer}" in d.files:
                safe_acts.append(d[f"layer_{layer}"])
                safe_prompts.extend(list(d["prompts"]))
    for n in UNSAFE_FILES:
        p = ACTIVATIONS / f"{n}.npz"
        if p.exists():
            d = load_npz(p)
            if f"layer_{layer}" in d.files:
                unsafe_acts.append(d[f"layer_{layer}"])
                unsafe_prompts.extend(list(d["prompts"]))
    safe = np.vstack(safe_acts) if safe_acts else np.zeros((0, 1))
    unsafe = np.vstack(unsafe_acts) if unsafe_acts else np.zeros((0, 1))

    if len(safe) and len(unsafe):
        safe, unsafe, safe_prompts, unsafe_prompts = apply_length_matching(
            safe, unsafe, safe_prompts, unsafe_prompts,
            tol=MATCH_TOL, k=MATCH_K, seed=SEED, logger=logger,
        )

    return safe, unsafe


def split_safe(n_safe, val_frac=0.1, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_safe)
    n_val = int(val_frac * n_safe)
    return idx[n_val:], idx[:n_val]


def evaluate_direction(direction, X_safe, X_unsafe):
    s_safe = baseline_score_with_direction(X_safe, direction)
    s_unsafe = baseline_score_with_direction(X_unsafe, direction)
    labels = np.concatenate([np.zeros(len(s_safe)), np.ones(len(s_unsafe))])
    scores = np.concatenate([s_safe, s_unsafe])
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "mean_safe": float(s_safe.mean()),
        "mean_unsafe": float(s_unsafe.mean()),
    }


def evaluate_vanilla(extractor, prompts, max_new_tokens=128):
    refused = 0
    records = []
    for p in tqdm(prompts, desc="vanilla", leave=False):
        try:
            formatted = format_prompt(p, extractor.tokenizer)
            out = extractor.generate(formatted, max_new_tokens=max_new_tokens)
            r = is_refusal(out)
            if r:
                refused += 1
            records.append({"prompt": p, "response": out, "refused": r})
        except Exception:
            log_exception(logger, f"vanilla failed: {p[:60]!r}")
    N = len(records)
    return {
        "n_prompts": N,
        "refusal_rate": refused / N if N > 0 else 0.0,
        "attack_success_rate": (N - refused) / N if N > 0 else 0.0,
        "records": records,
    }


def get_prompts(name, n_max):
    path = RAW / name
    if not path.exists():
        return []
    try:
        ds = load_from_disk(str(path))
    except Exception:
        log_exception(logger, f"load failed: {name}")
        return []
    fields = DATASET_FIELDS.get(name, [])
    prompts = []
    for i in range(min(n_max, len(ds))):
        ex = ds[i]
        for f in fields:
            v = ex.get(f)
            if isinstance(v, str) and len(v.strip()) > 0:
                prompts.append(v)
                break
    return prompts


def main():
    import mlflow
    set_seed()
    decision = load_json(METRICS / "localization_decision.json")
    layer = decision["best_layer"]
    logger.info(f"baselines for layer {layer}")

    with setup_mlflow("05_baselines"):
        mlflow.log_param("layer", layer)

        safe_acts, unsafe_acts = load_layer_acts(layer)
        logger.info(f"safe={safe_acts.shape} unsafe={unsafe_acts.shape}")

        try:
            n_unsafe = len(unsafe_acts)
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(n_unsafe)
            n_sel = int(0.5 * n_unsafe)
            select_idx = perm[:n_sel]
            eval_idx = perm[n_sel:]

            safe_train_idx, safe_val_idx = split_safe(len(safe_acts), seed=SEED)

            direction = baseline_refusal_direction(
                unsafe_acts[select_idx], safe_acts[safe_train_idx]
            )
            ev = evaluate_direction(
                direction, safe_acts[safe_val_idx], unsafe_acts[eval_idx]
            )
            mlflow.log_metric("refusal_direction_auroc", ev["auroc"])
            logger.info(f"refusal direction AUROC={ev['auroc']:.4f} "
                        f"(matched corpus, fit on U_select, eval on U_eval)")
        except Exception:
            log_exception(logger, "refusal direction failed")
            ev = None

        extractor = GemmaResidualExtractor()

        jb = get_prompts("jailbreakbench", 100)
        vanilla_jb = evaluate_vanilla(extractor, jb) if jb else None
        if vanilla_jb:
            mlflow.log_metric("vanilla_jbb_asr", vanilla_jb["attack_success_rate"])
            logger.info(f"vanilla JBB ASR={vanilla_jb['attack_success_rate']:.3f}")

        xs = get_prompts("xstest", 100)
        vanilla_xs = evaluate_vanilla(extractor, xs) if xs else None
        if vanilla_xs:
            mlflow.log_metric("vanilla_xstest_refusal", vanilla_xs["refusal_rate"])
            logger.info(f"vanilla XSTest refusal={vanilla_xs['refusal_rate']:.3f}")

        out = {
            "layer": layer,
            "refusal_direction": ev,
            "vanilla_jailbreakbench": {
                "asr": vanilla_jb["attack_success_rate"] if vanilla_jb else None,
                "refusal_rate": vanilla_jb["refusal_rate"] if vanilla_jb else None,
                "n": vanilla_jb["n_prompts"] if vanilla_jb else 0,
            },
            "vanilla_xstest": {
                "over_refusal_rate": vanilla_xs["refusal_rate"] if vanilla_xs else None,
                "n": vanilla_xs["n_prompts"] if vanilla_xs else 0,
            },
        }
        save_json(out, METRICS / "baselines.json")
        mlflow.log_artifact(str(METRICS / "baselines.json"))
        logger.info("baselines complete")


if __name__ == "__main__":
    main()