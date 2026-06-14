import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from datasets import load_from_disk

from src.model import GemmaResidualExtractor, format_prompt
from src.dataset_fields import DATASET_LABEL, get_text
from src.utils import (
    ACTIVATIONS,
    CANDIDATE_LAYERS,
    RAW,
    Timer,
    get_logger,
    gpu_info,
    log_exception,
    save_npz,
    set_seed,
    setup_mlflow,
)

logger = get_logger("extract")

CORPUS_MAX = {
    "alpaca": 3000,
    "triviaqa": 1500,
    "mmlu": 1500,
    "gsm8k": 800,
    "advbench": 520,
    "harmbench": 400,
    "jailbreakbench": 100,
    "xstest": 250,
    "inthewild": 800,
}


def collect_prompts(corpus_name, n_max_cfg):
    path = RAW / corpus_name
    if not path.exists():
        logger.warning(f"missing {corpus_name}")
        return []
    try:
        ds = load_from_disk(str(path))
    except Exception:
        log_exception(logger, f"failed loading {corpus_name}")
        return []
    prompts = []
    n_max = min(n_max_cfg, len(ds))
    for i in range(n_max):
        ex = ds[i]
        t = get_text(ex, corpus_name)
        if t:
            prompts.append(t)
    return prompts


def extract_for_corpus(extractor, prompts, layers, max_length=256):
    formatted = [format_prompt(p, extractor.tokenizer) for p in prompts]
    by_layer, valid = extractor.batch_residual_activations(formatted, layers, max_length=max_length)
    kept_prompts = [p for p, v in zip(prompts, valid) if v]
    return by_layer, kept_prompts


def main():
    set_seed()
    with setup_mlflow("01_extract_activations") as mlflow:
        mlflow.log_dict(gpu_info(), "gpu_info.json")
        mlflow.log_param("layers", CANDIDATE_LAYERS)
        try:
            extractor = GemmaResidualExtractor()
        except Exception:
            log_exception(logger, "model load failed")
            raise

        all_counts = {}
        for name, n_max_cfg in CORPUS_MAX.items():
            out_path = ACTIVATIONS / f"{name}.npz"
            if out_path.exists():
                logger.info(f"skip {name} (exists)")
                continue
            prompts = collect_prompts(name, n_max_cfg)
            if not prompts:
                logger.warning(f"no prompts for {name}")
                continue
            label = DATASET_LABEL[name]
            logger.info(f"extracting {name}: {len(prompts)} prompts (label={label})")
            try:
                with Timer(f"extract {name}", logger=logger):
                    by_layer, kept_prompts = extract_for_corpus(extractor, prompts, CANDIDATE_LAYERS)
                save_npz(
                    out_path,
                    prompts=np.array(kept_prompts, dtype=object),
                    label=np.array([label] * len(kept_prompts)),
                    **{f"layer_{L}": by_layer[L] for L in CANDIDATE_LAYERS},
                )
                all_counts[name] = len(kept_prompts)
                logger.info(f"saved {out_path}")
                mlflow.log_metric(f"n_extracted_{name}", len(kept_prompts))
            except Exception:
                log_exception(logger, f"extraction failed for {name}")

        mlflow.log_param("counts", all_counts)
        logger.info(f"extraction complete: {all_counts}")


if __name__ == "__main__":
    main()