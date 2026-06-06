import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_from_disk

from src.eval import (
    evaluate_attack_success,
    evaluate_density_auroc,
    evaluate_gsm8k_subset,
    evaluate_mmlu_subset,
    evaluate_over_refusal,
)
from src.model import GemmaResidualExtractor
from src.monitor import SafetyMonitor
from src.pc import HCLT
from src.sae import JumpReLUSAE
from src.dataset_fields import DATASET_FIELDS
from src.utils import (
    METRICS,
    PC_DIR,
    RAW,
    SAE_FEATURES,
    Timer,
    get_logger,
    log_exception,
    load_json,
    load_npz,
    save_json,
    set_seed,
    setup_mlflow,
)

logger = get_logger("evaluate")

MAX_PER_BENCH = {
    "jailbreakbench": 100,
    "harmbench": 100,
    "advbench": 100,
    "xstest": 100,
    "mmlu": 150,
    "gsm8k": 80,
}


def load_monitor(device="cuda"):
    config = load_json(PC_DIR / "monitor" / "config.json")
    pc = HCLT.load(PC_DIR / "monitor" / "pc", device=device)
    sae = JumpReLUSAE.from_gemma_scope(config["layer_idx"], device=device)
    return SafetyMonitor(
        pc=pc,
        sae=sae,
        layer_idx=config["layer_idx"],
        refusal_features=config["refusal_features"],
        alpha=config["alpha"],
        threshold=config["threshold"],
        feature_index_map=np.array(config["feature_index_map"], dtype=np.int64),
        feature_descriptions={int(k): v for k, v in config["feature_descriptions"].items()},
    )


def get_prompts(name, n_max):
    """Usa los campos centralizados de dataset_fields."""
    path = RAW / name
    if not path.exists():
        logger.warning(f"missing {name}")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with setup_mlflow("04_evaluate"):
        monitor = load_monitor(device=device)
        extractor = GemmaResidualExtractor()
        mlflow.log_params({
            "layer": monitor.layer_idx,
            "threshold": monitor.threshold,
            "alpha": monitor.alpha,
        })

        results = {}

        # --- density AUROC SIN leakage: usa val_idx persistido en 03 ---
        try:
            layer = monitor.layer_idx
            split = load_json(METRICS / "pc_split.json")
            val_idx = np.array(split["val_idx"], dtype=np.int64)

            safe_feat = load_npz(SAE_FEATURES / f"layer_{layer}_safe.npz")["z_binary"]
            unsafe_feat = load_npz(SAE_FEATURES / f"layer_{layer}_unsafe.npz")["z_binary"]

            # restringe a las features del PC, igual que en 03
            X_safe_all = safe_feat[:, monitor.feature_index_map].astype(np.float32)
            X_unsafe = unsafe_feat[:, monitor.feature_index_map].astype(np.float32)

            # SOLO el conjunto de validacion safe (no visto por el PC)
            X_safe_val = X_safe_all[val_idx]
            logger.info(f"density eval: n_safe_val={len(X_safe_val)} n_unsafe={len(X_unsafe)}")

            density = evaluate_density_auroc(monitor, X_safe_val, X_unsafe)
            results["density_auroc"] = density
            mlflow.log_metric("density_auroc", density["auroc"])
            logger.info(f"density AUROC={density['auroc']:.4f}")
        except Exception:
            log_exception(logger, "density auroc failed")

        for bench in ["jailbreakbench", "harmbench", "advbench"]:
            prompts = get_prompts(bench, MAX_PER_BENCH[bench])
            if not prompts:
                continue
            logger.info(f"{bench}: {len(prompts)} prompts")
            try:
                with Timer(bench, logger=logger):
                    r = evaluate_attack_success(extractor, prompts, monitor=monitor)
                results[bench] = r
                mlflow.log_metric(f"{bench}_asr", r["attack_success_rate"])
                mlflow.log_metric(f"{bench}_monitor_block", r["monitor_block_rate"])
                logger.info(f"{bench} ASR={r['attack_success_rate']:.3f} "
                            f"block={r['monitor_block_rate']:.3f}")
            except Exception:
                log_exception(logger, f"{bench} eval failed")

        prompts = get_prompts("xstest", MAX_PER_BENCH["xstest"])
        if prompts:
            try:
                with Timer("xstest", logger=logger):
                    r = evaluate_over_refusal(extractor, prompts, monitor=monitor)
                results["xstest"] = r
                mlflow.log_metric("xstest_overrefusal", r["over_refusal_rate"])
                logger.info(f"XSTest OR={r['over_refusal_rate']:.3f}")
            except Exception:
                log_exception(logger, "xstest failed")

        mmlu_path = RAW / "mmlu"
        if mmlu_path.exists():
            try:
                ds = load_from_disk(str(mmlu_path))
                examples = [ds[i] for i in range(min(MAX_PER_BENCH["mmlu"], len(ds)))]
                with Timer("mmlu", logger=logger):
                    r = evaluate_mmlu_subset(extractor, examples)
                results["mmlu"] = r
                mlflow.log_metric("mmlu_accuracy", r["accuracy"])
                logger.info(f"MMLU acc={r['accuracy']:.3f}")
            except Exception:
                log_exception(logger, "mmlu failed")

        gsm_path = RAW / "gsm8k"
        if gsm_path.exists():
            try:
                ds = load_from_disk(str(gsm_path))
                examples = [ds[i] for i in range(min(MAX_PER_BENCH["gsm8k"], len(ds)))]
                with Timer("gsm8k", logger=logger):
                    r = evaluate_gsm8k_subset(extractor, examples)
                results["gsm8k"] = r
                mlflow.log_metric("gsm8k_accuracy", r["accuracy"])
                logger.info(f"GSM8K acc={r['accuracy']:.3f}")
            except Exception:
                log_exception(logger, "gsm8k failed")

        summary = {
            "layer": monitor.layer_idx,
            "threshold": float(monitor.threshold),
            "alpha": float(monitor.alpha),
            "density_auroc": results.get("density_auroc", {}).get("auroc"),
            "jailbreakbench_asr": results.get("jailbreakbench", {}).get("attack_success_rate"),
            "harmbench_asr": results.get("harmbench", {}).get("attack_success_rate"),
            "advbench_asr": results.get("advbench", {}).get("attack_success_rate"),
            "xstest_overrefusal": results.get("xstest", {}).get("over_refusal_rate"),
            "mmlu_accuracy": results.get("mmlu", {}).get("accuracy"),
            "gsm8k_accuracy": results.get("gsm8k", {}).get("accuracy"),
        }
        save_json(summary, METRICS / "evaluation_summary.json")
        save_json(
            {k: {kk: vv for kk, vv in v.items() if kk != "records"} if isinstance(v, dict) else v
             for k, v in results.items()},
            METRICS / "evaluation_full.json",
        )
        mlflow.log_artifact(str(METRICS / "evaluation_summary.json"))
        mlflow.log_artifact(str(METRICS / "evaluation_full.json"))
        logger.info(f"summary: {summary}")


if __name__ == "__main__":
    main()