import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from huggingface_hub import snapshot_download

from src.utils import (
    CANDIDATE_LAYERS,
    MODEL_REPO,
    RAW,
    SAE_DIR,
    SAE_REPO,
    SAE_WIDTH,
    get_logger,
    log_exception,
    setup_hf_auth,
    setup_mlflow,
    gpu_info,
)

logger = get_logger("download")

DATASETS = {
    "alpaca": ("tatsu-lab/alpaca", None, "train"),
    "mmlu": ("cais/mmlu", "all", "test"),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "validation"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "advbench": ("walledai/AdvBench", None, "train"),
    "harmbench": ("walledai/HarmBench", "standard", "train"),
    "jailbreakbench": ("JailbreakBench/JBB-Behaviors", "behaviors", "harmful"),
    "harmful_hirundo": ("hirundo-io/harmful-prompts-refusals", None, "train"),
    "xstest": ("natolambert/xstest-v2-copy", None, "gpt4"),
}


def download_model_weights():
    setup_hf_auth()
    logger.info(f"downloading model weights {MODEL_REPO}")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=None,
        allow_patterns=["*.json", "*.safetensors", "*.model", "tokenizer*"],
    )
    logger.info("model weights cached")


def download_sae():
    setup_hf_auth()
    SAE_DIR.mkdir(parents=True, exist_ok=True)
    for layer in CANDIDATE_LAYERS:
        try:
            pattern = f"layer_{layer}/width_{SAE_WIDTH}/average_l0_*/*"
            logger.info(f"downloading SAE layer {layer}")
            snapshot_download(
                repo_id=SAE_REPO,
                allow_patterns=[pattern],
                local_dir=str(SAE_DIR),
            )
        except Exception:
            log_exception(logger, f"failed SAE layer {layer}")


def download_datasets():
    RAW.mkdir(parents=True, exist_ok=True)
    success = []
    failed = []
    for name, (repo, config, split) in DATASETS.items():
        target = RAW / name
        if target.exists() and any(target.iterdir()):
            logger.info(f"skip {name} (exists)")
            success.append(name)
            continue
        logger.info(f"downloading {name}")
        try:
            ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
            ds.save_to_disk(str(target))
            logger.info(f"saved {name}: {len(ds)} examples")
            success.append(name)
        except Exception:
            log_exception(logger, f"failed {name}")
            failed.append(name)
    return success, failed


def main():
    with setup_mlflow("00_download") as mlflow:
        mlflow.log_dict(gpu_info(), "gpu_info.json")
        try:
            download_model_weights()
            mlflow.log_param("model_downloaded", True)
        except Exception:
            log_exception(logger, "model download failed")
            mlflow.log_param("model_downloaded", False)
        try:
            download_sae()
            mlflow.log_param("sae_downloaded", True)
        except Exception:
            log_exception(logger, "sae download failed")
            mlflow.log_param("sae_downloaded", False)
        success, failed = download_datasets()
        mlflow.log_param("datasets_success", success)
        mlflow.log_param("datasets_failed", failed)
        logger.info(f"download done. success={success} failed={failed}")


if __name__ == "__main__":
    main()