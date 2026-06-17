import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def detect_root():
    candidates = [
        Path("/kaggle/working/pc-sae"),
        Path("/kaggle/working"),
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "src").exists() and (c / "scripts").exists():
            return c
    return Path.cwd()


ROOT = detect_root()
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
RAW = DATA / "raw"
ACTIVATIONS = DATA / "activations"
SAE_FEATURES = DATA / "sae_features"
CHECKPOINTS = OUTPUTS / "checkpoints"
SAE_DIR = CHECKPOINTS / "sae"
PC_DIR = CHECKPOINTS / "pc"
RESULTS = OUTPUTS / "results"
METRICS = RESULTS / "metrics"
FIGURES = RESULTS / "figures"
LOG_DIR = OUTPUTS / "logs"
MLRUNS_DIR = OUTPUTS / "mlruns"

for p in [OUTPUTS, DATA, RAW, ACTIVATIONS, SAE_FEATURES,
          CHECKPOINTS, SAE_DIR, PC_DIR,
          RESULTS, METRICS, FIGURES, LOG_DIR, MLRUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


MODEL_REPO = "google/gemma-2-2b-it"
SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_WIDTH = "16k"
CANDIDATE_LAYERS = [13, 17, 20]

TOP_K_FEATURES = 64
PC_NUM_TOP_FEATURES = 128
PC_LATENT_STATES = 8
PC_NUM_EPOCHS = 120
PC_LR = 1e-2
PC_BATCH_SIZE = 128
PC_TRANS_INIT_SCALE = 2.0
PC_COLLAPSE_ENTROPY_THRESHOLD = 0.95
PC_REAL_FEATURE_PROBE_THRESHOLD = 0.01
PC_DROP_PADDING = True

SAFETY_ALPHA = 0.7
FPR_TARGET = 0.05

SEED = 42

MLFLOW_EXPERIMENT = "pc-sae"
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_hf_token():
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.environ.get("HF_TOKEN")
    except ImportError:
        pass
    return None


def setup_hf_auth():
    token = get_hf_token()
    if token is None:
        raise RuntimeError(
            "no HF_TOKEN found. set via Kaggle Secrets (Add-ons -> Secrets), "
            "or env variable, or .env file"
        )
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    return token


def get_logger(name):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(LOG_DIR / f"{name}.log", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    err_fh = logging.FileHandler(LOG_DIR / "errors.log", mode="a")
    err_fh.setLevel(logging.ERROR)
    err_fh.setFormatter(fmt)
    logger.addHandler(err_fh)
    logger.propagate = False
    return logger


def log_exception(logger, msg=""):
    tb = traceback.format_exc()
    logger.error(f"{msg}\n{tb}")


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_npz(path):
    return np.load(path, allow_pickle=True)


class _MlflowStub:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, _name):
        def _noop(*args, **kwargs):
            return None
        return _noop


def setup_mlflow(run_name=None):
    return _MlflowStub()


def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False}
        info = {"cuda": True, "num_gpus": torch.cuda.device_count(), "gpus": []}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
            })
        return info
    except Exception as e:
        return {"error": str(e)}


class Timer:
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        dt = time.time() - self.t0
        msg = f"{self.name} took {dt:.2f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)