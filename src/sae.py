import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from .utils import SAE_DIR, SAE_REPO, SAE_WIDTH, get_logger, log_exception, setup_hf_auth

logger = get_logger("sae")

DEFAULT_TARGET_L0 = 68


def _parse_l0(path):
    m = re.search(r"average_l0_(\d+)", path.name)
    return int(m.group(1)) if m else None


def find_sae_path(layer, width=SAE_WIDTH, base=SAE_DIR, target_l0=DEFAULT_TARGET_L0):
    base = Path(base)
    candidates = list(base.glob(f"layer_{layer}/width_{width}/average_l0_*"))
    if not candidates:
        raise FileNotFoundError(
            f"no SAE checkpoint for layer {layer} width {width} under {base}"
        )
    parsed = [(p, _parse_l0(p)) for p in candidates]
    parsed = [(p, l0) for p, l0 in parsed if l0 is not None]
    if not parsed:
        candidates.sort()
        chosen = candidates[0]
        logger.warning(f"could not parse l0; falling back to {chosen.name}")
        return chosen
    parsed.sort(key=lambda t: (abs(t[1] - target_l0), t[1]))
    chosen, l0 = parsed[0]
    logger.info(f"layer {layer}: selected SAE {chosen.name} (l0={l0}, target={target_l0})")
    return chosen


def download_sae(layer, width=SAE_WIDTH):
    setup_hf_auth()
    pattern = f"layer_{layer}/width_{width}/average_l0_*/*"
    logger.info(f"downloading SAE layer {layer} width {width}")
    snapshot_download(
        repo_id=SAE_REPO,
        allow_patterns=[pattern],
        local_dir=str(SAE_DIR),
    )
    return find_sae_path(layer, width=width)


class JumpReLUSAE(nn.Module):
    def __init__(self, W_enc, W_dec, b_enc, b_dec, threshold, device="cuda", dtype=torch.float32):
        super().__init__()
        self.W_enc = nn.Parameter(torch.tensor(W_enc, dtype=dtype, device=device), requires_grad=False)
        self.W_dec = nn.Parameter(torch.tensor(W_dec, dtype=dtype, device=device), requires_grad=False)
        self.b_enc = nn.Parameter(torch.tensor(b_enc, dtype=dtype, device=device), requires_grad=False)
        self.b_dec = nn.Parameter(torch.tensor(b_dec, dtype=dtype, device=device), requires_grad=False)
        self.threshold = nn.Parameter(torch.tensor(threshold, dtype=dtype, device=device), requires_grad=False)
        self.d_model = self.W_enc.shape[0]
        self.n_features = self.W_enc.shape[1]
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_gemma_scope(cls, layer, width=SAE_WIDTH, device="cuda", target_l0=DEFAULT_TARGET_L0):
        try:
            path = find_sae_path(layer, width=width, target_l0=target_l0)
        except FileNotFoundError:
            download_sae(layer, width=width)
            path = find_sae_path(layer, width=width, target_l0=target_l0)
        npz_files = list(path.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"no .npz file in {path}")
        d = np.load(npz_files[0])
        logger.info(f"loaded SAE from {npz_files[0]}, keys: {list(d.keys())}")
        return cls(
            W_enc=d["W_enc"],
            W_dec=d["W_dec"],
            b_enc=d["b_enc"],
            b_dec=d["b_dec"],
            threshold=d["threshold"],
            device=device,
        )

    @torch.no_grad()
    def encode(self, h):
        if isinstance(h, np.ndarray):
            h = torch.tensor(h, dtype=self.dtype, device=self.device)
        if h.dim() == 1:
            h = h.unsqueeze(0)
        h = h.to(self.device, dtype=self.dtype)
        pre = h @ self.W_enc + self.b_enc
        mask = (pre > self.threshold).to(self.dtype)
        z = pre * mask
        return z

    @torch.no_grad()
    def encode_np(self, h):
        z = self.encode(h)
        return z.cpu().numpy()

    @torch.no_grad()
    def decode(self, z):
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=self.dtype, device=self.device)
        return z @ self.W_dec + self.b_dec

    @torch.no_grad()
    def reconstruct(self, h):
        z = self.encode(h)
        return self.decode(z)

    @torch.no_grad()
    def reconstruction_error(self, h):
        if isinstance(h, np.ndarray):
            h_t = torch.tensor(h, dtype=self.dtype, device=self.device)
        else:
            h_t = h.to(self.device, dtype=self.dtype)
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        h_hat = self.reconstruct(h_t)
        num = torch.linalg.norm(h_t - h_hat, dim=-1)
        den = torch.linalg.norm(h_t, dim=-1) + 1e-8
        rel = num / den
        return float(rel.mean().cpu())


def topk_binarize(z, k):
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    if z.ndim == 1:
        z = z[None, :]
    n, d = z.shape
    out = np.zeros((n, d), dtype=np.uint8)
    kk = min(k, d)
    if kk <= 0:
        return out
    idx = np.argpartition(-z, kk - 1, axis=1)[:, :kk]
    rows = np.repeat(np.arange(n), kk)
    cols = idx.reshape(-1)
    vals = z[rows, cols]
    keep = vals > 0
    out[rows[keep], cols[keep]] = 1
    return out