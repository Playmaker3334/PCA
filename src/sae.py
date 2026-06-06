from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download

from .utils import SAE_DIR, SAE_REPO, SAE_WIDTH, get_logger, log_exception, setup_hf_auth

logger = get_logger("sae")


def find_sae_path(layer, width=SAE_WIDTH, base=SAE_DIR):
    base = Path(base)
    candidates = list(base.glob(f"layer_{layer}/width_{width}/average_l0_*"))
    if not candidates:
        raise FileNotFoundError(
            f"no SAE checkpoint for layer {layer} width {width} under {base}"
        )
    candidates.sort()
    return candidates[0]


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
    def from_gemma_scope(cls, layer, width=SAE_WIDTH, device="cuda"):
        try:
            path = find_sae_path(layer, width=width)
        except FileNotFoundError:
            path = download_sae(layer, width=width)
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
    out = np.zeros_like(z, dtype=np.uint8)
    for i in range(z.shape[0]):
        row = z[i]
        if (row > 0).sum() == 0:
            continue
        n_pos = int((row > 0).sum())
        kk = min(k, n_pos)
        idx = np.argpartition(-row, kk - 1)[:kk]
        idx = idx[row[idx] > 0]
        out[i, idx] = 1
    return out
