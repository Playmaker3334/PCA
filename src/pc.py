from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .utils import (
    PC_BATCH_SIZE,
    PC_LATENT_STATES,
    PC_LR,
    PC_NUM_EPOCHS,
    get_logger,
)

logger = get_logger("pc")


def pairwise_mutual_information(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    p1 = X.mean(axis=0).clip(eps, 1 - eps)
    p0 = 1.0 - p1
    M = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        Xi = X[:, i:i+1]
        joint11 = (X * Xi).mean(axis=0)
        joint01 = X.mean(axis=0) - joint11
        joint10 = Xi.mean() - joint11
        joint00 = 1.0 - joint01 - joint10 - joint11
        joints = np.stack([joint00, joint01, joint10, joint11], axis=0).clip(eps, 1)
        marg = np.stack([p0[i] * p0, p0[i] * p1, p1[i] * p0, p1[i] * p1], axis=0).clip(eps, 1)
        mi = (joints * np.log(joints / marg)).sum(axis=0)
        M[i] = mi
    np.fill_diagonal(M, 0.0)
    M = 0.5 * (M + M.T)
    return M


def build_chow_liu_tree(mi_matrix, root=0):
    d = mi_matrix.shape[0]
    G = nx.Graph()
    for i in range(d):
        G.add_node(i)
        for j in range(i + 1, d):
            G.add_edge(i, j, weight=-mi_matrix[i, j])
    T = nx.minimum_spanning_tree(G, algorithm="prim")
    parents = {root: -1}
    order = [root]
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in T.neighbors(u):
            if v not in visited:
                visited.add(v)
                parents[v] = u
                order.append(v)
                stack.append(v)
    return parents, order


class HCLT(nn.Module):
    def __init__(self, num_vars, num_states=PC_LATENT_STATES, parents=None, order=None, device="cuda"):
        super().__init__()
        self.num_vars = num_vars
        self.K = num_states
        self.parents = parents
        self.order = order
        self.device = device
        self.logit_root = nn.Parameter(torch.randn(num_states) * 0.1)
        self.logit_emit = nn.Parameter(torch.randn(num_vars, num_states, 2) * 0.1)
        trans_params = {}
        for v in order:
            if parents[v] != -1:
                trans_params[f"t_{v}"] = nn.Parameter(torch.randn(num_states, num_states) * 0.1)
        self.logit_trans = nn.ParameterDict(trans_params)
        self.to(device)
        self._children = {v: [] for v in order}
        for v in order:
            p = parents[v]
            if p != -1:
                self._children[p].append(v)
        self._rev_order = list(reversed(order))

    @classmethod
    def from_data(cls, X, num_states=PC_LATENT_STATES, root=0, device="cuda"):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        logger.info(f"computing MI matrix for {X.shape[1]} vars over {X.shape[0]} samples")
        mi = pairwise_mutual_information(X)
        parents, order = build_chow_liu_tree(mi, root=root)
        logger.info(f"tree built: {len(parents)} nodes, root={root}")
        return cls(num_vars=X.shape[1], num_states=num_states,
                   parents=parents, order=order, device=device)

    def _log_prob_impl(self, X, observed_mask=None):
        B, D = X.shape
        K = self.K
        log_root = F.log_softmax(self.logit_root, dim=-1)
        log_emit = F.log_softmax(self.logit_emit, dim=-1)
        log_trans = {v: F.log_softmax(self.logit_trans[f"t_{v}"], dim=-1)
                     for v in self.order if self.parents[v] != -1}

        if observed_mask is None:
            obs_mask = torch.ones(B, D, device=X.device, dtype=X.dtype)
        else:
            obs_mask = observed_mask

        log_p1 = log_emit[:, :, 1]
        log_p0 = log_emit[:, :, 0]
        x_exp = X.unsqueeze(-1)
        emit_log = x_exp * log_p1.unsqueeze(0) + (1.0 - x_exp) * log_p0.unsqueeze(0)
        emit_log = obs_mask.unsqueeze(-1) * emit_log

        upward = {}
        for v in self._rev_order:
            if self.parents[v] == -1:
                continue
            m = emit_log[:, v, :]
            for c in self._children[v]:
                m = m + upward[c]
            trans_v = log_trans[v]
            combined = trans_v.unsqueeze(0) + m.unsqueeze(1)
            msg_to_parent = torch.logsumexp(combined, dim=-1)
            upward[v] = msg_to_parent

        root = self.order[0]
        m_root = emit_log[:, root, :]
        for c in self._children[root]:
            m_root = m_root + upward[c]
        log_p = torch.logsumexp(log_root.unsqueeze(0) + m_root, dim=-1)
        return log_p

    def log_prob(self, X, observed_mask=None):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device, dtype=torch.float32)
        if observed_mask is not None and isinstance(observed_mask, np.ndarray):
            observed_mask = torch.tensor(observed_mask, dtype=torch.float32, device=self.device)
        return self._log_prob_impl(X, observed_mask=observed_mask)

    def fit(self, X, num_epochs=PC_NUM_EPOCHS, batch_size=PC_BATCH_SIZE,
            lr=PC_LR, val_X=None, log_every=1, mlflow_log=False):
        if isinstance(X, np.ndarray):
            X_t = torch.tensor(X, dtype=torch.float32)
        else:
            X_t = X.float()
        dataset = TensorDataset(X_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = []
        for epoch in range(num_epochs):
            self.train()
            losses = []
            pbar = tqdm(loader, desc=f"epoch {epoch+1}/{num_epochs}", leave=False)
            for (batch,) in pbar:
                batch = batch.to(self.device)
                lp = self._log_prob_impl(batch)
                loss = -lp.mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                losses.append(float(loss))
                pbar.set_postfix(loss=f"{losses[-1]:.3f}")
            mean_train = float(np.mean(losses))
            entry = {"epoch": epoch + 1, "train_nll": mean_train}
            if val_X is not None:
                self.eval()
                with torch.no_grad():
                    val_t = (torch.tensor(val_X, dtype=torch.float32)
                             if isinstance(val_X, np.ndarray) else val_X.float())
                    val_t = val_t.to(self.device)
                    val_lp = self._log_prob_impl(val_t)
                    entry["val_nll"] = float(-val_lp.mean())
            if (epoch + 1) % log_every == 0:
                logger.info(f"epoch {epoch+1}: " + " ".join(
                    [f"{k}={v:.4f}" for k, v in entry.items() if k != "epoch"]
                ))
            if mlflow_log:
                try:
                    import mlflow
                    mlflow.log_metric("train_nll", mean_train, step=epoch + 1)
                    if "val_nll" in entry:
                        mlflow.log_metric("val_nll", entry["val_nll"], step=epoch + 1)
                except Exception:
                    pass
            history.append(entry)
        return history

    @torch.no_grad()
    def mpe(self, X_observed, observed_mask=None):
        if isinstance(X_observed, np.ndarray):
            X = torch.tensor(X_observed, dtype=torch.float32, device=self.device)
        else:
            X = X_observed.to(self.device, dtype=torch.float32)
        if X.dim() == 1:
            X = X.unsqueeze(0)
        B, D = X.shape
        log_root = F.log_softmax(self.logit_root, dim=-1)
        log_emit = F.log_softmax(self.logit_emit, dim=-1)
        log_trans = {v: F.log_softmax(self.logit_trans[f"t_{v}"], dim=-1)
                     for v in self.order if self.parents[v] != -1}
        if observed_mask is None:
            obs_mask = torch.ones(B, D, device=self.device, dtype=X.dtype)
        else:
            obs_mask = observed_mask.to(self.device)

        log_p1 = log_emit[:, :, 1]
        log_p0 = log_emit[:, :, 0]
        x_exp = X.unsqueeze(-1)
        emit_log = x_exp * log_p1.unsqueeze(0) + (1.0 - x_exp) * log_p0.unsqueeze(0)
        emit_log = obs_mask.unsqueeze(-1) * emit_log

        upward = {}
        argmax_table = {}
        for v in self._rev_order:
            if self.parents[v] == -1:
                continue
            m = emit_log[:, v, :]
            for c in self._children[v]:
                m = m + upward[c]
            trans_v = log_trans[v]
            combined = trans_v.unsqueeze(0) + m.unsqueeze(1)
            msg, arg = combined.max(dim=-1)
            upward[v] = msg
            argmax_table[v] = arg

        root = self.order[0]
        m_root = emit_log[:, root, :]
        for c in self._children[root]:
            m_root = m_root + upward[c]
        root_combined = log_root.unsqueeze(0) + m_root
        z_root = root_combined.argmax(dim=-1)

        z_assign = {root: z_root}
        for v in self.order[1:]:
            parent = self.parents[v]
            zp = z_assign[parent]
            z_assign[v] = argmax_table[v][torch.arange(B, device=self.device), zp]

        x_mpe = torch.zeros(B, D, device=self.device)
        for v in range(D):
            zv = z_assign[v]
            p0 = log_emit[v, zv, 0]
            p1 = log_emit[v, zv, 1]
            x_mpe[:, v] = (p1 > p0).float()

        latents = torch.stack([z_assign[v] for v in range(D)], dim=1)
        return x_mpe.cpu().numpy(), latents.cpu().numpy()

    @torch.no_grad()
    def feature_contributions(self, X):
        if isinstance(X, np.ndarray):
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_t = X.to(self.device, dtype=torch.float32)
        if X_t.dim() == 1:
            X_t = X_t.unsqueeze(0)
        B, D = X_t.shape
        full_lp = self._log_prob_impl(X_t)
        contributions = torch.zeros(B, D, device=self.device)
        for v in range(D):
            mask = torch.ones(B, D, device=self.device)
            mask[:, v] = 0.0
            lp_masked = self._log_prob_impl(X_t, observed_mask=mask)
            contributions[:, v] = full_lp - lp_masked
        return contributions.cpu().numpy()

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "num_vars": self.num_vars,
            "K": self.K,
            "parents": self.parents,
            "order": self.order,
        }, path / "hclt.pt")
        logger.info(f"saved HCLT to {path}")

    @classmethod
    def load(cls, path, device="cuda"):
        path = Path(path)
        d = torch.load(path / "hclt.pt", map_location=device)
        pc = cls(
            num_vars=d["num_vars"],
            num_states=d["K"],
            parents=d["parents"],
            order=d["order"],
            device=device,
        )
        pc.load_state_dict(d["state_dict"])
        pc.to(device)
        pc.eval()
        return pc
