"""联邦学习核心组件（第 6 周 · T008）。

包含：模型定义、本地训练、评估、FedAvg 加权聚合。
聚合公式（计划原文）：
    w^{t+1} = Σ_{i∈S_t} (n_i / n_{S_t}) · w_i^t
其中 S_t 为第 t 轮实际参与的客户端集合（稀疏客户端场景下 |S_t| << N）。
"""
from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """窗口统计特征分类器（轻量，便于 CPU 上快速跑通联邦闭环）。"""

    def __init__(self, in_dim: int, n_classes: int = 3, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(in_dim: int, n_classes: int = 3, seed: int = 0) -> MLP:
    torch.manual_seed(seed)
    return MLP(in_dim, n_classes)


def train_local(global_model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                epochs: int = 1, lr: float = 1e-2, batch_size: int = 64,
                weight_decay: float = 1e-4, seed: int = 0,
                prox_mu: float = 0.0, prox_skip_prefix: str | None = None,
                sample_weights: torch.Tensor | None = None) -> tuple[nn.Module, int, float]:
    """客户端本地训练：从全局参数出发做 K 个 epoch 优化，返回 (本地模型, 样本数, 平均损失)。

    prox_mu > 0 时（FedProx）目标为 L_i(w) + (μ/2)·‖w − w^t‖²：
    在每一步对参数与初始化参数（即下发的 w^t）的平方距离做惩罚，抑制本地漂移。
    prox_skip_prefix 非空时，名字以该前缀开头的参数（例如个性化头 net.5.）不计入 prox 项
    —— SA-PFA 的设计要求：prox 只约束共享 backbone，个性化头自由更新。
    sample_weights 非空时按样本加权交叉熵（用于行程等权训练），默认 None 与原行为完全一致。
    """
    torch.manual_seed(seed)
    model = copy.deepcopy(global_model)
    model.train()
    ref_params = [(name, p.detach().clone()) for name, p in model.named_parameters()]  # 下发的 w^t（prox 锚点）
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(X)
    bs = max(1, min(batch_size, n))
    if sample_weights is None:
        dl = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)
    else:
        dl = DataLoader(TensorDataset(torch.arange(n)), batch_size=bs, shuffle=True)
    total, seen = 0.0, 0
    for _ in range(epochs):
        for batch in dl:
            if sample_weights is None:
                xb, yb = batch
                wb = None
            else:
                (ib,) = batch
                xb, yb = X[ib], y[ib]
                wb = sample_weights[ib]
            opt.zero_grad()
            out = model(xb)
            if wb is None:
                loss = F.cross_entropy(out, yb)
            else:
                loss = (F.cross_entropy(out, yb, reduction="none") * wb).sum() / wb.sum().clamp_min(1e-12)
            if prox_mu > 0.0:
                prox = 0.0
                for name, p in model.named_parameters():
                    if prox_skip_prefix and name.startswith(prox_skip_prefix):
                        continue
                    p0 = dict(ref_params)[name]
                    prox = prox + ((p - p0) ** 2).sum()
                loss = loss + 0.5 * prox_mu * prox
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            seen += len(yb)
    return model, n, total / max(seen, 1)


def train_local_scope(global_model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                      scope: str = "head", head_prefix: str = "net.5.",
                      epochs: int = 1, lr: float = 1e-2, batch_size: int = 64,
                      weight_decay: float = 1e-4, seed: int = 0):
    """只更新一部分参数（FedRep）：scope='head' 只训练个性化头，'body' 只训练 backbone。

    与 train_local 相同的优化器/批大小/损失；差别只是 requires_grad 的筛选。
    返回 (本地模型, 样本数, 平均损失)。
    """
    torch.manual_seed(seed)
    model = copy.deepcopy(global_model)
    model.train()
    for name, p in model.named_parameters():
        is_head = name.startswith(head_prefix)
        p.requires_grad = is_head if scope == "head" else (not is_head)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    n = len(X)
    bs = max(1, min(batch_size, n))
    dl = DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True)
    total, seen = 0.0, 0
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            seen += len(yb)
    for p in model.parameters():
        p.requires_grad = True
    return model, n, total / max(seen, 1)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, n_classes: int = 3) -> Dict[str, float]:
    model.eval()
    logits = model(X)
    loss = F.cross_entropy(logits, y).item()
    pred = logits.argmax(dim=1).cpu().numpy()
    yt = y.cpu().numpy()
    acc = float((pred == yt).mean())
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (yt, pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return {"loss": loss, "acc": acc, "macro_f1": float(f1.mean())}


def fedavg_aggregate(state_dicts: Sequence[OrderedDict], weights: Sequence[float]) -> OrderedDict:
    """FedAvg 加权聚合：w = Σ (n_i / Σn_j) · w_i（先升 float64 再还原 dtype）。"""
    wsum = float(sum(weights))
    assert wsum > 0
    agg = OrderedDict()
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].to(torch.float64) * (w / wsum) for sd, w in zip(state_dicts, weights)])
        agg[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return agg


def state_bytes(state: Dict[str, torch.Tensor]) -> int:
    """模型参数字节数（float32 = 4 bytes/参数），用于通信量统计。"""
    return int(sum(p.numel() for p in state.values()) * 4)


def standardize(train_X: np.ndarray, others: List[np.ndarray]):
    """按训练集统计量标准化特征（集中预处理，正式版改为客户端本地标准化）。"""
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_X - mean) / std, [(X - mean) / std for X in others], mean, std
