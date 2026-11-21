"""第 11 周 · T043：SA-PFA（Sparse-Aware Personalized Federated Aggregation）实现。

与第 10 周设计文档（research/paper/notes/week10_SA-PFA方法设计.md）逐节对应：
  §2 ActivityScore      活跃度 A_i：EWMA 参与率（离线轮补齐）+ 新鲜度衰减 + 下限 A_min
  §3 QualityScore       数据质量 Q_i：完整性 κ · 邻域标签一致性 ψ · 近重复冲突率 ν · 样本充分度 s
  §4 ContributionScore  贡献度 Ĉ_i：全局验证集损失下降 → EWMA → 均值归一化 + 截断 [0.5, 1.5]
  §5 sparse_aware_weights  稀疏感知聚合：度量加权 → 时间平滑 η → 权重下限 ω → 归一化
   §5' 支持度门控（W15 v0.3，默认关闭）：n_i < gate_min_n 或 类别数 < gate_min_classes 的
       "支持度退化"参与者豁免 ω/K 下限（见 research/paper/notes/week15_T050_预注册判据.md §2）
  §6 结构：共享 backbone 聚合（Σα_i·φ_i）+ 本地个性化头（不参与聚合）+ 共享层 prox（μ）

自检（T044 的一部分）：
  /opt/anaconda3/bin/python research/code/federated/proposed.py
  - 活跃度：与逐轮 EWMA 暴力实现逐位对比（含离线 gap）
  - 质量：κ/ψ/ν/s 边界与几何平均手算对比
  - 权重：退化性（A=Q=C=1, ω=η=0 → α ∝ n^d）、下限性质（α_i ≥ ω/K）、归一化（Σα=1）
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# §2 活跃度
# --------------------------------------------------------------------------- #
@dataclass
class ClientEvalState:
    """每客户端的评估状态（服务器侧维护；A/Q 的原始量由客户端自报标量）。"""

    a_ema: float = 0.0          # ā_i：EWMA 参与率
    last_seen: int = -1         # τ_i：上次参与轮次（-1 = 从未参与）
    q_ema: float = 1.0          # Q̄_i：质量（本数据集为静态量，接口保留 EWMA）
    c_ema: float = 1.0          # C̄_i：贡献度 EWMA
    alpha_hat: float | None = None   # α̂_i：时间平滑后的权重
    n: int = 0                  # n_i：本地样本数


def activity_score(state: ClientEvalState, t: int, lam: float = 0.8, T: float = 5.0,
                   gamma: float = 0.5, a_min: float = 0.05) -> float:
    """客户端在第 t 轮参与时的活跃度 A_i(t)（设计文档 §2）。

    - ā_i 按"离线轮 p=0、本轮 p=1"的 EWMA 补齐：ā ← λ^gap·ā + (1−λ)，gap = t − τ_i
      （首次参与时 τ_i = −1，等价于"从第 0 轮起一直离线"，gap 取 t+1）
    - r_i 用**到达前的** staleness：r = exp(−(gap−1)/T)（gap=1 即连续参与 → r=1）
    - A_i = clip(γ·ā + (1−γ)·r, A_min, 1)；随后更新 τ_i ← t
    """
    first = state.last_seen < 0
    gap = (t + 1) if first else (t - state.last_seen)
    r = 1.0 if first else float(np.exp(-(gap - 1) / T))
    state.a_ema = lam ** gap * state.a_ema + (1.0 - lam)
    state.last_seen = t
    return float(np.clip(gamma * state.a_ema + (1.0 - gamma) * r, a_min, 1.0))


def activity_trace(participated: list[bool], lam: float = 0.8, T: float = 5.0,
                   gamma: float = 0.5, a_min: float = 0.05) -> list[float]:
    """给定每轮是否参与，返回 A_i(t) 序列（离线轮返回的是"若此刻上线"的 A；用于画图/自检）。"""
    state = ClientEvalState()
    out = []
    for t, p in enumerate(participated):
        if p:
            out.append(activity_score(state, t, lam=lam, T=T, gamma=gamma, a_min=a_min))
        else:
            # 离线轮：不更新状态，仅记录"此刻若上线"的参考值（不改变 last_seen）
            first = state.last_seen < 0
            gap = (t + 1) if first else (t - state.last_seen)
            a = lam ** gap * state.a_ema + (1.0 - lam)
            r = 1.0 if first else float(np.exp(-(gap - 1) / T))
            out.append(float(np.clip(gamma * a + (1.0 - gamma) * r, a_min, 1.0)))
    return out


# --------------------------------------------------------------------------- #
# §3 数据质量
# --------------------------------------------------------------------------- #
def neighborhood_stats(X: np.ndarray, y: np.ndarray, k: int = 5) -> dict:
    """本地数据的邻域结构（全部在客户端本地计算，只上传 2 个标量）。

    ψ （标签一致性）= 每窗口 k 近邻中同标签比例的平均
    ν （无近重复冲突）= 1 − 近重复对（距离 < 0.5×中位距离）中标签冲突的比例
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = len(X)
    if n < 3:
        return dict(psi=1.0, nu=1.0, n=n)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    kk = min(k, n - 1)
    nn = np.argpartition(d2, kk, axis=1)[:, :kk]
    psi = float((y[nn] == y[:, None]).mean())
    finite = d2[np.isfinite(d2)]
    thr = 0.5 * float(np.median(finite))
    dup = d2 < thr
    m = int(dup.sum())
    if m == 0:
        nu = 1.0
    else:
        clash = int((dup & (y[:, None] != y[None, :])).sum())
        nu = float(1.0 - clash / m)
    return dict(psi=psi, nu=nu, n=n)


def quality_score(n: int, n_ref: float, kappa: float = 1.0, psi: float = 1.0, nu: float = 1.0,
                  beta: float = 0.5, q_min: float = 0.05, use_size: bool = False) -> float:
    """Q_i（设计文档 §3）。

    use_size=False（默认）: Q = clip((κ·ψ·ν)^{1/3}) —— 样本量已经通过 α 里的 n_i^d 生效，
                            不重复计入质量项（避免对大数据客户端的二次偏置）
    use_size=True         : Q = clip((κ·ψ·ν·s)^{1/4})，s = min(1, n/n_ref)^β
    """
    n_ref = max(float(n_ref), 1.0)
    s = min(1.0, float(n) / n_ref) ** beta
    if use_size:
        q = (max(kappa, 0.0) * max(psi, 0.0) * max(nu, 0.0) * s) ** 0.25
    else:
        q = (max(kappa, 0.0) * max(psi, 0.0) * max(nu, 0.0)) ** (1.0 / 3.0)
    return float(np.clip(q, q_min, 1.0))


# --------------------------------------------------------------------------- #
# §4 贡献度
# --------------------------------------------------------------------------- #
class ContributionScore:
    """服务器侧贡献度：C^raw = clip((ℓ(φ^t;D_val) − ℓ(φ_i^t;D_val)) / ℓ(φ^t;D_val), 0, 1)。"""

    def __init__(self, rho: float = 0.3, c_min: float = 0.5, c_max: float = 1.5):
        self.rho = rho
        self.c_min = c_min
        self.c_max = c_max
        self.ema: dict[str, float] = {}
        self.last_raw: dict[str, float] = {}
        self.last_norm: dict[str, float] = {}

    def observe(self, cid: str, loss_global: float, loss_candidate: float) -> float:
        if loss_global <= 1e-12:
            raw = 0.0
        else:
            raw = float(np.clip((loss_global - loss_candidate) / loss_global, 0.0, 1.0))
        self.ema[cid] = (1.0 - self.rho) * self.ema.get(cid, 1.0) + self.rho * raw
        self.last_raw[cid] = raw
        return raw

    def normalized(self, participants: list[str]) -> dict[str, float]:
        vals = np.array([self.ema.get(c, 1.0) for c in participants], dtype=np.float64)
        mean = float(vals.mean()) if len(vals) else 1.0
        mean = mean if mean > 1e-12 else 1.0
        out = {}
        for c in participants:
            out[c] = float(np.clip(self.ema.get(c, 1.0) / mean, self.c_min, self.c_max))
        self.last_norm = out
        return out


# --------------------------------------------------------------------------- #
# §5 稀疏感知聚合
# --------------------------------------------------------------------------- #
def sparse_aware_weights(entries: list[dict], states: dict[str, ClientEvalState],
                         exp_a: float = 1.0, exp_b: float = 1.0, exp_c: float = 1.0,
                         exp_d: float = 1.0, floor: float = 0.2, momentum: float = 0.6,
                         normalize: bool = True,
                         gate_min_n: float | None = None,
                         gate_min_classes: int | None = None) -> dict[str, float]:
    """α_i：度量加权 → 时间平滑 → 权重下限 → 归一化（设计文档 §5）。

    entries: [{'cid':str, 'A':float, 'Q':float, 'C':float, 'n':int}, ...]（本轮参与者）

    W15 v0.3：gate_min_n / gate_min_classes 给出时启用支持度门控——命中者豁免 ω/K 下限
    （直接用平滑权重 α̂ 参与归一化），不改变其他参与者的任何算式。
    """
    if len(entries) == 0:
        return {}
    gated = gated_clients(entries, gate_min_n=gate_min_n, gate_min_classes=gate_min_classes)
    raw = {}
    for e in entries:
        raw[e["cid"]] = ((max(e["A"], 0.0) ** exp_a) * (max(e["Q"], 0.0) ** exp_b)
                         * (max(e["C"], 0.0) ** exp_c) * (max(float(e["n"]), 1.0) ** exp_d))
    tot = float(sum(raw.values()))
    K = len(entries)
    alpha = {}
    for cid, v in raw.items():
        tilde = v / tot
        st = states[cid]
        prev = tilde if st.alpha_hat is None else st.alpha_hat
        hat = momentum * prev + (1.0 - momentum) * tilde
        st.alpha_hat = hat
        alpha[cid] = hat if cid in gated else (1.0 - floor) * hat + floor / K
    if normalize:
        s = float(sum(alpha.values()))
        alpha = {k: v / s for k, v in alpha.items()}
    return alpha


def gated_clients(entries: list[dict], gate_min_n: float | None = None,
                  gate_min_classes: int | None = None) -> set:
    """支持度门控（W15 §v0.3）：返回本轮参与者中被判定为"支持度退化"的 cid 集合。

    规则（预注册写死，见 week15_T050_预注册判据.md §2）：
      n_i < gate_min_n 或 类别数 k_i < gate_min_classes（任一开启即生效；均关闭返回空集）。
    entries 需含 'n'；若提供 'n_cls'（本地训练类别数）则参与类别覆盖判定。
    """
    out: set = set()
    if not gate_min_n and not gate_min_classes:
        return out
    for e in entries:
        n, k = e.get("n"), e.get("n_cls")
        hit = False
        if gate_min_n and n is not None and float(n) < float(gate_min_n):
            hit = True
        if gate_min_classes and k is not None and float(k) < float(gate_min_classes):
            hit = True
        if hit:
            out.add(e["cid"])
    return out


def weights_from_schedule(sched: dict[int, list[int]], rounds: int, N: int,
                          A_of, Q_of, C_of, n_of, **kw) -> dict[int, dict[str, float]]:
    """离线工具：用给定参与调度推演 α 轨迹（第 12/14 周分析用）。"""
    states = {c: ClientEvalState() for c in range(N)}
    out = {}
    for t in range(rounds):
        part = sched.get(t, [])
        if part:
            entries = [dict(cid=c, A=A_of(c, t), Q=Q_of(c), C=C_of(c), n=n_of(c)) for c in part]
            alpha = sparse_aware_weights(entries, states, **kw)
            out[t] = alpha
    return out


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def _brute_force_activity(p: list[bool], lam: float, T: float, gamma: float, a_min: float) -> list[float]:
    """逐轮暴力 EWMA（p=0/1），用于对照 activity_score 的补齐公式。"""
    a = 0.0
    last = None
    out = []
    for t, pt in enumerate(p):
        a = lam * a + (1 - lam) * (1.0 if pt else 0.0)
        if pt:
            gap = 0 if last is None else t - last
            r = 1.0 if last is None else float(np.exp(-(gap - 1) / T))
            out.append(float(np.clip(gamma * a + (1 - gamma) * r, a_min, 1.0)))
            last = t
    return out


def _self_test() -> None:
    rng = np.random.default_rng(0)
    ok = True

    # 1) 活跃度：补齐公式 vs 暴力 EWMA
    for trial in range(20):
        p = (rng.random(60) < 0.4).tolist()
        if not any(p):
            continue
        st = ClientEvalState()
        fast = [activity_score(st, t) for t, pt in enumerate(p) if pt]
        slow = _brute_force_activity(p, 0.8, 5.0, 0.5, 0.05)
        if not np.allclose(fast, slow, atol=1e-12):
            ok = False
            print(f"[FAIL] activity trial {trial}: fast={fast[:4]} slow={slow[:4]}")
    print(f"[ok] activity_score 与暴力 EWMA 一致（lam=0.8, T=5, gamma=0.5）: {ok}")

    # 2) 质量：几何平均与样本充分度
    q1 = quality_score(100, 100)                       # 全 1 → 1
    q2 = quality_score(25, 100, use_size=True)         # s=0.5 → 0.5^{1/4}
    q3 = quality_score(0, 100, kappa=0.0)
    print(f"[ok] quality_score: Q(全1)={q1:.4f} (期望 1.0), Q(s=0.5)={q2:.4f} "
          f"(期望 {0.5 ** 0.25:.4f}), Q(κ=0)={q3:.4f} (期望 Q_min=0.05)")
    ok &= abs(q1 - 1.0) < 1e-12 and abs(q2 - 0.5 ** 0.25) < 1e-12 and abs(q3 - 0.05) < 1e-12

    # 3) 权重：退化性 / 下限 / 归一化
    states = {c: ClientEvalState() for c in range(4)}
    entries = [dict(cid=c, A=1.0, Q=1.0, C=1.0, n=n) for c, n in zip(range(4), [100, 50, 25, 25])]
    a_deg = sparse_aware_weights(entries, states, floor=0.0, momentum=0.0)
    expect = np.array([100, 50, 25, 25], dtype=float)
    expect = expect / expect.sum()
    got = np.array([a_deg[c] for c in range(4)])
    print(f"[ok] 退化性（floor=0, mom=0）: α={np.round(got, 4).tolist()} vs FedAvg 权重 {np.round(expect, 4).tolist()}"
          f" | max|Δ|={float(np.abs(got - expect).max()):.2e}")
    ok &= bool(np.allclose(got, expect, atol=1e-12))

    states2 = {c: ClientEvalState() for c in range(4)}
    entries2 = [dict(cid=c, A=0.01, Q=1.0, C=1.0, n=1000) for c in range(4)]
    a_floor = sparse_aware_weights(entries2, states2, floor=0.2, momentum=0.0)
    v = np.array([a_floor[c] for c in range(4)])
    print(f"[ok] 权重下限: min α={v.min():.4f} ≥ ω/K={0.2 / 4:.4f}，Σα={v.sum():.6f}")
    ok &= bool(v.min() >= 0.2 / 4 - 1e-12) and abs(v.sum() - 1.0) < 1e-12

    # 4) 贡献度归一化截断
    cs = ContributionScore()
    for _ in range(30):
        cs.observe("good", 1.0, 0.5)
        cs.observe("bad", 1.0, 1.05)
    norm = cs.normalized(["good", "bad"])
    print(f"[ok] 贡献度: good={norm['good']:.3f} (上限 1.5), bad={norm['bad']:.3f} (下限 0.5)")
    ok &= norm["good"] <= 1.5 + 1e-12 and norm["bad"] >= 0.5 - 1e-12

    print("[ALL OK]" if ok else "[SOME CHECKS FAILED]")


if __name__ == "__main__":
    _self_test()
