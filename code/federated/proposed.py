"""SA-PFA: Sparse-Aware Personalized Federated Aggregation.

ActivityScore          A_i: EWMA participation rate with offline-round backfill, staleness
                       decay and a lower bound A_min
QualityScore           Q_i: completeness kappa * neighbourhood label agreement psi *
                       near-duplicate conflict rate nu * sample sufficiency s
ContributionScore      C_i: drop in D_val loss -> EWMA -> mean normalisation, clipped to [0.5, 1.5]
sparse_aware_weights   measure weighting -> temporal smoothing eta -> weight floor omega ->
                       normalisation; the optional support gate exempts support-degraded
                       participants (n_i < gate_min_n or fewer than gate_min_classes classes)
                       from the omega/K floor
Structure              the shared backbone is aggregated as sum_i alpha_i * phi_i, the
                       personalised head stays local and is never aggregated, and a prox term
                       mu constrains the shared layers only

Self-check (run directly, needs no data):
  python code/federated/proposed.py
  - activity: compared bit for bit against a brute-force per-round EWMA (including gaps)
  - quality: kappa/psi/nu/s boundary cases against hand-computed geometric means
  - weights: degeneracy (A=Q=C=1, omega=eta=0 -> alpha proportional to n^d), floor property
    (alpha_i >= omega/K) and normalisation (sum alpha = 1)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Activity score
# --------------------------------------------------------------------------- #
@dataclass
class ClientEvalState:
    """Per-client evaluation state kept on the server (A/Q raw quantities are self-reported scalars)."""

    a_ema: float = 0.0          # a_bar_i: EWMA participation rate
    last_seen: int = -1         # tau_i: last participating round (-1 = never)
    q_ema: float = 1.0          # Q_bar_i: quality (static here; the EWMA interface is kept)
    c_ema: float = 1.0          # C_bar_i: contribution EWMA
    alpha_hat: float | None = None   # alpha_hat_i: temporally smoothed weight
    n: int = 0                  # n_i: local sample count


def activity_score(state: ClientEvalState, t: int, lam: float = 0.8, T: float = 5.0,
                   gamma: float = 0.5, a_min: float = 0.05) -> float:
    """Activity A_i(t) of a client that participates in round t.

    - a_bar_i is backfilled with offline rounds (p=0) and this round (p=1):
      a <- lam^gap * a + (1-lam), gap = t - tau_i
      (first participation: tau_i = -1, equivalent to being offline since round 0, gap = t+1)
    - r_i uses the staleness *before* arrival: r = exp(-(gap-1)/T) (gap=1 -> r=1)
    - A_i = clip(gamma*a + (1-gamma)*r, A_min, 1); then tau_i <- t
    """
    first = state.last_seen < 0
    gap = (t + 1) if first else (t - state.last_seen)
    r = 1.0 if first else float(np.exp(-(gap - 1) / T))
    state.a_ema = lam ** gap * state.a_ema + (1.0 - lam)
    state.last_seen = t
    return float(np.clip(gamma * state.a_ema + (1.0 - gamma) * r, a_min, 1.0))


def activity_trace(participated: list[bool], lam: float = 0.8, T: float = 5.0,
                   gamma: float = 0.5, a_min: float = 0.05) -> list[float]:
    """A_i(t) sequence for a given participation trace (offline rounds report the value the
    client would obtain if it came online now; used for plotting and self-checks)."""
    state = ClientEvalState()
    out = []
    for t, p in enumerate(participated):
        if p:
            out.append(activity_score(state, t, lam=lam, T=T, gamma=gamma, a_min=a_min))
        else:
            # offline round: state is not updated, only the "if online now" reference value is
            # recorded (last_seen is unchanged)
            first = state.last_seen < 0
            gap = (t + 1) if first else (t - state.last_seen)
            a = lam ** gap * state.a_ema + (1.0 - lam)
            r = 1.0 if first else float(np.exp(-(gap - 1) / T))
            out.append(float(np.clip(gamma * a + (1.0 - gamma) * r, a_min, 1.0)))
    return out


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def neighborhood_stats(X: np.ndarray, y: np.ndarray, k: int = 5) -> dict:
    """Neighbourhood structure of the local data (computed on the client; only 2 scalars are uploaded).

    psi (label agreement)  = mean fraction of same-label windows among the k nearest neighbours
    nu  (no near-duplicate conflict) = 1 - fraction of near-duplicate pairs (distance <
         0.5 * median distance) whose labels disagree
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
    """Quality score Q_i.

    use_size=False (default): Q = clip((kappa*psi*nu)^(1/3)) - sample size already enters through
                              n_i^d in alpha, so it is not counted twice (avoids a second bias
                              towards large clients)
    use_size=True           : Q = clip((kappa*psi*nu*s)^(1/4)), s = min(1, n/n_ref)^beta
    """
    n_ref = max(float(n_ref), 1.0)
    s = min(1.0, float(n) / n_ref) ** beta
    if use_size:
        q = (max(kappa, 0.0) * max(psi, 0.0) * max(nu, 0.0) * s) ** 0.25
    else:
        q = (max(kappa, 0.0) * max(psi, 0.0) * max(nu, 0.0)) ** (1.0 / 3.0)
    return float(np.clip(q, q_min, 1.0))


# --------------------------------------------------------------------------- #
# Contribution score
# --------------------------------------------------------------------------- #
class ContributionScore:
    """Server-side contribution: C_raw = clip((L(phi^t;D_val) - L(phi_i^t;D_val)) / L(phi^t;D_val), 0, 1)."""

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
# Sparse-aware aggregation
# --------------------------------------------------------------------------- #
def sparse_aware_weights(entries: list[dict], states: dict[str, ClientEvalState],
                         exp_a: float = 1.0, exp_b: float = 1.0, exp_c: float = 1.0,
                         exp_d: float = 1.0, floor: float = 0.2, momentum: float = 0.6,
                         normalize: bool = True,
                         gate_min_n: float | None = None,
                         gate_min_classes: int | None = None) -> dict[str, float]:
    """alpha_i: measure weighting -> temporal smoothing -> weight floor -> normalisation.

    entries: [{'cid':str, 'A':float, 'Q':float, 'C':float, 'n':int}, ...] (this round's participants)

    When gate_min_n / gate_min_classes are given the support gate is active: gated clients
    skip the omega/K floor (their smoothed weight enters normalisation unchanged) and no other
    participant's computation changes.
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
    """Support gate: returns the participants judged support-degraded in this round.

    Rule: n_i < gate_min_n or number of classes k_i < gate_min_classes (either check activates
    when its threshold is set; both unset returns the empty set).
    entries must contain 'n'; when 'n_cls' (local training classes) is present it also drives the
    class-coverage check.
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
    """Offline helper: replay the alpha trajectory for a given participation schedule (analysis only)."""
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
# Self-check
# --------------------------------------------------------------------------- #
def _brute_force_activity(p: list[bool], lam: float, T: float, gamma: float, a_min: float) -> list[float]:
    """Round-by-round brute-force EWMA (p=0/1), used to validate the backfill formula."""
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

    # 1) activity: backfill formula vs brute-force EWMA
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
    print(f"[ok] activity_score matches brute-force EWMA (lam=0.8, T=5, gamma=0.5): {ok}")

    # 2) quality: geometric mean and sample sufficiency
    q1 = quality_score(100, 100)                       # all ones -> 1
    q2 = quality_score(25, 100, use_size=True)         # s=0.5 -> 0.5^{1/4}
    q3 = quality_score(0, 100, kappa=0.0)
    print(f"[ok] quality_score: Q(all ones)={q1:.4f} (expect 1.0), Q(s=0.5)={q2:.4f} "
          f"(expect {0.5 ** 0.25:.4f}), Q(kappa=0)={q3:.4f} (expect Q_min=0.05)")
    ok &= abs(q1 - 1.0) < 1e-12 and abs(q2 - 0.5 ** 0.25) < 1e-12 and abs(q3 - 0.05) < 1e-12

    # 3) weights: degeneracy / floor / normalisation
    states = {c: ClientEvalState() for c in range(4)}
    entries = [dict(cid=c, A=1.0, Q=1.0, C=1.0, n=n) for c, n in zip(range(4), [100, 50, 25, 25])]
    a_deg = sparse_aware_weights(entries, states, floor=0.0, momentum=0.0)
    expect = np.array([100, 50, 25, 25], dtype=float)
    expect = expect / expect.sum()
    got = np.array([a_deg[c] for c in range(4)])
    print(f"[ok] degeneracy (floor=0, mom=0): alpha={np.round(got, 4).tolist()} "
          f"vs FedAvg weights {np.round(expect, 4).tolist()}"
          f" | max|delta|={float(np.abs(got - expect).max()):.2e}")
    ok &= bool(np.allclose(got, expect, atol=1e-12))

    states2 = {c: ClientEvalState() for c in range(4)}
    entries2 = [dict(cid=c, A=0.01, Q=1.0, C=1.0, n=1000) for c in range(4)]
    a_floor = sparse_aware_weights(entries2, states2, floor=0.2, momentum=0.0)
    v = np.array([a_floor[c] for c in range(4)])
    print(f"[ok] weight floor: min alpha={v.min():.4f} >= omega/K={0.2 / 4:.4f}, sum alpha={v.sum():.6f}")
    ok &= bool(v.min() >= 0.2 / 4 - 1e-12) and abs(v.sum() - 1.0) < 1e-12

    # 4) contribution normalisation and clipping
    cs = ContributionScore()
    for _ in range(30):
        cs.observe("good", 1.0, 0.5)
        cs.observe("bad", 1.0, 1.05)
    norm = cs.normalized(["good", "bad"])
    print(f"[ok] contribution: good={norm['good']:.3f} (cap 1.5), bad={norm['bad']:.3f} (floor 0.5)")
    ok &= norm["good"] <= 1.5 + 1e-12 and norm["bad"] >= 0.5 - 1e-12

    print("[ALL OK]" if ok else "[SOME CHECKS FAILED]")


if __name__ == "__main__":
    _self_test()
