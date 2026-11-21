"""Sparse-client (partial participation) schedule generation.

Two modes:
  uniform      each round draws ceil(rate*K) of the K clients without replacement
  availability each client has its own online probability p_i (default p_i = rate, with one
               nearly always online and one rarely), sampled as Bernoulli(p_i) per round with at
               least one client kept: some vehicles never drop out, others rarely join

Outputs (data/processed/UAH/sparse/):
  schedule_{mode}_r{rate}.json   participating clients per round
  sparse_summary.csv             per configuration: mean participants / min and max client
                                 coverage / number of all-down rounds
  schedule_heatmap.png           participation matrix for uniform and availability at 60%
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = CODE_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str(RESEARCH_DIR / "logs" / "mplcache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Songti SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

K = 6
ROUNDS = 30
RATES = [1.0, 0.8, 0.6, 0.4, 0.2]
SEED = 7


def uniform_schedule(rate: float, rounds: int, seed: int):
    rng = np.random.default_rng(seed)
    m = max(1, int(np.ceil(rate * K)))
    return {str(t): sorted(rng.choice(K, size=m, replace=False).tolist()) for t in range(rounds)}


def availability_schedule(rate: float, rounds: int, seed: int):
    """More realistic: uneven online probabilities (high activity x2, low activity x0.5, clipped)."""
    rng = np.random.default_rng(seed)
    pattern = np.array([1.0, 1.4, 0.4, 1.2, 0.7, 0.3])
    p = np.clip(pattern * rate, 0.05, 1.0)
    out = {}
    for t in range(rounds):
        part = np.where(rng.random(K) < p)[0]
        if len(part) == 0:
            part = np.array([int(rng.choice(K))])
        out[str(t)] = sorted(part.tolist())
    return out


def main():
    out_dir = RESEARCH_DIR / "data" / "processed" / "UAH" / "sparse"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [["mode", "rate", "rounds", "avg_participants", "min_client_coverage", "max_client_coverage", "rounds_all_down"]]
    examples = {}
    for mode, fn in [("uniform", uniform_schedule), ("availability", availability_schedule)]:
        for rate in RATES:
            sched = fn(rate, ROUNDS, SEED)
            with open(out_dir / f"schedule_{mode}_r{int(rate*100)}.json", "w", encoding="utf-8") as f:
                json.dump({"mode": mode, "rate": rate, "rounds": ROUNDS, "seed": SEED, "participants": sched}, f)
            counts = np.zeros(K, dtype=int)
            avg = []
            for t, part in sched.items():
                counts[part] += 1
                avg.append(len(part))
            rows.append([mode, rate, ROUNDS, f"{np.mean(avg):.2f}", int(counts.min()), int(counts.max()),
                         int((np.array(avg) == 0).sum())])
            if rate == 0.6:
                examples[mode] = sched

    with open(out_dir / "sparse_summary.csv", "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        _csv.writer(f).writerows(rows)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for ax, (mode, sched) in zip(axes, examples.items()):
        mat = np.zeros((ROUNDS, K))
        for t, part in sched.items():
            mat[int(t), part] = 1
        ax.imshow(mat.T, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"c{k}" for k in range(K)])
        ax.set_title(f"{mode} 60% participation matrix (rows=clients, columns=rounds)")
    axes[-1].set_xlabel("round")
    fig.tight_layout()
    fig.savefig(out_dir / "schedule_heatmap.png", dpi=150)
    print("sparse schedules saved ->", out_dir)
    for r in rows[1:]:
        print("  ", r)


if __name__ == "__main__":
    main()
