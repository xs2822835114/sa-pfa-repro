"""第 5 周 · T022：Sparse Client（部分参与）调度生成。

两种模式：
  uniform      每轮从 K 个客户端中随机抽 ceil(rate*K) 个参与（无放回）
  availability 每个客户端有自身"在线概率" p_i（默认 p_i = rate，含 1 个总是在线、
               1 个低活跃客户端），每轮按 Bernoulli(p_i) 采样，至少保留 1 个
                —— 模拟真实车队里"部分车辆从不掉线、部分车辆很少上线"的场景

产出（research/data/processed/UAH/sparse/）：
  schedule_{mode}_r{rate}.json   每轮参与客户端列表
  sparse_summary.csv             每配置：平均参与数 / 最低覆盖客户端 / 覆盖率
  schedule_heatmap.png           uniform 60% 与 availability 60% 的参与矩阵示意
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
    """真实感更强：客户端在线概率不均（高活跃 x2，低活跃 x0.5，严格范围裁剪）。"""
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
        ax.set_title(f"{mode} 60% 参与矩阵（行=客户端, 列=轮次）")
    axes[-1].set_xlabel("round")
    fig.tight_layout()
    fig.savefig(out_dir / "schedule_heatmap.png", dpi=150)
    print("sparse schedules saved ->", out_dir)
    for r in rows[1:]:
        print("  ", r)


if __name__ == "__main__":
    main()
