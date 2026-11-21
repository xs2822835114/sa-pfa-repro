"""第 5 周 · T020/T021：客户端划分与 Non-IID 构造（第四轮修订：无泄漏划分）。

第四轮修订（外部评审）：原实现在 50% 重叠窗口池上随机分 train/test，同一段原始
信号的相邻窗口会跨集，构成行程级泄漏。现改为**行程级**划分，再做客户端划分：

  1. 每个 (司机, 类别) 的最后一个行程 → test；只有一个行程的类别按时序 70/30 切，
     中间留 GAP=2 个窗口（同司机同类别内 train/test 不重叠）。
  2. natural     客户端 = 司机，直接使用上面的 train/test。
  3. iid         训练池按类别均匀分给 K 个客户端；测试池按类别均匀分给 K 个
                 客户端（各客户端测试分布一致，macro-F1 对每个客户端都有定义）。
  4. dirichlet_aX 训练池按类别用 Dirichlet(α) 分给 K 个客户端（保留严酷偏斜，
                 仅修补 < min_total 的客户端）；测试划分同 iid。

产出（research/data/processed/UAH/partitions/）：
  partition_<config>.json   每客户端 {"train": [...idx], "test": [...idx]}
  partition_stats.csv       每配置每客户端的类别计数（Non-IID 证据表）
  partition_overview.png    6 个配置 x 6 客户端的类别堆叠柱状图
"""
from __future__ import annotations

import argparse
import csv
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

LABEL_NAMES = ["Normal", "Drowsy", "Aggressive"]
K = 6            # 客户端数量（UAH: 6 名司机）
SEED = 42
GAP = 2          # 单行程切分时丢弃的窗口数（防止相邻窗口跨集）
MIN_TOTAL = 5    # Dirichlet 客户端的最少训练样本（不足则从最大客户端转移）


def trip_pool(y, driver, trip, road, seed: int = SEED, folds: int = 2):
    """行程级 train/test（无泄漏），返回 {fold: (train_idx, test_idx)}。

    留一行程时按道路类型分层：优先留出"同类型还有第二个行程"的行程，使训练集仍覆盖
    该道路类型。fold 1 与旧结果完全一致（同一个 rng 序列、同一个选择）；fold 2 留出
    同类别中另一个候选行程（两行程类别即另一个行程），用于检验单次留出的偶然性。
    单行程类别按时序 70/30 切，各 fold 相同（该类别没有第二个行程可留）。
    """
    rng = np.random.default_rng(seed)
    tr = {k: [] for k in range(1, folds + 1)}
    te = {k: [] for k in range(1, folds + 1)}
    for d in sorted(set(driver.tolist())):
        for c in np.unique(y):
            idx = np.where((driver == d) & (y == c))[0]
            if len(idx) == 0:
                continue
            trips = sorted(set(trip[idx].tolist()))
            if len(trips) >= 2:
                roads = {t: str(road[idx[trip[idx] == t][0]]) for t in trips}
                cnt = {r: sum(1 for t in trips if roads[t] == r) for r in set(roads.values())}
                cands = [t for t in trips if cnt[roads[t]] >= 2]
                pool = sorted(cands) if cands else trips
                held1 = str(rng.choice(pool))
                order = [held1] + [t for t in pool if t != held1] + \
                        [t for t in trips if t not in pool and t != held1]
                for k in range(1, folds + 1):
                    held = order[(k - 1) % len(order)]
                    te[k] += idx[trip[idx] == held].tolist()
                    tr[k] += idx[trip[idx] != held].tolist()
            else:
                n = len(idx)
                cut = int(round(0.7 * n))
                for k in range(1, folds + 1):
                    tr[k] += idx[:cut].tolist()
                    if cut + GAP < n:
                        te[k] += idx[cut + GAP:].tolist()
    return {k: (sorted(tr[k]), sorted(te[k])) for k in range(1, folds + 1)}


def split_even(idxs, y, k: int, rng):
    """按类别把 idxs 均匀分给 k 个客户端。"""
    out = [[] for _ in range(k)]
    for c in np.unique(y[idxs]):
        ii = idxs[y[idxs] == c].copy()
        rng.shuffle(ii)
        for j, part in enumerate(np.array_split(ii, k)):
            out[j] += part.tolist()
    return [sorted(o) for o in out]


def split_dirichlet(idxs, y, k: int, alpha: float, rng):
    """按类别用 Dirichlet(α) 分给 k 个客户端；只修补过小客户端，保留偏斜。"""
    out = [[] for _ in range(k)]
    for c in np.unique(y[idxs]):
        ii = idxs[y[idxs] == c].copy()
        rng.shuffle(ii)
        props = rng.dirichlet([alpha] * k)
        cuts = np.floor(np.cumsum(props) * len(ii)).astype(int)
        cuts[-1] = len(ii)
        start = 0
        for j in range(k):
            out[j] += ii[start:cuts[j]].tolist()
            start = cuts[j]
    for j in range(k):
        while len(out[j]) < MIN_TOTAL:
            donor = max(range(k), key=lambda q: len(out[q]))
            out[j].append(out[donor].pop())
    return [sorted(o) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=30, choices=[10, 30, 60])
    args = ap.parse_args()

    data = np.load(RESEARCH_DIR / "data" / "processed" / "UAH" / f"features_L{args.window}.npz",
                   allow_pickle=True)
    y, driver, trip = data["y"], data["driver"], data["trip"]
    folds = trip_pool(y, driver, trip, data["road"], folds=3)
    tr_pool, te_pool = folds[1]
    assert not (set(tr_pool) & set(te_pool))
    tr_arr, te_arr = np.asarray(tr_pool), np.asarray(te_pool)

    # 客户端方案：("natural" | "even" | "dirichlet", alpha)
    configs = {
        "natural": ("natural", None),
        "iid": ("even", None),
        "dirichlet_a1.0": ("dirichlet", 1.0),
        "dirichlet_a0.5": ("dirichlet", 0.5),
        "dirichlet_a0.1": ("dirichlet", 0.1),
        "dirichlet_a0.05": ("dirichlet", 0.05),
    }
    out_dir = RESEARCH_DIR / "data" / "processed" / "UAH" / "partitions"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = [["config", "client", "n_total", "n_train", "n_test"] + LABEL_NAMES]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = ["#4C9F70", "#E1A730", "#C2453A"]
    for ax, (name, (kind, alpha)) in zip(axes.ravel(), configs.items()):
        if kind == "natural":
            trainers = [sorted(i for i in tr_pool if driver[i] == d)
                        for d in sorted(set(driver.tolist()))]
            testers = [sorted(i for i in te_pool if driver[i] == d)
                       for d in sorted(set(driver.tolist()))]
        else:
            rng = np.random.default_rng(SEED)
            trainers = split_even(tr_arr, y, K, rng) if kind == "even" else \
                split_dirichlet(tr_arr, y, K, alpha, rng)
            rng_t = np.random.default_rng(SEED + 1)
            testers = split_even(te_arr, y, K, rng_t)

        payload = {}
        for j, (tr, te) in enumerate(zip(trainers, testers)):
            assert not (set(tr) & set(te)), f"{name} client_{j} train/test overlap"
            payload[f"client_{j}"] = {"train": tr, "test": te}
            idxs = tr + te
            counts = [int((y[idxs] == c).sum()) for c in range(3)]
            stats_rows.append([name, j, len(idxs), len(tr), len(te)] + counts)
        with open(out_dir / f"partition_{name}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

        bottoms = np.zeros(len(trainers))
        for c in range(3):
            vals = np.array([(y[np.asarray(t)] == c).sum() for t in trainers], dtype=float)
            frac = np.divide(vals, [max(len(t), 1) for t in trainers])
            ax.bar(range(len(trainers)), frac, bottom=bottoms, color=colors[c], label=LABEL_NAMES[c])
            bottoms += frac
        ax.set_title(f"{name}  (n_train={sum(len(t) for t in trainers)})")
        ax.set_xticks(range(len(trainers)))
        ax.set_xticklabels([f"c{j}" for j in range(len(trainers))])
        ax.set_ylim(0, 1)
        if ax is axes.ravel()[0]:
            ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(f"UAH 客户端划分对比（窗口 L={args.window}s，训练集标签比例，行程级无泄漏）")
    fig.tight_layout()
    fig.savefig(out_dir / "partition_overview.png", dpi=150)

    with open(out_dir / "partition_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(stats_rows)

    # 第二、三折（另一个/第三个留出行程），仅用于稳健性检验
    drivers = sorted(set(driver.tolist()))
    for k in (2, 3):
        trk, tek = folds[k]
        payload_k = {}
        for j, d in enumerate(drivers):
            payload_k[f"client_{j}"] = {
                "train": sorted(i for i in trk if driver[i] == d),
                "test": sorted(i for i in tek if driver[i] == d),
            }
        with open(out_dir / f"partition_natural_fold{k}.json", "w", encoding="utf-8") as f:
            json.dump(payload_k, f)
        print(f"  natural_fold{k}       train sizes: "
              f"{[len(v['train']) for v in payload_k.values()]}")

    print("partition configs saved:", ", ".join(configs))
    for name in configs:
        rows = [r for r in stats_rows[1:] if r[0] == name]
        sizes = [r[3] for r in rows]
        print(f"  {name:18s} train sizes: {sizes}")


if __name__ == "__main__":
    main()
