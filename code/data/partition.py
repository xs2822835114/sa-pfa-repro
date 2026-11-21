"""Client partitions and Non-IID construction for UAH-DriveSet (leak-free splits).

Splits are trip-level: neighbouring windows of the same raw signal must not land on both
sides, which random splitting over a 50%-overlapping window pool would allow.

  1. For each (driver, class) the last trip goes to test; classes with a single trip are cut
     chronologically 70/30 with a GAP=2 window buffer (train and test never overlap).
  2. natural      client = driver, using those trip-level splits.
  3. iid          the training pool is split evenly by class over K clients; the test pool is
                  split the same way (identical class distribution per client, so macro-F1 is
                  defined for every client).
  4. dirichlet_aX the training pool is split by class with Dirichlet(alpha) over K clients
                  (severe skew preserved; only clients below MIN_TOTAL are repaired); the test
                  split is the iid one.

Outputs (data/processed/UAH/partitions/):
  partition_<config>.json   per client {"train": [...idx], "test": [...idx]}
  partition_stats.csv       per-configuration, per-client class counts (Non-IID evidence)
  partition_overview.png    stacked class bars for 6 configurations x 6 clients
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
K = 6            # number of clients (UAH: 6 drivers)
SEED = 42
GAP = 2          # windows dropped between train and test when a single-trip class is cut
MIN_TOTAL = 5    # minimum training samples per Dirichlet client (topped up from the largest client)


def trip_pool(y, driver, trip, road, seed: int = SEED, folds: int = 2):
    """Trip-level train/test split (leak-free), returns {fold: (train_idx, test_idx)}.

    The held-out trip is stratified by road type: trips whose road type has a second trip are
    preferred, so the training set still covers that road type. Fold 1 reproduces the earlier
    results exactly (same rng sequence, same draw); fold 2 holds out another candidate trip of
    the same class, to test whether a single held-out trip was a lucky draw. Single-trip classes
    are cut chronologically 70/30 in every fold (there is no second trip to hold out).
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
    """Split idxs evenly by class over k clients."""
    out = [[] for _ in range(k)]
    for c in np.unique(y[idxs]):
        ii = idxs[y[idxs] == c].copy()
        rng.shuffle(ii)
        for j, part in enumerate(np.array_split(ii, k)):
            out[j] += part.tolist()
    return [sorted(o) for o in out]


def split_dirichlet(idxs, y, k: int, alpha: float, rng):
    """Split idxs by class with Dirichlet(alpha) over k clients; only tiny clients are repaired,
    the skew is kept."""
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

    # client configurations: ("natural" | "even" | "dirichlet", alpha)
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
    fig.suptitle(f"UAH client partitions (window L={args.window}s, training-label share, trip-level split)")
    fig.tight_layout()
    fig.savefig(out_dir / "partition_overview.png", dpi=150)

    with open(out_dir / "partition_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(stats_rows)

    # folds 2 and 3 (a different held-out trip), for robustness checks only
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
