"""Leak-free UCI HAR partitions: time-block natural split plus Dirichlet variants.

Rules:
  consecutive windows of the same (subject, label) form one time block (file order);
  block length n >= 10: first 70% train, 2 windows dropped as a gap, the rest test;
  block length n < 10: block dropped (too short to provide both train and test windows with a gap).

Dirichlet variants: the training pool is redistributed by class with Dirichlet(alpha) over the 30
clients (only tiny clients are repaired); the test split is the per-subject natural_block one, so
every client covers all 6 classes.

Outputs (data/processed/HAR/partitions/):
  partition_har_natural_block.json (also used to verify/rebuild the split)
  partition_har_dirichlet_a0.1.json
  partition_har_dirichlet_a0.05.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
DATA = R / "data" / "processed" / "HAR"
OUT = DATA / "partitions"
GAP = 2
MIN_TOTAL = 10


def blocks(subject, y):
    """Return [(subject, label, [idx...]), ...] in file order."""
    out, start = [], 0
    for i in range(1, len(y) + 1):
        if i == len(y) or subject[i] != subject[start] or y[i] != y[start]:
            out.append((str(subject[start]), int(y[start]), list(range(start, i))))
            start = i
    return out


def natural(subject, y):
    tr, te, dropped = [], [], 0
    for _, _, idx in blocks(subject, y):
        n = len(idx)
        if n >= 10:
            cut = int(round(0.7 * n))
            tr += idx[:cut]
            te += idx[cut + GAP:]
        else:
            dropped += n
    return tr, te, dropped


def split_dirichlet(idxs, y, k, alpha, rng):
    out = [[] for _ in range(k)]
    for c in np.unique(y[idxs]):
        ii = np.asarray(idxs)[y[idxs] == c].copy()
        rng.shuffle(ii)
        props = rng.dirichlet([alpha] * k)
        cuts = np.floor(np.cumsum(props) * len(ii)).astype(int)
        cuts[-1] = len(ii)
        s = 0
        for j in range(k):
            out[j] += ii[s:cuts[j]].tolist()
            s = cuts[j]
    for j in range(k):
        while len(out[j]) < MIN_TOTAL:
            donor = max(range(k), key=lambda q: len(out[q]))
            out[j].append(out[donor].pop())
    return [sorted(o) for o in out]


def main():
    d = np.load(DATA / "features_har.npz", allow_pickle=True)
    X, y, subject = d["X"], d["y"], d["subject"]
    tr_pool, te_pool, dropped = natural(subject, y)
    assert not (set(tr_pool) & set(te_pool))
    subs = sorted(set(subject.tolist()))
    print(f"windows={len(y)} train={len(tr_pool)} test={len(te_pool)} "
          f"dropped_short_blocks={dropped} subjects={len(subs)}")

    # natural: client = subject
    nat_tr = [sorted(i for i in tr_pool if subject[i] == s) for s in subs]
    nat_te = [sorted(i for i in te_pool if subject[i] == s) for s in subs]
    write("natural_block", nat_tr, nat_te, y)

    tr_arr = np.asarray(tr_pool)
    for a in (0.1, 0.05):
        rng = np.random.default_rng(42)
        tr_cl = split_dirichlet(tr_arr, y, len(subs), a, rng)
        write(f"dirichlet_a{a}", tr_cl, nat_te, y)


def write(name, tr_cl, te_cl, y):
    payload = {f"client_{j}": {"train": list(a), "test": list(b)}
               for j, (a, b) in enumerate(zip(tr_cl, te_cl))}
    out = OUT / f"partition_har_{name}.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    sizes = [(len(a), len(b)) for a, b in zip(tr_cl, te_cl)]
    print(f"{name:24s} train={sum(x for x,_ in sizes)} test={sum(z for _,z in sizes)} "
          f"min_train={min(x for x,_ in sizes)} min_test={min(z for _,z in sizes)}")


if __name__ == "__main__":
    main()
