#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Client-granularity extrapolation partitions (semi-synthetic).

No new data is collected: starting from the natural (per-driver) partition, each driver's trips
are split round-robin into m parts, giving 6m logical clients. The physical data is unchanged;
only the *client granularity* changes.

This has a counterpart in connected-vehicle fleets: one vehicle is rarely one driver, the same
driver may rotate across vehicles and one vehicle may be shared, so "one vehicle per client" is
already finer-grained (and each client holds less data) than "one driver per client".

Scope for the paper: the number of independent drivers does not increase, so these runs cannot
support claims of better cross-driver generalisation; they test whether the conclusions hold when
the client granularity changes.

Outputs:
  data/processed/UAH/partitions/partition_natural_k12.json
  data/processed/UAH/partitions/partition_natural_k18.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
FEAT = R / "data" / "processed" / "UAH" / "features_L30.npz"
SRC = R / "data" / "processed" / "UAH" / "partitions" / "partition_natural.json"
OUTDIR = SRC.parent

feat = np.load(FEAT, allow_pickle=True)
trip = feat["trip"]
y = feat["y"]
src = json.loads(SRC.read_text(encoding="utf-8"))
drivers = sorted(src.keys(), key=lambda s: int(s.split("_")[1]))

for m in (2, 3):
    out = {}
    for c in drivers:
        d = int(c.split("_")[1])
        for split in ("train", "test"):
            idx = np.asarray(src[c][split], dtype=int)
            lab = y[idx]
            rng = np.random.default_rng(1000 * m + d + (0 if split == "train" else 77))
            # stratified by class: the windows of each class are spread evenly over the m
            # sub-clients so that all 3 classes are covered (splitting by trip would produce
            # many single-class clients, making training and F1 incomparable)
            for gi in range(m):
                sel = []
                for k in np.unique(lab):
                    pos = np.where(lab == k)[0]
                    rng.shuffle(pos)
                    sel.append(idx[pos[gi::m]])
                take = np.sort(np.concatenate(sel)) if sel else np.array([], dtype=int)
                out.setdefault(f"client_{d * m + gi}", {"train": [], "test": []})
                out[f"client_{d * m + gi}"][split] = take.tolist()

    # renumber clients and check coverage
    out = {f"client_{i}": out[k] for i, k in enumerate(
        sorted(out, key=lambda s: int(s.split("_")[1])))}
    dst = OUTDIR / f"partition_natural_k{6 * m}.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"\n=== K={len(out)}  ->  {dst.name} ===")
    bad = 0
    for k, v in out.items():
        tr = np.asarray(v["train"], dtype=int)
        cls = np.bincount(y[tr], minlength=3)
        cover = int((cls > 0).sum())
        if cover < 3:
            bad += 1
        print(f"  {k}: train={len(tr):3d} test={len(v['test']):3d} "
              f"classes={cls.tolist()} cover={cover}/3")
    print(f"  clients covering fewer than 3 classes: {bad}/{len(out)}")
