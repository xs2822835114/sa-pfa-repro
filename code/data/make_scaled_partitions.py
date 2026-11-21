#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
W25：客户端规模外推的划分生成（半合成）

思路：不做新的数据采集，而是在 natural（按驾驶员）划分的基础上，把每个驾驶员的
行程按轮转方式拆成 m 份，得到 6m 个逻辑客户端。物理数据不变，改变的是**客户端粒度**。

这在车联网场景中有现实对应：真实车队里"一台车"通常不是"一个驾驶员"，同一司机
可能轮换多台车、同一台车也可能多人驾驶，因此"一车一客户端"本身就比"一人一客户端"
粒度更细、单客户端数据更少。

严格边界（须写进论文）：本实验**不增加独立驾驶员的数目**，因此不能用于主张跨驾驶员
泛化能力的提升；它检验的是结论在客户端粒度变化时是否稳健。

产出：
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
            # 按类别分层：每一类的窗口均匀分给 m 个子客户端，保证 3 类覆盖
            # （按行程切分会造成大量单类客户端，训练与 F1 均不可比）
            for gi in range(m):
                sel = []
                for k in np.unique(lab):
                    pos = np.where(lab == k)[0]
                    rng.shuffle(pos)
                    sel.append(idx[pos[gi::m]])
                take = np.sort(np.concatenate(sel)) if sel else np.array([], dtype=int)
                out.setdefault(f"client_{d * m + gi}", {"train": [], "test": []})
                out[f"client_{d * m + gi}"][split] = take.tolist()

    # 规整编号并检查
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
              f"类别={cls.tolist()} 覆盖={cover}/3")
    print(f"  类别覆盖不足 3 类的客户端: {bad}/{len(out)}")
