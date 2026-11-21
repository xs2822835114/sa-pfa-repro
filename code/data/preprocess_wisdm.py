"""WISDM 1.1 preprocessing: phone accelerometer, 6 coarse activities, block-level leak-free split.

Data: data/raw/WISDM/wisdm-dataset.zip (phone accelerometer files)
Format: subject,activity,timestamp_ns,x,y,z;
Activity codes: A walk, B jog, C stairs up, D stairs down, E sit, F stand (the other 12 classes
are discarded). Sampling 15 Hz, 3 s windows (45 points) with 50% overlap; 6 statistics per axis
plus 2 for the acceleration magnitude = 20 dims.

Splits:
  natural         for each (subject, class) the contiguous segment is cut 70/30 in time, with
                  2 windows dropped as a gap; client = subject (51 subjects).
  dirichlet_a0.05 training windows are redistributed by class with Dirichlet(alpha) over the 51
                  clients (skew preserved); the test split stays per subject, so every client
                  covers all 6 classes.

Outputs:
  data/processed/WISDM/features_wisdm.npz           X, y, subject, trip(segment id)
  data/processed/WISDM/partitions/partition_wisdm_natural.json
  data/processed/WISDM/partitions/partition_wisdm_dirichlet_a0.05.json
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2]
ZIP = R / "data" / "raw" / "WISDM" / "wisdm-dataset.zip"
OUT = R / "data" / "processed" / "WISDM"
LABELS = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
W, STRIDE, GAP = 45, 22, 2


def window_feats(w: np.ndarray) -> np.ndarray:
    """w: (W,3) -> 20 statistics."""
    f = []
    for j in range(3):
        x = w[:, j]
        f += [x.mean(), x.std(), x.min(), x.max(), np.percentile(x, 25), np.percentile(x, 75)]
    m = np.linalg.norm(w, axis=1)
    f += [m.mean(), m.std()]
    return np.asarray(f, dtype=np.float32)


def parse_file(zf, name):
    xs, ys, segs = [], [], []
    seg_id = 0
    cur_lab, cur = None, []

    def flush():
        nonlocal cur, seg_id
        if cur and len(cur) >= W:
            arr = np.asarray(cur, dtype=np.float32)
            for s in range(0, len(arr) - W + 1, STRIDE):
                xs.append(window_feats(arr[s:s + W]))
                ys.append(LABELS[cur_lab])
                segs.append(seg_id)
        cur = []
        seg_id += 1

    with zf.open(name) as f:
        for line in f:
            s = line.decode().strip().rstrip(";")
            if not s:
                continue
            parts = s.split(",")
            if len(parts) < 6:
                continue
            lab = parts[1]
            if lab not in LABELS:
                if cur:
                    flush()
                    cur_lab = None
                continue
            if lab != cur_lab:
                if cur:
                    flush()
                cur_lab = lab
            cur.append((float(parts[3]), float(parts[4]), float(parts[5])))
    if cur:
        flush()
    return xs, ys, segs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(ZIP)
    files = [n for n in zf.namelist() if "/phone/accel/" in n and n.endswith(".txt")]
    files.sort(key=lambda s: int(re.search(r"data_(\d+)_", s).group(1)))

    X, y, subj, trip = [], [], [], []
    for k, name in enumerate(files):
        sid = int(re.search(r"data_(\d+)_", name).group(1))
        xs, ys, segs = parse_file(zf, name)
        X += xs
        y += ys
        subj += [sid] * len(xs)
        trip += [(sid, sg) for sg in segs]
        print(f"  [{k+1}/{len(files)}] subject {sid}: {len(xs)} windows", flush=True)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    subject = np.asarray(subj, dtype=np.int64)
    trip_ids = {t: i for i, t in enumerate(sorted(set(trip)))}
    tripid = np.asarray([trip_ids[t] for t in trip], dtype=np.int64)
    np.savez_compressed(OUT / "features_wisdm.npz", X=X, y=y, subject=subject, trip=tripid)
    print(f"saved X={X.shape} classes={np.bincount(y)} subjects={len(set(subject))}")

    # natural: per (subject, class) chronological 70/30 with a 2-window gap
    subs = sorted(set(subject.tolist()))
    part = {}
    for j, s in enumerate(subs):
        tr, te = [], []
        for c in range(6):
            idx = np.where((subject == s) & (y == c))[0]
            if len(idx) == 0:
                continue
            n = len(idx)
            cut = int(round(0.7 * n))
            tr += idx[:cut].tolist()
            if cut + GAP < n:
                te += idx[cut + GAP:].tolist()
        part[f"client_{j}"] = {"train": sorted(tr), "test": sorted(te)}

    pdir = OUT / "partitions"
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "partition_wisdm_natural.json").write_text(json.dumps(part), encoding="utf-8")
    sizes = [(len(v["train"]), len(v["test"])) for v in part.values()]
    print(f"natural: {len(part)} clients, train={sum(a for a,_ in sizes)} "
          f"test={sum(b for _,b in sizes)} min={min(min(a,b) for a,b in sizes)}")

    # Dirichlet: only the training windows are redistributed, the test split stays per subject
    tr_pool = np.concatenate([np.asarray(v["train"], dtype=int) for v in part.values()])
    rng = np.random.default_rng(42)
    alloc = [[] for _ in subs]
    for c in range(6):
        ii = tr_pool[y[tr_pool] == c].copy()
        rng.shuffle(ii)
        props = rng.dirichlet([0.05] * len(subs))
        cuts = np.floor(np.cumsum(props) * len(ii)).astype(int)
        cuts[-1] = len(ii)
        st = 0
        for j in range(len(subs)):
            alloc[j] += ii[st:cuts[j]].tolist()
            st = cuts[j]
    for j in range(len(subs)):
        while len(alloc[j]) < 5:
            donor = max(range(len(subs)), key=lambda q: len(alloc[q]))
            alloc[j].append(alloc[donor].pop())

    part_d = {}
    for j, cl in enumerate(part):
        part_d[cl] = {"train": sorted(alloc[j]), "test": part[cl]["test"]}
    (pdir / "partition_wisdm_dirichlet_a0.05.json").write_text(json.dumps(part_d), encoding="utf-8")
    tr_sizes = [len(v["train"]) for v in part_d.values()]
    print(f"dirichlet_a0.05: train sizes min={min(tr_sizes)} max={max(tr_sizes)}")


if __name__ == "__main__":
    main()
