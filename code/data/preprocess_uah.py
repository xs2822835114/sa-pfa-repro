"""UAH-DriveSet preprocessing pipeline.

Stages: Raw -> Cleaning -> Resampling -> Normalization -> Windowing -> Label -> Client Split

- Input: data/raw/UAH-DriveSet/UAH-DRIVESET-v1/D*/<trip>/
  - RAW_ACCELEROMETERS.txt (11 columns: t, act, ax, ay, az, ax_kf, ay_kf, az_kf, roll, pitch, yaw)
  - RAW_GPS.txt (12 columns: t, speed, lat, lon, alt, v_acc, h_acc, course, difcourse, s1, s2, s3)
  - folder name: <timestamp>-<mileage>-<D#>-(NORMAL1|NORMAL2|NORMAL|DROWSY|AGGRESSIVE)-<SECONDARY|MOTORWAY>
- Output: data/processed/UAH/
  - windows_L{10,30,60}.npz  normalised (N, L*10, 7) time-series windows + label/driver/trip/road
  - features_L{10,30,60}.npz window statistics (N, 47) for the lightweight models and smoke tests
  - trips_summary.csv / client_label_distribution.csv / scaler_stats.csv

Notes:
- Normalisation is a per-driver z-score (driver = client, an approximation of clients that never
  share raw data; a strict protocol would fit the scaler on the training split only).
- The window label is the trip-level behaviour label (NORMAL/DROWSY/AGGRESSIVE).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = CODE_DIR.parent
RAW_ROOT = RESEARCH_DIR / "data" / "raw" / "UAH-DriveSet" / "UAH-DRIVESET-v1"
OUT_DIR = RESEARCH_DIR / "data" / "processed" / "UAH"

FS = 10.0            # target sampling rate (Hz); the accelerometer is natively ~10 Hz
GAP_SPLIT_S = 1.0    # start a new segment when consecutive samples are further apart (dropped samples)
WINDOW_LENGTHS = [10, 30, 60]   # seconds
LABEL_MAP = {"NORMAL": 0, "DROWSY": 1, "AGGRESSIVE": 2}
LABEL_NAMES = ["Normal", "Drowsy", "Aggressive"]
ACC_COLS = [0, 2, 3, 4, 8, 9, 10]     # t, ax, ay, az, roll, pitch, yaw
GPS_COLS = [0, 1]                     # t, speed
CHANNELS = ["speed", "ax", "ay", "az", "roll", "pitch", "yaw"]


def loadtxt_clean(path: Path, cols) -> np.ndarray:
    arr = np.genfromtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = arr[:, cols]
    arr = arr[np.isfinite(arr[:, 0])]
    # drop duplicate timestamps, keeping increasing time
    keep = np.concatenate([[True], np.diff(arr[:, 0]) > 0])
    return arr[keep]


def split_segments(t: np.ndarray, gap_s: float = GAP_SPLIT_S):
    """Split at time gaps, returns [(i0, i1), ...] (half-open intervals)."""
    if len(t) == 0:
        return []
    breaks = np.where(np.diff(t) > gap_s)[0]
    bounds = [0] + (breaks + 1).tolist() + [len(t)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 1]


def process_trip(trip_dir: Path):
    parts = trip_dir.name.split("-")
    driver, behavior, road = parts[2], parts[3], parts[4]
    label = LABEL_MAP[re.sub(r"\d+$", "", behavior)]

    acc = loadtxt_clean(trip_dir / "RAW_ACCELEROMETERS.txt", ACC_COLS)
    gps = loadtxt_clean(trip_dir / "RAW_GPS.txt", GPS_COLS)
    if len(acc) < 100 or len(gps) < 10:
        return None

    # align both streams to a 10 Hz grid over [max(t0), min(t1)]
    t0 = max(acc[0, 0], gps[0, 0])
    t1 = min(acc[-1, 0], gps[-1, 0])
    if t1 - t0 < 60:
        return None
    grid = np.arange(t0, t1, 1.0 / FS)

    # segment on the raw observations first (a long gap must not be hidden by interpolation)
    segs_acc = split_segments(acc[:, 0])
    segs = []
    for i0, i1 in segs_acc:
        a = acc[i0:i1]
        if len(a) < FS * 10:
            continue
        # grid slice covering the time span of this segment
        m = (grid >= a[0, 0]) & (grid <= a[-1, 0])
        if m.sum() < FS * 10:
            continue
        g = grid[m]
        chans = []
        for c in range(1, acc.shape[1]):
            chans.append(np.interp(g, a[:, 0], a[:, c]))
        sp = np.interp(g, gps[:, 0], gps[:, 1], left=np.nan, right=np.nan)
        ok = np.isfinite(sp)
        if (~ok).all():
            continue
        sp = np.interp(g, g[ok], sp[ok])  # fill beyond the observed endpoints
        X = np.stack([sp] + chans, axis=1)  # (T, 7) = speed, ax, ay, az, roll, pitch, yaw
        segs.append((g, X))
    if not segs:
        return None

    # cleaning: clip to physical ranges, then measure the actual time coverage
    for _, X in segs:
        np.clip(X[:, 0], 0, 300, out=X[:, 0])          # speed <= 300 km/h
        np.clip(X[:, 1:4], -4, 4, out=X[:, 1:4])       # accel <= 4g
        np.clip(X[:, 4:6], -90, 90, out=X[:, 4:6])     # roll/pitch

    total_len = float(np.sum([len(s) for _, s in segs])) / FS
    return dict(driver=driver, behavior=behavior, road=road, label=label, segs=segs, duration_s=total_len)


def normalize_by_driver(trips):
    """Per-driver mean/std (each driver's own data only). Returns {driver: (mean, std)}."""
    stats = {}
    for drv in sorted({t["driver"] for t in trips}):
        allX = np.concatenate([X for t in trips if t["driver"] == drv for _, X in t["segs"]], axis=0)
        mean = allX.mean(axis=0)
        std = allX.std(axis=0)
        std[std < 1e-6] = 1.0
        stats[drv] = (mean, std)
    return stats


def window_features(X: np.ndarray) -> np.ndarray:
    """Window statistics: 7 channels x 6 statistics + 5 speed/acceleration features = 47 dims."""
    feats, names = [], []
    for i, ch in enumerate(CHANNELS):
        v = X[:, i]
        feats += [v.mean(), v.std(), v.min(), v.max(), np.percentile(v, 25), np.percentile(v, 75)]
        names += [f"{ch}_{s}" for s in ["mean", "std", "min", "max", "p25", "p75"]]
    sp = X[:, 0]
    amag = np.linalg.norm(X[:, 1:4], axis=1)
    feats += [np.percentile(sp, 95), float((sp > 50).mean()), amag.mean(), amag.std(), amag.max()]
    names += ["speed_p95", "speed_frac_over50", "accmag_mean", "accmag_std", "accmag_max"]
    return np.asarray(feats, dtype=np.float32), names


def main():
    trips = []
    for d in sorted(RAW_ROOT.glob("D*")):
        for trip_dir in sorted([p for p in d.iterdir() if p.is_dir()]):
            t = process_trip(trip_dir)
            if t is None:
                print(f"[skip] {trip_dir.name}")
                continue
            t["trip"] = trip_dir.name
            trips.append(t)
    print(f"valid trips: {len(trips)} / drivers: {sorted({t['driver'] for t in trips})}")

    stats = normalize_by_driver(trips)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "scaler_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["driver", "channel", "mean", "std"])
        for drv, (mean, std) in stats.items():
            for i, ch in enumerate(CHANNELS):
                w.writerow([drv, ch, f"{mean[i]:.6f}", f"{std[i]:.6f}"])

    # window collection
    store = {L: dict(X=[], y=[], feats=[], driver=[], trip=[], road=[], t0=[]) for L in WINDOW_LENGTHS}
    feature_names = None
    trip_rows = []
    for t in trips:
        mean, std = stats[t["driver"]]
        n_win = {L: 0 for L in WINDOW_LENGTHS}
        for g, X in t["segs"]:
            X = (X - mean) / std
            for L in WINDOW_LENGTHS:
                W = int(L * FS)
                stride = W // 2
                for s in range(0, len(X) - W + 1, stride):
                    win = X[s:s + W]
                    feat, feature_names = window_features(win)
                    store[L]["X"].append(win.astype(np.float32))
                    store[L]["y"].append(t["label"])
                    store[L]["feats"].append(feat)
                    store[L]["driver"].append(t["driver"])
                    store[L]["trip"].append(t["trip"])
                    store[L]["road"].append(t["road"])
                    store[L]["t0"].append(float(g[s]))
                    n_win[L] += 1
        trip_rows.append([t["driver"], t["trip"], t["road"], LABEL_NAMES[t["label"]],
                          f"{t['duration_s']:.1f}", n_win[10], n_win[30], n_win[60]])

    with open(OUT_DIR / "trips_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["driver", "trip", "road", "label", "duration_s", "n_win_L10", "n_win_L30", "n_win_L60"])
        w.writerows(trip_rows)

    for L in WINDOW_LENGTHS:
        d = store[L]
        np.savez_compressed(
            OUT_DIR / f"windows_L{L}.npz",
            X=np.stack(d["X"]), y=np.asarray(d["y"], dtype=np.int64),
            driver=np.asarray(d["driver"]), trip=np.asarray(d["trip"]),
            road=np.asarray(d["road"]), t0=np.asarray(d["t0"]),
        )
        np.savez_compressed(
            OUT_DIR / f"features_L{L}.npz",
            X=np.stack(d["feats"]), y=np.asarray(d["y"], dtype=np.int64),
            feature_names=np.asarray(feature_names),
            driver=np.asarray(d["driver"]), trip=np.asarray(d["trip"]), road=np.asarray(d["road"]),
        )
        n = len(d["y"])
        cls = np.bincount(np.asarray(d["y"]), minlength=3)
        print(f"L={L:>2}s windows: {n:5d}  class(N/D/A)={cls.tolist()}  saved windows/features npz")

    # driver x class window distribution (L=30), evidence of the natural Non-IID split
    d = store[30]
    drv_arr = np.asarray(d["driver"]); y_arr = np.asarray(d["y"])
    with open(OUT_DIR / "client_label_distribution.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["driver", "n_windows_L30", "Normal", "Drowsy", "Aggressive"])
        for drv in sorted(set(drv_arr)):
            m = drv_arr == drv
            w.writerow([drv, int(m.sum())] + [int((y_arr[m] == k).sum()) for k in range(3)])
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
