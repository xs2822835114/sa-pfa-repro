"""第 5 周 · T018/T019/T020：UAH-DriveSet 数据管线。

流程（与 18 周计划一致）：
    Raw -> Cleaning -> Resampling -> Normalization -> Windowing -> Label -> Client Split

- 输入：research/data/raw/UAH-DriveSet/UAH-DRIVESET-v1/D*/<trip>/
  - RAW_ACCELEROMETERS.txt（11 列：t, act, ax, ay, az, ax_kf, ay_kf, az_kf, roll, pitch, yaw）
  - RAW_GPS.txt（12 列：t, speed, lat, lon, alt, v_acc, h_acc, course, difcourse, s1, s2, s3）
  - 文件夹名：<时间戳>-<里程>-<D#>-(NORMAL1|NORMAL2|NORMAL|DROWSY|AGGRESSIVE)-<SECONDARY|MOTORWAY>
- 输出：research/data/processed/UAH/
  - windows_L{10,30,60}.npz  归一化后的 (N, L*10, 7) 时序窗口 + 标签/司机/行程/道路
  - features_L{10,30,60}.npz 窗口统计特征 (N, 47)，供轻量模型与 FedAvg 冒烟实验使用
  - trips_summary.csv / client_label_distribution.csv / scaler_stats.csv

说明：
- 归一化用「每位司机各自」的 z-score（司机=客户端，互不共享原始数据的场景近似；
  严格协议应在训练集上拟合 scaler，此处记录，正式实验（第 13 周）切换）。
- 每窗口的标签 = 行程级行为标签（NORMAL/DROWSY/AGGRESSIVE）。
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

FS = 10.0            # 目标采样率 (Hz)，加速度计原生 ~10Hz
GAP_SPLIT_S = 1.0    # 相邻样本间隔超过该值则切段（原始数据丢包/断采）
WINDOW_LENGTHS = [10, 30, 60]   # 秒
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
    # 时间递增去重
    keep = np.concatenate([[True], np.diff(arr[:, 0]) > 0])
    return arr[keep]


def split_segments(t: np.ndarray, gap_s: float = GAP_SPLIT_S):
    """按时间间断分段，返回 [(i0, i1), ...]（闭开区间）。"""
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

    # 时间对齐到 [max(t0), min(t1)] 的 10Hz 网格
    t0 = max(acc[0, 0], gps[0, 0])
    t1 = min(acc[-1, 0], gps[-1, 0])
    if t1 - t0 < 60:
        return None
    grid = np.arange(t0, t1, 1.0 / FS)

    # 先用"原始观测"切段（防止长间断被插值掩盖）
    segs_acc = split_segments(acc[:, 0])
    segs = []
    for i0, i1 in segs_acc:
        a = acc[i0:i1]
        if len(a) < FS * 10:
            continue
        # 该段覆盖的时间范围对应的网格切片
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
        sp = np.interp(g, g[ok], sp[ok])  # 端点外推填充
        X = np.stack([sp] + chans, axis=1)  # (T, 7) = speed, ax, ay, az, roll, pitch, yaw
        segs.append((g, X))
    if not segs:
        return None

    # 清洗：幅度裁剪（物理范围），再统计原始时间覆盖
    for _, X in segs:
        np.clip(X[:, 0], 0, 300, out=X[:, 0])          # speed <= 300 km/h
        np.clip(X[:, 1:4], -4, 4, out=X[:, 1:4])       # accel <= 4g
        np.clip(X[:, 4:6], -90, 90, out=X[:, 4:6])     # roll/pitch

    total_len = float(np.sum([len(s) for _, s in segs])) / FS
    return dict(driver=driver, behavior=behavior, road=road, label=label, segs=segs, duration_s=total_len)


def normalize_by_driver(trips):
    """按司机统计 mean/std（仅用该司机自己的数据）。返回 {driver: (mean, std)}。"""
    stats = {}
    for drv in sorted({t["driver"] for t in trips}):
        allX = np.concatenate([X for t in trips if t["driver"] == drv for _, X in t["segs"]], axis=0)
        mean = allX.mean(axis=0)
        std = allX.std(axis=0)
        std[std < 1e-6] = 1.0
        stats[drv] = (mean, std)
    return stats


def window_features(X: np.ndarray) -> np.ndarray:
    """窗口统计特征：7 通道 x 6 统计量 + 5 个速度/合加速度特征 = 47 维。"""
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

    # 窗口收集
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

    # 司机 x 类别窗口分布（L=30，作为"天然 Non-IID"证据）
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
