# -*- coding: utf-8 -*-
"""Collect every archived run into results/summary.csv.

Each run directory is expected to contain config.json and final_metrics.json,
as produced by code/federated/runner.py.  Global metrics describe the shared
model; personalised metrics describe each client's own head (FedPer-style
methods only).  Worst-client values are the minimum over clients within a
seed; the paper averages that minimum over seeds.
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "baselines"
OUT = ROOT / "results" / "summary.csv"

COLUMNS = [
    "exp_id", "dataset", "algo", "partition", "rate", "seed", "n_clients", "rounds",
    "mean_global_f1", "worst_global_f1", "mean_pers_f1", "worst_pers_f1",
]


def read_run(d: Path):
    cfg_p, met_p = d / "config.json", d / "final_metrics.json"
    if not (cfg_p.exists() and met_p.exists()):
        return None
    cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
    met = json.loads(met_p.read_text(encoding="utf-8"))
    row = {
        "exp_id": d.name,
        "dataset": cfg.get("dataset", ""),
        "algo": cfg.get("algo", ""),
        "partition": cfg.get("partition", ""),
        "rate": cfg.get("rate", ""),
        "seed": cfg.get("seed", ""),
        "n_clients": cfg.get("n_clients", ""),
        "rounds": cfg.get("rounds", ""),
    }
    g = met.get("per_client_global_f1")
    if g:
        vals = list(g.values())
        row["mean_global_f1"] = round(st.mean(vals), 6)
        row["worst_global_f1"] = round(min(vals), 6)
    p = met.get("per_client_pers_f1")
    if p:
        vals = list(p.values())
        row["mean_pers_f1"] = round(st.mean(vals), 6)
        row["worst_pers_f1"] = round(min(vals), 6)
    return row


def main() -> int:
    rows = [r for d in sorted(RESULTS.iterdir()) if d.is_dir() for r in [read_run(d)] if r]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} runs to {OUT}")

    # Sanity check: HAR natural_block, full participation, mean over seeds.
    print("\nHAR natural_block r100 (mean over seeds of per-seed client means)")
    for algo in ("local", "fedavg", "fedper", "fedrep", "proposed"):
        sub = [r for r in rows if r["exp_id"].startswith(f"main-har-{algo}-r100-s")]
        if not sub:
            continue
        mg = [r["mean_global_f1"] for r in sub if "mean_global_f1" in r]
        wg = [r["worst_global_f1"] for r in sub if "worst_global_f1" in r]
        mp = [r["mean_pers_f1"] for r in sub if "mean_pers_f1" in r]
        wp = [r["worst_pers_f1"] for r in sub if "worst_pers_f1" in r]
        mean_val = st.mean(mp) if mp else (st.mean(mg) if mg else float("nan"))
        worst_val = st.mean(wp) if wp else (st.mean(wg) if wg else float("nan"))
        print(f"  {algo:10s} n={len(sub):2d} mean={mean_val:.4f} worst={worst_val:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
