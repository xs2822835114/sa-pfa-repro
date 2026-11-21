"""第四轮修订 · 全量重跑驱动（无泄漏划分 + 移除贡献项 C）。

背景：
  - UAH/HAR 的 train/test 改为行程级/时间块级划分（见 code/data/partition.py、
    partition_har_natural_block.json），消除 50% 重叠窗口泄漏。
  - proposed 默认 --no-c：不划 D_val、不评估候选完整模型，C≡1（第四轮决策）。
  - 所有算法 --val-frac 0：不再有服务器验证集，训练池一致。

组（--groups 逗号分隔；默认 main,har,scale,avail,extra）：
  main   natural 主矩阵（4 参与率、7 算法 + ablAQC + 10-seed 复现）、IID、α=0.05
  har    UCI HAR 时间块划分（21 runs）
  scale  K=12/18 客户端粒度外推
  avail  availability / markov / shift 调度 + FedPer+prox + FedProx 调参 + Ditto 调参 + pFedMe 调参
  extra  -prox、A/Q 消融

用法：
  python code/experiments/v4_rerun.py --dry-run
  python code/experiments/v4_rerun.py --workers 6
  python code/experiments/v4_rerun.py --groups main --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[2]
RUNNER = RESEARCH / "code" / "federated" / "runner.py"
RESULTS = RESEARCH / "results" / "baselines"
LOGS = RESEARCH / "logs" / "v4_rerun"

BASE = ["--window", "30", "--rounds", "80", "--lr", "0.01", "--local-epochs", "1",
        "--batch-size", "64", "--val-frac", "0"]

METHODS = ["fedavg", "fedprox", "pfedme", "fedper", "ditto", "proposed"]
SEEDS = [1, 2, 3, 4, 5, 6]


def job(exp, algo, part, rate, seed, extra, dataset="uah", mode="uniform"):
    cmd = [sys.executable, str(RUNNER), "--algo", algo, "--dataset", dataset,
           "--partition", part, "--rate", str(rate), "--mode", mode,
           "--seed", str(seed), "--exp-id", exp] + BASE + extra
    return exp, cmd


def build(groups: set[str]):
    jobs = []

    # ---------------- main ----------------
    if "main" in groups:
        # natural：7 算法（local 只 100%），4 个参与率
        for s in SEEDS:
            jobs.append(job(f"main-local-natural-r100-s{s}", "local", "natural", 1.0, s,
                            ["--val-frac", "0"]))
        for algo in METHODS:
            rates = [1.0, 0.6, 0.2] if algo != "proposed" else [1.0, 0.6, 0.2, 0.1]
            for r in rates:
                for s in SEEDS:
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"main-{algo}-natural-r{int(r*100)}-s{s}", algo, "natural", r, s, extra))
        # 10%：FedAvg / FedPer / Ditto / SA-PFA（正文四方法）
        for algo in ("fedavg", "fedper", "ditto", "proposed"):
            for s in SEEDS:
                extra = ["--no-c"] if algo == "proposed" else []
                jobs.append(job(f"main-{algo}-natural-r10-s{s}", algo, "natural", 0.1, s, extra))
        # ablAQC：同结构、无动态权重（a=b=0, d=1）
        for r in (1.0, 0.6, 0.2):
            for s in SEEDS:
                jobs.append(job(f"main-ablAQC-natural-r{int(r*100)}-s{s}", "proposed", "natural", r, s,
                                ["--no-c", "--a-exp", "0", "--b-exp", "0"]))
        # 10-seed 复现：FedAvg / FedPer / SA-PFA
        for algo in ("fedavg", "fedper", "proposed"):
            for r in (1.0, 0.6, 0.2):
                for s in (7, 8, 9, 10):
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"main-{algo}-natural-r{int(r*100)}-s{s}", algo, "natural", r, s, extra))
        # IID（100%，6 seeds）
        for algo in ("fedavg", "fedper", "proposed"):
            for s in SEEDS:
                extra = ["--no-c"] if algo == "proposed" else []
                jobs.append(job(f"main-{algo}-iid-r100-s{s}", algo, "iid", 1.0, s, extra))
        # α=0.05 参考头对照（3 seeds，含 ref-head 变体报告）
        for algo in ("fedavg", "fedper", "proposed"):
            for s in (1, 2, 3):
                extra = ["--ref-head-report", "1"]
                if algo == "proposed":
                    extra += ["--no-c"]
                jobs.append(job(f"main-{algo}-dirichlet_a0.05-r100-s{s}", algo, "dirichlet_a0.05", 1.0, s, extra))

    # ---------------- wisdm（第三个数据集：51 客户端，标签偏斜对照） ----------------
    if "wisdm" in groups:
        for algo in ("fedavg", "fedper", "fedrep", "proposed"):
            for r in (1.0, 0.2):
                for s in (1, 2, 3):
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"main-wisdm-{algo}-r{int(r*100)}-s{s}", algo, "natural", r, s,
                                    extra, dataset="wisdm"))
        for algo in ("fedavg", "fedper", "proposed"):
            for s in (1, 2, 3):
                extra = ["--no-c"] if algo == "proposed" else []
                jobs.append(job(f"v4-wisdm-skew-{algo}-s{s}", algo, "dirichlet_a0.05", 1.0, s,
                                extra, dataset="wisdm"))

    # ---------------- fedrep / fold2 / har_refhead6（第四轮补充） ----------------
    if "fedrep" in groups:
        for r in (1.0, 0.6, 0.2):
            for s in SEEDS:
                jobs.append(job(f"main-fedrep-natural-r{int(r*100)}-s{s}", "fedrep", "natural", r, s, []))
        for r in (1.0, 0.6, 0.2, 0.1):
            for s in SEEDS:
                jobs.append(job(f"main-har-fedrep-r{int(r*100)}-s{s}", "fedrep", "natural_block", r, s,
                                [], dataset="har"))
    if "fold2" in groups:
        for algo in ("fedavg", "fedper", "proposed"):
            for r in (1.0, 0.2):
                for s in (1, 2, 3):
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"v4-fold2-{algo}-r{int(r*100)}-s{s}", algo, "natural_fold2", r, s, extra))
    if "fold3" in groups:
        for algo in ("fedavg", "fedper", "proposed"):
            for r in (1.0, 0.2):
                for s in (1, 2, 3):
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"v4-fold3-{algo}-r{int(r*100)}-s{s}", algo, "natural_fold3", r, s, extra))
    if "har_refhead6" in groups:
        for algo in ("fedavg", "fedper", "proposed"):
            for s in (4, 5, 6):
                extra = ["--ref-head-report", "1"]
                if algo == "proposed":
                    extra += ["--no-c"]
                jobs.append(job(f"main-har-{algo}-a005-r100-s{s}", algo, "dirichlet_a0.05", 1.0, s,
                                extra, dataset="har"))

    # ---------------- har_main（第四轮：HAR 升为主实验） ----------------
    if "har_main" in groups:
        for s in (1, 2, 3):
            jobs.append(job(f"main-har-local-r100-s{s}", "local", "natural_block", 1.0, s,
                            ["--val-frac", "0"], dataset="har"))
        for algo in METHODS:
            for r in (1.0, 0.6, 0.2, 0.1):
                for s in SEEDS:
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"main-har-{algo}-r{int(r*100)}-s{s}", algo, "natural_block", r, s,
                                    extra, dataset="har"))
        for r in (1.0, 0.6, 0.2):
            for s in SEEDS:
                jobs.append(job(f"main-har-ablAQ-r{int(r*100)}-s{s}", "proposed", "natural_block", r, s,
                                ["--no-c", "--a-exp", "0", "--b-exp", "0"], dataset="har"))
        # availability 调度
        for algo in ("fedavg", "fedper", "proposed"):
            for r in (1.0, 0.6, 0.2, 0.1):
                for s in SEEDS:
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"v4-avail-har-{algo}-r{int(r*100)}-s{s}", algo, "natural_block", r, s,
                                    extra, dataset="har", mode="availability"))
        # 参考头规则（α=0.05，3 seeds）
        for algo in ("fedavg", "fedper", "proposed"):
            for s in (1, 2, 3):
                extra = ["--ref-head-report", "1"]
                if algo == "proposed":
                    extra += ["--no-c"]
                jobs.append(job(f"main-har-{algo}-a005-r100-s{s}", algo, "dirichlet_a0.05", 1.0, s,
                                extra, dataset="har"))

    # ---------------- har（旧的单次外部检查，保留兼容） ----------------
    if "har" in groups:
        for s in (1, 2, 3):
            jobs.append(job(f"v4-har-local-r100-s{s}", "local", "natural_block", 1.0, s,
                            ["--val-frac", "0"], dataset="har"))
        for algo in ("fedavg", "fedper", "proposed"):
            for r in (1.0, 0.2):
                for s in (1, 2, 3):
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"v4-har-{algo}-r{int(r*100)}-s{s}", algo, "natural_block", r, s,
                                    extra, dataset="har"))

    # ---------------- scale ----------------
    if "scale" in groups:
        for algo in ("fedavg", "fedper", "proposed"):
            for part in ("natural_k12", "natural_k18"):
                for r in (1.0, 0.2):
                    for s in (1, 2, 3):
                        extra = ["--no-c"] if algo == "proposed" else []
                        jobs.append(job(f"v4-scale-{algo}-{part}-r{int(r*100)}-s{s}", algo, part, r, s, extra))

    # ---------------- avail ----------------
    if "avail" in groups:
        # availability：4 个参与率
        for algo in ("fedavg", "fedper", "ditto", "proposed"):
            for r in (1.0, 0.6, 0.2, 0.1):
                for s in SEEDS:
                    extra = ["--no-c"] if algo == "proposed" else []
                    jobs.append(job(f"v4-avail-{algo}-r{int(r*100)}-s{s}", algo, "natural", r, s,
                                    extra, mode="availability"))
        # FedPer+prox：uniform 3 档 + availability 2 档
        for r in (1.0, 0.6, 0.2):
            for s in SEEDS:
                jobs.append(job(f"v4-fedperprox-u-r{int(r*100)}-s{s}", "fedperprox", "natural", r, s, []))
        for r in (1.0, 0.2):
            for s in SEEDS:
                jobs.append(job(f"v4-fedperprox-a-r{int(r*100)}-s{s}", "fedperprox", "natural", r, s, [],
                                mode="availability"))
        # 相关可用性：markov / shift
        for mode in ("markov", "shift"):
            for algo in ("fedavg", "fedper", "proposed"):
                for r in (1.0, 0.6, 0.2):
                    for s in SEEDS:
                        extra = ["--no-c"] if algo == "proposed" else []
                        jobs.append(job(f"v4-{mode}-{algo}-r{int(r*100)}-s{s}", algo, "natural", r, s,
                                        extra, mode=mode))
        # FedProx 调参（μ）
        for mu in ("0.001", "0.1", "1"):
            for s in SEEDS:
                jobs.append(job(f"v4-mu{mu}-fedprox-natural-r100-s{s}", "fedprox", "natural", 1.0, s,
                                ["--mu", mu]))
        # Ditto 调参（λ）
        for lam in ("0.1", "5"):
            for s in SEEDS:
                jobs.append(job(f"v4-lam{lam}-ditto-natural-r100-s{s}", "ditto", "natural", 1.0, s,
                                ["--lam", lam]))
        # pFedMe 调参（inner × outer）
        for inner in (1, 5):
            for outer in ("0.1", "0.3"):
                for s in (1, 2, 3):
                    jobs.append(job(f"v4-pfedme-i{inner}-o{outer}-s{s}", "pfedme", "natural", 1.0, s,
                                    ["--inner-epochs", str(inner), "--outer-lr", outer]))

    # ---------------- extra ----------------
    if "extra" in groups:
        # -prox：proposed 去掉 backbone 近端项
        for s in (1, 2, 3):
            jobs.append(job(f"v4-noprox-natural-r100-s{s}", "proposed", "natural", 1.0, s,
                            ["--no-c", "--mu-sa", "0"]))
            jobs.append(job(f"v4-noprox-avail-r20-s{s}", "proposed", "natural", 0.2, s,
                            ["--no-c", "--mu-sa", "0"], mode="availability"))
        # A / Q 单因子消融
        for name, extra in (("noA", ["--a-exp", "0"]), ("noQ", ["--b-exp", "0"])):
            for s in (1, 2, 3):
                jobs.append(job(f"v4-abl-{name}-natural-r100-s{s}", "proposed", "natural", 1.0, s,
                                ["--no-c"] + extra))
                jobs.append(job(f"v4-abl-{name}-avail-r60-s{s}", "proposed", "natural", 0.6, s,
                                ["--no-c"] + extra, mode="availability"))
                jobs.append(job(f"v4-abl-{name}-a01-r100-s{s}", "proposed", "dirichlet_a0.1", 1.0, s,
                                ["--no-c"] + extra))
    return jobs


def run_one(exp, cmd, logdir: Path):
    log = logdir / f"{exp}.log"
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=str(RESEARCH), stdout=f, stderr=subprocess.STDOUT, env=env)
    return exp, r.returncode, time.time() - t0, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="main,har,scale,avail,extra")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    groups = {g.strip() for g in a.groups.split(",") if g.strip()}

    jobs = build(groups)
    todo = [(e, c) for e, c in jobs if not (RESULTS / e / "final_metrics.json").exists()]
    print(f"jobs={len(jobs)}  done={len(jobs)-len(todo)}  todo={len(todo)}  groups={sorted(groups)}")
    if a.dry_run:
        for e, c in todo:
            print(" ".join(c))
        return 0
    if not todo:
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    fails = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, e, c, LOGS): e for e, c in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            exp, rc, dt, log = fut.result()
            status = "ok" if rc == 0 else f"FAIL rc={rc}"
            print(f"[{k}/{len(todo)}] {exp:52s} {status} {dt:6.1f}s", flush=True)
            if rc != 0:
                fails.append(exp)
    print(f"\n完成：{len(todo)-len(fails)} ok, {len(fails)} fail, "
          f"总耗时 {time.time()-t0:.0f}s")
    if fails:
        print("失败：", ", ".join(fails))
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
