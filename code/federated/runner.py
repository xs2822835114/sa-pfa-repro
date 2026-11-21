"""Unified experiment driver for the federated baselines and SA-PFA.

One run trains a single algorithm for a fixed number of rounds on one partition and writes
history.csv, config.json, final_metrics.json, curves.png and model_final.pt (plus pers_final.pt
for personalised methods) to results/baselines/<exp_id>/; the run is also appended to
results/experiment_records.csv. Features and partition JSONs are read from data/processed/.

Two protocols are reported: global (the shared model scored on every client's test split, with
mean/std/worst over clients) and personalized (each client scored with its own model). The
participation schedule is generated on the fly from --seed.

Example:
  python code/federated/runner.py --algo fedprox --partition dirichlet_a0.1 \\
      --rate 0.6 --rounds 40 --exp-id bl-fedprox-dirichlet_a0.1-r60
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
RESEARCH_DIR = CODE_DIR.parent
REPO_ROOT = RESEARCH_DIR.parent
sys.path.insert(0, str(CODE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(RESEARCH_DIR / "logs" / "mplcache"))

import torch  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Songti SC", "Heiti SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from federated import flcore, proposed  # noqa: E402
from common.record import append_record, git_commit, today  # noqa: E402

ALGOS = ["local", "fedavg", "fedprox", "fedper", "fedperprox", "fedrep", "pfedme", "ditto", "proposed"]
NON_IID_DESC = {
    "natural": "Natural(driver)", "iid": "IID",
    "dirichlet_a1.0": "Mild(a=1.0)", "dirichlet_a0.5": "Medium(a=0.5)",
    "dirichlet_a0.1": "Severe(a=0.1)", "dirichlet_a0.05": "Extreme(a=0.05)",
}
HEAD_PREFIX = "net.5."   # last MLP layer = personalised head; the rest = shared base
N_CLASSES = 3            # UAH has 3 classes; set to 6 for the HAR/WISDM checks


def head_keys(state):
    return [k for k in state.keys() if k.startswith(HEAD_PREFIX)]


def base_keys(state):
    return [k for k in state.keys() if not k.startswith(HEAD_PREFIX)]


def subset(state, keys):
    return OrderedDict((k, state[k]) for k in keys)


def online_probs(rate, n_clients):
    """Steady-state online probability per client (one nearly always online, one rarely),
    scaled by rate."""
    pattern = np.array([1.0, 1.4, 0.4, 1.2, 0.7, 0.3])
    # tile the pattern when K exceeds its length (scaled client-count experiments need K>6)
    pat = pattern[:n_clients] if n_clients <= pattern.size else np.resize(pattern, n_clients)
    return np.clip(pat * rate, 0.05, 1.0)


def make_schedule(mode, rate, rounds, seed, n_clients=6, allow_empty=False, dwell=8):
    """Build the participation schedule on the fly (same semantics as code/data/sparse.py,
    but seeded here).

    uniform      draw ceil(rate*K) clients per round (without replacement)
    availability each client has its own online probability (one nearly always online,
                 one rarely); independent Bernoulli sampling per round
    markov       same online probabilities, but each client follows a two-state Markov
                 chain: mean online dwell rounds, mean offline dwell*(1-p)/p
    shift        same online probabilities, one shift every dwell rounds: at the start of
                 a block the client is online for the whole block with probability p

    allow_empty  applies to availability/markov/shift only. False (default, as in the
                 existing experiments): if no client is online, draw one so training never
                 stalls. True: keep the empty round (no aggregation) so that "sparse
                 participation" and "forced single-client rotation" can be separated.
    """
    rng = np.random.default_rng(seed)
    if mode == "uniform":
        m = max(1, int(np.ceil(rate * n_clients)))
        return {t: sorted(rng.choice(n_clients, size=m, replace=False).tolist()) for t in range(rounds)}
    p = online_probs(rate, n_clients)
    out = {}

    def emit(t, on):
        part = np.where(on)[0]
        if len(part) == 0 and not allow_empty:
            part = np.array([int(rng.choice(n_clients))])
        out[t] = sorted(part.tolist())

    if mode == "markov":
        # two-state chain: on->off = 1/dwell, off->on = p/(dwell(1-p)), stationary p
        beta = np.where(p < 1.0, 1.0 / dwell, 0.0)
        alpha = np.where(p < 1.0, p * beta / np.maximum(1.0 - p, 1e-9), 0.0)
        on = rng.random(n_clients) < p
        for t in range(rounds):
            emit(t, on)
            u = rng.random(n_clients)
            on = np.where(on, u >= beta, u < alpha)
        return out

    if mode == "shift":
        on = None
        for t in range(rounds):
            if t % dwell == 0:
                on = rng.random(n_clients) < p
            emit(t, on)
        return out

    for t in range(rounds):
        emit(t, rng.random(n_clients) < p)
    return out


def evaluate_model(model, X, y):
    n_cls = model.net[-1].out_features if hasattr(model, "net") else N_CLASSES
    m = flcore.evaluate(model, X, y, n_classes=n_cls)
    return m["acc"], m["macro_f1"]


def per_client_metrics(model, client_test):
    """Global protocol: F1 of the shared model on each client's test split."""
    f1s, accs = {}, {}
    for c, (X, y) in client_test.items():
        acc, f1 = evaluate_model(model, X, y)
        f1s[c], accs[c] = f1, acc
    return f1s, accs


def personalized_metrics(pers_models, client_test):
    f1s, accs = {}, {}
    for c, (X, y) in client_test.items():
        acc, f1 = evaluate_model(pers_models[c], X, y)
        f1s[c], accs[c] = f1, acc
    return f1s, accs


def agg_stats(f1s):
    v = np.array(list(f1s.values()))
    return float(v.mean()), float(v.std()), float(v.min())


def ref_mean_head(heads, participants, ref_state, clients):
    """Mean of the participants' personalised heads (reference global head; used only for
    the global-protocol evaluation of fedper/proposed)."""
    out = OrderedDict()
    keys = [k for k in ref_state.keys() if k.startswith(HEAD_PREFIX)]
    for k in keys:
        stacked = torch.stack([heads[clients[int(cid)]][k].float() for cid in participants])
        out[k] = stacked.mean(dim=0).to(ref_state[k].dtype)
    return out


def ref_weighted_head(heads, entries, alpha, ref_state, clients):
    """Reference global head weighted by the same alpha (proposed; used for evaluation and
    as the next-round initialisation, never inside the shared-layer aggregation)."""
    out = OrderedDict()
    keys = [k for k in ref_state.keys() if k.startswith(HEAD_PREFIX)]
    for k in keys:
        acc = None
        for e in entries:
            t_ = heads[e["cid"]][k].float() * alpha[e["cid"]]
            acc = t_ if acc is None else acc + t_
        out[k] = acc.to(ref_state[k].dtype)
    return out


def macro_f1_from_preds(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> float:
    """Macro-F1 computed exactly as in flcore.evaluate (for non-parametric outputs such as
    prediction-level ensembles)."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


def ref_head_rule_variants(base_state, heads, participants, clients, client_test, val_xy,
                           n_cls=None, n_train=None, n_min=32.0, min_classes=2, calib_steps=300,
                           calib_lr=1e-2, seed=0, n_classes=3, in_dim=47):
    """Quality of the shared deliverable under four reference-head rules on the same shared
    backbone (evaluation side only; training is untouched).

    mean     uniform average of the participants' personalised heads (current default)
    gate     uniform average after dropping support-degraded participants (n_i < n_min or
             fewer than min_classes local training classes)
    ensemble prediction-level majority vote of "shared backbone + own head" per participant
             (skipped without heads; the server must hold every head, upper bound only)
    calib    refit the head on the server-side D_val with the backbone frozen (also applies
             without heads; a fair control for FedAvg)
    """
    import torch.nn.functional as F
    head_ks = [k for k in base_state.keys() if k.startswith(HEAD_PREFIX)]
    out = {}

    def mk(state):
        m = flcore.build_model(in_dim, n_classes=n_classes, seed=seed)
        m.load_state_dict(OrderedDict(state))
        return m

    def eval_state(state):
        m = mk(state)
        per = {}
        for c, (X, y) in client_test.items():
            _, f1 = evaluate_model(m, X, y)
            per[c] = float(f1)
        vals = np.array(list(per.values()), dtype=float)
        return dict(per_client_f1=per, mean=float(vals.mean()), worst=float(vals.min()))

    if heads:
        parts = list(participants)
        out["mean"] = eval_state({**base_state, **ref_mean_head(heads, parts, base_state, clients)})
        gated = []
        for p in parts:
            cname = clients[int(p)]
            by_cls = n_cls is not None and float(n_cls.get(cname, min_classes)) < min_classes
            by_n = n_train is not None and float(n_train.get(cname, n_min)) < n_min
            if by_cls or by_n:
                gated.append(p)
        keep = [p for p in parts if clients[int(p)] not in set(gated)] or parts
        keep = [p for p in parts if int(p) not in set(gated)] or parts
        out["gate"] = eval_state({**base_state, **ref_mean_head(heads, keep, base_state, clients)})
        out["gate"]["excluded"] = sorted(clients[int(p)] for p in gated)
        # prediction-level ensemble: every participant votes with its own head on the shared backbone
        preds = []
        for p in parts:
            m = mk({**base_state, **{k: heads[clients[int(p)]][k] for k in head_ks}})
            with torch.no_grad():
                preds.append([])
                for c, (X, y) in client_test.items():
                    preds[-1].append(m(X).argmax(dim=1).cpu().numpy())
        per = {}
        for j, (c, (X, y)) in enumerate(client_test.items()):
            stack = np.stack([preds[i][j] for i in range(len(preds))])
            yt = y.cpu().numpy()
            vote = np.array([np.bincount(stack[:, i], minlength=n_classes).argmax() for i in range(stack.shape[1])])
            per[c] = macro_f1_from_preds(yt, vote, n_classes)
        vals = np.array(list(per.values()), dtype=float)
        out["ensemble"] = dict(per_client_f1=per, mean=float(vals.mean()), worst=float(vals.min()))

    if val_xy is not None:
        X_val, y_val = val_xy
        m = mk(base_state)
        for name, p in m.named_parameters():
            p.requires_grad_(name.startswith(HEAD_PREFIX))
        opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=calib_lr)
        m.train()
        n_v = len(y_val)
        bs = max(1, min(64, n_v))
        torch.manual_seed(seed)
        for step in range(max(1, calib_steps)):
            idx = torch.randperm(n_v)[:bs] if n_v > bs else torch.arange(n_v)
            opt.zero_grad()
            loss = F.cross_entropy(m(X_val[idx]), y_val[idx])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            tr_acc = float((m(X_val).argmax(dim=1) == y_val).float().mean())
        calib = eval_state(m.state_dict())
        calib["val_acc"] = tr_acc
        calib["n_val"] = int(n_v)
        out["calib"] = calib
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=ALGOS)
    ap.add_argument("--dataset", default="uah", choices=["uah", "har", "wisdm"],
                    help="uah=UAH-DriveSet 3-class; har=UCI HAR and wisdm=WISDM 6-class external checks")
    ap.add_argument("--window", type=int, default=30, choices=[10, 30, 60])
    ap.add_argument("--partition", default="iid")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--mode", default="uniform", choices=["uniform", "availability", "markov", "shift"])
    ap.add_argument("--dwell", type=int, default=8,
                    help="mean online dwell (rounds) for markov/shift")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--inner-epochs", type=int, default=3, help="number of pfedme inner iterations")
    ap.add_argument("--outer-lr", type=float, default=0.3,
                    help="pfedme outer step size eta (the pseudo-gradient lam(w-theta) has the same scale as eta)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mu", type=float, default=0.01, help="FedProx mu")
    ap.add_argument("--lam", type=float, default=1.0, help="pfedme/ditto lambda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp-id", default=None)
    # ---- SA-PFA (proposed) hyperparameters; defaults frozen after the ablation sweep ----
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="fraction of each client's training split held out as the shared validation set D_val "
                         "(>0 gives every algorithm the same protocol)")
    ap.add_argument("--val-noise-rate", type=float, default=0.0,
                    help="symmetric label-noise rate on the server-side D_val (0=off; only active with val-frac>0)")
    ap.add_argument("--omega", type=float, default=0.4, help="SA-PFA weight floor omega (frozen at 0.4)")
    ap.add_argument("--no-c", action="store_true",
                    help="drop the contribution term C: no D_val split, no candidate full-model evaluation, C=1")
    ap.add_argument("--mu-sa", type=float, default=0.1, help="SA-PFA prox strength on the shared layers (frozen at 0.1)")
    ap.add_argument("--eta", type=float, default=0.6, help="SA-PFA temporal smoothing eta")
    ap.add_argument("--a-exp", type=float, default=1.0)
    ap.add_argument("--b-exp", type=float, default=1.0)
    ap.add_argument("--c-exp", type=float, default=1.0)
    ap.add_argument("--d-exp", type=float, default=1.0,
                    help="sample-size exponent d (with A=Q=C=1 it degenerates to FedAvg)")
    ap.add_argument("--a-lam", type=float, default=0.8, help="activity EWMA lambda")
    ap.add_argument("--a-T", type=float, default=5.0, help="activity staleness time constant T")
    ap.add_argument("--gamma-a", type=float, default=0.5, help="activity mixing weight gamma")
    ap.add_argument("--a-min", type=float, default=0.05)
    ap.add_argument("--q-beta", type=float, default=0.5, help="sample-sufficiency exponent beta")
    ap.add_argument("--q-min", type=float, default=0.05)
    ap.add_argument("--q-use-size", type=int, default=0,
                    help="whether Q counts sample size again: 0=(kappa*psi*nu)^(1/3) (default, avoids double "
                         "counting with n^d); 1=(kappa*psi*nu*s)^(1/4)")
    ap.add_argument("--c-rho", type=float, default=0.3, help="contribution EWMA rho")
    # ---- support gate (default off) ----
    ap.add_argument("--support-gate", type=int, default=0,
                    help="1=enable the support gate: degraded clients are excluded from the reference "
                         "head and exempt from the omega/K floor")
    ap.add_argument("--gate-min-n", type=float, default=32.0,
                    help="clients with fewer training samples are treated as support-degraded")
    ap.add_argument("--gate-min-classes", type=int, default=2,
                    help="clients with fewer local training classes are treated as support-degraded")
    # ---- reference-head rule report (evaluation side, all variants in one run; default off) ----
    ap.add_argument("--ref-head-report", type=int, default=0,
                    help="1=additionally evaluate the reference-head rules {mean,gate,ensemble,calib} after "
                         "training (the training procedure is unchanged)")
    ap.add_argument("--calib-steps", type=int, default=300, help="fitting steps for the calib head on D_val")
    ap.add_argument("--calib-lr", type=float, default=1e-2, help="learning rate for the calib head")
    ap.add_argument("--trip-balance", type=int, default=0,
                    help="1=weight the local loss by 1/(number of windows in the trip) (equal weight per trip; "
                         "default 0=equal weight per window)")
    ap.add_argument("--fedrep-head-epochs", type=int, default=5,
                    help="FedRep: head-only epochs per round (backbone frozen)")
    ap.add_argument("--fedrep-body-epochs", type=int, default=1,
                    help="FedRep: backbone-only epochs afterwards (head frozen)")
    ap.add_argument("--allow-empty-round", type=int, default=0,
                    help="availability only: 1=allow empty rounds (skip a round when nobody is online) to "
                         "separate sparse participation from forced single-client rotation")
    ap.add_argument("--holdout-client", default="",
                    help="leave-one-client-out cold start: this client does not train; afterwards evaluate "
                         "zero-shot and few-shot heads of the shared backbone")
    ap.add_argument("--cold-fracs", default="0.05,0.1,0.25",
                    help="fraction of the held-out client's training samples available for cold start "
                         "(comma separated; 0 = zero-shot only)")
    args = ap.parse_args()

    if args.no_c:
        # contribution term C removed: no D_val split (saves 10% of the training data and one
        # full forward pass) and no candidate full-model evaluation (saves compute and avoids
        # uploading heads), so C stays 1
        args.c_exp = 0.0
        args.val_frac = 0.0

    exp_id = args.exp_id or f"bl-{args.algo}-{args.partition}-r{int(args.rate*100)}-L{args.window}"
    out_dir = RESEARCH_DIR / "results" / "baselines" / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data and partitions ----------
    global N_CLASSES
    if args.dataset in ("har", "wisdm"):
        tag = "HAR" if args.dataset == "har" else "WISDM"
        data_dir = RESEARCH_DIR / "data" / "processed" / tag
        fname = "features_har.npz" if args.dataset == "har" else "features_wisdm.npz"
        feat = np.load(data_dir / fname, allow_pickle=True)
        prefix = "partition_har_" if args.dataset == "har" else "partition_wisdm_"
        part_path = data_dir / "partitions" / f"{prefix}{args.partition}.json"
        N_CLASSES = 6
    else:
        data_dir = RESEARCH_DIR / "data" / "processed" / "UAH"
        feat = np.load(data_dir / f"features_L{args.window}.npz", allow_pickle=True)
        part_path = data_dir / "partitions" / f"partition_{args.partition}.json"
        N_CLASSES = 3
    X_all, y_all = feat["X"], feat["y"]
    with open(part_path, encoding="utf-8") as f:
        partition_data = json.load(f)
    clients = sorted(partition_data.keys(), key=lambda s: int(s.split("_")[1]))

    # ---------- leave-one-client-out cold start: this client trains on nothing and keeps only its test split ----------
    holdout_train_idx = holdout_test_idx = None
    if args.holdout_client:
        if args.holdout_client in partition_data:
            holdout_train_idx = np.asarray(partition_data[args.holdout_client]["train"], dtype=int)
            holdout_test_idx = np.asarray(partition_data[args.holdout_client]["test"], dtype=int)
            clients = [c for c in clients if c != args.holdout_client]
        else:
            raise SystemExit(f"holdout client {args.holdout_client} not in partition")

    # ---------- shared validation set D_val (same protocol for all algorithms; carved out of the
    # training pool and never used for local training) ----------
    val_holdout = {c: [] for c in clients}
    if args.val_frac > 0:
        rng = np.random.default_rng(args.seed * 100 + 7)
        for c in clients:
            idx = np.asarray(partition_data[c]["train"], dtype=int)
            k = max(1, int(round(args.val_frac * len(idx))))
            pick = np.sort(rng.choice(len(idx), size=k, replace=False))
            val_holdout[c] = idx[pick].tolist()
            keep = np.setdiff1d(np.arange(len(idx)), pick)
            partition_data[c]["train"] = idx[keep].tolist()

    union_train = np.concatenate([np.asarray(partition_data[c]["train"], dtype=int) for c in clients])
    _, others, _, _ = flcore.standardize(X_all[union_train], [X_all])
    X_std = others[0]

    def tensors(idxs):
        return (torch.tensor(X_std[idxs], dtype=torch.float32), torch.tensor(y_all[idxs], dtype=torch.long))

    client_train = {c: tensors(np.asarray(partition_data[c]["train"], dtype=int)) for c in clients}
    client_test = {c: tensors(np.asarray(partition_data[c]["test"], dtype=int)) for c in clients}
    trip_weights = {}
    if args.trip_balance:
        trip_all = feat["trip"]
        for c in clients:
            idx = np.asarray(partition_data[c]["train"], dtype=int)
            tp = trip_all[idx]
            _, inv, cnt = np.unique(tp, return_inverse=True, return_counts=True)
            w = (1.0 / cnt.astype(np.float64))[inv]
            w = w / w.mean()
            trip_weights[c] = torch.tensor(w, dtype=torch.float32)
    X_te_all = torch.cat([client_test[c][0] for c in clients])
    y_te_all = torch.cat([client_test[c][1] for c in clients])

    val_idx = np.concatenate([np.asarray(val_holdout[c], dtype=int) for c in clients]) if args.val_frac > 0 else np.array([], dtype=int)
    X_val, y_val = tensors(val_idx) if len(val_idx) else (None, None)
    # symmetric label noise on the server-side D_val (default 0=off)
    if len(val_idx) and args.val_noise_rate > 0:
        rng_n = np.random.default_rng(args.seed * 1000 + 13)   # separate stream: other seeds untouched
        n_cls_noise = N_CLASSES
        y_np = y_val.cpu().numpy().copy()
        flip = rng_n.random(y_np.shape[0]) < args.val_noise_rate
        m_flip = int(flip.sum())
        if m_flip:
            off = rng_n.integers(0, n_cls_noise - 1, size=m_flip)   # one of the other two classes
            y_np[flip] = (y_np[flip] + 1 + off) % n_cls_noise
        y_val = torch.tensor(y_np, dtype=torch.long)
    # slice each client occupies in D_val (for leave-one-out contribution: C_i uses only the
    # validation samples of the *other* clients)
    val_masks = {}
    if len(val_idx):
        off = 0
        for c in clients:
            k = len(val_holdout[c])
            m = torch.zeros(len(val_idx), dtype=torch.bool)
            m[off:off + k] = True
            val_masks[c] = ~m          # exclude client c's own samples
            off += k

    def val_loss(model, exclude_client=None):
        if X_val is None:
            return float("nan")
        import torch.nn.functional as F
        with torch.no_grad():
            if exclude_client is None:
                return float(F.cross_entropy(model(X_val), y_val).item())
            m = val_masks[exclude_client]
            return float(F.cross_entropy(model(X_val[m]), y_val[m]).item())

    schedule = make_schedule(args.mode, args.rate, args.rounds, args.seed, n_clients=len(clients),
                             allow_empty=bool(args.allow_empty_round), dwell=args.dwell)
    in_dim = X_all.shape[1]
    torch.manual_seed(args.seed)
    global_model = flcore.build_model(in_dim, N_CLASSES, seed=args.seed)
    n_base_bytes = sum(v.numel() for k, v in global_model.state_dict().items() if k.startswith(HEAD_PREFIX) is False) * 4
    n_full_bytes = flcore.state_bytes(global_model.state_dict())
    n_head_bytes = n_full_bytes - n_base_bytes

    # ---------- algorithm state ----------
    heads = {c: subset(global_model.state_dict(), head_keys(global_model.state_dict())) for c in clients}  # fedper
    theta = {c: copy.deepcopy(global_model) for c in clients}   # ditto personalised models
    pfed_theta = {c: copy.deepcopy(global_model) for c in clients}  # pfedme personalised models (refreshed by participants)
    pers_models = None
    local_models = {c: flcore.build_model(in_dim, N_CLASSES, seed=args.seed + i) for i, c in enumerate(clients)}  # local

    # ---- SA-PFA state (proposed) ----
    estate = {c: proposed.ClientEvalState() for c in clients}
    contrib = proposed.ContributionScore(rho=args.c_rho)
    qstat = {}
    n_train = {c: len(np.asarray(partition_data[c]["train"], dtype=int)) for c in clients}
    # local training class coverage (support gate; computed for all algorithms so reporting is fair)
    n_cls = {c: int(np.unique(y_all[np.asarray(partition_data[c]["train"], dtype=int)]).size)
             for c in clients}
    if args.algo == "proposed":
        n_ref = float(np.median(list(n_train.values())))
        for c in clients:
            idx = np.asarray(partition_data[c]["train"], dtype=int)
            hs = proposed.neighborhood_stats(X_std[idx], y_all[idx])
            qstat[c] = proposed.quality_score(n_train[c], n_ref, kappa=1.0, psi=hs["psi"], nu=hs["nu"],
                                              beta=args.q_beta, q_min=args.q_min,
                                              use_size=bool(args.q_use_size))
            estate[c].q_ema = qstat[c]
            estate[c].n = n_train[c]
    alpha_hist = []
    gate_hist = []

    comm_mb = 0.0
    skipped_rounds = 0
    history = []
    t_start = time.time()
    for t in range(args.rounds):
        participants = schedule[t]
        if not participants:            # only possible with --allow-empty-round 1
            skipped_rounds += 1
            continue
        states, weights, losses = [], [], []
        pers_eval = None

        if args.algo == "local":
            # local-only training: no communication, used as the personalised reference
            for c in clients:
                local_models[c], n_i, loss_i = flcore.train_local(
                    local_models[c], *client_train[c], epochs=1, lr=args.lr,
                    batch_size=args.batch_size, seed=args.seed + t)
            pers_models = local_models

        elif args.algo in ("fedavg", "fedprox", "pfedme"):
            # fedavg:    w <- sum_i (n_i / sum_j n_j) * w_i over all parameters
            # fedprox:   local objective L_i(w) + (mu/2)||w - w^t||^2
            # pfedme:    inner theta_i = argmin L_i(theta) + (lam/2)||theta - w||^2, outer
            #            pseudo-gradient g_i = lam (w - theta_i); theta_i is the deployment model
            prox = args.mu if args.algo == "fedprox" else 0.0
            inner = args.inner_epochs if args.algo == "pfedme" else args.local_epochs
            thetas = {}
            for cid in participants:
                c = clients[cid]
                m, n_i, loss_i = flcore.train_local(
                    global_model, *client_train[c], epochs=inner, lr=args.lr,
                    batch_size=args.batch_size, seed=args.seed + t,
                    prox_mu=(args.lam if args.algo == "pfedme" else prox))
                states.append(m.state_dict())
                weights.append(n_i)
                losses.append(loss_i)
                if args.algo == "pfedme":
                    thetas[c] = m
            if args.algo == "pfedme":
                # outer step: w <- w - eta * lam * (w - theta_i), weighted mean (Moreau pseudo-gradient)
                w = global_model.state_dict()
                total = float(sum(weights))
                agg = OrderedDict()
                for k in w.keys():
                    g = 0.0
                    for m, wt in zip(states, weights):
                        g = g + (w[k].double() - m[k].double()) * args.lam * (wt / total)
                    agg[k] = (w[k].double() - args.outer_lr * g).to(w[k].dtype)
                global_model.load_state_dict(agg)
                for c, m in thetas.items():
                    pfed_theta[c] = m
                pers_models = pfed_theta
            else:
                global_model.load_state_dict(flcore.fedavg_aggregate(states, weights))
            comm_mb += (len(participants) * 2 * n_full_bytes) / 1e6

        elif args.algo in ("fedper", "fedperprox", "fedrep"):
            # fedper: shared base + local personalised head, only the base is aggregated;
            # fedperprox adds a prox term on the base; fedrep updates head, then backbone
            st = global_model.state_dict()
            base_upload = []
            for cid in participants:
                c = clients[cid]
                m = flcore.build_model(in_dim, N_CLASSES, seed=args.seed)
                full = OrderedDict(st)          # shared base (current global)
                full.update(heads[c])           # local personalised head
                m.load_state_dict(full)
                if args.algo == "fedrep":
                    # FedRep: update the head with the backbone frozen, then the backbone with the head frozen
                    m, n_i, _ = flcore.train_local_scope(
                        m, *client_train[c], scope="head", head_prefix=HEAD_PREFIX,
                        epochs=args.fedrep_head_epochs, lr=args.lr,
                        batch_size=args.batch_size, seed=args.seed + t)
                    m, n_i, loss_i = flcore.train_local_scope(
                        m, *client_train[c], scope="body", head_prefix=HEAD_PREFIX,
                        epochs=args.fedrep_body_epochs, lr=args.lr,
                        batch_size=args.batch_size, seed=args.seed + 1000 + t)
                else:
                    prox_kwargs = (dict(prox_mu=args.mu_sa, prox_skip_prefix=HEAD_PREFIX)
                                   if args.algo == "fedperprox" else {})
                    m, n_i, loss_i = flcore.train_local(
                        m, *client_train[c], epochs=args.local_epochs, lr=args.lr,
                        batch_size=args.batch_size, seed=args.seed + t, **prox_kwargs)
                heads[c] = subset(m.state_dict(), head_keys(st))   # the head stays local
                base_upload.append(subset(m.state_dict(), base_keys(st)))
                weights.append(n_i)
                losses.append(loss_i)
            new_base = flcore.fedavg_aggregate(base_upload, weights)
            merged = OrderedDict(st)
            merged.update(new_base)
            # reference global head: mean of this round's personalised heads (evaluation/reporting only;
            # avoids scoring a head that was never updated)
            merged.update(ref_mean_head(heads, participants, st, clients))
            global_model.load_state_dict(merged)
            pers_models = {}
            for c in clients:
                m = flcore.build_model(in_dim, N_CLASSES, seed=args.seed)
                full = OrderedDict(global_model.state_dict())
                full.update(heads[c])
                m.load_state_dict(full)
                pers_models[c] = m
            comm_mb += (len(participants) * 2 * n_base_bytes) / 1e6

        elif args.algo == "proposed":
            # SA-PFA: activity/quality/contribution-driven sparse-aware aggregation (weight floor
            # omega, temporal smoothing eta) plus a prox term on the shared backbone and a local
            # personalised head; contribution C is the drop in D_val loss
            st = global_model.state_dict()
            entries, base_upload, cand = [], [], {}
            for cid in participants:
                c = clients[cid]
                m = flcore.build_model(in_dim, N_CLASSES, seed=args.seed)
                full = OrderedDict(st)
                full.update(heads[c])
                m.load_state_dict(full)
                m, n_i, loss_i = flcore.train_local(
                    m, *client_train[c], epochs=args.local_epochs, lr=args.lr,
                    batch_size=args.batch_size, seed=args.seed + t,
                    prox_mu=args.mu_sa, prox_skip_prefix=HEAD_PREFIX,   # prox constrains the shared backbone only
                    sample_weights=trip_weights.get(c))
                heads[c] = subset(m.state_dict(), head_keys(st))
                base_upload.append(subset(m.state_dict(), base_keys(st)))
                cand[c] = m
                losses.append(loss_i)
                a_i = proposed.activity_score(estate[c], t, lam=args.a_lam, T=args.a_T,
                                              gamma=args.gamma_a, a_min=args.a_min)
                entries.append(dict(cid=c, A=a_i, Q=qstat[c], C=1.0, n=n_i, n_cls=n_cls[c]))
            # contribution: drop in D_val loss (global model vs the client's full model)
            # --no-c: contribution term removed (no D_val, no candidate evaluation), so C stays 1
            if args.no_c:
                cnorm = {e["cid"]: 1.0 for e in entries}
            else:
                loss_g = val_loss(global_model)
                for c in cand:
                    # leave-one-out: C_i uses only the other clients' validation samples, avoiding self-scoring
                    contrib.observe(c, val_loss(global_model, exclude_client=c),
                                    val_loss(cand[c], exclude_client=c))
                cnorm = contrib.normalized([e["cid"] for e in entries])
            for e in entries:
                e["C"] = cnorm[e["cid"]]
            # sparse-aware aggregation
            gate_kw = {}
            if args.support_gate:
                gate_kw = dict(gate_min_n=args.gate_min_n, gate_min_classes=args.gate_min_classes)
            alpha = proposed.sparse_aware_weights(
                entries, estate, exp_a=args.a_exp, exp_b=args.b_exp, exp_c=args.c_exp,
                exp_d=args.d_exp, floor=args.omega, momentum=args.eta, **gate_kw)
            gated = proposed.gated_clients(entries, **gate_kw) if args.support_gate else set()
            new_base = flcore.fedavg_aggregate(base_upload, [alpha[e["cid"]] for e in entries])
            merged = OrderedDict(st)
            merged.update(new_base)
            # reference global head: *uniform* mean of the participants' personalised heads (same
            # convention as fedper; evaluation/reporting only)
            # support gate: support-degraded clients are left out of the reference head (falls back to
            # the default convention if every participant is gated)
            head_parts = [p for p in participants if clients[int(p)] not in gated] or list(participants)
            refh = ref_mean_head(heads, head_parts, st, clients)
            if os.environ.get("SA_PFA_DEBUG"):
                exp_h = sum((heads[e["cid"]]["net.5.weight"].float() * alpha[e["cid"]]).double() for e in entries)
                glob_h = refh["net.5.weight"].double()
                print("  [dbg] t=", t, "|ref−Σαhead|=", float((glob_h - exp_h).abs().max()),
                      "| α=", {e["cid"].replace("client_", "c"): round(alpha[e["cid"]], 4) for e in entries})
            merged.update(refh)
            global_model.load_state_dict(merged)
            if args.support_gate:
                gate_hist.append(dict(round=t, gated=sorted(gated)))
            alpha_hist.append(dict(round=t, participants=len(participants),
                                   **{c: alpha[c] for c in alpha},
                                   **{f"C_raw_{c}": contrib.last_raw.get(c, float("nan")) for c in alpha}))
            pers_models = {}
            for c in clients:
                m = flcore.build_model(in_dim, N_CLASSES, seed=args.seed)
                full = OrderedDict(global_model.state_dict())
                full.update(heads[c])
                m.load_state_dict(full)
                pers_models[c] = m
            # communication accounting: only the backbone crosses clients (the head is never uploaded)
            comm_mb += (len(participants) * 2 * n_base_bytes) / 1e6

        elif args.algo == "ditto":
            # ditto: FedAvg for w, then each client re-anchors theta_i on w with
            # L_i(theta) + (lam/2)||theta - w||^2
            for cid in participants:
                c = clients[cid]
                m, n_i, loss_i = flcore.train_local(
                    global_model, *client_train[c], epochs=args.local_epochs, lr=args.lr,
                    batch_size=args.batch_size, seed=args.seed + t)
                states.append(m.state_dict())
                weights.append(n_i)
                losses.append(loss_i)
            global_model.load_state_dict(flcore.fedavg_aggregate(states, weights))
            comm_mb += (len(participants) * 2 * n_full_bytes) / 1e6
            for cid in participants:   # personalised model re-anchored on the new global w (never uploaded)
                c = clients[cid]
                theta[c], _, _ = flcore.train_local(
                    theta[c], *client_train[c], epochs=args.local_epochs, lr=args.lr,
                    batch_size=args.batch_size, seed=args.seed + t, prox_mu=args.lam)
            pers_models = theta

        # ---------- evaluation ----------
        if args.algo == "local":
            f1s, accs = personalized_metrics(pers_models, client_test)
            g_row = None
        else:
            f1s, accs = per_client_metrics(global_model, client_test)
            acc_g, f1_g = evaluate_model(global_model, X_te_all, y_te_all)
            g_row = (acc_g, f1_g)
        mean_f1, std_f1, worst_f1 = agg_stats(f1s)
        row = dict(round=t, participants=len(participants),
                   train_loss=float(np.mean(losses)) if losses else float("nan"),
                   acc=g_row[0] if g_row else float(np.mean(list(accs.values()))),
                   macro_f1=g_row[1] if g_row else mean_f1,
                   worst_client_f1=worst_f1, client_f1_std=std_f1, comm_mb=comm_mb)
        if pers_models is not None and args.algo != "local":
            pf1, pacc = personalized_metrics(pers_models, client_test)
            pmean, pstd, pworst = agg_stats(pf1)
            row.update(pers_mean_f1=pmean, pers_worst_f1=pworst, pers_std=pstd)
        else:
            row.update(pers_mean_f1=float("nan"), pers_worst_f1=float("nan"), pers_std=float("nan"))
        history.append(row)
        if t % 5 == 0 or t == args.rounds - 1:
            print(f"[{exp_id}] r{t:3d} part={len(participants)} acc={row['acc']:.4f} f1={row['macro_f1']:.4f} "
                  f"worst={row['worst_client_f1']:.4f} pers_worst={row['pers_worst_f1']:.4f}")

    elapsed = time.time() - t_start

    # ---------- save ----------
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    final = dict(
        global_metrics=dict(acc=history[-1]["acc"], macro_f1=history[-1]["macro_f1"],
                            worst_client_f1=history[-1]["worst_client_f1"],
                            client_f1_std=history[-1]["client_f1_std"]),
        per_client_global_f1=f1s, per_client_global_acc=accs,
        schedule=dict(mean_participants=float(np.mean([len(schedule[t]) for t in range(args.rounds)])),
                      min_participants=int(np.min([len(schedule[t]) for t in range(args.rounds)])),
                      max_participants=int(np.max([len(schedule[t]) for t in range(args.rounds)]))),
    )
    if pers_models is not None:
        pf1, pacc = personalized_metrics(pers_models, client_test)
        final["personalized_metrics"] = dict(mean_f1=agg_stats(pf1)[0], worst_f1=agg_stats(pf1)[2],
                                             std=agg_stats(pf1)[1])
        final["per_client_pers_f1"] = pf1
        torch.save({c: m.state_dict() for c, m in pers_models.items()}, out_dir / "pers_final.pt")

    # ---------- leave-one-client-out cold start (zero-shot / few-shot head / local training from scratch) ----------
    if holdout_train_idx is not None and args.algo != "local":
        import torch.nn.functional as F  # noqa: F401  (local import to keep it out of the main path)
        hx, hy = tensors(holdout_train_idx)
        tx, ty = tensors(holdout_test_idx)

        def _eval_cold(m):
            a, f1 = evaluate_model(m, tx, ty)
            return dict(acc=float(a), macro_f1=float(f1), n=int(len(ty)))

        cold = dict(holdout=args.holdout_client, n_train=int(len(holdout_train_idx)),
                    n_test=int(len(holdout_test_idx)), results={})
        cold["results"]["zero_shot"] = _eval_cold(global_model)
        rng_c = np.random.default_rng(args.seed * 100 + 99)
        for frac in [float(x) for x in args.cold_fracs.split(",") if x.strip()]:
            if frac <= 0:
                continue
            k = max(1, int(round(frac * len(holdout_train_idx))))
            pick = np.sort(rng_c.choice(len(holdout_train_idx), size=k, replace=False))
            sx, sy = hx[pick], hy[pick]
            # (a) shared backbone + head fitted on the k available samples only
            m = flcore.build_model(in_dim, N_CLASSES, seed=args.seed)
            m.load_state_dict(global_model.state_dict())
            for name, p in m.named_parameters():
                p.requires_grad = name.startswith(HEAD_PREFIX)
            opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                    lr=args.lr, weight_decay=1e-4)
            m.train()
            for _ in range(5):
                perm = torch.randperm(len(sx))
                for i0 in range(0, len(sx), args.batch_size):
                    b = perm[i0:i0 + args.batch_size]
                    opt.zero_grad()
                    loss = F.cross_entropy(m(sx[b]), sy[b])
                    loss.backward()
                    opt.step()
            m.eval()
            cold["results"][f"head_{frac:g}"] = _eval_cold(m)
            # (b) train the same k samples from scratch on the client
            m2 = flcore.build_model(in_dim, N_CLASSES, seed=args.seed)
            m2, _, _ = flcore.train_local(m2, sx, sy, epochs=5, lr=args.lr,
                                          batch_size=args.batch_size, seed=args.seed)
            m2.eval()
            cold["results"][f"local_{frac:g}"] = _eval_cold(m2)
        final["cold_start"] = cold

    # ---------- reference-head rule comparison (evaluation side; same backbone, only the definition
    # of the shared deliverable changes) ----------
    if args.ref_head_report and args.algo != "local":
        head_src = heads if args.algo in ("fedper", "fedperprox", "fedrep", "proposed") else None
        try:
            final["ref_head_variants"] = ref_head_rule_variants(
                global_model.state_dict(), head_src, participants, clients, client_test,
                (X_val, y_val) if X_val is not None else None,
                n_cls=n_cls, n_train=n_train, n_min=args.gate_min_n, min_classes=args.gate_min_classes,
                calib_steps=args.calib_steps, calib_lr=args.calib_lr, seed=args.seed, in_dim=in_dim,
                n_classes=N_CLASSES)
        except Exception as exc:  # a failed variant evaluation must not destroy the main results
            final["ref_head_variants"] = {"error": f"{type(exc).__name__}: {exc}"}
        final["ref_head_meta"] = dict(last_participants=[clients[int(p)] for p in participants],
                                      n_cls=n_cls, n_train=n_train)
    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(dict(vars(args), exp_id=exp_id, n_features=in_dim, n_clients=len(clients)), f, indent=2, ensure_ascii=False)
    if args.algo != "local":
        torch.save(global_model.state_dict(), out_dir / "model_final.pt")
    if args.algo == "proposed" and alpha_hist:
        keys = sorted({k for row in alpha_hist for k in row})
        with open(out_dir / "alpha_history.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in alpha_hist:
                w.writerow(row)
        with open(out_dir / "quality_stats.json", "w", encoding="utf-8") as f:
            json.dump(dict(Q=qstat, n_val=int(len(val_idx)), val_frac=args.val_frac), f, indent=2, ensure_ascii=False)
    if args.algo == "proposed" and args.support_gate:
        with open(out_dir / "gate_log.json", "w", encoding="utf-8") as f:
            json.dump(gate_hist, f, indent=2, ensure_ascii=False)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    rounds = [r["round"] for r in history]
    ax1.plot(rounds, [r["acc"] for r in history], "o-", label="Acc (global proto)", color="#2F6EA5")
    ax1.plot(rounds, [r["macro_f1"] for r in history], "s-", label="Macro-F1", color="#4C9F70")
    ax1.plot(rounds, [r["worst_client_f1"] for r in history], "^--", label="Worst-client F1", color="#C2453A")
    if pers_models is not None and args.algo != "local":
        ax1.plot(rounds, [r["pers_worst_f1"] for r in history], "d:", label="Worst-client F1 (personalized)", color="#7A4FA3")
    ax1.set_xlabel("Communication Round"); ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.02)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_title(f"{args.algo} on {args.dataset.upper()} ({args.partition}, rate={args.rate:g}, L={args.window}s)")
    fig.tight_layout(); fig.savefig(out_dir / "curves.png", dpi=150)

    ds_label = "UCI-HAR-feat" if args.dataset == "har" else f"UAH-DriveSet-feat-L{args.window}"
    append_record(RESEARCH_DIR / "results" / "experiment_records.csv", dict(
        exp_id=exp_id, date=today(), dataset=ds_label, seed=args.seed,
        num_clients=len(clients), participation_rate=args.rate,
        non_iid_level=NON_IID_DESC.get(args.partition, args.partition),
        model="MLP(64,32)", learning_rate=args.lr, local_epoch=args.local_epochs,
        communication_rounds=args.rounds,
        accuracy=f"{history[-1]['acc']:.4f}", macro_f1=f"{history[-1]['macro_f1']:.4f}",
        std=f"{history[-1]['client_f1_std']:.4f}", time_s=f"{elapsed:.1f}",
        communication_mb=f"{comm_mb:.3f}", commit=git_commit(REPO_ROOT),
        notes=f"algo={args.algo}; worstF1={history[-1]['worst_client_f1']:.4f}; "
              f"pers_worstF1={history[-1]['pers_worst_f1']}; mu={args.mu}; lam={args.lam}; mode={args.mode}; "
              f"mean_part={final['schedule']['mean_participants']:.2f}"
              + (f"; dwell={args.dwell}" if args.mode in ("markov", "shift") else "")
              + (f"; skipped_rounds={skipped_rounds}" if args.allow_empty_round else ""),
    ))
    print(f"[{exp_id}] done {elapsed:.1f}s | acc={history[-1]['acc']:.4f} f1={history[-1]['macro_f1']:.4f} "
          f"worst={history[-1]['worst_client_f1']:.4f} pers_worst={history[-1]['pers_worst_f1']}")


if __name__ == "__main__":
    main()
