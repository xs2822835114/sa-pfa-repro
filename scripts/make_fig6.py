# -*- coding: utf-8 -*-
"""Regenerate Fig. 6 with driver labels D1-D6 (the original PDF used D0-D5).

Left: paired difference in worst-client F1 against FedAvg and FedPer at three
uniform rates (ten seeds; 95% paired t intervals).
Right: per-driver deployment F1 at full participation (ten seeds, mean +- sd),
drivers ordered by FedAvg's F1, hardest first.
"""
import json
import os
import statistics as st
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results' / 'baselines'

NAVY = '#1F4E6B'
ORANGE = '#C25A0F'
GRAY = '#5B6770'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8.6,
    'axes.labelsize': 9.2,
    'xtick.labelsize': 8.4,
    'ytick.labelsize': 8.4,
    'axes.linewidth': 0.8,
})


def per_client(algo, rate, seed):
    p = os.path.join(BASE, f'main-{algo}-natural-{rate}-s{seed}', 'final_metrics.json')
    d = json.load(open(p))
    key = 'per_client_global_f1' if algo == 'fedavg' else 'per_client_pers_f1'
    return d[key]


def worst(algo, rate, seed):
    pc = per_client(algo, rate, seed)
    return min(pc.values())


RATES = [('r20', '20%'), ('r60', '60%'), ('r100', '100%')]
SEEDS = list(range(1, 11))
T95 = 2.262157  # two-sided 95%, df = 9


def paired(a, b):
    diff = [x - y for x, y in zip(a, b)]
    m = st.mean(diff)
    half = T95 * st.stdev(diff) / math.sqrt(len(diff))
    return m, half


fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(8.50, 3.45), gridspec_kw={'width_ratios': [1.0, 1.08], 'wspace': 0.27})

# ---- left panel -----------------------------------------------------------
rates = [r[0] for r in RATES]
xs = np.arange(len(rates))
# per-rate paired diffs: proposed - baseline, computed seed by seed
res = {}
for a in ('fedavg', 'fedper'):
    res[a] = [paired([worst('proposed', r, s) for s in SEEDS],
                     [worst(a, r, s) for s in SEEDS]) for r in rates]

ax1.axhline(0.0, color='#999999', lw=0.8, zorder=1)
for a, col, mk, ls, lab in (('fedavg', NAVY, 'o', '-', 'vs FedAvg'),
                            ('fedper', ORANGE, 's', '--', 'vs FedPer')):
    m = [v[0] for v in res[a]]
    h = [v[1] for v in res[a]]
    ax1.errorbar(xs, m, yerr=h, color=col, marker=mk, ms=5.2, lw=1.5,
                 ls=ls, capsize=3.0, elinewidth=1.0, zorder=3, label=lab)

ax1.set_xticks(xs)
ax1.set_xticklabels([r[1] for r in RATES])
ax1.set_xlim(-0.42, 2.42)
ax1.set_ylim(-0.105, 0.125)
ax1.set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
ax1.set_xlabel(r'Participation rate $\rho$')
ax1.set_ylabel(r'$\Delta$ worst-client F1 (UAH)')
ax1.grid(axis='y', color='#DDDDDD', lw=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.spines[['top', 'right']].set_visible(False)

ax1.annotate(f'{res["fedavg"][2][0]:+.3f}', (xs[2], res['fedavg'][2][0]),
             textcoords='offset points', xytext=(-4, 8), ha='right',
             color=NAVY, fontsize=8.0)
ax1.annotate(f'{res["fedper"][2][0]:+.3f}', (xs[2], res['fedper'][2][0]),
             textcoords='offset points', xytext=(-5, -24), ha='right',
             color=ORANGE, fontsize=8.0)
ax1.annotate(f'{res["fedavg"][1][0]:+.3f}', (xs[1], res['fedavg'][1][0]),
             textcoords='offset points', xytext=(8, 11), ha='left',
             color=NAVY, fontsize=8.0)
ax1.annotate(f'{res["fedper"][1][0]:+.3f}', (xs[1], res['fedper'][1][0]),
             textcoords='offset points', xytext=(3, -12), ha='left',
             color=ORANGE, fontsize=8.0)
ax1.annotate(f'{res["fedavg"][0][0]:+.3f}', (xs[0], res['fedavg'][0][0]),
             textcoords='offset points', xytext=(3, -13), ha='left',
             color=NAVY, fontsize=8.0)
ax1.annotate(f'{res["fedper"][0][0]:+.3f}', (xs[0], res['fedper'][0][0]),
             textcoords='offset points', xytext=(3, 6), ha='left',
             color=ORANGE, fontsize=8.0)

ax1.legend(frameon=False, fontsize=8.0, ncol=2, loc='upper center',
           bbox_to_anchor=(0.5, -0.20), handlelength=2.2, columnspacing=2.0)

# ---- right panel ----------------------------------------------------------
clients = sorted(per_client('fedavg', 'r100', 1).keys())
stats = {}
for a in ('fedavg', 'fedper', 'proposed'):
    vals = {c: [] for c in clients}
    for s in SEEDS:
        for c, v in per_client(a, 'r100', s).items():
            vals[c].append(v)
    stats[a] = {c: (st.mean(v), st.stdev(v)) for c, v in vals.items()}

order = sorted(clients, key=lambda c: stats['fedavg'][c][0])  # hardest first
labels = [f'D{int(c.split("_")[1]) + 1}' for c in order]

xpos = np.arange(len(order))
w = 0.26
for k, (a, col, lab) in enumerate((('fedavg', GRAY, 'FedAvg'),
                                   ('fedper', ORANGE, 'FedPer'),
                                   ('proposed', NAVY, 'SA-PFA'))):
    m = [stats[a][c][0] for c in order]
    e = [stats[a][c][1] for c in order]
    ax2.bar(xpos + (k - 1) * w, m, width=w, color=col, label=lab,
            yerr=e, error_kw=dict(ecolor='#333333', elinewidth=0.8, capsize=2.2),
            zorder=3)

ax2.set_xticks(xpos)
ax2.set_xticklabels(labels, fontsize=8.6)
ax2.set_ylim(0, 1.0)
ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_ylabel('Deployment F1')
ax2.set_xlabel("Drivers ordered by FedAvg F1 (hardest first)")
ax2.grid(axis='y', color='#DDDDDD', lw=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend(frameon=False, fontsize=8.0, ncol=3, loc='upper center',
           bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.3,
           handletextpad=0.5)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fig6.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
print('saved', out)
for a in ('fedavg', 'fedper'):
    print(a, [f'{m:+.3f}+-{h:.3f}' for m, h in res[a]])
print('order', labels)
