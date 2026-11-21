# Reproducibility package

Code and archived results for

> **Personalised Federated Learning under Intermittent Participation:
> A Split-Level Leakage-Free Evaluation of Decoupled Heads**

The paper asks when decoupled (personalised) heads help under intermittent
client participation. It contributes a leakage-free, split-level evaluation on
three datasets (UCI HAR, UAH-DriveSet, WISDM 1.1), a conditional account of the
head effect, and an explicit reference-head protocol for the shared deliverable.

The repository mirrors the run identifiers used in the manuscript: every table
and figure can be traced to per-seed `config.json` / `final_metrics.json` files
under `results/baselines/`.

---

## 1. Layout

```
code/federated/      runner.py, flcore.py, proposed.py  (all methods)
code/common/         record.py                          (experiment ledger)
code/data/           preprocessing and partition builders
code/experiments/    v4_rerun.py                        (main matrix driver)

data/processed/      derived features and partition JSONs
data/processed/*/partitions/   exact client splits used in the paper

results/baselines/   one directory per run: config.json, final_metrics.json,
                     history.csv (no model checkpoints)
results/experiment_records.csv  flat ledger of all runs

scripts/make_fig6.py  regenerates Fig. 6 from the archived per-seed metrics
summarize.py          collects all runs into results/summary.csv
```

No raw datasets, model checkpoints or intermediate `.pt` files are included.

---

## 2. Environment

Python 3.10+ with NumPy, PyTorch (CPU is sufficient) and Matplotlib:

```bash
python -m pip install -r requirements.txt
```

All commands below are run from the repository root.

Quick check that the archived results are readable and match the paper:

```bash
python summarize.py
```

This writes `results/summary.csv` and prints the HAR full-participation
sanity check (six seeds):

| Method | mean client F1 | worst client F1 |
|---|---|---|
| Local | 0.952 | 0.787 |
| FedAvg | 0.840 | 0.523 |
| FedPer | 0.948 | 0.734 |
| SA-PFA | 0.951 | 0.766 |

---

## 3. Data

The derived feature archives used by every run are included, together with the
exact partition files, so **no preprocessing is required to reproduce the
reported numbers or to retrain**.

| Dataset | Feature archive | Clients | Classes |
|---|---|---|---|
| UCI HAR | `data/processed/HAR/features_har.npz` | 30 | 6 |
| UAH-DriveSet (30 s) | `data/processed/UAH/features_L30.npz` | 6 | 3 |
| WISDM 1.1 | `data/processed/WISDM/features_wisdm.npz` | 51 | 6 |

Partition files follow the manuscript:

* HAR: `partition_har_natural_block.json` (leakage-free block split with a
  two-window gap; the paper's main HAR protocol),
  `partition_har_dirichlet_a0.05.json` (label-skewed training pool, same test
  sets), `partition_har_dirichlet_a0.1.json`.
* WISDM: `partition_wisdm_natural.json`, `partition_wisdm_dirichlet_a0.05.json`.
* UAH: `partition_natural.json` (trip-level split), `partition_natural_fold2.json`,
  `partition_natural_fold3.json` (holdout folds), `partition_natural_k12.json`,
  `partition_natural_k18.json` (client-scale tests),
  `partition_dirichlet_a0.05.json` (skewed reference-head check).
  `partitions/legacy_window_split/` contains the window-level partitions used
  only for the leakage comparison in Table 1.

To rebuild the features from the public raw data:

```bash
# UAH-DriveSet: unzip UAH-DRIVESET-v1 under data/raw/UAH-DriveSet/
python code/data/preprocess_uah.py
python code/data/partition.py --window 30

# WISDM 1.1: place the raw archive as expected by the script
python code/data/preprocess_wisdm.py

# HAR: features_har.npz is archived; make_har_partitions.py rebuilds the
# leakage-free and Dirichlet partitions from it
python code/data/make_har_partitions.py
```

* UAH-DriveSet: <http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/>
* UCI HAR: <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
* WISDM 1.1: <https://www.cis.fordham.edu/wisdm/dataset.php>

For the leakage comparison only, copy
`data/processed/UAH/partitions/legacy_window_split/partition_natural.json`
to `data/processed/UAH/partitions/partition_natural_window.json` and pass
`--partition natural_window`.

---

## 4. Reproducing the tables and figures without retraining

Every reported value is stored per seed in
`results/baselines/<exp-id>/final_metrics.json`. Metric conventions:

* `per_client_global_f1` — shared model evaluated on each client's test set
  (FedAvg, FedProx, Local, and the global protocol of every method).
* `per_client_pers_f1` — each client's own head (FedPer, FedRep, pFedMe, Ditto,
  SA-PFA, ablAQ).
* mean client F1 = mean over clients within a seed, then mean over seeds.
* worst-client F1 = minimum over clients within a seed, then mean over seeds.
* standard deviations are population values over seeds (`ddof=0`).

Run id prefixes map to the manuscript as follows:

| Manuscript item | Result directories |
|---|---|
| Table 2, Fig. 3–5 (HAR main) | `main-har-*` |
| Table 3 (structure vs modifiers, HAR) | `main-har-{fedavg,fedper,ablAQ,proposed}-*` |
| Table 4 (reference-head rules, HAR α=0.05) | `main-har-*-a005-r100-*` |
| Table 5, Fig. 6 (UAH main) | `main-*` (UAH, no `har`/`wisdm` in the id) |
| Table 6 (availability / correlated scheduling) | `v4-avail-*`, `v4-markov-*`, `v4-shift-*` |
| Table 7 (client scale, cold start) | `v4-scale-*`, `v4-cold-*` |
| Table 8 (WISDM) | `main-wisdm-*`, `v4-wisdm-skew-*` |
| Table 9 (per-driver counts) | derived from `features_L30.npz` |
| Table 10 (hyperparameters) | `config.json` of any run |
| Holdout folds (Section 5.5) | `v4-fold2-*`, `v4-fold3-*` |
| Proximal / tuning controls | `v4-noprox-*`, `v4-mu*`, `v4-lam*`, `v4-pfedme-*`, `v4-abl-*` |

The seed policy matches Section 5.1: six seeds by default; three for Local on
UCI HAR, WISDM, the scale check and the fold-2/3 checks; six for Local on
UAH-DriveSet; ten for FedAvg, FedPer and SA-PFA on UAH (seeds 1–6/7–10).

Regenerate Fig. 6 (writes `scripts/Fig6.pdf`):

```bash
python scripts/make_fig6.py
```

---

## 5. Retraining

`code/experiments/v4_rerun.py` is the driver used for the paper. It skips runs
whose `final_metrics.json` already exists, so it is safe to resume. A two-run
smoke test is provided in `scripts/run_smoke.sh`.

```bash
# one group at a time, e.g. the HAR main matrix
python code/experiments/v4_rerun.py --groups har_main --workers 6

# the full paper matrix
python code/experiments/v4_rerun.py \
  --groups main,har_main,wisdm,fedrep,fold2,fold3,har_refhead6,scale,avail,extra \
  --workers 6
```

Cold-start runs are not part of `v4_rerun.py`; run them directly:

```bash
python code/federated/runner.py --algo fedavg --dataset uah \
  --partition natural --holdout-client client_1 --rate 1.0 --rounds 80 \
  --seed 1 --exp-id v4-cold-fedavg-client_1-s1
```

Add `--no-c` for `--algo proposed` (this is the ablation that removes the
contribution term and is the default configuration reported for SA-PFA).

Every run writes `config.json`, `history.csv`, `final_metrics.json` and a
`curves.png` under `results/baselines/<exp-id>/`, and appends a row to
`results/experiment_records.csv`.

---

## 6. Notes

* The paper reports results only for the methods listed in Section 5.1. The
  SA-AHI pilot (an adaptive head-interpolation probe) is **not** part of this
  package and is not used by any manuscript table or figure.
* `results/baselines/` contains the runs referenced by the manuscript. Archived
  window-level baselines (Table 1) are not shipped; they can be regenerated with
  the legacy partition as described in Section 3.
* Released under the MIT License; see `LICENSE`.
  Repository URL: <https://github.com/xs2822835114/sa-pfa-repro>.

---

## 7. Citation

```bibtex
@article{liu2026personalised,
  title   = {Personalised Federated Learning under Intermittent Participation:
             A Split-Level Leakage-Free Evaluation of Decoupled Heads},
  author  = {Liu, Kehao and Guo, Hongfei and Zhao, Yulong},
  journal = {Applied Intelligence},
  year    = {2026}
}
```
