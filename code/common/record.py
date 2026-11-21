"""Experiment bookkeeping: append one run's key settings and results to a shared CSV."""
from __future__ import annotations

import csv
import datetime as _dt
import subprocess
from pathlib import Path

COLUMNS = [
    "exp_id", "date",
    "dataset", "seed", "num_clients", "participation_rate", "non_iid_level",
    "model", "learning_rate", "local_epoch", "communication_rounds",
    "accuracy", "macro_f1", "std", "time_s", "communication_mb",
    "commit", "notes",
]


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out or "-"
    except Exception:
        return "-"


def append_record(csv_path: Path, row: dict) -> None:
    """Write one row in COLUMNS order; missing fields stay empty; an unexpected header is an error."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    if exists:
        with open(csv_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            raise ValueError(f"experiment record header does not match the expected columns: {csv_path}")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in COLUMNS})


def today() -> str:
    return _dt.date.today().isoformat()
