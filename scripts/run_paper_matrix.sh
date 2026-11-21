#!/usr/bin/env bash
# Rebuild the full experiment matrix of the manuscript.
# Existing runs are skipped, so the command can be resumed after interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
python code/experiments/v4_rerun.py \
  --groups main,har_main,wisdm,fedrep,fold2,fold3,har_refhead6,scale,avail,extra \
  --workers "${WORKERS:-6}"
