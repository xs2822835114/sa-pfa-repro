#!/usr/bin/env bash
# Two-run smoke test: one UAH and one HAR run at full participation.
set -euo pipefail
cd "$(dirname "$0")/.."
python code/federated/runner.py --algo fedavg --dataset uah --partition natural \
  --rate 1.0 --rounds 80 --seed 1 --exp-id smoke-uah-fedavg-natural-r100-s1
python code/federated/runner.py --algo fedper --dataset har --partition natural_block \
  --rate 1.0 --rounds 80 --seed 1 --exp-id smoke-har-fedper-natural_block-r100-s1
