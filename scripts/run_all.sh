#!/usr/bin/env bash
# The whole pipeline, including the always-fuse control twin.
#
#   bash scripts/run_all.sh              full run
#   bash scripts/run_all.sh quick        small and fast, for checking a change
#
# Each stage writes files, so you can stop after any of them and pick up later
# by calling the individual scripts.

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "quick" ]]; then
  N=8000; EPOCHS=60; SUFFIX="_quick"
else
  N=50000; EPOCHS=300; SUFFIX=""
fi

echo "=== 1. data ==============================================="
python scripts/01_generate_data.py --name "main${SUFFIX}" --n "$N"
python scripts/01_generate_data.py --name "twin${SUFFIX}" --n "$N" --head fused

echo "=== 2. train =============================================="
python scripts/02_train.py --data "main${SUFFIX}" --run "baseline${SUFFIX}" --epochs "$EPOCHS"
python scripts/02_train.py --data "twin${SUFFIX}" --run "twin${SUFFIX}"     --epochs "$EPOCHS"

echo "=== 3. analyse ============================================"
python scripts/03_analyze.py --run "baseline${SUFFIX}" --twin "twin${SUFFIX}"

echo "=== 4. figures ============================================"
python scripts/04_figures.py --run "baseline${SUFFIX}"

echo
echo "done -> results/baseline${SUFFIX}/"
