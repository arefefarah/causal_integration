#!/usr/bin/env bash
# The whole pipeline for one config, including the always-fuse control twin.
#
#   bash scripts/run_all.sh                                    default config
#   bash scripts/run_all.sh --quick                            small and fast
#   bash scripts/run_all.sh --config configs/wide_rf.yaml      a variant
#   bash scripts/run_all.sh --config configs/wide_rf.yaml --quick
#   bash scripts/run_all.sh --config configs/wide_rf.yaml --name rf8 --epochs 500
#
# The run is named after the config file, so variants land in their own folders
# and never overwrite each other. Encoding and generative parameters reach the
# network through the dataset, so changing them means stage 1 has to rerun --
# which is what this script does anyway.
#
# Each stage writes files, so you can stop after any of them and pick up later
# with the individual scripts.

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/default.yaml"
NAME=""
N=""
EPOCHS=""
QUICK=0

need() { [[ -n "${2:-}" ]] || { echo "$1 needs a value" >&2; exit 2; }; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)  need "$1" "${2:-}"; CONFIG="$2"; shift 2 ;;
    --name)    need "$1" "${2:-}"; NAME="$2";   shift 2 ;;
    --n)       need "$1" "${2:-}"; N="$2";      shift 2 ;;
    --epochs)  need "$1" "${2:-}"; EPOCHS="$2"; shift 2 ;;
    --quick)   QUICK=1; shift ;;
    quick)     QUICK=1; shift ;;               # old positional form, still works
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -f "$CONFIG" ]] || { echo "no such config: $CONFIG" >&2; exit 2; }

# Default name = the config's filename, so results/ mirrors configs/.
# configs/default.yaml is the exception: its run is called "baseline".
if [[ -z "$NAME" ]]; then
  NAME="$(basename "$CONFIG" .yaml)"
  [[ "$NAME" == "default" ]] && NAME="baseline"
fi

if [[ "$QUICK" == 1 ]]; then
  NAME="${NAME}_quick"
  : "${N:=8000}"
  : "${EPOCHS:=60}"
fi

# Anything still unset is left to the config file itself. These are deliberately
# plain strings rather than arrays and are passed unquoted below: macOS ships
# bash 3.2, where expanding an empty array under `set -u` is an error. The values
# are numbers, so word splitting is safe here.
N_ARG=""
EPOCH_ARG=""
[[ -n "$N" ]] && N_ARG="--n $N"
[[ -n "$EPOCHS" ]] && EPOCH_ARG="--epochs $EPOCHS"

echo "config : $CONFIG"
echo "name   : $NAME"
echo

echo "=== 1. data ==============================================="
python scripts/01_generate_data.py --config "$CONFIG" --name "$NAME"        $N_ARG
python scripts/01_generate_data.py --config "$CONFIG" --name "${NAME}_twin" $N_ARG --head fused

echo "=== 2. train =============================================="
python scripts/02_train.py --data "$NAME"        --run "$NAME"        $EPOCH_ARG
python scripts/02_train.py --data "${NAME}_twin" --run "${NAME}_twin" $EPOCH_ARG

echo "=== 3. analyse ============================================"
python scripts/03_analyze.py --run "$NAME" --twin "${NAME}_twin"

echo "=== 4. figures ============================================"
python scripts/04_figures.py --run "$NAME"

echo
echo "done -> results/$NAME/"
