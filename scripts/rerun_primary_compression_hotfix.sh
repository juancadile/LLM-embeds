#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: rerun_primary_compression_hotfix.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}

cd "$RUN_DIR" || exit 1
status=0
trap 'printf "%s\n" "$status" > compression_primary_hotfix.exit' EXIT
export PYTHONPATH=analysis_v5
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

"$PYTHON" -P -m src.probe_experiments --config "$CONFIG" --stage compress \
  --model qwen-qwen3-14b --predicate knows --layer block29 --pool last \
  > compression_primary_hotfix.log 2>&1 || status=$?
exit "$status"
