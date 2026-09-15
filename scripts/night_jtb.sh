#!/usr/bin/env bash
# E2: JTB composition. Usage: night_jtb.sh RUN_DIR LAYER
set -uo pipefail

RUN_DIR=${1:?usage: night_jtb.sh RUN_DIR LAYER}
LAYER=${2:?usage: night_jtb.sh RUN_DIR LAYER}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}

cd "$RUN_DIR" || exit 1
status=0
trap 'printf "%s\n" "$status" > jtb.exit' EXIT

export PYTHONPATH=analysis_v10
export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -P -m src.probe_jtb --config "$CONFIG" --layer "$LAYER" \
  > "jtb.${LAYER}.log" 2>&1 || status=$?
exit "$status"
