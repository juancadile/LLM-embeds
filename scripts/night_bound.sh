#!/usr/bin/env bash
# E1: bound ladder. Usage: night_bound.sh RUN_DIR PREDICATE LAYER
set -uo pipefail

RUN_DIR=${1:?usage: night_bound.sh RUN_DIR PREDICATE LAYER}
PREDICATE=${2:?usage: night_bound.sh RUN_DIR PREDICATE LAYER}
LAYER=${3:?usage: night_bound.sh RUN_DIR PREDICATE LAYER}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}

cd "$RUN_DIR" || exit 1
status=0
trap 'printf "%s\n" "$status" > bound.exit' EXIT

export PYTHONPATH=analysis_v11
export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -P -m src.probe_bound --config "$CONFIG" \
  --predicate "$PREDICATE" --layer "$LAYER" > "bound.${PREDICATE}.${LAYER}.log" 2>&1 || status=$?
exit "$status"
