#!/usr/bin/env bash
# E3: paired scenario bootstrap for the cross-predicate ordering.
# Usage: night_ordering.sh RUN_DIR LAYER DRAWS
set -uo pipefail

RUN_DIR=${1:?usage: night_ordering.sh RUN_DIR LAYER DRAWS}
LAYER=${2:?usage: night_ordering.sh RUN_DIR LAYER DRAWS}
DRAWS=${3:?usage: night_ordering.sh RUN_DIR LAYER DRAWS}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
EXPERIMENT=${EXPERIMENT:-knows_mdl_v5_balanced}
# TAG names a parallel run (markers, log, output dir); SEEDS restricts probe seeds.
TAG=${TAG:-}
SEEDS=${SEEDS:-}

cd "$RUN_DIR" || exit 1
status=0
trap 'printf "%s\n" "$status" > "ordering${TAG}.exit"' EXIT

export PYTHONPATH=analysis_v12
export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false

extra=()
[[ -n $SEEDS ]] && extra+=(--seeds "$SEEDS")
[[ -n $TAG ]] && extra+=(--output "results/probe_experiments/$EXPERIMENT/ordering_bootstrap/${LAYER}${TAG}")

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -P -m src.probe_ordering --config "$CONFIG" \
  --layer "$LAYER" --draws "$DRAWS" --probe linear "${extra[@]}" \
  > "ordering${TAG}.${LAYER}.log" 2>&1 || status=$?
exit "$status"
