#!/usr/bin/env bash
# E4: depth profile for the four non-KNOWS predicates on already-extracted layers.
# Usage: night_depth_profile.sh RUN_DIR LAYER [LAYER...]
set -uo pipefail

RUN_DIR=${1:?usage: night_depth_profile.sh RUN_DIR LAYER [LAYER...]}
shift
LAYERS=("$@")
[[ ${#LAYERS[@]} -gt 0 ]] || { echo "at least one layer is required" >&2; exit 2; }
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
PREDICATES=(believes justified true lucky_guessed)

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > depth_profile.exit' EXIT

export PYTHONPATH=analysis_v9
export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false

# Layer-major so an interrupted run still leaves every predicate at the same depths.
for layer in "${LAYERS[@]}"; do
  for predicate in "${PREDICATES[@]}"; do
    marker="depth.${predicate}.${layer}.exit"
    [[ -f $marker && $(<"$marker") == 0 ]] && continue
    status=0
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -P -m src.probe_experiments --config "$CONFIG" \
      --stage mdl --model qwen-qwen3-14b --predicate "$predicate" --layer "$layer" --pool last \
      > "depth.${predicate}.${layer}.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "$marker"
    if [[ $status -ne 0 ]]; then overall=1; fi
  done
done
exit "$overall"
