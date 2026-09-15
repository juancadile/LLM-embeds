#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: continue_after_primary_mdl.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
MODEL=${MODEL:-qwen-qwen3-14b}
EXPERIMENT=${EXPERIMENT:-knows_mdl_v5_balanced}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
MODEL_ROOT="$RUN_DIR/results/probe_experiments/$EXPERIMENT/$MODEL"

cd "$RUN_DIR"
trap 'followup_status=$?; printf "%s\n" "$followup_status" > mdl_followup.exit' EXIT

while [[ ! -f mdl_primary.exit ]]; do
  wrapper=$(<mdl_primary.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "primary MDL wrapper stopped without writing mdl_primary.exit" >&2
    exit 3
  fi
  sleep 60
done

if [[ $(<mdl_primary.exit) != 0 ]]; then
  echo "primary MDL stage failed with status $(<mdl_primary.exit); gate and sweep suppressed" >&2
  exit 4
fi

(
  cd mdl_gate_tool
  "$PYTHON" -m src.probe_mdl_gate \
    --cells-root "$MODEL_ROOT/cells/knows" \
    --output "$MODEL_ROOT/primary_mdl_gate.json" \
    --layer emb --layer p25 --layer p50 --layer p75 --layer final \
    --pool last --minimum-macro-f1 0.75 --minimum-shuffled-improvement 0.10
)

layers=()
for number in $(seq -w 1 39); do
  layers+=(--layer "block$number")
done

export PYTHONPATH=analysis_v2
export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" -P -m src.probe_experiments --config "$CONFIG" --stage mdl \
  --model "$MODEL" --predicate knows --pool last "${layers[@]}" \
  > mdl_all_layers.log 2>&1
