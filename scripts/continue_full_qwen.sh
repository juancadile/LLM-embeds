#!/usr/bin/env bash
set -euo pipefail

RUN_DIR=${1:?usage: continue_full_qwen.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
MODEL=${MODEL:-qwen-qwen3-14b}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT to the configured output directory name}
LABEL_ROOT="$RUN_DIR/results/probe_experiments/$EXPERIMENT"

cd "$RUN_DIR"
trap 'continuation_status=$?; printf "%s\n" "$continuation_status" > continuation.exit' EXIT

while [[ ! -f labels.exit ]]; do
  wrapper=$(<labels.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "label wrapper stopped without writing labels.exit" >&2
    exit 3
  fi
  sleep 60
done

if [[ $(<labels.exit) != 0 ]]; then
  echo "label stage failed with status $(<labels.exit); extraction suppressed" >&2
  exit 4
fi

"$PYTHON" -m src.probe_gate \
  --labels "$LABEL_ROOT/$MODEL/labels.parquet" \
  --output "$LABEL_ROOT/$MODEL/label_gate.json" \
  --minimum 1000 --require knows

export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage extract \
  --model "$MODEL" > extraction.log 2>&1
