#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: continue_replication_data.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
EXPERIMENT=${EXPERIMENT:-knows_mdl_v5_balanced}
ROOT="$RUN_DIR/results/probe_experiments/$EXPERIMENT"

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > replication_data.exit' EXIT

while [[ ! -f id_compress_primary.exit ]]; do
  wrapper=$(<id_compress_primary.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "ID/compression wrapper stopped without writing id_compress_primary.exit" >&2
    overall=3
    exit "$overall"
  fi
  sleep 60
done
if [[ $(<id_compress_primary.exit) != 0 ]]; then
  echo "primary ID/compression failed; replication suppressed" >&2
  overall=4
  exit "$overall"
fi

export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_model() {
  local model_id=$1
  local slug=$2
  local prefix=$3
  local status=0

  "$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage labels \
    --model "$slug" > "$prefix.labels.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$prefix.labels.exit"
  if [[ $status -ne 0 ]]; then
    echo "$model_id labeling failed with status $status; extraction suppressed" >&2
    overall=1
    return
  fi

  local gate_status=0
  "$PYTHON" -m src.probe_gate --labels "$ROOT/$slug/labels.parquet" \
    --output "$ROOT/$slug/label_gate.json" --minimum 1000 \
    > "$prefix.gate.log" 2>&1 || gate_status=$?
  printf '%s\n' "$gate_status" > "$prefix.gate.exit"
  if [[ $gate_status -ne 0 ]]; then overall=1; fi

  local extraction_status=0
  "$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage extract \
    --model "$slug" > "$prefix.extraction.log" 2>&1 || extraction_status=$?
  printf '%s\n' "$extraction_status" > "$prefix.extraction.exit"
  if [[ $extraction_status -ne 0 ]]; then overall=1; fi
}

run_model 'Qwen/Qwen3-1.7B' 'qwen-qwen3-1.7b' 'replication_qwen17'
run_model 'meta-llama/Llama-3.1-8B-Instruct' \
  'meta-llama-llama-3.1-8b-instruct' 'replication_llama8b'

exit "$overall"
