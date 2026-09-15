#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: continue_quality_gates.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
EXPERIMENT=${EXPERIMENT:-knows_mdl_v5_balanced}
ROOT="$RUN_DIR/results/probe_experiments/$EXPERIMENT"

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > quality_gates.exit' EXIT

while [[ ! -f replication_mdl.exit ]]; do
  wrapper=$(<replication_mdl.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "replication-MDL wrapper stopped without writing replication_mdl.exit" >&2
    overall=3
    exit "$overall"
  fi
  sleep 60
done
if [[ $(<replication_mdl.exit) != 0 ]]; then
  echo "replication MDL failed; quality gates suppressed" >&2
  overall=4
  exit "$overall"
fi

run_gate() {
  local slug=$1
  local prefix=$2
  local model_root="$ROOT/$slug"
  local status=0
  if [[ ! -f "$model_root/label_gate.json" ]]; then
    echo "missing label gate for $slug" > "$prefix.quality_gate.log"
    printf '5\n' > "$prefix.quality_gate.exit"
    overall=1
    return
  fi
  (
    cd quality_gate_tool
    "$PYTHON" -m src.probe_quality_gate --model-root "$model_root" \
      --output "$model_root/quality_gate.json" \
      --layer emb --layer p25 --layer p50 --layer p75 --layer final \
      --pool last --minimum-macro-f1 0.75
  ) > "$prefix.quality_gate.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$prefix.quality_gate.exit"
  if [[ $status -ne 0 ]]; then overall=1; fi
}

run_gate 'qwen-qwen3-14b' 'primary'
run_gate 'qwen-qwen3-1.7b' 'replication_qwen17'
run_gate 'meta-llama-llama-3.1-8b-instruct' 'replication_llama8b'

exit "$overall"
