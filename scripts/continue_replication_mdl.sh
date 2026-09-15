#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: continue_replication_mdl.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
EXPERIMENT=${EXPERIMENT:-knows_mdl_v5_balanced}
ROOT="$RUN_DIR/results/probe_experiments/$EXPERIMENT"

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > replication_mdl.exit' EXIT

while [[ ! -f replication_data.exit ]]; do
  wrapper=$(<replication_data.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "replication-data wrapper stopped without writing replication_data.exit" >&2
    overall=3
    exit "$overall"
  fi
  sleep 60
done
if [[ $(<replication_data.exit) != 0 ]]; then
  echo "replication data failed; replication MDL suppressed" >&2
  overall=4
  exit "$overall"
fi

export PYTHONPATH=analysis_v5
export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

eligible_predicates() {
  local slug=$1
  local excluded=${2:-}
  "$PYTHON" -P - "$ROOT/$slug/label_gate.json" "$excluded" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
excluded = set(filter(None, sys.argv[2].split(',')))
for predicate, result in gate["predicates"].items():
    if result["eligible"] and predicate not in excluded:
        print(predicate)
PY
}

run_mdl() {
  local slug=$1
  local prefix=$2
  local excluded=${3:-}
  if [[ ! -f "$ROOT/$slug/label_gate.json" || ! -f "$ROOT/$slug/activations.sidecar.json" ]]; then
    echo "missing gate or activation sidecar for $slug" > "$prefix.mdl.log"
    printf '5\n' > "$prefix.mdl.exit"
    overall=1
    return
  fi
  local predicates=()
  while IFS= read -r predicate; do
    [[ -n $predicate ]] && predicates+=(--predicate "$predicate")
  done < <(eligible_predicates "$slug" "$excluded")
  if [[ ${#predicates[@]} -eq 0 ]]; then
    echo "no class-eligible predicates for $slug" > "$prefix.mdl.log"
    printf '0\n' > "$prefix.mdl.exit"
    return
  fi
  local status=0
  "$PYTHON" -P -m src.probe_experiments --config "$CONFIG" --stage mdl \
    --model "$slug" --pool last "${predicates[@]}" > "$prefix.mdl.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$prefix.mdl.exit"
  if [[ $status -ne 0 ]]; then overall=1; fi
}

# KNOWS already has a complete Qwen3-14B layer sweep.
run_mdl 'qwen-qwen3-14b' 'primary_other_predicates' 'knows'
run_mdl 'qwen-qwen3-1.7b' 'replication_qwen17' ''
run_mdl 'meta-llama-llama-3.1-8b-instruct' 'replication_llama8b' ''

exit "$overall"
