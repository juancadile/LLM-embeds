#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: continue_llc.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds-llc-v2/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
EXPERIMENT=${EXPERIMENT:-knows_mdl_v5_balanced}
ROOT="$RUN_DIR/results/probe_experiments/$EXPERIMENT"

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > llc.exit' EXIT

wait_for_job() {
  local marker=$1
  local pid_file=$2
  while [[ ! -f $marker ]]; do
    local wrapper
    wrapper=$(<"$pid_file")
    if ! kill -0 "$wrapper" 2>/dev/null; then
      echo "$pid_file process stopped without writing $marker" >&2
      return 3
    fi
    sleep 60
  done
  [[ $(<"$marker") == 0 ]]
}

if ! wait_for_job llc_environment.exit llc_environment.pid; then
  echo "pinned LLC environment or logistic validation failed" >&2
  overall=4
  exit "$overall"
fi
if ! wait_for_job artifact_audit.exit artifact_audit.pid; then
  echo "ID/compression prerequisite grid or artifact audit failed" >&2
  overall=5
  exit "$overall"
fi

predicates=()
while IFS= read -r predicate; do
  [[ -n $predicate ]] && predicates+=(--predicate "$predicate")
done < <("$PYTHON" -P - "$ROOT/qwen-qwen3-14b/quality_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
for predicate, result in gate["predicates"].items():
    if result["comparative_claim_eligible"]:
        print(predicate)
PY
)

if [[ ${#predicates[@]} -eq 0 ]]; then
  echo "no Qwen3-14B predicates passed the comparative quality gate" >&2
  overall=6
  exit "$overall"
fi

export PYTHONPATH=analysis_v8
export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -P -m src.probe_experiments --config "$CONFIG" \
  --stage llc --model qwen-qwen3-14b "${predicates[@]}" > llc.log 2>&1 || overall=$?
exit "$overall"
