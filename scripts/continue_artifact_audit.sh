#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: continue_artifact_audit.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds-llc-v2/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
EXPERIMENT=${EXPERIMENT:-knows_mdl_v5_balanced}
ROOT="$RUN_DIR/results/probe_experiments/$EXPERIMENT"

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > artifact_audit.exit' EXIT

while [[ ! -f id_compression_grid.exit ]]; do
  wrapper=$(<id_compression_grid.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "ID/compression wrapper stopped without writing its exit marker" >&2
    overall=3
    exit "$overall"
  fi
  sleep 60
done
if [[ $(<id_compression_grid.exit) != 0 ]]; then
  echo "ID/compression grid failed; artifact audit suppressed" >&2
  overall=4
  exit "$overall"
fi

export PYTHONPATH=analysis_v6

eligible_predicates() {
  "$PYTHON" -P - "$ROOT/$1/quality_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
for predicate, result in gate["predicates"].items():
    if result["comparative_claim_eligible"]:
        print(predicate)
PY
}

audit_model() {
  local slug=$1
  local layers=(emb p25 p50 p75 final)
  if [[ $slug == qwen-qwen3-14b ]]; then layers+=(block29); fi
  while IFS= read -r predicate; do
    [[ -z $predicate ]] && continue
    for layer in "${layers[@]}"; do
      local cell="$ROOT/$slug/cells/$predicate/$layer.last"
      local output="$cell/artifact_audit.json"
      local status=0
      "$PYTHON" -P -m src.probe_artifact_audit --config "$CONFIG" \
        --cell "$cell" --output "$output" \
        > "audit.${slug}.${predicate}.${layer}.log" 2>&1 || status=$?
      printf '%s\n' "$status" > "audit.${slug}.${predicate}.${layer}.exit"
      if [[ $status -ne 0 ]]; then overall=1; fi
    done
  done < <(eligible_predicates "$slug")
}

audit_model qwen-qwen3-14b
audit_model qwen-qwen3-1.7b
audit_model meta-llama-llama-3.1-8b-instruct
exit "$overall"
