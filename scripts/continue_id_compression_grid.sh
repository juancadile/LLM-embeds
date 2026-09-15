#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: continue_id_compression_grid.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}
EXPERIMENT=${EXPERIMENT:-knows_mdl_v5_balanced}
ROOT="$RUN_DIR/results/probe_experiments/$EXPERIMENT"

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > id_compression_grid.exit' EXIT

while [[ ! -f quality_gates.exit ]]; do
  wrapper=$(<quality_gates.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "quality-gate wrapper stopped without writing quality_gates.exit" >&2
    overall=3
    exit "$overall"
  fi
  sleep 60
done
if [[ $(<quality_gates.exit) != 0 ]]; then
  echo "quality gates failed; ID/compression grid suppressed" >&2
  overall=4
  exit "$overall"
fi

while [[ ! -f compression_primary_hotfix.exit ]]; do
  wrapper=$(<compression_primary_hotfix.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "primary compression hotfix stopped without writing its exit marker" >&2
    overall=5
    exit "$overall"
  fi
  sleep 60
done
if [[ $(<compression_primary_hotfix.exit) != 0 ]]; then
  echo "primary compression hotfix failed; grid suppressed" >&2
  overall=6
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
  "$PYTHON" -P - "$ROOT/$slug/quality_gate.json" <<'PY'
import json, sys
gate = json.load(open(sys.argv[1]))
for predicate, result in gate["predicates"].items():
    if result["comparative_claim_eligible"]:
        print(predicate)
PY
}

complete_table() {
  [[ -f $1 || -f ${1%.parquet}.tsv ]]
}

run_cell() {
  local slug=$1
  local predicate=$2
  local layer=$3
  local prefix="grid.${slug}.${predicate}.${layer}"
  local cell="$ROOT/$slug/cells/$predicate/$layer.last"
  local id_status=0 compression_status=0 id_pid='' compression_pid=''

  if complete_table "$cell/intrinsic_dimension.parquet"; then
    printf '0\n' > "$prefix.id.exit"
  else
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" -P -m src.probe_experiments --config "$CONFIG" \
      --stage id --model "$slug" --predicate "$predicate" --layer "$layer" --pool last \
      > "$prefix.id.log" 2>&1 &
    id_pid=$!
  fi

  if complete_table "$cell/compression.parquet"; then
    printf '0\n' > "$prefix.compression.exit"
  else
    CUDA_VISIBLE_DEVICES='' "$PYTHON" -P -m src.probe_experiments --config "$CONFIG" \
      --stage compress --model "$slug" --predicate "$predicate" --layer "$layer" --pool last \
      > "$prefix.compression.log" 2>&1 &
    compression_pid=$!
  fi

  if [[ -n $id_pid ]]; then
    wait "$id_pid" || id_status=$?
    printf '%s\n' "$id_status" > "$prefix.id.exit"
  fi
  if [[ -n $compression_pid ]]; then
    wait "$compression_pid" || compression_status=$?
    printf '%s\n' "$compression_status" > "$prefix.compression.exit"
  fi
  if [[ $id_status -ne 0 || $compression_status -ne 0 ]]; then overall=1; fi
}

ensure_primary_block29_mdl() {
  local predicates=()
  while IFS= read -r predicate; do
    [[ -z $predicate || $predicate == knows ]] && continue
    if ! complete_table "$ROOT/qwen-qwen3-14b/cells/$predicate/block29.last/mdl.mlp.parquet"; then
      predicates+=(--predicate "$predicate")
    fi
  done < <(eligible_predicates qwen-qwen3-14b)
  if [[ ${#predicates[@]} -eq 0 ]]; then
    printf '0\n' > grid.primary_block29_mdl.exit
    return
  fi
  local status=0
  "$PYTHON" -P -m src.probe_experiments --config "$CONFIG" --stage mdl \
    --model qwen-qwen3-14b --layer block29 --pool last "${predicates[@]}" \
    > grid.primary_block29_mdl.log 2>&1 || status=$?
  printf '%s\n' "$status" > grid.primary_block29_mdl.exit
  if [[ $status -ne 0 ]]; then overall=1; fi
}

run_model() {
  local slug=$1
  local layers=(emb p25 p50 p75 final)
  if [[ $slug == qwen-qwen3-14b ]]; then layers+=(block29); fi
  if [[ ! -f $ROOT/$slug/quality_gate.json ]]; then
    echo "missing quality gate for $slug" >&2
    overall=1
    return
  fi
  while IFS= read -r predicate; do
    [[ -z $predicate ]] && continue
    for layer in "${layers[@]}"; do
      run_cell "$slug" "$predicate" "$layer"
    done
  done < <(eligible_predicates "$slug")
}

ensure_primary_block29_mdl
run_model qwen-qwen3-14b
run_model qwen-qwen3-1.7b
run_model meta-llama-llama-3.1-8b-instruct

exit "$overall"
