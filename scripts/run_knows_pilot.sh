#!/usr/bin/env bash
set -euo pipefail
trap 'pilot_status=$?; printf "%s\n" "$pilot_status" > pilot.exit' EXIT
export TORCH_DISABLE_NATIVE_JIT=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON=/home/forestzhang001/venvs/llm-embeds/bin/python
CONFIG=${1:-configs/knows_pilot.yaml}
"$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage labels
"$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage extract
"$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage mdl --layer final --pool last
"$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage id --layer final --pool last
"$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage compress --layer final --pool last
"$PYTHON" -m src.probe_experiments --config "$CONFIG" --stage report
