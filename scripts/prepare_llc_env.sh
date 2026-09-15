#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: prepare_llc_env.sh RUN_DIR}
BASE_PYTHON=${BASE_PYTHON:-/home/forestzhang001/venvs/llm-embeds/bin/python}
LLC_VENV=${LLC_VENV:-/home/forestzhang001/venvs/llm-embeds-llc-v2}

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > llc_environment.exit' EXIT

if [[ ! -x $LLC_VENV/bin/python ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$LLC_VENV" || overall=$?
fi
if [[ $overall -eq 0 && (! -f llc_environment.install.exit || $(<llc_environment.install.exit) != 0) ]]; then
  "$LLC_VENV/bin/python" -m pip install --disable-pip-version-check \
    -r requirements-llc.txt > llc_environment.install.log 2>&1 || overall=$?
fi
printf '%s\n' "$overall" > llc_environment.install.exit

if [[ $overall -eq 0 ]]; then
  CUDA_VISIBLE_DEVICES='' PYTHONPATH=analysis_v4 "$LLC_VENV/bin/python" -P - <<'PY' \
    > llc_logistic_validation.log 2>&1 || overall=$?
from pathlib import Path
from src.probe_llc import validate_regular_logistic
from src.probe_utils import atomic_json

result = validate_regular_logistic(2026)
atomic_json(result, Path("results/probe_experiments/knows_mdl_v5_balanced/llc_logistic_validation.json"))
if not result["passed"]:
    raise SystemExit("regular logistic d/2 validation failed")
PY
fi
printf '%s\n' "$overall" > llc_logistic_validation.exit
exit "$overall"
