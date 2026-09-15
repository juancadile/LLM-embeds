#!/usr/bin/env bash
set -uo pipefail

RUN_DIR=${1:?usage: continue_report.sh RUN_DIR}
PYTHON=${PYTHON:-/home/forestzhang001/venvs/llm-embeds-llc-v2/bin/python}
CONFIG=${CONFIG:-configs/knows_mdl.yaml}

cd "$RUN_DIR" || exit 1
overall=0
trap 'printf "%s\n" "$overall" > report.exit' EXIT

while [[ ! -f llc.exit ]]; do
  wrapper=$(<llc.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "LLC wrapper stopped without writing llc.exit" >&2
    overall=3
    exit "$overall"
  fi
  sleep 60
done

# A rejected LLC estimate is itself a reportable result. The report stage runs
# after terminal LLC status rather than requiring LLC exit zero.
export PYTHONPATH=analysis_v9
"$PYTHON" -P -m src.probe_experiments --config "$CONFIG" --stage report \
  > report.log 2>&1 || overall=$?
exit "$overall"
