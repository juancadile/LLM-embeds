#!/usr/bin/env bash
# Waits for the running depth slice, then continues with the remaining layers.
# Usage: night_depth_chain.sh RUN_DIR LAYER [LAYER...]
set -uo pipefail

RUN_DIR=${1:?usage: night_depth_chain.sh RUN_DIR LAYER [LAYER...]}
shift
cd "$RUN_DIR" || exit 1

while [[ ! -f depth_profile.exit ]]; do
  wrapper=$(<depth_profile.pid)
  if ! kill -0 "$wrapper" 2>/dev/null; then
    echo "depth wrapper stopped without writing its marker" >&2
    exit 3
  fi
  sleep 60
done
# The first slice is terminal; its per-cell markers make the next slice resumable.
mv -f depth_profile.exit depth_profile.slice1.exit
exec bash night_depth_profile.sh "$RUN_DIR" "$@"
