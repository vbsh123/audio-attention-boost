#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ "${1:-run}" != plan ]]; then
  exec 9> "$PWD/.run.lock"
  flock -n 9 || { echo 'An experiment is already running in this checkout.' >&2; exit 1; }
fi
exec bash scripts/vast_audio_direct_boost.sh "${1:-run}"
