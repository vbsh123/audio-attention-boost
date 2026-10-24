#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
run_name="${RUN_NAME:-audio_direct_boost50_v24_tps}"
[[ "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
if [[ ! -f "runs/$run_name/summary.json" ]]; then
  echo "No completed question pairs yet. Check the run log. Expected: runs/$run_name/summary.json"
  exit 1
fi
python_bin="${AUDIO_VENV:-$PWD/.venv-audio}/bin/python"
if [[ ! -x "$python_bin" ]]; then python_bin="${PYTHON:-python3}"; fi
"$python_bin" -m experiments.audio_drift.run_direct_boost --out "runs/$run_name" --report-only
cat "runs/$run_name/REPORT.md"
