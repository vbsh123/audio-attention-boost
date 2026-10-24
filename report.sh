#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
samples="${SAMPLES:-50}"
case "${PROFILE:-original}" in
  original) default_run="audio_direct_boost${samples}_v24_tps" ;;
  early15_deep|early15_all) default_run="audio_direct_boost${samples}_${PROFILE}_v25_tps" ;;
  *) echo 'Unknown PROFILE'; exit 2 ;;
esac
run_name="${RUN_NAME:-$default_run}"
[[ "$run_name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_NAME'; exit 2; }
if [[ ! -f "runs/$run_name/summary.json" ]]; then
  echo "No completed question pairs yet. Check the run log. Expected: runs/$run_name/summary.json"
  exit 1
fi
python_bin="${AUDIO_VENV:-$PWD/.venv-audio}/bin/python"
if [[ ! -x "$python_bin" ]]; then python_bin="${PYTHON:-python3}"; fi
"$python_bin" -m experiments.audio_drift.run_direct_boost --out "runs/$run_name" --report-only
cat "runs/$run_name/REPORT.md"
