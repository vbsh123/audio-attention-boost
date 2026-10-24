#!/usr/bin/env bash
# Compare the two requested profiles sequentially on the same development pool.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
samples="${SAMPLES:-50}"
[[ "$samples" =~ ^[1-9][0-9]*$ ]] && (( samples <= 50 )) || { echo 'Tune on 1–50 development questions; preserve the larger validation pool.'; exit 2; }
exec 9> "$PWD/.run.lock"
flock -n 9 || { echo 'An experiment is already running in this checkout.' >&2; exit 1; }
for profile in early15_deep early15_all; do
  echo "Starting tuning profile: $profile on $samples questions"
  PROFILE="$profile" SAMPLES="$samples" RUN_NAME="audio_direct_boost${samples}_${profile}_v25_tps" \
    bash scripts/vast_audio_direct_boost.sh run
  PROFILE="$profile" SAMPLES="$samples" RUN_NAME="audio_direct_boost${samples}_${profile}_v25_tps" \
    bash report.sh
done
"${AUDIO_VENV:-$PWD/.venv-audio}/bin/python" -m experiments.audio_drift.compare_tuning \
  --deep-run "runs/audio_direct_boost${samples}_early15_deep_v25_tps" \
  --all-run "runs/audio_direct_boost${samples}_early15_all_v25_tps" \
  --out "runs/audio_direct_boost${samples}_tuning_v25"
