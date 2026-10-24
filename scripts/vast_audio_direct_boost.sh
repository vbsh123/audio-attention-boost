#!/usr/bin/env bash
# Frozen direct-attention intervention; separate from the old priority heuristic.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-run}"
case "$mode" in prepare|run|plan) ;; *) echo 'Usage: bash scripts/vast_audio_direct_boost.sh [prepare|run|plan]'; exit 2;; esac
protocol="$PWD/experiments/audio_drift/direct_boost_protocol_v24.json"
exclusions="$PWD/experiments/audio_drift/direct_boost_excluded_ids_v24.json"
manifest="${MANIFEST:-$PWD/data/audio_drift/gsm_direct_boost50_seed20261024_v24.jsonl}"
run_name="${RUN_NAME:-audio_direct_boost50_v24_tps}"
samples="${SAMPLES:-50}"
[[ "$run_name" =~ ^[a-zA-Z0-9_-]+$ && "$samples" =~ ^[1-9][0-9]*$ ]] && (( samples <= 50 )) || { echo 'Invalid run name/sample count'; exit 2; }
if [[ "$mode" == plan ]]; then
  "${PYTHON:-python3}" -m experiments.audio_drift.run_direct_boost --manifest "$manifest" \
    --out "runs/$run_name" --protocol "$protocol" --samples "$samples" --plan
  exit 0
fi
audio_venv="${AUDIO_VENV:-$PWD/.venv-audio}"
[[ -x "$audio_venv/bin/python" ]] || { echo 'First run: bash setup.sh'; exit 2; }
source "$audio_venv/bin/activate"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if [[ ! -f "$manifest" ]]; then
  python -u -m experiments.audio_drift.prepare_gsm --out "$manifest" --samples 50 \
    --revision e94caa2154433a54cd430702666c454977df3083 --seed 20261024 \
    --cohort direct_boost_v24 --exclude-id-files "$exclusions" --min-seconds 0 --max-seconds 30
fi
if [[ "$mode" == prepare ]]; then echo "Manifest: $manifest"; exit 0; fi
python -m experiments.audio_drift.checkout --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}"
python -u -m experiments.audio_drift.run_direct_boost --manifest "$manifest" --out "runs/$run_name" \
  --protocol "$protocol" --exclusions "$exclusions" --samples "$samples" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}"
