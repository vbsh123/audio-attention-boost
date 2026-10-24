#!/usr/bin/env bash
# Frozen direct-attention intervention; separate from the old priority heuristic.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${1:-run}"
case "$mode" in prepare|run|plan) ;; *) echo 'Usage: bash scripts/vast_audio_direct_boost.sh [prepare|run|plan]'; exit 2;; esac
protocol="$PWD/experiments/audio_drift/direct_boost_protocol_v24.json"
exclusions="$PWD/experiments/audio_drift/direct_boost_excluded_ids_v24.json"
samples="${SAMPLES:-50}"
run_name="${RUN_NAME:-audio_direct_boost${samples}_v24_tps}"
[[ "$run_name" =~ ^[a-zA-Z0-9_-]+$ && "$samples" =~ ^[1-9][0-9]*$ ]] && (( samples <= 500 )) || { echo 'Invalid run name/sample count (1–500)'; exit 2; }
pool_count=50
pool_seed=20261024
cohort=direct_boost_v24
if (( samples > 50 )); then
  pool_count=500
  pool_seed=20261025
  cohort=direct_boost_validation_v24
fi
manifest="${MANIFEST:-$PWD/data/audio_drift/gsm_direct_boost${pool_count}_seed${pool_seed}_v24.jsonl}"
if [[ "$mode" == plan ]]; then
  "${PYTHON:-python3}" -m experiments.audio_drift.run_direct_boost --manifest "$manifest" \
    --out "runs/$run_name" --protocol "$protocol" --samples "$samples" --plan
  exit 0
fi
audio_venv="${AUDIO_VENV:-$PWD/.venv-audio}"
[[ -x "$audio_venv/bin/python" ]] || { echo 'First run: bash setup.sh'; exit 2; }
source "$audio_venv/bin/activate"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
if (( samples > 50 )); then
  pilot_manifest="${PILOT_MANIFEST:-$PWD/data/audio_drift/gsm_direct_boost50_seed20261024_v24.jsonl}"
  [[ -f "$pilot_manifest" ]] || { echo 'Run the 50-question pilot first, or set PILOT_MANIFEST to its manifest.'; exit 2; }
  large_exclusions="${manifest%.jsonl}.exclusions.json"
  python - "$exclusions" "$pilot_manifest" "$large_exclusions" <<'PY'
import json,sys
from pathlib import Path
old=json.loads(Path(sys.argv[1]).read_text())
pilot_path=Path(sys.argv[2]);raw=pilot_path.read_text()
pilot=json.loads(raw) if raw.lstrip().startswith('[') else [json.loads(s) for s in raw.splitlines() if s.strip()]
ids={r['id'] for r in pilot}
assert len(pilot)==len(ids)==50,'Expected the complete 50-question pilot manifest'
assert all(r.get('source_dataset')==old['source_dataset'] and r.get('source_revision')==old['source_revision'] for r in pilot),'Pilot dataset/revision mismatch'
assert not ids&set(old['ids']),'Pilot overlaps original development/evaluation questions'
merged=dict(source_dataset=old['source_dataset'],source_revision=old['source_revision'],
            ids=sorted(set(old['ids'])|ids),note='Exclude 130 historical questions plus the 50-question direct-boost pilot.')
out=Path(sys.argv[3]);out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():
    assert json.loads(out.read_text())==merged,'Exclusion set changed; use a fresh manifest path'
else:
    out.write_text(json.dumps(merged,indent=2)+'\n')
print(f'Larger validation pool excludes {len(merged["ids"])} previously used questions',flush=True)
PY
  exclusions="$large_exclusions"
fi
if [[ ! -f "$manifest" ]]; then
  python -u -m experiments.audio_drift.prepare_gsm --out "$manifest" --samples "$pool_count" \
    --revision e94caa2154433a54cd430702666c454977df3083 --seed "$pool_seed" \
    --cohort "$cohort" --exclude-id-files "$exclusions" --min-seconds 0 --max-seconds 30
fi
python - "$manifest" "$pool_count" "$exclusions" <<'PY'
import json,sys
from pathlib import Path
text=Path(sys.argv[1]).read_text()
rows=json.loads(text) if text.lstrip().startswith('[') else [json.loads(s) for s in text.splitlines() if s.strip()]
old=json.loads(Path(sys.argv[3]).read_text());ids={r['id'] for r in rows}
assert len(rows)==len(ids)==int(sys.argv[2]),'Unexpected manifest pool size'
assert not ids&set(old['ids']),'Evaluation pool overlaps excluded questions'
assert all(r.get('source_dataset')==old['source_dataset'] and r.get('source_revision')==old['source_revision'] for r in rows),'Pool dataset/revision mismatch'
print(f'Validated {len(rows)} unique pool entries; the run uses the requested prefix without correctness filtering',flush=True)
PY
if [[ "$mode" == prepare ]]; then echo "Manifest: $manifest"; exit 0; fi
python -m experiments.audio_drift.checkout --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}"
python -u -m experiments.audio_drift.run_direct_boost --manifest "$manifest" --out "runs/$run_name" \
  --protocol "$protocol" --exclusions "$exclusions" --samples "$samples" \
  --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}"
