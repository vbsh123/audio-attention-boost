#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
audio_venv="${AUDIO_VENV:-$PWD/.venv-audio}"
python_bin="${PYTHON:-python3}"
if [[ ! -x "$audio_venv/bin/python" ]]; then
  "$python_bin" - <<'PY'
import sys
assert (3,10)<=sys.version_info[:2]<=(3,12), 'Use Python 3.10–3.12; e.g. PYTHON=python3.12 bash setup.sh'
PY
  "$python_bin" -m venv "$audio_venv"
fi
source "$audio_venv/bin/activate"
if [[ "${SKIP_INSTALL:-0}" != 1 ]]; then
  python -m pip install --upgrade pip
  python -m pip install "torch==${TORCH_VERSION:-2.6.0}" "torchaudio==${TORCHAUDIO_VERSION:-2.6.0}" \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
  python -m pip install -r experiments/audio_drift/requirements.txt
  python -m pip check
fi
python -m experiments.audio_drift.checkout --diffa-root "${DIFFA_ROOT:-$PWD/external/DIFFA}"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA GPU required for the full experiment'
assert torch.cuda.is_bf16_supported(), 'BF16-capable CUDA GPU required'
print('GPU:',torch.cuda.get_device_name())
print('Setup complete. Run: bash run.sh')
PY
