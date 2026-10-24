#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec bash scripts/vast_audio_direct_boost.sh "${1:-run}"
