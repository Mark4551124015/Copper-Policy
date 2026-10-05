#!/usr/bin/env bash
# Install the two independent inference environments (no models or GPU work).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/tools/uv_env.sh"
copper_use_env robotwin
copper_use_env libero
echo 'Ready: RoboTwin in .venv; LIBERO / LIBERO-Plus in .venv-libero.'
echo 'For RoboTwin CUDA extensions, also run: bash tools/setup_robotwin.sh'
