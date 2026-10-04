#!/usr/bin/env bash
# Install RoboTwin evaluation into this project's uv environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
unset UV_PROJECT_ENVIRONMENT PYTHONPATH PYTHONHOME
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
uv sync --frozen --extra robotwin --inexact
uv run --frozen --extra robotwin --no-sync python -m third_party.setup_sources robotwin

CUROBO_DIR="$ROOT/third_party/curobo"
CUROBO_REV=d64c4b005459db10c5dd867d8b30a87d5bda9bdb  # v0.7.8
if [[ ! -e "$CUROBO_DIR" ]]; then
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi
if [[ "$(git -C "$CUROBO_DIR" rev-parse HEAD)" != "$CUROBO_REV" ]]; then
  echo "Expected cuRobo v0.7.8 ($CUROBO_REV) at $CUROBO_DIR" >&2
  exit 1
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo 'Set CUDA_HOME to a CUDA toolkit matching the locked PyTorch (CUDA 13.0).' >&2
  exit 1
fi
# Compilation runs on the CPU. Supply architectures explicitly so the build
# works without probing or allocating a GPU. Override for your deployment GPU.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;12.0+PTX}"
export MAX_JOBS="${MAX_JOBS:-4}"
uv pip install --python "$ROOT/.venv/bin/python" --no-build-isolation --no-deps "$CUROBO_DIR"
uv run --frozen --extra robotwin --no-sync python tools/patch_robotwin_dependencies.py
uv run --frozen --extra robotwin --no-sync python tools/setup_oidn.py
echo 'RoboTwin uv environment ready. Prepare assets/weights, then run: bash test_robotwin_clean.sh'
