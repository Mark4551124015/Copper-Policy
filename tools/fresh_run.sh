#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
unset PYTHONPATH PYTHONHOME UV_PROJECT_ENVIRONMENT
export HF_HOME="$PWD/.fresh-cache/huggingface"
export TORCH_HOME="$PWD/.fresh-cache/torch"
export MPLCONFIGDIR="$PWD/.fresh-cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.fresh-cache/xdg"
export UV_CACHE_DIR="$PWD/.fresh-cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.fresh-python"
export UV_LINK_MODE=copy
export UV_NO_CACHE=1
export PYTHONNOUSERSITE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
PYTHON="$PWD/.venv/bin/python"
if [[ "${1:-}" == --env ]]; then
  case "${2:-}" in
    libero|libero-plus) PYTHON="$PWD/.venv-libero/bin/python" ;;
    robotwin) ;;
    *) echo 'Usage: fresh_run.sh [--env libero|libero-plus|robotwin] <Python arguments>' >&2; exit 2 ;;
  esac
  shift 2
fi
exec "$PYTHON" "$@"
