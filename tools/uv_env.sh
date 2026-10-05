#!/usr/bin/env bash
# Source from a launcher after setting ROOT to the repository root.
copper_use_env() {
  local backend="$1"
  unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
  if [[ "$backend" == libero || "$backend" == libero-plus ]]; then
    export UV_PROJECT_ENVIRONMENT="$ROOT/.venv-libero"
    uv sync --project "$ROOT/envs/libero" --frozen --inexact
    PYTHON="$ROOT/.venv-libero/bin/python"
  else
    export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
    uv sync --project "$ROOT" --frozen --extra robotwin --inexact
    PYTHON="$ROOT/.venv/bin/python"
  fi
  export PYTHON
}
