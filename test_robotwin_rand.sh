#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TASK_CONFIG=demo_randomized
export AUTO_EVAL_TASKS="${AUTO_EVAL_TASKS:-dump_bin_bigbin,shake_bottle,shake_bottle_horizontally,place_container_plate,grab_roller,stack_bowls_three}"
exec bash "$ROOT/test_robotwin_clean.sh" "$@"
