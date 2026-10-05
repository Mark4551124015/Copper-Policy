#!/usr/bin/env bash
# Fixed stratified sample: 3 spatial, 3 object, 2 goal, 2 libero_10 tasks.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/tools/uv_env.sh"
copper_use_env libero
if [[ " $* " != *" --dry-run "* ]]; then
  "$PYTHON" -m third_party.setup_sources libero
  "$PYTHON" -m evaluation.run libero --check
fi
exec "$PYTHON" -m evaluation.libero_mot.eval_tasks \
  --ckpt "${CKPT:-pretrained_weights/copper_policy/libero/policy.pt}" \
  --out-dir "${OUT_DIR:-outputs/inference/libero/$(date +%Y%m%d_%H%M%S)}" \
  --gpu-ids "${EVAL_GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --num-trials "${NUM_TRIALS:-50}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS:-10}" \
  --replan-steps "${REPLAN_STEPS:-10}" \
  --seed "${SEED:-7}" "$@"
