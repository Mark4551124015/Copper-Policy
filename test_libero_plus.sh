#!/usr/bin/env bash
# LIBERO-Plus: fixed stratified 500-instance sample, one episode per instance.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/tools/uv_env.sh"
copper_use_env libero
if [[ " $* " != *" --dry-run "* ]]; then
  "$PYTHON" -m third_party.setup_sources libero-plus
  "$PYTHON" -m evaluation.run libero-plus --check
fi
exec "$PYTHON" -m evaluation.libero_mot.eval_tasks \
  --backend libero-plus --sample-size "${PLUS_SAMPLE_SIZE:-500}" \
  --ckpt "${CKPT:-pretrained_weights/copper_policy/libero/policy.pt}" \
  --out-dir "${OUT_DIR:-outputs/inference/libero-plus/$(date +%Y%m%d_%H%M%S)}" \
  --gpu-ids "${EVAL_GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --num-trials "${NUM_TRIALS:-1}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS:-10}" \
  --replan-steps "${REPLAN_STEPS:-10}" \
  --seed "${SEED:-7}" "$@"
