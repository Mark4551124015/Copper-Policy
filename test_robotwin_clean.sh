#!/usr/bin/env bash
# Inference only. Run with: uv run --frozen bash test_robotwin_clean.sh [backend] [options]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash test_robotwin_clean.sh [robotwin|libero|libero-plus] [options]
  --ckpt PATH          Portable checkpoint (or set CKPT)
  --gpu-ids LIST       Physical GPU indices; RoboTwin default: 0,1,2,3,4,5,6,7
  --out-dir PATH       Results directory (or set OUT_DIR)
  --smoke              One episode per task; LIBERO task 0 only
  --dry-run            Print evaluation commands without loading models
  --no-compile         Disable compilation for debugging
  --all-tasks          Evaluate all 50 RoboTwin tasks
  -h, --help           Show this help

Default: full RoboTwin evaluation, exactly 10 tasks, 50 episodes/task,
one worker per GPU, compiled inference. No training or data preprocessing.
Environment: CKPT, OUT_DIR, EVAL_GPU_IDS, EVAL_NUM_EPISODES,
TASK_CONFIG, LIBERO_SUITE, NUM_TRIALS, NUM_INFERENCE_STEPS, REPLAN_STEPS, SEED.
LIBERO and LIBERO-Plus share the LIBERO checkpoint and use one GPU.
EOF
}

BACKEND=robotwin
DRY_RUN=0
SMOKE=0
NO_COMPILE=0
ALL_TASKS=0
GPU_IDS="${EVAL_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
CKPT="${CKPT:-}"
OUT_DIR="${OUT_DIR:-}"
while (($#)); do
  case "$1" in
    robotwin|libero|libero-plus) BACKEND="$1"; shift ;;
    --ckpt|--gpu-ids|--out-dir)
      if (($# < 2)) || [[ -z "$2" ]]; then
        echo "Missing value for $1" >&2; exit 2
      fi
      case "$1" in
        --ckpt) CKPT="$2" ;;
        --gpu-ids) GPU_IDS="$2" ;;
        --out-dir) OUT_DIR="$2" ;;
      esac
      shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-compile) NO_COMPILE=1; shift ;;
    --all-tasks) ALL_TASKS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU indices must be comma-separated numbers" >&2; exit 2
fi
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/' >&2; exit 1; }
source "$ROOT/tools/uv_env.sh"
copper_use_env "$BACKEND"
POLICY_BACKEND="$BACKEND"
[[ "$BACKEND" == libero-plus ]] && POLICY_BACKEND=libero
CKPT="${CKPT:-pretrained_weights/copper_policy/$POLICY_BACKEND/policy.pt}"
OUT_DIR="${OUT_DIR:-outputs/inference/${BACKEND}_${TASK_CONFIG:-demo_clean}/$(date +%Y%m%d_%H%M%S)}"

run() { printf 'Running:'; printf ' %q' "$@"; printf '\n'; "$@"; }
echo "Backend: $BACKEND; checkpoint: $CKPT"
echo "Output: $OUT_DIR"
echo "Python: $PYTHON"
if ((!DRY_RUN)); then
  "$PYTHON" - "$BACKEND" <<'PY'
import importlib.util
import sys
required = ["torch", "transformers", "diffusers", "numpy", "scipy", "yaml", "cv2"]
if sys.argv[1] == "robotwin":
    required += ["sapien", "mplib", "curobo"]
else:
    required += ["robosuite", "mujoco"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print(f"Interpreter: {sys.executable}", flush=True)
if missing:
    raise SystemExit(
        "Missing inference dependencies: " + ", ".join(missing)
        + "\nFor RoboTwin, run: bash tools/setup_robotwin.sh"
        + "\nSimulator installation instructions: README.md -> Evaluation setup."
    )
PY
fi
if [[ "$BACKEND" == robotwin ]]; then
  EPISODES="${EVAL_NUM_EPISODES:-50}"
  ((SMOKE)) && EPISODES=1
  COMMAND=("$PYTHON" -m evaluation.robotwin.eval_tasks
    --ckpt "$CKPT" --out-dir "$OUT_DIR" --gpu-ids "$GPU_IDS"
    --eval-num-episodes "$EPISODES" --task-config "${TASK_CONFIG:-demo_clean}"
    --replan-steps "${REPLAN_STEPS:-24}"
    --num-inference-steps "${NUM_INFERENCE_STEPS:-10}"
    --seed "${SEED:-0}"
  )
  if ((ALL_TASKS)); then
    COMMAND+=(--all-tasks)
  elif [[ -n "${AUTO_EVAL_TASKS:-}" ]]; then
    COMMAND+=(--tasks "$AUTO_EVAL_TASKS")
  fi
  ((NO_COMPILE)) && COMMAND+=(--no-compile)
  ((DRY_RUN)) && COMMAND+=(--dry-run)
  printf 'Running:'; printf ' %q' "${COMMAND[@]}"; printf '\n'
  exec "${COMMAND[@]}"
else
  TRIALS="${NUM_TRIALS:-50}"
  [[ "$BACKEND" == libero-plus ]] && TRIALS="${NUM_TRIALS:-1}"
  ((SMOKE)) && TRIALS=1
  COMMAND=("$PYTHON" -m evaluation.run "$BACKEND" --ckpt "$CKPT"
    --suite "${LIBERO_SUITE:-libero_spatial}" --num-trials "$TRIALS"
    --num-inference-steps "${NUM_INFERENCE_STEPS:-10}"
    --replan-steps "${REPLAN_STEPS:-10}"
    --seed "${SEED:-7}" --out-dir "$OUT_DIR" --device cuda:0)
  ((SMOKE)) && COMMAND+=(--task-range 0 1)
  ((NO_COMPILE)) && COMMAND+=(--no-compile)
  echo "LIBERO GPU: ${GPU_IDS%%,*}"
  if ((DRY_RUN)); then
    printf 'CUDA_VISIBLE_DEVICES=%q' "${GPU_IDS%%,*}"
    printf ' %q' "${COMMAND[@]}"; printf '\n'
  else
    export CUDA_VISIBLE_DEVICES="${GPU_IDS%%,*}"
    run "${COMMAND[@]}"
  fi
fi
