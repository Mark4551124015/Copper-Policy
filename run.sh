#!/usr/bin/env bash
# Run all four complete benchmarks sequentially on the same GPUs.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'HELP'
Usage: bash run.sh [options]
  --out-dir PATH       Batch output directory (default: outputs/release_validation/<timestamp>)
  --gpu-ids LIST       Physical GPU indices (default: 0,1,2,3,4,5,6,7)
  --order LIST         Comma-separated permutation of libero,libero_plus,robotwin_clean,robotwin_rand
  --resume             Skip completed evaluations in --out-dir; retry incomplete ones
  --smoke              One episode per task, retaining the complete task sets
  --dry-run            Preview launcher commands without starting evaluations
  -h, --help           Show help

Order: RoboTwin clean -> LIBERO -> RoboTwin randomized -> LIBERO-Plus.
Task sets: 50 / 40 / 50 / all LIBERO-Plus instances. No task sampling.
Default episodes per task: 50 / 50 / 50 / 1.
Each evaluation saves summary.json, videos and its own launcher.log.
A failure stops the batch. Resume skips whole completed evaluations; incomplete
ones continue in their latest attempt directory and resume individual episodes.
Environment: RUN_OUT_DIR, EVAL_GPU_IDS, ROBOTWIN_CKPT, LIBERO_CKPT,
ROBOTWIN_EPISODES, LIBERO_TRIALS, PLUS_TRIALS, NUM_INFERENCE_STEPS,
ROBOTWIN_REPLAN_STEPS, LIBERO_REPLAN_STEPS, ROBOTWIN_SEED, LIBERO_SEED.
HELP
}
BATCH_DIR="${RUN_OUT_DIR:-outputs/release_validation/$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${EVAL_GPU_IDS:-0,1,2,3,4,5,6,7}"
RESUME=0
DRY_RUN=0
SMOKE=0
OPTIONS=(--all-tasks)
ORDER=robotwin_clean,libero,robotwin_rand,libero_plus
while (($#)); do
  case "$1" in
    --out-dir|--gpu-ids|--order)
      (($# >= 2)) && [[ -n "$2" ]] || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --out-dir) BATCH_DIR="$2" ;;
        --gpu-ids) GPU_IDS="$2" ;;
        --order) ORDER="$2" ;;
      esac
      shift 2 ;;
    --resume) RESUME=1; shift ;;
    --include-clean) shift ;; # Compatibility: clean is always included.
    --smoke) SMOKE=1; OPTIONS+=(--smoke); shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid --gpu-ids' >&2; exit 2; }
export EVAL_GPU_IDS="$GPU_IDS"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"
IFS=',' read -r -a NAMES <<< "$ORDER"
[[ "$ORDER" != *, && ${#NAMES[@]} == 4 ]] || { echo '--order must include all four benchmarks exactly once.' >&2; exit 2; }
for expected in libero libero_plus robotwin_clean robotwin_rand; do
  occurrences=0
  for name in "${NAMES[@]}"; do
    if [[ "$name" == "$expected" ]]; then occurrences=$((occurrences + 1)); fi
  done
  ((occurrences == 1)) || { echo '--order must include all four benchmarks exactly once.' >&2; exit 2; }
done
ROBOTWIN_EPISODES="${ROBOTWIN_EPISODES:-50}"
LIBERO_TRIALS="${LIBERO_TRIALS:-50}"
PLUS_TRIALS="${PLUS_TRIALS:-1}"
if ((SMOKE)); then ROBOTWIN_EPISODES=1; LIBERO_TRIALS=1; PLUS_TRIALS=1; fi
for count in "$ROBOTWIN_EPISODES" "$LIBERO_TRIALS" "$PLUS_TRIALS"; do
  [[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo 'Episode counts must be positive integers.' >&2; exit 2; }
done
settings() {
  if [[ "$1" == robotwin_* ]]; then
    EPISODES="$ROBOTWIN_EPISODES"
    CHECKPOINT="${ROBOTWIN_CKPT:-pretrained_weights/copper_policy/robotwin/policy.pt}"
    REPLAN="${ROBOTWIN_REPLAN_STEPS:-24}"
    ACTION_SEED="${ROBOTWIN_SEED:-0}"
    TASK_VARIANT=demo_clean
    [[ "$1" == robotwin_rand ]] && TASK_VARIANT=demo_randomized
  else
    EPISODES="$LIBERO_TRIALS"
    [[ "$1" == libero_plus ]] && EPISODES="$PLUS_TRIALS"
    CHECKPOINT="${LIBERO_CKPT:-pretrained_weights/copper_policy/libero/policy.pt}"
    REPLAN="${LIBERO_REPLAN_STEPS:-10}"
    ACTION_SEED="${LIBERO_SEED:-7}"
    TASK_VARIANT=demo_clean
  fi
  return 0
}
if ((DRY_RUN)); then
  for name in "${NAMES[@]}"; do
    settings "$name"
    printf 'EVAL_GPU_IDS=%q OUT_DIR=%q CKPT=%q EVAL_NUM_EPISODES=%q NUM_TRIALS=%q TASK_CONFIG=%q REPLAN_STEPS=%q SEED=%q bash %q' \
      "$GPU_IDS" "$BATCH_DIR/$name/attempt_1" "$CHECKPOINT" "$EPISODES" "$EPISODES" "$TASK_VARIANT" "$REPLAN" "$ACTION_SEED" "test_${name}.sh"
    if ((${#OPTIONS[@]})); then printf ' %q' "${OPTIONS[@]}"; fi; printf '\n'
  done
  exit 0
fi
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
# Prevent inherited per-test OUT_DIR from placing all evaluations together.
unset OUT_DIR AUTO_EVAL_TASKS PLUS_SAMPLE_SIZE
mkdir -p "$BATCH_DIR"
BATCH_DIR="$(cd "$BATCH_DIR" && pwd)"
if [[ -e "$BATCH_DIR/status.tsv" ]] && ((!RESUME)); then
  echo 'Output directory already has a batch; use --resume or a new --out-dir.' >&2
  exit 2
fi
if [[ ! -f "$BATCH_DIR/status.tsv" ]]; then
  printf 'evaluation\tstatus\texit_code\toutput\ttimestamp\n' > "$BATCH_DIR/status.tsv"
fi
completed() {
  # uv only; validate task completion rather than treating low success rate as failure.
  uv run --frozen --no-sync python - "$1" "$2" "$3" <<'PY'
import ast, json, sys
from pathlib import Path
from inference.tasks import ALL_TASKS
try:
    summary = json.loads(Path(sys.argv[1]).read_text())
    name, episodes = sys.argv[2], int(sys.argv[3])
    valid = summary.get("all_tasks_completed") is True
    valid = valid and bool(summary.get("results"))
    valid = valid and all(row.get("exit_code") == 0 and row.get("episodes") == episodes for row in summary["results"])
    valid = valid and summary.get("episodes_per_task") == episodes
    if name.startswith("robotwin_"):
        expected = set(ALL_TASKS)
        actual = summary["tasks"]
        records = [row["task_name"] for row in summary["results"]]
        variant = "demo_clean" if name == "robotwin_clean" else "demo_randomized"
        valid = valid and summary.get("task_config") == variant
    else:
        suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
        if name == "libero":
            expected = {(suite, i) for suite in suites for i in range(10)}
        else:
            from third_party.setup_sources import source_dir
            path = source_dir("libero-plus") / "libero/libero/benchmark/libero_suite_task_map.py"
            node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in n.targets))
            task_map = ast.literal_eval(node.value)
            expected = {(suite, i) for suite in suites for i in range(len(task_map[suite]))}
        actual = [tuple(task) for task in summary["tasks"]]
        records = [(row["suite"], row["task_id"]) for row in summary["results"]]
        valid = valid and summary.get("backend") == ("libero" if name == "libero" else "libero-plus")
    valid = valid and len(actual) == len(expected) and set(actual) == expected
    valid = valid and len(records) == len(expected) and set(records) == expected
except (OSError, ValueError, KeyError, TypeError, StopIteration):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}
record() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$(date -Iseconds)" >> "$BATCH_DIR/status.tsv"
}
for name in "${NAMES[@]}"; do
  settings "$name"
  EVAL_DIR="$BATCH_DIR/$name"
  mkdir -p "$EVAL_DIR"
  if ((RESUME)) && [[ -f "$EVAL_DIR/completed_output.txt" ]]; then
    PREVIOUS="$(cat "$EVAL_DIR/completed_output.txt")"
    if completed "$PREVIOUS/summary.json" "$name" "$EPISODES"; then
      echo "[$name] already complete: $PREVIOUS"
      record "$name" skipped 0 "$PREVIOUS"
      continue
    fi
  fi
  ATTEMPT=1
  while [[ -e "$EVAL_DIR/attempt_$ATTEMPT" ]]; do ATTEMPT=$((ATTEMPT + 1)); done
  if ((RESUME && ATTEMPT > 1)); then
    ATTEMPT=$((ATTEMPT - 1))
  fi
  CURRENT="$EVAL_DIR/attempt_$ATTEMPT"
  mkdir -p "$CURRENT"
  echo "[$name] starting $(date -Iseconds); output: $CURRENT"
  record "$name" running - "$CURRENT"
  # PIPESTATUS preserves launcher failures even when tee succeeds.
  set +e
  OUT_DIR="$CURRENT" CKPT="$CHECKPOINT" EVAL_NUM_EPISODES="$EPISODES" NUM_TRIALS="$EPISODES" \
    TASK_CONFIG="$TASK_VARIANT" REPLAN_STEPS="$REPLAN" SEED="$ACTION_SEED" \
    bash "test_${name}.sh" "${OPTIONS[@]}" 2>&1 | tee -a "$CURRENT/launcher.log"
  PIPE_CODES=("${PIPESTATUS[@]}")
  set -e
  CODE="${PIPE_CODES[0]}"
  ((CODE == 0)) && CODE="${PIPE_CODES[1]}"
  if ((CODE == 0)) && ! completed "$CURRENT/summary.json" "$name" "$EPISODES"; then
    echo "[$name] missing or incomplete summary.json" >&2
    CODE=1
  fi
  if ((CODE != 0)); then
    record "$name" failed "$CODE" "$CURRENT"
    echo "[$name] FAILED (exit $CODE). Log: $CURRENT/launcher.log" >&2
    printf 'Retry batch: bash run.sh --resume --out-dir %q --gpu-ids %q --order %q' "$BATCH_DIR" "$GPU_IDS" "$ORDER" >&2
    ((SMOKE)) && printf ' --smoke' >&2
    printf '\n' >&2
    exit "$CODE"
  fi
  printf '%s\n' "$CURRENT" > "$EVAL_DIR/completed_output.txt"
  record "$name" completed 0 "$CURRENT"
  echo "[$name] completed: $CURRENT/summary.json"
done
echo "Batch completed. Status: $BATCH_DIR/status.tsv"
