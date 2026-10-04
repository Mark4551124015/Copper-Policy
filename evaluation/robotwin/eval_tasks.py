"""Evaluate selected RoboTwin tasks with one worker per selected GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading

from inference.tasks import TASKS, ALL_TASKS

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", default=os.environ.get("EVAL_GPU_IDS", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")))
    parser.add_argument("--eval-num-episodes", type=int, default=50)
    parser.add_argument("--tasks", default=None, help="Comma-separated task names; default: original ten tasks")
    parser.add_argument("--all-tasks", action="store_true", help="Evaluate all 50 RoboTwin benchmark tasks")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--instruction-type", default="seen")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replan-steps", type=int, default=24)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.all_tasks and args.tasks is not None:
        parser.error("--all-tasks cannot be combined with --tasks")
    tasks = list(ALL_TASKS) if args.all_tasks else ([value.strip() for value in args.tasks.split(",")] if args.tasks is not None else list(TASKS))
    if not tasks or any(not name or not all(c.isalnum() or c == "_" for c in name) for name in tasks) or len(set(tasks)) != len(tasks):
        parser.error("--tasks must contain distinct task names separated by commas")
    if not args.dry_run:
        from third_party.setup_sources import source_dir
        missing = [name for name in tasks if not (source_dir("robotwin") / "envs" / f"{name}.py").is_file()]
        if missing:
            parser.error(f"RoboTwin task source missing: {missing}")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",")]
    if not gpu_ids or any(not value.isdigit() for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpu-ids must contain distinct physical GPU indices, e.g. 0,1,2,3,4,5,6,7")
    if args.eval_num_episodes < 1:
        parser.error("--eval-num-episodes must be positive")
    checkpoint, output = args.ckpt.resolve(), args.out_dir.resolve()
    if not args.dry_run and not checkpoint.is_file():
        parser.error(f"Checkpoint missing: {checkpoint}")

    def completed_result(task):
        try:
            result = json.loads((output / task / "result.json").read_text())
            if (result["task_name"] == task and result["eval_num_episodes"] == args.eval_num_episodes
                    and result["task_config"] == args.task_config
                    and result["instruction_type"] == args.instruction_type
                    and Path(result["ckpt_setting"]).resolve() == checkpoint):
                return result
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def command(shard, gpu):
        task = shard[0]
        cmd = [sys.executable, "-m", "evaluation.run", "robotwin", "--ckpt", str(checkpoint),
               "--task-name", task, "--tasks", ",".join(shard), "--gpu-id", gpu, "--out-dir", str(output),
               "--eval-num-episodes", str(args.eval_num_episodes), "--task-config", args.task_config,
               "--instruction-type", args.instruction_type, "--seed", str(args.seed),
               "--replan-steps", str(args.replan_steps),
               "--num-inference-steps", str(args.num_inference_steps)]
        if args.no_compile:
            cmd.append("--no-compile")
        return cmd

    shards = [tasks[index::len(gpu_ids)] for index in range(len(gpu_ids))]
    print(f"Evaluating {len(tasks)} tasks on GPUs {','.join(gpu_ids)}; one resident model/GPU", flush=True)
    if args.dry_run:
        for gpu, shard in zip(gpu_ids, shards):
            if shard:
                print(shlex.join(command(shard, gpu)))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    active = {}
    lock = threading.Lock()
    stopping = threading.Event()

    def stop_workers(_signum=None, _frame=None):
        stopping.set()
        with lock:
            for proc in active.values():
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

    signal.signal(signal.SIGINT, stop_workers)
    signal.signal(signal.SIGTERM, stop_workers)

    def worker(gpu, shard):
        results, pending = [], []
        for task in shard:
            existing = completed_result(task)
            if existing is not None:
                results.append(dict(task_name=task, gpu_id=int(gpu), exit_code=0,
                                    successes=existing["successes"], episodes=existing["eval_num_episodes"],
                                    success_rate=existing["success_rate"], resumed=True))
                print(f"[GPU {gpu}] already complete: {task}; model loading skipped", flush=True)
            else:
                pending.append(task)
        if not pending or stopping.is_set():
            return results
        log = output / f"persistent_gpu{gpu}.log"
        print(f"[GPU {gpu}] starting resident worker: {','.join(pending)}", flush=True)
        with log.open("a") as stream:
            stream.write(shlex.join(command(pending, gpu)) + "\n")
            stream.flush()
            with lock:
                if stopping.is_set():
                    return results
                proc = subprocess.Popen(command(pending, gpu), cwd=ROOT, stdout=stream,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = proc
            code = proc.wait()
            with lock:
                active.pop(gpu, None)
        for task in pending:
            result = completed_result(task)
            record = {"task_name": task, "gpu_id": int(gpu), "exit_code": 0 if result is not None else (code or 1)}
            if result is not None:
                record.update(successes=result["successes"], episodes=result["eval_num_episodes"],
                              success_rate=result["success_rate"])
            results.append(record)
            print(f"[GPU {gpu}] finished {task}: exit={record['exit_code']}; log={log}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(tasks))) as pool:
        futures = [pool.submit(worker, gpu, shard) for gpu, shard in zip(gpu_ids, shards) if shard]
        records = [record for future in futures for record in future.result()]
    records.sort(key=lambda row: tasks.index(row["task_name"]))
    complete = len(records) == len(tasks) and all(row["exit_code"] == 0 for row in records)
    summary = {"tasks": tasks, "task_config": args.task_config, "gpu_ids": gpu_ids, "workers_per_gpu": 1,
               "dispatch_unit": "persistent_worker", "episodes_per_task": args.eval_num_episodes, "all_tasks_completed": complete,
               "results": records}
    if complete:
        summary["mean_task_success_rate"] = sum(row["success_rate"] for row in records) / len(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {output / 'summary.json'}; all tasks completed={complete}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
