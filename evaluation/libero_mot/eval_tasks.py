"""Evaluate complete LIBERO benchmarks or explicitly selected task samples."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("libero", "libero-plus"), default="libero")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-manifest", type=Path, help="Explicit JSON list of [suite, task_id]; overrides sampling")
    parser.add_argument("--all-tasks", action="store_true", help="Evaluate all 40 LIBERO tasks or all LIBERO-Plus instances")
    parser.add_argument("--sample-size", type=int, default=500, help="Plus stratified sample size; 0 selects every instance")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--max-tasks-per-gpu", type=int,
                        default=int(os.environ.get("MAX_TASKS_PER_GPU", "1")),
                        help="Concurrent environment clients per resident model; inference stays batch 1")
    parser.add_argument("--rpc-retry-interval", type=float, default=20.0)
    args = parser.parse_args()
    if args.all_tasks and args.task_manifest is not None:
        parser.error("--all-tasks cannot be combined with --task-manifest")
    gpu_ids = args.gpu_ids.split(",")
    if not gpu_ids or any(not v.isdigit() for v in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpu-ids must be distinct comma-separated GPU indices")
    if args.sample_size < 0:
        parser.error("--sample-size must be nonnegative")
    if args.num_trials < 1 or args.num_inference_steps < 1 or args.replan_steps < 1:
        parser.error("Trials, inference steps and replan steps must be positive")
    if args.max_tasks_per_gpu < 1 or args.rpc_retry_interval <= 0:
        parser.error("Environment clients and RPC retry interval must be positive")
    if args.smoke:
        args.num_trials = 1
    rng = random.Random(args.sample_seed)
    if args.all_tasks:
        tasks = [(suite, task_id) for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
                 for task_id in range(10)]
    else:
        tasks = [(suite, task_id) for suite, count in (
            ("libero_spatial", 3), ("libero_object", 3), ("libero_goal", 2), ("libero_10", 2)
        ) for task_id in sorted(rng.sample(range(10), count))]
    if args.backend == "libero-plus":
        from third_party.setup_sources import source_dir
        path = source_dir("libero-plus") / "libero/libero/benchmark/libero_suite_task_map.py"
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "libero_task_map" for t in n.targets))
        task_map = ast.literal_eval(node.value)
        tasks = [(suite, i) for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
                 for i in range(len(task_map[suite]))]
        if args.sample_size and not args.smoke and not args.all_tasks:
            classification = json.loads((source_dir("libero-plus") / "libero/libero/benchmark/task_classification.json").read_text())
            groups = {}
            for suite, items in classification.items():
                for item in items:
                    key = (suite, item["category"])
                    groups.setdefault(key, []).append((suite, int(item["id"]) - 1))
            size = min(args.sample_size, len(tasks))
            allocations = {key: size * len(group) // len(tasks) for key, group in groups.items()}
            remainder = size - sum(allocations.values())
            order = sorted(groups, key=lambda key: (-(size * len(groups[key]) % len(tasks)), key))
            for key in order[:remainder]:
                allocations[key] += 1
            tasks = sorted(task for key in sorted(groups)
                           for task in rng.sample(groups[key], allocations[key]))
        if args.smoke and not args.all_tasks:
            # Exercise both plaintext mapping and language bypass in each suite.
            tasks = [(suite, i) for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10")
                     for i in (0, next(i for i, name in enumerate(task_map[suite]) if "_language_" in name))]
    if args.task_manifest is not None:
        requested = json.loads(args.task_manifest.read_text())
        tasks = [(suite, int(task_id)) for suite, task_id in requested]
        if not tasks or len(set(tasks)) != len(tasks):
            parser.error("Task manifest must be nonempty with distinct tasks")
        for suite, task_id in tasks:
            limit = len(task_map.get(suite, [])) if args.backend == "libero-plus" else (10 if suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10") else 0)
            if not 0 <= task_id < limit:
                parser.error(f"Invalid manifest task: {suite}:{task_id}")
    output = args.out_dir.resolve()
    checkpoint = args.ckpt.resolve()
    if not args.dry_run and not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")

    def persistent_command():
        cmd = [sys.executable, "-m", "evaluation.libero_mot.eval_libero_persistent",
               "--task-file", str(output / "pending_tasks.csv"),
               "--gpu-ids", ",".join(gpu_ids[:min(len(gpu_ids), len(pending_tasks))]),
               "--log-dir", str(output / "logs"),
               "--max-tasks-per-gpu", str(args.max_tasks_per_gpu),
               "--rpc-retry-interval", str(args.rpc_retry_interval), "--per-task-output",
               "--ckpt", str(checkpoint), "--suite", tasks[0][0],
               "--num-trials", str(args.num_trials), "--num-inference-steps", str(args.num_inference_steps),
               "--replan-steps", str(args.replan_steps), "--seed", str(args.seed),
               "--description-source", "mapping", "--save-video", "--resume",
               "--t5-dir", str(ROOT / "pretrained_weights/text_encoder/wan22_ti2v_5b"),
               "--out-dir", str(output)]
        from wan_va.modules.backbone_presets import vision_cache_dir, vision_cache_ready
        if vision_cache_ready("dinov2_with_registers_large"):
            cmd += ["--dino-model", str(vision_cache_dir("dinov2_with_registers_large"))]
        if args.no_compile:
            cmd.append("--no-compile")
        return cmd

    def task_outcomes(task):
        suite, task_id = task
        outcomes = {}
        for path in sorted((output / f"{suite}_task{task_id}" / suite / "_parts").glob("*_results.json")):
            result = json.loads(path.read_text())
            if int(result["task_id"]) != task_id:
                continue
            for episode in result.get("success_episodes", []):
                outcomes[int(episode)] = True
            for episode in result.get("failure_episodes", []):
                outcomes[int(episode)] = False
        return {episode: success for episode, success in outcomes.items() if 0 <= episode < args.num_trials}

    pending_tasks = [task for task in tasks if len(task_outcomes(task)) < args.num_trials]

    print(f"{args.backend}: {len(tasks)} tasks; selection={'all' if args.all_tasks else 'subset'}; {args.num_trials} trials/task; GPUs={gpu_ids}", flush=True)
    if args.dry_run:
        print(shlex.join(persistent_command()) if pending_tasks else "All requested episodes are already complete.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    (output / "sample_tasks.json").write_text(json.dumps(tasks, indent=2))
    (output / "pending_tasks.csv").write_text("".join(f"{suite},{task_id}\n" for suite, task_id in pending_tasks))
    active = None

    def stop(*_):
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    exit_code = 0
    if pending_tasks:
        from evaluation.run import _prepare_libero_env
        env = _prepare_libero_env(args.backend)
        env["PYTHONUNBUFFERED"] = "1"
        print(f"Persistent dispatch: {len(pending_tasks)} pending tasks; one resident model/GPU; "
              f"{args.max_tasks_per_gpu} environment client(s)/GPU", flush=True)
        active = subprocess.Popen(persistent_command(), cwd=ROOT, env=env, start_new_session=True)
        exit_code = active.wait()
        if exit_code:
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    records = []
    for task in tasks:
        suite, task_id = task
        outcomes = task_outcomes(task)
        episodes = len(outcomes)
        successes = sum(outcomes.values())
        records.append(dict(suite=suite, task_id=task_id, exit_code=0 if episodes == args.num_trials else (exit_code or 1),
                            successes=successes, episodes=episodes,
                            success_rate=successes / episodes if episodes else None))
    records.sort(key=lambda r: tasks.index((r["suite"], r["task_id"])))
    complete = len(records) == len(tasks) and all(
        r["exit_code"] == 0 and r["episodes"] == args.num_trials for r in records
    )
    episodes = sum(r["episodes"] for r in records)
    successes = sum(r["successes"] for r in records)
    summary = dict(backend=args.backend, sample_size=len(tasks), tasks=tasks, sample_seed=args.sample_seed, seed=args.seed,
                   selection="all" if args.all_tasks else "subset",
                   dispatch_unit="persistent_worker", max_tasks_per_gpu=args.max_tasks_per_gpu,
                   compile=not args.no_compile,
                   description_source="mapping",
                   replan_steps=args.replan_steps, num_inference_steps=args.num_inference_steps,
                   episodes_per_task=args.num_trials,
                   all_tasks_completed=complete, results=records, total_successes=successes,
                   total_episodes=episodes, overall_success_rate=successes / episodes if episodes else None)
    if args.backend == "libero-plus":
        classification = json.loads((source_dir("libero-plus") / "libero/libero/benchmark/task_classification.json").read_text())
        categories_by_task = {(suite, int(item["id"]) - 1): item["category"]
                              for suite, items in classification.items() for item in items}
        categories = {}
        for record in records:
            category = categories_by_task[(record["suite"], record["task_id"])]
            stats = categories.setdefault(category, dict(tasks=0, episodes=0, successes=0))
            stats["tasks"] += 1
            stats["episodes"] += record["episodes"]
            stats["successes"] += record["successes"]
        for stats in categories.values():
            stats["success_rate"] = stats["successes"] / stats["episodes"] if stats["episodes"] else None
        summary["categories"] = categories
        rates = [v["success_rate"] for v in categories.values()]
        summary["seven_perturbation_macro_average"] = (
            sum(rates) / 7 if len(rates) == 7 and all(r is not None for r in rates) else None)
        summary["rate_units"] = "fraction (0 to 1)"
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary: {output / 'summary.json'}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
