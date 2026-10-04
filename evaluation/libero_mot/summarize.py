#!/usr/bin/env python3
"""Summarize LIBERO evaluation results produced by eval_libero.py.

Usage:
    python evaluation/libero_mot/summarize.py --out-dir outputs/eval/step16000
    python evaluation/libero_mot/summarize.py --out-dir outputs/eval/step16000 --suite libero_spatial
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def format_time(seconds: float) -> str:
    s = round(seconds)
    if s < 60:
        return f"{s:02d}s"
    if s < 3600:
        return f"{s // 60:02d}m{s % 60:02d}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h:02d}h{m:02d}m{s % 60:02d}s"


def _suite_result_files(suite_dir: Path) -> list[Path]:
    parts_dir = suite_dir / "_parts"
    if parts_dir.exists():
        return sorted(parts_dir.glob("*_results.json"))
    return sorted(suite_dir.glob("*_results.json"))


def _write_task_aggregates(out_dir: Path, task_results: dict) -> list[Path]:
    written_paths = []
    for result in task_results.values():
        suite_dir = out_dir / result["suite"]
        suite_dir.mkdir(parents=True, exist_ok=True)
        aggregate_path = suite_dir / f"task{result['task_id']}_results.json"
        with open(aggregate_path, "w") as f:
            json.dump(result, f, indent=2)
        written_paths.append(aggregate_path)
    return written_paths


def collect_results(out_dir: Path, suites: list, task_filter: dict[str, set[int]] | None = None) -> dict:
    """Scan output directory and return per-task results."""
    task_results = {}
    suite_stats = defaultdict(lambda: {
        "tasks": 0, "trials": 0, "successes": 0, "time": 0.0, "max_time": 0.0, "_task_ids": set()
    })
    global_start = None
    global_end = None

    for suite in suites:
        suite_dir = out_dir / suite
        if not suite_dir.exists():
            continue
        result_files = _suite_result_files(suite_dir)
        for fname in result_files:
            with open(fname) as f:
                r = json.load(f)
            task_id = r.get("task_id")
            if task_id is None:
                task_id = int(fname.stem.rsplit("_task", 1)[1].split("_", 1)[0])
            if task_filter is not None and task_id not in task_filter.get(suite, set()):
                continue
            key = f"{suite}_{task_id}"
            if key not in task_results:
                task_results[key] = {
                    "suite": suite,
                    "task_id": task_id,
                    "task_description": r.get("task_description", ""),
                    "successes": 0,
                    "total_episodes": 0,
                    "success_rate": 0.0,
                    "duration": 0.0,
                    "success_episodes": [],
                    "failure_episodes": [],
                }
            task_results[key]["successes"] += r["successes"]
            task_results[key]["total_episodes"] += r["total_episodes"]
            task_results[key]["duration"] += r.get("duration", 0.0)
            task_results[key]["success_episodes"].extend(r.get("success_episodes", []))
            task_results[key]["failure_episodes"].extend(r.get("failure_episodes", []))
            task_results[key]["success_rate"] = (
                task_results[key]["successes"] / max(task_results[key]["total_episodes"], 1) * 100
            )
            st = suite_stats[suite]
            st["_task_ids"].add(task_id)
            st["trials"] += r["total_episodes"]
            st["successes"] += r["successes"]
            st["time"] += r.get("duration", 0.0)
            start_time = r.get("start_time")
            end_time = r.get("end_time")
            if start_time:
                parsed = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
                global_start = parsed if global_start is None else min(global_start, parsed)
            if end_time:
                parsed = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
                global_end = parsed if global_end is None else max(global_end, parsed)

    for r in task_results.values():
        r["success_episodes"] = sorted(set(r["success_episodes"]))
        r["failure_episodes"] = sorted(set(r["failure_episodes"]))
        suite_stats[r["suite"]]["max_time"] = max(suite_stats[r["suite"]]["max_time"], r["duration"])

    for st in suite_stats.values():
        st["tasks"] = len(st.pop("_task_ids"))

    return task_results, suite_stats, global_start, global_end


def aggregate_single_task(out_dir: Path, suite: str, task_id: int) -> Path | None:
    task_results, _, _, _ = collect_results(out_dir, [suite], task_filter={suite: {task_id}})
    if not task_results:
        return None
    written = _write_task_aggregates(out_dir, task_results)
    return written[0] if written else None


def print_summary(out_dir: Path, suites: list):
    task_results, suite_stats, global_start, global_end = collect_results(out_dir, suites)

    if not suite_stats:
        print(f"No results found in {out_dir}")
        return

    print("\n=== Evaluation Summary ===")
    total_sr_sum, total_suites = 0.0, 0
    wall_time_seconds = None
    if global_start is not None and global_end is not None:
        wall_time_seconds = max((global_end - global_start).total_seconds(), 0.0)
        print(f"start time          : {global_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"end time            : {global_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"wall time           : {format_time(wall_time_seconds)}")

    rows = []
    for suite in suites:
        if suite not in suite_stats:
            continue
        st = suite_stats[suite]
        sr = st["successes"] / max(st["trials"], 1) * 100
        avg_t = st["time"] / max(st["tasks"], 1)
        print(f"\n{suite}:")
        print(f"  tasks completed : {st['tasks']}")
        print(f"  total trials    : {st['trials']}")
        print(f"  successes       : {st['successes']}")
        print(f"  success rate    : {sr:.2f}%")
        print(f"  total time      : {format_time(st['time'])}")
        print(f"  avg time/task   : {format_time(avg_t)}")
        print(f"  longest task    : {format_time(st['max_time'])}")
        rows.append((suite, sr, avg_t, st["max_time"]))
        total_sr_sum += sr
        total_suites += 1

    if total_suites > 1:
        all_tasks = sum(s["tasks"] for s in suite_stats.values())
        all_time = sum(s["time"] for s in suite_stats.values())
        max_t = max(s["max_time"] for s in suite_stats.values())
        print(f"\nOverall ({total_suites} suites):")
        print(f"  avg success rate : {total_sr_sum / total_suites:.2f}%")
        print(f"  total time       : {format_time(all_time)}")
        print(f"  avg time/task    : {format_time(all_time / max(all_tasks, 1))}")
        print(f"  longest task     : {format_time(max_t)}")

    # Per-task table
    print("\n=== Per-Task Success Rates ===")
    print(f"{'Task':<30} {'Desc':<50} {'Rate':>8} {'Trials':>8}")
    print("-" * 100)
    for key in sorted(task_results):
        r = task_results[key]
        desc = r["task_description"][:48]
        print(f"{key:<30} {desc:<50} {r['success_rate']:>7.1f}% {r['total_episodes']:>8}")

    _write_task_aggregates(out_dir, task_results)

    # Save summary JSON
    summary = {
        "suite_stats": {
            k: {
                "tasks": v["tasks"],
                "trials": v["trials"],
                "successes": v["successes"],
                "success_rate": v["successes"] / max(v["trials"], 1) * 100,
                "total_time": v["time"],
                "max_time": v["max_time"],
            }
            for k, v in suite_stats.items()
        },
        "overall": {
            "avg_success_rate": total_sr_sum / max(total_suites, 1),
            "num_suites": total_suites,
            "start_time": global_start.strftime("%Y-%m-%d %H:%M:%S") if global_start else None,
            "end_time": global_end.strftime("%Y-%m-%d %H:%M:%S") if global_end else None,
            "wall_time_seconds": wall_time_seconds,
        },
        "task_results": task_results,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    # Save CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("suite,success_rate,avg_time_s,max_time_s\n")
        for suite, sr, avg_t, max_t in rows:
            f.write(f"{suite},{sr:.2f},{avg_t:.1f},{max_t:.1f}\n")
        if total_suites > 1:
            all_time = sum(s["time"] for s in suite_stats.values())
            all_tasks = sum(s["tasks"] for s in suite_stats.values())
            f.write(f"overall,{total_sr_sum / total_suites:.2f},{all_time / max(all_tasks, 1):.1f},{max_t:.1f}\n")
    print(f"CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize eval_libero.py results")
    parser.add_argument("--out-dir", required=True, help="Root output directory")
    parser.add_argument(
        "--suite", nargs="+", default=SUITE_NAMES,
        choices=SUITE_NAMES, help="Which suites to summarize (default: all)"
    )
    parser.add_argument("--task-id", type=int, default=None, help="Aggregate only one task id")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.task_id is not None:
        if len(args.suite) != 1:
            raise SystemExit("--task-id requires exactly one --suite value")
        aggregate_path = aggregate_single_task(out_dir, args.suite[0], args.task_id)
        if aggregate_path is None:
            raise SystemExit(
                f"No partial results found for suite={args.suite[0]} task_id={args.task_id} under {out_dir}"
            )
        print(f"Task aggregate written to {aggregate_path}")
        return
    print_summary(out_dir, args.suite)


if __name__ == "__main__":
    main()
