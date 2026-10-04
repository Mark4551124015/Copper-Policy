#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "mot_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "train_out" in parts:
        idx = parts.index("train_out")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 2 < len(parts):
            return f"{parts[idx + 1]}_{parts[idx + 2]}"
    return ckpt_path.stem


def _format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _describe_return_code(return_code: int) -> str:
    if return_code >= 0:
        return f"return code {return_code}"
    signum = -return_code
    try:
        signal_name = signal.Signals(signum).name
    except ValueError:
        signal_name = f"SIG{signum}"
    return f"signal {signum} ({signal_name})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Semantic MoT checkpoint on RobotWin via RoboTwin.")
    parser.add_argument("--ckpt", required=True, help="Path to Semantic MoT checkpoint.")
    parser.add_argument("--config-name", default="config_robotwin_train", help="wan_va config key.")
    parser.add_argument("--task-name", required=True, help="RobotWin task name.")
    parser.add_argument("--tasks", help="Comma-separated tasks evaluated with one resident model.")
    parser.add_argument("--task-config", default="demo_clean", help="RoboTwin task config yaml basename.")
    parser.add_argument("--instruction-type", default="seen", help="Instruction split used by RoboTwin.")
    parser.add_argument("--eval-num-episodes", type=int, default=100, help="Number of episodes to evaluate.")
    parser.add_argument("--robotwin-root", default=str(PROJECT_ROOT / "third_party" / "RoboTwin"), help="Path to RoboTwin root.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id forwarded via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--device", default="cuda", help="Torch device visible inside policy.")
    parser.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--t5-dir", default="pretrained_weights/text_encoder/wan22_ti2v_5b", help="Directory containing tokenizer/text_encoder.")
    parser.add_argument("--dino-model", default="facebook/dinov2-with-registers-large")
    parser.add_argument("--vjepa-model", default="vjepa2_1_vit_large_384")
    parser.add_argument(
        "--replan-steps", type=int, default=24,
        help="Actions to execute per inference call (0 = auto = action_per_frame).",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--future-semantic-steps", type=int)
    parser.add_argument("--t5-max-length", type=int, help="-1 disables truncation; otherwise fixed max length.")
    parser.add_argument("--use-proprio", dest="use_proprio", action="store_true")
    parser.add_argument("--no-proprio", dest="use_proprio", action="store_false")
    joint_future_group = parser.add_mutually_exclusive_group()
    joint_future_group.add_argument(
        "--joint-future-denoising",
        dest="joint_future_denoising",
        action="store_true",
        help=(
            "Denoise pooled future semantic queries together with actions during eval, "
            "instead of following the config default."
        ),
    )
    joint_future_group.add_argument(
        "--no-joint-future-denoising",
        dest="joint_future_denoising",
        action="store_false",
        help="Force cached FastWAM-style action-only inference even if the config default is joint.",
    )
    parser.add_argument("--timing-enabled", action="store_true")
    parser.add_argument("--no-compile", action="store_true", help="Use eager inference for all policy stages")
    parser.add_argument("--skip-get-obs-within-replan", action="store_true", default=True)
    parser.add_argument("--no-skip-get-obs-within-replan", dest="skip_get_obs_within_replan", action="store_false")
    parser.add_argument("--out-dir", help="Optional explicit output dir.")
    parser.set_defaults(use_proprio=None, joint_future_denoising=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    ckpt_path = _resolve_path(args.ckpt, base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    robotwin_root = _resolve_path(args.robotwin_root, base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_source_dir = (PROJECT_ROOT / "evaluation" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    t5_dir = _resolve_path(args.t5_dir, base=PROJECT_ROOT)
    if not t5_dir.exists():
        raise FileNotFoundError(f"T5 directory not found: {t5_dir}")

    ckpt_tag = _resolve_ckpt_tag(ckpt_path)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        run_output_dir = _resolve_path(args.out_dir, base=PROJECT_ROOT)
    else:
        run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin_mot" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / f"eval_{args.task_name}_{run_ts}.log"
    task_output_dir = run_output_dir / args.task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    overrides: list[str] = []
    _append_override(overrides, "task_name", args.task_name)
    if args.tasks:
        overrides.extend(["--eval_tasks", repr(args.tasks.split(","))])
    _append_override(overrides, "task_config", args.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "config_name", args.config_name)
    _append_override(overrides, "t5_dir", str(t5_dir))
    _append_override(overrides, "seed", args.seed)
    _append_override(overrides, "policy_name", POLICY_NAME)
    _append_override(overrides, "instruction_type", args.instruction_type)
    _append_override(overrides, "eval_num_episodes", args.eval_num_episodes)
    _append_override(overrides, "eval_output_dir", str(run_output_dir if args.tasks else task_output_dir))
    _append_override(overrides, "device", args.device)
    _append_override(overrides, "mixed_precision", args.mixed_precision)
    _append_override(overrides, "dino_model", args.dino_model)
    _append_override(overrides, "vjepa_model", args.vjepa_model)
    _append_override(overrides, "replan_steps", args.replan_steps)
    _append_override(overrides, "num_inference_steps", args.num_inference_steps)
    _append_override(overrides, "use_proprio", args.use_proprio)
    _append_override(overrides, "joint_future_denoising", args.joint_future_denoising)
    _append_override(overrides, "future_semantic_steps", args.future_semantic_steps)
    _append_override(overrides, "t5_max_length", args.t5_max_length)
    _append_override(overrides, "timing_enabled", args.timing_enabled)
    _append_override(overrides, "skip_get_obs_within_replan", args.skip_get_obs_within_replan)

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "evaluation" / "robotwin" / "run_robotwin_eval.py"),
        "--robotwin-root",
        str(robotwin_root),
        "--config",
        str(policy_source_dir / "deploy_policy.yml"),
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONFAULTHANDLER", "1")
    if args.no_compile:
        env["COPPER_DISABLE_COMPILE"] = "1"
        env["ROBOTWIN_COMPILE_INFER_ACTION"] = "0"

    with open(log_file, "w", encoding="utf-8") as log_f:
        log_f.write(f"[robotwin-mot] timestamp: {run_ts}\n")
        log_f.write(f"[robotwin-mot] python: {sys.executable}\n")
        log_f.write(f"[robotwin-mot] cwd: {PROJECT_ROOT}\n")
        log_f.write(f"[robotwin-mot] robotwin_root: {robotwin_root}\n")
        log_f.write(f"[robotwin-mot] task: {args.task_name}\n")
        log_f.write(f"[robotwin-mot] command: {_format_command(cmd)}\n")
        log_f.write(f"[robotwin-mot] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        log_f.write(f"[robotwin-mot] PYTHONFAULTHANDLER={env.get('PYTHONFAULTHANDLER', '')}\n")
        log_f.write("\n")
        log_f.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()
        log_f.write(f"\n[robotwin-mot] process exit: {_describe_return_code(return_code)}\n")
        log_f.flush()

    if return_code != 0:
        raise RuntimeError(f"RobotWin evaluation failed with {_describe_return_code(return_code)}. Log: {log_file}")

    print(f"[robotwin-mot] log: {log_file}")
    print(f"[robotwin-mot] output: {task_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
