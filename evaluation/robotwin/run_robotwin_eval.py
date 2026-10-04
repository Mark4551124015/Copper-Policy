#!/usr/bin/env python3
from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


faulthandler.enable(all_threads=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RobotWin eval driver with configurable episode count/output dir.")
    parser.add_argument("--robotwin-root", required=True, help="Path to RoboTwin root.")
    parser.add_argument("--config", required=True, help="Path to RoboTwin deploy yaml, relative to robotwin root or absolute.")
    parser.add_argument("--overrides", nargs=argparse.REMAINDER, help="Key/value pairs identical to RoboTwin script/eval_policy.py")
    return parser.parse_args()


def parse_override_pairs(pairs: list[str] | None) -> dict[str, Any]:
    if not pairs:
        return {}
    if len(pairs) % 2 != 0:
        raise ValueError(f"Override pairs must be even, got {len(pairs)} items: {pairs}")

    override_dict: dict[str, Any] = {}
    for i in range(0, len(pairs), 2):
        key = pairs[i].lstrip("--")
        value = pairs[i + 1]
        try:
            value = eval(value)
        except Exception:
            pass
        override_dict[key] = value
    return override_dict


def resolve_config_path(config_arg: str, robotwin_root: Path) -> Path:
    path = Path(config_arg)
    if path.is_absolute():
        return path
    return (robotwin_root / path).resolve()


def prepare_robotwin_imports(robotwin_root: Path) -> None:
    os.chdir(robotwin_root)

    extra_paths = [
        robotwin_root,
        robotwin_root / "policy",
        robotwin_root / "description" / "utils",
        robotwin_root / "script",
    ]
    for path in reversed(extra_paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def validate_robotwin_imports() -> None:
    try:
        planner_mod = importlib.import_module("envs.robot.planner")
    except Exception as exc:
        raise ImportError(
            "Failed to import RoboTwin module `envs.robot.planner`. "
            "Please verify the RoboTwin evaluation environment and its dependencies."
        ) from exc

    if not hasattr(planner_mod, "CuroboPlanner"):
        raise ImportError(
            "RoboTwin imported `envs.robot.planner`, but `CuroboPlanner` is not defined. "
            "This usually means the internal curobo imports inside RoboTwin failed. "
            "No fallback is used here by design."
        )

    try:
        importlib.import_module("envs.robot.robot")
    except Exception as exc:
        raise ImportError(
            "Failed to import RoboTwin module `envs.robot.robot` even though `CuroboPlanner` exists. "
            "Please verify the RoboTwin evaluation environment."
        ) from exc


def maybe_run_render_self_test(render_checker, *, default_skip: bool = False) -> None:
    raw = os.environ.get("ROBOTWIN_SKIP_RENDER_TEST")
    if raw is None:
        skip = bool(default_skip)
    else:
        skip = str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}
    if skip:
        print("[robotwin] skip Sapien render self-test")
        return
    render_checker()


def _parse_bool_flag(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _task_instruction_fallback(robotwin_root: Path, task_name: str) -> str:
    instruction_path = robotwin_root / "description" / "task_instruction" / f"{task_name}.json"
    try:
        payload = json.loads(instruction_path.read_text(encoding="utf-8"))
    except Exception:
        return task_name.replace("_", " ")

    full_description = str(payload.get("full_description") or "").strip()
    if full_description:
        return full_description

    for key in ("seen", "unseen"):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            text = str(item).strip()
            if text and "{" not in text and "}" not in text:
                return text
    return task_name.replace("_", " ")


def _resolve_episode_instruction(
    *,
    robotwin_root: Path,
    task_name: str,
    instruction_type: str,
    episode_info: dict[str, Any] | None,
    scene_seed: int,
) -> str:
    if episode_info is not None:
        try:
            from description.utils.generate_episode_instructions import generate_episode_descriptions

            # The pinned upstream generator uses the global Python RNG.
            # Isolate its state so instructions depend on this scene, including
            # when resuming or assigning tasks to different workers.
            instruction_rng = random.Random(f"{task_name}:{scene_seed}:{instruction_type}")
            previous_random_state = random.getstate()
            try:
                random.setstate(instruction_rng.getstate())
                results = generate_episode_descriptions(task_name, [episode_info], 1)
            finally:
                random.setstate(previous_random_state)
            if results:
                candidates = results[0].get(instruction_type) or results[0].get("seen") or results[0].get("unseen") or []
                if candidates:
                    return str(instruction_rng.choice(candidates))
        except Exception:
            pass

    return _task_instruction_fallback(robotwin_root, task_name)


def _should_request_observation(model: Any, *, skip_get_obs_within_replan: bool) -> bool:
    if not skip_get_obs_within_replan:
        return True
    request_fn = getattr(model, "should_request_observation", None)
    if request_fn is None:
        return True
    try:
        return bool(request_fn())
    except Exception:
        return True


def _refresh_video_frame_cache(task_env) -> None:
    eval_video_path = getattr(task_env, "eval_video_path", None)
    if eval_video_path is None:
        return
    if not hasattr(task_env, "_update_render") or not hasattr(task_env, "cameras"):
        return

    try:
        task_env._update_render()
        task_env.cameras.update_picture()
        rgb = task_env.cameras.get_rgb()
        observation_cache = getattr(task_env, "now_obs", None)
        if not isinstance(observation_cache, dict):
            observation_cache = {}
            task_env.now_obs = observation_cache
        obs_dict = observation_cache.setdefault("observation", {})
        config = task_env.cameras.get_config()
        for camera_name, payload in config.items():
            obs_dict.setdefault(camera_name, {}).update(payload)
        for camera_name, payload in rgb.items():
            obs_dict.setdefault(camera_name, {}).update(payload)
    except Exception:
        pass


def _episode_log_dir(save_dir: Path) -> Path:
    return save_dir / "episodes"


def _load_existing_episode_records(save_dir: Path) -> list[dict[str, Any]]:
    episode_dir = _episode_log_dir(save_dir)
    if not episode_dir.is_dir():
        return []

    records_by_index: dict[int, dict[str, Any]] = {}
    for path in sorted(episode_dir.glob("episode_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            episode_index = int(payload.get("episode_index"))
        except Exception:
            continue
        records_by_index[episode_index] = payload

    records: list[dict[str, Any]] = []
    expected = 0
    while expected in records_by_index:
        records.append(records_by_index[expected])
        expected += 1
    return records


def _write_episode_record(save_dir: Path, payload: dict[str, Any]) -> None:
    episode_dir = _episode_log_dir(save_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    episode_index = int(payload["episode_index"])
    out_path = episode_dir / f"episode_{episode_index:05d}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_task_progress(save_dir: Path, payload: dict[str, Any]) -> None:
    progress_path = save_dir / "progress.json"
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_local_eval_policy(
    *,
    robotwin_root: Path,
    usr_args: dict[str, Any],
    robotwin_eval,
    task_name: str,
    task_env,
    env_args: dict[str, Any],
    model,
    st_seed: int,
    test_num: int,
    video_size: str | None,
    instruction_type: str,
    save_dir: Path,
) -> tuple[int, int]:
    # Upstream play_once initializes success-check fields and instruction
    # placeholders as well as filtering seeds. Reuse the same task instance
    # and rebuild the scene with the same seed for the policy rollout.
    expert_check = not _parse_bool_flag(usr_args.get("skip_expert_check"), default=False)
    skip_get_obs_within_replan = _parse_bool_flag(usr_args.get("skip_get_obs_within_replan"), default=True)

    existing_records = _load_existing_episode_records(save_dir)
    resumed_episode_count = len(existing_records)
    resumed_success_count = sum(1 for item in existing_records if bool(item.get("success", False)))

    task_env.suc = resumed_success_count
    task_env.test_num = resumed_episode_count

    now_id = resumed_episode_count
    succ_seed = resumed_episode_count
    policy_name = env_args["policy_name"]
    eval_func = robotwin_eval.eval_function_decorator(policy_name, "eval")
    reset_func = robotwin_eval.eval_function_decorator(policy_name, "reset_model")

    if existing_records:
        try:
            now_seed = int(existing_records[-1].get("seed")) + 1
        except Exception:
            now_seed = st_seed + resumed_episode_count
        print(
            f"[robotwin] resume task={task_name} from episode {resumed_episode_count} "
            f"(successes={resumed_success_count}, next_seed={now_seed})"
        )
    else:
        now_seed = st_seed
    clear_cache_freq = env_args["clear_cache_freq"]
    env_args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = env_args["render_freq"]
        env_args["render_freq"] = 0
        episode_info = None
        expert_setup_s = None

        if expert_check:
            try:
                expert_t0 = time.perf_counter()
                task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **env_args)
                episode_info = task_env.play_once()
                expert_setup_s = time.perf_counter() - expert_t0
                task_env.close_env()
            except robotwin_eval.UnStableError:
                task_env.close_env()
                now_seed += 1
                env_args["render_freq"] = render_freq
                continue
            except Exception:
                task_env.close_env()
                now_seed += 1
                env_args["render_freq"] = render_freq
                print("error occurs !")
                continue

            if not (task_env.plan_success and task_env.check_success()):
                now_seed += 1
                env_args["render_freq"] = render_freq
                continue

        succ_seed += 1
        env_args["render_freq"] = render_freq

        episode_t0 = time.perf_counter()
        try:
            setup_t0 = time.perf_counter()
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **env_args)
            setup_s = time.perf_counter() - setup_t0
        except robotwin_eval.UnStableError:
            task_env.close_env()
            succ_seed -= 1
            now_seed += 1
            continue
        except Exception:
            task_env.close_env()
            succ_seed -= 1
            now_seed += 1
            print("error occurs during eval setup!")
            continue
        episode_info_payload = None
        if isinstance(episode_info, dict):
            # Empty metadata is valid for tasks without placeholders.
            episode_info_payload = episode_info.get("info", {})
        if episode_info_payload is None:
            fallback_info = getattr(task_env, "info", None)
            if isinstance(fallback_info, dict):
                episode_info_payload = fallback_info.get("info")
        instruction_t0 = time.perf_counter()
        instruction = _resolve_episode_instruction(
            robotwin_root=robotwin_root,
            task_name=task_name,
            instruction_type=instruction_type,
            episode_info=episode_info_payload,
            scene_seed=now_seed,
        )
        if "{" in instruction or "}" in instruction:
            raise ValueError(
                f"Unresolved instruction placeholders for {task_name}, seed {now_seed}: {instruction!r}. "
                "Keep skip_expert_check=false to obtain episode-specific instruction metadata."
            )
        instruction_s = time.perf_counter() - instruction_t0
        task_env.set_instruction(instruction=instruction)

        episode_video_path = None
        if task_env.eval_video_path is not None:
            episode_video_path = Path(task_env.eval_video_path) / f"episode{task_env.test_num}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(episode_video_path),
                ],
                stdin=subprocess.PIPE,
            )
            task_env._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        reset_func(model)
        rollout_t0 = time.perf_counter()
        while task_env.take_action_cnt < task_env.step_lim:
            if _should_request_observation(model, skip_get_obs_within_replan=skip_get_obs_within_replan):
                observation = task_env.get_obs()
            else:
                observation = None
                _refresh_video_frame_cache(task_env)
            eval_func(task_env, model, observation)
            if task_env.eval_success:
                succ = True
                break
        rollout_wall_s = time.perf_counter() - rollout_t0

        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()
            if ffmpeg.returncode != 0:
                raise RuntimeError(f"Video encoding failed: {episode_video_path}")
            episode_video_path = episode_video_path.rename(
                episode_video_path.with_name(
                    f"{episode_video_path.stem}-{'succ' if succ else 'fail'}.mp4"
                )
            )

        if succ:
            task_env.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        close_env_t0 = time.perf_counter()
        now_id += 1
        task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        close_env_s = time.perf_counter() - close_env_t0
        total_episode_s = time.perf_counter() - episode_t0
        timing_payload = {
            "setup_s": float(setup_s),
            "instruction_s": float(instruction_s),
            "rollout_wall_s": float(rollout_wall_s),
            "close_env_s": float(close_env_s),
            "total_episode_s": float(total_episode_s),
        }
        if expert_setup_s is not None:
            timing_payload["expert_setup_s"] = float(expert_setup_s)

        episode_record = {
            "episode_index": int(task_env.test_num),
            "seed": int(now_seed),
            "instruction_seed": f"{task_name}:{now_seed}:{instruction_type}",
            "success": bool(succ),
            "instruction": str(instruction),
            "task_name": str(task_name),
            "instruction_type": str(instruction_type),
            "step_count": int(getattr(task_env, "take_action_cnt", 0)),
            "video_path": str(episode_video_path) if episode_video_path is not None else None,
            "timing": timing_payload,
        }
        if hasattr(model, "get_timing_rollout"):
            try:
                timing_payload["policy"] = dict(model.get_timing_rollout())
            except Exception:
                pass
        _write_episode_record(save_dir, episode_record)

        if task_env.render_freq:
            task_env.viewer.close()

        task_env.test_num += 1
        _write_task_progress(
            save_dir,
            {
                "task_name": str(task_name),
                "completed_episodes": int(task_env.test_num),
                "target_episodes": int(test_num),
                "successes": int(task_env.suc),
                "success_rate": (float(task_env.suc) / float(task_env.test_num)) if task_env.test_num > 0 else 0.0,
                "next_seed": int(now_seed + 1),
                "last_episode_timing": timing_payload,
            },
        )
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{env_args['policy_name']}\033[0m | "
            f"\033[92m{env_args['task_config']}\033[0m | \033[91m{env_args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => "
            f"\033[95m{round(task_env.suc / task_env.test_num * 100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    return now_seed, task_env.suc


def run_eval(usr_args: dict[str, Any], robotwin_eval, render_checker, *, robotwin_root: Path,
             resident_model: list | None = None) -> int:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]

    render_checker()

    get_model = robotwin_eval.eval_function_decorator(policy_name, "get_model")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["eval_video_log"] = _parse_bool_flag(usr_args.get("eval_video_log"), default=bool(args.get("eval_video_log", False)))

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(robotwin_eval.CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_name: str) -> str:
        robot_file = embodiment_types[embodiment_name]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(robotwin_eval.CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = robotwin_eval.get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = robotwin_eval.get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    eval_output_dir = usr_args.get("eval_output_dir")
    if eval_output_dir:
        save_dir = Path(str(eval_output_dir)).expanduser().resolve()
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_save_dir = None
    video_size = None
    if args["eval_video_log"]:
        video_save_dir = save_dir
        cfg = robotwin_eval.get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(cfg["w"]) + "x" + str(cfg["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))
    print(
        "\033[94mHead Camera Config:\033[0m "
        + str(args["camera"]["head_camera_type"])
        + f", {args['camera']['collect_head_camera']}"
    )
    print(
        "\033[94mWrist Camera Config:\033[0m "
        + str(args["camera"]["wrist_camera_type"])
        + f", {args['camera']['collect_wrist_camera']}"
    )
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print(f"\033[94mEval Episodes:\033[0m {int(usr_args['eval_num_episodes'])}")
    print(f"\033[94mEval Output:\033[0m {save_dir}")
    print("\n==================================")

    task_env = robotwin_eval.class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = int(usr_args["seed"])
    st_seed = 100000 * (1 + seed)
    test_num = int(usr_args.get("eval_num_episodes", 100))

    if resident_model is None:
        model = get_model(usr_args)
    else:
        if not resident_model:
            resident_model.append(get_model(usr_args))
        model = resident_model[0]
    # Record effective settings rather than only launcher defaults.
    settings = {
        "task_name": task_name,
        "checkpoint": str(ckpt_setting),
        "instruction_type": instruction_type,
        "instruction_seed_scheme": "task_name:scene_seed:instruction_type",
        "expert_check": not _parse_bool_flag(usr_args.get("skip_expert_check"), default=False),
        "video": bool(args["eval_video_log"]),
    }
    for key in (
        "current_view_resize_mode", "per_view_sizes", "action_horizon",
        "replan_steps", "num_inference_steps", "seed", "teacher_type",
        "robotwin_action_space", "robotwin_qpos_action_mode", "text_max_length",
        "joint_future_denoising", "future_semantic_steps", "compile_infer_action",
        "compile_vjepa", "compile_mode", "dtype",
    ):
        value = getattr(model, key, None)
        settings[key] = str(value) if key == "dtype" else value
    from importlib.metadata import PackageNotFoundError, version
    settings["packages"] = {}
    for package in ("torch", "sapien", "mplib", "toppra", "warp-lang", "numpy", "scipy", "transformers"):
        try:
            settings["packages"][package] = version(package)
        except PackageNotFoundError:
            settings["packages"][package] = None
    settings["resolved_config"] = {
        key: getattr(getattr(model, "cfg", None), key, None)
        for key in (
            "instruction_template", "text_emb_use_padding_mask", "default_joint_future_denoising",
            "action_attends_future_video", "action_frame_ratio", "robotwin_current_view_resize_mode",
        )
    }
    (save_dir / "evaluation_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )
    st_seed, suc_num = run_local_eval_policy(
        robotwin_root=robotwin_root,
        usr_args=usr_args,
        robotwin_eval=robotwin_eval,
        task_name=task_name,
        task_env=task_env,
        env_args=args,
        model=model,
        st_seed=st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        save_dir=save_dir,
    )

    result_payload = {
        "timestamp": current_time,
        "task_name": task_name,
        "task_config": task_config,
        "instruction_type": instruction_type,
        "ckpt_setting": ckpt_setting,
        "eval_num_episodes": test_num,
        "successes": int(suc_num),
        "success_rate": float(suc_num) / float(test_num) if test_num > 0 else 0.0,
        "save_dir": str(save_dir),
    }
    result_txt_path = save_dir / "_result.txt"
    with open(result_txt_path, "w", encoding="utf-8") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Successes: {suc_num}/{test_num}\n")
        file.write(f"Success Rate: {result_payload['success_rate']:.6f}\n")

    result_json_path = save_dir / "result.json"
    result_json_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(f"Data has been saved to {result_txt_path}")
    print(f"JSON summary saved to {result_json_path}")
    return 0


def main() -> int:
    args = parse_args()
    robotwin_root = Path(args.robotwin_root).expanduser().resolve()
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    config_path = resolve_config_path(args.config, robotwin_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    prepare_robotwin_imports(robotwin_root)
    # Resolve Copper's adapter locally, even if an existing upstream checkout
    # has a different mot_policy installed under its policy/ directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    validate_robotwin_imports()
    robotwin_eval = importlib.import_module("script.eval_policy")
    render_mod = importlib.import_module("test_render")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.update(parse_override_pairs(args.overrides))

    maybe_run_render_self_test(render_mod.Sapien_TEST, default_skip=False)
    tasks = config.pop("eval_tasks", None)
    if tasks is None:
        return run_eval(config, robotwin_eval, lambda: None, robotwin_root=robotwin_root)
    output_root = Path(config["eval_output_dir"])
    # Match the original batch evaluator: construct the policy before creating
    # the first task environment and keep it resident for the full task shard.
    get_model = robotwin_eval.eval_function_decorator(config["policy_name"], "get_model")
    resident_model = [get_model(config)]
    for task in tasks:
        task_config = dict(config, task_name=task, eval_output_dir=str(output_root / task))
        print(f"=== Resident worker: {task} ===", flush=True)
        run_eval(task_config, robotwin_eval, lambda: None, robotwin_root=robotwin_root,
                 resident_model=resident_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
