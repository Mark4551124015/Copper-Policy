# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Serve the MTLBot Semantic MoT RealBot policy over the openpi wire protocol.

Drop-in replacement for openpi's ``serve_pi05_bimanual_ros_policy.py``
(``_upstream_snapshots/agilex_deploy/openpi``): length-prefixed JSON over TCP,
same request/response schema, same CLI conventions, default port 8767.

The existing ROS bridge (``scripts/ros_pi05_piper_bridge*.py``) connects to
``--policy-host HOST --policy-port PORT`` unchanged; requests are:

    {"request_id", "prompt", "state": [14 floats],
     "images": {"cam_high", "cam_left_wrist", "cam_right_wrist": base64 JPEG}}

and responses are:

    {"actions": [[14] x N], "decoded_action_shape", "model_action_dim": 14,
     "real_action_dim": 14, "action_space": "ros_abs", "timing", "request_id"}

or ``{"error", "request_id"}`` on failure.

The new ``bridge_new`` sends ``session_id`` and ``unirobot_request_timing``
on the same primary endpoint.  The server caches the prior *full* chunk per
session, computes elapsed overlap and a dynamic delay from the timing metadata,
clamps only that hard prefix during denoising, and returns the complete chunk.

When ``--rtc-port`` is enabled, that second port uses the same envelope plus
``action_prefix`` (one to six already-committed physical 14-D targets).
The prefix is held clean at t=0 during
denoising and the response contains only actions K onward, so the bridge must
append it to its execution queue rather than execute the prefix again.

The served checkpoint is the *new* Semantic MoT FastWAM architecture
(``config_realbot_train``). Inference reuses
``evaluation/robotwin/mot_policy/deploy_policy.py`` (``SemanticMoTRobotWinPolicy``).
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib.util
import json
import logging
import os
import pathlib
import socketserver
import struct
import sys
import threading
import time
import math
from collections import defaultdict, deque
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("wan_va_server")

REAL_ACTION_DIM = 14
# RTC is a fixed-shape serving path.  A varying queue depth would otherwise
# create a distinct denoise graph for each K.
RTC_PREFIX_LENGTH = 6
# Bridge image key -> observation dict slot consumed by
# obs_to_model_input_robotwin().
#
# IMPORTANT: obs_to_model_input_robotwin() hardcodes the RobotWin view order
# (head_camera, left_camera, right_camera), but the realbot checkpoint was
# trained with obs_cam_keys = [front, right, left]
# (config_realbot_train.py). To reproduce the training view slots
# [front, right, left] we feed the RIGHT wrist image into the "left_camera"
# slot and the LEFT wrist image into the "right_camera" slot.
IMAGE_SLOT_KEYS = {
    "cam_high": "head_camera",
    "cam_right_wrist": "left_camera",
    "cam_left_wrist": "right_camera",
}

MTLBOT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_NAME = "config_realbot_train"
DEFAULT_T5_DIR = str(MTLBOT_ROOT / "pretrained_weights" / "text_encoder" / "wan22_ti2v_5b")


# ---------------------------------------------------------------------------
# Transport: length-prefixed JSON over TCP (identical to openpi).
# ---------------------------------------------------------------------------

def _read_exact(sock, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed while reading message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_message(sock) -> dict[str, Any]:
    header = _read_exact(sock, 4)
    (length,) = struct.unpack("!I", header)
    if length <= 0:
        raise ValueError(f"invalid message length: {length}")
    return json.loads(_read_exact(sock, length).decode("utf-8"))


def write_message(sock, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def _decode_jpeg_rgb(encoded: str) -> np.ndarray:
    import cv2

    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode JPEG image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Policy: wraps the Semantic MoT inference core from deploy_policy.py.
# ---------------------------------------------------------------------------

def _load_deploy_policy_module():
    """Load evaluation/robotwin/mot_policy/deploy_policy.py by path.

    evaluation/ is not a python package (no __init__.py), so importlib is used
    instead of a regular import. The module inserts MTLBot root into sys.path
    itself, so its internal ``from wan_va...`` imports resolve either way.
    """
    policy_path = (
        MTLBOT_ROOT / "evaluation" / "robotwin" / "mot_policy" / "deploy_policy.py"
    )
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Semantic MoT inference module not found: {policy_path}"
        )
    spec = importlib.util.spec_from_file_location("mtlbot_deploy_policy", policy_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MTLBotRealbotPolicy:
    """Stateless-per-request bimanual (14-D abs-qpos) RealBot policy."""

    def __init__(
        self,
        checkpoint: pathlib.Path,
        *,
        config_name: str,
        t5_dir: pathlib.Path,
        num_inference_steps: int,
        inference_seed: int | None,
        device: str,
        mixed_precision: str,
        compile_infer_action: bool = True,
        warmup: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        self.real_action_dim = REAL_ACTION_DIM
        self.num_inference_steps = int(num_inference_steps)
        self.inference_seed = inference_seed
        self.profile_infer = os.environ.get("PROFILE_INFER", "").strip().lower() in {"1", "true", "yes", "on"}
        self._infer_lock = threading.Lock()
        self._rtc_cache: dict[str, dict[str, Any]] = defaultdict(dict)
        self._rtc_latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=10))
        self._policy_hz = 30.0
        self._rtc_execution_horizon = 15
        self._rtc_max_delay_steps = RTC_PREFIX_LENGTH

        deploy = _load_deploy_policy_module()
        usr_args = {
            "ckpt_setting": str(checkpoint),
            "config_name": config_name,
            "device": device,
            "mixed_precision": mixed_precision,
            "t5_dir": str(t5_dir),
            # Absent/None values let the checkpoint's own config decide.
            "use_proprio": None,
            "future_semantic_steps": None,
            "joint_future_denoising": None,
            "t5_max_length": None,
            "replan_steps": 0,
            "num_inference_steps": self.num_inference_steps,
            "seed": inference_seed,
            # Static RTC graphs require compiled action denoising.
            "compile_infer_action": bool(compile_infer_action),
            "compile_mode": "reduce-overhead",
            # RTC may receive any committed queue depth from 0 through 6.
            # Materialize one static action-denoise CUDA Graph per depth.
            "static_rtc_prefix_graphs": True,
            "rtc_prefix_graph_max_length": RTC_PREFIX_LENGTH,
            "timing_enabled": False,
        }
        started = time.time()
        self.policy = deploy.get_model(usr_args)
        self.action_horizon = int(self.policy.action_horizon)
        self.model_dtype = str(self.policy.dtype)
        print(
            f"Loaded MTLBot Semantic MoT policy from {checkpoint} "
            f"in {time.time() - started:.2f}s; config={config_name}; "
            f"real_action_dim={self.real_action_dim}; "
            f"action_horizon={self.action_horizon}; "
            f"num_inference_steps={self.num_inference_steps}; "
            f"mixed_precision={mixed_precision}; "
            f"seed={inference_seed}",
            flush=True,
        )
        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Compile the real fixed inference shapes before listening.

        Warm both the normal request and the normal RTC replan length through
        the same private policy entry point as a socket request.  This covers
        the action denoise-step graphs. Reset the rollout generator afterwards
        so a configured inference seed keeps the exact sequence it would have
        had without this startup-only call.
        """
        started = time.perf_counter()
        logger.info("Starting inference warmup before opening policy ports...")
        images = {
            slot: np.zeros((height, width, 3), dtype=np.uint8)
            for slot, (height, width) in zip(
                ("head_camera", "left_camera", "right_camera"),
                self.policy.per_view_sizes,
            )
        }
        observation = {
            "observation": {
                "state": np.zeros(self.real_action_dim, dtype=np.float32),
                **{slot: {"rgb": image} for slot, image in images.items()},
            }
        }
        try:
            with self._infer_lock:
                # Compile/capture every fixed RTC queue depth before serving.
                # K=0 is the ordinary non-RTC action graph.
                self.policy._infer_action_chunk(None, observation, "warmup")
                for prefix_length in range(1, RTC_PREFIX_LENGTH + 1):
                    self.policy._infer_action_chunk(
                        None,
                        observation,
                        "warmup",
                        action_prefix=np.zeros(
                            (prefix_length, self.real_action_dim), dtype=np.float32
                        ),
                    )
        finally:
            # Never carry the warmup prompt or its RNG progression into a
            # customer request.  The compiled artifacts remain cached.
            self.policy.current_instruction = None
            self.policy.current_context = None
            self.policy.current_context_mask = None
            self.policy.generator = self.policy._build_rollout_generator()
        logger.info("Inference warmup finished in %.2f s.", time.perf_counter() - started)

    def infer(self, request: dict[str, Any], *, rtc: bool = False) -> dict[str, Any]:
        if request.get("type") == "capabilities":
            return {
                "type": "capabilities",
                "request_id": request.get("request_id"),
                "protocol": "mtlbot-unirobot-adapter-v1",
                "training_time_rtc_supported": True,
                "rtc_mode": "training_time",
                "action_horizon": self.action_horizon,
                "action_dim": self.real_action_dim,
                "action_space": "ros_abs",
                "policy_hz": self._policy_hz,
                "max_rtc_delay_steps": self._rtc_max_delay_steps,
            }
        timing_meta = request.get("unirobot_request_timing")
        native_training_rtc = isinstance(timing_meta, dict)
        timings: dict[str, float] = {}
        started = time.time()

        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (self.real_action_dim,):
            raise ValueError(
                f"expected {self.real_action_dim}-dim state, got {state.shape}"
            )

        images: dict[str, np.ndarray] = {}
        for request_key, slot_key in IMAGE_SLOT_KEYS.items():
            images[slot_key] = _decode_jpeg_rgb(request["images"][request_key])

        # RobotWin observation layout consumed by obs_to_model_input_robotwin()
        # and extract_proprio_robotwin() (joint mode reads observation.state).
        # NOTE: assumes the bridge's 14-D ROS state order
        # [left 6 joints, left gripper, right 6 joints, right gripper]
        # (ros_pi05_piper_bridge_hlx._joint_msgs_to_ros_state) matches the
        # RealBot dataset observation.state order. If the robot drifts to the
        # wrong pose, permute `state` here to the dataset order.
        observation = {
            "observation": {
                "state": state,
                **{
                    model_key: {"rgb": image}
                    for model_key, image in images.items()
                },
            }
        }
        timings["preprocess_ms"] = (time.time() - started) * 1000.0

        action_prefix = None
        if native_training_rtc:
            session = str(request.get("session_id", "default"))
            cache = self._rtc_cache[session]
            latencies = self._rtc_latencies[session]
            observation_index = timing_meta.get("observation_timeline_index")
            transition_epoch = timing_meta.get("transition_epoch")
            predicted = timing_meta.get("predicted_total_delay_steps")
            if predicted is None or float(predicted) < 0:
                predicted = math.ceil(max(latencies, default=0.0) * self._policy_hz)
            predicted = int(np.clip(int(predicted), 0, self._rtc_max_delay_steps))
            reference = None
            overlap = 0
            matched_index = None
            if (cache.get("actions") is not None and observation_index is not None
                    and cache.get("observation_timeline_index") is not None
                    and (transition_epoch is None or transition_epoch == cache.get("transition_epoch"))):
                elapsed = int(observation_index) - int(cache["observation_timeline_index"])
                if 0 <= elapsed < len(cache["actions"]):
                    # Match UniRobot training-time RTC: prefix phase is a
                    # shared observation-timeline index only.  Do not search
                    # the old chunk by physical joint state; that makes the
                    # prefix phase vary between replans and breaks the model's
                    # trained temporal contract when move_j has tracking lag.
                    actions_cache = np.asarray(cache["actions"], dtype=np.float32)
                    matched_index = elapsed
                    overlap = min(len(actions_cache) - matched_index,
                                  max(self._rtc_execution_horizon, predicted))
                    reference = actions_cache[matched_index:matched_index + overlap]
            hard_delay = min(predicted, self._rtc_max_delay_steps, overlap)
            if reference is not None and hard_delay > 0:
                action_prefix = np.asarray(reference[:hard_delay], dtype=np.float32)
        elif rtc:
            if "action_prefix" not in request:
                raise ValueError("RTC requests must include action_prefix: [[14 floats], ...]")
            action_prefix = np.asarray(request["action_prefix"], dtype=np.float32)
            if action_prefix.ndim != 2 or action_prefix.shape[1] != self.real_action_dim:
                raise ValueError(
                    f"RTC action_prefix must have shape [K,{self.real_action_dim}], got {action_prefix.shape}"
                )
            if not 0 < action_prefix.shape[0] <= RTC_PREFIX_LENGTH:
                raise ValueError(
                    f"RTC action_prefix must contain 1..{RTC_PREFIX_LENGTH} actions, "
                    f"got {action_prefix.shape[0]}"
                )

        infer_started = time.time()
        with self._infer_lock:
            actions = self.policy._infer_action_chunk(
                None, observation, str(request["prompt"]), action_prefix=action_prefix
            )
        timings["infer_ms"] = (time.time() - infer_started) * 1000.0
        if self.profile_infer:
            timings.update(getattr(self.policy, "last_inference_profile_ms", {}))

        post_started = time.time()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.real_action_dim:
            raise ValueError(
                f"model returned action shape {actions.shape}, "
                f"expected (N, {self.real_action_dim})"
            )
        timings["postprocess_ms"] = (time.time() - post_started) * 1000.0
        timings["total_ms"] = (time.time() - started) * 1000.0

        committed_action_count = 0 if action_prefix is None else int(action_prefix.shape[0])
        returned_actions = actions[committed_action_count:]
        if native_training_rtc:
            session = str(request.get("session_id", "default"))
            self._rtc_cache[session] = {
                "actions": actions.copy(),
                "observation_timeline_index": timing_meta.get("observation_timeline_index"),
                "transition_epoch": timing_meta.get("transition_epoch"),
                "ready_at": time.monotonic(),
            }
            self._rtc_latencies[session].append(time.time() - infer_started)
            returned_actions = actions
        return {
            "actions": returned_actions.tolist(),
            "decoded_action_shape": list(returned_actions.shape),
            "model_action_dim": self.real_action_dim,
            "real_action_dim": self.real_action_dim,
            "action_space": "ros_abs",
            "timing": timings,
            "request_id": request.get("request_id"),
            "mode": "training_time" if native_training_rtc else ("rtc" if rtc else "non_rtc"),
            "rtc_mode": "training_time" if native_training_rtc else ("rtc" if rtc else "off"),
            "committed_action_count": committed_action_count,
            "returned_action_shape": list(returned_actions.shape),
            "rtc": {
                "guided": bool(native_training_rtc and committed_action_count),
                "applied_delay_steps": committed_action_count if native_training_rtc else 0,
                "matched_action_index": matched_index,
            } if native_training_rtc else None,
        }


class _PolicyHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        policy: MTLBotRealbotPolicy = self.server.policy  # type: ignore[attr-defined]
        connection_opened_unix_ms = time.time() * 1000.0
        while True:
            read_started = time.monotonic()
            try:
                request = read_message(self.request)
            except ConnectionError:
                return
            request_received_unix_ms = time.time() * 1000.0
            request_read_ms = (time.monotonic() - read_started) * 1000.0
            try:
                response = policy.infer(request, rtc=bool(self.server.rtc))  # type: ignore[attr-defined]
            except Exception as exc:
                logger.exception("inference failed")
                response = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "request_id": request.get("request_id"),
                }
            # These timestamps are deliberately part of the reply rather than
            # only server logs, so the robot-side action trace has one
            # correlated record for a slow first request.
            response["transport"] = {
                "server_connection_opened_unix_ms": connection_opened_unix_ms,
                "server_request_received_unix_ms": request_received_unix_ms,
                "server_request_read_ms": request_read_ms,
            }
            write_message(self.request, response)


class _PolicyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_cls, policy: MTLBotRealbotPolicy, *, rtc: bool = False):
        super().__init__(server_address, handler_cls)
        self.policy = policy
        self.rtc = bool(rtc)


def _self_test(policy: MTLBotRealbotPolicy) -> None:
    import cv2

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    ok, jpg = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )
    if not ok:
        raise RuntimeError("failed to encode self-test image")
    encoded = base64.b64encode(jpg.tobytes()).decode("ascii")
    request = {
        "request_id": 0,
        "prompt": "pick up the object",
        "state": [0.0] * policy.real_action_dim,
        "images": {key: encoded for key in IMAGE_SLOT_KEYS},
    }
    for prefix_length in range(RTC_PREFIX_LENGTH + 1):
        rtc = prefix_length > 0
        if rtc:
            request["action_prefix"] = [[0.0] * policy.real_action_dim for _ in range(prefix_length)]
        else:
            request.pop("action_prefix", None)
        request["request_id"] = prefix_length
        response = policy.infer(request, rtc=rtc)
        actions = np.asarray(response["actions"])
        if not np.isfinite(actions).all():
            raise RuntimeError(f"self-test produced non-finite actions for RTC prefix length {prefix_length}")
        print(
            "self_test_ok "
            f"prefix_length={prefix_length} actions_shape={actions.shape} "
            f"space={response['action_space']} timing={response['timing']}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# CLI compatible with the existing real-robot bridge flags.
# ---------------------------------------------------------------------------

def _resolve_checkpoint(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    # Directory form (openpi convention): pick the newest checkpoint_step_*.pt.
    candidates = sorted(path.rglob("checkpoint_step_*.pt"))
    if not candidates:
        candidates = sorted(path.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint .pt found under {path}")
    resolved = candidates[-1]
    print(f"[checkpoint] resolved --checkpoint-dir {path} -> {resolved}", flush=True)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", "--checkpoint", "--ckpt", dest="checkpoint_dir", required=True)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--t5-dir", default=DEFAULT_T5_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--rtc-port", type=int, default=None,
                        help="Optional RTC endpoint. It requires action_prefix and returns only uncommitted actions.")
    parser.add_argument("--real-action-dim", type=int, choices=(7, 14), default=14)
    parser.add_argument("--num-steps", "--num-inference-steps", dest="num_inference_steps", type=int, default=20)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--inference-seed", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument(
        "--compile-infer-action",
        type=int,
        choices=(0, 1),
        default=1,
        help="enable torch.compile for the action inference callable (opt-in)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        choices=(0, 1),
        default=0,
        help="run a dummy fixed-shape inference before opening ports (default: 0)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--norm-asset-id", default=None,
                        help="openpi compatibility; ignored (stats are stored in the checkpoint).")
    parser.add_argument(
        "--delta-joint-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="openpi compatibility. The MTLBot realbot checkpoint predicts absolute "
        "next-qpos directly, so --no-delta-joint-actions (the default) is required.",
    )
    parser.add_argument(
        "--use-quantile-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="openpi compatibility; action/proprio normalization always comes from the checkpoint stats.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.real_action_dim != REAL_ACTION_DIM:
        parser.error(
            f"--real-action-dim {args.real_action_dim} is not supported: "
            f"the realbot checkpoint is a {REAL_ACTION_DIM}-D bimanual model."
        )
    if args.delta_joint_actions:
        parser.error(
            "--delta-joint-actions is not supported: the realbot checkpoint "
            "predicts absolute next-qpos directly. Use --no-delta-joint-actions."
        )
    if args.rtc_port is not None and args.rtc_port == args.port:
        parser.error("--rtc-port must differ from --port")

    checkpoint = _resolve_checkpoint(args.checkpoint_dir)
    t5_dir = pathlib.Path(args.t5_dir).expanduser()
    if not t5_dir.is_absolute():
        t5_dir = (MTLBOT_ROOT / t5_dir)
    if not t5_dir.exists():
        raise FileNotFoundError(f"t5 dir not found: {t5_dir}")

    with contextlib.ExitStack():
        policy = MTLBotRealbotPolicy(
            checkpoint,
            config_name=args.config_name,
            t5_dir=t5_dir,
            num_inference_steps=args.num_inference_steps,
            inference_seed=None if args.inference_seed < 0 else args.inference_seed,
            device=args.device,
            mixed_precision=args.mixed_precision,
            compile_infer_action=bool(args.compile_infer_action),
            warmup=bool(args.warmup),
        )
        if args.action_horizon is not None and args.action_horizon != policy.action_horizon:
            print(
                f"[warning] --action-horizon {args.action_horizon} ignored: "
                f"model horizon is fixed at {policy.action_horizon} "
                f"(future_blocks x action_per_frame).",
                flush=True,
            )
        if args.self_test:
            _self_test(policy)
            return

        server = _PolicyServer((args.host, args.port), _PolicyHandler, policy, rtc=False)
        if args.rtc_port is None:
            print(f"Serving non-RTC MTLBot policy on {args.host}:{args.port}", flush=True)
            with server:
                server.serve_forever()
            return

        rtc_server = _PolicyServer((args.host, args.rtc_port), _PolicyHandler, policy, rtc=True)
        print(
            f"Serving MTLBot policy: non-RTC={args.host}:{args.port}, RTC={args.host}:{args.rtc_port}",
            flush=True,
        )
        with server, rtc_server:
            rtc_thread = threading.Thread(target=rtc_server.serve_forever, name="mtlbot-rtc-server", daemon=True)
            rtc_thread.start()
            try:
                server.serve_forever()
            finally:
                rtc_server.shutdown()
                rtc_thread.join()


if __name__ == "__main__":
    main()
