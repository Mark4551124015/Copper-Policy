from __future__ import annotations

import copy
import html
import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.config import VA_CONFIGS
from inference.robotwin_stats import (
    normalize_robotwin_qpos_action_mode,
    resolve_robotwin_action_space_stats,
    resolve_robotwin_feature_stats,
    select_action_space_stats,
)
from inference.precision import _keep_selected_tensors_fp32
from inference.model import build_semantic_mot_model
from wan_va.modules.vjepa_hub import load_vjepa_model

logger = logging.getLogger(__name__)

DEFAULT_TEXT_EMB_MAX_TOKENS = 64
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _prompt_clean(text: str) -> str:
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def build_instruction_text(task_description: str, instruction_template: str | None) -> str:
    task = _prompt_clean(task_description or "")
    if not instruction_template:
        return task
    return instruction_template.format(task=task)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if _is_none_like(value):
        return None
    return _parse_bool(value)


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _maybe_compile_callable(
    fn,
    *,
    name: str,
    enabled: bool,
    mode: str,
    disable_cudagraphs: bool = False,
):
    if not enabled or os.environ.get("COPPER_DISABLE_COMPILE") == "1":
        return fn

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        logger.warning("torch.compile is unavailable; keeping %s in eager mode.", name)
        return fn

    try:
        if disable_cudagraphs:
            # A compiled V-JEPA returns graph-owned inference tensors. Keep
            # Inductor fusion for action denoising but do not nest a second
            # CUDA Graph capture after that vision graph.
            compiled_fn = compile_fn(
                fn,
                fullgraph=False,
                dynamic=False,
                options={"triton.cudagraphs": False},
            )
        else:
            compiled_fn = compile_fn(fn, mode=mode, fullgraph=False, dynamic=False)
    except Exception as exc:
        logger.warning("torch.compile setup failed for %s (%s); keeping eager mode.", name, exc)
        return fn

    state = {"enabled": True}

    def _wrapped(*args, **kwargs):
        if not state["enabled"]:
            return fn(*args, **kwargs)
        try:
            return compiled_fn(*args, **kwargs)
        except Exception:
            state["enabled"] = False
            logger.exception("torch.compile runtime failed for %s; falling back to eager mode.", name)
            return fn(*args, **kwargs)

    logger.info(
        "Enabled torch.compile for %s with mode=%s%s.",
        name,
        "default" if disable_cudagraphs else mode,
        " (CUDA Graph disabled)" if disable_cudagraphs else "",
    )
    return _wrapped


def _initialize_cuda_blas_for_graph_capture(device: str) -> None:
    """Create the cuBLAS handle before any Inductor CUDA-Graph capture.

    cuBLAS is initialized lazily.  Creating its handle while a graph is being
    captured is illegal and invalidates that capture (and any later capture on
    the request stream).  A tiny eager matmul performs the same one-time
    initialization outside capture; it has no model-output effect.
    """
    if not device.startswith("cuda"):
        return
    with torch.no_grad():
        warm = torch.ones((1, 1), device=device, dtype=torch.float32)
        torch.mm(warm, warm)
    torch.cuda.synchronize(device)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _config_from_checkpoint_or_registry(payload: dict, config_name: str):
    cfg = copy.deepcopy(VA_CONFIGS[config_name])
    model_config = payload.get("model_config") if isinstance(payload, dict) else None
    if isinstance(model_config, dict):
        for key, value in model_config.items():
            cfg[key] = copy.deepcopy(value)
    config_overrides = payload.get("config_overrides") if isinstance(payload, dict) else None
    if isinstance(config_overrides, dict):
        for key, value in config_overrides.items():
            cfg[key] = copy.deepcopy(value)
    dataset_env = {
        "config_robotwin_train": "COPPER_ROBOTWIN_DATASET",
        "config_realbot_train": "COPPER_REALROBOT_DATASET",
    }.get(config_name)
    if dataset_env and os.environ.get(dataset_env):
        dataset_root = Path(os.environ[dataset_env]).expanduser()
        cfg.dataset_path = str(dataset_root)
        cfg.empty_emb_path = str(dataset_root / "empty_emb.pt")
        cfg.dino_spatial_norm_path = str(dataset_root / "vjepa_spatial_norm.json")
    cfg.pretrain_video_backbone = ""
    return cfg


def _resolve_project_path(path_like: Any) -> Any:
    if _is_none_like(path_like):
        return path_like
    path = Path(os.path.expanduser(os.path.expandvars(str(path_like))))
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def _resolve_eval_cfg_paths(cfg) -> None:
    for key in (
        "dataset_path",
        "empty_emb_path",
        "dino_spatial_norm_path",
    ):
        if hasattr(cfg, key):
            setattr(cfg, key, _resolve_project_path(getattr(cfg, key)))


def _validate_robotwin_eval_cfg(cfg) -> None:
    required_keys = (
        "num_views",
        "per_view_image_sizes",
        "image_patch_size",
        "teacher_type",
        "use_proprio",
        "proprio_dim",
        "proprio_norm_mode",
        "proprio_norm_stat",
        "action_norm_mode",
        "norm_stat",
        "robotwin_qpos_action_mode",
        "robotwin_current_view_resize_mode",
        "future_blocks",
        "action_per_frame",
    )
    missing = [key for key in required_keys if not hasattr(cfg, key)]
    if missing:
        raise ValueError(
            "RobotWin eval requires checkpoint-resolved config keys to be present. "
            f"Missing: {missing}"
        )

    num_views = int(getattr(cfg, "num_views"))
    if num_views != 3:
        raise ValueError(f"RobotWin eval expects num_views=3, got {num_views}")

    raw_per_view_sizes = getattr(cfg, "per_view_image_sizes")
    if raw_per_view_sizes is None:
        raise ValueError(
            "RobotWin eval requires explicit per_view_image_sizes from training config; "
            "implicit replication from per_view_image_size is disabled."
        )
    if len(raw_per_view_sizes) != num_views:
        raise ValueError(
            "per_view_image_sizes must align with num_views for RobotWin eval: "
            f"got {len(raw_per_view_sizes)} sizes for num_views={num_views}"
        )

    teacher_type = str(getattr(cfg, "teacher_type") or "").lower()
    if teacher_type not in {"vjepa", "dino"}:
        raise ValueError(
            f"RobotWin eval requires explicit teacher_type in {{'vjepa', 'dino'}}, got {teacher_type!r}"
        )

    use_proprio = bool(getattr(cfg, "use_proprio"))
    proprio_dim = int(getattr(cfg, "proprio_dim") or 0)
    action_space = infer_robotwin_action_space(cfg)
    qpos_action_mode = infer_robotwin_qpos_action_mode(cfg)
    expected_proprio_dim = 14 if action_space == "joint" else 16
    if use_proprio and proprio_dim != expected_proprio_dim:
        raise ValueError(
            f"RobotWin eval expects {expected_proprio_dim}D {action_space} proprio when proprio is enabled, "
            f"got proprio_dim={proprio_dim}"
        )
    if action_space == "joint" and qpos_action_mode == "rel_qpos" and (not use_proprio or proprio_dim != 14):
        raise ValueError("RobotWin rel_qpos eval requires use_proprio=True and proprio_dim=14.")


def infer_robotwin_action_space(cfg) -> str:
    explicit = str(getattr(cfg, "robotwin_action_space", "") or "").strip().lower()
    if explicit:
        if explicit not in {"eef", "joint"}:
            raise ValueError(
                f"Unsupported robotwin_action_space={explicit!r}. Expected one of: eef, joint."
            )
        return explicit

    action_dim = int(getattr(cfg, "action_dim", 0) or 0)
    proprio_dim = int(getattr(cfg, "proprio_dim", 0) or 0)
    if action_dim == 14 or proprio_dim == 14:
        return "joint"
    return "eef"


def infer_robotwin_action_type(cfg) -> str:
    explicit = str(getattr(cfg, "robotwin_action_type", "") or "").strip().lower()
    if explicit:
        return explicit
    return "qpos" if infer_robotwin_action_space(cfg) == "joint" else "ee"


def infer_robotwin_qpos_action_mode(cfg) -> str:
    return normalize_robotwin_qpos_action_mode(getattr(cfg, "robotwin_qpos_action_mode", "abs_qpos"))


def load_model(
    ckpt_path: str,
    config_name: str,
    device: str,
    dtype: torch.dtype,
    use_proprio: bool | None = None,
):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg = _config_from_checkpoint_or_registry(payload, config_name)
    if use_proprio is not None:
        cfg.use_proprio = bool(use_proprio)
    elif not hasattr(cfg, "use_proprio"):
        raise ValueError(
            "Checkpoint/config is missing explicit `use_proprio`. "
            "RobotWin eval disables implicit inference to avoid train/eval mismatch."
        )
    _resolve_eval_cfg_paths(cfg)
    if not bool(getattr(cfg, "use_proprio", False)):
        cfg.proprio_dim = 0
    _validate_robotwin_eval_cfg(cfg)

    model, _model_cfg = build_semantic_mot_model(cfg, device=device, dtype=dtype)
    state_dict = payload.get("state_dict", payload)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing_trained_dino = [
        key for key in incompatible.missing_keys if "_dino_encoder.model." in key
    ]
    if missing_trained_dino:
        preview = ", ".join(missing_trained_dino[:3])
        suffix = " ..." if len(missing_trained_dino) > 3 else ""
        raise RuntimeError(
            "Checkpoint is missing trained DINO weights required for deployment "
            f"({len(missing_trained_dino)} tensors; e.g. {preview}{suffix}). "
            "Refusing to use randomly initialized DINO parameters."
        )
    model = model.to(device=device, dtype=dtype)

    use_custom_fp32_precision = bool(
        getattr(cfg, "enable_custom_fp32_precision", False)
        or getattr(cfg, "enable_fp32_modules", False)
    )
    if use_custom_fp32_precision:
        _keep_selected_tensors_fp32(model, getattr(model, "_keep_in_fp32_modules", None))
    if hasattr(model, "apply_precision_policy"):
        model.apply_precision_policy()
    return model.eval(), cfg


def load_text_encoder(t5_dir: str, device: str, dtype: torch.dtype):
    from transformers import AutoTokenizer, UMT5EncoderModel

    encoder_device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(t5_dir, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        t5_dir,
        torch_dtype=dtype,
        subfolder="text_encoder",
    ).to(encoder_device).eval()
    return tokenizer, encoder, encoder_device


@torch.no_grad()
def encode_text(
    text: str,
    tokenizer,
    encoder,
    encoder_device: str,
    output_device: str,
    max_length: int | None = DEFAULT_TEXT_EMB_MAX_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    text = _prompt_clean(text)
    if max_length is None:
        inputs = tokenizer(
            [text],
            padding=True,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
    else:
        length_inputs = tokenizer(
            [text],
            padding=False,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=False,
            return_tensors="pt",
        )
        token_len = int(length_inputs.input_ids.shape[1])
        if token_len > max_length:
            raise ValueError(
                f"Instruction token length {token_len} exceeds text_emb_max_tokens={max_length}. "
                "This eval path matches offline text embedding preprocessing and will not truncate implicitly."
            )
        inputs = tokenizer(
            [text],
            padding="max_length",
            max_length=max_length,
            truncation=False,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
    ids = inputs.input_ids.to(encoder_device)
    mask = inputs.attention_mask.to(device=encoder_device, dtype=torch.bool)
    seq_len = int(mask[0].gt(0).sum().item())
    embeds = encoder(ids, attention_mask=mask).last_hidden_state[0]
    embeds = embeds[:seq_len]
    if max_length is not None and seq_len < max_length:
        pad = torch.zeros(max_length - seq_len, embeds.shape[1], dtype=embeds.dtype, device=encoder_device)
        embeds = torch.cat([embeds, pad], dim=0)
    embeds = embeds.unsqueeze(0).to(output_device)
    mask = mask[:, : embeds.shape[1]].to(device=output_device, dtype=torch.bool)
    return embeds, mask


def _resolve_text_emb_max_tokens(cfg, override: Optional[int]) -> int | None:
    if override is not None:
        return None if int(override) < 0 else int(override)
    raw_value = getattr(cfg, "text_emb_max_tokens", DEFAULT_TEXT_EMB_MAX_TOKENS)
    max_tokens = DEFAULT_TEXT_EMB_MAX_TOKENS if raw_value is None else int(raw_value)
    if max_tokens == 0 or max_tokens < -1:
        raise ValueError(f"text_emb_max_tokens must be -1 or a positive integer, got {max_tokens}")
    return None if max_tokens < 0 else max_tokens


def _resolve_text_emb_use_padding_mask(cfg) -> bool:
    raw_value = getattr(cfg, "text_emb_use_padding_mask", True)
    return True if raw_value is None else bool(raw_value)


def load_dino_encoder(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    return model


@torch.no_grad()
def extract_anchor_dino(
    views: list[torch.Tensor],
    dino_model,
    per_view_num_spatial_tokens: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.tensor(_IMAGENET_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=dtype, device=device).view(1, 3, 1, 1)

    cls_list, reg_list, spatial_list = [], [], []
    if len(views) != len(per_view_num_spatial_tokens):
        raise ValueError(
            "per_view_num_spatial_tokens must align with runtime views: "
            f"got {len(per_view_num_spatial_tokens)} sizes for {len(views)} views"
        )
    for view, num_spatial_tokens in zip(views, per_view_num_spatial_tokens):
        x = view.to(device=device, dtype=dtype)
        x = x * 0.5 + 0.5
        x = (x - mean) / std
        seq = dino_model(pixel_values=x).last_hidden_state
        n_extra = seq.shape[1] - int(num_spatial_tokens)
        if n_extra < 1:
            raise ValueError(
                "DINO output token count is too small for the configured view size: "
                f"seq_len={seq.shape[1]}, expected spatial={int(num_spatial_tokens)} plus cls/register tokens"
            )
        n_reg = n_extra - 1
        cls_list.append(seq[:, :1].float())
        reg_list.append(seq[:, 1 : 1 + n_reg].float())
        spatial_list.append(seq[:, 1 + n_reg :].float())

    return (
        torch.cat(cls_list, dim=1),
        torch.cat(reg_list, dim=1),
        torch.cat(spatial_list, dim=1),
    )


def load_vjepa_encoder(model_name: str, device: str, dtype: torch.dtype):
    model, _predictor = load_vjepa_model(model_name)
    return model.to(device=device, dtype=dtype).eval()


def _normalize_vjepa_output(output) -> torch.Tensor:
    if torch.is_tensor(output):
        seq = output
    elif hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        seq = output.last_hidden_state
    elif isinstance(output, (tuple, list)) and output:
        seq = output[0]
    else:
        raise TypeError(f"Unsupported VJEPA output type: {type(output)}")
    if seq.ndim == 2:
        seq = seq.unsqueeze(1)
    elif seq.ndim > 3:
        seq = seq.reshape(seq.shape[0], -1, seq.shape[-1])
    if seq.ndim != 3:
        raise ValueError(f"Expected normalized VJEPA features [B,N,D], got {tuple(seq.shape)}")
    return seq.float()


@torch.no_grad()
def extract_anchor_vjepa(
    views: list[torch.Tensor],
    vjepa_model,
    per_view_num_spatial_tokens: tuple[int, ...],
    device: str,
    dtype: torch.dtype,
    prev_views: list[torch.Tensor] | None = None,
    compiled_vjepa: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del prev_views
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    spatial_by_view: list[torch.Tensor | None] = [None] * len(views)
    if len(views) != len(per_view_num_spatial_tokens):
        raise ValueError(
            "per_view_num_spatial_tokens must align with runtime views: "
            f"got {len(per_view_num_spatial_tokens)} sizes for {len(views)} views"
        )

    # The realbot geometry is [256x320, 128x160, 128x160].  Run views with
    # equal spatial shapes in one V-JEPA batch: this preserves every view's
    # token geometry while reducing three encoder launches to two.
    shape_groups: dict[tuple[int, int], list[int]] = {}
    for view_idx, view in enumerate(views):
        shape_groups.setdefault((int(view.shape[-2]), int(view.shape[-1])), []).append(view_idx)

    for view_indices in shape_groups.values():
        x_parts = []
        for view_idx in view_indices:
            x_cur = views[view_idx].to(device=device, dtype=torch.float32)
            x_cur = x_cur * 0.5 + 0.5
            x_parts.append((x_cur - mean) / std)
        x_cur = torch.cat(x_parts, dim=0)
        video_input = x_cur.unsqueeze(2)

        # `torch.compile(..., mode="reduce-overhead")` captures V-JEPA with
        # CUDA Graph. Its output must remain a normal no-grad tensor because
        # the downstream compactor/action graphs own mutable replay buffers.
        # Eager V-JEPA can retain inference_mode's small overhead advantage.
        execution_context = torch.no_grad if compiled_vjepa else torch.inference_mode
        with execution_context():
            if x_cur.device.type == "cuda" and dtype != torch.float32:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = vjepa_model(video_input)
            elif x_cur.device.type == "cpu" and dtype == torch.bfloat16:
                with torch.autocast(device_type="cpu", dtype=dtype):
                    out = vjepa_model(video_input)
            else:
                out = vjepa_model(video_input)
        # Materialize output outside the vision graph before passing it to a
        # separately captured compactor/action graph.  Values and token
        # semantics are unchanged; this only gives the next graph independent
        # replay-safe storage.
        seq = _normalize_vjepa_output(out).detach().clone()
        batch_size = int(views[view_indices[0]].shape[0])
        expected_batch = batch_size * len(view_indices)
        if seq.shape[0] != expected_batch:
            raise ValueError(
                f"VJEPA output batch mismatch: expected {expected_batch}, got {seq.shape[0]}"
            )
        for group_idx, view_idx in enumerate(view_indices):
            seq_view = seq[group_idx * batch_size : (group_idx + 1) * batch_size]
            num_spatial_tokens = per_view_num_spatial_tokens[view_idx]
            if seq_view.shape[1] != int(num_spatial_tokens):
                raise ValueError(
                    f"VJEPA output spatial tokens mismatch: expected {int(num_spatial_tokens)}, "
                    f"got {seq_view.shape[1]}"
                )
            spatial_by_view[view_idx] = seq_view

    if any(spatial is None for spatial in spatial_by_view):
        raise RuntimeError("VJEPA did not produce features for every input view")
    spatial_list = [spatial for spatial in spatial_by_view if spatial is not None]
    spatial = torch.cat(spatial_list, dim=1)
    empty = torch.empty(1, 0, spatial.shape[-1], dtype=torch.float32, device=spatial.device)
    return empty, empty, spatial


def _build_action_affine_params(mode: str, center: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mode == "z-score":
        affine_scale = 1.0 / (scale + 1e-8)
        affine_offset = -center / (scale + 1e-8)
        return affine_scale, affine_offset

    input_min = center.copy()
    input_max = scale.copy()
    input_range = input_max - input_min
    ignore_dim = input_range < 1e-4
    safe_range = input_range.copy()
    safe_range[ignore_dim] = 2.0

    affine_scale = 2.0 / safe_range
    affine_offset = -1.0 - affine_scale * input_min
    affine_offset[ignore_dim] = -input_min[ignore_dim]
    return affine_scale, affine_offset


def _checkpoint_normalization(cfg, name: str) -> dict[str, Any] | None:
    """Return normalization captured at training time, if this is a new checkpoint."""
    payload = getattr(cfg, "checkpoint_normalization", None)
    if not isinstance(payload, dict):
        return None
    value = payload.get(name)
    return dict(value) if isinstance(value, dict) else None


def build_action_denormalizer(cfg, clip_norm: float | None = 5.0):
    embedded = _checkpoint_normalization(cfg, "action")
    if embedded is not None:
        mode = str(embedded["mode"]).lower()
        center = np.asarray(embedded["center"], dtype=np.float32)
        scale = np.asarray(embedded["scale"], dtype=np.float32)
        zscore_exclude_dims = tuple(int(x) for x in embedded.get("zscore_exclude_dims", ()))
        if center.ndim != 1 or scale.ndim != 1 or center.shape != scale.shape:
            raise ValueError("Invalid checkpoint action normalization statistics.")
        affine_scale, affine_offset = _build_action_affine_params(mode, center, scale)

        def denormalize(action_norm: np.ndarray) -> np.ndarray:
            if clip_norm is not None:
                action_norm = np.clip(action_norm, -float(clip_norm), float(clip_norm))
            physical = (action_norm - affine_offset) / affine_scale
            if mode == "z-score" and zscore_exclude_dims:
                physical[..., list(zscore_exclude_dims)] = action_norm[..., list(zscore_exclude_dims)]
            return physical

        return denormalize

    norm_stat = dict(cfg.norm_stat)
    zscore_exclude_dims = tuple(int(x) for x in (getattr(cfg, "action_zscore_exclude_dims", ()) or ()))
    mode = str(getattr(cfg, "action_norm_mode", "auto") or "auto").lower()
    action_space = infer_robotwin_action_space(cfg)
    qpos_action_mode = infer_robotwin_qpos_action_mode(cfg)
    action_horizon = int(getattr(cfg, "future_blocks")) * int(getattr(cfg, "action_per_frame"))
    if action_space == "joint":
        try:
            selected = select_action_space_stats(
                norm_stat,
                qpos_action_mode=qpos_action_mode,
                action_horizon=action_horizon,
                allow_top_level=True,
            )
            norm_stat = dict(selected)
        except KeyError:
            norm_stat = resolve_robotwin_action_space_stats(
                dataset_path=str(getattr(cfg, "dataset_path")),
                qpos_action_mode=qpos_action_mode,
                action_horizon=action_horizon,
                dataset_variant=getattr(cfg, "dataset_variant", "mix"),
                dataset_repo_allowlist=getattr(cfg, "dataset_repo_allowlist", None),
            )
    try:
        if mode in {"auto", ""}:
            if "mean" in norm_stat and "std" in norm_stat:
                mode = "z-score"
            elif "min" in norm_stat and "max" in norm_stat:
                mode = "min/max"
            elif "q01" in norm_stat and "q99" in norm_stat:
                mode = "q01/q99"
            else:
                raise KeyError(
                    "norm_stat must contain either mean/std, min/max, or q01/q99. "
                    f"Available keys: {sorted(norm_stat.keys())}"
                )

        if mode == "z-score":
            center = np.array(norm_stat["mean"], dtype=np.float32)
            scale = np.array(norm_stat["std"], dtype=np.float32)
        elif mode == "min/max":
            center = np.array(norm_stat["min"], dtype=np.float32)
            scale = np.array(norm_stat["max"], dtype=np.float32)
        elif mode == "q01/q99":
            center = np.array(norm_stat["q01"], dtype=np.float32)
            scale = np.array(norm_stat["q99"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action_norm_mode={mode!r}")
    except KeyError:
        if action_space == "joint":
            merged = resolve_robotwin_action_space_stats(
                dataset_path=str(getattr(cfg, "dataset_path")),
                qpos_action_mode=qpos_action_mode,
                action_horizon=action_horizon,
                dataset_variant=getattr(cfg, "dataset_variant", "mix"),
                dataset_repo_allowlist=getattr(cfg, "dataset_repo_allowlist", None),
            )
        else:
            merged = resolve_robotwin_feature_stats(
                dataset_path=str(getattr(cfg, "dataset_path")),
                feature_key="action",
                dataset_variant=getattr(cfg, "dataset_variant", "mix"),
                dataset_repo_allowlist=getattr(cfg, "dataset_repo_allowlist", None),
            )
        merged.update(norm_stat)
        norm_stat = merged
        if mode in {"auto", ""}:
            if "mean" in norm_stat and "std" in norm_stat:
                mode = "z-score"
            elif "min" in norm_stat and "max" in norm_stat:
                mode = "min/max"
            elif "q01" in norm_stat and "q99" in norm_stat:
                mode = "q01/q99"
            else:
                raise KeyError(
                    "Resolved RobotWin action stats must contain either mean/std, min/max, or q01/q99. "
                    f"Available keys: {sorted(norm_stat.keys())}"
                )
        if mode == "z-score":
            center = np.array(norm_stat["mean"], dtype=np.float32)
            scale = np.array(norm_stat["std"], dtype=np.float32)
        elif mode == "min/max":
            center = np.array(norm_stat["min"], dtype=np.float32)
            scale = np.array(norm_stat["max"], dtype=np.float32)
        elif mode == "q01/q99":
            center = np.array(norm_stat["q01"], dtype=np.float32)
            scale = np.array(norm_stat["q99"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported action_norm_mode={mode!r}")
    if center.ndim == 0 or scale.ndim == 0:
        raise ValueError("Invalid scalar RobotWin action normalization stats.")

    affine_scale, affine_offset = _build_action_affine_params(mode, center, scale)

    def denormalize(action_norm: np.ndarray) -> np.ndarray:
        if clip_norm is not None:
            action_norm = np.clip(action_norm, -float(clip_norm), float(clip_norm))
        physical = (action_norm - affine_offset) / affine_scale
        if mode == "z-score" and zscore_exclude_dims:
            physical[..., list(zscore_exclude_dims)] = action_norm[..., list(zscore_exclude_dims)]
        return physical

    return denormalize


def _lookup_feature_stats(stats: Dict[str, Any], feature_key: str) -> Dict[str, Any]:
    if feature_key in stats and isinstance(stats[feature_key], dict):
        return stats[feature_key]
    fallback_paths = (("state", "default"), ("observation", "state"))
    for path in fallback_paths:
        cur: Any = stats
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if isinstance(cur, dict):
            return cur
    raise KeyError(f"Could not find proprio stats for {feature_key!r}")


def build_proprio_normalizer(cfg) -> dict[str, Any] | None:
    if not bool(getattr(cfg, "use_proprio", False)) or int(getattr(cfg, "proprio_dim", 0) or 0) <= 0:
        return None
    mode = str(getattr(cfg, "proprio_norm_mode", "none") or "none").lower()
    if mode in {"none", "false", "off", "disabled"}:
        return None
    if mode not in {"z-score", "min/max", "q01/q99"}:
        raise ValueError(
            f"Unsupported proprio_norm_mode={mode!r}. "
            "Only 'z-score', 'min/max', 'q01/q99', and 'none' are allowed."
        )

    embedded = _checkpoint_normalization(cfg, "proprio")
    if embedded is not None:
        if str(embedded.get("mode", "")).lower() != mode:
            raise ValueError(
                "Checkpoint proprio normalization mode does not match the resolved model config: "
                f"checkpoint={embedded.get('mode')!r}, config={mode!r}."
            )
        center = torch.as_tensor(embedded["center"], dtype=torch.float32)
        scale = torch.as_tensor(embedded["scale"], dtype=torch.float32)
        return {
            "mode": mode,
            "center": center.flatten(),
            "scale": scale.flatten(),
            "clip": float(embedded.get("clip", getattr(cfg, "proprio_norm_clip", 5.0))),
            "zscore_exclude_dims": tuple(int(x) for x in embedded.get("zscore_exclude_dims", ())),
        }

    explicit = getattr(cfg, "proprio_norm_stat", None)
    if explicit is None:
        field_stats = resolve_robotwin_feature_stats(
            dataset_path=str(getattr(cfg, "dataset_path")),
            feature_key="observation.state",
            dataset_variant=getattr(cfg, "dataset_variant", "mix"),
            dataset_repo_allowlist=getattr(cfg, "dataset_repo_allowlist", None),
        )
    else:
        field_stats = _lookup_feature_stats(explicit, "observation.state")
    if mode == "z-score":
        center = torch.as_tensor(field_stats["mean"], dtype=torch.float32)
        scale = torch.as_tensor(field_stats["std"], dtype=torch.float32)
    elif mode == "q01/q99":
        center = torch.as_tensor(field_stats["q01"], dtype=torch.float32)
        scale = torch.as_tensor(field_stats["q99"], dtype=torch.float32)
    else:
        center = torch.as_tensor(field_stats["min"], dtype=torch.float32)
        scale = torch.as_tensor(field_stats["max"], dtype=torch.float32)
    return {
        "mode": mode,
        "center": center.flatten(),
        "scale": scale.flatten(),
        "clip": float(getattr(cfg, "proprio_norm_clip", 5.0)),
        "zscore_exclude_dims": tuple(int(x) for x in (getattr(cfg, "proprio_zscore_exclude_dims", ()) or ())),
    }


def normalize_proprio(proprio: torch.Tensor, normalizer: dict[str, Any] | None) -> torch.Tensor:
    if normalizer is None:
        return proprio
    mode = str(normalizer["mode"])
    center = normalizer["center"].to(device=proprio.device, dtype=proprio.dtype)
    scale = normalizer["scale"].to(device=proprio.device, dtype=proprio.dtype)
    clip = float(normalizer["clip"])
    if center.numel() != proprio.shape[-1] or scale.numel() != proprio.shape[-1]:
        raise ValueError(
            "proprio stats dimension mismatch: "
            f"proprio_dim={proprio.shape[-1]}, center={center.numel()}, scale={scale.numel()}"
        )

    if mode == "z-score":
        out = (proprio - center) / (scale + 1e-8)
        zscore_exclude_dims = tuple(int(x) for x in normalizer.get("zscore_exclude_dims", ()))
        if zscore_exclude_dims:
            exclude_idx = torch.as_tensor(zscore_exclude_dims, device=proprio.device, dtype=torch.long)
            out = out.clone()
            out.index_copy_(1, exclude_idx, proprio.index_select(1, exclude_idx))
    else:  # min/max or q01/q99: map the selected robust interval to [-1, 1].
        input_range = scale - center
        valid_range = input_range >= 1e-4
        safe_range = torch.where(valid_range, input_range, torch.ones_like(input_range) * 2.0)
        out = (proprio - center) * (2.0 / safe_range) - 1.0
        out = torch.where(valid_range, out, proprio - center)
    return torch.clamp(out, -clip, clip)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _resize_rgb_center_crop(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Match SemanticChunkLeRobotDataset._resize_frame for per-view RobotWin inputs."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for RobotWin per-view resizing") from exc

    h0, w0 = image.shape[:2]
    scale = max(out_h / h0, out_w / w0)
    rh, rw = int(round(h0 * scale)), int(round(w0 * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    top = max((rh - out_h) // 2, 0)
    left = max((rw - out_w) // 2, 0)
    return resized[top : top + out_h, left : left + out_w, :]


def _build_robotwin_view_tensors(
    camera_images: tuple[np.ndarray, ...],
    per_view_sizes: tuple[tuple[int, int], ...],
    *,
    resize_mode: str,
    device: str,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    mode = str(resize_mode).strip().lower()
    if mode not in {"center_crop", "direct"}:
        raise ValueError(f"Unsupported RobotWin per-view resize_mode={resize_mode!r}")

    views: list[torch.Tensor] = []
    for image, (h, w) in zip(camera_images, per_view_sizes):
        if mode == "center_crop":
            resized = _resize_rgb_center_crop(image, int(h), int(w))
        else:
            resized = _resize_rgb(image, (int(w), int(h)))
        x = torch.tensor(resized, dtype=dtype).permute(2, 0, 1).unsqueeze(0).to(device)
        x = x * (2.0 / 255.0) - 1.0
        views.append(x)
    return views


def _compose_robotwin_frame(observation: Dict[str, Any], out_h: int, out_w: int) -> np.ndarray:
    obs_data = observation["observation"]
    top_h = (int(out_h) * 2) // 3
    bottom_h = int(out_h) - top_h
    left_w = int(out_w) // 2
    right_w = int(out_w) - left_w
    head = _resize_rgb(obs_data["head_camera"]["rgb"], (int(out_w), top_h))
    left = _resize_rgb(obs_data["left_camera"]["rgb"], (left_w, bottom_h))
    right = _resize_rgb(obs_data["right_camera"]["rgb"], (right_w, bottom_h))
    bottom = np.concatenate([left, right], axis=1)
    return np.concatenate([head, bottom], axis=0)


def obs_to_model_input_robotwin(
    observation: Dict[str, Any],
    per_view_sizes: tuple[tuple[int, int], ...],
    device: str,
    dtype: torch.dtype,
    *,
    current_resize_mode: str = "center_crop",
) -> tuple[list[torch.Tensor], list[torch.Tensor], np.ndarray]:
    obs_data = observation["observation"]
    camera_images = (
        obs_data["head_camera"]["rgb"],
        obs_data["left_camera"]["rgb"],
        obs_data["right_camera"]["rgb"],
    )
    if len(per_view_sizes) != len(camera_images):
        raise ValueError(
            f"RobotWin expects {len(camera_images)} camera sizes, got {len(per_view_sizes)}"
        )

    # Keep two per-view image geometries at eval time:
    # - `views_direct`: frozen anchor extraction path, aligned with cached-anchor
    #   training geometry.
    # - `views`: extra runtime vision path (for example DINO/grid-sampler
    #   branches), aligned with the train-time per-view center-crop geometry.
    views = _build_robotwin_view_tensors(
        camera_images,
        per_view_sizes,
        resize_mode=current_resize_mode,
        device=device,
        dtype=dtype,
    )
    views_direct = _build_robotwin_view_tensors(
        camera_images,
        per_view_sizes,
        resize_mode="direct",
        device=device,
        dtype=dtype,
    )

    stitched_h, stitched_w = per_view_sizes[0]
    stitched = _compose_robotwin_frame(observation, int(stitched_h), int(stitched_w))
    return views, views_direct, stitched


def _get_nested_observation_value(observation: Dict[str, Any], path: tuple[str, ...]) -> tuple[Any, bool]:
    cur: Any = observation
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None, False
        cur = cur[key]
    return cur, True


def _to_flat_float32_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().float().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _summarize_vector(value: Any, max_dims: int) -> list[float] | None:
    if value is None:
        return None
    arr = _to_flat_float32_array(value)
    if arr.size == 0:
        return []
    limit = max(0, int(max_dims))
    if limit <= 0:
        return []
    return [float(x) for x in arr[:limit]]


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu", dtype=torch.float32)
    return {
        "shape": list(tensor.shape),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
    }


def extract_proprio_robotwin(
    observation: Dict[str, Any],
    proprio_dim: int,
    action_space: str,
    device: str,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if proprio_dim <= 0:
        return None

    if action_space == "joint":
        # RoboTwin online observations expose the same 14D qpos used as
        # LeRobot observation.state through joint_action.vector.
        candidate_paths = (
            (("joint_action", "vector"), "joint_action.vector"),
            (("observation.state",), "observation.state"),
            (("observation", "state"), "observation.state"),
            (("state",), "state"),
        )
        found_dims: list[str] = []
        for path, label in candidate_paths:
            value, exists = _get_nested_observation_value(observation, path)
            if not exists:
                continue
            state = _to_flat_float32_array(value)
            if state.shape[0] == proprio_dim:
                return torch.from_numpy(state).to(device=device, dtype=dtype).unsqueeze(0)
            found_dims.append(f"{label}={state.shape[0]}")
        if found_dims:
            raise ValueError(
                "RobotWin joint-mode proprio does not match training proprio_dim. "
                f"Expected {proprio_dim}, found {', '.join(found_dims)}"
            )
        raise KeyError(
            "RobotWin joint-mode proprio requires one of: "
            "`joint_action.vector`, `observation.state`, "
            "`observation['observation']['state']`, or `state`."
        )

    endpose = observation.get("endpose")
    if isinstance(endpose, dict):
        try:
            state = np.asarray(
                endpose["left_endpose"]
                + [endpose["left_gripper"]]
                + endpose["right_endpose"]
                + [endpose["right_gripper"]],
                dtype=np.float32,
            ).reshape(-1)
        except KeyError as exc:
            raise KeyError(f"Missing RobotWin endpose field: {exc.args[0]}") from exc
        if state.shape[0] == proprio_dim:
            return torch.from_numpy(state).to(device=device, dtype=dtype).unsqueeze(0)
        raise ValueError(
            "RobotWin endpose exists but does not match training proprio_dim. "
            f"Expected {proprio_dim}, got {state.shape[0]}"
        )
    raise KeyError("RobotWin eval requires `observation.endpose` for EEF-mode proprio.")


@torch.no_grad()
def predict_action_chunk(
    infer_action_fn,
    views,
    anchor_dino_cls,
    anchor_dino_spatial,
    anchor_dino_registers,
    context,
    context_mask,
    proprio,
    num_inference_steps: int,
    generator: torch.Generator | None,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    joint_future_denoising: bool | None = None,
    future_semantic_steps: int | None = None,
    action_prefix: np.ndarray | None = None,
) -> np.ndarray:
    autocast_enabled = device != "cpu" and dtype in (torch.float16, torch.bfloat16)
    # A request is one CUDA-Graph step.  This prevents graph-tree outputs from
    # a previous request being treated as live inputs to the next request.
    mark_step_begin = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if autocast_enabled and mark_step_begin is not None:
        mark_step_begin()
    with torch.autocast(device_type="cuda" if device != "cpu" else "cpu", dtype=dtype, enabled=autocast_enabled):
        action = infer_action_fn(
            input_pixels=views,
            anchor_dino_cls=anchor_dino_cls,
            anchor_dino_spatial=anchor_dino_spatial,
            anchor_dino_registers=anchor_dino_registers,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_inference_steps=num_inference_steps,
            generator=generator,
            joint_future_denoising=joint_future_denoising,
            future_semantic_steps=future_semantic_steps,
            action_prefix=(
                None
                if action_prefix is None
                else torch.as_tensor(action_prefix, device=device, dtype=dtype).unsqueeze(0)
            ),
        )
    return action[0].cpu().float().numpy()


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def add_eef_pose(new_pose: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    new_pose = np.asarray(new_pose, dtype=np.float32)
    init_pose = np.asarray(init_pose, dtype=np.float32)
    out_rot = (R.from_quat(init_pose[3:7][None]) * R.from_quat(new_pose[3:7][None])).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]], axis=0).astype(np.float32)


def add_init_pose(action_rel: np.ndarray, init_pose: np.ndarray) -> np.ndarray:
    left_pose = add_eef_pose(action_rel[:8], init_pose[:8])
    right_pose = add_eef_pose(action_rel[8:16], init_pose[8:16])
    out = np.concatenate([left_pose, right_pose], axis=0).astype(np.float32)
    out[3:7] = _normalize_quaternion(out[3:7])
    out[11:15] = _normalize_quaternion(out[11:15])
    out[7] = float(np.clip(out[7], 0.0, 1.0))
    out[15] = float(np.clip(out[15], 0.0, 1.0))
    return out


class SemanticMoTRobotWinPolicy:
    def __init__(
        self,
        *,
        ckpt_path: str,
        config_name: str,
        t5_dir: str,
        vjepa_model_name: str,
        dino_model_name: str,
        device: str,
        model_dtype: torch.dtype,
        use_proprio: Optional[bool],
        replan_steps: int,
        num_inference_steps: int,
        seed: Optional[int],
        joint_future_denoising: Optional[bool],
        future_semantic_steps: Optional[int],
        t5_max_length: Optional[int],
        timing_enabled: bool,
        compile_infer_action: bool,
        compile_mode: str,
        static_rtc_prefix_graphs: bool = False,
        rtc_prefix_graph_max_length: int = 6,
    ) -> None:
        self.model, self.cfg = load_model(
            ckpt_path=ckpt_path,
            config_name=config_name,
            device=device,
            dtype=model_dtype,
            use_proprio=use_proprio,
        )
        self.device = device
        self.dtype = model_dtype
        self.teacher_type = str(getattr(self.cfg, "teacher_type")).lower()
        raw_per_view_sizes = getattr(self.cfg, "per_view_image_sizes")
        self.per_view_sizes = tuple(
            (int(view_h), int(view_w))
            for view_h, view_w in raw_per_view_sizes
        )
        self.patch_size = int(getattr(self.cfg, "image_patch_size"))
        self.per_view_num_spatial_tokens = tuple(
            (view_h // self.patch_size) * (view_w // self.patch_size)
            for view_h, view_w in self.per_view_sizes
        )
        self.current_view_resize_mode = str(
            getattr(self.cfg, "robotwin_current_view_resize_mode", "center_crop") or "center_crop"
        ).strip().lower()
        if self.current_view_resize_mode not in {"center_crop", "direct"}:
            raise ValueError(
                "robotwin_current_view_resize_mode must be 'center_crop' or 'direct', "
                f"got {self.current_view_resize_mode!r}"
            )
        self.action_horizon = int(getattr(self.cfg, "future_blocks")) * int(getattr(self.cfg, "action_per_frame"))
        self.action_dim = int(getattr(self.cfg, "action_dim"))
        replan_steps_resolved = int(getattr(self.cfg, "action_per_frame")) if replan_steps <= 0 else int(replan_steps)
        self.replan_steps = int(max(1, min(replan_steps_resolved, self.action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.seed = seed
        self.generator = self._build_rollout_generator()
        self.joint_future_denoising = joint_future_denoising
        if future_semantic_steps is None:
            cfg_future_semantic_steps = getattr(self.cfg, "future_semantic_steps", None)
            self.future_semantic_steps = (
                None if _is_none_like(cfg_future_semantic_steps) else int(cfg_future_semantic_steps)
            )
        else:
            self.future_semantic_steps = int(future_semantic_steps)
        self.timing_enabled = bool(timing_enabled)
        self.profile_infer = str(os.environ.get("MTLBOT_PROFILE_INFER", "")).strip().lower() in {
            "1", "true", "yes", "on",
        }
        self.compile_infer_action = bool(compile_infer_action and device != "cpu")
        self.static_rtc_prefix_graphs = bool(static_rtc_prefix_graphs and self.compile_infer_action)
        self.rtc_prefix_graph_max_length = max(0, int(rtc_prefix_graph_max_length))
        self.compile_vjepa = self.teacher_type == "vjepa" and device != "cpu" and os.environ.get("COPPER_DISABLE_COMPILE") != "1"
        self.compile_mode = str(compile_mode).strip() or "reduce-overhead"
        # No-spatial and VLA ablations have a different (shorter) action-cache
        # layout than the regular model: the former removes current spatial
        # tokens and the latter removes abstract anchor tokens. With PyTorch
        # 2.9, materializing their seven RTC prefix-specialized action CUDA
        # Graphs can corrupt the CUDA-Graph checkpoint allocator
        # (``curr_block->next`` assertion during replay).
        # This is a CUDA-Graph runtime bug, not an Inductor compilation
        # limitation: retain torch.compile fusion but keep these action calls
        # out of CUDA Graph capture.  Spatial models retain their established
        # reduce-overhead CUDA-Graph path unchanged.
        self.disable_action_cudagraphs = (
            str(getattr(self.cfg, "current_spatial_mode", "off")).strip().lower() == "off"
            or not bool(getattr(self.cfg, "use_abstract_anchor_tokens", True))
        )
        self._cuda_blas_initialized_threads: set[int] = set()

        if device.startswith("cuda"):
            _initialize_cuda_blas_for_graph_capture(device)

        self.tokenizer, self.text_encoder, self.text_device = load_text_encoder(t5_dir, device, model_dtype)
        self.text_max_length = _resolve_text_emb_max_tokens(self.cfg, t5_max_length)
        self.text_use_padding_mask = _resolve_text_emb_use_padding_mask(self.cfg)

        if self.teacher_type == "vjepa":
            self.vjepa_model = load_vjepa_encoder(vjepa_model_name, device, model_dtype)
            # Profile this fixed-shape encoder independently from the action
            # graph.  RealBot has one head shape and one shared wrist shape.
            self.vjepa_model = _maybe_compile_callable(
                self.vjepa_model,
                name="SemanticMoTRobotWinPolicy.vjepa_encoder",
                enabled=self.compile_vjepa,
                mode=self.compile_mode,
            )
            self.dino_model = None
        else:
            self.dino_model = load_dino_encoder(dino_model_name, device, model_dtype)
            self.vjepa_model = None
        compactor = getattr(self.model, "compactor", None)
        if compactor is not None and device != "cpu":
            # The compact control latent encoder consumes V-JEPA spatial
            # tokens once per request before video prefill.
            self.model._compactor_infer_fn = _maybe_compile_callable(
                compactor,
                name="SemanticMoTRobotWinPolicy.compactor",
                enabled=True,
                mode=self.compile_mode,
            )
        # Optional, inference-only representation surgery.  The basis is fit
        # offline from training episodes and applied on the compact feature
        # axis before compact_anchor_proj.  With the environment variables
        # unset this block is inert and the normal evaluation path is exact.
        surgery_path = str(os.environ.get("COPPER_Z_SURGERY_PATH", "")).strip()
        surgery_mode = str(os.environ.get("COPPER_Z_SURGERY_MODE", "none")).strip().lower()
        surgery_rank = int(os.environ.get("COPPER_Z_SURGERY_RANK", "10"))
        if surgery_path and surgery_mode not in {"", "none", "original"}:
            if surgery_mode not in {"nuisance", "temporal", "random"}:
                raise ValueError(f"Unsupported COPPER_Z_SURGERY_MODE={surgery_mode!r}")
            payload = np.load(surgery_path)
            basis_key = f"{surgery_mode}_basis"
            if basis_key not in payload or "mean" not in payload:
                raise KeyError(f"{surgery_path} must contain {basis_key!r} and 'mean'")
            basis_np = np.asarray(payload[basis_key], dtype=np.float32)
            if not 0 < surgery_rank <= basis_np.shape[1]:
                raise ValueError(f"Surgery rank must be in [1,{basis_np.shape[1]}], got {surgery_rank}")
            surgery_basis = torch.from_numpy(basis_np[:, :surgery_rank]).to(device=device)
            surgery_mean = torch.from_numpy(np.asarray(payload["mean"], dtype=np.float32)).to(device=device).reshape(1, 1, -1)
            compactor_fn = getattr(self.model, "_compactor_infer_fn", compactor)

            def _compactor_with_surgery(tokens, *, conditioning_context=None, conditioning_mask=None):
                z = compactor_fn(
                    tokens,
                    conditioning_context=conditioning_context,
                    conditioning_mask=conditioning_mask,
                )
                u = surgery_basis.to(dtype=z.dtype)
                mu = surgery_mean.to(dtype=z.dtype)
                return z - ((z - mu) @ u) @ u.transpose(0, 1)

            self.model._compactor_infer_fn = _compactor_with_surgery
            logger.info("Enabled Copper Z surgery: mode=%s rank=%d basis=%s", surgery_mode, surgery_rank, surgery_path)
            print(
                f"[copper-surgery] enabled mode={surgery_mode} rank={surgery_rank} basis={surgery_path}",
                flush=True,
            )
        current_dino_encoder = getattr(self.model, "current_spatial_dino_encoder", None)
        if current_dino_encoder is not None and device != "cpu":
            self.model._current_spatial_dino_encoder_infer_fn = _maybe_compile_callable(
                current_dino_encoder,
                name="SemanticMoTRobotWinPolicy.current_spatial_dino_encoder",
                enabled=True,
                mode=self.compile_mode,
            )
        # Video prefill has a fixed 30-layer tensor-tuple cache schema.  It
        # is a separate graph from denoising so its cache is materialized at
        # the graph boundary before repeated action replays.
        self.model._video_prefill_fn = _maybe_compile_callable(
            self.model.mot.prefill_video_cache_flat,
            name="SemanticMoTRobotWinPolicy.video_prefill",
            enabled=device != "cpu",
            mode=self.compile_mode,
        )

        self.proprio_dim = int(getattr(self.cfg, "proprio_dim", 0) or 0)
        self.robotwin_action_space = infer_robotwin_action_space(self.cfg)
        self.robotwin_action_type = infer_robotwin_action_type(self.cfg)
        self.robotwin_qpos_action_mode = infer_robotwin_qpos_action_mode(self.cfg)
        self.proprio_normalizer = build_proprio_normalizer(self.cfg)
        self.denormalize_action = build_action_denormalizer(self.cfg)
        # The denormalizer is affine (including identity dimensions).  Recover
        # its inverse once so RTC's physical joint targets can be supplied to
        # the model in the same normalized action space used during training.
        zero = np.zeros((1, self.action_dim), dtype=np.float32)
        one = np.ones((1, self.action_dim), dtype=np.float32)
        self._action_norm_zero = self.denormalize_action(zero)[0]
        self._action_norm_scale = self.denormalize_action(one)[0] - self._action_norm_zero
        if np.any(np.abs(self._action_norm_scale) < 1e-8):
            raise ValueError("action normalization has a non-invertible dimension")
        # The static tensor stages are compiled separately; image decoding,
        # token assembly and request metadata remain outside CUDA Graph.
        self.model._action_denoise_step_fn = _maybe_compile_callable(
            self.model._action_denoise_step,
            name="SemanticMoTRobotWinPolicy.action_denoise_step",
            enabled=self.compile_infer_action,
            mode=self.compile_mode,
            disable_cudagraphs=self.disable_action_cudagraphs,
        )
        self.model._action_denoise_step_fns_by_committed_len = {}
        if self.static_rtc_prefix_graphs:
            max_prefix_len = min(self.rtc_prefix_graph_max_length, self.action_horizon - 1)
            # Each committed-prefix length gets its own compiled callable and
            # CUDA Graph allocation.  Although all K values share the action
            # horizon, keeping separate callables prevents graph-cache reuse
            # across RTC queue depths and avoids any K-dependent recompilation
            # on the control path.
            for prefix_len in range(max_prefix_len + 1):
                self.model._action_denoise_step_fns_by_committed_len[prefix_len] = _maybe_compile_callable(
                    self.model._action_denoise_step,
                    name=f"SemanticMoTRobotWinPolicy.action_denoise_step_prefix_{prefix_len}",
                    enabled=True,
                    mode=self.compile_mode,
                    disable_cudagraphs=self.disable_action_cudagraphs,
                )
            self.model._action_denoise_step_fn = self.model._action_denoise_step_fns_by_committed_len[0]
            logger.info(
                "Enabled %d static RTC action-denoise %s for committed prefix lengths 0..%d.",
                max_prefix_len + 1,
                "compiled callables (CUDA Graph disabled)" if self.disable_action_cudagraphs else "graphs",
                max_prefix_len,
            )
        self.infer_action_fn = self.model.infer_action

        self.pending_actions: deque[np.ndarray] = deque()
        self.current_instruction: Optional[str] = None
        self.current_context: Optional[torch.Tensor] = None
        self.current_context_mask: Optional[torch.Tensor] = None
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self.debug_hold_enabled = str(os.environ.get("ROBOTWIN_HOLD_DEBUG", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.debug_hold_trace_path = os.environ.get("ROBOTWIN_HOLD_TRACE_PATH")
        self.debug_hold_cmd_threshold = float(os.environ.get("ROBOTWIN_HOLD_CMD_THRESHOLD", "0.05"))
        self.force_identity_hold = str(os.environ.get("ROBOTWIN_IDENTITY_HOLD", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.debug_chunk_enabled = str(os.environ.get("ROBOTWIN_CHUNK_DEBUG", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.debug_chunk_trace_path = os.environ.get("ROBOTWIN_CHUNK_TRACE_PATH")
        self.debug_chunk_max_actions = max(1, int(os.environ.get("ROBOTWIN_CHUNK_MAX_ACTIONS", "3")))
        self.debug_chunk_max_dims = max(1, int(os.environ.get("ROBOTWIN_CHUNK_MAX_DIMS", "6")))
        self.debug_step_enabled = str(os.environ.get("ROBOTWIN_STEP_DEBUG", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.debug_step_trace_path = os.environ.get("ROBOTWIN_STEP_TRACE_PATH")
        self.debug_step_max_dims = max(1, int(os.environ.get("ROBOTWIN_STEP_MAX_DIMS", "8")))
        self._debug_chunk_index = 0
        self._debug_chunk_printed_norm = False
        self._debug_prev_action: np.ndarray | None = None
        self._debug_prev_obs: np.ndarray | None = None

    def _build_rollout_generator(self) -> torch.Generator | None:
        if self.seed is None:
            return None
        return torch.Generator(device=torch.device(self.device)).manual_seed(int(self.seed))

    def _extract_joint_state_np(self, observation: Optional[Dict[str, Any]]) -> np.ndarray | None:
        if observation is None:
            return None
        if self.robotwin_action_space != "joint":
            return None
        try:
            proprio = extract_proprio_robotwin(
                observation,
                self.proprio_dim,
                self.robotwin_action_space,
                "cpu",
                torch.float32,
            )
        except Exception:
            return None
        if proprio is None:
            return None
        return np.asarray(proprio.detach().cpu().numpy().reshape(-1), dtype=np.float32)

    def _write_hold_debug_record(self, record: dict[str, Any]) -> None:
        if not self.debug_hold_trace_path:
            return
        trace_path = Path(self.debug_hold_trace_path).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_chunk_debug_record(self, record: dict[str, Any]) -> None:
        if not self.debug_chunk_trace_path:
            return
        trace_path = Path(self.debug_chunk_trace_path).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_step_debug_record(self, record: dict[str, Any]) -> None:
        if not self.debug_step_trace_path:
            return
        trace_path = Path(self.debug_step_trace_path).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _debug_chunk_plan(
        self,
        task_env,
        *,
        observation: Dict[str, Any],
        instruction: str,
        views: list[torch.Tensor],
        proprio_raw: torch.Tensor | None,
        proprio_norm: torch.Tensor | None,
        action_chunk: np.ndarray,
    ) -> None:
        if not self.debug_chunk_enabled:
            return

        step_idx = int(getattr(task_env, "take_action_cnt", -1))
        raw_state = self._extract_joint_state_np(observation)
        num_actions = min(int(action_chunk.shape[0]), self.debug_chunk_max_actions)
        action_preview = action_chunk[:num_actions]
        delta_preview = None
        if raw_state is not None and self.robotwin_action_space == "joint":
            delta_preview = action_preview - raw_state[None, :]

        view_stats = [_tensor_stats(view) for view in views]
        record = {
            "chunk_index": int(self._debug_chunk_index),
            "step": step_idx,
            "instruction": str(instruction),
            "robotwin_action_space": self.robotwin_action_space,
            "robotwin_qpos_action_mode": self.robotwin_qpos_action_mode,
            "replan_steps": int(self.replan_steps),
            "action_horizon": int(action_chunk.shape[0]),
            "raw_state_head": _summarize_vector(raw_state, self.debug_chunk_max_dims),
            "proprio_raw_head": _summarize_vector(proprio_raw, self.debug_chunk_max_dims),
            "proprio_norm_head": _summarize_vector(proprio_norm, self.debug_chunk_max_dims),
            "view_stats": view_stats,
            "action_preview_head": [
                _summarize_vector(action_preview[i], self.debug_chunk_max_dims) for i in range(num_actions)
            ],
            "action_delta_head": (
                [_summarize_vector(delta_preview[i], self.debug_chunk_max_dims) for i in range(num_actions)]
                if delta_preview is not None
                else None
            ),
            "action_delta_norms": (
                [float(np.linalg.norm(delta_preview[i])) for i in range(num_actions)]
                if delta_preview is not None
                else None
            ),
            "chunk_action_jump_norms": (
                [float(np.linalg.norm(action_preview[i] - action_preview[i - 1])) for i in range(1, num_actions)]
                if num_actions > 1
                else []
            ),
        }
        if not self._debug_chunk_printed_norm and self.proprio_normalizer is not None:
            record["proprio_norm_center_head"] = _summarize_vector(
                self.proprio_normalizer["center"], self.debug_chunk_max_dims
            )
            record["proprio_norm_scale_head"] = _summarize_vector(
                self.proprio_normalizer["scale"], self.debug_chunk_max_dims
            )
            self._debug_chunk_printed_norm = True

        summary = {
            "chunk": record["chunk_index"],
            "step": record["step"],
            "raw_state_head": record["raw_state_head"],
            "proprio_norm_head": record["proprio_norm_head"],
            "action0_head": (record["action_preview_head"][0] if record["action_preview_head"] else None),
            "delta0_head": (record["action_delta_head"][0] if record["action_delta_head"] else None),
            "delta_norms": record["action_delta_norms"],
            "view_means": [round(float(item["mean"]), 4) for item in view_stats],
            "view_stds": [round(float(item["std"]), 4) for item in view_stats],
        }
        print(f"[chunk-debug] {json.dumps(summary, ensure_ascii=False)}")
        self._write_chunk_debug_record(record)
        self._debug_chunk_index += 1

    def _debug_step_action(
        self,
        *,
        task_env,
        observation: Optional[Dict[str, Any]],
        action: np.ndarray,
        queue_len_after_pop: int,
    ) -> None:
        if not self.debug_step_enabled:
            return

        step_idx = int(getattr(task_env, "take_action_cnt", -1))
        raw_state = self._extract_joint_state_np(observation)
        proprio_raw_t = None
        proprio_norm_t = None
        if observation is not None and self.proprio_dim > 0:
            try:
                proprio_raw_t = extract_proprio_robotwin(
                    observation,
                    self.proprio_dim,
                    self.robotwin_action_space,
                    "cpu",
                    torch.float32,
                )
                if proprio_raw_t is not None:
                    proprio_norm_t = normalize_proprio(proprio_raw_t.clone(), self.proprio_normalizer)
            except Exception as exc:
                proprio_raw_t = None
                proprio_norm_t = None
                logger.warning("step debug failed to extract proprio at step %s: %s", step_idx, exc)

        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        delta = None if raw_state is None else (action_np - raw_state)
        record = {
            "step": step_idx,
            "queue_len_after_pop": int(queue_len_after_pop),
            "raw_state_head": _summarize_vector(raw_state, self.debug_step_max_dims),
            "proprio_raw_head": _summarize_vector(proprio_raw_t, self.debug_step_max_dims),
            "proprio_norm_head": _summarize_vector(proprio_norm_t, self.debug_step_max_dims),
            "action_head": _summarize_vector(action_np, self.debug_step_max_dims),
            "action_delta_head": _summarize_vector(delta, self.debug_step_max_dims),
            "action_delta_norm": (None if delta is None else float(np.linalg.norm(delta))),
        }
        summary = {
            "step": record["step"],
            "queue": record["queue_len_after_pop"],
            "state": record["raw_state_head"],
            "proprio_norm": record["proprio_norm_head"],
            "action": record["action_head"],
            "delta": record["action_delta_head"],
            "delta_norm": record["action_delta_norm"],
        }
        print(f"[step-debug] {json.dumps(summary, ensure_ascii=False)}")
        self._write_step_debug_record(record)

    def _debug_hold_step(
        self,
        task_env,
        *,
        observation_before: Optional[Dict[str, Any]],
        action: np.ndarray,
    ) -> None:
        if not self.debug_hold_enabled:
            return
        obs_before = self._extract_joint_state_np(observation_before)
        obs_after_payload = getattr(task_env, "now_obs", None)
        if hasattr(task_env, "get_obs"):
            try:
                obs_after_payload = task_env.get_obs()
            except Exception:
                obs_after_payload = getattr(task_env, "now_obs", None)
        obs_after = self._extract_joint_state_np(obs_after_payload)
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        hold_error = None if obs_before is None else (action_np - obs_before)
        tracking_error = None if obs_after is None else (obs_after - action_np)
        action_delta = None if self._debug_prev_action is None else (action_np - self._debug_prev_action)
        obs_delta = None if self._debug_prev_obs is None or obs_before is None else (obs_before - self._debug_prev_obs)
        hold_norm = None if hold_error is None else float(np.linalg.norm(hold_error))
        track_norm = None if tracking_error is None else float(np.linalg.norm(tracking_error))
        action_delta_norm = None if action_delta is None else float(np.linalg.norm(action_delta))
        obs_delta_norm = None if obs_delta is None else float(np.linalg.norm(obs_delta))
        is_hold_like = bool(hold_norm is not None and hold_norm <= self.debug_hold_cmd_threshold)

        step_idx = int(getattr(task_env, "take_action_cnt", -1))
        prefix = "[hold-debug]" if is_hold_like else "[action-debug]"
        print(
            f"{prefix} step={step_idx} hold_norm={hold_norm!s} "
            f"track_norm={track_norm!s} action_delta_norm={action_delta_norm!s} obs_delta_norm={obs_delta_norm!s}"
        )
        self._write_hold_debug_record(
            {
                "step": step_idx,
                "is_hold_like": is_hold_like,
                "hold_norm": hold_norm,
                "track_norm": track_norm,
                "action_delta_norm": action_delta_norm,
                "obs_delta_norm": obs_delta_norm,
                "action": action_np.tolist(),
                "obs_before": None if obs_before is None else obs_before.tolist(),
                "obs_after": None if obs_after is None else obs_after.tolist(),
                "hold_error": None if hold_error is None else hold_error.tolist(),
                "tracking_error": None if tracking_error is None else tracking_error.tolist(),
            }
        )
        self._debug_prev_action = action_np.copy()
        if obs_before is not None:
            self._debug_prev_obs = obs_before.copy()

    def _encode_instruction(self, instruction: str) -> None:
        instruction = build_instruction_text(instruction, getattr(self.cfg, "instruction_template", None))
        if instruction == self.current_instruction and self.current_context is not None:
            return
        context, text_token_mask = encode_text(
            instruction,
            self.tokenizer,
            self.text_encoder,
            self.text_device,
            self.device,
            max_length=self.text_max_length,
        )
        if self.text_use_padding_mask:
            context_mask = text_token_mask[:, : context.shape[1]]
        else:
            context_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
        self.current_instruction = instruction
        self.current_context = context
        self.current_context_mask = context_mask

    def _extract_init_pose(self, observation: Dict[str, Any]) -> np.ndarray:
        return np.array(
            observation["endpose"]["left_endpose"]
            + [observation["endpose"]["left_gripper"]]
            + observation["endpose"]["right_endpose"]
            + [observation["endpose"]["right_gripper"]],
            dtype=np.float32,
        )

    def _infer_action_chunk(
        self,
        task_env,
        observation: Dict[str, Any],
        instruction: str,
        action_prefix: np.ndarray | None = None,
    ) -> np.ndarray:
        # socketserver executes requests on worker threads, whereas startup
        # warmup runs on the main thread. cuBLAS handles are thread-local, so
        # initialize it again here before this worker enters a V-JEPA CUDA
        # Graph capture. The operation is tiny and happens before any model
        # work for the request.
        thread_id = threading.get_ident()
        if self.compile_vjepa and thread_id not in self._cuda_blas_initialized_threads:
            _initialize_cuda_blas_for_graph_capture(self.device)
            self._cuda_blas_initialized_threads.add(thread_id)
        self._encode_instruction(instruction)
        views, views_direct, _stitched = obs_to_model_input_robotwin(
            observation,
            self.per_view_sizes,
            self.device,
            self.dtype,
            current_resize_mode=self.current_view_resize_mode,
        )
        proprio = extract_proprio_robotwin(
            observation,
            self.proprio_dim,
            self.robotwin_action_space,
            self.device,
            self.dtype,
        )
        proprio_raw = None if proprio is None else proprio.detach().clone()
        if proprio is not None:
            proprio = normalize_proprio(proprio, self.proprio_normalizer)

        anchor_views = views_direct
        vjepa_start = None
        if self.profile_infer and self.device != "cpu":
            torch.cuda.synchronize()
            vjepa_start = torch.cuda.Event(enable_timing=True)
            vjepa_start.record()
        if self.teacher_type == "vjepa":
            anchor_dino_cls, anchor_dino_registers, anchor_dino_spatial = extract_anchor_vjepa(
                anchor_views,
                self.vjepa_model,
                self.per_view_num_spatial_tokens,
                self.device,
                self.dtype,
                compiled_vjepa=self.compile_vjepa,
            )
        else:
            anchor_dino_cls, anchor_dino_registers, anchor_dino_spatial = extract_anchor_dino(
                anchor_views,
                self.dino_model,
                self.per_view_num_spatial_tokens,
                self.device,
                self.dtype,
            )
        vjepa_ms = None
        if vjepa_start is not None:
            vjepa_end = torch.cuda.Event(enable_timing=True)
            vjepa_end.record()

        prefix_norm = None
        if action_prefix is not None:
            action_prefix = np.asarray(action_prefix, dtype=np.float32)
            if action_prefix.ndim != 2 or action_prefix.shape[1] != self.action_dim:
                raise ValueError(
                    f"action_prefix must be [K,{self.action_dim}], got {action_prefix.shape}"
                )
            if not 0 < action_prefix.shape[0] < self.action_horizon:
                raise ValueError(
                    f"action_prefix length must be in [1,{self.action_horizon - 1}], got {action_prefix.shape[0]}"
                )
            if self.robotwin_action_space == "joint" and self.robotwin_qpos_action_mode == "rel_qpos":
                if proprio_raw is None:
                    raise ValueError("rel_qpos RTC requires current joint observation.")
                action_prefix = action_prefix - proprio_raw.detach().cpu().float().numpy().reshape(1, -1)
            prefix_norm = (action_prefix - self._action_norm_zero) / self._action_norm_scale

        if self.profile_infer:
            self.model._inference_profile_enabled = True
            core_start = torch.cuda.Event(enable_timing=True) if self.device != "cpu" else None
            if core_start is not None:
                core_start.record()
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        chunk_norm = predict_action_chunk(
            self.infer_action_fn,
            views,
            anchor_dino_cls,
            anchor_dino_spatial,
            anchor_dino_registers,
            self.current_context,
            self.current_context_mask,
            proprio,
            self.num_inference_steps,
            self.generator,
            self.device,
            dtype=self.dtype,
            joint_future_denoising=self.joint_future_denoising,
            future_semantic_steps=self.future_semantic_steps,
            action_prefix=prefix_norm,
        )
        if self.profile_infer:
            core_end = torch.cuda.Event(enable_timing=True) if self.device != "cpu" else None
            if core_end is not None:
                core_end.record()
                torch.cuda.synchronize()
                vjepa_ms = float(vjepa_start.elapsed_time(vjepa_end)) if vjepa_start is not None else None
                self.last_inference_profile_ms = {
                    "core_model_ms": float(core_start.elapsed_time(core_end)),
                    **self.model.get_inference_profile_ms(),
                }
                if vjepa_ms is not None:
                    self.last_inference_profile_ms["vjepa_ms"] = vjepa_ms
            else:
                self.last_inference_profile_ms = {}
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        action = self.denormalize_action(chunk_norm).astype(np.float32, copy=False)
        if self.robotwin_action_space == "joint" and self.robotwin_qpos_action_mode == "rel_qpos":
            if proprio_raw is None:
                raise ValueError("rel_qpos inference requires current 14D joint observation.")
            current_qpos = proprio_raw.detach().cpu().float().numpy().reshape(-1)
            if current_qpos.shape[0] != action.shape[-1]:
                raise ValueError(
                    "rel_qpos inference dim mismatch: "
                    f"current_qpos_dim={current_qpos.shape[0]}, action_dim={action.shape[-1]}"
                )
            action = action + current_qpos[None, :]
        self._debug_chunk_plan(
            task_env,
            observation=observation,
            instruction=instruction,
            views=views,
            proprio_raw=proprio_raw,
            proprio_norm=proprio,
            action_chunk=action,
        )
        if self.robotwin_action_space == "joint":
            return action

        init_pose = self._extract_init_pose(observation)
        return np.stack([add_init_pose(step, init_pose) for step in action], axis=0)

    def _fill_action_queue(self, task_env, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(task_env, observation, instruction)
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError("Observation is required when action queue is empty.")
            if self.force_identity_hold:
                obs_state = self._extract_joint_state_np(observation)
                if obs_state is None:
                    raise ValueError("ROBOTWIN_IDENTITY_HOLD requires joint-mode observation.state.")
                self.pending_actions.append(obs_state.astype(np.float32, copy=True))
            else:
                instruction = task_env.get_instruction()
                self._fill_action_queue(task_env, observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        self._debug_step_action(
            task_env=task_env,
            observation=observation,
            action=action,
            queue_len_after_pop=len(self.pending_actions),
        )
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type=self.robotwin_action_type)
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self._debug_hold_step(task_env, observation_before=observation, action=action)

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.current_instruction = None
        self.current_context = None
        self.current_context_mask = None
        self.generator = self._build_rollout_generator()
        self._debug_chunk_index = 0
        self._debug_chunk_printed_norm = False
        self._debug_prev_action = None
        self._debug_prev_obs = None
        self.reset_timing_rollout()


def get_model(usr_args: Dict[str, Any]):
    ckpt_path = usr_args.get("ckpt_setting")
    if _is_none_like(ckpt_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")
    config_name = str(usr_args.get("config_name") or "config_robotwin_train")

    device = str(usr_args.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"
    mixed_precision = str(usr_args.get("mixed_precision") or "bf16")
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    t5_dir = usr_args.get("t5_dir")
    if _is_none_like(t5_dir):
        raise ValueError("`t5_dir` is required for Semantic MoT RobotWin eval.")
    t5_dir = str(Path(str(t5_dir)).expanduser().resolve())
    if not Path(t5_dir).exists():
        raise FileNotFoundError(f"T5 directory not found: {t5_dir}")

    use_proprio = _parse_optional_bool(usr_args.get("use_proprio"))
    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    seed = _parse_optional_int(usr_args.get("seed"))
    future_semantic_steps = _parse_optional_int(usr_args.get("future_semantic_steps"))
    t5_max_length = _parse_optional_int(usr_args.get("t5_max_length"))
    compile_infer_action = _parse_bool(usr_args.get("compile_infer_action", True))
    compile_mode = str(usr_args.get("compile_mode") or "reduce-overhead")
    static_rtc_prefix_graphs = _parse_bool(usr_args.get("static_rtc_prefix_graphs", False))
    rtc_prefix_graph_max_length = _parse_optional_int(usr_args.get("rtc_prefix_graph_max_length"))
    env_compile_infer_action = os.environ.get("ROBOTWIN_COMPILE_INFER_ACTION")
    env_compile_mode = os.environ.get("ROBOTWIN_COMPILE_MODE")

    if not _is_none_like(env_compile_infer_action):
        compile_infer_action = _parse_bool(env_compile_infer_action)
    if not _is_none_like(env_compile_mode):
        compile_mode = str(env_compile_mode).strip()

    if replan_steps is None:
        replan_steps = 0
    if num_inference_steps is None:
        num_inference_steps = 20
    if rtc_prefix_graph_max_length is None:
        rtc_prefix_graph_max_length = 6

    policy = SemanticMoTRobotWinPolicy(
        ckpt_path=str(ckpt_path),
        config_name=config_name,
        t5_dir=t5_dir,
        vjepa_model_name=str(usr_args.get("vjepa_model") or "vjepa2_1_vit_large_384"),
        dino_model_name=str(usr_args.get("dino_model") or "facebook/dinov2-with-registers-large"),
        device=device,
        model_dtype=model_dtype,
        use_proprio=use_proprio,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        seed=seed,
        joint_future_denoising=_parse_optional_bool(usr_args.get("joint_future_denoising")),
        future_semantic_steps=future_semantic_steps,
        t5_max_length=t5_max_length,
        timing_enabled=bool(_parse_bool(usr_args.get("timing_enabled", False))),
        compile_infer_action=compile_infer_action,
        compile_mode=compile_mode,
        static_rtc_prefix_graphs=static_rtc_prefix_graphs,
        rtc_prefix_graph_max_length=rtc_prefix_graph_max_length,
    )
    logger.info(
        "Initialized SemanticMoTRobotWinPolicy | ckpt=%s | config=%s | horizon=%d | replan=%d | teacher=%s | compile_infer_action=%s | action_cudagraphs=%s | compile_vjepa=True | compile_mode=%s",
        ckpt_path,
        config_name,
        policy.action_horizon,
        policy.replan_steps,
        str(getattr(policy.cfg, "teacher_type", "dino")),
        policy.compile_infer_action,
        not policy.disable_action_cudagraphs,
        policy.compile_mode,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    model.step(TASK_ENV, observation)


def reset_model(model):
    model.reset()
