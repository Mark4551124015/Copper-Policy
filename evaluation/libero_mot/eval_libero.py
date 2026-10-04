#!/usr/bin/env python3
"""Direct LIBERO evaluation for Copper-Policy checkpoints.

Does NOT require a websocket server — the model is loaded and run in-process.

Usage:
    # Full suite
    python evaluation/libero_mot/eval_libero.py \
        --ckpt pretrained_weights/copper_policy/libero/policy.pt \
        --suite libero_spatial \
        --out-dir outputs/eval/libero \
        --t5-dir pretrained_weights/text_encoder/wan22_ti2v_5b \
        --dino-model facebook/dinov2-with-registers-large

    # Task subset (tasks 0..4)
    python evaluation/libero_mot/eval_libero.py \
        --ckpt ... --suite libero_spatial --task-range 0 5

    # Summarize after evaluation
    python evaluation/libero_mot/summarize.py --out-dir outputs/eval/libero
"""

import argparse
import copy
import contextlib
import html
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

# Project root must be on sys.path before any local imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TEXT_EMB_MAX_TOKENS = 64
DEFAULT_TEXT_EMB_STORAGE_LENGTH = 512

import imageio
import numpy as np
import torch
from easydict import EasyDict
LIBERO_IMPORT_ERROR = None
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as exc:
    LIBERO_IMPORT_ERROR = exc
    benchmark = None
    OffScreenRenderEnv = None
from tqdm import tqdm

if not hasattr(np, "float_"):
    np.float_ = np.float64

from inference.config import VA_CONFIGS
from inference.model import (
    build_semantic_mot_model,
    load_semantic_mot_checkpoint,
)
from wan_va.modules.vjepa_hub import load_vjepa_model
from inference.precision import _keep_selected_tensors_fp32

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LIBERO_RENDER_RES = 256
DINO_PATCH_SIZE = 14  # All DINOv2 variants use 14×14 patches
VJEPA_PATCH_SIZE = 16  # All VJEPA variants use 16×16 patches
VJEPA_TUBELET_SIZE = 2  # VJEPA compresses 2 input frames into 1 temporal token
DATE_TIME = time.strftime("%Y%m%d_%H%M%S")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
FASTWAM_INSTRUCTION_TEMPLATE = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
SEMANTIC_NEUTRAL_SUFFIXES = (
    "thank you",
    "thanks",
    "many thanks",
    "thank you very much",
)

SUITE_MAX_STEPS = {
    "libero_spatial": 400,
    "libero_object": 400,
    "libero_goal": 400,
    "libero_10": 700,
    "libero_90": 700,
}

ALL_SUITES = list(SUITE_MAX_STEPS.keys())
_TASK_VARIANT_SUFFIX_PATTERNS = (
    re.compile(r"_initstate_\d+$"),
    re.compile(r"_noise_\d+$"),
    re.compile(r"_view(?:_\d+)+$"),
    re.compile(r"_language_\d+$"),
    re.compile(r"_light_\d+$"),
    re.compile(r"_level\d+_sample\d+$"),
    re.compile(r"_add_\d+$"),
    re.compile(r"_table_\d+$"),
    re.compile(r"_tb_\d+$"),
)


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


def _suite_dataset_name(suite: str) -> str:
    return f"{suite}_no_noops_lerobot"


def _canonical_task_key(text: str) -> str:
    text = _prompt_clean(text).lower()
    text = re.sub(r"\.bddl$", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return re.sub(r"^[a-z]+(?:_[a-z]+)*_scene\d+_", "", text)


def _strip_task_variant_suffixes(task_key: str) -> str:
    stripped = task_key
    while stripped:
        updated = stripped
        for pattern in _TASK_VARIANT_SUFFIX_PATTERNS:
            updated = pattern.sub("", updated)
        updated = updated.strip("_")
        if updated == stripped:
            return updated
        stripped = updated
    return task_key


def _candidate_task_keys(text: str) -> list[str]:
    task_key = _canonical_task_key(text)
    if not task_key:
        return []
    stripped_key = _strip_task_variant_suffixes(task_key)
    if stripped_key == task_key:
        return [task_key]
    return [task_key, stripped_key]


def _suite_result_files(suite_dir: Path) -> list[Path]:
    parts_dir = suite_dir / "_parts"
    if parts_dir.exists():
        result_paths = parts_dir.glob("*_results.json")
    else:
        result_paths = suite_dir.glob("*_results.json")
    return sorted(result_paths, key=lambda path: (path.stat().st_mtime, path.name))


def _result_episode_outcomes(payload: dict) -> dict[int, bool]:
    outcomes: dict[int, bool] = {}
    for episode_idx in payload.get("failure_episodes", []):
        outcomes[int(episode_idx)] = False
    for episode_idx in payload.get("success_episodes", []):
        outcomes[int(episode_idx)] = True
    if outcomes:
        return outcomes

    episode_indices = payload.get("episode_indices")
    if isinstance(episode_indices, list) and len(episode_indices) == 1:
        total_episodes = int(payload.get("total_episodes", 0) or 0)
        successes = int(payload.get("successes", 0) or 0)
        if total_episodes == 1:
            outcomes[int(episode_indices[0])] = successes > 0
    return outcomes


def _load_existing_episode_outcomes(suite_dir: Path) -> dict[int, dict[int, bool]]:
    task_outcomes: dict[int, dict[int, bool]] = {}
    for result_file in _suite_result_files(suite_dir):
        try:
            payload = json.loads(result_file.read_text())
        except Exception as exc:
            print(f"Warning: failed to read existing result file {result_file}: {exc}")
            continue

        payload_task_id = payload.get("task_id")
        if payload_task_id is None:
            match = re.search(r"_task(\d+)", result_file.stem)
            if match is None:
                continue
            payload_task_id = int(match.group(1))
        payload_task_id = int(payload_task_id)
        task_episode_outcomes = task_outcomes.setdefault(payload_task_id, {})
        for episode_idx, success in _result_episode_outcomes(payload).items():
            task_episode_outcomes[int(episode_idx)] = bool(success)
    return task_outcomes


def _summarize_existing_episodes(
    suite_name: str,
    task_id: int,
    task_description: str,
    requested_episode_indices: list[int],
    episode_outcomes: dict[int, bool],
) -> dict:
    success_episodes = [idx for idx in requested_episode_indices if episode_outcomes.get(idx) is True]
    failure_episodes = [idx for idx in requested_episode_indices if episode_outcomes.get(idx) is False]
    completed_episode_indices = success_episodes + failure_episodes
    return {
        "task_suite": suite_name,
        "task_id": task_id,
        "task_description": task_description,
        "successes": len(success_episodes),
        "total_episodes": len(completed_episode_indices),
        "episode_indices": completed_episode_indices,
        "success_episodes": success_episodes,
        "failure_episodes": failure_episodes,
        "start_time": None,
        "duration": 0.0,
        "end_time": None,
    }


def _is_contiguous_episode_range(episode_indices: list[int]) -> bool:
    return bool(episode_indices) and episode_indices == list(range(episode_indices[0], episode_indices[-1] + 1))


def _make_resume_result_tag(base_tag: str, episode_indices: list[int]) -> str:
    if not episode_indices:
        return f"{base_tag}_resume_empty"
    if _is_contiguous_episode_range(episode_indices):
        return f"{base_tag}_resume_ep{episode_indices[0]:04d}_{episode_indices[-1] + 1:04d}"
    return (
        f"{base_tag}_resume_first{episode_indices[0]:04d}"
        f"_last{episode_indices[-1]:04d}_n{len(episode_indices)}"
    )


def _load_dataset_task_descriptions(dataset_root: str, suite: str) -> dict:
    root = Path(dataset_root)
    if not root.is_absolute():
        root = ROOT / root

    task_file = root / _suite_dataset_name(suite) / "meta" / "tasks.jsonl"
    if not task_file.exists():
        raise FileNotFoundError(f"Dataset task metadata not found: {task_file}")

    descriptions = {}
    with open(task_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            task_text = str(item["task"])
            for task_key in _candidate_task_keys(task_text):
                descriptions.setdefault(task_key, task_text)
    return descriptions


def _resolve_task_description(task, dataset_descriptions: dict | None, *,
                              suite_name=None, task_id=None,
                              description_source="mapping", prompt_mapping_path=None) -> str:
    if "_language_" in task.name:
        return task.language
    if description_source == "mapping":
        from inference.prompt_mapping import resolve_training_prompt
        return resolve_training_prompt(task.language, task_name=task.name,
                                       mapping_path=prompt_mapping_path)
    if dataset_descriptions is None:
        return task.language

    for key_source in (task.name, task.language):
        for task_key in _candidate_task_keys(key_source):
            if task_key in dataset_descriptions:
                return dataset_descriptions[task_key]

    print(
        "Warning: failed to map LIBERO benchmark task to dataset description; "
        "using raw benchmark language before templated re-encoding: "
        f"name={task.name!r}, language={task.language!r}"
    )
    return _prompt_clean(task.language)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def _config_from_checkpoint_or_registry(payload, config_name: str):
    cfg = copy.deepcopy(VA_CONFIGS[config_name])
    model_config = payload.get("model_config") if isinstance(payload, dict) else None
    if isinstance(model_config, dict):
        for key, value in model_config.items():
            cfg[key] = copy.deepcopy(value)
        cfg_source = "registry+checkpoint"
    else:
        cfg_source = "registry"
    config_overrides = payload.get("config_overrides") if isinstance(payload, dict) else None
    if isinstance(config_overrides, dict):
        for key, value in config_overrides.items():
            cfg[key] = copy.deepcopy(value)
        cfg_source += "+overrides"
    if config_name == "config_libero_train" and os.environ.get("COPPER_LIBERO_DATASET"):
        dataset_root = Path(os.environ["COPPER_LIBERO_DATASET"]).expanduser()
        cfg.dataset_path = str(dataset_root)
        cfg.empty_emb_path = str(dataset_root / "empty_emb.pt")
        cfg.dino_spatial_norm_path = str(dataset_root / "vjepa_spatial_norm.json")
    # Eval should restore checkpoint weights only; avoid re-initializing from WAN.
    cfg.pretrain_video_backbone = ""
    return cfg, cfg_source


def _resolve_future_semantic_steps(cfg) -> int:
    future_blocks = int(getattr(cfg, "future_blocks"))
    future_pool_size = int(getattr(cfg, "future_pool_size", 1) or 1)
    action_per_frame = int(getattr(cfg, "action_per_frame"))
    teacher_compression_frames = int(getattr(cfg, "teacher_compression_frames"))
    teacher_compression_skip = int(getattr(cfg, "teacher_compression_skip", 1) or 1)
    full_horizon_teacher_coverage = bool(getattr(cfg, "full_horizon_teacher_coverage", False))

    if future_pool_size <= 0:
        raise ValueError(f"future_pool_size must be positive, got {future_pool_size}")
    if teacher_compression_frames <= 0:
        raise ValueError(
            f"teacher_compression_frames must be positive, got {teacher_compression_frames}"
        )
    if action_per_frame % teacher_compression_frames != 0:
        raise ValueError(
            "action_per_frame must be divisible by teacher_compression_frames, "
            f"got {action_per_frame} and {teacher_compression_frames}"
        )

    if full_horizon_teacher_coverage:
        if teacher_compression_frames % teacher_compression_skip != 0:
            raise ValueError(
                "teacher_compression_frames must be divisible by teacher_compression_skip, "
                f"got {teacher_compression_frames} and {teacher_compression_skip}"
            )
        teacher_cache_stride = teacher_compression_frames // teacher_compression_skip
        if action_per_frame % teacher_cache_stride != 0:
            raise ValueError(
                "action_per_frame must be divisible by teacher_cache_stride, "
                f"got {action_per_frame} and {teacher_cache_stride}"
            )
        teacher_samples_per_action_block = action_per_frame // teacher_cache_stride
    else:
        teacher_samples_per_action_block = 1

    num_teacher_samples = future_blocks * teacher_samples_per_action_block
    if num_teacher_samples % future_pool_size != 0:
        raise ValueError(
            "num_teacher_samples must be divisible by future_pool_size, "
            f"got {num_teacher_samples} and {future_pool_size}"
        )
    return num_teacher_samples // future_pool_size


def load_model(ckpt_path: str, config_name: str, device: str, dtype: torch.dtype, use_proprio: bool | None = None):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
    cfg, cfg_source = _config_from_checkpoint_or_registry(payload, config_name)
    if use_proprio is not None:
        cfg.use_proprio = bool(use_proprio)
    elif not hasattr(cfg, "use_proprio"):
        cfg.use_proprio = bool(getattr(cfg, "proprio_dim", 0) or 0)
    if not bool(getattr(cfg, "use_proprio", False)):
        cfg.proprio_dim = 0
    model, model_cfg = build_semantic_mot_model(cfg, device=device, dtype=dtype)
    state_dict = payload.get("state_dict", payload)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device=device, dtype=dtype)
    use_custom_fp32_precision = bool(
        getattr(cfg, "enable_custom_fp32_precision", False)
        or getattr(cfg, "enable_fp32_modules", False)
    )
    if use_custom_fp32_precision:
        _keep_selected_tensors_fp32(model, getattr(model, "_keep_in_fp32_modules", None))
    if hasattr(model, "apply_precision_policy"):
        model.apply_precision_policy()
    model = model.eval()
    step = payload.get("step", "?") if isinstance(payload, dict) else "?"
    print(f"Loaded checkpoint from step {step}: {ckpt_path}")
    print(f"Eval config source: {cfg_source} ({config_name})")
    return model, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Optional T5 text encoder
# ──────────────────────────────────────────────────────────────────────────────

def load_text_encoder(t5_dir: str, device: str, dtype: torch.dtype):
    from transformers import UMT5EncoderModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(t5_dir, subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(t5_dir, torch_dtype=dtype, subfolder="text_encoder").to(device).eval()
    print(f"Loaded T5 encoder from {t5_dir}")
    return tokenizer, encoder


@torch.no_grad()
def encode_text(
    text: str,
    tokenizer,
    encoder,
    device: str,
    max_length: int | None = DEFAULT_TEXT_EMB_MAX_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns embeddings and a real-token mask with zero-padding past real tokens.

    Resize observations to the checkpoint input geometry.
    """
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
        inputs = tokenizer(
            [text],
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
    ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device=device, dtype=torch.bool)
    seq_len = int(mask[0].gt(0).sum().item())
    embeds = encoder(ids, attention_mask=mask).last_hidden_state[0]  # [L, D]
    embeds = embeds[:seq_len]
    if max_length is not None and seq_len < max_length:
        pad = torch.zeros(max_length - seq_len, embeds.shape[1], dtype=embeds.dtype, device=device)
        embeds = torch.cat([embeds, pad], dim=0)
    return embeds.unsqueeze(0), mask[:, : embeds.shape[0]]  # [1, L, D], [1, L]


def _resolve_text_emb_max_tokens(cfg) -> int:
    raw_value = getattr(cfg, "text_emb_max_tokens", DEFAULT_TEXT_EMB_MAX_TOKENS)
    max_tokens = DEFAULT_TEXT_EMB_MAX_TOKENS if raw_value is None else int(raw_value)
    if max_tokens == 0 or max_tokens < -1:
        raise ValueError(f"text_emb_max_tokens must be -1 or a positive integer, got {max_tokens}")
    return max_tokens


def _resolve_text_emb_use_padding_mask(cfg) -> bool:
    raw_value = getattr(cfg, "text_emb_use_padding_mask", True)
    return True if raw_value is None else bool(raw_value)


def append_semantic_neutral_suffix(
    text: str,
    task_id: int,
    enabled: bool,
    base_seed: int | None = None,
) -> tuple[str, str | None]:
    """Append a task-specific courtesy suffix without changing task semantics."""
    if not enabled:
        return text, None
    rng = random.Random(None if base_seed is None else int(base_seed) + int(task_id))
    suffix = rng.choice(SEMANTIC_NEUTRAL_SUFFIXES)
    stripped = text.rstrip()
    if not stripped:
        return suffix, suffix
    if stripped[-1] in ".!?":
        return f"{stripped} {suffix}", suffix
    return f"{stripped}, {suffix}", suffix


# ──────────────────────────────────────────────────────────────────────────────
# Frozen DINO encoder (anchor features for SemanticFastWAM)
# ──────────────────────────────────────────────────────────────────────────────

def load_dino_encoder(model_name: str, device: str, dtype: torch.dtype):
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    print(f"Loaded DINO encoder: {model_name}")
    return model


@torch.no_grad()
def extract_anchor_dino(
    views: list,
    dino_model,
    num_spatial_per_view: int,
    device: str,
    dtype: torch.dtype,
) -> tuple:
    """Extract frozen anchor DINO features from the current observation.

    Args:
        views: list of per-view tensors [1, 3, H, W] in [-1, 1].
        num_spatial_per_view: H//patch * W//patch (e.g. 8*8=64 for 112x112/patch14).

    Returns:
        (anchor_dino_cls, anchor_dino_registers, anchor_dino_spatial) — all float32.
        Shapes: [1, V, D], [1, V*R, D], [1, V*N_s, D]
    """
    mean = torch.tensor(_IMAGENET_MEAN, dtype=dtype, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  dtype=dtype, device=device).view(1, 3, 1, 1)

    cls_list, reg_list, spatial_list = [], [], []
    for view in views:
        x = view.to(device=device, dtype=dtype)
        x = x * 0.5 + 0.5          # [-1, 1] → [0, 1]
        x = (x - mean) / std        # ImageNet normalize
        seq = dino_model(pixel_values=x).last_hidden_state  # [1, 1+R+N_s, D]
        n_extra = seq.shape[1] - num_spatial_per_view
        n_reg   = n_extra - 1       # tokens after CLS that are registers
        cls_list.append(seq[:, :1].float())
        reg_list.append(seq[:, 1 : 1 + n_reg].float())
        spatial_list.append(seq[:, 1 + n_reg :].float())

    return (
        torch.cat(cls_list,     dim=1),   # [1, V,       D]
        torch.cat(reg_list,     dim=1),   # [1, V*R,     D]
        torch.cat(spatial_list, dim=1),   # [1, V*N_s,   D]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Frozen VJEPA encoder (anchor features for SemanticFastWAM with VJEPA teacher)
# ──────────────────────────────────────────────────────────────────────────────

VJEPA_MODEL_CHOICES = [
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant_384",
    "vjepa2_1_vit_gigantic_384",
]


def load_vjepa_encoder(model_name: str, device: str, dtype: torch.dtype):
    model, _predictor = load_vjepa_model(model_name)
    model = model.to(device=device, dtype=torch.float32).eval()
    print(f"Loaded VJEPA encoder: {model_name}")
    return model


def _normalize_vjepa_output(output) -> torch.Tensor:
    """Normalize VJEPA model output to [B, N, D] float32 tensor."""
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
    views: list,
    vjepa_model,
    num_spatial_per_view: int,
    device: str,
    dtype: torch.dtype,
    prev_views: list | None = None,
    temporal_mode: str = "single_frame",
) -> tuple:
    """Extract frozen anchor VJEPA features from the current observation.

    Supports both the current single-frame eval path and the legacy prev-cur
    pair path used by older LIBERO eval runs.

    Args:
        views: list of per-view tensors [1, 3, H, W] in [-1, 1] (current frame).
        num_spatial_per_view: H//patch * W//patch (e.g. 8*8=64 for 128x128/patch16).
        prev_views: optional list of per-view tensors [1, 3, H, W] in [-1, 1]
                    from the previous replan step.
        temporal_mode: "single_frame" or "prev_or_self".

    Returns:
        (cls, registers, spatial) — all float32.
        cls and registers are empty [1, 0, D] since VJEPA has no CLS/register tokens.
        spatial: [1, V*N_s, D]
    """
    mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32, device=device).view(1, 3, 1, 1)

    temporal_mode = str(temporal_mode or "single_frame").strip().lower()
    if temporal_mode not in {"single_frame", "prev_or_self"}:
        raise ValueError(
            f"Unsupported V-JEPA anchor temporal_mode={temporal_mode!r}. "
            "Expected 'single_frame' or 'prev_or_self'."
        )
    if prev_views is not None and len(prev_views) != len(views):
        raise ValueError(
            f"prev_views must align with views: got {len(prev_views)} previous views for {len(views)} current views."
        )

    spatial_list = []
    for i, view in enumerate(views):
        x_cur = view.to(device=device, dtype=torch.float32)
        x_cur = x_cur * 0.5 + 0.5          # [-1, 1] → [0, 1]
        x_cur = (x_cur - mean) / std        # ImageNet normalize
        if temporal_mode == "single_frame":
            video_input = x_cur.unsqueeze(2)
        else:
            if prev_views is not None:
                x_prev = prev_views[i].to(device=device, dtype=torch.float32)
                x_prev = x_prev * 0.5 + 0.5
                x_prev = (x_prev - mean) / std
            else:
                x_prev = x_cur
            video_input = torch.stack([x_prev, x_cur], dim=2)

        with torch.inference_mode():
            if x_cur.device.type == "cuda" and dtype != torch.float32:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = vjepa_model(video_input)
            elif x_cur.device.type == "cpu" and dtype == torch.bfloat16:
                with torch.autocast(device_type="cpu", dtype=dtype):
                    out = vjepa_model(video_input)
            else:
                out = vjepa_model(video_input)
        seq = _normalize_vjepa_output(out).detach()
        # seq: [1, N_s, D] where N_s = num_spatial_per_view
        if seq.shape[1] != num_spatial_per_view:
            actual_per_view = seq.shape[1]
            raise ValueError(
                f"VJEPA output spatial tokens mismatch: expected {num_spatial_per_view} "
                f"per view (from per_view_image_size/patch_size), got {actual_per_view}. "
                f"Check that the input image size matches the VJEPA model's expected resolution."
            )
        spatial_list.append(seq)

    spatial = torch.cat(spatial_list, dim=1)   # [1, V*N_s, D]
    empty = torch.empty(1, 0, spatial.shape[-1], dtype=torch.float32, device=spatial.device)
    return empty, empty, spatial


def _resolve_vjepa_anchor_temporal_mode(cfg, cli_mode: str | None) -> str:
    if cli_mode is not None:
        mode = str(cli_mode).strip().lower()
    else:
        mode = str(getattr(cfg, "eval_vjepa_anchor_temporal_mode", "single_frame") or "single_frame").strip().lower()
    if mode not in {"single_frame", "prev_or_self"}:
        raise ValueError(
            f"Unsupported eval_vjepa_anchor_temporal_mode={mode!r}. "
            "Expected 'single_frame' or 'prev_or_self'."
        )
    return mode


def _resolve_vjepa_anchor_prev_offset(cfg, cli_offset: int | None) -> int:
    raw = getattr(cfg, "eval_vjepa_anchor_prev_offset", 1) if cli_offset is None else cli_offset
    offset = int(raw)
    if offset <= 0:
        raise ValueError(f"V-JEPA anchor prev offset must be positive, got {offset}")
    return offset


# ──────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _center_crop_resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(img)
    sw, sh = pil.size
    scale = max(w / sw, h / sh)
    resized = pil.resize((round(sw * scale), round(sh * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - w) // 2, 0)
    top = max((rh - h) // 2, 0)
    return np.asarray(resized.crop((left, top, left + w, top + h)), dtype=np.uint8)


def _orient_libero_image(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "vertical":
        return np.ascontiguousarray(img[::-1])
    if mode == "vertical_horizontal":
        return np.ascontiguousarray(img[::-1, ::-1])
    if mode == "none":
        return np.ascontiguousarray(img)
    raise ValueError(f"Unsupported image flip mode: {mode}")


def extract_obs(obs: dict, image_flip: str = "vertical_horizontal") -> dict:
    agentview = _orient_libero_image(obs["agentview_image"], image_flip)
    wrist = _orient_libero_image(obs["robot0_eye_in_hand_image"], image_flip)
    return {"image": agentview, "wrist_image": wrist}


def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternion to axis-angle (3,) used by LIBERO proprio."""
    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {tuple(q.shape)}")
    w = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * np.float32(2.0 * math.acos(w) / den)).astype(np.float32)


def extract_proprio(obs: dict, proprio_dim: int, device: str, dtype: torch.dtype):
    """Build [1, proprio_dim] proprio tensor from env observation.

    Training uses LIBERO's observation.state:
    [robot0_eef_pos(3), quat2axisangle(robot0_eef_quat)(3), gripper(2)].
    Gripper convention is handled before normalization.
    """
    if proprio_dim <= 0:
        return None

    state = None
    if "observation.state" in obs:
        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
    elif all(k in obs for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")):
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        eef_aa = _quat_to_axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if gripper.shape[0] == 1 and proprio_dim == 8:
            gripper = np.repeat(gripper, 2)
        state = np.concatenate([eef_pos, eef_aa, gripper], axis=0).astype(np.float32, copy=False)
    elif "state" in obs:
        raw_state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if raw_state.ndim == 1:
            if raw_state.shape[0] == 7 and proprio_dim == 8:
                raw_state = np.concatenate([raw_state[:6], np.repeat(raw_state[6:7], 2)], axis=0)
            state = raw_state

    if state is None:
        keys_preview = sorted(obs.keys())
        raise KeyError(
            "Cannot extract proprio from observation. Expected one of: "
            "`observation.state`, `state`, or (`robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`). "
            f"Available keys: {keys_preview}"
        )

    if state.shape[0] != proprio_dim:
        raise ValueError(
            f"Extracted proprio dim {state.shape[0]} does not match proprio_dim={proprio_dim}. "
            "Expected [eef_pos(3), axis_angle(3), gripper_qpos(2)] exactly."
        )

    return torch.from_numpy(state).to(device=device, dtype=dtype).unsqueeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Proprio normalization (checkpoint proprio normalization)
# ──────────────────────────────────────────────────────────────────────────────

def _load_proprio_norm_stats(dataset_path: str, suite: str | None = None) -> dict:
    """Load shared proprio stats for observation.state from dataset root only."""
    import json as _json

    del suite
    base = Path(dataset_path)
    if not base.is_absolute():
        base = ROOT / base

    global_stats_file = base / "global_stats.json"
    if not global_stats_file.exists():
        raise FileNotFoundError(
            "Shared proprio stats are required for evaluation. "
            f"Expected {global_stats_file}."
        )
    payload = _json.loads(global_stats_file.read_text())
    field_stats = payload.get("observation.state")
    if field_stats is None:
        raise KeyError(f"Missing observation.state in {global_stats_file}")
    return {"observation.state": field_stats}


def _load_action_norm_stats(dataset_path: str) -> dict:
    import json as _json

    base = Path(dataset_path)
    if not base.is_absolute():
        base = ROOT / base

    global_stats_file = base / "global_stats.json"
    if not global_stats_file.exists():
        raise FileNotFoundError(
            "Shared action stats are required when cfg.norm_stat does not contain the requested mode. "
            f"Expected {global_stats_file}."
        )
    payload = _json.loads(global_stats_file.read_text())
    field_stats = payload.get("action")
    if field_stats is None:
        raise KeyError(f"Missing action in {global_stats_file}")
    return field_stats


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


def build_proprio_normalizer(
    cfg,
    suite: str | None = None,
) -> dict | None:
    """Return a normalizer dict (from checkpoint normalization metadata) or None."""
    if not bool(getattr(cfg, "use_proprio", False)) or int(getattr(cfg, "proprio_dim", 0) or 0) <= 0:
        return None
    mode = str(getattr(cfg, "proprio_norm_mode", "none") or "none").lower()
    if mode in {"none", "false", "off", "disabled"}:
        return None
    if mode not in {"z-score", "min/max"}:
        raise ValueError(
            f"Unsupported proprio_norm_mode={mode!r} in eval. "
            "Only 'z-score', 'min/max', and 'none' are allowed for proprio normalization."
        )

    embedded = (getattr(cfg, "checkpoint_normalization", None) or {}).get("proprio")
    if embedded is not None:
        if str(embedded["mode"]).lower() != mode:
            raise ValueError("Checkpoint proprio normalization mode differs from model config")
        return {
            "mode": mode,
            "center": torch.as_tensor(embedded["center"], dtype=torch.float32),
            "scale": torch.as_tensor(embedded["scale"], dtype=torch.float32),
            "clip": float(embedded.get("clip", getattr(cfg, "proprio_norm_clip", 5.0))),
            "suite": suite,
            "zscore_exclude_dims": tuple(int(x) for x in embedded.get("zscore_exclude_dims", ())),
        }

    if not hasattr(cfg, "dataset_path"):
        raise ValueError("proprio_norm_mode is set but cfg has no dataset_path")

    stats = _load_proprio_norm_stats(cfg.dataset_path, suite=suite)
    field_stats = stats["observation.state"]
    if mode == "z-score":
        center = torch.as_tensor(field_stats["mean"], dtype=torch.float32)
        scale = torch.as_tensor(field_stats["std"], dtype=torch.float32)
    else:
        center = torch.as_tensor(field_stats["min"], dtype=torch.float32)
        scale = torch.as_tensor(field_stats["max"], dtype=torch.float32)
    clip = float(getattr(cfg, "proprio_norm_clip", 5.0))

    return {
        "mode": mode,
        "center": center,
        "scale": scale,
        "clip": clip,
        "suite": suite,
        "zscore_exclude_dims": tuple(int(x) for x in (getattr(cfg, "proprio_zscore_exclude_dims", ()) or ())),
    }


def _format_float_list(values, ndigits: int = 4) -> str:
    return "[" + ",".join(f"{float(v):.{ndigits}f}" for v in values) + "]"




def print_eval_proprio(
    tag: str,
    obs: dict,
    proprio_dim: int,
    device: str,
    dtype: torch.dtype,
    normalizer: dict | None,
) -> None:
    proprio_raw = extract_proprio(obs, proprio_dim, device, dtype)
    proprio = normalize_proprio(proprio_raw, normalizer)
    prop_raw_g = proprio_raw[0, -2:].detach().cpu().float().numpy()
    prop_norm_g = proprio[0, -2:].detach().cpu().float().numpy()
    print(
        f"[debug-proprio] {tag} "
        f"prop_grip_raw=[{prop_raw_g[0]:.4f},{prop_raw_g[1]:.4f}] "
        f"prop_grip_norm=[{prop_norm_g[0]:.3f},{prop_norm_g[1]:.3f}]"
    )


def normalize_proprio(proprio: "torch.Tensor", normalizer: dict | None) -> "torch.Tensor":
    """Apply proprio normalization matching training."""
    if normalizer is None:
        return proprio

    mode = normalizer["mode"]
    center = normalizer["center"].to(device=proprio.device, dtype=proprio.dtype)
    scale = normalizer["scale"].to(device=proprio.device, dtype=proprio.dtype)
    clip = normalizer["clip"]
    zscore_exclude_dims = tuple(int(x) for x in normalizer.get("zscore_exclude_dims", ()))
    proprio = proprio.clone()

    if mode == "z-score":
        out = (proprio - center) / (scale + 1e-8)
        if zscore_exclude_dims:
            exclude_idx = torch.as_tensor(zscore_exclude_dims, device=proprio.device, dtype=torch.long)
            out[..., exclude_idx] = proprio[..., exclude_idx]
    else:
        input_min, input_max = center, scale
        input_range = input_max - input_min
        valid_range = input_range >= 1e-4
        safe_range = torch.where(valid_range, input_range, torch.ones_like(input_range) * 2.0)
        out = (proprio - input_min) * (2.0 / safe_range) - 1.0
        out = torch.where(valid_range, out, proprio - input_min)

    return torch.clamp(out, -clip, clip)


def obs_to_model_input(
    obs_dict: dict,
    per_view_size: tuple,
    device: str,
    dtype: torch.dtype,
) -> list:
    """Convert raw obs dict to list of per-view tensors [1, 3, H, W] in [-1, 1]."""
    h, w = per_view_size
    views = []
    for key in ("image", "wrist_image"):
        img = _center_crop_resize(obs_dict[key], h, w)
        x = torch.tensor(img, dtype=dtype).permute(2, 0, 1).unsqueeze(0).to(device)
        x = x * (2.0 / 255.0) - 1.0
        views.append(x)
    return views


# ──────────────────────────────────────────────────────────────────────────────
# Action denormalization
# ──────────────────────────────────────────────────────────────────────────────

def build_action_denormalizer(cfg, clip_norm: float | None = 5.0):
    """Return a function that converts normalized model output to 7-DOF EEF actions."""
    embedded = (getattr(cfg, "checkpoint_normalization", None) or {}).get("action")
    if embedded is not None:
        mode = str(embedded["mode"]).lower()
        if mode not in {"z-score", "min/max", "q01/q99"}:
            raise ValueError(f"Unsupported embedded action normalization mode: {mode}")
        center = np.asarray(embedded["center"], dtype=np.float32)
        scale = np.asarray(embedded["scale"], dtype=np.float32)
        zscore_exclude_dims = tuple(int(x) for x in embedded.get("zscore_exclude_dims", ()))
        affine_scale, affine_offset = _build_action_affine_params(mode, center, scale)

        def denormalize(action_norm):
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
        if not hasattr(cfg, "dataset_path"):
            raise
        merged = _load_action_norm_stats(cfg.dataset_path)
        merged.update(norm_stat)
        norm_stat = merged
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
    affine_scale, affine_offset = _build_action_affine_params(mode, center, scale)

    def denormalize(action_norm: np.ndarray) -> np.ndarray:
        # action_norm: [horizon, action_dim=7]
        if clip_norm is not None:
            action_norm = np.clip(action_norm, -float(clip_norm), float(clip_norm))
        physical = (action_norm - affine_offset) / affine_scale
        if mode == "z-score" and zscore_exclude_dims:
            physical[..., list(zscore_exclude_dims)] = action_norm[..., list(zscore_exclude_dims)]
        return physical                           # [horizon, 7]

    return denormalize


def apply_gripper_transform(
    actions: np.ndarray,
    mode: str,
    binarize_gripper: bool = True,
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    """Convert denormalized training actions to the gripper convention used by env.step."""
    if mode == "none":
        out = actions.copy()
        if binarize_gripper:
            out[:, -1] = np.where(out[:, -1] > gripper_threshold, 1.0, 0.0)
        return out
    elif mode in {"open_minus1_close1", "legacy_1_minus_2x"}:
        out = actions.copy()
        gripper_open = out[:, -1].copy()
        out[:, -1] = 1.0 - 2.0 * gripper_open
        if binarize_gripper:
            out[:, -1] = np.where(gripper_open > gripper_threshold, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported gripper transform: {mode}")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Single-step inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_action_chunk(
    model,
    views: list,
    anchor_dino_cls,
    anchor_dino_spatial,
    anchor_dino_registers,
    context,
    context_mask,
    proprio,
    num_inference_steps: int,
    seed: int | None,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    joint_future_denoising: bool = False,
    future_semantic_steps: int | None = None,
) -> np.ndarray:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=torch.device(device)).manual_seed(int(seed))
    autocast_enabled = device != "cpu" and dtype in (torch.float16, torch.bfloat16)
    with torch.autocast(device_type="cuda" if device != "cpu" else "cpu", dtype=dtype, enabled=autocast_enabled):
        action = model.infer_action(
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
        )  # [1, horizon, action_dim]
    return action[0].cpu().float().numpy()  # [horizon, action_dim]


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_env(task, retries: int = 5) -> OffScreenRenderEnv:
    from libero.libero import get_libero_path
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(bddl),
        "camera_heights": LIBERO_RENDER_RES,
        "camera_widths": LIBERO_RENDER_RES,
    }
    for attempt in range(retries):
        try:
            env = OffScreenRenderEnv(**env_args)
            return env
        except Exception as e:
            print(f"Env creation attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(3)
    raise RuntimeError(f"Failed to create env after {retries} attempts")


def save_video(frames: list, path: Path, fps: int = 15):
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    video_writer = imageio.get_writer(str(path), fps=fps)
    for f in frames:
        if isinstance(f, dict):
            row = np.concatenate([f["image"], f["wrist_image"]], axis=1)
        else:
            row = np.asarray(f)
        video_writer.append_data(row.astype(np.uint8))
    video_writer.close()


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(
    env,
    init_state,
    task_description: str,
    model,
    cfg,
    context,
    context_mask,
    *,
    per_view_size: tuple,
    device: str,
    dtype: torch.dtype,
    num_inference_steps: int,
    replan_steps: int,
    max_steps: int,
    num_steps_wait: int,
    denormalize,
    gripper_transform: str,
    binarize_gripper: bool,
    gripper_threshold: float,
    debug_actions: bool,
    seed: int | None,
    episode_idx: int,
    task_id: int | None = None,
    dino_model,
    vjepa_model,
    teacher_type: str,
    num_spatial_per_view: int,
    proprio_normalizer: dict | None = None,
    zero_proprio: bool = False,
    joint_future_denoising: bool = False,
    future_semantic_steps: int | None = None,
    vjepa_anchor_temporal_mode: str = "single_frame",
    vjepa_anchor_prev_offset: int = 1,
    inference_lock=None,
    action_client=None,
    suite_name: str | None = None,
) -> tuple:
    env.reset()
    obs = env.set_init_state(init_state)

    replay_frames = []
    obs_history: list[dict] = []
    pending: list = []
    done = False
    t = 0
    proprio_dim = int(getattr(cfg, "proprio_dim", 0) or 0)

    if debug_actions and proprio_dim > 0:
        print_eval_proprio(
            f"ep={episode_idx} after_set_init_state",
            obs,
            proprio_dim,
            device,
            dtype,
            proprio_normalizer,
        )

    while t < max_steps + num_steps_wait:
        if t < num_steps_wait:
            warmup_action = [0.0] * 7
            if gripper_transform in {"open_minus1_close1", "legacy_1_minus_2x"}:
                warmup_action[-1] = -1.0
            obs, _, done, _ = env.step(warmup_action)
            if debug_actions and proprio_dim > 0:
                print_eval_proprio(
                    f"ep={episode_idx} warmup_t={t + 1}",
                    obs,
                    proprio_dim,
                    device,
                    dtype,
                    proprio_normalizer,
                )
            t += 1
            continue

        obs_dict = extract_obs(obs)
        replay_frames.append(obs_dict)
        obs_history.append(obs_dict)

        if not pending:
            if action_client is not None:
                prev_obs_dict = None
                if vjepa_anchor_temporal_mode == "prev_or_self" and len(obs_history) > vjepa_anchor_prev_offset:
                    prev_obs_dict = obs_history[-1 - vjepa_anchor_prev_offset]
                physical = action_client.predict(
                    task_description=task_description,
                    task_id=task_id if task_id is not None else episode_idx,
                    suite_name=suite_name,
                    obs=obs,
                    prev_obs_dict=prev_obs_dict,
                )
                pending = physical[:replan_steps].tolist()
                action = pending.pop(0)
                obs, _, done, _ = env.step(action)
                t += 1
                if done:
                    break
                continue
            views = obs_to_model_input(obs_dict, per_view_size, device, dtype)
            proprio_raw = extract_proprio(obs, proprio_dim, device, dtype)
            proprio = normalize_proprio(proprio_raw, proprio_normalizer)
            if zero_proprio and proprio is not None:
                proprio = torch.zeros_like(proprio)
            # Multiple env clients share one GPU-resident model. Serialize the
            # complete anchor + denoise request so every call stays batch size 1.
            lock_context = inference_lock if inference_lock is not None else contextlib.nullcontext()
            with lock_context:
                if teacher_type == "vjepa":
                    prev_views = None
                    if vjepa_anchor_temporal_mode == "prev_or_self" and len(obs_history) > vjepa_anchor_prev_offset:
                        prev_obs_dict = obs_history[-1 - vjepa_anchor_prev_offset]
                        prev_views = obs_to_model_input(prev_obs_dict, per_view_size, device, dtype)
                    anchor_dino_cls, anchor_dino_registers, anchor_dino_spatial = extract_anchor_vjepa(
                        views, vjepa_model, num_spatial_per_view, device, dtype,
                        prev_views=prev_views, temporal_mode=vjepa_anchor_temporal_mode,
                    )
                else:
                    anchor_dino_cls, anchor_dino_registers, anchor_dino_spatial = extract_anchor_dino(
                        views, dino_model, num_spatial_per_view, device, dtype
                    )
                chunk_norm = predict_action_chunk(
                    model, views, anchor_dino_cls, anchor_dino_spatial, anchor_dino_registers,
                    context, context_mask, proprio, num_inference_steps, seed, device,
                    dtype=dtype, joint_future_denoising=joint_future_denoising,
                    future_semantic_steps=future_semantic_steps,
                )
            physical_raw = denormalize(chunk_norm)  # [horizon, 7], same convention as training data/server.
            physical = apply_gripper_transform(
                physical_raw,
                gripper_transform,
                binarize_gripper=binarize_gripper,
                gripper_threshold=gripper_threshold,
            )
            if debug_actions:
                raw_xyz = physical_raw[:replan_steps, :3]
                env_xyz = physical[:replan_steps, :3]
                raw_rpy = physical_raw[:replan_steps, 3:6]
                raw_g = physical_raw[:replan_steps, -1]
                env_g = physical[:replan_steps, -1]
                prop_raw_g = proprio_raw[0, -2:].detach().cpu().float().numpy()
                prop_norm_g = proprio[0, -2:].detach().cpu().float().numpy()
                raw_g_seq = ",".join(f"{v:.3f}" for v in raw_g.tolist())
                env_g_seq = ",".join(f"{v:.3f}" for v in env_g.tolist())
                print(
                    f"[debug-actions] ep={episode_idx} t={t} "
                    f"prop_grip_raw=[{prop_raw_g[0]:.4f},{prop_raw_g[1]:.4f}] "
                    f"prop_grip_norm=[{prop_norm_g[0]:.3f},{prop_norm_g[1]:.3f}] "
                    f"raw_xyz[min/mean/max]={raw_xyz.min():.3f}/{raw_xyz.mean():.3f}/{raw_xyz.max():.3f} "
                    f"env_xyz[min/mean/max]={env_xyz.min():.3f}/{env_xyz.mean():.3f}/{env_xyz.max():.3f} "
                    f"raw_rpy[min/mean/max]={raw_rpy.min():.3f}/{raw_rpy.mean():.3f}/{raw_rpy.max():.3f} "
                    f"raw_gripper[min/mean/max]={raw_g.min():.3f}/{raw_g.mean():.3f}/{raw_g.max():.3f} "
                    f"env_gripper[min/mean/max]={env_g.min():.3f}/{env_g.mean():.3f}/{env_g.max():.3f} "
                    f"raw_g_seq=[{raw_g_seq}] env_g_seq=[{env_g_seq}]"
                )
            pending = physical[:replan_steps].tolist()

        action = pending.pop(0)
        obs, _, done, _ = env.step(action)
        t += 1

        if done:
            break

    return bool(done), replay_frames


# ──────────────────────────────────────────────────────────────────────────────
# Task runner
# ──────────────────────────────────────────────────────────────────────────────

def run_task(
    task,
    init_states,
    task_id: int,
    suite_name: str,
    task_description: str,
    model,
    cfg,
    tokenizer,
    text_encoder,
    dino_model,
    vjepa_model,
    teacher_type: str,
    num_spatial_per_view: int,
    *,
    per_view_size: tuple,
    device: str,
    dtype: torch.dtype,
    num_trials: int,
    num_inference_steps: int,
    replan_steps: int,
    max_steps: int,
    num_steps_wait: int,
    denormalize,
    gripper_transform: str,
    binarize_gripper: bool,
    gripper_threshold: float,
    debug_actions: bool,
    seed: int | None,
    video_dir: Path,
    save_video_flag: bool,
    proprio_normalizer: dict | None = None,
    zero_proprio: bool = False,
    joint_future_denoising: bool = False,
    future_semantic_steps: int | None = None,
    vjepa_anchor_temporal_mode: str = "single_frame",
    vjepa_anchor_prev_offset: int = 1,
    lang_random_suffix: bool = False,
    lang_suffix_seed: int | None = None,
    episode_indices: list[int] | None = None,
    inference_lock=None,
    action_client=None,
) -> dict:
    instruction_template = getattr(cfg, "instruction_template", FASTWAM_INSTRUCTION_TEMPLATE)

    # Build text context (reuse same embedding for all trials)
    if action_client is None and (tokenizer is None or text_encoder is None):
        raise RuntimeError(
            "Text encoder is required but not loaded. Pass --t5-dir or use --no-text to disable."
        )
    instruction_text = build_instruction_text(task_description, instruction_template)
    instruction_text, _ = append_semantic_neutral_suffix(
        instruction_text,
        task_id=task_id,
        enabled=lang_random_suffix,
        base_seed=lang_suffix_seed,
    )
    text_emb_max_tokens = _resolve_text_emb_max_tokens(cfg)
    use_padding_mask = _resolve_text_emb_use_padding_mask(cfg)
    if text_emb_max_tokens > 0:
        encode_max_length: int | None = text_emb_max_tokens
    elif use_padding_mask:
        encode_max_length = None
    else:
        # Preserve legacy "all tokens valid" semantics when text padding is intentionally unmasked.
        encode_max_length = DEFAULT_TEXT_EMB_STORAGE_LENGTH
    lock_context = inference_lock if inference_lock is not None else contextlib.nullcontext()
    if action_client is None:
        with lock_context:
            context, text_token_mask = encode_text(
                instruction_text, tokenizer, text_encoder, device, max_length=encode_max_length,
            )
        context = context.to(dtype=dtype)
        if use_padding_mask:
            context_mask = text_token_mask[:, : context.shape[1]]
        else:
            context_mask = torch.ones(context.shape[:2], device=context.device, dtype=torch.bool)
    else:
        context = context_mask = None

    env = make_env(task)

    selected_episode_indices = (
        list(range(num_trials)) if episode_indices is None else list(episode_indices)
    )

    result = {
        "task_suite": suite_name,
        "task_id": task_id,
        "task_description": task_description,
        "successes": 0,
        "total_episodes": len(selected_episode_indices),
        "episode_indices": selected_episode_indices,
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0.0,
    }
    t0 = time.time()

    for trial_idx in selected_episode_indices:
        init_state = init_states[trial_idx]
        success, frames = run_episode(
            env=env,
            init_state=init_state,
            task_description=task_description,
            model=model,
            cfg=cfg,
            context=context,
            context_mask=context_mask,
            per_view_size=per_view_size,
            device=device,
            dtype=dtype,
            num_inference_steps=num_inference_steps,
            replan_steps=replan_steps,
            max_steps=max_steps,
            num_steps_wait=num_steps_wait,
            denormalize=denormalize,
            gripper_transform=gripper_transform,
            binarize_gripper=binarize_gripper,
            gripper_threshold=gripper_threshold,
            debug_actions=debug_actions,
            seed=seed,
            task_id=task_id,
            episode_idx=trial_idx,
            dino_model=dino_model,
            vjepa_model=vjepa_model,
            teacher_type=teacher_type,
            num_spatial_per_view=num_spatial_per_view,
            proprio_normalizer=proprio_normalizer,
            zero_proprio=zero_proprio,
            joint_future_denoising=joint_future_denoising,
            future_semantic_steps=future_semantic_steps,
            vjepa_anchor_temporal_mode=vjepa_anchor_temporal_mode,
            vjepa_anchor_prev_offset=vjepa_anchor_prev_offset,
            inference_lock=inference_lock,
            action_client=action_client,
            suite_name=suite_name,
        )
        if success:
            result["successes"] += 1
            result["success_episodes"].append(trial_idx)
        else:
            result["failure_episodes"].append(trial_idx)

        tag = task_description.lower().replace(" ", "_")[:40]

        if save_video_flag and frames:
            fname = f"{DATE_TIME}--task{task_id}--trial{trial_idx}--{tag}--{'succ' if success else 'fail'}.mp4"
            save_video(frames, video_dir / fname)

    env.close()
    result["duration"] = time.time() - t0
    result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def configure_compiled_inference(model, anchor_encoder, args, device):
    """Compile tensor stages while preserving the existing sampling loop.

    Disable CUDA Graph replay: independently compiled encoder/prefill/denoising
    stages pass outputs to one another and must not reuse graph-owned buffers.
    """
    enabled = not args.no_compile and str(device).startswith("cuda")
    if enabled:
        def compile_stage(fn):
            return torch.compile(fn, fullgraph=False, dynamic=False,
                                 options={"triton.cudagraphs": False})
        for attr, fn in (
            ("_compactor_infer_fn", getattr(model, "compactor", None)),
            ("_current_spatial_dino_encoder_infer_fn", getattr(model, "current_spatial_dino_encoder", None)),
            ("_video_prefill_fn", model.mot.prefill_video_cache_flat),
            ("_action_denoise_step_fn", model._action_denoise_step),
        ):
            if fn is not None:
                setattr(model, attr, compile_stage(fn))
        if anchor_encoder is not None:
            anchor_encoder = compile_stage(anchor_encoder)
    print(f"Compile tensor stages: {enabled}; CUDA Graph replay: disabled", flush=True)
    settings_path = Path(args.out_dir) / args.suite / "evaluation_settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({
        "compile": enabled, "cudagraphs": False,
        "num_inference_steps": args.num_inference_steps,
        "replan_steps": args.replan_steps if args.replan_steps > 0 else int(model.config.action_per_frame),
        "seed": args.seed, "num_trials": args.num_trials,
        "description_source": args.description_source,
        "prompt_mapping": str(args.prompt_map) if args.description_source == "mapping" else None,
    }, indent=2))
    return anchor_encoder


def create_persistent_runtime(args):
    """Load the GPU-resident policy components once for a persistent worker."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model, cfg = load_model(args.ckpt, args.config_name, device, dtype, use_proprio=args.use_proprio)
    action_clip_norm = None if args.no_action_clip else float(args.action_clip_norm)
    denormalize = build_action_denormalizer(cfg, clip_norm=action_clip_norm)
    per_view_size = tuple(cfg.per_view_image_size)
    action_horizon = int(cfg.future_blocks) * int(cfg.action_per_frame)
    replan_steps = args.replan_steps if args.replan_steps > 0 else int(cfg.action_per_frame)
    future_semantic_steps = _resolve_future_semantic_steps(cfg)
    vjepa_anchor_temporal_mode = _resolve_vjepa_anchor_temporal_mode(cfg, args.vjepa_anchor_temporal_mode)
    vjepa_anchor_prev_offset = _resolve_vjepa_anchor_prev_offset(cfg, args.vjepa_anchor_prev_offset)
    resolved_joint_future_denoising = (
        bool(getattr(cfg, "default_joint_future_denoising", False))
        if args.joint_future_denoising is None
        else bool(args.joint_future_denoising)
    )
    patch_size = int(cfg.image_patch_size)
    num_spatial_per_view = (per_view_size[0] // patch_size) * (per_view_size[1] // patch_size)
    teacher_type = str(getattr(cfg, "teacher_type", "dino") or "dino").lower()
    spatial_grid_size = int(num_spatial_per_view ** 0.5)
    if spatial_grid_size * spatial_grid_size != num_spatial_per_view:
        raise ValueError(f"num_spatial_per_view={num_spatial_per_view} is not a perfect square")
    encoder_image_size = per_view_size if teacher_type == "vjepa" else (
        spatial_grid_size * DINO_PATCH_SIZE,
        spatial_grid_size * DINO_PATCH_SIZE,
    )
    if args.no_text or not args.t5_dir:
        raise ValueError("Persistent evaluation requires --t5-dir and text conditioning")
    tokenizer, text_encoder = load_text_encoder(args.t5_dir, device, dtype)
    if teacher_type == "vjepa":
        dino_model = None
        vjepa_model = load_vjepa_encoder(args.vjepa_model, device, dtype)
    else:
        dino_model = load_dino_encoder(args.dino_model, device, dtype)
        vjepa_model = None
    compiled_anchor = configure_compiled_inference(model, vjepa_model or dino_model, args, device)
    if teacher_type == "vjepa":
        vjepa_model = compiled_anchor
    else:
        dino_model = compiled_anchor
    return {
        "device": device, "dtype": dtype, "model": model, "cfg": cfg,
        "denormalize": denormalize, "replan_steps": replan_steps,
        "action_horizon": action_horizon, "future_semantic_steps": future_semantic_steps,
        "vjepa_anchor_temporal_mode": vjepa_anchor_temporal_mode,
        "vjepa_anchor_prev_offset": vjepa_anchor_prev_offset,
        "resolved_joint_future_denoising": resolved_joint_future_denoising,
        "num_spatial_per_view": num_spatial_per_view, "teacher_type": teacher_type,
        "encoder_image_size": encoder_image_size, "tokenizer": tokenizer,
        "text_encoder": text_encoder, "dino_model": dino_model, "vjepa_model": vjepa_model,
        "suite_cache": {},
    }


@torch.no_grad()
def predict_persistent_action(runtime, args, request):
    """Run one batch-1 policy request for an out-of-process environment client."""
    cfg = runtime["cfg"]
    task_description = request["task_description"]
    task_id = int(request["task_id"])
    suite_name = request["suite_name"]
    context_cache = runtime.setdefault("remote_text_cache", {})
    cache_key = (suite_name, task_id, task_description)
    if cache_key not in context_cache:
        text = build_instruction_text(task_description, getattr(cfg, "instruction_template", FASTWAM_INSTRUCTION_TEMPLATE))
        text, _ = append_semantic_neutral_suffix(
            text, task_id=task_id, enabled=args.lang_random_suffix, base_seed=args.lang_suffix_seed,
        )
        max_tokens = _resolve_text_emb_max_tokens(cfg)
        use_padding_mask = _resolve_text_emb_use_padding_mask(cfg)
        max_length = max_tokens if max_tokens > 0 else (None if use_padding_mask else DEFAULT_TEXT_EMB_STORAGE_LENGTH)
        context, token_mask = encode_text(text, runtime["tokenizer"], runtime["text_encoder"], runtime["device"], max_length=max_length)
        context = context.to(dtype=runtime["dtype"])
        context_mask = token_mask[:, :context.shape[1]] if use_padding_mask else torch.ones(
            context.shape[:2], device=context.device, dtype=torch.bool
        )
        context_cache[cache_key] = (context, context_mask)
    context, context_mask = context_cache[cache_key]

    obs = request["obs"]
    obs_dict = extract_obs(obs)
    views = obs_to_model_input(obs_dict, runtime["encoder_image_size"], runtime["device"], runtime["dtype"])
    proprio_dim = int(getattr(cfg, "proprio_dim", 0) or 0)
    proprio_cache = runtime.setdefault("remote_proprio_cache", {})
    if suite_name not in proprio_cache:
        proprio_cache[suite_name] = build_proprio_normalizer(cfg, suite=suite_name)
    proprio = normalize_proprio(
        extract_proprio(obs, proprio_dim, runtime["device"], runtime["dtype"]),
        proprio_cache[suite_name],
    )
    if args.zero_proprio and proprio is not None:
        proprio = torch.zeros_like(proprio)
    if runtime["teacher_type"] == "vjepa":
        prev_views = None
        prev_obs_dict = request.get("prev_obs_dict")
        if runtime["vjepa_anchor_temporal_mode"] == "prev_or_self" and prev_obs_dict is not None:
            prev_views = obs_to_model_input(prev_obs_dict, runtime["encoder_image_size"], runtime["device"], runtime["dtype"])
        cls, registers, spatial = extract_anchor_vjepa(
            views, runtime["vjepa_model"], runtime["num_spatial_per_view"], runtime["device"], runtime["dtype"],
            prev_views=prev_views, temporal_mode=runtime["vjepa_anchor_temporal_mode"],
        )
    else:
        cls, registers, spatial = extract_anchor_dino(
            views, runtime["dino_model"], runtime["num_spatial_per_view"], runtime["device"], runtime["dtype"]
        )
    chunk_norm = predict_action_chunk(
        runtime["model"], views, cls, spatial, registers, context, context_mask, proprio,
        args.num_inference_steps, args.seed, runtime["device"], dtype=runtime["dtype"],
        joint_future_denoising=runtime["resolved_joint_future_denoising"],
        future_semantic_steps=runtime["future_semantic_steps"],
    )
    physical = apply_gripper_transform(
        runtime["denormalize"](chunk_norm), args.gripper_transform,
        binarize_gripper=args.binarize_gripper, gripper_threshold=args.gripper_threshold,
    )
    return physical


def run_persistent_task(runtime, args, suite_name: str, task_id: int, result_tag: str, inference_lock=None, action_client=None):
    """Evaluate one task with an already loaded model and encoder stack."""
    cache = runtime["suite_cache"]
    if suite_name not in cache:
        task_suite = benchmark.get_benchmark_dict()[suite_name]()
        descriptions = None
        if args.description_source == "dataset":
            descriptions = _load_dataset_task_descriptions(args.dataset_task_root, suite_name)
        proprio_normalizer = build_proprio_normalizer(runtime["cfg"], suite=suite_name)
        suite_dir = Path(args.out_dir) / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)
        video_dir = suite_dir / "videos"
        if args.save_video:
            video_dir.mkdir(parents=True, exist_ok=True)
        cache[suite_name] = (task_suite, descriptions, proprio_normalizer, suite_dir, video_dir)
    task_suite, descriptions, proprio_normalizer, suite_dir, video_dir = cache[suite_name]
    if getattr(args, "task_output_root", None):
        suite_dir = Path(args.task_output_root) / f"{suite_name}_task{task_id}" / suite_name
        suite_dir.mkdir(parents=True, exist_ok=True)
        video_dir = suite_dir / "videos"
        if args.save_video:
            video_dir.mkdir(parents=True, exist_ok=True)
    if task_id < 0 or task_id >= task_suite.n_tasks:
        raise ValueError(f"Invalid task {suite_name}:{task_id}")
    episode_start, episode_end = (0, int(args.num_trials)) if args.episode_range is None else args.episode_range
    if episode_start < 0 or episode_end > int(args.num_trials) or episode_start >= episode_end:
        raise ValueError(f"Invalid episode range [{episode_start}, {episode_end})")
    episode_indices = list(range(episode_start, episode_end))
    task = task_suite.get_task(task_id)
    task_description = _resolve_task_description(
        task, descriptions, suite_name=suite_name, task_id=task_id,
        description_source=args.description_source, prompt_mapping_path=args.prompt_map)
    existing = _load_existing_episode_outcomes(suite_dir).get(task_id, {}) if args.resume else {}
    requested = [idx for idx in episode_indices if idx not in existing]
    if not requested:
        print(f"Task {suite_name}:{task_id}: already complete, skip")
        return {"skipped": True}
    init_states = task_suite.get_task_init_states(task_id)
    while len(init_states) < episode_end:
        init_states = np.concatenate([init_states, init_states], axis=0)
    result = run_task(
        task=task, init_states=init_states, task_id=task_id, suite_name=suite_name,
        task_description=task_description, model=runtime.get("model"), cfg=runtime["cfg"],
        tokenizer=runtime.get("tokenizer"), text_encoder=runtime.get("text_encoder"),
        dino_model=runtime.get("dino_model"), vjepa_model=runtime.get("vjepa_model"),
        teacher_type=runtime.get("teacher_type", "dino"), num_spatial_per_view=runtime.get("num_spatial_per_view", 0),
        per_view_size=runtime.get("encoder_image_size", (128, 128)), device=runtime.get("device", "cpu"), dtype=runtime.get("dtype", torch.float32),
        num_trials=args.num_trials, num_inference_steps=args.num_inference_steps,
        replan_steps=runtime["replan_steps"], max_steps=SUITE_MAX_STEPS[suite_name],
        num_steps_wait=args.num_steps_wait, denormalize=runtime["denormalize"],
        gripper_transform=args.gripper_transform, binarize_gripper=args.binarize_gripper,
        gripper_threshold=args.gripper_threshold, debug_actions=args.debug_actions, seed=args.seed,
        video_dir=video_dir, save_video_flag=args.save_video, proprio_normalizer=proprio_normalizer,
        zero_proprio=args.zero_proprio, joint_future_denoising=runtime["resolved_joint_future_denoising"],
        future_semantic_steps=runtime["future_semantic_steps"],
        vjepa_anchor_temporal_mode=runtime["vjepa_anchor_temporal_mode"],
        vjepa_anchor_prev_offset=runtime["vjepa_anchor_prev_offset"],
        lang_random_suffix=args.lang_random_suffix, lang_suffix_seed=args.lang_suffix_seed,
        episode_indices=requested,
        inference_lock=inference_lock,
        action_client=action_client,
    )
    parts_dir = suite_dir / "_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    with open(parts_dir / f"{result_tag}_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Task {suite_name}:{task_id}: {result['successes']}/{result['total_episodes']} | {result['duration']:.0f}s")
    return result


def run_evaluation(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model, cfg = load_model(args.ckpt, args.config_name, device, dtype, use_proprio=args.use_proprio)
    action_clip_norm = None if args.no_action_clip else float(args.action_clip_norm)
    denormalize = build_action_denormalizer(cfg, clip_norm=action_clip_norm)
    print(f"Action normalized clip: {action_clip_norm}")
    proprio_normalizer = build_proprio_normalizer(
        cfg,
        suite=args.suite,
    )
    if bool(getattr(cfg, "use_proprio", False)) and int(getattr(cfg, "proprio_dim", 0) or 0) > 0:
        print(f"Proprio input: enabled (dim={int(cfg.proprio_dim)})")
    else:
        print("Proprio input: disabled")
    if args.zero_proprio:
        print("Eval proprio mode: zero normalized proprio (keeps proprio token, keeps the checkpoint proprio token)")
    if proprio_normalizer is not None:
        print(
            "Proprio normalization: "
            f"{proprio_normalizer['mode']} suite={proprio_normalizer['suite']} "
            f"(clip={proprio_normalizer['clip']})"
        )


    per_view_size = tuple(cfg.per_view_image_size)
    action_horizon = int(cfg.future_blocks) * int(cfg.action_per_frame)
    # Auto mode: execute only a short prefix of each predicted chunk before replanning.
    # This is much less jerky than executing the full horizon in one open-loop rollout.
    replan_steps = args.replan_steps if args.replan_steps > 0 else int(cfg.action_per_frame)
    future_pool_size = int(getattr(cfg, "future_pool_size", 1) or 1)
    future_semantic_steps = _resolve_future_semantic_steps(cfg)
    vjepa_anchor_temporal_mode = _resolve_vjepa_anchor_temporal_mode(cfg, args.vjepa_anchor_temporal_mode)
    vjepa_anchor_prev_offset = _resolve_vjepa_anchor_prev_offset(cfg, args.vjepa_anchor_prev_offset)
    resolved_joint_future_denoising = (
        bool(getattr(cfg, "default_joint_future_denoising", False))
        if args.joint_future_denoising is None
        else bool(args.joint_future_denoising)
    )

    patch_size = int(cfg.image_patch_size)
    num_spatial_per_view = (per_view_size[0] // patch_size) * (per_view_size[1] // patch_size)

    # Detect teacher type from config (set by _apply_teacher_mode_overrides).
    teacher_type = str(getattr(cfg, "teacher_type", "dino") or "dino").lower()

    # Image size for anchor extraction. For DINO the input is resized to match
    # DINO_PATCH_SIZE=14 grid size; for VJEPA we use per_view_image_size directly
    # since VJEPA uses the same patch size as the model config (16×16).
    spatial_grid_size = int(num_spatial_per_view ** 0.5)
    if spatial_grid_size * spatial_grid_size != num_spatial_per_view:
        raise ValueError(
            f"num_spatial_per_view={num_spatial_per_view} is not a perfect square. "
            f"per_view_image_size={per_view_size}, image_patch_size={patch_size}"
        )
    if teacher_type == "vjepa":
        encoder_image_size = per_view_size  # 128×128 for VJEPA
    else:
        encoder_image_size = (spatial_grid_size * DINO_PATCH_SIZE, spatial_grid_size * DINO_PATCH_SIZE)

    if args.no_text:
        raise SystemExit("--no-text is not supported: this model requires text conditioning. Provide --t5-dir.")
    if not args.t5_dir:
        raise SystemExit("--t5-dir is required. Provide the path to the pretrained model (e.g. pretrained_weights/text_encoder/wan22_ti2v_5b).")
    tokenizer, text_encoder = load_text_encoder(args.t5_dir, device, dtype)

    # Load anchor encoder based on teacher type.
    if teacher_type == "vjepa":
        dino_model = None
        vjepa_model = load_vjepa_encoder(args.vjepa_model, device, dtype)
    else:
        dino_model = load_dino_encoder(args.dino_model, device, dtype)
        vjepa_model = None

    compiled_anchor = configure_compiled_inference(model, vjepa_model or dino_model, args, device)
    if teacher_type == "vjepa":
        vjepa_model = compiled_anchor
    else:
        dino_model = compiled_anchor
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.suite]()
    num_tasks = task_suite.n_tasks
    dataset_descriptions = None
    if args.description_source == "dataset":
        dataset_descriptions = _load_dataset_task_descriptions(
            dataset_root=args.dataset_task_root,
            suite=args.suite,
        )

    task_start, task_end = (0, num_tasks) if args.task_range is None else tuple(args.task_range)
    task_end = min(task_end, num_tasks)
    max_steps = SUITE_MAX_STEPS[args.suite]
    if args.episode_range is None:
        episode_start = 0
        episode_end = int(args.num_trials)
    else:
        episode_start, episode_end = tuple(args.episode_range)
    if episode_start < 0 or episode_end > int(args.num_trials) or episode_start >= episode_end:
        raise ValueError(
            f"Invalid episode range [{episode_start}, {episode_end}) for num_trials={args.num_trials}"
        )
    episode_indices = list(range(episode_start, episode_end))

    out_dir = Path(args.out_dir)
    suite_dir = out_dir / args.suite
    suite_dir.mkdir(parents=True, exist_ok=True)
    video_dir = suite_dir / "videos"
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== LIBERO evaluation: {args.suite}, tasks [{task_start}, {task_end}) ===")
    print(f"  checkpoint        : {args.ckpt}")
    print(f"  config            : {args.config_name}")
    print(f"  mot structure     : {getattr(cfg, 'mot_structure_mode', 'fastwam')}")
    print(f"  trials/task       : {args.num_trials}")
    print(f"  episode range     : [{episode_start}, {episode_end})")
    print(f"  replan_steps      : {replan_steps} / {action_horizon}")
    print(f"  infer_steps       : {args.num_inference_steps}")
    print(f"  seed              : {args.seed}")
    print(f"  lang rnd suffix   : {args.lang_random_suffix}")
    print(f"  lang suffix seed  : {args.lang_suffix_seed}")
    print(f"  text cond         : {'yes' if tokenizer is not None else 'no'}")
    print(f"  teacher type      : {teacher_type}")
    if teacher_type == "vjepa":
        print(f"  vjepa model       : {args.vjepa_model}")
        print(
            f"  vjepa anchor mode : {vjepa_anchor_temporal_mode}"
            + (
                f" (prev_offset={vjepa_anchor_prev_offset})"
                if vjepa_anchor_temporal_mode == "prev_or_self"
                else ""
            )
        )
    else:
        print(f"  dino model        : {args.dino_model}")
    print(f"  spatial/view      : {num_spatial_per_view}")
    print(f"  encoder image size: {encoder_image_size}")
    print(f"  binarize gripper  : {args.binarize_gripper}")
    print(f"  gripper threshold : {args.gripper_threshold}")
    print(
        f"  joint future      : {resolved_joint_future_denoising} "
        f"(semantic_steps={future_semantic_steps}, pool={future_pool_size})"
    )
    print(f"  resume            : {args.resume}")

    suite_episode_outcomes = _load_existing_episode_outcomes(suite_dir) if args.resume else {}
    overall_succ = 0
    overall_total = 0
    for task_id in tqdm(range(task_start, task_end), desc=args.suite):
        task = task_suite.get_task(task_id)
        task_description = _resolve_task_description(
            task, dataset_descriptions, suite_name=args.suite, task_id=task_id,
            description_source=args.description_source, prompt_mapping_path=args.prompt_map)
        requested_episode_indices = list(episode_indices)
        existing_episode_outcomes = {}
        existing_result_summary = None
        if args.resume:
            existing_episode_outcomes = suite_episode_outcomes.get(int(task_id), {})
            completed_episode_indices = [
                idx for idx in requested_episode_indices if idx in existing_episode_outcomes
            ]
            if completed_episode_indices:
                existing_result_summary = _summarize_existing_episodes(
                    suite_name=args.suite,
                    task_id=task_id,
                    task_description=task_description,
                    requested_episode_indices=requested_episode_indices,
                    episode_outcomes=existing_episode_outcomes,
                )
                if len(completed_episode_indices) == len(requested_episode_indices):
                    overall_succ += existing_result_summary["successes"]
                    overall_total += existing_result_summary["total_episodes"]
                    rate = existing_result_summary["successes"] / max(existing_result_summary["total_episodes"], 1) * 100
                    print(
                        f"  Task {task_id}: skip existing results for "
                        f"{len(completed_episode_indices)}/{len(requested_episode_indices)} episodes "
                        f"({rate:.1f}% from JSON)"
                    )
                    continue
                overall_succ += existing_result_summary["successes"]
                overall_total += existing_result_summary["total_episodes"]
                requested_episode_indices = [
                    idx for idx in requested_episode_indices if idx not in existing_episode_outcomes
                ]
                print(
                    f"  Task {task_id}: resume from JSON, "
                    f"skip {len(completed_episode_indices)} completed episodes, "
                    f"run remaining {len(requested_episode_indices)}"
                )

        init_states = task_suite.get_task_init_states(task_id)
        while len(init_states) < episode_end:
            init_states = np.concatenate([init_states, init_states], axis=0)

        result = run_task(
            task=task,
            init_states=init_states,
            task_id=task_id,
            suite_name=args.suite,
            task_description=task_description,
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            dino_model=dino_model,
            vjepa_model=vjepa_model,
            teacher_type=teacher_type,
            num_spatial_per_view=num_spatial_per_view,
            per_view_size=encoder_image_size,
            device=device,
            dtype=dtype,
            num_trials=args.num_trials,
            num_inference_steps=args.num_inference_steps,
            replan_steps=replan_steps,
            max_steps=max_steps,
            num_steps_wait=args.num_steps_wait,
            denormalize=denormalize,
            gripper_transform=args.gripper_transform,
            binarize_gripper=args.binarize_gripper,
            gripper_threshold=args.gripper_threshold,
            debug_actions=args.debug_actions,
            seed=args.seed,
            video_dir=video_dir,
            save_video_flag=args.save_video,
            proprio_normalizer=proprio_normalizer,
            zero_proprio=args.zero_proprio,
            joint_future_denoising=resolved_joint_future_denoising,
            future_semantic_steps=future_semantic_steps,
            vjepa_anchor_temporal_mode=vjepa_anchor_temporal_mode,
            vjepa_anchor_prev_offset=vjepa_anchor_prev_offset,
            lang_random_suffix=args.lang_random_suffix,
            lang_suffix_seed=args.lang_suffix_seed,
            episode_indices=requested_episode_indices,
        )
        overall_succ += result["successes"]
        overall_total += result["total_episodes"]

        result_tag = args.result_tag or f"gpu0_task{task_id}"
        if args.resume and len(requested_episode_indices) != len(episode_indices):
            result_tag = _make_resume_result_tag(result_tag, requested_episode_indices)
        parts_dir = suite_dir / "_parts"
        parts_dir.mkdir(parents=True, exist_ok=True)
        out_file = parts_dir / f"{result_tag}_results.json"
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        if existing_result_summary is not None:
            combined_successes = existing_result_summary["successes"] + result["successes"]
            combined_total = existing_result_summary["total_episodes"] + result["total_episodes"]
            rate = combined_successes / combined_total * 100
            print(
                f"  Task {task_id}: new {result['successes']}/{result['total_episodes']} "
                f"| combined {combined_successes}/{combined_total} ({rate:.1f}%) | {result['duration']:.0f}s"
            )
        else:
            rate = result["successes"] / result["total_episodes"] * 100
            print(
                f"  Task {task_id}: {result['successes']}/{result['total_episodes']} "
                f"({rate:.1f}%) | {result['duration']:.0f}s"
            )

    if overall_total > 0:
        print(f"\nOverall: {overall_succ}/{overall_total} ({overall_succ / overall_total * 100:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate SemanticFastWAM on LIBERO")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path (.pt)")
    parser.add_argument(
        "--config-name", default="config_libero_train",
        help="Config key in VA_CONFIGS (default: config_libero_train)"
    )
    parser.add_argument(
        "--suite", default="libero_spatial", choices=ALL_SUITES,
        help="LIBERO task suite"
    )
    parser.add_argument(
        "--task-range", type=int, nargs=2, default=None, metavar=("START", "END"),
        help="[start, end) task index range; default: all tasks in suite"
    )
    parser.add_argument(
        "--episode-range", type=int, nargs=2, default=None, metavar=("START", "END"),
        help="[start, end) episode index range within each task; default: all episodes in [0, num_trials)",
    )
    parser.add_argument(
        "--description-source",
        choices=["mapping", "dataset", "benchmark"],
        default="mapping",
        help=(
            "Where to read task descriptions from. 'mapping' uses the bundled "
            "benchmark-to-training prompt map, falling back to benchmark language. "
            "'dataset' uses <dataset-task-root>/<suite>_no_noops_lerobot/meta/tasks.jsonl "
            "and maps by task name; 'benchmark' uses LIBERO task.language."
        ),
    )
    parser.add_argument("--prompt-map", default=str(ROOT / "inference/libero_prompt_mapping.json"),
                        help="Portable task-to-training prompt map used by --description-source=mapping")
    parser.add_argument(
        "--dataset-task-root",
        default=os.environ.get("COPPER_LIBERO_DATASET", "lerobot/datasets_libero"),
        help="Root containing per-suite LeRobot metadata used when --description-source=dataset.",
    )
    parser.add_argument("--num-trials", type=int, default=50, help="Episodes per task")
    parser.add_argument("--num-inference-steps", type=int, default=10, help="Flow matching steps")
    parser.add_argument("--no-compile", action="store_true", help="Disable compilation of inference tensor stages")
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Optional fixed action-noise seed for each replan, matching FastWAM-style deterministic inference",
    )
    parser.add_argument(
        "--replan-steps", type=int, default=10,
        help="Actions to execute per inference call (0 = auto = action_per_frame)"
    )
    parser.add_argument("--num-steps-wait", type=int, default=30, help="Warm-up no-op steps")
    parser.add_argument(
        "--gripper-transform",
        choices=["open_minus1_close1", "none", "legacy_1_minus_2x"],
        default="open_minus1_close1",
        help=(
            "Post-denormalization gripper conversion before env.step. "
            "'open_minus1_close1' maps training action 1=open/0=close to LIBERO env -1=open/+1=close; "
            "'none' leaves the denormalized training-data convention unchanged; "
            "'legacy_1_minus_2x' is an alias for open_minus1_close1."
        ),
    )
    parser.add_argument(
        "--no-binarize-gripper",
        dest="binarize_gripper",
        action="store_false",
        help="Disable default binary gripper execution after converting to the env convention.",
    )
    parser.set_defaults(binarize_gripper=True)
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for binary gripper execution. For open_minus1_close1 this is applied "
            "to the denormalized training gripper open value before converting to LIBERO env "
            "(-1=open, +1=close)."
        ),
    )
    parser.add_argument("--debug-actions", action="store_true", help="Print gripper stats for each planned chunk")
    joint_future_group = parser.add_mutually_exclusive_group()
    joint_future_group.add_argument(
        "--joint-future-denoising",
        dest="joint_future_denoising",
        action="store_true",
        help=(
            "Denoise pooled future semantic queries together with actions during eval, "
            "matching the training joint-token count instead of using the cached anchor-only fast path."
        ),
    )
    joint_future_group.add_argument(
        "--no-joint-future-denoising",
        dest="joint_future_denoising",
        action="store_false",
        help="Force cached FastWAM-style action-only inference even if the config default is joint.",
    )
    parser.set_defaults(joint_future_denoising=None)
    parser.add_argument(
        "--action-clip-norm",
        type=float,
        default=5.0,
        help="Clip normalized action outputs before denormalization, matching training target clipping.",
    )
    parser.add_argument("--no-action-clip", action="store_true", help="Disable normalized action clipping")
    parser.add_argument("--out-dir", default="outputs/eval", help="Output directory")
    parser.add_argument(
        "--result-tag",
        default=None,
        help="Optional output filename stem written as <out-dir>/<suite>/_parts/<result-tag>_results.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing *_results.json under <out-dir>/<suite>. "
            "Completed episode_indices are skipped; partial tasks only run missing episodes."
        ),
    )
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    proprio_group = parser.add_mutually_exclusive_group()
    proprio_group.add_argument(
        "--use-proprio",
        dest="use_proprio",
        action="store_true",
        help="Force-enable proprio input token at eval time (override config).",
    )
    proprio_group.add_argument(
        "--no-proprio",
        dest="use_proprio",
        action="store_false",
        help="Force-disable proprio input token at eval time (override config).",
    )
    parser.set_defaults(use_proprio=None)
    parser.add_argument(
        "--zero-proprio",
        action="store_true",
        help=(
            "Keep the proprio token enabled, but replace the normalized proprio vector with zeros. "
            "Keep the proprio token while supplying zeros."
        ),
    )
    parser.add_argument(
        "--t5-dir", default=None,
        help="Path to T5 text encoder dir (e.g. pretrained_weights/text_encoder/wan22_ti2v_5b)"
    )
    parser.add_argument("--no-text", action="store_true", help="Disable text conditioning")
    parser.add_argument(
        "--dino-model", default="facebook/dinov2-with-registers-large",
        help="HuggingFace model name for frozen DINO anchor encoder (default: facebook/dinov2-with-registers-large)"
    )
    parser.add_argument(
        "--vjepa-model",
        default="vjepa2_1_vit_large_384",
        choices=VJEPA_MODEL_CHOICES,
        help="torch.hub model name for frozen VJEPA anchor encoder (default: vjepa2_1_vit_large_384)",
    )
    parser.add_argument(
        "--vjepa-anchor-temporal-mode",
        choices=["single_frame", "prev_or_self"],
        default=None,
        help=(
            "Override the LIBERO eval V-JEPA current-anchor temporal mode. "
            "'single_frame' uses the current frame only; "
            "'prev_or_self' forms a 2-frame pair using a previous replan frame when available, "
            "and falls back to a self-pair at the start of an episode."
        ),
    )
    parser.add_argument(
        "--vjepa-anchor-prev-offset",
        type=int,
        default=None,
        help=(
            "When --vjepa-anchor-temporal-mode=prev_or_self, use the observation from this many "
            "replan steps earlier as the previous frame. Default is 1."
        ),
    )
    parser.add_argument("--save-video", action="store_true", help="Save rollout videos")
    parser.add_argument(
        "--lang-random-suffix",
        action="store_true",
        help=(
            "Append one semantic-neutral courtesy suffix to the instruction text during eval. "
            "Examples include 'thank you' and 'many thanks'."
        ),
    )
    parser.add_argument(
        "--lang-suffix-seed",
        type=int,
        default=None,
        help="Optional base seed used to pick a task-specific courtesy suffix.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if benchmark is None or OffScreenRenderEnv is None:
        raise RuntimeError("LIBERO import failed; check simulator dependencies and PYTHONPATH. "
                           f"Original error: {LIBERO_IMPORT_ERROR}") from LIBERO_IMPORT_ERROR
    run_evaluation(args)


if __name__ == "__main__":
    main()
