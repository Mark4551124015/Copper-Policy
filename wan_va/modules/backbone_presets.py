"""Pretrained encoder identities, dimensions, and local cache locations.

Add a new entry with its real output width before selecting a different model.
Cached teacher latents must be regenerated when the world encoder changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = ROOT / "pretrained_weights"
WAN_SOURCE_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"


@dataclass(frozen=True)
class WorldEncoderPreset:
    hub_entry: str
    output_dim: int
    checkpoint_name: str
    checkpoint_url: str
    checkpoint_key: str = "ema_encoder"
    source_repo: str = "https://github.com/facebookresearch/vjepa2.git"


@dataclass(frozen=True)
class VisionEncoderPreset:
    model_id: str
    output_dim: int


@dataclass(frozen=True)
class TextEncoderPreset:
    model_id: str
    output_dim: int


WORLD_ENCODER_PRESETS = {
    "vjepa2_1_vit_large_384": WorldEncoderPreset(
        hub_entry="vjepa2_1_vit_large_384",
        output_dim=1024,
        checkpoint_name="vjepa2_1_vitl_dist_vitG_384.pt",
        checkpoint_url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
    ),
}

VISION_ENCODER_PRESETS = {
    "dinov2_with_registers_large": VisionEncoderPreset(
        model_id="facebook/dinov2-with-registers-large",
        output_dim=1024,
    ),
}

TEXT_ENCODER_PRESETS = {
    "wan22_ti2v_5b": TextEncoderPreset(
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        output_dim=4096,
    ),
}


def world_cache_dir(name: str) -> Path:
    if name not in WORLD_ENCODER_PRESETS:
        raise ValueError(f"Unknown world encoder {name!r}; choices: {tuple(WORLD_ENCODER_PRESETS)}")
    return CACHE_ROOT / "world_encoder" / name


def vision_cache_dir(name: str) -> Path:
    if name not in VISION_ENCODER_PRESETS:
        raise ValueError(f"Unknown vision encoder {name!r}; choices: {tuple(VISION_ENCODER_PRESETS)}")
    return CACHE_ROOT / "vision_encoder" / name


def text_cache_dir(name: str) -> Path:
    if name not in TEXT_ENCODER_PRESETS:
        raise ValueError(f"Unknown text encoder {name!r}; choices: {tuple(TEXT_ENCODER_PRESETS)}")
    return CACHE_ROOT / "text_encoder" / name


def wan_source_dir() -> Path:
    return CACHE_ROOT / "WAN22" / "transformer"


def world_cache_ready(name: str) -> bool:
    preset = WORLD_ENCODER_PRESETS[name]
    root = world_cache_dir(name)
    return (root / "source" / "hubconf.py").is_file() and (root / preset.checkpoint_name).is_file()


def vision_cache_ready(name: str) -> bool:
    root = vision_cache_dir(name)
    return (root / "config.json").is_file() and any(
        (root / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
    )


def text_cache_ready(name: str) -> bool:
    root = text_cache_dir(name)
    tokenizer = root / "tokenizer"
    encoder = root / "text_encoder"
    if not (tokenizer / "tokenizer_config.json").is_file() or not (encoder / "config.json").is_file():
        return False
    index = encoder / "model.safetensors.index.json"
    if index.is_file():
        import json

        shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
        return all((encoder / shard).is_file() for shard in shards)
    return (encoder / "model.safetensors").is_file()


def wan_source_ready() -> bool:
    root = wan_source_dir()
    index = root / "diffusion_pytorch_model.safetensors.index.json"
    if not index.is_file():
        return (root / "diffusion_pytorch_model.safetensors").is_file()
    import json

    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    return all((root / shard).is_file() for shard in shards)


__all__ = [
    "WORLD_ENCODER_PRESETS", "VISION_ENCODER_PRESETS", "TEXT_ENCODER_PRESETS",
    "world_cache_dir", "vision_cache_dir", "text_cache_dir",
    "world_cache_ready", "vision_cache_ready", "text_cache_ready",
    "WAN_SOURCE_MODEL_ID", "wan_source_dir", "wan_source_ready",
]
