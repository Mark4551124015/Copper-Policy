from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

from .backbone_presets import WORLD_ENCODER_PRESETS, world_cache_dir, world_cache_ready


_DEFAULT_VJEPA_REPO = "facebookresearch/vjepa2"
_DEFAULT_VJEPA_CACHE_DIRNAME = "facebookresearch_vjepa2_main"


def _candidate_local_repo_paths() -> tuple[Path, ...]:
    paths: list[Path] = []

    env_override = os.environ.get("VJEPA2_HUB_DIR")
    if env_override:
        paths.append(Path(env_override).expanduser())

    try:
        torch_hub_dir = Path(torch.hub.get_dir()).expanduser()
        paths.append(torch_hub_dir / _DEFAULT_VJEPA_CACHE_DIRNAME)
    except Exception:
        pass

    paths.append(Path.home() / ".cache" / "torch" / "hub" / _DEFAULT_VJEPA_CACHE_DIRNAME)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return tuple(deduped)


def resolve_local_vjepa_repo() -> Optional[Path]:
    for repo_path in _candidate_local_repo_paths():
        if repo_path.is_dir() and (repo_path / "hubconf.py").is_file():
            return repo_path
    return None


def load_vjepa_model(model_name: str):
    if model_name in WORLD_ENCODER_PRESETS:
        if not world_cache_ready(model_name):
            raise FileNotFoundError(
                f"V-JEPA cache missing at {world_cache_dir(model_name)}. "
                f"Run 'python -m tools.cache_backbones world {model_name}' first."
            )
        preset = WORLD_ENCODER_PRESETS[model_name]
        cache = world_cache_dir(model_name)
        encoder, predictor = torch.hub.load(str(cache / "source"), preset.hub_entry, source="local", pretrained=False)
        payload = torch.load(cache / preset.checkpoint_name, map_location="cpu", mmap=True, weights_only=False)
        if not isinstance(payload, dict) or preset.checkpoint_key not in payload or "predictor" not in payload:
            raise ValueError(f"Invalid V-JEPA checkpoint in {cache / preset.checkpoint_name}")

        def clean_keys(state):
            return {key.removeprefix("module.").removeprefix("backbone."): value for key, value in state.items()}

        encoder.load_state_dict(clean_keys(payload[preset.checkpoint_key]), strict=True)
        predictor.load_state_dict(clean_keys(payload["predictor"]), strict=True)
        return encoder, predictor
    local_repo = resolve_local_vjepa_repo()
    if local_repo is not None:
        return torch.hub.load(str(local_repo), model_name, source="local")
    return torch.hub.load(_DEFAULT_VJEPA_REPO, model_name)


__all__ = ["load_vjepa_model", "resolve_local_vjepa_repo"]
