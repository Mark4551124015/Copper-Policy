from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SemanticMoTPreset:
    name: str
    approx_params_m: int
    hidden_dim: int
    action_hidden_dim: int
    num_layers: int
    num_heads: int
    head_dim: int
    mlp_ratio: float
    num_register_tokens: int
    action_mlp_ratio: Optional[float] = None


PRESET_REGISTRY: Dict[str, SemanticMoTPreset] = {
    # real params ~500M
    "small": SemanticMoTPreset(
        "small",
        500,
        1280,  # hidden_dim (video expert)
        768,   # action_hidden_dim
        12,    # num_layers
        10,    # num_heads
        128,   # head_dim
        4.0,
        4,
    ),

    # Budgeted WAN-1.3B-style preset: keep the shared attention space at
    # 12 x 128 and trim FFNs, which are intentionally not pretrained.
    "big": SemanticMoTPreset(
        "big",
        990,
        1536,  # hidden_dim (video expert)
        768,   # action_hidden_dim
        16,    # num_layers
        12,    # num_heads
        128,   # head_dim
        4.0,
        4,
        action_mlp_ratio=8.0,
    ),

    "good_params": SemanticMoTPreset(
        "good_params",
        940 + 860,
        1280,  # hidden_dim (video expert), matches WAN 5B width exactly
        1280,  # action_hidden_dim
        30,    # num_layers, reduced to stay near the 2B budget
        10,    # num_heads
        128,   # head_dim
        14336.0 / 3072.0,  # WAN-5B FFN ratio
        4,
        action_mlp_ratio=4.0,
    ),
    # real params ~1.5B
    "large": SemanticMoTPreset(
        "large",
        1500,
        1280,  # hidden_dim (video expert)
        1024,  # action_hidden_dim
        28,    # num_layers
        10,    # num_heads
        128,   # head_dim
        4.0,
        4,
    ),
    "fastWAM": SemanticMoTPreset(
        "fastWAM",
        6000,
        3072,  # hidden_dim (video expert)
        1024,  # action_hidden_dim
        30,    # num_layers
        24,    # num_heads
        128,   # head_dim
        14336.0 / 3072.0,
        4,
    ),
}


__all__ = ["SemanticMoTPreset", "PRESET_REGISTRY"]
