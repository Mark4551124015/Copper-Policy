from __future__ import annotations
from typing import Any, Mapping, Optional
import torch
from wan_va.modules.semantic_mot_model import SemanticMoTConfig
from wan_va.modules.compact_semantic_mot_model import CompactSemanticFastWAM

def _cfg_get(cfg: Any, key: str) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, None)
    return getattr(cfg, key, None)

def _cfg_require(cfg: Any, key: str) -> Any:
    value = _cfg_get(cfg, key)
    if value is None:
        raise KeyError(f"Missing required config key: {key}")
    return value

def _cfg_get_int_tuple(cfg: Any, key: str) -> Optional[tuple[int, ...]]:
    raw = _cfg_get(cfg, key)
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return tuple(int(v) for v in raw)
    return (int(raw),)

def _normalize_per_view_image_sizes(cfg_like: Any) -> tuple[tuple[int, int], ...]:
    raw_sizes = _cfg_get(cfg_like, "per_view_image_sizes")
    num_views = int(_cfg_require(cfg_like, "num_views"))
    if raw_sizes is None:
        image_h, image_w = tuple(_cfg_require(cfg_like, "per_view_image_size"))
        return tuple((int(image_h), int(image_w)) for _ in range(num_views))
    view_sizes = tuple((int(h), int(w)) for h, w in raw_sizes)
    if len(view_sizes) != num_views:
        raise ValueError(
            "per_view_image_sizes must provide one (H,W) pair per view, "
            f"got {len(view_sizes)} for num_views={num_views}"
        )
    return view_sizes

def build_semantic_mot_config(cfg_like: Any) -> SemanticMoTConfig:
    compact_latent_enabled = bool(_cfg_get(cfg_like, "compact_latent_enabled") or False)
    if not compact_latent_enabled:
        raise ValueError(
            "Legacy dense/cls semantic model construction is disabled. "
            "Set compact_latent_enabled=True and use the compact model path."
        )
    per_view_image_sizes = _normalize_per_view_image_sizes(cfg_like)
    per_view_image_size = tuple(per_view_image_sizes[0])
    image_patch_size = _cfg_require(cfg_like, "image_patch_size")
    if isinstance(image_patch_size, (tuple, list)):
        raise TypeError(
            f"image_patch_size must be a scalar integer for the semantic model, got {image_patch_size}"
        )

    config = SemanticMoTConfig(
        per_view_image_size=per_view_image_size,
        per_view_image_sizes=per_view_image_sizes,
        num_views=int(_cfg_require(cfg_like, "num_views")),
        image_patch_size=int(image_patch_size),
        teacher_dim=int(_cfg_require(cfg_like, "teacher_dim")),
        text_dim=int(_cfg_require(cfg_like, "text_dim")),
        action_dim=int(_cfg_require(cfg_like, "action_dim")),
        used_action_channel_ids=(
            tuple(int(i) for i in _cfg_get(cfg_like, "used_action_channel_ids"))
            if _cfg_get(cfg_like, "used_action_channel_ids") is not None
            else None
        ),
        action_per_frame=int(_cfg_require(cfg_like, "action_per_frame")),
        future_blocks=int(_cfg_require(cfg_like, "future_blocks")),
        model_size=str(_cfg_require(cfg_like, "model_size")),
        enable_future_dense=False,
        enable_future_cls=False,
        keep_registers=bool(_cfg_require(cfg_like, "keep_registers")),
        dropout=float(_cfg_require(cfg_like, "dropout")),
        eps=float(_cfg_require(cfg_like, "eps")),
        flow_eps=float(_cfg_require(cfg_like, "flow_eps")),
        use_gradient_checkpointing=False,
        enable_fp32_modules=bool(_cfg_get(cfg_like, "enable_fp32_modules") is not False),
        future_pool_size=int(_cfg_get(cfg_like, "future_pool_size") or 1),
        future_semantic_steps=_cfg_get(cfg_like, "future_semantic_steps"),
        action_frame_ratio=_cfg_get(cfg_like, "action_frame_ratio"),
        use_runtime_current_vjepa_anchor=bool(_cfg_get(cfg_like, "use_runtime_current_vjepa_anchor") or False),
        vjepa_model_name_or_path=(
            _cfg_get(cfg_like, "vjepa_model_name_or_path")
            or _cfg_get(cfg_like, "current_vjepa_model_name_or_path")
            or _cfg_get(cfg_like, "future_vjepa_model_name_or_path")
        ),
        hidden_dim=_cfg_get(cfg_like, "hidden_dim"),
        num_layers=_cfg_get(cfg_like, "num_layers"),
        num_heads=_cfg_get(cfg_like, "num_heads"),
        head_dim=_cfg_get(cfg_like, "head_dim"),
        mlp_ratio=_cfg_get(cfg_like, "mlp_ratio"),
        num_register_tokens=_cfg_get(cfg_like, "num_register_tokens"),
        dino_spatial_mean=(
            tuple(float(v) for v in _cfg_get(cfg_like, "dino_spatial_mean"))
            if _cfg_get(cfg_like, "dino_spatial_mean") is not None
            else None
        ),
        dino_spatial_global_scale=_cfg_get(cfg_like, "dino_spatial_global_scale"),
        teacher_latent_norm_enabled=bool(_cfg_get(cfg_like, "teacher_latent_norm_enabled") is not False),
        action_hidden_dim=_cfg_get(cfg_like, "action_hidden_dim"),
        proprio_dim=(
            _cfg_get(cfg_like, "proprio_dim")
            if bool(_cfg_get(cfg_like, "use_proprio")) and _cfg_get(cfg_like, "proprio_dim") is not None
            else 0
        ),
        video_context_uses_proprio=bool(_cfg_get(cfg_like, "video_context_uses_proprio") or False),
        use_view_embed=(
            bool(_cfg_get(cfg_like, "use_view_embed"))
            if _cfg_get(cfg_like, "use_view_embed") is not None
            else True
        ),
        view_rope_layout=str(_cfg_get(cfg_like, "view_rope_layout") or "horizontal"),
        compact_view_rope_layout=_cfg_get(cfg_like, "compact_view_rope_layout"),
        # Compact latent config
        compact_latent_enabled=True,
        compact_encoder_type=str(_cfg_get(cfg_like, "compact_encoder_type") or "vjepa"),
        compact_encoder_dim=int(_cfg_get(cfg_like, "compact_encoder_dim") or 1024),
        compact_use_cached=bool(_cfg_get(cfg_like, "compact_use_cached") is not False),
        compact_num_tokens=int(_cfg_get(cfg_like, "compact_num_tokens") or 8),
        compact_dim=int(_cfg_get(cfg_like, "compact_dim") or 384),
        compact_proj_dim=int(_cfg_get(cfg_like, "compact_proj_dim") or 0),
        compact_proj_hidden_mult=int(_cfg_get(cfg_like, "compact_proj_hidden_mult") or 2),
        compact_depth=int(_cfg_get(cfg_like, "compact_depth") or 2),
        compact_num_heads=int(_cfg_get(cfg_like, "compact_num_heads") or 6),
        compact_head_dim=int(_cfg_get(cfg_like, "compact_head_dim") or 64),
        compact_mlp_ratio=float(_cfg_get(cfg_like, "compact_mlp_ratio") or 4.0),
        compact_dropout=float(_cfg_get(cfg_like, "compact_dropout") or 0.0),
        compact_norm_output=bool(_cfg_get(cfg_like, "compact_norm_output") is not False),
        compact_pool_views=bool(_cfg_get(cfg_like, "compact_pool_views") or False),
        compact_all_global=(
            bool(_cfg_get(cfg_like, "compact_all_global"))
            if _cfg_get(cfg_like, "compact_all_global") is not None
            else True
        ),
        compact_token_layout=str(_cfg_get(cfg_like, "compact_token_layout") or ""),
        compact_token_scope=str(_cfg_get(cfg_like, "compact_token_scope") or ""),
        compact_spatial_h_tokens=int(_cfg_get(cfg_like, "compact_spatial_h_tokens") or 0),
        compact_spatial_v_tokens=int(_cfg_get(cfg_like, "compact_spatial_v_tokens") or 0),
        compact_global_tokens=int(_cfg_get(cfg_like, "compact_global_tokens") or 0),
        compact_view_global_tokens=(
            None
            if _cfg_get(cfg_like, "compact_view_global_tokens") is None
            else tuple(int(v) for v in _cfg_get(cfg_like, "compact_view_global_tokens"))
        ),
        compact_spatial_view_indices=(
            None
            if _cfg_get(cfg_like, "compact_spatial_view_indices") is None
            else tuple(int(v) for v in _cfg_get(cfg_like, "compact_spatial_view_indices"))
        ),
        compact_spatial_mode=str(_cfg_get(cfg_like, "compact_spatial_mode") or "axis_spatial"),
        compact_spatial_band_rows=int(_cfg_get(cfg_like, "compact_spatial_band_rows") or 2),
        compact_spatial_band_cols=int(_cfg_get(cfg_like, "compact_spatial_band_cols") or 2),
        compact_use_abs_pos_embed=bool(_cfg_get(cfg_like, "compact_use_abs_pos_embed") or False),
        compact_use_fourier_film_pos=bool(_cfg_get(cfg_like, "compact_use_fourier_film_pos") or False),
        compact_fourier_num_bands=int(_cfg_get(cfg_like, "compact_fourier_num_bands") or 8),
        compact_fourier_include_local_xy=(
            bool(_cfg_get(cfg_like, "compact_fourier_include_local_xy"))
            if _cfg_get(cfg_like, "compact_fourier_include_local_xy") is not None
            else True
        ),
        compact_fourier_include_global_xy=(
            bool(_cfg_get(cfg_like, "compact_fourier_include_global_xy"))
            if _cfg_get(cfg_like, "compact_fourier_include_global_xy") is not None
            else True
        ),
        compact_fourier_include_view_embed=(
            bool(_cfg_get(cfg_like, "compact_fourier_include_view_embed"))
            if _cfg_get(cfg_like, "compact_fourier_include_view_embed") is not None
            else True
        ),
        compact_fourier_film_init_scale=float(_cfg_get(cfg_like, "compact_fourier_film_init_scale") or 0.0),
        compact_conditioning_mode=str(_cfg_get(cfg_like, "compact_conditioning_mode") or "none"),
        compact_conditioning_dim=_cfg_get(cfg_like, "compact_conditioning_dim"),
        compact_conditioning_use_patch_logit_bias=bool(
            _cfg_get(cfg_like, "compact_conditioning_use_patch_logit_bias") or False
        ),
        compact_use_attn_spatial_writeback=bool(_cfg_get(cfg_like, "compact_use_attn_spatial_writeback") or False),
        compact_attention_mode=str(_cfg_get(cfg_like, "compact_attention_mode") or "standard"),
        compact_competitive_temperature=float(_cfg_get(cfg_like, "compact_competitive_temperature") or 1.0),
        compact_competitive_eps=float(_cfg_get(cfg_like, "compact_competitive_eps") or 1e-6),
        compact_spatial_writeback_hidden_mult=int(_cfg_get(cfg_like, "compact_spatial_writeback_hidden_mult") or 4),
        compact_spatial_writeback_init_scale=float(_cfg_get(cfg_like, "compact_spatial_writeback_init_scale") or 0.0),
        compact_spatial_writeback_detach_attn=bool(_cfg_get(cfg_like, "compact_spatial_writeback_detach_attn") or False),
        compact_spatial_moment_mode=str(_cfg_get(cfg_like, "compact_spatial_moment_mode") or "view_center_spread"),
        compact_core_tokens=int(_cfg_get(cfg_like, "compact_core_tokens") or 8),
        enable_compact_future=bool(_cfg_get(cfg_like, "enable_compact_future") is not False),
        compact_action_uses_all_tokens=bool(_cfg_get(cfg_like, "compact_action_uses_all_tokens") is not False),
        compact_future_uses_core_tokens=bool(_cfg_get(cfg_like, "compact_future_uses_core_tokens") is not False),
        learnable_compact_vision_encoder=bool(_cfg_get(cfg_like, "learnable_compact_vision_encoder") or False),
        compact_dino_model_name_or_path=str(
            _cfg_get(cfg_like, "compact_dino_model_name_or_path")
            or "facebook/dinov2-with-registers-large"
        ),
        use_abstract_anchor_tokens=bool(_cfg_get(cfg_like, "use_abstract_anchor_tokens") is not False),
        current_spatial_mode=str(_cfg_get(cfg_like, "current_spatial_mode") or "off"),
        mot_structure_mode=str(_cfg_get(cfg_like, "mot_structure_mode") or "fastwam"),
        default_joint_future_denoising=_cfg_get(cfg_like, "default_joint_future_denoising"),
        action_attends_future_video=_cfg_get(cfg_like, "action_attends_future_video"),
        future_video_attends_current_spatial=bool(
            _cfg_get(cfg_like, "future_video_attends_current_spatial") or False
        ),
        grid_sampler_num_tokens=int(_cfg_get(cfg_like, "grid_sampler_num_tokens") or 16),
        grid_sampler_num_tokens_per_view=_cfg_get_int_tuple(cfg_like, "grid_sampler_num_tokens_per_view"),
        grid_sampler_global_canvas=bool(_cfg_get(cfg_like, "grid_sampler_global_canvas") or False),
        grid_sampler_share_coord_head=bool(_cfg_get(cfg_like, "grid_sampler_share_coord_head") or False),
        grid_sampler_coord_embed=bool(_cfg_get(cfg_like, "grid_sampler_coord_embed") is not False),
        learnable_spatial_vision_encoder=bool(_cfg_get(cfg_like, "learnable_spatial_vision_encoder") or False),
        current_spatial_dino_model_name_or_path=str(
            _cfg_get(cfg_like, "current_spatial_dino_model_name_or_path")
            or "facebook/dinov2-with-registers-large"
        ),
        legacy_batched_view_dino_encoder=bool(_cfg_get(cfg_like, "legacy_batched_view_dino_encoder") or False),
    )
    return config.resolved()

def build_semantic_mot_model(cfg_like, device=None, dtype=None):
    config = build_semantic_mot_config(cfg_like)
    model = CompactSemanticFastWAM(config)
    if dtype is not None:
        model = model.to(dtype=dtype)
    if device is not None:
        model = model.to(device=device)
    model.apply_precision_policy()
    return model, config


def load_semantic_mot_checkpoint(model, path, strict=False, map_location="cpu"):
    # Legacy checkpoints contain trusted Python metadata; public exports also
    # support weights_only=True. Only load checkpoints from trusted sources.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = payload.get("state_dict", payload)
    model.load_state_dict(state, strict=strict)
    return payload
