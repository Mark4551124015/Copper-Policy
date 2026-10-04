from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .wan_video_dit import precompute_freqs_cis
from .semantic_mot_blocks import (
    ExpertStack,
    FastWAMStyleMoT,
    build_fastwam_style_joint_mask,
    sinusoidal_embedding_1d,
)
from .semantic_mot_presets import PRESET_REGISTRY

import torch


class WanContinuousFlowMatchScheduler:
    """Continuous-time Flow-Matching scheduler with shift-based sampling."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0, eps: float = 1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)






    def build_inference_schedule(
        self,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        shift_override: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be positive, got {num_inference_steps}")
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}")

        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return timesteps.to(dtype=dtype), deltas.to(dtype=dtype)

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *([1] * (sample.ndim - 1)))
        return sample + model_output * delta



@dataclass
class SemanticMoTConfig:
    per_view_image_size: Tuple[int, int]
    num_views: int
    image_patch_size: int
    teacher_dim: int
    text_dim: int
    action_dim: int
    action_per_frame: int
    future_blocks: int
    model_size: str
    enable_future_dense: bool
    enable_future_cls: bool
    # If False: future dense branch uses spatial patch tokens only.
    # If True: future dense branch includes register tokens as input/output.
    keep_registers: bool
    # If False and keep_registers=True: registers are present in the future dense stream,

    dropout: float
    eps: float
    flow_eps: float
    use_gradient_checkpointing: bool

    flow_shift: float = 5.0
    flow_num_train_timesteps: int = 1000
    enable_fp32_modules: bool = True
    enable_custom_fp32_precision: bool = False
    future_pool_size: int = 1
    future_semantic_steps: Optional[int] = None
    action_frame_ratio: Optional[float] = None
    # policy to continue a chunk after a prefix has already been committed.
    use_runtime_current_vjepa_anchor: bool = False
    vjepa_model_name_or_path: Optional[str] = None
    per_view_image_sizes: Optional[Tuple[Tuple[int, int], ...]] = None

    hidden_dim: Optional[int] = None
    action_hidden_dim: Optional[int] = None  # action expert hidden dim; defaults to preset
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    mlp_ratio: Optional[float] = None
    action_mlp_ratio: Optional[float] = None
    num_register_tokens: Optional[int] = None
    dino_spatial_mean: Optional[Tuple[float, ...]] = None
    dino_spatial_global_scale: Optional[float] = None
    teacher_latent_norm_enabled: bool = True
    proprio_dim: Optional[int] = None
    proprio_injection_mode: str = "context"  # action_prefix | context | none
    # Match original FastWAM's shared [language; proprio] cross-attention
    # context. Disabled by default so existing checkpoints/configs preserve
    # the action-only proprio routing.
    video_context_uses_proprio: bool = False
    used_action_channel_ids: Optional[Tuple[int, ...]] = None
    use_view_embed: bool = False
    view_rope_layout: str = "horizontal"  # horizontal | vertical | robotwin_tshape
    compact_view_rope_layout: Optional[str] = None  # defaults to view_rope_layout when unset

    # Compact predictive-control latent settings
    compact_latent_enabled: bool = False
    compact_encoder_type: str = "vjepa"
    compact_encoder_dim: int = 1024
    compact_use_cached: bool = True
    compact_num_tokens: int = 8
    compact_dim: int = 384
    compact_proj_dim: int = 0  # if > 0, add MLP projector compact_dim -> compact_proj_dim
    compact_proj_hidden_mult: int = 2  # hidden multiplier for the MLP projector
    compact_depth: int = 2
    compact_num_heads: int = 6
    compact_head_dim: int = 64
    compact_mlp_ratio: float = 4.0
    compact_dropout: float = 0.0
    compact_norm_output: bool = True
    compact_pool_views: bool = False
    compact_all_global: bool = True
    compact_token_layout: str = "all_global"  # all_global | spatial_global
    compact_token_scope: str = "merged_views"  # merged_views | hybrid
    compact_spatial_h_tokens: int = 0
    compact_spatial_v_tokens: int = 0
    compact_global_tokens: int = 0
    compact_view_global_tokens: Optional[Tuple[int, ...]] = None
    compact_spatial_view_indices: Optional[Tuple[int, ...]] = None
    compact_spatial_mode: str = "axis_spatial"  # axis_spatial | v_anchor
    compact_spatial_band_rows: int = 2
    compact_spatial_band_cols: int = 2
    compact_use_abs_pos_embed: bool = False
    compact_use_fourier_film_pos: bool = False
    compact_fourier_num_bands: int = 8
    compact_fourier_include_local_xy: bool = True
    compact_fourier_include_global_xy: bool = True
    compact_fourier_include_view_embed: bool = True
    compact_fourier_film_init_scale: float = 0.0
    compact_conditioning_mode: str = "none"  # none | patch_motivation_key
    compact_conditioning_dim: Optional[int] = None
    compact_conditioning_use_patch_logit_bias: bool = False
    # Spatial grounding (Phase 2) — attention-mode + writeback
    compact_use_attn_spatial_writeback: bool = False
    compact_attention_mode: str = "standard"        # "standard" | "competitive"
    compact_competitive_temperature: float = 1.0
    compact_competitive_eps: float = 1e-6
    compact_spatial_writeback_hidden_mult: int = 4
    compact_spatial_writeback_init_scale: float = 0.0
    compact_spatial_writeback_detach_attn: bool = False
    compact_spatial_moment_mode: str = "view_center_spread"
    compact_core_tokens: int = 8
    enable_compact_future: bool = True
    compact_action_uses_all_tokens: bool = True
    compact_future_uses_core_tokens: bool = True
    learnable_compact_vision_encoder: bool = False
    compact_dino_model_name_or_path: str = "facebook/dinov2-with-registers-large"
    use_abstract_anchor_tokens: bool = True
    current_spatial_mode: str = "off"  # off | sampled | full
    mot_structure_mode: str = "fastwam"  # fastwam | fastwam_joint
    default_joint_future_denoising: Optional[bool] = None
    action_attends_future_video: Optional[bool] = None  # None => follow mot_structure_mode; True/False => override
    # Let future/video queries read the current-frame spatial branch V_t.
    # False preserves the original strict FastWAM routing.
    future_video_attends_current_spatial: bool = False
    grid_sampler_num_tokens: int = 16
    grid_sampler_num_tokens_per_view: Optional[tuple[int, ...]] = None
    grid_sampler_global_canvas: bool = False
    grid_sampler_share_coord_head: bool = False
    grid_sampler_coord_embed: bool = True
    learnable_spatial_vision_encoder: bool = False
    current_spatial_dino_model_name_or_path: str = "facebook/dinov2-with-registers-large"
    # Reproduce the pre-per-view-geometry DINO execution: stack all views into
    # one B*V batch. This changes BF16 attention numerics and is kept opt-in.
    legacy_batched_view_dino_encoder: bool = False

    def resolved(self) -> "SemanticMoTConfig":
        if self.model_size not in PRESET_REGISTRY:
            raise KeyError(f"Unknown model_size={self.model_size}. Expected one of {list(PRESET_REGISTRY.keys())}")
        preset = PRESET_REGISTRY[self.model_size]
        cfg = SemanticMoTConfig(**asdict(self))
        cfg.hidden_dim = int(cfg.hidden_dim or preset.hidden_dim)
        cfg.action_hidden_dim = int(cfg.action_hidden_dim or preset.action_hidden_dim)
        cfg.num_layers = int(cfg.num_layers or preset.num_layers)
        cfg.num_heads = int(cfg.num_heads or preset.num_heads)
        cfg.head_dim = int(cfg.head_dim or preset.head_dim)
        cfg.mlp_ratio = float(cfg.mlp_ratio or preset.mlp_ratio)
        cfg.future_pool_size = int(cfg.future_pool_size or 1)
        if cfg.future_pool_size <= 0:
            raise ValueError(f"future_pool_size must be positive, got {cfg.future_pool_size}")
        if cfg.future_semantic_steps is not None:
            cfg.future_semantic_steps = int(cfg.future_semantic_steps)
            if cfg.future_semantic_steps <= 0:
                raise ValueError(
                    f"future_semantic_steps must be positive when set, got {cfg.future_semantic_steps}"
                )
        if cfg.action_frame_ratio is not None:
            cfg.action_frame_ratio = float(cfg.action_frame_ratio)
            if cfg.action_frame_ratio <= 0:
                raise ValueError(
                    f"action_frame_ratio must be positive when set, got {cfg.action_frame_ratio}"
                )
        cfg.action_mlp_ratio = float(
            cfg.action_mlp_ratio
            if cfg.action_mlp_ratio is not None
            else (preset.action_mlp_ratio if preset.action_mlp_ratio is not None else cfg.mlp_ratio)
        )
        if cfg.per_view_image_sizes is None:
            cfg.per_view_image_sizes = tuple(
                tuple(int(v) for v in cfg.per_view_image_size)
                for _ in range(int(cfg.num_views))
            )
        else:
            cfg.per_view_image_sizes = tuple(
                (int(h), int(w))
                for h, w in cfg.per_view_image_sizes
            )
        if len(cfg.per_view_image_sizes) != int(cfg.num_views):
            raise ValueError(
                "per_view_image_sizes must provide one (H,W) pair per view, "
                f"got {len(cfg.per_view_image_sizes)} for num_views={cfg.num_views}"
            )
        cfg.per_view_image_size = tuple(cfg.per_view_image_sizes[0])
        cfg.num_register_tokens = int(cfg.num_register_tokens or preset.num_register_tokens)
        if cfg.used_action_channel_ids is None:
            cfg.used_action_channel_ids = tuple(range(int(cfg.action_dim)))
        else:
            used_ids = tuple(int(i) for i in cfg.used_action_channel_ids)
            if not used_ids:
                raise ValueError("used_action_channel_ids cannot be empty")
            invalid = [i for i in used_ids if i < 0 or i >= int(cfg.action_dim)]
            if invalid:
                raise ValueError(
                    f"used_action_channel_ids contains out-of-range ids {invalid} for action_dim={cfg.action_dim}"
                )
            cfg.used_action_channel_ids = used_ids

        mode = str(cfg.current_spatial_mode or "").strip().lower()
        if not mode:
            mode = "off"
        if mode not in {"off", "sampled", "full"}:
            raise ValueError(
                f"Unsupported current_spatial_mode={cfg.current_spatial_mode!r}. Expected one of: off, sampled, full."
            )
        cfg.current_spatial_mode = mode

        legacy_layout = str(cfg.compact_token_layout or "").strip().lower()
        if legacy_layout and legacy_layout not in {
            "all_global",
            "global",
            "global_only",
            "spatial_global",
            "spatial+global",
            "hv_global",
            "axis_global",
        }:
            raise ValueError(
                f"Unsupported compact_token_layout={cfg.compact_token_layout!r}. "
                "Expected one of: all_global, spatial_global."
            )

        cfg.compact_all_global = bool(cfg.compact_all_global)
        if legacy_layout:
            cfg.compact_all_global = legacy_layout in {"all_global", "global", "global_only"}
        cfg.compact_token_layout = "all_global" if cfg.compact_all_global else "spatial_global"

        token_scope = str(cfg.compact_token_scope or "").strip().lower()
        if cfg.compact_token_layout == "all_global":
            if token_scope in {"", "merged", "merged_views", "cross_view"}:
                token_scope = "merged_views"
            else:
                raise ValueError(
                    f"Unsupported compact_token_scope={cfg.compact_token_scope!r} for all_global layout. "
                    "Expected: merged_views."
                )
        else:
            if token_scope in {"", "hybrid"}:
                token_scope = "hybrid"
            elif token_scope in {"shared", "shared_views", "cross_view_shared"}:
                token_scope = "shared_views"
            else:
                raise ValueError(
                    f"Unsupported compact_token_scope={cfg.compact_token_scope!r} for spatial_global layout. "
                    "Expected: hybrid or shared_views."
                )
        cfg.compact_token_scope = token_scope

        conditioning_mode = str(cfg.compact_conditioning_mode or "").strip().lower()
        if conditioning_mode in {"", "none", "off", "disabled"}:
            conditioning_mode = "none"
        elif conditioning_mode in {"patch_motivation_key", "patch_motivation", "task_key"}:
            conditioning_mode = "patch_motivation_key"
        else:
            raise ValueError(
                f"Unsupported compact_conditioning_mode={cfg.compact_conditioning_mode!r}. "
                "Expected one of: none, patch_motivation_key."
            )
        cfg.compact_conditioning_mode = conditioning_mode
        if cfg.compact_conditioning_dim is None:
            cfg.compact_conditioning_dim = int(cfg.text_dim)
        else:
            cfg.compact_conditioning_dim = int(cfg.compact_conditioning_dim)
        cfg.compact_conditioning_use_patch_logit_bias = bool(cfg.compact_conditioning_use_patch_logit_bias)
        spatial_mode = str(cfg.compact_spatial_mode or "").strip().lower()
        if spatial_mode in {"", "axis", "axis_spatial", "7v7h", "hv", "h_v"}:
            spatial_mode = "axis_spatial"
        elif spatial_mode in {"v", "v_anchor", "v-anchor"}:
            spatial_mode = "v_anchor"
        else:
            raise ValueError(
                f"Unsupported compact_spatial_mode={cfg.compact_spatial_mode!r}. "
                "Expected one of: axis_spatial, v_anchor."
            )
        cfg.compact_spatial_mode = spatial_mode

        band_rows = int(cfg.compact_spatial_band_rows or 0)
        band_cols = int(cfg.compact_spatial_band_cols or 0)
        if band_rows <= 0 or band_cols <= 0:
            raise ValueError(
                "compact_spatial_band_rows and compact_spatial_band_cols must be positive, "
                f"got rows={band_rows}, cols={band_cols}"
            )
        cfg.compact_spatial_band_rows = band_rows
        cfg.compact_spatial_band_cols = band_cols

        per_view_image_sizes = tuple(tuple(int(v) for v in size) for size in cfg.per_view_image_sizes)
        per_view_grid_sizes = tuple(
            (image_h // int(cfg.image_patch_size), image_w // int(cfg.image_patch_size))
            for image_h, image_w in per_view_image_sizes
        )

        global_tokens = int(cfg.compact_global_tokens or 0)
        if global_tokens <= 0 and cfg.compact_token_layout == "all_global":
            fallback_global_tokens = int(cfg.compact_num_tokens or 0)
            if fallback_global_tokens <= 0:
                fallback_global_tokens = 8
            global_tokens = fallback_global_tokens
        cfg.compact_global_tokens = global_tokens

        if cfg.compact_token_layout == "spatial_global":
            if cfg.compact_view_global_tokens is None:
                view_global_tokens = tuple(0 for _ in range(int(cfg.num_views)))
            else:
                view_global_tokens = tuple(int(v) for v in cfg.compact_view_global_tokens)
                if len(view_global_tokens) != int(cfg.num_views):
                    raise ValueError(
                        "compact_view_global_tokens must provide one entry per view, "
                        f"got {len(view_global_tokens)} for num_views={cfg.num_views}."
                    )
                if any(v < 0 for v in view_global_tokens):
                    raise ValueError(
                        f"compact_view_global_tokens must be non-negative, got {view_global_tokens}."
                    )
            cfg.compact_view_global_tokens = view_global_tokens

            if cfg.compact_spatial_view_indices is None:
                spatial_view_indices = tuple(range(int(cfg.num_views)))
            else:
                spatial_view_indices = tuple(int(v) for v in cfg.compact_spatial_view_indices)
                if len(set(spatial_view_indices)) != len(spatial_view_indices):
                    raise ValueError(
                        f"compact_spatial_view_indices must be unique, got {spatial_view_indices}."
                    )
                invalid_views = [v for v in spatial_view_indices if v < 0 or v >= int(cfg.num_views)]
                if invalid_views:
                    raise ValueError(
                        "compact_spatial_view_indices contains out-of-range ids "
                        f"{invalid_views} for num_views={cfg.num_views}."
                    )
            cfg.compact_spatial_view_indices = spatial_view_indices

            derived_h_tokens_per_view = tuple(
                (grid_h + cfg.compact_spatial_band_rows - 1) // cfg.compact_spatial_band_rows
                for grid_h, _grid_w in per_view_grid_sizes
            )
            derived_v_tokens_per_view = tuple(
                (grid_w + cfg.compact_spatial_band_cols - 1) // cfg.compact_spatial_band_cols
                for _grid_h, grid_w in per_view_grid_sizes
            )
            explicit_h_tokens = int(cfg.compact_spatial_h_tokens or 0)
            explicit_v_tokens = int(cfg.compact_spatial_v_tokens or 0)
            if len(set(per_view_grid_sizes)) == 1:
                derived_h_tokens = int(derived_h_tokens_per_view[0])
                derived_v_tokens = int(derived_v_tokens_per_view[0])
                if explicit_h_tokens not in {0, derived_h_tokens}:
                    raise ValueError(
                        "compact_spatial_h_tokens must match the band-derived token count. "
                        f"Expected {derived_h_tokens}, got {explicit_h_tokens}."
                    )
                if explicit_v_tokens not in {0, derived_v_tokens}:
                    raise ValueError(
                        "compact_spatial_v_tokens must match the band-derived token count. "
                        f"Expected {derived_v_tokens}, got {explicit_v_tokens}."
                    )
                cfg.compact_spatial_h_tokens = derived_h_tokens
                cfg.compact_spatial_v_tokens = derived_v_tokens
            else:
                if cfg.compact_token_scope == "shared_views":
                    raise ValueError(
                        "compact_token_scope='shared_views' is not supported with heterogeneous per_view_image_sizes."
                    )
                if explicit_h_tokens or explicit_v_tokens:
                    raise ValueError(
                        "compact_spatial_h_tokens / compact_spatial_v_tokens must be 0 when per-view grids differ. "
                        "Use the derived per-view band layout instead."
                    )
                cfg.compact_spatial_h_tokens = 0
                cfg.compact_spatial_v_tokens = 0
            if cfg.compact_spatial_mode == "axis_spatial":
                per_view_spatial_tokens = tuple(
                    int(h_tokens) + int(v_tokens)
                    for h_tokens, v_tokens in zip(derived_h_tokens_per_view, derived_v_tokens_per_view)
                )
            else:
                per_view_spatial_tokens = tuple(int(v_tokens) for v_tokens in derived_v_tokens_per_view)
            per_view_global_total = sum(cfg.compact_view_global_tokens)
            if cfg.compact_token_scope == "shared_views":
                cfg.compact_num_tokens = int(per_view_spatial_tokens[0]) + int(cfg.compact_global_tokens) + per_view_global_total
            else:
                cfg.compact_num_tokens = (
                    sum(int(per_view_spatial_tokens[view_idx]) for view_idx in cfg.compact_spatial_view_indices)
                    + int(cfg.compact_global_tokens)
                    + per_view_global_total
                )
        else:
            cfg.compact_spatial_h_tokens = int(cfg.compact_spatial_h_tokens or 0)
            cfg.compact_spatial_v_tokens = int(cfg.compact_spatial_v_tokens or 0)
            cfg.compact_view_global_tokens = None
            cfg.compact_spatial_view_indices = None
            cfg.compact_num_tokens = int(cfg.compact_global_tokens)
            if cfg.compact_num_tokens <= 0:
                raise ValueError(f"compact_num_tokens must be positive, got {cfg.compact_num_tokens}")
        cfg.compact_core_tokens = int(cfg.compact_num_tokens)

        mot_structure_mode = str(cfg.mot_structure_mode or "").strip().lower()
        if mot_structure_mode in {"", "fastwam"}:
            mot_structure_mode = "fastwam"
        elif mot_structure_mode in {"fastwam_joint", "joint", "joint_denoise"}:
            mot_structure_mode = "fastwam_joint"
        else:
            raise ValueError(
                f"Unsupported mot_structure_mode={cfg.mot_structure_mode!r}. "
                "Expected one of: fastwam, fastwam_joint."
            )
        cfg.mot_structure_mode = mot_structure_mode
        if cfg.default_joint_future_denoising is None:
            cfg.default_joint_future_denoising = mot_structure_mode == "fastwam_joint"
        else:
            cfg.default_joint_future_denoising = bool(cfg.default_joint_future_denoising)
        if cfg.action_attends_future_video is None:
            cfg.action_attends_future_video = mot_structure_mode == "fastwam_joint"
        else:
            cfg.action_attends_future_video = bool(cfg.action_attends_future_video)

        cfg.enable_compact_future = bool(cfg.enable_compact_future)

        proprio_mode = str(cfg.proprio_injection_mode or "").strip().lower()
        if proprio_mode in {"", "action", "prefix", "action_prefix"}:
            proprio_mode = "action_prefix"
        elif proprio_mode in {"context", "context_token"}:
            proprio_mode = "context"
        elif proprio_mode in {"none", "off", "disabled"}:
            proprio_mode = "none"
        else:
            raise ValueError(
                f"Unsupported proprio_injection_mode={cfg.proprio_injection_mode!r}. "
                "Expected one of: action_prefix, context, none."
            )
        cfg.proprio_injection_mode = proprio_mode
        return cfg

class ContinuousFlowMatcher(nn.Module):
    def __init__(
        self,
        eps: float,
        shift: float = 5.0,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            eps=eps,
        )
        self.num_train_timesteps = int(num_train_timesteps)





    def build_inference_schedule(
        self,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scheduler.build_inference_schedule(num_steps, device, dtype)

    def step(self, x_t: torch.Tensor, pred_velocity: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return self.scheduler.step(pred_velocity, delta, x_t)

class SemanticFastWAM(nn.Module):
    def __init__(self, config: SemanticMoTConfig):
        super().__init__()
        self.config = config.resolved()
        cfg = self.config
        self.num_views = int(cfg.num_views)
        self.hidden_dim = int(cfg.hidden_dim)
        self.teacher_dim = int(cfg.teacher_dim)
        self.text_dim = int(cfg.text_dim)
        self.action_dim = int(cfg.action_dim)
        action_channel_mask = torch.zeros(self.action_dim, dtype=torch.bool)
        action_channel_mask[list(cfg.used_action_channel_ids)] = True
        self.register_buffer("action_channel_mask", action_channel_mask, persistent=False)
        self.future_blocks = int(cfg.future_blocks)
        self.action_per_frame = int(cfg.action_per_frame)
        self.enable_future_dense = bool(cfg.enable_future_dense)
        self.enable_future_cls = bool(cfg.enable_future_cls)
        self.anchor_keep_registers = bool(cfg.keep_registers)

        self.per_view_image_sizes = tuple((int(h), int(w)) for h, w in cfg.per_view_image_sizes)
        self.per_view_grid_sizes = []
        for image_h, image_w in self.per_view_image_sizes:
            if image_h % int(cfg.image_patch_size) != 0 or image_w % int(cfg.image_patch_size) != 0:
                raise ValueError(
                    f"per_view_image_size={(image_h, image_w)} must be divisible by image_patch_size={cfg.image_patch_size}"
                )
            self.per_view_grid_sizes.append((image_h // int(cfg.image_patch_size), image_w // int(cfg.image_patch_size)))
        self.per_view_grid_sizes = tuple(self.per_view_grid_sizes)
        self.per_view_spatial_tokens = tuple(grid_h * grid_w for grid_h, grid_w in self.per_view_grid_sizes)
        self.per_view_spatial_offsets = []
        cursor = 0
        for tokens_per_view in self.per_view_spatial_tokens:
            self.per_view_spatial_offsets.append(cursor)
            cursor += int(tokens_per_view)
        self.per_view_spatial_offsets = tuple(self.per_view_spatial_offsets)
        self.total_num_spatial_tokens = int(cursor)
        self.grid_h, self.grid_w = self.per_view_grid_sizes[0]
        self.use_view_embed = bool(cfg.use_view_embed)
        self.view_rope_layout = str(cfg.view_rope_layout)
        if self.view_rope_layout not in {"horizontal", "vertical", "robotwin_tshape"}:
            raise ValueError(
                f"Unsupported view_rope_layout={self.view_rope_layout!r}. "
                "Expected one of: horizontal, vertical, robotwin_tshape."
            )
        if self.view_rope_layout == "horizontal":
            self.virtual_grid_h = max(grid_h for grid_h, _grid_w in self.per_view_grid_sizes)
            self.virtual_grid_w = sum(grid_w for _grid_h, grid_w in self.per_view_grid_sizes)
        elif self.view_rope_layout == "vertical":
            self.virtual_grid_h = sum(grid_h for grid_h, _grid_w in self.per_view_grid_sizes)
            self.virtual_grid_w = max(grid_w for _grid_h, grid_w in self.per_view_grid_sizes)
        else:
            top_h, top_w = self.per_view_grid_sizes[0]
            bottom_row_h = max((grid_h for grid_h, _grid_w in self.per_view_grid_sizes[1:]), default=0)
            bottom_row_w = sum(grid_w for _grid_h, grid_w in self.per_view_grid_sizes[1:])
            self.virtual_grid_h = top_h + bottom_row_h
            self.virtual_grid_w = max(top_w, bottom_row_w)
        self.num_spatial_tokens = self.total_num_spatial_tokens
        self.num_register_tokens = int(cfg.num_register_tokens)
        self.cls_tokens_per_step = self.num_views
        # Future semantic generation is spatial-only even when current-frame anchor
        # retains register tokens for alignment with the cached frozen DINO input.
        self.keep_registers = False

        self.per_view_dense_tokens = tuple(
            int(tokens_per_view) + (self.num_register_tokens if self.keep_registers else 0)
            for tokens_per_view in self.per_view_spatial_tokens
        )
        self.dense_tokens_per_step = int(sum(self.per_view_dense_tokens))
        # Register slot embeddings are needed only when at least one stream
        # actually carries register tokens.
        self._use_anchor_register_slots = bool(self.anchor_keep_registers and self.num_register_tokens > 0)
        self._use_future_register_slots = bool(self.keep_registers and self.num_register_tokens > 0)
        self._use_register_slot_embed = bool(self._use_anchor_register_slots or self._use_future_register_slots)

        self.action_hidden_dim = int(cfg.action_hidden_dim)
        rope_cache_len = max(
            2048,
            self.future_blocks + 1,
            self.action_per_frame * self.future_blocks,
            self.virtual_grid_h,
            self.virtual_grid_w,
        )
        video_f, video_h, video_w = self._precompute_video_rope_freqs(cfg.head_dim, rope_cache_len)
        # Store as plain attributes (not register_buffer) so that model.to(dtype=bf16)
        # does NOT cast complex128 → bf16, which would discard the imaginary part
        # and completely destroy RoPE.  Device transfer is handled in the build methods.
        self._video_rope_f_freqs = video_f
        self._video_rope_h_freqs = video_h
        self._video_rope_w_freqs = video_w
        self._action_rope_freqs = precompute_freqs_cis(cfg.head_dim, end=rope_cache_len)

        self.frozen_anchor_proj = nn.Linear(cfg.teacher_dim, cfg.hidden_dim)
        # Separate context projectors for video and action experts (FastWAM-style).
        self.video_context_proj = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.action_context_proj = nn.Sequential(
            nn.Linear(cfg.text_dim, self.action_hidden_dim),
            nn.GELU(),
            nn.Linear(self.action_hidden_dim, self.action_hidden_dim),
        )

        # Video-stream positional / type embeddings (dim = hidden_dim)
        # Slots: 0=frozen_cls, 1=frozen_reg, 2=frozen_spatial,
        #        3=future_cls, 4=future_reg, 5=future_dense
        self.view_embed = nn.Embedding(self.num_views, cfg.hidden_dim) if self.use_view_embed else None
        self.token_type_embed = nn.Embedding(6, cfg.hidden_dim)
        self.register_slot_embed = nn.Embedding(self.num_register_tokens, cfg.hidden_dim)
        if not self._use_register_slot_embed:
            # Keep tensor shape for checkpoint compatibility, but exclude it from
            # optimization when register slots are never consumed in this config.
            self.register_slot_embed.weight.requires_grad_(False)

        # Action-stream positional / type embeddings (dim = action_hidden_dim)
        self.action_type_embed = nn.Parameter(torch.zeros(1, 1, self.action_hidden_dim))

        self.future_dense_in = nn.Linear(cfg.teacher_dim, cfg.hidden_dim) if cfg.enable_future_dense else None
        self.future_cls_in = nn.Linear(cfg.teacher_dim, cfg.hidden_dim) if cfg.enable_future_cls else None
        self.action_in = nn.Linear(cfg.action_dim, self.action_hidden_dim)

        self.video_time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 6),
        )
        self.action_time_mlp = nn.Sequential(
            nn.Linear(self.action_hidden_dim, self.action_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.action_hidden_dim, self.action_hidden_dim * 6),
        )

        video_ffn_dim = int(cfg.hidden_dim * cfg.mlp_ratio)
        action_ffn_dim = int(self.action_hidden_dim * cfg.action_mlp_ratio)
        video_stack = ExpertStack(
            num_layers=cfg.num_layers,
            hidden_dim=cfg.hidden_dim,
            ffn_dim=video_ffn_dim,
            context_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            eps=cfg.eps,
            dropout=cfg.dropout,
        )
        action_stack = ExpertStack(
            num_layers=cfg.num_layers,
            hidden_dim=self.action_hidden_dim,
            ffn_dim=action_ffn_dim,
            context_dim=self.action_hidden_dim,
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            eps=cfg.eps,
            dropout=cfg.dropout,
        )
        self.mot = FastWAMStyleMoT(
            expert_stacks={"video": video_stack, "action": action_stack},
            num_heads=cfg.num_heads,
            head_dim=cfg.head_dim,
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        )
        self.dit = self.mot

        self.dense_head = nn.Linear(cfg.hidden_dim, cfg.teacher_dim) if cfg.enable_future_dense else None
        self.cls_head = nn.Linear(cfg.hidden_dim, cfg.teacher_dim) if cfg.enable_future_cls else None
        self.action_head = nn.Linear(self.action_hidden_dim, cfg.action_dim)
        self.flow = ContinuousFlowMatcher(
            eps=cfg.flow_eps,
            shift=cfg.flow_shift,
            num_train_timesteps=cfg.flow_num_train_timesteps,
        )

        self.proprio_dim = cfg.proprio_dim if cfg.proprio_dim is not None else 0
        self.proprio_injection_mode = str(cfg.proprio_injection_mode)
        if self.proprio_dim > 0 and self.proprio_injection_mode == "context":
            self.context_proprio_in = nn.Linear(self.proprio_dim, cfg.text_dim)
        else:
            self.context_proprio_in = None
        if self.proprio_dim > 0 and self.proprio_injection_mode == "action_prefix":
            self.action_proprio_in = nn.Linear(self.proprio_dim, self.action_hidden_dim)
            self.action_proprio_type_embed = nn.Parameter(torch.zeros(1, 1, self.action_hidden_dim))
        else:
            self.action_proprio_in = None
            self.action_proprio_type_embed = None

        # Match the old WAN/LingBot intent using this model's actual structure:
        # time-conditioning MLPs plus each expert block's modulation and attention norms.
        self._keep_in_fp32_modules = self._fp32_tensor_name_tokens()

        dino_spatial_mean, dino_spatial_scale = self._load_dino_spatial_norm(cfg)
        # Keep this tensor device-aligned with the model.  Moving it inside
        # _normalize_dino_spatial() creates an aten._to_copy/DeviceCopy op in
        # torch.compile graphs and forces CUDA Graph partitioning.
        if dino_spatial_mean is None:
            dino_spatial_mean = torch.empty(0, dtype=torch.float32)
        # Keep teacher normalization in the state dict: deployment must not
        self.register_buffer("_dino_spatial_mean", dino_spatial_mean, persistent=True)
        self.register_buffer(
            "_dino_spatial_scale",
            torch.tensor(float(dino_spatial_scale), dtype=torch.float32),
            persistent=True,
        )
        # Deployment may replace this with a compiled callable.  Keep eager
        self._action_denoise_step_fn = self._action_denoise_step



    @staticmethod
    def _runtime_compute_dtype(device: torch.device, fallback: torch.dtype) -> torch.dtype:
        if device.type == "cuda" and torch.is_autocast_enabled():
            try:
                return torch.get_autocast_dtype("cuda")
            except TypeError:
                return torch.get_autocast_gpu_dtype()
        if device.type == "cpu" and torch.is_autocast_enabled():
            try:
                return torch.get_autocast_dtype("cpu")
            except TypeError:
                return torch.get_autocast_cpu_dtype()
        return fallback

    @staticmethod
    def _autocast_device_type(device: torch.device) -> str:
        return device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

    @staticmethod
    def _module_compute_dtype(
        module: nn.Module,
        *,
        fallback: torch.dtype = torch.float32,
    ) -> torch.dtype:
        for param in module.parameters(recurse=True):
            if torch.is_floating_point(param):
                return param.dtype
        for buf in module.buffers(recurse=True):
            if torch.is_floating_point(buf):
                return buf.dtype
        return fallback

    def _forward_module_fp32(
        self,
        module: nn.Module,
        x: torch.Tensor,
        *,
        out_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        runtime_out_dtype = out_dtype
        if runtime_out_dtype is None:
            runtime_out_dtype = self._runtime_compute_dtype(x.device, x.dtype)
        if not self._use_custom_fp32_precision():
            y = module(x)
            if runtime_out_dtype is not None:
                y = y.to(dtype=runtime_out_dtype)
            return y
        target_dtype = self._module_compute_dtype(module, fallback=torch.float32)
        with torch.autocast(device_type=self._autocast_device_type(x.device), enabled=False):
            y = module(x.to(device=x.device, dtype=target_dtype))
        if runtime_out_dtype is not None:
            y = y.to(dtype=runtime_out_dtype)
        return y

    def _use_custom_fp32_precision(self) -> bool:
        return bool(
            getattr(self.config, "enable_custom_fp32_precision", False)
            or getattr(self.config, "enable_fp32_modules", False)
        )

    @staticmethod
    def _load_dino_spatial_norm(cfg: SemanticMoTConfig) -> tuple[Optional[torch.Tensor], float]:
        if (not bool(getattr(cfg, "teacher_latent_norm_enabled", True))) or cfg.dino_spatial_mean is None:
            return None, 1.0
        mean = torch.tensor(cfg.dino_spatial_mean, dtype=torch.float32)
        scale = float(cfg.dino_spatial_global_scale or 1.0)
        if scale <= 0:
            raise ValueError(f"dino spatial global scale must be positive, got {scale}")
        return mean, scale

    def _normalize_dino_spatial(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None or self._dino_spatial_mean.numel() == 0:
            return x
        # _dino_spatial_mean is a buffer and follows model.to(device).
        # Do not call .to(device=...) here: it becomes DeviceCopy in a
        # compiled CUDA Graph region.
        mean = self._dino_spatial_mean
        if self._use_custom_fp32_precision():
            x_work = x.to(torch.float32)
        else:
            x_work = x
            mean = mean.to(dtype=x.dtype)
        y = (x_work - mean) / self._dino_spatial_scale
        return y.to(dtype=x.dtype)

    def denormalize_dino_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if self._dino_spatial_mean.numel() == 0:
            return x
        mean = self._dino_spatial_mean
        if self._use_custom_fp32_precision():
            x_work = x.to(torch.float32)
        else:
            x_work = x
            mean = mean.to(dtype=x.dtype)
        y = x_work * self._dino_spatial_scale + mean
        return y.to(dtype=x.dtype)

    @staticmethod
    def _precompute_axis_rope_freqs(num_pairs: int, end: int, theta: float = 10000.0) -> torch.Tensor:
        if num_pairs <= 0:
            return torch.empty(end, 0, dtype=torch.complex128)
        real_dim = num_pairs * 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, real_dim, 2, dtype=torch.float64) / real_dim)
        )
        angles = torch.outer(torch.arange(end, dtype=torch.float64), inv_freq)
        return torch.polar(torch.ones_like(angles), angles)

    @classmethod
    def _precompute_video_rope_freqs(cls, head_dim: int, end: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """3D RoPE cache with exactly head_dim/2 complex pairs.

        WAN/FastWAM split the head dimension across frame, height, and width.
        The semantic presets use head_dim=64, so the safe split below preserves
        the same 3-axis semantics while avoiding a dropped complex pair when one
        axis would otherwise receive an odd real dimension.
        """
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}")
        total_pairs = head_dim // 2
        f_pairs = (head_dim - 2 * (head_dim // 3)) // 2
        h_pairs = (head_dim // 3) // 2
        w_pairs = total_pairs - f_pairs - h_pairs
        if f_pairs < 0 or h_pairs < 0 or w_pairs < 0:
            raise ValueError(
                f"Invalid 3D RoPE split for head_dim={head_dim}: "
                f"pairs=({f_pairs}, {h_pairs}, {w_pairs})"
            )
        return (
            cls._precompute_axis_rope_freqs(f_pairs, end),
            cls._precompute_axis_rope_freqs(h_pairs, end),
            cls._precompute_axis_rope_freqs(w_pairs, end),
        )

    @staticmethod
    def num_parameters(model: nn.Module, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

    def parameter_report(self) -> Dict[str, float]:
        total = self.num_parameters(self) / 1e6
        trainable = self.num_parameters(self, trainable_only=True) / 1e6
        return {"total_params_m": total, "trainable_params_m": trainable}

    def _resolve_attr_path(self, attr_path: str):
        obj = self
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        return obj

    def _fp32_module_paths(self) -> tuple[str, ...]:
        paths = [
            "video_time_mlp",
            "action_time_mlp",
        ]
        for expert_name in self.mot.expert_order:
            for layer_idx in range(self.mot.num_layers):
                prefix = f"mot.experts.{expert_name}.blocks.{layer_idx}"
                paths.extend(
                    (
                        f"{prefix}.norm1",
                        f"{prefix}.norm2",
                        f"{prefix}.norm3",
                        f"{prefix}.self_attn.norm_q",
                        f"{prefix}.self_attn.norm_k",
                        f"{prefix}.cross_attn.norm_q",
                        f"{prefix}.cross_attn.norm_k",
                    )
                )
        return tuple(paths)

    def _fp32_parameter_paths(self) -> tuple[str, ...]:
        paths = []
        for expert_name in self.mot.expert_order:
            for layer_idx in range(self.mot.num_layers):
                paths.append(f"mot.experts.{expert_name}.blocks.{layer_idx}.modulation")
        return tuple(paths)

    def _fp32_tensor_name_tokens(self) -> tuple[str, ...]:
        tokens = [
            "video_time_mlp",
            "action_time_mlp",
            "_dino_spatial_mean",
        ]
        for expert_name in self.mot.expert_order:
            for layer_idx in range(self.mot.num_layers):
                prefix = f"mot.experts.{expert_name}.blocks.{layer_idx}"
                tokens.extend(
                    (
                        f"{prefix}.modulation",
                        f"{prefix}.self_attn.norm_q",
                        f"{prefix}.self_attn.norm_k",
                        f"{prefix}.cross_attn.norm_q",
                        f"{prefix}.cross_attn.norm_k",
                    )
                )
        return tuple(tokens)


    def apply_precision_policy(self) -> None:
        if not self._use_custom_fp32_precision():
            return
        keep_tokens = tuple(getattr(self, "_keep_in_fp32_modules", ()))
        for path in self._fp32_module_paths():
            try:
                module = self._resolve_attr_path(path)
            except AttributeError:
                continue
            if isinstance(module, nn.Module):
                module.to(dtype=torch.float32)

        for name, param in self.named_parameters(recurse=True):
            if torch.is_floating_point(param) and any(token in name for token in keep_tokens):
                param.data = param.data.to(dtype=torch.float32)

        for name, buf in self.named_buffers(recurse=True):
            if torch.is_floating_point(buf) and any(token in name for token in keep_tokens):
                buf.data = buf.data.to(dtype=torch.float32)

        for path in self._fp32_parameter_paths():
            try:
                param = self._resolve_attr_path(path)
            except AttributeError:
                continue
            if isinstance(param, nn.Parameter):
                param.data = param.data.to(dtype=torch.float32)


    def _normalize_pixels(self, input_pixels):
        if not isinstance(input_pixels, (list, tuple)):
            raise ValueError(f"input_pixels must be a list/tuple of per-view tensors, got {type(input_pixels)}")
        if len(input_pixels) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(input_pixels)}")
        out = []
        for tensor in input_pixels:
            if tensor.ndim != 4:
                raise ValueError(f"Each view tensor must be [B,3,H,W], got {tuple(tensor.shape)}")
            out.append(tensor)
        return out

    def _normalize_action_chunk(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 4:
            actions = actions.unsqueeze(0)
        if actions.ndim != 5:
            raise ValueError(f"actions must be [B,C,T,N,1] or [C,T,N,1], got {tuple(actions.shape)}")
        if actions.shape[-1] != 1:
            raise ValueError(f"Expected trailing singleton in actions, got {tuple(actions.shape)}")
        actions = actions.squeeze(-1).permute(0, 2, 3, 1).contiguous()
        return actions.view(actions.shape[0], actions.shape[1] * actions.shape[2], actions.shape[3])

    def _normalize_action_mask(self, action_is_pad: Optional[torch.Tensor], batch_size: int, horizon: int, device):
        if action_is_pad is None:
            return torch.ones(batch_size, horizon, dtype=torch.bool, device=device)
        if action_is_pad.ndim == 2:
            if batch_size != 1:
                raise ValueError(
                    f"action_is_pad with ndim=2 implies a single sample, got batch_size={batch_size} and shape={tuple(action_is_pad.shape)}"
                )
            action_is_pad = action_is_pad.unsqueeze(0)
        if action_is_pad.ndim != 3:
            raise ValueError(f"Unsupported action_is_pad shape: {tuple(action_is_pad.shape)}")
        if action_is_pad.shape[0] != batch_size:
            raise ValueError(
                f"action_is_pad batch mismatch: expected batch_size={batch_size}, got shape={tuple(action_is_pad.shape)}"
            )
        flat = action_is_pad.reshape(batch_size, -1)
        if flat.shape[1] != horizon:
            raise ValueError(f"Flattened action_is_pad horizon mismatch: expected {horizon}, got {flat.shape[1]}")
        return ~flat.to(device=device, dtype=torch.bool)

    def _normalize_action_feature_mask(
        self,
        actions_mask: Optional[torch.Tensor],
        batch_size: int,
        horizon: int,
        action_dim: int,
        device,
    ) -> torch.Tensor:
        if actions_mask is None:
            return torch.ones(batch_size, horizon, action_dim, dtype=torch.bool, device=device)
        if actions_mask.ndim == 4:
            actions_mask = actions_mask.unsqueeze(0)
        if actions_mask.ndim != 5:
            raise ValueError(f"actions_mask must be [B,C,T,N,1] or [C,T,N,1], got {tuple(actions_mask.shape)}")
        if actions_mask.shape[0] != batch_size:
            raise ValueError(
                f"actions_mask batch mismatch: expected batch_size={batch_size}, got shape={tuple(actions_mask.shape)}"
            )
        if actions_mask.shape[-1] != 1:
            raise ValueError(f"Expected trailing singleton in actions_mask, got {tuple(actions_mask.shape)}")
        mask = actions_mask.squeeze(-1).permute(0, 2, 3, 1).contiguous()
        mask = mask.view(mask.shape[0], mask.shape[1] * mask.shape[2], mask.shape[3])
        if mask.shape[1] != horizon or mask.shape[2] != action_dim:
            raise ValueError(
                f"Flattened actions_mask mismatch: expected [B,{horizon},{action_dim}], got {tuple(mask.shape)}"
            )
        return mask.to(device=device, dtype=torch.bool)

    def _expand_action_channel_mask(self, batch_size: int, horizon: int, device: torch.device) -> torch.Tensor:
        return self.action_channel_mask.to(device=device).view(1, 1, self.action_dim).expand(batch_size, horizon, -1)


    def _normalize_anchor_dino(
        self,
        anchor_dino_cls: Optional[torch.Tensor],
        anchor_dino_registers: Optional[torch.Tensor],
        anchor_dino_spatial: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if anchor_dino_cls is None or anchor_dino_spatial is None:
            raise ValueError("anchor_dino_cls and anchor_dino_spatial are required for the frozen current-frame anchor.")
        if anchor_dino_cls.ndim == 2:
            anchor_dino_cls = anchor_dino_cls.unsqueeze(0)
        if anchor_dino_spatial.ndim == 2:
            anchor_dino_spatial = anchor_dino_spatial.unsqueeze(0)
        if anchor_dino_cls.ndim != 3 or anchor_dino_spatial.ndim != 3:
            raise ValueError(
                "anchor DINO tensors must be [B,N,D] or [N,D], got "
                f"cls={tuple(anchor_dino_cls.shape)}, spatial={tuple(anchor_dino_spatial.shape)}"
            )
        if anchor_dino_cls.shape[-1] != self.teacher_dim or anchor_dino_spatial.shape[-1] != self.teacher_dim:
            raise ValueError(
                f"anchor DINO teacher dim mismatch: expected {self.teacher_dim}, "
                f"got cls={anchor_dino_cls.shape[-1]}, spatial={anchor_dino_spatial.shape[-1]}"
            )
        if anchor_dino_cls.shape[1] not in (0, self.num_views):
            raise ValueError(
                f"anchor_dino_cls token count mismatch: expected {self.num_views} (or 0 for VJEPA), got {anchor_dino_cls.shape[1]}"
            )
        expected_spatial = self.total_num_spatial_tokens
        if anchor_dino_spatial.shape[1] != expected_spatial:
            raise ValueError(
                f"anchor_dino_spatial token count mismatch: expected {expected_spatial}, got {anchor_dino_spatial.shape[1]}"
            )
        if anchor_dino_registers is None:
            anchor_dino_registers = anchor_dino_cls[:, 0:0, :]
        elif anchor_dino_registers.ndim == 2:
            anchor_dino_registers = anchor_dino_registers.unsqueeze(0)
        if anchor_dino_registers.ndim != 3 or anchor_dino_registers.shape[-1] != self.teacher_dim:
            raise ValueError(
                "anchor_dino_registers must be [B,N,D] or [N,D] with teacher_dim last axis, got "
                f"{tuple(anchor_dino_registers.shape)}"
            )
        expected_regs = self.num_views * self.num_register_tokens
        if self.anchor_keep_registers and anchor_dino_registers.shape[1] != expected_regs:
            raise ValueError(
                f"anchor_dino_registers token count mismatch: expected {expected_regs}, got {anchor_dino_registers.shape[1]}"
            )
        if (not self.anchor_keep_registers) and anchor_dino_registers.shape[1] not in (0, expected_regs):
            raise ValueError(
                "anchor_dino_registers token count mismatch for keep_registers=False: "
                f"expected 0 or {expected_regs}, got {anchor_dino_registers.shape[1]}"
            )
        anchor_dino_spatial = self._normalize_dino_spatial(anchor_dino_spatial)
        return anchor_dino_cls, anchor_dino_registers, anchor_dino_spatial

    def _normalize_context(
        self, context, context_mask, device, dtype, proprio=None
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Returns per-expert contexts and masks.

        In ``context`` proprio mode, proprio is appended only to the action-side
        context. This keeps future/video denoising from reading proprio through
        cross-attention and bypassing the compact state bottleneck.
        """
        if context is None:
            raise ValueError("text context (text_emb) is required but was not provided.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context.ndim != 3:
            raise ValueError(f"context must be [B,L,D] or [L,D], got {tuple(context.shape)}")
        context = context.to(device=device, dtype=dtype)

        if context_mask is not None:
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            context_mask = context_mask.to(device=device, dtype=torch.bool)

        video_context = context
        video_context_mask = context_mask
        action_context = context
        action_context_mask = context_mask

        if self._use_proprio_context_token():
            proprio_vec = self._normalize_action_proprio(
                proprio,
                batch_size=context.shape[0],
                device=device,
                dtype=dtype,
            )
            proprio_token = self._forward_module_fp32(
                self.context_proprio_in,
                proprio_vec,
                out_dtype=action_context.dtype,
            ).unsqueeze(1)
            action_context = torch.cat([action_context, proprio_token], dim=1)
            if action_context_mask is not None:
                proprio_mask = torch.ones(
                    (action_context_mask.shape[0], 1),
                    dtype=torch.bool,
                    device=action_context_mask.device,
                )
                action_context_mask = torch.cat([action_context_mask, proprio_mask], dim=1)

            if self.config.video_context_uses_proprio:
                video_context = torch.cat([video_context, proprio_token], dim=1)
                if video_context_mask is not None:
                    proprio_mask = torch.ones(
                        (video_context_mask.shape[0], 1),
                        dtype=torch.bool,
                        device=video_context_mask.device,
                    )
                    video_context_mask = torch.cat([video_context_mask, proprio_mask], dim=1)

        video_ctx = self.video_context_proj(video_context)
        action_ctx = self.action_context_proj(action_context)

        video_cross_attn_mask = None
        if video_context_mask is not None:
            video_cross_attn_mask = video_context_mask.unsqueeze(1).unsqueeze(1)

        action_cross_attn_mask = None
        if action_context_mask is not None:
            action_cross_attn_mask = action_context_mask.unsqueeze(1).unsqueeze(1)

        return video_ctx, action_ctx, video_cross_attn_mask, action_cross_attn_mask

    def _normalize_action_proprio(
        self,
        proprio: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self.proprio_dim <= 0:
            return None
        if proprio is None:
            raise ValueError(
                f"proprio is required (proprio_dim={self.proprio_dim}) but was not provided."
            )
        p = proprio.to(device=device, dtype=dtype)
        if p.ndim == 1:
            p = p.unsqueeze(0)
        if p.ndim != 2 or p.shape[-1] != self.proprio_dim:
            raise ValueError(f"proprio must be [B, {self.proprio_dim}], got {tuple(p.shape)}")
        if p.shape[0] != batch_size:
            raise ValueError(f"proprio batch mismatch: expected {batch_size}, got {p.shape[0]}")
        return p






    def _build_anchor_tokens(
        self,
        input_pixels,
        anchor_dino_cls: torch.Tensor,
        anchor_dino_registers: Optional[torch.Tensor],
        anchor_dino_spatial: torch.Tensor,
    ):
        anchor_dino_cls, anchor_dino_registers, anchor_dino_spatial = self._normalize_anchor_dino(
            anchor_dino_cls,
            anchor_dino_registers,
            anchor_dino_spatial,
        )
        view_token_groups = []
        anchor_cls_total = 0
        anchor_regs_total = 0
        anchor_spatial_total = 0
        for view_idx, _view_img in enumerate(input_pixels):
            cls_slice = anchor_dino_cls[:, view_idx : view_idx + 1, :]
            spatial_start = self.per_view_spatial_offsets[view_idx]
            spatial_end = spatial_start + self.per_view_spatial_tokens[view_idx]
            spatial_slice = anchor_dino_spatial[:, spatial_start:spatial_end, :]
            if self.anchor_keep_registers and self.num_register_tokens > 0:
                reg_start = view_idx * self.num_register_tokens
                reg_end = reg_start + self.num_register_tokens
                reg_slice = anchor_dino_registers[:, reg_start:reg_end, :]
            else:
                reg_slice = anchor_dino_registers[:, 0:0, :]

            frozen_cls = self._forward_module_fp32(self.frozen_anchor_proj, cls_slice)
            frozen_regs = self._forward_module_fp32(self.frozen_anchor_proj, reg_slice)
            frozen_spatial = self._forward_module_fp32(self.frozen_anchor_proj, spatial_slice)

            dev, dt = frozen_spatial.device, frozen_spatial.dtype
            if self.view_embed is not None:
                view_emb = self.view_embed.weight[view_idx].view(1, 1, self.hidden_dim).to(device=dev, dtype=dt)
            else:
                view_emb = torch.zeros(1, 1, self.hidden_dim, device=dev, dtype=dt)
            cls_type = self.token_type_embed.weight[0].view(1, 1, self.hidden_dim).to(device=dev, dtype=dt)
            reg_type = self.token_type_embed.weight[1].view(1, 1, self.hidden_dim).to(device=dev, dtype=dt)
            spatial_type = self.token_type_embed.weight[2].view(1, 1, self.hidden_dim).to(device=dev, dtype=dt)

            frozen_cls = frozen_cls + view_emb + cls_type
            if frozen_regs.shape[1] > 0:
                reg_slots = self.register_slot_embed.weight[: frozen_regs.shape[1]].unsqueeze(0).to(device=dev, dtype=dt)
                frozen_regs = frozen_regs + view_emb + reg_type + reg_slots
            frozen_spatial = frozen_spatial + view_emb + spatial_type

            view_token_groups.append(torch.cat([frozen_cls, frozen_regs, frozen_spatial], dim=1))
            anchor_cls_total += frozen_cls.shape[1]
            anchor_regs_total += frozen_regs.shape[1]
            anchor_spatial_total += frozen_spatial.shape[1]

        anchor = torch.cat(view_token_groups, dim=1)
        meta = {
            "anchor_cls": anchor_cls_total,
            "anchor_regs": anchor_regs_total,
            "anchor_spatial": anchor_spatial_total,
            "anchor_total": anchor.shape[1],
            "anchor_cls_per_view": (
                anchor_cls_total // self.num_views if self.num_views > 0 else 0
            ),
        }
        return anchor, meta

    def _video_rope_from_coords(
        self,
        frame_ids: torch.Tensor,
        row_ids: torch.Tensor,
        col_ids: torch.Tensor,
    ) -> torch.Tensor:
        max_frame = int(frame_ids.max().item()) if frame_ids.numel() else 0
        max_row = int(row_ids.max().item()) if row_ids.numel() else 0
        max_col = int(col_ids.max().item()) if col_ids.numel() else 0
        if (
            max_frame >= self._video_rope_f_freqs.shape[0]
            or max_row >= self._video_rope_h_freqs.shape[0]
            or max_col >= self._video_rope_w_freqs.shape[0]
        ):
            raise ValueError(
                "Video RoPE cache too short for requested coords: "
                f"frame={max_frame}, row={max_row}, col={max_col}"
            )
        device = frame_ids.device
        freqs = torch.cat(
            [
                self._video_rope_f_freqs.to(device=device)[frame_ids],
                self._video_rope_h_freqs.to(device=device)[row_ids],
                self._video_rope_w_freqs.to(device=device)[col_ids],
            ],
            dim=-1,
        )
        return freqs.view(freqs.shape[0], 1, -1)

    def _spatial_coord_ids(self, device: torch.device, view_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        grid_h, grid_w = self.per_view_grid_sizes[int(view_idx)]
        rows = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
        cols = torch.arange(grid_w, device=device).repeat(grid_h)
        return rows, cols

    def _view_origin_coord(self, view_idx: int) -> tuple[int, int]:
        if self.view_rope_layout == "horizontal":
            return 0, sum(grid_w for _grid_h, grid_w in self.per_view_grid_sizes[: int(view_idx)])
        if self.view_rope_layout == "vertical":
            return sum(grid_h for grid_h, _grid_w in self.per_view_grid_sizes[: int(view_idx)]), 0
        if int(view_idx) == 0:
            top_w = self.per_view_grid_sizes[0][1]
            return 0, max(0, (self.virtual_grid_w - top_w) // 2)
        top_h = self.per_view_grid_sizes[0][0]
        left_col = sum(grid_w for _grid_h, grid_w in self.per_view_grid_sizes[1 : int(view_idx)])
        return top_h, left_col

    def _spatial_coord_ids_for_view(self, view_idx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rows, cols = self._spatial_coord_ids(device, view_idx)
        row0, col0 = self._view_origin_coord(view_idx)
        return rows + int(row0), cols + int(col0)

    def _build_anchor_video_freqs(
        self,
        device: torch.device,
        anchor_cls_per_view: int = 1,
    ) -> torch.Tensor:
        # Per-view layout: [cls+regs (prefix_len)] [frozen_spatial]
        # With multi-view inputs we assign each view a unique virtual 2D offset.
        # This mirrors "concatenate then RoPE" behavior without a VAE grid.
        if anchor_cls_per_view < 0:
            raise ValueError(f"anchor_cls_per_view must be >= 0, got {anchor_cls_per_view}")
        if anchor_cls_per_view > 1:
            raise ValueError(f"anchor_cls_per_view must be <= 1, got {anchor_cls_per_view}")
        frame_chunks = []
        row_chunks = []
        col_chunks = []
        for view_idx in range(self.num_views):
            row0, col0 = self._view_origin_coord(view_idx)
            prefix_len = int(anchor_cls_per_view) + (self.num_register_tokens if self.anchor_keep_registers else 0)
            frame_chunks.append(torch.zeros(prefix_len, device=device, dtype=torch.long))
            row_chunks.append(torch.full((prefix_len,), int(row0), device=device, dtype=torch.long))
            col_chunks.append(torch.full((prefix_len,), int(col0), device=device, dtype=torch.long))
            spatial_rows, spatial_cols = self._spatial_coord_ids_for_view(view_idx, device)
            frame_chunks.append(torch.zeros(self.per_view_spatial_tokens[view_idx], device=device, dtype=torch.long))
            row_chunks.append(spatial_rows)
            col_chunks.append(spatial_cols)
        return self._video_rope_from_coords(
            torch.cat(frame_chunks, dim=0),
            torch.cat(row_chunks, dim=0),
            torch.cat(col_chunks, dim=0),
        ).to(device=device)

    def _dense_step_coord_ids(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        row_chunks = []
        col_chunks = []
        for view_idx in range(self.num_views):
            row0, col0 = self._view_origin_coord(view_idx)
            if self.keep_registers and self.num_register_tokens > 0:
                row_chunks.append(torch.full((self.num_register_tokens,), int(row0), device=device, dtype=torch.long))
                col_chunks.append(torch.full((self.num_register_tokens,), int(col0), device=device, dtype=torch.long))
            spatial_rows, spatial_cols = self._spatial_coord_ids_for_view(view_idx, device)
            row_chunks.append(spatial_rows)
            col_chunks.append(spatial_cols)
        return torch.cat(row_chunks, dim=0), torch.cat(col_chunks, dim=0)

    def _build_future_video_freqs(
        self,
        device: torch.device,
        future_steps: Optional[int] = None,
        include_cls: Optional[bool] = None,
        include_dense: Optional[bool] = None,
        future_frame_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        future_steps = self.future_blocks if future_steps is None else int(future_steps)
        include_cls = self.enable_future_cls if include_cls is None else bool(include_cls)
        include_dense = self.enable_future_dense if include_dense is None else bool(include_dense)
        if future_steps <= 0:
            return torch.empty(0, 1, self.config.head_dim // 2, device=device, dtype=self._video_rope_f_freqs.dtype)

        # Base temporal frame IDs: provided offsets take priority over the default 1..T.
        if future_frame_offsets is not None:
            base_frame_ids = future_frame_offsets.to(device=device, dtype=torch.long)
            if base_frame_ids.ndim > 1:
                base_frame_ids = base_frame_ids[0]
        else:
            base_frame_ids = torch.arange(1, future_steps + 1, device=device, dtype=torch.long)

        frame_chunks = []
        row_chunks = []
        col_chunks = []
        if include_cls:
            cls_per_step = self.cls_tokens_per_step
            frame_ids = base_frame_ids.repeat_interleave(cls_per_step)
            cls_rows = []
            cls_cols = []
            for view_idx in range(self.num_views):
                row0, col0 = self._view_origin_coord(view_idx)
                cls_rows.append(int(row0))
                cls_cols.append(int(col0))
            cls_rows = torch.tensor(cls_rows, device=device, dtype=torch.long)
            cls_cols = torch.tensor(cls_cols, device=device, dtype=torch.long)
            frame_chunks.append(frame_ids)
            row_chunks.append(cls_rows.repeat(future_steps))
            col_chunks.append(cls_cols.repeat(future_steps))

        if include_dense:
            rows_per_step, cols_per_step = self._dense_step_coord_ids(device)
            frame_ids = base_frame_ids.repeat_interleave(self.dense_tokens_per_step)
            frame_chunks.append(frame_ids)
            row_chunks.append(rows_per_step.repeat(future_steps))
            col_chunks.append(cols_per_step.repeat(future_steps))

        if not frame_chunks:
            return torch.empty(0, 1, self.config.head_dim // 2, device=device, dtype=self._video_rope_f_freqs.dtype)
        return self._video_rope_from_coords(
            torch.cat(frame_chunks, dim=0),
            torch.cat(row_chunks, dim=0),
            torch.cat(col_chunks, dim=0),
        ).to(device=device)

    def _build_action_freqs(self, horizon: int, device: torch.device, *, action_prefix_len: int = 0) -> torch.Tensor:
        action_prefix_len = int(action_prefix_len)
        if horizon < action_prefix_len:
            raise ValueError(f"action_prefix_len={action_prefix_len} exceeds horizon={horizon}")
        body_horizon = int(horizon) - action_prefix_len
        if body_horizon > self._action_rope_freqs.shape[0]:
            raise ValueError(f"action horizon {body_horizon} exceeds RoPE cache {self._action_rope_freqs.shape[0]}")
        body_freqs = self._action_rope_freqs[:body_horizon].view(body_horizon, 1, -1).to(device=device)
        if action_prefix_len <= 0:
            return body_freqs
        prefix_freqs = torch.zeros(action_prefix_len, 1, body_freqs.shape[-1], device=device, dtype=body_freqs.dtype)
        return torch.cat([prefix_freqs, body_freqs], dim=0)

    def _dense_slot_embeddings(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokens = []
        reg_type = self.token_type_embed.weight[4].view(1, 1, self.hidden_dim).to(device=device, dtype=dtype)
        spatial_type = self.token_type_embed.weight[5].view(1, 1, self.hidden_dim).to(device=device, dtype=dtype)

        for view_idx in range(self.num_views):
            if self.view_embed is not None:
                view_emb = self.view_embed.weight[view_idx].view(1, 1, self.hidden_dim).to(device=device, dtype=dtype)
            else:
                view_emb = torch.zeros(1, 1, self.hidden_dim, device=device, dtype=dtype)

            if self.keep_registers and self.num_register_tokens > 0:
                reg_slots = self.register_slot_embed.weight[: self.num_register_tokens].unsqueeze(0).to(device=device, dtype=dtype)
                tokens.append(view_emb + reg_type + reg_slots)

            tokens.append((view_emb + spatial_type).expand(1, self.per_view_spatial_tokens[view_idx], -1))

        return torch.cat(tokens, dim=1).expand(batch_size, -1, -1)

    def _cls_slot_embeddings(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tokens = []
        cls_type = self.token_type_embed.weight[3].view(1, 1, self.hidden_dim).to(device=device, dtype=dtype)
        for view_idx in range(self.num_views):
            if self.view_embed is not None:
                view_emb = self.view_embed.weight[view_idx].view(1, 1, self.hidden_dim).to(device=device, dtype=dtype)
            else:
                view_emb = torch.zeros(1, 1, self.hidden_dim, device=device, dtype=dtype)
            tokens.append(view_emb + cls_type)
        return torch.cat(tokens, dim=1).expand(batch_size, -1, -1)

    def _build_future_video_tokens(self, dense_noisy, cls_noisy):
        chunks = []
        slices: Dict[str, slice] = {}
        cursor = 0
        if self.enable_future_cls and cls_noisy is not None:
            batch_size, future_steps, num_cls, _ = cls_noisy.shape
            if num_cls != self.cls_tokens_per_step:
                raise ValueError(f"dino_cls tokens per step mismatch: expected {self.cls_tokens_per_step}, got {num_cls}")
            cls_tokens = self._forward_module_fp32(self.future_cls_in, cls_noisy)
            cls_slot = self._cls_slot_embeddings(batch_size, cls_noisy.device, cls_noisy.dtype).unsqueeze(1).expand(batch_size, future_steps, -1, -1)
            cls_tokens = (cls_tokens + cls_slot).view(batch_size, -1, self.hidden_dim)
            chunks.append(cls_tokens)
            slices["cls"] = slice(cursor, cursor + cls_tokens.shape[1])
            cursor += cls_tokens.shape[1]

        if self.enable_future_dense and dense_noisy is not None:
            batch_size, future_steps, num_dense, _ = dense_noisy.shape
            if num_dense != self.dense_tokens_per_step:
                raise ValueError(f"dino_dense tokens per step mismatch: expected {self.dense_tokens_per_step}, got {num_dense}")
            dense_tokens = self._forward_module_fp32(self.future_dense_in, dense_noisy)
            dense_slot = self._dense_slot_embeddings(batch_size, dense_noisy.device, dense_noisy.dtype).unsqueeze(1).expand(batch_size, future_steps, -1, -1)
            dense_tokens = (dense_tokens + dense_slot).view(batch_size, -1, self.hidden_dim)
            chunks.append(dense_tokens)
            slices["dense"] = slice(cursor, cursor + dense_tokens.shape[1])
            cursor += dense_tokens.shape[1]

        if not chunks:
            return None, slices
        return torch.cat(chunks, dim=1), slices

    def _use_proprio_action_prefix(self) -> bool:
        return self.proprio_dim > 0 and self.proprio_injection_mode == "action_prefix"

    def _use_proprio_context_token(self) -> bool:
        return self.proprio_dim > 0 and self.proprio_injection_mode == "context"

    def _action_proprio_prefix_len(self) -> int:
        return 1 if self._use_proprio_action_prefix() else 0

    def _build_action_tokens(self, action_noisy: torch.Tensor, proprio: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, horizon, _ = action_noisy.shape
        if horizon != self.future_blocks * self.action_per_frame:
            raise ValueError(f"action horizon mismatch: expected {self.future_blocks * self.action_per_frame}, got {horizon}")
        action_tokens = self._forward_module_fp32(self.action_in, action_noisy)  # [B, horizon, action_hidden_dim]
        action_type = self.action_type_embed.to(device=action_tokens.device, dtype=action_tokens.dtype)
        action_tokens = action_tokens + action_type
        if not self._use_proprio_action_prefix():
            return action_tokens
        proprio_vec = self._normalize_action_proprio(
            proprio,
            batch_size=batch_size,
            device=action_noisy.device,
            dtype=action_noisy.dtype,
        )
        proprio_token = self._forward_module_fp32(self.action_proprio_in, proprio_vec, out_dtype=action_tokens.dtype).unsqueeze(1)
        proprio_type = self.action_proprio_type_embed.to(device=action_tokens.device, dtype=action_tokens.dtype)
        return torch.cat([proprio_token + proprio_type, action_tokens], dim=1)

    def _time_mod(self, timestep: torch.Tensor, hidden_dim: int, mlp: nn.Module) -> torch.Tensor:
        t_emb = sinusoidal_embedding_1d(hidden_dim, timestep, out_dtype=torch.float32)
        return self._forward_module_fp32(mlp, t_emb)

    def _action_token_time_mod(
        self,
        timestep: torch.Tensor,
        *,
        action_body_len: int,
        action_prefix_len: int = 0,
    ) -> torch.Tensor:
        if timestep.ndim == 1:
            batch_size = timestep.shape[0]
            body_timestep = timestep.unsqueeze(1).expand(-1, int(action_body_len))
        elif timestep.ndim == 2:
            batch_size = timestep.shape[0]
            if timestep.shape[1] != int(action_body_len):
                raise ValueError(
                    "Per-action timestep shape mismatch: "
                    f"expected [B,{int(action_body_len)}], got {tuple(timestep.shape)}"
                )
            body_timestep = timestep
        else:
            raise ValueError(f"timestep must be [B] or [B,T], got {tuple(timestep.shape)}")
        if action_prefix_len > 0:
            token_timestep = torch.cat(
                [
                    torch.zeros(
                        batch_size,
                        int(action_prefix_len),
                        device=timestep.device,
                        dtype=timestep.dtype,
                    ),
                    body_timestep,
                ],
                dim=1,
            )
        else:
            token_timestep = body_timestep
        t_emb = sinusoidal_embedding_1d(self.action_hidden_dim, token_timestep.reshape(-1), out_dtype=torch.float32)
        return self._forward_module_fp32(self.action_time_mlp, t_emb).view(batch_size, token_timestep.shape[1], -1)


    def _video_token_time_mod(
        self,
        timestep: torch.Tensor,
        anchor_len: int,
        future_video_len: int,
    ) -> torch.Tensor:
        batch_size = timestep.shape[0]
        token_timestep = torch.zeros(
            batch_size,
            int(anchor_len) + int(future_video_len),
            device=timestep.device,
            dtype=timestep.dtype,
        )
        if future_video_len > 0:
            token_timestep[:, int(anchor_len) :] = timestep.unsqueeze(1)
        t_emb = sinusoidal_embedding_1d(self.hidden_dim, token_timestep.reshape(-1), out_dtype=torch.float32)
        return self._forward_module_fp32(self.video_time_mlp, t_emb).view(batch_size, token_timestep.shape[1], -1)

    def _decode_future_outputs(self, video_out: torch.Tensor, future_slices: Dict[str, slice], batch_size: int, future_steps: int):
        outputs = {"dense": None, "cls": None}
        if "cls" in future_slices:
            cls_out = video_out[:, future_slices["cls"]]
            outputs["cls"] = self._forward_module_fp32(self.cls_head, cls_out).view(
                batch_size, future_steps, self.cls_tokens_per_step, self.teacher_dim
            )
        if "dense" in future_slices:
            dense_out = video_out[:, future_slices["dense"]]
            outputs["dense"] = self._forward_module_fp32(self.dense_head, dense_out).view(
                batch_size, future_steps, self.dense_tokens_per_step, self.teacher_dim
            )
        return outputs


    def _action_denoise_step(
        self,
        action_state: torch.Tensor,
        timestep: torch.Tensor,
        delta: torch.Tensor,
        action_channel_mask_f: torch.Tensor,
        action_freqs: torch.Tensor,
        video_cache_k: tuple[torch.Tensor, ...],
        video_cache_v: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_context: Optional[torch.Tensor],
        action_context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor],
        action_prefix_len: int,
    ) -> torch.Tensor:
        """One fixed-shape action denoising step for inference compilation.

        Anchor encoding, RoPE/mask construction, and video-cache prefill stay
        outside this method because they contain dynamic Python metadata.  The
        repeated MoT action step is tensor-only for a fixed deployment shape.
        """
        action_tmod = self._action_token_time_mod(
            timestep,
            action_body_len=action_state.shape[1],
            action_prefix_len=action_prefix_len,
        )
        action_state = action_state * action_channel_mask_f
        action_tokens = self._build_action_tokens(action_state, proprio=proprio)
        action_out = self.mot.forward_action_with_video_cache_flat(
            action_tokens=action_tokens,
            action_time_mod=action_tmod,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            action_freqs=action_freqs,
            action_context=action_context,
            action_context_mask=action_context_mask,
        )
        pred_velocity = self._forward_module_fp32(
            self.action_head,
            action_out[:, action_prefix_len:],
        ) * action_channel_mask_f
        return self.flow.step(action_state, pred_velocity, delta) * action_channel_mask_f

    def _normal_tensor_for_cuda_graph(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Materialize action-graph inputs after a compiled V-JEPA forward."""
        if tensor is not None and tensor.is_inference():
            return tensor.detach().clone()
        return tensor

    def _normal_video_cache_for_cuda_graph(self, video_cache):
        return [
            {
                key: self._normal_tensor_for_cuda_graph(value)
                if torch.is_tensor(value)
                else value
                for key, value in layer_cache.items()
            }
            for layer_cache in video_cache
        ]

    def _normal_flat_video_cache_for_cuda_graph(self, video_cache):
        """Normalize the fixed (K-cache, V-cache) pytree at graph boundary."""
        return tuple(
            tuple(self._normal_tensor_for_cuda_graph(value) for value in cache_part)
            for cache_part in video_cache
        )

    def _reset_inference_profile(self) -> None:
        """Start a CUDA-event profile for one inference request when enabled."""
        if getattr(self, "_inference_profile_enabled", False):
            self._inference_profile_events = {}

    def _profile_cuda_start(self, name: str, device: torch.device):
        if not getattr(self, "_inference_profile_enabled", False) or device.type != "cuda":
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(device))
        return event

    def _profile_cuda_end(self, name: str, start_event) -> None:
        if start_event is None:
            return
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._inference_profile_events[name] = (start_event, end_event)

    def get_inference_profile_ms(self) -> dict[str, float]:
        """Return completed event durations; caller must have synchronized first."""
        return {
            name: float(start.elapsed_time(end))
            for name, (start, end) in getattr(self, "_inference_profile_events", {}).items()
        }

    @torch.no_grad()
    def infer_action(
        self,
        input_pixels,
        anchor_dino_cls: torch.Tensor,
        anchor_dino_spatial: torch.Tensor,
        anchor_dino_registers: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        generator: Optional[torch.Generator] = None,
        joint_future_denoising: Optional[bool] = None,
        future_semantic_steps: Optional[int] = None,
        action_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_pixels = self._normalize_pixels(input_pixels)
        device = input_pixels[0].device
        self._reset_inference_profile()
        dtype = self._runtime_compute_dtype(device, input_pixels[0].dtype)
        batch_size = input_pixels[0].shape[0]
        horizon = self.future_blocks * self.action_per_frame

        anchor_event = self._profile_cuda_start("anchor_tokens_ms", device)
        anchor_tokens, anchor_meta = self._build_anchor_tokens(
            input_pixels,
            anchor_dino_cls=anchor_dino_cls.to(device=device, dtype=torch.float32),
            anchor_dino_registers=(
                anchor_dino_registers.to(device=device, dtype=torch.float32)
                if anchor_dino_registers is not None
                else None
            ),
            anchor_dino_spatial=anchor_dino_spatial.to(device=device, dtype=torch.float32),
        )
        self._profile_cuda_end("anchor_tokens_ms", anchor_event)
        video_ctx, action_ctx, video_cross_attn_mask, action_cross_attn_mask = self._normalize_context(context, context_mask, device, dtype, proprio=proprio)
        action_channel_mask_f = self._expand_action_channel_mask(batch_size, horizon, device).to(dtype=dtype)
        action_state = torch.randn(batch_size, horizon, self.action_dim, device=device, dtype=dtype, generator=generator)
        action_state = action_state * action_channel_mask_f
        committed_prefix = None
        committed_len = 0
        if action_prefix is not None:
            committed_prefix = action_prefix.to(device=device, dtype=dtype)
            if committed_prefix.ndim != 3 or committed_prefix.shape[0] != batch_size or committed_prefix.shape[2] != self.action_dim:
                raise ValueError(
                    "action_prefix must be [B,K,action_dim], got "
                    f"{tuple(committed_prefix.shape)} for B={batch_size}, action_dim={self.action_dim}"
                )
            committed_len = int(committed_prefix.shape[1])
            if not 0 < committed_len < horizon:
                raise ValueError(f"action_prefix length must be in [1,{horizon - 1}], got {committed_len}")
            committed_prefix = committed_prefix * action_channel_mask_f[:, :committed_len]
            action_state[:, :committed_len] = committed_prefix
        action_prefix_len = self._action_proprio_prefix_len()
        action_freqs = self._build_action_freqs(horizon + action_prefix_len, device, action_prefix_len=action_prefix_len)
        timesteps, deltas = self.flow.build_inference_schedule(
            num_inference_steps,
            device=device,
            dtype=dtype,
        )

        if joint_future_denoising is None:
            joint_future_denoising = bool(self.config.default_joint_future_denoising)

        if joint_future_denoising:
            future_steps = self.future_blocks if future_semantic_steps is None else int(future_semantic_steps)
            if future_steps <= 0:
                raise ValueError(f"future_semantic_steps must be positive, got {future_steps}")
            has_future_cls = bool(self.enable_future_cls)
            has_future_dense = bool(self.enable_future_dense)

            # Build independent inference schedules for action and future streams,
            # matching FastWAM's separate video/action schedulers.
            sem_timesteps, sem_deltas = self.flow.build_inference_schedule(
                num_inference_steps, device=device, dtype=dtype,
            )
            act_timesteps, act_deltas = self.flow.build_inference_schedule(
                num_inference_steps, device=device, dtype=dtype,
            )
            dense_state = (
                torch.randn(
                    batch_size,
                    future_steps,
                    self.dense_tokens_per_step,
                    self.teacher_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                if self.enable_future_dense
                else None
            )
            cls_state = (
                torch.randn(
                    batch_size,
                    future_steps,
                    self.cls_tokens_per_step,
                    self.teacher_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                if self.enable_future_cls
                else None
            )
            anchor_freqs = self._build_anchor_video_freqs(
                device,
                anchor_cls_per_view=int(anchor_meta.get("anchor_cls_per_view", 1)),
            )
            future_freqs = self._build_future_video_freqs(
                device,
                future_steps,
                include_cls=has_future_cls,
                include_dense=has_future_dense,
            )

            for step_idx in range(num_inference_steps):
                t_sem = sem_timesteps[step_idx].expand(batch_size)
                delta_sem = sem_deltas[step_idx].expand(batch_size)
                t_act = act_timesteps[step_idx].expand(batch_size)
                if committed_len:
                    t_act = t_act.unsqueeze(1).expand(-1, horizon).clone()
                    t_act[:, :committed_len] = 0
                delta_act = act_deltas[step_idx].expand(batch_size)
                action_tmod = self._action_token_time_mod(
                    t_act,
                    action_body_len=horizon,
                    action_prefix_len=action_prefix_len,
                )
                action_state = action_state * action_channel_mask_f
                action_tokens = self._build_action_tokens(action_state, proprio=proprio)
                future_video_tokens, future_slices = self._build_future_video_tokens(dense_state, cls_state)
                if future_video_tokens is None:
                    video_tokens = anchor_tokens
                    video_freqs = anchor_freqs
                    future_video_len = 0
                else:
                    video_tokens = torch.cat([anchor_tokens, future_video_tokens], dim=1)
                    if future_freqs.shape[0] != future_video_tokens.shape[1]:
                        raise ValueError(
                            "Future video token/frequency length mismatch during inference: "
                            f"tokens={future_video_tokens.shape[1]}, freqs={future_freqs.shape[0]}, "
                            f"include_cls={has_future_cls}, include_dense={has_future_dense}, future_steps={future_steps}"
                        )
                    video_freqs = torch.cat([anchor_freqs, future_freqs], dim=0)
                    future_video_len = future_video_tokens.shape[1]
                self_mask = build_fastwam_style_joint_mask(
                    batch_size,
                    anchor_meta["anchor_total"],
                    future_video_len,
                    action_tokens.shape[1],
                    device,
                    action_prefix_len=action_prefix_len,
                    action_attends_future_video=self.config.action_attends_future_video,
                )
                video_tmod = self._video_token_time_mod(
                    t_sem,
                    anchor_len=anchor_meta["anchor_total"],
                    future_video_len=future_video_len,
                )
                outputs = self.mot(
                    expert_tokens={"video": video_tokens, "action": action_tokens},
                    expert_time_mod={"video": video_tmod, "action": action_tmod},
                    self_attn_mask=self_mask,
                    expert_freqs={"video": video_freqs, "action": action_freqs},
                    expert_context={"video": video_ctx, "action": action_ctx},
                    expert_context_mask={"video": video_cross_attn_mask, "action": action_cross_attn_mask},
                )
                pred_velocity = self._forward_module_fp32(self.action_head, outputs["action"][:, action_prefix_len:]) * action_channel_mask_f
                action_state = self.flow.step(action_state, pred_velocity, delta_act) * action_channel_mask_f
                if committed_prefix is not None:
                    action_state[:, :committed_len] = committed_prefix

                if future_video_len > 0:
                    future_video_out = outputs["video"][:, anchor_meta["anchor_total"] :]
                    future_preds = self._decode_future_outputs(
                        future_video_out,
                        future_slices,
                        batch_size,
                        future_steps,
                    )
                    if dense_state is not None and future_preds["dense"] is not None:
                        dense_state = self.flow.step(dense_state, future_preds["dense"], delta_sem)
                    if cls_state is not None and future_preds["cls"] is not None:
                        cls_state = self.flow.step(cls_state, future_preds["cls"], delta_sem)

            return action_state * action_channel_mask_f

        self_mask = build_fastwam_style_joint_mask(
            batch_size,
            anchor_meta["anchor_total"],
            0,
            horizon + action_prefix_len,
            device,
            action_prefix_len=action_prefix_len,
            action_attends_future_video=self.config.action_attends_future_video,
        )
        video_freqs = self._build_anchor_video_freqs(
            device,
            anchor_cls_per_view=int(anchor_meta.get("anchor_cls_per_view", 1)),
        )
        zero_t = torch.zeros(batch_size, device=device, dtype=dtype)
        video_tmod = self._time_mod(zero_t, self.hidden_dim, self.video_time_mlp)
        prefill_event = self._profile_cuda_start("video_prefill_ms", device)
        video_prefill_fn = getattr(self, "_video_prefill_fn", self.mot.prefill_video_cache_flat)
        video_cache = video_prefill_fn(
            video_tokens=anchor_tokens,
            video_time_mod=video_tmod,
            video_freqs=video_freqs,
            video_context=video_ctx,
            video_context_mask=video_cross_attn_mask,
        )
        self._profile_cuda_end("video_prefill_ms", prefill_event)

        # A compiled frozen vision encoder may propagate inference tensors into
        # this cache.  CUDA Graph capture forbids in-place updates to them.
        action_state = self._normal_tensor_for_cuda_graph(action_state)
        timesteps = self._normal_tensor_for_cuda_graph(timesteps)
        deltas = self._normal_tensor_for_cuda_graph(deltas)
        action_channel_mask_f = self._normal_tensor_for_cuda_graph(action_channel_mask_f)
        action_freqs = self._normal_tensor_for_cuda_graph(action_freqs)
        video_cache = self._normal_flat_video_cache_for_cuda_graph(video_cache)
        self_mask = self._normal_tensor_for_cuda_graph(self_mask)
        action_ctx = self._normal_tensor_for_cuda_graph(action_ctx)
        action_cross_attn_mask = self._normal_tensor_for_cuda_graph(action_cross_attn_mask)
        proprio = self._normal_tensor_for_cuda_graph(proprio)

        denoise_event = self._profile_cuda_start("action_denoise_ms", device)
        action_denoise_step_fn = getattr(
            self,
            "_action_denoise_step_fns_by_committed_len",
            {},
        ).get(committed_len, self._action_denoise_step_fn)
        for step_idx in range(num_inference_steps):
            # Keep the action-step input rank identical for normal and RTC
            # requests.  A scalar timestep is semantically broadcast across
            # the horizon; RTC overwrites its fixed committed prefix with 0.
            t_cur = timesteps[step_idx].expand(batch_size).unsqueeze(1).expand(-1, horizon)
            if committed_len:
                t_cur = t_cur.clone()
                t_cur[:, :committed_len] = 0
            delta = deltas[step_idx].expand(batch_size)
            # Compiled CUDA Graph outputs are graph-owned storage.  Clone at
            # the eager boundary before feeding this state to the next denoise
            # iteration, otherwise the next graph replay overwrites its input.
            action_state = action_denoise_step_fn(
                action_state,
                t_cur,
                delta,
                action_channel_mask_f,
                action_freqs,
                video_cache[0],
                video_cache[1],
                self_mask,
                anchor_meta["anchor_total"],
                action_ctx,
                action_cross_attn_mask,
                proprio,
                action_prefix_len,
            ).clone()
            if committed_prefix is not None:
                action_state[:, :committed_len] = committed_prefix
        self._profile_cuda_end("action_denoise_ms", denoise_event)
        return action_state * action_channel_mask_f
