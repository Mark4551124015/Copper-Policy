from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantic_mot_model import SemanticFastWAM, SemanticMoTConfig
from .semantic_mot_blocks import build_fastwam_style_joint_mask
from .compact_control_latent import CompactControlLatent, build_ema_compactor
from .vjepa_hub import load_vjepa_model


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


_TRAINED_DINO_V2_REGISTERS_LARGE = "facebook/dinov2-with-registers-large"


def _build_policy_dino_encoder(model_name_or_path: str) -> nn.Module:
    """Construct the trainable DINO backbone without fetching Hub weights.

    RealBot checkpoints contain the fine-tuned DINO parameters. Downloading
    public pretrained weights before loading such a checkpoint wastes time and
    makes offline deployment fail. A local model directory remains supported
    for custom architectures; the released DINOv2-registers-large identifier
    is built from its fixed public architecture config.
    """
    from transformers import AutoConfig, AutoModel, Dinov2WithRegistersConfig
    from .backbone_presets import VISION_ENCODER_PRESETS, vision_cache_dir, vision_cache_ready

    source = Path(model_name_or_path).expanduser()
    if source.is_dir():
        if torch.get_default_device().type == "meta":
            return AutoModel.from_config(AutoConfig.from_pretrained(str(source), local_files_only=True))
        return AutoModel.from_pretrained(str(source), local_files_only=True, torch_dtype=torch.float32)
    for name, preset in VISION_ENCODER_PRESETS.items():
        if model_name_or_path == preset.model_id and vision_cache_ready(name):
            cache = str(vision_cache_dir(name))
            if torch.get_default_device().type == "meta":
                return AutoModel.from_config(AutoConfig.from_pretrained(cache, local_files_only=True))
            return AutoModel.from_pretrained(
                cache, local_files_only=True, torch_dtype=torch.float32
            )
    if model_name_or_path != _TRAINED_DINO_V2_REGISTERS_LARGE:
        raise ValueError(
            "Trainable DINO must be a local model directory or "
            f"{_TRAINED_DINO_V2_REGISTERS_LARGE!r}; got {model_name_or_path!r}. "
            "Remote Hugging Face initialization is disabled because the checkpoint "
            "must provide the trained DINO weights."
        )

    # Exact architecture of facebook/dinov2-with-registers-large/config.json.
    config = Dinov2WithRegistersConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        image_size=518,
        patch_size=14,
        num_register_tokens=4,
        mlp_ratio=4,
        qkv_bias=True,
        layer_norm_eps=1e-6,
        layerscale_value=1.0,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        use_swiglu_ffn=False,
    )
    return AutoModel.from_config(config).to(dtype=torch.float32)


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


class AbstractConditionedGridSampler(nn.Module):
    """Sample current spatial tokens either per-view or from a stitched global canvas."""

    def __init__(
        self,
        *,
        source_dim: int,
        hidden_dim: int,
        num_views: int,
        num_tokens: int,
        num_tokens_per_view: Optional[tuple[int, ...]] = None,
        per_view_grid_sizes: tuple[tuple[int, int], ...],
        view_rope_layout: str = "horizontal",
        global_canvas: bool = False,
        share_coord_head: bool = False,
        coord_embed: bool = True,
    ):
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.num_tokens = int(num_tokens)
        self.global_canvas = bool(global_canvas)
        self.share_coord_head = bool(share_coord_head)
        self.per_view_grid_sizes = tuple((int(h), int(w)) for h, w in per_view_grid_sizes)
        if len(self.per_view_grid_sizes) != self.num_views:
            raise ValueError(
                "per_view_grid_sizes must provide one (grid_h, grid_w) pair per view, "
                f"got {len(self.per_view_grid_sizes)} for num_views={self.num_views}"
            )
        self.view_rope_layout = str(view_rope_layout or "horizontal").strip().lower()
        if self.view_rope_layout not in {"horizontal", "vertical", "robotwin_tshape"}:
            raise ValueError(
                f"Unsupported view_rope_layout={view_rope_layout!r}. "
                "Expected one of: horizontal, vertical, robotwin_tshape."
            )
        if self.global_canvas:
            if num_tokens_per_view is not None:
                raise ValueError(
                    "global_canvas grid sampler expects a single global token budget; "
                    "set grid_sampler_num_tokens_per_view=None."
                )
            self.num_tokens_per_view = None
        elif num_tokens_per_view is None:
            self.num_tokens_per_view = tuple(self.num_tokens for _ in range(self.num_views))
        else:
            self.num_tokens_per_view = tuple(int(v) for v in num_tokens_per_view)
            if len(self.num_tokens_per_view) != self.num_views:
                raise ValueError(
                    "num_tokens_per_view must provide one token count per view, "
                    f"got {len(self.num_tokens_per_view)} for num_views={self.num_views}"
                )
            if any(v <= 0 for v in self.num_tokens_per_view):
                raise ValueError(
                    f"num_tokens_per_view must be positive for all views, got {self.num_tokens_per_view}"
                )
        if self.share_coord_head and not self.global_canvas:
            if len(set(self.num_tokens_per_view)) != 1:
                raise ValueError(
                    "grid_sampler_share_coord_head requires an equal token budget for every view, "
                    f"got {self.num_tokens_per_view}"
                )
        self.per_view_spatial_tokens = tuple(grid_h * grid_w for grid_h, grid_w in self.per_view_grid_sizes)
        self.per_view_offsets = []
        cursor = 0
        for tokens_per_view in self.per_view_spatial_tokens:
            self.per_view_offsets.append(cursor)
            cursor += int(tokens_per_view)
        self.per_view_offsets = tuple(self.per_view_offsets)
        self.total_spatial_tokens = int(cursor)
        self.coord_embed_enabled = bool(coord_embed)
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

        if self.global_canvas or self.share_coord_head:
            self.coord_head = nn.Sequential(
                nn.Linear(self.source_dim, 512),
                nn.ReLU(),
                nn.Linear(
                    512,
                    int(self.num_tokens if self.global_canvas else self.num_tokens_per_view[0]) * 2,
                ),
                nn.Sigmoid(),
            )
            self.coord_heads = None
        else:
            self.coord_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.source_dim, 512),
                        nn.ReLU(),
                        nn.Linear(512, int(view_tokens) * 2),
                        nn.Sigmoid(),
                    )
                    for view_tokens in self.num_tokens_per_view
                ]
            )
            self.coord_head = None
        self.sampled_visual_proj = nn.Linear(self.source_dim, self.hidden_dim)
        if self.global_canvas:
            self.view_type_embed_for_coord = None
            self.view_type_embed_for_token = None
        else:
            self.view_type_embed_for_coord = nn.Embedding(self.num_views, self.source_dim)
            self.view_type_embed_for_token = nn.Embedding(self.num_views, self.hidden_dim)
            nn.init.zeros_(self.view_type_embed_for_coord.weight)
            nn.init.zeros_(self.view_type_embed_for_token.weight)
        if self.coord_embed_enabled:
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Linear(256, self.hidden_dim),
            )
        else:
            self.coord_encoder = None

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

    def _build_stitched_feature_map(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        stitched = spatial_tokens.new_zeros(
            spatial_tokens.shape[0],
            self.source_dim,
            self.virtual_grid_h,
            self.virtual_grid_w,
        )
        for view_idx in range(self.num_views):
            grid_h, grid_w = self.per_view_grid_sizes[view_idx]
            spatial_start = self.per_view_offsets[view_idx]
            spatial_end = spatial_start + self.per_view_spatial_tokens[view_idx]
            feature_tokens = spatial_tokens[:, spatial_start:spatial_end, :]
            feature_map = feature_tokens.reshape(spatial_tokens.shape[0], grid_h, grid_w, self.source_dim)
            feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
            row0, col0 = self._view_origin_coord(view_idx)
            stitched[:, :, row0 : row0 + grid_h, col0 : col0 + grid_w] = feature_map
        return stitched

    def forward(
        self,
        spatial_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial_tokens.ndim != 3:
            raise ValueError(f"spatial_tokens must be [B,V*N,D], got {tuple(spatial_tokens.shape)}")

        batch_size = spatial_tokens.shape[0]
        expected_spatial = self.total_spatial_tokens
        if spatial_tokens.shape[1] != expected_spatial:
            raise ValueError(
                f"spatial_tokens length mismatch: expected {expected_spatial}, got {spatial_tokens.shape[1]}"
            )
        if spatial_tokens.shape[-1] != self.source_dim:
            raise ValueError(
                f"spatial_tokens dim mismatch: expected {self.source_dim}, got {spatial_tokens.shape[-1]}"
            )

        if self.global_canvas:
            pooled_spatial = spatial_tokens.mean(dim=1)
            pred_coords_01 = self.coord_head(pooled_spatial).view(batch_size, self.num_tokens, 2)
            coords_norm = pred_coords_01 * 2.0 - 1.0
            feature_map = self._build_stitched_feature_map(spatial_tokens)
            sampled_raw = F.grid_sample(
                feature_map,
                coords_norm.unsqueeze(1),
                align_corners=False,
                mode="bilinear",
            ).squeeze(2).transpose(1, 2)
            sampled_tok = self.sampled_visual_proj(sampled_raw)
            if self.coord_encoder is not None:
                sampled_tok = sampled_tok + self.coord_encoder(coords_norm)
            return sampled_tok, coords_norm

        sampled_tokens_per_view = []
        coords_per_view = []
        max_tokens = max(self.num_tokens_per_view)
        for view_idx in range(self.num_views):
            grid_h, grid_w = self.per_view_grid_sizes[view_idx]
            num_tokens = int(self.num_tokens_per_view[view_idx])
            spatial_start = self.per_view_offsets[view_idx]
            spatial_end = spatial_start + self.per_view_spatial_tokens[view_idx]
            feature_tokens = spatial_tokens[:, spatial_start:spatial_end, :]
            pooled_spatial = feature_tokens.mean(dim=1)
            view_cond = pooled_spatial + self.view_type_embed_for_coord.weight[view_idx].view(1, -1).to(
                device=pooled_spatial.device, dtype=pooled_spatial.dtype
            )
            coord_head = self.coord_head if self.share_coord_head else self.coord_heads[view_idx]
            pred_coords_01 = coord_head(view_cond).view(batch_size, num_tokens, 2)
            coords_norm = pred_coords_01 * 2.0 - 1.0

            feature_map = feature_tokens.reshape(batch_size, grid_h, grid_w, self.source_dim)
            feature_map = feature_map.permute(0, 3, 1, 2).contiguous()
            sampled_raw = F.grid_sample(
                feature_map,
                coords_norm.unsqueeze(1),
                align_corners=False,
                mode="bilinear",
            ).squeeze(2).transpose(1, 2)
            sampled_tok = self.sampled_visual_proj(sampled_raw)
            if self.coord_encoder is not None:
                sampled_tok = sampled_tok + self.coord_encoder(coords_norm)
            sampled_tok = sampled_tok + self.view_type_embed_for_token.weight[view_idx].view(1, 1, -1).to(
                device=sampled_tok.device, dtype=sampled_tok.dtype
            )

            sampled_tokens_per_view.append(sampled_tok)
            padded_coords = coords_norm.new_zeros(batch_size, max_tokens, 2)
            padded_coords[:, :num_tokens, :] = coords_norm
            coords_per_view.append(padded_coords)

        sampled_visual_tokens = torch.cat(sampled_tokens_per_view, dim=1)
        coords = torch.stack(coords_per_view, dim=1)
        return sampled_visual_tokens, coords


class FullCurrentSpatialTokenEncoder(nn.Module):
    """Project all current-frame spatial tokens into video hidden tokens."""

    def __init__(
        self,
        *,
        source_dim: int,
        hidden_dim: int,
        num_views: int,
        per_view_num_spatial_tokens: tuple[int, ...],
    ):
        super().__init__()
        self.source_dim = int(source_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.per_view_num_spatial_tokens = tuple(int(v) for v in per_view_num_spatial_tokens)
        if len(self.per_view_num_spatial_tokens) != self.num_views:
            raise ValueError(
                "per_view_num_spatial_tokens must provide one token count per view, "
                f"got {len(self.per_view_num_spatial_tokens)} for num_views={self.num_views}"
            )
        self.total_spatial_tokens = int(sum(self.per_view_num_spatial_tokens))

        self.spatial_proj = nn.Linear(self.source_dim, self.hidden_dim)
        self.view_type_embed_for_token = nn.Embedding(self.num_views, self.hidden_dim)
        nn.init.zeros_(self.view_type_embed_for_token.weight)

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        if spatial_tokens.ndim != 3:
            raise ValueError(f"spatial_tokens must be [B,V*N,D], got {tuple(spatial_tokens.shape)}")
        expected_spatial = self.total_spatial_tokens
        if spatial_tokens.shape[1] != expected_spatial:
            raise ValueError(
                f"spatial_tokens length mismatch: expected {expected_spatial}, got {spatial_tokens.shape[1]}"
            )
        if spatial_tokens.shape[-1] != self.source_dim:
            raise ValueError(
                f"spatial_tokens dim mismatch: expected {self.source_dim}, got {spatial_tokens.shape[-1]}"
            )

        batch_size = spatial_tokens.shape[0]
        encoded_per_view = []
        offset = 0
        for view_idx in range(self.num_views):
            spatial_tokens_per_view = self.per_view_num_spatial_tokens[view_idx]
            encoded = self.spatial_proj(spatial_tokens[:, offset : offset + spatial_tokens_per_view])
            encoded = encoded + self.view_type_embed_for_token.weight[view_idx].view(1, 1, -1).to(
                device=encoded.device, dtype=encoded.dtype
            )
            encoded_per_view.append(encoded)
            offset += spatial_tokens_per_view
        return torch.cat(encoded_per_view, dim=1)


class LearnableCurrentDinoSpatialEncoder(nn.Module):
    """Current-frame DINO encoder with the same architecture and initialization as DINO."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        num_views: int,
        expected_spatial_tokens: int,
        per_view_grid_sizes: tuple[tuple[int, int], ...],
        expected_hidden_dim: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
        legacy_batched_views: bool = False,
    ):
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.num_views = int(num_views)
        self.expected_spatial_tokens = int(expected_spatial_tokens)
        self.per_view_grid_sizes = tuple((int(h), int(w)) for h, w in per_view_grid_sizes)
        if len(self.per_view_grid_sizes) != self.num_views:
            raise ValueError(
                "per_view_grid_sizes must provide one (grid_h, grid_w) pair per view, "
                f"got {len(self.per_view_grid_sizes)} for num_views={self.num_views}"
            )
        self.per_view_spatial_tokens = tuple(grid_h * grid_w for grid_h, grid_w in self.per_view_grid_sizes)
        if int(sum(self.per_view_spatial_tokens)) != self.expected_spatial_tokens:
            raise ValueError(
                "expected_spatial_tokens must match the sum of per-view spatial tokens, "
                f"got expected_spatial_tokens={self.expected_spatial_tokens} and per_view_spatial_tokens={self.per_view_spatial_tokens}"
            )
        self.expected_hidden_dim = None if expected_hidden_dim is None else int(expected_hidden_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.legacy_batched_views = bool(legacy_batched_views)
        if self.legacy_batched_views and len(set(self.per_view_grid_sizes)) != 1:
            raise ValueError(
                "legacy_batched_views requires identical per-view grid sizes, "
                f"got {self.per_view_grid_sizes}"
            )

        self.model = _build_policy_dino_encoder(self.model_name_or_path)
        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if hidden_size <= 0:
            raise ValueError(
                f"Could not resolve hidden_size from current DINO config for {self.model_name_or_path}"
            )
        if self.expected_hidden_dim is not None and hidden_size != self.expected_hidden_dim:
            raise ValueError(
                f"Current DINO hidden size mismatch: expected {self.expected_hidden_dim}, got {hidden_size}"
            )
        self.hidden_dim = hidden_size
        embeddings = getattr(self.model, "embeddings", None)
        mask_token = getattr(embeddings, "mask_token", None)
        if isinstance(mask_token, nn.Parameter):
            mask_token.requires_grad_(False)
        patch_size = int(getattr(self.model.config, "patch_size", 14) or 14)
        self.patch_size = patch_size
        self.per_view_target_image_sizes = tuple(
            (grid_h * patch_size, grid_w * patch_size)
            for grid_h, grid_w in self.per_view_grid_sizes
        )

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=True)
        self.register_buffer("imagenet_std", std, persistent=True)

    def forward(self, input_pixels) -> torch.Tensor:
        if not isinstance(input_pixels, (list, tuple)):
            raise ValueError(f"input_pixels must be a list/tuple of per-view tensors, got {type(input_pixels)}")
        if len(input_pixels) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(input_pixels)}")

        batch_size = input_pixels[0].shape[0]
        for view in input_pixels:
            if view.ndim != 4 or view.shape[1] != 3:
                raise ValueError(f"Each view tensor must be [B,3,H,W], got {tuple(view.shape)}")
            if view.shape[0] != batch_size:
                raise ValueError("All view tensors must share the same batch dimension")

        if self.legacy_batched_views:
            # Exact pre-heterogeneous-view path. Besides layout, B*V versus B
            # changes BF16 attention reductions, so this is needed for legacy
            stacked = torch.stack(input_pixels, dim=1)
            flat = stacked.reshape(batch_size * self.num_views, 3, stacked.shape[-2], stacked.shape[-1])
            flat = flat.to(device=self.imagenet_mean.device, dtype=self.imagenet_mean.dtype)
            target_image_size = self.per_view_target_image_sizes[0]
            if tuple(flat.shape[-2:]) != target_image_size:
                flat = F.interpolate(flat, size=target_image_size, mode="bilinear", align_corners=False, antialias=True)
            flat = flat * 0.5 + 0.5
            flat = (flat - self.imagenet_mean) / self.imagenet_std

            seq = self.model(pixel_values=flat).last_hidden_state
            expected_tokens = self.per_view_spatial_tokens[0]
            num_extra = int(seq.shape[1]) - expected_tokens
            if num_extra < 0:
                raise ValueError(
                    f"DINO output token count {seq.shape[1]} is smaller than expected spatial tokens {expected_tokens}"
                )
            spatial = seq[:, num_extra:, :]
            if spatial.shape[1] != expected_tokens:
                raise ValueError(
                    f"DINO spatial token count mismatch: expected {expected_tokens}, got {spatial.shape[1]}"
                )
            return spatial.reshape(batch_size, self.num_views * expected_tokens, self.hidden_dim)

        per_view_spatial = []
        for view_idx, view in enumerate(input_pixels):
            target_image_size = self.per_view_target_image_sizes[view_idx]
            flat = view.to(device=self.imagenet_mean.device, dtype=self.imagenet_mean.dtype)
            if tuple(flat.shape[-2:]) != target_image_size:
                flat = F.interpolate(flat, size=target_image_size, mode="bilinear", align_corners=False, antialias=True)
            flat = flat * 0.5 + 0.5
            flat = (flat - self.imagenet_mean) / self.imagenet_std

            seq = self.model(pixel_values=flat).last_hidden_state
            expected_tokens = self.per_view_spatial_tokens[view_idx]
            num_extra = int(seq.shape[1]) - expected_tokens
            if num_extra < 0:
                raise ValueError(
                    f"DINO output token count {seq.shape[1]} is smaller than expected spatial tokens {expected_tokens}"
                )
            spatial = seq[:, num_extra:, :]
            if spatial.shape[1] != expected_tokens:
                raise ValueError(
                    f"DINO spatial token count mismatch: expected {expected_tokens}, got {spatial.shape[1]}"
                )
            per_view_spatial.append(spatial)
        return torch.cat(per_view_spatial, dim=1)


class FrozenCurrentVJEPASpatialEncoder(nn.Module):
    """Frozen current-frame V-JEPA encoder using true single-frame forward."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        num_views: int,
        expected_spatial_tokens: int,
        per_view_grid_sizes: tuple[tuple[int, int], ...],
        expected_hidden_dim: int,
    ):
        super().__init__()
        self.model_name_or_path = str(model_name_or_path)
        self.num_views = int(num_views)
        self.expected_spatial_tokens = int(expected_spatial_tokens)
        self.expected_hidden_dim = int(expected_hidden_dim)
        self.per_view_grid_sizes = tuple((int(h), int(w)) for h, w in per_view_grid_sizes)
        if len(self.per_view_grid_sizes) != self.num_views:
            raise ValueError(
                "per_view_grid_sizes must provide one (grid_h, grid_w) pair per view, "
                f"got {len(self.per_view_grid_sizes)} for num_views={self.num_views}"
            )
        self.per_view_spatial_tokens = tuple(grid_h * grid_w for grid_h, grid_w in self.per_view_grid_sizes)
        if int(sum(self.per_view_spatial_tokens)) != self.expected_spatial_tokens:
            raise ValueError(
                "expected_spatial_tokens must match the sum of per-view spatial tokens, "
                f"got expected_spatial_tokens={self.expected_spatial_tokens} and per_view_spatial_tokens={self.per_view_spatial_tokens}"
            )

        model, _predictor = load_vjepa_model(self.model_name_or_path)
        self.model = model.to(dtype=torch.float32).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.hidden_dim = self.expected_hidden_dim
        self.patch_size = 16
        self.per_view_target_image_sizes = tuple(
            (grid_h * self.patch_size, grid_w * self.patch_size)
            for grid_h, grid_w in self.per_view_grid_sizes
        )

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=True)
        self.register_buffer("imagenet_std", std, persistent=True)

    def forward(self, input_pixels) -> torch.Tensor:
        if not isinstance(input_pixels, (list, tuple)):
            raise ValueError(f"input_pixels must be a list/tuple of per-view tensors, got {type(input_pixels)}")
        if len(input_pixels) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(input_pixels)}")

        batch_size = input_pixels[0].shape[0]
        for view in input_pixels:
            if view.ndim != 4 or view.shape[1] != 3:
                raise ValueError(f"Each view tensor must be [B,3,H,W], got {tuple(view.shape)}")
            if view.shape[0] != batch_size:
                raise ValueError("All view tensors must share the same batch dimension")

        per_view_spatial = []
        for view_idx, view in enumerate(input_pixels):
            target_image_size = self.per_view_target_image_sizes[view_idx]
            x_cur = view.to(device=self.imagenet_mean.device, dtype=self.imagenet_mean.dtype)
            if tuple(x_cur.shape[-2:]) != target_image_size:
                x_cur = F.interpolate(x_cur, size=target_image_size, mode="bilinear", align_corners=False, antialias=True)
            x_cur = x_cur * 0.5 + 0.5
            x_cur = (x_cur - self.imagenet_mean) / self.imagenet_std
            video_input = x_cur.unsqueeze(2)

            with torch.inference_mode():
                out = self.model(video_input)
            seq = _normalize_vjepa_output(out)
            expected_tokens = self.per_view_spatial_tokens[view_idx]
            if seq.shape[1] != expected_tokens:
                raise ValueError(
                    f"VJEPA spatial token count mismatch: expected {expected_tokens}, got {seq.shape[1]}"
                )
            if seq.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"VJEPA hidden dim mismatch: expected {self.hidden_dim}, got {seq.shape[-1]}"
                )
            per_view_spatial.append(seq)
        return torch.cat(per_view_spatial, dim=1)


class CompactSemanticFastWAM(SemanticFastWAM):
    """SemanticFastWAM variant with compact predictive-control latent backbone.

    Key changes from SemanticFastWAM:
    - Frozen encoder features -> Compactor C -> compact latent z (few tokens)
    - Future branch predicts z_future (compact) instead of raw teacher tokens
    - EMA compactor provides stop-grad future targets
    """

    def __init__(self, config: SemanticMoTConfig):
        super().__init__(config)
        cfg = self.config

        # ---- Compactor ----
        self.compact_source_dim = int(cfg.teacher_dim)
        if int(cfg.compact_encoder_dim) != self.compact_source_dim:
            raise ValueError(
                f"compact_encoder_dim must match teacher_dim for CompactSemanticFastWAM: "
                f"got compact_encoder_dim={int(cfg.compact_encoder_dim)} and teacher_dim={self.compact_source_dim}"
            )
        self.compactor = CompactControlLatent(
            input_dim=self.compact_source_dim,
            output_dim=int(cfg.compact_dim),
            num_tokens=int(cfg.compact_num_tokens),
            depth=int(cfg.compact_depth),
            num_heads=int(cfg.compact_num_heads),
            head_dim=int(cfg.compact_head_dim),
            mlp_ratio=float(cfg.compact_mlp_ratio),
            dropout=float(cfg.compact_dropout),
            eps=float(cfg.eps),
            norm_output=bool(cfg.compact_norm_output),
            proj_dim=int(cfg.compact_proj_dim),
            proj_hidden_mult=int(cfg.compact_proj_hidden_mult),
            token_layout=str(cfg.compact_token_layout),
            token_scope=str(cfg.compact_token_scope),
            use_abs_pos_embed=bool(cfg.compact_use_abs_pos_embed),
            use_fourier_film_pos=bool(getattr(cfg, "compact_use_fourier_film_pos", False)),
            fourier_num_bands=int(getattr(cfg, "compact_fourier_num_bands", 8)),
            fourier_include_local_xy=bool(getattr(cfg, "compact_fourier_include_local_xy", True)),
            fourier_include_global_xy=bool(getattr(cfg, "compact_fourier_include_global_xy", True)),
            fourier_include_view_embed=bool(getattr(cfg, "compact_fourier_include_view_embed", True)),
            fourier_film_init_scale=float(getattr(cfg, "compact_fourier_film_init_scale", 0.0)),
            conditioning_mode=str(getattr(cfg, "compact_conditioning_mode", "none")),
            conditioning_dim=getattr(cfg, "compact_conditioning_dim", None),
            conditioning_use_patch_logit_bias=bool(
                getattr(cfg, "compact_conditioning_use_patch_logit_bias", False)
            ),
            view_rope_layout=str(
                getattr(cfg, "compact_view_rope_layout", None)
                or getattr(cfg, "view_rope_layout", "horizontal")
            ),
            num_views=self.num_views,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            per_view_grid_sizes=self.per_view_grid_sizes,
            spatial_h_tokens=int(cfg.compact_spatial_h_tokens),
            spatial_v_tokens=int(cfg.compact_spatial_v_tokens),
            global_tokens=int(cfg.compact_global_tokens),
            view_global_tokens=getattr(cfg, "compact_view_global_tokens", None),
            spatial_view_indices=getattr(cfg, "compact_spatial_view_indices", None),
            spatial_mode=str(cfg.compact_spatial_mode),
            spatial_band_rows=int(cfg.compact_spatial_band_rows),
            spatial_band_cols=int(cfg.compact_spatial_band_cols),
            use_attn_spatial_writeback=bool(cfg.compact_use_attn_spatial_writeback),
            attention_mode=str(cfg.compact_attention_mode),
            competitive_temperature=float(cfg.compact_competitive_temperature),
            competitive_eps=float(cfg.compact_competitive_eps),
            spatial_writeback_hidden_mult=int(cfg.compact_spatial_writeback_hidden_mult),
            spatial_writeback_init_scale=float(cfg.compact_spatial_writeback_init_scale),
            spatial_writeback_detach_attn=bool(cfg.compact_spatial_writeback_detach_attn),
            spatial_moment_mode=str(cfg.compact_spatial_moment_mode),
        )
        self.compactor_ema = build_ema_compactor(self.compactor) if cfg.enable_compact_future else None

        # ---- Projections ----
        compact_proj_dim = int(cfg.compact_proj_dim)
        self.effective_compact_dim = compact_proj_dim if compact_proj_dim > 0 else int(cfg.compact_dim)
        self.compact_anchor_proj = nn.Linear(self.effective_compact_dim, int(cfg.hidden_dim))
        self.compact_token_layout = str(cfg.compact_token_layout)
        self.compact_spatial_band_rows = int(cfg.compact_spatial_band_rows)
        self.compact_spatial_band_cols = int(cfg.compact_spatial_band_cols)
        self.use_abstract_anchor_tokens = bool(getattr(cfg, "use_abstract_anchor_tokens", True))
        if self.compact_token_layout == "spatial_global":
            self.compact_hidden_query_embed = nn.Parameter(
                torch.zeros(1, int(cfg.compact_num_tokens), int(cfg.hidden_dim))
            )
            nn.init.normal_(self.compact_hidden_query_embed, std=0.02)
        else:
            self.compact_hidden_query_embed = None

        self.enable_compact_future = bool(cfg.enable_compact_future)
        self.compact_core_tokens = int(cfg.compact_core_tokens)
        self.future_pool_size = int(cfg.future_pool_size)
        if self.compact_core_tokens > int(cfg.compact_num_tokens):
            raise ValueError(
                f"compact_core_tokens ({self.compact_core_tokens}) cannot exceed compact_num_tokens ({int(cfg.compact_num_tokens)})"
            )
        self.compact_future_in = nn.Linear(self.effective_compact_dim, int(cfg.hidden_dim))
        self.compact_future_head = nn.Linear(int(cfg.hidden_dim), self.effective_compact_dim)

        self.learnable_compact_vision_encoder = bool(cfg.learnable_compact_vision_encoder)
        self.compact_dino_model_name_or_path = str(cfg.compact_dino_model_name_or_path)
        self.current_spatial_mode = str(cfg.current_spatial_mode)
        self._use_current_spatial_branch = self.current_spatial_mode != "off"
        if not self.use_abstract_anchor_tokens and not self._use_current_spatial_branch:
            raise ValueError(
                "use_abstract_anchor_tokens=False requires current_spatial_mode to provide a non-empty spatial branch."
            )
        self.grid_sampler_num_tokens = int(cfg.grid_sampler_num_tokens)
        raw_grid_sampler_num_tokens_per_view = getattr(cfg, "grid_sampler_num_tokens_per_view", None)
        self.grid_sampler_num_tokens_per_view = (
            None if raw_grid_sampler_num_tokens_per_view is None
            else tuple(int(v) for v in raw_grid_sampler_num_tokens_per_view)
        )
        self.grid_sampler_global_canvas = bool(getattr(cfg, "grid_sampler_global_canvas", False))
        self.grid_sampler_share_coord_head = bool(getattr(cfg, "grid_sampler_share_coord_head", False))
        self.grid_sampler_coord_embed = bool(cfg.grid_sampler_coord_embed)
        self.use_runtime_current_vjepa_anchor = bool(getattr(cfg, "use_runtime_current_vjepa_anchor", False))
        self.vjepa_model_name_or_path = str(
            getattr(cfg, "vjepa_model_name_or_path", None) or "vjepa2_1_vit_large_384"
        )
        self.learnable_spatial_vision_encoder = bool(cfg.learnable_spatial_vision_encoder) and self._use_current_spatial_branch
        self.current_spatial_dino_model_name_or_path = str(cfg.current_spatial_dino_model_name_or_path)
        self.current_spatial_source_dim = int(self.teacher_dim)

        if self.use_runtime_current_vjepa_anchor:
            if str(cfg.compact_encoder_type).lower() != "vjepa":
                raise ValueError(
                    "runtime VJEPA anchor/teacher requires compact_encoder_type='vjepa', "
                    f"got {cfg.compact_encoder_type!r}"
                )
            self.vjepa_encoder = FrozenCurrentVJEPASpatialEncoder(
                model_name_or_path=self.vjepa_model_name_or_path,
                num_views=int(self.num_views),
                expected_spatial_tokens=int(self.total_num_spatial_tokens),
                per_view_grid_sizes=self.per_view_grid_sizes,
                expected_hidden_dim=int(self.teacher_dim),
            )
        else:
            self.vjepa_encoder = None

        if self.learnable_compact_vision_encoder:
            if str(cfg.compact_encoder_type).lower() != "dino":
                raise ValueError(
                    f"learnable_compact_vision_encoder currently supports compact_encoder_type='dino' only, got {cfg.compact_encoder_type!r}"
                )
            if not self.compact_dino_model_name_or_path:
                raise ValueError("compact_dino_model_name_or_path must be set when learnable_compact_vision_encoder=True")
            self.compact_dino_encoder = LearnableCurrentDinoSpatialEncoder(
                model_name_or_path=self.compact_dino_model_name_or_path,
                num_views=int(self.num_views),
                expected_spatial_tokens=int(self.total_num_spatial_tokens),
                per_view_grid_sizes=self.per_view_grid_sizes,
                expected_hidden_dim=int(self.teacher_dim),
                use_gradient_checkpointing=bool(cfg.use_gradient_checkpointing),
            )
        else:
            self.compact_dino_encoder = None

        if self.learnable_spatial_vision_encoder:
            if not self.current_spatial_dino_model_name_or_path:
                raise ValueError("current_spatial_dino_model_name_or_path must be set when learnable_spatial_vision_encoder=True")
            self.current_spatial_dino_encoder = LearnableCurrentDinoSpatialEncoder(
                model_name_or_path=self.current_spatial_dino_model_name_or_path,
                num_views=int(self.num_views),
                expected_spatial_tokens=int(self.total_num_spatial_tokens),
                per_view_grid_sizes=self.per_view_grid_sizes,
                use_gradient_checkpointing=bool(cfg.use_gradient_checkpointing),
                legacy_batched_views=bool(getattr(cfg, "legacy_batched_view_dino_encoder", False)),
            )
            self.current_spatial_source_dim = int(self.current_spatial_dino_encoder.hidden_dim)
        else:
            self.current_spatial_dino_encoder = None

        if self.current_spatial_mode == "sampled":
            if self.grid_sampler_num_tokens <= 0:
                raise ValueError(
                    f"grid_sampler_num_tokens must be positive, got {self.grid_sampler_num_tokens}"
                )
            if self.grid_sampler_num_tokens_per_view is not None:
                if len(self.grid_sampler_num_tokens_per_view) != int(self.num_views):
                    raise ValueError(
                        "grid_sampler_num_tokens_per_view must align with num_views: "
                        f"got {self.grid_sampler_num_tokens_per_view} for num_views={self.num_views}"
                    )
                if any(v <= 0 for v in self.grid_sampler_num_tokens_per_view):
                    raise ValueError(
                        "grid_sampler_num_tokens_per_view must contain only positive values, "
                        f"got {self.grid_sampler_num_tokens_per_view}"
                    )
            self.abstract_grid_sampler = AbstractConditionedGridSampler(
                source_dim=int(self.current_spatial_source_dim),
                hidden_dim=int(cfg.hidden_dim),
                num_views=int(self.num_views),
                num_tokens=self.grid_sampler_num_tokens,
                num_tokens_per_view=self.grid_sampler_num_tokens_per_view,
                per_view_grid_sizes=self.per_view_grid_sizes,
                view_rope_layout=self.view_rope_layout,
                global_canvas=self.grid_sampler_global_canvas,
                share_coord_head=self.grid_sampler_share_coord_head,
                coord_embed=self.grid_sampler_coord_embed,
            )
        else:
            self.abstract_grid_sampler = None

        if self.current_spatial_mode == "full":
            self.full_spatial_token_encoder = FullCurrentSpatialTokenEncoder(
                source_dim=int(self.current_spatial_source_dim),
                hidden_dim=int(cfg.hidden_dim),
                num_views=int(self.num_views),
                per_view_num_spatial_tokens=self.per_view_spatial_tokens,
            )
        else:
            self.full_spatial_token_encoder = None

        # Remove parent-class modules that are unused in the compact path.
        for attr in (
            "frozen_anchor_proj",
            "future_dense_in",
            "future_cls_in",
            "dense_head",
            "cls_head",
            "token_type_embed",
            "register_slot_embed",
            "view_embed",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------

    def _fp32_module_paths(self) -> tuple[str, ...]:
        return super()._fp32_module_paths()

    def _fp32_parameter_paths(self) -> tuple[str, ...]:
        return super()._fp32_parameter_paths()




    # ------------------------------------------------------------------
    # Encoder token packing
    # ------------------------------------------------------------------

    def _normalize_compact_anchor_spatial(self, anchor_dino_spatial: torch.Tensor) -> torch.Tensor:
        if anchor_dino_spatial.ndim == 2:
            anchor_dino_spatial = anchor_dino_spatial.unsqueeze(0)
        if anchor_dino_spatial.ndim != 3:
            raise ValueError(
                "anchor_dino_spatial must be [B,N,D] or [N,D], "
                f"got {tuple(anchor_dino_spatial.shape)}"
            )
        if anchor_dino_spatial.shape[-1] != self.teacher_dim:
            raise ValueError(
                f"anchor_dino_spatial teacher dim mismatch: expected {self.teacher_dim}, "
                f"got {anchor_dino_spatial.shape[-1]}"
            )
        expected_spatial = self.total_num_spatial_tokens
        if anchor_dino_spatial.shape[1] != expected_spatial:
            raise ValueError(
                f"anchor_dino_spatial token count mismatch: expected {expected_spatial}, got {anchor_dino_spatial.shape[1]}"
            )
        return self._normalize_dino_spatial(anchor_dino_spatial)

    def _forward_current_dino_encoder_fp32(self, input_pixels) -> torch.Tensor:
        if self.current_spatial_dino_encoder is None:
            raise RuntimeError("current_spatial_dino_encoder is not initialized")
        event = self._profile_cuda_start("extra_dino_ms", input_pixels[0].device)
        runtime_out_dtype = self._runtime_compute_dtype(input_pixels[0].device, input_pixels[0].dtype)
        encoder_fn = getattr(self, "_current_spatial_dino_encoder_infer_fn", self.current_spatial_dino_encoder)
        if not self._use_custom_fp32_precision():
            output = encoder_fn(input_pixels).to(dtype=runtime_out_dtype)
            self._profile_cuda_end("extra_dino_ms", event)
            return output
        target_dtype = self._module_compute_dtype(self.current_spatial_dino_encoder, fallback=torch.float32)
        with torch.autocast(device_type=self._autocast_device_type(input_pixels[0].device), enabled=False):
            encoder_views = [view.to(device=view.device, dtype=target_dtype) for view in input_pixels]
            y = self.current_spatial_dino_encoder(encoder_views)
        output = y.to(dtype=runtime_out_dtype)
        self._profile_cuda_end("extra_dino_ms", event)
        return output

    def _forward_current_vjepa_encoder_fp32(self, input_pixels) -> torch.Tensor:
        if self.vjepa_encoder is None:
            raise RuntimeError("vjepa_encoder is not initialized")
        runtime_out_dtype = self._runtime_compute_dtype(input_pixels[0].device, input_pixels[0].dtype)
        if not self._use_custom_fp32_precision():
            return self.vjepa_encoder(input_pixels).to(dtype=runtime_out_dtype)
        target_dtype = self._module_compute_dtype(self.vjepa_encoder, fallback=torch.float32)
        with torch.autocast(device_type=self._autocast_device_type(input_pixels[0].device), enabled=False):
            encoder_views = [view.to(device=view.device, dtype=target_dtype) for view in input_pixels]
            y = self.vjepa_encoder(encoder_views)
        return y.to(dtype=runtime_out_dtype)



    def _build_compact_encoder_source(self, input_pixels, anchor_dino_spatial: torch.Tensor) -> torch.Tensor:
        if self.compact_dino_encoder is not None:
            runtime_out_dtype = self._runtime_compute_dtype(input_pixels[0].device, input_pixels[0].dtype)
            if not self._use_custom_fp32_precision():
                return self.compact_dino_encoder(input_pixels).to(dtype=runtime_out_dtype)
            target_dtype = self._module_compute_dtype(self.compact_dino_encoder, fallback=torch.float32)
            with torch.autocast(device_type=self._autocast_device_type(input_pixels[0].device), enabled=False):
                encoder_views = [view.to(device=view.device, dtype=target_dtype) for view in input_pixels]
                y = self.compact_dino_encoder(encoder_views)
            return y.to(dtype=runtime_out_dtype)
        return self._normalize_compact_anchor_spatial(anchor_dino_spatial)

    def _build_current_spatial_source(self, input_pixels, anchor_dino_spatial: torch.Tensor) -> torch.Tensor:
        if self.current_spatial_dino_encoder is not None:
            return self._forward_current_dino_encoder_fp32(input_pixels)
        return self._normalize_compact_anchor_spatial(anchor_dino_spatial)

    def _forward_compactor_fp32(
        self,
        module: nn.Module,
        tokens: torch.Tensor,
        *,
        conditioning_context: Optional[torch.Tensor] = None,
        conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        runtime_out_dtype = self._runtime_compute_dtype(tokens.device, tokens.dtype)
        if not self._use_custom_fp32_precision():
            return module(
                tokens,
                conditioning_context=conditioning_context,
                conditioning_mask=conditioning_mask,
            ).to(dtype=runtime_out_dtype)
        target_dtype = self._module_compute_dtype(module, fallback=torch.float32)
        with torch.autocast(device_type=self._autocast_device_type(tokens.device), enabled=False):
            cast_context = None
            if conditioning_context is not None:
                cast_context = conditioning_context.to(device=tokens.device, dtype=target_dtype)
            y = module(
                tokens.to(device=tokens.device, dtype=target_dtype),
                conditioning_context=cast_context,
                conditioning_mask=conditioning_mask,
            )
        return y.to(dtype=runtime_out_dtype)

    def _pack_spatial_tokens(self, spatial: torch.Tensor) -> torch.Tensor:
        if spatial.shape[-2] != self.total_num_spatial_tokens:
            raise ValueError(
                f"spatial token count mismatch: expected {self.total_num_spatial_tokens}, got {spatial.shape[-2]}"
            )
        per_view_tokens = []
        for view_idx in range(self.num_views):
            spatial_start = self.per_view_spatial_offsets[view_idx]
            spatial_end = spatial_start + self.per_view_spatial_tokens[view_idx]
            per_view_tokens.append(spatial[:, spatial_start:spatial_end, :])
        return torch.cat(per_view_tokens, dim=-2)

    # ------------------------------------------------------------------
    # RoPE for compact tokens
    # ------------------------------------------------------------------

    def _build_compact_anchor_freqs(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        if self.compact_token_layout == "spatial_global":
            row_ids, col_ids = self._build_spatial_global_compact_coords(num_tokens, device)
            frame_ids = torch.zeros(num_tokens, device=device, dtype=torch.long)
            return self._video_rope_from_coords(frame_ids, row_ids, col_ids).to(device=device)
        frame_ids = torch.zeros(num_tokens, device=device, dtype=torch.long)
        row_ids = torch.arange(num_tokens, device=device, dtype=torch.long)
        col_ids = torch.zeros(num_tokens, device=device, dtype=torch.long)
        return self._video_rope_from_coords(frame_ids, row_ids, col_ids).to(device=device)

    def _build_current_spatial_ordered_freqs(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        frame_ids = torch.zeros(num_tokens, device=device, dtype=torch.long)
        row_ids = torch.arange(num_tokens, device=device, dtype=torch.long)
        col_ids = torch.zeros(num_tokens, device=device, dtype=torch.long)
        return self._video_rope_from_coords(frame_ids, row_ids, col_ids).to(device=device)

    def _build_full_current_spatial_freqs(self, device: torch.device) -> torch.Tensor:
        frame_chunks = []
        row_chunks = []
        col_chunks = []
        for view_idx in range(self.num_views):
            rows, cols = self._spatial_coord_ids_for_view(view_idx, device)
            frame_chunks.append(torch.zeros(self.per_view_spatial_tokens[view_idx], device=device, dtype=torch.long))
            row_chunks.append(rows)
            col_chunks.append(cols)
        return self._video_rope_from_coords(
            torch.cat(frame_chunks, dim=0),
            torch.cat(row_chunks, dim=0),
            torch.cat(col_chunks, dim=0),
        ).to(device=device)

    def _build_current_anchor_freqs(
        self,
        *,
        abstract_anchor_len: int,
        current_spatial_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        chunks = []
        if int(abstract_anchor_len) > 0:
            chunks.append(self._build_compact_anchor_freqs(int(abstract_anchor_len), device))
        if int(current_spatial_len) > 0:
            if self.current_spatial_mode == "full" and int(current_spatial_len) == self.total_num_spatial_tokens:
                chunks.append(self._build_full_current_spatial_freqs(device))
            else:
                chunks.append(self._build_current_spatial_ordered_freqs(int(current_spatial_len), device))
        if not chunks:
            return self._build_compact_anchor_freqs(0, device)
        return torch.cat(chunks, dim=0)

    def _build_compact_future_freqs(
        self,
        future_steps: int,
        tokens_per_step: int,
        device: torch.device,
        future_frame_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.compact_token_layout == "spatial_global":
            if future_frame_offsets is not None:
                base_ids = future_frame_offsets.to(device=device, dtype=torch.long)
                if base_ids.ndim > 1:
                    base_ids = base_ids[0]
            else:
                base_ids = (
                    torch.arange(1, future_steps + 1, device=device, dtype=torch.long) * self.action_per_frame
                )
            row_ids_step, col_ids_step = self._build_spatial_global_compact_coords(tokens_per_step, device)
            frame_ids = base_ids.repeat_interleave(tokens_per_step)
            row_ids = row_ids_step.repeat(future_steps)
            col_ids = col_ids_step.repeat(future_steps)
            return self._video_rope_from_coords(frame_ids, row_ids, col_ids).to(device=device)
        total = future_steps * tokens_per_step
        if future_frame_offsets is not None:
            base_ids = future_frame_offsets.to(device=device, dtype=torch.long)
            if base_ids.ndim > 1:
                base_ids = base_ids[0]
        else:
            base_ids = (
                torch.arange(1, future_steps + 1, device=device, dtype=torch.long) * self.action_per_frame
            )
        frame_ids = base_ids.repeat_interleave(tokens_per_step)
        row_ids = torch.arange(tokens_per_step, device=device, dtype=torch.long).repeat(future_steps)
        col_ids = torch.zeros(total, device=device, dtype=torch.long)
        return self._video_rope_from_coords(frame_ids, row_ids, col_ids).to(device=device)

    # ------------------------------------------------------------------
    # Anchor tokens (override)
    # ------------------------------------------------------------------

    def _build_spatial_global_compact_coords(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compact_token_layout != "spatial_global":
            raise RuntimeError("_build_spatial_global_compact_coords is only valid for spatial_global layout")
        type_ids = self.compactor.layout_query_type_ids[:num_tokens].to(device=device)
        slot_ids = self.compactor.layout_query_slot_ids[:num_tokens].to(device=device)
        view_ids = self.compactor.layout_query_view_ids[:num_tokens].to(device=device)
        row_ids = torch.zeros(num_tokens, device=device, dtype=torch.long)
        col_ids = torch.zeros(num_tokens, device=device, dtype=torch.long)
        for token_idx in range(num_tokens):
            view_idx = int(view_ids[token_idx].item())
            slot_idx = int(slot_ids[token_idx].item())
            type_id = int(type_ids[token_idx].item())
            if view_idx < 0:
                if type_id == 0:
                    row_ids[token_idx] = min(slot_idx * self.compact_spatial_band_rows, self.virtual_grid_h - 1)
                    col_ids[token_idx] = 0
                elif type_id == 1:
                    row_ids[token_idx] = 0
                    col_ids[token_idx] = min(slot_idx * self.compact_spatial_band_cols, self.virtual_grid_w - 1)
                else:
                    row_ids[token_idx] = token_idx
                    col_ids[token_idx] = 0
                continue
            row0, col0 = self._view_origin_coord(view_idx)
            grid_h, grid_w = self.per_view_grid_sizes[view_idx]
            if type_id == 0:
                row_ids[token_idx] = row0 + min(slot_idx * self.compact_spatial_band_rows, grid_h - 1)
                col_ids[token_idx] = col0
            elif type_id == 1:
                row_ids[token_idx] = row0
                col_ids[token_idx] = col0 + min(slot_idx * self.compact_spatial_band_cols, grid_w - 1)
            else:
                row_ids[token_idx] = row0 + min(slot_idx, grid_h - 1)
                col_ids[token_idx] = col0
        return row_ids, col_ids

    def _apply_compact_hidden_query_embed(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.compact_hidden_query_embed is None or tokens.shape[-2] <= 0:
            return tokens
        hidden = self.compact_hidden_query_embed[:, : tokens.shape[-2], :].to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        global_prefix = int(self.config.compact_global_tokens) if self.compact_token_layout == "spatial_global" else 0
        if global_prefix > 0:
            hidden = hidden.clone()
            hidden[:, :global_prefix, :] = 0
        if tokens.ndim == 3:
            return tokens + hidden
        if tokens.ndim == 4:
            return tokens + hidden.unsqueeze(1)
        raise ValueError(f"Unsupported compact token shape for hidden query embed: {tuple(tokens.shape)}")

    def _build_anchor_tokens(
        self,
        input_pixels,
        anchor_dino_cls: torch.Tensor,
        anchor_dino_registers: Optional[torch.Tensor],
        anchor_dino_spatial: torch.Tensor,
        *,
        compact_context: Optional[torch.Tensor] = None,
        compact_context_mask: Optional[torch.Tensor] = None,
    ):
        del anchor_dino_cls, anchor_dino_registers
        z_now = None
        batch_size = input_pixels[0].shape[0]
        runtime_dtype = self._runtime_compute_dtype(input_pixels[0].device, input_pixels[0].dtype)
        abstract_anchor_tokens = torch.empty(
            batch_size,
            0,
            self.hidden_dim,
            device=input_pixels[0].device,
            dtype=runtime_dtype,
        )
        abstract_anchor_len = 0
        anchor_spatial_source = anchor_dino_spatial
        if self.vjepa_encoder is not None and self.use_runtime_current_vjepa_anchor:
            anchor_spatial_source = self._forward_current_vjepa_encoder_fp32(input_pixels)
        if self.use_abstract_anchor_tokens:
            compact_encoder_tokens = self._build_compact_encoder_source(input_pixels, anchor_spatial_source)
            now_tokens_raw = self._pack_spatial_tokens(compact_encoder_tokens)
            compactor_fn = getattr(self, "_compactor_infer_fn", self.compactor)
            compactor_event = self._profile_cuda_start("compactor_ms", input_pixels[0].device)
            z_now = self._forward_compactor_fp32(
                compactor_fn,
                now_tokens_raw,
                conditioning_context=compact_context,
                conditioning_mask=compact_context_mask,
            )
            self._profile_cuda_end("compactor_ms", compactor_event)
            # A compiled compactor may return CUDA-Graph-owned inference
            # storage.  The following projection/prefill path is eager, so
            # materialize only that exceptional tensor representation.
            if z_now.is_inference():
                z_now = z_now.detach().clone()
            abstract_anchor_tokens = self._forward_module_fp32(self.compact_anchor_proj, z_now)
            abstract_anchor_tokens = self._apply_compact_hidden_query_embed(abstract_anchor_tokens)
            abstract_anchor_len = int(abstract_anchor_tokens.shape[1])

        if self.abstract_grid_sampler is not None or self.full_spatial_token_encoder is not None:
            current_spatial_tokens = self._build_current_spatial_source(input_pixels, anchor_spatial_source)
            if self.abstract_grid_sampler is not None:
                current_spatial_branch, _coords = self.abstract_grid_sampler(current_spatial_tokens)
            else:
                current_spatial_branch = self.full_spatial_token_encoder(current_spatial_tokens)
        else:
            current_spatial_branch = abstract_anchor_tokens[:, 0:0, :]
        current_spatial_len = int(current_spatial_branch.shape[1])

        if current_spatial_branch.shape[-1] != abstract_anchor_tokens.shape[-1]:
            raise ValueError(
                "Anchor token hidden dim mismatch: "
                f"abstract={abstract_anchor_tokens.shape[-1]}, current_spatial={current_spatial_branch.shape[-1]}"
            )
        anchor_tokens = torch.cat([abstract_anchor_tokens, current_spatial_branch], dim=1)
        meta = {
            "anchor_cls": 0,
            "anchor_regs": 0,
            "anchor_spatial": current_spatial_len,
            "anchor_total": anchor_tokens.shape[1],
            "anchor_cls_per_view": 0,
            "abstract_anchor_len": abstract_anchor_len,
            "current_spatial_len": current_spatial_len,
        }
        return anchor_tokens, meta, z_now, abstract_anchor_tokens

    # ------------------------------------------------------------------
    # Future video tokens (override)
    # ------------------------------------------------------------------

    def _build_future_video_tokens(self, future_noisy: torch.Tensor) -> torch.Tensor:
        B, T, K_core, _ = future_noisy.shape
        future_tokens = self._forward_module_fp32(self.compact_future_in, future_noisy)
        future_tokens = self._apply_compact_hidden_query_embed(future_tokens)
        return future_tokens.reshape(B, T * K_core, self.hidden_dim)

    # ------------------------------------------------------------------
    # Decode future outputs (override)
    # ------------------------------------------------------------------

    def _decode_future_outputs(self, future_out: torch.Tensor, batch_size: int, future_steps: int) -> torch.Tensor:
        K_core = int(self.compact_core_tokens)
        pred = self._forward_module_fp32(self.compact_future_head, future_out)
        return pred.view(batch_size, future_steps, K_core, self.effective_compact_dim)




    # ------------------------------------------------------------------
    # Regularization
    # ------------------------------------------------------------------



    # ------------------------------------------------------------------
    # ------------------------------------------------------------------


    # ------------------------------------------------------------------
    # Inference (override)
    # ------------------------------------------------------------------

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
        anchor_tokens, anchor_meta, _, _ = self._build_anchor_tokens(
            input_pixels,
            anchor_dino_cls=anchor_dino_cls.to(device=device, dtype=torch.float32),
            anchor_dino_registers=(
                anchor_dino_registers.to(device=device, dtype=torch.float32)
                if anchor_dino_registers is not None
                else None
            ),
            anchor_dino_spatial=anchor_dino_spatial.to(device=device, dtype=torch.float32),
            compact_context=context,
            compact_context_mask=context_mask,
        )
        self._profile_cuda_end("anchor_tokens_ms", anchor_event)
        anchor_len = int(anchor_meta["anchor_total"])
        abstract_anchor_len = int(anchor_meta["abstract_anchor_len"])
        current_spatial_len = int(anchor_meta["current_spatial_len"])
        K_core = int(self.compact_core_tokens)

        video_ctx, action_ctx, video_cross_attn_mask, action_cross_attn_mask = self._normalize_context(
            context, context_mask, device, dtype, proprio=proprio,
        )
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
        timesteps, deltas = self.flow.build_inference_schedule(num_inference_steps, device=device, dtype=dtype)

        current_anchor_freqs = self._build_current_anchor_freqs(
            abstract_anchor_len=abstract_anchor_len,
            current_spatial_len=current_spatial_len,
            device=device,
        )

        if joint_future_denoising is None:
            joint_future_denoising = bool(self.config.default_joint_future_denoising)

        if joint_future_denoising and self.enable_compact_future and K_core > 0:
            future_steps = self.future_blocks if future_semantic_steps is None else int(future_semantic_steps)
            compact_dim = self.effective_compact_dim
            future_state = torch.randn(
                batch_size, future_steps, K_core, compact_dim,
                device=device, dtype=dtype, generator=generator,
            )
            future_freqs = self._build_compact_future_freqs(future_steps, K_core, device)
            video_freqs = torch.cat([current_anchor_freqs, future_freqs], dim=0)

            # Build independent inference schedules for action and future streams.
            sem_timesteps, sem_deltas = self.flow.build_inference_schedule(
                num_inference_steps, device=device, dtype=dtype,
            )
            act_timesteps, act_deltas = self.flow.build_inference_schedule(
                num_inference_steps, device=device, dtype=dtype,
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
                action_tokens_local = self._build_action_tokens(action_state, proprio=proprio)

                future_video_tokens = self._build_future_video_tokens(future_state)
                future_video_len = future_video_tokens.shape[1]
                video_tokens = torch.cat([anchor_tokens, future_video_tokens], dim=1)

                self_mask = build_fastwam_style_joint_mask(
                    batch_size,
                    anchor_len,
                    future_video_len,
                    action_tokens_local.shape[1],
                    device,
                    abstract_anchor_len=abstract_anchor_len,
                    current_spatial_len=current_spatial_len,
                    action_prefix_len=action_prefix_len,
                    action_attends_future_video=self.config.action_attends_future_video,
                    future_video_attends_current_spatial=self.config.future_video_attends_current_spatial,
                )
                video_tmod = self._video_token_time_mod(t_sem, anchor_len=anchor_len, future_video_len=future_video_len)

                outputs_local = self.mot(
                    expert_tokens={"video": video_tokens, "action": action_tokens_local},
                    expert_time_mod={"video": video_tmod, "action": action_tmod},
                    self_attn_mask=self_mask,
                    expert_freqs={"video": video_freqs, "action": action_freqs},
                    expert_context={"video": video_ctx, "action": action_ctx},
                    expert_context_mask={"video": video_cross_attn_mask, "action": action_cross_attn_mask},
                )

                pred_velocity = self._forward_module_fp32(self.action_head, outputs_local["action"][:, action_prefix_len:]) * action_channel_mask_f
                action_state = self.flow.step(action_state, pred_velocity, delta_act) * action_channel_mask_f
                if committed_prefix is not None:
                    action_state[:, :committed_len] = committed_prefix

                future_out = outputs_local["video"][:, anchor_len:]
                future_pred = self._decode_future_outputs(future_out, batch_size, future_steps)
                future_state = self.flow.step(future_state, future_pred, delta_sem)

            return action_state * action_channel_mask_f

        self_mask = build_fastwam_style_joint_mask(
            batch_size,
            anchor_len,
            0,
            horizon + action_prefix_len,
            device,
            abstract_anchor_len=abstract_anchor_len,
            current_spatial_len=current_spatial_len,
            action_prefix_len=action_prefix_len,
            action_attends_future_video=self.config.action_attends_future_video,
        )
        current_video_mask = build_fastwam_style_joint_mask(
            batch_size,
            anchor_len,
            0,
            0,
            device,
            abstract_anchor_len=abstract_anchor_len,
            current_spatial_len=current_spatial_len,
        )
        zero_t = torch.zeros(batch_size, device=device, dtype=dtype)
        video_tmod = self._time_mod(zero_t, self.hidden_dim, self.video_time_mlp)
        prefill_event = self._profile_cuda_start("video_prefill_ms", device)
        video_prefill_fn = getattr(self, "_video_prefill_fn", self.mot.prefill_video_cache_flat)
        video_cache = video_prefill_fn(
            video_tokens=anchor_tokens,
            video_time_mod=video_tmod,
            video_freqs=current_anchor_freqs,
            video_context=video_ctx,
            video_context_mask=video_cross_attn_mask,
            video_attention_mask=current_video_mask,
        )
        self._profile_cuda_end("video_prefill_ms", prefill_event)

        # See SemanticFastWAM.infer_action(): compiled V-JEPA can leave
        # inference tensors in the action-step inputs. Materialize normal
        # tensors only at this CUDA-Graph boundary.
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
            # Use [B,T] in both modes so normal and fixed-K RTC requests
            # replay the same denoise graph. Values remain unchanged.
            t_cur = timesteps[step_idx].expand(batch_size).unsqueeze(1).expand(-1, horizon)
            if committed_len:
                t_cur = t_cur.clone()
                t_cur[:, :committed_len] = 0
            delta = deltas[step_idx].expand(batch_size)
            # See SemanticFastWAM.infer_action(): retain this state outside
            # CUDA-Graph-owned output storage before the next denoise replay.
            action_state = action_denoise_step_fn(
                action_state,
                t_cur,
                delta,
                action_channel_mask_f,
                action_freqs,
                video_cache[0],
                video_cache[1],
                self_mask,
                anchor_len,
                action_ctx,
                action_cross_attn_mask,
                proprio,
                action_prefix_len,
            ).clone()
            if committed_prefix is not None:
                action_state[:, :committed_len] = committed_prefix

        self._profile_cuda_end("action_denoise_ms", denoise_event)
        return action_state * action_channel_mask_f
