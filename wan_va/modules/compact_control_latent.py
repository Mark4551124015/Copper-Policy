from __future__ import annotations

import copy
import math
from typing import Optional

import torch
import torch.nn as nn

from .semantic_mot_blocks import (
    CrossAttention,
    FeedForward,
    PerceiverResampler,
    RMSNorm,
    scaled_dot_product_attention,
    sinusoidal_embedding_1d,
)


QUERY_TYPE_H = 0
QUERY_TYPE_V = 1
QUERY_TYPE_G = 2


class TaskAwareSlotReadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim

        self.q = nn.Linear(self.dim, self.inner_dim)
        self.k = nn.Linear(self.dim, self.inner_dim)
        self.v = nn.Linear(self.dim, self.inner_dim)
        self.o = nn.Linear(self.inner_dim, self.dim)
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

    def forward(
        self,
        query_tokens: torch.Tensor,
        *,
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        q = self.norm_q(self.q(query_tokens))
        k = self.norm_k(self.k(key_tokens))
        v = self.v(value_tokens)

        if attn_bias is None and not return_attn:
            out = scaled_dot_product_attention(q, k, v, num_heads=self.num_heads)
            return self.o(out)

        batch_size, q_len, _ = q.shape
        kv_len = k.shape[1]
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attn_bias is not None:
            logits = logits + attn_bias
        attn = logits.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, q_len, self.inner_dim)
        out = self.o(out)
        return (out, attn) if return_attn else out


class TaskAwareResamplerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        *,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=eps)
        self.read_attn = TaskAwareSlotReadAttention(dim=dim, num_heads=num_heads, head_dim=head_dim, eps=eps)
        self.norm_ffn = nn.LayerNorm(dim, eps=eps)
        self.ffn = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(
        self,
        query_tokens: torch.Tensor,
        *,
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        read_out = self.read_attn(
            self.norm_q(query_tokens),
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            attn_bias=attn_bias,
            return_attn=return_attn,
        )
        if return_attn:
            delta, attn = read_out
        else:
            delta = read_out
            attn = None
        query_tokens = query_tokens + delta
        query_tokens = query_tokens + self.ffn(self.norm_ffn(query_tokens))
        return (query_tokens, attn) if return_attn else query_tokens


class CompactControlLatent(nn.Module):
    """Compress encoder features into compact latent tokens.

    Supports two layouts:
    - `all_global`: standard Perceiver resampler over all spatial tokens.
    - `spatial_global`: masked horizontal/vertical/global queries that preserve
      explicit spatial structure (the "7v7h decomp" path when the patch grid is
      14x14 and the band size is 2x2).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        eps: float = 1e-6,
        norm_output: bool = True,
        proj_dim: int = 0,
        proj_hidden_mult: int = 2,
        use_abs_pos_embed: bool = False,
        use_fourier_film_pos: bool = False,
        fourier_num_bands: int = 8,
        fourier_include_local_xy: bool = True,
        fourier_include_global_xy: bool = True,
        fourier_include_view_embed: bool = True,
        fourier_film_init_scale: float = 0.0,
        conditioning_mode: str = "none",
        conditioning_dim: Optional[int] = None,
        conditioning_use_patch_logit_bias: bool = False,
        view_rope_layout: str = "horizontal",
        num_views: int = 1,
        grid_h: int = 16,
        grid_w: int = 16,
        per_view_grid_sizes: Optional[tuple[tuple[int, int], ...]] = None,
        use_attn_spatial_writeback: bool = False,
        attention_mode: str = "standard",
        competitive_temperature: float = 1.0,
        competitive_eps: float = 1e-6,
        spatial_writeback_hidden_mult: int = 4,
        spatial_writeback_init_scale: float = 0.0,
        spatial_writeback_detach_attn: bool = False,
        spatial_moment_mode: str = "view_center_spread",
        token_layout: str = "all_global",
        token_scope: str = "merged_views",
        spatial_h_tokens: int = 0,
        spatial_v_tokens: int = 0,
        global_tokens: int = 0,
        view_global_tokens: Optional[tuple[int, ...]] = None,
        spatial_view_indices: Optional[tuple[int, ...]] = None,
        spatial_mode: str = "axis_spatial",
        spatial_band_rows: int = 2,
        spatial_band_cols: int = 2,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_tokens = int(num_tokens)
        self.num_views = int(num_views)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        if per_view_grid_sizes is None:
            self.per_view_grid_sizes = tuple((self.grid_h, self.grid_w) for _ in range(self.num_views))
        else:
            self.per_view_grid_sizes = tuple((int(h), int(w)) for h, w in per_view_grid_sizes)
        if len(self.per_view_grid_sizes) != self.num_views:
            raise ValueError(
                "per_view_grid_sizes must provide one (grid_h, grid_w) pair per view, "
                f"got {len(self.per_view_grid_sizes)} for num_views={self.num_views}"
            )
        self.per_view_tokens = tuple(int(h) * int(w) for h, w in self.per_view_grid_sizes)
        self.per_view_offsets = []
        cursor = 0
        for tokens_per_view in self.per_view_tokens:
            self.per_view_offsets.append(cursor)
            cursor += int(tokens_per_view)
        self.per_view_offsets = tuple(self.per_view_offsets)
        self.total_spatial_tokens = int(cursor)
        self.tokens_per_view = self.per_view_tokens[0]
        self.view_rope_layout = str(view_rope_layout or "horizontal").strip().lower()
        if self.view_rope_layout not in {"horizontal", "vertical", "robotwin_tshape"}:
            raise ValueError(
                f"Unsupported view_rope_layout={view_rope_layout!r}. "
                "Expected one of: horizontal, vertical, robotwin_tshape."
            )
        if self.view_rope_layout == "horizontal":
            self.virtual_grid_h = max(grid_h_i for grid_h_i, _grid_w_i in self.per_view_grid_sizes)
            self.virtual_grid_w = sum(grid_w_i for _grid_h_i, grid_w_i in self.per_view_grid_sizes)
        elif self.view_rope_layout == "vertical":
            self.virtual_grid_h = sum(grid_h_i for grid_h_i, _grid_w_i in self.per_view_grid_sizes)
            self.virtual_grid_w = max(grid_w_i for _grid_h_i, grid_w_i in self.per_view_grid_sizes)
        else:
            top_h, top_w = self.per_view_grid_sizes[0]
            bottom_row_h = max((grid_h_i for grid_h_i, _grid_w_i in self.per_view_grid_sizes[1:]), default=0)
            bottom_row_w = sum(grid_w_i for _grid_h_i, grid_w_i in self.per_view_grid_sizes[1:])
            self.virtual_grid_h = top_h + bottom_row_h
            self.virtual_grid_w = max(top_w, bottom_row_w)
        self.token_layout = str(token_layout or "").strip().lower()
        if self.token_layout in {"", "all_global", "global", "global_only"}:
            self.token_layout = "all_global"
        elif self.token_layout in {"spatial_global", "spatial+global", "axis_global", "hv_global"}:
            self.token_layout = "spatial_global"
        else:
            raise ValueError(
                f"Unsupported token_layout={token_layout!r}. Expected all_global or spatial_global."
            )
        self.token_scope = str(token_scope or "merged_views").strip().lower()
        if self.token_layout == "spatial_global":
            if self.token_scope in {"", "hybrid"}:
                self.token_scope = "hybrid"
            elif self.token_scope in {"shared", "shared_views", "cross_view_shared"}:
                self.token_scope = "shared_views"
            else:
                raise ValueError(
                    f"Unsupported token_scope={token_scope!r} for spatial_global layout. "
                    "Expected hybrid or shared_views."
                )
        self.spatial_h_tokens = int(spatial_h_tokens)
        self.spatial_v_tokens = int(spatial_v_tokens)
        self.global_tokens = int(global_tokens)
        if view_global_tokens is None:
            self.view_global_tokens = tuple(0 for _ in range(self.num_views))
        else:
            self.view_global_tokens = tuple(int(v) for v in view_global_tokens)
        if spatial_view_indices is None:
            self.spatial_view_indices = tuple(range(self.num_views))
        else:
            self.spatial_view_indices = tuple(int(v) for v in spatial_view_indices)
        self.spatial_mode = str(spatial_mode or "").strip().lower()
        if self.spatial_mode in {"", "axis", "axis_spatial", "7v7h", "hv", "h_v"}:
            self.spatial_mode = "axis_spatial"
        elif self.spatial_mode in {"v", "v_anchor", "v-anchor"}:
            self.spatial_mode = "v_anchor"
        else:
            raise ValueError(
                f"Unsupported spatial_mode={spatial_mode!r}. Expected axis_spatial or v_anchor."
            )
        self.spatial_band_rows = int(spatial_band_rows)
        self.spatial_band_cols = int(spatial_band_cols)
        self.per_view_spatial_h_tokens = tuple(
            (grid_h_i + self.spatial_band_rows - 1) // self.spatial_band_rows
            for grid_h_i, _grid_w_i in self.per_view_grid_sizes
        )
        self.per_view_spatial_v_tokens = tuple(
            (grid_w_i + self.spatial_band_cols - 1) // self.spatial_band_cols
            for _grid_h_i, grid_w_i in self.per_view_grid_sizes
        )

        self.resampler = PerceiverResampler(
            dim_input=input_dim,
            dim=output_dim,
            num_queries=num_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            eps=eps,
            dropout=dropout,
            learnable_queries=(self.token_layout != "spatial_global"),
        )

        proj_dim = int(proj_dim)
        if proj_dim > 0:
            proj_hidden = int(output_dim) * int(proj_hidden_mult)
            self.proj = nn.Sequential(
                nn.Linear(int(output_dim), proj_hidden),
                nn.GELU(),
                nn.Linear(proj_hidden, proj_dim),
            )
            self.effective_output_dim = proj_dim
        else:
            self.proj = nn.Identity()
            self.effective_output_dim = int(output_dim)

        self.norm_output = (
            nn.LayerNorm(self.effective_output_dim, eps=eps) if norm_output else nn.Identity()
        )

        self.use_abs_pos_embed = bool(use_abs_pos_embed)
        if self.use_abs_pos_embed:
            if len(set(self.per_view_grid_sizes)) != 1:
                raise ValueError("use_abs_pos_embed=True is not supported with heterogeneous per_view_grid_sizes")
            self.view_embed = nn.Embedding(self.num_views, self.input_dim)
            nn.init.zeros_(self.view_embed.weight)
            self.pos_embed = nn.Parameter(torch.zeros(self.grid_h, self.grid_w, self.input_dim))
        else:
            self.view_embed = None
            self.pos_embed = None

        self.use_fourier_film_pos = bool(use_fourier_film_pos)
        self.fourier_num_bands = int(fourier_num_bands)
        self.fourier_include_local_xy = bool(fourier_include_local_xy)
        self.fourier_include_global_xy = bool(fourier_include_global_xy)
        self.fourier_include_view_embed = bool(fourier_include_view_embed)
        self.conditioning_mode = str(conditioning_mode or "none").strip().lower()
        if self.conditioning_mode in {"", "none", "off", "disabled"}:
            self.conditioning_mode = "none"
        elif self.conditioning_mode not in {"patch_motivation_key"}:
            raise ValueError(
                f"Unsupported conditioning_mode={conditioning_mode!r}. Expected none or patch_motivation_key."
            )
        self.conditioning_dim = None if conditioning_dim is None else int(conditioning_dim)
        self.conditioning_use_patch_logit_bias = bool(conditioning_use_patch_logit_bias)
        if self.use_fourier_film_pos and self.token_layout != "all_global":
            raise ValueError("use_fourier_film_pos=True is only supported for token_layout='all_global'")
        if self.use_fourier_film_pos and self.fourier_num_bands <= 0:
            raise ValueError(f"fourier_num_bands must be positive, got {self.fourier_num_bands}")
        if self.use_fourier_film_pos and not (
            self.fourier_include_local_xy
            or self.fourier_include_global_xy
            or self.fourier_include_view_embed
        ):
            raise ValueError("Fourier FiLM requires at least one of local_xy, global_xy, or view_embed")
        self.fourier_xy_feature_dim = 4 * self.fourier_num_bands
        self.fourier_view_feature_dim = self.fourier_xy_feature_dim if self.fourier_include_view_embed else 0
        self.fourier_feature_dim = (
            int(self.fourier_include_local_xy) * self.fourier_xy_feature_dim
            + int(self.fourier_include_global_xy) * self.fourier_xy_feature_dim
            + self.fourier_view_feature_dim
        )
        if self.use_fourier_film_pos:
            self._init_fourier_film_state(eps=eps, init_scale=fourier_film_init_scale)
        else:
            self.fourier_film_norm = None
            self.fourier_film_mlp = None
            self.fourier_view_embed = None
            self.fourier_film_alpha = None
            self.register_buffer("_fourier_local_xy", torch.empty(0, 2), persistent=False)
            self.register_buffer("_fourier_global_xy", torch.empty(0, 2), persistent=False)
            self.register_buffer("_fourier_view_ids", torch.empty(0, dtype=torch.long), persistent=False)

        self.attention_mode = str(attention_mode)
        self.competitive_temperature = float(competitive_temperature)
        self.competitive_eps = float(competitive_eps)

        self.use_attn_spatial_writeback = bool(use_attn_spatial_writeback)
        self.spatial_writeback_detach_attn = bool(spatial_writeback_detach_attn)
        self.spatial_moment_mode = str(spatial_moment_mode)
        if self.use_attn_spatial_writeback:
            coords_chunks = []
            view_ids_chunks = []
            for view_idx, (grid_h_i, grid_w_i) in enumerate(self.per_view_grid_sizes):
                col = torch.arange(grid_w_i, dtype=torch.float32)
                row = torch.arange(grid_h_i, dtype=torch.float32)
                x = (col + 0.5) / float(grid_w_i) * 2.0 - 1.0
                y = (row + 0.5) / float(grid_h_i) * 2.0 - 1.0
                grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
                coords_chunks.append(torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1))
                view_ids_chunks.append(torch.full((grid_h_i * grid_w_i,), view_idx, dtype=torch.long))
            coords = torch.cat(coords_chunks, dim=0)
            self.register_buffer("_spatial_coords", coords, persistent=True)
            view_ids = torch.cat(view_ids_chunks, dim=0)
            self.register_buffer("_spatial_view_ids", view_ids, persistent=True)
            if spatial_moment_mode != "view_center_spread":
                raise ValueError(
                    f"Unsupported spatial_moment_mode={spatial_moment_mode!r}. Expected 'view_center_spread'."
                )
            moment_dim = 5 * self.num_views
            hidden = int(self.output_dim) * int(spatial_writeback_hidden_mult)
            self.spatial_writeback_mlp = nn.Sequential(
                nn.Linear(moment_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.output_dim),
            )
            self.spatial_writeback_scale = nn.Parameter(torch.tensor(float(spatial_writeback_init_scale)))
        else:
            self._spatial_coords = None
            self._spatial_view_ids = None
            self.spatial_writeback_mlp = None
            self.spatial_writeback_scale = None

        if self.token_layout == "spatial_global":
            if self.spatial_mode != "axis_spatial":
                raise ValueError("Only axis_spatial is implemented for spatial_global layout in this branch.")
            self._init_spatial_global_layout()
        else:
            self.query_content = None
            self.query_type_embed = None
            self.query_view_embed = None
            self.register_buffer("layout_query_type_ids", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("layout_query_slot_ids", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("layout_query_view_ids", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("layout_query_attn_mask", torch.empty(0), persistent=False)

        if self.conditioning_mode != "none":
            if self.conditioning_dim is None or self.conditioning_dim <= 0:
                raise ValueError(
                    "conditioning_dim must be a positive integer when conditioning_mode is enabled."
                )
            # The task-conditioned path keeps the original learnable slot queries
            # but bypasses the legacy Perceiver resampler blocks entirely.
            # as active trainable parameters.
            for param in self.resampler.layers.parameters():
                param.requires_grad_(False)
            comparator_hidden = max(int(self.output_dim) * 4, 1536)
            self.conditioning_patch_proj = nn.Linear(self.input_dim, self.output_dim)
            self.conditioning_motivation_proj = nn.Linear(self.conditioning_dim, self.output_dim)
            self.conditioning_patch_to_motivation = CrossAttention(
                hidden_dim=self.output_dim,
                context_dim=self.output_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                eps=eps,
            )
            self.conditioning_key_mlp = nn.Sequential(
                nn.Linear(self.output_dim * 3, comparator_hidden),
                nn.GELU(),
                nn.Linear(comparator_hidden, self.output_dim),
            )
            self.conditioning_value_proj = nn.Linear(self.output_dim, self.output_dim)
            if self.conditioning_use_patch_logit_bias:
                self.conditioning_patch_score = nn.Sequential(
                    nn.Linear(self.output_dim * 3, comparator_hidden),
                    nn.GELU(),
                    nn.Linear(comparator_hidden, 1),
                )
            else:
                self.conditioning_patch_score = None
            self.conditioning_read_blocks = nn.ModuleList(
                [
                    TaskAwareResamplerBlock(
                        dim=self.output_dim,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        mlp_ratio=mlp_ratio,
                        eps=eps,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.conditioning_patch_proj = None
            self.conditioning_motivation_proj = None
            self.conditioning_patch_to_motivation = None
            self.conditioning_key_mlp = None
            self.conditioning_value_proj = None
            self.conditioning_patch_score = None
            self.conditioning_read_blocks = None

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

    def _normalize_xy(self, rows: torch.Tensor, cols: torch.Tensor, *, grid_h: int, grid_w: int) -> torch.Tensor:
        x = (cols + 0.5) / max(float(grid_w), 1.0) * 2.0 - 1.0
        y = (rows + 0.5) / max(float(grid_h), 1.0) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    def _build_fourier_film_coords(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_chunks = []
        global_chunks = []
        view_id_chunks = []
        for view_idx, (grid_h_i, grid_w_i) in enumerate(self.per_view_grid_sizes):
            row = torch.arange(grid_h_i, dtype=torch.float32)
            col = torch.arange(grid_w_i, dtype=torch.float32)
            grid_row, grid_col = torch.meshgrid(row, col, indexing="ij")
            local_chunks.append(
                self._normalize_xy(grid_row.reshape(-1), grid_col.reshape(-1), grid_h=grid_h_i, grid_w=grid_w_i)
            )
            row0, col0 = self._view_origin_coord(view_idx)
            global_chunks.append(
                self._normalize_xy(
                    grid_row.reshape(-1) + float(row0),
                    grid_col.reshape(-1) + float(col0),
                    grid_h=self.virtual_grid_h,
                    grid_w=self.virtual_grid_w,
                )
            )
            view_id_chunks.append(torch.full((grid_h_i * grid_w_i,), view_idx, dtype=torch.long))
        return (
            torch.cat(local_chunks, dim=0),
            torch.cat(global_chunks, dim=0),
            torch.cat(view_id_chunks, dim=0),
        )

    def _init_fourier_film_state(self, *, eps: float, init_scale: float) -> None:
        local_xy, global_xy, view_ids = self._build_fourier_film_coords()
        self.register_buffer("_fourier_local_xy", local_xy, persistent=False)
        self.register_buffer("_fourier_global_xy", global_xy, persistent=False)
        self.register_buffer("_fourier_view_ids", view_ids, persistent=False)
        self.fourier_film_norm = nn.LayerNorm(self.input_dim, eps=eps)
        hidden_dim = max(self.input_dim, self.fourier_feature_dim * 2)
        self.fourier_film_mlp = nn.Sequential(
            nn.Linear(self.fourier_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.input_dim * 2),
        )
        if self.fourier_include_view_embed:
            self.fourier_view_embed = nn.Embedding(self.num_views, self.fourier_view_feature_dim)
            nn.init.normal_(self.fourier_view_embed.weight, std=0.02)
        else:
            self.fourier_view_embed = None
        self.fourier_film_alpha = nn.Parameter(torch.tensor(float(init_scale)))

    def _add_spatial_embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        n = self.tokens_per_view
        emb_parts = []
        for view_idx in range(self.num_views):
            v_emb = self.view_embed.weight[view_idx].unsqueeze(0).expand(n, -1)
            p_emb = self.pos_embed.reshape(n, self.input_dim)
            emb_parts.append(v_emb + p_emb)
        ctx_emb = torch.cat(emb_parts, dim=0)
        if tokens.ndim == 4:
            ctx_emb = ctx_emb.unsqueeze(0).unsqueeze(0)
        elif tokens.ndim == 3:
            ctx_emb = ctx_emb.unsqueeze(0)
        else:
            raise ValueError(f"Expected tokens with ndim=3 or 4, got shape {tuple(tokens.shape)}")
        return tokens + ctx_emb.to(dtype=tokens.dtype, device=tokens.device)

    def _attention_to_spatial_moment(self, attn: torch.Tensor) -> torch.Tensor:
        bsz, _nh, _k, _vn = attn.shape
        a = attn.mean(dim=1)
        moment_parts: list[torch.Tensor] = []
        for view_idx in range(self.num_views):
            view_mask = self._spatial_view_ids == view_idx
            a_v = a[..., view_mask]
            view_mass = a_v.sum(dim=-1)
            a_v_norm = a_v / a_v.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            coords_v = self._spatial_coords[view_mask]
            centre = (a_v_norm.unsqueeze(-1) * coords_v.unsqueeze(0).unsqueeze(0)).sum(dim=-2)
            delta = coords_v.unsqueeze(0).unsqueeze(0) - centre.unsqueeze(-2)
            spread = (a_v_norm.unsqueeze(-1) * delta.pow(2)).sum(dim=-2)
            moment_parts.extend([view_mass, centre[..., 0], centre[..., 1], spread[..., 0], spread[..., 1]])
        return torch.stack(moment_parts, dim=-1).view(bsz, a.shape[1], -1)

    def _build_resampler_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.use_attn_spatial_writeback:
            kwargs["return_attn"] = True
        if self.attention_mode != "standard":
            kwargs["attention_mode"] = self.attention_mode
            kwargs["competitive_temperature"] = self.competitive_temperature
            kwargs["competitive_eps"] = self.competitive_eps
        return kwargs

    def _apply_spatial_writeback(self, q: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
        moment = self._attention_to_spatial_moment(attn)
        if self.spatial_writeback_detach_attn:
            moment = moment.detach()
        delta = self.spatial_writeback_mlp(moment)
        return q + self.spatial_writeback_scale * delta

    def _fourier_encode_xy(
        self,
        coords: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        freqs = (2.0 ** torch.arange(self.fourier_num_bands, device=device, dtype=torch.float32)) * math.pi
        angles = coords.to(device=device, dtype=torch.float32).unsqueeze(-1) * freqs.view(1, 1, -1)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return torch.cat([sin[..., 0, :], cos[..., 0, :], sin[..., 1, :], cos[..., 1, :]], dim=-1).to(dtype=dtype)

    def _build_fourier_film_features(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        parts = []
        if self.fourier_include_global_xy:
            parts.append(self._fourier_encode_xy(self._fourier_global_xy, device=device, dtype=dtype))
        if self.fourier_include_local_xy:
            parts.append(self._fourier_encode_xy(self._fourier_local_xy, device=device, dtype=dtype))
        if self.fourier_view_embed is not None:
            parts.append(self.fourier_view_embed(self._fourier_view_ids.to(device=device)).to(dtype=dtype))
        if not parts:
            raise RuntimeError("Fourier FiLM features are enabled but no position components were constructed")
        return torch.cat(parts, dim=-1)

    def _apply_fourier_film_pos(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.use_fourier_film_pos:
            return tokens
        if tokens.shape[1] != self.total_spatial_tokens:
            raise ValueError(
                f"Fourier FiLM expects {self.total_spatial_tokens} spatial tokens, got {tokens.shape[1]}"
            )
        film_dtype = self.fourier_film_mlp[0].weight.dtype
        pos = self._build_fourier_film_features(device=tokens.device, dtype=film_dtype)
        gamma_beta = self.fourier_film_mlp(pos).to(dtype=tokens.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        x_norm = self.fourier_film_norm(tokens)
        return tokens + self.fourier_film_alpha.to(dtype=tokens.dtype) * (
            gamma.unsqueeze(0) * x_norm + beta.unsqueeze(0)
        )

    def _init_spatial_global_layout(self) -> None:
        if any(grid_h_i <= 0 or grid_w_i <= 0 for grid_h_i, grid_w_i in self.per_view_grid_sizes):
            raise ValueError("grid_h/grid_w must be positive for spatial_global layout")
        if self.spatial_band_rows <= 0 or self.spatial_band_cols <= 0:
            raise ValueError("spatial_band_rows/spatial_band_cols must be positive for spatial_global layout")
        if self.token_scope == "shared_views":
            if self.spatial_h_tokens <= 0 or self.spatial_v_tokens <= 0:
                raise ValueError("spatial_h_tokens/spatial_v_tokens must be positive for spatial_global layout")
            if len(set(self.per_view_grid_sizes)) != 1:
                raise ValueError("shared_views spatial_global layout requires uniform per_view_grid_sizes")

        if self.token_scope == "shared_views":
            expected_tokens = self.global_tokens + sum(self.view_global_tokens) + self.spatial_h_tokens + self.spatial_v_tokens
        else:
            expected_tokens = (
                self.global_tokens
                + sum(self.view_global_tokens)
                + sum(
                    int(self.per_view_spatial_h_tokens[view_idx]) + int(self.per_view_spatial_v_tokens[view_idx])
                    for view_idx in self.spatial_view_indices
                )
            )
        if expected_tokens != self.num_tokens:
            raise ValueError(
                f"spatial_global layout expects num_tokens={expected_tokens}, got {self.num_tokens}"
            )

        def _view_token_indices(view_idx: int) -> torch.Tensor:
            start = self.per_view_offsets[view_idx]
            end = start + self.per_view_tokens[view_idx]
            return torch.arange(start, end, dtype=torch.long)

        def _horizontal_band_indices(view_idx: int, slot_idx: int) -> torch.Tensor:
            grid_h_i, grid_w_i = self.per_view_grid_sizes[view_idx]
            row_start = slot_idx * self.spatial_band_rows
            row_end = min(row_start + self.spatial_band_rows, grid_h_i)
            rows = []
            base = self.per_view_offsets[view_idx]
            for row_idx in range(row_start, row_end):
                start = base + row_idx * grid_w_i
                rows.append(torch.arange(start, start + grid_w_i, dtype=torch.long))
            return torch.cat(rows, dim=0)

        def _vertical_band_indices(view_idx: int, slot_idx: int) -> torch.Tensor:
            grid_h_i, grid_w_i = self.per_view_grid_sizes[view_idx]
            col_start = slot_idx * self.spatial_band_cols
            col_end = min(col_start + self.spatial_band_cols, grid_w_i)
            cols = []
            base = self.per_view_offsets[view_idx]
            for row_idx in range(grid_h_i):
                row_base = base + row_idx * grid_w_i
                cols.append(torch.arange(row_base + col_start, row_base + col_end, dtype=torch.long))
            return torch.cat(cols, dim=0)

        def _build_mask_row(visible_indices: torch.Tensor) -> torch.Tensor:
            row = torch.full(
                (self.total_spatial_tokens,),
                float("-inf"),
                dtype=torch.float32,
            )
            row[visible_indices] = 0.0
            return row

        query_type_ids = []
        query_slot_ids = []
        query_view_ids = []
        mask_rows = []

        all_view_indices = torch.cat([_view_token_indices(view_idx) for view_idx in range(self.num_views)], dim=0)
        for slot_idx in range(self.global_tokens):
            query_type_ids.append(QUERY_TYPE_G)
            query_slot_ids.append(slot_idx)
            query_view_ids.append(-1)
            mask_rows.append(_build_mask_row(all_view_indices))

        for view_idx in range(self.num_views):
            for slot_idx in range(self.view_global_tokens[view_idx]):
                query_type_ids.append(QUERY_TYPE_G)
                query_slot_ids.append(slot_idx)
                query_view_ids.append(view_idx)
                mask_rows.append(_build_mask_row(_view_token_indices(view_idx)))

        if self.token_scope == "shared_views":
            for slot_idx in range(self.spatial_h_tokens):
                query_type_ids.append(QUERY_TYPE_H)
                query_slot_ids.append(slot_idx)
                query_view_ids.append(-1)
                visible = torch.cat(
                    [_horizontal_band_indices(view_idx, slot_idx) for view_idx in range(self.num_views)],
                    dim=0,
                )
                mask_rows.append(_build_mask_row(visible))
            for slot_idx in range(self.spatial_v_tokens):
                query_type_ids.append(QUERY_TYPE_V)
                query_slot_ids.append(slot_idx)
                query_view_ids.append(-1)
                visible = torch.cat(
                    [_vertical_band_indices(view_idx, slot_idx) for view_idx in range(self.num_views)],
                    dim=0,
                )
                mask_rows.append(_build_mask_row(visible))
        else:
            for view_idx in self.spatial_view_indices:
                for slot_idx in range(self.per_view_spatial_h_tokens[view_idx]):
                    query_type_ids.append(QUERY_TYPE_H)
                    query_slot_ids.append(slot_idx)
                    query_view_ids.append(view_idx)
                    mask_rows.append(_build_mask_row(_horizontal_band_indices(view_idx, slot_idx)))
                for slot_idx in range(self.per_view_spatial_v_tokens[view_idx]):
                    query_type_ids.append(QUERY_TYPE_V)
                    query_slot_ids.append(slot_idx)
                    query_view_ids.append(view_idx)
                    mask_rows.append(_build_mask_row(_vertical_band_indices(view_idx, slot_idx)))

        self.query_content = nn.Parameter(torch.zeros(1, self.num_tokens, self.output_dim))
        nn.init.normal_(self.query_content, std=0.02)
        self.query_type_embed = nn.Embedding(3, self.output_dim)
        self.query_view_embed = None if self.token_scope == "shared_views" else nn.Embedding(self.num_views, self.output_dim)
        nn.init.normal_(self.query_type_embed.weight, std=0.02)
        if self.query_view_embed is not None:
            nn.init.normal_(self.query_view_embed.weight, std=0.02)

        self.register_buffer("layout_query_type_ids", torch.tensor(query_type_ids, dtype=torch.long), persistent=False)
        self.register_buffer("layout_query_slot_ids", torch.tensor(query_slot_ids, dtype=torch.long), persistent=False)
        self.register_buffer("layout_query_view_ids", torch.tensor(query_view_ids, dtype=torch.long), persistent=False)
        self.register_buffer(
            "layout_query_attn_mask",
            torch.stack(mask_rows, dim=0).unsqueeze(0).unsqueeze(0),
            persistent=False,
        )

    def _build_masked_spatial_queries(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.query_content is None or self.query_type_embed is None:
            raise RuntimeError("masked spatial query layout is not initialized")
        type_ids = self.layout_query_type_ids.to(device=device)
        slot_ids = self.layout_query_slot_ids.to(device=device)
        query = self.query_content.to(device=device, dtype=dtype)
        query = query + self.query_type_embed(type_ids).unsqueeze(0).to(dtype=dtype)

        # Do not use boolean-indexed in-place writes here.  Inductor lowers
        # them to index_put_, which CUDA Graph capture does not permit.  The
        # masks below produce exactly the same per-query H/V positions while
        # remaining a pure, capture-safe tensor expression.
        h_mask = (type_ids == QUERY_TYPE_H).to(dtype=dtype).view(1, -1, 1)
        h_pos = (slot_ids * self.spatial_band_rows).to(dtype=torch.float32)
        h_embed = sinusoidal_embedding_1d(self.output_dim, h_pos, out_dtype=dtype).unsqueeze(0)
        v_mask = (type_ids == QUERY_TYPE_V).to(dtype=dtype).view(1, -1, 1)
        v_pos = (slot_ids * self.spatial_band_cols).to(dtype=torch.float32)
        v_embed = sinusoidal_embedding_1d(self.output_dim, v_pos, out_dtype=dtype).unsqueeze(0)
        query = query + h_embed * h_mask + v_embed * v_mask

        if self.query_view_embed is not None:
            view_ids = self.layout_query_view_ids.to(device=device)
            valid_view = (view_ids >= 0).to(dtype=dtype).view(1, -1, 1)
            safe_view_ids = view_ids.clamp_min(0)
            view_embed = self.query_view_embed(safe_view_ids).unsqueeze(0).to(dtype=dtype)
            query = query + view_embed * valid_view

        return query.expand(batch_size, -1, -1)

    def _forward_all_global(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.use_fourier_film_pos:
            tokens = self._apply_fourier_film_pos(tokens)
        resampler_kwargs = self._build_resampler_kwargs()
        result = self.resampler(tokens, **resampler_kwargs)
        if self.use_attn_spatial_writeback:
            q, attn = result
            q = self._apply_spatial_writeback(q, attn)
        else:
            q = result
        return self.norm_output(self.proj(q))

    def _build_conditioning_mask(
        self,
        conditioning_mask: torch.Tensor,
        *,
        num_queries: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        mask = conditioning_mask.to(device=device, dtype=torch.bool)
        if mask.ndim != 2:
            raise ValueError(f"conditioning_mask must be [B,L], got {tuple(mask.shape)}")
        has_valid = mask.any(dim=-1)
        safe_mask = mask.clone()
        if (~has_valid).any():
            safe_mask[~has_valid, 0] = True
        attn_mask = torch.zeros(
            safe_mask.shape[0],
            1,
            num_queries,
            safe_mask.shape[1],
            device=device,
            dtype=dtype,
        )
        attn_mask.masked_fill_(~safe_mask[:, None, None, :], float("-inf"))
        return attn_mask, has_valid.to(device=device)

    def _forward_conditioned_queries(
        self,
        query_tokens: torch.Tensor,
        tokens: torch.Tensor,
        *,
        conditioning_context: Optional[torch.Tensor],
        conditioning_mask: Optional[torch.Tensor],
        query_attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if conditioning_context is None:
            if self.token_layout == "spatial_global":
                return self._forward_spatial_global(tokens)
            return self._forward_all_global(tokens)
        if conditioning_context.ndim != 3:
            raise ValueError(
                "conditioning_context must be [B,L,D], "
                f"got {tuple(conditioning_context.shape)}"
            )
        if conditioning_context.shape[0] != tokens.shape[0]:
            raise ValueError(
                "conditioning_context batch mismatch: "
                f"context={conditioning_context.shape[0]} tokens={tokens.shape[0]}"
            )
        if conditioning_context.shape[-1] != self.conditioning_dim:
            raise ValueError(
                f"conditioning_context dim mismatch: expected {self.conditioning_dim}, "
                f"got {conditioning_context.shape[-1]}"
            )

        if self.use_fourier_film_pos:
            tokens = self._apply_fourier_film_pos(tokens)

        x = self.conditioning_patch_proj(tokens)
        motivation = self.conditioning_motivation_proj(conditioning_context.to(device=x.device, dtype=x.dtype))
        patch_attn_mask = None
        valid_samples = None
        if conditioning_mask is not None:
            patch_attn_mask, valid_samples = self._build_conditioning_mask(
                conditioning_mask,
                num_queries=x.shape[1],
                device=x.device,
                dtype=x.dtype,
            )
        patch_context = self.conditioning_patch_to_motivation(
            x,
            motivation,
            context_mask=patch_attn_mask,
        )
        if valid_samples is not None and (~valid_samples).any():
            patch_context = patch_context.clone()
            patch_context[~valid_samples] = 0

        fused = torch.cat([x, patch_context, x * patch_context], dim=-1)
        key_tokens = self.conditioning_key_mlp(fused)
        value_tokens = self.conditioning_value_proj(x)

        attn_bias = query_attn_bias
        if self.conditioning_patch_score is not None:
            patch_score = self.conditioning_patch_score(fused).transpose(1, 2).unsqueeze(1)
            attn_bias = patch_score if attn_bias is None else attn_bias + patch_score

        last_attn = None
        for block_idx, block in enumerate(self.conditioning_read_blocks):
            need_attn = self.use_attn_spatial_writeback and block_idx == len(self.conditioning_read_blocks) - 1
            out = block(
                query_tokens,
                key_tokens=key_tokens,
                value_tokens=value_tokens,
                attn_bias=attn_bias,
                return_attn=need_attn,
            )
            if need_attn:
                query_tokens, last_attn = out
            else:
                query_tokens = out

        if self.use_attn_spatial_writeback and last_attn is not None:
            query_tokens = self._apply_spatial_writeback(query_tokens, last_attn)
        return self.norm_output(self.proj(query_tokens))

    def _forward_all_global_conditioned(
        self,
        tokens: torch.Tensor,
        *,
        conditioning_context: Optional[torch.Tensor],
        conditioning_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.resampler.queries is None:
            raise RuntimeError("Task-conditioned all_global compactor expects learnable Perceiver queries.")
        query_tokens = self.resampler.queries.expand(tokens.shape[0], -1, -1).to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        return self._forward_conditioned_queries(
            query_tokens,
            tokens,
            conditioning_context=conditioning_context,
            conditioning_mask=conditioning_mask,
        )

    def _forward_spatial_global(self, tokens: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.total_spatial_tokens
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"spatial_global expects {expected_tokens} merged spatial tokens, got {tokens.shape[1]}"
            )

        resampler_kwargs = self._build_resampler_kwargs()
        query_tokens = self._build_masked_spatial_queries(tokens.shape[0], tokens.device, tokens.dtype)
        context_mask = self.layout_query_attn_mask.to(device=tokens.device, dtype=tokens.dtype).expand(
            tokens.shape[0], -1, -1, -1
        )
        result = self.resampler(
            tokens,
            query_tokens=query_tokens,
            context_mask=context_mask,
            **resampler_kwargs,
        )
        if self.use_attn_spatial_writeback:
            q, attn = result
            q = self._apply_spatial_writeback(q, attn)
        else:
            q = result
        return self.norm_output(self.proj(q))

    def _forward_spatial_global_conditioned(
        self,
        tokens: torch.Tensor,
        *,
        conditioning_context: Optional[torch.Tensor],
        conditioning_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        expected_tokens = self.total_spatial_tokens
        if tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"spatial_global expects {expected_tokens} merged spatial tokens, got {tokens.shape[1]}"
            )
        query_tokens = self._build_masked_spatial_queries(tokens.shape[0], tokens.device, tokens.dtype)
        query_attn_bias = self.layout_query_attn_mask.to(device=tokens.device, dtype=tokens.dtype).expand(
            tokens.shape[0], -1, -1, -1
        )
        return self._forward_conditioned_queries(
            query_tokens,
            tokens,
            conditioning_context=conditioning_context,
            conditioning_mask=conditioning_mask,
            query_attn_bias=query_attn_bias,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning_context: Optional[torch.Tensor] = None,
        conditioning_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_abs_pos_embed:
            tokens = self._add_spatial_embedding(tokens)

        if tokens.ndim == 3:
            if self.token_layout == "spatial_global":
                if self.conditioning_mode != "none":
                    return self._forward_spatial_global_conditioned(
                        tokens,
                        conditioning_context=conditioning_context,
                        conditioning_mask=conditioning_mask,
                    )
                return self._forward_spatial_global(tokens)
            if self.conditioning_mode != "none":
                return self._forward_all_global_conditioned(
                    tokens,
                    conditioning_context=conditioning_context,
                    conditioning_mask=conditioning_mask,
                )
            return self._forward_all_global(tokens)

        if tokens.ndim == 4:
            batch_size, steps, num_tokens, dim = tokens.shape
            flat = tokens.reshape(batch_size * steps, num_tokens, dim)
            flat_conditioning_context = None
            flat_conditioning_mask = None
            if conditioning_context is not None:
                if conditioning_context.ndim != 3 or conditioning_context.shape[0] != batch_size:
                    raise ValueError(
                        "conditioning_context must be [B,L,D] when tokens are [B,T,N,D], "
                        f"got {tuple(conditioning_context.shape)} for batch_size={batch_size}"
                    )
                flat_conditioning_context = conditioning_context.unsqueeze(1).expand(-1, steps, -1, -1)
                flat_conditioning_context = flat_conditioning_context.reshape(
                    batch_size * steps,
                    conditioning_context.shape[1],
                    conditioning_context.shape[2],
                )
            if conditioning_mask is not None:
                if conditioning_mask.ndim != 2 or conditioning_mask.shape[0] != batch_size:
                    raise ValueError(
                        "conditioning_mask must be [B,L] when tokens are [B,T,N,D], "
                        f"got {tuple(conditioning_mask.shape)} for batch_size={batch_size}"
                    )
                flat_conditioning_mask = conditioning_mask.unsqueeze(1).expand(-1, steps, -1)
                flat_conditioning_mask = flat_conditioning_mask.reshape(batch_size * steps, conditioning_mask.shape[1])
            if self.token_layout == "spatial_global":
                if self.conditioning_mode != "none":
                    q = self._forward_spatial_global_conditioned(
                        flat,
                        conditioning_context=flat_conditioning_context,
                        conditioning_mask=flat_conditioning_mask,
                    )
                else:
                    q = self._forward_spatial_global(flat)
            elif self.conditioning_mode != "none":
                q = self._forward_all_global_conditioned(
                    flat,
                    conditioning_context=flat_conditioning_context,
                    conditioning_mask=flat_conditioning_mask,
                )
            else:
                q = self._forward_all_global(flat)
            return q.view(batch_size, steps, self.num_tokens, self.effective_output_dim)

        raise ValueError(f"Expected tokens with ndim=3 or 4, got shape {tuple(tokens.shape)}")




def build_ema_compactor(compactor: CompactControlLatent) -> CompactControlLatent:
    ema = copy.deepcopy(compactor)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema
