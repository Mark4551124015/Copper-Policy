from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import rope_apply


# ------------------------------------------------------------
# Core helpers
# ------------------------------------------------------------


def gradient_checkpoint_forward(module: nn.Module, use_gradient_checkpointing: bool, *args, **kwargs):
    if use_gradient_checkpointing and torch.is_grad_enabled():
        return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    while shift.ndim < x.ndim:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1.0 + scale) + shift


def sinusoidal_embedding_1d(
    dim: int,
    position: torch.Tensor,
    *,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    half = dim // 2
    device = position.device
    dtype = position.dtype if out_dtype is None else out_dtype
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(half, 1))
    angles = position.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb.to(dtype)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bsz, seq_q, _ = q.shape
    _, seq_k, _ = k.shape
    head_dim = q.shape[-1] // num_heads
    q = q.view(bsz, seq_q, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_k, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_k, num_heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = out.transpose(1, 2).contiguous().view(bsz, seq_q, num_heads * head_dim)
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return ((x.float() * scale).to(x.dtype) * self.weight).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim

        self.q = nn.Linear(hidden_dim, self.inner_dim)
        self.k = nn.Linear(hidden_dim, self.inner_dim)
        self.v = nn.Linear(hidden_dim, self.inner_dim)
        self.o = nn.Linear(self.inner_dim, hidden_dim)
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

    def project_qkv(
        self,
        x: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        if freqs is not None:
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, k, v = self.project_qkv(x, freqs=freqs)
        return self.o(scaled_dot_product_attention(q, k, v, num_heads=self.num_heads, attn_mask=attn_mask))


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim

        self.q = nn.Linear(hidden_dim, self.inner_dim)
        self.k = nn.Linear(context_dim, self.inner_dim)
        self.v = nn.Linear(context_dim, self.inner_dim)
        self.o = nn.Linear(self.inner_dim, hidden_dim)
        self.norm_q = RMSNorm(self.inner_dim, eps=eps)
        self.norm_k = RMSNorm(self.inner_dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        attention_mode: str = "standard",
        competitive_temperature: float = 1.0,
        competitive_eps: float = 1e-6,
    ):
        if context is None:
            out = torch.zeros_like(x)
            return (out, None) if return_attn else out

        q = self.norm_q(self.q(x))        # [B, K, inner_dim]
        k = self.norm_k(self.k(context))  # [B, N, inner_dim]
        v = self.v(context)               # [B, N, inner_dim]

        # Fast path: no attn needed and standard softmax → use flash attention
        if not return_attn and attention_mode == "standard":
            return self.o(scaled_dot_product_attention(
                q, k, v, num_heads=self.num_heads, attn_mask=context_mask,
            ))

        # Manual multi-head path (when we need attn or custom attention mode)
        B = q.shape[0]
        K_tokens = q.shape[1]
        N_tokens = k.shape[1]

        q = q.reshape(B, K_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, K, hd]
        k = k.reshape(B, N_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, N, hd]
        v = v.reshape(B, N_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, N, hd]

        scale = self.head_dim ** -0.5
        logits = q @ k.transpose(-2, -1) * scale  # [B, nh, K, N]

        if context_mask is not None:
            # context_mask is expected as additive mask [B, 1, K, N] or broadcastable
            logits = logits + context_mask

        if attention_mode == "standard":
            attn = logits.softmax(dim=-1)  # [B, nh, K, N]
        elif attention_mode == "competitive":
            # Slot-attention style: each patch chooses its query first (softmax
            # over queries), then each query normalises over its patches.
            attn = (logits / competitive_temperature).softmax(dim=-2)  # over queries
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(float(competitive_eps))
        else:
            raise ValueError(
                f"Unsupported attention_mode={attention_mode!r}. "
                "Expected 'standard' or 'competitive'."
            )

        out = attn @ v                                     # [B, nh, K, hd]
        out = out.transpose(1, 2).reshape(B, K_tokens, self.inner_dim)
        out = self.o(out)

        return (out, attn) if return_attn else out


class PerceiverResampler(nn.Module):
    """Compress arbitrary-length visual tokens -> k fixed summary tokens via cross-attention.

    Learnable queries [k, dim] attend to input visual tokens [N, dim_input].
    K/V are projected directly from dim_input inside each CrossAttention layer,
    so no separate kv_proj is needed and the output is already in hidden_dim.
    """

    def __init__(
        self,
        dim_input: int,
        dim: int,
        num_queries: int,
        num_heads: int,
        head_dim: int,
        depth: int = 1,
        mlp_ratio: float = 4.0,
        eps: float = 1e-6,
        dropout: float = 0.0,
        learnable_queries: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        if learnable_queries:
            self.queries = nn.Parameter(torch.zeros(1, num_queries, dim))
            nn.init.normal_(self.queries, std=0.02)
        else:
            self.register_parameter("queries", None)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim, eps=eps),
                CrossAttention(dim, dim_input, num_heads, head_dim, eps=eps),
                nn.LayerNorm(dim, eps=eps),
                FeedForward(dim, int(dim * mlp_ratio), dropout=dropout),
            ]))

    def forward(
        self,
        x: torch.Tensor,
        query_tokens: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
        attention_mode: str = "standard",
        competitive_temperature: float = 1.0,
        competitive_eps: float = 1e-6,
    ):
        # x: [B, N, dim_input]  ->  [B, num_queries, dim]
        if query_tokens is None:
            if self.queries is None:
                raise RuntimeError("PerceiverResampler was constructed without learnable queries; query_tokens are required.")
            q = self.queries.expand(x.shape[0], -1, -1)
        else:
            q = query_tokens
        last_attn = None

        for layer_idx, (norm_q, cross_attn, norm_ffn, ffn) in enumerate(self.layers):
            is_last = layer_idx == len(self.layers) - 1
            need_attn = return_attn and is_last

            ca_out = cross_attn(
                norm_q(q), x,
                context_mask=context_mask,
                return_attn=need_attn,
                attention_mode=attention_mode,
                competitive_temperature=competitive_temperature,
                competitive_eps=competitive_eps,
            )
            if need_attn:
                delta, last_attn = ca_out
            else:
                delta = ca_out

            q = q + delta
            q = q + ffn(norm_ffn(q))

        return (q, last_attn) if return_attn else q


class ExpertBlock(nn.Module):
    """A DiT-style expert block whose self-attention can be mixed across experts."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.self_attn = SelfAttention(hidden_dim, num_heads, head_dim, eps=eps)
        self.cross_attn = CrossAttention(hidden_dim, context_dim, num_heads, head_dim, eps=eps)
        self.ffn = FeedForward(hidden_dim, ffn_dim, dropout=dropout)
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / math.sqrt(hidden_dim))

    def _split_modulation(self, time_mod: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # time_mod can be shared per sample [B, 6*D] or per token [B, S, 6*D].
        if time_mod.ndim == 3:
            base = self.modulation.to(device=time_mod.device, dtype=time_mod.dtype).unsqueeze(1)
            mod = base + time_mod.view(time_mod.shape[0], time_mod.shape[1], 6, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=2)
            squeeze_dim = 2
        else:
            base = self.modulation.to(device=time_mod.device, dtype=time_mod.dtype)
            mod = base + time_mod.view(time_mod.shape[0], 6, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
            squeeze_dim = 1
        return (
            shift_msa.squeeze(squeeze_dim),
            scale_msa.squeeze(squeeze_dim),
            gate_msa.squeeze(squeeze_dim),
            shift_mlp.squeeze(squeeze_dim),
            scale_mlp.squeeze(squeeze_dim),
            gate_mlp.squeeze(squeeze_dim),
        )

    def build_attention_io(
        self,
        x: torch.Tensor,
        time_mod: torch.Tensor,
        freqs: Optional[torch.Tensor] = None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(time_mod)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.self_attn.project_qkv(attn_in, freqs=freqs)
        return q, k, v, x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def apply_post(
        self,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        while gate_msa.ndim < residual_x.ndim:
            gate_msa = gate_msa.unsqueeze(1)
        x = residual_x + gate_msa * self.self_attn.o(mixed_attn_out)
        if context is not None:
            x = x + self.cross_attn(self.norm3(x), context, context_mask)
        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        while gate_mlp.ndim < x.ndim:
            gate_mlp = gate_mlp.unsqueeze(1)
        x = x + gate_mlp * self.ffn(mlp_in)
        return x

    def forward(
        self,
        *,
        mode: str,
        x: Optional[torch.Tensor] = None,
        time_mod: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None,
        residual_x: Optional[torch.Tensor] = None,
        mixed_attn_out: Optional[torch.Tensor] = None,
        gate_msa: Optional[torch.Tensor] = None,
        shift_mlp: Optional[torch.Tensor] = None,
        scale_mlp: Optional[torch.Tensor] = None,
        gate_mlp: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ):
        if mode == "build_attention_io":
            if x is None or time_mod is None:
                raise ValueError("`x` and `time_mod` are required for mode='build_attention_io'")
            return self.build_attention_io(x=x, time_mod=time_mod, freqs=freqs)
        if mode == "apply_post":
            required = {
                "residual_x": residual_x,
                "mixed_attn_out": mixed_attn_out,
                "gate_msa": gate_msa,
                "shift_mlp": shift_mlp,
                "scale_mlp": scale_mlp,
                "gate_mlp": gate_mlp,
            }
            missing = [k for k, v in required.items() if v is None]
            if missing:
                raise ValueError(
                    f"Missing required tensors for mode='apply_post': {missing}"
                )
            return self.apply_post(
                residual_x=residual_x,
                mixed_attn_out=mixed_attn_out,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context=context,
                context_mask=context_mask,
            )
        raise ValueError(f"Unsupported ExpertBlock forward mode: {mode}")


class ExpertStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        ffn_dim: int,
        context_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ExpertBlock(
                    hidden_dim=hidden_dim,
                    ffn_dim=ffn_dim,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    eps=eps,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )


class FastWAMStyleMoT(nn.Module):
    """Shared-attention MoT.

    Each expert has its own norms / FFN / cross-attn, while self-attention is
    computed jointly on concatenated expert tokens using a structured mask.
    """

    def __init__(
        self,
        expert_stacks: Dict[str, ExpertStack],
        num_heads: int,
        head_dim: int,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if not expert_stacks or set(expert_stacks.keys()) != {"video", "action"}:
            raise ValueError("expert_stacks must contain exactly {'video', 'action'}")
        self.experts = nn.ModuleDict(expert_stacks)
        self.expert_order = ["video", "action"]
        self.num_layers = len(self.experts["video"].blocks)
        if len(self.experts["action"].blocks) != self.num_layers:
            raise ValueError("video and action experts must have the same number of layers")
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def forward(
        self,
        expert_tokens: Dict[str, torch.Tensor],
        expert_time_mod: Dict[str, torch.Tensor],
        self_attn_mask: Optional[torch.Tensor],
        expert_freqs: Optional[Dict[str, torch.Tensor]] = None,
        expert_context: Optional[Dict[str, torch.Tensor]] = None,
        expert_context_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        x = {k: v for k, v in expert_tokens.items()}
        for layer_idx in range(self.num_layers):
            packed = {}
            sizes = []
            for name in self.expert_order:
                block = self.experts[name].blocks[layer_idx]
                freqs = expert_freqs[name] if expert_freqs is not None else None
                q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    mode="build_attention_io",
                    x=x[name],
                    time_mod=expert_time_mod[name],
                    freqs=freqs,
                )
                packed[name] = {
                    "block": block,
                    "q": q,
                    "k": k,
                    "v": v,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                }
                sizes.append(q.shape[1])
            video_len, action_len = sizes
            use_factorized_fastwam = (
                self_attn_mask is not None
                and self.expert_order == ["video", "action"]
            )

            if use_factorized_fastwam:
                video_mask = self_attn_mask[:, :, :video_len, :video_len]
                action_mask = self_attn_mask[:, :, video_len:, : video_len + action_len]

                mixed_video = scaled_dot_product_attention(
                    packed["video"]["q"],
                    packed["video"]["k"],
                    packed["video"]["v"],
                    num_heads=self.num_heads,
                    attn_mask=video_mask,
                )
                k_action_cat = torch.cat([packed["video"]["k"], packed["action"]["k"]], dim=1)
                v_action_cat = torch.cat([packed["video"]["v"], packed["action"]["v"]], dim=1)
                mixed_action = scaled_dot_product_attention(
                    packed["action"]["q"],
                    k_action_cat,
                    v_action_cat,
                    num_heads=self.num_heads,
                    attn_mask=action_mask,
                )
                mixed_by_name = {
                    "video": mixed_video,
                    "action": mixed_action,
                }
            else:
                q_cat = torch.cat([packed[name]["q"] for name in self.expert_order], dim=1)
                k_cat = torch.cat([packed[name]["k"] for name in self.expert_order], dim=1)
                v_cat = torch.cat([packed[name]["v"] for name in self.expert_order], dim=1)
                mixed = scaled_dot_product_attention(
                    q_cat,
                    k_cat,
                    v_cat,
                    num_heads=self.num_heads,
                    attn_mask=self_attn_mask,
                )
                mixed_by_name = {}
                cursor = 0
                for name, size in zip(self.expert_order, sizes):
                    mixed_by_name[name] = mixed[:, cursor : cursor + size]
                    cursor += size

            for name in self.expert_order:
                block = packed[name]["block"]
                ctx = expert_context[name] if expert_context is not None else None
                ctx_mask = expert_context_mask[name] if expert_context_mask is not None else None
                x[name] = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    mode="apply_post",
                    residual_x=packed[name]["residual_x"],
                    mixed_attn_out=mixed_by_name[name],
                    gate_msa=packed[name]["gate_msa"],
                    shift_mlp=packed[name]["shift_mlp"],
                    scale_mlp=packed[name]["scale_mlp"],
                    gate_mlp=packed[name]["gate_mlp"],
                    context=ctx,
                    context_mask=ctx_mask,
                )
        return x

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_time_mod: torch.Tensor,
        video_freqs: Optional[torch.Tensor] = None,
        video_context: Optional[torch.Tensor] = None,
        video_context_mask: Optional[torch.Tensor] = None,
        video_attention_mask: Optional[torch.Tensor] = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Run the video expert once and cache per-layer K/V for action-only inference.

        This mirrors FastWAM's action inference path where the visual condition is
        encoded once with zero video timestep, then reused for all action denoising
        steps instead of being re-forwarded with the changing action timestep.
        """
        x_video = video_tokens
        video_len = int(video_tokens.shape[1])
        if video_attention_mask is None:
            video_mask = torch.ones((video_len, video_len), dtype=torch.bool, device=video_tokens.device)
            video_mask = video_mask.unsqueeze(0).unsqueeze(0).expand(video_tokens.shape[0], 1, video_len, video_len)
        else:
            video_mask = video_attention_mask
        cache: list[dict[str, torch.Tensor]] = []

        for layer_idx in range(self.num_layers):
            block = self.experts["video"].blocks[layer_idx]
            q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = block(
                mode="build_attention_io",
                x=x_video,
                time_mod=video_time_mod,
                freqs=video_freqs,
            )
            mixed = scaled_dot_product_attention(
                q,
                k,
                v,
                num_heads=self.num_heads,
                attn_mask=video_mask,
            )
            cache.append({"k": k, "v": v})
            x_video = block(
                mode="apply_post",
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context=video_context,
                context_mask=video_context_mask,
            )

        return cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_time_mod: torch.Tensor,
        video_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_freqs: Optional[torch.Tensor] = None,
        action_context: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise action tokens while attending to a fixed, cached video branch."""
        if len(video_cache) != self.num_layers:
            raise ValueError(
                f"video_cache length mismatch: expected {self.num_layers}, got {len(video_cache)}"
            )

        x_action = action_tokens
        action_mask = attention_mask[:, :, video_seq_len:, :]

        for layer_idx in range(self.num_layers):
            block = self.experts["action"].blocks[layer_idx]
            q_a, k_a, v_a, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = block(
                mode="build_attention_io",
                x=x_action,
                time_mod=action_time_mod,
                freqs=action_freqs,
            )
            k_cat = torch.cat([video_cache[layer_idx]["k"], k_a], dim=1)
            v_cat = torch.cat([video_cache[layer_idx]["v"], v_a], dim=1)
            mixed = scaled_dot_product_attention(
                q_a,
                k_cat,
                v_cat,
                num_heads=self.num_heads,
                attn_mask=action_mask,
            )
            x_action = block(
                mode="apply_post",
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context=action_context,
                context_mask=action_context_mask,
            )

        return x_action

    def prefill_video_cache_flat(
        self,
        video_tokens: torch.Tensor,
        video_time_mod: torch.Tensor,
        video_freqs: Optional[torch.Tensor] = None,
        video_context: Optional[torch.Tensor] = None,
        video_context_mask: Optional[torch.Tensor] = None,
        video_attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Prefill with a fixed tensor-tuple cache for compiled serving.

        It is numerically identical to ``prefill_video_cache``.  The only
        difference is representation: two immutable length-``num_layers``
        tuples replace a Python list of ``{\"k\", \"v\"}`` dictionaries.
        """
        x_video = video_tokens
        video_len = int(video_tokens.shape[1])
        if video_attention_mask is None:
            video_mask = torch.ones((video_len, video_len), dtype=torch.bool, device=video_tokens.device)
            video_mask = video_mask.unsqueeze(0).unsqueeze(0).expand(video_tokens.shape[0], 1, video_len, video_len)
        else:
            video_mask = video_attention_mask
        cache_k: list[torch.Tensor] = []
        cache_v: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            block = self.experts["video"].blocks[layer_idx]
            q, k, v, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = block(
                mode="build_attention_io", x=x_video, time_mod=video_time_mod, freqs=video_freqs,
            )
            mixed = scaled_dot_product_attention(q, k, v, num_heads=self.num_heads, attn_mask=video_mask)
            cache_k.append(k)
            cache_v.append(v)
            x_video = block(
                mode="apply_post", residual_x=residual_x, mixed_attn_out=mixed,
                gate_msa=gate_msa, shift_mlp=shift_mlp, scale_mlp=scale_mlp, gate_mlp=gate_mlp,
                context=video_context, context_mask=video_context_mask,
            )
        return tuple(cache_k), tuple(cache_v)

    def forward_action_with_video_cache_flat(
        self,
        action_tokens: torch.Tensor,
        action_time_mod: torch.Tensor,
        video_cache_k: tuple[torch.Tensor, ...],
        video_cache_v: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        action_freqs: Optional[torch.Tensor] = None,
        action_context: Optional[torch.Tensor] = None,
        action_context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise with the fixed tensor-tuple cache used by CUDA Graphs."""
        x_action = action_tokens
        action_mask = attention_mask[:, :, video_seq_len:, :]
        for layer_idx in range(self.num_layers):
            block = self.experts["action"].blocks[layer_idx]
            q_a, k_a, v_a, residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = block(
                mode="build_attention_io", x=x_action, time_mod=action_time_mod, freqs=action_freqs,
            )
            k_cat = torch.cat([video_cache_k[layer_idx], k_a], dim=1)
            v_cat = torch.cat([video_cache_v[layer_idx], v_a], dim=1)
            mixed = scaled_dot_product_attention(q_a, k_cat, v_cat, num_heads=self.num_heads, attn_mask=action_mask)
            x_action = block(
                mode="apply_post", residual_x=residual_x, mixed_attn_out=mixed,
                gate_msa=gate_msa, shift_mlp=shift_mlp, scale_mlp=scale_mlp, gate_mlp=gate_mlp,
                context=action_context, context_mask=action_context_mask,
            )
        return x_action


# ------------------------------------------------------------
# Structured masks
# ------------------------------------------------------------

def build_fastwam_style_joint_mask(
    batch_size: int,
    anchor_len: int,
    future_video_len: int,
    action_len: int,
    device: torch.device,
    *,
    abstract_anchor_len: Optional[int] = None,
    current_spatial_len: int = 0,
    action_prefix_len: int = 0,
    action_attends_future_video: bool = False,
    future_video_attends_current_spatial: bool = False,
) -> torch.Tensor:
    """Build a FastWAM-style mask.

    Query rows attend to key/value columns.

    Default behavior matches the original implementation when `abstract_anchor_len`
    is omitted. When `abstract_anchor_len` is provided, the current-frame anchor is
    split into `[abstract anchor][current spatial branch]` with stricter routing:

    - abstract anchor queries attend to abstract anchor + current spatial tokens
    - current spatial queries attend to abstract anchor + current spatial tokens
    - future video queries attend to abstract anchor + future video, and optionally
      current spatial tokens when ``future_video_attends_current_spatial`` is True
    - action prefix queries attend to abstract anchor + current spatial tokens + action prefix
      (plus future video if ``action_attends_future_video`` is True)
    - action body queries attend to abstract anchor + current spatial tokens + all action tokens
      (plus future video if ``action_attends_future_video`` is True)
    """
    anchor_len = int(anchor_len)
    future_video_len = int(future_video_len)
    action_len = int(action_len)
    current_spatial_len = int(current_spatial_len)
    action_prefix_len = int(action_prefix_len)
    if abstract_anchor_len is None:
        abstract_anchor_len = anchor_len
        current_spatial_len = 0
    abstract_anchor_len = int(abstract_anchor_len)
    if abstract_anchor_len < 0 or current_spatial_len < 0 or action_prefix_len < 0:
        raise ValueError("abstract_anchor_len, current_spatial_len, and action_prefix_len must be non-negative")
    if abstract_anchor_len + current_spatial_len != anchor_len:
        raise ValueError(
            f"anchor split mismatch: abstract={abstract_anchor_len}, current_spatial={current_spatial_len}, anchor_len={anchor_len}"
        )

    if action_prefix_len > action_len:
        raise ValueError(f"action_prefix_len={action_prefix_len} exceeds action_len={action_len}")

    total_video = anchor_len + future_video_len
    total = total_video + action_len

    mask = torch.zeros(total, total, dtype=torch.bool, device=device)

    abstract_start = 0
    abstract_end = abstract_anchor_len
    sampled_start = abstract_end
    sampled_end = sampled_start + current_spatial_len
    future_start = anchor_len
    future_end = future_start + future_video_len
    action_start = total_video
    action_prefix_end = action_start + action_prefix_len
    action_end = total

    if abstract_anchor_len > 0:
        mask[abstract_start:abstract_end, abstract_start:abstract_end] = True
        if current_spatial_len > 0:
            mask[abstract_start:abstract_end, sampled_start:sampled_end] = True

    if current_spatial_len > 0:
        mask[sampled_start:sampled_end, sampled_start:sampled_end] = True
        if abstract_anchor_len > 0:
            mask[sampled_start:sampled_end, abstract_start:abstract_end] = True

    if future_video_len > 0:
        if abstract_anchor_len > 0:
            mask[future_start:future_end, abstract_start:abstract_end] = True
        if future_video_attends_current_spatial and current_spatial_len > 0:
            mask[future_start:future_end, sampled_start:sampled_end] = True
        mask[future_start:future_end, future_start:future_end] = True

    if action_len > 0:
        action_body_start = action_prefix_end
        if action_prefix_len > 0:
            if abstract_anchor_len > 0:
                mask[action_start:action_prefix_end, abstract_start:abstract_end] = True
            if current_spatial_len > 0:
                mask[action_start:action_prefix_end, sampled_start:sampled_end] = True
            mask[action_start:action_prefix_end, action_start:action_prefix_end] = True
            if action_attends_future_video and future_video_len > 0:
                mask[action_start:action_prefix_end, future_start:future_end] = True
        if action_body_start < action_end:
            if abstract_anchor_len > 0:
                mask[action_body_start:action_end, abstract_start:abstract_end] = True
            if current_spatial_len > 0:
                mask[action_body_start:action_end, sampled_start:sampled_end] = True
            mask[action_body_start:action_end, action_start:action_end] = True
            if action_attends_future_video and future_video_len > 0:
                mask[action_body_start:action_end, future_start:future_end] = True
        if action_prefix_len == 0:
            mask[action_start:action_end, action_start:action_end] = True

    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, total, total)
