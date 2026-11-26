"""Scale-adaptive mixed-precision (BF16 / FP8 E4M3) layers and wrappers for NCM."""

from __future__ import annotations

import functools
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FP8_E4M3 = torch.float8_e4m3fn
FP8_E5M2 = torch.float8_e5m2
FP8_E4M3_MAX = float(torch.finfo(FP8_E4M3).max)
FP8_E5M2_MAX = float(torch.finfo(FP8_E5M2).max)


class PrecisionContext:
    """Tracks the active NCM concept scale (level_idx) and precision policy."""

    def __init__(
        self,
        mode: str = "pure_bf16",
        num_levels: int = 12,
        split_level: Optional[int] = None,
        fp8_prefix: Optional[bool] = None,
    ) -> None:
        valid = {
            "fp32",
            "bf16",
            "official_bf16",
            "pure_bf16",
            "fp8_all",
            "coarse_bf16_fine_fp8",
            "coarse_bf16_fine_fp8_w8",
            "coarse_fp8_fine_bf16",
        }
        if mode not in valid:
            raise ValueError(f"Unknown precision mode: {mode}; expected one of {sorted(valid)}")
        self.mode = mode
        self.num_levels = int(num_levels)
        self.split_level = int(split_level if split_level is not None else num_levels // 2)
        self.fp8_prefix = (mode == "fp8_all") if fp8_prefix is None else bool(fp8_prefix)
        self.level_idx: int = -1  # -1 = prefix, 0..num_levels-1 = target level
        self.calibrating: bool = False

    def is_fp8_level(self, level_idx: Optional[int] = None) -> bool:
        lvl = self.level_idx if level_idx is None else level_idx
        if self.mode in ("fp32", "bf16", "official_bf16", "pure_bf16"):
            return False
        if lvl < 0:
            return self.fp8_prefix
        if self.mode == "fp8_all":
            return True
        if self.mode in ("coarse_bf16_fine_fp8", "coarse_bf16_fine_fp8_w8"):
            return lvl >= self.split_level
        if self.mode == "coarse_fp8_fine_bf16":
            return lvl < self.split_level
        return False


def _quantize_e4m3_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.abs().amax().float().clamp(min=1e-12)) / FP8_E4M3_MAX
    q = (x.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_E4M3)
    return q, scale


def _quantize_e5m2_tensorwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (x.abs().amax().float().clamp(min=1e-12)) / FP8_E5M2_MAX
    q = (x.float() / scale).clamp(-FP8_E5M2_MAX, FP8_E5M2_MAX).to(FP8_E5M2)
    return q, scale


class ScaleAwareFP8Linear(nn.Module):
    """Inference linear layer supporting BF16, single-copy FP8, and scale-adaptive switching."""

    def __init__(
        self,
        src: nn.Linear,
        ctx: PrecisionContext,
        weight_storage: str = "dual",
        act_scaling: str = "dynamic",
        fixed_level: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_features = src.in_features
        self.out_features = src.out_features
        self.ctx = ctx
        self.weight_storage = weight_storage  # "dual", "fp8_only", "bf16_only"
        self.act_scaling = act_scaling        # "dynamic" or "static"
        self.fixed_level = fixed_level        # For per-level heads (no cross-level sharing)
        self.fp8_supported = (self.in_features % 16 == 0) and (self.out_features % 16 == 0)

        w_bf16 = src.weight.detach().to(torch.bfloat16).contiguous()
        if src.bias is not None:
            self.register_buffer("bias", src.bias.detach().to(torch.bfloat16).contiguous())
        else:
            self.bias = None

        if not self.fp8_supported or weight_storage == "bf16_only":
            self.register_buffer("weight_bf16", w_bf16)
            self.w_fp8 = None
            self.w_scale = None
        elif weight_storage == "fp8_only":
            w_fp8, w_scale = _quantize_e4m3_tensorwise(w_bf16)
            self.register_buffer("w_fp8", w_fp8.contiguous())  # (N, K) row-major -> .t() is col-major
            self.register_buffer("w_scale", w_scale)
            self.weight_bf16 = None
        elif weight_storage == "dual":
            self.register_buffer("weight_bf16", w_bf16)
            w_fp8, w_scale = _quantize_e4m3_tensorwise(w_bf16)
            self.register_buffer("w_fp8", w_fp8.contiguous())
            self.register_buffer("w_scale", w_scale)
        else:
            raise ValueError(f"Unknown weight_storage: {weight_storage}")

        # Per-level static activation scales indexed by (level_idx + 1), supporting -1..num_levels.
        dev = w_bf16.device
        n_slots = ctx.num_levels + 2
        self.register_buffer("static_amax", torch.zeros(n_slots, dtype=torch.float32, device=dev))
        self.register_buffer("static_scale", torch.ones(n_slots, dtype=torch.float32, device=dev))
        self.register_buffer("static_inv_scale", torch.ones(n_slots, dtype=torch.bfloat16, device=dev))
        self._has_static = [False] * n_slots

    def finalize_calibration(self) -> None:
        for slot in range(len(self._has_static)):
            amax = self.static_amax[slot].clamp(min=1e-12)
            if self._has_static[slot]:
                s = amax / FP8_E4M3_MAX
                self.static_scale[slot].copy_(s)
                self.static_inv_scale[slot].copy_((1.0 / s).to(torch.bfloat16))

    def _slot(self) -> int:
        lvl = self.ctx.level_idx if self.fixed_level is None else self.fixed_level
        return max(0, min(lvl + 1, len(self._has_static) - 1))

    def _run_bf16(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_bf16 is not None:
            w = self.weight_bf16
        else:
            w = self.w_fp8.to(torch.bfloat16) * self.w_scale.to(torch.bfloat16)
        x_in = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        return F.linear(x_in, w, self.bias)

    def _run_fp8(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        if x_2d.dtype != torch.bfloat16:
            x_2d = x_2d.to(torch.bfloat16)

        if self.act_scaling == "static" and not self.ctx.calibrating and self._has_static[self._slot()]:
            slot = self._slot()
            sx = self.static_scale[slot]
            inv_sx = self.static_inv_scale[slot]
            x_q = (x_2d * inv_sx).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_E4M3)
        else:
            x_q, sx = _quantize_e4m3_tensorwise(x_2d)

        out = torch._scaled_mm(
            x_q,
            self.w_fp8.t(),
            scale_a=sx,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
        )
        return out.reshape(*orig_shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ctx.calibrating:
            slot = self._slot()
            cur_amax = x.detach().abs().amax().float()
            self.static_amax[slot] = torch.maximum(self.static_amax[slot], cur_amax)
            self._has_static[slot] = True

        lvl = self.ctx.level_idx if self.fixed_level is None else self.fixed_level
        if self.fp8_supported and self.w_fp8 is not None and self.ctx.is_fp8_level(lvl):
            return self._run_fp8(x)
        return self._run_bf16(x)


class FP8Embedding(nn.Module):
    """FP8 E4M3 quantized codebook embedding (1 byte/param + 1 float32 scale)."""

    def __init__(self, src: nn.Embedding) -> None:
        super().__init__()
        self.num_embeddings = src.num_embeddings
        self.embedding_dim = src.embedding_dim
        w_q, w_scale = _quantize_e4m3_tensorwise(src.weight.detach())
        self.register_buffer("w_fp8", w_q.contiguous())
        self.register_buffer("w_scale", w_scale)

    @property
    def weight(self) -> torch.Tensor:
        return self.w_fp8.to(torch.bfloat16) * self.w_scale.to(torch.bfloat16)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices, self.w_fp8).to(torch.bfloat16) * self.w_scale.to(torch.bfloat16)


class FP8TiedCodebookHead(nn.Module):
    """FP8-aware replacement for models.next_concept._TiedCodebookHead."""

    def __init__(
        self,
        src,
        ctx: PrecisionContext,
        weight_storage: str = "dual",
        act_scaling: str = "dynamic",
        fixed_level: Optional[int] = None,
        quantize_codebook: bool = True,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.fixed_level = fixed_level
        self.proj = ScaleAwareFP8Linear(
            src.proj,
            ctx,
            weight_storage=weight_storage,
            act_scaling=act_scaling,
            fixed_level=fixed_level,
        )
        self.out_features = src.out_features
        self.d_scale = src.d_scale.detach().to(torch.bfloat16)
        self.log_inv_T = src.log_inv_T.detach().to(torch.bfloat16) if torch.is_tensor(src.log_inv_T) else src.log_inv_T
        self.bias = src.bias.detach().to(torch.bfloat16) if src.bias is not None else None

        E_bf16 = src.E.detach().to(torch.bfloat16).contiguous()
        E_sq = src.E_sq.detach().to(torch.bfloat16).contiguous()
        self.register_buffer("E_sq", E_sq)

        is_fp8 = ctx.is_fp8_level(fixed_level)
        if quantize_codebook and is_fp8 and E_bf16.shape[0] % 16 == 0 and E_bf16.shape[1] % 16 == 0:
            E_fp8, E_scale = _quantize_e4m3_tensorwise(E_bf16)
            self.register_buffer("E_fp8", E_fp8.contiguous())
            self.register_buffer("E_scale", E_scale)
            self.E_bf16 = None
            n_slots = ctx.num_levels + 2
            dev = E_bf16.device
            self.register_buffer("static_amax", torch.zeros(n_slots, dtype=torch.float32, device=dev))
            self.register_buffer("static_scale", torch.ones(n_slots, dtype=torch.float32, device=dev))
            self.register_buffer("static_inv_scale", torch.ones(n_slots, dtype=torch.bfloat16, device=dev))
            self._has_static = [False] * n_slots
            self.act_scaling = act_scaling
        else:
            self.register_buffer("E_bf16", E_bf16)
            self.E_fp8 = None
            self.E_scale = None
            self._has_static = []
            self.act_scaling = act_scaling

    def finalize_calibration(self) -> None:
        self.proj.finalize_calibration()
        for slot in range(len(self._has_static)):
            amax = self.static_amax[slot].clamp(min=1e-12)
            if self._has_static[slot]:
                s = amax / FP8_E4M3_MAX
                self.static_scale[slot].copy_(s)
                self.static_inv_scale[slot].copy_((1.0 / s).to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)
        if self.E_fp8 is not None:
            orig_shape = z.shape
            z_2d = z.reshape(-1, orig_shape[-1])
            slot = max(0, min((self.fixed_level if self.fixed_level is not None else self.ctx.level_idx) + 1, len(self._has_static) - 1))
            if self.ctx.calibrating:
                cur_amax = z_2d.detach().abs().amax().float()
                self.static_amax[slot] = torch.maximum(self.static_amax[slot], cur_amax)
                self._has_static[slot] = True
            if self.act_scaling == "static" and not self.ctx.calibrating and self._has_static[slot]:
                sz = self.static_scale[slot]
                inv_sz = self.static_inv_scale[slot]
                z_q = (z_2d * inv_sz).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_E4M3)
            else:
                z_q, sz = _quantize_e4m3_tensorwise(z_2d)
            dot = torch._scaled_mm(
                z_q,
                self.E_fp8.t(),
                scale_a=sz,
                scale_b=self.E_scale,
                out_dtype=torch.bfloat16,
            ).reshape(*orig_shape[:-1], self.out_features)
        else:
            dot = z.to(torch.bfloat16) @ self.E_bf16.t()
        neg_d2 = (2.0 * dot - self.E_sq) / self.d_scale
        out = neg_d2 * torch.exp(self.log_inv_T)
        return out if self.bias is None else out + self.bias


def _replace_backbone_linears(
    module: nn.Module,
    ctx: PrecisionContext,
    weight_storage: str,
    act_scaling: str,
    skip_names: tuple[str, ...] = ("lm_heads", "lm_head_l0"),
) -> None:
    for name, child in list(module.named_children()):
        if name in skip_names:
            continue
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                ScaleAwareFP8Linear(child, ctx, weight_storage=weight_storage, act_scaling=act_scaling),
            )
        else:
            _replace_backbone_linears(child, ctx, weight_storage, act_scaling, skip_names=skip_names)


def apply_inference_precision(
    net: nn.Module,
    tk,
    mode: str,
    split_level: Optional[int] = None,
    act_scaling: str = "dynamic",
) -> Optional[PrecisionContext]:
    """Convert net & vqvae to the requested precision regime without altering model files."""
    if mode == "official_bf16":
        return None

    levels = net.levels
    ctx = PrecisionContext(mode=mode, num_levels=len(levels), split_level=split_level)
    net._precision_ctx = ctx

    # Convert whole model to BF16 first (eliminates FP32 weight copies).
    net.to(torch.bfloat16)
    tk.vqvae.to(torch.bfloat16)

    if mode == "pure_bf16":
        return ctx

    backbone_storage = "fp8_only" if mode in ("fp8_all", "coarse_bf16_fine_fp8_w8") else "dual"
    _replace_backbone_linears(net.transformer, ctx, weight_storage=backbone_storage, act_scaling=act_scaling)

    # Per-level heads: store ONLY the needed precision per level (no duplication!).
    lvl_first = net._lvl_first
    for i in range(len(net.lm_heads)):
        abs_lvl = i + lvl_first
        head = net.lm_heads[i]
        head_storage = "fp8_only" if ctx.is_fp8_level(abs_lvl) else "bf16_only"
        if isinstance(head, nn.Linear):
            net.lm_heads[i] = ScaleAwareFP8Linear(
                head, ctx, weight_storage=head_storage, act_scaling=act_scaling, fixed_level=abs_lvl
            )
        elif hasattr(head, "proj") and hasattr(head, "E"):
            net.lm_heads[i] = FP8TiedCodebookHead(
                head,
                ctx,
                weight_storage=head_storage,
                act_scaling=act_scaling,
                fixed_level=abs_lvl,
                quantize_codebook=ctx.is_fp8_level(abs_lvl),
            )

    if getattr(net, "predict_level0", False) and hasattr(net, "lm_head_l0"):
        l0_storage = "fp8_only" if ctx.is_fp8_level(0) else "bf16_only"
        head0 = net.lm_head_l0
        if isinstance(head0, nn.Linear):
            net.lm_head_l0 = ScaleAwareFP8Linear(
                head0, ctx, weight_storage=l0_storage, act_scaling=act_scaling, fixed_level=0
            )
        elif hasattr(head0, "proj") and hasattr(head0, "E"):
            net.lm_head_l0 = FP8TiedCodebookHead(
                head0,
                ctx,
                weight_storage=l0_storage,
                act_scaling=act_scaling,
                fixed_level=0,
                quantize_codebook=ctx.is_fp8_level(0),
            )

    # Per-level VQ codebook embeddings: quantize FP8 levels to 1 byte/param.
    vq = tk.vqvae.vq
    for lvl_i in range(len(vq.embeddings)):
        if ctx.is_fp8_level(lvl_i):
            vq.embeddings[lvl_i] = FP8Embedding(vq.embeddings[lvl_i])

    # Wrap level-tracking hooks around generate / _kv_prefix / _kv_level / get_next_autoregressive_input.
    orig_generate = net.generate
    orig_kv_prefix = net._kv_prefix
    orig_kv_level = net._kv_level
    orig_decode_level = net._decode_level_logits
    orig_next_input = vq.get_next_autoregressive_input

    @functools.wraps(orig_kv_prefix)
    def wrapped_kv_prefix(h, mask, rope):
        ctx.level_idx = 0 if net.query_level0 else -1
        return orig_kv_prefix(h, mask, rope)

    @functools.wraps(orig_kv_level)
    def wrapped_kv_level(x, prefix_kv, mask, rope, hidx):
        ctx.level_idx = int(hidx) + (0 if net.query_level0 else 1)
        return orig_kv_level(x, prefix_kv, mask, rope, hidx)

    @functools.wraps(orig_decode_level)
    def wrapped_decode_level(x, prefix_emb, level_idx, pad_bias=None, hidx=None):
        eff_hidx = level_idx if hidx is None else hidx
        ctx.level_idx = int(eff_hidx) + (0 if net.query_level0 else 1)
        return orig_decode_level(x, prefix_emb, level_idx, pad_bias=pad_bias, hidx=hidx)

    @functools.wraps(orig_next_input)
    def wrapped_next_input(level_idx, z_hat, z_q_level):
        # Update ctx before transformer.wte(x) runs for level_idx + 1.
        ctx.level_idx = min(int(level_idx) + 1, len(levels) - 1)
        return orig_next_input(level_idx, z_hat, z_q_level)

    @functools.wraps(orig_generate)
    def wrapped_generate(*args, **kwargs):
        ctx.level_idx = -1
        try:
            return orig_generate(*args, **kwargs)
        finally:
            ctx.level_idx = -1

    net._kv_prefix = wrapped_kv_prefix
    net._kv_level = wrapped_kv_level
    net._decode_level_logits = wrapped_decode_level
    vq.get_next_autoregressive_input = wrapped_next_input
    net.generate = wrapped_generate

    return ctx


def finalize_calibration(net: nn.Module) -> None:
    for m in net.modules():
        if isinstance(m, (ScaleAwareFP8Linear, FP8TiedCodebookHead)):
            m.finalize_calibration()


# ---------------------------------------------------------------------------
# Training-time FP8 Autograd Op & Scale-Adaptive Split-Precision Forward
# ---------------------------------------------------------------------------


class _FP8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        orig_shape = x.shape
        x_2d = x.reshape(-1, weight.shape[1]).to(torch.bfloat16).contiguous()
        w_bf16 = weight.to(torch.bfloat16).contiguous()

        x_q, sx = _quantize_e4m3_tensorwise(x_2d)
        w_q, sw = _quantize_e4m3_tensorwise(w_bf16)

        out = torch._scaled_mm(
            x_q,
            w_q.t(),
            scale_a=sx,
            scale_b=sw,
            bias=bias.to(torch.bfloat16) if bias is not None else None,
            out_dtype=torch.bfloat16,
        )
        # Store quantized FP8 tensors to save activation memory!
        ctx.save_for_backward(x_q, sx, w_q, sw)
        ctx.has_bias = bias is not None
        ctx.orig_dtype = x.dtype
        ctx.weight_dtype = weight.dtype
        return out.reshape(*orig_shape[:-1], weight.shape[0]).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x_q, sx, w_q, sw = ctx.saved_tensors
        N = w_q.shape[0]
        K = w_q.shape[1]
        grad_2d = grad_out.reshape(-1, N).to(torch.bfloat16).contiguous()

        gy_q_e5, sgy = _quantize_e5m2_tensorwise(grad_2d)

        # grad_x = grad_y @ w : (M, N) @ (N, K) -> w_col = w_q.t().contiguous().t() is (N, K) col-major
        w_col = w_q.t().contiguous().t()
        grad_x = torch._scaled_mm(
            gy_q_e5,
            w_col,
            scale_a=sgy,
            scale_b=sw,
            out_dtype=torch.bfloat16,
        ).reshape(*grad_out.shape[:-1], K).to(ctx.orig_dtype)

        # grad_w = grad_y^T @ x : (N, M) @ (M, K)
        # mat1: grad_y^T row-major contiguous (N, M); mat2: x_q column-major (M, K) = x_q.t().contiguous().t()
        gy_t = grad_2d.t().contiguous()
        gy_t_q, sgy_t = _quantize_e5m2_tensorwise(gy_t)
        x_col = x_q.t().contiguous().t()
        grad_w = torch._scaled_mm(
            gy_t_q,
            x_col,
            scale_a=sgy_t,
            scale_b=sx,
            out_dtype=torch.bfloat16,
        ).to(ctx.weight_dtype)

        grad_b = grad_2d.sum(dim=0).to(ctx.weight_dtype) if ctx.has_bias else None
        return grad_x, grad_w, grad_b


class ScaleAdaptiveTrainLinear(nn.Module):
    """Training linear layer keeping FP32/BF16 weights and switching between BF16 and FP8 by ctx."""

    def __init__(self, src: nn.Linear, ctx: PrecisionContext) -> None:
        super().__init__()
        self.in_features = src.in_features
        self.out_features = src.out_features
        self.weight = src.weight
        self.bias = src.bias
        self.ctx = ctx
        self.fp8_supported = (self.in_features % 16 == 0) and (self.out_features % 16 == 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.fp8_supported
            and self.ctx.is_fp8_level()
            and x.reshape(-1, self.in_features).shape[0] % 16 == 0
        ):
            return _FP8LinearFn.apply(x, self.weight, self.bias)
        w = self.weight.to(x.dtype) if self.weight.dtype != x.dtype else self.weight
        b = self.bias.to(x.dtype) if (self.bias is not None and self.bias.dtype != x.dtype) else self.bias
        return F.linear(x, w, b)


def patch_model_for_train_precision(
    net: nn.Module,
    mode: str,
    split_level: Optional[int] = None,
) -> PrecisionContext:
    """Wrap transformer block linears with ScaleAdaptiveTrainLinear for split-precision training."""
    levels = net.levels
    ctx = PrecisionContext(mode=mode, num_levels=len(levels), split_level=split_level, fp8_prefix=(mode == "fp8_all"))
    net._precision_ctx = ctx

    if mode in ("fp32", "bf16", "official_bf16", "pure_bf16"):
        return ctx

    def _patch(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear):
                setattr(mod, name, ScaleAdaptiveTrainLinear(child, ctx))
            else:
                _patch(child)

    _patch(net.transformer.blocks)
    return ctx


def _block_diag_chunk_mask(P: int, chunk_levels: list[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    T_c = sum(chunk_levels)
    d = torch.cat([torch.full((l,), i, device=device) for i, l in enumerate(chunk_levels)]).view(T_c, 1)
    dT = d.transpose(0, 1)
    br = torch.where(d == dT, 0.0, float("-inf"))
    bl = torch.zeros(T_c, P, device=device, dtype=dtype)
    return torch.cat([bl, br.to(dtype)], dim=1).reshape(1, 1, T_c, P + T_c)


def _rope_for_levels(net, P: int, lvl_start: int, lvl_end: int):
    if not net.use_rope:
        return None
    # Concatenate prefix RoPE [0..P-1] with target levels' per-level RoPE chunks.
    offsets = [0]
    for l in net.levels[net._lvl_first:]:
        offsets.append(offsets[-1] + l)
    s0 = P + offsets[lvl_start]
    s1 = P + offsets[lvl_end]
    cos = torch.cat([net.rope_cos[:P], net.rope_cos[s0:s1]], dim=0)
    sin = torch.cat([net.rope_sin[:P], net.rope_sin[s0:s1]], dim=0)
    return cos, sin


def scale_adaptive_train_forward(
    net,
    x: torch.Tensor,
    mode: str = "bf16",
    split_level: Optional[int] = None,
) -> torch.Tensor:
    """Equivalent to NextConceptTransformer.forward() for block_diagonal models, splitting coarse & fine levels."""
    if getattr(net.config, "attn_pattern", "block_diagonal") != "block_diagonal":
        raise ValueError("scale_adaptive_train_forward requires attn_pattern == 'block_diagonal'")
    if getattr(net, "use_ctx_encoder", False) or getattr(net, "predict_level0", False):
        raise ValueError("scale_adaptive_train_forward expects standard prefix + target levels")

    P = net.prefix_len
    target_levels = net.levels[net._lvl_first:]
    K = len(target_levels)
    split = int(split_level if split_level is not None else len(net.levels) // 2)
    split_target_idx = max(0, min(split - net._lvl_first, K))

    ctx: Optional[PrecisionContext] = getattr(net, "_precision_ctx", None)

    z_prefix = x[:, :P, :]
    x_target = x[:, P:, :]
    if net.use_post_upsample_conv:
        x_target = net._apply_post_upsample_convs_multi(x_target)

    if ctx is not None:
        ctx.level_idx = -1

    prefix_emb = net.transformer.wte(z_prefix)
    tgt_emb = net.transformer.wte(x_target)
    if not net.use_rope:
        prefix_emb = prefix_emb + net.wpe(net.wp_prefix)
        tgt_emb = tgt_emb + net.wpe(net.wp_1L)
    if net.config.use_level_encoding:
        tgt_emb = tgt_emb + net.transformer.wle(net.lvl_1L)

    # 1. Encode prefix once with gradients enabled (prefix_kv).
    rope_p = (net.rope_cos[:P], net.rope_sin[:P]) if net.use_rope else None
    pfx_mask = torch.zeros(1, 1, P, P, device=x.device, dtype=prefix_emb.dtype)
    h_p = prefix_emb
    prefix_kv = []
    for block in net.transformer.blocks:
        h_p, layer_kv = block(h_p, attn_mask=pfx_mask, rope_cos_sin=rope_p, return_kv=True)
        prefix_kv.append(layer_kv)

    coarse_len = sum(target_levels[:split_target_idx])
    out_chunks = []

    def _run_target_chunk(chunk_emb: torch.Tensor, lvl_a: int, lvl_b: int, level_tag: int) -> torch.Tensor:
        if ctx is not None:
            ctx.level_idx = level_tag
        mask = _block_diag_chunk_mask(P, target_levels[lvl_a:lvl_b], x.device, chunk_emb.dtype)
        rope = _rope_for_levels(net, P, lvl_a, lvl_b)
        h = chunk_emb
        for block, layer_kv in zip(net.transformer.blocks, prefix_kv):
            h = block(h, attn_mask=mask, rope_cos_sin=rope, past_kv=layer_kv)
        return net.transformer.ln_f(h)

    if split_target_idx > 0:
        out_chunks.append(_run_target_chunk(tgt_emb[:, :coarse_len, :], 0, split_target_idx, 0))
    if split_target_idx < K:
        out_chunks.append(_run_target_chunk(tgt_emb[:, coarse_len:, :], split_target_idx, K, split))

    x_out = torch.cat(out_chunks, dim=1) if len(out_chunks) > 1 else out_chunks[0]

    logits_list = []
    offset = 0
    for i, level in enumerate(target_levels):
        logits_list.append(net._pad_vocab(net.lm_heads[i](x_out[:, offset:offset + level, :])))
        offset += level
    return torch.cat(logits_list, dim=1)


def measure_module_vram_mib(net: nn.Module, tk=None) -> Dict[str, float]:
    """Compute actual GPU tensor bytes for parameters + buffers of net (and optional tk.vqvae)."""
    def _bytes(m: nn.Module) -> int:
        seen = set()
        total = 0
        for t in list(m.parameters()) + list(m.buffers()):
            if id(t) not in seen:
                seen.add(id(t))
                total += t.numel() * t.element_size()
        return total

    gen_mib = _bytes(net) / (1024 ** 2)
    vq_mib = _bytes(tk.vqvae) / (1024 ** 2) if tk is not None else 0.0
    return {
        "generator_mib": gen_mib,
        "vqvae_mib": vq_mib,
        "total_mib": gen_mib + vq_mib,
    }
