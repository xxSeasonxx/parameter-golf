#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pickle
import sys
import time
import zlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx  # MLX arrays are lazily evaluated — ops build a compute graph, nothing runs until mx.eval()
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten  # MLX uses nested dicts for params; these flatten/unflatten them

from train_gpt_common import (
    Hyperparameters as _CommonHyperparameters,
    POLAR_EXPRESS_COEFFS_5,
    BASELINE_NS_COEFFS,
    CONTROL_TENSOR_NAME_PATTERNS,
    INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
    INT8_KEEP_FLOAT_MAX_NUMEL,
    INT8_CLIP_PERCENTILE,
    INT8_CLIP_Q,
)

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

# MLX runs on Apple Silicon unified memory (CPU+GPU share RAM). bfloat16 is natively supported.
COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
#
# The shared env-var fields (model architecture, optimizer, QAT, deep supervision,
# eval_stride, use_polar_express, use_dyt_norm, etc.) live on the common base
# class. This file overrides MLX-specific defaults and adds MLX-only fields.
class Hyperparameters(_CommonHyperparameters):
    # MLX defaults to no in-loop validation (rely on the final eval after warmdown);
    # the common base defaults to 1000 to match PyTorch. Preserve MLX's behavior.
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # MLX historically accepted TRAIN_MAX_SEQ_LEN as a fallback; keep it.
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))

    # MLX-only memory and graph control fields.
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    # Force MLX to materialize the graph after every sub-batch, preventing lazy
    # graph buildup across accumulation steps. Keeps peak memory low on 16GB machines.
    # Disable on 32GB+ unified memory for better throughput (MLX_EAGER_EVAL=0).
    mlx_eager_eval = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    logit_chunk_tokens = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))

    # Byte-weighted loss: weight each token's CE by its decoded byte count.
    # Directly optimizes BPB instead of token-level CE.
    byte_weighted_loss = bool(int(os.environ.get("BYTE_WEIGHTED_LOSS", "0")))
    byte_weight_clamp_lo = float(os.environ.get("BYTE_WEIGHT_CLAMP_LO", 0.5))
    byte_weight_clamp_hi = float(os.environ.get("BYTE_WEIGHT_CLAMP_HI", 3.0))

    # Sequence length curriculum: progressive seq_len phases based on wallclock fraction.
    # Format: "seq_len:end_fraction,..." e.g. "256:0.25,512:0.55,1024:1.0"
    seq_len_curriculum = os.environ.get("SEQ_LEN_CURRICULUM", "")

    out_dir = os.environ.get("OUT_DIR", "logs")

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    """Split a microbatch into smaller sub-batches for MLX memory control.

    Unlike PyTorch where CUDA handles memory paging, MLX's lazy graph can grow unbounded.
    Splitting into chunks and calling mx.eval() between them (see mlx_eager_eval) keeps
    the compute graph small and peak unified memory usage predictable.
    """
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    """Flatten MLX's nested grad tree into a flat dict and accumulate weighted gradients.

    MLX returns grads as nested dicts mirroring model structure. We flatten them to
    flat "dotted.key" dicts so we can do simple per-key accumulation across sub-batches.
    """
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale  # lazy — just extends the graph until mx.eval()
    return accum


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
    ):
        self.stream = TokenStream(pattern, log_fn=log_fn, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    # Stores weights in fp32 and casts to compute dtype on the fly. MLX has no nn.Parameter;
    # any mx.array attribute on an nn.Module is automatically a parameter.
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        # Manual matmul instead of nn.Linear — MLX doesn't have F.linear(); x @ W^T is idiomatic.
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - causal masked SDPA
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        # MLX provides nn.RoPE as a built-in module (PyTorch version uses a custom implementation).
        # traditional=False selects the "non-interleaved" / GPT-NeoX style rotation.
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        # mx.fast.scaled_dot_product_attention dispatches to Metal-optimized attention kernels.
        # mask="causal" enables the fused causal mask path (no explicit mask tensor needed).
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc(x)
        x = mx.where(x > 0, x, 0.5 * x)  # LeakyReLU(0.5) — preserves negative gradient flow
        return self.proj(x * x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
    ):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = mx.ones((dim,), dtype=mx.float32)
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32)
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))

    def __call__(self, x: mx.array, x0: mx.array) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - tied embeddings for the LM head (the baseline default setup)
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int,
                 logit_chunk_tokens: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float, freq_skip_gating: bool = False, freq_skip_window: int = 32,
                 mlp_mult_asymmetric: str = "",
                 byte_weighted_loss: bool = False, byte_weight_clamp_lo: float = 0.5, byte_weight_clamp_hi: float = 3.0,
                 deep_supervision: bool = False, deep_supervision_alpha: float = 0.1, deep_supervision_tap_layers: list[int] | None = None):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self._byte_weighted_loss = byte_weighted_loss
        self._byte_weight_clamp_lo = byte_weight_clamp_lo
        self._byte_weight_clamp_hi = byte_weight_clamp_hi
        self._deep_supervision = deep_supervision
        self._deep_supervision_alpha = deep_supervision_alpha
        self._aux_tap_set = set(deep_supervision_tap_layers) if deep_supervision_tap_layers else set()

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.freq_skip_gating = freq_skip_gating
        self.freq_skip_window = freq_skip_window
        if freq_skip_gating:
            self.skip_lo_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
            self.skip_hi_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        else:
            self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        # Per-layer MLP width: asymmetric allows different encoder/decoder widths
        if mlp_mult_asymmetric:
            enc_m, dec_m = (int(x) for x in mlp_mult_asymmetric.split(","))
            layer_mlp_mults = [enc_m] * self.num_encoder_layers + [dec_m] * self.num_decoder_layers
        else:
            layer_mlp_mults = [mlp_mult] * num_layers
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, layer_mlp_mults[i], rope_base, qk_gain_init)
            for i in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def set_byte_luts(self, base_bytes_lut: np.ndarray, has_leading_space_lut: np.ndarray, is_boundary_token_lut: np.ndarray) -> None:
        """Store byte count LUTs for byte-weighted loss. Called once before training."""
        self._base_bytes_lut = mx.array(base_bytes_lut.astype(np.float32))
        self._has_leading_space_lut = mx.array(has_leading_space_lut.astype(np.float32))
        self._is_boundary_lut = mx.array(is_boundary_token_lut.astype(np.float32))

    def _byte_weights(self, input_ids_flat: mx.array, target_ids_flat: mx.array) -> mx.array:
        """Compute per-token byte weights matching the BPB eval metric."""
        bw = self._base_bytes_lut[target_ids_flat]
        # Add 1 byte for leading space when prev token is not a boundary token
        bw = bw + self._has_leading_space_lut[target_ids_flat] * (1.0 - self._is_boundary_lut[input_ids_flat])
        return mx.clip(bw, self._byte_weight_clamp_lo, self._byte_weight_clamp_hi)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            # Odd layer counts have one more decoder block than encoder block. The baseline only
            # applies a skip connection when one exists, then runs the remaining decoder block(s)
            # without an added skip.
            if skips:
                skip = skips.pop()
                if self.freq_skip_gating:
                    W = self.freq_skip_window
                    # Decompose: low-freq = block means, high-freq = residual
                    s = skip.reshape(*skip.shape[:-1], -1, W)
                    lo = mx.repeat(s.mean(axis=-1, keepdims=True), W, axis=-1).reshape(skip.shape)
                    hi = skip - lo
                    x = x + (self.skip_lo_weights[i].astype(x.dtype)[None, None, :] * lo +
                             self.skip_hi_weights[i].astype(x.dtype)[None, None, :] * hi)
                else:
                    x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skip
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def _forward_aux(self, input_ids: mx.array) -> tuple[mx.array, list[mx.array]]:
        """Forward pass with intermediate hidden state taps for deep supervision."""
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        x0 = x
        skips: list[mx.array] = []
        aux_states: list[mx.array] = []

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
            if i in self._aux_tap_set:
                aux_states.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                skip = skips.pop()
                if self.freq_skip_gating:
                    W = self.freq_skip_window
                    s = skip.reshape(*skip.shape[:-1], -1, W)
                    lo = mx.repeat(s.mean(axis=-1, keepdims=True), W, axis=-1).reshape(skip.shape)
                    hi = skip - lo
                    x = x + (self.skip_lo_weights[i].astype(x.dtype)[None, None, :] * lo +
                             self.skip_hi_weights[i].astype(x.dtype)[None, None, :] * hi)
                else:
                    x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skip
            x = self.blocks[self.num_encoder_layers + i](x, x0)
            if (self.num_encoder_layers + i) in self._aux_tap_set:
                aux_states.append(x)
        return self.final_norm(x), aux_states

    def _compute_main_loss(self, x: mx.array, y: mx.array, input_ids: mx.array) -> mx.array:
        """Compute main CE loss with optional byte weighting."""
        logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits_proj)
        if self._byte_weighted_loss:
            per_token_ce = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
            bw = self._byte_weights(input_ids.reshape(-1), y)
            return (per_token_ce * bw).sum() / bw.sum()
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

    def loss_train(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        """Training loss — byte weighting + deep supervision. Used by value_and_grad."""
        dim = self.tok_emb.weight.shape[1]
        y = target_ids.reshape(-1)

        if self._deep_supervision:
            final_x, aux_states = self._forward_aux(input_ids)
            x = final_x.reshape(-1, dim)
            main_loss = self._compute_main_loss(x, y, input_ids)
            # Auxiliary losses: project each tap through shared embedding, geometric decay
            n_aux = len(aux_states)
            for i, aux_x in enumerate(aux_states):
                ax = rms_norm(aux_x).reshape(-1, dim)
                aux_logits = self.softcap(ax @ self.tok_emb.weight.astype(ax.dtype).T)
                aux_ce = nn.losses.cross_entropy(aux_logits.astype(mx.float32), y, reduction="mean")
                weight = self._deep_supervision_alpha ** (n_aux - i)
                main_loss = main_loss + weight * aux_ce
            return main_loss

        x = self(input_ids).reshape(-1, dim)
        return self._compute_main_loss(x, y, input_ids)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        """Eval loss — always standard token-level CE (BPB is computed separately)."""
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

        loss_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits_proj = x[s:e] @ self.tok_emb.weight.astype(x.dtype).T
            logits = self.softcap(logits_proj)
            loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
        return loss_sum / float(n)

    def loss_per_token(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        """Return per-token cross-entropy losses (no reduction). Shape: (batch * seq_len,)."""
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits_proj)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    # Unlike PyTorch train_gpt.py, there is no distributed allreduce — MLX runs single-device only.
    # Momentum buffers are plain dicts of mx.arrays (no torch optimizer state_dict machinery).
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}
        # Pre-compute per-key layer LR scales (deeper layers get higher LR).
        self.lr_scales: dict[str, float] = {}
        if args.layer_lr_scale != 0:
            n = max(args.num_layers - 1, 1)
            for k in keys:
                # Extract layer index from key like "blocks.3.attn.c_q.weight"
                parts = k.split(".")
                layer_idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                self.lr_scales[k] = 1.0 + args.layer_lr_scale * (layer_idx / n)

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        base_lr = self.args.matrix_lr * lr_mul
        # Warmdown-aware WD: increase WD as LR drops during warmdown.
        # When lr_mul=1.0 (full LR), wd=base_wd. When lr_mul→0 (warmdown end), wd→2*base_wd.
        # This counteracts the natural weakening of WD's effect as updates shrink.
        wd = self.args.muon_weight_decay * (2.0 - lr_mul) if self.args.muon_weight_decay > 0 else 0.0
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            update = (g_ortho * scale).astype(p.dtype)
            if wd > 0:
                update = update + wd * p
            lr = base_lr * self.lr_scales.get(k, 1.0) if self.lr_scales else base_lr
            out[k] = p - lr * update
        return out


class SplitOptimizers:
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: GPT, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k
            for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2 and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k in ("skip_weights", "skip_lo_weights", "skip_hi_weights") or (k.startswith("blocks.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        # MLX optimizers don't mutate params in-place like PyTorch. apply_gradients() returns
        # new param dicts, which we collect into `updated` and then push back to the model.
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        # MLX optim.Adam.learning_rate is a mutable property — we set it directly each step
        # instead of using a scheduler callback (PyTorch uses param_groups or LR schedulers).
        self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            )
        )

        self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        # model.update() replaces parameters in the module tree. This is how MLX "applies"
        # optimizer results — there's no .step() that modifies tensors in-place.
        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# - per-row int8 for 2D float tensors
# - per-tensor int8 for other float tensors
# - fp16 passthrough for small float tensors
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

QUANT_BITS = int(os.environ.get("QUANT_BITS", 8))
QUANT_MAX_VAL = {6: 31, 8: 127}[QUANT_BITS]  # 2^(bits-1) - 1
# Tensors matching these patterns are kept as FP16 regardless of size (not int8 quantized).
# Use for critical tensors like tied embeddings where quantization error hurts disproportionately.
INT8_KEEP_FLOAT_FP16_NAME_PATTERNS = tuple(
    os.environ.get("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS", "").split(",")
) if os.environ.get("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS") else ()
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
# INT8_KEEP_FLOAT_MAX_NUMEL, INT8_CLIP_PERCENTILE, INT8_CLIP_Q come from train_gpt_common.


def sim_quant_roundtrip(w: mx.array) -> mx.array:
    """Simulate int8 quantize→dequantize roundtrip in MLX ops (stays on GPU).
    Per-row for 2D, per-tensor for 1D. Mirrors the actual int8 quantization path."""
    f = w.astype(mx.float32)
    qmax = float(QUANT_MAX_VAL)
    if f.ndim == 2:
        row_max = mx.maximum(mx.max(mx.abs(f), axis=1, keepdims=True), 1.0 / qmax)
        scale = row_max / qmax
        q = mx.clip(mx.round(f / scale), -qmax, qmax)
        return (q * scale).astype(w.dtype)
    amax = mx.maximum(mx.max(mx.abs(f)), mx.array(1.0 / qmax))
    scale = amax / qmax
    q = mx.clip(mx.round(f / scale), -qmax, qmax)
    return (q * scale).astype(w.dtype)


def _np_float32(arr: mx.array) -> np.ndarray:
    # MLX arrays live in unified memory — np.array() triggers mx.eval() and copies to a numpy view.
    # This is the MLX equivalent of tensor.cpu().numpy() in PyTorch.
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / QUANT_MAX_VAL, 1.0 / QUANT_MAX_VAL).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -QUANT_MAX_VAL, QUANT_MAX_VAL).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -QUANT_MAX_VAL, QUANT_MAX_VAL).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    # Quantization is done via numpy intermediaries (not MLX ops) since we need precise
    # control over rounding/clipping and the result is serialized to disk anyway.
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        # Named patterns can force specific large tensors to stay as FP16 (e.g., tied embeddings).
        if INT8_KEEP_FLOAT_FP16_NAME_PATTERNS and any(p in name for p in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS):
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": f"int{QUANT_BITS}_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest.json
    # (in the datasets parent dir) records which tokenizer produced which shard set,
    # letting the script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(pattern: str, seq_len: int) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = np.ascontiguousarray(np.concatenate([load_data_shard(file) for file in files], axis=0))
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
    seq_len: int | None = None,
) -> tuple[mx.array, dict]:
    """Compute loss+grads for one microbatch, chunked into sub-batches for memory control.

    This is the inner loop of gradient accumulation. Each microbatch is further split via
    token_chunks() so the MLX lazy graph stays small. The outer training loop then
    accumulates across grad_accum_steps microbatches before calling opt.step().
    """
    effective_seq_len = seq_len or args.train_seq_len
    chunk_sizes = token_chunks(args.microbatch_tokens, effective_seq_len, args.mlx_max_microbatch_tokens)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, effective_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)  # still lazy — nothing computed yet
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)  # force graph execution now, freeing intermediates
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: bits-per-byte, a tokenizer-agnostic compression metric used by the challenge.
    #   BPB byte counting uses numpy arrays (base_bytes_lut, has_leading_space_lut, etc.) because
    #   it needs SentencePiece token-to-byte mappings which are integer lookups, not differentiable.
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)  # force evaluation — without this, the graph would grow across batches
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def eval_val_sliding(
    args: Hyperparameters,
    compiled_loss_per_token,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    """Sliding-window validation with batching for efficiency.

    Processes overlapping windows where each scored token has (seq_len - stride) context.
    Windows are batched together so multiple forward passes happen in parallel.
    """
    stride = args.eval_stride
    seq_len = args.train_seq_len
    n_tokens = val_tokens.size - 1  # -1 because targets are shifted by 1

    # Build list of (win_start, score_start_in_val, n_scored) for each window
    windows = []
    # First window: score all seq_len tokens
    first_win_len = min(seq_len, n_tokens)
    windows.append((0, 0, first_win_len))
    # Subsequent windows: slide by stride, score last stride tokens
    pos = seq_len
    while pos < n_tokens:
        win_end = min(pos + stride, n_tokens)
        win_start = max(win_end - seq_len, 0)
        n_scored = min(stride, win_end - win_start)
        score_start_in_val = win_end - n_scored
        windows.append((win_start, score_start_in_val, n_scored))
        pos = win_end

    total_windows = len(windows)
    if total_windows == 0:
        return 0.0, 0.0

    # Determine batch size: how many windows fit in val_batch_size tokens
    batch_size = max(1, args.val_batch_size // (args.grad_accum_steps * seq_len))

    total_loss_sum = 0.0
    total_scored_tokens = 0.0
    total_bytes = 0.0

    for batch_start in range(0, total_windows, batch_size):
        batch_end = min(batch_start + batch_size, total_windows)
        batch_windows = windows[batch_start:batch_end]
        actual_batch = len(batch_windows)

        # Stack windows into a batch: all padded/truncated to seq_len
        x_batch = np.zeros((actual_batch, seq_len), dtype=np.int32)
        y_batch = np.zeros((actual_batch, seq_len), dtype=np.int32)
        for i, (ws, _, _) in enumerate(batch_windows):
            win_len = min(seq_len, n_tokens - ws)
            x_batch[i, :win_len] = val_tokens[ws:ws + win_len]
            y_batch[i, :win_len] = val_tokens[ws + 1:ws + win_len + 1]

        x = mx.array(x_batch, dtype=mx.int32)
        y = mx.array(y_batch, dtype=mx.int32)
        per_tok_loss = compiled_loss_per_token(x, y)
        mx.eval(per_tok_loss)
        losses_np = np.array(per_tok_loss, dtype=np.float32).reshape(actual_batch, seq_len)

        for i, (ws, score_start_val, n_scored) in enumerate(batch_windows):
            # Score offset within this window
            score_offset_in_win = score_start_val - ws
            scored = losses_np[i, score_offset_in_win:score_offset_in_win + n_scored]

            total_loss_sum += float(np.sum(scored, dtype=np.float64))
            total_scored_tokens += n_scored

            # BPB byte counting
            tgt_ids = val_tokens[score_start_val + 1:score_start_val + n_scored + 1]
            prev_ids = val_tokens[score_start_val:score_start_val + n_scored]
            bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
            bytes_np += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).astype(np.int16, copy=False)
            total_bytes += float(bytes_np.astype(np.float64).sum())

        batch_idx = batch_start // batch_size + 1
        total_batches = (total_windows + batch_size - 1) // batch_size
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")

    val_loss = total_loss_sum / total_scored_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_scored_tokens / total_bytes)
    return val_loss, val_bpb


# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    if not args.tie_embeddings:
        raise NotImplementedError("train_gpt_mlx.py only supports tied embeddings")
    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
        args.data_path,
        args.tokenizer_path,
    )
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size
    )

    # ==============================================================================
    # TRAINING SETUP
    # ==============================================================================
    mx.random.seed(args.seed)

    train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # ==============================================================================
    # MODEL + OPTIMIZER SETUP
    # ==============================================================================
    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        logit_chunk_tokens=args.logit_chunk_tokens,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        tied_embed_init_std=args.tied_embed_init_std,
        qk_gain_init=args.qk_gain_init,
        freq_skip_gating=args.freq_skip_gating,
        freq_skip_window=args.freq_skip_window,
        mlp_mult_asymmetric=args.mlp_mult_asymmetric,
        byte_weighted_loss=args.byte_weighted_loss,
        byte_weight_clamp_lo=args.byte_weight_clamp_lo,
        byte_weight_clamp_hi=args.byte_weight_clamp_hi,
        deep_supervision=args.deep_supervision,
        deep_supervision_alpha=args.deep_supervision_alpha,
        deep_supervision_tap_layers=(
            [int(x) for x in args.deep_supervision_layers.split(",") if x]
            if args.deep_supervision_layers
            else list(range(1, args.num_layers - 1, 2))  # default: every 2nd layer
        ) if args.deep_supervision else None,
    )
    if args.byte_weighted_loss:
        model.set_byte_luts(base_bytes_lut, has_leading_space_lut, is_boundary_token_lut)
        # Freeze byte LUTs so they're not trained or included in gradients
        model.freeze(keys=["_base_bytes_lut", "_has_leading_space_lut", "_is_boundary_lut"])
        log(f"byte_weighted_loss:enabled clamp=[{args.byte_weight_clamp_lo}, {args.byte_weight_clamp_hi}]")
    if args.deep_supervision:
        tap_layers = sorted(model._aux_tap_set)
        log(f"deep_supervision:enabled alpha={args.deep_supervision_alpha} tap_layers={tap_layers}")
    opt = SplitOptimizers(model, args)

    # ==============================================================================
    # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
    # ==============================================================================
    # mx.compile() traces and caches the compute graph for repeated execution (like torch.compile).
    # inputs/outputs=model.state tells MLX that model weights are "live state" that can change between
    # calls (e.g., after optimizer updates). Without this, MLX would bake initial weight values into
    # the compiled graph.
    #
    # The model also contains non-trainable arrays (e.g., RoPE frequency tables). Passing model.state
    # (not just model.parameters()) captures everything, avoiding "uncaptured input" errors.
    #
    # Two separate compiled functions are needed because MLX traces different graphs:
    # - compiled_loss: forward-only (used for validation)
    # - compiled_loss_and_grad: forward + backward via nn.value_and_grad (used for training)
    # They cannot share a compiled graph because the backward pass changes the trace structure.
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_per_token = mx.compile(lambda x, y: model.loss_per_token(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss_train(x, y)),
        inputs=model.state,
        outputs=model.state,
    )

    # Print config once so logs are self-describing.
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    log(f"run_id:{args.run_id}")
    log(f"mlx_version:{mx.__version__}")
    log(f"train_loader:shards pattern={args.train_files}")
    log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
    if expected_train_files is None:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
    elif actual_train_files < expected_train_files:
        log(
            f"WARNING: train_loader:subset dataset:{dataset_name} "
            f"train_shards:{actual_train_files}/{expected_train_files} "
            f"new epochs will arrive sooner than the full dataset"
        )
    else:
        log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
    log(f"tokenizer_path:{args.tokenizer_path}")
    log(
        f"model_params:{n_params} vocab_size:{args.vocab_size} layers:{args.num_layers} "
        f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
        f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
    )
    log(
        f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
        f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
        f"val_batch_size:{args.val_batch_size} "
        f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
    log(
        f"optimizer:muon+adam muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
        f"embed_lr:{args.tied_embed_lr} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
    )
    log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log(f"compute_dtype:{COMPUTE_DTYPE} compile:True")
    log(
        f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
        f"linear_weight:{model.blocks[0].attn.c_q.weight.dtype} "
        f"skip_weights:{model.skip_lo_weights.dtype if model.freq_skip_gating else model.skip_weights.dtype}"
    )

    # ==============================================================================
    # TRAINING LOOP
    # ==============================================================================
    if args.warmup_steps > 0:
        # Warmup primes MLX's compile cache and Metal memory allocator without affecting training.
        # Unlike PyTorch, we do NOT snapshot/restore model weights here. Why? On unified memory Macs,
        # saving a full copy of model + optimizer state doubles peak memory. Instead, we simply
        # run forward+backward but skip the optimizer step, then reset the data loader so training
        # starts from the correct token position. The weights remain at their initial values.
        for warmup_step in range(args.warmup_steps):
            accum: dict[str, mx.array] | None = None
            warmup_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                warmup_loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
                accum = accumulate_flat_grads(accum, grads, grad_scale)
            mx.eval(warmup_loss, accum)  # mx.eval() starts async GPU work
            mx.synchronize()  # mx.synchronize() blocks until all GPU work completes (like torch.cuda.synchronize())
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

        # Prime the compiled_loss graph separately — it has a different trace than compiled_loss_and_grad.
        val_batch_tokens = args.val_batch_size // args.grad_accum_steps
        if val_batch_tokens < args.train_seq_len:
            raise ValueError(
                "VAL_BATCH_SIZE must provide at least one sequence; "
                f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                f"TRAIN_SEQ_LEN={args.train_seq_len}"
            )
        warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
        warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        if args.eval_stride > 0:
            # Prime the per-token loss graph for sliding window eval with expected batch shape
            sw_batch_size = max(1, args.val_batch_size // (args.grad_accum_steps * args.train_seq_len))
            warm_n = min(sw_batch_size, (val_tokens.size - 1) // args.train_seq_len)
            warm_x = mx.array(val_tokens[:warm_n * args.train_seq_len].reshape(warm_n, -1), dtype=mx.int32)
            warm_y = mx.array(val_tokens[1:warm_n * args.train_seq_len + 1].reshape(warm_n, -1), dtype=mx.int32)
            warm_per_tok = compiled_loss_per_token(warm_x, warm_y)
            mx.eval(warm_per_tok)
        mx.synchronize()

        train_loader = TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    # Parse sequence length curriculum phases
    curriculum_phases: list[tuple[int, float]] = []  # [(seq_len, end_wallclock_fraction), ...]
    if args.seq_len_curriculum:
        for phase_str in args.seq_len_curriculum.split(","):
            sl, frac = phase_str.strip().split(":")
            curriculum_phases.append((int(sl), float(frac)))
        log(f"seq_len_curriculum:phases={curriculum_phases}")
    current_seq_len = curriculum_phases[0][0] if curriculum_phases else args.train_seq_len
    current_phase_idx = 0

    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    stop_after_step: int | None = None
    t0 = time.perf_counter()
    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            # Validation always scans the same fixed full validation split.
            val_loss, val_bpb = eval_val(
                args,
                compiled_loss,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
            )
            if step % 25 == 0 or last_step:
                log(
                    f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                    f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                )
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
            break

        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        # Curriculum: check if we should transition to next seq_len phase
        if curriculum_phases and max_wallclock_ms:
            approx_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
            frac = approx_ms / max_wallclock_ms
            while current_phase_idx < len(curriculum_phases) - 1 and frac >= curriculum_phases[current_phase_idx][1]:
                current_phase_idx += 1
                new_seq_len = curriculum_phases[current_phase_idx][0]
                if new_seq_len != current_seq_len:
                    log(f"curriculum_transition: seq_len {current_seq_len} -> {new_seq_len} at wallclock_frac={frac:.3f} step={step}")
                    current_seq_len = new_seq_len

        accum: dict[str, mx.array] | None = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        # Gradient accumulation is explicit here (no scaler or DDP averaging).
        # PyTorch train_gpt.py divides by world_size * accum_steps; MLX has no distributed, so just accum_steps.
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad, seq_len=current_seq_len)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)  # materialize each microbatch to cap peak memory

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())  # .item() triggers mx.eval() for this scalar
        opt.step(model, grads, step=step, lr_mul=lr_mul)

        # QAT: nudge weights toward their quantized form during pre-warmdown.
        if args.qat_prewarmdown and lr_mul >= args.qat_stop_lr_mul and step % args.qat_every == 0:
            flat = {k: v for k, v in tree_flatten(model.state)}
            qat_updates = {}
            for name, w in flat.items():
                if not mx.issubdtype(w.dtype, mx.floating) or w.size <= INT8_KEEP_FLOAT_MAX_NUMEL:
                    continue
                if INT8_KEEP_FLOAT_FP16_NAME_PATTERNS and any(p in name for p in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS):
                    continue
                w_q = sim_quant_roundtrip(w)
                qat_updates[name] = w + args.qat_strength * (w_q - w)
            if qat_updates:
                model.update(tree_unflatten(list(qat_updates.items())))

        # mx.synchronize() is the timing fence — everything above is lazy/async. This blocks
        # until all Metal GPU work finishes, giving accurate wall-clock step timing.
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = args.train_batch_tokens / (step_ms / 1000.0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
            log(
                f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
                + (f" seq_len:{current_seq_len}" if curriculum_phases else "")
            )
        if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step

    # ==============================================================================
    # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
    # ==============================================================================
    # We always write a raw artifact and a quantized artifact, then validate the
    # quantized roundtrip directly by loading the dequantized tensors back into the
    # model and running one final validation pass.
    out_path = out_dir / f"{args.run_id}_mlx_model.npz"
    # Exclude byte LUTs (non-model arrays used only during training) from serialization
    _byte_lut_keys = {"_base_bytes_lut", "_has_leading_space_lut", "_is_boundary_lut"}
    flat_state = {k: v for k, v in tree_flatten(model.state) if k not in _byte_lut_keys}
    mx.savez(str(out_path), **flat_state)  # MLX's native save format (numpy-compatible .npz)
    log(f"saved_model:{out_path} bytes:{out_path.stat().st_size}")

    # Quantized checkpoint uses pickle + zlib (not torch.save) since MLX has no native
    # checkpoint format for quantized models. The numpy intermediaries are pickle-friendly.
    quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    quant_serialized_bytes = len(quant_raw)
    quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
    with quant_path.open("wb") as f:
        f.write(quant_blob)
    quant_file_bytes = quant_path.stat().st_size
    ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    log(
        f"serialized_model_int8_zlib:{quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_pickle:{quant_serialized_bytes} payload_ratio:{ratio:.2f}x)"
    )

    with quant_path.open("rb") as f:
        quant_blob_disk = f.read()
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob_disk)))
    model.update(tree_unflatten(list(quant_flat.items())))
    q_t0 = time.perf_counter()
    if args.eval_stride > 0:
        q_val_loss, q_val_bpb = eval_val_sliding(
            args,
            compiled_loss_per_token,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log,
        )
    else:
        q_val_loss, q_val_bpb = eval_val(
            args,
            compiled_loss,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            log_fn=log,
        )
    q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
    log(f"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms")
    log(f"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


if __name__ == "__main__":
    main()
