"""Baseline training script. Keep under 1500 lines."""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import zlib
from pathlib import Path

try:
    import zstandard as zstd_mod
except ImportError:
    zstd_mod = None

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from train_gpt_common import (
    Hyperparameters as _CommonHyperparameters,
    POLAR_EXPRESS_COEFFS_5 as _POLAR_EXPRESS_COEFFS_5,
    BASELINE_NS_COEFFS as _BASELINE_NS_COEFFS,
    CONTROL_TENSOR_NAME_PATTERNS,
    INT8_KEEP_FLOAT_FP32_NAME_PATTERNS,
    INT8_KEEP_FLOAT_MAX_NUMEL,
    INT8_CLIP_PERCENTILE,
    INT8_CLIP_Q,
    get_ds_tap_layers,
    get_compression_name,
    describe_feature_flags,
    final_eval_weight_source,
    sim_quant_roundtrip_torch as sim_quant_roundtrip,
    quantize_float_tensor_calibrated_torch,
    DyTTorch as DyT,
    compute_bpb_from_sums,
)
# Backwards-compat alias (we kept the underscore-prefixed name for callers
# inside this module; common drops the underscore).
_get_ds_tap_layers = get_ds_tap_layers


class Hyperparameters(_CommonHyperparameters):
    # PyTorch-only fields (TTT-LoRA, EMA, layer growth, zstd, calibrated quant,
    # plus train_files/val_files convenience strings and the non-tied embed/head LRs).
    train_files = os.path.join(_CommonHyperparameters.data_path, "fineweb_train_*.bin")
    val_files = os.path.join(_CommonHyperparameters.data_path, "fineweb_val_*.bin")

    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))

    ema_decay = float(os.environ.get("EMA_DECAY", 0.0))
    int8_keep_float_fp16_name_patterns = os.environ.get("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS", "")
    calibrated_quant = bool(int(os.environ.get("CALIBRATED_QUANT", "0")))
    grow_layers_from = int(os.environ.get("GROW_LAYERS_FROM", 0))
    grow_at_wallclock_frac = float(os.environ.get("GROW_AT_WALLCLOCK_FRAC", 0.35))
    use_zstd = bool(int(os.environ.get("USE_ZSTD", "0")))
    zstd_level = int(os.environ.get("ZSTD_LEVEL", 22))

    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    ttt_batch_size = int(os.environ.get("TTT_BATCH_SIZE", 64))

# Muon optimizer (from modded-nanogpt, see https://kellerjordan.github.io/posts/muon/)


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7,
                                 use_polar_express: bool = False) -> Tensor:
    """Map G -> UV^T (nearest orthogonal matrix) via degree-5 Newton-Schulz iteration.

    use_polar_express: when True, use Chebyshev-minimax coefficients per iteration.
    Otherwise: modded-nanogpt baseline (constant a, b, c).
    """
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    if use_polar_express:
        for i in range(steps):
            a, b, c = _POLAR_EXPRESS_COEFFS_5[min(i, len(_POLAR_EXPRESS_COEFFS_5) - 1)]
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
    else:
        a, b, c = _BASELINE_NS_COEFFS
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),
        )

    def set_layer_lr_scales(self, param_to_scale: dict):
        self._param_lr_scales = param_to_scale

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps, use_polar_express=Hyperparameters.use_polar_express)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            wd = group.get("weight_decay", 0.0)
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                lr_scale = self._param_lr_scales.get(id(p), 1.0) if hasattr(self, '_param_lr_scales') else 1.0
                if wd > 0:
                    base_lr_val = group.get("base_lr", lr)
                    lr_mul = lr / base_lr_val if base_lr_val > 0 else 1.0
                    wd_eff = wd * (2.0 - lr_mul)  # warmdown-aware WD
                    p.data.mul_(1.0 - lr * lr_scale * wd_eff)
                p.add_(g, alpha=-lr * lr_scale)
                curr += p.numel()

        return loss

# Tokenizer-agnostic BPB evaluation: bits-per-byte instead of token-level loss.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """Build LUTs for token-level loss -> byte-level BPB conversion."""
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)       # UTF-8 byte count per token
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_) # True if token has "▁" prefix
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)  # True for control/unk/unused
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]  # strip the "▁" before counting raw text bytes
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    # Keep only a multiple of seq_len tokens (+1 for the next-token prediction shift).
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss_f, val_bpb = compute_bpb_from_sums(
        val_loss_sum.item(), val_token_count.item(), val_byte_count.item()
    )
    model.train()
    return float(val_loss_f), float(val_bpb)


def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    """Sliding-window evaluation: each scored token sees (seq_len - stride) of context.

    For each window of `seq_len` tokens, only the last `stride` tokens are scored.
    With stride=64 and seq_len=1024, each token gets 960 context tokens.

    Cost: ~seq_len/stride more forward passes than chunked. With stride == seq_len,
    degenerates to chunked eval.
    """
    stride = args.eval_stride if args.eval_stride > 0 else args.train_seq_len
    seq_len = args.train_seq_len

    n_tokens = val_tokens.numel()
    n_windows_total = max(1, (n_tokens - seq_len) // stride + 1)
    win_per_rank = (n_windows_total + world_size - 1) // world_size
    win_start_idx = rank * win_per_rank
    win_end_idx = min(win_start_idx + win_per_rank, n_windows_total)

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for win_idx in range(win_start_idx, win_end_idx):
            start = win_idx * stride
            end = min(start + seq_len + 1, n_tokens)
            local = val_tokens[start:end].to(device=device, dtype=torch.int64, non_blocking=True)
            if local.numel() < 2:
                continue
            x = local[:-1].unsqueeze(0)
            y = local[1:].unsqueeze(0)
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16, enabled=device.type == "cuda",
            ):
                batch_loss = model(x, y).detach()

            n_predicted = int(end - start - 1)
            if win_idx == 0:
                scored_tokens = n_predicted
                tgt_subset = y.reshape(-1)
                prev_subset = x.reshape(-1)
            else:
                scored_tokens = min(stride, n_predicted)
                tgt_subset = y.reshape(-1)[-scored_tokens:]
                prev_subset = x.reshape(-1)[-scored_tokens:]

            val_loss_sum += batch_loss.to(torch.float64) * float(scored_tokens)
            val_token_count += float(scored_tokens)

            token_bytes = base_bytes_lut[tgt_subset].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_subset] & ~is_boundary_token_lut[prev_subset]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss_f, val_bpb = compute_bpb_from_sums(
        val_loss_sum.item(), val_token_count.item(), val_byte_count.item()
    )
    model.train()
    return float(val_loss_f), float(val_bpb)

# Post-training int8 quantization + compression.
# Pattern tuples + numerical constants come from train_gpt_common; only the
# framework-specific store dtypes stay here.
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()  # downcast to fp16 to save space
    return t

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Per-tensor quantization for 1D params.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale

# Calibrated per-row int8 quant lives in train_gpt_common; bind a thin wrapper
# that injects the per-tensor fallback (the upstream quantize_float_tensor below).
def quantize_float_tensor_calibrated(tensor: Tensor) -> tuple:
    return quantize_float_tensor_calibrated_torch(tensor, quantize_float_tensor)

def quantize_state_dict_int8(state_dict: dict[str, Tensor], calibrated: bool = False):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    fp16_pats = tuple(p for p in os.environ.get("INT8_KEEP_FLOAT_FP16_NAME_PATTERNS", "").split(",") if p)

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        if fp16_pats and any(p in name for p in fp16_pats):
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        if calibrated and t.ndim == 2:
            q, s = quantize_float_tensor_calibrated(t)
            q = torch.from_numpy(q) if not isinstance(q, torch.Tensor) else q
            s = torch.from_numpy(s) if not isinstance(s, torch.Tensor) else s
        else:
            q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
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

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# Data loading.

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# Transformer modules.

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
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
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)

    def forward(self, x: Tensor, q_delta=None, v_delta=None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x) + (q_delta if q_delta is not None else 0)
        k = self.c_k(x)
        v = self.c_v(x) + (v_delta if v_delta is not None else 0)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = F.leaky_relu(x, 0.5)
        return self.proj(x.square())


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
        if Hyperparameters.use_dyt_norm:
            self.attn_norm = DyT(dim)
            self.mlp_norm = DyT(dim)
        else:
            self.attn_norm = RMSNorm()
            self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor, q_delta_fn=None, v_delta_fn=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        n = self.attn_norm(x)
        qd = q_delta_fn(n) if q_delta_fn is not None else None
        vd = v_delta_fn(n) if v_delta_fn is not None else None
        attn_out = self.attn(n, qd, vd)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mlp_mult_asymmetric: str = "",
        freq_skip_gating: bool = False,
        freq_skip_window: int = 32,
        deep_supervision: bool = False,
        deep_supervision_alpha: float = 0.1,
        deep_supervision_tap_layers: list | None = None,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self._deep_supervision = deep_supervision
        self._ds_alpha = deep_supervision_alpha
        self._ds_taps = set(deep_supervision_tap_layers or [])
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.freq_skip_gating = freq_skip_gating
        self.freq_skip_window = freq_skip_window
        if freq_skip_gating and model_dim % freq_skip_window != 0:
            raise ValueError(f"model_dim ({model_dim}) must be divisible by freq_skip_window ({freq_skip_window})")
        if freq_skip_gating:
            self.skip_lo_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
            self.skip_hi_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        if mlp_mult_asymmetric:
            enc_m, dec_m = (int(x) for x in mlp_mult_asymmetric.split(","))
            layer_mlp_mults = [enc_m] * self.num_encoder_layers + [dec_m] * self.num_decoder_layers
        else:
            layer_mlp_mults = [mlp_mult] * num_layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    layer_mlp_mults[i],
                    rope_base,
                    qk_gain_init,
                )
                for i in range(num_layers)
            ]
        )
        self.final_norm = DyT(model_dim) if Hyperparameters.use_dyt_norm else RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def forward(self, input_ids: Tensor, target_ids: Tensor, lora=None) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        aux_states: list[Tensor] = []

        for i in range(self.num_encoder_layers):
            qd = lora.q_loras[i] if lora else None
            vd = lora.v_loras[i] if lora else None
            x = self.blocks[i](x, x0, qd, vd)
            skips.append(x)
            if self._deep_supervision and i in self._ds_taps:
                aux_states.append(x)
        for i in range(self.num_decoder_layers):
            bi = self.num_encoder_layers + i
            if skips:
                skip = skips.pop()
                if self.freq_skip_gating:
                    W = self.freq_skip_window
                    s = skip.reshape(*skip.shape[:-1], -1, W)
                    lo = s.mean(dim=-1, keepdim=True).expand_as(s).reshape(skip.shape)
                    hi = skip - lo
                    x = x + (self.skip_lo_weights[i].to(dtype=x.dtype)[None, None, :] * lo +
                             self.skip_hi_weights[i].to(dtype=x.dtype)[None, None, :] * hi)
                else:
                    x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skip
            qd = lora.q_loras[bi] if lora else None
            vd = lora.v_loras[bi] if lora else None
            x = self.blocks[bi](x, x0, qd, vd)
            if self._deep_supervision and bi in self._ds_taps:
                aux_states.append(x)
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits = F.linear(x, self.tok_emb.weight)
        else:
            logits = self.lm_head(x)
        logits = logits + (lora.lm_head_lora(x) if lora else 0)
        logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        if lora:
            bsz, sl, V = logits.shape
            return F.cross_entropy(
                logits.float().reshape(-1, V), target_ids.reshape(-1), reduction="none").reshape(bsz, sl)
        main_loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction="mean")
        if self._deep_supervision and aux_states:
            emb_w = self.tok_emb.weight
            n_aux = len(aux_states)
            for idx, ax in enumerate(aux_states):
                ax_norm = F.rms_norm(ax, (ax.size(-1),))
                aux_logits = F.linear(ax_norm, emb_w)
                aux_logits = self.logit_softcap * torch.tanh(aux_logits / self.logit_softcap)
                aux_ce = F.cross_entropy(aux_logits.float().reshape(-1, aux_logits.size(-1)), target_ids.reshape(-1), reduction="mean")
                weight = self._ds_alpha ** (n_aux - idx)
                main_loss = main_loss + weight * aux_ce
        return main_loss


# Test-time training (LoRA): per-document low-rank adaptation at eval time.

BOS_ID = 1

class BatchedLinearLoRA(nn.Module):
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))     # down-projection
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))    # up-projection (zero-init)
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.A.transpose(1, 2)) @ self.B.transpose(1, 2)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()

# Compatible with DyT (replaces RMSNorm only; attn weight slots c_q, c_v unchanged).
class BatchedTTTLoRA(nn.Module):
    def __init__(self, bsz: int, model: GPT, rank: int):
        super().__init__()
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        for block in model.blocks:
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_q.weight.shape[0], rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, block.attn.c_v.weight.shape[0], rank))

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()

def _reset_ttt_optimizer(opt):
    for group in opt.param_groups:
        for p in group['params']:
            s = opt.state.get(p)
            if not s:
                continue
            s['exp_avg'].zero_()
            s['exp_avg_sq'].zero_()
            s['step'].fill_(0)

def _build_ttt_optimizer(lora, args: Hyperparameters):
    return torch.optim.Adam(lora.parameters(), lr=args.ttt_lora_lr, betas=(args.beta1, args.beta2), eps=1e-10)

def _find_docs(all_tokens: Tensor, include_next_bos: bool = True) -> list[tuple[int, int]]:
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].numpy()
    docs = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else all_tokens.numel()
        if include_next_bos and i + 1 < len(bos_positions):
            end += 1
        assert end - start >= 2
        docs.append((start, end - start))
    return docs

def _compute_chunk_window(ci: int, pred_len: int, num_chunks: int, chunk_size: int, eval_seq_len: int):
    chunk_start = ci * chunk_size
    chunk_end = pred_len if ci == num_chunks - 1 else (ci + 1) * chunk_size
    win_start = max(0, chunk_end - eval_seq_len)
    win_len = chunk_end - win_start
    chunk_offset = chunk_start - win_start
    chunk_len = chunk_end - chunk_start
    return win_start, win_len, chunk_offset, chunk_len

def _accumulate_bpb(
    ptl: Tensor, x: Tensor, y: Tensor,
    batch_i: int, chunk_offset: int, chunk_len: int,
    base_bytes_lut: Tensor, has_leading_space_lut: Tensor, is_boundary_token_lut: Tensor,
    loss_sum: Tensor, byte_sum: Tensor, token_count: Tensor,
):
    lbl = ptl[batch_i, chunk_offset:chunk_offset + chunk_len].to(torch.float64)
    prev = x[batch_i, chunk_offset:chunk_offset + chunk_len]
    tgt = y[batch_i, chunk_offset:chunk_offset + chunk_len]
    tok_bytes = base_bytes_lut[tgt].to(torch.float64)
    tok_bytes += has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]
    loss_sum += lbl.sum()
    byte_sum += tok_bytes.sum()
    token_count += chunk_len

def eval_val_ttt_lora(
    args: Hyperparameters,
    base_model: GPT,
    rank: int,
    world_size: int,
    device: torch.device,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    files = sorted(glob.glob(args.val_files))
    all_tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    docs = _find_docs(all_tokens)

    rank_docs = docs[(len(docs) * rank) // world_size : (len(docs) * (rank + 1)) // world_size]
    chunk_size = args.ttt_chunk_size
    eval_seq_len = args.ttt_eval_seq_len
    batch_size = args.ttt_batch_size
    lora_rank = args.ttt_lora_rank

    rank_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    lora = BatchedTTTLoRA(batch_size, base_model, lora_rank).to(device)
    opt = _build_ttt_optimizer(lora, args)

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    byte_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)

    for bi in range(0, len(rank_docs), batch_size):
        batch = rank_docs[bi:bi + batch_size]
        bsz = len(batch)

        if bsz == batch_size:
            cur_lora, cur_opt = lora, opt
            cur_lora.reset()
            _reset_ttt_optimizer(cur_opt)
        else:
            cur_lora = BatchedTTTLoRA(bsz, base_model, lora_rank).to(device)
            cur_opt = _build_ttt_optimizer(cur_lora, args)

        pred_lens = [doc_len - 1 for _, doc_len in batch]
        num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
        max_nc = max(num_chunks)

        for ci in range(max_nc):
            chunk_stats = _compute_chunk_window(ci, (ci + 1) * chunk_size, ci + 1, chunk_size, eval_seq_len)
            context_size, chunk_offset = chunk_stats[1], chunk_stats[2]

            active = [ci < nc for nc in num_chunks]
            needs_train = any(ci < nc - 1 for nc in num_chunks)

            x = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, context_size, dtype=torch.int64, device=device)
            doc_info = []  # (chunk_offset, chunk_len) per doc
            for b in range(bsz):
                if not active[b]:
                    doc_info.append((0, 0))
                    continue
                ds, dl = batch[b]
                ws, wl, co, cl = _compute_chunk_window(ci, pred_lens[b], num_chunks[b], chunk_size, eval_seq_len)
                chunk = all_tokens[ds + ws: ds + ws + wl + 1]
                toks = chunk.to(dtype=torch.int64, device=device)
                x[b, :wl] = toks[:-1]
                y[b, :wl] = toks[1:]
                doc_info.append((co, cl))

            if needs_train:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)
            else:
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    ptl = base_model(x, y, lora=cur_lora)

            with torch.no_grad():
                for b in range(bsz):
                    if not active[b]:
                        continue
                    co, cl = doc_info[b]
                    _accumulate_bpb(
                        ptl, x, y, b, co, cl, base_bytes_lut, has_leading_space_lut,
                        is_boundary_token_lut, loss_sum, byte_sum, token_count)

            if needs_train:
                mask = torch.tensor([float(ci < num_chunks[b] - 1) for b in range(bsz)], device=device)
                per_doc = ptl[:, chunk_offset:chunk_offset + chunk_size].mean(dim=-1)
                cur_opt.zero_grad()
                (per_doc * mask).sum().backward()
                cur_opt.step()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

    val_loss = float(loss_sum.item() / token_count.item())
    val_bpb = float((loss_sum.item() / math.log(2.0)) / byte_sum.item())
    return val_loss, val_bpb

def grow_model(
    small_model: GPT,
    args,
    device,
) -> GPT:
    """Grow model to full depth by inserting new zero-init layers."""
    small_state = small_model.state_dict()
    small_enc = small_model.num_encoder_layers
    small_dec = small_model.num_decoder_layers

    big = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mlp_mult_asymmetric=args.mlp_mult_asymmetric, freq_skip_gating=args.freq_skip_gating,
        freq_skip_window=args.freq_skip_window, deep_supervision=args.deep_supervision,
        deep_supervision_alpha=args.deep_supervision_alpha,
        deep_supervision_tap_layers=_get_ds_tap_layers(args, args.num_layers),
    ).to(device).bfloat16()

    big_enc = big.num_encoder_layers
    big_dec = big.num_decoder_layers
    big_state = big.state_dict()

    for k, v in small_state.items():
        if not k.startswith("blocks."):
            if k in big_state and big_state[k].shape == v.shape:
                big_state[k] = v.clone()

    for i in range(small_enc):
        for k, v in small_state.items():
            if k.startswith(f"blocks.{i}."):
                if k in big_state:
                    big_state[k] = v.clone()

    for i in range(small_dec):
        small_idx = small_enc + i
        big_idx = big_enc + (big_dec - small_dec) + i
        for k, v in small_state.items():
            if k.startswith(f"blocks.{small_idx}."):
                suffix = k[len(f"blocks.{small_idx}."):]
                big_key = f"blocks.{big_idx}.{suffix}"
                if big_key in big_state and big_state[big_key].shape == v.shape:
                    big_state[big_key] = v.clone()

    big.load_state_dict(big_state)
    return big


def build_optimizers(base_model: GPT, args: Hyperparameters, num_layers: int, lr_scale: float = 1.0) -> tuple:
    """Build all optimizer groups. Returns (optimizers_list, muon_optimizer)."""
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [p for n, p in block_named_params
                     if p.ndim == 2 and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    scalar_params = [p for n, p in block_named_params
                     if p.ndim < 2 or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)]
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    if hasattr(base_model, "skip_lo_weights"):
        scalar_params.append(base_model.skip_lo_weights)
        scalar_params.append(base_model.skip_hi_weights)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr * lr_scale, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    optimizer_muon = Muon(matrix_params, lr=args.matrix_lr * lr_scale,
        momentum=args.muon_momentum, backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay)
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    if args.layer_lr_scale != 0:
        param_to_scale = {}
        n = max(num_layers - 1, 1)
        for name, p in base_model.blocks.named_parameters():
            if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS):
                layer_idx = int(name.split(".")[0])
                param_to_scale[id(p)] = 1.0 + args.layer_lr_scale * (layer_idx / n)
        optimizer_muon.set_layer_lr_scales(param_to_scale)
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr * lr_scale, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
    opts: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr * lr_scale, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2), eps=args.adam_eps, fused=True)
        opts.insert(1, optimizer_head)
    return opts, optimizer_muon


def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    effective_num_layers = args.grow_layers_from if args.grow_layers_from > 0 else args.num_layers
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=effective_num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mlp_mult_asymmetric=args.mlp_mult_asymmetric,
        freq_skip_gating=args.freq_skip_gating,
        freq_skip_window=args.freq_skip_window,
        deep_supervision=args.deep_supervision,
        deep_supervision_alpha=args.deep_supervision_alpha,
        deep_supervision_tap_layers=_get_ds_tap_layers(args, effective_num_layers),
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
        if isinstance(module, Rotary):
            module.inv_freq.data = module.inv_freq.data.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True) if distributed else compiled_model
    optimizers, optimizer_muon = build_optimizers(base_model, args, effective_num_layers)

    ema_state = None
    if args.ema_decay > 0:
        ema_state = {k: v.clone() for k, v in base_model.state_dict().items()}
        log0(f"ema:enabled decay={args.ema_decay}")

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{args.tied_embed_lr if args.tie_embeddings else args.embed_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")
    for line in describe_feature_flags(args, effective_num_layers):
        log0(line)

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    training_time_ms = 0.0
    stop_after_step: int | None = None
    layer_grown = args.grow_layers_from == 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            eval_fn = eval_val_sliding if args.eval_stride > 0 else eval_val
            val_loss, val_bpb = eval_fn(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)

        # Progressive layer growing
        if not layer_grown:
            elapsed_frac = elapsed_ms / max_wallclock_ms if max_wallclock_ms else 0
            should_grow = elapsed_frac >= args.grow_at_wallclock_frac
            if distributed:
                grow_tensor = torch.tensor(int(should_grow), device=device)
                dist.all_reduce(grow_tensor, op=dist.ReduceOp.MAX)
                should_grow = bool(grow_tensor.item())
            if should_grow:
                layer_grown = True
                log0(f"layer_growth: {args.grow_layers_from}L -> {args.num_layers}L at wallclock_frac={elapsed_frac:.3f} step={step}")
                base_model = grow_model(base_model, args, device)
                for module in base_model.modules():
                    if isinstance(module, CastedLinear):
                        module.float()
                    if isinstance(module, Rotary):
                        module.inv_freq.data = module.inv_freq.data.float()
                restore_low_dim_params_to_fp32(base_model)
                compiled_model = torch.compile(base_model, dynamic=False, fullgraph=False)
                model = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True) if distributed else compiled_model
                optimizers, optimizer_muon = build_optimizers(base_model, args, args.num_layers, lr_scale=scale)
                if ema_state is not None:
                    ema_state = {k: v.clone() for k, v in base_model.state_dict().items()}
                log0(f"layer_growth: rebuilt model, optimizers, EMA. New params: {sum(p.numel() for p in base_model.parameters())}")
                if distributed:
                    dist.barrier()  # sync all ranks before resuming training

        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        if ema_state is not None:
            with torch.no_grad():
                for k, v in base_model.state_dict().items():
                    ema_state[k].lerp_(v, 1.0 - args.ema_decay)

        if args.qat_prewarmdown and scale >= args.qat_stop_lr_mul and step % args.qat_every == 0:
            fp16_pats = tuple(p for p in args.int8_keep_float_fp16_name_patterns.split(",") if p)
            with torch.no_grad():
                for name, p in base_model.named_parameters():
                    if not p.is_floating_point() or p.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
                        continue
                    if fp16_pats and any(pat in name for pat in fp16_pats):
                        continue
                    p_q = sim_quant_roundtrip(p.data)
                    p.data.add_(p_q - p.data, alpha=args.qat_strength)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    eval_weight_source = final_eval_weight_source(ema_state is not None)
    log0(f"final_eval_weights:{eval_weight_source}")
    if ema_state is not None:
        log0("Loading EMA weights for final evaluation and serialization")
        base_model.load_state_dict(ema_state)
    else:
        log0("Using live model weights for final evaluation and serialization")
    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    if master_process:
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict(), calibrated=args.calibrated_quant)
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        if args.use_zstd and zstd_mod is not None:
            compressor = zstd_mod.ZstdCompressor(level=args.zstd_level)
            quant_blob = compressor.compress(quant_raw)
            compress_name = get_compression_name(args)
        else:
            quant_blob = zlib.compress(quant_raw, level=9)
            compress_name = get_compression_name(args)
        quant_raw_bytes = len(quant_raw)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+{compress_name}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)"
        )
        log0(f"Total submission size int8+{compress_name}: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    if args.use_zstd and zstd_mod is not None:
        decompressor = zstd_mod.ZstdDecompressor()
        quant_raw_disk = decompressor.decompress(quant_blob_disk)
    else:
        quant_raw_disk = zlib.decompress(quant_blob_disk)
    quant_state = torch.load(io.BytesIO(quant_raw_disk), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    eval_fn = eval_val_sliding if args.eval_stride > 0 else eval_val
    # NOTE: a previous compile-warmup that ran only on rank 0 caused a hard
    # DDP deadlock — eval_fn calls dist.all_reduce internally, and rank 0 alone
    # cannot complete that collective while ranks 1-7 sit at a barrier. Removed.
    # Compile cost (~5-10s) is included in the first eval call's eval_time log.
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_fn(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_zlib_roundtrip eval_stride:{args.eval_stride} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_int8_zlib_roundtrip_exact eval_stride:{args.eval_stride} val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    torch._dynamo.reset()
    torch.cuda.synchronize()
    t_ttt = time.perf_counter()
    ttt_val_loss, ttt_val_bpb = eval_val_ttt_lora(
        args, base_model, rank, world_size, device,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_ttt_lora val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
    )

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
