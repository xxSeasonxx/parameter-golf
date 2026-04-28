"""Common utilities shared by train_gpt.py (PyTorch) and train_gpt_mlx.py (MLX).

This module contains:
- Framework-agnostic code (Hyperparameters base, NS coefficient tables, name
  patterns, int8 constants, pure-Python feature-flag helpers).
- Framework-typed helpers gated by best-effort imports of torch and mlx. If
  torch is available, torch-flavored helpers (suffix _torch) are defined; same
  for mlx. The two train scripts each import only the flavor they need.

Top-level imports of torch/mlx are guarded so that environments missing one
of them still load this module cleanly.
"""

from __future__ import annotations

import os
import uuid

# ----------------------------------------------------------------------------
# Best-effort framework imports.
# ----------------------------------------------------------------------------
try:
    import torch as _torch  # noqa: N816 (private alias for guarded usage)
    _HAS_TORCH = True
except ImportError:  # pragma: no cover — environment-dependent
    _torch = None
    _HAS_TORCH = False

try:
    import mlx.core as _mx  # noqa: N816
    _HAS_MLX = True
except ImportError:  # pragma: no cover
    _mx = None
    _HAS_MLX = False


# ============================================================================
# HYPERPARAMETERS BASE CLASS
# ============================================================================
# Holds the env-var fields that are identical between train_gpt.py and
# train_gpt_mlx.py. Each file's Hyperparameters class subclasses this and adds
# framework-specific fields (e.g., MLX adds mlx_eager_eval, PyTorch adds ttt_*).
#
# All values are class attributes evaluated at import time, mirroring the
# original behavior (env vars must be set before importing the train script).
class Hyperparameters:
    # Data / tokenizer.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Training loop schedule.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    iterations = int(os.environ.get("ITERATIONS", 20_000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model architecture.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    mlp_mult_asymmetric = os.environ.get("MLP_MULT_ASYMMETRIC", "")
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    freq_skip_gating = bool(int(os.environ.get("FREQ_SKIP_GATING", "0")))
    freq_skip_window = int(os.environ.get("FREQ_SKIP_WINDOW", 32))

    # Optimizer hyperparameters (Muon for matrices, Adam for embeddings + scalars).
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.0))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    layer_lr_scale = float(os.environ.get("LAYER_LR_SCALE", 0.0))

    # Quantization-aware training (QAT).
    qat_prewarmdown = bool(int(os.environ.get("QAT_PREWARMDOWN", "0")))
    qat_strength = float(os.environ.get("QAT_STRENGTH", 0.1))
    qat_stop_lr_mul = float(os.environ.get("QAT_STOP_LR_MUL", 0.8))
    qat_every = int(os.environ.get("QAT_EVERY", 10))

    # Deep supervision (auxiliary next-token-prediction losses at intermediate layers).
    deep_supervision = bool(int(os.environ.get("DEEP_SUPERVISION", "0")))
    deep_supervision_alpha = float(os.environ.get("DEEP_SUPERVISION_ALPHA", 0.1))
    deep_supervision_layers = os.environ.get("DEEP_SUPERVISION_LAYERS", "")

    # Sliding window evaluation: each scored token sees (seq_len - eval_stride) of context.
    # 0 disables sliding window. 64 is the competition-best stride.
    eval_stride = int(os.environ.get("EVAL_STRIDE", 0))

    # L1' cheap winners: Polar Express NS coefficients, DyT replacing RMSNorm.
    use_polar_express = bool(int(os.environ.get("USE_POLAR_EXPRESS", "0")))
    use_dyt_norm = bool(int(os.environ.get("USE_DYT_NORM", "0")))


# ============================================================================
# NEWTON-SCHULZ COEFFICIENT TABLES
# ============================================================================
# Polar Express coefficients (Chebyshev-style minimax-optimal, ICML 2025).
# Reference: arxiv 2505.16932; impl: github.com/Dao-AILab/gram-newton-schulz
POLAR_EXPRESS_COEFFS_5 = (
    (8.28721202544396, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.948690853482295, -2.908902115962949, 0.5518191394370137),
    (3.318419657370526, -2.488488024314215, 0.5099255672945051),
    (2.300652019954817, -1.665307437232293, 0.3737887369687766),
)

# Baseline (modded-nanogpt) Newton-Schulz coefficients used per iteration.
BASELINE_NS_COEFFS = (3.4445, -4.7750, 2.0315)


# ============================================================================
# TENSOR NAME PATTERNS (env-overridable)
# ============================================================================
# Tensors whose names match these patterns are treated as "control" / scalar
# parameters: routed to Adam (not Muon), and kept at fp32 even when the model
# is bfloat16. The default list is the union of patterns referenced by the two
# train scripts; train_gpt.py was already using the superset, train_gpt_mlx.py
# previously had a strict subset (missing skip_lo_weight, skip_hi_weight). The
# missing-from-MLX patterns were harmless in practice because MLX's optimizer
# selects skip_lo/hi by exact key match, but unifying here removes the drift.
CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_lo_weight,skip_hi_weight",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)


# ============================================================================
# INT8 QUANTIZATION NUMERICAL CONSTANTS
# ============================================================================
# Float tensors with at most this many elements are kept un-quantized in the
# serialized payload (they are too small for per-row int8 to amortize the
# scale overhead).
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536

# Percentile used for amplitude clipping during int8 quantization. 99.99984 was
# chosen empirically to keep quant error tiny on the long-tail row maxima while
# sacrificing negligible amplitude on outliers.
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


# ============================================================================
# PURE-PYTHON FEATURE-FLAG HELPERS (framework-agnostic)
# ============================================================================
# These helpers route on Hyperparameters fields and produce strings/lists for
# logging. They have no torch or mlx dependency and are shared by both train
# scripts.

def get_ds_tap_layers(args, num_layers: int) -> list | None:
    """Compute deep-supervision tap layer indices from args."""
    if not args.deep_supervision:
        return None
    if args.deep_supervision_layers:
        return [int(x) for x in args.deep_supervision_layers.split(",") if x]
    return list(range(1, num_layers - 1, 2))


def get_compression_name(args, zstd_available: bool | None = None) -> str:
    """Return the compression algorithm label used for the final artifact."""
    if zstd_available is None:
        # Local lazy import: zstd is an optional dep, callers may not have it.
        try:
            import zstandard as _zstd_mod
            zstd_available = _zstd_mod is not None
        except ImportError:
            zstd_available = False
    if args.use_zstd and zstd_available:
        return f"zstd-{args.zstd_level}"
    return "zlib-9"


def describe_feature_flags(args, effective_num_layers: int, zstd_available: bool | None = None) -> list[str]:
    """Render the feature_flags log lines for run startup."""
    lines = [
        "feature_flags: "
        f"ema={'on' if args.ema_decay > 0 else 'off'} "
        f"calibrated_quant={'on' if args.calibrated_quant else 'off'} "
        f"compression={get_compression_name(args, zstd_available=zstd_available)} "
        f"deep_supervision={'on' if args.deep_supervision else 'off'} "
        f"layer_growth={'on' if args.grow_layers_from > 0 else 'off'}"
    ]
    if args.deep_supervision:
        tap_layers = get_ds_tap_layers(args, effective_num_layers)
        lines.append(
            f"deep_supervision:alpha={args.deep_supervision_alpha:.2f} tap_layers={tap_layers}"
        )
    if args.grow_layers_from > 0:
        lines.append(
            "layer_growth:"
            f"start_layers={args.grow_layers_from} "
            f"target_layers={args.num_layers} "
            f"grow_at_frac={args.grow_at_wallclock_frac:.3f}"
        )
    return lines


def final_eval_weight_source(ema_enabled: bool) -> str:
    """Return the label of the weight source used for the final-eval pass."""
    return "ema" if ema_enabled else "live"


# ============================================================================
# QAT-REGULARIZER + CALIBRATED-INT8-QUANT HELPERS (framework-typed)
# ============================================================================
# sim_quant_roundtrip is the per-step "fake quant noise" used by the
# pre-warmdown QAT regularizer (exp_051). quantize_float_tensor_calibrated is
# the MSE-optimal clip-percentile sweep used during the final int8 serializer
# (exp_055 era). Both are OUR additions over upstream.

if _HAS_TORCH:
    def sim_quant_roundtrip_torch(w):
        """Simulate int8 quantize -> dequantize roundtrip in PyTorch ops.

        Per-row for 2D, per-tensor for 1D. Mirrors the actual int8 quantization
        path. Used by the pre-warmdown QAT regularizer to inject quant noise
        every QAT_EVERY steps during training.
        """
        f = w.float()
        qmax = 127.0
        if f.ndim == 2:
            row_max = f.abs().amax(dim=1, keepdim=True).clamp(min=1.0 / qmax)
            scale = row_max / qmax
            q = (f / scale).round().clamp(-qmax, qmax)
            return (q * scale).to(w.dtype)
        amax = f.abs().amax().clamp(min=1.0 / qmax)
        scale = amax / qmax
        q = (f / scale).round().clamp(-qmax, qmax)
        return (q * scale).to(w.dtype)

    def quantize_float_tensor_calibrated_torch(tensor, fallback_per_tensor):
        """Per-row int8 quantization with MSE-optimal clip percentile selection.

        Sweeps a small set of clip quantiles, picks the one with smallest
        squared-error per row. Returns numpy arrays (int8 codes + fp16 scales)
        ready to embed in the serialized state dict. fallback_per_tensor is the
        non-2D fallback function (typically the upstream quantize_float_tensor
        living in train_gpt.py).
        """
        if tensor.ndim != 2:
            return fallback_per_tensor(tensor)
        f32 = tensor.detach().float().cpu()
        candidates = [0.999, 0.9995, 0.9999, 0.99999, 1.0]
        best_q = _torch.zeros_like(f32, dtype=_torch.int8)
        best_scale = _torch.zeros(f32.size(0), dtype=_torch.float32)
        best_mse = _torch.full((f32.size(0),), float('inf'))
        abs_f32 = f32.abs()
        for clip_q in candidates:
            clip_abs = abs_f32.amax(dim=1) if clip_q >= 1.0 else _torch.quantile(abs_f32, clip_q, dim=1)
            scale = (clip_abs / 127.0).clamp(min=1.0 / 127.0)
            clipped = _torch.clamp(f32, -clip_abs[:, None], clip_abs[:, None])
            q = (clipped / scale[:, None]).round().clamp(-127, 127).to(_torch.int8)
            mse = ((f32 - q.float() * scale[:, None]) ** 2).mean(dim=1)
            improved = mse < best_mse
            best_mse[improved] = mse[improved]
            best_q[improved] = q[improved]
            best_scale[improved] = scale[improved]
        return best_q.numpy(), best_scale.to(_torch.float16).numpy()


if _HAS_MLX:
    def sim_quant_roundtrip_mlx(w, qmax_val: float = 127.0):
        """MLX equivalent of sim_quant_roundtrip_torch.

        qmax_val is parameterized because train_gpt_mlx.py defines QUANT_MAX_VAL
        as a module-level constant; passing it through keeps the helper pure.
        """
        f = w.astype(_mx.float32)
        qmax = float(qmax_val)
        if f.ndim == 2:
            row_max = _mx.maximum(_mx.max(_mx.abs(f), axis=1, keepdims=True), 1.0 / qmax)
            scale = row_max / qmax
            q = _mx.clip(_mx.round(f / scale), -qmax, qmax)
            return (q * scale).astype(w.dtype)
        amax = _mx.maximum(_mx.max(_mx.abs(f)), _mx.array(1.0 / qmax))
        scale = amax / qmax
        q = _mx.clip(_mx.round(f / scale), -qmax, qmax)
        return (q * scale).astype(w.dtype)


# ============================================================================
# DyT (Dynamic Tanh) — drop-in replacement for RMSNorm
# ============================================================================
# Reference: Zhu et al., CVPR 2025 (arxiv 2503.10622).
# DyT(x) = gamma * tanh(alpha * x) + beta
# alpha is a scalar learnable parameter (init 0.5 per paper);
# gamma/beta are per-channel learnable vectors.
#
# We define one class per framework so each train script gets a native
# nn.Module. Importers do `from train_gpt_common import DyTTorch as DyT`.

if _HAS_TORCH:
    import torch.nn as _torch_nn

    class DyTTorch(_torch_nn.Module):
        """PyTorch DyT. See module-level docstring for the formula."""

        def __init__(self, dim: int, alpha_init: float = 0.5):
            super().__init__()
            self.alpha = _torch_nn.Parameter(_torch.tensor(alpha_init, dtype=_torch.float32))
            self.gamma = _torch_nn.Parameter(_torch.ones(dim, dtype=_torch.float32))
            self.beta = _torch_nn.Parameter(_torch.zeros(dim, dtype=_torch.float32))

        def forward(self, x):
            return (
                self.gamma.to(dtype=x.dtype)
                * _torch.tanh(self.alpha.to(dtype=x.dtype) * x)
                + self.beta.to(dtype=x.dtype)
            )


if _HAS_MLX:
    import mlx.nn as _mlx_nn

    class DyTMLX(_mlx_nn.Module):
        """MLX DyT. MLX treats any mx.array attribute as a parameter.

        Keep alpha/gamma/beta in fp32: they are tiny scalars/per-channel vectors,
        and the MLX optimizer + quantizer paths preserve fp32 control tensors via
        the CONTROL_TENSOR_NAME_PATTERNS list. Note: train_gpt_mlx.py also has
        a SplitOptimizers leak detector that asserts these scalars are routed
        to the Adam group — if you rename or relocate DyT, verify the leak
        detector still binds the params.
        """

        def __init__(self, dim: int, alpha_init: float = 0.5):
            super().__init__()
            self.alpha = _mx.array(alpha_init, dtype=_mx.float32)
            self.gamma = _mx.ones((dim,), dtype=_mx.float32)
            self.beta = _mx.zeros((dim,), dtype=_mx.float32)

        def __call__(self, x):
            return (
                self.gamma.astype(x.dtype) * _mx.tanh(self.alpha.astype(x.dtype) * x)
                + self.beta.astype(x.dtype)
            )
