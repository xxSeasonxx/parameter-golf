"""Common utilities shared by train_gpt.py (PyTorch) and train_gpt_mlx.py (MLX).

This module contains framework-agnostic code: env-var-driven hyperparameter base class,
Newton-Schulz coefficient tables, control-tensor name patterns, and int8 quantization
constants. It must NOT import torch or mlx.core — both files import from here.
"""

from __future__ import annotations

import os
import uuid


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
