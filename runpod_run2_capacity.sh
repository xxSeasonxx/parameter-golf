#!/bin/bash
# =============================================================================
# Run 2: COMPRESSION FOR CAPACITY
# =============================================================================
# Question: Does bigger model + lossy quant beat smaller + lossless?
# Config:   13 layers, int6, zstd — uses compression headroom for more depth
# Expected: Artifact ~5-6MB. If capacity wins over int6 gap → push this path
# =============================================================================
set -euo pipefail

export RUN_ID=h100_run2_capacity
export SEED=1337

# Architecture — 13 layers fits in ~6MB artifact with int6+zstd
export NUM_LAYERS=13
export MLP_MULT_ASYMMETRIC=2,4
export FREQ_SKIP_GATING=1

# Optimizer
export MUON_WEIGHT_DECAY=0.10
export GRAD_CLIP_NORM=0.5
export LAYER_LR_SCALE=0.5
export WARMUP_STEPS=50

# QAT (pre-warmdown, targeting int6 quantization)
export QAT_PREWARMDOWN=1
export QAT_STRENGTH=0.1
export QAT_STOP_LR_MUL=0.8
export QAT_EVERY=10

# Training
export TRAIN_BATCH_TOKENS=524288
export ITERATIONS=20000
export VAL_LOSS_EVERY=500
export MAX_WALLCLOCK_SECONDS=600

# Serialization — int6 + zstd for maximum compression
export COMPRESSION=zstd
export QUANT_BITS=6
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb

torchrun --nproc_per_node=8 train_gpt_h100.py 2>&1 | tee run2_capacity.log
