#!/bin/bash
# =============================================================================
# Run 3: H100-ONLY TECHNIQUES (SWA + ZSTD)
# =============================================================================
# Question: How much does SWA add with 1500+ steps? How much does zstd save?
# Config:   11 layers, int8, zstd, SWA (last 10% of LR, every 5 steps)
# Expected: SWA was killed on Mac (700 steps, no oscillation). H100 has 1500+
#           steps — enough oscillation for averaging to help. Zstd is free.
# =============================================================================
set -euo pipefail

export RUN_ID=h100_run3_swa
export SEED=1337

# Architecture
export NUM_LAYERS=11
export MLP_MULT_ASYMMETRIC=2,4
export FREQ_SKIP_GATING=1

# Optimizer
export MUON_WEIGHT_DECAY=0.10
export GRAD_CLIP_NORM=0.5
export LAYER_LR_SCALE=0.5
export WARMUP_STEPS=50

# QAT (pre-warmdown only)
export QAT_PREWARMDOWN=1
export QAT_STRENGTH=0.1
export QAT_STOP_LR_MUL=0.8
export QAT_EVERY=10

# SWA — average snapshots when lr_mul < 0.1 (last ~10%), every 5 steps
export SWA_START_LR_MUL=0.1
export SWA_EVERY=5

# Training
export TRAIN_BATCH_TOKENS=524288
export ITERATIONS=20000
export VAL_LOSS_EVERY=500
export MAX_WALLCLOCK_SECONDS=600

# Serialization — zstd for better compression, keep int8 (clean comparison vs Run 1)
export COMPRESSION=zstd
export QUANT_BITS=8
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb

torchrun --nproc_per_node=8 train_gpt_h100.py 2>&1 | tee run3_swa.log
