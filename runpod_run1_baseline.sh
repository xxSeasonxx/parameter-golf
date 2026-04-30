#!/bin/bash
# =============================================================================
# Run 1: FAITHFUL BASELINE PORT
# =============================================================================
# Question: What's our floor on H100? Do Mac findings transfer?
# Config:   11 layers, int8, zlib — identical to Mac best + extra layer
# Expected: val_bpb ~1.18-1.20
# =============================================================================
set -euo pipefail

export RUN_ID=h100_run1_baseline
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

# Training
export TRAIN_BATCH_TOKENS=524288
export ITERATIONS=20000
export VAL_LOSS_EVERY=500
export MAX_WALLCLOCK_SECONDS=600

# Serialization
export USE_ZSTD=0
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb

torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee run1_baseline.log
