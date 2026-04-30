#!/usr/bin/env bash
# L0-only control: sliding-window stride 64 with all L1' flags at default.
# Lets us attribute L0's BPB delta separately from L1' (Polar Express + DyT + extended TTT).
#
# Plan: docs/superpowers/plans/2026-04-28-stacked-h100-sprint.md Task 20.
# Pair with h100_sprint.sh; run this first to validate the sliding-window timing budget on H100.

set -euo pipefail

export RUN_ID=h100_l0_only
export NUM_LAYERS=11
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
export MUON_WEIGHT_DECAY=0.10
export FREQ_SKIP_GATING=1
export TRAIN_BATCH_TOKENS=524288
export MLP_MULT_ASYMMETRIC=2,4
export GRAD_CLIP_NORM=0.5
export LAYER_LR_SCALE=0.5
export WARMUP_STEPS=50
export QAT_PREWARMDOWN=1
export QAT_STRENGTH=0.1
export QAT_STOP_LR_MUL=0.8
export QAT_EVERY=10
export USE_ZSTD=1
export ZSTD_LEVEL=22

# L0 only
export EVAL_STRIDE=64

# L1' flags OFF for clean attribution
export USE_POLAR_EXPRESS=0
export USE_DYT_NORM=0

# TTT defaults (rank 8, chunk 256)

export ITERATIONS=20000
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=600

mkdir -p runs
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee runs/h100_l0_only.log
