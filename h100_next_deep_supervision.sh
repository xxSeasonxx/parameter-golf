#!/usr/bin/env bash
# H100 next experiment: clean 11L baseline plus isolated deep supervision.
# Purpose: test DEEP_SUPERVISION=1 without stacking DyT, Polar Express, EMA, 13L,
# calibrated quant, layer growth, or extended TTT.
#
# Read first:
# - PROJECT.md
# - BASELINE.md
# - NEXT_EXPERIMENT.md

set -euo pipefail

cd "$(dirname "$0")"

export RUN_ID=h100_next_deep_supervision
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
export EMA_DECAY=0
export EXPECTED_TRAIN_SHARDS=195

# Isolated experimental variable.
export DEEP_SUPERVISION=1
export DEEP_SUPERVISION_ALPHA=0.05
export DEEP_SUPERVISION_LAYERS=3,7

# Final evaluation.
export EVAL_STRIDE=64

# Keep L1' flags off. They are not part of this experiment.
export USE_POLAR_EXPRESS=0
export USE_DYT_NORM=0

export ITERATIONS=20000
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=600

mkdir -p runs
RUN_LOG="runs/h100_next_deep_supervision_$(date +%Y%m%d_%H%M%S).log"
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RUN_LOG"
