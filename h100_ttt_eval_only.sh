#!/usr/bin/env bash
# H100 eval-only TTT ablation on an existing corrected checkpoint.
# Purpose: sweep TTT params without spending another 10-minute training run.
#
# Read first:
# - PROJECT.md
# - BASELINE.md
# - NEXT_EXPERIMENT.md

set -euo pipefail

cd "$(dirname "$0")"

export RUN_ID="${RUN_ID:-h100_ttt_eval_only}"
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

# Eval-only checkpoint path. Defaults to the artifact produced by h100_l0_only.sh.
export EVAL_ONLY_CHECKPOINT="${EVAL_ONLY_CHECKPOINT:-./final_model.int8.ptz}"
export EVAL_ONLY_SKIP_ROUNDTRIP="${EVAL_ONLY_SKIP_ROUNDTRIP:-1}"
export EVAL_STRIDE=64

# TTT parameters are intentionally env-overridable for ablation sweeps.
export TTT_LORA_RANK="${TTT_LORA_RANK:-8}"
export TTT_LORA_LR="${TTT_LORA_LR:-0.01}"
export TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-256}"
export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
export TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-64}"

# Explicitly keep killed/deferred paths off for attribution.
export DEEP_SUPERVISION=0
export USE_POLAR_EXPRESS=0
export USE_DYT_NORM=0
export CALIBRATED_QUANT=0
export GROW_LAYERS_FROM=0

export ITERATIONS=20000
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=600

mkdir -p runs
RUN_LOG="runs/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log"
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RUN_LOG"
