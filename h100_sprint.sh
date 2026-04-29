#!/usr/bin/env bash
# H100 sprint: L0 (sliding-window stride 64) + L1' (Polar Express + DyT + extended TTT).
# Single run, ~$15 RunPod cost. Outcome decides what comes next per the plan's decision gate.
#
# Plan: docs/superpowers/plans/2026-04-28-stacked-h100-sprint.md
# Spec: docs/superpowers/specs/2026-04-28-stacked-h100-sprint-design.md
# Baseline: git tag baseline-3e34098 (1.2102 BPB).

set -euo pipefail

export RUN_ID=h100_sprint_v1
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

# L0
export EVAL_STRIDE=64

# L1' cheap winners
export USE_POLAR_EXPRESS=1
export USE_DYT_NORM=1

# Extended TTT (rank 16 for more adaptation capacity, smaller chunks for more passes)
export TTT_LORA_RANK=16
export TTT_CHUNK_SIZE=128

export ITERATIONS=20000
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=600

mkdir -p runs
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee runs/h100_sprint_v1.log
