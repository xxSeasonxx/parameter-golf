#!/bin/bash
# Run 6: Everything, Everywhere, All At Once
# Full kitchen-sink: 13 layers, deep supervision at 4 intermediate checkpoints,
# and calibrated quantization. Tests whether all proven techniques combine
# additively or interfere with each other.
set -e

export RUN_ID=h100_run6_everything
export NUM_LAYERS=13
export MUON_WEIGHT_DECAY=0.12
export DEEP_SUPERVISION=1
export DEEP_SUPERVISION_ALPHA=0.05
export DEEP_SUPERVISION_LAYERS=3,5,9,11
export CALIBRATED_QUANT=1
export MLP_MULT_ASYMMETRIC=2,4
export FREQ_SKIP_GATING=1
export GRAD_CLIP_NORM=0.5
export LAYER_LR_SCALE=0.5
export WARMUP_STEPS=50
export QAT_PREWARMDOWN=1
export QAT_STRENGTH=0.1
export QAT_STOP_LR_MUL=0.8
export QAT_EVERY=10
export EMA_DECAY=0.997
export USE_ZSTD=1
export ZSTD_LEVEL=22
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
export ITERATIONS=20000
export VAL_LOSS_EVERY=1000

torchrun --nproc_per_node=8 train_gpt.py
