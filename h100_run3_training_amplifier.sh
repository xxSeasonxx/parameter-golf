#!/bin/bash
# Run 3: Training Amplifier
# Proven foundation (11 layers) plus deep supervision at intermediate layers.
# Deep supervision provides auxiliary gradient signal to early layers, improving
# representational quality without changing the final model size.
set -e

export RUN_ID=h100_run3_training_amplifier
export NUM_LAYERS=11
export MUON_WEIGHT_DECAY=0.10
export DEEP_SUPERVISION=1
export DEEP_SUPERVISION_ALPHA=0.05
export DEEP_SUPERVISION_LAYERS=3,7
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
