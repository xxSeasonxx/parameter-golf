#!/bin/bash
# =============================================================================
# Phase 1: Quick fix re-run — just Run 1 (11L) and Run 4 (13L calibrated)
# Fixes: fullgraph=False, WD=0.15 for 13L
# =============================================================================
set -eo pipefail

REPO_URL="https://github.com/xxSeasonxx/parameter-golf.git"
BRANCH="lab/mar29"

# Auto-detect repo dir
if [ -f "train_gpt.py" ]; then
    WORKDIR="$(pwd)"
else
    WORKDIR="/workspace/parameter-golf"
fi
RESULTS_DIR="$WORKDIR/results/h100_phase1_$(date +%Y%m%d_%H%M%S)"

cd "$WORKDIR"
if [ -d ".git" ]; then
    echo "Fetching latest..."
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "ERROR: No git repo at $WORKDIR"
    exit 1
fi

pip install -q zstandard 2>/dev/null || true

python3 -c "
import torch
print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')
assert torch.cuda.is_available() and torch.cuda.device_count() >= 8
"

# Download data if needed
ACTUAL_SHARDS=$(find data/datasets/fineweb10B_sp1024 -name "fineweb_train_*.bin" 2>/dev/null | wc -l || echo 0)
if [ "$ACTUAL_SHARDS" -lt 195 ]; then
    echo "Downloading 195 shards..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 195
fi

mkdir -p "$RESULTS_DIR"

# Common env vars
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
export MAX_WALLCLOCK_SECONDS=600
export TRAIN_BATCH_TOKENS=524288

run_experiment() {
    local RUN_NAME="$1"
    shift
    local RUN_DIR="$RESULTS_DIR/$RUN_NAME"
    mkdir -p "$RUN_DIR"

    echo ""
    echo "============================================================"
    echo "RUN: $RUN_NAME — $(date)"
    echo "============================================================"

    for kv in "$@"; do export "$kv"; done
    export RUN_ID="$RUN_NAME"

    local START=$(date +%s)
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RUN_DIR/train.log"
    local ELAPSED=$(( $(date +%s) - START ))

    cp -f final_model.int8.ptz "$RUN_DIR/" 2>/dev/null || true
    cp -f final_model.pt "$RUN_DIR/" 2>/dev/null || true

    local VAL_BPB=$(grep "final_int8_ttt_lora" "$RUN_DIR/train.log" | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    local POST_QUANT=$(grep "final_int8_zlib_roundtrip_exact" "$RUN_DIR/train.log" | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    local ARTIFACT=$(stat -c%s "$RUN_DIR/final_model.int8.ptz" 2>/dev/null || stat -f%z "$RUN_DIR/final_model.int8.ptz" 2>/dev/null || echo "N/A")

    echo "$RUN_NAME | post_quant=$POST_QUANT | ttt=$VAL_BPB | artifact=$ARTIFACT | time=${ELAPSED}s" >> "$RESULTS_DIR/summary.txt"
    echo "--- $RUN_NAME: post_quant=$POST_QUANT ttt=$VAL_BPB artifact=$ARTIFACT ---"

    unset NUM_LAYERS MUON_WEIGHT_DECAY DEEP_SUPERVISION DEEP_SUPERVISION_ALPHA
    unset DEEP_SUPERVISION_LAYERS CALIBRATED_QUANT GROW_LAYERS_FROM GROW_AT_WALLCLOCK_FRAC
}

# ── Run 1: Proven Foundation (11L) ───────────────────────────────────────────
run_experiment "p1_run1_proven_11L" \
    "NUM_LAYERS=11" \
    "MUON_WEIGHT_DECAY=0.10"

# ── Run 2: Calibrated 13L (higher WD for artifact size) ─────────────────────
run_experiment "p1_run2_calibrated_13L" \
    "NUM_LAYERS=13" \
    "MUON_WEIGHT_DECAY=0.15" \
    "CALIBRATED_QUANT=1"

# ── Run 3: Proven 11L + Deep Supervision ─────────────────────────────────────
run_experiment "p1_run3_proven_11L_DS" \
    "NUM_LAYERS=11" \
    "MUON_WEIGHT_DECAY=0.10" \
    "DEEP_SUPERVISION=1" \
    "DEEP_SUPERVISION_ALPHA=0.05" \
    "DEEP_SUPERVISION_LAYERS=3,7"

echo ""
echo "============================================================"
echo "PHASE 1 COMPLETE"
echo "============================================================"
cat "$RESULTS_DIR/summary.txt"
echo "Results: $RESULTS_DIR"
