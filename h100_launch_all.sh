#!/bin/bash
# =============================================================================
# H100 Competition: Master Launch Script
# =============================================================================
# Runs on RunPod 8xH100. Does everything:
#   1. Clone repo & checkout the right branch
#   2. Install dependencies
#   3. Download full dataset (195 shards)
#   4. Run all 6 competition experiments sequentially
#
# Usage (on RunPod):
#   curl -sL <raw-url-to-this-file> | bash
#   — OR —
#   git clone ... && cd parameter-golf && bash h100_launch_all.sh
#
# Each run saves logs to results/<run_id>/ and the final artifact.
# Total time: ~6 runs × 10 min + data download + overhead ≈ 90 min
# =============================================================================
set -eo pipefail

REPO_URL="https://github.com/xxSeasonxx/parameter-golf.git"
BRANCH="lab/mar29"
# Auto-detect: if we're already in the repo, use current dir. Otherwise use default.
if [ -f "train_gpt.py" ]; then
    WORKDIR="$(pwd)"
else
    WORKDIR="/workspace/parameter-golf"
fi
RESULTS_DIR="$WORKDIR/results/h100_campaign_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# 1. SETUP: Clone, checkout, install
# =============================================================================
echo "============================================================"
echo "STEP 1: Repository setup"
echo "============================================================"

cd "$WORKDIR"
if [ -d ".git" ]; then
    echo "Repo found at $WORKDIR, fetching latest..."
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "ERROR: No git repo at $WORKDIR. Clone first:"
    echo "  git clone $REPO_URL $WORKDIR"
    exit 1
fi

echo "Branch: $(git branch --show-current)"
echo "Commit: $(git log --oneline -1)"

# =============================================================================
# 2. DEPENDENCIES
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 2: Installing dependencies"
echo "============================================================"

pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q zstandard  # for zstd compression

# Verify critical imports
python3 -c "
import torch
import sentencepiece
import numpy
print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA not available!'
assert torch.cuda.device_count() >= 8, f'Need 8 GPUs, found {torch.cuda.device_count()}'
print('All dependencies OK')
"

# =============================================================================
# 3. DOWNLOAD FULL DATASET (195 shards)
# =============================================================================
echo ""
echo "============================================================"
echo "STEP 3: Downloading full dataset (195 shards)"
echo "============================================================"

EXPECTED_SHARDS=195
ACTUAL_SHARDS=$(find data/datasets/fineweb10B_sp1024 -name "fineweb_train_*.bin" 2>/dev/null | wc -l || echo 0)

if [ "$ACTUAL_SHARDS" -ge "$EXPECTED_SHARDS" ]; then
    echo "Dataset already complete: $ACTUAL_SHARDS shards found"
else
    echo "Found $ACTUAL_SHARDS shards, downloading all $EXPECTED_SHARDS..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$EXPECTED_SHARDS"
fi

# Verify dataset
ACTUAL_SHARDS=$(find data/datasets/fineweb10B_sp1024 -name "fineweb_train_*.bin" 2>/dev/null | wc -l || echo 0)
VAL_SHARDS=$(find data/datasets/fineweb10B_sp1024 -name "fineweb_val_*.bin" 2>/dev/null | wc -l || echo 0)
echo "Dataset: $ACTUAL_SHARDS training shards, $VAL_SHARDS validation shards"
if [ "$ACTUAL_SHARDS" -lt "$EXPECTED_SHARDS" ]; then
    echo "WARNING: Only $ACTUAL_SHARDS/$EXPECTED_SHARDS shards downloaded!"
fi

# =============================================================================
# 4. PREPARE RESULTS DIRECTORY
# =============================================================================
mkdir -p "$RESULTS_DIR"
echo ""
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# =============================================================================
# 5. RUN ALL 6 EXPERIMENTS
# =============================================================================

# Common env vars shared by all runs
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
    echo "RUN: $RUN_NAME"
    echo "Started: $(date)"
    echo "============================================================"

    # Set per-run env vars (passed as KEY=VALUE arguments)
    for kv in "$@"; do
        export "$kv"
    done
    export RUN_ID="$RUN_NAME"
    export OUT_DIR="$RUN_DIR"

    # Run training
    local START_TIME=$(date +%s)
    torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RUN_DIR/train.log"
    local END_TIME=$(date +%s)
    local ELAPSED=$(( END_TIME - START_TIME ))

    # Collect artifacts
    cp -f final_model.int8.ptz "$RUN_DIR/" 2>/dev/null || true
    cp -f final_model.pt "$RUN_DIR/" 2>/dev/null || true

    # Extract key metrics from log
    local VAL_BPB=$(grep "final_int8.*val_bpb:" "$RUN_DIR/train.log" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    local TTT_BPB=$(grep "final_int8_ttt.*val_bpb:" "$RUN_DIR/train.log" | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    local ARTIFACT_SIZE=$(stat -f%z "$RUN_DIR/final_model.int8.ptz" 2>/dev/null || stat -c%s "$RUN_DIR/final_model.int8.ptz" 2>/dev/null || echo "N/A")

    echo ""
    echo "--- $RUN_NAME RESULTS ---"
    echo "  val_bpb (int8): $VAL_BPB"
    echo "  val_bpb (TTT):  $TTT_BPB"
    echo "  artifact:       $ARTIFACT_SIZE bytes"
    echo "  wall time:      ${ELAPSED}s"
    echo "  log:            $RUN_DIR/train.log"
    echo ""

    # Write summary line
    echo "$RUN_NAME | val_bpb=$VAL_BPB | ttt_bpb=$TTT_BPB | artifact=$ARTIFACT_SIZE | time=${ELAPSED}s" >> "$RESULTS_DIR/summary.txt"

    # Clean up per-run env overrides
    unset NUM_LAYERS MUON_WEIGHT_DECAY DEEP_SUPERVISION DEEP_SUPERVISION_ALPHA
    unset DEEP_SUPERVISION_LAYERS CALIBRATED_QUANT GROW_LAYERS_FROM GROW_AT_WALLCLOCK_FRAC
}

# ── Run 1: Proven Foundation (SAFE) ──────────────────────────────────────────
run_experiment "run1_proven" \
    "NUM_LAYERS=11" \
    "MUON_WEIGHT_DECAY=0.10"

# ── Run 2: Deep Capacity (13L) ──────────────────────────────────────────────
run_experiment "run2_deep_capacity" \
    "NUM_LAYERS=13" \
    "MUON_WEIGHT_DECAY=0.12"

# ── Run 3: Training Amplifier (deep supervision) ────────────────────────────
run_experiment "run3_training_amplifier" \
    "NUM_LAYERS=11" \
    "MUON_WEIGHT_DECAY=0.10" \
    "DEEP_SUPERVISION=1" \
    "DEEP_SUPERVISION_ALPHA=0.05" \
    "DEEP_SUPERVISION_LAYERS=3,7"

# ── Run 4: Calibrated Compression (13L + calibrated quant) ──────────────────
run_experiment "run4_calibrated" \
    "NUM_LAYERS=13" \
    "MUON_WEIGHT_DECAY=0.12" \
    "CALIBRATED_QUANT=1"

# ── Run 5: Progressive Deep (8L→13L moonshot) ───────────────────────────────
run_experiment "run5_progressive" \
    "NUM_LAYERS=13" \
    "MUON_WEIGHT_DECAY=0.12" \
    "GROW_LAYERS_FROM=8" \
    "GROW_AT_WALLCLOCK_FRAC=0.35" \
    "DEEP_SUPERVISION=1" \
    "DEEP_SUPERVISION_ALPHA=0.1" \
    "DEEP_SUPERVISION_LAYERS=3,5" \
    "CALIBRATED_QUANT=1"

# ── Run 6: Everything, Everywhere, All At Once ──────────────────────────────
run_experiment "run6_everything" \
    "NUM_LAYERS=13" \
    "MUON_WEIGHT_DECAY=0.12" \
    "DEEP_SUPERVISION=1" \
    "DEEP_SUPERVISION_ALPHA=0.05" \
    "DEEP_SUPERVISION_LAYERS=3,5,9,11" \
    "CALIBRATED_QUANT=1"

# =============================================================================
# 6. FINAL SUMMARY
# =============================================================================
echo ""
echo "============================================================"
echo "ALL 6 RUNS COMPLETE"
echo "============================================================"
echo ""
cat "$RESULTS_DIR/summary.txt"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Finished: $(date)"
