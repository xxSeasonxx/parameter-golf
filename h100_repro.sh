#!/bin/bash
# =============================================================================
# Minimal H100 diagnostic cycle
# 1. Clean 11L repro, no EMA
# 2. 13L int8+zstd capacity, no EMA
# 3. 11L + EMA only
# =============================================================================
set -eo pipefail

BRANCH="${BRANCH:-lab/mar29}"
RUN_SET="${RUN_SET:-all}"  # clean_11l | capacity_13l | ema_11l | all

if [ -f "train_gpt.py" ]; then
    WORKDIR="$(pwd)"
else
    WORKDIR="/workspace/parameter-golf"
fi
RESULTS_DIR="${RESULTS_DIR:-$WORKDIR/results/h100_repro_$(date +%Y%m%d_%H%M%S)}"

cd "$WORKDIR"
if [ ! -d ".git" ]; then
    echo "ERROR: No git repo at $WORKDIR"
    exit 1
fi

echo "Fetching latest branch state..."
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH"

python3 -m pip install -q zstandard 2>/dev/null || true

python3 - <<'PY'
import torch
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
assert torch.cuda.is_available() and torch.cuda.device_count() >= 8
PY

ACTUAL_SHARDS=$(python3 - <<'PY'
from pathlib import Path
print(len(list(Path("data/datasets/fineweb10B_sp1024").glob("fineweb_train_*.bin"))))
PY
)
if [ "$ACTUAL_SHARDS" -lt 195 ]; then
    echo "Downloading 195 shards..."
    python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 195
fi

mkdir -p "$RESULTS_DIR"

# Common validated env vars
export MLP_MULT_ASYMMETRIC=2,4
export FREQ_SKIP_GATING=1
export GRAD_CLIP_NORM=0.5
export LAYER_LR_SCALE=0.5
export WARMUP_STEPS=50
export QAT_PREWARMDOWN=1
export QAT_STRENGTH=0.1
export QAT_STOP_LR_MUL=0.8
export QAT_EVERY=10
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

    local LOCAL_ENV=()
    for kv in "$@"; do
        LOCAL_ENV+=("$kv")
    done
    export RUN_ID="$RUN_NAME"

    local START
    START=$(date +%s)
    env "${LOCAL_ENV[@]}" torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee "$RUN_DIR/train.log"
    local ELAPSED=$(( $(date +%s) - START ))

    cp -f final_model.int8.ptz "$RUN_DIR/" 2>/dev/null || true
    cp -f final_model.pt "$RUN_DIR/" 2>/dev/null || true

    local RAW_BPB
    RAW_BPB=$(python3 - "$RUN_DIR/train.log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read().splitlines()
vals = []
for line in text:
    m = re.match(r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:(\d+)ms", line.strip())
    if m:
        vals.append(float(m.group(4)))
print(vals[-1] if vals else "N/A")
PY
)
    local POST_QUANT
    POST_QUANT=$(python3 - "$RUN_DIR/train.log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"final_int8_zlib_roundtrip_exact\s+val_loss:[\d.]+\s+val_bpb:([\d.]+)", text)
print(m.group(1) if m else "N/A")
PY
)
    local TTT_BPB
    TTT_BPB=$(python3 - "$RUN_DIR/train.log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"final_int8_ttt_lora\s+val_loss:[\d.]+\s+val_bpb:([\d.]+)", text)
print(m.group(1) if m else "N/A")
PY
)
    local STEP_INFO
    STEP_INFO=$(python3 - "$RUN_DIR/train.log" <<'PY'
import re, sys
text = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"stopping_early: wallclock_cap train_time:(\d+)ms step:(\d+)/(\d+)", text)
if m:
    print(f"steps={m.group(2)}/{m.group(3)} train_time_ms={m.group(1)}")
else:
    print("steps=N/A train_time_ms=N/A")
PY
)
    local ARTIFACT
    ARTIFACT=$(stat -c%s "$RUN_DIR/final_model.int8.ptz" 2>/dev/null || stat -f%z "$RUN_DIR/final_model.int8.ptz" 2>/dev/null || echo "N/A")

    echo "$RUN_NAME | raw=$RAW_BPB | post_quant=$POST_QUANT | ttt=$TTT_BPB | artifact=$ARTIFACT | $STEP_INFO | time=${ELAPSED}s" >> "$RESULTS_DIR/summary.txt"
    echo "--- $RUN_NAME: raw=$RAW_BPB post_quant=$POST_QUANT ttt=$TTT_BPB artifact=$ARTIFACT $STEP_INFO ---"
}

if [ "$RUN_SET" = "clean_11l" ] || [ "$RUN_SET" = "all" ]; then
    run_experiment "clean_11l_no_ema" \
        "NUM_LAYERS=11" \
        "MUON_WEIGHT_DECAY=0.10" \
        "EMA_DECAY=0"
fi

if [ "$RUN_SET" = "capacity_13l" ] || [ "$RUN_SET" = "all" ]; then
    run_experiment "capacity_13l_no_ema" \
        "NUM_LAYERS=13" \
        "MUON_WEIGHT_DECAY=0.15" \
        "CALIBRATED_QUANT=1" \
        "EMA_DECAY=0"
fi

if [ "$RUN_SET" = "ema_11l" ] || [ "$RUN_SET" = "all" ]; then
    run_experiment "ema_11l" \
        "NUM_LAYERS=11" \
        "MUON_WEIGHT_DECAY=0.10" \
        "EMA_DECAY=0.997"
fi

echo ""
echo "============================================================"
echo "H100 REPRO COMPLETE"
echo "============================================================"
cat "$RESULTS_DIR/summary.txt"
echo "Results: $RESULTS_DIR"
