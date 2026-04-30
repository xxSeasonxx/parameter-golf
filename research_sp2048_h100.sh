#!/usr/bin/env bash
# H100 SP2048 research run. This is intentionally not an active baseline runner.

set -euo pipefail

cd "$(dirname "$0")"

export RUN_ID="${RUN_ID:-h100_sp2048}"
export DATA_PATH="${DATA_PATH:-./data/research_tokenizers/datasets/fineweb10B_sp2048}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/research_tokenizers/tokenizers/fineweb_2048_bpe.model}"
export VOCAB_SIZE=2048

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

dataset_dir="${DATA_PATH%/}"
if [ ! -d "$dataset_dir" ]; then
  echo "missing DATA_PATH: $dataset_dir" >&2
  echo "build it with data/download_hf_docs_and_tokenize.py and data/tokenizer_specs_sp2048.json" >&2
  exit 1
fi
actual_train_shards="$(find "$dataset_dir" -maxdepth 1 -name 'fineweb_train_*.bin' | wc -l | tr -d ' ')"
export EXPECTED_TRAIN_SHARDS="${EXPECTED_TRAIN_SHARDS:-$actual_train_shards}"

export EVAL_STRIDE=64
export TTT_LORA_RANK="${TTT_LORA_RANK:-8}"
export TTT_LORA_LR="${TTT_LORA_LR:-0.003}"
export TTT_CHUNK_SIZE="${TTT_CHUNK_SIZE:-128}"
export TTT_EVAL_SEQ_LEN="${TTT_EVAL_SEQ_LEN:-1024}"
export TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-64}"
export TTT_EPOCHS="${TTT_EPOCHS:-1}"

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
