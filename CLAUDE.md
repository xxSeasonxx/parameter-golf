# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parameter Golf is an OpenAI challenge to train the best language model that fits in a **16MB artifact** while training in **under 10 minutes on 8xH100 GPUs**. The metric is compression on the FineWeb validation set, measured in tokenizer-agnostic **bits-per-byte (BPB)**. Lower BPB is better.

## Environment

All Python commands **must** run inside the `openai` conda environment:
```bash
conda activate openai
# or prefix each command:
conda run -n openai --no-capture-output <command>
```

## Key Commands

### Data Download
```bash
# Full dataset (80 shards, sp1024 tokenizer)
conda run -n openai --no-capture-output python3 data/cached_challenge_fineweb.py --variant sp1024

# Smaller download for local dev (10 shards)
conda run -n openai --no-capture-output python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

### Training (MLX, Mac Apple Silicon) — primary development target
```bash
# Smoke test (~1-2 min)
RUN_ID=exp_001 ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log

# Medium run (~5-10 min)
RUN_ID=exp_001 ITERATIONS=2000 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=500 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

### Training (PyTorch, GPU)
```bash
# Single GPU
torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (competition setup)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Post-Run Analysis
```bash
# Parse run.log, archive to .lab/<commit>/, generate plots and analysis.md
conda run -n openai --no-capture-output python3 analyze.py
```

### Tests
```bash
conda run -n openai --no-capture-output python3 -m pytest test_analyze.py -v
```

## Architecture

### Code Layout
- **`train_gpt_mlx.py`** — MLX training script for Apple Silicon (~1200 lines). The **only file modified** during experiments. Contains model definition, optimizers, data loading, evaluation, quantization, and training loop.
- **`train_gpt.py`** — PyTorch GPU version (~1450 lines). **Read-only** reference — a source of proven techniques to port to MLX.
- **`analyze.py`** — Post-run analysis engine. Parses training logs (both MLX and PyTorch formats), archives artifacts to `.lab/`, generates comparison plots, writes `analysis.md`.
- **`test_analyze.py`** — Unit tests for the log parser.
- **`program.md`** — Autonomous experiment agent instructions. Defines the setup, experiment loop, decision protocol, and run tiers.
- **`data/`** — Dataset download scripts, tokenizer files, and cached binary shards.
- **`records/`** — Submission history. Each record has a README, `submission.json`, code snapshot, and training logs.

### Experiment Tracking (`.lab/`)
```
.lab/
  results.tsv                    # Master log: commit, val_bpb, artifact_bytes, status, description
  insights.md                    # Validated learnings + current best tracking
  ideas_queue.md                 # Prioritized hypothesis queue
  techniques_from_pytorch.md     # Cross-pollination tracker (ported vs not-yet-ported)
  <commit>/                      # Per-run archive
    analysis.md, run.log, train_gpt_mlx.py, metrics.jsonl, summary.json, *.png
```

### Model Architecture (in both train_gpt.py and train_gpt_mlx.py)
- Transformer with U-Net-style learned skip connections between first-half and second-half layers
- Grouped Query Attention (GQA): 8 query heads, 4 KV heads by default
- RoPE positional embeddings, ReLU² MLP activation, RMSNorm
- Tied input/output embeddings
- Default: 9 layers, 512 dim, vocab size 1024
- **Current best config**: 10 layers, 512 dim, Muon WD=0.02, FP16 tok_emb (val_bpb=1.8291 on Apple Silicon)

### Configuration
All hyperparameters are set via **environment variables** (no CLI args). Key ones:
- Model: `NUM_LAYERS`, `MODEL_DIM`, `NUM_HEADS`, `NUM_KV_HEADS`, `VOCAB_SIZE`, `MLP_MULT`
- Training: `ITERATIONS`, `TRAIN_BATCH_TOKENS`, `TRAIN_SEQ_LEN`, `MAX_WALLCLOCK_SECONDS`
- Optimizer: `EMBED_LR`, `MATRIX_LR`, `SCALAR_LR`, `MUON_MOMENTUM`, `MUON_WEIGHT_DECAY`
- Evaluation: `EVAL_STRIDE` (sliding window, 0=off, 64=competition standard)
- Quantization: `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS` (comma-separated tensor names to keep as FP16)
- LoRA TTT: `TTT_LORA_RANK`, `TTT_LORA_LR`, `TTT_CHUNK_SIZE`, `TTT_EVAL_SEQ_LEN`
- MLX-specific: `MLX_MAX_MICROBATCH_TOKENS`, `MLX_EAGER_EVAL`, `GRAD_ACCUM_STEPS`

### Key Technical Details
- **Muon optimizer** for matrix parameters (Newton-Schulz orthogonalization), Adam for embeddings and scalars
- **Muon weight decay** (`MUON_WEIGHT_DECAY`, default 0.0): Decoupled WD on matrix params. 0.02 validated as beneficial.
- **Post-training quantization**: int8 per-row quantization + zlib compression to fit 16MB limit. Control tensors stay FP32. Embeddings can be kept as FP16 via `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb`.
- **Sliding window evaluation** (`EVAL_STRIDE`): Overlapping context windows for better eval. Too slow on Apple Silicon (~50min), use on 8xH100 only.
- **Tokenizer-agnostic BPB evaluation**: lookup tables mapping token IDs to byte lengths
- **Test-Time Training (LoRA)**: per-document LoRA adaptation during evaluation — test-time compute is free
- **Wallclock-based LR scheduling**: warmdown computed from remaining wall time, not iteration count. Warmdown=1200 is validated as optimal.
- **MLX lazy evaluation**: graphs build up until `mx.eval()` — use `MLX_MAX_MICROBATCH_TOKENS` and `MLX_EAGER_EVAL` to control memory pressure on Apple Silicon

### Key Documentation
- **`EXPERIMENT_LOG.md`** — Full narrative of all past experiments, learnings, failures, and code changes. **Read this first in any new session.**
- **`program.md`** — Autonomous experiment agent instructions with research philosophy, validation protocol, and experiment loop.
- **`.lab/insights.md`** — Validated technical knowledge base and current best tracking.
- **`.lab/ideas_queue.md`** — Prioritized research queue with original ideas (not just parameter sweeps).

### Autonomous Experiment Workflow
When running as an autonomous agent (see `program.md`):
1. **Read `EXPERIMENT_LOG.md` and `.lab/insights.md` first** — prevents repeating failed experiments
2. Only modify `train_gpt_mlx.py` — everything else is read-only
3. **Be a researcher, not a copier** — develop original ideas, use competition as inspiration only
4. Use tiered runs: smoke test (200 iters) → medium (2000 iters) → full (10000+ iters)
5. **Validate novel code** before running: syntax check → shape/value sanity → 5-step micro-run (see program.md section 6.5)
6. After each run: `python3 analyze.py` to archive and generate analysis
7. Compare against all-time best in `.lab/insights.md`, not previous run
8. Keep improvements, revert regressions by restoring from `.lab/<best_commit>/train_gpt_mlx.py`
