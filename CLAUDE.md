# CLAUDE.md

## Project Overview

Parameter Golf: train the best language model fitting in **16MB** in **<10 min on 8xH100**. Metric: **bits-per-byte (BPB)** on FineWeb validation. Lower is better.

## Environment

All Python commands **must** use the `openai` conda env:
```bash
conda run -n openai --no-capture-output <command>
```

## Key Commands

```bash
# Data (10 shards for local dev)
conda run -n openai --no-capture-output python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Training — see .lab/insights.md for current best env vars
RUN_ID=exp_NNN NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.1 \
  ITERATIONS=2000 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=500 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log

# Post-run analysis
conda run -n openai --no-capture-output python3 analyze.py

# Tests
conda run -n openai --no-capture-output python3 -m pytest test_analyze.py -v
```

## Files

- **`train_gpt_mlx.py`** — The ONLY file modified during experiments. Model, optimizer, training loop.
- **`train_gpt.py`** — PyTorch reference. READ-ONLY.
- **`analyze.py`** — Post-run analysis. READ-ONLY.
- **`program.md`** — Experiment loop process (how to run, analyze, decide, iterate).

## State (single source of truth for each)

| What | Where | Read when |
|------|-------|-----------|
| Current best config + validated learnings | `.lab/insights.md` | Every session start, every decision |
| What to try next | `.lab/ideas_queue.md` | Before each experiment |
| Full experiment history + narratives | `EXPERIMENT_LOG.md` | Session start, deep reference |
| Per-run results table | `.lab/results.tsv` | Comparing runs |
| Per-run detailed metrics | `.lab/<commit>/` | Deep investigation |

## Architecture (quick reference)

- Transformer with U-Net skip connections, GQA (8Q/4KV), RoPE, ReLU² MLP, RMSNorm
- Tied embeddings, vocab 1024, seq_len 1024
- Muon optimizer (matrix params) + Adam (embeddings, scalars)
- Wallclock-based LR warmdown, int8+zlib quantization
- All config via env vars (see `train_gpt_mlx.py` Hyperparameters class)

## Rules

- Only modify `train_gpt_mlx.py`
- Artifact must be < 16,000,000 bytes after int8+zlib
- Be a researcher, not a copier — develop original ideas
- Never run experiments concurrently on Apple Silicon
- Do NOT read `record/` — those are other teams' submissions. We develop our own approaches independently.
