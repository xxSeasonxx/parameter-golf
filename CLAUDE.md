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

# Training — see .lab/insights.md for BEST_CONFIG_VARS (batch size, WD, etc.)
RUN_ID=exp_NNN <BEST_CONFIG_VARS> ITERATIONS=2000 VAL_LOSS_EVERY=500 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log

# Post-run analysis
conda run -n openai --no-capture-output python3 analyze.py

# Tests
conda run -n openai --no-capture-output python3 -m pytest test_analyze.py -v
```

## Files

- **`train_gpt_mlx.py`** — Apple Silicon experiment file. Model, optimizer, training loop.
- **`train_gpt.py`** — H100 active development. Trusted baseline result (1.2102 BPB) is at git tag `baseline-3e34098` (commit 3e34098). For A/B comparison, check out the tag in a worktree.
- **`analyze.py`** — Post-run analysis. READ-ONLY.
- **`program.md`** — Experiment loop process (how to run, analyze, decide, iterate).
- **`docs/superpowers/specs/`** — Sprint design docs.
- **`docs/superpowers/plans/`** — Sprint implementation plans.
- **`TODOS.md`** — Deferred research bets per sprint.

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

- For Mac iteration: only modify `train_gpt_mlx.py`
- For H100 sprint: only modify `train_gpt.py`. The trusted baseline is at git tag `baseline-3e34098`.
- Artifact must be < 16,000,000 bytes after int8+zstd
- Be a researcher, not a copier — develop original ideas
- Never run experiments concurrently on Apple Silicon
- Do NOT read `record/` — those are other teams' submissions. We develop our own approaches independently.
