# CLAUDE.md

## Project Overview

Parameter Golf: train the best language model fitting in **16MB** in **<10 min on 8xH100**. Metric: **bits-per-byte (BPB)** on FineWeb validation. Lower is better.

## Read First

1. `PROJECT.md` - original challenge vs active research state
2. `BASELINE.md` - trusted baseline and killed non-baselines
3. `NEXT_EXPERIMENT.md` - next H100 run and decision rule
4. `.lab/insights.md` - validated learnings
5. `.lab/ideas_queue.md` - prioritized research queue

## Environment

All Python commands **must** use the `openai` conda env:

```bash
conda run -n openai --no-capture-output python -m pytest --version
```

## Key Commands

```bash
# Data (10 shards for local dev)
conda run -n openai --no-capture-output python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# Baseline H100 control
./h100_l0_only.sh

# Next H100 experiment
./h100_next_deep_supervision.sh

# Post-run analysis
conda run -n openai --no-capture-output python analyze.py

# Focused cleanup/repo tests
conda run -n openai --no-capture-output python -m pytest test_analyze.py tests/test_runner_targets.py tests/test_import_safety.py tests/test_eval_sliding.py tests/test_active_project_state.py -q
```

## Files

- `README.md` - original OpenAI challenge context and rules, not the active runbook.
- `PROJECT.md` - current repo orientation.
- `BASELINE.md` - trusted baseline at tag `baseline-3e34098`, TTT BPB `1.2102`.
- `NEXT_EXPERIMENT.md` - current next experiment.
- `train_gpt.py` - H100 active development and active RunPod entrypoint.
- `train_gpt_common.py` - shared constants/helpers for PyTorch and MLX paths.
- `train_gpt_mlx.py` - Apple Silicon experiment file; useful for local smoke checks, not the H100 oracle.
- `analyze.py` - post-run analysis.
- `.lab/insights.md` - validated learnings.
- `.lab/ideas_queue.md` - prioritized ideas.
- `.lab/results.tsv` - per-run result table.
- `EXPERIMENT_LOG.md` - full experiment history.
- `TODOS.md` - deferred research bets.

## Architecture

- Transformer with U-Net skip connections, GQA (8Q/4KV), RoPE, LeakyReLU(0.5)^2 MLP, RMSNorm by default.
- Tied embeddings, vocab 1024, seq_len 1024.
- Muon optimizer for matrix params plus Adam for embeddings and scalars.
- Warmdown-aware Muon weight decay, per-layer LR scaling, pre-warmdown QAT.
- Active H100 artifact path is int8 + zstd-22.
- Legal LoRA TTT is used for final comparison.

## Rules

- For Mac iteration: only modify `train_gpt_mlx.py`.
- For H100 work: only modify `train_gpt.py` unless the plan explicitly says otherwise.
- The trusted baseline is git tag `baseline-3e34098`.
- Active runners are `h100_l0_only.sh` and `h100_next_deep_supervision.sh`.
- Artifact must be < 16,000,000 bytes after int8+zstd.
- Be a researcher, not a copier - develop original ideas.
- Never run experiments concurrently on Apple Silicon.
- Do not inspect or edit `records/**`.
- Do not use deprecated H100 scripts from git history for new runs.
