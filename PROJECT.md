# Parameter Golf Project State

This repository has three different kinds of context. Keep them separate.

## Original challenge

`README.md` is the original OpenAI Parameter Golf challenge description: rules, public leaderboard, getting started notes, and submission process.

Do not use README.md as the active runbook for this fork. It is challenge context, not the current experiment plan.

## Trusted baseline

The trusted H100 baseline is documented in `BASELINE.md`.

Short version:

- tag: `baseline-3e34098`
- script: `train_gpt.py`
- run: clean 11L full-shard H100 repro
- post-quant BPB: `1.2316`
- TTT BPB: `1.2102`
- artifact: `14.48MB`

EMA(0.997), the current 13L recipe, SWA, int6 without STE/GPTQ, progressive growth, and old feature-stacked runners are not the baseline.

## Next experiment

The active next experiment is documented in `NEXT_EXPERIMENT.md`.

Short version: run the clean 11L H100 baseline plus isolated deep supervision:

- `DEEP_SUPERVISION=1`
- `DEEP_SUPERVISION_ALPHA=0.05`
- `DEEP_SUPERVISION_LAYERS=3,7`
- `EMA_DECAY=0`

Use `h100_l0_only.sh` as the sliding-window control and `h100_next_deep_supervision.sh` as the next experiment runner.

## Required reading order

1. `PROJECT.md`
2. `BASELINE.md`
3. `NEXT_EXPERIMENT.md`
4. `.lab/insights.md`
5. `.lab/ideas_queue.md`

## Active files

- `train_gpt.py` - active H100 training and evaluation entrypoint
- `train_gpt_common.py` - shared constants and helpers
- `train_gpt_mlx.py` - Apple Silicon experiment path; useful for local smoke checks, not the H100 oracle
- `analyze.py` - post-run parsing and `.lab` archival
- `.lab/insights.md` - validated learnings
- `.lab/ideas_queue.md` - prioritized next ideas
- `.lab/results.tsv` - run result table
- `EXPERIMENT_LOG.md` - historical narrative
- `TODOS.md` - deferred research bets

## Off limits

Do not inspect or edit `records/**` while developing our own approach.
