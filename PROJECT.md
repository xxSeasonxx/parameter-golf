# Parameter Golf Project State

This repository has three different kinds of context. Keep them separate.

## Original challenge

`README.md` is the original OpenAI Parameter Golf challenge description: rules, public leaderboard, getting started notes, and submission process.

Do not use README.md as the active runbook for this fork. It is challenge context, not the current experiment plan.

## Trusted baseline

The trusted H100 baseline is documented in `BASELINE.md`.

Short version:

- commit: `87b4a2f`
- script: `train_gpt.py`
- run: corrected clean 11L full-shard H100 control
- post-quant exact BPB: `1.20303259`
- TTT BPB: `1.2162` (harmful; do not use unless retuned)
- artifact: `14.18MB`

EMA(0.997), the current 13L recipe, SWA, int6 without STE/GPTQ, progressive growth, and old feature-stacked runners are not the baseline.

## Next experiment

The active next experiment is documented in `NEXT_EXPERIMENT.md`.

Short version: deep supervision was run and discarded. The next RunPod step is
TTT eval-only ablation on the corrected control checkpoint, because current TTT
regresses from post-quant `1.20303259` to `1.2162`.

Use `EVAL_ONLY_CHECKPOINT=./final_model.int8.ptz` with
`EVAL_ONLY_SKIP_ROUNDTRIP=1` to sweep TTT settings without retraining.

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
