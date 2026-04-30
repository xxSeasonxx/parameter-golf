# AGENTS.md

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

## Active Commands

```bash
# Baseline H100 control
./h100_l0_only.sh

# Next H100 experiment
./h100_next_deep_supervision.sh

# Post-run analysis
conda run -n openai --no-capture-output python analyze.py

# Focused tests
conda run -n openai --no-capture-output python -m pytest test_analyze.py tests/test_runner_targets.py tests/test_import_safety.py tests/test_eval_sliding.py tests/test_active_project_state.py -q
```

## Rules

- `README.md` is original challenge context, not the active runbook.
- The trusted baseline is `baseline-3e34098`, TTT BPB `1.2102`.
- The next run is isolated deep supervision on clean 11L, with `EMA_DECAY=0`.
- Active runners are `h100_l0_only.sh` and `h100_next_deep_supervision.sh`.
- Do not enable EMA, 13L capacity, calibrated quant, progressive growth, DyT, or Polar Express unless a new plan explicitly reopens them.
- Do not inspect or edit `records/**`.
- Do not change model math during repository cleanup.
