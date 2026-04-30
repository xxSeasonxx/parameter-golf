# Project Prune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prune the active repository so future work is clear on the original challenge, the trusted baseline, and the next H100 experiment.

**Architecture:** Add active-state guard tests first, then create canonical orientation docs and active runners, then remove stale tracked files, then move ignored bulky artifacts to an external archive. The cleanup is repo organization only; `train_gpt.py` model math, optimizer math, quantization math, and BPB calculation are not changed.

**Tech Stack:** Python 3.12, pytest, shell scripts, git, conda `openai` env.

**Spec:** [docs/superpowers/specs/2026-04-30-project-prune-design.md](../specs/2026-04-30-project-prune-design.md)

---

## File Structure

Create:

- `PROJECT.md` - root orientation: original challenge vs active research state.
- `BASELINE.md` - trusted baseline identity, metrics, and killed non-baselines.
- `NEXT_EXPERIMENT.md` - exact next H100 experiment and decision rule.
- `h100_next_deep_supervision.sh` - single next-experiment runner.
- `tests/test_active_project_state.py` - regression tests for docs/runners/prune boundary.

Modify:

- `h100_l0_only.sh` - keep as the baseline-control runner and make `EMA_DECAY=0` explicit.
- `CLAUDE.md` - point future agents to `PROJECT.md`, `BASELINE.md`, and `NEXT_EXPERIMENT.md`; update stale architecture/compression wording.
- `AGENTS.md` - add repo-local Codex guidance matching `CLAUDE.md`; track this file so agent instructions are not untracked local state.
- `.gitignore` - ignore local tool state such as `.codex/` and `.firecrawl/`.
- `EXPERIMENT_LOG.md` - correct the known stale top-of-file H100 best line.
- `.lab/insights.md` - replace stale H100 porting/next-session text with the completed diagnostic state and next experiment.
- `.lab/ideas_queue.md` - correct stale `1.2087` target wording in the older sprint section.

Delete from active tree:

- stale runners: `h100_launch_all.sh`, `h100_phase1.sh`, `h100_repro.sh`, `h100_run1_proven.sh`, `h100_run2_deep_capacity.sh`, `h100_run3_training_amplifier.sh`, `h100_run4_calibrated.sh`, `h100_run5_progressive.sh`, `h100_run6_everything.sh`, `h100_sprint.sh`, `runpod_run1_baseline.sh`, `runpod_run2_capacity.sh`, `runpod_run3_swa.sh`
- deprecated code: `train_gpt_h100.py`
- stale/generated docs: `TRAINING_WALKTHROUGH.md`, `TRAINING_WALKTHROUGH.pdf`, old `docs/superpowers/**` plans/specs listed in Task 5
- stale helpers/tests: `experiment_results.xlsx`, `generate_results_excel.py`, `test_h100_smoke.py`

Archive outside repo:

- `logs/`
- `results/`
- `runs/`
- generated root run outputs such as `final_model.*`, `run.log`, `run_*.log`, `*.ptz`, `*.npz`
- `.firecrawl/`

Do not touch:

- `records/**`
- `data/datasets/**`
- `data/tokenizers/**`
- `.lab/e8addc5/**` and other `.lab/` hash-named archived analysis directories

---

### Task 1: Add Active-State Guard Tests

**Files:**
- Create: `tests/test_active_project_state.py`

- [ ] **Step 1: Create the failing test file**

Add `tests/test_active_project_state.py` with this exact content:

```python
"""Regression tests for active project orientation after aggressive prune."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RUNNERS = {"h100_l0_only.sh", "h100_next_deep_supervision.sh"}


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_orientation_docs_exist_and_name_the_three_project_states():
    expected_markers = {
        "PROJECT.md": [
            "Original challenge",
            "Trusted baseline",
            "Next experiment",
            "Do not use README.md as the active runbook",
        ],
        "BASELINE.md": [
            "baseline-3e34098",
            "TTT BPB: 1.2102",
            "Post-quant BPB: 1.2316",
            "EMA(0.997) is not the baseline",
        ],
        "NEXT_EXPERIMENT.md": [
            "H100 next experiment",
            "DEEP_SUPERVISION=1",
            "DEEP_SUPERVISION_ALPHA=0.05",
            "DEEP_SUPERVISION_LAYERS=3,7",
            "EMA_DECAY=0",
            "1.2102",
        ],
    }

    for rel_path, markers in expected_markers.items():
        path = ROOT / rel_path
        assert path.exists(), f"missing {rel_path}"
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            assert marker in text, f"{rel_path} missing marker: {marker}"


def test_only_active_h100_runners_remain():
    root_runner_names = {p.name for p in ROOT.glob("h100_*.sh")}
    root_runner_names |= {p.name for p in ROOT.glob("runpod_*.sh")}
    assert root_runner_names == ACTIVE_RUNNERS


def test_active_runners_do_not_enable_killed_or_deprecated_paths():
    banned_snippets = [
        "train_gpt_h100.py",
        "EMA_DECAY=0.997",
        "GROW_LAYERS_FROM",
        "NUM_LAYERS=13",
        "CALIBRATED_QUANT=1",
        "USE_DYT_NORM=1",
        "USE_POLAR_EXPRESS=1",
    ]

    for runner in ACTIVE_RUNNERS:
        text = _read(runner)
        for snippet in banned_snippets:
            assert snippet not in text, f"{runner} contains banned snippet {snippet!r}"


def test_baseline_runner_is_l0_only_control():
    text = _read("h100_l0_only.sh")
    required = [
        "RUN_ID=h100_l0_only",
        "NUM_LAYERS=11",
        "EMA_DECAY=0",
        "EVAL_STRIDE=64",
        "USE_POLAR_EXPRESS=0",
        "USE_DYT_NORM=0",
        "VAL_LOSS_EVERY=0",
    ]
    for snippet in required:
        assert snippet in text
    assert "DEEP_SUPERVISION=1" not in text


def test_next_experiment_runner_is_isolated_deep_supervision():
    text = _read("h100_next_deep_supervision.sh")
    required = [
        "RUN_ID=h100_next_deep_supervision",
        "NUM_LAYERS=11",
        "EMA_DECAY=0",
        "DEEP_SUPERVISION=1",
        "DEEP_SUPERVISION_ALPHA=0.05",
        "DEEP_SUPERVISION_LAYERS=3,7",
        "EVAL_STRIDE=64",
        "USE_ZSTD=1",
        "VAL_LOSS_EVERY=0",
    ]
    for snippet in required:
        assert snippet in text
```

- [ ] **Step 2: Run the new test to confirm it fails before implementation**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_active_project_state.py -q
```

Expected: FAIL because `PROJECT.md`, `BASELINE.md`, `NEXT_EXPERIMENT.md`, and `h100_next_deep_supervision.sh` do not exist yet and stale runners still remain.

- [ ] **Step 3: Commit is delayed**

Do not commit yet. This test should be committed together with the implementation that makes it pass.

---

### Task 2: Create Canonical Orientation Docs

**Files:**
- Create: `PROJECT.md`
- Create: `BASELINE.md`
- Create: `NEXT_EXPERIMENT.md`

- [ ] **Step 1: Create `PROJECT.md`**

Add `PROJECT.md` with this exact content:

```markdown
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
```

- [ ] **Step 2: Create `BASELINE.md`**

Add `BASELINE.md` with this exact content:

```markdown
# Trusted Baseline

## Identity

- Git tag: `baseline-3e34098`
- Commit: `3e34098`
- Script: `train_gpt.py`
- Run name: `clean_11l_no_ema`
- Dataset: full FineWeb SP1024 training set, `195/195` train shards
- Hardware target: 8xH100

## Result

| Metric | Value |
|---|---:|
| Steps | `8007/20000` |
| Step time | `74.95ms/step` |
| Raw BPB | `1.2282` |
| Post-quant BPB | `1.2316` |
| TTT BPB | `1.2102` |
| Artifact | `14,479,660 bytes` |

This is the comparison point for the next round.

## Baseline Stack

The baseline keeps our validated core stack:

- 11 transformer layers
- SP1024 tokenizer
- tied embeddings with `tok_emb` kept fp16 in serialization
- GQA with 8 query heads and 4 KV heads
- U-Net skip connections
- frequency-decomposed skip gating
- asymmetric MLP, encoder `2x`, decoder `4x`
- LeakyReLU(0.5)^2 MLP activation
- Muon + Adam optimizers
- warmdown-aware Muon weight decay
- per-layer Muon LR scaling
- pre-warmdown int8 QAT regularizer
- zstd-22 final artifact compression
- legal LoRA TTT evaluation path

## Not Baseline

- EMA(0.997) is not the baseline. It produced normal raw BPB but collapsed after final EMA load to post-quant `1.3724` and TTT `1.3035`.
- The current 13L int8+zstd+calibrated recipe is not the baseline. It reached TTT `1.2280`, worse than clean 11L.
- SWA is not the baseline. It regressed on Mac and on H100.
- Int6 without STE/GPTQ is not the baseline. Its H100 quantization gap was too large.
- Progressive layer growth is not the baseline. It failed local screening and should not be promoted without a new reason.

## Reproducing The Baseline Control

Use `h100_l0_only.sh` for the active control path. It uses the clean 11L baseline stack with stride-64 final eval and explicitly disables EMA, DyT, and Polar Express.
```

- [ ] **Step 3: Create `NEXT_EXPERIMENT.md`**

Add `NEXT_EXPERIMENT.md` with this exact content:

```markdown
# H100 Next Experiment

## Goal

Test whether deep supervision improves the trusted clean 11L H100 baseline without stacking unrelated features.

## Baseline

Compare against `baseline-3e34098` / `clean_11l_no_ema`:

- post-quant BPB: `1.2316`
- TTT BPB: `1.2102`
- artifact: `14.48MB`

## Experiment

Run `h100_next_deep_supervision.sh`.

This starts from the clean 11L baseline stack and adds only:

```bash
DEEP_SUPERVISION=1
DEEP_SUPERVISION_ALPHA=0.05
DEEP_SUPERVISION_LAYERS=3,7
```

It keeps:

```bash
EMA_DECAY=0
USE_ZSTD=1
ZSTD_LEVEL=22
EVAL_STRIDE=64
```

It does not enable DyT, Polar Express, EMA, 13L capacity, calibrated quant, layer growth, or extended TTT.

## Control

Run `h100_l0_only.sh` first if the stride-64 timing budget has not been verified on the current RunPod image.

If stride-64 eval exceeds the evaluation budget, update this file before launching the deep-supervision run. Do not silently change the runner.

## Decision Rule

Promote deep supervision only if:

- TTT BPB improves by at least `0.005` versus `1.2102`, and
- training plus final eval stay inside the challenge budget, and
- artifact remains under `16,000,000` bytes.

Kill or revise deep supervision if:

- TTT BPB regresses, or
- eval exceeds the budget, or
- artifact exceeds `16,000,000` bytes.

Treat the result as neutral if:

- TTT BPB improves by less than `0.005`.

If neutral, the next round should focus on TTT evaluation improvements, not EMA, SWA, current 13L capacity, or progressive growth.
```

- [ ] **Step 4: Do not commit yet**

These docs should be committed after the runner and prune changes make `tests/test_active_project_state.py` pass.

---

### Task 3: Create Active Runners

**Files:**
- Modify: `h100_l0_only.sh`
- Create: `h100_next_deep_supervision.sh`

- [ ] **Step 1: Replace `h100_l0_only.sh` with the active control runner**

Replace `h100_l0_only.sh` with this exact content:

```bash
#!/usr/bin/env bash
# H100 L0 control: clean 11L baseline stack with stride-64 final eval.
# Purpose: measure sliding-window timing and score without deep supervision or L1' flags.
#
# Read first:
# - PROJECT.md
# - BASELINE.md
# - NEXT_EXPERIMENT.md

set -euo pipefail

export RUN_ID=h100_l0_only
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

# Final evaluation.
export EVAL_STRIDE=64

# Explicitly keep L1' flags off for attribution.
export USE_POLAR_EXPRESS=0
export USE_DYT_NORM=0

export ITERATIONS=20000
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=600

mkdir -p runs
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee runs/h100_l0_only.log
```

- [ ] **Step 2: Create `h100_next_deep_supervision.sh`**

Add `h100_next_deep_supervision.sh` with this exact content:

```bash
#!/usr/bin/env bash
# H100 next experiment: clean 11L baseline plus isolated deep supervision.
# Purpose: test DEEP_SUPERVISION=1 without stacking DyT, Polar Express, EMA, 13L,
# calibrated quant, layer growth, or extended TTT.
#
# Read first:
# - PROJECT.md
# - BASELINE.md
# - NEXT_EXPERIMENT.md

set -euo pipefail

export RUN_ID=h100_next_deep_supervision
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

# Isolated experimental variable.
export DEEP_SUPERVISION=1
export DEEP_SUPERVISION_ALPHA=0.05
export DEEP_SUPERVISION_LAYERS=3,7

# Final evaluation.
export EVAL_STRIDE=64

# Keep L1' flags off. They are not part of this experiment.
export USE_POLAR_EXPRESS=0
export USE_DYT_NORM=0

export ITERATIONS=20000
export VAL_LOSS_EVERY=0
export MAX_WALLCLOCK_SECONDS=600

mkdir -p runs
torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee runs/h100_next_deep_supervision.log
```

- [ ] **Step 3: Make the new runner executable**

Run:

```bash
chmod +x h100_l0_only.sh h100_next_deep_supervision.sh
```

Expected: command exits 0.

- [ ] **Step 4: Do not commit yet**

The runner set test will still fail until stale runners are removed in Task 5.

---

### Task 4: Update Agent Docs And Active Summaries

**Files:**
- Modify: `CLAUDE.md`
- Add/modify: `AGENTS.md`
- Modify: `.gitignore`
- Modify: `EXPERIMENT_LOG.md`
- Modify: `.lab/insights.md`
- Modify: `.lab/ideas_queue.md`

- [ ] **Step 1: Replace `CLAUDE.md` with current project guidance**

Replace `CLAUDE.md` with this exact content:

```markdown
# CLAUDE.md

## Project Overview

Parameter Golf: train the best language model fitting in **16MB** in **<10 min on 8xH100**. Metric: **bits-per-byte (BPB)** on FineWeb validation. Lower is better.

## Read First

1. `PROJECT.md` — original challenge vs active research state
2. `BASELINE.md` — trusted baseline and killed non-baselines
3. `NEXT_EXPERIMENT.md` — next H100 run and decision rule
4. `.lab/insights.md` — validated learnings
5. `.lab/ideas_queue.md` — prioritized research queue

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

- `README.md` — original OpenAI challenge context and rules, not the active runbook.
- `PROJECT.md` — current repo orientation.
- `BASELINE.md` — trusted baseline at tag `baseline-3e34098`, TTT BPB `1.2102`.
- `NEXT_EXPERIMENT.md` — current next experiment.
- `train_gpt.py` — H100 active development and active RunPod entrypoint.
- `train_gpt_common.py` — shared constants/helpers for PyTorch and MLX paths.
- `train_gpt_mlx.py` — Apple Silicon experiment file; useful for local smoke checks, not the H100 oracle.
- `analyze.py` — post-run analysis.
- `.lab/insights.md` — validated learnings.
- `.lab/ideas_queue.md` — prioritized ideas.
- `.lab/results.tsv` — per-run result table.
- `EXPERIMENT_LOG.md` — full experiment history.
- `TODOS.md` — deferred research bets.

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
- Be a researcher, not a copier — develop original ideas.
- Never run experiments concurrently on Apple Silicon.
- Do not inspect or edit `records/**`.
- Do not use deprecated H100 scripts from git history for new runs.
```

- [ ] **Step 2: Replace or create `AGENTS.md` with matching current guidance**

Add or replace `AGENTS.md` with this exact content:

```markdown
# AGENTS.md

## Project Overview

Parameter Golf: train the best language model fitting in **16MB** in **<10 min on 8xH100**. Metric: **bits-per-byte (BPB)** on FineWeb validation. Lower is better.

## Read First

1. `PROJECT.md` — original challenge vs active research state
2. `BASELINE.md` — trusted baseline and killed non-baselines
3. `NEXT_EXPERIMENT.md` — next H100 run and decision rule
4. `.lab/insights.md` — validated learnings
5. `.lab/ideas_queue.md` — prioritized research queue

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
```

- [ ] **Step 3: Update `.gitignore`**

Add these lines near the existing local-tool/cache entries:

```gitignore
.codex/
.firecrawl/
```

Expected result: `git status --short` no longer shows `.codex/` or `.firecrawl/` as untracked.

- [ ] **Step 4: Correct the top summary in `EXPERIMENT_LOG.md`**

Replace the first three summary lines:

```markdown
**Final best**: val_bpb=**1.6215** (commit `e8addc5`, exp_063 LeakyReLU(0.5)²)
**Latest**: H100 repro cycle (clean 11L full-shard baseline established; 13L recipe and EMA discarded)
**H100 best**: TTT BPB **1.2087** (run1_baseline: 11L int8 zlib, 6421 steps, 80/195 shards)
```

with:

```markdown
**Final Mac best**: val_bpb=**1.6215** (commit `e8addc5`, exp_063 LeakyReLU(0.5)^2)
**Trusted H100 baseline**: TTT BPB **1.2102** (`clean_11l_no_ema`: 11L int8+zstd-22, full 195 shards, tag `baseline-3e34098`)
**Next H100 experiment**: isolated 11L deep supervision, EMA off (`h100_next_deep_supervision.sh`)
```

- [ ] **Step 5: Replace the stale H100 porting section in `.lab/insights.md`**

Replace the section beginning with:

```markdown
## Porting to 8xH100
```

through the line:

```markdown
Expected baseline: ~1500-2000 steps, val_bpb ~1.18-1.20.
```

with:

```markdown
## Active H100 Baseline Stack

The trusted H100 baseline is `baseline-3e34098` / `clean_11l_no_ema`: full `195/195` shards, `8007` steps, `74.95ms/step`, post-quant `1.2316`, TTT `1.2102`, artifact `14.48MB`.

Keep for baseline/control runs:
- `NUM_LAYERS=11`
- `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb`
- `MUON_WEIGHT_DECAY=0.10`
- `FREQ_SKIP_GATING=1`
- `TRAIN_BATCH_TOKENS=524288`
- `MLP_MULT_ASYMMETRIC=2,4`
- `GRAD_CLIP_NORM=0.5`
- `LAYER_LR_SCALE=0.5`
- `WARMUP_STEPS=50`
- `QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10`
- `USE_ZSTD=1 ZSTD_LEVEL=22`
- `EMA_DECAY=0`

Next H100 run: isolated deep supervision on this clean 11L stack (`DEEP_SUPERVISION=1`, `DEEP_SUPERVISION_ALPHA=0.05`, `DEEP_SUPERVISION_LAYERS=3,7`) with EMA off.
```

- [ ] **Step 6: Replace the stale final takeaway in `.lab/insights.md`**

Replace:

```markdown
- **Next H100 session should be diagnostic, not feature-stacked**: Since the local sprint found no portable winner, the next RunPod budget should go to (1) a clean 11L repro on `train_gpt.py` with full 195 shards and no EMA, (2) a 13L `int8+zstd` capacity run with no EMA, and (3) an isolated EMA run only after the clean repro is trustworthy.
```

with:

```markdown
- **The H100 diagnostic session is complete**: clean 11L full-shard repro is trusted; current 13L recipe and EMA(0.997) are killed in this codebase. The next RunPod budget should test isolated 11L deep supervision with EMA off.
```

- [ ] **Step 7: Correct stale target wording in `.lab/ideas_queue.md`**

Replace:

```markdown
**Target**: H100 competition (leaderboard 1.1194, our best 1.2087, gap 0.089 BPB)
```

with:

```markdown
**Target**: H100 competition (leaderboard reference 1.1194 in this historical sprint section; trusted H100 baseline 1.2102, gap 0.091 BPB)
```

- [ ] **Step 8: Run focused tests that do not depend on pruning yet**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_import_safety.py test_train_gpt_logging.py -q
```

Expected: PASS.

- [ ] **Step 9: Do not commit yet**

Commit after Task 5 makes the active-state test pass.

---

### Task 5: Prune Stale Tracked Files

**Files:**
- Delete stale tracked files listed below.

- [ ] **Step 1: Remove stale runners and deprecated H100 script**

Run:

```bash
git rm \
  h100_launch_all.sh \
  h100_phase1.sh \
  h100_repro.sh \
  h100_run1_proven.sh \
  h100_run2_deep_capacity.sh \
  h100_run3_training_amplifier.sh \
  h100_run4_calibrated.sh \
  h100_run5_progressive.sh \
  h100_run6_everything.sh \
  h100_sprint.sh \
  runpod_run1_baseline.sh \
  runpod_run2_capacity.sh \
  runpod_run3_swa.sh \
  train_gpt_h100.py
```

Expected: each listed file is removed from the working tree and staged for deletion.

- [ ] **Step 2: Remove stale generated/convenience docs and helpers**

Run:

```bash
git rm \
  TRAINING_WALKTHROUGH.md \
  TRAINING_WALKTHROUGH.pdf \
  experiment_results.xlsx \
  generate_results_excel.py \
  test_h100_smoke.py
```

Expected: each listed file is removed from the working tree and staged for deletion.

- [ ] **Step 3: Remove superseded design and plan docs**

Run:

```bash
git rm \
  docs/superpowers/plans/2026-04-04-h100-port-and-runs.md \
  docs/superpowers/plans/2026-04-28-stacked-h100-sprint.md \
  docs/superpowers/plans/2026-04-29-project-stabilization.md \
  docs/superpowers/specs/2026-03-28-h100-competition-prep-design.md \
  docs/superpowers/specs/2026-04-03-first-principles-innovations-design.md \
  docs/superpowers/specs/2026-04-03-moonshot-sprint-design.md \
  docs/superpowers/specs/2026-04-04-h100-five-runs-design.md \
  docs/superpowers/specs/2026-04-28-stacked-h100-sprint-design.md \
  docs/superpowers/specs/2026-04-29-project-stabilization-summary.md
```

Expected: only the current prune spec and this implementation plan remain under `docs/superpowers/**`.

- [ ] **Step 4: Run active-state tests**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_active_project_state.py tests/test_runner_targets.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit docs, runners, tests, and tracked prune**

Run:

```bash
git add PROJECT.md BASELINE.md NEXT_EXPERIMENT.md AGENTS.md CLAUDE.md .gitignore .lab/insights.md .lab/ideas_queue.md EXPERIMENT_LOG.md h100_l0_only.sh h100_next_deep_supervision.sh tests/test_active_project_state.py
git commit -m "chore: prune repo to active H100 experiment state"
```

Expected: commit succeeds. This commit includes file deletions staged by `git rm`.

---

### Task 6: Move Ignored Artifacts To External Archive

**Files:**
- Create temporarily: `archive_manifest_2026-04-30.md`
- Move to external archive: `/Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune/MANIFEST.md`
- Move ignored local paths if present: `logs/`, `results/`, `runs/`, `.firecrawl/`, root generated model/run outputs.

This task writes outside the workspace. When executing, request escalated permissions for the `mkdir` and `mv` commands that target `/Users/Season_Yang/Development/parameter-golf-archive`.

- [ ] **Step 1: Create temporary manifest in repo**

Add `archive_manifest_2026-04-30.md` with this exact content:

```markdown
# Parameter Golf Pre-Prune Artifact Archive

Archive date: 2026-04-30
Source repo: `/Users/Season_Yang/Development/parameter-golf`
Archive target: `/Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune`

## Why This Archive Exists

The active repository was pruned to make the original challenge, trusted baseline, and next H100 experiment unambiguous. Bulky ignored outputs were moved out of the active tree instead of deleted.

## Moved If Present

- `logs/`
- `results/`
- `runs/`
- `.firecrawl/`
- root generated files matching `final_model.*`
- root generated files matching `run.log`
- root generated files matching `run_*.log`
- root generated files matching `*.ptz`
- root generated files matching `*.npz`

## Not Moved

- `records/**` — off-limits public submission material
- `.lab/**` — canonical local analysis state
- `data/datasets/**` — local development dataset cache
- `data/tokenizers/**` — local tokenizer cache

## Active Summary Files Kept In Repo

- `PROJECT.md`
- `BASELINE.md`
- `NEXT_EXPERIMENT.md`
- `.lab/insights.md`
- `.lab/ideas_queue.md`
- `.lab/results.tsv`
- `EXPERIMENT_LOG.md`
- `TODOS.md`

## Trusted Baseline Reminder

- tag: `baseline-3e34098`
- run: `clean_11l_no_ema`
- post-quant BPB: `1.2316`
- TTT BPB: `1.2102`
- artifact: `14.48MB`
```

- [ ] **Step 2: Create archive directory outside repo**

Run with escalated permissions:

```bash
mkdir -p /Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune
```

Expected: command exits 0.

- [ ] **Step 3: Move bulky ignored artifact directories**

Run with escalated permissions:

```bash
ARCHIVE=/Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune
for path in logs results runs .firecrawl; do
  if [ -e "$path" ]; then
    mv "$path" "$ARCHIVE/"
  fi
done
```

Expected: any existing listed directories are moved into the archive.

- [ ] **Step 4: Move root generated run outputs**

Run with escalated permissions:

```bash
ARCHIVE=/Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune
find . -maxdepth 1 -type f \( -name 'final_model.*' -o -name 'run.log' -o -name 'run_*.log' -o -name '*.ptz' -o -name '*.npz' \) -exec mv {} "$ARCHIVE/" \;
```

Expected: matching generated files are moved if they exist. Source files are untouched.

- [ ] **Step 5: Move manifest into archive**

Run with escalated permissions:

```bash
mv archive_manifest_2026-04-30.md /Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune/MANIFEST.md
```

Expected: manifest exists at `/Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune/MANIFEST.md` and no temporary manifest remains in repo.

- [ ] **Step 6: Confirm archive and active tree**

Run:

```bash
du -sh /Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune
git status --short
```

Expected:

- archive has nonzero size if artifacts existed
- `git status --short` does not show `logs/`, `results/`, `runs/`, `.firecrawl/`, or `archive_manifest_2026-04-30.md`

No commit is needed for this task unless `.gitignore` was missed earlier.

---

### Task 7: Final Verification

**Files:**
- No new files.

- [ ] **Step 1: Run pytest collection**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest --collect-only -q
```

Expected: collection succeeds. The exact test count will be lower than before because `test_h100_smoke.py` was removed.

- [ ] **Step 2: Run focused verification suite**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest test_analyze.py tests/test_runner_targets.py tests/test_import_safety.py tests/test_eval_sliding.py tests/test_active_project_state.py -q
```

Expected: PASS.

- [ ] **Step 3: Check active runners for banned paths**

Run:

```bash
rg -n "train_gpt_h100|EMA_DECAY=0\\.997|GROW_LAYERS_FROM|NUM_LAYERS=13|CALIBRATED_QUANT=1|USE_DYT_NORM=1|USE_POLAR_EXPRESS=1" -- h100_l0_only.sh h100_next_deep_supervision.sh
```

Expected: command exits 1 with no matches because only active runners remain and they do not contain banned snippets.

- [ ] **Step 4: Check `records/**` was not touched**

Run:

```bash
git status --short -- records
```

Expected: no output.

- [ ] **Step 5: Check final git status**

Run:

```bash
git status --short
```

Expected: no unstaged source changes. Ignored `.codex/` may remain on disk but should not appear.

- [ ] **Step 6: Final commit if verification caused any small doc/test fix**

If Task 7 required a fix, commit it:

```bash
git add -A
git commit -m "fix: align prune verification"
```

Expected: commit succeeds. If no files changed, skip this step.
