# Project Prune And Next Experiment Design

**Date:** 2026-04-30
**Status:** Approved design
**Goal:** Aggressively prune the active repository so it is clear what is original challenge context, what is the trusted baseline, and what the next H100 experiment is.

## Problem

The project has accumulated many iterations, run scripts, plans, and local artifacts. That history is useful, but the active tree is now ambiguous:

- `README.md` describes the original OpenAI challenge and public leaderboard, not our current runbook.
- `.lab/insights.md` and `.lab/ideas_queue.md` identify the trusted H100 baseline and next candidate, but stale scripts still encode killed ideas.
- Several H100 scripts still enable EMA, 13L capacity, calibrated quant, progressive growth, or old stacked plans even though current learnings killed or deprioritized those paths.
- Ignored generated outputs, especially `logs/` and `results/`, dominate disk usage and distract from the active research state.

The cleanup must reduce ambiguity without changing model math or losing the evidence needed to understand prior decisions.

## Design

### 1. Canonical Map

The active repo will make three concepts explicit.

**Original challenge**

- Keep `README.md` as the upstream/OpenAI challenge context and rules.
- Do not treat `README.md` as the active project runbook.
- Add or update a small root-level orientation doc, `PROJECT.md`, that tells future agents and humans where to look first.

**Trusted baseline**

- Document the baseline in `BASELINE.md`.
- Baseline identity: git tag `baseline-3e34098`.
- Trusted result: clean 11L full-shard H100 repro, post-quant BPB `1.2316`, TTT BPB `1.2102`, artifact `14.48MB`.
- Make explicit that EMA(0.997), the current 13L recipe, SWA, and int6 without stronger quantization machinery are not baseline candidates.

**Next experiment**

- Document the next runnable experiment in `NEXT_EXPERIMENT.md`.
- Next experiment: clean 11L H100 baseline plus isolated deep supervision.
- Keep the exact env vars and success/failure criteria in the doc so RunPod spend is not driven by stale scripts.

Required reading order for future work:

1. `PROJECT.md`
2. `BASELINE.md`
3. `NEXT_EXPERIMENT.md`
4. `.lab/insights.md`
5. `.lab/ideas_queue.md`

### 2. Active Repository Boundary

Keep active:

- `README.md`, `PROJECT.md`, `BASELINE.md`, `NEXT_EXPERIMENT.md`
- `train_gpt.py`, `train_gpt_common.py`, `train_gpt_mlx.py`, `analyze.py`
- test files and data scripts
- `.lab/insights.md`, `.lab/ideas_queue.md`, `.lab/results.tsv`
- `EXPERIMENT_LOG.md`, `TODOS.md`
- active runner scripts only

Prune from active tree:

- stale H100 runner scripts that encode killed or superseded paths
- superseded design and plan docs under `docs/superpowers/**` when they conflict with the current strategy
- generated or convenience docs that are not needed for active development, such as old PDFs
- untracked local tool state that does not serve the active experiment
- `train_gpt_h100.py`; it is deprecated, no active runner should call it, and git history preserves it

Do not inspect, edit, delete, or reorganize `records/**`.

### 3. External Artifact Archive

Move bulky ignored artifacts out of the active repo instead of deleting them.

Archive target:

```text
/Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune/
```

Move these if present:

- `logs/`
- `results/`
- generated model files such as `final_model.*`
- other ignored generated run outputs that are not part of source control

Write:

```text
/Users/Season_Yang/Development/parameter-golf-archive/2026-04-30-pre-prune/MANIFEST.md
```

The manifest will list what was moved, why it was moved, and which active files preserve the canonical summaries.

### 4. Active Runners

Keep or create only the runners needed for the next H100 work.

**Baseline control:** `h100_l0_only.sh`

- clean 11L baseline stack
- `EMA_DECAY=0`
- `USE_POLAR_EXPRESS=0`
- `USE_DYT_NORM=0`
- `EVAL_STRIDE=64`
- purpose: isolate sliding-window timing and score

**Next experiment:** `h100_next_deep_supervision.sh`

- starts from the clean 11L H100 baseline
- adds only:
  - `DEEP_SUPERVISION=1`
  - `DEEP_SUPERVISION_ALPHA=0.05`
  - `DEEP_SUPERVISION_LAYERS=3,7`
- keeps:
  - `EMA_DECAY=0`
  - `USE_ZSTD=1`
  - `EVAL_STRIDE=64`
- does not stack DyT, Polar Express, EMA, 13L, calibrated quant, or layer growth

If the L0 control shows stride-64 eval exceeds the budget, update `NEXT_EXPERIMENT.md`
before launching the deep-supervision run rather than silently changing the runner.

Remove or archive active-tree scripts for old campaigns, 13L capacity, EMA/SWA comparisons, progressive growth, and feature-stacked runs.

### 5. Decision Rule

Use the trusted H100 baseline as the comparison point:

- baseline TTT BPB: `1.2102`
- baseline post-quant BPB: `1.2316`

For the next deep-supervision run:

- If TTT BPB improves by at least `0.005` versus `1.2102` and training/eval stay within budget, promote deep supervision.
- If TTT BPB regresses or eval exceeds the budget, kill or revise the idea.
- If the delta is between `0.000` and `0.005`, treat it as neutral and shift the next round toward TTT evaluation improvements rather than capacity, EMA, or the current 13L path.

### 6. Verification

This cleanup is a repository-organization change. It must not change model math.

Before claiming completion, run:

```bash
conda run -n openai --no-capture-output python -m pytest --collect-only -q
conda run -n openai --no-capture-output python -m pytest test_analyze.py tests/test_runner_targets.py tests/test_import_safety.py tests/test_eval_sliding.py -q
```

Also check active runner scripts for killed or deprecated paths:

- no active runner should call `train_gpt_h100.py`
- no active next-experiment runner should enable `EMA_DECAY=0.997`
- no active next-experiment runner should enable `GROW_LAYERS_FROM`
- no active next-experiment runner should enable the killed 13L calibrated path

## Non-Goals

- Do not change architecture, optimizer math, quantization math, or BPB calculation.
- Do not run a new ML experiment during cleanup.
- Do not touch `records/**`.
- Do not delete bulky artifacts; move them to the external archive.
- Do not rewrite the full historical experiment log; only correct the known active-summary drift, such as the stale top-of-file H100 best line.

## Risks

- Removing too much history can make future research decisions harder. Mitigation: preserve canonical summaries in `.lab/*`, `EXPERIMENT_LOG.md`, `BASELINE.md`, `NEXT_EXPERIMENT.md`, and the external archive manifest.
- Keeping old docs inside the active tree can reintroduce stale instructions. Mitigation: prune conflicting docs aggressively or label them as archived if they must remain.
- Moving ignored artifacts can break ad hoc local references. Mitigation: manifest the archive path and preserve small summaries in the repo.
