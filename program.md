# parameter-golf: Autonomous Experiment Agent

This document instructs an AI agent to autonomously run experiments on the parameter-golf challenge, iteratively improving val_bpb on Apple Silicon using `train_gpt_mlx.py`.

**The goal is simple: get the lowest val_bpb on 8xH100 (RunPod).** Apple Silicon is our development environment for fast iteration, but the real competition runs on 8xH100 with 64x more data per step. An idea that wins on Mac usually wins on GPU too — but keep in mind that some tradeoffs differ (e.g. 11 layers is too slow on Mac but likely better on 8xH100 where step time is batch-dominated). The only hard constraints are the 16MB artifact limit and the 10-minute time budget.

## Setup (once per session)

1. **Agree on a branch tag** (e.g. `lab/mar27`). The branch must not already exist.
2. **Create the branch**: `git checkout -b <tag>` from current HEAD.
3. **Read prior work** — this prevents repeating failed experiments:
   - `.lab/insights.md` — current best config + validated learnings (the source of truth)
   - `.lab/ideas_queue.md` — prioritized research queue with originality tags
   - `EXPERIMENT_LOG.md` — full narratives (skim the summary table, read relevant sections)
4. **Read the codebase**:
   - `train_gpt_mlx.py` — **the file you modify**. Model, optimizer, training loop.
   - `train_gpt.py` — PyTorch reference. Read-only, but full of ideas to port.
5. **Verify data**: `./data/datasets/fineweb10B_sp1024/` has shards and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run the download script.
6. **Confirm and go**: Kick off the experiment loop.

## Constraints

**What you CAN do:**
- Modify `train_gpt_mlx.py` — this is the only file you edit. Architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization, evaluation approach.
- Write and run Python analysis scripts to explore training dynamics.
- Change environment variables passed to the training script.

**What you CANNOT do:**
- Modify `train_gpt.py`, `analyze.py`, or anything in `data/`. They are read-only.
- Install new packages beyond what's available (mlx, numpy, sentencepiece, matplotlib, huggingface_hub, datasets, tqdm).
- Modify the evaluation metric. The BPB calculation is ground truth.

**Size constraint**: Artifact must be < **16,000,000 bytes** after int8 quantization + zlib compression. The script reports this as `serialized_model_int8_zlib`. If a change pushes you over, it's invalid.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. A 0.001 BPB improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep.

## Research Philosophy

**This is a competition, not a homework assignment.** Do NOT simply copy techniques from other submissions or papers. Use the competition leaderboard, `train_gpt.py`, and academic literature as *inspiration* — understand *why* techniques work, then develop your own original approaches.

**Think like a researcher:**
- **Form hypotheses from first principles.** The next step after "WD=0.10 works" isn't WD=0.12 — it's asking "what other ways can we reduce weight entropy?"
- **Combine insights in new ways.** The best ideas connect two known observations into something novel.
- **Challenge defaults.** Why is MLP width uniform across layers? Why are skip connections symmetric? Why is quantization bit-width uniform?
- **Mark originality clearly.** Tag ideas as ORIGINAL / OUR TWIST / KNOWN / SWEEP in the ideas queue.

**What counts as original work:**
- Novel combinations of existing ideas that nobody has tried together
- New scheduling strategies (WD, momentum, LR) motivated by our own experiments
- Architectural modifications motivated by our own analysis
- Training dynamics insights unique to this setup

## Environment

All commands MUST run inside the `openai` conda environment:
```bash
conda run -n openai --no-capture-output <command>
```

## Run Commands

Get current best env vars from `.lab/insights.md` (under "Best config env vars"). Template:

```bash
# Smoke test (~2 min) — for untested ideas
RUN_ID=exp_NNN <BEST_CONFIG_VARS> ITERATIONS=200 VAL_LOSS_EVERY=0 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log

# Medium run (~15-20 min including eval) — for promising changes
RUN_ID=exp_NNN <BEST_CONFIG_VARS> ITERATIONS=2000 VAL_LOSS_EVERY=500 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

Note: `TRAIN_BATCH_TOKENS` is part of `<BEST_CONFIG_VARS>` from `.lab/insights.md`. Do NOT hardcode it here — it changes as we optimize.

**Tiered approach**: Always start with a smoke test. If the change looks promising (loss trending better at same step count), promote to medium. Only do a full run (10000+ iters) for changes showing clear medium-scale improvement.

## Output Format

The MLX training script prints structured log lines:
```
step:N/TOTAL train_loss:X.XXXX train_time:NNNms step_avg:NN.NNms tok_s:NNNN
step:N/TOTAL val_loss:X.XXXX val_bpb:X.XXXX train_time:NNNms step_avg:NN.NNms
```

At the end:
```
final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
```

The **authoritative metric** is `final_int8_zlib_roundtrip_exact val_bpb`. For smoke tests without full eval, compare `train_loss` at equivalent step counts.

## Logging Results

After each run, `analyze.py` appends a row to `.lab/results.tsv` (tab-separated, 5 columns):
```
commit	val_bpb	artifact_bytes	status	description
```
Status starts as `pending`. You update it to `keep` or `discard` after deciding.

## The Experiment Loop

**LOOP FOREVER:**

### 1. IMPLEMENT (main agent)

Read `insights.md` and `ideas_queue.md`. Pick the top idea. Modify `train_gpt_mlx.py`.

Be a researcher:
- Form a hypothesis: "I expect X to improve because Y"
- Design a minimal test: change one thing at a time when possible
- Predict the outcome before running

### 2. CODE REVIEW (mandatory — DO NOT SKIP)

**Before any validation or run, review your own implementation.** Spawn a code-review subagent or carefully self-review every changed line. Check for:
- **Correctness**: Does the code match the hypothesis? Are tensor shapes right? Are operations applied in the correct order?
- **Off-by-one errors**: Layer indices, slice boundaries, loop ranges.
- **Dtype/device issues**: MLX dtype mismatches (e.g. float32 vs bfloat16), unintended casts.
- **Side effects**: Does the change accidentally affect other code paths (e.g. breaking the baseline when the feature is disabled)?
- **Env var defaults**: Is the feature disabled by default so the baseline is unchanged?

If anything looks wrong, fix it before proceeding. A buggy experiment wastes 15+ minutes of compute and produces misleading results.

### 3. VALIDATE & COMMIT (main agent)

For code changes (skip for pure env-var changes):
```bash
# A. Syntax check (~5s)
conda run -n openai --no-capture-output python3 -c "import train_gpt_mlx; print('OK')"

# B. 5-step micro-run (~30s)
RUN_ID=validate ITERATIONS=5 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=60 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tail -5
```

Loss should start ~6.93 (ln(1024)) and decrease. Red flags: NaN, Inf, loss stuck, loss >> 7.0.

**Always git commit before running.** Use a descriptive message with the experiment ID and hypothesis. `analyze.py` archives code into `.lab/<commit>/` after each run.

### 4. RUN (main agent)

Launch with the appropriate tier (smoke or medium). Get current best env vars from `.lab/insights.md`.

### 5. POST-RUN BOOKKEEPING (mandatory subagent — DO NOT SKIP)

**After every run, spawn a subagent to handle all bookkeeping.** This is the most critical step. The main agent MUST NOT proceed to the next experiment until the subagent completes and confirms all updates are done.

**Spawn an Agent** with `subagent_type: "general-purpose"` and the following prompt (fill in the bracketed values):

```
You are the post-run bookkeeping agent for a parameter-golf experiment.

## Your task
Process the results of experiment [EXP_ID] and update ALL state files.
Do NOT skip any file. You must confirm each update at the end.

## Context
- Run log: [LOG_FILE_PATH] (e.g., logs/exp_034_gradclip025.txt)
- Current best val_bpb: [BEST_VAL_BPB] from [BEST_EXP_ID]
- Best config env vars: [BEST_CONFIG_VARS]
- Hypothesis: [WHAT_WAS_TESTED_AND_WHY]

## Steps (do ALL of them)

### A. Run analyze.py
```
conda run -n openai --no-capture-output python3 analyze.py
```
Read the generated `.lab/<commit>/analysis.md`.

### B. Extract results from run log
Parse the log for: final val_bpb (int8), artifact size, steps completed,
step time, and any other notable metrics.

### C. Decide: KEEP or DISCARD
Compare final val_bpb against current best.
- If LOWER → NEW BEST
- If EQUAL or HIGHER → DISCARD

### D. Update `.lab/results.tsv`
Append a row. Set status to `keep` or `discard`.

### E. Update `EXPERIMENT_LOG.md`
1. Add a row to the summary table with: exp number, run ID, change description,
   steps, val_bpb, artifact size, status, and key takeaway.
2. If this was a NEW BEST or reveals an important learning, add a detailed
   narrative section.

### F. Update `.lab/insights.md`
- If NEW BEST: update the "Current Best" section (commit, val_bpb, artifact,
  log path, next_exp number, and best config env vars).
- Add any new validated learnings to the appropriate section.
- Delete any disproven hypotheses.

### G. Update `.lab/ideas_queue.md`
- Mark the tested idea as completed (strikethrough + result).
- Kill any ideas that this result disproves.
- Add any new ideas spawned by this result.
- Reprioritize: does this result make queued ideas more/less promising?

### H. Research reflection (write in analysis notes)
Append to `.lab/<commit>/analysis.md` under "Agent Investigation Notes":
- What deeper principle did this experiment reveal?
- Chain: Observation → Principle → Implication → New idea / Killed idea
- How does the loss trajectory compare to previous runs?
- What surprised you?

### I. If DISCARD: restore best code
```
cp .lab/<best_commit>/train_gpt_mlx.py ./train_gpt_mlx.py
```
(Only if the best commit's code is archived. Otherwise use `git checkout <best_commit> -- train_gpt_mlx.py`.)

### J. Git commit all updates
Stage all changed files (use `git add -f` for .lab/ files) and commit with
a message summarizing the result.

### K. Confirm completion
End your response with this exact checklist (fill in ✅ or ❌):
- [ ] analyze.py ran
- [ ] results.tsv updated
- [ ] EXPERIMENT_LOG.md updated (table + narrative if needed)
- [ ] insights.md updated
- [ ] ideas_queue.md updated
- [ ] analysis notes written
- [ ] code restored (if discard)
- [ ] git committed
- [ ] RESULT: [KEEP/DISCARD] val_bpb=[VALUE] vs best=[BEST_VALUE]
```

**GATE: Do not start step 1 for the next experiment until the subagent returns with all items checked.**

### 6. REPEAT
Re-read `program.md` (context gets long and you forget steps), then go back to step 1.

## Important Rules

**Never run concurrent experiments** on Apple Silicon. Unified memory contention degrades throughput 2-3x.

**Timeout**: Smoke tests should take <2 min, medium <15 min. Kill anything exceeding 15 min.

**Crashes**: If it's a typo/import error, fix and re-run. If fundamentally broken, log as `crash` in results.tsv and move on.

**Bookkeeping is mandatory**: The post-run subagent (step 4) is a hard gate. NEVER skip it. NEVER start the next experiment before the subagent confirms all files are updated. This is the #1 source of lost context and repeated mistakes.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask "should I keep going?" The human might be asleep and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read your insights, look at the loss curves more carefully, try combining previous near-misses, study `train_gpt.py` for new inspiration, try more radical architectural changes. The loop runs until the human interrupts you. As an example, if each experiment takes ~15 minutes, you can run ~4/hour, ~32 over an 8-hour sleep. The human wakes up to results.

**Be a researcher, not a search script**: Don't just grid-search hyperparameters. Investigate why things work. Load `metrics.jsonl` files and compare loss trajectories across runs. Form hypotheses and test them. The per-step data is there for a reason — use it.
