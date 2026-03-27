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
RUN_ID=exp_NNN <BEST_CONFIG_VARS> ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log

# Medium run (~15-20 min including eval) — for promising changes
RUN_ID=exp_NNN <BEST_CONFIG_VARS> ITERATIONS=2000 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=500 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

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

### 0. REFRESH
Re-read this file (`program.md`). Context gets long and you will forget steps.

### 1. ANALYZE
```bash
conda run -n openai --no-capture-output python3 analyze.py
```
Read `.lab/<commit>/analysis.md` and the generated plots (loss curves, val_bpb, timing).

### 2. INVESTIGATE
Dig into `metrics.jsonl` for the run. Compare loss trajectories across runs. Look for:
- **Loss trajectory shape**: Still decreasing at the end? Plateaued? Diverging?
- **Step timing**: Any slowdowns from architectural changes?
- **Quantization headroom**: How close to 16MB?
- **Where differences emerge**: At what step does this run diverge from previous?

### 3. SYNTHESIZE
**MANDATORY**: Append investigation notes to `.lab/<commit>/analysis.md` under the "Agent Investigation Notes" header. Write what you observed, how it compares to the best, what surprised you, what to try next. No notes = no memory and you'll repeat mistakes.

### 4. UPDATE
Update `.lab/insights.md` with any new validated learnings. Delete disproven hypotheses. Keep it clean and accurate.

### 4.5. RE-EVALUATE: Research-driven idea reprioritization (DO NOT SKIP)

**Every experiment teaches you something bigger than its result.** This step is what separates parameter sweeping from research.

**A. What deeper principle did this experiment reveal?**

Don't just record "WD=0.10 improved BPB." Ask *why*. Chain: Observation → Principle → Implication → New idea / Killed idea. Write this reasoning in your analysis notes.

**B. Reprioritize `ideas_queue.md` based on the new principle.**

For each queued idea: does the new result make it more promising (move up), less promising (move down/kill), or spawn a better version (replace)?

**C. Keep the queue short and actionable** — max ~15 ideas. If an idea has sat untested for 10+ experiments, either test it now or kill it.

**D. Look for idea combinations** — the most powerful experiments often combine two insights that reinforce each other.

### 5. DECIDE (compare against ALL-TIME BEST in insights.md)

- **val_bpb lower than best** → this is a NEW BEST. Update `results.tsv` to `keep`. Update `insights.md` current best.
- **val_bpb equal or higher** → DISCARD. Update `results.tsv` to `discard`. Restore the best code: `cp .lab/<best_commit>/train_gpt_mlx.py ./train_gpt_mlx.py`. Git commit the revert.

### 6. IMPLEMENT
Read `insights.md` and `ideas_queue.md`. Pick the top idea. Modify `train_gpt_mlx.py`.

Be a researcher:
- Form a hypothesis: "I expect X to improve because Y"
- Design a minimal test: change one thing at a time when possible
- Predict the outcome before running

Git commit your change with a clear description of the hypothesis.

### 6.5. VALIDATE (for code changes — skip for pure env-var changes)

```bash
# A. Syntax check (~5s)
conda run -n openai --no-capture-output python3 -c "import train_gpt_mlx; print('OK')"

# B. Sanity check (~30s) — instantiate model, verify shapes/values

# C. 5-step micro-run (~30s)
RUN_ID=validate ITERATIONS=5 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=60 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tail -5
```

Loss should start ~6.93 (ln(1024)) and decrease. Red flags: NaN, Inf, loss stuck, loss >> 7.0.

If a novel idea is >0.1 BPB worse than baseline at same step count, **suspect a bug first**. Never discard an idea on a crashed/NaN run — fix the bug and retry.

### 7. RUN
Launch with the appropriate tier (smoke or medium).

### 8. POST-RUN
```bash
conda run -n openai --no-capture-output python3 analyze.py
```
Then update **all** state files:
- **`.lab/results.tsv`** — append a row with the result
- **`EXPERIMENT_LOG.md`** — append to the summary table AND write a detailed narrative
- **`.lab/insights.md`** — update current best if applicable, add new learnings
- **`.lab/ideas_queue.md`** — mark completed ideas, kill disproven ones, add new ideas

Do NOT batch these updates. Update after EVERY run before starting the next.

### 9. REPEAT
Go back to step 0.

## Important Rules

**Never run concurrent experiments** on Apple Silicon. Unified memory contention degrades throughput 2-3x.

**Timeout**: Smoke tests should take <2 min, medium <15 min. Kill anything exceeding 15 min.

**Crashes**: If it's a typo/import error, fix and re-run. If fundamentally broken, log as `crash` in results.tsv and move on.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask "should I keep going?" The human might be asleep and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read your insights, look at the loss curves more carefully, try combining previous near-misses, study `train_gpt.py` for new inspiration, try more radical architectural changes. The loop runs until the human interrupts you. As an example, if each experiment takes ~15 minutes, you can run ~4/hour, ~32 over an 8-hour sleep. The human wakes up to results.

**Be a researcher, not a search script**: Don't just grid-search hyperparameters. Investigate why things work. Load `metrics.jsonl` files and compare loss trajectories across runs. Form hypotheses and test them. The per-step data is there for a reason — use it.
