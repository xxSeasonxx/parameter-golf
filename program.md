# parameter-golf: Autonomous Experiment Agent

## Setup (once per session)

1. **Agree on a branch tag** (e.g. `lab/mar27`) and create it.
2. **Read prior work** — prevents repeating failed experiments:
   - `.lab/insights.md` — current best + validated learnings
   - `.lab/ideas_queue.md` — prioritized research queue
   - `EXPERIMENT_LOG.md` — full narratives (skim table, read relevant sections)
3. **Verify data**: `./data/datasets/fineweb10B_sp1024/` has shards and `./data/tokenizers/fineweb_1024_bpe.model` exists.
4. **Start the loop.**

## Constraints

- **Only modify** `train_gpt_mlx.py`. Everything else is read-only.
- Artifact must be < **16,000,000 bytes** after int8+zlib.
- Available packages: mlx, numpy, sentencepiece, matplotlib, huggingface_hub, datasets, tqdm.
- Simplicity: a small improvement that adds ugly complexity is not worth it.

## Research Philosophy

**Be a researcher, not a copier.** Use the competition and `train_gpt.py` as inspiration — understand *why* techniques work, then develop original approaches.

- **Form hypotheses from first principles.** The next step after "WD=0.10 works" isn't WD=0.12 — it's asking "what other ways can we reduce weight entropy?"
- **Combine insights in new ways.** The best ideas connect two known observations into something novel.
- **Challenge defaults.** Why is MLP width uniform? Why are skip connections symmetric?
- **Mark originality clearly.** Tag ideas as ORIGINAL / OUR TWIST / KNOWN / SWEEP in the ideas queue.

## Run Commands

Get current best env vars from `.lab/insights.md`. Template:

```bash
# Smoke test (~2 min) — for untested ideas
RUN_ID=exp_NNN <BEST_CONFIG_VARS> ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log

# Medium run (~15-20 min) — for promising changes
RUN_ID=exp_NNN <BEST_CONFIG_VARS> ITERATIONS=2000 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=500 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

**Tiered approach**: Smoke test first → promote if promising → full run only for significant improvements.

**Authoritative metric**: `final_int8_zlib_roundtrip_exact val_bpb`. For smoke tests, compare `train_loss` at equivalent steps.

## The Experiment Loop

**LOOP FOREVER** (do NOT ask "should I keep going?" — run until interrupted):

### 0. REFRESH
Re-read this file. Context gets long and you forget steps.

### 1. ANALYZE
```bash
conda run -n openai --no-capture-output python3 analyze.py
```
Read `.lab/<commit>/analysis.md` and the generated plots.

### 2. INVESTIGATE
Dig into `metrics.jsonl` for the run. Compare loss trajectories across runs. Look for:
- Loss shape (decreasing? plateaued?), timing changes, quantization headroom
- At what step does this run diverge from previous?

### 3. SYNTHESIZE
**MANDATORY**: Append investigation notes to `.lab/<commit>/analysis.md`. Write what you observed, what surprised you, what to try next. No notes = no memory.

### 4. UPDATE
Update `.lab/insights.md` with any new validated learnings. Delete disproven hypotheses.

### 4.5. RE-EVALUATE (the research step — DO NOT SKIP)

**A. What deeper principle did this experiment reveal?**

Chain: Observation → Principle → Implication → New idea / Killed idea. Write this reasoning in your analysis notes.

**B. Reprioritize `ideas_queue.md`.**

For each queued idea: does the new result make it more promising (up), less promising (down/kill), or spawn a better version (replace)?

**C. Keep the queue short** (max ~15). Kill ideas untested for 10+ experiments.

**D. Look for idea combinations** — pairs of insights that reinforce each other.

### 5. DECIDE (compare against ALL-TIME BEST in insights.md)

- **New best** → Update `results.tsv` status to `keep`, update `insights.md` current best.
- **Not better** → Update `results.tsv` to `discard`, restore best code: `cp .lab/<best_commit>/train_gpt_mlx.py ./train_gpt_mlx.py`

### 6. IMPLEMENT
Pick the top idea from `ideas_queue.md`. Form a hypothesis, predict the outcome, make a minimal change. Git commit with hypothesis description.

### 6.5. VALIDATE (for code changes, skip for env-var-only)

```bash
# A. Syntax check
conda run -n openai --no-capture-output python3 -c "import train_gpt_mlx; print('OK')"

# B. Sanity check (model instantiation, value ranges)

# C. 5-step micro-run
RUN_ID=validate ITERATIONS=5 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=60 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tail -5
```
Loss should start ~6.93 and decrease. Red flags: NaN, Inf, loss >> 7.0.

If a novel idea is >0.1 BPB worse than baseline, **suspect a bug first**. Never discard on a crashed run.

### 7. RUN
Launch with the appropriate tier (smoke or medium).

### 8. POST-RUN
```bash
conda run -n openai --no-capture-output python3 analyze.py
```

### 9. REPEAT
Go to step 0.

## Important Rules

- **Never run concurrent experiments** on Apple Silicon (2-3x throughput degradation).
- **Timeout**: Kill runs exceeding 15 min (for smoke/medium).
- **Experiment numbering**: Check `results.tsv` for the last exp number and increment.
- **Never stop**: Run autonomously until interrupted. If stuck, re-read insights, try combinations, study `train_gpt.py` for inspiration.
