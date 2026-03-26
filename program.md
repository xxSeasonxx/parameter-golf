# parameter-golf: Autonomous Experiment Agent

This document instructs an AI agent to autonomously run experiments on the parameter-golf challenge, iteratively improving val_bpb on Apple Silicon using `train_gpt_mlx.py`.

## Environment

All commands MUST be run inside the `openai` conda environment. Either activate it first:
```bash
conda activate openai
```
Or prefix every command with `conda run -n openai --no-capture-output`:
```bash
conda run -n openai --no-capture-output python3 train_gpt_mlx.py
```

The agent should ensure the correct environment is active before running any Python commands.

## Setup

Work with the user to complete these steps before starting the experiment loop:

1. **Agree on a run tag**: Propose a tag based on today's date (e.g. `lab/mar26`). The branch must not already exist.
2. **Create the branch**: `git checkout -b <tag>` from current main.
3. **Read prior work first** (critical — prevents repeating failed experiments):
   - `EXPERIMENT_LOG.md` — Full narrative of every past experiment, learnings, and failures
   - `.lab/insights.md` — Validated technical knowledge base
   - `.lab/ideas_queue.md` — Prioritized list of what to try next
   - `.lab/techniques_from_pytorch.md` — Cross-pollination tracker
4. **Read the key codebase files**:
   - `CLAUDE.md` — project instructions and architecture reference
   - `train_gpt_mlx.py` — **the file you modify**. Model architecture, optimizer, training loop for Apple Silicon.
   - `train_gpt.py` — PyTorch reference. Read-only but full of ideas to port.
   - `README.md` — competition overview and leaderboard
5. **Study the records** (if not already done in prior sessions): Read READMEs and submission.json files in `records/track_10min_16mb/` to understand what techniques have been tried and what worked. Update `.lab/techniques_from_pytorch.md` with any new findings.
6. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains data shards and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run:
   ```bash
   conda run -n openai --no-capture-output python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
   ```
7. **Initialize tracking**: Verify `.lab/` directory exists with `insights.md`, `ideas_queue.md`, `techniques_from_pytorch.md`, and `results.tsv`. If not, create them.
8. **Confirm and go**: Confirm setup looks good, then kick off the experiment loop.

## Constraints

**What you CAN do:**
- Modify `train_gpt_mlx.py` — this is the only training file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization strategy, evaluation approach.
- Write and run Python analysis scripts to explore training dynamics.
- Generate plots with matplotlib and save them to `.lab/<commit>/`.
- Change environment variables passed to the training script.

**What you CANNOT do:**
- Modify `train_gpt.py` (PyTorch version). It is read-only reference material.
- Modify anything in `data/` — the dataset and tokenizer are fixed.
- Install new packages or add dependencies beyond what's already available (mlx, numpy, sentencepiece, matplotlib, huggingface_hub, datasets, tqdm).
- Modify the evaluation metric. The BPB calculation is the ground truth.
- Modify `analyze.py` — it handles bookkeeping automatically.

**The goal is simple: get the lowest val_bpb.**

**Size constraint**: The final artifact (model + code) must fit under **16,000,000 bytes** (16 MB decimal) after int8 quantization + zlib compression. The script reports this as "Total submission size int8+zlib". If a change pushes you over the limit, it's invalid — find a way to fit.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome.

## Research Philosophy

**This is a competition, not a homework assignment.** Do NOT simply copy techniques from other submissions or papers. Use the competition leaderboard, academic literature, and `train_gpt.py` as *learning and inspiration* — understand *why* techniques work, then develop your own original approaches.

**Think like a researcher:**
- **Understand the binding constraint.** At 16MB, the core challenge is *bits per parameter* — information density. Every idea should either pack more signal per byte of artifact, or extract more signal at eval time for free.
- **Form hypotheses from first principles.** "Weight decay improves compressibility" led us to WD=0.02. The next step isn't WD=0.03 — it's asking "what else makes weights more compressible?" and inventing Compression-Aware Training.
- **Combine insights in new ways.** The best ideas often emerge from connecting two known observations. Our warmdown insight (long warmdown = regularizer) + our WD insight (WD = regularizer + compressor) suggests warmdown-aware WD scheduling.
- **Challenge defaults.** Why is MLP width uniform across layers? Why are skip connections symmetric? Why is quantization bit-width uniform? Question every fixed choice.
- **Validate, don't assume.** An elegant hypothesis is worthless without a clean experiment. Use the tiered run approach (smoke → medium → full) to validate cheaply.

**What counts as original work:**
- Novel combinations of existing ideas that nobody has tried together
- New scheduling strategies (WD, momentum, LR, noise)
- Architectural modifications motivated by our own analysis (asymmetric capacity, frequency gating)
- Quantization innovations (per-row adaptive precision, compression-aware training)
- Training dynamics insights unique to this setup (progressive growing, dual-phase)

**What does NOT count:**
- Copying someone's submission code
- Blindly porting a technique without understanding why it works
- Parameter grid searches without a hypothesis

## Run Commands

The MLX script auto-saves a log to `logs/<RUN_ID>.txt` (no need to redirect stdout). Also pipe to `run.log` so `analyze.py` can find it.

**Smoke test** (~1-2min on Apple Silicon, for quick iteration):
```bash
RUN_ID=exp_NNN NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.02 \
  ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

**Medium run** (~15-20min including eval, for validating promising changes):
```bash
RUN_ID=exp_NNN NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.02 \
  ITERATIONS=2000 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=500 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

**Full run** (~30-60min+ on Mac, for final validation of significant improvements):
```bash
RUN_ID=exp_NNN NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.02 \
  ITERATIONS=10000 VAL_LOSS_EVERY=2000 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

Note: The env vars above include the current best config (10L, FP16 tok_emb, Muon WD=0.02). Adjust as needed for your experiment.

Replace `NNN` with the experiment number (e.g. `exp_001`, `exp_002`, ...).
The log is also saved to `logs/exp_NNN.txt` by the script itself.

**Tiered approach**: Always start with a smoke test. If the change looks promising (loss curve trending better than baseline at the same step count), promote to a medium run. Only do a full run for changes that show clear improvement at medium scale.

## Output Format

The MLX training script prints structured log lines:
```
step:N/TOTAL train_loss:X.XXXX train_time:NNNms step_avg:NN.NNms tok_s:NNNN
step:N/TOTAL val_loss:X.XXXX val_bpb:X.XXXX train_time:NNNms step_avg:NN.NNms
```

At the end:
```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:NNNms
final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
```

Note: Unlike the PyTorch version, the MLX script does **not** print peak memory or artifact size. Check artifact size manually with `ls -la logs/*_mlx_model.int8.ptz`. The model is saved to `logs/<RUN_ID>_mlx_model.int8.ptz`.

The **authoritative metric** is `final_int8_zlib_roundtrip_exact val_bpb` — this is what the competition scores. For smoke tests without full eval, use the last `val_bpb` from training or compare `train_loss` curves at equivalent steps.

## Current Best & Prior Work

Multiple experiment sessions have been run. The current best and all learnings are documented in:
- **`EXPERIMENT_LOG.md`** — Full narrative of every experiment, what worked, what failed, and why
- **`.lab/insights.md`** — Validated technical learnings (the authoritative knowledge base)
- **`.lab/ideas_queue.md`** — Prioritized queue of what to try next
- **`.lab/results.tsv`** — Master results table

**You MUST read `EXPERIMENT_LOG.md` and `.lab/insights.md` before starting any new experiments.** They contain hard-won knowledge that will prevent you from repeating failed experiments.

### Current Best (as of lab/mar26b session, 2026-03-26)

```
commit: 6ed4a1d
val_bpb: 1.8291 (Apple Silicon, 10L, TRAIN_BATCH_TOKENS=8192)
artifact: 14.6MB (1.4MB headroom)
config:
  NUM_LAYERS=10
  INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
  MUON_WEIGHT_DECAY=0.02
  WARMDOWN_ITERS=1200 (default)
  TRAIN_BATCH_TOKENS=8192
```

### Key Validated Insights (see EXPERIMENT_LOG.md for full details)

- **10 layers > 9 layers**: ~0.05 BPB improvement, +1.2MB artifact. Non-negotiable.
- **Muon WD=0.02**: -0.031 BPB AND -1.1MB artifact. Highest-impact single change.
- **FP16 tok_emb**: Reduces quantization gap to +0.0001 BPB. +0.5MB artifact.
- **Warmdown=1200 is optimal**: Reducing to 400 was WORSE on both BPB and artifact size. Acts as regularizer.
- **Sliding window eval**: Implemented (`EVAL_STRIDE=64`), ~0.03 BPB gain. Too slow on Apple Silicon (~50min). Use for final submission on 8xH100 only.
- **Never run experiments concurrently**: Unified memory contention degrades throughput 2-3x.
- **Solo throughput**: ~1600 steps in 600s at ~370ms/step (10L, batch=8192).
- **Weight decay improves compressibility**: Regularized weights compress better under zlib.

### Failed Experiments (do NOT repeat)

- **Warmdown=400**: Worse BPB (1.878 vs 1.861) AND artifact over 16MB (16.5MB). Long warmdown is good.
- **INT8_KEEP_FLOAT_MAX_NUMEL=600000**: Accidentally kept all large tensors as FP16 (35MB artifact). Use name patterns instead.
- **Concurrent runs on Apple Silicon**: 2-3x slower. Always run sequentially.

### Original Baseline (for reference)

```
Unmodified train_gpt_mlx.py, 200 iterations:
  val_bpb=2.4109 train_loss=3.9161
  ~320ms/step, ~25.7k tok/s, ~64s total training time
  Artifact: ~11.3MB
```

### Experiment Numbering

Previous sessions used exp_001 through exp_009. **Start new experiments at exp_010.**

## Tracking and Analysis

All experiment data lives in `.lab/`:

```
.lab/
  results.tsv                    # Master log (append-only TSV)
  insights.md                    # Validated learnings + CURRENT BEST tracking
  ideas_queue.md                 # Queue of ideas to try next
  techniques_from_pytorch.md     # Cross-pollination tracker
  <commit>/
    analysis.md                  # Auto-generated + YOUR investigation notes
    run.log                      # Training output
    train_gpt_mlx.py             # Exact code snapshot
    metrics.jsonl                # Parsed per-step metrics
    summary.json                 # Parsed final metrics
    loss_curve.png               # Auto-generated
    val_bpb_curve.png            # Auto-generated (if val steps exist)
    step_timing.png              # Auto-generated
    *.png                        # Any additional plots you generate
```

**results.tsv** has 5 tab-separated columns:
```
commit	val_bpb	artifact_bytes	status	description
```

`analyze.py` appends rows with status=`pending`. You update the status to `keep` or `discard` after deciding.

## The Experiment Loop

**LOOP FOREVER:**

### 0. REFRESH: Re-read program.md

At the start of every iteration, re-read this file (`program.md`) to remind yourself of the full process. Context gets long and you will forget steps otherwise.

### 1. ANALYZE: Run the analysis

After each run, run `analyze.py` and read the generated analysis:

```bash
conda run -n openai --no-capture-output python3 analyze.py
```

Then read `.lab/<commit>/analysis.md`.

### 2. INVESTIGATE: Dig deeper

`analyze.py` generates comparison plots overlaid with the previous run. **Start by looking at these plots** in `.lab/<commit>/`.

Then dig deeper. Each run's per-step metrics are saved as `metrics.jsonl`. Each line is JSON:

```python
# Example: Load a run's step-level metrics
import json
commit = "abc1234"
train_metrics = [json.loads(l) for l in open(f".lab/{commit}/metrics.jsonl")
                 if json.loads(l)["type"] == "train"]
losses = [m["train_loss"] for m in train_metrics]
```

```python
# Example: Find where two runs diverge
import json
cur = [json.loads(l) for l in open(f".lab/{cur_commit}/metrics.jsonl")
       if json.loads(l)["type"] == "train"]
prev = [json.loads(l) for l in open(f".lab/{prev_commit}/metrics.jsonl")
        if json.loads(l)["type"] == "train"]

for i in range(min(len(cur), len(prev))):
    delta = cur[i]["train_loss"] - prev[i]["train_loss"]
    if abs(delta) > 0.05:
        print(f"Divergence at step {cur[i]['step']}: delta={delta:.4f}")
        break
```

Things to look for:
- **Loss trajectory shape**: Still decreasing at the end? Plateaued? Diverging?
- **Learning rate sensitivity**: How does loss change during warmup vs warmdown?
- **Step timing**: Any slowdowns from architectural changes? Apple Silicon memory pressure?
- **Cross-run patterns**: Do certain types of changes have consistent effects?
- **Quantization headroom**: How close to the 16MB limit? Room for more parameters?
- **Where differences emerge**: At what step does this run diverge from the previous one?

### 3. SYNTHESIZE: Write your findings

**MANDATORY**: Open `.lab/<commit>/analysis.md` and APPEND your investigation notes at the bottom (under the "Agent Investigation Notes" header). Every run must have your written analysis. Write:
- What you observed in the loss curves and plots
- How this run compares to the previous one and the all-time best
- What was surprising or expected
- What this tells you about what to try next

If you skip this step, your research has no memory and you'll repeat mistakes.

### 4. UPDATE your research notes

**insights.md** — Your validated knowledge base:
- "Increasing matrix_lr beyond 0.06 causes training instability"
- "10 layers fits within 16MB after int8 quantization"
- "MLX step time increases 15% with seq_len 2048 vs 1024"

Delete disproven hypotheses. Keep this file clean and accurate.

**techniques_from_pytorch.md** — Update the cross-pollination tracker:
- Check off techniques you've ported
- Move failed ports to "Tried and Discarded" with a note on why
- Add new techniques discovered from reading PyTorch records

### 4.5. RE-EVALUATE: Research-driven idea reprioritization

**Every experiment teaches you something bigger than its result.** Before moving on, step back and ask:

**A. What deeper principle did this experiment reveal?**

Don't just record "WD=0.02 improved BPB." Ask *why* it worked. Example chain:
- *Observation*: WD=0.02 improved BPB by 0.031 AND reduced artifact by 1.1MB.
- *Principle*: Regularized weights have lower entropy → better zlib compression.
- *Implication*: **Anything that reduces weight entropy is doubly valuable** (better model + smaller artifact).
- *New idea*: Compression-Aware Training — directly optimize for compressibility during training.
- *Killed idea*: Warmdown reduction — we now know regularization during warmdown helps, not hurts.

Write this reasoning in your analysis notes. The chain from observation → principle → new idea is the core research skill.

**B. Reprioritize `ideas_queue.md` based on the new principle.**

For each idea currently in the queue, ask:
1. Does this experiment's result make this idea **more promising**? (Move up)
2. Does it make this idea **less promising or invalid**? (Move down or kill)
3. Does the new principle **spawn a better version** of this idea? (Replace)

Example: If you discover that "late-training changes have outsized impact on final BPB," then:
- Move up: warmdown-aware WD scheduling (targets late training)
- Move down: better initialization (only helps early training)
- Spawn: "late-phase architecture modification" — what if you enable extra capacity only during warmdown?

**C. Keep the queue short and actionable.**

- Maximum ~15 ideas. More than that means you're hoarding, not prioritizing.
- Every idea must have a clear hypothesis and a way to test it.
- If an idea has sat untested for 10+ experiments, either test it now or kill it — it's probably not actually high priority.
- Group related ideas. If you have 3 ideas about quantization, test the most promising one and let it inform the others.

**D. Look for idea combinations.**

The most powerful experiments often combine two insights. After each run, scan the queue for pairs that reinforce each other:
- WD success + skip connection analysis → "regularized skip weights"
- Asymmetric MLP + entropy-guided quantization → "allocate both compute AND bits to later layers"
- Embedding perturbation + frequency decomposition → "perturbation in frequency space"

If a combination looks promising, add it as a new idea and prioritize it above its individual components.

**This step is what separates parameter sweeping from research.** A sweep tries 50 values. A researcher tries 5 values, learns a principle, and uses it to skip the other 45.

### 5. DECIDE: Keep or reject? (compare against ALL-TIME BEST)

**CRITICAL**: Do NOT compare against the previous run. Compare against the **all-time best val_bpb** recorded in `insights.md` under "Current Best".

- If this run's val_bpb **is lower than the current best**: this is a NEW BEST.
  - Update `results.tsv`: change status from `pending` to `keep`
  - Update `insights.md`: set the "Current Best" section to this commit and val_bpb
  - Leave the code as is.
- If this run's val_bpb **is equal to or higher than the current best**: DISCARD.
  - Update `results.tsv`: change status from `pending` to `discard`
  - **Restore the best version**: `cp .lab/<best_commit>/train_gpt_mlx.py ./train_gpt_mlx.py`
  - Git commit the revert with a clear message.

This ensures you always build on top of the historically best configuration, not just the last thing you tried.

**Comparing smoke tests**: For smoke tests without full eval, use `train_loss` at equivalent step counts as a proxy. A smoke test that's clearly worse at step 200 than the baseline at step 200 can be discarded without a longer run. But if it's close or better, promote to a medium run before making a keep/discard decision.

### 6. IMPLEMENT: Make your next change

Read `insights.md`, `ideas_queue.md`, and `techniques_from_pytorch.md`. Pick the top idea. Modify `train_gpt_mlx.py`.

Think about what the investigation told you — don't just try random things. Be a researcher:
- Form a hypothesis: "I expect X to improve because Y"
- Design a minimal test: change one thing at a time when possible
- Predict the outcome before running

Git commit your change with a clear description of the hypothesis.

### 6.5. VALIDATE: Pre-flight checks before running

**Novel code changes are bug-prone.** Before launching a training run with new code, run this lightweight validation checklist. Skip this for pure env-var changes (those can't introduce bugs).

**A. Syntax & import check** (~5 seconds):
```bash
conda run -n openai --no-capture-output python3 -c "import train_gpt_mlx; print('OK')"
```
If this fails, fix the syntax error before proceeding.

**B. Shape & value sanity check** (~30 seconds):
Write a quick inline test that instantiates the model and verifies your change does what you expect. Examples:

```python
# For a new init strategy:
conda run -n openai --no-capture-output python3 -c "
from train_gpt_mlx import *
args = Hyperparameters()
model = GPT(args.vocab_size, args.num_layers, args.model_dim, args.num_heads, args.num_kv_heads,
            args.mlp_mult, args.logit_chunk_tokens, args.logit_softcap, args.rope_base,
            args.tied_embed_init_std, args.qk_gain_init)
# Check shapes, value ranges, or whatever your change affects
print('tok_emb shape:', model.tok_emb.weight.shape)
print('tok_emb std:', float(model.tok_emb.weight.astype(mx.float32).var()**0.5))
print('PASS')
"
```

```python
# For an optimizer change:
# Verify the new schedule/WD/momentum produces expected values at key steps
conda run -n openai --no-capture-output python3 -c "
from train_gpt_mlx import Hyperparameters
args = Hyperparameters()
# Check lr_mul at start, middle, end
for step, ms in [(0, 0), (500, 180000), (1500, 550000), (1600, 600000)]:
    print(f'step={step} lr_mul={args.lr_mul(step, ms):.4f}')
print('PASS')
"
```

**C. 5-step micro-run** (~30 seconds):
Run 5 training steps and check loss is finite and decreasing:
```bash
RUN_ID=validate ITERATIONS=5 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=60 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tail -5
```
- Loss should be ~6.9 at step 1 (random init cross-entropy over vocab 1024 ≈ ln(1024) ≈ 6.93)
- Loss should decrease over 5 steps
- **Red flags**: NaN, Inf, loss increasing, loss stuck at exactly the same value, loss >> 7.0

**D. When a run fails or produces suspicious results**:

Classify the failure:

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Crash before step 1 | Shape mismatch, bad init, import error | Fix and re-run |
| NaN after N steps | Gradient explosion, bad LR/WD interaction | Check value ranges, reduce LR, add gradient clipping |
| Loss stuck at ~6.93 | Model not learning (zero gradients, broken optimizer) | Print gradient norms, check optimizer is updating params |
| Loss much worse than baseline | Bug in forward pass, wrong masking, bad scale | Compare output shapes/values vs unmodified code |
| Loss close but slightly worse | The idea just didn't work (not a bug) | Discard the idea, not the approach |
| Artifact over 16MB | Model too large or weights not compressible | Check param count, try stronger WD |

**Key rule**: If a novel idea produces a result >0.1 BPB worse than baseline at the same step count, **suspect a bug first** before blaming the idea. Run the sanity checks above. Only classify as "idea failed" after confirming the implementation is correct.

**Never discard an idea on a crashed/NaN run.** Fix the bug and retry once. Only discard after a clean run shows the idea doesn't help.

### 7. RUN: Launch the experiment

Use the appropriate tier:
```bash
# Smoke test for untested ideas
RUN_ID=exp_NNN ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log

# Medium run for promising changes
RUN_ID=exp_NNN ITERATIONS=2000 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=500 \
  conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

### 8. POST-RUN: Run analysis

```bash
conda run -n openai --no-capture-output python3 analyze.py
```

### 9. REPEAT

Go back to step 0.

## Learning from the Ecosystem (Inspire, Don't Copy)

The PyTorch `train_gpt.py`, `records/`, and the broader competition community are sources of *understanding*, not copy-paste material. Use them to learn **why** things work, then develop **your own** approaches.

**How to learn from references:**
1. **`train_gpt.py`** — Read for architectural understanding. Ask "why is this designed this way?" not "how do I copy this?"
2. **`records/track_10min_16mb/*/README.md`** — Study the progression of ideas. What patterns emerge across winners?
3. **Competition leaderboard** — Understand the frontier, but aim to push it with original work.
4. **Academic literature** — Search for recent papers on small model training, quantization, optimizer design. Web search is available.

**Already ported from PyTorch (do not re-do):**
- ~~10-layer architecture~~ (NUM_LAYERS=10)
- ~~Muon weight decay~~ (MUON_WEIGHT_DECAY=0.02)
- ~~FP16 tied embeddings~~ (INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb)
- ~~Sliding window eval~~ (EVAL_STRIDE=64, too slow on Mac, use on 8xH100)

**Remaining reference techniques** (understand these, then innovate beyond them):
- Spectral/overtone embedding init — understand the principle (structured low-entropy init), then try our own variant
- Mixed-precision quantization — understand per-layer sensitivity, then try entropy-guided allocation
- LoRA TTT — understand eval-time adaptation, then explore novel adaptation strategies

**Key insight from competition analysis**: The binding constraint is *bits per parameter* in the artifact. The winners co-optimize the entire pipeline: init → training → quantization → compression → evaluation. Don't optimize one stage in isolation.

## Important Rules

**Timeout**: Smoke tests should take <2 min, medium runs <10 min. If a run exceeds 15 minutes (for a smoke/medium), kill it and treat it as a failure.

**Crashes**: If a run crashes (OOM, bug, etc.), fix trivial issues and re-run, or skip fundamentally broken ideas. Log crashes in results.tsv with status `crash`.

**Apple Silicon specifics**:
- Monitor memory pressure. Unified memory means CPU and GPU share RAM.
- Use `MLX_MAX_MICROBATCH_TOKENS` and `MLX_EAGER_EVAL` env vars to control memory usage.
- Step time can vary — close other applications for consistent benchmarking.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?". The human might be asleep and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read your insights, look at the loss curves more carefully, try combining previous near-misses, study the PyTorch records for new inspiration, try more radical architectural changes. The loop runs until the human interrupts you.

**Be a researcher, not a search script**: Don't just grid-search hyperparameters. Investigate why things work. Look at the auto-generated plots. Load `metrics.jsonl` files and compare loss trajectories across runs. Form hypotheses and test them. The per-step data is there for a reason — use it.
