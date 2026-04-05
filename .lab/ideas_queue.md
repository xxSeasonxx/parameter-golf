# Ideas Queue

Prioritized by expected impact. Each idea is one experiment, one commit.

**Originality key**:
- **ORIGINAL** — Novel technique we invented
- **OUR TWIST** — Known technique applied in a novel way
- **KNOWN** — Established technique we haven't tried yet
- **SWEEP** — Pure parameter/env-var exploration

---

## START HERE: Minimal H100 Session

**Goal**: Spend the next RunPod budget only on the highest-information runs. The local-first follow-up cycle found no shared-stack winner worth promoting.
**Current best**: val_bpb=1.6215 (exp_063, commit e8addc5)
**H100 best**: TTT BPB 1.2087 | Leaderboard: 1.1194 | Gap: 0.089 BPB

Run these in order:

### H100-Next-1: Clean 11L repro, no EMA [KNOWN, H100-ONLY]
**What**: Reproduce the best known PyTorch baseline in `train_gpt.py` with full 195 shards, `int8 + zstd`, and **EMA off**.
**Why**: Re-establish trustworthy measurement in the real scoring path before testing any new feature.

### H100-Next-2: 13L int8+zstd capacity test, no EMA [KNOWN, H100-ONLY]
**What**: Re-run the 13-layer capacity direction with `int8 + zstd` instead of int6, still with **EMA off**.
**Why**: H100 logs already showed 13L has better pre-quant quality. The missing question is whether int8 preserves enough of that gain under the size cap.

### H100-Next-3: EMA isolated on the clean 11L stack [KNOWN, H100-ONLY]
**What**: Add `EMA_DECAY=0.997` to the clean 11L repro only. No other changes.
**Why**: EMA is a real H100-only hypothesis, but the current code path can make it look much worse if stacked with other uncertainty. Test it in isolation.

### ~~Experiment 062: Progressive Layer Growing 7L→10L~~ COMPLETED -- DISCARD
**Result**: Tested via exp_076/076b with a forced early trigger. Even after actual growth, the run lands at **2.1457** BPB at step 200 vs baseline **2.0717**. Faster early steps do not repay the shallow-model quality debt. Kill on Mac and do not promote.

### ~~Experiment 063: LeakyReLU(0.5)²~~ COMPLETED -- NEW BEST
**Result**: val_bpb=1.6215, -0.0084 BPB vs previous best (1.6299). Largest non-batch/non-clip win. Dead neuron elimination via 50% negative slope. Pre-quant 1.6190 (also best ever). All remaining experiments now stack on top of this win.

### ~~Experiment 064: EMA (decay=0.997)~~ COMPLETED -- DISCARD
**Result**: val_bpb=1.8437, +0.22 BPB. EMA window (333 steps) = 48% of 689 total steps — dominated by early under-trained weights. Pre-EMA model was 1.6224 (fine). **EMA killed on Mac.** Reserve for H100 where 6000+ steps make decay=0.997 reasonable (~5.5% window).

### ~~Experiment 065: Multi-Band Skip Gating v2 (3 bands)~~ COMPLETED -- DISCARD
**Result**: val_bpb=1.6238, +0.0023 BPB. 2-band (W=32) already captures the useful spectral decomposition. Ultra-low band (W=128) is redundant — splits 4 dims from 16 into a separate channel but provides no new information. Extra parameters (5 skip_ulo_weights vectors) are under-constrained. Kill multi-band experiments.

### ~~Experiment 066: XSA on Decoder Layers~~ COMPLETED -- DISCARD
**Result**: val_bpb=1.6222, +0.0007 BPB (neutral). Self-exclusion removes 1/1024 context — too small at 10L. No speed penalty from custom mask. Worth trying on H100 with 11L+ where deeper layers benefit more from pure-context signals.

### ~~Experiment 067: Partial RoPE (25% dims)~~ COMPLETED -- DISCARD
**Result**: val_bpb=1.6240, +0.0025 BPB. At seq_len=1024, full RoPE is better — every position matters for short sequences. Partial RoPE may help at longer seq_len (4K+) where more attention patterns are position-invariant.

### ~~Experiment 068: GELU² Activation~~ COMPLETED -- DISCARD
**Result**: val_bpb=1.6373, +0.0158 BPB. GELU's Gaussian gating kills negative inputs exponentially, while LeakyReLU(0.5) preserves 50% linearly. With squaring providing sparsity, negative gradient preservation is the key mechanism — more flow is better. Activation exploration is CLOSED.

### ~~Experiment 069: LeakyReLU(0.7)² — higher negative slope~~ COMPLETED -- DISCARD
**Result**: val_bpb=1.6231, +0.0016 BPB. Slope 0.7 reduces sparsity too much (0.7²=0.49 on negatives vs 0.5²=0.25). Full activation slope sweep: ReLU²(1.6299) < LeakyReLU(0.5)²(**1.6215**) > LeakyReLU(0.7)²(1.6231) > GELU²(1.6373). Slope 0.5 is the sweet spot. Activation exploration KILLED.

### ~~Experiment 070: Wider Model MODEL_DIM=544~~ COMPLETED -- DISCARD
**Result**: val_bpb=1.6446, +0.023 BPB. 27.2M params (+13%) but 1018ms/step (+19%), only 590 steps vs 700. Step speed dominates capacity on Apple Silicon. Matches 11L finding. Model width increases KILLED for Mac — reserve for H100 where step time is batch-dominated.

---

## First-Principles Innovation Sprint (exp_071-075)

**Goal**: Three original innovations, each tested in A/B isolation against current best baseline.
**Baseline**: val_bpb=1.6215 (exp_063, commit e8addc5)
**Target**: H100 competition (leaderboard 1.1194, our best 1.2087, gap 0.089 BPB)
**Philosophy**: Fix the training-eval objective mismatch, inject gradient signal deeper, use compute more efficiently.

### ~~Experiment 071: Baseline Reconfirm~~ COMPLETED -- CONTROL
**Result**: val_bpb=1.6251 post-quant (1.6224 pre-quant). 688 steps, 873ms/step, 13.1MB artifact. Matches exp_063 (1.6215) within noise (+0.0009).

### ~~Experiment 072: Byte-Weighted Loss~~ COMPLETED -- DISCARD
**Result**: val_bpb=1.6659 post-quant, **+0.041 BPB regression**. 679 steps, 885ms/step. Byte weighting concentrates gradients on multi-byte tokens and under-trains frequent single-byte tokens. Standard token-level CE already optimizes BPB effectively through its correlation with per-token prediction quality. Training-eval objective mismatch is NOT the bottleneck. **Kill byte-weighted loss.**

### ~~Experiment 073: Deep Supervision~~ COMPLETED -- DISCARD (Mac), PROMISING (H100)
**Result**: val_bpb=1.6351 post-quant, **+0.010 BPB on Mac**. 651 steps at 923ms/step (5% overhead). BUT per-step quality was BETTER (1.7223 vs 1.7307 at step 500, -0.008 BPB). The 5% overhead cost 37 steps (688→651), killing the gain. **On H100 (6000+ steps), 5% overhead = ~300 steps lost but per-step improvement compounds over 5700+ steps.**
**Follow-up**: A lighter variant (layers `3,7`, `alpha=0.05`) removed the overhead penalty but also removed the quality gain (`2.0766` vs baseline `2.0717` at step 200). Keep deep supervision as a low-confidence H100-only candidate, not a local winner.

### ~~Experiment 074: Seq Len Curriculum~~ COMPLETED -- DISCARD (Mac)
**Result**: Original curriculum was neutral on Mac (`1.6271`). A refined follow-up schedule `256:0.10,512:0.30,1024:1.0` is worse: **703** steps and **1.6387** BPB in 600s. Kill curriculum for local iteration.

### Experiment 075: Combined Winners [STACK] — SKIPPED
No clear winners on Mac. Deep supervision and curriculum are both H100 candidates but not additive on Mac.

---

## Backlog: Not Yet Run (low priority)

- Zstd compression on Mac — validated on H100, saves 1.6MB. Not urgent locally.
- Decoder-only pre-warmdown QAT [ORIGINAL, LOW PRIORITY] — smoke had a tiny win, but medium regressed to **1.6283** at 654 steps. Not worth H100 budget unless future evidence changes.

---

## Killed Ideas

See EXPERIMENT_LOG.md for full history and details.
