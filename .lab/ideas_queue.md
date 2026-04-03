# Ideas Queue

Prioritized by expected impact. Each idea is one experiment, one commit.

**Originality key**:
- **ORIGINAL** — Novel technique we invented
- **OUR TWIST** — Known technique applied in a novel way
- **KNOWN** — Established technique we haven't tried yet
- **SWEEP** — Pure parameter/env-var exploration

---

## START HERE: Moonshot Sprint (exp_062-067)

**Goal**: Validate 6 experiments locally on Mac. Take winners to H100.
**Current best**: val_bpb=1.6215 (exp_063, commit e8addc5)
**H100 best**: TTT BPB 1.2087 | Leaderboard: 1.1194 | Gap: 0.089 BPB

Run these in order. Each = one commit per program.md.

### Experiment 062: Progressive Layer Growing 7L→10L [ORIGINAL, HIGH IMPACT]
**What**: Start training with 7 layers for first 50% of wallclock, then grow to 10 layers. Zero-init new layers' output projections so they start as identity.
**Hypothesis**: 7L is ~15% faster per step → ~20% more total steps. Shallow features learned in Phase 1 transfer perfectly. Net: more total computation + pre-learned features = better BPB.
**Code change**: Add GROW_LAYERS_FROM=7 env var. In main(), reconstruct model at 50% wallclock: copy existing block weights, add new blocks with zero-init projections, rebuild optimizer. Expand U-Net skip structure (3enc+4dec → 5enc+5dec).
**Run**: Best config + GROW_LAYERS_FROM=7.
**Expect**: More total steps. BPB should be within 0.005 of constant 10L — if so, the step advantage makes it a clear win on H100.

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

## Backlog: Not Yet Run (low priority)

- Zstd compression on Mac — validated on H100, saves 1.6MB. Not urgent locally.
- Int6 + Zstd combined — int6 is killed without STE/GPTQ.

---

## Backlog: Not Yet Run (low priority)

- Zstd compression on Mac — validated on H100, saves 1.6MB. Not urgent locally.

---

## Killed Ideas

See EXPERIMENT_LOG.md for full history and details.
