# Ideas Queue

Prioritized by expected impact. Organized by research direction, not just parameter sweeps.

**Originality key**: Each idea is tagged:
- **ORIGINAL** — Novel technique we invented, not found in competition or literature
- **OUR TWIST** — Known technique applied in a novel way specific to our setup
- **KNOWN** — Established technique we haven't tried yet (porting, not inventing)
- **SWEEP** — Pure parameter/env-var exploration

**Guiding principles** (updated for 8xH100 competition focus):
1. **Weight entropy reduction is the master lever** — WD proved that reducing weight entropy improves BOTH BPB and compressibility simultaneously.
2. **Late-phase training changes have outsized impact** — The warmdown phase determines final quality. Target ideas at the last 25% of training.
3. **On 8xH100, step time is batch-dominated** — 11L, MLP3x, and other capacity increases are nearly free. Artifact size (16MB) is the binding constraint, not step time.
4. **Maximize effective params in 16MB** — Int6, mixed precision, and compression advances directly translate to more model capacity within the artifact limit.
5. **Eval-time compute is free** — TTT, sliding window, and other eval techniques cost nothing in the competition.
6. **Originality matters** — We develop our own approaches. Use competition as inspiration, not a copying target.

---

## 🎯 Competition Strategy: "Stack of Originals on a Solid Foundation"

**Target**: val_bpb ≤ 1.12 on 8xH100 (current leaderboard top: 1.1194)

| Layer | Techniques | Expected BPB | Status |
|-------|-----------|-------------|--------|
| Foundation | 11L, batch=524K, SWA, int6, TTT, sliding eval | ~1.14-1.15 | To port |
| Proven Originals | Warmdown-aware WD, freq skip gating | ~1.13-1.14 | Done |
| New Originals | Warmdown QAT, adaptive NS, layer-wise quant | ~1.11-1.12 | To test |

---

## Tier 0: Port to 8xH100 (prerequisite for everything)

### P1. Port Best Config to PyTorch [KNOWN]
**What**: Port our 3 code changes to `train_gpt.py`: (1) Muon WD + warmdown schedule, (2) FP16 tok_emb, (3) freq-decomposed skip gating. Set NUM_LAYERS=11, MLP_MULT=3, GRAD_CLIP=0.5, MUON_WEIGHT_DECAY=0.10.
**Priority**: BLOCKING — everything else depends on this.
**Effort**: Medium. 3 focused code ports.

---

## Tier 1: High Priority — Original Ideas

### 1. Warmdown-Phase QAT (Quantization-Aware Training) [**ORIGINAL**]
**Hypothesis**: Integrate int6 quantization noise into our warmdown schedule rather than running QAT as a separate phase. During warmdown, every K steps, replace weights with `dequant(quant_int6(w))`. Three forces act in concert: ↓LR (finer adjustments), ↑WD (push weights toward zero), and quant noise (teach model to tolerate int6 rounding).
**Why original**: Standard QAT is a separate training phase with its own LR schedule. Our QAT is co-designed with warmdown-aware WD — the three forces are synchronized, not independent. Nobody in competition does this.
**Why high priority**: Int6 is what gets submissions from ~1.17 to ~1.13. Our warmdown integration could do it better.
**Expected impact**: -0.01 to -0.03 BPB.
**Effort**: Medium. ~30 lines: int6 quant/dequant functions + injection in warmdown.
**Test on Mac**: Yes — can validate the mechanism at int8 scale first.

### 2. Adaptive Newton-Schulz Scheduling [**ORIGINAL**]
**Hypothesis**: Muon's Newton-Schulz always runs 5 iterations. Schedule based on training phase: 3 steps early (don't over-orthogonalize noise), 5 mid-training (standard), 7 during warmdown (precise conditioning for fine convergence).
**Why original**: Everyone uses fixed NS steps. Scheduling them is unexplored.
**Expected impact**: -0.005 to -0.01 BPB.
**Effort**: Very low. 3 lines of code.
**Test on Mac**: Yes — trivial to test.

### 3. Layer-Wise Quantization Budget Allocation [**ORIGINAL**]
**Hypothesis**: Instead of uniform int6 everywhere, measure quantization sensitivity per layer (how much val_loss degrades when only that layer is quantized). Give sensitive layers int8, insensitive layers int5/int4. Maximize effective capacity in 16MB.
**Why original**: Mixed-precision approaches exist but decide by architecture position. Ours is empirically data-driven — we measure and allocate.
**Expected impact**: -0.01 to -0.02 BPB via more effective parameter use.
**Effort**: Medium. Analysis script + modified quantization.
**Test on Mac**: Yes — the analysis part. Apply on H100.

### 4. Depth-Recurrent Warmdown [**ORIGINAL**, HIGH RISK]
**Hypothesis**: During warmdown, gradually tie adjacent layer pairs: blend `w_layer_i` toward `w_layer_{i+1}` with increasing strength. By end of warmdown, pairs share ~50% of weights. Creates a quasi-recurrent transformer that compresses dramatically (shared weights = huge zlib win) while preserving most performance.
**Why original**: Depth recurrence is a training-time architecture choice. Making it a warmdown regularization technique is entirely novel — train deep, converge to recurrent.
**Expected impact**: -0.01 to -0.02 BPB + major compression gains.
**Effort**: High. Complex to implement correctly.
**Risk**: High — might catastrophically degrade like EMA blending. Start with very gentle blending (α=0.1).
**Test on Mac**: Yes — the mechanism. Small scale first.

---

## Tier 2: Foundation + Known Wins (for 8xH100)

### 5. SWA (Stochastic Weight Averaging) [KNOWN]
**Hypothesis**: Average the last N checkpoints during warmdown. Unlike our FAILED EMA blending (exp_018), SWA averages discrete checkpoints at the END, not a running average blended during training. This is fundamentally different and proven to work.
**Why different from killed EMA**: EMA blending interpolated toward stale averaged weights during training → catastrophic. SWA just averages final checkpoints after training → safe.
**Expected impact**: -0.005 to -0.01 BPB.
**Effort**: Low. Already in train_gpt.py, just port.

### 6. Int6 Quantization [KNOWN]
**Hypothesis**: 6-bit quantization for MLP/attention weights. Fits ~40% more effective params in 16MB.
**Expected impact**: -0.01 to -0.02 BPB (via more capacity).
**Effort**: Medium. Port int6 quant from competition PRs.

### 7. TTT LoRA at Eval Time [KNOWN]
**Hypothesis**: Per-document LoRA adaptation during evaluation. Already implemented in train_gpt.py.
**Expected impact**: -0.01 to -0.02 BPB.
**Effort**: Low. Already in codebase, just enable.

### 8. 11 Layers [KNOWN]
**Hypothesis**: 11L is better per-step. On 8xH100 where step time is batch-dominated, the extra layer is nearly free.
**Expected impact**: -0.01 to -0.02 BPB.
**Effort**: Env var only.

### 9. Zstd Compression (replace zlib) [KNOWN]
**Hypothesis**: Zstd at level 22 compresses ~10-15% better than zlib level 9. More compression = more params in 16MB.
**Expected impact**: +0.5-1MB headroom.
**Effort**: Low. Swap compressor.

---

## Tier 3: Speculative / Eval-Time

### 10. Cascaded TTT with Document Entropy Priors [**ORIGINAL**]
**Hypothesis**: Before TTT adaptation per document, initialize LoRA based on document's token entropy. High-entropy (diverse vocab) documents get different init than low-entropy (repetitive) ones. Gives TTT a head start.
**Expected impact**: -0.005 BPB over vanilla TTT.
**Effort**: Medium.

### 11. Cross-Layer KV Sharing [**OUR TWIST**]
**Hypothesis**: Share K/V projections across groups of 2-3 consecutive layers. Dramatically reduces params (saves artifact bytes for more capacity elsewhere).
**Expected impact**: -0.005 BPB + 1-2MB saved.
**Effort**: Medium.

### 12. Per-Layer Learning Rate Decay [**OUR TWIST**]
**Hypothesis**: `lr_layer_i = base_lr * decay^(num_layers - i)`. Later layers get higher LR. Used in fine-tuning but never with Muon.
**Expected impact**: -0.003 to -0.005 BPB.
**Effort**: Low.

### 13. Cyclical Batch Size [**ORIGINAL**]
**Hypothesis**: Cycle batch between 256K and 1M every ~200 steps. Small batches explore, large batches exploit. Same total token budget.
**Expected impact**: -0.003 to -0.005 BPB.
**Effort**: Low.

---

## Completed (lab/mar26b)

- ~~10-layer architecture~~ [KNOWN] — DONE: -0.05 BPB
- ~~Muon weight decay=0.02~~ [KNOWN] — DONE: -0.031 BPB, -1.1MB artifact
- ~~Muon weight decay=0.05, 0.10~~ [SWEEP] — DONE: -0.099 BPB total, 11.0MB artifact
- ~~Warmdown-aware WD scheduling~~ [**ORIGINAL**] — DONE: -0.110 BPB total, 9.9MB artifact
- ~~FP16 tied embeddings~~ [KNOWN] — DONE: quant gap +0.0001
- ~~Sliding window eval~~ [KNOWN] — DONE: implemented, EVAL_STRIDE=64
- ~~Warmdown tuning~~ [SWEEP] — TESTED: warmdown=400 failed. Keep 1200.
- ~~11 layers (Mac)~~ [SWEEP] — TESTED: worse on Mac (slower steps), better per-step.
- ~~Frequency-decomposed skip gating~~ [**ORIGINAL**] — DONE: -0.010 BPB
- ~~MLP_MULT=3~~ [KNOWN] — DONE: -0.003 BPB. Free capacity win.
- ~~Gradient clipping=0.5~~ [SWEEP] — DONE: -0.016 BPB. Sweet spot (clip=0.25 too aggressive).
- ~~Batch scaling~~ [SWEEP] — DONE: batch=24576 optimal on Mac. Marginal returns above this.

---

## Killed Ideas (Evidence Against)

- **Warmdown < 1200**: Warmdown=400 was strictly worse. Long warmdown is beneficial.
- **INT8_KEEP_FLOAT_MAX_NUMEL increase**: Accidentally keeps all tensors as FP16. Use name patterns.
- **MoE at 16MB**: Research shows unviable below 500M params.
- **Label smoothing**: Failed in systematic competition testing.
- **Concurrent runs on Apple Silicon**: 2-3x throughput degradation.
- **Muon Momentum Warmdown Ramp**: exp_016 +0.035 BPB regression. High momentum destabilizes Newton-Schulz.
- **EMA Warmdown Blending**: exp_018 catastrophic. EMA too stale, destroys convergence. (Note: SWA is DIFFERENT — averaging final checkpoints, not blending during training.)
- **ROPE_BASE=50000**: exp_020 +0.021 BPB regression.
- **QK_GAIN_INIT=1.0**: exp_021 +0.014 BPB regression.
- **Shorter warmdown at large batch**: exp_023 +0.029 worse.
- **DropHead**: exp_025/026. Gradient noise hurts when WD is already strong.
- **batch=32k with MLP3x**: exp_028 too slow.
- **11L+MLP3x on Mac**: exp_030 too heavy for Apple Silicon.
- **WD=0.15/0.20**: Over-regularized.
- **batch=16k with MLP3x**: exp_031 gradient quality too poor.
- **Grad clip=0.25**: exp_034 too aggressive, clips useful gradients.
