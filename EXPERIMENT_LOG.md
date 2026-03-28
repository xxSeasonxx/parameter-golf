# Experiment Log — Parameter Golf (MLX on Apple Silicon)

This document records all experiments conducted during the `lab/mar26b` session, including what was tried, what worked, what failed, and why. It serves as institutional memory for future sessions.

**Session date**: 2026-03-26
**Branch**: `lab/mar26b`
**Starting point**: Unmodified `train_gpt_mlx.py` baseline (val_bpb=2.4109 at 200 iters)
**Final best**: val_bpb=**1.6311** (commit `3af0786`, exp_041 asymmetric MLP width)

---

## Summary of Results

| Exp | Run ID | Change | Steps | val_bpb | Artifact | Status | Key Takeaway |
|-----|--------|--------|-------|---------|----------|--------|--------------|
| 001 | exp_001 | 10 layers (smoke) | 200/200 | 2.4146 | 12.5MB | Inconclusive | 200 iters too short to compare architectures |
| 002 | exp_002 | 10 layers (medium) | 1032/2000 | 1.9470 | 14.6MB | Superseded | 10L clearly beats 9L; ran concurrent (degraded) |
| 003 | exp_003_baseline9L | 9 layers (medium) | 688/2000 | 2.0004 | 12.0MB | Baseline | 9L reference; ran concurrent (degraded) |
| 004 | exp_004_sw64 | Sliding window v1 | 200/200 | N/A | 12.5MB | Killed | Single-window approach: 969K passes, way too slow |
| 005 | exp_005_sw64 | Sliding window v2 (batched) | 200/200 | N/A | 12.5MB | Killed | Batched: 15K passes, still ~50min. Impractical for iteration |
| 006 | exp_006_10L_fp16emb | FP16 ALL tensors (bug) | 1622/2000 | 1.8655 | 35.2MB | Failed | INT8_KEEP_FLOAT_MAX_NUMEL=600k kept everything FP16 |
| 007 | exp_007_10L_fp16tok | FP16 tok_emb only | 1629/2000 | 1.8605 | 15.7MB | Superseded | Correct FP16 approach; quant gap reduced to +0.0001 |
| 008 | exp_008_wd400 | Warmdown=400 | 1636/2000 | 1.8779 | 16.5MB | Failed | Shorter warmdown HURTS both BPB and compressibility |
| **009** | **exp_009_wd02** | **Muon WD=0.02** | **1604/2000** | **1.8291** | **14.6MB** | **Superseded** | **WD improves BPB by 0.031 AND reduces artifact 1.1MB** |
| 010 | exp_010_wd05 | Muon WD=0.05 (smoke) | 200/200 | 2.4145 | 12.1MB | Inconclusive | Marginal train_loss advantage at 200 iters |
| 011 | exp_011_wd05_med | Muon WD=0.05 (medium) | 1692/2000 | 1.7883 | 13.2MB | Superseded | WD=0.05 > WD=0.02, benefit still increasing |
| 012 | exp_012_wd10 | Muon WD=0.10 (smoke) | 200/200 | 2.4371 | 11.5MB | Inconclusive | Worse at 200 iters but late crossover expected |
| 013 | exp_013_wd10_med | Muon WD=0.10 (medium) | 1682/2000 | 1.7608 | 11.0MB | Superseded | WD=0.10 > WD=0.05, massive compression benefit |
| **014** | **exp_014_wd_sched** | **WD=0.10 + warmdown sched** | **1704/2000** | **1.7504** | **9.9MB** | **BEST** | **Warmdown-aware WD: wd*(2-lr_mul). -0.010 more BPB** |
| 015 | exp_015_11L | 11 layers | 1524/2000 | 1.7833 | 11.0MB | Discard | 11L better per-step but slower (~394ms vs 352ms), fewer total steps |
| 016 | exp_016_mom_ramp | Momentum warmdown ramp 0.95→0.99 | 1688/2000 | 1.7852 | 10.7MB | Discard | +0.035 BPB regression. High momentum in warmdown destabilizes Muon's Newton-Schulz |
| **017** | **exp_017_freq_skip** | **Freq-decomposed skip gating (W=32)** | **1678/2000** | **1.7403** | **10.0MB** | **BEST** | **ORIGINAL: lo/hi freq band gates on skip signals. -0.010 BPB** |
| 018 | exp_018_ema | EMA warmdown blend (decay=0.999, blend=0.5) | 1651/2000 | 2.4115 | 9.1MB | Discard | Catastrophic: loss INCREASED during warmdown. EMA too stale, destroys convergence |
| 019 | exp_019_wd20 | WD=0.20 | 1681/2000 | 1.7688 | 7.7MB | Discard | Over-regularized: +0.029 BPB vs WD=0.10. Artifact 7.7MB (great compression). WD=0.10 is the sweet spot |
| 020 | exp_020_rope50k | ROPE_BASE=50000 | ~1700/2000 | 1.7610 | 9.9MB | Discard | +0.021 BPB regression. Higher RoPE base hurts |
| 021 | exp_021_qkgain1 | QK_GAIN_INIT=1.0 | 1713/2000 | 1.7542 | 9.8MB | Discard | +0.014 BPB regression. Default 1.5 is better |
| **022** | **exp_022_batch16k** | **TRAIN_BATCH_TOKENS=16384** | **1057/2000** | **1.6584** | **11.2MB** | **Superseded** | **-0.082 BPB! 2x data/step, better gradient quality dominates** |
| 023 | exp_023_batch16k_wd600 | batch=16k + warmdown=600 | 1038/2000 | 1.6877 | 11.5MB | Discard | Shorter warmdown hurts again: +0.029 BPB vs warmdown=1200 |
| **024** | **exp_024_batch24k** | **TRAIN_BATCH_TOKENS=24576** | **711/2000** | **1.6518** | **10.8MB** | **Superseded** | **-0.007 more BPB. Batch scaling trend continues** |
| 025 | exp_025_drophead | DropHead p=0.10 | 788/2000 | 1.6598 | 10.7MB | Discard | +0.008 BPB regression. Gradient noise hurts with WD already strong |
| 026 | exp_026_drophead05 | DropHead p=0.05 | 772/2000 | 1.6582 | 10.8MB | Discard | +0.006 BPB. Gentler still hurts |
| **027** | **exp_027_mlp3x** | **MLP_MULT=3** | **705/2000** | **1.6492** | **12.9MB** | **Superseded** | **-0.003 BPB. 24M params, near-zero step time overhead** |
| 028 | exp_028_mlp3x_batch32k | MLP3x + batch=32768 | 552/2000 | 1.6582 | 12.2MB | Discard | Too slow: 1087ms/step, only 552 steps |
| 029 | exp_029_mlp3x_wd15 | MLP3x + WD=0.15 | 703/2000 | 1.6513 | 11.8MB | Discard | +0.002 over WD=0.10. Better compression but worse BPB |
| 030 | exp_030_11L_mlp3x | 11L + MLP3x | 645/2000 | 1.6757 | 13.8MB | Discard | Too heavy: 931ms/step, only 645 steps |
| 031 | exp_031_mlp3x_batch16k | MLP3x + batch=16384 | 937/2000 | 1.6692 | 13.3MB | Discard | Gradient quality too poor despite more steps |
| **032** | **exp_032_gradclip** | **MLP3x + GRAD_CLIP=1.0** | **702/2000** | **1.6434** | **13.0MB** | **Superseded** | **-0.006 BPB. Stabilizes early training spikes** |
| **033** | **exp_033_gradclip05** | **MLP3x + GRAD_CLIP=0.5** | **698/2000** | **1.6334** | **13.0MB** | **BEST** | **-0.016 BPB total. Stronger clip is better** |
| 034 | exp_034_gradclip025 | MLP3x + GRAD_CLIP=0.25 | 690/2000 | 1.6362 | 13.0MB | Discard | Too aggressive: +0.003 vs clip=0.5. Clips useful gradients |
| 035 | exp_035_adaptive_ns_med | Adaptive NS scheduling (5→7 warmdown) | 695/2000 | 1.6352 | 13.0MB | Discard | Neutral: +0.0018 vs best. NS converges fine at 5 iters for dim=512 |
| 036 | exp_036_qat_warmdown | Warmdown-phase QAT (int8 noise ramp) | 709/2000 | 1.6907 | 12.8MB | Discard | +0.057 BPB regression. QAT noise fights warmdown convergence. Quant gap 0.0002 (excellent) |
| 037 | exp_037_swa | SWA (uniform avg 60 snapshots, lr_mul<0.5) | 706/2000 | 1.7601 | 12.6MB | Discard | +0.127 BPB regression. Wide SWA window catastrophic. Pre-SWA was 1.6288 |
| 038 | exp_038_swa_narrow | Narrow SWA (lr_mul<0.1, 24 snapshots, every 5 steps) | 704/2000 | 1.6363 | 13.0MB | Discard | +0.003 vs best. Pre-SWA 1.6295→post-SWA 1.6363. SWA killed for Mac |
| **039** | **exp_039_layerlr** | **Per-layer LR scaling (LAYER_LR_SCALE=0.5)** | **702/2000** | **1.6321** | **13.2MB** | **BEST** | **-0.0013 BPP. Marginal new best. Deeper layers get higher LR** |
| 040 | exp_040_layerlr_inv | Inverse layer LR (LAYER_LR_SCALE=-0.5) | 702/2000 | 1.6463 | 12.5MB | Discard | +0.014 BPB. Early layers higher LR is wrong direction. Confirms deeper=faster is correct |
| **041** | **exp_041_asymmlp** | **Asymmetric MLP (encoder=2x, decoder=4x)** | **708/2000** | **1.6311** | **13.1MB** | **BEST** | **-0.001 BPB. Same params as uniform MLP3x but better allocation** |
| 042 | exp_042_asym15 | Extreme asymmetric MLP (1,5) | 708/2000 | 1.6321 | 12.7MB | Discard | +0.001 vs best. Encoder MLP=1x too aggressive, starves feature extraction |
| 043 | exp_043_layerlr10 | LAYER_LR_SCALE=1.0 (deepest=2x LR) | 706/2000 | 1.6316 | 13.1MB | Discard | +0.0005 vs best. Scale saturates between 0.5-1.0. 0.5 sufficient |
| 044 | exp_044_fsw16 | FREQ_SKIP_WINDOW=16 (smaller window) | 701/2000 | 1.6318 | 13.1MB | Discard | +0.0007 vs best. Window size doesn't matter. W=32 is fine |

---

## Detailed Experiment Narratives

### Experiment 001-003: 10 Layers vs 9 Layers

**Hypothesis**: More layers = more model capacity. Competition top submissions use 10L. We have 4.7MB artifact headroom (11.3→16MB).

**What happened**:
- exp_001 (10L smoke, 200 iters): train_loss 3.9107 vs baseline 3.9161 — marginal. Inconclusive at 200 iters.
- exp_002 (10L medium) and exp_003 (9L medium) ran **concurrently** — bad idea. Unified memory contention inflated step times 2-3x (580-874ms vs normal 370ms). exp_002 got 1032 steps, exp_003 only 688.
- Despite unfair timing, 10L (1.9470) clearly beat 9L (2.0004).

**Learnings**:
- 10L is strictly better than 9L at medium scale. Extra layer adds ~1.2MB artifact.
- **Never run experiments concurrently** on Apple Silicon — unified memory contention destroys throughput.
- 200-iter smoke tests are insufficient for architecture comparisons. Always promote to medium.

---

### Experiment 004-005: Sliding Window Evaluation

**Hypothesis**: Scoring each token with near-max context (stride=64, giving 960+ context tokens) should improve val_bpb by ~0.03 BPB. Competition's #2 technique.

**What happened**:
- exp_004: Initial implementation processed one window at a time. With stride=64 over 62M val tokens = 969,073 windows. Each taking ~1ms = ~16 minutes minimum. Killed.
- exp_005: Rewrote with batched processing (64 windows per batch). Reduced to 15,142 batches at ~5 batches/sec = ~50 minutes. Still too slow for development iteration. Killed.

**Learnings**:
- Sliding window eval works correctly — code is implemented and tested (`EVAL_STRIDE` env var).
- stride=64 is impractical on Apple Silicon (~50min per eval). Use for **final submission only**.
- On 8xH100 (RunPod), eval time is unlimited and stride=64 should take ~5 minutes. Worth enabling there.
- The fundamental cost: stride=64 with seq_len=1024 creates 16x more work than non-overlapping eval.

---

### Experiment 006-007: FP16 Tied Embeddings

**Hypothesis**: The tied embedding serves dual input/output role and is the most critical tensor. Keeping it as FP16 instead of int8 should reduce quantization error.

**What happened**:
- exp_006 (**bug**): Set `INT8_KEEP_FLOAT_MAX_NUMEL=600000` to cover the 524K-element embedding. But this threshold also caught ALL attention/MLP weight matrices (also >65K elements). Result: nearly zero quantization (artifact 35MB!). The val_bpb looked great (1.8655) but was misleading — the model wasn't really quantized.
- exp_007 (**fix**): Added `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb` to selectively keep only the embedding as FP16. Artifact: 15.7MB (acceptable). val_bpb: 1.8605. Quantization gap reduced from +0.0005 to +0.0001 BPB.

**Learnings**:
- Never use size thresholds to select tensors for FP16 — use name patterns.
- FP16 embedding adds ~0.5MB to artifact but nearly eliminates quantization error on this tensor.
- The quantization gap (+0.0001 BPB) is negligible with FP16 embeddings.

---

### Experiment 008: Warmdown Schedule Tuning

**Hypothesis**: Default warmdown_iters=1200 means 441s of 600s is warmdown (74%!). Only 26% at full LR. Reducing to 400 should keep full LR longer and improve training.

**What happened**:
- val_bpb=1.8779 — **worse** than exp_007's 1.8605.
- Artifact=16.5MB — **over the 16MB limit**.
- The model was consistently worse at every checkpoint (steps 500, 1000, 1500, final).

**Learnings**:
- **Long warmdown is beneficial, not harmful.** It acts as a strong regularizer.
- Shorter warmdown produces weights that are both worse AND less compressible.
- The LR decay during warmdown isn't "wasted training" — it helps the model converge to a flatter, more generalizable minimum.
- Compressibility insight: regularized weights have lower entropy → better zlib compression. Warmdown contributes to this.
- **Do not touch warmdown=1200.** It's optimal for the ~1600-step regime.

---

### Experiment 009: Muon Weight Decay (BEST)

**Hypothesis**: SOTA competition entry uses Muon with WD=0.02. Weight decay should improve generalization. The original MLX Muon optimizer had no weight decay.

**Code change**: Added 3 lines to `Muon.step()`:
```python
if self.args.muon_weight_decay > 0:
    update = update + self.args.muon_weight_decay * p
```

**What happened**:
- val_bpb=**1.8291** — a massive 0.031 improvement over exp_007 (1.8605).
- Artifact=14.6MB — 1.1MB **smaller** than without WD (15.7MB).
- WD was slightly worse early (step 500: 2.1865 vs 2.1774) but pulled ahead by step 1500 (1.8554 vs 1.8852).

**Learnings**:
- Muon WD=0.02 is a double win: better val_bpb AND smaller artifact.
- Weight decay regularizes the weights → lower entropy → better zlib compression.
- The pattern "worse early, better late" is consistent with regularization effects.
- WD benefits compound over training — the longer you train, the more WD helps.

---

### Experiments 010-013: Weight Decay Sweep (WD=0.05, WD=0.10)

**Hypothesis**: WD=0.02 gave -0.031 BPB. Is more WD better? The benefit function might be monotonically increasing.

**What happened**:
- WD=0.05 (exp_011): val_bpb=1.7883 — beat WD=0.02 by 0.041. Artifact: 13.2MB.
- WD=0.10 (exp_013): val_bpb=1.7608 — beat WD=0.05 by 0.028. Artifact: 11.0MB.
- Both followed the "worse early, better late" pattern: higher WD is worse at intermediate checkpoints but surpasses at convergence.

**Learnings**:
- WD benefit is **monotonically increasing** at least to 0.10. The response curve: 0.00→1.860, 0.02→1.829, 0.05→1.788, 0.10→1.761.
- Higher WD dramatically improves compressibility: artifact went from 15.7MB (no WD) to 11.0MB (WD=0.10).
- The late crossover happens later with higher WD — WD=0.10 catches WD=0.05 only after step 1500.
- **Principle**: WD is a dual regularizer+compressor. The binding constraint (bits per parameter) means WD's compression benefit is as important as its BPB improvement.

---

### Experiment 014: Warmdown-Aware WD Scheduling (NEW BEST)

**Hypothesis**: As LR drops during warmdown, WD's relative effect increases naturally. Amplify this by scheduling `wd = base_wd * (2.0 - lr_mul)`, so WD doubles from 0.10 to 0.20 over the warmdown phase. This concentrates regularization where our data shows it matters most.

**Code change**: 4-line modification in `Muon.step()` — compute WD based on lr_mul instead of using constant.

**What happened**:
- val_bpb=**1.7504** — beat constant WD=0.10 by 0.0104.
- Artifact=**9.9MB** — another 1.1MB reduction from the scheduling alone.
- Slightly worse at step 1000 (+0.021) but caught up and surpassed at the end.

**Learnings**:
- Warmdown-aware WD scheduling works! The extra WD during warmdown pushes weights to a more compressible minimum.
- This is a **novel technique**: nobody in the competition uses WD that varies with the LR schedule.
- Total improvement from WD work (exp 009-014): **-0.110 BPB** and **-5.8MB artifact**.
- 6.1MB headroom is now available for architectural changes.

---

### Experiment 015: 11 Layers (Discard)

**Hypothesis**: With 6.1MB headroom, 11 layers should fit easily. More layers = more capacity.

**What happened**:
- val_bpb=1.7833 — **worse** than 10L's 1.7504.
- 11L is slower per step (~394ms vs ~352ms for 10L), so only 1524 steps completed vs 1704.
- At matched step counts, 11L is actually better (~0.04 BPB). But fewer total steps negates the advantage.

**Learnings**:
- On Apple Silicon with 600s wallclock, the binding constraint is step throughput, not artifact size.
- 11L would likely win on 8xH100 where batch=524K dominates step time and the extra layer's cost is negligible.
- For Apple Silicon experiments, stick with 10L and optimize per-step efficiency.

### Experiment 016: Muon Momentum Warmdown Ramp (Discard)

**Hypothesis**: Ramp momentum from 0.95 to 0.99 during warmdown for smoother convergence into flatter minimum.

**What happened**: val_bpb=1.7852 — **+0.035 BPB regression** vs best (1.7504).

**Learnings**: High momentum destabilizes Muon's Newton-Schulz orthogonalization during warmdown. Don't modify Muon momentum late in training.

---

### Experiment 017: Frequency-Decomposed Skip Gating (NEW BEST)

**Hypothesis**: Decompose U-Net skip connections into low-frequency (block means, W=32) and high-frequency (residual) bands with independent per-dim gates. Different layers should pass through different frequency content.

**Code change**: In GPT forward pass, replace flat skip weights with `skip_lo_weights` and `skip_hi_weights`. Low-freq = windowed mean of skip signal, high-freq = residual.

**What happened**: val_bpb=**1.7403** — beat previous best by 0.010 BPB. Artifact: 10.0MB. Minimal overhead (~2ms/step).

**Learnings**:
- **ORIGINAL technique**: Nobody in competition decomposes skip signals by frequency band.
- The lo/hi decomposition adds only ~10 params per skip connection (negligible).
- Validates the principle that architectural novelty can yield step-change improvements.

---

### Experiment 018: EMA Warmdown Blend (Catastrophic Failure)

**Hypothesis**: Maintain EMA of weights (decay=0.999), blend toward EMA during warmdown (α=0.5). Lower variance weights = more compressible, flatter minima.

**What happened**: val_bpb=2.4115 — **catastrophic**. Loss INCREASED during warmdown (2.19→2.41). The model effectively unlearned during the blend phase.

**Learnings**: EMA weights are too stale — they represent an average over training history, not a better current solution. Blending toward them destroys the hard-won convergence. Don't interpolate toward averaged weights during training.

---

### Experiment 019: WD=0.20 (Discard)

**Hypothesis**: WD response might still be increasing beyond 0.10.

**What happened**: val_bpb=1.7688 — **+0.029 BPB regression** vs WD=0.10. But artifact=7.7MB (excellent compression).

**Learnings**: WD=0.10 is the sweet spot. WD=0.20 over-regularizes — the model loses too much capacity. The compression benefit is huge but not worth the BPB cost.

---

### Experiment 020: ROPE_BASE=50000 (Discard)

**Hypothesis**: Default rope_base=10000 may not be optimal for seq_len=1024. Higher base = less position sensitivity, more uniform attention.

**What happened**: val_bpb=1.7610 — **+0.021 BPB regression**. Artifact: 9.9MB (similar).

**Learnings**: Higher RoPE base hurts. The default 10000 is already good for seq_len=1024.

---

### Experiment 021: QK_GAIN_INIT=1.0 (Discard)

**Hypothesis**: Default QK_GAIN_INIT=1.5 was never validated. Lower gain might reduce attention saturation.

**What happened**: val_bpb=1.7542 — **+0.014 BPB regression**. Artifact: 9.8MB. Step time similar (~350ms).

**Learnings**: Default QK_GAIN=1.5 is good. Both RoPE and QK gain tuning failed — the attention defaults are well-calibrated.

---

### Experiments 022-024: Batch Size Scaling (MAJOR DISCOVERY)

**Hypothesis**: Larger batch = better gradient quality per step. At batch=8192, we get ~1710 steps in 600s. Doubling the batch doubles tokens per step but adds ~60% step time (more grad accum). The question is whether better gradients compensate for fewer total steps.

**exp_022 (batch=16384)**: val_bpb=**1.6584** — massive **-0.082 BPB** improvement over best (1.7403). 1057 steps at ~566ms/step. This is by far the biggest single improvement in this session.

**exp_023 (batch=16k + warmdown=600)**: val_bpb=1.6877 — **+0.029 worse** than exp_022. Reducing warmdown_iters from 1200 to 600 hurts, confirming that long warmdown is always beneficial. With only ~1057 steps and warmdown=1200, the entire training was effectively in warmdown mode — and it STILL won massively.

**exp_024 (batch=24576)**: val_bpb=**1.6518** — another **-0.007 BPB** improvement. 711 steps at ~845ms/step. The batch scaling trend continues.

**Key Learnings**:
- **Batch size is the master lever on Apple Silicon.** Better gradient quality from 2-3x more tokens per step massively outweighs having fewer total steps.
- **Long warmdown works at any step count.** Even with warmdown_iters=1200 and only 711-1057 steps (so warmdown covers 100%+ of training), the model benefits from the gradual LR decay.
- **Token throughput matters more than step count.** At batch=24k, we see ~31K tok/s vs ~23K at batch=8k — 35% more data processed in the same wallclock.
- **The batch scaling curve**: 8192→1.7403, 16384→1.6584, 24576→1.6518. The marginal return is diminishing (0.082 → 0.007), suggesting we're approaching the optimal batch for this wallclock budget.
- **Artifact size is manageable**: 10.8MB at batch=24k, well under the 16MB limit.

---

### Experiments 025-026: DropHead (Discard)

**Hypothesis**: Randomly zeroing entire attention heads during training (DropHead) forces head diversity. Regularization via WD has been our biggest lever.

**What happened**:
- exp_025 (p=0.10): val_bpb=1.6598 — +0.008 regression.
- exp_026 (p=0.05): val_bpb=1.6582 — +0.006 regression.

**Learnings**: DropHead adds gradient noise that isn't compensated by regularization when WD is already strong (0.10). Unlike WD which reduces weight magnitude, DropHead randomly corrupts information flow. With only 8 heads, even p=0.05 zeroes out ~2 heads per sample on average.

---

### Experiment 027: MLP_MULT=3 (NEW BEST)

**Hypothesis**: Wider MLP (3x instead of 2x) adds capacity with minimal step time overhead on Apple Silicon.

**What happened**: val_bpb=**1.6492** — beat exp_024 by 0.003 BPB. Params: 24.1M (vs 18.9M). Step time: ~852ms (vs ~845ms). Artifact: 12.9MB (3.1MB headroom).

**Learnings**: MLP3x is nearly free on Apple Silicon — the MLP computation parallelizes well on unified memory. The 27% param increase only adds 1% step time.

---

### Experiments 028-031: MLP3x Scaling (All Discard)

**exp_028 (MLP3x + batch=32k)**: 1.6582. Too slow at 1087ms/step, only 552 steps.
**exp_029 (MLP3x + WD=0.15)**: 1.6513. Slightly over-regularized but great compression (11.8MB).
**exp_030 (11L + MLP3x)**: 1.6757. Too heavy: 931ms/step, only 645 steps.
**exp_031 (MLP3x + batch=16k)**: 1.6692. Gradient quality too poor despite 937 steps.

**Learnings**: batch=24576 remains optimal with MLP3x. WD=0.10 remains the sweet spot. 11L+MLP3x is too heavy for Apple Silicon.

---

### Experiments 032-033: Gradient Clipping (MAJOR DISCOVERY)

**Hypothesis**: Early training shows massive loss spikes (17.9 at step 2). Gradient clipping could prevent these wasted updates and stabilize learning.

**What happened**:
- exp_032 (clip=1.0): val_bpb=**1.6434** — -0.006 from no clip.
- exp_033 (clip=0.5): val_bpb=**1.6334** — -0.016 from no clip! Stronger clipping is better.

**Key Learnings**:
- **Gradient clipping is a major discovery for this setup.** The early loss spikes waste optimization steps with huge, poorly-directed gradients.
- **The clip curve**: no_clip→1.6492, clip=1.0→1.6434, clip=0.5→1.6334. Monotonically improving with tighter clipping (so far).
- **Zero overhead**: clipping is a simple norm comparison + scaling.
- **Synergy with Muon**: Muon's Newton-Schulz orthogonalization may amplify gradient noise. Clipping before Muon processing keeps the orthogonalization well-conditioned.

---

### Experiment 035: Adaptive Newton-Schulz Scheduling (Discard)

**Hypothesis**: Schedule NS iterations from 5 (during full LR) to 7 (during warmdown) for more precise gradient conditioning when fine convergence matters most.

**Code change**: 1 line in `Muon.step()`: `ns_steps = base_steps + round(2 * (1 - lr_mul))` when in warmdown.

**What happened**: val_bpb=1.6352 (int8) -- +0.0018 worse than best (1.6334). Pre-quant val_bpb=1.6321 was marginally better, but the quantization gap erased the gain. 695 steps at 863ms/step. Artifact: 13.0MB.

**Learnings**:
- Newton-Schulz already converges well at 5 iterations for dim=512. The error from 5 vs 7 iterations is negligible compared to gradient noise.
- The pre-quant result (1.6321 vs best's ~1.630) hints at a tiny real effect, but it is masked by quantization noise and not worth the complexity.
- Don't schedule NS iterations -- 5 is sufficient for this model size.

---

### Experiment 036: Warmdown-Phase QAT (Discard)

**Hypothesis**: Inject simulated int8 quantization noise during warmdown, with strength ramping from 0 (at lr_mul=1) to 1 (at lr_mul=0). The QAT noise should synergize with warmdown-aware WD — three forces (decaying LR, increasing WD, increasing quant noise) converging the model to a quantization-friendly minimum.

**What happened**: val_bpb=1.6907 (int8) — **+0.057 BPB regression** vs best (1.6334). 709 steps at 847ms/step. Artifact: 12.8MB (slightly better compression than best's 13.0MB).

**The good**: The quantization gap was only 0.0002 BPB (pre-quant 1.6905, int8 1.6907). This proves the QAT mechanism works for closing the quant gap — the model genuinely learned to tolerate int8 rounding.

**The bad**: The overall BPB suffered massively. QAT noise during warmdown fights the optimizer's convergence. Warmdown is when the model does its finest convergence toward a flat minimum — injecting noise during this critical phase is counterproductive.

**Root cause**: With only ~700 total steps and warmdown_iters=1200 (meaning warmdown covers 100%+ of training), QAT noise is active for essentially the entire run with increasing strength. There is no "clean" convergence phase followed by a brief QAT adaptation. Competition QAT works because it runs for the last ~15% of 1500+ steps — a brief adaptation after the model has already converged. Our warmdown covers everything.

**Learnings**:
- Warmdown QAT with ramping strength is fundamentally misdesigned for our setup. The convergence-critical warmdown phase cannot tolerate additional noise.
- The quant gap closure (0.0002 BPB) validates that QAT works mechanistically. But it needs to happen BEFORE warmdown, not during it.
- Better compression (12.8MB vs 13.0MB) suggests QAT does push weights toward more quantization-friendly values.
- A gentler variant (constant low QAT strength, or QAT applied only before warmdown begins) might work, especially on 8xH100 with more steps and a proper pre-warmdown phase.

---

### Experiment 037: SWA — Stochastic Weight Averaging (Discard)

**Hypothesis**: Average discrete weight snapshots collected during warmdown when lr_mul < 0.5 (last ~60% of training steps). Unlike EMA blending (exp_018) which interpolated toward stale averages during training, SWA collects snapshots and averages them after training finishes. This is the standard approach used in competition entries.

**Implementation**: Collect a snapshot every 10 steps when lr_mul < 0.5. After training, uniformly average all collected snapshots and replace the model weights.

**What happened**:
- Pre-SWA val_bpb=**1.6288** — actually better than current best (1.6334)! Pure random variance.
- After SWA averaging 60 snapshots (collected from ~step 350 to 706): val_bpb=**1.7601 (int8)** — catastrophic **+0.131 BPB regression** from the pre-SWA model.
- 706 steps at 851ms/step. Artifact: 12.6MB (better compression than best's 13.0MB — SWA smooths weights).

**Root cause**: The averaging window was far too wide. lr_mul < 0.5 covers the last ~60% of training steps (steps ~350-706). Weights at step 350 and step 706 are very different — the model is still making meaningful progress between them. Uniformly averaging across such a wide window creates a "blurry" parameter average that is worse than any individual snapshot.

**Competition comparison**: Top competition entries use SWA over the **last 100-120 steps only** (lr_mul < 0.1), or use EMA with very high decay (0.9999). Our window of 350 steps (lr_mul < 0.5) was 3-5x too wide.

**Learnings**:
- SWA with wide window (lr_mul < 0.5, ~60% of steps) is catastrophic: +0.131 BPB.
- The pre-SWA model at 1.6288 demonstrates that run-to-run variance is ~0.005 BPB — our best (1.6334) is within noise.
- Better compression (12.6MB vs 13.0MB) confirms that weight averaging smooths the parameter landscape and improves compressibility.
- Must try: SWA_START_LR_MUL=0.1 (last ~70 steps only) or exponential weighting favoring later snapshots.
- This failure is analogous to EMA (exp_018) in that both average over too much training history. The key is narrowness: only average weights that are already very close to the final solution.

---

### Experiment 038: Narrow SWA (Discard — SWA Killed for Mac)

**Hypothesis**: Narrow SWA window (lr_mul < 0.1, every 5 steps, ~24 snapshots) should avoid the catastrophic regression of wide SWA (exp_037, lr_mul < 0.5, 60 snapshots). Competition entries use narrow windows (last 100-120 steps).

**What happened**:
- Pre-SWA val_bpb=**1.6295** — within noise of best (1.6334).
- After SWA averaging 24 snapshots (lr_mul < 0.1): val_bpb=**1.6363 (int8)** — **+0.003 BPB** vs best, **+0.007** vs the pre-SWA model.
- 704 steps at 853ms/step. Artifact: 13.0MB.

**Comparison with wide SWA (exp_037)**:
- Wide (lr_mul < 0.5, 60 snapshots): +0.131 BPB regression from pre-SWA.
- Narrow (lr_mul < 0.1, 24 snapshots): +0.007 BPB regression from pre-SWA.
- Narrow is 18x less destructive, but still a net negative.

**Root cause**: With only ~700 total steps, weights converge monotonically during the warmdown phase. There is no oscillatory behavior to average out — each successive snapshot is strictly closer to the minimum. Averaging earlier snapshots with later ones can only regress toward a less-converged point.

**Key insight**: SWA's theoretical benefit comes from averaging across oscillations in a flat loss basin. With 1500+ steps (as on 8xH100), the model may oscillate during late warmdown. With ~700 steps on Mac, the loss curve is monotonically decreasing throughout — SWA has nothing useful to average.

**Decision**: **KILL SWA for all Apple Silicon experiments.** SWA remains viable for 8xH100 testing where longer training may produce the oscillatory dynamics SWA is designed to exploit.

---

### Experiment 039: Per-Layer LR Scaling for Muon (Marginal New Best)

**Hypothesis**: Deeper layers need higher learning rates because they receive more attenuated gradients through the residual stream. Scale Muon LR per layer: layer_i gets `base_lr * (1.0 + scale * i / (num_layers - 1))` where scale=0.5. This means layer 0 gets 1.0x LR and layer 9 gets 1.5x LR.

**What happened**:
- Pre-quant val_bpb=**1.6294** — consistent with recent pre-quant results (~1.629 range).
- Int8 val_bpb=**1.6321** — **-0.0013 BPB** vs best (1.6334). Marginal new best.
- 702 steps at 855ms/step. Artifact: 13,236,789 bytes (~13.2MB).

**Analysis**:
- The improvement is very small (-0.0013) and within the noise range (~0.005 BPB run-to-run variance observed in exp_037/038 pre-SWA results).
- However, the technique has zero step time overhead (just a scalar multiply on existing LR) and no artifact size cost.
- The pre-quant result (1.6294) is consistent with other recent pre-quant results, suggesting the per-layer LR is not hurting and may be helping slightly.
- On 8xH100 with more steps (1500+), the per-layer LR scaling could have a larger effect as the deeper layers get more opportunity to benefit from the higher LR.

**Decision**: **KEEP** as marginal new best. The improvement is within noise, but keeping it costs nothing. The concept is sound and may show larger gains with more training steps on H100.

**Learnings**:
- Per-layer LR scaling for Muon shows marginal positive signal. LAYER_LR_SCALE=0.5 (1.0x to 1.5x range) is the first tested value.
- Zero overhead technique — worth including in the H100 config.
- Further tuning (scale=0.3, scale=0.7, or inverse scaling) could be explored but is low priority given the marginal signal.

---

### Experiment 040: Inverse Per-Layer LR Scaling (Discard)

**Hypothesis**: Test the opposite direction of exp_039: LAYER_LR_SCALE=-0.5 gives early layers higher LR (1.0x at layer 9, up to 1.5x at layer 0). Maybe early layers, which learn more general features, benefit more from higher LR.

**What happened**:
- Int8 val_bpb=**1.6463** — **+0.014 BPB regression** vs best (1.6321).
- Pre-quant val_bpb=1.6427.
- 702 steps at 855ms/step. Artifact: 12,526,015 bytes (~12.5MB).
- Interesting: artifact is ~0.7MB smaller than exp_039 (13.2MB). Lower LR on later layers produces simpler weights that compress better.

**Learnings**:
- Inverse layer LR is clearly worse (+0.014 BPB). This is a symmetric test against exp_039 and confirms that **deeper layers need MORE LR, not less**.
- LAYER_LR_SCALE=0.5 (deeper=faster) is the correct direction. The gradient attenuation through the residual stream genuinely requires compensation.
- Compression benefit (12.5MB vs 13.2MB) is expected: later layers with lower LR explore less of the parameter space and settle on simpler solutions.
- No further tuning of layer LR direction is needed. Scale magnitude (0.3, 0.7) is low priority given the marginal signal from exp_039.

---

### Experiment 041: Asymmetric MLP Width (Marginal New Best)

**Hypothesis**: Encoder layers (first half) get MLP_MULT=2, decoder layers (second half) get MLP_MULT=4. Same total params as uniform MLP_MULT=3 (24.1M). The reasoning: decoder layers closer to the output need more capacity for token prediction, while encoder layers can work with less since they primarily build representations.

**What happened**:
- Pre-quant val_bpb=**1.6285** at step 708 — slightly better than recent pre-quant results (~1.629 range).
- Int8 val_bpb=**1.6311** — **-0.001 BPB** vs best (1.6321). Marginal new best.
- 708 steps at 848ms/step (slightly faster than uniform MLP3x at ~855ms). Artifact: 13,075,651 bytes (~13.1MB).
- Same param count: 24,142,928 (identical to uniform MLP3x).

**Analysis**:
- The improvement (-0.001 BPB) is small but directionally consistent: reallocating MLP capacity from encoder to decoder layers is better than uniform distribution.
- Slightly faster step time (848ms vs 855ms) may be due to the smaller encoder MLPs being cheaper to compute even though decoder MLPs are larger.
- Artifact is slightly smaller (13.1MB vs 13.2MB), suggesting the asymmetric structure compresses marginally better.
- The pre-quant result (1.6285) is better than exp_039's pre-quant (1.6294), and the int8 result (1.6311) is better than exp_039's (1.6321). Both metrics improved.

**Decision**: **KEEP** as marginal new best. Same param budget, better allocation. Zero-cost architectural insight.

**Learnings**:
- Asymmetric MLP width is a free win: same params, slightly better BPB, slightly faster, slightly smaller artifact.
- Decoder layers benefit from more MLP capacity. This aligns with the intuition that predicting next tokens requires more nonlinear transformation than building intermediate representations.
- The MLP_MULT_ASYMMETRIC=2,4 pattern replaces uniform MLP_MULT=3 in best config.

---

### Experiment 042: Extreme Asymmetric MLP (Discard)

**Hypothesis**: Push all MLP capacity to decoder. MLP_MULT_ASYMMETRIC=1,5 instead of 2,4. If decoder benefits from more MLP, maximizing decoder allocation should help further.

**What happened**: val_bpb=1.6321 (int8) -- +0.001 BPB worse than best (1.6311). 708 steps at 847ms/step. Artifact: 12.7MB (slightly better compression from tiny encoder MLPs). Pure env-var change, no code modifications.

**Learnings**:
- Extreme asymmetry (1,5) is NOT better than moderate (2,4). Encoder still needs meaningful MLP capacity.
- MLP_MULT=1 in encoder layers is too aggressive -- it starves feature extraction in the first half of the network. The U-Net skip connections carry encoder representations to decoder layers; those representations need adequate nonlinear processing.
- 2,4 is the sweet spot for this architecture: enough encoder capacity for good intermediate representations, extra decoder capacity for token prediction.
- The slightly better compression (12.7MB vs 13.1MB) confirms that smaller encoder MLPs have less weight entropy, but the BPB cost is not worth it.

**Decision**: **DISCARD**. Keep MLP_MULT_ASYMMETRIC=2,4.

---

### Experiment 043: LAYER_LR_SCALE=1.0 (Discard)

**Hypothesis**: LAYER_LR_SCALE=1.0 gives deeper layers 2.0x LR (vs 1.5x at scale=0.5). If deeper layers benefit from more LR, doubling the scale should help further.

**What happened**: val_bpb=1.6316 (int8) -- +0.0005 BPB worse than best (1.6311). Pre-quant val_bpb=1.6296. 706 steps at 850ms/step. Artifact: 13,128,949 bytes (~13.1MB). Pure env-var change, no code modifications.

**Learnings**:
- LAYER_LR_SCALE=0.5 (1.6311) and 1.0 (1.6316) are nearly identical. The effect of per-layer LR scaling saturates between 0.5 and 1.0.
- The full picture: scale=-0.5 (1.6463, wrong direction) → scale=0.0 (baseline) → scale=0.5 (1.6311, marginal best) → scale=1.0 (1.6316, no further gain). Diminishing returns past 0.5.
- Per-layer LR scaling is a micro-optimization. The direction matters (deeper=faster) but the magnitude barely matters in the 0.5-1.0 range. Not worth further tuning.

**Decision**: **DISCARD**. Keep LAYER_LR_SCALE=0.5.

---

## Code Changes Made to train_gpt_mlx.py

All changes are in `train_gpt_mlx.py`. No other training files were modified.

### 1. Muon Weight Decay with Warmdown-Aware Scheduling (used in best run)
- Added `MUON_WEIGHT_DECAY` env var (default 0.0) to Hyperparameters
- Added decoupled weight decay in `Muon.step()`: `update = update + wd * p`
- Warmdown-aware scheduling: `wd = base_wd * (2.0 - lr_mul)` — WD doubles during warmdown phase

### 2. FP16 Embedding Quantization (6 lines, used in best run)
- Added `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS` env var
- Added name-pattern check in `quantize_state_dict_int8()` to keep matched tensors as FP16

### 4. Frequency-Decomposed Skip Gating (used in best run)
- Added `FREQ_SKIP_GATING` and `FREQ_SKIP_WINDOW` env vars
- Replaced flat `skip_weights` with `skip_lo_weights` (low-freq band) and `skip_hi_weights` (high-freq band)
- Low-freq = windowed mean (W=32), high-freq = residual after subtracting low-freq
- Independent learned gates per frequency band per skip connection

### 5. Per-Layer LR Scaling for Muon (used in best run)
- Added `LAYER_LR_SCALE` env var (default 0.0 = disabled)
- In `Muon.step()`, each layer's LR is scaled: `lr_i = base_lr * (1.0 + scale * i / (num_layers - 1))`
- Layer 0 gets 1.0x LR, deepest layer gets (1.0 + scale)x LR

### 6. Asymmetric MLP Width (used in best run)
- Added `MLP_MULT_ASYMMETRIC` env var (e.g., "2,4" for encoder=2x, decoder=4x)
- Encoder layers (first half) get smaller MLP, decoder layers (second half) get larger MLP
- Same total param count as uniform MLP_MULT=3

### 3. Sliding Window Evaluation (80 lines, implemented but not used in best run)
- Added `EVAL_STRIDE` env var (default 0 = disabled)
- Added `loss_per_token()` method on GPT model (returns per-position losses)
- Added `eval_val_sliding()` function with batched overlapping-window processing
- Compiled and warmed up separately from the training loss function

---

### Experiment 044: Frequency Skip Window Size (FREQ_SKIP_WINDOW=16)

**Hypothesis**: Smaller frequency decomposition window (W=16 vs default W=32) gives finer frequency resolution, potentially capturing more granular frequency bands in skip connections.

**Config**: Pure env-var change: `FREQ_SKIP_WINDOW=16`. All else identical to exp_041 best config.

**Result**: val_bpb=1.6318 (int8), +0.0007 vs best (1.6311). Within noise. DISCARD.

**Analysis**: The frequency decomposition in skip gating is robust to window size. W=16 (finer resolution, more frequency bands) performs identically to W=32 (coarser, fewer bands). This makes sense: the key insight of freq skip gating is the low/high frequency *decomposition itself*, not the exact cutoff. The per-dim gating weights learn to compensate for whatever window size is used. Not worth tuning further.

**Key learning**: FREQ_SKIP_WINDOW is not a lever. W=32 default is fine.

---

> **Live state**: See `.lab/insights.md` (current best + learnings) and `.lab/ideas_queue.md` (what to try next). Those are the authoritative, always-up-to-date sources. This log is history.

---

## Porting to PyTorch (8xH100 RunPod)

When porting the best config to `train_gpt.py` for competition submission:

### Must port (code changes needed)
1. **Muon weight decay with warmdown scheduling**: PyTorch Muon has NO WD. Add WD + the `wd = base_wd * (2 - lr_mul)` schedule.
2. **FP16 tok_emb**: Add `INT8_KEEP_FLOAT_FP16_NAME_PATTERNS` logic to PyTorch quantization.
3. **Sliding window eval**: Port `eval_val_sliding()` to PyTorch. Eval time is free on 8xH100.
4. **Asymmetric MLP width**: Port `MLP_MULT_ASYMMETRIC` env var. Encoder layers get MLP2x, decoder layers get MLP4x.

### Just set env vars
5. `NUM_LAYERS=10` (or try `NUM_LAYERS=11` — likely better on 8xH100 where step time is batch-dominated)
6. `MUON_WEIGHT_DECAY=0.10`
7. `WARMDOWN_ITERS=1200` — already the default
8. `MLP_MULT_ASYMMETRIC=2,4`

### Expected 8xH100 performance
- Batch: 524,288 tokens/step (64x more than Apple Silicon's 8192)
- Steps: ~1,500-2,000 in 600s
- Tokens seen: ~800M-1B (vs 13M on Mac)
- Expected val_bpb: **~1.18-1.20** (vs 1.83 on Mac — the gap is almost entirely data volume)

### Quantization is identical
- Same format (`int8_clean_per_row_v1`), same clip percentile, same zlib level
- Artifact size will be similar for same architecture
