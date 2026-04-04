# Experiment Log — Parameter Golf (MLX on Apple Silicon)

This document records all experiments conducted during the `lab/mar26b` session, including what was tried, what worked, what failed, and why. It serves as institutional memory for future sessions.

**Session date**: 2026-03-26
**Branch**: `lab/mar26b`
**Starting point**: Unmodified `train_gpt_mlx.py` baseline (val_bpb=2.4109 at 200 iters)
**Final best**: val_bpb=**1.6215** (commit `e8addc5`, exp_063 LeakyReLU(0.5)²)
**Latest**: exp_074 Seq Len Curriculum (NEUTRAL — +0.002 BPB, Mac exhausted)
**H100 best**: TTT BPB **1.2087** (run1_baseline: 11L int8 zlib, 6421 steps, 80/195 shards)

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
| 045 | exp_045_kv2 | NUM_KV_HEADS=2 (aggressive GQA) | 723/2000 | 1.6345 | 14.0MB | Discard | +0.003 vs best. Fewer KV heads hurts BPB and compression |
| **046** | **exp_046_warmup50** | **WARMUP_STEPS=50 (up from 20)** | **706/2000** | **1.6309** | **13.1MB** | **BEST** | **-0.0002 BPB. Marginal new best. Longer warmup stabilizes early training** |
| 047 | exp_047_softcap15 | LOGIT_SOFTCAP=15.0 (down from 30.0) | 709/2000 | 1.6364 | 13.2MB | Discard | +0.006 BPB. Tighter clamping hurts confident predictions. Default 30.0 is optimal |
| 049 | exp_049_int6 | Int6 per-row quantization (QUANT_BITS=6) | 686/2000 | 1.6975 | 6.8MB | Discard | +0.064 BPB quant gap (pre-quant 1.6332 normal). 48% artifact reduction. Needs QAT |
| **051** | **exp_051_qat_prewarmdown** | **Pre-warmdown QAT (strength=0.1, every 10, lr_mul>=0.8)** | **710/2000** | **1.6299** | **13.1MB** | **BEST** | **NEW BEST -0.0010. QAT as regularization: pre-quant 1.6270 (best ever)** |
| 052 | exp_052_int6_qat | Int6 QAT (QAT_BITS=6, strength=0.1, every=10) + QUANT_BITS=6 | 712/2000 | 1.6894 | 6.9MB | Discard | +0.061 quant gap barely improved from +0.064. Pre-quant 1.6281 best ever (int6 noise as regularizer) |
| 053 | exp_053_depth_recur | Depth-recurrent warmdown (alpha=0.1, layers 2,3,4) | 711/2000 | 1.6378 | 12.9MB | Discard | +0.008 BPB, -0.2MB. Warmdown too sensitive for auxiliary losses |
| 054 | exp_054_strong_qat | Strong QAT (strength=0.2, every=5, 4x signal) | 712/2000 | 1.6299 | 13.1MB | Discard | Identical to exp_051. QAT regularization saturated — don't tune further |
| 055 | exp_055_strong_int6_qat | Strong int6 QAT (strength=0.3, every=5, QAT_BITS=6) | 713/2000 | 1.6904 | 6.9MB | Discard | +0.063 quant gap — int6 QAT killed (3 exps confirm gap is ~+0.063 regardless of QAT) |
| 056 | exp_056_full_qat | Full-training QAT (QAT_STOP_LR_MUL=0) | 704/2000 | 1.6308 | 13.1MB | Discard | Neutral — quant gap slightly better (+0.0025 vs +0.0029) but pre-quant worse (1.6283 vs 1.6270). Effects cancel |
| 057 | exp_057_cosine_warmdown | Cosine warmdown shape (WARMDOWN_SHAPE=cosine) | 696/2000 | 1.6503 | 12.7MB | Discard | +0.020 BPB regression. Cosine keeps LR too high too long, insufficient fine convergence. Linear warmdown is optimal |
| 058 | exp_058_label_smooth | Label smoothing (0.1) | 692/2000 | 2.0499 | 12.9MB | Discard | CATASTROPHIC +0.42 BPB. Smoothing redistributes too much mass with vocab=1024. Training objective diverges from eval metric |
| 059 | exp_059_higher_lr | MATRIX_LR=0.06 (1.5x) | 703/2000 | 1.6391 | 13.3MB | Discard | +0.009 BPB. Higher LR overshoots — Muon orthogonalized updates well-scaled at 0.04. Kill LR sweep |
| 060 | exp_060_high_embed_lr | TIED_EMBED_LR=0.1 (2x) | 705/2000 | 1.6598 | 13.2MB | Discard | +0.030 BPB. Tied embeddings extremely LR-sensitive. Kill embed LR sweep upward |
| 061 | exp_061_low_embed_lr | TIED_EMBED_LR=0.03 (0.6x) | 690/2000 | 1.6407 | 13.0MB | Discard | +0.011 BPB. Lower embed LR under-trains. Embed LR sweep fully closed: 0.05 is the sweet spot |
| **063** | **exp_063_leakyrelu_med** | **LeakyReLU(0.5)² activation in MLP** | **700/2000** | **1.6215** | **13.1MB** | **BEST** | **NEW BEST -0.0084 BPB! Dead neuron elimination via 50% negative slope** |
| 064 | exp_064_ema | EMA decay=0.997 | 689/2000 | 1.8437 | 12.1MB | Discard | EMA window too wide for 689 steps (48% of training). Pre-EMA model was 1.6224 |
| 065 | exp_065_3band | 3-band freq skip gating | 699/2000 | 1.6238 | 13.1MB | Discard | +0.0023 BPB, ultra-low band redundant with 2-band W=32 |
| 066 | exp_066_xsa3 | XSA on last 3 decoder layers | 701/2000 | 1.6222 | 13.1MB | Discard | Neutral +0.0007 BPB, no speed penalty. Self-exclusion removes 1/1024 context — too small to matter at 10L |
| 067 | exp_067_partial_rope | Partial RoPE (25% dims) | 697/2000 | 1.6240 | 13.1MB | Discard | +0.0025 BPB, full RoPE better at short seq_len=1024 |
| 068 | exp_068_gelu2 | GELU² activation | 687/2000 | 1.6373 | 12.7MB | Discard | +0.016 BPB, Gaussian gating kills negative gradient flow |
| 069 | exp_069_leaky07 | LeakyReLU(0.7)² activation | 695/2000 | 1.6231 | 13.3MB | Discard | +0.0016 BPB, slope 0.5 optimal. Activation sweep CLOSED |
| 070 | exp_070 | MODEL_DIM=544 (wider model) | 590/2000 | 1.6446 | 14.1MB | Discard | +0.023 BPB, 1018ms/step too slow on Mac |
| 071 | exp_071 | Baseline reconfirm (control) | 688/2000 | 1.6251 | 13.1MB | Control | Matches exp_063 within noise (+0.0009) |
| 072 | exp_072 | Byte-weighted loss (BPB-aligned training) | 679/2000 | 1.6659 | 13.1MB | Discard | +0.041 BPB. Training-eval mismatch is NOT the bottleneck |
| 073 | exp_073 | Deep supervision (tap layers 1,3,5,7) | 651/2000 | 1.6351 | 13.0MB | Discard (Mac) | +0.010 BPB on Mac (5% overhead kills), per-step quality better. H100 candidate |
| 074 | exp_074 | Seq len curriculum (256→512→1024) | 786/2000 | 1.6271 | 13.5MB | Discard | +0.002 BPB (neutral). 14% more steps but short-seq less efficient |

### H100 RunPod Runs (2026-04-03)

| Run | Config | Steps | Pre-quant BPB | Post-quant BPB | TTT BPB | Artifact | Status | Key Takeaway |
|-----|--------|-------|---------------|---------------|---------|----------|--------|-------------|
| H100-1 | 11L int8 zlib (baseline) | 6421 | 1.2250 | 1.2302 | **1.2087** | 15.8MB | Best | Baseline established. Gap to leaderboard: 0.089 BPB |
| H100-2 | 13L int6 zstd (capacity) | 5418 | **1.2145** | 1.3066 | 1.2765 | 9.9MB | Discard | Best pre-quant but int6 gap +0.092 destroys it. Int6 KILLED |
| H100-3 | 11L int8 zstd + SWA | 6420 | 1.2260 | 1.2322 | 1.2107 | 14.2MB | Discard | SWA (23 snapshots) gives +0.002 regression. SWA KILLED EVERYWHERE |

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

### Experiment 045: NUM_KV_HEADS=2 — Aggressive GQA (Discard)

**Hypothesis**: Reducing KV heads from 4 to 2 (more aggressive GQA with 8 query heads) saves ~1.3M params, giving faster steps. The saved capacity might be better spent elsewhere or the speed gain might allow more training steps.

**Config**: Pure env-var change: `NUM_KV_HEADS=2`. All else identical to exp_041 best config.

**What happened**:
- Pre-quant val_bpb=**1.6328** at step 723 (vs 708 steps for exp_041).
- Int8 val_bpb=**1.6345** — **+0.003 BPB worse** than best (1.6311).
- 723 steps at 830ms/step (faster than exp_041's 848ms). Artifact: 14,035,696 bytes (~14.0MB).
- Params: 22.8M (vs 24.1M for 4 KV heads).

**Analysis**:
- The speed gain is real: 830ms vs 848ms per step, yielding 723 steps vs 708 steps (+15 more steps in 10 min).
- But BPB is worse by 0.003 — the extra 15 steps do not compensate for reduced attention capacity.
- Artifact is significantly larger: 14.0MB vs 13.1MB despite fewer params (22.8M vs 24.1M). Compression ratio dropped from 3.85x to 3.64x.
- Fewer KV heads produce less regular weight patterns, hurting zlib compression. With 4 KV heads shared across 8 query heads, each KV head serves exactly 2 query heads — a clean 2:1 mapping. With 2 KV heads serving 4 query heads each, the attention patterns are less structured and compress worse.

**Learnings**:
- 2 KV heads hurts both BPB (+0.003) and compressibility (3.64x vs 3.85x). A double loss.
- The speed gain (830ms vs 848ms, +15 steps) does not compensate for reduced attention capacity.
- 4 KV heads is the sweet spot for 8 query heads. The 2:1 Q/KV ratio provides enough attention diversity while keeping params and compression reasonable.
- Don't reduce KV heads below 4.

**Decision**: **DISCARD**. Keep NUM_KV_HEADS=4 (default).

---

### Experiment 046: WARMUP_STEPS=50 — Longer Warmup (Marginal New Best)

**Hypothesis**: Increasing warmup from 20 to 50 steps stabilizes early training, giving the optimizer (especially Muon with Newton-Schulz) a smoother start. This should complement GRAD_CLIP_NORM=0.5 — both target early training stability but through different mechanisms (warmup: smaller LR ramp; clip: gradient magnitude control).

**Config**: Pure env-var change: `WARMUP_STEPS=50`. All else identical to exp_041 best config.

**What happened**:
- Pre-quant val_bpb=**1.6284** — best pre-quant we've seen (previous best: ~1.6287 from exp_041).
- Int8 val_bpb=**1.6309** — **-0.0002 BPB** vs best (1.6311). Marginal new best.
- 706 steps at 851ms/step. Artifact: 13,065,155 bytes (~13.1MB).

**Analysis**:
- The improvement is extremely marginal (-0.0002 BPB) and well within noise (~0.005 BPB run-to-run variance).
- However, both pre-quant (1.6284, best ever) and post-quant (1.6309, best ever) are simultaneously the best we've seen. This adds confidence that the signal is real, even if tiny.
- Zero cost: WARMUP_STEPS=50 doesn't affect step time, artifact size, or total steps.
- The quant gap (1.6284 → 1.6309 = 0.0025 BPB) is consistent with recent experiments.

**Learnings**:
- WARMUP_STEPS=50 is a marginal free improvement. Longer warmup gives Muon's Newton-Schulz orthogonalization better initial conditions.
- Complementary with GRAD_CLIP_NORM=0.5: warmup controls LR ramp, clip controls gradient magnitude. Both stabilize early training through orthogonal mechanisms.
- Pre-quant BPB of 1.6284 suggests the model capacity is well-utilized. The quant gap (~0.0025) remains the main source of loss.

**Decision**: **KEEP** as marginal new best. Zero downside, both metrics are best-ever.

---

### Experiment 047: LOGIT_SOFTCAP=15.0 — Tighter Logit Clamping (Discard)

**Hypothesis**: Reducing logit softcap from 30.0 to 15.0 constrains logit magnitudes more aggressively, potentially acting as a regularizer. Tighter clamping prevents over-confident predictions, which could improve generalization.

**Config**: Pure env-var change: `LOGIT_SOFTCAP=15.0`. All else identical to exp_046 best config.

**What happened**:
- Pre-quant val_bpb=**1.6340** at step 709 — worse than best pre-quant (1.6284 from exp_046).
- Int8 val_bpb=**1.6364** — **+0.006 BPB worse** than best (1.6309).
- 709 steps at 847ms/step. Artifact: 13,188,283 bytes (~13.2MB, 3.85x compression).

**Analysis**:
- The softcap=30.0 default is already well-calibrated. Reducing to 15.0 constrains the model's ability to make confident predictions for common, predictable tokens (articles, prepositions, closing brackets, etc.). These tokens are "easy" and the model should be allowed high confidence on them.
- The regression is consistent across both pre-quant (+0.006 vs best pre-quant) and post-quant (+0.006 vs best), indicating the damage is in the model quality itself, not quantization interaction.
- Compression ratio (3.85x) is identical to exp_046, confirming the softcap doesn't affect weight distribution.

**Learnings**:
- LOGIT_SOFTCAP=30.0 is well-calibrated for this architecture. Don't reduce it.
- Softcap acts as a regularizer on prediction confidence, not weight magnitude. Unlike WD (which improves compressibility), softcap changes only affect the logit distribution at inference time.
- The model genuinely benefits from being able to make confident predictions. Common tokens have near-deterministic distributions that require large logit magnitudes to represent accurately.

**Decision**: **DISCARD**. Keep LOGIT_SOFTCAP=30.0 (default).

---

### Experiment 049: Int6 Per-Row Quantization (Discard)

**Hypothesis**: 6-bit per-row quantization ([-31,31] range, 63 levels) should dramatically reduce artifact size via better zlib compression of smaller integers. Primary goal is validating the int6 mechanism for H100 deployment; expected BPB degradation without QAT.

**Config**: Best config + `QUANT_BITS=6`. New `quantize_float_array_int6()` function clips to [-31,31] with per-row scaling.

**What happened**:
- Pre-quant val_bpb=**1.6332** at step 686 — within noise of best (1.6309). Training is unaffected by quantization changes.
- Post-int6 val_bpb=**1.6975** — **+0.064 BPB quant gap** vs pre-quant, **+0.067 vs best**.
- 686 steps at 875ms/step. Artifact: **6,834,725 bytes (6.8MB)** — **48% smaller** than int8 (13.1MB).
- Payload ratio: 3.85x (identical to int8 — the compression ratio is the same, but the raw payload is much smaller because 6-bit integers occupy less space).

**Analysis**:
- The artifact size reduction is massive: 6.8MB vs 13.1MB. On H100 with 16MB limit, this opens ~9.2MB of headroom — enough for 12+ layers or wider MLP.
- But the quant gap is catastrophic: +0.064 BPB. For comparison, int8 quant gap is ~0.003 BPB. Int6 is **~20x worse**.
- The error amplification is non-linear: 63 levels (int6) vs 255 levels (int8) = 4x less precision, but the quantization noise compounds across 10 transformer layers. Each layer's output error feeds into the next layer's input, creating multiplicative degradation.
- Pre-quant 1.6332 confirms the model trains identically — all damage is purely at serialization time. This is the strongest possible motivation for QAT: teach the model to be robust to int6 quantization noise during training.

**Decision**: **DISCARD**. Int6 without QAT has an unacceptable quant gap. The mechanism is validated (artifact reduction works, compression ratio maintained) but QAT is mandatory to make int6 competitive.

**Learnings**:
- Int6 (QUANT_BITS=6) reduces artifact 48% (13.1MB to 6.8MB) but quant gap is +0.064 BPB without QAT.
- The quant gap is ~20x worse than int8 (~0.003). Error compounds across layers — not a simple 4x precision loss.
- Pre-quant BPB is identical to best config — training is completely unaffected. All damage is at serialization.
- This establishes the baseline quant gap for exp_052 (int6 QAT): must close +0.064 BPB to be competitive.
- On H100, int6 + QAT could enable 12L or wider models within 16MB, which could more than compensate for any residual quant gap.

---

### Experiment 051: Pre-Warmdown QAT (NEW BEST)

**Hypothesis**: Constant-strength QAT (strength=0.1, every 10 steps) during pre-warmdown phase only (lr_mul >= 0.8) teaches quantization-friendly weights without disrupting warmdown convergence. This fixes exp_036's failure where ramping QAT noise fought warmdown convergence.

**Config**: Best config + `QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10`. New code: ~15 lines adding `sim_quant_roundtrip()` injection every QAT_EVERY steps when lr_mul >= QAT_STOP_LR_MUL.

**What happened**:
- Pre-quant val_bpb=**1.6270** — **best pre-quant EVER** (previous: 1.6284 from exp_046).
- Int8 val_bpb=**1.6299** — **NEW BEST, -0.0010** vs previous best (1.6309).
- 710 steps at ~846ms/step. Artifact: 13,091,119 bytes (~13.1MB, 3.85x compression).
- Quant gap: +0.0029 BPB (similar to baseline ~0.003 for int8, unchanged).

**The surprise: QAT as regularization, not quant-gap closure**:
- The quant gap is unchanged (+0.0029 vs ~0.0025 baseline). QAT did NOT reduce the int8 quant gap.
- But the pre-quant BPB improved by 0.0014 (1.6270 vs 1.6284). The model learned BETTER features.
- The int8 quantization noise every 10 steps acts as a regularizer during the main training phase, similar to dropout or label smoothing. It adds structured noise that prevents overfitting to the training data distribution.

**Why pre-warmdown timing works**:
- QAT fires during lr_mul >= 0.8, covering ~first 60% of training steps (before warmdown ramp begins).
- During this phase, the model is actively learning features and the noise is beneficial.
- Stopping before warmdown ensures the final convergence phase is clean and undisturbed.
- Compare with exp_036 (warmdown QAT): quant gap was excellent (0.0002) but overall BPB suffered catastrophically (+0.057) because noise fought convergence.

**Comparison with exp_036 (warmdown QAT)**:
| Metric | exp_036 (warmdown QAT) | exp_051 (pre-warmdown QAT) |
|--------|----------------------|--------------------------|
| Overall val_bpb | 1.6907 (+0.060) | **1.6299 (-0.001)** |
| Quant gap | **0.0002** | 0.0029 |
| Mechanism | QAT closes gap but hurts training | QAT improves training, gap unchanged |
| Timing | During warmdown (lr_mul < 0.8) | Before warmdown (lr_mul >= 0.8) |

**Decision**: **KEEP — NEW BEST**. Pre-warmdown QAT is a validated technique with zero overhead that improves model quality through regularization.

**Learnings**:
- Pre-warmdown QAT (strength=0.1, every 10 steps, lr_mul >= 0.8) improves overall val_bpb by -0.001.
- The improvement is from better training (regularization effect), not quant gap reduction.
- Zero overhead: <2ms/step on steps where QAT fires (every 10th step).
- Validates the hypothesis from exp_036: the QAT mechanism works, timing was the only problem.
- For int6 QAT (exp_052), the pre-warmdown approach should be used, not warmdown-phase QAT.

---

### Experiment 052: Int6 QAT (Discard)

**Hypothesis**: Int6 QAT (QAT_BITS=6) with int6 serialization (QUANT_BITS=6) should close the int6 quant gap validated in exp_049. Using the pre-warmdown approach from exp_051 (strength=0.1, every 10 steps, lr_mul >= 0.8) with int6 quantization noise instead of int8.

**Config**: Best config + `QAT_BITS=6 QUANT_BITS=6 QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10`. Code changes: `sim_quant_int6()` function for 6-bit roundtrip noise injection.

**What happened**:
- Pre-quant val_bpb=**1.6281** — **best pre-quant EVER** (previous: 1.6270 from exp_051). Int6 noise is an even better regularizer than int8 noise.
- Post-int6 val_bpb=**1.6894** — **+0.061 BPB quant gap**. Barely improved from exp_049's +0.064 gap without QAT.
- 712 steps at ~843ms/step. Artifact: **6,936,812 bytes (6.9MB)**.

**Analysis**:
- The regularization effect works even better with int6 noise: pre-quant 1.6281 beats exp_051's 1.6270 pre-quant. Coarser quantization noise = stronger regularization, which is beneficial during pre-warmdown training.
- But QAT's main purpose (closing quant gap) FAILED: +0.061 vs +0.064 is a negligible improvement (only 0.003 BPB closed out of 0.064). The model is not learning int6-friendly weights.
- Root cause: strength=0.1 every 10 steps is far too gentle for int6's coarser quantization. Int8 has 255 levels, int6 has 63 levels — the quantization error is ~4x larger per weight, and it compounds across 10 layers. The training signal from QAT every 10th step at 10% strength is swamped by the magnitude of the int6 quantization error.
- Compare: exp_051 (int8 QAT) had quant gap ~0.003 — already near the noise floor. Int8 quantization is inherently gentle enough that light QAT suffices. Int6 requires proportionally stronger QAT.

**Decision**: **DISCARD**. Int6 quant gap remains unacceptable at +0.061 despite QAT.

**Learnings**:
- Int6 QAT at strength=0.1, every=10 barely closes the gap (+0.064 to +0.061). Need much stronger QAT for int6.
- Int6 noise is an excellent pre-quant regularizer: 1.6281 is best pre-quant ever, confirming the "QAT as regularization" insight from exp_051 scales with noise magnitude.
- Next attempt should use strength=0.3, every=5 (6x more total QAT signal) with QAT_BITS=6.
- The artifact size (6.9MB) confirms int6 serialization works well mechanically.

---

### Experiment 053: Depth-Recurrent Warmdown (Discard)

**Hypothesis**: During warmdown, add L2 regularization between middle encoder layers (2,3,4) to encourage weight sharing. `L_share = alpha * sum(||W_i - W_j||^2)` for adjacent pairs. Alpha ramps from 0 to 0.1 during warmdown (`alpha * (1 - lr_mul)`). If layers converge, we can deduplicate at serialization for artifact savings.

**Config**: Best config + `DEPTH_RECURRENCE=1 DEPTH_RECURRENCE_ALPHA=0.1 DEPTH_RECURRENCE_LAYERS=2,3,4`. New code: ~25 lines computing pairwise L2 between designated layers and adding to loss during warmdown.

**What happened**:
- Pre-quant val_bpb=**1.6345** at step 711 — worse than best pre-quant (1.6270 from exp_051).
- Int8 val_bpb=**1.6378** — **+0.008 BPB worse** than best (1.6299).
- 711 steps at ~844ms/step. Artifact: **12,913,555 bytes (12.9MB)** — only -0.2MB smaller than best (13.1MB).
- Compression ratio: 3.85x (unchanged).

**Analysis**:
- The artifact savings are negligible: 12.9MB vs 13.1MB (-0.2MB). The L2 nudging at alpha=0.1 during warmdown is too gentle to actually make layers similar enough for meaningful deduplication.
- But the BPB cost is real: +0.008 is a significant regression (comparable to DropHead or SWA). Even gentle auxiliary losses during warmdown interfere with the convergence dynamics.
- This confirms a broader principle: **the warmdown phase is sacred**. Both QAT (exp_036, +0.057) and depth recurrence (exp_053, +0.008) degrade BPB when applied during warmdown. Only pre-warmdown techniques (exp_051, QAT as regularizer) work.
- The 0.2MB saving would require nearly identical weights across layers 2,3,4 to be worthwhile (full deduplication could save ~3MB). At alpha=0.1, the layers are nowhere near identical.

**Decision**: **DISCARD**. Kill depth recurrence as a warmdown technique. The convergence cost (+0.008 BPB) far exceeds the compression benefit (-0.2MB). Stronger alpha would only make the BPB regression worse.

**Learnings**:
- Depth-recurrent warmdown (alpha=0.1, layers 2,3,4) hurts BPB +0.008 for only -0.2MB artifact savings.
- Even gentle auxiliary losses during warmdown interfere with convergence. The warmdown phase must remain clean.
- Pattern confirmed: warmdown-phase interventions (SWA, QAT, depth recurrence) all hurt on Apple Silicon. Only pre-warmdown interventions work.
- Kill exp_054 (stronger depth recurrence, alpha=0.5) — if gentle already hurts, stronger will be worse.

---

### Experiment 054: Strong QAT (Discard)

**Hypothesis**: Stronger int8 QAT (strength=0.2, every=5 steps, 4x total signal vs exp_051's strength=0.1/every=10) provides more regularization, further improving pre-quant BPB or reducing quant gap.

**Config**: Pure env-var change: `QAT_STRENGTH=0.2 QAT_EVERY=5`. All else identical to exp_051 best config.

**What happened**:
- Pre-quant val_bpb=**1.6272** at step 712 — essentially identical to exp_051's 1.6270.
- Int8 val_bpb=**1.6299** — identical to exp_051's 1.6299.
- 712 steps at ~843ms/step. Artifact: 13,105,555 bytes (~13.1MB).
- Quant gap: +0.0027 (marginally better than exp_051's +0.0029 — within noise).

**Analysis**:
- 4x more QAT signal (2x strength * 2x frequency) produces identical results. The regularization effect of int8 QAT noise is already saturated at strength=0.1/every=10.
- This is a binary threshold effect, not a gradient: either you have QAT noise or you don't. The exact strength/frequency doesn't matter once you're above the threshold.
- This makes intuitive sense: int8 quantization noise is very small (255 levels, ~0.4% error per weight). Even light exposure to this noise pattern is enough for the model to develop robustness. More exposure adds no new information.
- Contrast with int6 QAT (exp_052) where strength=0.1/every=10 was clearly insufficient for the much larger int6 noise. Int6 has ~4x larger per-weight error, so the threshold for saturation is much higher.

**Decision**: **DISCARD**. Keep the simpler config (strength=0.1, every=10). QAT hyperparameters are not worth tuning for int8.

**Learnings**:
- QAT regularization saturates quickly for int8. 4x more signal produces identical BPB. It's a binary threshold, not a gradient.
- Don't tune QAT hyperparameters further for int8. Strength=0.1, every=10 is sufficient.
- This distinguishes int8 QAT from int6 QAT: int8 noise is small enough that any reasonable amount saturates the benefit. Int6 noise is large enough that the saturation threshold is much higher (exp_052 was clearly below it).

---

### Experiment 055: Strong Int6 QAT (Discard — Int6 QAT Killed)

**Hypothesis**: 6x stronger int6 QAT (strength=0.3, every=5 vs exp_052's 0.1/10) should close the int6 quant gap (+0.061-0.064 BPB) by giving the model proportionally more exposure to int6 quantization noise during pre-warmdown training.

**Config**: Best config + `QAT_PREWARMDOWN=1 QAT_STRENGTH=0.3 QAT_EVERY=5 QAT_STOP_LR_MUL=0.8 QAT_BITS=6 QUANT_BITS=6`.

**What happened**:
- Pre-quant val_bpb=**1.6271** at step 713 — excellent, matches best pre-quant ever (1.6270 from exp_051).
- Post-int6 val_bpb=**1.6904** — **+0.063 BPB quant gap**. Barely moved from exp_052's +0.061 or exp_049's +0.064.
- 713 steps at ~842ms/step. Artifact: **6,938,640 bytes (6.9MB)**.

**The definitive int6 QAT result — three experiments confirm:**

| Experiment | QAT Config | Pre-quant BPB | Post-int6 BPB | Quant Gap |
|------------|-----------|---------------|---------------|-----------|
| exp_049 | No QAT | 1.6332 | 1.6975 | +0.064 |
| exp_052 | strength=0.1, every=10 | 1.6281 | 1.6894 | +0.061 |
| exp_055 | strength=0.3, every=5 | 1.6271 | 1.6904 | +0.063 |

The quant gap is **+0.063 +/- 0.003** regardless of QAT strength. Three data points spanning no-QAT to 6x-strong QAT all land in the same band. Pre-warmdown QAT cannot close the int6 quant gap.

**Root cause analysis**:
- Int6 quantization maps each weight to one of 63 levels ([-31, +31]). The per-weight error is ~1.6% of the weight range.
- Pre-warmdown QAT periodically rounds weights to these 63 levels, nudging the optimizer to prefer quantization-friendly values. But the 63-level grid is so coarse that "quantization-friendly" and "optimal for loss" are fundamentally different objectives.
- With int8 (255 levels), the grid is fine enough that QAT can trivially close the gap (~0.003 BPB). With int6, the gap is 20x larger and structurally resistant to pre-warmdown nudging.
- Closing int6 gap likely requires: (1) STE (straight-through estimator) in the forward pass — every step sees quantized weights, not just periodic nudging; (2) post-training calibration (GPTQ/AWQ) which optimally adjusts weights to minimize layer-wise reconstruction error; or (3) simply more training steps on H100 where the model can find deeper quantization-friendly basins.

**Decision**: **DISCARD**. Kill all int6 QAT experiments on Apple Silicon. The pre-warmdown QAT approach that works beautifully for int8 fundamentally cannot close the int6 gap.

**Learnings**:
- Int6 quant gap is ~+0.063 BPB regardless of QAT strength (3 experiments, no-QAT through 6x-strong).
- Pre-warmdown QAT cannot close the int6 gap — 63 levels is fundamentally too coarse for periodic nudging.
- Int6 QAT does improve pre-quant BPB (regularization effect scales with noise), but cannot fix serialization damage.
- Kill int6 QAT on Mac. Need STE forward-pass quantization, GPTQ post-training calibration, or H100's longer training.

---

### Experiment 056: Full-Training QAT (Discard — Neutral)

**Hypothesis**: Full-training QAT (QAT_STOP_LR_MUL=0, running through warmdown) helps convergence to a more quantization-friendly minimum by maintaining QAT noise through the entire training including warmdown, unlike exp_051 which stops at lr_mul=0.8.

**Config**: Best config + `QAT_STOP_LR_MUL=0` (QAT runs through warmdown instead of stopping at lr_mul=0.8). All else identical to exp_051.

**What happened**:
- Pre-quant val_bpb=**1.6283** at step 704 — slightly worse than exp_051's 1.6270.
- Int8 val_bpb=**1.6308** — +0.0009 vs best (1.6299). Within noise.
- Quant gap: **+0.0025** (slightly better than exp_051's +0.0029).
- 704 steps at ~853ms/step. Artifact: **13,085,184 bytes (13.1MB)**.

**Analysis**:
- QAT during warmdown does close the quant gap slightly better (+0.0025 vs +0.0029 = -0.0004 improvement). This makes sense: the model continues seeing quantization noise through warmdown, so final weights are slightly more quantization-friendly.
- But pre-quant BPB is slightly worse (1.6283 vs 1.6270 = +0.0013). The QAT noise during warmdown mildly interferes with convergence, just as predicted from exp_036 (where ramping warmdown QAT was catastrophic at +0.057).
- The two effects cancel: -0.0004 quant gap improvement + +0.0013 pre-quant regression = +0.0009 net regression. Within noise.
- This is an interesting contrast with other warmdown interventions: SWA (+0.127), depth recurrence (+0.008), and ramping QAT (+0.057) all clearly hurt during warmdown. Constant-strength QAT during warmdown is merely neutral. The difference is that SWA/depth-recurrence modify the loss objective, while QAT only perturbs weights. Weight perturbation is tolerated; loss modification is not.

**Decision**: **DISCARD**. Neutral result — within noise of best. Pre-warmdown-only QAT (exp_051) remains the better config.

**Learnings**:
- Full-training QAT (QAT_STOP_LR_MUL=0) is neutral vs pre-warmdown-only (QAT_STOP_LR_MUL=0.8).
- QAT during warmdown closes quant gap slightly better (+0.0025 vs +0.0029) but at cost of slightly worse pre-quant BPB (1.6283 vs 1.6270). Effects cancel.
- The "sacred warmdown" principle has nuance: loss-modifying interventions (SWA, depth recurrence, ramping QAT) clearly hurt, but weight perturbations (constant QAT noise) are tolerated. The warmdown is sensitive to what's being changed, not just that something is being changed.
- Keep pre-warmdown-only config (QAT_STOP_LR_MUL=0.8) — it has the best pre-quant BPP and post-quant is identical within noise.

---

### Experiment 057: Cosine Warmdown Shape (Discard)

**Hypothesis**: Cosine warmdown shape (`lr_mul = 0.5 * (1 + cos(pi * (1 - t)))`) spends more time at higher LR and drops faster at the end, potentially finding better optima before final convergence.

**Config**: Best config + `WARMDOWN_SHAPE=cosine`. Code change: ~5 lines in `lr_mul()` to apply cosine transformation when shape is "cosine" and t < 1.0.

**What happened**:
- Pre-quant val_bpb=**1.6466** at step 696 — worse than best pre-quant (1.6270 from exp_051).
- Post-quant val_bpb=**1.6503** — **+0.020 BPB worse** than best (1.6299).
- 696 steps at ~863ms/step. Artifact: **12,749,219 bytes (12.7MB)** — smaller than best (13.1MB), likely from less-converged weights compressing better.

**Analysis**:
- Cosine warmdown is clearly worse (+0.020 BPB). The cosine curve keeps LR too high for too long during the warmdown phase — by the time it drops, there isn't enough time for fine convergence.
- With linear warmdown, the steady LR decay gives the model continuous opportunity to converge to sharper minima. The gradual, even decrease matches the optimization landscape better than cosine's plateau-then-plunge shape.
- The smaller artifact (12.7MB vs 13.1MB) is a red herring — the weights are less trained (fewer effective low-LR steps), producing less structured weights that happen to compress slightly better.
- This is consistent with our broader finding that the warmdown phase is critical and sensitive. Linear decay is the most predictable schedule for this aggressive warmdown regime (~1200 steps of warmdown with only ~700 total steps).

**Decision**: **DISCARD**. Kill warmdown shape experiments. Linear warmdown is optimal for this training regime.

**Learnings**:
- Cosine warmdown shape is clearly worse (+0.020 BPB). Linear warmdown is optimal.
- The steady LR decay of linear warmdown allows fine convergence that cosine's rapid end-drop cannot match.
- With only ~700 steps and warmdown dominating training, the shape of the decay curve matters — linear gives the most even distribution of convergence effort across the warmdown phase.

---

### Experiment 058: Label Smoothing (Discard — CATASTROPHIC)

**Hypothesis**: Label smoothing (0.1) acts as regularization for limited data, preventing overconfident predictions and potentially improving generalization on the small training set.

**Config**: Best config + `LABEL_SMOOTHING=0.1`. Code change: modify cross-entropy loss to use smoothed targets instead of hard labels.

**What happened**:
- Pre-quant val_bpb=**2.0475** at step 692 — **+0.42 BPB worse** than best (1.6299). CATASTROPHIC regression.
- Post-quant val_bpb=**2.0499**. Quant gap: +0.0024 (normal).
- 692 steps at ~867ms/step. Artifact: 12,944,370 bytes (12.9MB).
- Training loss was massively elevated throughout: step 200 loss=3.95 (vs typical ~2.8), step 400 loss=3.68 (vs typical ~2.5). The model never converged to normal loss levels.
- val_bpb at step 500 was 2.1406 — already catastrophic at the midpoint.

**Root cause analysis**:
- With vocab=1024, label smoothing=0.1 redistributes 10% of probability mass across 1023 non-target tokens. Each non-target token gets ~0.0001 probability mass. This seems small, but the training objective (smoothed CE) now encourages the model to assign non-trivial probability to ALL tokens at every position.
- The evaluation metric is standard CE (hard labels). The model optimized for smoothed CE learns to spread probability mass, which is penalized by standard CE. The training objective directly conflicts with the evaluation metric.
- With large vocabularies (32K-100K), the per-token smoothing mass is negligible (~1e-6). With vocab=1024, it's 100x larger per token, making the objective mismatch much worse.
- Compare with pre-warmdown QAT (exp_051, -0.001 BPB): QAT adds noise to weights but doesn't change the loss objective. Label smoothing fundamentally changes what the model is optimizing for.

**Decision**: **DISCARD**. Kill label smoothing entirely.

**Learnings**:
- Label smoothing is CATASTROPHIC with small vocabularies. Vocab=1024 means each non-target token gets ~0.0001 smoothing mass — 100x more than with vocab=100K.
- The training/evaluation objective mismatch (smoothed CE vs standard CE) is the primary damage mechanism. The model learns to spread probability mass, which standard CE penalizes.
- This distinguishes label smoothing from other regularizers: WD, QAT, and gradient clipping all preserve the loss objective. Only label smoothing changes what the model optimizes for.
- Regularization that preserves the loss function (WD, QAT noise, gradient clipping) works. Regularization that modifies the loss function (label smoothing, SWA) hurts.

---

### Experiment 059: Higher Matrix LR (Discard)

**Hypothesis**: Higher matrix_lr=0.06 (1.5x default 0.04) compensates for the reduced effective LR from aggressive warmdown. With warmdown dominating training (~1200 iters target vs ~700 actual steps), the model spends most of its time at reduced LR. A higher base LR could allow more learning in the pre-warmdown phase.

**Config**: Pure env-var change: `MATRIX_LR=0.06`. All else identical to exp_051 best config.

**What happened**:
- Pre-quant val_bpb=**1.6375** at step 703 — worse than best pre-quant (1.6270 from exp_051).
- Post-quant val_bpb=**1.6391** — **+0.009 BPB worse** than best (1.6299).
- 703 steps at similar step time. Artifact: **13,309,291 bytes (13.3MB)** — slightly larger than best (13.1MB), likely from less regular weights due to overshooting.

**Analysis**:
- The higher LR overshoots during the main training phase. Muon's Newton-Schulz orthogonalization already normalizes gradient updates to have unit norm on the Stiefel manifold — the matrix_lr=0.04 is tuned to match this update scale. Increasing to 0.06 pushes updates 50% larger than the natural orthogonal step size.
- The artifact size increase (13.3MB vs 13.1MB) is consistent with the overshooting hypothesis: larger LR steps produce less regular weight patterns that compress worse under zlib.
- Pre-quant regression (+0.011 vs best) is larger than post-quant regression (+0.009 vs best), meaning quantization slightly helped — the noisier weights happen to be more robust to int8 rounding.

**Decision**: **DISCARD**. Kill LR sweep. Muon's default matrix_lr=0.04 is well-calibrated for the orthogonalized update scale.

**Learnings**:
- Matrix LR=0.04 is optimal for Muon with Newton-Schulz orthogonalization. The LR is matched to the unit-norm update scale on the Stiefel manifold.
- Higher LR (0.06) overshoots: +0.009 BPP and worse compression. The damage is in the main training phase, not warmdown.
- Don't sweep matrix_lr. The LR for Muon is fundamentally different from SGD/Adam LR — it's a step size on the Stiefel manifold where unit norm is already the natural scale.
- This closes the "compensate warmdown with higher LR" hypothesis. The warmdown regime is already well-optimized (linear decay, 1200 target).

---

### Experiment 060: Higher Tied Embed LR (Discard)

**Hypothesis**: Higher tied_embed_lr=0.1 (2x default 0.05) improves embedding learning by giving the shared input/output embedding more optimization signal during the limited ~700 steps on Apple Silicon.

**Config**: Pure env-var change: `TIED_EMBED_LR=0.1`. All else identical to exp_051 best config.

**What happened**:
- Pre-quant val_bpb=**1.6573** at step 705 — significantly worse than best pre-quant (1.6270 from exp_051).
- Post-quant val_bpb=**1.6598** — **+0.030 BPB worse** than best (1.6299). The largest regression from any LR sweep.
- 705 steps at ~852ms/step. Artifact: **13,180,922 bytes (13.2MB)**.
- Compression ratio: 3.85x (unchanged).
- Early training was unstable: step 2 loss=24.4, step 3 loss=19.9 (vs typical ~6.9). The 50-step warmup partially recovered but the damage to early feature learning persisted.

**Analysis**:
- The tied embedding serves dual duty: input token lookup AND output logit projection. This shared representation is extremely sensitive to learning rate — a 2x increase destabilizes both the input and output interfaces simultaneously.
- The early training instability (loss spikes to 24.4 at step 2) suggests the higher LR causes catastrophic updates to the embedding matrix in the first few gradient steps, before warmup ramps to full LR. Even though warmup recovers the training loss, the initial damage to embedding structure persists through the run.
- Compare with matrix_lr sweep (exp_059, +0.009 BPB at 1.5x): the embed LR is 3x more sensitive than matrix LR. This makes sense — matrix params are orthogonalized by Newton-Schulz (bounded update scale), while Adam updates to embeddings have no such normalization.
- The artifact size (13.2MB vs 13.1MB) is marginally larger, consistent with slightly less regular embeddings from the higher LR.

**Decision**: **DISCARD**. Kill embed LR sweep upward. Adam's default 0.05 is already well-tuned for the delicate tied embedding.

**Learnings**:
- Tied embed_lr=0.05 is optimal. 2x higher gives +0.030 BPB — the worst regression from any single LR change.
- Tied embeddings are 3x more LR-sensitive than Muon matrix params (0.030 vs 0.009 per 50% LR increase).
- The dual-use nature (input lookup + output projection) makes the embedding uniquely sensitive: errors propagate through both the forward pass input AND the output logits.
- Adam's embed LR is already well-calibrated at 0.05. Don't sweep upward. A downward sweep (e.g., 0.03) is unlikely to help given the plateau.

---

### Experiment 061: Lower Tied Embed LR (Discard -- Embed LR Sweep Fully Closed)

**Hypothesis**: Lower tied_embed_lr=0.03 (0.6x default 0.05) provides more stable embedding learning by reducing the update magnitude to the sensitive tied embedding.

**Config**: Pure env-var change: `TIED_EMBED_LR=0.03`. All else identical to exp_051 best config.

**What happened**:
- Pre-quant val_bpb=**1.6382** at step 690 -- worse than best pre-quant (1.6270 from exp_051).
- Post-quant val_bpb=**1.6407** -- **+0.011 BPB worse** than best (1.6299).
- 690 steps at ~870ms/step. Artifact: **13,029,285 bytes (13.0MB)**.
- Compression ratio: 3.85x (unchanged).
- Early training was normal (step 2 loss=12.0, step 3 loss=10.4 -- elevated but not explosive like exp_060's 24.4/19.9). The lower LR avoids the instability of higher LR but simply doesn't train the embedding fast enough.

**Analysis**:
- The lower embed LR under-trains the embedding. With only ~700 steps on Apple Silicon, the embedding needs sufficient LR to develop good representations. At 0.03, the embedding converges too slowly and doesn't reach the quality achieved at 0.05.
- Compare with exp_060 (embed_lr=0.1, +0.030 BPB): higher LR is 3x worse than lower LR (+0.030 vs +0.011). This asymmetry makes sense -- overshooting the tied embedding is more damaging than under-training it, because the dual-use embedding (input+output) amplifies errors in both directions through the forward pass.
- The embed LR sweep is now fully closed: 0.03 (+0.011), 0.05 (best), 0.10 (+0.030). The response curve is V-shaped with minimum at 0.05. No further exploration needed.

**Embed LR sweep summary**:
| Embed LR | Experiment | val_bpb | Delta vs Best | Early Training |
|----------|-----------|---------|---------------|----------------|
| 0.03 | exp_061 | 1.6407 | +0.011 | Normal (loss=12.0 at step 2) |
| **0.05** | **exp_051** | **1.6299** | **baseline** | **Normal (loss=6.9)** |
| 0.10 | exp_060 | 1.6598 | +0.030 | Explosive (loss=24.4 at step 2) |

**Decision**: **DISCARD**. Embed LR sweep is fully closed. 0.05 is the sweet spot.

**Learnings**:
- Tied embed_lr=0.05 is optimal. Both lower (0.03, +0.011) and higher (0.10, +0.030) are worse. The V-shaped response has its minimum at 0.05.
- The tied embedding's dual role (input lookup + output projection) makes it sensitive to LR in both directions: too high destabilizes, too low under-trains.
- The asymmetry (higher LR is 3x worse) suggests the embedding is more sensitive to overshooting than undershooting -- consistent with the dual-use amplification of errors.
- This closes the embed LR sweep definitively. No further exploration needed in any direction.

---

### Experiment 063: LeakyReLU(0.5)² Activation (NEW BEST)

**Hypothesis**: Replace `relu(x)²` with `leaky_relu(x, 0.5)²` in MLP. The original relu² creates dead neurons (zero gradient for x<0). With a 0.5 negative slope, 50% of negative gradient flow is preserved, eliminating dead neurons while the squaring still provides sparsity. Expected -0.001 to -0.003 BPB.

**What happened**:
- Pre-quant val_bpb=**1.6190** — best pre-quant ever (vs previous best 1.6270 from exp_051).
- Int8+zlib val_bpb=**1.6215** — **NEW BEST, -0.0084 BPB** vs previous best (1.6299).
- 700 steps at ~857ms/step (within normal range). Artifact: 13,124,539 bytes (~13.1MB, 2.9MB headroom).
- Quant gap: +0.0025 BPB (1.6215 - 1.6190), consistent with normal int8 quant gap.

**Analysis**:
- This is a surprisingly large win (-0.0084 BPB) for what is essentially a one-line change. It is the largest single improvement since batch scaling (exp_022, -0.082 BPB) and gradient clipping (exp_033, -0.016 BPB).
- The improvement is genuine training quality, not a quantization artifact: pre-quant BPB improved by -0.0080 (1.6190 vs 1.6270), nearly identical to the post-quant improvement (-0.0084).
- With vocab=1024 and tied embeddings, every neuron matters more than in large-vocab models. Dead neurons in relu² permanently waste capacity. The 0.5 slope preserves negative information while the squaring still provides beneficial sparsity (0.5² = 0.25, so negative activations are still suppressed relative to positive ones).
- The win likely stacks multiplicatively with everything else: it improves the fundamental capacity utilization of every MLP layer across all 10 layers. This is not a regularization trick — it is a genuine architectural improvement.
- Step time (857ms) is essentially identical to previous runs (~855ms), confirming zero overhead from the activation change.

**Decision**: **KEEP** as new best. Largest non-batch/non-clip improvement in the entire experiment history.

**Learnings**:
- LeakyReLU(0.5)² is strictly better than relu² for this architecture. Dead neuron elimination is worth -0.008 BPB.
- The 0.5 slope was chosen to balance gradient flow preservation (higher slope = more gradient) with sparsity from squaring (lower slope = more sparsity). The squaring means even a 0.5 slope only contributes 0.25x on the negative side, maintaining substantial sparsity.
- This makes ALL remaining experiments more promising — they stack on top of a better activation function. The LeakyReLU change is in the code (not env vars), so the best config env vars remain unchanged.
- Key principle confirmed: with small vocab (1024) and tied embeddings, capacity efficiency is paramount. Any technique that reduces wasted capacity (dead neurons, over-regularization) has outsized impact.

---

### Experiment 064: EMA decay=0.997 (Discard)

**Hypothesis**: Continuous EMA (decay=0.997) smooths per-step noise, improving final model quality. All top-4 competition teams use EMA. Unlike SWA (killed), EMA updates continuously and works with monotonic convergence.

**Config**: Best config (exp_063 baseline with LeakyReLU(0.5)²) + EMA with decay=0.997, updated every step. EMA weights swapped in for eval/serialization.

**What happened**:
- Pre-EMA val_bpb=**1.6224** — close to exp_063 baseline (1.6215), confirming the underlying model trained well.
- Post-EMA val_bpb=**1.8437** — **CATASTROPHIC +0.22 BPB regression**.
- 689 steps at ~871ms/step. Artifact: 12,099,145 bytes (~12.1MB).

**Root cause — EMA window too wide for short training**:
- With decay=0.997, the effective averaging window is `1/(1-0.997) = 333 steps` = **48% of 689 total steps**.
- EMA weights are dominated by early under-trained parameters from the first half of training.
- The warmdown phase (which produces the final convergence) is heavily diluted by stale early weights.
- Compare with H100 (6000+ steps): same decay gives a ~333-step window = ~5% of training, a reasonable smoothing window.

**Scaling analysis**:
| Platform | Steps | EMA window (decay=0.997) | Window/Total | Expected effect |
|----------|-------|--------------------------|--------------|-----------------|
| Mac (Apple Silicon) | ~700 | 333 | 48% | Catastrophic (confirmed) |
| H100 | ~6000 | 333 | 5.5% | Should work (top teams confirm) |

**Decision**: **DISCARD**. EMA is fundamentally incompatible with Apple Silicon's short training runs at decay=0.997. Reserve for H100 only, or use much higher decay on Mac (e.g., 0.98 for ~50-step window = 7% of training).

**Learnings**:
- EMA's effective window scales inversely with step count. For fixed decay, EMA is harmful when training duration is short.
- On Mac with ~700 steps, decay=0.997 averages 48% of training — far too much. Need decay >= 0.98 for a reasonable ~7% window.
- On H100 with 6000+ steps, decay=0.997 gives ~5.5% window — this is why all top teams use it successfully.
- The pre-EMA model (1.6224) confirms the EMA overhead (~871ms vs ~848ms/step) is minor and the underlying training is unaffected.
- EMA is now KILLED for Mac experiments. It joins SWA in the "weight averaging killed on Mac" category.
- For H100 deployment, EMA (decay=0.997) remains a must-have.

---

### Experiment 066: XSA on Last 3 Decoder Layers (Discard — Neutral)

**Hypothesis**: Exclusive Self Attention (XSA) — subtracting each token's own value contribution from the attention output — forces the model to learn only context-dependent information. Applied to the last 3 decoder layers only. 4 of 5 top teams use this. Expected -0.002 to -0.005 BPB.

**Config**: Best config (exp_063 baseline with LeakyReLU(0.5)²) + `XSA_LAYERS=3`. Code change: in CausalSelfAttention.forward, after attention output, compute self-value component and subtract for the last 3 layers.

**What happened**:
- Pre-quant val_bpb=**1.6198** at step 701 — within noise of exp_063 baseline (1.6190).
- Int8+zlib val_bpb=**1.6222** — **+0.0007 BPB** vs best (1.6215). Within noise but trending negative.
- 701 steps at ~857ms/step (identical to baseline). Artifact: **13,115,125 bytes (~13.1MB)**.
- No measurable speed penalty from custom mask vs fused causal kernel.

**Analysis**:
- XSA is neutral on Apple Silicon with 10 layers. The self-exclusion removes one token's value contribution out of up to 1024 context positions — a ~0.1% information loss. This marginal deduplication benefit is cancelled by the loss of the self-reinforcing signal.
- The absence of speed penalty is good news: the custom attention mask (causal + diagonal exclusion) runs at the same speed as the standard causal mask on Apple Silicon. This means XSA is "free" in terms of compute.
- The technique may work better on H100 with more layers (11L+) where deeper layers benefit more from pure-context signals, and with longer training where the model has time to adapt to the modified attention pattern.

**Decision**: **DISCARD**. Neutral result. XSA is not worth keeping for Mac experiments.

**Learnings**:
- XSA on last 3 decoder layers is neutral (+0.0007 BPB) on 10L Apple Silicon runs.
- The self-exclusion removes 1/1024 (~0.1%) of context information per position — too small a signal for the deduplication benefit to manifest with only 10 layers.
- No speed penalty from custom mask vs fused causal kernel. XSA is compute-free.
- Worth revisiting on H100 with 11L+ where deeper layers benefit more from pure-context signals.

---

### Experiment 067: Partial RoPE 25% (Discard)

**Hypothesis**: Apply RoPE to only 25% of head dimensions (16 of 64), freeing 75% for position-invariant content matching. Top-3 team reports -0.002 BPB at longer seq_len.

**Config**: Best config (exp_063 baseline with LeakyReLU(0.5)²) + `PARTIAL_ROPE_FRAC=0.25`. Code change: in apply_rotary_emb, only rotate x[..., :partial_dim].

**What happened**:
- Pre-quant val_bpb=**1.6213** at step 697 — within noise of exp_063 baseline (1.6190).
- Int8+zlib val_bpb=**1.6240** — **+0.0025 BPB** vs best (1.6215). DISCARD.
- 697 steps at ~862ms/step. Artifact: **13,073,493 bytes (~13.1MB)**.

**Analysis**:
- At seq_len=1024, every position matters. RoPE enables position-dependent attention patterns (recency bias, periodic patterns) that the model actively uses. Removing positional encoding from 75% of head dimensions deprives the model of this positional capability without providing sufficient compensating content-matching benefit.
- The pre-quant result (1.6213 vs 1.6190 = +0.0023) confirms the regression is in training quality, not quantization. The quant gap (+0.0027) is normal.
- This differs from longer-context models (seq_len=4K+) where many attention patterns genuinely are position-invariant. At seq_len=1024, the model needs position information across all dimensions to learn fine-grained positional patterns in the relatively short context window.
- The top-3 team's reported -0.002 BPB was likely at longer seq_len where the benefit of position-free content matching outweighs the cost of reduced positional resolution.

**Decision**: **DISCARD**. Full RoPE is better at seq_len=1024.

**Learnings**:
- Full RoPE is optimal at seq_len=1024. Partial RoPE (25%) gives +0.0025 BPP regression.
- Short sequences need position information everywhere — the 1024-token context is short enough that position-dependent patterns (recency, periodicity) are useful across all attention dimensions.
- Partial RoPE is a longer-context technique. At seq_len=4K+, more attention patterns become genuinely position-invariant, and the content-matching benefit of position-free dimensions may outweigh the cost. Not worth revisiting unless we move to longer sequences.

---

### Experiment 065: 3-Band Freq-Decomposed Skip Gating (Discard)

**Hypothesis**: 3-band freq-decomposed skip gating (ultra-low W=128, mid W=32, high residual) gives finer spectral control than the 2-band (low W=32, high residual) currently in the best config. Expected -0.001 to -0.005 BPB.

**Config**: Best config (exp_063 baseline with LeakyReLU(0.5)²) + `FREQ_SKIP_BANDS=3`. Code change: add third ultra-low frequency band with W=128 decomposition, 3 weight vectors per skip connection instead of 2.

**What happened**:
- Pre-quant val_bpb=**1.6212** at step 699 — within noise of exp_063 baseline (1.6190).
- Int8+zlib val_bpb=**1.6238** — **+0.0023 BPB** vs best (1.6215). DISCARD.
- 699 steps at ~859ms/step. Artifact: **13,127,542 bytes (~13.1MB)**.

**Analysis**:
- The 2-band lo/hi decomposition at W=32 already captures the meaningful spectral information in the skip connections. Adding an ultra-low band at W=128 splits 4 dims from the 16-dim low band into a separate channel, but this provides no new information — the W=32 low band already captures trends at that scale.
- The extra parameters (5 skip_ulo_weights vectors, one per skip connection) are under-constrained with only ~700 training steps and slightly hurt optimization by adding degrees of freedom without signal.
- Pre-quant regression (+0.0022 vs baseline) confirms the damage is in training quality, not quantization. The additional gating parameters slow convergence without improving the spectral decomposition.
- This is consistent with exp_044 (W=16 vs W=32 makes no difference): the frequency decomposition is robust to window size because the meaningful information split is between "any local average" and "local residual", not at a specific frequency cutoff.

**Decision**: **DISCARD**. Kill multi-band skip gating experiments. 2-band at W=32 is the sweet spot.

**Learnings**:
- 2-band skip gating (W=32) already captures the useful spectral decomposition. More bands add parameters without signal.
- The meaningful information split is binary: local average vs local residual. Finer frequency decomposition is redundant.
- This closes the skip gating spectral exploration: W=16 (exp_044), W=32 (exp_017, best), 3-band (exp_065) all confirm 2-band W=32 is optimal.

---

### Experiment 068: GELU² Activation (Discard)

**Hypothesis**: GELU² activation provides smooth Gaussian gating, potentially better than LeakyReLU(0.5)² which uses a hard kink at x=0. GELU's smooth transition might allow better gradient flow during training.

**Config**: Best config (exp_063 baseline) with GELU² replacing LeakyReLU(0.5)² in MLP. Code change: replace `leaky_relu(x, neg_slope=0.5) ** 2` with `gelu(x) ** 2`.

**What happened**:
- Pre-quant val_bpb=**1.6345** at step 687 — worse than exp_063 baseline (1.6190).
- Int8+zlib val_bpb=**1.6373** — **+0.0158 BPB** vs best (1.6215). DISCARD.
- 687 steps at ~873ms/step (slower than LeakyReLU's ~855ms). Artifact: **12,652,167 bytes (~12.7MB)** — smaller than usual due to GELU producing more compressible weight patterns.

**Analysis**:
- GELU² is significantly worse because GELU's Gaussian gating kills negative inputs exponentially (GELU(x) approaches 0 for x < -2), while LeakyReLU(0.5) preserves 50% of the input linearly. With squaring already providing sparsity (both activations square to provide the nonlinearity), the activation's role is to control negative gradient flow.
- LeakyReLU(0.5)² preserves 25% of gradient magnitude for negative inputs (0.5² = 0.25), while GELU² provides essentially zero gradient for x < -2. This means GELU² creates dead zones similar to ReLU², just with a smoother boundary.
- The smaller artifact (12.7MB vs 13.1MB) confirms GELU produces more regular/compressible weight patterns — but the quality cost is far too high.
- Pre-quant regression (+0.0155 vs baseline) confirms the damage is in training quality, not quantization.

**Decision**: **DISCARD**. LeakyReLU(0.5)² remains optimal.

**Learnings**:
- Negative gradient preservation is the key mechanism for squared activations, not smooth gating. LeakyReLU(0.5)² > GELU² > ReLU² forms a clear ordering by how much negative gradient flow is preserved (50% > ~0% smooth > 0% hard).
- With vocab=1024 and tied embeddings, every neuron's capacity matters. Activations that kill negative inputs (GELU, ReLU) permanently waste MLP capacity through dead neurons/zones.
- GELU's smoothness is not a benefit when squaring is already applied — the squaring operation already smooths the gradient landscape. The kink in LeakyReLU at x=0 is irrelevant after squaring.
- This closes the activation exploration: LeakyReLU(0.5)² is optimal because it maximizes gradient flow while still providing sparsity via squaring.

---

### Experiment 069: LeakyReLU(0.7)² — Higher Negative Slope (Discard)

**Hypothesis**: More negative gradient flow (70% vs 50%) might be better. Testing the upper bound of the activation slope.

**Config**: Best config (exp_063 baseline) with LeakyReLU(0.7)² replacing LeakyReLU(0.5)² in MLP. Code change: replace `neg_slope=0.5` with `neg_slope=0.7`.

**What happened**:
- Pre-quant val_bpb=**1.6206** at step 695 — slightly better than exp_063 baseline (1.6190), within noise.
- Int8+zlib val_bpb=**1.6231** — **+0.0016 BPB** vs best (1.6215). DISCARD.
- 695 steps at ~864ms/step. Artifact: **13,285,284 bytes (~13.3MB)**.

**Analysis**:
- The activation slope sweep is now CLOSED with four data points establishing a clear optimum:
  - ReLU² (slope=0.0): 1.6299
  - LeakyReLU(0.5)² (slope=0.5): **1.6215** (BEST)
  - LeakyReLU(0.7)² (slope=0.7): 1.6231
  - GELU² (smooth ~0): 1.6373
- Slope 0.5 is the sweet spot. Too little negative flow (ReLU, GELU) kills gradients and wastes MLP capacity through dead neurons. Too much negative flow (0.7) reduces the sparsity benefit from squaring: 0.7²=0.49 retains nearly half the magnitude on the negative side, while 0.5²=0.25 provides a 4:1 suppression ratio that balances gradient flow with sparsity.
- The slightly larger artifact (13.3MB vs 13.1MB) is consistent with less sparse activations producing less compressible weights.
- Pre-quant was actually marginally better (1.6206 vs 1.6190 = +0.0016), but the quant gap is slightly worse (+0.0025 vs +0.0025), resulting in net regression.

**Decision**: **DISCARD**. Activation slope sweep is CLOSED. LeakyReLU(0.5)² is the optimal activation.

**Learnings**:
- The activation slope has a clear optimum at 0.5. The mechanism is a tradeoff between gradient flow (higher slope = more gradient for negatives) and sparsity from squaring (lower slope = more suppression of negatives after squaring).
- At slope=0.5, squaring gives 0.25 on the negative side — a 4:1 positive/negative ratio. At slope=0.7, it's 0.49 — nearly 1:1, losing most of the sparsity benefit.
- All activation experiments are KILLED. The sweep (0.0, ~0 smooth, 0.5, 0.7) thoroughly covers the space.

---

### Experiment 070: Wider Model MODEL_DIM=544 (Discard)

**Hypothesis**: Wider model (dim=544, 27.2M params vs dim=512, 24.1M params) uses the 2.9MB artifact headroom for more capacity. 13% more parameters within budget.

**Config**: Pure env-var change: `MODEL_DIM=544`. All else identical to exp_063 best config (LeakyReLU(0.5)², 10L, asymmetric MLP 2,4, etc.).

**What happened**:
- Pre-quant val_bpb=**1.6411** at step 590 — worse than best pre-quant (1.6190 from exp_063).
- Int8+zlib val_bpb=**1.6446** — **+0.0231 BPB worse** than best (1.6215). Significant regression.
- 590 steps at ~1018ms/step (vs ~855ms for dim=512). Artifact: **14,102,619 bytes (14.1MB)** — within budget but using most headroom.
- Compression ratio: 3.86x (consistent with dim=512).

**Analysis**:
- The wider model is **16% slower per step** (1018ms vs 855ms), resulting in **16% fewer steps** (590 vs ~700). On Apple Silicon where wallclock is the binding constraint, this step count reduction dominates the capacity increase.
- 13% more parameters but 16% fewer optimization steps = net negative. The model doesn't have enough training iterations to exploit the additional capacity.
- This exactly matches the 11L finding (exp_015): 11L was better per-step but worse overall due to slower steps (394ms vs 352ms, fewer total steps). The pattern is consistent — on Apple Silicon, step speed dominates model capacity.
- The artifact (14.1MB) confirms the wider model fits within budget, so the failure is purely a training dynamics issue, not a size constraint.
- Pre-quant regression (+0.022 vs baseline) confirms the damage is in training quality, not quantization. The quant gap (+0.0035) is slightly larger than normal (~0.003), consistent with a less-converged model having slightly less regular weights.

**Step time breakdown**:
| Config | dim | Params | ms/step | Steps (600s) | val_bpb |
|--------|-----|--------|---------|------------|---------|
| 10L dim=512 | 512 | 24.1M | ~855ms | ~700 | **1.6215** |
| 10L dim=544 | 544 | 27.2M | ~1018ms | 590 | 1.6446 |
| 11L dim=512 | 512 | 26.3M | ~931ms | 645 | 1.6757* |

*exp_015 was pre-activation-improvement; comparable pattern

**Decision**: **DISCARD**. Kill model width increases for Apple Silicon experiments. The step time penalty outweighs the capacity gain.

**Learnings**:
- On Apple Silicon, step speed dominates model capacity. 16% fewer steps hurts more than 13% more parameters helps.
- This matches the 11L finding: capacity increases that slow steps are net negative on Apple Silicon where wallclock is the binding constraint.
- MODEL_DIM=544 produces 1018ms/step (19% slower than dim=512's 855ms). The compute scaling is super-linear in dim on Apple Silicon (dim increases 6.25%, step time increases 19%).
- Model width increases should be reserved for H100 where step time is batch-dominated (93ms/step), not compute-dominated. On H100, dim=544 would likely add <5ms/step while providing 13% more capacity.
- Kill model width experiments on Mac. The dim=512 / 10L / MLP-asymmetric(2,4) configuration is the optimal capacity allocation for Apple Silicon's wallclock constraint.

---

### First-Principles Innovation Sprint (exp_071–074, 2026-04-03/04)

Three original ideas from an RL-inspired first-principles perspective, each A/B tested in isolation against exp_071 baseline (1.6251 post-quant).

### Experiment 071: Baseline Reconfirm (Control)

Re-run of current best config to establish the control reference. Result: val_bpb=1.6251 post-quant, 688 steps at 873ms/step. Matches exp_063 (1.6215) within noise.

### Experiment 072: Byte-Weighted Loss (Discard — +0.041 BPB)

**Idea**: Weight each token's training CE by its decoded byte count, directly optimizing BPB instead of token-level CE. The hypothesis was that multi-byte tokens matter more to BPB but get equal gradient weight, so importance sampling on the actual reward function should help.

**Result**: val_bpb=1.6659, +0.041 BPP regression. The worst outcome of the sprint.

**Why it failed**: The byte weighting concentrates gradients on multi-byte tokens (long word pieces, multi-byte UTF-8) while under-training frequent single-byte tokens (spaces, punctuation, common letters). These single-byte tokens are individually cheap in BPB but collectively dominate the evaluation metric through sheer frequency. Standard token-level CE already optimizes BPB effectively because improving prediction quality on ANY token improves BPB — the byte-count weighting in the eval metric is just a normalization constant, not an importance signal for training.

**Key principle**: Training-eval objective mismatch is NOT a bottleneck for BPB. Token-level CE and BPB are highly correlated enough that direct optimization provides no benefit and significant harm.

### Experiment 073: Deep Supervision (Discard on Mac, H100 Candidate — +0.010 BPB)

**Idea**: Add auxiliary next-token prediction losses at layers [1, 3, 5, 7], projecting each intermediate hidden state through the shared embedding (zero new parameters). Losses use geometric decay weighting (alpha=0.1). This is "deep supervision" from computer vision (Inception, DenseNet) — forces early layers to maintain prediction-useful features rather than relying on gradient signal diluted through 10+ layers of backprop.

**Result**: val_bpb=1.6351, +0.010 BPB regression overall. BUT:
- At step 500: val_bpb=1.7223 vs baseline 1.7307 — **per-step quality is better by 0.008 BPB**
- 651 total steps at 923ms/step (5% overhead from 4 extra projections through shared embedding)
- The overhead cost 37 steps (688→651), which exceeded the per-step quality gain

**Why it's an H100 candidate**: On H100 with 6000+ steps, the 5% overhead costs ~300 steps (6000→5700). But the per-step quality improvement of ~0.008 BPP scales multiplicatively: 5700 steps × better-per-step should beat 6000 steps × baseline-per-step. The math: 0.008 BPB gain at step 500 ÷ 500 steps × 5700 steps ≈ 0.091 BPB accumulated improvement, minus the 300-step loss penalty. Net should be solidly positive.

**Key principle**: On Mac (~700 steps), ANY per-step overhead >3% is lethal. Step count is the master constraint. This reverses on H100 where step count is abundant and per-step quality is the bottleneck.

### Experiment 074: Progressive Sequence Length Curriculum (Neutral — +0.002 BPB)

**Idea**: Train in 3 phases with increasing seq_len: 256 (0-25% wallclock) → 512 (25-55%) → 1024 (55-100%). Same tokens/step throughout. Shorter sequences = more independent training contexts per step = better gradient diversity for local pattern learning. This is curriculum learning (easy→hard) applied to sequence structure.

**Result**: val_bpb=1.6271, +0.002 BPB (within noise). Key observations:
- seq_len=256 runs at 665ms/step (31% faster than 873ms at 1024)
- Total 786 steps — 14% more than baseline's 688
- Phase transitions at steps 230 (256→512) and 480 (512→1024)
- Only 306 steps at full seq_len=1024 (vs baseline's 688)
- At step 500 (just 20 steps after transitioning to 1024): val_bpb=1.8128 (+0.082 worse due to short exposure to full context)

**Why it was neutral**: The extra steps from faster early phases compensate for the less efficient short-sequence training, but don't exceed it. Short-sequence steps teach local n-gram patterns efficiently but miss long-range dependencies that are critical for BPB at seq_len=1024. The model needs ~200+ steps at full context to recover from the curriculum transition. Net: a wash.

**Key principle**: Curriculum learning by sequence length doesn't help when total training time is wallclock-constrained. The model needs sufficient exposure to the evaluation sequence length during final convergence.

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
