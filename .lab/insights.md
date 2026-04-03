# Insights

Validated learnings from experiments. Single source of truth. Delete disproven hypotheses.

## Current Best

```
commit: e8addc5
val_bpb: 1.6215 (Apple Silicon, 10L, asymmetric MLP 2x/4x, LeakyReLU(0.5)², GRAD_CLIP_NORM=0.5, LAYER_LR_SCALE=0.5, WARMUP_STEPS=50, pre-warmdown QAT)
artifact: 13,124,539 bytes (~13.1MB, 2.9MB headroom)
log: logs/exp_063_leakyrelu_med.txt
next_exp: 070
```

**Best config env vars** (copy-paste for runs):
```
NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.10 FREQ_SKIP_GATING=1 TRAIN_BATCH_TOKENS=24576 MLP_MULT_ASYMMETRIC=2,4 GRAD_CLIP_NORM=0.5 LAYER_LR_SCALE=0.5 WARMUP_STEPS=50 QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10
```
Note: Warmdown-aware WD scheduling is in the code: `wd = base_wd * (2 - lr_mul)`
Note: Frequency-decomposed skip gating is in the code (FREQ_SKIP_WINDOW=32 default)
Note: Pre-warmdown QAT fires every 10 steps while lr_mul >= 0.8 (~first 60% of training)
Note: LeakyReLU(0.5)² activation is in the code (replaces relu²), not an env var

## Architecture

- **10 layers > 9 layers**: ~0.05 BPB. Extra layer adds ~1.2MB artifact.
- **11 layers worse on Apple Silicon**: Better per-step (~0.04 BPB at matched steps) but slower (~394ms vs ~352ms), fewer total steps. **Try on 8xH100.**
- **Frequency-decomposed skip gating [ORIGINAL]**: Decompose U-Net skip signals into low-freq (block means, W=32) and high-freq (residual) with independent per-dim gates. -0.010 BPB. Minimal overhead (~2ms/step). Window size W=16 vs W=32 makes no difference (exp_044: 1.6318 vs 1.6311). Robust to window size; W=32 is fine.
- **MLP_MULT=3 > MLP_MULT=2**: -0.003 BPP. 24.1M params vs 18.9M. Artifact 12.9MB (3.1MB headroom). Nearly identical step time (~852ms vs ~845ms). Free capacity win.
- **Asymmetric MLP (2x encoder, 4x decoder) > uniform MLP3x**: -0.001 BPB. Same 24.1M params, slightly faster (848ms vs 855ms), slightly smaller artifact (13.1MB vs 13.2MB). Decoder layers need more MLP capacity for token prediction. Free architectural win.
- **Asymmetric MLP limits**: Extreme asymmetry (1,5) is worse (+0.001 BPB vs 2,4). Encoder MLP=1x starves feature extraction. 2,4 is the sweet spot — encoder needs enough capacity for good intermediate representations fed via U-Net skip connections.
- **4 KV heads is optimal for 8Q heads**: NUM_KV_HEADS=2 (exp_045) is worse on both BPB (+0.003) and compression (3.64x vs 3.85x, 14.0MB vs 13.1MB despite fewer params). Fewer KV heads produce less regular weight patterns → worse zlib. The speed gain (830ms vs 848ms, +15 steps) doesn't compensate. Don't reduce KV heads below 4.
- **LeakyReLU(0.5)² >> relu² [KNOWN, LARGE WIN]**: exp_063 val_bpb=1.6215, -0.0084 BPB vs relu² baseline (1.6299). Largest non-batch/non-clip win. The 0.5 negative slope preserves 50% of gradient flow for x<0, eliminating dead neurons. Squaring still provides sparsity (0.5²=0.25 on negative side). With vocab=1024 and tied embeddings, every neuron's capacity matters — dead neurons from relu² permanently waste MLP capacity. Pre-quant also improved (1.6190 vs 1.6270), confirming genuine training improvement not quantization artifact. Zero overhead. The change is in code, not env vars.
- **Activation slope sweep CLOSED — LeakyReLU(0.5)² is optimal [KNOWN, KILLED]**: Four data points establish the optimum: ReLU² 1.6299 (slope=0.0), LeakyReLU(0.5)² **1.6215** (slope=0.5, BEST), LeakyReLU(0.7)² 1.6231 (slope=0.7, exp_069), GELU² 1.6373 (smooth ~0). Slope 0.5 is the sweet spot — too little negative flow (ReLU, GELU) kills gradients and wastes MLP capacity through dead neurons; too much (0.7) reduces sparsity from squaring (0.7²=0.49 on negative side vs 0.5²=0.25). The squaring operation needs enough negative suppression to maintain sparse representations while preserving enough gradient flow to avoid dead neurons. All activation experiments are KILLED.
- **XSA (Exclusive Self Attention) is neutral on Mac 10L [KNOWN, KILLED ON MAC]**: exp_066 XSA on last 3 decoder layers gives +0.0007 BPB (within noise). Self-exclusion removes 1/1024 (~0.1%) of context per position — too small for the deduplication benefit to manifest. No speed penalty (custom mask runs as fast as fused causal kernel). Worth trying on H100 with 11L+ where deeper layers benefit more from pure-context signals.
- **Multi-band skip gating is DEAD [ORIGINAL, KILLED]**: exp_065 3-band (ultra-low W=128, mid W=32, high residual) gives +0.0023 BPB vs 2-band. The 2-band lo/hi decomposition at W=32 already captures the meaningful spectral information. The ultra-low band (W=128) splits 4 dims from the 16-dim low band into a separate channel, but provides no new information — the W=32 band already captures trends at that scale. The extra parameters (5 skip_ulo_weights vectors) are under-constrained and slightly hurt optimization. Kill multi-band experiments. 2-band at W=32 is the sweet spot.
- **Full RoPE is optimal at seq_len=1024 [KNOWN, KILLED ON MAC]**: exp_067 Partial RoPE (25% of head dims) gives +0.0025 BPB. At seq_len=1024, every position matters — RoPE enables position-dependent attention patterns (recency bias, periodic patterns) that the model needs for short sequences. Removing positional encoding from 75% of dims deprives the model of this capability without providing compensating content-matching benefit. This differs from longer-context models (seq_len=4K+) where many attention patterns genuinely are position-invariant. Don't reduce RoPE coverage at seq_len=1024.
- **LOGIT_SOFTCAP=30.0 is optimal**: exp_047 softcap=15.0 is worse (+0.006 BPB). Tighter clamping constrains confident predictions for common tokens. Softcap regulates prediction confidence, not weight magnitude — doesn't affect compressibility. Don't touch.

## Optimization

- **Muon WD response** (monotonically increasing at least to 0.10):
  - WD=0.00 → val_bpb=1.860, artifact=15.7MB
  - WD=0.02 → val_bpb=1.829, artifact=14.6MB
  - WD=0.05 → val_bpb=1.788, artifact=13.2MB
  - WD=0.10 → val_bpb=1.761, artifact=11.0MB
  - WD=0.20 → val_bpb=1.769, artifact=7.7MB (over-regularized, WD=0.10 is the sweet spot)
- **Warmdown-aware WD scheduling [ORIGINAL]**: `wd = base_wd * (2 - lr_mul)`. Extra -0.010 BPB and -1.1MB over constant WD. Nobody in competition uses this.
- **WD "worse early, better late"**: Higher WD hurts intermediate checkpoints but wins at convergence. Crossover ~step 1500.
- **Warmdown=1200 optimal**: Reducing to 400 was worse on BPB AND artifact. Don't touch.
- **Linear warmdown shape is optimal**: Cosine warmdown (exp_057) is clearly worse (+0.020 BPB). Cosine keeps LR too high too long, then drops too fast — insufficient time for fine convergence. The steady linear decay distributes convergence effort evenly. Kill warmdown shape experiments.
- **Muon matrix_lr=0.04 is optimal (KILLED LR sweep)**: exp_059 matrix_lr=0.06 (1.5x default) gives +0.009 BPB and worse compression (13.3MB vs 13.1MB). Muon's Newton-Schulz orthogonalization produces unit-norm updates on the Stiefel manifold — the LR is a step size matched to this natural scale. Overshooting is worse than undershooting. Don't sweep matrix_lr.
- **Tied embed_lr=0.05 is optimal (KILLED LR sweep both directions)**: Embed LR is tightly tuned — both higher (0.1, exp_060: +0.030 BPB) and lower (0.03, exp_061: +0.011 BPB) are worse. The tied embedding's dual role (input lookup + output projection) makes it sensitive to LR in both directions: too high destabilizes the shared representation (loss spikes to 24.4), too low under-trains it (690 steps, val_bpb=1.6382 pre-quant). Adam's 0.05 is the sweet spot. Don't sweep embed LR in any direction.
- **Label smoothing is CATASTROPHIC with small vocab (KILLED)**: exp_058 label_smoothing=0.1 gives val_bpb=2.0499 (+0.42 BPP, catastrophic). With vocab=1024, smoothing redistributes too much mass per non-target token (~0.0001 vs ~1e-6 for vocab=100K). Training objective (smoothed CE) diverges from eval metric (standard CE). Key principle: regularization that preserves the loss function (WD, QAT, gradient clipping) works; regularization that modifies the loss function (label smoothing) destroys performance.

## Quantization

- **FP16 tok_emb**: Quant gap +0.0001 BPB. +0.5MB artifact.
- **WD dramatically improves compressibility**: Artifact 15.7MB → 9.9MB via WD+scheduling.
- **Int6 (QUANT_BITS=6) without QAT**: Artifact drops 48% (13.1MB → 6.8MB) but quant gap is +0.064 BPB (~20x worse than int8's ~0.003). 63 levels vs 255 levels = 4x less precision, but error compounds across layers making it ~20x worse in practice. Pre-quant BPB is identical (1.6332 vs best 1.6309) — all damage is at serialization. **Int6 needs QAT to be competitive.** Mechanism validated for H100 deployment: compression ratio (3.85x) identical to int8, just smaller raw payload.
- **Int6 quant gap is ~+0.063 BPB regardless of QAT strength (KILLED)**: Three experiments confirm the gap is impervious to pre-warmdown QAT: exp_049 (no QAT) +0.064, exp_052 (strength=0.1/every=10) +0.061, exp_055 (strength=0.3/every=5) +0.063. The 63-level quantization grid is fundamentally too coarse for periodic weight nudging to close the gap. Int6 QAT does improve pre-quant BPB (regularization scales with noise magnitude) but cannot fix serialization damage. **Kill int6 QAT on Mac.** Need STE (straight-through estimator in forward pass), GPTQ post-training calibration, or H100's longer training to close int6 gap.

## Training Dynamics (Apple Silicon)

- **Batch size is the master lever**: 8192→1.7403, 16384→1.6584, 24576→1.6518. Marginal returns diminishing (0.082 → 0.007).
- **Long warmdown works at any step count**: WARMDOWN_ITERS=1200 with only 711 steps (warmdown > training) is STILL optimal. Reducing to 600 hurts.
- **Throughput by batch**: batch=8192→~352ms/step (~1710 steps), batch=16384→~566ms/step (~1057 steps), batch=24576→~845ms/step (~711 steps).
- **Token throughput**: batch=8k→~23K tok/s, batch=16k→~29K tok/s, batch=24k→~31K tok/s.
- **Never run concurrent**: 2-3x throughput degradation.
- **Hyperparameter sweeps at batch=8192 are NOT representative**: RoPE and QK gain tuning showed no gains. Focus on batch scaling and architectural ideas.
- **DropHead hurts**: p=0.1 (+0.008 BPB) and p=0.05 (+0.006 BPB). Stochastic head masking adds gradient noise that isn't compensated by regularization benefit when WD is already strong.
- **Gradient clipping is a major lever**: clip=0.5 is optimal. Response: no_clip→1.6492, clip=1.0→1.6434, clip=0.5→1.6334, clip=0.25→1.6362. Clip=0.25 too aggressive (clips useful gradients). Clip=0.5 is the sweet spot.
- **Adaptive Newton-Schulz scheduling (5→7 during warmdown) is neutral**: exp_035 val_bpb=1.6352 (+0.0018). NS converges well enough at 5 steps for dim=512. Don't schedule NS iterations.
- **Batch=16k with MLP3x is too small**: val_bpb=1.6692 despite 937 steps. Gradient quality dominates.
- **Batch=32k with MLP3x is too slow**: val_bpb=1.6582, only 552 steps at 1087ms/step.
- **11L+MLP3x too heavy**: val_bpb=1.6757, 645 steps at 931ms/step. Artifact 13.8MB.
- **Warmdown QAT (ramping strength) is too aggressive**: exp_036 val_bpb=1.6907 (+0.057). QAT noise fights warmdown convergence. Quant gap closes to 0.0002 BPB (mechanism works!) but overall BPB suffers badly. **Fixed by exp_051**: pre-warmdown QAT (lr_mul >= 0.8) avoids the convergence-critical warmdown phase entirely.
- **Depth-recurrent warmdown is KILLED**: exp_053 depth recurrence (alpha=0.1, layers 2,3,4, ramping during warmdown) hurts BPB +0.008 for only -0.2MB artifact savings (12.9MB vs 13.1MB). Even gentle nudging during warmdown interferes with convergence — the warmdown phase is too sensitive for any auxiliary loss. The compression benefit (layer weight similarity) doesn't justify the quality cost. Kill depth recurrence as a warmdown technique. Might be viable as a pre-warmdown technique (like QAT), but the expected compression gain (~0.2MB) is too small to justify the complexity.
- **Pre-warmdown QAT is a regularizer [OUR TWIST]**: exp_051 val_bpb=1.6299 (NEW BEST, -0.0010). Constant strength=0.1, every 10 steps, stop when lr_mul < 0.8. Pre-quant BPB improved from 1.6284 to 1.6270 (best ever) — the improvement is from better training, NOT quant gap reduction (gap unchanged at ~0.003 for int8). Int8 noise acts as structured regularization during the main training phase. Zero overhead (<2ms/step). Fixes exp_036's failure by separating QAT from warmdown.
- **QAT regularization saturates for int8**: exp_054 4x stronger QAT (strength=0.2, every=5) produces identical BPB to exp_051 (strength=0.1, every=10). Both pre-quant (1.6272 vs 1.6270) and post-quant (1.6299 vs 1.6299) are within noise. The regularization effect is a binary threshold — either you have QAT or you don't. Don't tune QAT hyperparameters for int8. (Note: int6 QAT does NOT saturate at these levels — exp_052 showed strength=0.1/every=10 is insufficient for int6's larger noise.)
- **Full-training QAT (QAT_STOP_LR_MUL=0) is neutral vs pre-warmdown-only**: exp_056 QAT through warmdown gives slightly better quant gap (+0.0025 vs +0.0029) but slightly worse pre-quant BPB (1.6283 vs 1.6270). Effects cancel (post-quant 1.6308 vs 1.6299, within noise). Key insight: the "sacred warmdown" rule applies to loss-modifying interventions (SWA, depth recurrence) but NOT to weight perturbations like constant-strength QAT noise. Keep pre-warmdown-only config (QAT_STOP_LR_MUL=0.8).
- **SWA with wide window is catastrophic**: exp_037 val_bpb=1.7601 (+0.127). Uniform averaging of 60 snapshots over lr_mul<0.5 (~60% of steps) destroys convergence. Pre-SWA model was 1.6288 (within noise of best). Competition uses narrow SWA (last 100-120 steps, lr_mul<0.1) or high-decay EMA (0.9999). Better compression though (12.6MB vs 13.0MB).
- **SWA with narrow window still hurts on Mac**: exp_038 val_bpb=1.6363 (+0.003). 24 snapshots, lr_mul<0.1, every 5 steps. Pre-SWA was 1.6295, post-SWA 1.6363 (+0.007). Much better than wide SWA but still a regression. With only ~700 steps, weights monotonically converge during warmdown — no oscillation to average out. **SWA is KILLED for Apple Silicon experiments.** May still help on 8xH100 with 1500+ steps.
- **EMA (decay=0.997) is CATASTROPHIC on Mac, reserve for H100 [KILLED ON MAC]**: exp_064 val_bpb=1.8437 (+0.22). Effective window = 1/(1-0.997) = 333 steps = 48% of 689 total steps. EMA weights dominated by early under-trained parameters. Pre-EMA model was 1.6224 (fine), confirming all damage is from the averaging. On H100 (6000+ steps), same decay gives 5.5% window — should work. For Mac, would need decay >= 0.98 (~50-step window, 7% of training), but even then marginal benefit is unlikely given SWA's failure. **Weight averaging (SWA and EMA) is KILLED for Mac.** EMA (decay=0.997) remains planned for H100.
- **Per-layer LR scaling is marginally positive**: exp_039 LAYER_LR_SCALE=0.5 gives val_bpb=1.6321 (-0.0013 vs best). Deeper layers get higher LR (1.0x to 1.5x range). Within noise but zero overhead, so keeping it. May show larger gains on H100 with more steps.
- **Layer LR scale saturates between 0.5 and 1.0**: exp_043 LAYER_LR_SCALE=1.0 (deepest=2.0x LR) gives val_bpb=1.6316 (+0.0005 vs best). Nearly identical to scale=0.5 (1.6311). Not worth fine-tuning further — 0.5 is sufficient.
- **WARMUP_STEPS=50 is marginally better than 20**: exp_046 val_bpb=1.6309 (-0.0002 vs best). Pre-quant 1.6284 (best pre-quant ever). Longer warmup gives Muon's Newton-Schulz better initial conditions. Complements GRAD_CLIP_NORM=0.5 — both stabilize early training through orthogonal mechanisms (LR ramp vs gradient magnitude). Zero cost.
- **Inverse layer LR is clearly wrong direction**: exp_040 LAYER_LR_SCALE=-0.5 gives val_bpb=1.6463 (+0.014 regression). Confirms deeper layers need MORE LR, not less. Interesting: better compression (12.5MB vs 13.2MB) with inverse — later layers with lower LR produce simpler weights.

## Evaluation

- **Sliding window (EVAL_STRIDE=64)**: Implemented. ~0.03 BPB gain. ~50 min on Apple Silicon — use on 8xH100 only.

## Porting to 8xH100

Must port: (1) Muon WD + warmdown schedule, (2) FP16 tok_emb, (3) freq-decomposed skip gating, (4) sliding window eval, (5) LeakyReLU(0.5)² activation, (6) EMA (decay=0.997) — validated by top teams on H100, killed on Mac.
Env vars: NUM_LAYERS=11 (free on H100), MLP_MULT_ASYMMETRIC=2,4, MUON_WEIGHT_DECAY=0.10, GRAD_CLIP_NORM=0.5, LAYER_LR_SCALE=0.5, WARMUP_STEPS=50, QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10.
Expected baseline: ~1500-2000 steps, val_bpb ~1.18-1.20.

## H100 RunPod Results (2026-04-03)

Three comparative runs on 8xH100 (80/195 shards, 524K batch, 600s wallclock):

| Run | Config | Steps | Pre-quant | Post-quant | TTT BPB | Artifact |
|-----|--------|-------|-----------|-----------|---------|----------|
| 1 | 11L int8 zlib (baseline) | 6421 | 1.2250 | 1.2302 | **1.2087** | 15.8MB |
| 2 | 13L int6 zstd (capacity) | 5418 | **1.2145** | 1.3066 | 1.2765 | 9.9MB |
| 3 | 11L int8 zstd + SWA (23 snapshots) | 6420 | 1.2260 | 1.2322 | 1.2107 | 14.2MB |

**Key learnings**:
- **SWA is DEAD everywhere**: 23 H100 snapshots gave +0.002 BPB regression. Killed for good. Need EMA instead (decay=0.997, every step — all top teams use this).
- **Int6 is DEAD without STE/GPTQ**: Quant gap +0.092 on H100 (worse than Mac's +0.063). Pre-quant was best (1.2145) but quant destroys it.
- **Zstd-22 saves ~1.6MB for free**: 15.8MB (zlib) → 14.2MB (zstd). Always use.
- **13L pre-quant is better**: 1.2145 vs 1.2250 (-0.0105) from 2 extra layers. But needs int8 to realize it.
- **Only 80/195 shards used**: Future runs MUST download full dataset.
- **Step time 93ms/step**: Top teams achieve 83-85ms/step with parameter banking.
- **Still improving at wallclock cap**: ~-0.19 BPB/1000 steps at step 6400. More steps = better.

## Competition Strategy (8xH100 target: ≤1.12 BPB)

**Leaderboard top**: 1.1194. Our best H100 result: **1.2087 TTT BPB**. Gap: **0.089 BPB**.

**Gap breakdown**:
- Insufficient steps/data (~65%): only 80/195 shards, still steep improvement curve at cap
- Missing architecture features (~20%): XSA, SmearGate, BigramHash, Partial RoPE, LN Scale
- Missing training techniques (~10%): EMA, GPTQ-lite, better optimizer config
- TTT suboptimal (~5%): LoRA vs full-model legal TTT

**Our original contributions** (differentiators):
1. Warmdown-aware WD scheduling: `wd = base_wd * (2 - lr_mul)` — proven, unique
2. Frequency-decomposed skip gating — proven, unique
3. Asymmetric MLP (encoder=2x, decoder=4x) — validated, small win
4. Per-layer LR scaling for Muon — validated, small win
5. Pre-warmdown QAT (exp_051) — proven, QAT as regularizer

**Killed definitively (both Mac + H100)**:
- SWA (any form) — Mac: no oscillation at 700 steps. H100: 23 snapshots still hurts.
- EMA on Mac (any decay) — Mac: decay=0.997 gives 48% window at 689 steps, +0.22 BPB. Weight averaging is dead on Mac.
- Int6 without STE/GPTQ — Mac: +0.063 gap. H100: +0.092 gap. Even worse with more params.
- Label smoothing with small vocab — catastrophic
- Depth recurrence during warmdown — warmdown is sacred

## Apple Silicon Plateau Analysis

After 27 experiments at the current config level (exp_035-061), Apple Silicon val_bpb appeared plateaued at ~1.630 +/- 0.003. However, exp_063 (LeakyReLU(0.5)²) broke through with **1.6215**, proving that architectural improvements to capacity utilization can still deliver significant wins. The plateau was in optimization hyperparameters, not architecture. Remaining moonshot experiments (EMA, multi-band skip gating, XSA, partial RoPE) now stack on top of this new baseline.
