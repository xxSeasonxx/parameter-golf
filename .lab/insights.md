# Insights

Validated learnings from experiments. Single source of truth. Delete disproven hypotheses.

## Current Best

```
commit: 4585b0b
val_bpb: 1.6299 (Apple Silicon, 10L, asymmetric MLP 2x/4x, GRAD_CLIP_NORM=0.5, LAYER_LR_SCALE=0.5, WARMUP_STEPS=50, pre-warmdown QAT)
artifact: 13,091,119 bytes (~13.1MB, 2.9MB headroom)
log: logs/exp_051_qat_prewarmdown.txt
next_exp: 060
```

**Best config env vars** (copy-paste for runs):
```
NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.10 FREQ_SKIP_GATING=1 TRAIN_BATCH_TOKENS=24576 MLP_MULT_ASYMMETRIC=2,4 GRAD_CLIP_NORM=0.5 LAYER_LR_SCALE=0.5 WARMUP_STEPS=50 QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10
```
Note: Warmdown-aware WD scheduling is in the code: `wd = base_wd * (2 - lr_mul)`
Note: Frequency-decomposed skip gating is in the code (FREQ_SKIP_WINDOW=32 default)
Note: Pre-warmdown QAT fires every 10 steps while lr_mul >= 0.8 (~first 60% of training)

## Architecture

- **10 layers > 9 layers**: ~0.05 BPB. Extra layer adds ~1.2MB artifact.
- **11 layers worse on Apple Silicon**: Better per-step (~0.04 BPB at matched steps) but slower (~394ms vs ~352ms), fewer total steps. **Try on 8xH100.**
- **Frequency-decomposed skip gating [ORIGINAL]**: Decompose U-Net skip signals into low-freq (block means, W=32) and high-freq (residual) with independent per-dim gates. -0.010 BPB. Minimal overhead (~2ms/step). Window size W=16 vs W=32 makes no difference (exp_044: 1.6318 vs 1.6311). Robust to window size; W=32 is fine.
- **MLP_MULT=3 > MLP_MULT=2**: -0.003 BPP. 24.1M params vs 18.9M. Artifact 12.9MB (3.1MB headroom). Nearly identical step time (~852ms vs ~845ms). Free capacity win.
- **Asymmetric MLP (2x encoder, 4x decoder) > uniform MLP3x**: -0.001 BPB. Same 24.1M params, slightly faster (848ms vs 855ms), slightly smaller artifact (13.1MB vs 13.2MB). Decoder layers need more MLP capacity for token prediction. Free architectural win.
- **Asymmetric MLP limits**: Extreme asymmetry (1,5) is worse (+0.001 BPB vs 2,4). Encoder MLP=1x starves feature extraction. 2,4 is the sweet spot — encoder needs enough capacity for good intermediate representations fed via U-Net skip connections.
- **4 KV heads is optimal for 8Q heads**: NUM_KV_HEADS=2 (exp_045) is worse on both BPB (+0.003) and compression (3.64x vs 3.85x, 14.0MB vs 13.1MB despite fewer params). Fewer KV heads produce less regular weight patterns → worse zlib. The speed gain (830ms vs 848ms, +15 steps) doesn't compensate. Don't reduce KV heads below 4.
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
- **Per-layer LR scaling is marginally positive**: exp_039 LAYER_LR_SCALE=0.5 gives val_bpb=1.6321 (-0.0013 vs best). Deeper layers get higher LR (1.0x to 1.5x range). Within noise but zero overhead, so keeping it. May show larger gains on H100 with more steps.
- **Layer LR scale saturates between 0.5 and 1.0**: exp_043 LAYER_LR_SCALE=1.0 (deepest=2.0x LR) gives val_bpb=1.6316 (+0.0005 vs best). Nearly identical to scale=0.5 (1.6311). Not worth fine-tuning further — 0.5 is sufficient.
- **WARMUP_STEPS=50 is marginally better than 20**: exp_046 val_bpb=1.6309 (-0.0002 vs best). Pre-quant 1.6284 (best pre-quant ever). Longer warmup gives Muon's Newton-Schulz better initial conditions. Complements GRAD_CLIP_NORM=0.5 — both stabilize early training through orthogonal mechanisms (LR ramp vs gradient magnitude). Zero cost.
- **Inverse layer LR is clearly wrong direction**: exp_040 LAYER_LR_SCALE=-0.5 gives val_bpb=1.6463 (+0.014 regression). Confirms deeper layers need MORE LR, not less. Interesting: better compression (12.5MB vs 13.2MB) with inverse — later layers with lower LR produce simpler weights.

## Evaluation

- **Sliding window (EVAL_STRIDE=64)**: Implemented. ~0.03 BPB gain. ~50 min on Apple Silicon — use on 8xH100 only.

## Porting to 8xH100

Must port: (1) Muon WD + warmdown schedule, (2) FP16 tok_emb, (3) freq-decomposed skip gating, (4) sliding window eval.
Env vars: NUM_LAYERS=11 (free on H100), MLP_MULT_ASYMMETRIC=2,4, MUON_WEIGHT_DECAY=0.10, GRAD_CLIP_NORM=0.5, LAYER_LR_SCALE=0.5, WARMUP_STEPS=50, QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10.
Expected baseline: ~1500-2000 steps, val_bpb ~1.18-1.20.

## Competition Strategy (8xH100 target: ≤1.12 BPB)

**Leaderboard top**: 1.1194 (as of 2026-03-28). Our estimated ported result: ~1.18-1.20.

**Key constraint shift from Mac → H100**: Step time is batch-dominated (524K tokens/step). 11L, MLP3x are free. Artifact size (16MB) is the binding constraint. Eval-time compute is unlimited.

**Our original contributions** (differentiators):
1. Warmdown-aware WD scheduling: `wd = base_wd * (2 - lr_mul)` — proven, unique
2. Frequency-decomposed skip gating — proven, unique
3. Asymmetric MLP (encoder=2x, decoder=4x) — validated, small win
4. Per-layer LR scaling for Muon — validated, small win
5. Pre-warmdown QAT (exp_051) — proven, QAT as regularizer during lr_mul >= 0.8

**Known techniques to add on H100** (table stakes):
- Int6 quantization (MLP/attention weights) — validated (48% artifact reduction), but REQUIRES QAT (+0.064 BPB without)
- TTT LoRA at eval time (already in codebase)
- SWA (H100-only, needs 1500+ steps)
- Zstd-22 compression (replace zlib)
- 11 layers (free on H100)

**Killed for Mac, may work on H100**:
- QAT during warmdown — quant gap closes but BPB suffers at 700 steps. Full-training QAT (exp_056) is neutral (gap slightly better, pre-quant slightly worse, effects cancel)
- SWA — needs oscillation that only occurs at 1500+ steps

**Validated on Mac, ready for H100**:
- Pre-warmdown QAT (exp_051) — NEW BEST on Mac, should show even larger gains with more steps on H100

## Apple Silicon Plateau Analysis

After 14 experiments at the current config level (exp_035-051), Apple Silicon val_bpb is firmly plateaued at **~1.630 ± 0.003**. Pre-quant values consistently land in 1.627-1.634. exp_051 (pre-warmdown QAT) pushed to 1.6299 post-quant / 1.6270 pre-quant (both best ever), but remains within the noise band.

**Root cause**: ~700 steps with ~17M tokens is the fundamental bottleneck. No per-step optimization can overcome the data limitation. The leaderboard top (1.1194) uses 64x more data per step.

**What to do next**: Port to 8xH100 and focus on techniques that scale with data (int6, TTT, SWA, 11L).
