# Insights

Validated learnings from experiments. Single source of truth. Delete disproven hypotheses.

## Current Best

```
commit: 966ddeb
val_bpb: 1.6309 (Apple Silicon, 10L, asymmetric MLP 2x/4x, GRAD_CLIP_NORM=0.5, LAYER_LR_SCALE=0.5, WARMUP_STEPS=50)
artifact: 13,065,155 bytes (~13.1MB, 2.9MB headroom)
log: logs/exp_046_warmup50.txt
next_exp: 048
```

**Best config env vars** (copy-paste for runs):
```
NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.10 FREQ_SKIP_GATING=1 TRAIN_BATCH_TOKENS=24576 MLP_MULT_ASYMMETRIC=2,4 GRAD_CLIP_NORM=0.5 LAYER_LR_SCALE=0.5 WARMUP_STEPS=50
```
Note: Warmdown-aware WD scheduling is in the code: `wd = base_wd * (2 - lr_mul)`
Note: Frequency-decomposed skip gating is in the code (FREQ_SKIP_WINDOW=32 default)

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

## Quantization

- **FP16 tok_emb**: Quant gap +0.0001 BPB. +0.5MB artifact.
- **WD dramatically improves compressibility**: Artifact 15.7MB → 9.9MB via WD+scheduling.

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
- **Warmdown QAT (ramping strength) is too aggressive**: exp_036 val_bpb=1.6907 (+0.057). QAT noise fights warmdown convergence. Quant gap closes to 0.0002 BPB (mechanism works!) but overall BPB suffers badly. Don't inject quant noise during the convergence-critical warmdown phase. A pre-warmdown QAT phase or constant low-strength QAT might work on H100 with more steps.
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
Env vars: NUM_LAYERS=11 (free on H100), MLP_MULT_ASYMMETRIC=2,4, MUON_WEIGHT_DECAY=0.10, GRAD_CLIP_NORM=0.5, LAYER_LR_SCALE=0.5, WARMUP_STEPS=50.
Expected baseline: ~1500-2000 steps, val_bpb ~1.18-1.20.

## Competition Strategy (8xH100 target: ≤1.12 BPB)

**Leaderboard top**: 1.1194 (as of 2026-03-28). Our estimated ported result: ~1.18-1.20.

**Key constraint shift from Mac → H100**: Step time is batch-dominated (524K tokens/step). 11L, MLP3x are free. Artifact size (16MB) is the binding constraint. Eval-time compute is unlimited.

**Our original contributions** (differentiators):
1. Warmdown-aware WD scheduling: `wd = base_wd * (2 - lr_mul)` — proven, unique
2. Frequency-decomposed skip gating — proven, unique
3. ~~Warmdown-phase QAT~~ — FAILED (exp_036, +0.057 BPB). Ramping QAT during warmdown kills convergence. Quant gap closes (0.0002) but BPB too bad. Variant: pre-warmdown constant-strength QAT on H100 may work.
5. ~~SWA~~ — FAILED on Mac. Wide (exp_037, +0.127) and narrow (exp_038, +0.003). With ~700 steps, weights converge monotonically — no oscillation to average. H100-only technique.
4. Layer-wise quantization budget allocation (NEW, to test) — data-driven per-layer precision

**Known techniques to add** (table stakes):
- ~~SWA~~ — KILLED for Mac (exp_037 wide +0.127, exp_038 narrow +0.003). H100-only (needs 1500+ steps for oscillation).
- Int6 quantization (MLP/attention weights)
- TTT LoRA at eval time (already in codebase)
- Zstd-22 compression (replace zlib)

**Priority order for Mac testing** (validate mechanisms before H100):
1. Warmdown QAT mechanism (int8 on Mac, int6 on H100)
2. Depth-recurrent warmdown (high risk, start gentle)
