# Insights

Validated learnings from experiments. Single source of truth. Delete disproven hypotheses.

## Current Best

```
commit: 5dffd6e
val_bpb: 1.6334 (Apple Silicon, 10L, MLP_MULT=3, GRAD_CLIP_NORM=0.5, TRAIN_BATCH_TOKENS=24576)
artifact: 13,023,930 bytes (~13.0MB, 3.0MB headroom)
log: logs/exp_033_gradclip05.txt
next_exp: 036
```

**Best config env vars** (copy-paste for runs):
```
NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.10 FREQ_SKIP_GATING=1 TRAIN_BATCH_TOKENS=24576 MLP_MULT=3 GRAD_CLIP_NORM=0.5
```
Note: Warmdown-aware WD scheduling is in the code: `wd = base_wd * (2 - lr_mul)`
Note: Frequency-decomposed skip gating is in the code (FREQ_SKIP_WINDOW=32 default)

## Architecture

- **10 layers > 9 layers**: ~0.05 BPB. Extra layer adds ~1.2MB artifact.
- **11 layers worse on Apple Silicon**: Better per-step (~0.04 BPB at matched steps) but slower (~394ms vs ~352ms), fewer total steps. **Try on 8xH100.**
- **Frequency-decomposed skip gating [ORIGINAL]**: Decompose U-Net skip signals into low-freq (block means, W=32) and high-freq (residual) with independent per-dim gates. -0.010 BPB. Minimal overhead (~2ms/step).
- **MLP_MULT=3 > MLP_MULT=2**: -0.003 BPP. 24.1M params vs 18.9M. Artifact 12.9MB (3.1MB headroom). Nearly identical step time (~852ms vs ~845ms). Free capacity win.

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

## Evaluation

- **Sliding window (EVAL_STRIDE=64)**: Implemented. ~0.03 BPB gain. ~50 min on Apple Silicon — use on 8xH100 only.

## Porting to 8xH100

Must port: (1) Muon WD + warmdown schedule, (2) FP16 tok_emb, (3) freq-decomposed skip gating, (4) sliding window eval.
Env vars: NUM_LAYERS=11 (free on H100), MLP_MULT=3, MUON_WEIGHT_DECAY=0.10, GRAD_CLIP_NORM=0.5.
Expected baseline: ~1500-2000 steps, val_bpb ~1.18-1.20.

## Competition Strategy (8xH100 target: ≤1.12 BPB)

**Leaderboard top**: 1.1194 (as of 2026-03-28). Our estimated ported result: ~1.18-1.20.

**Key constraint shift from Mac → H100**: Step time is batch-dominated (524K tokens/step). 11L, MLP3x are free. Artifact size (16MB) is the binding constraint. Eval-time compute is unlimited.

**Our original contributions** (differentiators):
1. Warmdown-aware WD scheduling: `wd = base_wd * (2 - lr_mul)` — proven, unique
2. Frequency-decomposed skip gating — proven, unique
3. Warmdown-phase QAT (NEW, to test) — integrate quant noise into warmdown schedule
4. Layer-wise quantization budget allocation (NEW, to test) — data-driven per-layer precision

**Known techniques to add** (table stakes):
- SWA (average final checkpoints, NOT EMA blending which failed)
- Int6 quantization (MLP/attention weights)
- TTT LoRA at eval time (already in codebase)
- Zstd-22 compression (replace zlib)

**Priority order for Mac testing** (validate mechanisms before H100):
1. Warmdown QAT mechanism (int8 on Mac, int6 on H100)
2. Depth-recurrent warmdown (high risk, start gentle)
