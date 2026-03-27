# Insights

Validated learnings from experiments. Single source of truth. Delete disproven hypotheses.

## Current Best

```
commit: c8c2028
val_bpb: 1.6518 (Apple Silicon, 10L, TRAIN_BATCH_TOKENS=24576)
artifact: 10,804,460 bytes (~10.8MB, 5.2MB headroom)
log: logs/exp_024_batch24k.txt
next_exp: 025
```

**Best config env vars** (copy-paste for runs):
```
NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.10 FREQ_SKIP_GATING=1 TRAIN_BATCH_TOKENS=24576
```
Note: Warmdown-aware WD scheduling is in the code: `wd = base_wd * (2 - lr_mul)`
Note: Frequency-decomposed skip gating is in the code (FREQ_SKIP_WINDOW=32 default)

## Architecture

- **10 layers > 9 layers**: ~0.05 BPB. Extra layer adds ~1.2MB artifact.
- **11 layers worse on Apple Silicon**: Better per-step (~0.04 BPB at matched steps) but slower (~394ms vs ~352ms), fewer total steps. **Try on 8xH100.**
- **Frequency-decomposed skip gating [ORIGINAL]**: Decompose U-Net skip signals into low-freq (block means, W=32) and high-freq (residual) with independent per-dim gates. -0.010 BPB. Minimal overhead (~2ms/step).

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

## Evaluation

- **Sliding window (EVAL_STRIDE=64)**: Implemented. ~0.03 BPB gain. ~50 min on Apple Silicon — use on 8xH100 only.

## Porting to 8xH100

Must port: (1) Muon WD + warmdown schedule, (2) FP16 tok_emb, (3) sliding window eval.
Env vars: NUM_LAYERS=10 (or 11), MUON_WEIGHT_DECAY=0.10.
Expected: ~1500-2000 steps, val_bpb ~1.18-1.20 (gap is data volume, not architecture).
