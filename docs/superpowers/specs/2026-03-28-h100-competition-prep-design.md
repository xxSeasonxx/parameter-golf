# H100 Competition Prep: Local Validation of Breakthrough Techniques

**Date**: 2026-03-28
**Goal**: Implement and validate 3 breakthrough techniques on Apple Silicon so they're ready for 8xH100 deployment.
**Target**: val_bpb <= 1.12 on 8xH100 (leaderboard top: 1.1194)

## Context

Apple Silicon val_bpb is plateaued at ~1.630 after 47 experiments. The remaining gains come from:
1. More data (64x on H100) — gives ~0.45 BPB improvement
2. Better compression (int6 + zstd) — fits ~40% more effective params
3. Eval-time techniques (TTT, sliding window) — free BPB
4. Novel architecture (depth recurrence) — 2x effective params if it works

We implement and validate all three locally, then deploy together on H100.

## Technique 1: Int6 Quantization + Zstd Compression

### What
Replace int8 quantization with int6 for large matrix weights (attention, MLP). Keep embeddings and control tensors at int8/fp16/fp32. Replace zlib-9 with zstd-22 for compression.

### Why
Int6 saves 25% per quantized weight (6 bits vs 8 bits). At 24M params, this saves ~6MB of payload. Zstd-22 compresses ~10-15% better than zlib-9. Combined: artifact drops from ~13MB to ~8-9MB, freeing 7-8MB for more parameters (larger model, more layers, wider MLP).

### Implementation
- Add `quantize_float_array_int6()`: 6-bit per-row quantization with values in [-31, 31] and fp16 scales
- Pack two int6 values per 12-bit pair (or store as int8 with clipping to 6-bit range for simplicity)
- Simple approach first: store as int8 but clip to [-31, 31] with scale = max/31. This wastes 2 bits per byte but is trivial to implement and still compresses much better.
- Add `QUANT_BITS` env var (default 8, set to 6 for int6)
- Replace `zlib.compress(data, 9)` with `zstd.compress(data, 22)` — needs `import zstandard` (check if available)
- Fallback to zlib if zstandard not installed

### Validation on Mac
- Run best config with QUANT_BITS=6
- Verify artifact size drops significantly (~8-9MB)
- Measure BPB degradation from int6 vs int8 (expect +0.01-0.03 on Mac, less on H100 with more training)
- The mechanism is what we're validating, not the absolute BPB

### Success criteria
- Artifact < 10MB with int6
- Model loads back and produces reasonable val_bpb
- Quant gap (pre-quant vs post-quant) is < 0.05 BPB

## Technique 2: Pre-Warmdown Constant-Strength QAT

### What
Apply quantization-aware training (simulated quant noise) at constant low strength during the first 80% of training, then stop for clean warmdown convergence.

### Why
Our ramping QAT (exp_036) failed because it injected increasing noise during warmdown, fighting convergence. The fix: inject constant, gentle noise BEFORE warmdown, then let warmdown converge cleanly. This teaches weights to be quantization-friendly without disrupting the final convergence.

### Implementation
- Reuse `sim_quant_int8()` (or add `sim_quant_int6()` for int6)
- Apply every K steps (QAT_EVERY=10) when lr_mul >= QAT_STOP_LR_MUL (e.g., 0.8)
- Constant strength QAT_STRENGTH=0.1 (not ramping)
- `w = (1 - s) * w + s * dequant(quant(w))` with s=0.1 fixed
- Stop when warmdown starts (lr_mul drops below threshold)

### Validation on Mac
- Run with QAT_WARMDOWN=0, QAT_PREWARMDOWN=1, QAT_STRENGTH=0.1, QAT_STOP_LR_MUL=0.8
- Compare int8-quantized val_bpb vs baseline
- Look for: smaller quant gap (gap < 0.002 would be great), similar or better final BPB
- On Mac with 700 steps, QAT runs for ~560 steps, clean convergence for ~140

### Success criteria
- Quant gap reduced vs baseline (currently ~0.003)
- Overall val_bpb not worse than baseline (within noise ±0.003)

## Technique 3: Depth-Recurrent Warmdown (Soft Weight Sharing)

### What
Train with 10 (or 11) independent layers, then during the last 20% of training, add a soft weight-sharing loss between adjacent middle layers that gradually increases. By the end, the middle layers converge to nearly identical weights, effectively creating a depth-recurrent model that compresses dramatically.

### Why
On the 16MB frontier, every byte matters. If 2-3 middle layers share weights, we save ~2-4MB of artifact. This space can be used for more unique layers, wider model, or simply better compression. The model gets the training benefit of independent layers but the compression benefit of shared layers.

### Implementation
- Add `DEPTH_RECURRENCE_WARMDOWN` env var (0=disabled)
- Add `DEPTH_RECURRENCE_LAYERS` env var (e.g., "3,4,5" — which layers to tie)
- Add `DEPTH_RECURRENCE_STRENGTH` env var (max alpha, default 0.5)
- During warmdown (lr_mul < 1.0), add L2 regularization between designated layers:
  `L_share = alpha * sum(||W_i - W_j||^2 for i,j in pairs)`
  where alpha = DEPTH_RECURRENCE_STRENGTH * (1 - lr_mul)
- This doesn't require architecture changes — it's just a regularization loss
- After training, check if tied layers are similar enough to share weights at quantization time

### Validation on Mac
- Run with DEPTH_RECURRENCE_WARMDOWN=1, layers 3,4,5 (middle of the U-Net)
- Measure: (a) val_bpb, (b) weight similarity between tied layers, (c) artifact size after sharing
- If layers 3,4,5 end up with <1% L2 distance, we can literally replace them with a single set of weights at serialization → massive compression win

### Success criteria
- Weight similarity > 95% between tied layers after training
- val_bpb degradation < 0.005 from forced weight sharing
- Artifact size reduction > 2MB if weights are shared at serialization

## Experiment Sequence

| Exp | Technique | Config Change | What to Look For |
|-----|-----------|--------------|-----------------|
| 048 | Zstd compression | Replace zlib with zstd-22 | Artifact size reduction |
| 049 | Int6 quantization | QUANT_BITS=6 | Artifact size, quant gap |
| 050 | Pre-warmdown QAT | QAT constant strength 0.1 | Quant gap reduction |
| 051 | Int6 + QAT combined | Both together | Best compression + quality |
| 052 | Depth recurrence (gentle) | Tie layers 3,4,5, alpha=0.1 | Weight similarity |
| 053 | Depth recurrence (stronger) | alpha=0.5 | Weight similarity, BPB |

## Code Changes Summary

All changes in `train_gpt_mlx.py` only:
- ~30 lines: int6 quantization functions
- ~10 lines: zstd compression (with zlib fallback)
- ~15 lines: pre-warmdown QAT (modify existing QAT code)
- ~20 lines: depth-recurrent weight sharing loss
- Total: ~75 lines of new code

## Non-Goals (for later, on H100)
- TTT LoRA implementation (already in train_gpt.py, just enable)
- Porting to train_gpt.py (will copy validated techniques)
- seq_len=2048 training (env var change on H100)
- Actual leaderboard submission
