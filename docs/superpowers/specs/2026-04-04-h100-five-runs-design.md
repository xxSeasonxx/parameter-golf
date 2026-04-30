# H100 Competition: 6 Differentiated Runs — Design Spec

**Date**: 2026-04-04
**Target**: Leaderboard placement (<=1.12 BPB), ideally top 3 (<=1.11)
**Current best**: 1.2087 TTT BPB (H100), 1.6215 (Mac)
**Leaderboard**: 1.1194

## The 6 Runs

| # | Name | Philosophy | Layers | Unique Feature | Risk | Expected BPB |
|---|------|-----------|--------|---------------|------|-------------|
| 1 | Proven Foundation | Safe baseline | 11L | All proven innovations stacked | Low | 1.14-1.16 |
| 2 | Deep Capacity | Bet on depth | 13L | WD scheduling enables 13L in 16MB | Medium | 1.12-1.14 |
| 3 | Training Amplifier | Bet on per-step quality | 11L | Deep supervision (2 taps, alpha=0.05) | Medium | 1.13-1.15 |
| 4 | Calibrated Compression | Maximize capacity | 13L | Our own per-row MSE-optimal clipping | High | 1.11-1.14 |
| 5 | Progressive Deep | Moonshot | 8L->13L | Layer growing + deep supervision on new layers | High | 1.12-1.17 |
| **6** | **Everything All At Once** | **Stack all innovations** | **13L** | **13L + deep supervision + calibrated quant** | **Medium** | **1.10-1.13** |

## Common Foundation

All runs share: full 195 shards, EMA(0.997), warmdown WD(0.10), freq skip gating, LeakyReLU(0.5)², asymmetric MLP 2x/4x, grad clip 0.5, pre-warmdown QAT, zstd-22, sliding window eval, per-layer LR scaling, FP16 embeddings.

## Implementation: Port MLX innovations to train_gpt.py

1. Warmdown-aware WD in Muon: `wd = base_wd * (2 - lr_mul)`
2. Frequency-decomposed skip gating (lo/hi bands, W=32)
3. Pre-warmdown QAT (constant strength, stop at lr_mul<0.8)
4. LeakyReLU(0.5)² activation
5. Asymmetric MLP (encoder=2x, decoder=4x)
6. Per-layer LR scaling
7. EMA shadow model
8. Zstd-22 serialization
9. Calibrated per-row quantization (Runs 4, 6)
10. Progressive layer growing (Run 5)
11. Deep supervision (Runs 3, 5, 6)

## Run Order

1 -> 2 -> 3 -> 4 -> 6 -> 5
