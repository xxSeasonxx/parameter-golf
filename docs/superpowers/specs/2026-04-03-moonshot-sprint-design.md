# Moonshot Sprint: Local Validation → H100 Leaderboard Push

**Date**: 2026-04-03
**Goal**: Close the 0.089 BPB gap (1.2087 → ≤1.12) via original techniques validated locally
**Target**: Leaderboard competitive (≤1.12 BPB TTT on 8xH100)

## Context

After 61 local experiments (2.41 → 1.63 BPB) and 3 H100 runs (best TTT 1.2087), we need ~0.09 BPB improvement. The competition top is 1.1194. Our H100 runs confirmed: SWA is dead everywhere, int6 is dead without STE, zstd saves 1.6MB free. The remaining gap comes from insufficient steps/data (65%), missing architecture features (20%), missing training techniques (10%), and suboptimal TTT (5%).

## Strategy: Moonshots First, Then Foundations

1. Test 6 experiments locally on Mac (relative ordering transfers to H100)
2. Take winners to H100 with full 195 shards + zstd compression
3. Stack with high-value foundations if moonshots alone aren't enough

## Experiments (in priority order)

### exp_062: Progressive Layer Growing 7L→10L [ORIGINAL]
- Train 7L for first 50% wallclock, grow to 10L. Zero-init new output projections.
- Tests: can dynamic architecture get 20% more total steps without quality loss?
- Success: BPB within 0.005 of constant 10L (the step advantage makes it a net win on H100)

### exp_063: LeakyReLU(0.5)² [KNOWN]
- One-line: replace relu² with leaky_relu(0.5)². Zero risk.
- Expected: -0.001 to -0.003 BPB

### exp_064: EMA decay=0.997 [KNOWN]
- Continuous weight averaging, replaces killed SWA. All top-4 teams use this.
- Expected: -0.001 to -0.005 BPB

### exp_065: Multi-Band Skip Gating v2 [ORIGINAL]
- 3 frequency bands (W=128, W=32, residual) instead of 2.
- Evolves our proven -0.010 BPB technique.
- Expected: -0.001 to -0.005 additional BPB

### exp_066: XSA on Decoder Layers [KNOWN]
- Exclusive Self Attention on last 3-4 layers. Zero new parameters.
- 4 of 5 top teams use this.
- Expected: -0.002 to -0.005 BPB

### exp_067: Partial RoPE 25% [KNOWN]
- RoPE on first 16/64 head dims only. Zero cost.
- Expected: -0.001 to -0.002 BPB

## H100 Deployment (after local validation)

Combine all local winners into one H100 config:
- Full 195 shards (currently only 80)
- Zstd-22 compression (free 1.6MB)
- All validated improvements from local sprint

## Success Criteria

- Local sprint: combined improvements yield -0.010+ BPB vs Mac best (1.6299)
- H100 deployment: TTT BPB ≤ 1.12
