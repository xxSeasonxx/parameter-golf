# First-Principles Innovation Sprint: Design Spec

**Date**: 2026-04-03
**Target**: H100 competition (leaderboard 1.1194, our best 1.2087, gap 0.089 BPB)
**Baseline**: val_bpb=1.6215 (Mac), 1.2087 TTT BPB (H100)

## Problem Statement

After 70 experiments, all conventional levers (hyperparameters, architecture variants, weight averaging, quantization) are exhausted. We need fundamentally new ideas to close the 0.089 BPB gap to the leaderboard.

## Three Innovations

### A. Byte-Weighted Loss (BPB-Aligned Training)

**Core insight**: Training optimizes token-level cross-entropy, but the competition metric is bits-per-byte (BPB). These are different objectives. A 6-byte token contributes 6x more to BPB than a 1-byte token, but both get equal gradient weight during training.

**Implementation**:
1. Pass `base_bytes_lut` and `has_leading_space_lut` into the model's loss function
2. Compute per-token cross-entropy (reduction="none")
3. Weight each token's loss by its byte count: `weighted_loss = (per_token_ce * byte_weights).sum() / byte_weights.sum()`
4. Clamp byte weights to [0.5, 3.0] for gradient stability

**Why novel**: Nobody in the competition optimizes BPB directly during training. Everyone trains on token CE and hopes the BPB correlation holds. We're doing importance sampling on the actual evaluation metric.

**Artifact impact**: Zero. No new parameters.
**Compute impact**: Zero. Per-token CE is the same cost as mean CE.

### B. Deep Supervision with Disposable Auxiliary Heads

**Core insight**: In 10+ layer transformers, gradients reaching early layers are heavily diluted. Early layers learn slowly, wasting the most time-constrained compute budget.

**Implementation**:
1. Add `aux_heads`: list of `CastedLinear(dim, vocab_size)` — one per tap point
2. Tap hidden states at layers [1, 3, 5, 7] (every 2nd layer through the network)
3. Each tap: project hidden state to logits, compute CE loss
4. Total loss = `main_loss + sum(alpha^(num_taps - i) * aux_loss_i)` where alpha=0.1
5. Before serialization: strip all `aux_heads` from state dict — zero artifact cost

**Why novel**: Deep supervision is well-established in computer vision (Inception, DenseNet, U-Net) but has never been applied to GPT pre-training in this competition setting. The key insight is that auxiliary heads are disposable — they exist only during training.

**Artifact impact**: Zero. Auxiliary heads are stripped before quantization.
**Compute impact**: ~5% overhead (4 extra dim->vocab linear projections + CE losses per step).

### C. Progressive Sequence Length Curriculum

**Core insight**: Most language model learning is local patterns (character n-grams, word boundaries, grammar). Long-range dependencies are important but sparse. Training at seq_len=1024 from step 1 wastes early compute on long-range patterns the model can't yet use.

**Implementation**:
1. Define 3 phases based on wallclock percentage:
   - Phase 1 (0-25%): seq_len=256, batch_seqs=2048 (4x more independent sequences)
   - Phase 2 (25-55%): seq_len=512, batch_seqs=1024 (2x more)
   - Phase 3 (55-100%): seq_len=1024, batch_seqs=512 (standard)
2. At each transition: update effective seq_len, recompile loss functions
3. Total tokens per step stays constant (524K)
4. RoPE naturally supports variable sequence length

**Why novel**: Curriculum learning by sequence length hasn't been applied in this competition. The insight that local patterns dominate early learning and can be trained more efficiently with shorter sequences is from the RL curriculum learning literature.

**Artifact impact**: Zero. Final model architecture is identical.
**Compute impact**: Phase transitions require recompilation (~2-3s each, twice total). Phase 1 may be faster per step due to shorter sequences.

## Experimental Plan (Mac A/B Isolation)

| Exp | Innovation | Compare vs | Code branch |
|-----|-----------|-----------|-------------|
| 071 | None (baseline reconfirm) | exp_063 | Current best |
| 072 | A only (byte-weighted loss) | 071 | Current + byte loss |
| 073 | B only (deep supervision) | 071 | Current + aux heads |
| 074 | C only (seq len curriculum) | 071 | Current + curriculum |
| 075 | Stack winners | 071 | Combined |

Each experiment: ITERATIONS=2000, VAL_LOSS_EVERY=500, best config env vars.

## Success Criteria

- Any single innovation showing >= -0.003 BPB on Mac is a winner (take to H100)
- Combined stack should show >= -0.005 BPB on Mac
- On H100, combined innovations target closing >= 0.03 BPB of the 0.089 gap
