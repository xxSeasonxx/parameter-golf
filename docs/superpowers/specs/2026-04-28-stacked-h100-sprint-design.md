# Stacked H100 Sprint — Design

**Date:** 2026-04-28 (revised 2026-04-28 after eng review + outside voice)
**Branch (target):** new feature branch off `lab/mar29`
**Author:** Season + Claude (team brainstorm)
**Status:** REVISED — cheap-winners-first redirect

## Revision note

Original plan stacked L0 (sliding window) + L1 (LSQ int4 QAT + sensitivity sweep) + L2 (DiffAttn + DyT + Polar Express + QK-Norm v2). Eng review (`/plan-eng-review`) plus an independent Claude subagent challenge surfaced four substantial concerns:

1. **Post-hoc sensitivity sweep is unsound.** Sweeping a non-LSQ-trained checkpoint with naive int4 PTQ measures PTQ-int4 sensitivity, not LSQ-int4 sensitivity. Different rankings. Would burn one full H100 run on a misallocated manifest.
2. **DiffAttn × shared-KV breaks the noise-cancellation prior.** Original DiffAttn splits both Q and K. Sharing K (forced by GQA) means the two softmaxes are correlated by construction. Likely a regression at our scale.
3. **Expected impact was double-counted.** Realistic ceiling ~0.08 BPB, not 0.13. Most likely landing 1.13–1.16, not 1.10.
4. **Highest-variance work was front-loaded.** Plan committed to LSQ + DiffAttn (research bets) before checking whether cheap drop-in changes (Polar Express, DyT, extended TTT) close most of the gap.

**Redirect: cheap winners first.** New L1' = Polar Express NS coefficients + DyT (replaces RMSNorm) + extended TTT (rank 16, smaller chunks). One H100 run, low risk, drop-in changes. Then a decision gate based on actual BPB delta. LSQ and DiffAttn move to `TODOS.md` as deferred research bets, runnable as separate sprints if the cheap winners don't close the gap.

## Current sprint scope (post-redirect)

| Phase | Components | Risk | Estimated H100 cost |
|---|---|---|---|
| L0 | Port `eval_val_sliding` (stride 64) from `train_gpt_mlx.py` to `train_gpt.py`. Add 50-step compile warmup. Regression test for `EVAL_STRIDE=0`. | Low | 1 run (~$15) |
| L1' (cheap winners) | Polar Express NS coefficients + DyT replacing RMSNorm + extended TTT (rank 16, chunk 128). Single ablation. | Low–medium | 1 run (~$15) |
| Decision gate | If BPB ≤ 1.15: ship. If 1.15 < BPB ≤ 1.17: evaluate one deferred bet from TODOS.md. If BPB > 1.17: escalate. | — | 0 |

**Hard target:** post-quant BPB ≤ 1.15 with sliding-window stride-64 eval after L1'.
**Stretch target:** ≤ 1.13 (top-10 territory).
**Cost ceiling:** 2 H100 runs (~$30) before decision gate.

The originally-planned L1 (LSQ + sensitivity + serializer) and L2 (DiffAttn + QK-Norm v2) detailed designs remain in §4.2, §4.3.1, §4.3.4 below as deferred reference. Do not implement until the decision gate fires.

## Decisions taken in eng review (locked)

1. **Three-PR split → single PR.** With LSQ and DiffAttn deferred, scope is small enough for one PR.
2. **Sensitivity sweep design.** N/A in revised scope (LSQ deferred). When we revisit, use proper short LSQ-aware probes per tensor, not post-hoc PTQ.
3. **Pre-flight DiffAttn smoke.** N/A in revised scope (DiffAttn deferred).
4. **Baseline reference.** Use git tag `baseline-3e34098`, no `train_gpt_baseline.py` snapshot file.
5. **MLX role.** Crash-screen only. Never trust MLX BPB. H100 is the quality oracle.
6. **Hard-kill criterion.** N/A in revised scope (no LSQ).
7. **Inline fixes that still apply:** instance-level Hyperparameter access for new flags (existing convention), TTT × DyT compatibility check, sliding-window compile warmup.
8. **Three regression tests:** L0 (`EVAL_STRIDE=0` byte-identical to baseline), L1' (`USE_POLAR_EXPRESS=0 + USE_DYT_NORM=0` byte-identical to baseline), L1' (TTT path unaffected when extended-TTT flags are at default).

## 1. Problem

The `openai/parameter-golf` leaderboard top moved to **1.0810 BPB** (`bigbag`: SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT). Our trusted H100 baseline is **1.2102 BPB** (`train_gpt.py` @ commit `3e34098`, clean 11L full-shard repro). **Gap: 0.13 BPB.**

Information-theoretic floor at 16MB / 8xH100 / 10min is ~**1.00–1.05 BPB** (NNCP-class neural compression on Wikipedia-grade text; FineWeb is noisier so the practical floor sits near 1.00). Top-3 are ~0.08 above floor; we are ~0.21 above floor.

We are **not capacity-bound**. A 16MB int8 model has ~16M effective params; at this regime ~1.05 BPB is achievable. The 0.13 gap is dominated by eval-time techniques and parameter-utilization quality, not by parameter count.

### Goal

Land BPB in the **1.05–1.10 range** through stacked original techniques. No copies of leaderboard entries. Inspiration only.

### Non-goals

- Porting "score-first TTT", "depth recurrence", or SP8192 directly (those are the leaderboard's stack — uninteresting if we just clone)
- Beating 1.0810 on the first try (multi-iteration sprint; first integration target is 1.10)
- Mac iteration (Apple Silicon is exhausted per `.lab/insights.md`)

## 2. Architecture

Three orthogonal layers, each with a kill switch. Sequencing is from cheapest-strict-win to highest-variance-bet.

```
Layer 0 (Pre-flight, ~4h)        — Sliding-window stride-64 eval on H100
Layer 1 (Capacity, ~2 days)      — LSQ int4 QAT + per-tensor sensitivity allocation
Layer 2 (Training quality, ~2d)  — DiffAttn + DyT + Polar Express NS + QK-Norm
                                   (stacked with our existing freq-decomposed skip gating)
```

The layers are designed to compose: each touches a different part of the system.

| Layer | Touches | Doesn't touch |
|---|---|---|
| L0 | Inference path only | Training, model weights, optimizer |
| L1 | Quant op (STE), serializer, optimizer (LSQ scales) | Block internals, eval path |
| L2 | Block internals (attn, norm, NS coefficients) | Quant op, eval path |

### Sequencing logic

L0 first because it's a free measurement re-baseline — every later experiment needs the new eval to compare correctly.

L1 second because it sets the *parameter budget*. If int4 holds, the optimal L2 hyperparameters (depth, dim, MLP mult) shift since we're effectively at 32M params instead of 16M. We want L2 to be designed against the post-L1 envelope.

L2 last, layered onto whichever quantization regime survived L1 (int4 if successful, int8 if collapsed).

### Kill criteria (each gates the next)

- **L0:** sliding-window eval pass time ≤ 8 min within the 10-min eval budget. If overruns, fall back to stride 128.
- **L1:** post-quant gap ≤ 0.030 BPB → proceed to L2 with current allocation. Gap 0.030–0.050 BPB → re-run with stricter sensitivity threshold (more tensors at int8). Gap > 0.050 BPB → revert all int4 to int8, proceed to L2 directly.
- **L2:** each architectural component held to a clean A/B against post-L1 baseline; any −0.005 BPB regression treated as failure → drop that component, keep the rest. (Per competition rules a strict significance test requires p<0.01 across multiple runs; for internal kill decisions we treat single-run ≥0.005 deltas as real.)

### Budget estimate

Happy-path: ~6 RunPod H100 runs (10–15 min wallclock + setup each). Fallback paths add up to 2 more runs (e.g. int4 collapse retry, sliding-window stride drop).

1. L0 baseline re-run (sliding window enabled, otherwise unchanged)
2. L1 sensitivity sweep (200-step probe runs per tensor — bundled into one launch script)
3. L1 main training run (LSQ int4 + mixed precision)
4. L2 ablation A: DiffAttn + DyT only
5. L2 ablation B: + Polar Express NS + QK-Norm
6. Final composite + verification

### Expected impact per layer (informed estimate, not commitment)

| Layer | Lower bound | Upper bound | Source of estimate |
|---|---|---|---|
| L0 sliding-window | 0.03 BPB | 0.10 BPB | MLX measurement (0.03 on Mac) + Agent D floor analysis (0.05–0.10 on web text) |
| L1 LSQ int4 + mixed | 0.00 BPB | 0.10 BPB | Agent B's compression survey: 0 if int4 collapses, 0.05–0.10 if effective param budget doubles |
| L2 DiffAttn + DyT + PolarExp + QKN | 0.03 BPB | 0.07 BPB | Agent A's architecture survey, sum of single-component estimates with 30% interaction discount |

**Combined ceiling (if all three land independently):** ~0.13 BPB → ~1.08 BPB. **Floor (only L0 lands):** ~1.18 BPB. Most likely landing zone: **1.10 ± 0.04 BPB.**

## 3. Code structure

**Decision:** snapshot current `train_gpt.py` → `train_gpt_baseline.py` (frozen reference, holds the 1.2102 trusted result). Continue development on `train_gpt.py`.

| File | Role | Mutability |
|---|---|---|
| `train_gpt_baseline.py` | Frozen H100 reference, last validated result 1.2102 | READ-ONLY |
| `train_gpt.py` | Active H100 development for this sprint | Edit |
| `train_gpt_mlx.py` | Apple Silicon dev (separate track, exhausted for now) | Edit |
| `train_gpt_h100.py` | Older H100 path before the cleanup | DEPRECATED |

CLAUDE.md to be updated to reflect the new READ-ONLY assignment. The "best practice coding style" of the current `train_gpt.py` is preserved by the snapshot.

## 4. Components

### 4.1 — L0: Sliding-window stride-64 eval

**Env var:** `EVAL_STRIDE` (default 0 = chunked; 64 = sliding).

**New function:** `eval_val_sliding(model, val_iter, args)` ported from `train_gpt_mlx.py`'s `eval_val_sliding`.

**Mechanism:** for each batch, slide window by `stride` tokens; score only the last `stride` tokens of each window. Each scored token sees up to `seq_len - stride` = 960 tokens of context.

**Memory:** identical to chunked eval (one batch at a time).

**Compute:** ~16× more forward passes (stride 64 over seq_len 1024).

**Integration:**
- Main eval flow gates on `args.eval_stride > 0`
- Both the `final_int8_zlib_roundtrip` eval and `eval_val_ttt_lora` use sliding window when enabled
- Reported metrics gain a `_sw64` suffix to distinguish from old chunked numbers

**Kill criterion:** eval pass time > 8 min on 8xH100 → fall back to stride 128 (still 7× context vs chunked).

### 4.2 — L1: LSQ int4 QAT + sensitivity allocation

#### 4.2.1 — `LSQQuantizer(bits, per_channel=True)` class

- Learnable scalar `s` (per output channel for matrices, per tensor for embeds)
- `s` registered as a parameter; trained by the Adam optimizer (matrix-side optimizer is Muon, embed-side is Adam, and LSQ scales go with Adam regardless of which weight they wrap)
- Forward: `y = s * round(clip(x/s, q_min, q_max))` with STE
- Backward through `s`: `dL/ds = sum(grad * (sign(round(x/s)*s - x) + clip indicator * x/s))` (LSQ paper formula)
- Initialization: `s_init = max(|W|) / 2^(b-1)` then unfrozen at step 200

#### 4.2.2 — `quantize_lsq(W, s, bits)`

Pure functional helper used in forward pass; no state. Activated **from step 1**, not post-training. This is the critical anti-collapse fix per the 2026 Pythia-160M paper "When Flat Minima Fail."

#### 4.2.3 — Sensitivity sweep (`analyze_quant_sensitivity.py`)

For each weight tensor in the model:
1. Train a 200-step probe run with that tensor at int4 (and all others at int8)
2. Record final pre-quant BPB and post-quant BPB; the post-pre gap is the per-tensor sensitivity score
3. Output: JSON file mapping tensor name → sensitivity score

Estimated cost: ~24 weight tensors × 200 steps × 0.075s/step × 8 H100s ÷ 8 = ~6 min total when batched into one launch script with sequential probes.

#### 4.2.4 — Bitwidth allocator

Given sensitivity scores + target artifact size:
- Sort tensors by sensitivity (most-sensitive first)
- Greedy allocation: assign int8 to most-sensitive tensors until artifact would exceed target, then int4 for the rest
- Embeddings default to int8 (most sensitive in our prior experience: per `.lab/insights.md`, FP16 tok_emb wins)
- Output: `bitwidth_manifest.json` consumed by both training and serializer

#### 4.2.5 — Serializer

- Writes `bitwidth_manifest.json` + per-tensor scales (fp16) + packed integer codes (4-bit packs use 2 weights/byte; 8-bit is 1 weight/byte)
- Whole blob compressed with zstd-22
- Manifest size negligible (~1KB)
- Round-trip on eval: parse manifest → unpack codes → multiply by scales → run model

### 4.3 — L2: Architectural stack

#### 4.3.1 — DiffAttn

Replace `Attention.forward`:

```
Q1, Q2 = project queries (each MODEL_DIM/2 per head, 4 paired query heads)
K, V   = project shared (4 KV heads, our existing GQA unchanged)
A1     = softmax(Q1 @ K^T / sqrt(d_head))
A2     = softmax(Q2 @ K^T / sqrt(d_head))
λ      = exp(λ_q1 · λ_k1) - exp(λ_q2 · λ_k2) + λ_init
λ_init = 0.8 - 0.6 * exp(-0.3 * layer_idx)  # per-layer constant
out    = (A1 - λ * A2) @ V
```

- Halves the effective number of Q heads to keep params equal to the 8Q baseline
- λ_q1, λ_k1, λ_q2, λ_k2 are learnable scalars per layer
- Combines with our existing freq-decomposed skip gating (different code path; no conflict)

#### 4.3.2 — DyT (Dynamic Tanh)

Drop-in replacement for RMSNorm:

```
DyT(x) = γ ⊙ tanh(α · x) + β
```

- α: learnable scalar per norm instance (init 0.5, paper default)
- γ, β: per-channel learnable (γ init 1, β init 0)
- 23 instances at 11L (pre-attn × 11 + pre-MLP × 11 + final = 23)
- Saves the mean/var reduction; tanh saturation provides natural activation clipping helpful under LSQ quantization

#### 4.3.3 — Polar Express NS coefficients

Swap the 5-iteration Newton-Schulz coefficient table in `zeropower_via_newtonschulz5`:

- Same code shape, same number of matmuls, same bf16 stability behavior
- Coefficients are Chebyshev-style minimax-optimal (from Dao-AILab/`gram-newton-schulz`)
- Pure constant change; falls back to baseline coefficients with one env var (`POLAR_EXPRESS=0`)

#### 4.3.4 — QK-Norm

l2-normalize Q and K along the head dim before the dot product:

```
Q_n = Q / (||Q||_head_dim + ε)
K_n = K / (||K||_head_dim + ε)
attn_logits = (Q_n · K_n^T) * γ  # γ learnable per head, init log(sqrt(d_head))
```

- Replaces the `1/√d_k` scale with a learnable per-head γ
- Makes attention scale-invariant to logit magnitudes; prevents drift at deeper layers
- Compatible with DiffAttn (apply QK-Norm to both Q1/K and Q2/K branches)
- 8 learnable scalars per layer × 11 layers = 88 extra parameters (negligible)

## 5. Data flow

### Training step (all layers active, post-warmup-50)

1. Batch arrives (524K tokens, 8 GPUs × 65K tokens each)
2. Embedding lookup: `tok_emb` quantized via LSQ int8 (most sensitive — kept at higher precision)
3. For each of 11 transformer blocks:
   - DyT(α₁) → DiffAttn (Q1, Q2 LSQ-quantized; QK-Norm; freq-decomposed skip gating active on residual stream) → +residual
   - DyT(α₂) → MLP (W_up LSQ, W_down LSQ, LeakyReLU(0.5)²) → +residual
4. DyT(final) → tied output projection (LSQ int8 same as input embedding)
5. Cross-entropy loss vs target tokens
6. Backward: gradients flow through all STE LSQ ops, through DyT (smooth tanh derivative), through DiffAttn
7. Optimizer step:
   - Muon (with Polar Express NS coefficients) on 2D matrix params
   - Adam on embeds, LSQ scales `s`, DyT params (α, γ, β), DiffAttn λ scalars, QK-Norm γ
8. Pre-warmdown QAT regularizer continues to apply (lr_mul ≥ 0.8, every 10 steps) — kept from current best

### Eval step

1. Load saved artifact: parse `bitwidth_manifest.json` + scales + packed integer codes
2. Dequantize per tensor: integer codes × scales → fp16 weights
3. Run model with **sliding window stride 64** through validation tokens
4. Compute total NLL → BPB
5. Optional TTT path: instantiate `BatchedTTTLoRA`, do legal LoRA adaptation on already-graded tokens, re-eval with sliding window
6. Report `final_int8_zlib_roundtrip_sw64` and `final_int8_ttt_lora_sw64`

## 6. Error handling and kill criteria (full list)

| Condition | Action |
|---|---|
| L0 sliding window eval > 8 min on 8xH100 | Fall back to stride 128 (still 7× context vs chunked) |
| L0 BPB delta < +0.02 vs current 1.2102 | Document and proceed; small win still useful |
| L1 quant gap > 0.05 BPB at end of training | Revert all int4 tensors to int8, proceed to L2 directly |
| L1 quant gap 0.03–0.05 BPB | Re-run with stricter sensitivity threshold (more int8) |
| L1 training diverges (loss > 5 by step 500) | Abort, halve LSQ scale init, retry once; if still diverges, kill |
| L2 DiffAttn alone regresses > 0.005 BPB | Drop DiffAttn; keep DyT + Polar Express + QK-Norm |
| L2 DyT alone regresses > 0.005 BPB | Drop DyT; keep RMSNorm |
| L2 Polar Express alone regresses > 0.005 BPB | Drop; keep baseline NS5 |
| L2 QK-Norm alone regresses > 0.005 BPB | Drop; keep baseline scaling |
| Composite (all L2) regresses > 0.01 BPB | Roll back to single-component winner |

Each L2 component is held to its own A/B against the post-L1 baseline. Components do not get to "carry" each other.

## 7. Testing

### Unit-level (pre-launch)

- `LSQQuantizer.forward` STE forward+backward gradient correctness vs pytorch autograd numerical check
- `eval_val_sliding` matches `eval_val` when `stride == seq_len` (degenerate case)
- `bitwidth_allocator` sums to expected artifact size on synthetic sensitivity inputs
- DiffAttn forward matches reference (Microsoft DiffAttn release) on a 2-layer toy model
- DyT replaces RMSNorm without shape mismatch
- Polar Express NS converges on random matrices to same Stiefel manifold as baseline NS5 (within 1e-4)
- QK-Norm forward matches manual l2-norm + dot product

### Integration-level (smoke runs)

- 200-step smoke on 8xH100 with each layer enabled in isolation
- Verify pre-quant BPB at step 200 is within 0.05 of `train_gpt_baseline.py` result
- Verify artifact size after each layer matches expectation (L1 should drop ~30% on int4 portions)

### Full-run validation

- Each H100 run logs: `pre_quant_bpb`, `post_quant_bpb_chunked`, `post_quant_bpb_sw64`, `ttt_bpb_sw64`, `artifact_bytes`, `step_time_ms`, `total_steps`
- Append to `.lab/results.tsv` with run ID and component flags
- Append narrative to `EXPERIMENT_LOG.md`
- Update `.lab/insights.md` for any keep/discard decisions

### Significance gate

- Per competition rules: 0.005 BPB improvement with p<0.01 is the minimum significance
- For internal kill criteria, treat single-run differences ≥0.005 as real (we cannot afford full p<0.01 multi-run testing on H100 budget)
- Re-run any composite that lands within 0.005 of `train_gpt_baseline.py` at least once for confirmation

## 8. What we're explicitly NOT doing this sprint

- Tokenizer change to SP8192 (leaderboard top has it, but it's structural; would invalidate all our prior insights and require re-tokenizing data)
- Depth recurrence / parameter-tying across layers (leaderboard core technique)
- PaLM-style parallel residuals (leaderboard core technique)
- Score-first TTT (leaderboard core technique)
- 13L+ depth experiments (killed in prior cycle; LayerScale could re-open this but is out of scope here)
- Mamba/SSM/RWKV hybrids (~2 day implementations, low confidence at our scale)
- AQLM/QuIP# vector quantization (built for outlier-heavy 7B+ models)
- 2:4 sparsity (capacity-cutting at our scale)
- Online float-teacher distillation (Option C from brainstorm — saved as fallback if this sprint stalls)

## 9. Original differentiators we keep from prior work

| Technique | Source | Status |
|---|---|---|
| Warmdown-aware WD scheduling | exp_014, ours | Keep |
| Frequency-decomposed skip gating (W=32) | exp_017, ours | Keep |
| Asymmetric MLP (encoder 2x, decoder 4x) | exp_041, ours | Keep |
| Per-layer LR scaling (LAYER_LR_SCALE=0.5) | exp_039, ours | Keep |
| Pre-warmdown QAT regularizer | exp_051, ours | Keep |
| LeakyReLU(0.5)² activation | exp_063, ours | Keep |

The full original-stack-plus-this-sprint becomes our unique fingerprint: nobody else has this combination.

## 10. Success criteria for the sprint

- **Hard target:** post-quant BPB ≤ 1.10 with sliding window stride-64 eval
- **Stretch target:** ≤ 1.08 (top-5 territory)
- **Time bound:** 5 calendar days, ~6 H100 runs
- **Originality bound:** zero leaderboard techniques copied; "inspiration only" rule honored

If we land ≥ 1.10 but < 1.12, document everything and pivot to Option C (online float-teacher distillation).

## 11. Open questions / risks

- **Sliding window cost.** The 8-min budget estimate is from the MLX path (50 min on Apple Silicon, scaled by H100/Mac throughput ratio). Could be faster or slower in practice. First L0 run is the empirical check.
- **LSQ scale init drift.** The `s_init = max(|W|)/2^(b-1)` heuristic assumes mature weight distributions. At step 1, weights are near initialization — `s` may need a warmup. Backup plan: hold `s` frozen for the first 200 steps.
- **DiffAttn × GQA interaction.** The original DiffAttn paper does not test with GQA (it tests at full multi-head attention). Splitting Q into Q1/Q2 with shared KV may have unexpected gradient dynamics. Smoke run will catch divergence early.
- **DyT + LSQ interaction.** DyT's tanh saturates at high `α`; under LSQ this can cause dead-channel patterns where most of the channel sits in tanh's flat region. Watch the α distribution during the smoke run.
- **Compounding integration risk.** All four L2 components share the residual stream and gradient flow. Single-component A/Bs help, but the composite could still surprise us.

## 12. Path to writing-plans

After this design is approved, transition to the `superpowers:writing-plans` skill to produce a step-by-step implementation plan covering:

- File creation (snapshot, new test files, sensitivity sweep script)
- Component implementation order
- Smoke test gates
- H100 launch scripts per layer
- Result-recording protocol
- Memory updates (`.lab/insights.md`, `.lab/results.tsv`, `EXPERIMENT_LOG.md`)
