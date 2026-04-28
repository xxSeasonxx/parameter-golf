# TODOS — Deferred Research Bets

Captured during the H100 sprint plan-eng-review on 2026-04-28. These are ideas that survived the eng review and outside-voice (independent Claude subagent) challenge but were deferred so the sprint could test cheap drop-in changes first. Re-evaluate after the sprint's decision gate (see `docs/superpowers/plans/2026-04-28-stacked-h100-sprint.md` Task 22).

Ordered roughly by expected value-per-cost.

---

## 1. LSQ-from-step-1 int4 QAT (with proper sensitivity sweep)

**What:** Wrap weight tensors in a learnable-step-size quantizer (LSQ, Esser et al. ICLR 2020) with STE during training, applied from step 1 (not post-training). Allocate per-tensor bitwidth (int4 vs int8) via a sensitivity sweep that uses **proper LSQ-aware short probes**, not post-hoc PTQ on a non-LSQ-trained checkpoint.

**Why:** If int4 holds at our scale, our effective parameter budget doubles from ~16M to ~32M at the same artifact size. Could close 0.05–0.10 BPB.

**Pros:**
- Largest single-component upside in the original sprint plan
- Falls back cleanly to int8 if int4 collapses
- Compresses well with our existing zstd-22 path
- Original differentiator (no top-5 leaderboard entry uses LSQ)

**Cons:**
- 2026 Pythia-160M paper "When Flat Minima Fail" shows naive int4 collapses (gap 11% → 517%) without LSQ-from-step-1. We assume LSQ closes this at our regime, but no published evidence at <50M params on byte-level perplexity.
- Cost: at least 2 H100 runs (sensitivity probes + main run, both fully LSQ-aware). With proper probes, sensitivity sweep adds ~30 min H100.
- Many code-level interactions: DDP gradient sync for LSQ scales (must eager-init before DDP wrap to avoid deadlock), serializer round-trip, pre-warmdown QAT × LSQ collision (must disable existing QAT regularizer when LSQ is on).

**Context:** The original sprint plan included LSQ as Phase L1 with a post-hoc sensitivity sweep on the L0 checkpoint. The eng review's outside voice flagged the post-hoc design as fundamentally flawed (would burn one H100 run on a manifest derived from PTQ-int4 sensitivity, not LSQ-int4 sensitivity, which have different rankings). The fix: each per-tensor probe must do a short (~3 min) LSQ training run with that tensor at int4. Total sweep cost: ~72 min × 1 GPU = ~9 min on 8xH100 if we batch the probes. Better: skip the sweep entirely, use a flat policy `tok_emb=int8, all-others=int4`, and let the H100 run tell us if int4 holds.

**Depends on:** L1' cheap-winners run. If cheap winners hit BPB ≤ 1.13, LSQ may not be worth the research cost. If we land 1.17–1.20, LSQ is the next bet.

**Implementation reference:** Original plan content at `docs/superpowers/plans/2026-04-28-stacked-h100-sprint.md` git history (pre-revision) Tasks 8-15. Rewrite needed for the corrected sensitivity sweep design.

---

## 2. Gated Output Attention (NeurIPS 2025 best paper)

**What:** Apply input-dependent sigmoid gate to attention output before output projection: `out = sigmoid(W_g · x) ⊙ Attn(x) → proj(out)`. Reference: Qiu et al., NeurIPS 2025 ("Gated Attention for LLMs"), deployed in Qwen3-Next.

**Why:** Eliminates attention sinks without sink tokens. Validated at 1.7B+ dense and 15B MoE. Compatible with our existing GQA (unlike DiffAttn, which we deferred because it requires independent K). Likely 0.02–0.04 BPB at our scale.

**Pros:**
- GQA-compatible (DiffAttn isn't, that's why we cut it)
- Few new params (one gate matrix per attention block)
- Stabilizes higher LR — pairs well with QK-Norm v2 (see #3)
- NeurIPS 2025 best paper

**Cons:**
- Adds one matmul per attention block (~3% step time)
- Requires shrinking `MODEL_DIM` slightly (~8%) to stay in artifact budget after int8

**Context:** Considered as an alternative to DiffAttn during the eng review's research phase. Plan deferred all attention-block changes to keep cheap-winners scope tight.

**Depends on:** L1' cheap-winners outcome. Strong candidate for a follow-up sprint if L1' lands marginal.

**Implementation reference:** [arxiv 2505.06708](https://arxiv.org/pdf/2505.06708)

---

## 3. QK-Norm v2 refactor (l2-norm + log gain init + scale=1.0)

**What:** Refactor the existing `F.rms_norm(q,...) * q_gain` path inside `CausalSelfAttention` to use explicit l2-norm, initialize per-head gain to `log(sqrt(head_dim))`, and pass `scale=1.0` to `F.scaled_dot_product_attention` (let the gain absorb both the QK normalization scale and the `1/√d_k` factor).

**Why:** Current code has redundant scaling (RMS-normalized Q × per-head gain × default 1/√d_k inside SDPA). Modded-nanoGPT speedrun community treats this as a "free win" that lets you push LR higher safely. Likely 0.005–0.01 BPB.

**Pros:**
- Pure refactor, no new params
- Single source of attention scale (just one gain per head)
- Pairs with Gated Output Attention (#2) — both stabilize attention dynamics
- Validated in modded-nanoGPT speedrun at 124M

**Cons:**
- Refactor risk: must verify gain-init reproduces baseline behavior exactly (baseline is RMS norm + 1/√d_k + gain=1.5; new is l2 norm + log(√d_k) ≈ 2.08 + scale=1.0)
- Marginal expected gain (0.005–0.01) — only worth it as part of a follow-up sprint with other tightenings

**Context:** Was originally L2 component #4 in the sprint plan. Deferred because cheap winners (Polar Express + DyT + extended TTT) do not need this.

**Depends on:** Could ship anytime as a small refactor. Best paired with Gated Output Attention (#2) for compounding stabilization.

---

## 4. LayerScale to re-open 13L+ depth

**What:** Add per-channel learnable residual gate `γ_l` initialized small (~0.1) to each residual branch: `x = x + γ_l ⊙ Block(x)`. CaiT-style (Touvron et al. 2021).

**Why:** 13L was killed in our prior cycle (`H100 capacity_13l_no_ema` regressed vs 11L). LayerScale was specifically designed to make deep networks trainable. Could unlock 12L or 13L that previously failed.

**Pros:**
- Minimal new params (`D` per layer = 512 × 11 layers = 5.6K)
- Stable deepening mechanism, well-validated
- Combines naturally with our existing residual + U-Net skip stream
- Could re-enable 13L which has more capacity for complex patterns

**Cons:**
- Depth-only test: runs at 12L need a parallel control to attribute (LayerScale alone? Or LayerScale × depth?)
- 13L step time was 85.92 ms vs 11L's 74.95 ms in our prior run — fewer total steps in the 10-min budget
- Could compound interestingly with DyT (DyT also helps deep training stability)

**Context:** Considered in research phase, deferred because depth experiments are higher-variance than the cheap winners.

**Depends on:** L1' outcome. Specifically, if L1' shows DyT helps (which would suggest depth-related stability is a lever), LayerScale becomes a natural next step.

**Implementation reference:** [arxiv 2103.17239](https://arxiv.org/abs/2103.17239)

---

## 5. Online float-teacher distillation (Option C from brainstorm)

**What:** Train a 60M-parameter float teacher in the first 4 minutes of the 10-min budget on 50% of FineWeb shards. Cache teacher logits at every 200 steps. Student gets standard 6000-step recipe but with KD: `loss = 0.7 * CE(student, label) + 0.3 * KL(student/T, teacher/T)`, T=2.0, only on tokens where teacher confidence > 0.4.

**Why:** ACL 2025 "Pre-training Distillation for LLMs" reports lower validation loss vs no-KD at Pythia/OLMo scale. KD especially powerful when student is heavily quantized — teacher's float logits give a smoother target than one-hot CE.

**Pros:**
- Original budget-allocation framing (no top-5 leaderboard entry co-trains a teacher)
- Falls back to plain CE if KD weight = 0 (no recipe corruption)
- Compounds with int8 student quality

**Cons:**
- Largest implementation surface (teacher training, logit caching, KD loss path)
- Tight on the 10-min budget (teacher gets 30% of compute, student 70%)
- If teacher is poorly-trained at 4 min, student gets noisy targets

**Context:** Surfaced as Option C in the original brainstorm. Deferred because it's a 2-day implementation and cheap winners are 3-hour implementations.

**Depends on:** L1' outcome. If L1' lands 1.17–1.20, distillation is the next big swing. If L1' lands ≤ 1.15, distillation is the third sprint.

**Implementation reference:** [aclanthology.org/2025.acl-long.181.pdf](https://aclanthology.org/2025.acl-long.181.pdf)

---

## 6. Tokenizer experiment: SP2048 / SP4096 middle ground

**What:** Re-tokenize FineWeb with sp2048 or sp4096 BPE (current is sp1024). Top of leaderboard uses sp8192. Going sp1024 → sp2048 is a single-step move that doesn't invalidate prior insights as severely as jumping to sp8192.

**Why:** With more vocab, fewer tokens per byte, fewer prediction steps. Potentially a different point on the BPB-vs-artifact trade-off curve. Top of leaderboard suggests bigger vocab wins.

**Pros:**
- Single-axis experiment, easy to attribute
- Does not commit to the structural rewrite that sp8192 implies (sp1024 is nearly byte-level at 8 bytes/token, sp8192 is closer to word-level)
- Reusable: re-tokenized data is a one-time cost

**Cons:**
- Embedding table grows (sp2048 = 2× embed params at fixed dim)
- Prior insights are calibrated for sp1024 — many learnings may not transfer
- 1-day data prep + at least one calibration H100 run

**Context:** Mentioned during research and ranked low because the existing sp1024 stack is well-tuned. Reconsider once cheap-winners results are in and we have a clearer picture of which lever (compression vs vocab vs architecture) is gating us.

**Depends on:** L1' outcome and decision gate. Likely a third or fourth sprint.

---

## Disposition rules

- After the H100 sprint completes, re-rank these by remaining gap (BPB delta needed to land top-5 vs. our post-sprint position).
- Each item gets its own focused sprint with a fresh `/plan-eng-review` invocation.
- Items can be combined only if they're orthogonal (e.g., LayerScale + LSQ) and budget allows multiple H100 runs.
- Items that fail in a follow-up sprint move to a "Killed" section here with one-line rationale, similar to `.lab/insights.md`.

## Killed (post-original eng review)

- **DiffAttn × shared-KV in original L2 plan**: independent Claude subagent showed that sharing K (forced by GQA) breaks the noise-cancellation prior structurally. `A1 - λ·A2` becomes near-rescaling, not noise subtraction. Replace with Gated Output Attention (#2 above) when ready.
- **Post-hoc PTQ sensitivity sweep**: would measure PTQ-int4 sensitivity, not LSQ-int4 sensitivity. See LSQ entry (#1) for corrected design.
- **Stacked 1-PR plan**: split decisions to 3 PRs, then redirected to 1 cheap-winners PR. Follow-up bets each get their own.

## Deferred from refactor pass (2026-04-28)

These items are confirmed OURS (not in upstream `openai/parameter-golf` main) but were left in `train_gpt.py` rather than moved to `train_gpt_common.py` because:
- They are PyTorch-only (no MLX equivalents → moving them gives zero deduplication benefit).
- They reference upstream-derived classes (`Muon`, `GPT`) defined locally in `train_gpt.py`, so moving them creates circular-import risk.

**Items left in `train_gpt.py`:**
- `BatchedLinearLoRA`, `BatchedTTTLoRA`, `_build_ttt_optimizer`, `_reset_ttt_optimizer`, `_find_docs`, `_compute_chunk_window`, `_accumulate_bpb`, `eval_val_ttt_lora` — TTT support stack (~250 lines).
- `grow_model` — progressive layer growth helper. Constructs a new GPT, references many `args` fields and the local GPT class.
- `build_optimizers` — Muon + Adam optimizer construction. References local `Muon` class (upstream-derived, must stay in `train_gpt.py`).

**Revisit triggers:** if MLX ever gets a TTT path, or if we extract the GPT factory into common, these become mechanical moves.

**Sliding-window window-list:** `compute_sliding_windows` was sketched but not landed because the two `eval_val_sliding` implementations (PyTorch per-window, MLX batched) use different conventions for `n_tokens` (one passes `val_tokens.numel()`, the other `val_tokens.size - 1`). Reconciling would require code-level changes to one of the eval paths. Module docstring of `compute_bpb_from_sums` records this for the next refactor.
