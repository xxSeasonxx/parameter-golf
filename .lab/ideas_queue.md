# Ideas Queue

Prioritized by expected impact. Organized by research direction, not just parameter sweeps.

**Originality key**: Each idea is tagged:
- **ORIGINAL** — Novel technique we invented, not found in competition or literature
- **OUR TWIST** — Known technique applied in a novel way specific to our setup
- **KNOWN** — Established technique we haven't tried yet (porting, not inventing)
- **SWEEP** — Pure parameter/env-var exploration

**Guiding principles** (updated after exp 010-015):
1. **Weight entropy reduction is the master lever** — WD proved that reducing weight entropy improves BOTH BPB and compressibility simultaneously. Any technique with this dual benefit is high priority.
2. **Late-phase training changes have outsized impact** — The "worse early, better late" crossover pattern means the warmdown phase determines final quality. Target ideas at the last 25% of training.
3. **On Apple Silicon, steps/second is the binding constraint** — Not artifact size (6.1MB headroom). Ideas that don't add per-step time are "free." Capacity increases that slow steps are penalized.
4. **WD already handles compression** — Artifact went from 15.7MB to 9.9MB. Explicit compression techniques have diminishing marginal returns.

## Completed (lab/mar26b)

- ~~10-layer architecture~~ [KNOWN] — DONE: -0.05 BPB
- ~~Muon weight decay=0.02~~ [KNOWN] — DONE: -0.031 BPB, -1.1MB artifact
- ~~Muon weight decay=0.05, 0.10~~ [SWEEP] — DONE: -0.099 BPB total, 11.0MB artifact
- ~~Warmdown-aware WD scheduling~~ [**ORIGINAL**] — DONE: -0.110 BPB total, 9.9MB artifact. `wd = base_wd * (2 - lr_mul)`. Nobody in competition or literature uses WD that co-varies with LR schedule.
- ~~FP16 tied embeddings~~ [KNOWN] — DONE: quant gap +0.0001
- ~~Sliding window eval~~ [KNOWN] — DONE: implemented, EVAL_STRIDE=64
- ~~Warmdown tuning~~ [SWEEP] — TESTED: warmdown=400 failed. Keep 1200.
- ~~11 layers~~ [SWEEP] — TESTED: worse on Apple Silicon (slower steps), better per-step. Try on 8xH100.
- ~~Frequency-decomposed skip gating~~ [**ORIGINAL**] — DONE: -0.010 BPB. Lo/hi band gates with W=32 on U-Net skip connections.
- ~~Muon momentum warmdown ramp~~ [OUR TWIST] — FAILED: +0.035 BPB regression. Killed.

---

## Tier 1: High Priority (aligned with all principles)

### 1. Frequency-Decomposed Skip Gating [**ORIGINAL**]
**Hypothesis**: U-Net skip connections currently use flat per-dim weights. Decompose the skip signal into low-frequency (local mean over a window) and high-frequency (residual) bands, with independent learned gates for each. Different layers should pass through different frequency content.
**Why original**: Standard U-Net skips use a single learned scale per dimension. Nobody decomposes skip signals by frequency band. This is signal-processing-inspired neural architecture design.
**Why high priority**: P2 — skips feed decoder (late) layers, so improving skip quality targets the critical phase. P3 — adds only ~10 params per skip connection (negligible cost). Architectural novelty could yield a step-change, not incremental gain.
**Expected impact**: -0.005 to -0.02 BPB.
**Effort**: Medium. Modify GPT forward pass to decompose skip signals.

### 2. ~~Warmdown-Phase Weight Averaging~~ [**ORIGINAL**] — KILLED (exp_018)
**Hypothesis**: Maintain an exponential moving average (EMA) of weights throughout training. During warmdown, gradually interpolate current weights toward EMA: `w = (1-α)*w + α*EMA` where α increases from 0 to 0.5. EMA weights have lower variance → more compressible, and averaging finds flatter minima → better generalization.
**Why original**: SWA (Izmailov 2018) averages at the end of training with a cyclical LR. Our approach is different: continuous EMA blending that activates specifically during the Muon warmdown phase, co-designed with our warmdown-aware WD scheduling. The combination of EMA + WD scheduling + Muon is novel.
**Why aligned**: P1 (lower weight variance = lower entropy = dual benefit), P2 (targets warmdown specifically), P3 (EMA update is cheap — one multiply-add per param per step).
**Expected impact**: -0.005 to -0.02 BPB + -0.5MB artifact.
**Effort**: Medium. ~30 lines: maintain EMA dict, blend during warmdown.

### 3. Dual-Phase Optimizer Config [**OUR TWIST**]
**Hypothesis**: Phase 1 (0-75% of wallclock): standard LR, lower WD=0.05 for exploration. Phase 2 (75-100%): lower LR, higher WD=0.15 for exploitation/regularization. Our data shows late-phase regularization is what matters (P2).
**What's known**: Phase-based training (cyclical LR, warm restarts) is established. Two-phase with explicit WD transitions exists in some contexts.
**What's our twist**: Specific combination with Muon optimizer + the insight that WD matters most in late phase (derived from our warmdown-aware scheduling experiments). The phase boundary is wallclock-aware, not step-aware.
**Expected impact**: -0.005 to -0.02 BPB.
**Effort**: Low. ~10-line change to lr_mul() and Muon.step().

### 4. ~~Muon Momentum Warmdown Ramp~~ [**OUR TWIST**] — KILLED (exp_016)
**Hypothesis**: Instead of constant momentum=0.95, ramp momentum from 0.95 to 0.99 during warmdown. Higher momentum in late phase = smoother convergence into flatter minimum, combining with our warmdown-aware WD for a coordinated late-phase optimization strategy.
**What's known**: 1cycle policy (Smith 2018) varies momentum inversely with LR. Super-convergence uses momentum scheduling.
**What's our twist**: Applied specifically to Muon's Newton-Schulz orthogonalization (not standard SGD), coordinated with our warmdown-aware WD schedule. The compound effect of {increasing WD + increasing momentum + decreasing LR} during warmdown is unexplored.
**Expected impact**: -0.003 to -0.01 BPB.
**Effort**: Low. 3-line change in Muon.step().

---

## Tier 2: Medium Priority (aligned with 2-3 principles)

### 5. TTT LoRA (Test-Time Training) [KNOWN]
**Hypothesis**: Per-document LoRA adaptation during evaluation. Already implemented in the codebase (`TTT_LORA_RANK`, `TTT_LORA_LR`, etc.) but never tested.
**What's known**: TTT with LoRA is an established technique in this competition. Several top submissions use it.
**Why still valuable**: Free eval-time compute (P3). Even known techniques matter if they yield BPB gains.
**Effort**: None — just set env vars.
**Risk**: May be slow on Apple Silicon. Test timing first.

### 6. Asymmetric MLP Capacity [KNOWN]
**Hypothesis**: MLP_MULT=1.5 for layers 0-4, MLP_MULT=3 for layers 5-9. Same total params → same step time (P3). Better allocation of capacity to later layers.
**What's known**: Non-uniform layer widths appear in several architectures (e.g., EfficientNet scaling, some transformer variants). The specific application to this U-Net transformer is not common but not novel.
**Effort**: Medium. Modify Block init to accept per-layer MLP width.

### 7. Attention Head Diversity via Stochastic Masking [KNOWN]
**Hypothesis**: During training, randomly zero out entire attention heads (p=0.1). Forces diverse, non-redundant head patterns. At inference, all heads active.
**What's known**: DropHead (Zhou et al. 2020) does exactly this. It's a known regularization technique.
**Why still valuable**: P1 — regularization works (proven by WD). P3 — zero extra per-step cost. Low effort.
**Expected impact**: -0.005 to -0.015 BPB.
**Effort**: Low. ~10 lines in CausalSelfAttention.

### 8. ~~Larger Batch Tokens~~ [SWEEP]
DONE: batch=16384 (-0.082 BPB!), batch=24576 (-0.007 more). batch=24k is now the default. Marginal returns diminishing — try batch=32768 as a final test.

---

## Tier 3: Quick Tests (env var only, < 5 min each) [all SWEEP]

### 9. RoPE Base Tuning (1000)
~~50000~~: exp_020 +0.021 BPB regression. Try 1000 (shorter context bias for seq_len=1024).

### 10. ~~QK Gain Init Tuning (1.0, 2.0)~~
exp_021: QK_GAIN=1.0 was +0.014 BPB regression. Default 1.5 is good.

### 11. ~~WD=0.20~~
exp_019: Over-regularized +0.029 BPB. WD=0.10 is the sweet spot.

---

## Killed Ideas (Evidence Against)

- **Warmdown < 1200**: Warmdown=400 was strictly worse. Long warmdown is beneficial.
- **INT8_KEEP_FLOAT_MAX_NUMEL increase**: Accidentally keeps all tensors as FP16. Use name patterns.
- **MoE at 16MB**: Research shows unviable below 500M params.
- **Label smoothing**: Failed in systematic competition testing.
- **Full-training QAT**: Late QAT (final 15%) is strictly better per competition data.
- **Concurrent runs on Apple Silicon**: 2-3x throughput degradation.
- **Compression-Aware Training (CAT)**: WD=0.10+sched already reduced artifact from 15.7→9.9MB. Explicit compression penalty adds complexity for marginal gain. (Killed by P4)
- **Self-Compressing Orthogonal Init**: Init only affects early training. With 1700 steps, model overwrites init quickly. Late-phase matters more (P2).
- **Cyclic Embedding Perturbation**: Perturbation decays to zero during warmdown — so it's absent during the phase that matters most (P2).
- **Progressive Layer Growing**: High effort, complex implementation. 11L experiment showed extra layers hurt on Apple Silicon anyway (P3).
- **Entropy-Guided Dynamic Precision**: With 6.1MB headroom, saving artifact space is no longer urgent (P4). Could revisit if we need to fit more capacity.
- **Muon Momentum Warmdown Ramp**: exp_016 showed +0.035 BPB regression. High momentum destabilizes Newton-Schulz orthogonalization during warmdown. Don't modify Muon momentum late in training.
- **EMA Warmdown Blending**: exp_018 catastrophic — loss increased during warmdown (2.19→2.41). EMA weights too stale, blending destroys convergence. Don't interpolate toward averaged weights during training.
- **ROPE_BASE=50000**: exp_020 +0.021 BPB regression. Default 10000 is already good.
- **QK_GAIN_INIT=1.0**: exp_021 +0.014 BPB regression. Default 1.5 is well-calibrated.
- **Shorter warmdown at large batch**: exp_023 warmdown=600 with batch=16k was +0.029 worse than warmdown=1200. Long warmdown is ALWAYS beneficial.

---

## Promising Combinations

- **Freq skip gating + SWA warmdown**: Better skip information flow + smoother final weights — architectural + optimization synergy
- **Dual-phase + momentum ramp**: Coordinated late-phase strategy: higher WD + higher momentum + lower LR all during warmdown
- **Asymmetric MLP + asymmetric WD**: Stronger WD on early layers (they need less capacity), weaker on late layers
- **TTT LoRA + sliding window**: Both are eval-time improvements. Stack them on 8xH100 for maximum eval-time gain
