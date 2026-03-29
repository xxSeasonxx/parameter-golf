# Ideas Queue

Prioritized by expected impact. Each idea is one experiment, one commit.

**Originality key**:
- **ORIGINAL** — Novel technique we invented
- **OUR TWIST** — Known technique applied in a novel way
- **KNOWN** — Established technique we haven't tried yet
- **SWEEP** — Pure parameter/env-var exploration

---

## Next Session Plan: Local Validation for H100

**Goal**: Validate 3 breakthrough techniques on Mac so they're ready for H100 deployment.
**Approach**: Top-down — test each independently, learn, then combine.
**Each experiment = one commit per program.md.**

### Experiment 048: Zstd Compression [KNOWN]
**What**: Replace `zlib.compress(data, 9)` with zstd level 22.
**Code change**: ~10 lines in serialization section of `train_gpt_mlx.py`. Add `import zstandard`, swap compressor, add fallback to zlib if not installed.
**Run**: Best config, same as exp_046.
**Expect**: Artifact drops ~10-15% (13.1MB → ~11-12MB). BPB unchanged.
**Why first**: Zero risk, pure compression win, tells us how much headroom we gain.

### ~~Experiment 049: Int6 Quantization [KNOWN]~~ COMPLETED — DISCARD
**Result**: Artifact 6.8MB (48% reduction, better than expected). But quant gap +0.064 BPB (~20x worse than int8). Pre-quant 1.6332 (normal). Post-int6 1.6975. Mechanism validated but needs QAT. See exp_052.

### Experiment 050: Int6 + Zstd Combined [KNOWN]
**What**: Both int6 and zstd together.
**Run**: Best config + `QUANT_BITS=6` + zstd enabled.
**Expect**: Artifact ~7-8MB. Massive headroom for larger model on H100.

### ~~Experiment 051: Pre-Warmdown QAT (Fix exp_036) [OUR TWIST]~~ COMPLETED — NEW BEST
**Result**: val_bpb=1.6299 (NEW BEST, -0.0010 vs 1.6309). Pre-quant 1.6270 (best ever). QAT acts as regularizer — improvement is from better training, NOT quant gap reduction (gap unchanged at ~0.003). Zero overhead. Fixes exp_036's timing problem. Config: `QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10`.

### ~~Experiment 052: Pre-Warmdown QAT + Int6 [OUR TWIST]~~ COMPLETED — DISCARD
**Result**: Post-int6 val_bpb=1.6894 (+0.061 quant gap, barely improved from exp_049's +0.064). Pre-quant val_bpb=1.6281 (best ever — int6 noise is excellent regularizer). Strength=0.1 every 10 steps is far too gentle for int6's coarser quantization. Artifact 6.9MB (mechanism works). Need 6x stronger QAT for int6.

### Experiment 055: Strong Int6 QAT [OUR TWIST]
**What**: Int6 QAT with 6x stronger signal: `QAT_STRENGTH=0.3 QAT_EVERY=5 QAT_BITS=6 QUANT_BITS=6`. Same pre-warmdown timing (lr_mul >= 0.8).
**Motivation**: exp_052 showed strength=0.1/every=10 barely closes int6 gap (+0.064 to +0.061). Int6 has 63 vs 255 levels — need proportionally stronger training signal. 6x more total QAT signal (3x strength * 2x frequency).
**Run**: Best config + `QAT_PREWARMDOWN=1 QAT_STRENGTH=0.3 QAT_EVERY=5 QAT_STOP_LR_MUL=0.8 QAT_BITS=6 QUANT_BITS=6`.
**Expect**: Quant gap should close significantly. Risk: stronger QAT noise may hurt pre-quant BPB (regularization has diminishing returns). Target: quant gap < +0.020 BPB.
**Baseline**: exp_052 quant gap +0.061. exp_049 (no QAT) +0.064.

### ~~Experiment 053: Depth-Recurrent Warmdown (Gentle) [ORIGINAL, HIGH RISK]~~ COMPLETED — DISCARD
**Result**: val_bpb=1.6378 (+0.008 BPB), artifact 12.9MB (-0.2MB). Even gentle alpha=0.1 during warmdown hurts convergence. Negligible compression benefit. Warmdown phase is sacred — no auxiliary losses allowed.

### ~~Experiment 054: Depth Recurrence (Stronger) [ORIGINAL]~~ KILLED
**Reason**: exp_053 (gentle, alpha=0.1) already hurts +0.008 BPB. Stronger alpha would be worse. Depth recurrence during warmdown is a dead end.

---

## Completed (lab/mar26b sessions 1+2)

### Session 1 (exp_001-034): val_bpb 2.4109 → 1.6334
- ~~10-layer architecture~~ [KNOWN] — -0.05 BPB
- ~~Muon weight decay 0.02→0.10~~ [KNOWN] — -0.099 BPB total
- ~~Warmdown-aware WD scheduling~~ [**ORIGINAL**] — -0.010 BPB extra
- ~~FP16 tied embeddings~~ [KNOWN] — quant gap +0.0001
- ~~Sliding window eval~~ [KNOWN] — implemented
- ~~Frequency-decomposed skip gating~~ [**ORIGINAL**] — -0.010 BPB
- ~~MLP_MULT=3~~ [KNOWN] — -0.003 BPB
- ~~Gradient clipping=0.5~~ [SWEEP] — -0.016 BPB
- ~~Batch scaling to 24576~~ [SWEEP] — -0.082 BPB (biggest single win)

### Session 3 (exp_048-053): val_bpb 1.6309 → 1.6299
- ~~Pre-warmdown QAT (fix exp_036)~~ [**OUR TWIST**] — -0.001 BPB, NEW BEST (QAT as regularizer)
- ~~Int6 quantization (QUANT_BITS=6)~~ [KNOWN] — +0.064 BPB quant gap, needs QAT
- ~~Int6 QAT (strength=0.1, every=10)~~ [OUR TWIST] — +0.061 quant gap, too gentle. Pre-quant 1.6281 best ever
- ~~Depth-recurrent warmdown (alpha=0.1)~~ [**ORIGINAL**] — +0.008 BPB, -0.2MB. KILLED (warmdown is sacred)
- Zstd compression — not yet run
- Int6 + Zstd combined — not yet run

### Session 2 (exp_035-047): val_bpb 1.6334 → 1.6309
- ~~Adaptive Newton-Schulz~~ [ORIGINAL] — neutral, killed
- ~~Warmdown QAT (ramping)~~ [ORIGINAL] — +0.057 BPB, killed (mechanism works, timing wrong)
- ~~SWA (wide + narrow)~~ [KNOWN] — killed for Mac, H100-only
- ~~Per-layer LR scaling~~ [OUR TWIST] — -0.001 BPB, kept (LAYER_LR_SCALE=0.5)
- ~~Asymmetric MLP (2,4)~~ [OUR TWIST] — -0.001 BPB, kept
- ~~WARMUP_STEPS=50~~ [SWEEP] — -0.0002 BPB, kept
- ~~Extreme asymmetry (1,5)~~ — worse, killed
- ~~LAYER_LR_SCALE=1.0~~ — saturated, closed
- ~~FREQ_SKIP_WINDOW=16~~ — no difference, closed
- ~~NUM_KV_HEADS=2~~ — worse BPB AND compression, killed
- ~~LOGIT_SOFTCAP=15~~ — worse, killed

---

## Killed Ideas (full list)

See EXPERIMENT_LOG.md for details. Key killed ideas:
- Warmdown < 1200, DropHead, EMA blending, momentum ramp
- ROPE_BASE=50000, QK_GAIN=1.0, WD=0.15/0.20
- Batch=32k (too slow), 11L+MLP3x on Mac (too heavy)
- Ramping warmdown QAT, SWA on Mac, inverse layer LR
- Extreme MLP asymmetry (1,5), KV_HEADS=2, SOFTCAP=15
- Depth-recurrent warmdown (any alpha) — warmdown too sensitive for auxiliary losses
