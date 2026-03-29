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

### ~~Experiment 055: Strong Int6 QAT [OUR TWIST]~~ COMPLETED — DISCARD
**Result**: Post-int6 val_bpb=1.6904 (+0.063 quant gap). Pre-quant 1.6271 (matches best). 3 experiments now confirm int6 quant gap is ~+0.063 regardless of QAT strength. Pre-warmdown QAT cannot close the int6 gap — 63 levels is fundamentally too coarse. **Int6 QAT killed on Apple Silicon.** Need STE forward-pass quantization or GPTQ for int6.

### ~~Experiment 053: Depth-Recurrent Warmdown (Gentle) [ORIGINAL, HIGH RISK]~~ COMPLETED — DISCARD
**Result**: val_bpb=1.6378 (+0.008 BPB), artifact 12.9MB (-0.2MB). Even gentle alpha=0.1 during warmdown hurts convergence. Negligible compression benefit. Warmdown phase is sacred — no auxiliary losses allowed.

### ~~Experiment 054: Strong Int8 QAT [SWEEP]~~ COMPLETED — DISCARD
**Result**: val_bpb=1.6299 (identical to exp_051). 4x stronger QAT (strength=0.2, every=5) produces identical results. QAT regularization saturates — it's a binary threshold, not a gradient. Don't tune QAT hyperparameters for int8. Pure env-var sweep, no code change.

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

### Future: Int6 Gap Closure (H100 or advanced techniques)

### ~~Experiment 056: Full-Training QAT (QAT_STOP_LR_MUL=0) [SWEEP]~~ COMPLETED — DISCARD
**Result**: val_bpb=1.6308 (+0.0009 vs best). Pre-quant 1.6283 (slightly worse than 1.6270). Quant gap +0.0025 (slightly better than +0.0029). Effects cancel. Full-training QAT is neutral — weight perturbations during warmdown are tolerated (unlike loss-modifying interventions) but don't help. Keep pre-warmdown-only config.

### ~~Experiment 057: Cosine Warmdown Shape [SWEEP]~~ COMPLETED — DISCARD
**Result**: val_bpb=1.6503 (+0.020 BPB). Cosine keeps LR too high too long, insufficient fine convergence. Linear warmdown is optimal. Kill warmdown shape experiments.

### ~~Experiment 058: Label Smoothing (0.1) [KNOWN]~~ COMPLETED — DISCARD (CATASTROPHIC)
**Result**: val_bpb=2.0499 (+0.42 BPB, catastrophic). With vocab=1024, smoothing redistributes too much mass per non-target token. Training objective (smoothed CE) diverges from eval metric (standard CE). Label smoothing is KILLED for small vocab.

---

### Session 3 (exp_048-055): val_bpb 1.6309 → 1.6299
- ~~Pre-warmdown QAT (fix exp_036)~~ [**OUR TWIST**] — -0.001 BPB, NEW BEST (QAT as regularizer)
- ~~Int6 quantization (QUANT_BITS=6)~~ [KNOWN] — +0.064 BPB quant gap, needs QAT
- ~~Int6 QAT (strength=0.1, every=10)~~ [OUR TWIST] — +0.061 quant gap, too gentle. Pre-quant 1.6281 best ever
- ~~Depth-recurrent warmdown (alpha=0.1)~~ [**ORIGINAL**] — +0.008 BPB, -0.2MB. KILLED (warmdown is sacred)
- ~~Strong int8 QAT (strength=0.2, every=5)~~ [SWEEP] — identical to exp_051. QAT saturated. Pure env-var sweep
- ~~Strong int6 QAT (strength=0.3, every=5)~~ [OUR TWIST] — +0.063 gap, KILLED. 3 exps confirm int6 gap impervious to QAT
- ~~Full-training QAT (QAT_STOP_LR_MUL=0)~~ [SWEEP] — neutral, quant gap slightly better but pre-quant worse. Effects cancel
- ~~Cosine warmdown shape~~ [SWEEP] — +0.020 BPB, linear is optimal. Kill warmdown shape experiments
- ~~Label smoothing (0.1)~~ [KNOWN] — +0.42 BPB CATASTROPHIC. Vocab=1024 too small, loss objective mismatch
- ~~Higher matrix_lr=0.06 (1.5x)~~ [SWEEP] — +0.009 BPB, Muon already well-scaled at 0.04. Kill LR sweep
- ~~Higher tied_embed_lr=0.1 (2x)~~ [SWEEP] — +0.030 BPB, tied embeddings extremely LR-sensitive
- ~~Lower tied_embed_lr=0.03 (0.6x)~~ [SWEEP] — +0.011 BPB, lower embed LR under-trains. Embed LR sweep fully closed: 0.05 is the V-shaped minimum
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
- Int6 pre-warmdown QAT at any strength — 3 experiments confirm gap is ~+0.063 regardless (exp_049/052/055)
- Full-training QAT (QAT_STOP_LR_MUL=0) — neutral vs pre-warmdown-only, no benefit (exp_056)
- Cosine warmdown shape — +0.020 BPB, linear is optimal. Kill warmdown shape experiments (exp_057)
- Label smoothing — +0.42 BPB CATASTROPHIC with vocab=1024. Loss objective mismatch kills performance (exp_058)
- Matrix LR sweep — matrix_lr=0.06 overshoots (+0.009 BPB), Muon's orthogonalized updates are well-scaled at 0.04 (exp_059)
- Embed LR sweep (both directions) — embed_lr=0.1 (2x) gives +0.030 BPB, embed_lr=0.03 (0.6x) gives +0.011 BPB. V-shaped response with minimum at 0.05. Fully closed (exp_060/061)
