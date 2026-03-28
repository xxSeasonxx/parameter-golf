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

### Experiment 049: Int6 Quantization [KNOWN]
**What**: 6-bit per-row quantization for matrix weights (clip to [-31,31], scale=max/31). Embeddings stay fp16, control tensors stay fp32.
**Code change**: ~30 lines. Add `QUANT_BITS` env var. New `quantize_float_array_int6()` function. Modify `quantize_state_dict_int8` to dispatch based on bits.
**Run**: Best config + `QUANT_BITS=6`.
**Expect**: Artifact drops to ~8-9MB. BPB degrades +0.01-0.03 (more on Mac than H100). The point is validating the mechanism, not the absolute BPB.
**Why**: Fits ~40% more effective params in 16MB. This is what gets competition entries from 1.17 to 1.13.

### Experiment 050: Int6 + Zstd Combined [KNOWN]
**What**: Both int6 and zstd together.
**Run**: Best config + `QUANT_BITS=6` + zstd enabled.
**Expect**: Artifact ~7-8MB. Massive headroom for larger model on H100.

### Experiment 051: Pre-Warmdown QAT (Fix exp_036) [OUR TWIST]
**What**: Constant-strength QAT (strength=0.1) during pre-warmdown phase only (lr_mul >= 0.8). Stop QAT when warmdown begins. This fixes exp_036's failure where ramping noise fought warmdown convergence.
**Code change**: ~15 lines. Reuse `sim_quant_int8()`. Add `QAT_PREWARMDOWN` env var, `QAT_STRENGTH` (fixed, not ramping), `QAT_STOP_LR_MUL` (threshold to stop).
**Run**: Best config + `QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10`.
**Expect**: Quant gap shrinks (from ~0.003 to ~0.001). Overall BPB similar or slightly better.
**Key insight from exp_036**: The mechanism works (quant gap closed to 0.0002) but timing was wrong. Pre-warmdown QAT separates "learning to quantize" from "final convergence."

### Experiment 052: Pre-Warmdown QAT + Int6 [OUR TWIST]
**What**: QAT with int6 simulation instead of int8.
**Code change**: Add `sim_quant_int6()`, ~10 more lines.
**Run**: Best config + int6 QAT.
**Expect**: Model learns int6-friendly weights. Quant gap at int6 level should shrink.

### Experiment 053: Depth-Recurrent Warmdown (Gentle) [ORIGINAL, HIGH RISK]
**What**: During warmdown, add L2 regularization between middle layers (3,4,5) to encourage weight sharing. `L_share = alpha * ||W_3 - W_4||^2 + alpha * ||W_4 - W_5||^2`. Alpha ramps from 0 to 0.1 during warmdown.
**Code change**: ~20 lines. Add `DEPTH_RECURRENCE_WARMDOWN` env var, compute pairwise L2 between designated layers, add to loss.
**Run**: Best config + `DEPTH_RECURRENCE_WARMDOWN=1`.
**Expect**: Middle layer weights converge toward each other. Measure L2 distance between layers before/after. If distance < 5% of weight norm, we can share weights at serialization → ~2-3MB artifact savings.
**Why original**: Nobody does depth recurrence as a warmdown regularizer. This is "train deep, compress to recurrent."
**Risk**: Could hurt BPB if alpha too high. Start gentle (0.1).

### Experiment 054: Depth Recurrence (Stronger) [ORIGINAL]
**What**: Same as 053 but alpha=0.5.
**Run**: If 053 shows weight convergence, push harder.
**Expect**: Layers become nearly identical. Artifact savings 3-4MB.

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
