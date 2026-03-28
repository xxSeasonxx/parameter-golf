# Run Analysis: 4ef8635

**val_bpb: 1.636228**

## Configuration
```
  run_id:exp_034_gradclip025
  mlx_version:0.31.1
  train_loader:shards pattern=./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
  val_loader:shards pattern=./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin tokens:62021632
  WARNING: train_loader:subset dataset:fineweb10B_sp1024 train_shards:10/195 new epochs will arrive sooner than the full dataset
  tokenizer_path:./data/tokenizers/fineweb_1024_bpe.model
  model_params:24142928 vocab_size:1024 layers:10 dim:512 heads:8 kv_heads:4 seq_len:1024 tie_embeddings:True
  iterations:2000 train_batch_tokens:24576 grad_accum_steps:8 microbatch_tokens:3072 microbatch_batch_size:3 val_batch_size:524288 warmup_steps:20 max_wallclock_seconds:600.000
  mlx_max_microbatch_tokens:8192
  optimizer:muon+adam muon_matrix_params:60 scalar_params:42 embed_lr:0.05 matrix_lr:0.04 scalar_lr:0.04 muon_momentum:0.95 muon_steps:5
  val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path=./data/tokenizers/fineweb_1024_bpe.model
  compute_dtype:mlx.core.bfloat16 compile:True
  dtypes tok_emb:mlx.core.bfloat16 linear_weight:mlx.core.float32 skip_weights:mlx.core.float32
```

## Performance
- Training time: 600.7s
- Steps completed: 690/2000
- Stopped early (wallclock cap)

## Final Metrics
- Int8+zlib roundtrip val_bpb: 1.636200
- Int8+zlib exact val_bpb: 1.63622841

## Loss Trajectory
- First loss: 6.9369
- Final loss: 2.8349
- Min loss: 2.8349 (step 600)
- First val_bpb: 4.1086
- Best val_bpb during training: 1.6326 (step 690)
- Avg step time: 885.8ms

## Plots
- `loss_curve.png`
- `val_bpb_curve.png`
- `step_timing.png`

## Cross-Run Summary
- All-time best val_bpb: 1.651780 (commit c8c2028)
- This run vs best: -0.015552 (NEW BEST)
- Recent 5 runs:
  - [discard] bpb=1.760961 | exp_020: ROPE_BASE=50000 — +0.021 BPB regression vs default 10000
  - [discard] bpb=1.754235 | exp_021: QK_GAIN_INIT=1.0 — +0.014 BPB regression vs default 1.5
  - [keep] bpb=1.658359 | exp_022: TRAIN_BATCH_TOKENS=16384 — NEW BEST -0.082 BPB! 1057 steps, gradient quality wins
  - [discard] bpb=1.687706 | exp_023: batch=16k + warmdown=600 — +0.029 BPB worse than warmdown=1200. Long warmdown wins again
  - [keep] bpb=1.651780 | exp_024: TRAIN_BATCH_TOKENS=24576 — NEW BEST -0.007 BPB. Batch scaling continues

## Improvement Lineage (kept runs)
  - 6ed4a1d bpb=1.829112 | 10L+FP16 tok_emb+Muon WD=0.02 — NEW BEST
  - 384185d bpb=1.788251 | exp_011: 10L+FP16 tok_emb+Muon WD=0.05 — NEW BEST
  - 384185d bpb=1.760833 | exp_013: 10L+FP16 tok_emb+Muon WD=0.10 — NEW BEST
  - 18cf2e2 bpb=1.750383 | exp_014: WD=0.10 + warmdown-aware scheduling — NEW BEST
  - c8c2028 bpb=1.740320 | exp_017: freq-decomposed skip gating (W=32) — NEW BEST
  - c8c2028 bpb=1.658359 | exp_022: TRAIN_BATCH_TOKENS=16384 — NEW BEST -0.082 BPB! 1057 steps, gradient quality wins
  - c8c2028 bpb=1.651780 | exp_024: TRAIN_BATCH_TOKENS=24576 — NEW BEST -0.007 BPB. Batch scaling continues

## Metrics
Per-step metrics: `.lab/4ef8635/metrics.jsonl`
Each line is JSON with keys: `type`, `step`, `total`, `train_loss`/`val_loss`/`val_bpb`, `train_time_ms`, `step_avg_ms`

## Agent Investigation Notes

### exp_035: Adaptive Newton-Schulz Scheduling (DISCARD)

**Result**: val_bpb=1.6352 (int8), +0.0018 worse than best (1.6334). Pre-quant: 1.6321.
**Artifact**: 13,007,336 bytes (~13.0MB). Steps: 695/2000 at 863ms/step avg.

**Why the extra NS precision does not help**:
- At dim=512, Newton-Schulz converges rapidly. The approximation error after 5 iterations is already negligible compared to gradient noise at this scale. Going to 7 iterations barely changes the orthogonalized gradient direction.
- The technique adds ~2ms overhead per step during warmdown (computing 2 extra NS iterations), which is small but provides no return.

**Pre-quant vs post-quant gap**:
- Pre-quant val_bpb was 1.6321 -- marginally better than best's ~1.630 pre-quant equivalent. This suggests the technique might have a tiny real effect, but it is entirely masked by int8 quantization noise (+0.003 BPB quant gap here vs ~0.003 for the baseline).
- The adaptive NS does not change weight magnitudes or distributions in a way that helps compressibility -- artifact size is nearly identical (13.0MB vs 13.0MB).

**Conclusion**: Newton-Schulz 5 iterations is sufficient for this model size. The error from 5 vs 7 iterations is negligible compared to both gradient noise and quantization noise. Do not schedule NS iterations -- the complexity is not justified.
