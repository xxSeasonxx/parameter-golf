# Run Analysis: a040c35

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
- All-time best val_bpb: 1.629900 (commit 4585b0b)
- This run vs best: +0.006328 (worse)
- Recent 5 runs:
  - [discard] bpb=1.6975 | exp_049: Int6 per-row quant (QUANT_BITS=6) — +0.064 BPB quant gap, artifact 48% smaller (6.8MB). Needs QAT
  - [keep] bpb=1.6299 | exp_051: Pre-warmdown QAT (strength=0.1, every 10 steps, lr_mul>=0.8) — NEW BEST (-0.0010 vs 1.6309)
  - [discard] bpb=1.6894 | exp_052: Int6 QAT (QAT_BITS=6, strength=0.1, every=10) — +0.061 quant gap barely improved from exp_049. Pre-quant 1.6281 best ever (int6 noise as regularizer)
  - [discard] bpb=1.6378 | exp_053: Depth-recurrent warmdown (alpha=0.1, layers 2,3,4) — +0.008 BPB, -0.2MB artifact. Convergence cost > compression benefit
  - [discard] bpb=1.6299 | exp_054: Strong QAT (strength=0.2, every=5) — identical to exp_051. QAT regularization saturated

## Improvement Lineage (kept runs)
  - 6ed4a1d bpb=1.829112 | 10L+FP16 tok_emb+Muon WD=0.02 — NEW BEST
  - 384185d bpb=1.788251 | exp_011: 10L+FP16 tok_emb+Muon WD=0.05 — NEW BEST
  - 384185d bpb=1.760833 | exp_013: 10L+FP16 tok_emb+Muon WD=0.10 — NEW BEST
  - 18cf2e2 bpb=1.750383 | exp_014: WD=0.10 + warmdown-aware scheduling — NEW BEST
  - c8c2028 bpb=1.740320 | exp_017: freq-decomposed skip gating (W=32) — NEW BEST
  - c8c2028 bpb=1.658359 | exp_022: TRAIN_BATCH_TOKENS=16384 — NEW BEST -0.082 BPB! 1057 steps, gradient quality wins
  - c8c2028 bpb=1.651780 | exp_024: TRAIN_BATCH_TOKENS=24576 — NEW BEST -0.007 BPB. Batch scaling continues
  - 996244b bpb=1.6334 | exp_033: MLP3x + GRAD_CLIP=0.5 — BEST
  - 50dbc88 bpb=1.6321 | exp_039: per-layer LR scaling (LAYER_LR_SCALE=0.5) — MARGINAL NEW BEST
  - 3af0786 bpb=1.6311 | exp_041: asymmetric MLP width (encoder MLP2x/decoder MLP4x) — MARGINAL NEW BEST
  - 966ddeb bpb=1.6309 | exp_046: WARMUP_STEPS=50 — MARGINAL NEW BEST (-0.0002 vs 1.6311)
  - 4585b0b bpb=1.6299 | exp_051: Pre-warmdown QAT (strength=0.1, every 10 steps, lr_mul>=0.8) — NEW BEST (-0.0010 vs 1.6309)

## Metrics
Per-step metrics: `.lab/a040c35/metrics.jsonl`
Each line is JSON with keys: `type`, `step`, `total`, `train_loss`/`val_loss`/`val_bpb`, `train_time_ms`, `step_avg_ms`

## Agent Investigation Notes

**Experiment 055: Strong Int6 QAT — DISCARD (Int6 QAT Killed)**

Note: analyze.py picked up stale code on this commit (exp_034 config). Actual exp_055 results from run log:
- Pre-quant val_bpb: 1.6271 (713 steps, 842ms/step)
- Post-int6 val_bpb: 1.6904 (+0.063 quant gap)
- Artifact: 6,938,640 bytes (6.9MB)

**Key finding**: Definitive result for int6 pre-warmdown QAT. Three experiments confirm gap is +0.063 +/- 0.003 regardless of QAT strength (no-QAT, 1x, 6x). Pre-warmdown QAT fundamentally cannot close the int6 gap — 63-level quantization is too coarse for periodic weight nudging.

**Surprising**: Pre-quant BPB (1.6271) matches best ever. Stronger int6 noise is an excellent regularizer, but regularization and quant-gap closure are independent effects.

**Next**: Kill int6 QAT on Mac. Need STE forward-pass quantization, GPTQ, or accept int6 is H100-only.
