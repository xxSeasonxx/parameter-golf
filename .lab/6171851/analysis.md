# Run Analysis: 6171851

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
  - [keep] bpb=1.6309 | exp_046: WARMUP_STEPS=50 — MARGINAL NEW BEST (-0.0002 vs 1.6311)
  - [discard] bpb=1.6364 | exp_047: LOGIT_SOFTCAP=15.0 — +0.006 BPB worse. Default 30.0 is well-calibrated
  - [discard] bpb=1.6975 | exp_049: Int6 per-row quant (QUANT_BITS=6) — +0.064 BPB quant gap, artifact 48% smaller (6.8MB). Needs QAT
  - [keep] bpb=1.6299 | exp_051: Pre-warmdown QAT (strength=0.1, every 10 steps, lr_mul>=0.8) — NEW BEST (-0.0010 vs 1.6309)
  - [discard] bpb=1.6894 | exp_052: Int6 QAT (QAT_BITS=6, strength=0.1, every=10) — +0.061 quant gap barely improved from exp_049. Pre-quant 1.6281 best ever (int6 noise as regularizer)

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
Per-step metrics: `.lab/6171851/metrics.jsonl`
Each line is JSON with keys: `type`, `step`, `total`, `train_loss`/`val_loss`/`val_bpb`, `train_time_ms`, `step_avg_ms`

## Agent Investigation Notes

**YOU MUST write your analysis findings below this line before moving on.**
Look at the plots above. Load metrics.jsonl and compare with previous runs.
What patterns do you see? What was surprising? What should you try next?

### exp_053: Depth-Recurrent Warmdown — DISCARD

**Result**: Post-quant val_bpb=1.6378 (+0.008 vs best 1.6299). Pre-quant val_bpb=1.6345 (+0.008 vs best pre-quant 1.6270). Artifact: 12,913,555 bytes (12.9MB, only -0.2MB savings vs 13.1MB).

**Key findings**:
1. The warmdown phase is too sensitive for any auxiliary loss. This is the third warmdown-phase intervention that failed (SWA +0.003/+0.127, warmdown QAT +0.057, depth recurrence +0.008). The pattern is clear and definitive.
2. The compression benefit is negligible. alpha=0.1 ramping from 0 to 0.1 during warmdown is far too gentle to make layers 2,3,4 actually similar. Would need alpha > 1.0 to see meaningful weight convergence, but that would destroy BPB.
3. The "train deep, compress to recurrent" idea is theoretically sound but the warmdown phase is the wrong time to apply it. The model needs clean convergence during warmdown. A pre-warmdown approach (like QAT in exp_051) might work, but the expected 0.2MB savings doesn't justify the complexity.
4. Step time (844ms) is normal — the L2 penalty computation has negligible overhead.

**Broader insight**: Warmdown-phase interventions on Apple Silicon (700 steps) are fundamentally different from H100 (1500+ steps). With only 700 steps, warmdown represents ~60% of all training. Any noise during this phase has outsized impact. On H100, warmdown is a smaller fraction and the model has more time to converge through noise.

**What to try next**: Focus on remaining H100-prep experiments (zstd compression, strong int6 QAT). Apple Silicon has hit its plateau at ~1.630 BPB.
