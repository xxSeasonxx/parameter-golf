# Run Analysis: 79261f5

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
- All-time best val_bpb: 1.630900 (commit 966ddeb)
- This run vs best: +0.005328 (worse)
- Recent 5 runs:
  - [discard] bpb=1.6316 | exp_043: LAYER_LR_SCALE=1.0 — +0.0005 vs best. Scale saturates, 0.5 sufficient
  - [discard] bpb=1.6318 | exp_044: FREQ_SKIP_WINDOW=16 — +0.0007 vs best. Window size doesn't matter, W=32 fine
  - [discard] bpb=1.6345 | exp_045: NUM_KV_HEADS=2 — +0.003 BPB vs best, worse compression (3.64x vs 3.85x). DISCARD
  - [keep] bpb=1.6309 | exp_046: WARMUP_STEPS=50 — MARGINAL NEW BEST (-0.0002 vs 1.6311)
  - [discard] bpb=1.6364 | exp_047: LOGIT_SOFTCAP=15.0 — +0.006 BPB worse. Default 30.0 is well-calibrated

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

## Metrics
Per-step metrics: `.lab/79261f5/metrics.jsonl`
Each line is JSON with keys: `type`, `step`, `total`, `train_loss`/`val_loss`/`val_bpb`, `train_time_ms`, `step_avg_ms`

## Agent Investigation Notes

### Int6 Quantization Analysis (exp_049)

**Note**: analyze.py evaluated this commit using its default int8 quantization pipeline, producing val_bpb=1.6362. The actual experiment used int6 quantization (QUANT_BITS=6), which produced val_bpb=1.6975 from the run log. The key results are from the run log, not analyze.py.

**Key findings**:

1. **Int6 quant gap (+0.064 BPB) is ~20x worse than int8 gap (~0.003)**. This is dramatically non-linear: 63 levels vs 255 levels = 4x less precision, but the error compounds across 10 transformer layers. Each layer's quantization error feeds into the next layer's input, creating multiplicative error amplification. Rough estimate: 4x precision loss * ~5x layer compounding = ~20x effective degradation.

2. **The model trains identically**: Pre-quant val_bpb=1.6332 is within noise of best (1.6309). The training loop, optimizer, and model architecture are completely unaffected by the quantization change. All damage is purely at serialization time when float32 weights are rounded to 6-bit integers.

3. **Artifact size reduction is excellent**: 6.8MB vs 13.1MB (48% reduction). The compression ratio (3.85x) is identical to int8, confirming zlib compresses int6 and int8 payloads equally well. The size reduction comes entirely from the smaller raw payload (6-bit integers pack more tightly).

4. **This strongly motivates QAT**: Since all damage is at serialization, QAT (quantization-aware training) can teach the model to be robust to int6 rounding noise during training. The model would learn weight distributions that are naturally int6-friendly — clustering around the 63 quantization levels rather than spreading continuously. exp_036 already proved the QAT mechanism works (quant gap closed to 0.0002 for int8), just with wrong timing. Pre-warmdown QAT (exp_051/052) should fix this.

5. **H100 implications**: With int6 + QAT, the 16MB artifact budget could fit a 12-layer model or wider MLP. The capacity gain from 48% artifact reduction could more than compensate for any residual int6 quant gap after QAT. This is potentially the single biggest lever for competition performance.
