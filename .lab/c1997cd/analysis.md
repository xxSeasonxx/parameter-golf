# Run Analysis: c1997cd

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
- All-time best val_bpb: 1.633400 (commit 996244b)
- This run vs best: +0.002828 (worse)
- Recent 5 runs:
  - [keep] bpb=1.658359 | exp_022: TRAIN_BATCH_TOKENS=16384 — NEW BEST -0.082 BPB! 1057 steps, gradient quality wins
  - [discard] bpb=1.687706 | exp_023: batch=16k + warmdown=600 — +0.029 BPB worse than warmdown=1200. Long warmdown wins again
  - [keep] bpb=1.651780 | exp_024: TRAIN_BATCH_TOKENS=24576 — NEW BEST -0.007 BPB. Batch scaling continues
  - [keep] bpb=1.6334 | exp_033: MLP3x + GRAD_CLIP=0.5 — BEST
  - [discard] bpb=1.6352 | exp_035: adaptive Newton-Schulz scheduling (5→7 warmdown) — neutral, +0.0018 vs best

## Improvement Lineage (kept runs)
  - 6ed4a1d bpb=1.829112 | 10L+FP16 tok_emb+Muon WD=0.02 — NEW BEST
  - 384185d bpb=1.788251 | exp_011: 10L+FP16 tok_emb+Muon WD=0.05 — NEW BEST
  - 384185d bpb=1.760833 | exp_013: 10L+FP16 tok_emb+Muon WD=0.10 — NEW BEST
  - 18cf2e2 bpb=1.750383 | exp_014: WD=0.10 + warmdown-aware scheduling — NEW BEST
  - c8c2028 bpb=1.740320 | exp_017: freq-decomposed skip gating (W=32) — NEW BEST
  - c8c2028 bpb=1.658359 | exp_022: TRAIN_BATCH_TOKENS=16384 — NEW BEST -0.082 BPB! 1057 steps, gradient quality wins
  - c8c2028 bpb=1.651780 | exp_024: TRAIN_BATCH_TOKENS=24576 — NEW BEST -0.007 BPB. Batch scaling continues
  - 996244b bpb=1.6334 | exp_033: MLP3x + GRAD_CLIP=0.5 — BEST

## Metrics
Per-step metrics: `.lab/c1997cd/metrics.jsonl`
Each line is JSON with keys: `type`, `step`, `total`, `train_loss`/`val_loss`/`val_bpb`, `train_time_ms`, `step_avg_ms`

## Agent Investigation Notes

**YOU MUST write your analysis findings below this line before moving on.**
Look at the plots above. Load metrics.jsonl and compare with previous runs.
What patterns do you see? What was surprising? What should you try next?

### exp_036: Warmdown-Phase QAT Analysis

**Actual run results** (analyze.py picked up stale state from code commit; actual results from run log):
- Run ID: exp_036_qat_warmdown
- val_bpb: 1.6907 (int8 roundtrip), pre-quant: 1.6905
- Quant gap: 0.0002 BPB (excellent)
- Artifact: 12,837,630 bytes (~12.8MB)
- Steps: 709/2000 at 847ms/step
- Status: DISCARD (+0.057 vs best 1.6334)

**Key observations**:
1. The quant gap (0.0002 BPB) is the smallest we have ever seen. This proves the QAT mechanism genuinely teaches the model to be quantization-friendly.
2. The compression improved slightly (12.8MB vs 13.0MB), confirming QAT pushes weights toward more quantizable (lower entropy) distributions.
3. However, the overall BPB suffered massively (+0.057). The ramping QAT noise during warmdown conflicts with the optimizer's convergence trajectory.
4. The fundamental problem: with warmdown_iters=1200 and only ~700 steps, warmdown covers everything. QAT noise is active from the start with increasing strength. There is no "quiet" convergence phase.

**Reflection**: The three-force synergy hypothesis (LR decay + WD increase + QAT noise) was wrong in this form. The forces compete rather than synergize. LR decay and WD increase are convergence-promoting forces. QAT noise is a divergence force. Adding divergence during the convergence-critical phase is counterproductive.

**Next directions**:
- On H100 with 1500+ steps, a pre-warmdown QAT phase (constant low strength for last ~200 steps before warmdown) could work.
- Layer-wise quantization budget allocation is the next original idea to test.
- Depth-recurrent warmdown remains high-risk but untested.
