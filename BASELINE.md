# Trusted Baseline

## Identity

- Git tag: `baseline-3e34098`
- Commit: `3e34098`
- Script: `train_gpt.py`
- Run name: `clean_11l_no_ema`
- Dataset: full FineWeb SP1024 training set, `195/195` train shards
- Hardware target: 8xH100

## Result

| Metric | Value |
|---|---:|
| Steps | `8007/20000` |
| Step time | `74.95ms/step` |
| Raw BPB | `1.2282` |
| Post-quant BPB | `1.2316` |
| TTT BPB | `1.2102` |
| Artifact | `14,479,660 bytes` |

This is the comparison point for the next round.

Post-quant BPB: 1.2316
TTT BPB: 1.2102

## Baseline Stack

The baseline keeps our validated core stack:

- 11 transformer layers
- SP1024 tokenizer
- tied embeddings with `tok_emb` kept fp16 in serialization
- GQA with 8 query heads and 4 KV heads
- U-Net skip connections
- frequency-decomposed skip gating
- asymmetric MLP, encoder `2x`, decoder `4x`
- LeakyReLU(0.5)^2 MLP activation
- Muon + Adam optimizers
- warmdown-aware Muon weight decay
- per-layer Muon LR scaling
- pre-warmdown int8 QAT regularizer
- zstd-22 final artifact compression
- legal LoRA TTT evaluation path

## Not Baseline

- EMA(0.997) is not the baseline. It produced normal raw BPB but collapsed after final EMA load to post-quant `1.3724` and TTT `1.3035`.
- The current 13L int8+zstd+calibrated recipe is not the baseline. It reached TTT `1.2280`, worse than clean 11L.
- SWA is not the baseline. It regressed on Mac and on H100.
- Int6 without STE/GPTQ is not the baseline. Its H100 quantization gap was too large.
- Progressive layer growth is not the baseline. It failed local screening and should not be promoted without a new reason.

## Reproducing The Baseline Control

Use `h100_l0_only.sh` for the active control path. It uses the clean 11L baseline stack with stride-64 final eval and explicitly disables EMA, DyT, and Polar Express.
