# Trusted Baseline

## Identity

- Commit: `87b4a2f`
- Script: `train_gpt.py`
- Run name: `h100_l0_only_20260430_042516`
- Dataset: full FineWeb SP1024 training set, `195/195` train shards
- Hardware target: 8xH100

## Result

| Metric | Value |
|---|---:|
| Steps | `6535/20000` |
| Step time | `91.50ms/step` |
| Raw checkpoint | `107,117,755 bytes` |
| Post-quant exact BPB | `1.20303259` |
| TTT BPB | `1.2162` |
| Artifact | `14,184,763 bytes` |
| Total submission size | `14,275,796 bytes` |

This is the corrected comparison point for the next round.

The post-quant exact score is the active baseline. The current LoRA TTT path is
legal but harmful on this checkpoint, so it must not be treated as the best score
unless a TTT-only ablation beats `1.20303259`.

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
- stride-64 post-quant final evaluation

## Previous Baseline

The old `baseline-3e34098` / `clean_11l_no_ema` row is now historical because
the eval/accounting repair changed the final BPB accounting:

| Metric | Old Value |
|---|---:|
| Post-quant BPB | `1.2316` |
| TTT BPB | `1.2102` |
| Artifact | `14.48MB` |

Keep the old row for lineage, but do not use it as the active decision gate.

## Not Baseline

- Deep supervision (`DEEP_SUPERVISION=1`, alpha `0.05`, layers `3,7`) is not the baseline. Corrected H100 result: post-quant `1.20610810`, TTT `1.2192`, worse than the corrected control.
- EMA(0.997) is not the baseline. It produced normal raw BPB but collapsed after final EMA load to post-quant `1.3724` and TTT `1.3035`.
- The current 13L int8+zstd+calibrated recipe is not the baseline. It reached TTT `1.2280`, worse than clean 11L.
- SWA is not the baseline. It regressed on Mac and on H100.
- Int6 without STE/GPTQ is not the baseline. Its H100 quantization gap was too large.
- Progressive growth is not the baseline. It failed local screening and should not be promoted without a new reason.

## Reproducing The Baseline Control

Use `h100_l0_only.sh` for a full retraining control. It uses the clean 11L stack
with stride-64 final eval and explicitly disables EMA, DyT, and Polar Express.

For TTT-only follow-up on an existing checkpoint, use the eval-only path:

```bash
EVAL_ONLY_CHECKPOINT=./final_model.int8.ptz EVAL_ONLY_SKIP_ROUNDTRIP=1
```
