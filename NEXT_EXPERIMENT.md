# H100 Next Experiment

## Current State

The previous next experiment, isolated deep supervision on the clean 11L stack,
has been run and discarded.

The follow-up TTT sweep on commit `d0b36ca` rescued the legal score-first TTT
path. A fresh clean 11L run produced post-quant exact `1.19775040`, and the best
eval-only TTT setting was:

```bash
TTT_LORA_RANK=8 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=128
```

with TTT BPB `1.1966`.

Corrected H100 control on commit `87b4a2f`:

| Metric | Value |
|---|---:|
| Run | `h100_l0_only_20260430_042516` |
| Train shards | `195/195` |
| Steps | `6535/20000` |
| Step time | `91.50ms/step` |
| Post-quant exact BPB | `1.20303259` |
| TTT BPB | `1.2162` |
| Artifact | `14,184,763 bytes` |

Deep supervision on the same commit:

| Metric | Value |
|---|---:|
| Run | `h100_next_deep_supervision_20260430_044023` |
| Train shards | `195/195` |
| Steps | `6549/20000` |
| Post-quant exact BPB | `1.20610810` |
| TTT BPB | `1.2192` |
| Artifact | `14,085,158 bytes` |

Decision: **kill deep supervision for now**. It regressed both post-quant and
TTT versus the corrected control.

## Next RunPod Step

Do not spend another full H100 training run on deep supervision, QK gain, Polar
Express, or DyT. Local MLX screens on 2026-04-30 did not produce a survivor.

First run a tight eval-only sweep around the current TTT winner:

```bash
EVAL_ONLY_CHECKPOINT=./final_model.int8.ptz EVAL_ONLY_SKIP_ROUNDTRIP=1
TTT_LORA_RANK=8 TTT_LORA_LR=0.002 TTT_CHUNK_SIZE=64
TTT_LORA_RANK=8 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=64
TTT_LORA_RANK=8 TTT_LORA_LR=0.004 TTT_CHUNK_SIZE=64
TTT_LORA_RANK=8 TTT_LORA_LR=0.002 TTT_CHUNK_SIZE=128
TTT_LORA_RANK=8 TTT_LORA_LR=0.004 TTT_CHUNK_SIZE=128
TTT_LORA_RANK=4 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=128
TTT_LORA_RANK=16 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=128
```

Then start the SP2048 full-run gamble:

```bash
python3 data/download_hf_docs_and_tokenize.py \
  --output-root ./data/research_tokenizers \
  --tokenizer-config ./data/tokenizer_specs_sp2048.json \
  --tokenizer-train-docs 200000

./research_sp2048_h100.sh
```

## Decision Rule

Promote a TTT config if it beats the current best:

- strong promote: TTT BPB `< 1.1950`
- weak promote: TTT BPB `< 1.1966`
- otherwise keep `TTT_LORA_RANK=8 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=128`

Promote SP2048 only if its best legal final score beats `1.1966` and remains
under 16MB total submission size. Otherwise, keep the tuned SP1024 TTT result as
the fallback and do not spend more time on SP1024 micro-toggles.

## Local Tokenizer Prep

Keep `data/tokenizer_specs.json` as the SP1024 default. Use
`data/tokenizer_specs_sp2048.json` for the one-day full-run target, and
`data/tokenizer_specs_research.json` only for intentional SP2048/SP4096 smoke
exports.

First run a smoke export with a small docs/training slice:

```bash
python3 data/download_hf_docs_and_tokenize.py \
  --output-root ./data/research_tokenizers \
  --tokenizer-config ./data/tokenizer_specs_research.json \
  --tokenizer-train-docs 200000 \
  --chunk-tokens 1000000
```

After the smoke export, run a short local MLX check by pointing `DATA_PATH`,
`TOKENIZER_PATH`, and `VOCAB_SIZE` at either exported tokenizer. Promote to a
full RunPod export only if the smoke path verifies tokenizer/data pairing,
artifact accounting, and final BPB eval.
