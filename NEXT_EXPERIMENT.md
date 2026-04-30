# H100 Next Experiment

## Current State

The previous next experiment, isolated deep supervision on the clean 11L stack,
has been run and discarded.

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

The next RunPod work should tune/evaluate the TTT path on the existing corrected
control checkpoint, because current TTT is actively harmful:

- post-quant exact: `1.20303259`
- current TTT: `1.2162`
- regression: `+0.0132 BPB`

Use `EVAL_ONLY_CHECKPOINT=./final_model.int8.ptz` with `EVAL_ONLY_SKIP_ROUNDTRIP=1`
to run TTT ablations without retraining. First reproduce the current TTT score,
then sweep only TTT params:

```bash
TTT_LORA_RANK=8 TTT_LORA_LR=0.01 TTT_CHUNK_SIZE=256
TTT_LORA_RANK=8 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=256
TTT_LORA_RANK=8 TTT_LORA_LR=0.001 TTT_CHUNK_SIZE=256
TTT_LORA_RANK=4 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=256
TTT_LORA_RANK=4 TTT_LORA_LR=0.001 TTT_CHUNK_SIZE=256
TTT_LORA_RANK=8 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=128
TTT_LORA_RANK=8 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=512
TTT_LORA_RANK=8 TTT_LORA_LR=0.003 TTT_CHUNK_SIZE=256 TTT_EPOCHS=2
```

## Decision Rule

Promote a TTT config only if it beats the corrected post-quant exact baseline:

- strong promote: TTT BPB `<= 1.2000`
- weak promote: TTT BPB `< 1.20303259`
- kill TTT path: no ablation beats `1.20303259`

If TTT remains worse, report/post only the post-quant exact score and move the
next research sprint to tokenizer/data work (`SP2048` or `SP4096`) or a larger
architecture change. Small SP1024 toggles are not closing the leaderboard gap.

## Local Tokenizer Prep

Keep `data/tokenizer_specs.json` as the SP1024 default. Use
`data/tokenizer_specs_research.json` only for intentional SP2048/SP4096 research.

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
