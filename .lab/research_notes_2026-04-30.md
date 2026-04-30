# Research Notes - 2026-04-30

## H100 Results Reviewed

Downloaded RunPod artifacts under `runpod-results/`.

| Run | Decision | Post-quant exact BPB | TTT BPB | Artifact | Notes |
|---|---|---:|---:|---:|---|
| `h100_l0_only_20260430_042516` | keep | `1.20303259` | `1.2162` | `14,184,763` | Corrected clean 11L control. TTT is harmful. |
| `h100_next_deep_supervision_20260430_044023` | discard | `1.20610810` | `1.2192` | `14,085,158` | Deep supervision regressed vs corrected control. |

Bookkeeping correction: `.lab/results.tsv` records the best legal final score,
which is post-quant exact for these runs. The previous pending rows used TTT
scores and were misleading because TTT regressed.

## Local MLX Screens

Dataset: `data/datasets/fineweb10B_sp1024_research`, one train shard plus a
small 262,144-token validation slice. These are screening results only, not H100
oracle results.

Common config:

```bash
NUM_LAYERS=10 INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb MUON_WEIGHT_DECAY=0.10 FREQ_SKIP_GATING=1 TRAIN_BATCH_TOKENS=24576 MLP_MULT_ASYMMETRIC=2,4 GRAD_CLIP_NORM=0.5 LAYER_LR_SCALE=0.5 WARMUP_STEPS=50 QAT_PREWARMDOWN=1 QAT_STRENGTH=0.1 QAT_STOP_LR_MUL=0.8 QAT_EVERY=10 VAL_LOSS_EVERY=0 MAX_WALLCLOCK_SECONDS=0 VAL_BATCH_SIZE=8192
```

| Run | Diff | Steps | Train Loss | Val BPB | Final Int8 BPB | Artifact | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| `research_002_fast_control` | control | 200 | `3.9013` | `2.2485` | `2.26271482` | `9,805,167` | control |
| `research_003_qkgain5` | `QK_GAIN_INIT=5.0` | 200 | `3.8806` | `2.2396` | `2.25437326` | `9,878,059` | smoke win |
| `research_004_qkgain8` | `QK_GAIN_INIT=8.0` | 200 | `3.9412` | `2.2793` | `2.29508829` | `9,877,547` | discard |
| `research_005_qkgain3` | `QK_GAIN_INIT=3.0` | 200 | `3.8348` | `2.2148` | `2.23207368` | `9,863,466` | best smoke |
| `research_006_qkgain4` | `QK_GAIN_INIT=4.0` | 200 | `3.8511` | `2.2245` | `2.23975275` | `9,871,515` | smoke win |
| `research_007_qkgain2` | `QK_GAIN_INIT=2.0` | 200 | `3.8463` | `2.2203` | `2.23547521` | `9,840,264` | smoke win |
| `research_008_medium_control` | control | 700 | `2.8402` at step 600 | `1.6175` | `1.61976100` | `13,060,364` | medium control |
| `research_009_medium_qkgain3` | `QK_GAIN_INIT=3.0` | 700 | `2.8432` at step 600 | `1.6182` | `1.62054350` | `13,065,236` | discard |
| `research_010_polar200` | `USE_POLAR_EXPRESS=1` | 200 | `3.9018` | `2.2495` | `2.26224394` | `9,983,982` | neutral |
| `research_011_dyt200` | `USE_DYT_NORM=1` | 200 | `3.9147` | `2.2542` | `2.26216288` | `10,115,161` | neutral |

## Conclusions

- Do not promote QK gain. It looked strong at 200 steps, but the matched
  700-step A/B regressed by `+0.0007825` BPB and was slower.
- Do not promote Polar Express from this evidence. It is neutral at 200 steps.
- Do not promote DyT from this evidence. It is neutral at 200 steps and adds
  parameters/artifact bytes.
- Do not rerun deep supervision. H100 already killed it.
- The highest-signal next RunPod action is TTT-only ablation on the existing
  corrected checkpoint. If TTT cannot beat `1.20303259`, move to tokenizer/data
  work rather than more SP1024 micro-toggles.
