# H100 Next Experiment

## Goal

Test whether deep supervision improves the trusted clean 11L H100 baseline without stacking unrelated features.

## Baseline

Compare against `baseline-3e34098` / `clean_11l_no_ema`:

- post-quant BPB: `1.2316`
- TTT BPB: `1.2102`
- artifact: `14.48MB`

## Experiment

Run `h100_next_deep_supervision.sh`.

This starts from the clean 11L baseline stack and adds only:

```bash
DEEP_SUPERVISION=1
DEEP_SUPERVISION_ALPHA=0.05
DEEP_SUPERVISION_LAYERS=3,7
```

It keeps:

```bash
EMA_DECAY=0
USE_ZSTD=1
ZSTD_LEVEL=22
EVAL_STRIDE=64
```

It does not enable DyT, Polar Express, EMA, 13L capacity, calibrated quant, layer growth, or extended TTT.

## Control

Run `h100_l0_only.sh` first if the stride-64 timing budget has not been verified on the current RunPod image.

If stride-64 eval exceeds the evaluation budget, update this file before launching the deep-supervision run. Do not silently change the runner.

## Decision Rule

Promote deep supervision only if:

- TTT BPB improves by at least `0.005` versus `1.2102`, and
- training plus final eval stay inside the challenge budget, and
- artifact remains under `16,000,000` bytes.

Kill or revise deep supervision if:

- TTT BPB regresses, or
- eval exceeds the budget, or
- artifact exceeds `16,000,000` bytes.

Treat the result as neutral if:

- TTT BPB improves by less than `0.005`.

If neutral, the next round should focus on TTT evaluation improvements, not EMA, SWA, current 13L capacity, or progressive growth.
