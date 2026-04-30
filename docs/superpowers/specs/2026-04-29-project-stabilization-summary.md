# Project Stabilization Summary

Implementation plan: `docs/superpowers/plans/2026-04-29-project-stabilization.md`.

## What Changed

- Removed credential-bearing Git remote from local config and documented token rotation.
- Made `train_gpt_common.py` safe to import from PyTorch-only paths without importing MLX.
- Made analyzer parsing compression-agnostic for zstd and zlib final artifacts.
- Made final quantized eval log labels reflect the actual compression algorithm.
- Routed active RunPod scripts to `train_gpt.py` and marked `train_gpt_h100.py` deprecated.
- Clarified ignored/generated artifact rules and kept only canonical `.lab` state tracked.
- Updated local command docs to use `conda run -n openai --no-capture-output python`.

## What Did Not Change

- No model architecture recipe changed.
- No optimizer math changed.
- No BPB claims changed.
- No `records/**` submission code was inspected or edited.

## Verified

- `conda run -n openai --no-capture-output python -m py_compile train_gpt.py train_gpt_mlx.py train_gpt_h100.py train_gpt_common.py analyze.py data/cached_challenge_fineweb.py data/download_hf_docs_and_tokenize.py` passed.
- Focused pytest suite passed: 48 passed, 3 warnings.
- Full pytest collection completed without MLX import abort: 52 tests collected, 2 warnings.
- Full pytest suite passed locally: 52 passed, 3 warnings.
- Local `remote.origin.url` is `git@github.com:xxSeasonxx/parameter-golf.git`.

## Remaining Risks

- GitHub-side token rotation cannot be verified locally; confirm the leaked `ghp_...` token has been revoked before relying on repository credential safety.
- H100 runtime behavior still needs a real 8xH100 smoke run before spending a full experiment budget.
- Historical docs may still mention older plans or old leaderboard numbers; they are history, not active run instructions.
- Large local generated files may still exist on disk after being untracked; this stabilization intentionally avoids deleting user data.
