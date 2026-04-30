# Project Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current research workspace into a safe, testable, and unambiguous H100 experiment repo without changing the model recipe or chasing BPB improvements.

**Architecture:** Stabilize from the outside in: first remove credential risk, then make imports/tests reliable, then fix metrics parsing/logging, then remove stale runner ambiguity, then clean repo hygiene and docs. Keep all behavioral changes guarded by tests and avoid refactoring model internals unless needed for testability.

**Tech Stack:** Python 3.12, PyTorch, MLX, pytest, shell scripts, git, conda `openai` env.

---

## File Structure

Primary files to modify:

- `train_gpt_common.py` - make MLX imports lazy so PyTorch tests can import the shared module on machines without a usable Metal device.
- `train_gpt_mlx.py` - adjust MLX imports if the common module exposes lazy MLX helpers through factory functions.
- `train_gpt.py` - make final quantized eval labels match actual compression and expose enough state for analyzer parsing.
- `analyze.py` - parse zstd artifact lines and final eval lines with optional `eval_stride`.
- `test_analyze.py` - add regression coverage for zstd logs and `eval_stride` final metrics.
- `tests/test_import_safety.py` - add import tests proving PyTorch/common imports do not touch MLX.
- `tests/test_runner_targets.py` - add shell-script target tests so active runner scripts do not call deprecated entrypoints.
- `runpod_run1_baseline.sh`, `runpod_run2_capacity.sh`, `runpod_run3_swa.sh` - switch from `train_gpt_h100.py` to `train_gpt.py` and active env var names.
- `train_gpt_h100.py` - mark deprecated with a hard runtime message or leave read-only after scripts stop referencing it.
- `.gitignore` - make ignored/generated directories explicit while allowing canonical `.lab` files to remain tracked.
- `CLAUDE.md` - correct the local command contract from `python3` to `python` inside the `openai` conda env and state the active H100 entrypoint.

Files intentionally not modified in this stabilization pass:

- `records/**` - repo-local rule says not to read or edit other teams' submissions.
- `README.md` leaderboard content - public challenge docs may be upstream-like and not the fork's operational source of truth.
- Model architecture in `GPT`, `Block`, `CausalSelfAttention`, or optimizer math - no recipe changes in this plan.

---

### Task 1: Remove GitHub Token From Remote

**Files:**
- Modify local git config only: `.git/config`
- No source files changed.

- [ ] **Step 1: Rotate the leaked token**

Open GitHub token settings for the account and revoke the token currently embedded in `remote.origin.url`.

The leaked token prefix observed during audit:

```text
ghp_uFrd...
```

Expected result: the old token no longer authenticates for `git fetch`, GitHub API calls, or repository access.

- [ ] **Step 2: Replace remote with SSH URL**

Run:

```bash
git remote set-url origin git@github.com:xxSeasonxx/parameter-golf.git
```

Expected: command exits 0.

- [ ] **Step 3: Verify no token remains in remote config**

Run:

```bash
git config --get remote.origin.url
```

Expected output:

```text
git@github.com:xxSeasonxx/parameter-golf.git
```

- [ ] **Step 4: Verify local git config has no GitHub token**

Run:

```bash
git config --show-origin --get remote.origin.url | grep -E 'ghp_|github_pat_' && exit 1 || exit 0
```

Expected: command exits 0 and prints nothing.

- [ ] **Step 5: Commit**

No commit is needed because this changes only local `.git/config`, which is not versioned.

---

### Task 2: Add Import-Safety Regression Tests

**Files:**
- Create: `tests/test_import_safety.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_import_safety.py` with:

```python
"""Import-safety tests for shared training modules.

PyTorch-only imports must not import MLX. On machines where MLX is installed
but Metal is unavailable, importing mlx.core can abort the interpreter instead
of raising ImportError. These tests block regressions by making MLX imports
raise immediately and asserting common/PyTorch modules still import.
"""

from __future__ import annotations

import builtins
import importlib
import sys


def _block_mlx_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlx" or name.startswith("mlx."):
            raise AssertionError(f"unexpected MLX import while importing {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_train_gpt_common_import_does_not_import_mlx(monkeypatch):
    sys.modules.pop("train_gpt_common", None)
    _block_mlx_imports(monkeypatch)

    module = importlib.import_module("train_gpt_common")

    assert hasattr(module, "Hyperparameters")
    assert hasattr(module, "DyTTorch")
    assert hasattr(module, "sim_quant_roundtrip_torch")


def test_train_gpt_import_does_not_import_mlx(monkeypatch):
    for module_name in ["train_gpt", "train_gpt_common"]:
        sys.modules.pop(module_name, None)
    _block_mlx_imports(monkeypatch)

    module = importlib.import_module("train_gpt")

    assert hasattr(module, "GPT")
    assert hasattr(module, "eval_val")
```

- [ ] **Step 2: Run tests and verify they fail before implementation**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_import_safety.py -q
```

Expected before implementation: FAIL with `AssertionError: unexpected MLX import`.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_import_safety.py
git commit -m "test: capture training import safety"
```

---

### Task 3: Make MLX Imports Lazy in `train_gpt_common.py`

**Files:**
- Modify: `train_gpt_common.py`
- Modify if needed: `train_gpt_mlx.py`
- Test: `tests/test_import_safety.py`

- [ ] **Step 1: Remove top-level MLX import block**

In `train_gpt_common.py`, delete the top-level guarded MLX import block:

```python
try:
    import mlx.core as _mx  # noqa: N816
    _HAS_MLX = True
except ImportError:  # pragma: no cover
    _mx = None
    _HAS_MLX = False
```

Keep the guarded Torch import block unchanged.

- [ ] **Step 2: Replace MLX helper section with lazy local imports**

Replace the current `if _HAS_MLX:` section that defines `sim_quant_roundtrip_mlx` with:

```python
def sim_quant_roundtrip_mlx(w, qmax_val: float = 127.0):
    """MLX equivalent of sim_quant_roundtrip_torch.

    Import MLX lazily because importing mlx.core can abort on hosts without a
    usable Metal device. PyTorch-only tests and scripts must be able to import
    train_gpt_common without touching MLX.
    """
    import mlx.core as mx

    f = w.astype(mx.float32)
    qmax = float(qmax_val)
    if f.ndim == 2:
        row_max = mx.maximum(mx.max(mx.abs(f), axis=1, keepdims=True), 1.0 / qmax)
        scale = row_max / qmax
        q = mx.clip(mx.round(f / scale), -qmax, qmax)
        return (q * scale).astype(w.dtype)
    amax = mx.maximum(mx.max(mx.abs(f)), mx.array(1.0 / qmax))
    scale = amax / qmax
    q = mx.clip(mx.round(f / scale), -qmax, qmax)
    return (q * scale).astype(w.dtype)
```

- [ ] **Step 3: Replace `DyTMLX` with a lazy factory class**

Replace the current `if _HAS_MLX:` block that defines `DyTMLX` with:

```python
class DyTMLX:
    """Lazy MLX DyT factory.

    Calling DyTMLX(dim) returns an mlx.nn.Module instance, but importing this
    symbol does not import mlx. This keeps PyTorch-only imports safe on hosts
    where MLX cannot initialize Metal.
    """

    def __new__(cls, dim: int, alpha_init: float = 0.5):
        import mlx.core as mx
        import mlx.nn as nn

        class _DyTMLX(nn.Module):
            def __init__(self, dim: int, alpha_init: float = 0.5):
                super().__init__()
                self.alpha = mx.array(alpha_init, dtype=mx.float32)
                self.gamma = mx.ones((dim,), dtype=mx.float32)
                self.beta = mx.zeros((dim,), dtype=mx.float32)

            def __call__(self, x):
                return (
                    self.gamma.astype(x.dtype) * mx.tanh(self.alpha.astype(x.dtype) * x)
                    + self.beta.astype(x.dtype)
                )

        return _DyTMLX(dim, alpha_init)
```

- [ ] **Step 4: Run import-safety tests**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_import_safety.py -q
```

Expected: PASS.

- [ ] **Step 5: Run PyTorch-adjacent tests that previously aborted**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_regression.py tests/test_polar_express.py tests/test_dyt.py test_train_gpt_logging.py -q
```

Expected: tests collect and run. If any assertion fails, fix the assertion or implementation in this task before proceeding.

- [ ] **Step 6: Commit**

```bash
git add train_gpt_common.py tests/test_import_safety.py
git commit -m "fix: keep PyTorch imports independent of MLX"
```

---

### Task 4: Add Analyzer Coverage for Zstd and `eval_stride`

**Files:**
- Modify: `test_analyze.py`

- [ ] **Step 1: Add a zstd PyTorch log fixture**

In `test_analyze.py`, add this fixture after `SAMPLE_PYTORCH_LOG`:

```python
SAMPLE_PYTORCH_ZSTD_LOG = """\
feature_flags: ema=off calibrated_quant=off compression=zstd-22 deep_supervision=off layer_growth=off
step:8007/20000 val_loss:2.0850 val_bpb:1.2282 train_time:600001ms step_avg:74.95ms
stopping_early: wallclock_cap train_time:600001ms step:8007/20000
Serialized model int8+zstd-22: 14479660 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zstd-22: 14551234 bytes
final_int8_zstd-22_roundtrip eval_stride:64 val_loss:2.0908 val_bpb:1.2316 eval_time:490000ms
final_int8_zstd-22_roundtrip_exact eval_stride:64 val_loss:2.09081234 val_bpb:1.23163600
final_int8_ttt_lora val_loss:2.0545 val_bpb:1.2102 eval_time:310000ms
"""
```

- [ ] **Step 2: Add failing tests for zstd artifact and final metric parsing**

Add this test class after `TestPyTorchLogParsing`:

```python
class TestPyTorchZstdLogParsing:
    def test_parses_zstd_artifact_size(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        assert parsed["summary"]["artifact_bytes"] == 14479660

    def test_parses_zstd_total_submission_size(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        assert parsed["summary"]["total_submission_bytes"] == 14551234

    def test_parses_eval_stride_final_roundtrip_exact(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        assert parsed["summary"]["final_val_loss"] == 2.09081234
        assert parsed["summary"]["final_val_bpb"] == 1.23163600

    def test_get_val_bpb_still_prefers_ttt(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        assert get_val_bpb(parsed) == 1.2102
```

- [ ] **Step 3: Run zstd parser tests and verify they fail before implementation**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest test_analyze.py::TestPyTorchZstdLogParsing -q
```

Expected before implementation: FAIL because current parser only matches `int8+zlib` artifact lines and final roundtrip lines without `eval_stride`.

- [ ] **Step 4: Commit failing tests**

```bash
git add test_analyze.py
git commit -m "test: cover zstd training log parsing"
```

---

### Task 5: Fix `analyze.py` for Compression-Agnostic Logs

**Files:**
- Modify: `analyze.py`
- Test: `test_analyze.py`

- [ ] **Step 1: Update final exact roundtrip regex**

In `parse_log`, replace the final exact regex block with:

```python
        # Final int8 roundtrip exact (authoritative metric)
        m = re.match(
            r"final_int8_[\w.+-]+_roundtrip_exact(?:\s+eval_stride:\d+)?\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)",
            line,
        )
        if m:
            summary["final_val_loss"] = float(m.group(1))
            summary["final_val_bpb"] = float(m.group(2))
            continue
```

- [ ] **Step 2: Update timed roundtrip regex**

Replace the timed roundtrip regex block with:

```python
        # Final int8 roundtrip (with eval time)
        m = re.match(
            r"final_int8_[\w.+-]+_roundtrip(?:\s+eval_stride:\d+)?\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+eval_time:(\d+)ms",
            line,
        )
        if m:
            summary["roundtrip_val_loss"] = float(m.group(1))
            summary["roundtrip_val_bpb"] = float(m.group(2))
            summary["roundtrip_eval_time_ms"] = int(m.group(3))
            continue
```

- [ ] **Step 3: Update artifact size regexes**

Replace the two zlib-only artifact regexes with:

```python
        # Submission size (int8 + selected compression)
        m = re.match(r"Serialized model int8\+[\w.+-]+:\s*(\d+)\s*bytes", line)
        if m:
            summary["artifact_bytes"] = int(m.group(1))
            continue

        m = re.match(r"Total submission size int8\+[\w.+-]+:\s*(\d+)\s*bytes", line)
        if m:
            summary["total_submission_bytes"] = int(m.group(1))
            continue
```

- [ ] **Step 4: Run analyzer tests**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest test_analyze.py -q
```

Expected: `25 passed` or the current total plus the four new zstd tests.

- [ ] **Step 5: Commit**

```bash
git add analyze.py test_analyze.py
git commit -m "fix: parse compression-agnostic training logs"
```

---

### Task 6: Make `train_gpt.py` Final Eval Labels Match Compression

**Files:**
- Modify: `train_gpt.py`
- Modify: `test_train_gpt_logging.py`

- [ ] **Step 1: Add a helper test for final roundtrip label**

In `test_train_gpt_logging.py`, add:

```python
def test_roundtrip_log_label_uses_compression_name():
    assert tg.final_roundtrip_log_prefix("zstd-22") == "final_int8_zstd-22_roundtrip"
    assert tg.final_roundtrip_log_prefix("zlib-9") == "final_int8_zlib-9_roundtrip"
```

- [ ] **Step 2: Run helper test and verify it fails before implementation**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest test_train_gpt_logging.py::test_roundtrip_log_label_uses_compression_name -q
```

Expected before implementation: FAIL with `AttributeError: module 'train_gpt' has no attribute 'final_roundtrip_log_prefix'`.

- [ ] **Step 3: Add helper function**

In `train_gpt.py`, add after `_get_ds_tap_layers = get_ds_tap_layers`:

```python
def final_roundtrip_log_prefix(compression_name: str) -> str:
    """Return the final eval log prefix for the selected artifact compression."""
    return f"final_int8_{compression_name}_roundtrip"
```

- [ ] **Step 4: Use helper in final eval logging**

In `main()`, before the quantized serialization `if master_process:` block, add:

```python
    compress_name = get_compression_name(args)
```

Inside the existing serialization block, remove the duplicate assignments to `compress_name` in both branches and reuse the variable above.

Replace:

```python
        f"final_int8_zlib_roundtrip eval_stride:{args.eval_stride} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
```

with:

```python
        f"{final_roundtrip_log_prefix(compress_name)} eval_stride:{args.eval_stride} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
```

Replace:

```python
    log0(f"final_int8_zlib_roundtrip_exact eval_stride:{args.eval_stride} val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
```

with:

```python
    log0(f"{final_roundtrip_log_prefix(compress_name)}_exact eval_stride:{args.eval_stride} val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
```

- [ ] **Step 5: Run logging and analyzer tests**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest test_train_gpt_logging.py test_analyze.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add train_gpt.py test_train_gpt_logging.py
git commit -m "fix: label final eval logs by artifact compression"
```

---

### Task 7: Stop Root Runner Scripts From Calling Deprecated H100 Entrypoint

**Files:**
- Create: `tests/test_runner_targets.py`
- Modify: `runpod_run1_baseline.sh`
- Modify: `runpod_run2_capacity.sh`
- Modify: `runpod_run3_swa.sh`

- [ ] **Step 1: Write failing script-target test**

Create `tests/test_runner_targets.py`:

```python
"""Runner script invariants.

Root-level active shell scripts should invoke train_gpt.py, not the deprecated
train_gpt_h100.py path.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_shell_scripts_do_not_call_deprecated_h100_entrypoint():
    offenders = []
    for path in sorted(ROOT.glob("*.sh")):
        text = path.read_text(encoding="utf-8")
        if "train_gpt_h100.py" in text:
            offenders.append(path.name)

    assert offenders == []
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_runner_targets.py -q
```

Expected before implementation: FAIL listing `runpod_run1_baseline.sh`, `runpod_run2_capacity.sh`, and `runpod_run3_swa.sh`.

- [ ] **Step 3: Update `runpod_run1_baseline.sh`**

Replace:

```bash
export COMPRESSION=zlib
export QUANT_BITS=8
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb

torchrun --nproc_per_node=8 train_gpt_h100.py 2>&1 | tee run1_baseline.log
```

with:

```bash
export USE_ZSTD=0
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb

torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee run1_baseline.log
```

- [ ] **Step 4: Update `runpod_run2_capacity.sh`**

Replace the serialization block:

```bash
export COMPRESSION=zstd
export QUANT_BITS=6
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
```

with:

```bash
export USE_ZSTD=1
export ZSTD_LEVEL=22
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
```

Replace the final command:

```bash
torchrun --nproc_per_node=8 train_gpt_h100.py 2>&1 | tee run2_capacity.log
```

with:

```bash
torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee run2_capacity.log
```

- [ ] **Step 5: Update `runpod_run3_swa.sh`**

Replace the serialization block:

```bash
export COMPRESSION=zstd
export QUANT_BITS=8
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
```

with:

```bash
export USE_ZSTD=1
export ZSTD_LEVEL=22
export INT8_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb
```

Replace the final command:

```bash
torchrun --nproc_per_node=8 train_gpt_h100.py 2>&1 | tee run3_swa.log
```

with:

```bash
torchrun --nproc_per_node=8 train_gpt.py 2>&1 | tee run3_swa.log
```

- [ ] **Step 6: Run target test**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_runner_targets.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_runner_targets.py runpod_run1_baseline.sh runpod_run2_capacity.sh runpod_run3_swa.sh
git commit -m "fix: route RunPod scripts to active H100 entrypoint"
```

---

### Task 8: Mark `train_gpt_h100.py` Deprecated Without Breaking Imports

**Files:**
- Modify: `train_gpt_h100.py`

- [ ] **Step 1: Add deprecation header**

At the top of `train_gpt_h100.py`, replace the existing module docstring with:

```python
"""
DEPRECATED: older H100 competition variant.

Use train_gpt.py for all active H100 runs. This file is retained only for
historical comparison against older RunPod logs and should not be used by root
runner scripts.
"""
```

- [ ] **Step 2: Verify no active root script calls it**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest tests/test_runner_targets.py -q
```

Expected: PASS.

- [ ] **Step 3: Compile deprecated file**

Run:

```bash
conda run -n openai --no-capture-output python -m py_compile train_gpt_h100.py
```

Expected: exits 0.

- [ ] **Step 4: Commit**

```bash
git add train_gpt_h100.py
git commit -m "docs: mark legacy H100 script deprecated"
```

---

### Task 9: Fix `.gitignore` Around Generated Research Artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Replace current generated-artifact section**

Replace the current `.gitignore` contents with:

```gitignore
data/tokenizers/
data/datasets/
data/manifest.json
data/docs_selected.jsonl
data/docs_selected.source_manifest.json

__pycache__/
.pytest_cache/
.mypy_cache/
.venv/
.DS_Store

modded-nanogpt/

logs/
results/
runs/
run.log
run_*.log
run_baseline.log
final_model.*
*.ptz
*.npz

.lab/*
!.lab/
!.lab/insights.md
!.lab/ideas_queue.md
!.lab/results.tsv

.claude/
openspec/
~$*.xlsx
```

- [ ] **Step 2: Verify canonical `.lab` files are not ignored**

Run:

```bash
git check-ignore -v .lab/insights.md .lab/ideas_queue.md .lab/results.tsv; test $? -eq 1
```

Expected: command exits 0 because `git check-ignore` returns 1 for not ignored and `test $? -eq 1` validates that.

- [ ] **Step 3: Verify generated outputs are ignored**

Run:

```bash
git check-ignore -v logs/example.txt results/example.log runs/example.log run_999.log final_model.pt .claude/example openspec/example
```

Expected: each path prints a matching `.gitignore` rule.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: clarify generated artifact ignores"
```

---

### Task 10: Untrack Ignored Generated Artifacts But Keep Files Locally

**Files:**
- Git index only.
- Keep working tree files in place.

- [ ] **Step 1: Preview tracked ignored files**

Run:

```bash
git ls-files --ignored --exclude-standard --cached
```

Expected before cleanup: includes ignored tracked files such as `.lab/*` archives or `logs/exp_069_leaky07.txt`.

- [ ] **Step 2: Remove generated artifacts from the index only**

Run:

```bash
git rm --cached -r logs .lab
git add .lab/insights.md .lab/ideas_queue.md .lab/results.tsv
```

Expected: generated files are staged for removal from git, but still exist on disk.

- [ ] **Step 3: Verify canonical `.lab` files are staged as tracked**

Run:

```bash
git status --short .lab/insights.md .lab/ideas_queue.md .lab/results.tsv
```

Expected: the three canonical files are not staged for deletion.

- [ ] **Step 4: Verify ignored tracked files list is empty or only intentionally tracked files remain**

Run:

```bash
git ls-files --ignored --exclude-standard --cached
```

Expected: no `logs/` files and no `.lab/<commit>/` archive files.

- [ ] **Step 5: Commit**

```bash
git add .gitignore .lab/insights.md .lab/ideas_queue.md .lab/results.tsv
git commit -m "chore: untrack generated experiment artifacts"
```

---

### Task 11: Correct Local Command Contract in `CLAUDE.md`

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Replace `python3` with `python` in conda commands**

In `CLAUDE.md`, replace:

```bash
conda run -n openai --no-capture-output python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

with:

```bash
conda run -n openai --no-capture-output python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

Replace:

```bash
conda run -n openai --no-capture-output python3 train_gpt_mlx.py 2>&1 | tee run.log
```

with:

```bash
conda run -n openai --no-capture-output python train_gpt_mlx.py 2>&1 | tee run.log
```

Replace:

```bash
conda run -n openai --no-capture-output python3 analyze.py
```

with:

```bash
conda run -n openai --no-capture-output python analyze.py
```

Replace:

```bash
conda run -n openai --no-capture-output python3 -m pytest test_analyze.py -v
```

with:

```bash
conda run -n openai --no-capture-output python -m pytest test_analyze.py -v
```

- [ ] **Step 2: Clarify active H100 entrypoint**

In the `Files` section, ensure these bullets exist:

```markdown
- **`train_gpt.py`** - H100 active development and active RunPod entrypoint. Trusted baseline result (1.2102 BPB) is at git tag `baseline-3e34098`.
- **`train_gpt_h100.py`** - Deprecated historical H100 variant. Do not use for new runs.
```

- [ ] **Step 3: Run documentation grep**

Run:

```bash
rg -n "conda run -n openai --no-capture-output python3|train_gpt_h100.py.*active|active.*train_gpt_h100.py" CLAUDE.md
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: correct local execution contract"
```

---

### Task 12: Run Focused Verification Suite

**Files:**
- No source edits.

- [ ] **Step 1: Compile main Python files**

Run:

```bash
conda run -n openai --no-capture-output python -m py_compile train_gpt.py train_gpt_mlx.py train_gpt_h100.py train_gpt_common.py analyze.py data/cached_challenge_fineweb.py data/download_hf_docs_and_tokenize.py
```

Expected: exits 0.

- [ ] **Step 2: Run non-H100 test suite**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest test_analyze.py test_train_gpt_logging.py tests/test_import_safety.py tests/test_runner_targets.py tests/test_regression.py tests/test_eval_sliding.py tests/test_polar_express.py tests/test_dyt.py -q
```

Expected: PASS. The exact test count depends on the new tests added above.

- [ ] **Step 3: Run full collection**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest --collect-only -q
```

Expected: collection completes without Python abort.

- [ ] **Step 4: Run full tests if collection succeeds**

Run:

```bash
conda run -n openai --no-capture-output python -m pytest -q
```

Expected: PASS, except tests that explicitly require CUDA/H100 should be skipped or isolated. If `test_h100_smoke.py` is too slow or unsuitable for local CPU/MPS, mark it with an explicit pytest marker in a follow-up task rather than silently ignoring failure.

- [ ] **Step 5: Commit verification-only marker if no files changed**

No commit is needed if verification changes no files.

---

### Task 13: Write Stabilization Summary

**Files:**
- Create: `docs/superpowers/specs/2026-04-29-project-stabilization-summary.md`

- [ ] **Step 1: Create summary doc**

Create `docs/superpowers/specs/2026-04-29-project-stabilization-summary.md`:

```markdown
# Project Stabilization Summary

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

- `python -m py_compile` passed for main Python files.
- Focused pytest suite passed.
- Full pytest collection no longer aborts from MLX import.

## Remaining Risks

- H100 runtime behavior still needs a real 8xH100 smoke run before spending a full experiment budget.
- Historical docs may still mention older plans or old leaderboard numbers; they are history, not active run instructions.
- Large local generated files may still exist on disk after being untracked; this plan intentionally avoids deleting user data.
```

- [ ] **Step 2: Commit summary**

```bash
git add docs/superpowers/specs/2026-04-29-project-stabilization-summary.md
git commit -m "docs: summarize project stabilization"
```

---

## Self-Review

**Spec coverage:**

- Credential leak: Task 1.
- MLX import abort/full pytest collection failure: Tasks 2, 3, 12.
- zstd analyzer mismatch: Tasks 4, 5, 6.
- Deprecated H100 path ambiguity: Tasks 7, 8, 11.
- Repo hygiene/generated artifacts: Tasks 9, 10.
- Local command mismatch (`python3` vs `python`): Task 11.
- Verification reporting: Task 12.
- Final documentation: Task 13.

**Placeholder scan:** This plan intentionally avoids `TBD`, `TODO`, "similar to", and open-ended "add tests" steps. Each code-changing task includes concrete code or exact replacement text.

**Type and naming consistency:** New helper names are `final_roundtrip_log_prefix`, `SAMPLE_PYTORCH_ZSTD_LOG`, and `tests/test_import_safety.py`; later tasks refer to the same names.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-29-project-stabilization.md`. Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.

