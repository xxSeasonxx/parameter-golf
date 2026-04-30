"""Tests for root-level shell runner targets."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_shell_scripts_do_not_call_deprecated_h100_entrypoint():
    offenders = []
    for script_path in ROOT.glob("*.sh"):
        if "train_gpt_h100.py" in script_path.read_text(encoding="utf-8"):
            offenders.append(script_path.name)

    assert offenders == []
