"""Regression tests for active project orientation after aggressive prune."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RUNNERS = {"h100_l0_only.sh", "h100_ttt_eval_only.sh"}
RETIRED_RUNNERS = {"h100_next_deep_supervision.sh"}


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _runner_env(path: str) -> dict[str, str]:
    assignments = {}
    assignment_re = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

    for line in _read(path).splitlines():
        effective_line = line.split("#", 1)[0].strip()
        if not effective_line:
            continue
        match = assignment_re.match(effective_line)
        if not match:
            continue
        name, value = match.groups()
        assignments[name] = value.strip().strip("\"'")

    return assignments


def test_orientation_docs_exist_and_name_the_three_project_states():
    expected_markers = {
        "PROJECT.md": [
            "Original challenge",
            "Trusted baseline",
            "Next experiment",
            "Do not use README.md as the active runbook",
        ],
        "BASELINE.md": [
            "87b4a2f",
            "Post-quant exact BPB",
            "1.20303259",
            "TTT BPB",
            "1.2162",
            "EMA(0.997) is not the baseline",
        ],
        "NEXT_EXPERIMENT.md": [
            "H100 Next Experiment",
            "EVAL_ONLY_CHECKPOINT=./final_model.int8.ptz",
            "EVAL_ONLY_SKIP_ROUNDTRIP=1",
            "TTT_LORA_RANK",
            "1.20303259",
        ],
    }

    for rel_path, markers in expected_markers.items():
        path = ROOT / rel_path
        assert path.exists(), f"missing {rel_path}"
        text = path.read_text(encoding="utf-8")
        for marker in markers:
            assert marker in text, f"{rel_path} missing marker: {marker}"


def test_only_active_h100_runners_remain():
    root_runner_names = {p.name for p in ROOT.glob("h100_*.sh")}
    root_runner_names |= {p.name for p in ROOT.glob("runpod_*.sh")}
    assert root_runner_names == ACTIVE_RUNNERS | RETIRED_RUNNERS


def test_active_runners_do_not_enable_killed_or_deprecated_paths():
    banned_assignments = {
        "EMA_DECAY": "0.997",
        "NUM_LAYERS": "13",
        "CALIBRATED_QUANT": "1",
        "USE_DYT_NORM": "1",
        "USE_POLAR_EXPRESS": "1",
    }

    for runner in ACTIVE_RUNNERS:
        text = _read(runner)
        assert "train_gpt_h100.py" not in text

        assignments = _runner_env(runner)
        assert assignments.get("GROW_LAYERS_FROM", "0") == "0"
        for name, banned_value in banned_assignments.items():
            assert assignments.get(name) != banned_value, (
                f"{runner} sets banned assignment {name}={banned_value}"
            )


def test_baseline_runner_is_l0_only_control():
    assignments = _runner_env("h100_l0_only.sh")
    required = {
        "RUN_ID": "${RUN_ID:-h100_l0_only}",
        "NUM_LAYERS": "11",
        "EMA_DECAY": "0",
        "EVAL_STRIDE": "64",
        "USE_POLAR_EXPRESS": "0",
        "USE_DYT_NORM": "0",
        "VAL_LOSS_EVERY": "0",
        "EXPECTED_TRAIN_SHARDS": "195",
    }
    for name, value in required.items():
        assert assignments.get(name) == value
    assert assignments.get("DEEP_SUPERVISION") != "1"


def test_next_experiment_runner_is_eval_only_ttt_ablation():
    assignments = _runner_env("h100_ttt_eval_only.sh")
    required = {
        "NUM_LAYERS": "11",
        "EMA_DECAY": "0",
        "DEEP_SUPERVISION": "0",
        "EVAL_ONLY_CHECKPOINT": "${EVAL_ONLY_CHECKPOINT:-./final_model.int8.ptz}",
        "EVAL_ONLY_SKIP_ROUNDTRIP": "${EVAL_ONLY_SKIP_ROUNDTRIP:-1}",
        "TTT_LORA_RANK": "${TTT_LORA_RANK:-8}",
        "TTT_LORA_LR": "${TTT_LORA_LR:-0.003}",
        "TTT_CHUNK_SIZE": "${TTT_CHUNK_SIZE:-128}",
        "TTT_EPOCHS": "${TTT_EPOCHS:-1}",
        "EVAL_STRIDE": "64",
        "USE_ZSTD": "1",
        "VAL_LOSS_EVERY": "0",
        "EXPECTED_TRAIN_SHARDS": "195",
    }
    for name, value in required.items():
        assert assignments.get(name) == value


def test_deep_supervision_runner_is_retired():
    text = _read("h100_next_deep_supervision.sh")
    assert "is retired" in text
    assert "DEEP_SUPERVISION=1" not in text
    assert "exit 1" in text


def test_retired_deep_supervision_config_is_not_active():
    assignments = _runner_env("h100_next_deep_supervision.sh")
    assert assignments == {}


def test_ttt_eval_runner_keeps_baseline_stack():
    assignments = _runner_env("h100_ttt_eval_only.sh")
    required = {
        "RUN_ID": "${RUN_ID:-h100_ttt_eval_only}",
        "NUM_LAYERS": "11",
        "EMA_DECAY": "0",
        "DEEP_SUPERVISION": "0",
        "EVAL_STRIDE": "64",
        "USE_ZSTD": "1",
        "USE_POLAR_EXPRESS": "0",
        "USE_DYT_NORM": "0",
        "VAL_LOSS_EVERY": "0",
        "EXPECTED_TRAIN_SHARDS": "195",
    }
    for name, value in required.items():
        assert assignments.get(name) == value
