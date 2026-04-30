#!/usr/bin/env python3

from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import train_gpt as tg


def _args(**overrides):
    base = dict(
        ema_decay=0.0,
        calibrated_quant=False,
        use_zstd=False,
        zstd_level=22,
        deep_supervision=False,
        deep_supervision_alpha=0.1,
        deep_supervision_layers="",
        grow_layers_from=0,
        grow_at_wallclock_frac=0.35,
        num_layers=11,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_compression_name_respects_zstd():
    args = _args(use_zstd=True, zstd_level=22)
    assert tg.get_compression_name(args, zstd_available=True) == "zstd-22"


def test_feature_summary_marks_clean_repro_state():
    args = _args()
    lines = tg.describe_feature_flags(args, effective_num_layers=11, zstd_available=True)
    assert lines[0] == "feature_flags: ema=off calibrated_quant=off compression=zlib-9 deep_supervision=off layer_growth=off"


def test_feature_summary_marks_optional_features():
    args = _args(
        ema_decay=0.997,
        calibrated_quant=True,
        use_zstd=True,
        deep_supervision=True,
        deep_supervision_alpha=0.05,
        deep_supervision_layers="3,7",
        grow_layers_from=8,
        grow_at_wallclock_frac=0.35,
        num_layers=13,
    )
    lines = tg.describe_feature_flags(args, effective_num_layers=8, zstd_available=True)
    assert lines[0] == "feature_flags: ema=on calibrated_quant=on compression=zstd-22 deep_supervision=on layer_growth=on"
    assert lines[1] == "deep_supervision:alpha=0.05 tap_layers=[3, 7]"
    assert lines[2] == "layer_growth:start_layers=8 target_layers=13 grow_at_frac=0.350"


def test_final_eval_weight_source():
    assert tg.final_eval_weight_source(False) == "live"
    assert tg.final_eval_weight_source(True) == "ema"


def test_roundtrip_log_label_uses_compression_name():
    assert tg.final_roundtrip_log_prefix("zstd-22") == "final_int8_zstd-22_roundtrip"
    assert tg.final_roundtrip_log_prefix("zlib-9") == "final_int8_zlib-9_roundtrip"


def test_submission_code_bytes_include_train_and_common_files():
    expected = (
        len(Path("train_gpt.py").read_bytes())
        + len(Path("train_gpt_common.py").read_bytes())
    )
    assert tg.compute_submission_code_bytes() == expected


def test_zstd_required_path_fails_when_module_unavailable():
    args = _args(use_zstd=True)
    with pytest.raises(RuntimeError, match="USE_ZSTD=1"):
        tg.require_zstd_if_requested(args, zstd_available=False)


def test_stopping_early_train_time_includes_active_elapsed_segment():
    assert tg.current_train_time_ms(
        accumulated_train_time_ms=1000.0,
        active_segment_start=10.0,
        now=10.25,
    ) == pytest.approx(1250.0)


def test_expected_train_shards_guard_rejects_partial_dataset():
    with pytest.raises(RuntimeError, match="train_shards:80/195"):
        tg.validate_expected_train_shards(actual=80, expected=195)


def test_load_eval_checkpoint_state_raw_state_dict(tmp_path):
    path = tmp_path / "model.pt"
    torch.save({"w": torch.tensor([1.0, 2.0])}, path)
    loaded = tg.load_eval_checkpoint_state(path)
    assert torch.equal(loaded["w"], torch.tensor([1.0, 2.0]))
