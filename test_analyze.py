#!/usr/bin/env python3
"""Unit tests for analyze.py log parsing against real MLX baseline output."""

import json
import os
import tempfile
from pathlib import Path

from analyze import (
    archive_run,
    compute_trajectory_stats,
    get_val_bpb,
    parse_log,
    select_log_path,
)


# Minimal MLX log excerpt (source code dump omitted, just the meaningful parts)
SAMPLE_MLX_LOG = """\
run_id:mlx_smoke
mlx_version:0.31.1
train_loader:shards pattern=./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
val_loader:shards pattern=./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin tokens:62021632
WARNING: train_loader:subset dataset:fineweb10B_sp1024 train_shards:10/195
tokenizer_path:./data/tokenizers/fineweb_1024_bpe.model
model_params:17059912 vocab_size:1024 layers:9 dim:512 heads:8 kv_heads:4 seq_len:1024 tie_embeddings:True
iterations:200 train_batch_tokens:8192 grad_accum_steps:8 microbatch_tokens:1024 microbatch_batch_size:1 val_batch_size:8192 warmup_steps:20 max_wallclock_seconds:600.000
mlx_max_microbatch_tokens:8192
optimizer:muon+adam muon_matrix_params:54 scalar_params:37 embed_lr:0.05 matrix_lr:0.04 scalar_lr:0.04 muon_momentum:0.95 muon_steps:5
val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path=./data/tokenizers/fineweb_1024_bpe.model
compute_dtype:mlx.core.bfloat16 compile:True
dtypes tok_emb:mlx.core.bfloat16 linear_weight:mlx.core.float32 skip_weights:mlx.core.float32
warmup_step:1/20
warmup_step:20/20
step:1/200 train_loss:6.9428 train_time:263ms step_avg:263.06ms tok_s:31142
step:2/200 train_loss:18.7847 train_time:730ms step_avg:365.11ms tok_s:18029
step:3/200 train_loss:14.4104 train_time:1032ms step_avg:343.85ms tok_s:27243
step:4/200 train_loss:10.0896 train_time:1368ms step_avg:342.08ms tok_s:25216
step:5/200 train_loss:7.7972 train_time:1693ms step_avg:338.66ms tok_s:26433
step:6/200 train_loss:6.9897 train_time:2023ms step_avg:337.13ms tok_s:25976
step:7/200 train_loss:6.7241 train_time:2345ms step_avg:334.97ms tok_s:26609
step:8/200 train_loss:6.6770 train_time:2654ms step_avg:331.78ms tok_s:26551
step:9/200 train_loss:6.5220 train_time:2964ms step_avg:329.30ms tok_s:26530
step:10/200 train_loss:6.3753 train_time:3279ms step_avg:327.91ms tok_s:26108
step:200/200 train_loss:3.9161 train_time:63994ms step_avg:319.97ms tok_s:25709
step:200/200 val_loss:4.0703 val_bpb:2.4106 train_time:64008ms step_avg:320.04ms
final_int8_zlib_roundtrip val_loss:4.0707 val_bpb:2.4109 eval_time:506446ms
final_int8_zlib_roundtrip_exact val_loss:4.07065555 val_bpb:2.41087150
"""

# PyTorch-style log (different format, no tok_s, has peak memory + artifact size)
SAMPLE_PYTORCH_LOG = """\
step:0/20000 val_loss:6.9370 val_bpb:4.0978 train_time:0ms step_avg:0.01ms
step:1/20000 train_loss:6.9408 train_time:24ms step_avg:23.99ms
step:2/20000 train_loss:16.8763 train_time:67ms step_avg:33.39ms
step:200/20000 train_loss:2.8041 train_time:8677ms step_avg:43.38ms
step:200/20000 val_loss:2.8397 val_bpb:1.6774 train_time:8699ms step_avg:43.49ms
step:13780/20000 val_loss:2.0606 val_bpb:1.2172 train_time:600038ms step_avg:43.54ms
stopping_early: wallclock_cap train_time:600038ms step:13780/20000
peak memory allocated: 10184 MiB reserved: 10200 MiB
Serialized model int8+zlib: 15815847 bytes (payload:17178912 raw_torch:17224025 payload_ratio:3.91x)
Total submission size int8+zlib: 15863489 bytes
final_int8_zlib_roundtrip val_loss:2.0727 val_bpb:1.2244 eval_time:1401ms
final_int8_zlib_roundtrip_exact val_loss:2.07269931 val_bpb:1.22436570
"""

SAMPLE_PYTORCH_ZSTD_LOG = """\
run_id:h100_l0_only
feature_flags: ema=off calibrated_quant=off compression=zstd-22 deep_supervision=off layer_growth=off
deep_supervision:alpha=0.05 tap_layers=[3, 7]
layer_growth:start_layers=8 target_layers=11 grow_at_frac=0.350
world_size:8 grad_accum_steps:1
sdp_backends:cudnn=False flash=True mem_efficient=False math=False
attention_mode:gqa num_heads:8 num_kv_heads:4
tie_embeddings:True embed_lr:0.05 head_lr:0.0 matrix_lr:0.04 scalar_lr:0.04
seed:1337
step:0/20000 val_loss:6.9370 val_bpb:4.0978 train_time:0ms step_avg:0.01ms
step:1/20000 train_loss:6.9408 train_time:24ms step_avg:23.99ms
step:200/20000 val_loss:2.8397 val_bpb:1.6774 train_time:8699ms step_avg:43.49ms
Serialized model int8+zstd-22: 14479660 bytes
Total submission size int8+zstd-22: 14551234 bytes
final_int8_zstd-22_roundtrip eval_stride:64 val_loss:2.0908 val_bpb:1.2316 eval_time:490000ms
final_int8_zstd-22_roundtrip_exact eval_stride:64 val_loss:2.09081234 val_bpb:1.23163600
final_int8_ttt_lora val_loss:2.0545 val_bpb:1.2102 eval_time:310000ms
"""

SAMPLE_PYTORCH_TTT_REGRESSION_LOG = """\
run_id:h100_l0_only
Serialized model int8+zstd-22: 14184763 bytes
final_int8_zstd-22_roundtrip eval_stride:64 val_loss:2.0313 val_bpb:1.2030 eval_time:82909ms
final_int8_zstd-22_roundtrip_exact eval_stride:64 val_loss:2.03127014 val_bpb:1.20303259
final_int8_ttt_lora val_loss:2.0536 val_bpb:1.2162 eval_time:86924ms
"""

SAMPLE_MLX_ARTIFACT_LOG = """\
run_id:mlx_smoke
step:1/200 train_loss:6.9428 train_time:263ms step_avg:263.06ms tok_s:31142
serialized_model_int8_zlib:13124539 bytes
final_int8_zlib_roundtrip_exact val_loss:2.7402 val_bpb:1.6215
"""


def _write_temp_log(content):
    """Write content to a temp file and return its Path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def test_explicit_log_path_wins(tmp_path):
    explicit = tmp_path / "explicit.log"
    explicit.write_text("run_id:explicit\n")
    runs = tmp_path / "runs"
    runs.mkdir()
    newer = runs / "newer.log"
    newer.write_text("run_id:newer\n")
    assert select_log_path(explicit, root=tmp_path) == explicit


def test_newest_runs_log_selected_when_no_argument(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    older = runs / "older.log"
    newer = runs / "newer.log"
    older.write_text("run_id:older\n")
    newer.write_text("run_id:newer\n")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    assert select_log_path(None, root=tmp_path) == newer


def test_root_run_log_fallback_when_runs_empty(tmp_path):
    root_log = tmp_path / "run.log"
    root_log.write_text("run_id:root\n")
    assert select_log_path(None, root=tmp_path) == root_log


class TestMLXLogParsing:
    def test_parses_train_steps(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        assert parsed is not None
        assert len(parsed["train_steps"]) == 11  # steps 1-10 + step 200

    def test_train_step_fields(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        first = parsed["train_steps"][0]
        assert first["step"] == 1
        assert first["total"] == 200
        assert abs(first["train_loss"] - 6.9428) < 0.001
        assert first["train_time_ms"] == 263
        assert abs(first["step_avg_ms"] - 263.06) < 0.01
        assert first["tok_s"] == 31142

    def test_parses_val_steps(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        assert len(parsed["val_steps"]) == 1
        val = parsed["val_steps"][0]
        assert val["step"] == 200
        assert abs(val["val_loss"] - 4.0703) < 0.001
        assert abs(val["val_bpb"] - 2.4106) < 0.001

    def test_final_roundtrip_exact(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        s = parsed["summary"]
        assert abs(s["final_val_loss"] - 4.07065555) < 1e-6
        assert abs(s["final_val_bpb"] - 2.41087150) < 1e-6

    def test_roundtrip_with_eval_time(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        s = parsed["summary"]
        assert abs(s["roundtrip_val_bpb"] - 2.4109) < 0.001
        assert s["roundtrip_eval_time_ms"] == 506446

    def test_config_lines_captured(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        config = parsed["config_lines"]
        # Should capture config lines, not source code
        assert any("model_params:" in c for c in config)
        assert any("optimizer:" in c for c in config)
        assert any("iterations:" in c for c in config)
        # Should NOT contain python source code
        assert not any("import " in c for c in config)
        assert not any("def " in c for c in config)

    def test_mlx_serialized_artifact_size(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_ARTIFACT_LOG))
        assert parsed["summary"]["artifact_bytes"] == 13124539

    def test_get_val_bpb_prefers_exact(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        bpb = get_val_bpb(parsed)
        # Should return the exact roundtrip value, not the val step value
        assert abs(bpb - 2.41087150) < 1e-6

    def test_trajectory_stats(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        stats = compute_trajectory_stats(parsed)
        assert stats["num_train_steps"] == 11
        assert abs(stats["first_loss"] - 6.9428) < 0.001
        assert abs(stats["final_loss"] - 3.9161) < 0.001
        assert stats["min_val_bpb_step"] == 200

    def test_last_step_has_final_time(self):
        parsed = parse_log(_write_temp_log(SAMPLE_MLX_LOG))
        s = parsed["summary"]
        assert s["final_train_time_ms"] == 63994
        assert s["final_step"] == 200


class TestPyTorchLogParsing:
    def test_parses_without_tok_s(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_LOG))
        assert parsed is not None
        assert len(parsed["train_steps"]) == 3  # steps 1, 2, 200
        # No tok_s field in PyTorch logs
        assert "tok_s" not in parsed["train_steps"][0]

    def test_parses_val_steps(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_LOG))
        assert len(parsed["val_steps"]) == 3  # steps 0, 200, 13780 (roundtrip is in summary)

    def test_stopping_early(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_LOG))
        s = parsed["summary"]
        assert s["stopped_early"] is True
        assert s["final_train_time_ms"] == 600038
        assert s["final_step"] == 13780
        assert s["total_iterations"] == 20000

    def test_peak_memory(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_LOG))
        assert parsed["summary"]["peak_memory_mib"] == 10184

    def test_artifact_size(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_LOG))
        assert parsed["summary"]["artifact_bytes"] == 15815847

    def test_total_submission_size(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_LOG))
        assert parsed["summary"]["total_submission_bytes"] == 15863489

    def test_final_exact_bpb(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_LOG))
        bpb = get_val_bpb(parsed)
        assert abs(bpb - 1.22436570) < 1e-6

    def test_zstd_artifact_size(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        assert parsed["summary"]["artifact_bytes"] == 14479660

    def test_zstd_total_submission_size(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        assert parsed["summary"]["total_submission_bytes"] == 14551234

    def test_zstd_final_roundtrip_exact(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        s = parsed["summary"]
        assert abs(s["final_val_loss"] - 2.09081234) < 1e-8
        assert abs(s["final_val_bpb"] - 1.23163600) < 1e-8

    def test_zstd_get_val_bpb_uses_ttt_when_it_improves(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        assert abs(get_val_bpb(parsed) - 1.2102) < 1e-6

    def test_zstd_get_val_bpb_uses_roundtrip_when_ttt_regresses(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_TTT_REGRESSION_LOG))
        assert abs(get_val_bpb(parsed) - 1.20303259) < 1e-8

    def test_active_feature_config_lines_captured(self):
        parsed = parse_log(_write_temp_log(SAMPLE_PYTORCH_ZSTD_LOG))
        config = parsed["config_lines"]
        for expected in [
            "feature_flags:",
            "deep_supervision:",
            "layer_growth:",
            "world_size:",
            "sdp_backends:",
            "attention_mode:",
            "tie_embeddings:",
            "seed:",
        ]:
            assert any(line.startswith(expected) for line in config)


def test_repeated_same_commit_archives_do_not_collide(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text("run_id:h100_l0_only\nstep:1/10 train_loss:1.0 train_time:1ms step_avg:1.0ms\n")
    script_path = tmp_path / "train_gpt_mlx.py"
    script_path.write_text("# script\n")
    parsed = parse_log(log_path)
    lab_dir = tmp_path / ".lab"

    first = archive_run("abc1234", parsed, log_path=log_path, train_script=script_path, lab_dir=lab_dir)
    second = archive_run("abc1234", parsed, log_path=log_path, train_script=script_path, lab_dir=lab_dir)

    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.name.startswith("abc1234_h100_l0_only_")
    assert second.name.startswith("abc1234_h100_l0_only_")


class TestEdgeCases:
    def test_empty_log(self):
        parsed = parse_log(_write_temp_log(""))
        assert parsed is not None
        assert len(parsed["train_steps"]) == 0
        assert len(parsed["val_steps"]) == 0

    def test_nonexistent_file(self):
        result = parse_log(Path("/nonexistent/path.log"))
        assert result is None

    def test_get_val_bpb_no_data(self):
        parsed = parse_log(_write_temp_log("just some random text\n"))
        assert get_val_bpb(parsed) is None

    def test_trajectory_stats_empty(self):
        parsed = parse_log(_write_temp_log(""))
        stats = compute_trajectory_stats(parsed)
        assert stats == {}


class TestWithRealBaseline:
    """Test against the actual baseline log if it exists."""

    LOG_PATH = Path("logs/mlx_smoke.txt")

    def test_real_baseline_parses(self):
        if not self.LOG_PATH.exists():
            return  # skip if log not available
        parsed = parse_log(self.LOG_PATH)
        assert parsed is not None
        # Should have train steps
        assert len(parsed["train_steps"]) > 0
        # Should have the final roundtrip metric
        bpb = get_val_bpb(parsed)
        assert bpb is not None
        assert abs(bpb - 2.41087150) < 0.001
        # Should parse config lines
        assert len(parsed["config_lines"]) > 0
        # Config should NOT contain 1000+ lines of source code
        assert len(parsed["config_lines"]) < 20


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
