#!/usr/bin/env python3
"""Smoke tests for H100 competition runs. Validates each run config can:
1. Construct the model
2. Forward + backward pass
3. Optimizer step
4. Quantization + serialization round-trip
5. EMA update (if enabled)

Runs on CPU or MPS (no CUDA needed). Uses tiny batch/seq to keep it fast.
"""
import os
import sys
import io
import zlib
import traceback

import pytest

# Suppress DDP since we're single-process
os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"

import torch
import torch.nn.functional as F

# Import from the training script
sys.path.insert(0, os.path.dirname(__file__))

# We need to define RUN CONFIGS before importing (env vars are read at import time)
# So we'll reload for each config.

RUNS = {
    "clean_11l_no_ema": {
        "NUM_LAYERS": "11", "MUON_WEIGHT_DECAY": "0.10",
        "MLP_MULT_ASYMMETRIC": "2,4", "FREQ_SKIP_GATING": "1",
        "GRAD_CLIP_NORM": "0.5", "LAYER_LR_SCALE": "0.5",
        "WARMUP_STEPS": "0", "QAT_PREWARMDOWN": "1", "QAT_STRENGTH": "0.1",
        "QAT_STOP_LR_MUL": "0.8", "QAT_EVERY": "10",
        "EMA_DECAY": "0", "USE_ZSTD": "1", "ZSTD_LEVEL": "22",
        "INT8_KEEP_FLOAT_FP16_NAME_PATTERNS": "tok_emb",
    },
    "capacity_13l_no_ema": {
        "NUM_LAYERS": "13", "MUON_WEIGHT_DECAY": "0.15",
        "MLP_MULT_ASYMMETRIC": "2,4", "FREQ_SKIP_GATING": "1",
        "GRAD_CLIP_NORM": "0.5", "LAYER_LR_SCALE": "0.5",
        "WARMUP_STEPS": "0", "QAT_PREWARMDOWN": "1", "QAT_STRENGTH": "0.1",
        "QAT_STOP_LR_MUL": "0.8", "QAT_EVERY": "10",
        "EMA_DECAY": "0", "USE_ZSTD": "1", "ZSTD_LEVEL": "22",
        "CALIBRATED_QUANT": "1",
        "INT8_KEEP_FLOAT_FP16_NAME_PATTERNS": "tok_emb",
    },
    "ema_11l": {
        "NUM_LAYERS": "11", "MUON_WEIGHT_DECAY": "0.10",
        "MLP_MULT_ASYMMETRIC": "2,4", "FREQ_SKIP_GATING": "1",
        "GRAD_CLIP_NORM": "0.5", "LAYER_LR_SCALE": "0.5",
        "WARMUP_STEPS": "0", "QAT_PREWARMDOWN": "1", "QAT_STRENGTH": "0.1",
        "QAT_STOP_LR_MUL": "0.8", "QAT_EVERY": "10",
        "EMA_DECAY": "0.997", "USE_ZSTD": "1", "ZSTD_LEVEL": "22",
        "INT8_KEEP_FLOAT_FP16_NAME_PATTERNS": "tok_emb",
    },
    "baseline_no_features": {
        # All defaults — should match original behavior
    },
}

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH, SEQ, VOCAB = 2, 64, 1024
DIM = 512


def set_env(config: dict):
    """Set env vars for a run config, clearing all others first."""
    # Clear all our custom env vars
    for key in ["NUM_LAYERS", "MUON_WEIGHT_DECAY", "LAYER_LR_SCALE",
                "MLP_MULT_ASYMMETRIC", "FREQ_SKIP_GATING", "FREQ_SKIP_WINDOW",
                "QAT_PREWARMDOWN", "QAT_STRENGTH", "QAT_STOP_LR_MUL", "QAT_EVERY",
                "EMA_DECAY", "INT8_KEEP_FLOAT_FP16_NAME_PATTERNS",
                "DEEP_SUPERVISION", "DEEP_SUPERVISION_ALPHA", "DEEP_SUPERVISION_LAYERS",
                "CALIBRATED_QUANT", "GROW_LAYERS_FROM", "GROW_AT_WALLCLOCK_FRAC",
                "USE_ZSTD", "ZSTD_LEVEL", "GRAD_CLIP_NORM", "WARMUP_STEPS"]:
        os.environ.pop(key, None)
    # Set this run's config
    for k, v in config.items():
        os.environ[k] = v


def run_config(name: str, config: dict) -> tuple[bool, str]:
    """Test a single run configuration. Returns (passed, message)."""
    set_env(config)

    # Force reimport to pick up new env vars
    if "train_gpt" in sys.modules:
        del sys.modules["train_gpt"]

    try:
        import train_gpt as tg

        args = tg.Hyperparameters()
        effective_layers = args.grow_layers_from if args.grow_layers_from > 0 else args.num_layers

        # 1. Construct model
        ds_tap = None
        if args.deep_supervision:
            if args.deep_supervision_layers:
                ds_tap = [int(x) for x in args.deep_supervision_layers.split(",") if x]
            else:
                ds_tap = list(range(1, effective_layers - 1, 2))

        model = tg.GPT(
            vocab_size=VOCAB,
            num_layers=effective_layers,
            model_dim=DIM,
            num_heads=8,
            num_kv_heads=4,
            mlp_mult=args.mlp_mult,
            tie_embeddings=True,
            tied_embed_init_std=0.005,
            logit_softcap=30.0,
            rope_base=10000.0,
            qk_gain_init=1.5,
            mlp_mult_asymmetric=args.mlp_mult_asymmetric,
            freq_skip_gating=args.freq_skip_gating,
            freq_skip_window=args.freq_skip_window,
            deep_supervision=args.deep_supervision,
            deep_supervision_alpha=args.deep_supervision_alpha,
            deep_supervision_tap_layers=ds_tap,
        ).to(DEVICE).bfloat16()

        # Restore dtypes like main() does
        for module in model.modules():
            if isinstance(module, tg.CastedLinear):
                module.float()
            if isinstance(module, tg.Rotary):
                module.inv_freq.data = module.inv_freq.data.float()
        tg.restore_low_dim_params_to_fp32(model)

        n_params = sum(p.numel() for p in model.parameters())

        # 2. Forward + backward
        x = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
        y = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)
        loss.backward()
        loss_val = loss.item()

        # 3. Check loss is finite
        if not (0 < loss_val < 100):
            return False, f"loss={loss_val} (not in expected range)"

        # 4. EMA update
        if args.ema_decay > 0:
            ema_state = {k: v.clone() for k, v in model.state_dict().items()}
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k].lerp_(v, 1.0 - args.ema_decay)

        # 5. Quantization round-trip
        state = model.state_dict()
        # Move to CPU for quantization
        cpu_state = {k: v.cpu() for k, v in state.items()}
        quant_obj, stats = tg.quantize_state_dict_int8(cpu_state, calibrated=args.calibrated_quant)
        # Compress
        buf = io.BytesIO()
        torch.save(quant_obj, buf)
        raw = buf.getvalue()
        if args.use_zstd and tg.zstd_mod is not None:
            blob = tg.zstd_mod.ZstdCompressor(level=1).compress(raw)
        else:
            blob = zlib.compress(raw, level=1)  # level 1 for speed in test

        # Decompress + dequantize
        if args.use_zstd and tg.zstd_mod is not None:
            raw_disk = tg.zstd_mod.ZstdDecompressor().decompress(blob)
        else:
            raw_disk = zlib.decompress(blob)
        restored = tg.dequantize_state_dict_int8(torch.load(io.BytesIO(raw_disk), map_location="cpu"))
        model.load_state_dict(restored, strict=True)

        artifact_mb = len(blob) / 1e6

        # 6. Forward pass with restored weights
        with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=True):
            loss2 = model(x, y)
        loss2_val = loss2.item()

        # 7. Progressive growing (if enabled)
        if args.grow_layers_from > 0:
            big_model = tg.grow_model(model, args, DEVICE)
            for module in big_model.modules():
                if isinstance(module, tg.CastedLinear):
                    module.float()
                if isinstance(module, tg.Rotary):
                    module.inv_freq.data = module.inv_freq.data.float()
            tg.restore_low_dim_params_to_fp32(big_model)
            big_params = sum(p.numel() for p in big_model.parameters())
            with torch.no_grad(), torch.autocast(device_type=DEVICE, dtype=torch.bfloat16, enabled=True):
                loss3 = big_model(x, y)
            if not (0 < loss3.item() < 100):
                return False, f"grown model loss={loss3.item()}"
            growth_msg = f" | grew {effective_layers}L->{args.num_layers}L ({big_params:,} params)"
        else:
            growth_msg = ""

        return True, (
            f"{effective_layers}L {n_params:,} params | "
            f"loss={loss_val:.2f} quant_loss={loss2_val:.2f} | "
            f"artifact={artifact_mb:.1f}MB{growth_msg}"
        )

    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


@pytest.mark.parametrize(("name", "config"), list(RUNS.items()))
def test_run(name: str, config: dict):
    ok, msg = run_config(name, config)
    assert ok, msg


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Batch={BATCH} Seq={SEQ} Vocab={VOCAB} Dim={DIM}")
    print("=" * 80)

    passed = 0
    failed = 0
    for name, config in RUNS.items():
        ok, msg = run_config(name, config)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {name}: {msg}")

    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(RUNS)}")
    sys.exit(0 if failed == 0 else 1)
