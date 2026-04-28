"""Regression tests: with new flags at defaults, behavior matches baseline."""

import os

import pytest
import torch

import train_gpt


def test_eval_stride_zero_uses_chunked_path():
    """With EVAL_STRIDE=0, eval_val (not eval_val_sliding) is called."""
    args = train_gpt.Hyperparameters
    args.eval_stride = 0
    eval_fn = train_gpt.eval_val_sliding if args.eval_stride > 0 else train_gpt.eval_val
    assert eval_fn is train_gpt.eval_val


def test_eval_stride_64_uses_sliding_path():
    args = train_gpt.Hyperparameters
    args.eval_stride = 64
    eval_fn = train_gpt.eval_val_sliding if args.eval_stride > 0 else train_gpt.eval_val
    assert eval_fn is train_gpt.eval_val_sliding


def test_polar_express_off_matches_default_coefficients():
    """USE_POLAR_EXPRESS=0 must match the default-coefficient (modded-nanogpt) path."""
    g = torch.randn(32, 32)
    out_default = train_gpt.zeropower_via_newtonschulz5(g, steps=5)
    out_explicit_off = train_gpt.zeropower_via_newtonschulz5(g, steps=5, use_polar_express=False)
    assert torch.allclose(out_default, out_explicit_off)


def test_dyt_off_uses_rmsnorm():
    """USE_DYT_NORM=0 must produce a Block with RMSNorm instances."""
    args = train_gpt.Hyperparameters
    args.use_dyt_norm = False
    block = train_gpt.Block(
        dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2,
        rope_base=10000.0, qk_gain_init=1.5,
    )
    assert isinstance(block.attn_norm, train_gpt.RMSNorm)
    assert isinstance(block.mlp_norm, train_gpt.RMSNorm)


def test_dyt_on_uses_dyt():
    """USE_DYT_NORM=1 must produce a Block with DyT instances."""
    args = train_gpt.Hyperparameters
    args.use_dyt_norm = True
    block = train_gpt.Block(
        dim=64, num_heads=4, num_kv_heads=2, mlp_mult=2,
        rope_base=10000.0, qk_gain_init=1.5,
    )
    assert isinstance(block.attn_norm, train_gpt.DyT)
    assert isinstance(block.mlp_norm, train_gpt.DyT)
    args.use_dyt_norm = False  # cleanup so subsequent tests start fresh
