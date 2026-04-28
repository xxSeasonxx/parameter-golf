"""Unit tests for Polar Express NS coefficients (L1')."""

import torch

import train_gpt


def test_polar_express_converges_to_orthogonal_matrix():
    """Polar Express NS should map any matrix toward its nearest orthogonal."""
    g = torch.randn(64, 64)
    out = train_gpt.zeropower_via_newtonschulz5(g, steps=10, use_polar_express=True)
    eye = torch.eye(64, dtype=out.dtype, device=out.device)
    err = (out.float() @ out.float().T - eye).abs().max().item()
    assert err < 0.05


def test_polar_express_off_matches_default():
    """use_polar_express=False (explicit) must match the default-coefficient path."""
    g = torch.randn(32, 32)
    out_default = train_gpt.zeropower_via_newtonschulz5(g, steps=5)
    out_explicit_off = train_gpt.zeropower_via_newtonschulz5(g, steps=5, use_polar_express=False)
    assert torch.allclose(out_default, out_explicit_off)
