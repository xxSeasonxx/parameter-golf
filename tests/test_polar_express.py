"""Unit tests for Polar Express NS coefficients (L1')."""

import torch

import train_gpt


def test_polar_express_improves_orthogonality_over_baseline():
    """Polar Express NS should produce a more orthogonal output than baseline NS at the same step count.

    The minimax-optimal coefficients converge faster per iteration. After 10 steps,
    Polar Express err < baseline err on the same input. For random Gaussian inputs,
    neither scheme reaches numerical orthogonality (the input's singular values are
    far from 1 after Frobenius normalization), so we test relative improvement.
    """
    g = torch.randn(64, 64)
    out_pe = train_gpt.zeropower_via_newtonschulz5(g, steps=10, use_polar_express=True)
    out_baseline = train_gpt.zeropower_via_newtonschulz5(g, steps=10, use_polar_express=False)
    eye = torch.eye(64, dtype=out_pe.dtype, device=out_pe.device)
    err_pe = (out_pe.float() @ out_pe.float().T - eye).abs().max().item()
    err_baseline = (out_baseline.float() @ out_baseline.float().T - eye).abs().max().item()
    assert err_pe < err_baseline, f"Polar Express err {err_pe:.4f} should beat baseline {err_baseline:.4f}"
    assert err_pe < 0.5, f"Polar Express err {err_pe:.4f} should be reasonable (sanity)"


def test_polar_express_preserves_shape():
    """Output shape must match input shape (within transposition logic)."""
    for shape in [(32, 32), (16, 64), (64, 16)]:
        g = torch.randn(*shape)
        out = train_gpt.zeropower_via_newtonschulz5(g, steps=5, use_polar_express=True)
        assert out.shape == g.shape, f"shape {out.shape} != {g.shape}"
        assert torch.isfinite(out.float()).all(), "Polar Express produced NaN/Inf"


def test_polar_express_off_matches_default():
    """use_polar_express=False (explicit) must match the default-coefficient path."""
    g = torch.randn(32, 32)
    out_default = train_gpt.zeropower_via_newtonschulz5(g, steps=5)
    out_explicit_off = train_gpt.zeropower_via_newtonschulz5(g, steps=5, use_polar_express=False)
    assert torch.allclose(out_default, out_explicit_off)
