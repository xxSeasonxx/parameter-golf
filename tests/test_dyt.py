"""Unit tests for Dynamic Tanh (DyT) — L1'."""

import pytest
import torch

import train_gpt


def test_dyt_output_shape_matches_input():
    layer = train_gpt.DyT(32, alpha_init=0.5)
    x = torch.randn(2, 16, 32)
    assert layer(x).shape == x.shape


def test_dyt_with_alpha_one_gamma_one_beta_zero_is_tanh():
    layer = train_gpt.DyT(16, alpha_init=1.0)
    with torch.no_grad():
        layer.gamma.fill_(1.0)
        layer.beta.fill_(0.0)
    x = torch.randn(1, 4, 16)
    assert torch.allclose(layer(x), torch.tanh(x), atol=1e-5)


def test_dyt_gradients_flow():
    layer = train_gpt.DyT(8, alpha_init=0.5)
    x = torch.randn(1, 4, 8)
    layer(x).sum().backward()
    assert layer.alpha.grad is not None and torch.isfinite(layer.alpha.grad)
    assert layer.gamma.grad is not None and torch.isfinite(layer.gamma.grad).all()
    assert layer.beta.grad is not None and torch.isfinite(layer.beta.grad).all()


def test_dyt_default_alpha_is_paper_default():
    assert train_gpt.DyT(16).alpha.item() == pytest.approx(0.5, abs=1e-6)
