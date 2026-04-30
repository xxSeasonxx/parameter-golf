"""Unit tests for PyTorch sliding-window validation accounting."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import train_gpt


class _GlobalPositionLossModel(nn.Module):
    """Per-token loss equals the global prediction index.

    Tests use ``val_tokens = arange(...)``. For next-token prediction at global
    prediction index ``p``, the target token id is ``p + 1``. Returning
    ``y - 1`` makes the loss expose exactly which global targets were scored.
    """

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lora=None,
        return_per_token: bool = False,
    ) -> torch.Tensor:
        per_token = y.to(torch.float32) - 1.0 + 0 * self._dummy
        if return_per_token:
            return per_token
        return per_token.mean()


class _ConstantLossModel(nn.Module):
    def __init__(self, loss: float):
        super().__init__()
        self.loss = loss
        self._dummy = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lora=None,
        return_per_token: bool = False,
    ) -> torch.Tensor:
        per_token = torch.full_like(y, self.loss, dtype=torch.float32) + 0 * self._dummy
        if return_per_token:
            return per_token
        return per_token.mean()


def _common_luts(vocab: int = 1024, bytes_per_token: int = 1):
    base = torch.full((vocab,), bytes_per_token, dtype=torch.int16)
    has_space = torch.zeros(vocab, dtype=torch.bool)
    is_boundary = torch.zeros(vocab, dtype=torch.bool)
    return base, has_space, is_boundary


def _args(seq_len: int, stride: int, val_batch_size: int | None = None):
    args = train_gpt.Hyperparameters
    args.train_seq_len = seq_len
    args.eval_stride = stride
    args.val_batch_size = val_batch_size or seq_len * 4
    return args


def test_sliding_eval_stride_equal_seqlen_matches_global_chunked_mean():
    seq_len = 8
    n_pred = seq_len * 4
    val_tokens = torch.arange(n_pred + 1, dtype=torch.int32)
    base, hs, ib = _common_luts(vocab=n_pred + 2)
    args = _args(seq_len=seq_len, stride=seq_len)
    model = _GlobalPositionLossModel()
    device = torch.device("cpu")

    chunked_loss, chunked_bpb = train_gpt.eval_val(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base,
        has_leading_space_lut=hs,
        is_boundary_token_lut=ib,
    )
    sliding_loss, sliding_bpb = train_gpt.eval_val_sliding(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base,
        has_leading_space_lut=hs,
        is_boundary_token_lut=ib,
    )

    expected_loss = sum(range(n_pred)) / n_pred
    assert chunked_loss == pytest.approx(expected_loss, rel=1e-6)
    assert sliding_loss == pytest.approx(chunked_loss, rel=1e-6)
    assert sliding_bpb == pytest.approx(chunked_bpb, rel=1e-6)


def test_sliding_eval_stride_less_than_seqlen_scores_each_prediction_once():
    seq_len = 8
    stride = 4
    n_pred = 20
    val_tokens = torch.arange(n_pred + 1, dtype=torch.int32)
    base, hs, ib = _common_luts(vocab=n_pred + 2)
    args = _args(seq_len=seq_len, stride=stride)
    model = _GlobalPositionLossModel()
    device = torch.device("cpu")

    val_loss, _bpb = train_gpt.eval_val_sliding(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base,
        has_leading_space_lut=hs,
        is_boundary_token_lut=ib,
    )

    expected_mean = sum(range(n_pred)) / n_pred
    assert val_loss == pytest.approx(expected_mean, rel=1e-6)


def test_sliding_eval_byte_count_uses_the_same_scored_targets():
    seq_len = 8
    stride = 4
    n_pred = 20
    val_tokens = torch.arange(n_pred + 1, dtype=torch.int32)
    base = torch.arange(n_pred + 2, dtype=torch.int16)
    has_space = torch.zeros(n_pred + 2, dtype=torch.bool)
    is_boundary = torch.zeros(n_pred + 2, dtype=torch.bool)
    args = _args(seq_len=seq_len, stride=stride)
    model = _ConstantLossModel(loss=math.log(2.0))
    device = torch.device("cpu")

    val_loss, bpb = train_gpt.eval_val_sliding(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=1,
        val_tokens=val_tokens,
        base_bytes_lut=base,
        has_leading_space_lut=has_space,
        is_boundary_token_lut=is_boundary,
    )

    expected_bytes = sum(range(1, n_pred + 1))
    assert val_loss == pytest.approx(math.log(2.0), rel=1e-6)
    assert bpb == pytest.approx(n_pred / expected_bytes, rel=1e-6)
