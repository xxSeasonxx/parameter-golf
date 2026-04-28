"""Unit tests for sliding-window evaluation (L0 of H100 sprint)."""

import math

import pytest
import torch
from torch import nn

import train_gpt


class _ConstantLossModel(nn.Module):
    """Returns constant cross-entropy loss for any (x, y) pair."""

    def __init__(self, constant_loss: float = 0.5):
        super().__init__()
        self.constant_loss = constant_loss
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.constant_loss, device=x.device, dtype=torch.float32) + 0 * self._dummy.sum()


def test_sliding_eval_with_stride_equal_seqlen_matches_chunked():
    """With stride == seq_len, sliding-window degenerates to chunked eval."""
    seq_len = 64
    val_tokens = torch.arange(seq_len * 4 + 1, dtype=torch.int32)
    base_bytes_lut = torch.full((1024,), 4, dtype=torch.int16)
    has_leading_space_lut = torch.zeros(1024, dtype=torch.int16)
    is_boundary_token_lut = torch.zeros(1024, dtype=torch.int16)

    args = train_gpt.Hyperparameters
    args.train_seq_len = seq_len
    args.val_batch_size = seq_len * 2
    args.eval_stride = seq_len  # degenerate

    model = _ConstantLossModel(constant_loss=math.log(2.0))
    device = torch.device("cpu")

    val_loss_chunked, bpb_chunked = train_gpt.eval_val(
        args, model, rank=0, world_size=1, device=device, grad_accum_steps=1,
        val_tokens=val_tokens, base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut, is_boundary_token_lut=is_boundary_token_lut,
    )
    val_loss_sw, bpb_sw = train_gpt.eval_val_sliding(
        args, model, rank=0, world_size=1, device=device, grad_accum_steps=1,
        val_tokens=val_tokens, base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut, is_boundary_token_lut=is_boundary_token_lut,
    )

    assert val_loss_sw == pytest.approx(val_loss_chunked, rel=1e-5)
    assert bpb_sw == pytest.approx(bpb_chunked, rel=1e-5)


def test_sliding_eval_constant_loss_gives_expected_bpb():
    """log(2) loss -> 1 bit/token. With 4 bytes/token -> 0.25 BPB."""
    seq_len = 64
    val_tokens = torch.arange(seq_len * 4 + 1, dtype=torch.int32)
    base_bytes_lut = torch.full((1024,), 4, dtype=torch.int16)
    has_leading_space_lut = torch.zeros(1024, dtype=torch.int16)
    is_boundary_token_lut = torch.zeros(1024, dtype=torch.int16)

    args = train_gpt.Hyperparameters
    args.train_seq_len = seq_len
    args.val_batch_size = seq_len * 2
    args.eval_stride = 16

    model = _ConstantLossModel(constant_loss=math.log(2.0))
    device = torch.device("cpu")

    _, bpb = train_gpt.eval_val_sliding(
        args, model, rank=0, world_size=1, device=device, grad_accum_steps=1,
        val_tokens=val_tokens, base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut, is_boundary_token_lut=is_boundary_token_lut,
    )
    assert bpb == pytest.approx(0.25, abs=1e-4)
