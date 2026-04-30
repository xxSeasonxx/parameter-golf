"""Unit tests for sliding-window evaluation (L0 of H100 sprint).

Important: tests use POSITION-DEPENDENT loss (not constant). Constant-loss
tests pass even when the eval is mathematically wrong (mean-of-all-positions
scaled by scored-count instead of mean-of-scored-positions). The first
sliding-window port shipped a bug exactly because constant-loss tests gave
false confidence.
"""

import math

import pytest
import torch
from torch import nn

import train_gpt


class _ConstantLossModel(nn.Module):
    """Returns a constant per-token loss tensor of shape (B, seq_len) when
    return_per_token=True, scalar mean when False. Used for the
    stride==seq_len degenerate-case sanity check ONLY.
    """

    def __init__(self, constant_loss: float = 0.5):
        super().__init__()
        self.constant_loss = constant_loss
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, lora=None,
                return_per_token: bool = False) -> torch.Tensor:
        if return_per_token:
            B, S = x.shape
            return torch.full((B, S), self.constant_loss,
                              device=x.device, dtype=torch.float32) + 0 * self._dummy.sum()
        return torch.tensor(self.constant_loss, device=x.device, dtype=torch.float32) + 0 * self._dummy.sum()


class _PositionDependentLossModel(nn.Module):
    """Returns per-token loss[b, t] = float(t).

    Lets us verify exactly which positions get scored. With this model,
    if a window scores positions [a:a+n], the contributed loss is
    sum(t for t in range(a, a+n)) = (a + a+n-1) * n / 2.

    Any bug that scores the wrong positions (e.g., the previous bug that
    used scalar mean × n_scored) gives a numerically different answer.
    """

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, y: torch.Tensor, lora=None,
                return_per_token: bool = False) -> torch.Tensor:
        B, S = x.shape
        per_token = torch.arange(S, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, S)
        per_token = per_token + 0 * self._dummy.sum()
        if return_per_token:
            return per_token
        return per_token.mean()


def _common_luts(vocab: int = 1024, bytes_per_tok: int = 4):
    base = torch.full((vocab,), bytes_per_tok, dtype=torch.int16)
    has_space = torch.zeros(vocab, dtype=torch.int16)
    is_boundary = torch.zeros(vocab, dtype=torch.int16)
    return base, has_space, is_boundary


def test_sliding_eval_with_stride_equal_seqlen_matches_chunked():
    """With stride == seq_len, sliding-window degenerates to chunked eval."""
    seq_len = 64
    val_tokens = torch.arange(seq_len * 4 + 1, dtype=torch.int32)
    base, hs, ib = _common_luts()

    args = train_gpt.Hyperparameters
    args.train_seq_len = seq_len
    args.val_batch_size = seq_len * 2
    args.eval_stride = seq_len  # degenerate

    model = _ConstantLossModel(constant_loss=math.log(2.0))
    device = torch.device("cpu")

    val_loss_chunked, bpb_chunked = train_gpt.eval_val(
        args, model, rank=0, world_size=1, device=device, grad_accum_steps=1,
        val_tokens=val_tokens, base_bytes_lut=base,
        has_leading_space_lut=hs, is_boundary_token_lut=ib,
    )
    val_loss_sw, bpb_sw = train_gpt.eval_val_sliding(
        args, model, rank=0, world_size=1, device=device, grad_accum_steps=1,
        val_tokens=val_tokens, base_bytes_lut=base,
        has_leading_space_lut=hs, is_boundary_token_lut=ib,
    )

    assert val_loss_sw == pytest.approx(val_loss_chunked, rel=1e-3)
    assert bpb_sw == pytest.approx(bpb_chunked, rel=1e-3)


def test_sliding_eval_scores_correct_positions_with_position_dependent_loss():
    """Position-dependent loss model: loss[t] = t. Verify the sum we
    compute matches the analytic sum over the SCORED positions only.

    With seq_len=8, stride=4, val_tokens length 17 (= 8 + 4*2 + 1):
    - Window 0: ws=0, score_start=0, n_scored=7. Scored loss positions
      in row 0 of per_token: [0, 1, 2, 3, 4, 5, 6]. Sum = 21.
    - Window 1: ws=4, score_start=11, n_scored=4. Local positions: [7..10] →
      but row only has 8 positions. local_score_start = 11 - 4 = 7.
      So row 1 positions [7], n_scored capped to 1 (per the impl). Sum = 7.
    - Window 2: ws=8, score_start=15, n_scored=4. local = 7. Sum = 7.

    Adjust expectations dynamically by replicating the window plan rather
    than hard-coding (the plan is part of what we want to validate).
    """
    seq_len = 8
    stride = 4
    n_tokens_total = 17
    val_tokens = torch.arange(n_tokens_total, dtype=torch.int32)
    base, hs, ib = _common_luts(vocab=n_tokens_total + 16)

    args = train_gpt.Hyperparameters
    args.train_seq_len = seq_len
    args.val_batch_size = seq_len * 4
    args.eval_stride = stride

    model = _PositionDependentLossModel()
    device = torch.device("cpu")

    val_loss, _bpb = train_gpt.eval_val_sliding(
        args, model, rank=0, world_size=1, device=device, grad_accum_steps=1,
        val_tokens=val_tokens, base_bytes_lut=base,
        has_leading_space_lut=hs, is_boundary_token_lut=ib,
    )

    # Recompute the expected sum by replaying the window plan.
    n_pred = n_tokens_total - 1
    expected_loss_sum = 0.0
    expected_count = 0
    first_n = min(seq_len - 1, n_pred)
    expected_count += first_n
    for t in range(0, first_n):
        expected_loss_sum += float(t)
    pos = seq_len - 1
    while pos < n_pred:
        win_end = min(pos + stride, n_pred)
        win_start = max(win_end - (seq_len - 1), 0)
        n_scored = min(stride, win_end - win_start)
        local_score_start = (win_end - n_scored) - win_start
        for k in range(n_scored):
            t = local_score_start + k
            expected_loss_sum += float(t)
        expected_count += n_scored
        pos = win_end

    expected_mean_loss = expected_loss_sum / expected_count
    assert val_loss == pytest.approx(expected_mean_loss, rel=1e-5), (
        f"sliding eval scored wrong positions or weighted wrong: "
        f"got {val_loss}, expected {expected_mean_loss}"
    )


def test_sliding_eval_old_buggy_calculation_would_fail_this_test():
    """Sanity check: the OLD (buggy) calc — mean-of-all-positions scaled by
    scored-count — produces a clearly different number than the correct one
    on this position-dependent model. Documents the bug shape.

    With seq_len=8, stride=4, n_tokens_total=17:
    The correct mean is the analytic sum of scored positions / scored count.
    The buggy version was: per-window mean(all 8 positions) × scored_count.
    For the position-dependent model, mean(0..7) = 3.5 across all windows,
    so buggy answer is always 3.5. Correct answer is different.
    """
    seq_len = 8
    stride = 4
    n_tokens_total = 17
    val_tokens = torch.arange(n_tokens_total, dtype=torch.int32)
    base, hs, ib = _common_luts(vocab=n_tokens_total + 16)

    args = train_gpt.Hyperparameters
    args.train_seq_len = seq_len
    args.val_batch_size = seq_len * 4
    args.eval_stride = stride

    model = _PositionDependentLossModel()
    device = torch.device("cpu")

    val_loss, _ = train_gpt.eval_val_sliding(
        args, model, rank=0, world_size=1, device=device, grad_accum_steps=1,
        val_tokens=val_tokens, base_bytes_lut=base,
        has_leading_space_lut=hs, is_boundary_token_lut=ib,
    )
    # Buggy answer would be ~mean(0..seq_len-1) = 3.5 regardless of stride.
    # Correct answer must NOT be 3.5 (since first window scores [0..6], later windows score later positions).
    assert val_loss != pytest.approx(3.5, abs=1e-3), (
        f"val_loss == 3.5 implies the old buggy mean×count calculation is back!"
    )
