from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train_gpt


class _FakeLoRA(nn.Module):
    def __init__(self, bsz: int, model: nn.Module, rank: int):
        super().__init__()
        del bsz, model, rank
        self.param = nn.Parameter(torch.zeros(()))
        self.update_count = 0

    def to(self, device):
        return super().to(device)

    def reset(self):
        self.update_count = 0


class _FakeOptimizer:
    def __init__(self, lora: _FakeLoRA):
        self.lora = lora

    def zero_grad(self):
        if self.lora.param.grad is not None:
            self.lora.param.grad.zero_()

    def step(self):
        self.lora.update_count += 1


class _FakeTTTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(()))

    def forward(self, x, y, lora=None, return_per_token: bool = False):
        del x
        update_count = 0 if lora is None else lora.update_count
        per_token = torch.full_like(y, float(update_count), dtype=torch.float32)
        if lora is not None:
            per_token = per_token + 0.0 * lora.param
        if return_per_token:
            return per_token
        return per_token


def _patch_ttt_dependencies(monkeypatch, tokens):
    monkeypatch.setattr(train_gpt.glob, "glob", lambda _pattern: ["dummy.bin"])
    monkeypatch.setattr(
        train_gpt,
        "load_data_shard",
        lambda _path: torch.tensor(tokens, dtype=torch.uint16),
    )
    monkeypatch.setattr(train_gpt, "BatchedTTTLoRA", _FakeLoRA)
    monkeypatch.setattr(train_gpt, "_build_ttt_optimizer", lambda lora, _args: _FakeOptimizer(lora))


def _args(ttt_epochs: int):
    return SimpleNamespace(
        val_files="dummy",
        beta1=0.9,
        beta2=0.95,
        ttt_lora_lr=0.01,
        ttt_lora_rank=1,
        ttt_chunk_size=2,
        ttt_eval_seq_len=8,
        ttt_batch_size=2,
        ttt_epochs=ttt_epochs,
    )


def _byte_luts(vocab_size: int):
    base_bytes = torch.ones(vocab_size, dtype=torch.int16)
    has_space = torch.zeros(vocab_size, dtype=torch.bool)
    is_boundary = torch.zeros(vocab_size, dtype=torch.bool)
    return base_bytes, has_space, is_boundary


def test_ttt_scores_chunk_before_training_on_that_chunk(monkeypatch):
    _patch_ttt_dependencies(monkeypatch, tokens=[1, 10, 11, 12, 13])
    base, has_space, is_boundary = _byte_luts(16)

    val_loss, _bpb = train_gpt.eval_val_ttt_lora(
        _args(ttt_epochs=1),
        _FakeTTTModel(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        base_bytes_lut=base,
        has_leading_space_lut=has_space,
        is_boundary_token_lut=is_boundary,
    )

    # Four predictions in two chunks of two. The first chunk is scored before
    # the first legal update, then the second chunk sees one previous update.
    assert val_loss == pytest.approx(0.5)


def test_ttt_epochs_only_affect_future_chunks(monkeypatch):
    _patch_ttt_dependencies(monkeypatch, tokens=[1, 10, 11, 12, 13])
    base, has_space, is_boundary = _byte_luts(16)

    val_loss, _bpb = train_gpt.eval_val_ttt_lora(
        _args(ttt_epochs=2),
        _FakeTTTModel(),
        rank=0,
        world_size=1,
        device=torch.device("cpu"),
        base_bytes_lut=base,
        has_leading_space_lut=has_space,
        is_boundary_token_lut=is_boundary,
    )

    # Two legal post-score updates after chunk 0; only chunk 1 observes them.
    assert val_loss == pytest.approx(1.0)


def test_ttt_epochs_must_be_positive(monkeypatch):
    _patch_ttt_dependencies(monkeypatch, tokens=[1, 10, 11])
    base, has_space, is_boundary = _byte_luts(16)

    with pytest.raises(ValueError, match="TTT_EPOCHS"):
        train_gpt.eval_val_ttt_lora(
            _args(ttt_epochs=0),
            _FakeTTTModel(),
            rank=0,
            world_size=1,
            device=torch.device("cpu"),
            base_bytes_lut=base,
            has_leading_space_lut=has_space,
            is_boundary_token_lut=is_boundary,
        )
