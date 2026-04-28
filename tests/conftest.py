"""Shared pytest fixtures for H100 sprint tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed():
    """Each test starts from a fixed seed for reproducibility."""
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)


@pytest.fixture
def small_dim() -> int:
    """Small dim for fast unit tests."""
    return 32


@pytest.fixture
def small_batch() -> int:
    return 2


@pytest.fixture
def small_seq() -> int:
    return 16
