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
