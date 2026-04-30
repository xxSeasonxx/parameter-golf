"""Regression tests for importing PyTorch modules without touching MLX."""

import builtins
import importlib
import sys


def _clear_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _fail_on_mlx_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlx" or name.startswith("mlx."):
            raise AssertionError(f"unexpected MLX import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_train_gpt_common_import_does_not_import_mlx(monkeypatch):
    _clear_modules("train_gpt_common")
    _fail_on_mlx_import(monkeypatch)

    module = importlib.import_module("train_gpt_common")

    assert hasattr(module, "Hyperparameters")
    assert hasattr(module, "DyTTorch")
    assert hasattr(module, "sim_quant_roundtrip_torch")


def test_train_gpt_import_does_not_import_mlx(monkeypatch):
    _clear_modules("train_gpt_common", "train_gpt")
    _fail_on_mlx_import(monkeypatch)

    module = importlib.import_module("train_gpt")

    assert hasattr(module, "GPT")
    assert hasattr(module, "eval_val")
