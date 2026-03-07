"""Tests for training config loading."""

from pathlib import Path

from lale.train.train import load_config


def test_load_default_config():
    config = load_config(Path("lale/train/config.yaml"))
    assert config.lora.r == 64
    assert config.lora.alpha == 128
    assert config.training.epochs == 3
    assert config.training.bf16 is True
    assert config.data.eval_split == 0.02
    assert "lale" in config.output.hub_model_id
