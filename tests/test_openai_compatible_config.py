"""Tests for OpenAI-compatible backend configuration pipeline."""

from __future__ import annotations

import tempfile
from unittest import mock

import pytest
import yaml

import scripts.eval_only as eval_only_script
import scripts.train as train_script
from skillopt.config import flatten_config, load_config


def test_flatten_config_maps_openai_compatible_keys() -> None:
    structured_cfg = {
        "model": {
            "backend": "openai_compatible",
            "openai_compatible_base_url": "https://api.deepseek.com/v1",
            "openai_compatible_api_key": "sk-deepseek-test",
            "openai_compatible_model": "deepseek-chat",
            "openai_compatible_temperature": 0.3,
            "openai_compatible_timeout_seconds": 180.0,
            "openai_compatible_max_tokens": 4096,
            "optimizer_openai_compatible_base_url": "https://api.together.xyz/v1",
            "optimizer_openai_compatible_api_key": "sk-together-test",
            "optimizer_openai_compatible_model": "together-model",
            "optimizer_openai_compatible_temperature": 0.1,
            "optimizer_openai_compatible_timeout_seconds": 240.0,
            "optimizer_openai_compatible_max_tokens": 8192,
            "target_openai_compatible_base_url": "https://api.groq.com/openai/v1",
            "target_openai_compatible_api_key": "sk-groq-test",
            "target_openai_compatible_model": "groq-model",
            "target_openai_compatible_temperature": 0.7,
            "target_openai_compatible_timeout_seconds": 90.0,
            "target_openai_compatible_max_tokens": 2048,
        },
        "train": {"num_epochs": 1},
        "env": {"name": "searchqa"},
    }

    flat = flatten_config(structured_cfg)

    assert flat["model_backend"] == "openai_compatible"
    assert flat["openai_compatible_base_url"] == "https://api.deepseek.com/v1"
    assert flat["openai_compatible_api_key"] == "sk-deepseek-test"
    assert flat["openai_compatible_model"] == "deepseek-chat"
    assert flat["openai_compatible_temperature"] == 0.3
    assert flat["openai_compatible_timeout_seconds"] == 180.0
    assert flat["openai_compatible_max_tokens"] == 4096

    assert flat["optimizer_openai_compatible_base_url"] == "https://api.together.xyz/v1"
    assert flat["optimizer_openai_compatible_api_key"] == "sk-together-test"
    assert flat["optimizer_openai_compatible_model"] == "together-model"
    assert flat["optimizer_openai_compatible_temperature"] == 0.1
    assert flat["optimizer_openai_compatible_timeout_seconds"] == 240.0
    assert flat["optimizer_openai_compatible_max_tokens"] == 8192

    assert flat["target_openai_compatible_base_url"] == "https://api.groq.com/openai/v1"
    assert flat["target_openai_compatible_api_key"] == "sk-groq-test"
    assert flat["target_openai_compatible_model"] == "groq-model"
    assert flat["target_openai_compatible_temperature"] == 0.7
    assert flat["target_openai_compatible_timeout_seconds"] == 90.0
    assert flat["target_openai_compatible_max_tokens"] == 2048


def test_load_config_with_cfg_options_overrides() -> None:
    raw = {
        "model": {
            "backend": "openai_compatible",
            "openai_compatible_base_url": "http://localhost:11434/v1",
            "openai_compatible_model": "llama3",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        path = f.name

    overrides = [
        "model.openai_compatible_base_url=http://custom-host:8000/v1",
        "model.openai_compatible_temperature=0.8",
    ]
    cfg = load_config(path, overrides=overrides)
    flat = flatten_config(cfg)

    assert flat["openai_compatible_base_url"] == "http://custom-host:8000/v1"
    assert flat["openai_compatible_temperature"] == 0.8
    assert flat["openai_compatible_model"] == "llama3"


def test_train_script_cli_to_structured_mapping() -> None:
    cli_args = [
        "--config", "configs/base.yaml",
        "--openai_compatible_base_url", "https://api.openai-compat.com/v1",
        "--openai_compatible_model", "custom-model",
        "--openai_compatible_temperature", "0.4",
        "--openai_compatible_timeout_seconds", "150",
        "--openai_compatible_max_tokens", "3000",
        "--optimizer_openai_compatible_base_url", "https://api.opt.com/v1",
        "--target_openai_compatible_base_url", "https://api.target.com/v1",
    ]

    with mock.patch("sys.argv", ["train.py"] + cli_args):
        args = train_script.parse_args()

    assert args.openai_compatible_base_url == "https://api.openai-compat.com/v1"
    assert args.openai_compatible_model == "custom-model"
    assert args.openai_compatible_temperature == 0.4
    assert args.openai_compatible_timeout_seconds == 150.0
    assert args.openai_compatible_max_tokens == 3000
    assert args.optimizer_openai_compatible_base_url == "https://api.opt.com/v1"
    assert args.target_openai_compatible_base_url == "https://api.target.com/v1"

    for flag in [
        "openai_compatible_base_url",
        "openai_compatible_api_key",
        "openai_compatible_model",
        "openai_compatible_temperature",
        "openai_compatible_timeout_seconds",
        "openai_compatible_max_tokens",
        "optimizer_openai_compatible_base_url",
        "optimizer_openai_compatible_api_key",
        "optimizer_openai_compatible_model",
        "optimizer_openai_compatible_temperature",
        "optimizer_openai_compatible_timeout_seconds",
        "optimizer_openai_compatible_max_tokens",
        "target_openai_compatible_base_url",
        "target_openai_compatible_api_key",
        "target_openai_compatible_model",
        "target_openai_compatible_temperature",
        "target_openai_compatible_timeout_seconds",
        "target_openai_compatible_max_tokens",
    ]:
        assert flag in train_script._LEGACY_TO_STRUCTURED
        assert train_script._LEGACY_TO_STRUCTURED[flag] == f"model.{flag}"


def test_eval_only_parse_args_and_map() -> None:
    raw = {
        "model": {
            "backend": "openai_compatible",
        },
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    cli_args = [
        "--config", config_path,
        "--skill", "skills/test.md",
        "--backend", "openai_compatible",
        "--openai_compatible_base_url", "https://eval.example/v1",
        "--openai_compatible_model", "eval-model",
        "--target_openai_compatible_temperature", "0.6",
    ]

    with mock.patch("sys.argv", ["eval_only.py"] + cli_args):
        args = eval_only_script.parse_args()
        cfg = eval_only_script.load_config(args)

    assert args.backend == "openai_compatible"
    assert args.openai_compatible_base_url == "https://eval.example/v1"
    assert args.openai_compatible_model == "eval-model"
    assert args.target_openai_compatible_temperature == 0.6
    assert cfg["openai_compatible_base_url"] == "https://eval.example/v1"
    assert cfg["openai_compatible_model"] == "eval-model"
    assert cfg["target_openai_compatible_temperature"] == 0.6


def test_train_script_credential_warnings_for_openai_compatible() -> None:
    raw = {
        "model": {"backend": "openai_compatible"},
        "env": {"name": "searchqa"},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw, f)
        config_path = f.name

    cli_args = [
        "--config", config_path,
        "--openai_compatible_api_key", "sk-secret-test",
    ]

    with mock.patch("sys.argv", ["train.py"] + cli_args):
        with pytest.deprecated_call(match="OPENAI_COMPATIBLE_API_KEY"):
            train_script.load_config(train_script.parse_args())


def test_trainer_initialization_configures_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    import skillopt.engine.trainer as trainer_mod
    from skillopt.envs.base import EnvAdapter

    configured_kwargs: dict = {}

    def fake_configure(**kwargs):
        configured_kwargs.update(kwargs)

    monkeypatch.setattr(trainer_mod, "configure_openai_compatible", fake_configure)

    class EarlyExit(Exception):
        pass

    def stop_after_model_config(*args, **kwargs):
        raise EarlyExit()

    monkeypatch.setattr(trainer_mod, "_configure_trace_to_optimizer_gates", stop_after_model_config)

    cfg = {
        "model_backend": "openai_compatible",
        "optimizer_backend": "openai_compatible",
        "target_backend": "openai_compatible",
        "optimizer_model": "opt-model",
        "target_model": "target-model",
        "openai_compatible_base_url": "https://api.test.com/v1",
        "openai_compatible_api_key": "test-key",
        "openai_compatible_model": "test-model",
        "openai_compatible_temperature": 0.5,
        "openai_compatible_timeout_seconds": 120.0,
        "openai_compatible_max_tokens": 4096,
        "skill_init": "skills/empty.md",
        "num_epochs": 1,
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "seed": 42,
        "out_root": "/tmp/out",
    }

    mock_adapter = mock.create_autospec(EnvAdapter, instance=True)
    mock_adapter.requires_ray.return_value = False
    mock_adapter.get_dataloader.return_value = None

    trainer = trainer_mod.ReflACTTrainer(cfg, mock_adapter)
    with pytest.raises(EarlyExit):
        trainer.train()

    assert configured_kwargs["base_url"] == "https://api.test.com/v1"
    assert configured_kwargs["api_key"] == "test-key"
    assert configured_kwargs["model"] == "test-model"
    assert configured_kwargs["temperature"] == 0.5
    assert configured_kwargs["timeout_seconds"] == 120.0
    assert configured_kwargs["max_tokens"] == 4096

