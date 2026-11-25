# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml

from tfm.extras.capability import extra_available

RECIPE = Path("recipes/blueprint_v1")
REQUIRED = (
    "resolved_config.yaml",
    "data_validation_report.json",
    "split_manifest.json",
    "tokenizer_manifest.json",
    "checkpoint_manifest.json",
    "embeddings_manifest.json",
    "metrics.json",
    "metrics.md",
    "model_card.md",
    "environment.json",
)


def test_cached_full_size_config_differs_only_in_its_token_cache() -> None:
    from tfm.config import load_recipe_config

    full = load_recipe_config(RECIPE / "config.yaml").model_dump(mode="json")
    cached = load_recipe_config(RECIPE / "config.cached.yaml").model_dump(mode="json")
    assert full["data"].pop("token_cache") is None
    assert cached["data"].pop("token_cache")
    assert cached == full


@pytest.mark.parametrize("world_size", [1, 2])
def test_full_size_config_trains_data_parallel_on_one_or_more_gpus(world_size: int) -> None:
    from tfm.config import load_recipe_config
    from tfm.training.backends import backend_capabilities, check_backend_supports

    cfg = load_recipe_config(RECIPE / "config.yaml")
    assert cfg.training.backend == "ddp"
    check_backend_supports(
        backend_capabilities(cfg.training.backend),
        backbone=cfg.model.backbone,
        packing=cfg.training.packing,
        device_type="cuda",
        world_size=world_size,
        layout="flat_fields",
        precision=cfg.training.precision,
    )


@pytest.mark.parametrize("name", ["config.small.yaml", "config.yaml", "config.cached.yaml"])
def test_tabformer_configs_fit_their_rows(name: str) -> None:
    from tfm.config import load_recipe_config

    cfg = load_recipe_config(RECIPE / name)
    fields = cfg.features.fields
    assert fields is not None and "CUST" not in fields
    per_event = len(fields) + 1
    assert cfg.data.max_tokens == per_event * cfg.data.sequence_length + 1
    decoder = cfg.model.composition.granularity.decoder
    assert decoder.max_seq_len >= cfg.data.max_tokens
    windows = cfg.data.pretrain_windows
    if windows is not None:
        assert windows.max_tokens is not None
        assert per_event * windows.events + 1 <= windows.max_tokens <= decoder.max_seq_len
    assert cfg.model.backbone == "huggingface"
    assert cfg.model.composition.position == "rope"
    assert cfg.evaluation.history_window.enabled
    assert cfg.evaluation.tabular_baselines.exclude_keys == ["user"]


def test_blueprint_v1_cpu_smoke(tmp_path: Path) -> None:
    from tfm.runtime import run_smoke

    out = run_smoke(
        run_dir=tmp_path / "run", config_path=RECIPE / "config.smoke.yaml", device="cpu"
    )
    for name in REQUIRED:
        assert (out / name).exists(), name
    env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    assert env["device"] == "cpu"
    report = json.loads((out / "data_validation_report.json").read_text(encoding="utf-8"))
    assert report["adapter"] == "synthetic"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert {"history_window_combined", "single_event_combined"} <= set(metrics)
    # The control's raw row is the history row's; only its embedding is scored.
    assert "single_event_raw" not in metrics


@pytest.mark.skipif(
    not all(extra_available(m) for m in ("transformers", "xgboost", "optuna")),
    reason="needs the huggingface and baselines extras",
)
def test_blueprint_v1_llama_arms(tmp_path: Path) -> None:
    from tfm.runtime import run_smoke

    raw = yaml.safe_load((RECIPE / "config.smoke.yaml").read_text(encoding="utf-8"))
    raw["model"]["backbone"] = "huggingface"
    raw["model"]["composition"]["position"] = "rope"
    raw["evaluation"]["tabular_baselines"]["xgboost_trials"] = 2
    config = tmp_path / "llama_smoke.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = run_smoke(run_dir=tmp_path / "run", config_path=config, device="cpu")
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    for arm in (
        "history_window_xgboost_raw",
        "history_window_xgboost_raw_embedding",
        "history_window_xgboost_raw_embedding:single_event",
        "history_window_tabular_xgboost",
    ):
        assert arm in metrics, arm
    # With the XGBoost arms on, the logistic stack is not scored.
    assert not {"raw", "combined", "history_window_combined"} & set(metrics)
    card = (out / "model_card.md").read_text(encoding="utf-8")
    assert "## Headline" in card and "| history_window |" in card
