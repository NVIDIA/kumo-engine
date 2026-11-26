# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flat BERT-style encoder (flat-encoder guide) through v1's pipeline on a generated book."""

import json
from pathlib import Path

import yaml

from tfm.config import load_recipe_config

ABLATION = Path("benchmarks/ablations/flat_encoder_tabformer.yaml")
V1 = Path("recipes/blueprint_v1/config.yaml")


def test_ablation_differs_from_v1_only_in_the_encoder_slots() -> None:
    encoder = load_recipe_config(ABLATION).model_dump()
    v1 = load_recipe_config(V1).model_dump()
    for section in ("seed", "data", "features", "training", "fraud_task", "evaluation"):
        assert encoder[section] == v1[section], section
    composition = encoder["model"]["composition"]
    assert composition["granularity"]["type"] == "flat_encoder"
    assert composition["objective"] == "masked_fields"
    assert composition["readout"] == "last_event_mean"


def test_flat_encoder_cpu_smoke(tmp_path: Path) -> None:
    from tfm.runtime import run_smoke

    raw = yaml.safe_load(Path("recipes/blueprint_v1/config.smoke.yaml").read_text("utf-8"))
    composition = raw["model"]["composition"]
    composition["granularity"]["type"] = "flat_encoder"
    composition["granularity"]["decoder"].pop("mask")
    composition.update(
        position="rope", objective="masked_fields", readout="last_event_mean", heads={}
    )
    config = tmp_path / "flat_encoder_smoke.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    out = run_smoke(run_dir=tmp_path / "run", config_path=config, device="cpu")
    stats = json.loads((out / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert stats
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert {"history_window_combined", "single_event_combined"} <= set(metrics)
