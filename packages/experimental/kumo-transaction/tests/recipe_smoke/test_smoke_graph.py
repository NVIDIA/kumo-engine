# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tfm.cli.recipe import REQUIRED_ARTIFACTS, run_command


@pytest.mark.parametrize("name", ["gnn_encoder", "gnn_sequence"])
@pytest.mark.parametrize("mode", ["binary", "future_subgraph"])
@pytest.mark.parametrize("sharing", ["shared", "disjoint"])
def test_graph_recipe_complete_and_reusable(
    name: str, sharing: str, mode: str, tmp_path: Path
) -> None:
    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires the optional kumo extra")
    config = (
        Path("recipes")
        / name
        / ("config.smoke.yaml" if mode == "binary" else "config.pretrain_smoke.yaml")
    )
    if sharing == "disjoint":
        control = yaml.safe_load(config.read_text())
        control["data"]["training_graphs"] = sharing
        control["runtime"]["device"] = "cuda:999"  # CPU CLI override must be persisted.
        config = tmp_path / "control.yaml"
        config.write_text(yaml.safe_dump(control))
    out = run_command("smoke", config_path=config, run_dir=tmp_path / name, device="cpu")
    assert all((out / f).is_file() for f in REQUIRED_ARTIFACTS)
    assert yaml.safe_load((out / "resolved_config.yaml").read_text())["runtime"]["device"] == "cpu"
    metrics = json.loads((out / "metrics.json").read_text())
    assert {"raw", "embedding", "combined", "graph_tabular", "sequence_only_embedding"} <= set(
        metrics
    )
    assert {"sequence_supervised_embedding", "sequence_supervised_task_head"} <= set(metrics)
    identity = json.loads((out / "checkpoint_manifest.json").read_text())
    assert identity["architecture"] == name
    assert identity["format"] == "tfm.graph-experiment.v1"
    assert identity["graph_objective"] == mode
    assert identity["hop_trimming"] is True
    assert identity["sequence_attention"] == "sdpa"
    assert (identity["target_transform"] is None) == (mode == "binary")
    if mode == "binary" or name == "gnn_sequence":
        assert "task_head" in metrics
    assert identity["kumo_engine_version"]
    assert identity["split"]["training_graphs"] == sharing
    assert identity["split"]["validation_graphs"] == "disjoint"
    assert identity["stages"]["graph"]["best_step"] > 0
    splits = json.loads((out / "split_manifest.json").read_text())
    assert set(splits["ssl_train_owners"]).isdisjoint(splits["ssl_validation_owners"])
    owners = splits["probe_owners"]
    assert set(owners["train"]).isdisjoint(owners["test"])
    original = dict(np.load(out / "embeddings.npz"))
    run_command("evaluate", config_path=config, run_dir=out, device="cpu", reuse_checkpoint=True)
    for key, values in dict(np.load(out / "embeddings.npz")).items():
        np.testing.assert_array_equal(values, original[key])
    metadata_path = out / "model.manifest.json"
    saved = json.loads(metadata_path.read_text())
    metadata_path.write_text(json.dumps({**saved, "format": "graph-v1"}))
    run_command("infer", config_path=config, run_dir=out, device="cpu", reuse_checkpoint=True)
    assert json.loads(metadata_path.read_text()) == saved
    changed = yaml.safe_load(config.read_text())
    changed["seed"] += 1
    path = tmp_path / "changed.yaml"
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="config identity"):
        run_command("infer", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)
