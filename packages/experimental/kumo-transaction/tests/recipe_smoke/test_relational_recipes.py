# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end shared dataset recipes; synthetic data only, CPU-safe."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tfm.cli.recipe import REQUIRED_ARTIFACTS, run_command


@pytest.mark.parametrize("variant", ["smoke", "graph_smoke", "hybrid_smoke"])
@pytest.mark.parametrize("loader", ["python", "kumo_pytorch"])
def test_relational_recipe_and_checkpoint_reuse(variant, loader, tmp_path):
    if loader == "kumo_pytorch":
        pytest.importorskip("kumo.pytorch.api", reason="requires optional native extra")
    if variant != "smoke":
        pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    config = yaml.safe_load(
        (Path("recipes/relational_inputs") / f"config.{variant}.yaml").read_text()
    )
    config["data"]["loader"] = loader
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    assert all((out / name).exists() for name in REQUIRED_ARTIFACTS)
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert metadata["schema"]["target_level"] == "entity"
    split = json.loads((out / "split_manifest.json").read_text())
    assert len(split["anchors"]) == 8
    assert split["loader"]["name"] == loader
    before = dict(np.load(out / "embeddings.npz"))
    # The embedded identity is authoritative and repairs a corrupt sidecar.
    (out / "model.manifest.json").write_text("broken")
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)
    after = dict(np.load(out / "embeddings.npz"))
    assert before.keys() == after.keys()
    for name in before:
        np.testing.assert_array_equal(before[name], after[name])
    assert json.loads((out / "model.manifest.json").read_text()) == metadata


def test_sequence_mode_never_builds_or_transfers_a_graph(monkeypatch, tmp_path):
    from tfm.data.relational_batches import RelationalBatches

    def forbidden(*args, **kwargs):
        raise AssertionError("sequence mode must not construct a graph")

    monkeypatch.setattr(RelationalBatches, "graph_sampler", forbidden)
    run_command(
        "smoke",
        config_path=Path("recipes/relational_inputs/config.smoke.yaml"),
        run_dir=tmp_path,
        device="cpu",
    )


@pytest.mark.parametrize("variant", ["graph_smoke", "hybrid_smoke"])
def test_relational_future_pretraining_and_full_downstream(variant, tmp_path):
    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    config = yaml.safe_load(
        (Path("recipes/relational_inputs") / f"config.{variant}.yaml").read_text()
    )
    config["training"]["objective"] = "future_subgraph"
    path = tmp_path / "future.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert "future_subgraph" in metadata["stages"]
    assert metadata["target_transform"]["selected"]
    assert (out / "encoder.pt").exists()
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)


def test_label_free_encoder_export_and_graph_downstream_fit(tmp_path):
    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    pretrain = Path("recipes/relational_inputs/config.encoder_pretrain_smoke.yaml")
    out = run_command("train", config_path=pretrain, run_dir=tmp_path / "pretrain", device="cpu")
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert metadata["tokenizer"] is None
    assert metadata["target_transform"] is not None
    config = yaml.safe_load(Path("recipes/relational_inputs/config.graph_smoke.yaml").read_text())
    config["runtime"].update(stage="downstream", encoder_checkpoint=str(out / "encoder.pt"))
    path = tmp_path / "downstream.yaml"
    path.write_text(yaml.safe_dump(config))
    result = run_command("smoke", config_path=path, run_dir=tmp_path / "downstream", device="cpu")
    loaded = json.loads((result / "model.manifest.json").read_text())
    assert "graph_binary" in loaded["stages"]  # The downstream head cannot remain random.
