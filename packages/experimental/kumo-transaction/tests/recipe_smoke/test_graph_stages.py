# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
import yaml

from tfm.cli.recipe import run_command
from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture
from tfm.training.checkpointing import load_encoder_state


def config_for(book):
    raw = yaml.safe_load(Path("recipes/gnn_sequence/config.pretrain_smoke.yaml").read_text())
    raw["data"].update(
        adapter="pragmatiq",
        path=str(book),
        n_entities=12,
        pretrain_end="2024-01-15T00:00:00Z",
        horizon_days=1,
        observed_through="2024-01-30T12:00:00Z",
    )
    raw["training"]["max_steps"] = 1
    raw["sequence_training"]["max_steps"] = 1
    return raw


def test_label_free_pretrain_export_and_downstream_reuse(tmp_path, monkeypatch):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.api.graph import graph_specs
    from tfm.config import RecipeConfig
    from tfm.data import graph_recipe
    from tfm.layers.gnn import GraphEncoder

    book = generate_pragmatiq_fixture(tmp_path / "unlabelled", n_users=12, events_per_user=30)
    (book / "labels/fraud.parquet").unlink()
    raw = config_for(book)
    raw["runtime"]["stage"] = "pretrain"
    path = tmp_path / "pretrain.yaml"
    path.write_text(yaml.safe_dump(raw))
    original = graph_recipe.available_label_value

    def forbidden(*args, **kwargs):
        raise AssertionError("Self-supervised pretrain must not inspect task labels")

    monkeypatch.setattr(graph_recipe, "available_label_value", forbidden)
    pretrained = run_command(
        "train", config_path=path, run_dir=tmp_path / "pretrained", device="cpu"
    )
    weights = torch.load(pretrained / "encoder.pt", map_location="cpu", weights_only=True)
    assert "state_dict" not in weights and all(
        isinstance(value, torch.Tensor) for value in weights.values()
    )
    assert not (pretrained / "metrics.json").exists()
    manifest = json.loads((pretrained / "model.manifest.json").read_text())
    assert set(manifest["stages"]) == {"graph"}
    assert not manifest["split"]["requires_task_labels"]
    # Public reuse in an unrelated model needs only a spec and plain state_dict.
    data = graph_recipe.prepare_graph_data(RecipeConfig.model_validate(raw))
    graph, _ = graph_specs(RecipeConfig.model_validate(raw), data)
    independent = GraphEncoder(graph)
    independent.load_state_dict(weights)
    torch.testing.assert_close(
        independent.state_dict()[next(iter(weights))], weights[next(iter(weights))]
    )
    # A same-shaped swap must be rejected before state installation.
    swapped = {key: value.clone() for key, value in weights.items()}
    swapped[next(iter(swapped))].add_(1)
    torch.save(swapped, pretrained / "encoder.pt")
    with pytest.raises(ValueError, match="weights_sha256"):
        load_encoder_state(independent, pretrained / "encoder.pt", expected={})
    torch.save(weights, pretrained / "encoder.pt")
    from tfm.training.checkpointing import save_encoder_state

    save_encoder_state(independent, pretrained / "encoder.pt", metadata=manifest)
    monkeypatch.setattr(graph_recipe, "available_label_value", original)
    labelled = generate_pragmatiq_fixture(tmp_path / "labelled", n_users=12, events_per_user=30)
    raw["data"].update(
        path=str(labelled), probe_cutoff="2024-01-28T12:00:00Z", pretrain_end="2099-01-01T00:00:00Z"
    )
    raw["runtime"].update(stage="downstream", encoder_checkpoint=str(pretrained / "encoder.pt"))
    path = tmp_path / "downstream.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(graph_recipe, "future_subgraph_targets", forbidden)
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "downstream", device="cpu")
    saved = json.loads((out / "model.manifest.json").read_text())
    assert "graph" not in saved["stages"] and saved["encoder_source"]
    assert saved["execution_stage"] == "downstream"
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)
    assert (out / "metrics.json").exists()


def test_full_training_keeps_missing_label_rejection(tmp_path):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    book = generate_pragmatiq_fixture(tmp_path / "book", n_users=12, events_per_user=30)
    (book / "labels/fraud.parquet").unlink()
    raw = config_for(book)
    raw["data"]["probe_cutoff"] = "2024-01-28T12:00:00Z"
    path = tmp_path / "full.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="fraud label coverage"):
        run_command("train", config_path=path, run_dir=tmp_path / "out", device="cpu")


@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
@pytest.mark.parametrize("selection", ["registered", "imported", "configured_calendar"])
def test_graph_yaml_extensions_train_and_reuse(tmp_path, objective, selection):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tests.fixtures.graph_components import UserReadout, head_factory, history_factory

    from tfm.api.graph import register_context_history, register_task_head
    from tfm.pooling.registry import register_readout

    fixture = "tests.fixtures.graph_components"
    raw = yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())
    raw["features"] = {"tokenizer": f"{fixture}:UserTokenizer"}
    raw["training"].update(objective=objective, max_steps=1)
    raw["sequence_training"]["max_steps"] = 1
    gran = raw["model"]["composition"]["granularity"]
    if selection != "configured_calendar":
        raw["model"]["composition"]["position"] = "nope"
    if selection == "registered":
        register_context_history("test_stage_history", history_factory)
        register_task_head("test_stage_head", head_factory)
        register_readout("test_stage_readout", UserReadout)
        gran.update(history_model="test_stage_history", task_head="test_stage_head")
        raw["model"]["composition"]["readout"] = "test_stage_readout"
    elif selection == "imported":
        gran.update(history_model=f"{fixture}:history_factory", task_head=f"{fixture}:head_factory")
        raw["model"]["composition"]["readout"] = f"{fixture}:UserReadout"
    else:
        gran["time_features"] = {"custom_clock": 27}
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(raw))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    manifest = json.loads((out / "model.manifest.json").read_text())
    if selection == "configured_calendar":
        assert manifest["sequence_schema"]["time_features"] == [
            {"array": "custom_clock", "period": 27.0, "origin": 0.0}
        ]
    else:
        assert any("UserHistory" in value for value in manifest["module_types"].values())
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)
