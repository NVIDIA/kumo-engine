# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader selection and resource capabilities without optional dependencies."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tfm.config import load_recipe_config
from tfm.contracts.schema import TransactionEvent
from tfm.data import native_history as native
from tfm.data.graph import GraphSampler

NOW = datetime(2024, 1, 2, tzinfo=UTC)


def events():
    return [
        TransactionEvent("a", str(i), NOW - timedelta(seconds=i), numeric={"amount": float(i)})
        for i in range(3)
    ]


def test_loader_default_and_explicit_reference():
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    assert config.data.loader == "kumo_pytorch"
    assert config.data.model_copy(update={"loader": "python"}).loader == "python"


def test_explicit_python_loader_does_not_import_native(monkeypatch):
    def forbidden():
        raise AssertionError("Explicit Python loader must be usable without native imports")

    monkeypatch.setattr(native, "native_api", forbidden)
    assert native.build_entity_sequences(events(), NOW, max_events=2, loader="python")


def test_default_sequence_helpers_do_not_require_native(monkeypatch):
    def forbidden():
        raise AssertionError("Python defaults must not initialize a native backend")

    monkeypatch.setattr(native, "native_api", forbidden)
    assert native.build_entity_sequences(events(), NOW, max_events=2)
    assert native.build_entity_windows(events(), NOW, window_events=2)
    labels = [SimpleNamespace(entity_id="a", timestamp=NOW)]
    assert native.build_sequences_at_label_cutoffs(events(), labels, max_events=2)


@pytest.mark.parametrize("failure", [None, "manifest", "trainer", "progress", "fit"])
def test_runner_closes_structural_batch_resource(failure, monkeypatch, tmp_path):
    import torch

    from tfm.contracts.resources import Closable
    from tfm.runtime import pipeline

    class CustomBatches:
        closed = 0

        def __len__(self):
            return 1

        def close(self):
            self.closed += 1

    batches = CustomBatches()
    assert isinstance(batches, Closable)
    assert not isinstance([], Closable)

    def fail(*args, **kwargs):
        raise RuntimeError(failure)

    monkeypatch.setattr(pipeline, "training_batches", lambda artifacts: (batches, {}))
    monkeypatch.setattr(pipeline, "is_primary_process", lambda: True)
    monkeypatch.setattr(
        pipeline, "write_json", fail if failure == "manifest" else lambda *a, **k: None
    )
    trainer = SimpleNamespace(fit=fail if failure == "fit" else lambda *a, **k: {"loss": 0.0})
    monkeypatch.setattr(
        pipeline, "build_trainer", fail if failure == "trainer" else lambda *a, **k: trainer
    )
    monkeypatch.setattr(
        pipeline,
        "TrainingProgress",
        fail if failure == "progress" else lambda *a, **k: SimpleNamespace(curve=[]),
    )
    monkeypatch.setattr(pipeline, "stage", lambda *a: None)
    monkeypatch.setattr(pipeline, "save_run_checkpoint", lambda *a: None)
    monkeypatch.setattr(pipeline, "_composition_summary", lambda *a: {})
    artifacts = SimpleNamespace(
        config=load_recipe_config("recipes/blueprint_v1/config.smoke.yaml"),
        run_dir=tmp_path,
        model=None,
        device=torch.device("cpu"),
        tokenizer=SimpleNamespace(vocab_size=10, layout="token"),
    )
    if failure is None:
        assert pipeline.run_train(artifacts) == {"loss": 0.0}
    else:
        with pytest.raises(RuntimeError, match=failure):
            pipeline.run_train(artifacts)
    assert batches.closed == 1


@pytest.mark.parametrize("failure", [False, True])
def test_graph_loader_supplies_provenance_and_owns_cleanup(failure, monkeypatch):
    from tfm.data import graph_recipe

    created = []

    class CustomLoader:
        loader_manifest = {"name": "custom-native", "revision": "fixture"}
        tie_policy = "custom tie policy"

        def __init__(self, *args, **kwargs):
            self.delegate = GraphSampler(*args, **kwargs)
            self.closed = 0
            created.append(self)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def close(self):
            self.closed += 1

    monkeypatch.setattr(graph_recipe, "NativeGraphSampler", CustomLoader)
    if failure:

        def fail(*args, **kwargs):
            raise RuntimeError("preparation")

        monkeypatch.setattr(graph_recipe, "_require_training_amount", fail)
    config = load_recipe_config("recipes/gnn_encoder/config.smoke.yaml")
    config = config.model_copy(
        update={
            "data": config.data.model_copy(update={"loader": "kumo_pytorch"}),
            "runtime": config.runtime.model_copy(update={"stage": "pretrain"}),
        }
    )
    if failure:
        with pytest.raises(RuntimeError, match="preparation"):
            graph_recipe.prepare_graph_data(config)
        assert created[0].closed == 1
    else:
        data = graph_recipe.prepare_graph_data(config)
        try:
            assert data.split_manifest["input_loader"] == CustomLoader.loader_manifest
            assert data.split_manifest["sampling"]["ties"] == CustomLoader.tie_policy
            assert created[0].closed == 0
        finally:
            data.sampler.close()
        assert created[0].closed == 1


def test_every_canonical_recipe_selects_native():
    from pathlib import Path

    for path in Path("recipes").glob("*/config*.yaml"):
        assert load_recipe_config(path).data.loader == "kumo_pytorch", str(path)


def test_omitted_loader_uses_native_in_typed_and_yaml_configs():
    from pathlib import Path

    import yaml

    from tfm.config import DataConfig, RecipeConfig

    data = yaml.safe_load(Path("recipes/blueprint_v1/config.smoke.yaml").read_text())
    data["data"].pop("loader")
    assert RecipeConfig.model_validate(data).data.loader == "kumo_pytorch"
    assert DataConfig.model_validate(data["data"]).loader == "kumo_pytorch"


def test_default_loader_missing_extra_raises_without_python_fallback(monkeypatch):
    from tfm.contracts.errors import ExtraUnavailableError

    calls = []

    def unavailable():
        calls.append(True)
        raise ExtraUnavailableError("Native integration is unavailable")

    monkeypatch.setattr(native, "native_api", unavailable)
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    with pytest.raises(ExtraUnavailableError, match="Native integration is unavailable"):
        native.build_entity_sequences(events(), NOW, max_events=2, loader=config.data.loader)
    assert calls == [True]
