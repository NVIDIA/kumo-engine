# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical source-native execution must never reconstruct Python histories."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tfm.cli.recipe import REQUIRED_ARTIFACTS, run_command


@pytest.mark.parametrize("variant", ["smoke", "graph_smoke", "hybrid_smoke"])
def test_source_native_cli_bypasses_object_histories_and_python_tokenizers(
    variant, monkeypatch, tmp_path
):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    if variant != "smoke":
        pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.data import native_history
    from tfm.data.adapters import registry
    from tfm.data.graph import GraphSampler
    from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer
    from tfm.runtime import relational

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Source-native recipes must consume native tensors, not Python events/windows/tokenization"
        )

    monkeypatch.setattr(relational, "load_adapter_events", forbidden)
    monkeypatch.setattr(registry, "load_adapter_events", forbidden)
    monkeypatch.setattr(native_history, "NativeHistories", forbidden)
    monkeypatch.setattr(native_history, "NativeSequenceBatches", forbidden)
    monkeypatch.setattr(GraphSampler, "sample", forbidden)
    monkeypatch.setattr(KeyValueTimeTokenizer, "fit", forbidden)
    monkeypatch.setattr(KeyValueTimeTokenizer, "collate", forbidden)
    config = Path(f"recipes/relational_inputs/config.{variant}.yaml")
    out = run_command("smoke", config_path=config, run_dir=tmp_path / "run", device="cpu")
    assert all((out / name).is_file() for name in REQUIRED_ARTIFACTS)
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert metadata["input_loader"]["pipeline"] == "source_tables_native_tensors_v1"
    assert (out / "anchors.parquet").is_file()
    assert list((out / "native-fitted").glob("*/fitted-state.json"))
    before = dict(np.load(out / "embeddings.npz"))
    if variant == "hybrid_smoke":
        assert "train_graph" in before
        metrics = json.loads((out / "metrics.json").read_text())
        assert {"graph_only_embedding", "graph_only_combined"} <= metrics.keys()
    assert before["train_entity_ids"].tolist() == ["entity0", "entity1", "entity2", "entity3"]
    (out / "model.manifest.json").write_text("corrupt")
    run_command("evaluate", config_path=config, run_dir=out, device="cpu", reuse_checkpoint=True)
    assert json.loads((out / "model.manifest.json").read_text()) == metadata
    after = dict(np.load(out / "embeddings.npz"))
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert not list((out / "native-work").iterdir())
    assert (
        metadata["input_loader"]["history_tie_policy"] == "event_time_then_typed_event_identity_v1"
    )
    assert metadata["input_loader"]["graph_tie_policy"] == "canonical_key_asc"
    if variant == "smoke":
        import torch

        from tfm.training.checkpointing import read_checkpoint

        payload = read_checkpoint(out / "model.pt")
        payload["metadata"]["input_loader"].pop("history_tie_policy")
        torch.save(payload, out / "model.pt")
        monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
        with pytest.raises(ValueError, match="source/anchor identity"):
            run_command(
                "evaluate", config_path=config, run_dir=out, device="cpu", reuse_checkpoint=True
            )


@pytest.mark.parametrize("variant", ["graph_smoke", "hybrid_smoke"])
def test_source_native_future_targets_and_matched_downstream(variant, tmp_path):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    config = yaml.safe_load(Path(f"recipes/relational_inputs/config.{variant}.yaml").read_text())
    config["training"]["objective"] = "future_subgraph"
    if variant == "graph_smoke":
        config["model"]["composition"]["objective"] = "future_subgraph"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert metadata["target_transform"]["selected"]
    assert "future_subgraph" in metadata["stages"]
    if variant == "hybrid_smoke":
        assert {"hybrid", "sequence_control"} <= metadata["stages"].keys()
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)


@pytest.mark.parametrize("recipe", ["gnn_encoder", "gnn_sequence"])
@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
def test_stock_graph_recipes_stream_sources_without_preparing_graph_or_history_lists(
    recipe, objective, monkeypatch, tmp_path
):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.data import graph_recipe, native_history
    from tfm.data.adapters import registry
    from tfm.data.graph import GraphSampler
    from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Canonical native graph recipes must not prepare Python histories/batch lists"
        )

    monkeypatch.setattr(graph_recipe, "prepare_graph_data", forbidden)
    monkeypatch.setattr(registry, "load_adapter_events", forbidden)
    monkeypatch.setattr(native_history, "NativeSequenceBatches", forbidden)
    monkeypatch.setattr(GraphSampler, "sample", forbidden)
    monkeypatch.setattr(KeyValueTimeTokenizer, "fit", forbidden)
    monkeypatch.setattr(KeyValueTimeTokenizer, "collate", forbidden)
    config = yaml.safe_load(Path(f"recipes/{recipe}/config.smoke.yaml").read_text())
    config["training"]["objective"] = objective
    if recipe == "gnn_encoder":
        config["model"]["composition"]["objective"] = objective
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    metadata = json.loads((out / "model.manifest.json").read_text())
    assert metadata["input_loader"]["pipeline"] == "source_tables_native_tensors_v1"
    assert metadata["schema"]["target_level"] == "sequence"
    assert metadata["input_loader"]["target_level"] == "sequence"
    encoder = json.loads((out / "encoder.manifest.json").read_text())
    assert encoder["schema"]["amount_field"] == config["data"].get("amount_field", "amount")
    assert encoder["schema"]["feature_transform"] == "transaction-signed-log-v1"
    before = dict(np.load(out / "embeddings.npz"))
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)
    after = dict(np.load(out / "embeddings.npz"))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("recipe", ["blueprint_v1", "blueprint_v2", "pragma_like"])
def test_stock_sequence_cli_uses_continuous_native_tensor_delivery(recipe, monkeypatch, tmp_path):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    from tfm.data import native_history
    from tfm.data.adapters import registry
    from tfm.features.tokenizers.financial import TransactionFieldTokenizer
    from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer
    from tfm.runtime import data

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Canonical native sequence recipes must not build Python corpora or tokenized windows"
        )

    monkeypatch.setattr(data, "prepare_data", forbidden)
    monkeypatch.setattr(registry, "load_adapter_events", forbidden)
    monkeypatch.setattr(native_history, "NativeHistories", forbidden)
    monkeypatch.setattr(native_history, "NativeSequenceBatches", forbidden)
    for cls in (TransactionFieldTokenizer, KeyValueTimeTokenizer):
        monkeypatch.setattr(cls, "fit", forbidden)
        monkeypatch.setattr(cls, "collate", forbidden)
    config = Path(f"recipes/{recipe}/config.smoke.yaml")
    out = run_command("smoke", config_path=config, run_dir=tmp_path / "run", device="cpu")
    assert all((out / name).is_file() for name in REQUIRED_ARTIFACTS)
    manifest = json.loads((out / "model.manifest.json").read_text())
    assert manifest["input_loader"]["pipeline"] == "source_tables_native_tensors_v1"
    before = dict(np.load(out / "embeddings.npz"))
    run_command("evaluate", config_path=config, run_dir=out, device="cpu", reuse_checkpoint=True)
    after = dict(np.load(out / "embeddings.npz"))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    if recipe == "blueprint_v1":
        metrics = json.loads((out / "metrics.json").read_text())
        assert {
            "tabular_logistic",
            "history_window_raw",
            "single_event_embedding",
        } <= metrics.keys()


def test_source_native_partial_prediction_labels_keep_known_negative_alignment(
    tmp_path, monkeypatch
):
    from datetime import timedelta

    import pyarrow as pa
    import pyarrow.parquet as pq

    from tfm.runtime import native

    pytest.importorskip("kumo.pytorch.api")
    base = native.table_source

    def source(*args, **kwargs):
        declaration = base(*args, **kwargs)
        table = pq.read_table(declaration.files["outcomes"])
        rows = table.to_pylist()
        prediction = {
            **rows[0],
            "id": "prediction",
            "split": "predict",
            "cutoff": rows[0]["cutoff"] + timedelta(days=3),
            "available": rows[0]["available"] + timedelta(days=3),
            "outcome": 0.0,
        }
        rows.append(prediction)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema), declaration.files["outcomes"]
        )
        anchors = [
            {
                "sample_id": f"{row['split']}-{row['owner']}",
                "id": row["owner"],
                "cutoff": row["cutoff"],
                "split": row["split"],
            }
            for row in rows
        ]
        anchors.append(
            {
                "sample_id": "predict-entity1",
                "id": "entity1",
                "cutoff": prediction["cutoff"],
                "split": "predict",
            }
        )
        pq.write_table(pa.Table.from_pylist(anchors), tmp_path / "anchors.parquet")
        return declaration

    monkeypatch.setattr(native, "table_source", source)
    config = yaml.safe_load(Path("recipes/relational_inputs/config.smoke.yaml").read_text())
    config["data"]["relational"]["anchors"] = str(tmp_path / "anchors.parquet")
    config["data"]["relational"]["label_cutoffs"]["predict"] = "2024-01-21"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("infer", config_path=path, run_dir=tmp_path / "run", device="cpu")
    values = dict(np.load(out / "embeddings.npz"))
    assert values["predict_sample_ids"].tolist() == ["predict-entity0", "predict-entity1"]
    assert values["predict_labels_valid"].tolist() == [True, False]
    assert values["predict_labels"][0] == 0.0
    assert np.isnan(values["predict_labels"][1])


def test_source_native_same_width_feature_reorder_rejected_before_weights(tmp_path, monkeypatch):
    from dataclasses import replace

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    from tfm.runtime import native

    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    base, reverse = native.table_source, False

    def source(*args, **kwargs):
        declaration = base(*args, **kwargs)
        table = pq.read_table(declaration.files["activity"])
        table = table.append_column(
            "other_value", pa.array([value + 100.0 for value in table["value"].to_pylist()])
        )
        pq.write_table(table, declaration.files["activity"])
        fields = [("amount", "value"), ("other", "other_value")]
        if reverse:
            fields.reverse()
        return replace(
            declaration,
            tables=tuple(
                replace(item, numeric=dict(fields)) if item.name == "activity" else item
                for item in declaration.tables
            ),
        )

    monkeypatch.setattr(native, "table_source", source)
    config = Path("recipes/relational_inputs/config.graph_smoke.yaml")
    out = run_command("train", config_path=config, run_dir=tmp_path / "run", device="cpu")
    reverse = True

    def forbidden(*args, **kwargs):
        raise AssertionError("Weights installed before source/schema identity validation")

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
    with pytest.raises(ValueError, match="source/anchor identity"):
        run_command("infer", config_path=config, run_dir=out, device="cpu", reuse_checkpoint=True)


@pytest.mark.parametrize("loader", ["python", "kumo_pytorch"])
def test_downstream_reuse_needs_only_its_bundled_encoder_weights(loader, tmp_path, monkeypatch):
    if loader == "kumo_pytorch":
        pytest.importorskip("kumo.pytorch.api")
        pytest.importorskip("diskgraph")
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    config = yaml.safe_load(
        Path("recipes/relational_inputs/config.encoder_pretrain_smoke.yaml").read_text()
    )
    config["data"]["loader"] = loader
    pretrain_path = tmp_path / "pretrain.yaml"
    pretrain_path.write_text(yaml.safe_dump(config))
    pretrain = run_command(
        "train", config_path=pretrain_path, run_dir=tmp_path / "pretrain", device="cpu"
    )
    config = yaml.safe_load(Path("recipes/relational_inputs/config.graph_smoke.yaml").read_text())
    config["data"]["loader"] = loader
    config["runtime"].update(stage="downstream", encoder_checkpoint=str(pretrain / "encoder.pt"))
    path = tmp_path / "downstream.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "downstream", device="cpu")
    before = dict(np.load(out / "embeddings.npz"))
    (pretrain / "encoder.pt").rename(pretrain / "encoder.offline")
    run_command("evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True)
    after = dict(np.load(out / "embeddings.npz"))
    for name, values in before.items():
        np.testing.assert_array_equal(values, after[name])
    if loader == "python":
        import torch

        from tfm.training.checkpointing import read_checkpoint

        payload = read_checkpoint(out / "model.pt")
        payload["metadata"]["encoder_source"]["schema"]["root"] = "wrong-root"
        torch.save(payload, out / "model.pt")

        def forbidden(*args, **kwargs):
            raise AssertionError("State installed before checking the saved encoder schema")

        monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
        with pytest.raises(ValueError, match="encoder schema"):
            run_command(
                "evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True
            )


def test_native_sequence_raw_features_use_the_selected_amount_alias(tmp_path, monkeypatch):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    from dataclasses import replace

    from tfm.data.adapters.registry import table_source as original_source
    from tfm.runtime import native_sequence

    def custom_source(data, *, seed, fixture_dir):
        source = original_source(
            data.model_copy(update={"amount_field": "amount"}), seed=seed, fixture_dir=fixture_dir
        )
        tables = tuple(
            replace(
                t,
                numeric={
                    ("payment_total" if key == "amount" else key): column
                    for key, column in t.numeric.items()
                },
            )
            if t.name == "events"
            else t
            for t in source.tables
        )
        future = {**source.future_numeric, "events": {"amount": "payment_total"}}
        return replace(source, tables=tables, future_numeric=future)

    monkeypatch.setattr(native_sequence, "table_source", custom_source)
    config = yaml.safe_load(Path("recipes/blueprint_v2/config.smoke.yaml").read_text())
    config["data"].update(loader="kumo_pytorch", amount_field="payment_total")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    values = np.load(out / "embeddings.npz")
    assert np.isfinite(values["train_raw"]).all()
    assert (values["train_raw"][:, :2] > 0).all()


@pytest.mark.parametrize("finite_training_amount", [False, True])
def test_label_free_precomputed_sequence_accepts_equal_partition_cutoffs(
    tmp_path, monkeypatch, finite_training_amount
):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    from tfm.data.adapters.native_transactions import _ScalarBook
    from tfm.runtime import native_sequence

    def source_factory(data, *, seed, fixture_dir):
        instant = 1704067200000000
        with _ScalarBook(fixture_dir, data.amount_field) as book:
            for partition in ("train", "validation", "test"):
                for index in range(3):
                    book.event(
                        "one",
                        f"{partition}-{index}",
                        instant + (index - 2) * 1_000_000,
                        "transaction",
                        {
                            "amount": index + 1
                            if finite_training_amount or partition != "train"
                            else float("nan")
                        },
                        {},
                        {},
                        None,
                        None,
                        partition=partition,
                    )
            return book.tables(data)

    monkeypatch.setattr(native_sequence, "table_source", source_factory)
    config = yaml.safe_load(Path("recipes/blueprint_v2/config.smoke.yaml").read_text())
    config["data"].update(
        loader="kumo_pytorch", load_labels=False, split_protocol="precomputed", sequence_length=16
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    if not finite_training_amount:
        from tfm.contracts.errors import SchemaError

        with pytest.raises(SchemaError, match="finite"):
            run_command("train", config_path=path, run_dir=tmp_path / "run", device="cpu")
        assert not (tmp_path / "run/model.pt").exists()
        return
    out = run_command("train", config_path=path, run_dir=tmp_path / "run", device="cpu")
    import pyarrow.parquet as pq

    anchors = pq.read_table(out / "anchors.parquet").to_pylist()
    assert {row["view"] for row in anchors} == {"train", "validation", "full"}
    assert len({row["cutoff"] for row in anchors}) == 1
    assert (out / "model.pt").is_file()


@pytest.mark.parametrize("variant", ["smoke", "graph_smoke", "hybrid_smoke", "stock_sequence"])
def test_source_loader_import_provider_runs_shared_recipe_pipeline(variant, monkeypatch, tmp_path):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    if variant in {"graph_smoke", "hybrid_smoke"}:
        pytest.importorskip("kumo_core.modeling.gnn.hetero")
    import sys
    from types import ModuleType

    from tfm.data.loaders import SourceLoader
    from tfm.extras.kumo_pytorch import temporary_engine

    created, closed = [], []

    def engine(resources, *, work_dir, num_threads):
        assert work_dir.is_dir() and num_threads > 0
        created.append(work_dir)
        resources.callback(lambda: closed.append(work_dir))
        return temporary_engine(resources, num_threads=2)

    module = ModuleType("contract_recipe_engine")
    module.provider = SourceLoader(engine, {"implementation": "contract-engine", "revision": "v1"})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = yaml.safe_load(
        Path(
            "recipes/blueprint_v2/config.smoke.yaml"
            if variant == "stock_sequence"
            else f"recipes/relational_inputs/config.{variant}.yaml"
        ).read_text()
    )
    config["data"]["loader"] = "contract_recipe_engine:provider"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    assert created and len(closed) == len(created)
    manifest = json.loads((out / "model.manifest.json").read_text())
    assert manifest["input_loader"]["name"] == "contract_recipe_engine:provider"
    assert manifest["input_loader"]["engine_provider"]["revision"] == "v1"
    if variant == "smoke":
        module.provider = SourceLoader(
            engine, {"implementation": "contract-engine", "revision": "v2"}
        )
        with pytest.raises(ValueError, match="source/anchor identity"):
            run_command(
                "evaluate", config_path=path, run_dir=out, device="cpu", reuse_checkpoint=True
            )
        assert len(closed) == len(created)


def test_source_loader_rejects_missing_anchor_capability_before_training(monkeypatch, tmp_path):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    import sys
    from types import ModuleType, SimpleNamespace

    from tfm.contracts.errors import UnsupportedCombinationError
    from tfm.data.loaders import SourceLoader

    closed = []

    def engine(resources, *, work_dir, num_threads):
        resources.callback(lambda: closed.append(True))
        return SimpleNamespace(open=lambda *args: None)

    module = ModuleType("contract_unsupported_engine")
    module.provider = SourceLoader(engine, {"implementation": "missing-anchors", "revision": "v1"})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = yaml.safe_load(Path("recipes/relational_inputs/config.smoke.yaml").read_text())
    config["data"]["loader"] = "contract_unsupported_engine:provider"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(UnsupportedCombinationError, match="AnchorTable"):
        run_command("train", config_path=path, run_dir=tmp_path / "run", device="cpu")
    assert closed == [True]
    assert not (tmp_path / "run/model.pt").exists()


@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
def test_legacy_native_sequence_controls_keep_aligned_combined_probes(
    objective, monkeypatch, tmp_path
):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.layers.gnn import SequenceBinaryModel
    from tfm.runtime import native, native_sequence

    controls, historical, probes = [], [], []
    predict, history_predict, probe = (
        native._recorded_predictions,
        native_sequence.predictions,
        native.fit_predict_probe,
    )

    def capture(model, *args, **kwargs):
        result = predict(model, *args, **kwargs)
        if isinstance(model, SequenceBinaryModel):
            controls.append(result[0])
        return result

    def capture_history(*args, **kwargs):
        result = history_predict(*args, **kwargs)
        historical.append(result)
        return result

    def capture_probe(train_x, train_y, test_x, *, seed):
        probes.append((train_x.copy(), train_y.copy(), test_x.copy(), seed))
        return probe(train_x, train_y, test_x, seed=seed)

    monkeypatch.setattr(native, "_recorded_predictions", capture)
    monkeypatch.setattr(native_sequence, "predictions", capture_history)
    monkeypatch.setattr(native, "fit_predict_probe", capture_probe)
    config = yaml.safe_load(Path("recipes/gnn_sequence/config.pretrain_smoke.yaml").read_text())
    config["training"].update(objective=objective, max_steps=1)
    config["sequence_training"]["max_steps"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    out = run_command("smoke", config_path=path, run_dir=tmp_path / "run", device="cpu")
    metrics = json.loads((out / "metrics.json").read_text())
    assert {
        "sequence_supervised_embedding",
        "sequence_supervised_combined",
        "sequence_only_embedding",
        "sequence_only_combined",
    } <= metrics.keys()

    def checked(train_raw, train_vectors, labels, test_raw, test_vectors):
        expected_train = np.concatenate((train_raw, train_vectors), 1)
        expected_test = np.concatenate((test_raw, test_vectors), 1)
        assert any(
            np.array_equal(x, expected_train)
            and np.array_equal(y, labels)
            and np.array_equal(z, expected_test)
            and seed == config["seed"]
            for x, y, z, seed in probes
        )

    assert len(controls) == 3
    checked(controls[0][3], controls[0][0], controls[0][2], controls[-1][3], controls[-1][0])
    if objective == "future_subgraph":
        assert len(historical) == 3
        checked(
            historical[0]["raw"],
            historical[0]["vectors"],
            historical[0]["labels"],
            historical[-1]["raw"],
            historical[-1]["vectors"],
        )
    else:
        assert metrics["sequence_only_combined"] == metrics["sequence_supervised_combined"]
