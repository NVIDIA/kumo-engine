# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestion validation remains independent of sequence tokenization."""

import json
import math
from dataclasses import replace

import pytest
import torch
from tests.fixtures.graph_components import adapter
from tests.recipe_smoke.test_graph_shared_slots import raw_config, run

from tfm.config import RecipeConfig
from tfm.contracts.errors import SchemaError
from tfm.data import graph_recipe
from tfm.data.adapters.result import AdapterResult
from tfm.data.graph_transactions import transaction_graph
from tfm.data.validation.schema import PROHIBITED_INPUT_KEYS


def loaded_events(raw):
    config = RecipeConfig.model_validate(raw)
    return adapter(config.data, seed=config.seed, flat_fields=False, fixture_dir=None).events


def source(monkeypatch, events):
    monkeypatch.setattr(
        graph_recipe,
        "load_adapter_events",
        lambda *args, **kwargs: AdapterResult(events=tuple(events)),
    )


@pytest.mark.parametrize("architecture", ["gnn_encoder", "gnn_sequence"])
def test_label_feature_is_rejected_before_pretraining_or_export(
    tmp_path, monkeypatch, architecture
):
    from tfm.runtime import graph

    raw = raw_config()
    raw["runtime"]["stage"] = "pretrain"
    raw["data"]["amount_field"] = "is_fraud"
    raw["model"]["composition"]["granularity"]["type"] = architecture
    if architecture == "gnn_encoder":
        raw["model"]["composition"]["objective"] = "future_subgraph"
    source(
        monkeypatch,
        [replace(e, numeric={**e.numeric, "is_fraud": 1.0}) for e in loaded_events(raw)],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid source must fail before any model or tokenizer work")

    monkeypatch.setattr(graph, "build_graph_architecture", forbidden)
    monkeypatch.setattr(graph_recipe, "tokenizer_for", forbidden)
    monkeypatch.setattr(graph_recipe, "build_entity_sequences", forbidden)
    with pytest.raises(SchemaError, match="is_fraud.*cannot be a model input"):
        run(raw, tmp_path)
    assert not (tmp_path / "run" / "encoder.pt").exists()


@pytest.mark.parametrize("column", ["numeric", "categorical", "text"])
@pytest.mark.parametrize("field", sorted(PROHIBITED_INPUT_KEYS))
@pytest.mark.parametrize("padding", ["", " "])
def test_shared_schema_rejects_unselected_label_features(monkeypatch, column, field, padding):
    raw = raw_config()
    raw["runtime"]["stage"] = "pretrain"
    events = list(loaded_events(raw))
    values = getattr(events[0], column)
    events[0] = replace(
        events[0],
        **{
            column: {
                **values,
                f"{padding}{field.upper()}{padding}": 1.0 if column == "numeric" else "yes",
            }
        },
    )
    source(monkeypatch, events)
    with pytest.raises(SchemaError, match="cannot be a model input"):
        graph_recipe.prepare_graph_data(RecipeConfig.model_validate(raw))


@pytest.mark.parametrize("field", ["entity_id", "event_id"])
def test_tokenizer_free_ingestion_retains_identity_validation(monkeypatch, field):
    raw = raw_config()
    raw["runtime"]["stage"] = "pretrain"
    events = list(loaded_events(raw))
    events[0] = replace(events[0], **{field: ""})
    source(monkeypatch, events)
    with pytest.raises(SchemaError, match=f"{field} is required"):
        graph_recipe.prepare_graph_data(RecipeConfig.model_validate(raw))


@pytest.mark.parametrize("field", sorted(PROHIBITED_INPUT_KEYS))
@pytest.mark.parametrize("padding", ["", " "])
def test_selected_amount_guard_uses_shared_prohibited_fields(field, padding):
    with pytest.raises(ValueError, match="amount_field"):
        transaction_graph([], amount_field=f"{padding}{field.upper()}{padding}")


@pytest.mark.parametrize("stage", ["pretrain", "full", "downstream"])
@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
def test_misspelled_amount_is_rejected_before_training(tmp_path, monkeypatch, stage, objective):
    from tfm.runtime import graph

    raw = raw_config()
    raw["runtime"].update(
        stage=stage, encoder_checkpoint="unused.pt" if stage == "downstream" else None
    )
    raw["training"]["objective"] = objective
    raw["data"]["amount_field"] = "amout"

    def forbidden(*args, **kwargs):
        raise AssertionError("Missing training amount must fail before model construction")

    monkeypatch.setattr(graph, "build_graph_architecture", forbidden)
    with pytest.raises(SchemaError, match="No train-side event.*amout"):
        run(raw, tmp_path)
    assert not (tmp_path / "run" / "encoder.pt").exists()


@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
@pytest.mark.parametrize("available_only", ["held_out", "future"])
@pytest.mark.parametrize("unusable", [None, math.nan], ids=["absent", "nan"])
def test_amount_presence_requires_training_history(
    monkeypatch, objective, available_only, unusable
):
    raw = raw_config()
    raw["runtime"]["stage"] = "pretrain"
    raw["training"]["objective"] = objective
    events = loaded_events(raw)
    source(monkeypatch, events)
    baseline = graph_recipe.prepare_graph_data(RecipeConfig.model_validate(raw))
    owners = {a.entity_id for b in baseline.pretrain for a in b.graph.anchors}
    cutoff = max(a.cutoff for b in baseline.pretrain for a in b.graph.anchors)
    restricted = [
        replace(
            e,
            numeric={
                **{k: v for k, v in e.numeric.items() if k != "amount"},
                **({} if unusable is None else {"amount": unusable}),
            },
        )
        if (e.entity_id in owners if available_only == "held_out" else e.event_timestamp <= cutoff)
        else e
        for e in events
    ]
    assert any("amount" in e.numeric for e in restricted)
    source(monkeypatch, restricted)
    with pytest.raises(SchemaError, match="No train-side event.*amount"):
        graph_recipe.prepare_graph_data(RecipeConfig.model_validate(raw))


@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
def test_individual_missing_amounts_remain_valid_in_tokenizer_free_training(
    tmp_path, monkeypatch, objective
):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    raw = raw_config()
    raw["runtime"]["stage"] = "pretrain"
    raw["training"]["objective"] = objective
    events = []
    for i, event in enumerate(loaded_events(raw)):
        numeric = dict(event.numeric)
        if i % 5 == 0:
            numeric.pop("amount")
        elif i % 5 in (1, 2, 3):
            numeric["amount"] = (math.nan, math.inf, -math.inf)[i % 5 - 1]
        events.append(replace(event, numeric=numeric))
    source(monkeypatch, events)
    data = graph_recipe.prepare_graph_data(RecipeConfig.model_validate(raw))
    missing = [
        row
        for row in data.sampler.graph.rows
        if row.table == "events" and not row.values["present"]
    ]
    assert missing and any(
        row.values["present"] for row in data.sampler.graph.rows if row.table == "events"
    )
    assert all(
        not math.isfinite(row.values["amount"]) and row.values["amount_log"] == 0 for row in missing
    )
    assert data.tokenizer is None
    out = run(raw, tmp_path)
    assert (out / "encoder.pt").is_file()
    weights = torch.load(out / "encoder.pt", weights_only=True, map_location="cpu")
    assert all(torch.isfinite(value).all() for value in weights.values())


@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
def test_hybrid_label_text_rejected_before_tokenizer_fit(tmp_path, monkeypatch, objective):
    from tfm.runtime import graph

    raw = raw_config()
    raw["training"]["objective"] = objective
    source(
        monkeypatch, [replace(e, text={**e.text, "is_fraud": "yes"}) for e in loaded_events(raw)]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Label text must not reach tokenizer fitting or model construction")

    monkeypatch.setattr(graph_recipe, "tokenizer_for", forbidden)
    monkeypatch.setattr(graph, "build_graph_architecture", forbidden)
    with pytest.raises(SchemaError, match="Text field 'is_fraud'.*cannot be a model input"):
        run(raw, tmp_path)
    assert not (tmp_path / "run" / "encoder.pt").exists()


@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
@pytest.mark.parametrize("stage", ["pretrain", "full", "downstream"])
@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf], ids=["nan", "inf", "negative-inf"]
)
@pytest.mark.parametrize("amount_field", ["amount", "gross_amount"])
def test_nonfinite_training_amount_rejected_before_export(
    tmp_path, monkeypatch, objective, stage, value, amount_field
):
    from tfm.runtime import graph

    raw = raw_config()
    raw["runtime"].update(
        stage=stage, encoder_checkpoint="unused.pt" if stage == "downstream" else None
    )
    raw["training"]["objective"] = objective
    raw["data"]["amount_field"] = amount_field
    # A finite value under another numeric name must not rescue the selected field.
    source(
        monkeypatch,
        [replace(e, numeric={**e.numeric, amount_field: value}) for e in loaded_events(raw)],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unusable training amounts must fail before model construction")

    monkeypatch.setattr(graph, "build_graph_architecture", forbidden)
    with pytest.raises(
        SchemaError, match=f"No train-side event.*finite numeric amount_field '{amount_field}'"
    ):
        run(raw, tmp_path)
    assert not (tmp_path / "run" / "encoder.pt").exists()


@pytest.mark.parametrize("architecture", ["gnn_encoder", "gnn_sequence"])
@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
@pytest.mark.parametrize("amount_field", ["amount", "gross_amount"])
def test_mixed_amounts_train_and_evaluate_with_checkpoint_reuse(
    tmp_path, monkeypatch, architecture, objective, amount_field
):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.runtime import graph

    raw = raw_config()
    raw["training"]["objective"] = objective
    raw["model"]["composition"]["granularity"]["type"] = architecture
    if architecture == "gnn_encoder":
        raw["model"]["composition"]["objective"] = objective
    raw["data"].update(
        adapter="tests.fixtures.graph_components:MIXED_AMOUNT_ADAPTER", amount_field=amount_field
    )
    data = graph_recipe.prepare_graph_data(RecipeConfig.model_validate(raw))
    for batches in data.probes.values():
        for batch in batches:
            assert torch.isfinite(batch.history.numeric_features).all()
    labels = torch.cat([batch.labels for batch in data.probes["train"] if batch.labels is not None])
    assert set(labels.tolist()) == {0.0, 1.0}
    out = run(raw, tmp_path)
    weights_before = (out / "model.pt").read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluate with reuse must not retrain the checkpoint")

    monkeypatch.setattr(graph, "fit_selected", forbidden)
    run(raw, tmp_path, command="evaluate", reuse=True)
    metrics = json.loads((out / "metrics.json").read_text())
    assert {"raw", "embedding", "combined"}.issubset(metrics)
    assert metrics["raw"]["n_eval"] > 0
    assert (out / "model.pt").read_bytes() == weights_before
