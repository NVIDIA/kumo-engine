# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph recipes honor shared amount, position, stage and cache contracts."""

import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
import torch
import yaml

from tfm.cli.recipe import run_command
from tfm.config import RecipeConfig
from tfm.data.collate import raw_feature_vector
from tfm.data.graph_recipe import prepare_graph_data
from tfm.data.graph_transactions import transaction_graph
from tfm.data.sequence_builders import build_entity_sequences
from tfm.position.none import NoPE
from tfm.position.time_rope import TimeRoPE

FIXTURE = "tests.fixtures.graph_components"


def raw_config():
    raw = yaml.safe_load(Path("recipes/gnn_sequence/config.pretrain_smoke.yaml").read_text())
    raw["training"]["max_steps"] = 1
    raw["sequence_training"]["max_steps"] = 1
    return raw


def run(raw, tmp_path, name="run", command="train", reuse=False):
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    return run_command(
        command,
        config_path=config_path,
        run_dir=tmp_path / name,
        device="cpu",
        reuse_checkpoint=reuse,
    )


@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
def test_configured_amount_reaches_graph_targets_and_raw_probes(objective):
    from tests.fixtures.graph_components import gross_amount_adapter

    raw = raw_config()
    raw["training"]["objective"] = objective
    raw["data"].update(adapter=f"{FIXTURE}:GROSS_AMOUNT_ADAPTER", amount_field="gross_amount")
    config = RecipeConfig.model_validate(raw)
    data = prepare_graph_data(config)
    source = gross_amount_adapter(
        config.data, seed=config.seed, flat_fields=False, fixture_dir=None
    )
    expected = {
        json.dumps([e.entity_id, e.event_id], ensure_ascii=False, separators=(",", ":")): e.numeric[
            "gross_amount"
        ]
        for e in source.events
    }
    for row in data.sampler.graph.rows:
        if row.table == "events":
            amount = expected[row.row_id]
            assert row.values == {
                "amount": amount,
                "amount_log": math.log1p(amount),
                "present": 1.0,
            }
    histories = {
        s.entity_id: s
        for s in build_entity_sequences(
            source.events,
            datetime.fromisoformat(config.data.probe_cutoff.replace("Z", "+00:00")),
            max_events=config.data.sequence_length,
        )
    }
    for batches in data.probes.values():
        for batch in batches:
            expected_raw = torch.tensor(
                [
                    raw_feature_vector(histories[a.entity_id], amount_key="gross_amount")
                    for a in batch.graph.anchors
                ]
            )
            torch.testing.assert_close(batch.history.numeric_features, expected_raw, rtol=0, atol=0)
    # Renaming the source leaves canonical targets and train-only normalization identical.
    canonical_raw = raw_config()
    canonical_raw["training"] = raw["training"]
    canonical_raw["data"]["adapter"] = f"{FIXTURE}:ADAPTER"
    canonical = prepare_graph_data(RecipeConfig.model_validate(canonical_raw))
    if objective == "future_subgraph":
        assert data.transform.input_columns == canonical.transform.input_columns
        torch.testing.assert_close(data.transform.mean, canonical.transform.mean, rtol=0, atol=0)
        for actual, expected_batch in zip(data.pretrain, canonical.pretrain, strict=True):
            torch.testing.assert_close(
                actual.targets.values, expected_batch.targets.values, rtol=0, atol=0
            )


def test_amount_identity_rejected_before_encoder_installation(tmp_path, monkeypatch):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    raw = raw_config()
    raw["data"].update(adapter=f"{FIXTURE}:GROSS_AMOUNT_ADAPTER", amount_field="gross_amount")
    raw["runtime"]["stage"] = "pretrain"
    out = run(raw, tmp_path, "encoder")
    for name in ("model.manifest.json", "encoder.manifest.json"):
        assert json.loads((out / name).read_text())["amount_field"] == "gross_amount"
    raw["runtime"].update(stage="downstream", encoder_checkpoint=str(out / "encoder.pt"))
    raw["data"].update(amount_field="amount", adapter=f"{FIXTURE}:ADAPTER")

    def forbidden(*args, **kwargs):
        raise AssertionError("Mismatched source must be rejected before installing weights")

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
    with pytest.raises(ValueError, match="amount_field"):
        run(raw, tmp_path, "downstream")


@pytest.mark.parametrize("field", ["fraud", "LABEL", "chargeback", "", " ", "\t"])
def test_amount_source_cannot_alias_forbidden_fields(field):
    with pytest.raises(ValueError, match="amount_field"):
        transaction_graph([], amount_field=field)


@pytest.mark.parametrize("position,kind", [("nope", NoPE), ("time_rope", TimeRoPE)])
def test_position_slot_controls_hybrid_and_both_sequence_arms(
    tmp_path, monkeypatch, position, kind
):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.layers.gnn import GNNSequenceModel, SequenceBinaryModel
    from tfm.layers.profile_event_decoder import ProfileEventDecoderModel
    from tfm.runtime import graph

    raw = raw_config()
    raw["model"]["composition"]["position"] = position
    models = []
    fit = graph.fit_selected

    def capture(model, *args, **kwargs):
        models.append(model)
        return fit(model, *args, **kwargs)

    monkeypatch.setattr(graph, "fit_selected", capture)
    out = run(raw, tmp_path)
    history = prepare_graph_data(RecipeConfig.model_validate(raw)).probes["validation"][0].history
    changed = replace(history, elapsed_log=history.elapsed_log * 9 + 0.7)
    seen = set()
    for model in models:
        if isinstance(model, (GNNSequenceModel, SequenceBinaryModel, ProfileEventDecoderModel)):
            seen.add(type(model))
            assert isinstance(model.position_spec, kind)
            model.eval()
            encode = (
                model.encode
                if isinstance(model, ProfileEventDecoderModel)
                else model.encode_history
            )
            with torch.no_grad():
                original, altered = encode(history)[0], encode(changed)[0]
            if position == "nope":
                torch.testing.assert_close(original, altered, rtol=0, atol=0)
            else:
                assert not torch.equal(original, altered)
    assert seen == {GNNSequenceModel, SequenceBinaryModel, ProfileEventDecoderModel}
    assert json.loads((out / "model.manifest.json").read_text())["sequence_position"] == position
    run(raw, tmp_path, command="evaluate", reuse=True)
    # A missing identity must fail before any bundle state is installed.
    payload = torch.load(out / "model.pt", weights_only=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("Position identity must be checked before loading weights")

    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)
    for identity in ("sequence_position", "amount_field"):
        invalid = {
            **payload,
            "metadata": {k: v for k, v in payload["metadata"].items() if k != identity},
        }
        torch.save(invalid, out / "model.pt")
        with pytest.raises(ValueError, match=identity):
            run(raw, tmp_path, command="evaluate", reuse=True)


@pytest.mark.parametrize("architecture", ["gnn_encoder", "gnn_sequence"])
@pytest.mark.parametrize("objective", ["binary", "future_subgraph"])
def test_pretrain_has_no_tokenizer_dependency(tmp_path, monkeypatch, architecture, objective):
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.api.graph import graph_specs
    from tfm.data import graph_recipe
    from tfm.runtime import graph

    raw = raw_config()
    raw["runtime"]["stage"] = "pretrain"
    raw["training"]["objective"] = objective
    raw["model"]["composition"]["granularity"]["type"] = architecture
    if architecture == "gnn_encoder":
        raw["model"]["composition"]["objective"] = objective
    raw["data"]["adapter"] = f"{FIXTURE}:LATE_HISTORY_ADAPTER"
    raw["features"] = {"tokenizer": "nonexistent.module:UnusedTokenizer"}

    def forbidden(*args, **kwargs):
        raise AssertionError("Graph pretraining must not resolve, fit or collate a tokenizer")

    monkeypatch.setattr(graph_recipe, "tokenizer_for", forbidden)
    monkeypatch.setattr(graph, "tokenizer_class", forbidden)
    if objective == "future_subgraph":
        monkeypatch.setattr(graph_recipe, "build_entity_sequences", forbidden)
    data = prepare_graph_data(RecipeConfig.model_validate(raw))
    assert data.tokenizer is None and graph_specs(RecipeConfig.model_validate(raw), data)[1] is None
    assert not data.probes and not data.hybrid_train and not data.sequence_pretrain
    assert data.split_manifest["codec_fit_cutoff"] is None
    out = run(raw, tmp_path)
    for name in ("model.manifest.json", "encoder.manifest.json"):
        manifest = json.loads((out / name).read_text())
        assert "tokenizer" not in manifest and "tokenizer_factory" not in manifest
    assert (out / "encoder.pt").exists()


def test_binary_pretrain_preserves_truncated_history_labels():
    raw = raw_config()
    raw["training"]["objective"] = "binary"
    full = prepare_graph_data(RecipeConfig.model_validate(raw))
    raw["runtime"]["stage"] = "pretrain"
    pretrain = prepare_graph_data(RecipeConfig.model_validate(raw))
    for actual, expected in zip(pretrain.pretrain, full.pretrain, strict=True):
        assert actual.graph.anchors == expected.graph.anchors
        torch.testing.assert_close(actual.labels, expected.labels, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["gnn_encoder", "gnn_sequence"])
@pytest.mark.parametrize("stage", ["pretrain", "full", "downstream"])
def test_graph_cache_is_rejected_before_source_loading(architecture, stage):
    raw = raw_config()
    raw["model"]["composition"]["granularity"]["type"] = architecture
    raw["runtime"].update(
        stage=stage, encoder_checkpoint="unused.pt" if stage == "downstream" else None
    )
    raw["data"]["token_cache"] = "/nonexistent/cache"
    with pytest.raises(ValueError, match="token_cache"):
        RecipeConfig.model_validate(raw)


def test_graph_position_default_is_explicit():
    raw = raw_config()
    del raw["model"]["composition"]["position"]
    assert RecipeConfig.model_validate(raw).model.composition.position == "time_rope"
    for family in ("gnn_encoder", "gnn_sequence"):
        for path in Path("recipes", family).glob("config*.yaml"):
            assert (
                yaml.safe_load(path.read_text())["model"]["composition"]["position"] == "time_rope"
            )
