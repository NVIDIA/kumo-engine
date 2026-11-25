# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real DiskGraph integration against the Python selection oracle."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import torch

from tfm.config import load_recipe_config
from tfm.contracts.graph import GraphAnchor
from tfm.contracts.schema import TransactionEvent
from tfm.data import native_history as native
from tfm.data.graph import GraphLink, GraphRow, GraphSampler, GraphTable, RelationalGraph
from tfm.data.native_graph import NativeGraphSampler

NOW = datetime(2024, 1, 2, tzinfo=UTC)


def events():
    return [
        TransactionEvent(
            owner, str(i), NOW + timedelta(microseconds=offset), numeric={"amount": float(i)}
        )
        for owner in ("a", "b")
        for i, offset in enumerate((-2_000_000, -1_000_000, -1, 0, 1, 700_000))
    ]


def test_loader_default_and_explicit_reference():
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    assert config.data.loader == "kumo_pytorch"
    assert config.data.model_copy(update={"loader": "python"}).loader == "python"


def test_histories_keep_exact_time_boundaries_and_ties():
    rows = events()
    rows.append(replace(rows[0], event_id="tied"))
    for window in (None, timedelta(seconds=1)):
        for limit in (0, 2, 20):
            args = dict(max_events=limit, observation_window=window)
            assert native.build_entity_sequences(
                rows, NOW, **args
            ) == native.build_entity_sequences(rows, NOW, loader="python", **args)
    for size in (1, 3, 20):
        assert native.build_entity_windows(
            rows, NOW, window_events=size
        ) == native.build_entity_windows(rows, NOW, window_events=size, loader="python")
    labels = [
        SimpleNamespace(entity_id=owner, timestamp=at)
        for owner, at in (("b", NOW), ("a", NOW - timedelta(seconds=1)), ("a", NOW))
    ]
    assert native.build_sequences_at_label_cutoffs(
        rows, labels, max_events=10
    ) == native.build_sequences_at_label_cutoffs(rows, labels, max_events=10, loader="python")


@pytest.mark.parametrize(
    "name", ["transaction_field_tokenizer", "flat_key_value", "key_value_time", "event_fields"]
)
def test_indexed_native_batches_preserve_window_membership(name):
    from tfm.data.collate import collate_batch
    from tfm.features.tokenizers import build_tokenizer

    rows = native.build_entity_windows(events(), NOW, window_events=3, loader="python")
    tokenizer = build_tokenizer(name)
    tokenizer.fit(rows)
    batches = native.NativeSequenceBatches(
        rows, tokenizer, batch_size=2, max_tokens=64, seed=13, amount_key="amount"
    )
    try:
        for i in reversed(range(len(batches))):
            actual = batches[i]
            expected = collate_batch(
                batches.rows[i * 2 : (i + 1) * 2], tokenizer, max_tokens=64, amount_key="amount"
            )

            def compare(a, b):
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                elif isinstance(a, Mapping):
                    assert set(a) == set(b)
                    for key in a:
                        compare(a[key], b[key])
                else:
                    assert a == b

            for field in fields(actual):
                if not field.name.startswith("_"):
                    compare(getattr(actual, field.name), getattr(expected, field.name))
        assert len(batches[:]) == len(batches)
        with pytest.raises(IndexError):
            _ = batches[len(batches)]
    finally:
        batches.close()


def graph_fixture():
    role = ("users", "activity", "events")
    return RelationalGraph(
        (GraphTable("users", ("x",), immutable=True), GraphTable("events", ("x",))),
        (role,),
        (
            GraphRow("users", "a", {"x": 1.0}),
            GraphRow("users", "b", {"x": 2.0}),
            GraphRow("users", "cold", {"x": 0.0}),
        )
        + tuple(
            GraphRow(
                "events",
                str(i),
                {"x": float(i)},
                NOW + timedelta(seconds=i),
                NOW + timedelta(seconds=i),
            )
            for i in (-2, -1, 0, 1)
        ),
        tuple(GraphLink(role, "a", str(i), NOW + timedelta(seconds=i)) for i in (-2, -1, 0, 1)),
        frozenset({"fraud"}),
        NOW + timedelta(days=1),
    )


@pytest.mark.parametrize("sharing", ["shared", "disjoint"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_native_graph_matches_uncapped_reference(sharing, depth):
    graph = graph_fixture()
    anchors = tuple(GraphAnchor(str(i), owner, NOW) for i, owner in enumerate(("a", "a", "cold")))
    sampler = NativeGraphSampler(graph, depth=depth, fanout=20)
    try:
        actual = sampler.sample(anchors, "users", sharing=sharing)
        expected = GraphSampler(graph, depth=depth, fanout=20).sample(
            anchors, "users", sharing=sharing
        )
        actual.assert_validated()
        assert actual.anchors == expected.anchors
        for name in actual.node_features:
            # Native row order is independent; compare ownership + features + clocks.
            def rows(batch, name=name):
                return sorted(
                    zip(
                        batch.node_sample[name].tolist(),
                        batch.node_features[name].tolist(),
                        batch.node_available_us[name].tolist(),
                        strict=True,
                    )
                )

            assert rows(actual) == rows(expected)
        assert _canonical_edges(actual) == _canonical_edges(expected)
        actual.node_features["users"].add_(1)
        with pytest.raises(ValueError):
            actual.assert_validated()
    finally:
        sampler.close()


def test_native_graph_rejects_unsupported_clocks():
    from tfm.contracts.errors import UnsupportedCombinationError

    sampler = NativeGraphSampler(graph_fixture(), depth=1, fanout=2)
    try:
        with pytest.raises(UnsupportedCombinationError, match="second-aligned"):
            sampler.sample((GraphAnchor("a", "a", NOW + timedelta(microseconds=1)),), "users")
    finally:
        sampler.close()


def test_native_anchor_fast_path_and_inference_construction(monkeypatch):
    from kumo.pytorch.engines.native_anchors import NativeAnchors

    from tfm.features.tokenizers.financial import TransactionFieldTokenizer

    def forbidden(*args, **kwargs):
        raise AssertionError("Recipe loaders must use native AnchorTable batches")

    monkeypatch.setattr(NativeAnchors, "_requests", forbidden)
    rows = native.build_entity_sequences(events(), NOW, max_events=10)
    tokenizer = TransactionFieldTokenizer()
    tokenizer.fit(rows)
    batches = native.NativeSequenceBatches(
        rows, tokenizer, batch_size=2, max_tokens=64, seed=None, amount_key="amount"
    )
    try:
        with torch.inference_mode():
            batch = batches[0]
        assert not batch.input_ids.is_inference()
    finally:
        root = batches.source.root
        index = batches.source.plan.engine.index if hasattr(batches.source.plan, "engine") else None
        batches.close()
        assert not root.exists()
        if index is not None:
            assert not index.exists()
    sampler = NativeGraphSampler(graph_fixture(), depth=2, fanout=20)
    try:
        with torch.inference_mode():
            graph = sampler.sample((GraphAnchor("x", "a", NOW),), "users")
        graph.assert_validated()
    finally:
        sampler.close()


def test_explicit_python_loader_does_not_import_native(monkeypatch):
    def forbidden():
        raise AssertionError("Explicit Python loader must be usable without native imports")

    monkeypatch.setattr(native, "native_api", forbidden)
    assert native.build_entity_sequences(events(), NOW, max_events=2, loader="python")


def test_native_and_reference_graph_model_outputs_and_gradients():
    pytest.importorskip("kumo_core.modeling.gnn.hetero")
    from tfm.compositions.gnn import GNNEncoder
    from tfm.layers.gnn import GraphEncoder

    torch.manual_seed(73)
    graph = graph_fixture()
    spec = GNNEncoder(
        tuple((t.name, len(t.features)) for t in graph.tables),
        graph.directed_relations,
        "users",
        foreign_key_relations=tuple((t, f"rev_{r}", s) for s, r, t in graph.relations),
        hidden_size=8,
        depth=2,
    )
    model = GraphEncoder(spec).eval()
    anchors = (GraphAnchor("first", "a", NOW), GraphAnchor("repeat", "a", NOW))
    native_sampler = NativeGraphSampler(graph, depth=2, fanout=20)
    try:
        actual = model(native_sampler.sample(anchors, "users"))
        actual.square().sum().backward()
        gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        expected = model(GraphSampler(graph, depth=2, fanout=20).sample(anchors, "users"))
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(gradients[name], parameter.grad, atol=1e-6, rtol=1e-5)
    finally:
        native_sampler.close()


def test_native_empty_optional_tables_keep_declared_schema():
    base = graph_fixture()
    graph = replace(
        base,
        tables=base.tables + (GraphTable("transfers", ("x",)),),
        relations=base.relations
        + (("users", "sender", "transfers"), ("users", "receiver", "transfers")),
    )
    sampler = NativeGraphSampler(graph, depth=2, fanout=10)
    try:
        batch = sampler.sample((GraphAnchor("anchor", "a", NOW),), "users")
        assert batch.node_features["transfers"].shape == (0, 1)
        for role in graph.directed_relations:
            assert role in batch.edge_index
            if "transfers" in role:
                assert batch.edge_index[role].shape == (2, 0)
        batch.assert_validated()
    finally:
        sampler.close()


def test_native_graph_checks_the_actual_sampling_clock():
    from tfm.contracts.graph import utc_us

    anchors = (GraphAnchor("s", "a", NOW),)
    metadata = {
        "entity_ids": (0,),
        "request_metadata": ({"cutoff_us": utc_us(NOW)},),
        "as_of_time": (utc_us(NOW) // 1_000_000,),
    }
    NativeGraphSampler._assert_bindings(SimpleNamespace(metadata=metadata), anchors, (0,))
    metadata["as_of_time"] = (utc_us(NOW) // 1_000_000 - 1,)
    with pytest.raises(ValueError, match="sampling clock"):
        NativeGraphSampler._assert_bindings(SimpleNamespace(metadata=metadata), anchors, (0,))


@pytest.mark.parametrize("peer", ["a", "b"])
def test_native_multiple_fk_roles_and_separate_future_targets(peer, monkeypatch):
    from tfm.contracts.graph import utc_us
    from tfm.data.graph import future_subgraph_targets

    base = graph_fixture()
    roles = (("users", "sender", "transfers"), ("users", "receiver", "transfers"))
    graph = replace(
        base,
        tables=base.tables + (GraphTable("transfers", ("x",), ("x",)),),
        relations=base.relations + roles,
        rows=base.rows
        + tuple(
            GraphRow(
                "transfers",
                name,
                {"x": value},
                NOW + timedelta(seconds=offset),
                NOW + timedelta(seconds=offset),
            )
            for name, value, offset in (("past", 5.0, -1), ("future", 99.0, 1))
        ),
        links=base.links
        + tuple(
            GraphLink(role, owner, name, NOW + timedelta(seconds=offset))
            for role, owner in zip(roles, ("a", peer), strict=True)
            for name, offset in (("past", -1), ("future", 1))
        ),
    )
    anchors = (GraphAnchor("first", "a", NOW), GraphAnchor("second", "b", NOW))
    sampler = NativeGraphSampler(graph, depth=2, fanout=20)
    reference = GraphSampler(graph, depth=2, fanout=20)
    try:
        actual, expected = sampler.sample(anchors, "users"), reference.sample(anchors, "users")
        assert actual.node_features["transfers"].max() == 5.0
        for name in actual.node_features:
            assert sorted(actual.node_features[name].tolist()) == sorted(
                expected.node_features[name].tolist()
            )
        assert _canonical_edges(actual) == _canonical_edges(expected)

        def forbidden(*args, **kwargs):
            raise AssertionError("Future target construction must not sample encoder inputs")

        monkeypatch.setattr(sampler, "sample", forbidden)
        kwargs = dict(
            horizon_us=2_000_000, depth=2, partition_end_us=utc_us(NOW + timedelta(days=1))
        )
        native_target = future_subgraph_targets(sampler, anchors, "users", **kwargs)
        reference_target = future_subgraph_targets(reference, anchors, "users", **kwargs)
        torch.testing.assert_close(native_target.values, reference_target.values, equal_nan=True)
        torch.testing.assert_close(native_target.valid, reference_target.valid)
    finally:
        sampler.close()


@pytest.mark.parametrize("failure", ["manifest", "trainer", "progress", "fit"])
def test_native_training_resources_close_on_setup_and_fit_errors(failure, monkeypatch, tmp_path):
    from tfm.features.tokenizers.financial import TransactionFieldTokenizer
    from tfm.runtime import pipeline

    rows = native.build_entity_sequences(events(), NOW, max_events=10, loader="python")
    tokenizer = TransactionFieldTokenizer()
    tokenizer.fit(rows)
    batches = native.NativeSequenceBatches(
        rows, tokenizer, batch_size=2, max_tokens=64, seed=None, amount_key="amount"
    )
    source, index = batches.source.root, batches.source.plan.engine.index

    def fail(*args, **kwargs):
        raise RuntimeError(failure)

    monkeypatch.setattr(pipeline, "training_batches", lambda artifacts: (batches, {}))
    monkeypatch.setattr(pipeline, "is_primary_process", lambda: True)
    monkeypatch.setattr(
        pipeline, "write_json", fail if failure == "manifest" else lambda *a, **k: None
    )
    monkeypatch.setattr(
        pipeline,
        "build_trainer",
        fail if failure == "trainer" else lambda *a, **k: SimpleNamespace(fit=fail),
    )
    monkeypatch.setattr(
        pipeline, "TrainingProgress", fail if failure == "progress" else lambda *a, **k: None
    )
    monkeypatch.setattr(pipeline, "stage", lambda *a: None)
    config = SimpleNamespace(
        seed=1,
        training=SimpleNamespace(
            backend="eager",
            log_every=1,
            batch_size=2,
            max_steps=1,
            learning_rate=0.01,
            optimizer=None,
            shuffle=False,
            precision="fp32",
        ),
    )
    artifacts = SimpleNamespace(
        config=config, run_dir=tmp_path, model=None, device=torch.device("cpu")
    )
    with pytest.raises(RuntimeError, match=failure):
        pipeline.run_train(artifacts)
    assert not source.exists()
    assert not index.exists()


def _canonical_edges(batch):
    def node(name, index):
        return (
            int(batch.node_sample[name][index]),
            tuple(batch.node_features[name][index].tolist()),
            int(batch.node_available_us[name][index]),
        )

    return {
        relation: sorted((node(relation[0], a), node(relation[2], b)) for a, b in edge.T.tolist())
        for relation, edge in batch.edge_index.items()
    }


def test_native_graph_sampling_does_not_rescan_links():
    class Counted(tuple):
        visits = 0

        def __iter__(self):
            for item in super().__iter__():
                self.visits += 1
                yield item

    base = graph_fixture()
    links = Counted(base.links)
    sampler = NativeGraphSampler(replace(base, links=links), depth=2, fanout=20)
    try:
        initial = links.visits
        for i in range(4):
            batch = sampler.sample((GraphAnchor(str(i), "a", NOW),), "users")
            assert batch.edge_index[base.relations[0]].shape[1] == 3
        assert links.visits == initial
    finally:
        sampler.close()


def test_native_graph_compares_clock_instants_across_dst_folds():
    from zoneinfo import ZoneInfo

    from tfm.contracts.errors import UnsupportedCombinationError

    first = datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/New_York"), fold=0)
    second = first.replace(fold=1)
    base = graph_fixture()
    row = replace(base.rows[3], available_at=second, event_time=first)
    with pytest.raises(UnsupportedCombinationError, match="clocks"):
        NativeGraphSampler(
            replace(base, rows=base.rows[:3] + (row,) + base.rows[4:]), depth=1, fanout=2
        )
    sampler = NativeGraphSampler(base, depth=1, fanout=2)
    try:
        batch = sampler.sample(
            (GraphAnchor("local", "a", first), GraphAnchor("utc", "a", first.astimezone(UTC))),
            "users",
            sharing="shared",
        )
        batch.assert_validated()
        with pytest.raises(ValueError, match="identical cutoffs"):
            sampler.sample(
                (GraphAnchor("first", "a", first), GraphAnchor("second", "a", second)),
                "users",
                sharing="shared",
            )
    finally:
        sampler.close()


def test_forecast_missing_owner_matches_reference_error():
    from tfm.contracts.errors import SchemaError

    labels = [SimpleNamespace(entity_id="absent", timestamp=NOW)]
    for loader in ("python", "kumo_pytorch"):
        with pytest.raises(SchemaError):
            native.build_sequences_at_label_cutoffs(events(), labels, max_events=10, loader=loader)
    # Existing owners with no history at their cutoff are intentionally skipped.
    labels = [SimpleNamespace(entity_id="a", timestamp=NOW - timedelta(days=1))]
    assert native.build_sequences_at_label_cutoffs(events(), labels, max_events=10) == []


def test_native_history_binding_is_atomic_and_single_use(monkeypatch):
    with native.NativeHistories(events()) as source:
        materialize = source.plan.materialize

        def fail(*args, **kwargs):
            raise RuntimeError("materialize")

        monkeypatch.setattr(source.plan, "materialize", fail)
        with pytest.raises(RuntimeError, match="materialize"):
            source.bind([("a", NOW)], ["train"])
        assert not hasattr(source, "requests")
        monkeypatch.setattr(source.plan, "materialize", materialize)
        task = source.bind([("a", NOW)], ["train"])
        batch = next(iter(task.batches(split="train", batch_size=1, output_format="model")))
        expected = source.decode(batch)
        with pytest.raises(ValueError, match="only bind once"):
            source.bind([("b", NOW)], ["train"])
        assert source.decode(batch) == expected


@pytest.mark.parametrize("window_events", [3, 20])
def test_native_windows_use_one_stage_and_bounded_history_with_repeated_ids(window_events):
    from tfm.data.collate import collate_batch
    from tfm.features.tokenizers.financial import TransactionFieldTokenizer

    values = [
        TransactionEvent("a", "repeated", NOW - timedelta(seconds=1), numeric={"amount": float(i)})
        for i in range(21)
    ]
    rows = native.build_entity_windows(values, NOW, window_events=window_events, loader="python")
    tokenizer = TransactionFieldTokenizer()
    tokenizer.fit(rows)
    batches = native.NativeSequenceBatches(
        rows, tokenizer, batch_size=2, max_tokens=64, seed=42, amount_key="amount"
    )
    try:
        assert len(set(batches.task._anchors.splits())) == 1
        assert max(Counter(e.entity_id for e in batches.source.events).values()) == window_events
        for i in reversed(range(len(batches))):
            actual = batches[i]
            expected = collate_batch(
                batches.rows[2 * i : 2 * i + 2], tokenizer, max_tokens=64, amount_key="amount"
            )
            torch.testing.assert_close(actual.input_ids, expected.input_ids, rtol=0, atol=0)
    finally:
        batches.close()


def test_native_graph_preparation_failure_closes_index(monkeypatch):
    from tfm.data import graph_recipe

    created = []

    class Tracked(NativeGraphSampler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    def fail(*args, **kwargs):
        raise RuntimeError("later preparation failure")

    monkeypatch.setattr(graph_recipe, "NativeGraphSampler", Tracked)
    monkeypatch.setattr(graph_recipe, "_require_training_amount", fail)
    config = load_recipe_config("recipes/gnn_encoder/config.smoke.yaml")
    with pytest.raises(RuntimeError, match="later preparation failure"):
        graph_recipe.prepare_graph_data(config)
    assert len(created) == 1
    assert not created[0].native_root.exists()
