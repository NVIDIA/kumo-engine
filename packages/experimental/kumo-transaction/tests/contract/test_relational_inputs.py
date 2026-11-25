# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared relational ownership and anchor semantics on generic synthetic rows."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from tfm.config import RelationalTaskConfig
from tfm.contracts.graph import GraphAnchor
from tfm.contracts.relational import (
    OwnershipVersion,
    RelationalDataset,
    RelationalForeignKey,
    RelationalTable,
)
from tfm.data.relational import RelationalSource, key_token

START = datetime(2024, 1, 1, tzinfo=UTC)


def day(n):
    return START + timedelta(days=n)


def declaration():
    return RelationalDataset(
        (
            RelationalTable("entities", ("id",), immutable=True),
            RelationalTable("containers", ("id",), event_time="created", available_at="known"),
            RelationalTable(
                "activity",
                ("id",),
                numeric={"amount": "value"},
                event_time="at",
                available_at="known",
            ),
        ),
        (
            RelationalForeignKey("owns", "entities", "containers", ("owner",), ownership=True),
            RelationalForeignKey("activity", "containers", "activity", ("container",)),
        ),
        {
            "entities": ({"id": "a"}, {"id": "b"}),
            "containers": ({"id": "g", "owner": "a", "created": day(0), "known": day(1)},),
            "activity": (
                {"id": "early", "container": "g", "value": 1.0, "at": day(2), "known": day(3)},
                {"id": "later", "container": "g", "value": 2.0, "at": day(5), "known": day(6)},
                {
                    "id": "not_observed",
                    "container": "g",
                    "value": 999.0,
                    "at": day(6),
                    "known": day(12),
                },
            ),
        },
        (
            OwnershipVersion("owns", ("a",), ("g",), day(0), day(1), known_until=day(9)),
            OwnershipVersion("owns", ("a",), ("g",), day(0), day(9), valid_until=day(4)),
            OwnershipVersion("owns", ("b",), ("g",), day(4), day(9)),
        ),
    )


def source(ownership="event_time"):
    return RelationalSource(
        declaration(),
        RelationalTaskConfig(
            root="entities",
            history_paths={"activity": ("owns", "activity")},
            cutoffs={"predict": day(10).isoformat()},
            ownership=ownership,
        ),
    )


def amounts(source, owner, cutoff):
    history = source.history(GraphAnchor("s", key_token((owner,)), cutoff), max_events=20)
    return [event.numeric["amount"] for event in history.events]


def test_relationship_versions_must_be_known_at_prediction_cutoff():
    resolved = source()
    assert amounts(resolved, "a", day(7)) == [1.0, 2.0]
    assert amounts(resolved, "b", day(7)) == []
    assert amounts(resolved, "a", day(10)) == [1.0]
    assert amounts(resolved, "b", day(10)) == [2.0]


def test_event_time_ownership_is_distinct_from_prediction_time_ownership():
    assert amounts(source("event_time"), "a", day(10)) == [1.0]
    assert amounts(source("prediction_time"), "a", day(10)) == []
    assert amounts(source("prediction_time"), "b", day(10)) == [1.0, 2.0]


def test_timed_intermediate_observation_is_required_even_for_older_events():
    original = declaration()
    rows = dict(original.rows)
    rows["containers"] = ({**rows["containers"][0], "known": day(8)},)
    resolved = RelationalSource(replace(original, rows=rows), source().task)
    assert amounts(resolved, "a", day(7)) == []


def test_multiple_paths_do_not_duplicate_terminal_events():
    task = source().task.model_copy(
        update={"history_paths": {"first": ("owns", "activity"), "second": ("owns", "activity")}}
    )
    resolved = RelationalSource(declaration(), task)
    assert amounts(resolved, "a", day(7)) == [1.0, 2.0]


def test_generated_anchors_preserve_root_and_cutoff_identity():
    anchors = source().anchors()
    assert {item.anchor.entity_id for item in anchors} == {key_token(("a",)), key_token(("b",))}
    assert all(item.anchor.cutoff == day(10) for item in anchors)
    assert all(item.target is None for item in anchors)


def test_declarations_snapshot_rows_and_feature_mappings():
    dataset = declaration()
    with pytest.raises(TypeError):
        dataset.rows["entities"][0]["id"] = "changed"
    with pytest.raises(TypeError):
        dataset.tables[-1].numeric["target"] = "value"


@pytest.mark.parametrize("kind", ["numeric", "categorical", "text"])
@pytest.mark.parametrize("alias,column", [(" Is_Fraud ", "value"), ("safe", " TaRgEt ")])
def test_all_feature_kinds_check_aliases_and_source_columns(kind, alias, column):
    from tfm.contracts.errors import SchemaError

    dataset = declaration()
    table = replace(
        dataset.tables[-1], numeric={}, **({kind: {alias: column}} if kind != "numeric" else {})
    )
    if kind == "numeric":
        table = replace(table, numeric={alias: column})
    with pytest.raises(SchemaError, match="cannot be model inputs"):
        RelationalSource(replace(dataset, tables=(*dataset.tables[:-1], table)), source().task)


def test_overlapping_ownership_versions_are_rejected_but_adjacent_intervals_are_valid():
    from tfm.contracts.errors import SchemaError

    dataset = declaration()
    with pytest.raises(SchemaError, match="owns: ambiguous overlapping"):
        RelationalSource(
            replace(
                dataset,
                ownership_versions=(*dataset.ownership_versions, dataset.ownership_versions[0]),
            ),
            source().task,
        )
    assert amounts(source(), "a", day(10)) == [1.0]


def test_history_retains_complete_path_observation_and_masks_missing_numeric_value():
    dataset = declaration()
    rows = dict(dataset.rows)
    rows["containers"] = ({**rows["containers"][0], "known": day(8)},)
    rows["activity"] = ({**rows["activity"][0], "value": None}, *rows["activity"][1:])
    resolved = RelationalSource(replace(dataset, rows=rows), source().task)
    events = resolved.resolve_history(GraphAnchor("s", key_token(("a",)), day(10)))
    assert events[0].path_available_at == day(
        9
    )  # corrected ownership fact is later than row clocks
    import math

    assert math.isnan(events[0].event.numeric["amount"])


def test_key_identity_is_typed_and_rejects_nonfinite_values():
    from tfm.contracts.errors import SchemaError

    assert (
        len({key_token((value,)) for value in (1, 1.0, True, "1", day(0), day(0).isoformat())}) == 6
    )
    for value in (float("nan"), float("inf"), object(), None):
        with pytest.raises(SchemaError):
            key_token((value,))


def six_hop_declaration():
    """Ten recent neighbors, a boundary tie, and an alternating-direction chain."""
    tables = (RelationalTable("level0", ("id",), immutable=True),) + tuple(
        RelationalTable(
            f"level{i}",
            ("id",),
            numeric={"marker": "marker"},
            event_time="at",
            available_at="known",
        )
        for i in range(1, 7)
    )
    roles = tuple(
        RelationalForeignKey(
            f"step{i}",
            f"level{i - 1}" if i % 2 else f"level{i}",
            f"level{i}" if i % 2 else f"level{i - 1}",
            (f"fk{i}",),
        )
        for i in range(1, 7)
    )
    neighbors = tuple(
        {
            "id": f"recent{i}",
            "marker": float(i),
            "fk1": "root",
            "at": day(20 + i),
            "known": day(20 + i),
        }
        for i in range(9)
    ) + (
        {
            "id": "tie-a",
            "marker": 10.0,
            "fk1": "root",
            "fk2": "chain2",
            "at": day(10),
            "known": day(10),
        },
        {"id": "tie-b", "marker": 11.0, "fk1": "root", "at": day(10), "known": day(10)},
        {"id": "old", "marker": 12.0, "fk1": "root", "at": day(1), "known": day(1)},
        {"id": "unavailable", "marker": 13.0, "fk1": "root", "at": day(29), "known": day(40)},
    )
    neighbors = tuple({"fk2": None, **row} for row in neighbors)
    rows = {"level0": ({"id": "root"},), "level1": neighbors}
    for i in range(2, 7):
        row = {"id": f"chain{i}", "marker": float(20 + i), "at": day(1), "known": day(5)}
        if i % 2:
            row[f"fk{i}"] = f"chain{i - 1}"
        if i < 6 and (i + 1) % 2 == 0:
            row[f"fk{i + 1}"] = f"chain{i + 1}"
        rows[f"level{i}"] = (row,)
    task = RelationalTaskConfig(
        root="level0",
        history_paths={"deep": tuple(f"step{i}" for i in range(1, 7))},
        cutoffs={"predict": day(30).isoformat()},
    )
    return RelationalDataset(tables, roles, rows), task


def test_six_hops_fanout_ten_filters_availability_before_canonical_boundary_tie():
    from tfm.contracts.graph import utc_us
    from tfm.data.graph import GraphSampler

    dataset, task = six_hop_declaration()
    graph = RelationalSource(dataset, task).graph()
    sampler = GraphSampler(graph, depth=6, fanout=10)
    root = ("level0", key_token(("root",)))
    nodes, edges = sampler.membership(root, utc_us(day(30)), depth=6)
    assert {key for (table, key) in nodes if table == "level1"} == {
        key_token((name,)) for name in (*[f"recent{i}" for i in range(9)], "tie-a")
    }
    assert nodes["level6", key_token(("chain6",))] == 6
    assert ("level6", key_token(("chain6",))) not in sampler.membership(
        root, utc_us(day(30)), depth=5
    )[0]
    assert len(nodes) == 16
    assert len(edges) > 0
    # One history event reached through timed intermediate tables and both FK directions.
    history = RelationalSource(dataset, task).history(
        GraphAnchor("s", root[1], day(30)), max_events=10
    )
    assert len(history.events) == 1
    assert history.events[0].numeric["marker"] == 26.0


def test_six_hop_shared_and_disjoint_batches_preserve_repeated_root_sample_identity():
    import torch

    from tfm.data.graph import GraphSampler

    dataset, task = six_hop_declaration()
    sampler = GraphSampler(RelationalSource(dataset, task).graph(), depth=6, fanout=10)
    anchors = tuple(GraphAnchor(f"sample{i}", key_token(("root",)), day(30)) for i in range(2))
    shared = sampler.sample(anchors, task.root, sharing="shared")
    disjoint = sampler.sample(anchors, task.root, sharing="disjoint")
    assert shared.anchors == disjoint.anchors == anchors
    assert shared.root_index.tolist() == [0, 0]
    assert disjoint.root_index.tolist() == [0, 1]
    assert sum(len(value) for value in shared.node_features.values()) == 16
    assert sum(len(value) for value in disjoint.node_features.values()) == 32
    for name in shared.node_features:
        assert torch.equal(
            disjoint.node_features[name][disjoint.node_sample[name] == 0],
            shared.node_features[name],
        )
    mixed = (anchors[0], replace(anchors[1], cutoff=day(15)))
    with pytest.raises(ValueError, match="identical cutoffs"):
        sampler.sample(mixed, task.root, sharing="shared")
    separate = sampler.sample(mixed, task.root)
    assert separate.anchors == mixed
    shared.node_features[task.root].add_(1)
    with pytest.raises(ValueError, match="mutat|changed|modified"):
        shared.assert_validated()


def test_fixed_graph_projection_keeps_independent_row_and_link_clocks():
    from tfm.contracts.graph import utc_us
    from tfm.data.graph import GraphSampler

    original = declaration()
    dataset = replace(original, ownership_versions=())
    rows = dict(dataset.rows)
    rows["containers"] = ({**rows["containers"][0], "known": day(8), "link_known": day(9)},)
    dataset = replace(
        dataset,
        rows=rows,
        relationships=(
            replace(dataset.relationships[0], available_at="link_known"),
            dataset.relationships[1],
        ),
    )
    resolved = RelationalSource(dataset, source().task)
    graph = resolved.graph()
    sampler = GraphSampler(graph, depth=2, fanout=10)
    root = ("entities", key_token(("a",)))
    assert len(sampler.membership(root, utc_us(day(8)), depth=2)[0]) == 1
    nodes, _ = sampler.membership(root, utc_us(day(10)), depth=2)
    assert {key for table, key in nodes if table == "activity"} == {
        key_token(("early",)),
        key_token(("later",)),
    }
    assert resolved.resolve_history(GraphAnchor("s", root[1], day(10)))[0].path_available_at == day(
        9
    )


def test_unsupported_graph_projection_does_not_flatten_versioned_ownership():
    from tfm.contracts.errors import UnsupportedCombinationError

    with pytest.raises(UnsupportedCombinationError, match="Graph path owns.*version-expanded"):
        source().graph()


def test_native_six_hop_frontiers_match_reference_with_canonical_fanout():
    import torch

    pytest.importorskip("kumo.pytorch.api", reason="requires optional native extra")
    from tfm.data.graph import GraphSampler
    from tfm.data.native_relational_graph import NativeRelationalGraphSampler

    dataset, task = six_hop_declaration()
    graph = RelationalSource(dataset, task).graph()
    native = NativeRelationalGraphSampler(graph, depth=6, fanout=10)
    reference = GraphSampler(graph, depth=6, fanout=10)
    try:
        anchors = tuple(GraphAnchor(f"s{i}", key_token(("root",)), day(30)) for i in range(2))
        for sharing in ("shared", "disjoint"):
            expected = reference.sample(anchors, task.root, sharing=sharing)
            actual = native.sample(anchors, task.root, sharing=sharing)
            assert actual.anchors == expected.anchors
            assert torch.equal(actual.root_index, expected.root_index)
            assert actual.num_sampled_nodes == expected.num_sampled_nodes
            assert actual.num_sampled_edges == expected.num_sampled_edges
            for name in expected.node_features:
                assert torch.equal(actual.node_features[name], expected.node_features[name])
                assert torch.equal(actual.node_sample[name], expected.node_sample[name])
                assert torch.equal(actual.node_available_us[name], expected.node_available_us[name])
            for role in expected.edge_index:
                assert torch.equal(actual.edge_index[role], expected.edge_index[role])
                assert torch.equal(actual.edge_available_us[role], expected.edge_available_us[role])
    finally:
        native.close()
    assert not native.snapshots


def test_observation_window_preserves_old_structural_rows_and_original_clocks():
    from tfm.data.graph import GraphSampler

    dataset, task = six_hop_declaration()
    resolved = RelationalSource(dataset, task)
    anchor = GraphAnchor("window", key_token(("root",)), day(30), observation_start=day(0))
    graph = GraphSampler(resolved.graph(), depth=6, fanout=10).sample((anchor,), task.root)
    assert "level1" in graph.window_exempt_types
    assert "level6" not in graph.window_exempt_types
    # Activity window excludes only terminal activity, not structural row creation.
    anchor = replace(anchor, observation_start=day(10))
    graph = GraphSampler(resolved.graph(), depth=6, fanout=10).sample((anchor,), task.root)
    assert len(graph.node_features["level5"]) == 1
    assert len(graph.node_features["level6"]) == 0
    graph.assert_validated()
    assert not resolved.history(anchor, max_events=10).events


def test_native_history_projection_preserves_versioned_ownership_and_anchor_identity():
    pytest.importorskip("kumo.pytorch.api", reason="requires optional native extra")
    from tfm.data.relational import RelationalAnchor
    from tfm.data.relational_batches import retrieve_histories

    resolved = source()
    anchors = tuple(
        RelationalAnchor(GraphAnchor(f"s{i}", key_token((owner,)), cutoff), "predict")
        for i, (owner, cutoff) in enumerate(
            (("a", day(7)), ("a", day(10)), ("b", day(10)), ("a", day(10)))
        )
    )
    expected = retrieve_histories(resolved, anchors, loader="python", max_events=10)
    actual = retrieve_histories(resolved, anchors, loader="kumo_pytorch", max_events=10)
    assert actual == expected


def test_label_tables_are_excluded_even_without_a_task_label_configuration():
    dataset, task = six_hop_declaration()
    dataset = replace(
        dataset,
        tables=(
            *dataset.tables,
            RelationalTable("outcomes", ("id",), immutable=True, role="labels"),
        ),
        rows={**dataset.rows, "outcomes": ({"id": "x", "target": 1},)},
    )
    graph = RelationalSource(dataset, task).graph()
    assert "outcomes" not in {table.name for table in graph.tables}
    with pytest.raises(ValueError, match="label tables cannot declare input features"):
        RelationalTable(
            "outcomes", ("id",), immutable=True, role="labels", numeric={"outcome": "target"}
        )


def test_numeric_target_names_cannot_overwrite_encoded_graph_features():
    dataset, task = six_hop_declaration()
    tables = list(dataset.tables)
    tables[-1] = replace(tables[-1], numeric={"x": "marker", "value:x": "second"})
    rows = dict(dataset.rows)
    rows["level6"] = ({**rows["level6"][0], "second": 987.0},)
    graph = RelationalSource(replace(dataset, tables=tuple(tables), rows=rows), task).graph()
    terminal = next(row for row in graph.rows if row.table == "level6")
    assert terminal.values["value:x"] == 26.0
    assert terminal.values["value:value:x"] == 987.0
    assert terminal.values["raw:value:x"] == 987.0


def test_replay_anchors_retain_windows_and_unlabeled_predictions(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tfm.contracts.graph import SequencePredictionBatch
    from tfm.data.relational_batches import RelationalBatches
    from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer

    dataset, task = six_hop_declaration()
    path = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sample_id": "replay1",
                    "id": "root",
                    "cutoff": day(30),
                    "observation_start": day(0),
                },
                {
                    "sample_id": "replay2",
                    "id": "root",
                    "cutoff": day(30),
                    "observation_start": day(10),
                },
            ]
        ),
        path,
    )
    task = task.model_copy(update={"anchors": str(path)})
    resolved = RelationalSource(dataset, task)
    anchors = resolved.anchors()
    assert [item.anchor.sample_id for item in anchors] == ["replay1", "replay2"]
    assert anchors[1].anchor.observation_start == day(10)
    with RelationalBatches(resolved) as inputs:
        tokenizer = KeyValueTimeTokenizer()
        tokenizer.fit(inputs.histories("predict"))
        (batch,) = inputs.batches(
            "predict", mode="sequence", tokenizer=tokenizer, batch_size=2, require_labels=False
        )
        assert isinstance(batch, SequencePredictionBatch)
        assert not batch.history.event_mask[1].any()
        assert batch.target_level == "entity"
        batch.history.value_ids.add_(1)
        with pytest.raises(ValueError, match="mutat|changed|modified"):
            batch.assert_validated()


def test_root_profiles_are_visible_allowlisted_and_fit_on_training_roots_only():
    from tfm.features.tokenizers.key_value_time import KeyValueTimeTokenizer

    dataset = declaration()
    rows = dict(dataset.rows)
    rows["entities"] = (
        {"id": "a", "region": "train_region"},
        {"id": "b", "region": "heldout_region"},
    )
    tables = (replace(dataset.tables[0], categorical={"region": "region"}), *dataset.tables[1:])
    resolved = RelationalSource(replace(dataset, tables=tables, rows=rows), source().task)
    anchor = GraphAnchor("train", key_token(("a",)), day(7))
    tokenizer = KeyValueTimeTokenizer()
    tokenizer.set_profiles(resolved.profiles((anchor,)))
    tokenizer.fit((resolved.history(anchor, max_events=10),))
    assert tokenizer.encode_profile(anchor.entity_id, anchor.cutoff)[0]
    assert "heldout_region" not in str(tokenizer.state_dict())
    assert "a" not in resolved.profiles((anchor,))[anchor.entity_id].attributes.values()


def test_native_six_hop_model_outputs_gradients_and_buffers_match_reference(monkeypatch):
    import copy

    import torch

    pytest.importorskip("kumo.pytorch.api", reason="requires optional native extra")
    pytest.importorskip("kumo_core.modeling.gnn.hetero", reason="requires optional kumo extra")
    from tfm.compositions.gnn import GNNEncoder
    from tfm.contracts.graph import GraphSupervisedBatch, utc_us
    from tfm.data.graph import GraphSampler
    from tfm.data.native_relational_graph import NativeRelationalGraphSampler
    from tfm.heads.binary import Binary
    from tfm.layers.gnn import GNNEncoderModel

    torch.manual_seed(42)
    dataset, task = six_hop_declaration()
    graph = RelationalSource(dataset, task).graph()
    anchors = tuple(GraphAnchor(f"s{i}", key_token(("root",)), day(30)) for i in range(2))
    native = NativeRelationalGraphSampler(graph, depth=6, fanout=10)
    try:
        first = GraphSampler(graph, depth=6, fanout=10).sample(anchors, task.root)
        second = native.sample(anchors, task.root)
    finally:
        native.close()
    spec = GNNEncoder(
        tuple((table.name, len(table.features)) for table in graph.tables),
        graph.directed_relations,
        task.root,
        foreign_key_relations=tuple(
            (target, f"rev_{role}", source) for source, role, target in graph.relations
        ),
        hidden_size=8,
        depth=6,
    )
    models = [GNNEncoderModel(spec, objective=Binary(target_level="entity"))]
    models.append(copy.deepcopy(models[0]))
    batches = [
        GraphSupervisedBatch(
            batch,
            torch.tensor([0.0, 1.0]),
            torch.full((2,), utc_us(day(30)), dtype=torch.long),
            utc_us(day(30)),
            target_level="entity",
        )
        for batch in (first, second)
    ]
    outputs = [model(batch) for model, batch in zip(models, batches, strict=True)]
    assert torch.equal(outputs[0].embeddings, outputs[1].embeddings)
    assert torch.equal(outputs[0].loss, outputs[1].loss)
    for output in outputs:
        output.loss.backward()
    for (_, left), (_, right) in zip(
        models[0].named_parameters(), models[1].named_parameters(), strict=True
    ):
        assert (left.grad is None) == (right.grad is None)
        if left.grad is not None:
            assert torch.equal(left.grad, right.grad)
    for (_, left), (_, right) in zip(
        models[0].named_buffers(), models[1].named_buffers(), strict=True
    ):
        assert torch.equal(left, right)

    def forbidden(*args, **kwargs):
        raise AssertionError("level validation must precede encoder execution")

    monkeypatch.setattr(models[0].encoder, "forward", forbidden)
    with pytest.raises(ValueError, match="label level differs"):
        models[0](replace(batches[0], target_level="sequence"))


def test_partial_version_logs_cannot_silently_drop_fixed_children():
    from tfm.contracts.errors import SchemaError

    dataset = declaration()
    rows = dict(dataset.rows)
    rows["containers"] = (
        *rows["containers"],
        {"id": "unversioned", "owner": "a", "created": day(0), "known": day(1)},
    )
    with pytest.raises(SchemaError, match="owns: version records must cover every child"):
        RelationalSource(replace(dataset, rows=rows), source().task)


@pytest.mark.parametrize("one_row", [False, True])
def test_missing_fk_columns_are_not_silent_optional_links(one_row):
    from tfm.contracts.errors import SchemaError

    dataset = declaration()
    rows = dict(dataset.rows)
    rows["activity"] = tuple(
        {key: value for key, value in row.items() if key != "container"}
        if not one_row or i == 0
        else dict(row)
        for i, row in enumerate(rows["activity"])
    )
    with pytest.raises(SchemaError, match="activity.*FK columns are absent"):
        RelationalSource(replace(dataset, rows=rows), source().task)


def test_explicit_null_fk_remains_a_valid_optional_link():
    dataset = declaration()
    rows = dict(dataset.rows)
    rows["activity"] = ({**rows["activity"][0], "container": None}, *rows["activity"][1:])
    resolved = RelationalSource(replace(dataset, rows=rows), source().task)
    assert amounts(resolved, "a", day(7)) == [2.0]


def test_window_exemption_is_limited_to_root_and_path_intermediates():
    from tfm.data.graph import GraphSampler

    dataset = RelationalDataset(
        (
            RelationalTable("entities", ("id",), event_time="created", available_at="known"),
            RelationalTable(
                "activity",
                ("id",),
                numeric={"amount": "value"},
                event_time="at",
                available_at="known",
            ),
            RelationalTable(
                "refunds",
                ("id",),
                numeric={"amount": "value"},
                event_time="at",
                available_at="known",
            ),
        ),
        (
            RelationalForeignKey("activity", "entities", "activity", ("owner",)),
            RelationalForeignKey("refunds", "entities", "refunds", ("owner",)),
        ),
        {
            "entities": ({"id": "a", "created": day(0), "known": day(1)},),
            "activity": (
                {"id": "event", "owner": "a", "at": day(7), "known": day(7), "value": 1.0},
            ),
            "refunds": (
                {"id": "old", "owner": "a", "at": day(2), "known": day(2), "value": 11.0},
                {"id": "recent", "owner": "a", "at": day(8), "known": day(8), "value": 22.0},
            ),
        },
    )
    task = RelationalTaskConfig(
        root="entities",
        history_paths={"activity": ("activity",)},
        cutoffs={"predict": day(10).isoformat()},
    )
    graph = RelationalSource(dataset, task).graph()
    anchor = GraphAnchor("s", key_token(("a",)), day(10), observation_start=day(5))
    batch = GraphSampler(graph, depth=1, fanout=10).sample((anchor,), task.root)
    assert batch.window_exempt_types == frozenset({"entities"})
    assert batch.node_features["refunds"][:, 0].tolist() == [22.0]
    assert len(batch.node_features["entities"]) == 1


def test_label_root_cutoff_cannot_cross_splits():
    from pathlib import Path

    from tfm.config import load_recipe_config
    from tfm.contracts.errors import SchemaError
    from tfm.data.adapters.registry import load_adapter_events

    config = load_recipe_config(Path("recipes/relational_inputs/config.smoke.yaml"))
    dataset = load_adapter_events(config, flat_fields=False).relational
    rows = dict(dataset.rows)
    rows["outcomes"] = (
        *rows["outcomes"],
        {**rows["outcomes"][0], "id": "duplicate", "split": "validation"},
    )
    with pytest.raises(SchemaError, match="root/cutoff appears in more than one split"):
        RelationalSource(replace(dataset, rows=rows), config.data.relational).anchors()


def test_generated_and_replayed_splits_cannot_repeat_root_cutoff(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from tfm.contracts.errors import SchemaError

    resolved = source()
    task = resolved.task.model_copy(
        update={"cutoffs": {"train": day(10).isoformat(), "validation": day(10).isoformat()}}
    )
    with pytest.raises(SchemaError, match="more than one split"):
        RelationalSource(declaration(), task).anchors()
    path = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"sample_id": split, "id": "a", "cutoff": day(10), "split": split}
                for split in ("train", "test")
            ]
        ),
        path,
    )
    with pytest.raises(SchemaError, match="more than one split"):
        RelationalSource(declaration(), task.model_copy(update={"anchors": str(path)})).anchors()
    different = task.model_copy(
        update={"cutoffs": {"train": day(7).isoformat(), "validation": day(10).isoformat()}}
    )
    assert len(RelationalSource(declaration(), different).anchors()) == 4


@pytest.mark.parametrize("field", ["valid_from", "known_from"])
def test_ownership_interval_starts_require_datetimes(field):
    values = dict(
        relation="owns", parent_key=("a",), child_key=("g",), valid_from=day(0), known_from=day(0)
    )
    values[field] = None
    with pytest.raises(ValueError, match=f"Ownership {field} must be a datetime"):
        OwnershipVersion(**values)


@pytest.mark.parametrize("stage", ["training", "sequence_training"])
@pytest.mark.parametrize("size", [0, -1])
def test_relational_batch_sizes_must_be_positive(stage, size):
    from pathlib import Path

    import yaml

    from tfm.config import RecipeConfig

    config = yaml.safe_load(Path("recipes/relational_inputs/config.smoke.yaml").read_text())
    config.setdefault(stage, {})["batch_size"] = size
    with pytest.raises(ValueError, match="positive steps and batch sizes"):
        RecipeConfig.model_validate(config)


@pytest.mark.parametrize("objective", ["binary", "next_event"])
def test_relational_graph_only_objectives_must_agree(objective):
    from pathlib import Path

    import yaml

    from tfm.config import RecipeConfig

    config = yaml.safe_load(Path("recipes/relational_inputs/config.graph_smoke.yaml").read_text())
    config["training"]["objective"] = "future_subgraph"
    config["model"]["composition"]["objective"] = objective
    with pytest.raises(ValueError, match="model and training objectives must agree"):
        RecipeConfig.model_validate(config)


def test_native_relational_cleanup_closes_all_snapshots_after_failure():
    from collections import OrderedDict
    from types import SimpleNamespace

    from tfm.data.native_relational_graph import NativeRelationalGraphSampler

    sampler = object.__new__(NativeRelationalGraphSampler)
    closed = []

    def close(name, fail=False):
        assert sampler.closed and not sampler.snapshots
        closed.append(name)
        if fail:
            raise OSError("cleanup failed")

    sampler.closed = False
    sampler.snapshots = OrderedDict(
        [
            ((1, 0), SimpleNamespace(close=lambda: close("first"))),
            ((2, 0), SimpleNamespace(close=lambda: close("second", fail=True))),
        ]
    )
    with pytest.raises(OSError, match="cleanup failed"):
        sampler.close()
    assert closed == ["second", "first"]
    sampler.close()


@pytest.mark.parametrize("failure", ["write", "binding"])
def test_native_anchor_file_is_removed_after_setup_failure(failure, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tfm.data import native_relational_graph as module

    snapshot = object.__new__(module._Snapshot)
    snapshot.root = tmp_path
    snapshot.calls = 0
    snapshot.cutoff = 0
    snapshot.relations = (("a", "edge", "b"),)
    snapshot.codes = {"a": {"root": 0}}
    monkeypatch.setattr(module._Snapshot, "_plan", lambda *args: None)

    def bind(*args, **kwargs):
        raise ValueError("binding failed")

    snapshot.api = SimpleNamespace(Parquet=lambda path: path, AnchorTable=bind)
    if failure == "write":

        def write(table, path):
            path.write_bytes(b"partial")
            raise ValueError("write failed")

        monkeypatch.setattr(module.pq, "write_table", write)
    with pytest.raises(ValueError, match=f"{failure} failed"):
        snapshot.neighbors(("a", "edge", "b"), ("root",))
    assert list(tmp_path.iterdir()) == []
