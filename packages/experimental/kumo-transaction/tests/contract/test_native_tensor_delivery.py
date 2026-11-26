# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native train-fitted source tensors reach all three views without event lookup."""

from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.relational import RelationalForeignKey, RelationalTable
from tfm.contracts.table_source import TableSource
from tfm.data.native_batches import NativeBatches
from tfm.data.native_source import NativeSourceFiles
from tfm.data.native_spec import AnchorFittedEngine, history_contract, with_graph
from tfm.data.native_tensors import NativeTensorAdapter
from tfm.extras.kumo_pytorch import native_api, temporary_engine

START = datetime(2024, 1, 1, tzinfo=UTC)


def source_files(tmp_path, *, late=False):
    pq.write_table(pa.table({"id": ["train", "heldout"]}), tmp_path / "users.parquet")
    times = [START + timedelta(seconds=s) for s in [0, 1, 2, 0]]
    pq.write_table(
        pa.table(
            {
                "id": ["first", "second", "third", "test"],
                "owner": ["train"] * 3 + ["heldout"],
                "at": pa.array(times, type=pa.timestamp("s", tz="UTC")),
                "known": pa.array(
                    [*times[:3], START + timedelta(days=4)] if late else times,
                    type=pa.timestamp("s", tz="UTC"),
                ),
                "amount": [1.0, float("nan"), 3.0, 100.0],
                "kind": ["fit"] * 3 + ["heldout-only"],
            }
        ),
        tmp_path / "events.parquet",
    )
    return TableSource(
        (
            RelationalTable("users", ("id",), immutable=True),
            RelationalTable(
                "events",
                ("id",),
                numeric={"amount": "amount"},
                categorical={"kind": "kind"},
                event_time="at",
                available_at="known" if late else "at",
            ),
        ),
        (RelationalForeignKey("activity", "users", "events", ("owner",)),),
        {name: tmp_path / f"{name}.parquet" for name in ("users", "events")},
    )


def task_for(stack, tmp_path, source, *, graph=True):
    api = native_api()
    spec = history_contract(
        source,
        root="users",
        paths={"activity": ("activity",)},
        max_events=8,
        batch_size=2,
        observation_seconds=2,
    )
    if graph:
        spec = with_graph(spec, source, root="users", depth=2, fanout=10)
    pq.write_table(
        pa.table(
            {
                "sid": ["train-at-two", "test-at-two"],
                "root": [0, 1],
                "at": pa.array(
                    [START + timedelta(seconds=2)] * 2, type=pa.timestamp("s", tz="UTC")
                ),
                "split": ["train", "test"],
                "target": [0.0, 1.0],
                "label_available_us": [1704067200000000] * 2,
                "label_cutoff_us": [1704067202000000] * 2,
            }
        ),
        tmp_path / "anchors.parquet",
    )
    anchors = api.AnchorTable(
        api.Parquet(str(tmp_path / "anchors.parquet")),
        entity_id="root",
        as_of_time="at",
        sample_id="sid",
        split="split",
        metadata=("target", "label_available_us", "label_cutoff_us"),
    )
    plan = stack.enter_context(
        api.prepare(
            spec,
            api.Dataset({name: api.Parquet(str(path)) for name, path in source.files.items()}),
            AnchorFittedEngine(temporary_engine(stack)),
        )
    )
    return stack.enter_context(plan.materialize(anchors, fit_split="train"))


@pytest.mark.parametrize("mode", ["history", "sequence", "graph", "hybrid"])
def test_source_tables_deliver_owned_tensor_views_with_train_only_encoding(tmp_path, mode):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    with ExitStack() as stack:
        normalized = stack.enter_context(
            NativeSourceFiles(source_files(tmp_path), tmp_path / "normalized", block_size=2)
        )
        source = normalized.source
        task = task_for(stack, tmp_path, source)
        fitted = task.fitted_state.to_dict()
        vocabularies = [
            e["entries"] for e in fitted["encoders"].values() if e["kind"] == "vocabulary"
        ]
        assert vocabularies == [[["fit", 2]]]
        adapter = NativeTensorAdapter(source, root="users", mode=mode, depth=2, require_labels=True)
        with stack.enter_context(
            NativeBatches(task, adapter, batch_size=2, seed=4, shuffle=False)
        ) as batches:
            (batch,) = list(batches)
        if mode != "history":
            batch.assert_validated()
        history = batch if mode == "history" else getattr(batch, "history", None)
        if history is not None:
            assert history.event_mask.sum() == 3  # inclusive lower and upper bounds
            assert history.numeric_features.tolist() == [[pytest.approx(4 / 3), 3.0, 3.0, 0.0, 0.0]]
            assert not history.value_ids.is_inference()
            assert not history.value_ids.requires_grad
        if mode in {"graph", "hybrid"}:
            assert batch.graph.batch_size == 1
            assert batch.graph.node_features["events"].shape == (3, 2)
            assert torch.isfinite(batch.graph.node_features["events"]).all()
            assert batch.graph.node_features["events"][:, 1].sum() == 2
            assert batch.graph.anchors[0].sample_id == "train-at-two"
            if history is not None:
                assert history.sequence_ids == ("train-at-two",)
            batch.graph.node_features["events"].add_(1)
            with pytest.raises(ValueError, match="mutat|changed"):
                batch.assert_validated()
        native = next(iter(task.batches(split="test", batch_size=2, output_format="model")))
        if mode in {"history", "sequence", "hybrid"}:
            assert 2 in native.inputs["history"]["value_ids"]  # unseen category is global UNK


def test_graph_rejects_independent_availability_before_fanout(tmp_path):
    pytest.importorskip("kumo.pytorch.api")
    with NativeSourceFiles(
        source_files(tmp_path, late=True), tmp_path / "normalized"
    ) as normalized:
        with pytest.raises(UnsupportedCombinationError, match="availability-before-fanout"):
            with_graph(None, normalized.source, root="users", depth=2, fanout=10)


def test_entity_labels_require_availability_metadata(tmp_path):
    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    with ExitStack() as stack:
        normalized = stack.enter_context(
            NativeSourceFiles(source_files(tmp_path), tmp_path / "normalized")
        )
        task = task_for(stack, tmp_path, normalized.source)
        native = next(iter(task.batches(split="train", batch_size=2, output_format="model")))
        metadata = {**native.metadata, "request_metadata": ({"target": 1.0},)}
        with pytest.raises(SchemaError, match="explicit availability"):
            NativeTensorAdapter(normalized.source, root="users", mode="hybrid", depth=2)(
                replace(native, metadata=metadata)
            )


def test_source_plan_reuses_one_anchor_table_for_streaming_views_and_closes_training(
    tmp_path, monkeypatch
):
    from tfm.api.builders import profile_event_decoder_from_config
    from tfm.config import load_recipe_config
    from tfm.data.native_inputs import NativeInputs
    from tfm.training.trainer import EagerTrainer

    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    anchors = tmp_path / "source-anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["one", "repeated", "heldout"],
                "id": ["train", "train", "heldout"],
                "cutoff": pa.array(
                    [START + timedelta(seconds=2)] * 3, type=pa.timestamp("s", tz="UTC")
                ),
                "split": ["train", "train", "test"],
                "target": [0.0, 0.0, 1.0],
                "label_available_us": [1704067200000000] * 3,
                "label_cutoff_us": [1704067202000000] * 3,
            }
        ),
        anchors,
    )
    with NativeInputs(
        source_files(tmp_path),
        root="users",
        history_paths={"activity": ("activity",)},
        anchors=anchors,
        work_dir=tmp_path / "runtime",
        max_events=8,
        batch_size=2,
        depth=2,
        fanout=10,
    ) as inputs:
        hybrid_task = inputs.task("hybrid")
        (hybrid,) = list(inputs.batches("train", mode="hybrid", require_labels=True))
        history_task = inputs.task("history")
        (history,) = list(inputs.batches("train", mode="history"))
        (graph,) = list(inputs.batches("train", mode="graph", require_labels=True))
        assert history.entity_ids == ("train", "train")
        assert history.sequence_ids == ("one", "repeated")
        assert torch.equal(hybrid.history.value_ids, history.value_ids)
        assert torch.equal(
            hybrid.graph.node_features["events"], graph.graph.node_features["events"]
        )
        assert inputs.manifest()["pipeline"] == "source_tables_native_tensors_v1"
        calls = []
        original = history_task.batches

        def recorded(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(history_task, "batches", recorded)
        batches = inputs.batches("train", mode="history", shuffle=True, seed=7)
        config = load_recipe_config("recipes/blueprint_v2/config.smoke.yaml")
        model = profile_event_decoder_from_config(
            config, vocab_size=history_task.spaces["model"].size
        )
        result = EagerTrainer().fit(
            model,
            batches,
            max_steps=3,
            learning_rate=0.001,
            seed=7,
            device=torch.device("cpu"),
            shuffle=False,
        )
        assert result["steps"] == 3
        assert [call["epoch"] for call in calls] == [0, 1, 2]
        assert batches._active is None  # stopping by max_steps releases lookahead
        assert len(list(batches)) == 1
        assert hybrid_task is inputs.task("hybrid")
    assert not list((tmp_path / "runtime").iterdir())


def test_generated_label_anchors_need_no_event_objects_or_histories(tmp_path):
    from tfm.config import load_recipe_config
    from tfm.data.adapters.registry import table_source
    from tfm.data.native_inputs import NativeInputs

    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    config = load_recipe_config("recipes/relational_inputs/config.yaml")
    task = config.data.relational
    assert task is not None
    source = table_source(config.data, seed=config.seed, fixture_dir=tmp_path / "source")
    with NativeInputs(
        source,
        root=task.root,
        history_paths=task.history_paths,
        task=task,
        work_dir=tmp_path / "runtime",
        max_events=8,
        batch_size=4,
        depth=2,
        fanout=10,
    ) as inputs:
        (history,) = list(inputs.batches("train", mode="sequence", require_labels=True))
        (hybrid,) = list(inputs.batches("train", mode="hybrid", require_labels=True))
        assert history.labels.tolist() == [0.0, 1.0, 0.0, 1.0]
        assert torch.equal(history.history.value_ids, hybrid.history.value_ids)
        assert history.anchors == hybrid.graph.anchors
        assert hybrid.graph.node_features["activity"].shape == (8, 2)
        assert history.label_cutoff_us == 1705363200000000


def test_late_known_history_is_rejected_before_native_fitting(tmp_path):
    from tfm.data.native_projection import history_projection

    with NativeSourceFiles(
        source_files(tmp_path, late=True), tmp_path / "normalized"
    ) as normalized:
        with pytest.raises(UnsupportedCombinationError, match="independent event/availability"):
            history_projection(
                normalized,
                root="users",
                paths={"activity": ("activity",)},
                directory=tmp_path / "projection",
            )


def test_native_profile_facts_keep_their_cutoff_and_independent_history_window(tmp_path):
    from tfm.data.native_inputs import NativeInputs

    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    source = source_files(tmp_path)
    pq.write_table(
        pa.table(
            {
                "id": ["old", "recent", "future"],
                "owner": ["train"] * 3,
                "at": pa.array(
                    [START + timedelta(seconds=s) for s in (-100, 1, 3)],
                    type=pa.timestamp("s", tz="UTC"),
                ),
                "fact": ["old-fact", "recent-fact", "future-fact"],
            }
        ),
        tmp_path / "facts.parquet",
    )
    source = replace(
        source,
        tables=(
            *source.tables,
            RelationalTable(
                "facts",
                ("id",),
                categorical={"milestone": "fact"},
                event_time="at",
                available_at="at",
            ),
        ),
        relationships=(
            *source.relationships,
            RelationalForeignKey("facts", "users", "facts", ("owner",)),
        ),
        files={**source.files, "facts": tmp_path / "facts.parquet"},
        profile_paths={"milestones": ("facts",)},
    )
    pq.write_table(
        pa.table(
            {
                "id": ["train"],
                "sample_id": ["one"],
                "split": ["train"],
                "cutoff": pa.array(
                    [START + timedelta(seconds=2)], type=pa.timestamp("s", tz="UTC")
                ),
            }
        ),
        tmp_path / "requested.parquet",
    )
    with NativeInputs(
        source,
        root="users",
        history_paths={"activity": ("activity",)},
        anchors=tmp_path / "requested.parquet",
        work_dir=tmp_path / "work",
        max_events=1,
        observation_seconds=1,
        batch_size=1,
    ) as inputs:
        task = inputs.task("history")
        (validated,) = list(inputs.batches("train", mode="sequence"))
        history = validated.history
        assert history.event_mask.sum() == 1
        dated = history.prof_time_log[history.prof_mask]
        assert dated.tolist() == pytest.approx(
            [
                0.0,
                8 * torch.log1p(torch.tensor(102.0 / 8)).item(),
                8 * torch.log1p(torch.tensor(1.0 / 8)).item(),
            ]
        )
        entries = [
            entry
            for encoder in task.fitted_state.to_dict()["encoders"].values()
            if encoder["kind"] == "vocabulary"
            for entry in encoder["entries"]
        ]
        assert any(entry[0] == "old-fact" for entry in entries)
        assert all(entry[0] != "future-fact" for entry in entries)
        history.prof_value_ids.add_(1)
        with pytest.raises(ValueError, match="mutat|changed"):
            validated.assert_validated()


def test_native_prefix_views_reuse_training_encodings_and_preserve_boundary_time(tmp_path):
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_views import SourceRows

    pytest.importorskip("kumo.pytorch.api")
    pytest.importorskip("diskgraph")
    source = source_files(tmp_path)
    pq.write_table(
        pa.table(
            {
                "id": ["train", "train"],
                "sample_id": ["train", "test"],
                "split": ["train", "test"],
                "cutoff": pa.array(
                    [START + timedelta(seconds=1), START + timedelta(seconds=2)],
                    type=pa.timestamp("s", tz="UTC"),
                ),
            }
        ),
        tmp_path / "requested.parquet",
    )
    with NativeInputs(
        source,
        root="users",
        history_paths={"activity": ("activity",)},
        anchors=tmp_path / "requested.parquet",
        work_dir=tmp_path / "work",
        max_events=8,
        batch_size=1,
        fit_view="train",
        views={"train": {"events": SourceRows(before_us=1704067201000000)}},
    ) as inputs:
        train = inputs.task("history")
        (training,) = list(inputs.batches("train", mode="history"))
        (testing,) = list(inputs.batches("test", mode="history"))
        assert training.event_mask.sum() == 1
        assert testing.event_mask.sum() == 3
        assert (
            inputs.task("history", view="full").fitted_state.fingerprint
            == train.fitted_state.fingerprint
        )
        assert training.time_log[0, 0] == 0
        assert training.elapsed_log[0, 0] == 0
        assert inputs.manifest()["fit_view"] == "train"


def test_six_hop_triangle_expands_both_fk_directions_in_one_layer(tmp_path):
    """A non-bipartite frontier needs both physical directions of one FK."""
    from tfm.data.native_inputs import NativeInputs

    records = {
        "roots": {"id": [0, 1], "value": [10.0, 20.0]},
        "middle": {"id": [0, 1], "owner": [0, 1], "value": [30.0, 40.0]},
        "activity": {
            "id": [0, 1],
            "owner": [0, 1],
            "middle": [1, 0],
            "value": [50.0, 60.0],
            "at": [START, START + timedelta(seconds=1)],
        },
    }
    for name, columns in records.items():
        pq.write_table(pa.table(columns), tmp_path / f"{name}.parquet")
    source = TableSource(
        (
            RelationalTable("roots", ("id",), numeric={"value": "value"}, immutable=True),
            RelationalTable("middle", ("id",), numeric={"value": "value"}, immutable=True),
            RelationalTable(
                "activity",
                ("id",),
                numeric={"amount": "value"},
                event_time="at",
                available_at="at",
            ),
        ),
        (
            RelationalForeignKey("owns_middle", "roots", "middle", ("owner",)),
            RelationalForeignKey("owns_activity", "roots", "activity", ("owner",)),
            RelationalForeignKey("middle_activity", "middle", "activity", ("middle",)),
        ),
        {name: tmp_path / f"{name}.parquet" for name in records},
    )
    anchors = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["triangle"],
                "id": [0],
                "cutoff": [START + timedelta(seconds=2)],
                "split": ["train"],
            }
        ),
        anchors,
    )
    with NativeInputs(
        source,
        root="roots",
        history_paths={"events": ("owns_activity",)},
        anchors=anchors,
        work_dir=tmp_path / "work",
        max_events=8,
        batch_size=1,
        depth=6,
        fanout=10,
    ) as inputs:
        with inputs.batches("train", mode="graph") as batches:
            actual = list(batches)
        assert len(actual) == 1
        graph = actual[0]
        graph.assert_validated()
        assert {name: len(values) for name, values in graph.node_features.items()} == {
            "roots": 2,
            "middle": 2,
            "activity": 2,
        }
        assert sum(edges.shape[1] for edges in graph.edge_index.values()) == 12
        # At layer two, middle -> activity and activity -> middle must both run.
        assert graph.num_sampled_nodes["middle"][2] == 1
        assert graph.num_sampled_nodes["activity"][2] == 1
        assert graph.num_sampled_nodes["roots"][3] == 1


def test_generated_anchors_preserve_date_key_types_and_utc_date_clocks(tmp_path):
    from datetime import date

    from tfm.config import RelationalTaskConfig
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_source import source_entity_id

    day = date(2024, 1, 1)
    pq.write_table(pa.table({"id": [1], "day": [day]}), tmp_path / "owners.parquet")
    pq.write_table(
        pa.table({"id": [1], "owner": [1], "day": [day], "amount": [2.0]}),
        tmp_path / "activity.parquet",
    )
    source = TableSource(
        (
            RelationalTable("owners", ("id", "day"), immutable=True),
            RelationalTable(
                "activity",
                ("id",),
                numeric={"amount": "amount"},
                event_time="day",
                available_at="day",
            ),
        ),
        (RelationalForeignKey("activity", "owners", "activity", ("owner", "day")),),
        {name: tmp_path / f"{name}.parquet" for name in ("owners", "activity")},
    )
    task = RelationalTaskConfig(
        root="owners",
        history_paths={"activity": ("activity",)},
        cutoffs={"train": "2024-01-02"},
    )
    with NativeInputs(
        source,
        root=task.root,
        history_paths=task.history_paths,
        task=task,
        work_dir=tmp_path / "work",
        max_events=8,
        batch_size=1,
    ) as inputs:
        with inputs.batches("train", mode="sequence") as batches:
            batch = next(iter(batches))
            batch.assert_validated()
            assert batch.anchors[0].entity_id == source_entity_id((1, day))
            assert batch.history.event_mask.tolist() == [[True]]


def test_native_source_watermark_includes_equal_cutoff_and_rejects_later(tmp_path):
    from tfm.data.native_inputs import NativeInputs

    source = replace(source_files(tmp_path), observed_through=START + timedelta(seconds=2))
    request = tmp_path / "requested.parquet"
    for seconds in (2, 3):
        pq.write_table(
            pa.table(
                {
                    "id": ["train"],
                    "sample_id": ["one"],
                    "split": ["train"],
                    "cutoff": [START + timedelta(seconds=seconds)],
                }
            ),
            request,
        )
        kwargs = dict(
            root="users",
            history_paths={"events": ("activity",)},
            anchors=request,
            work_dir=tmp_path / f"work{seconds}",
            max_events=8,
            batch_size=1,
        )
        if seconds == 2:
            with NativeInputs(source, **kwargs):
                pass
        else:
            with pytest.raises(SchemaError, match="source coverage"):
                NativeInputs(source, **kwargs)


def test_empty_optional_fit_view_table_retains_namespace_for_heldout_history(tmp_path):
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_views import SourceRows

    source = source_files(tmp_path)
    optional = RelationalTable(
        "messages",
        ("id",),
        numeric={"message_value": "value"},
        event_time="at",
        available_at="at",
    )
    pq.write_table(
        pa.table(
            {
                "id": ["late"],
                "owner": ["train"],
                "value": [7],
                "at": [START + timedelta(seconds=2)],
            }
        ),
        tmp_path / "messages.parquet",
    )
    source = replace(
        source,
        tables=(*source.tables, optional),
        relationships=(
            *source.relationships,
            RelationalForeignKey("messages", "users", "messages", ("owner",)),
        ),
        files={**source.files, "messages": tmp_path / "messages.parquet"},
    )
    anchors = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["train", "train"],
                "sample_id": ["train", "test"],
                "split": ["train", "test"],
                "cutoff": [START + timedelta(seconds=1), START + timedelta(seconds=2)],
            }
        ),
        anchors,
    )
    with NativeInputs(
        source,
        root="users",
        history_paths={"activity": ("activity",), "messages": ("messages",)},
        anchors=anchors,
        work_dir=tmp_path / "work",
        max_events=8,
        batch_size=1,
        fit_view="train",
        views={
            "train": {
                "events": SourceRows(before_us=1704067201000000),
                "messages": SourceRows(before_us=1704067201000000),
            }
        },
    ) as inputs:
        with inputs.batches("train", mode="history") as train:
            first = next(iter(train))
        with inputs.batches("test", mode="history") as test:
            second = next(iter(test))
        assert first.event_mask.sum() == 1
        assert second.event_mask.sum() == 4
        assert (
            inputs.task("history").fitted_state.fingerprint
            == inputs.task("history", view="full").fitted_state.fingerprint
        )


def test_integer_amounts_mix_with_histories_without_an_amount_field(tmp_path):
    from tfm.data.native_inputs import NativeInputs

    source = source_files(tmp_path)
    events = pq.read_table(source.files["events"])
    events = events.set_column(
        events.schema.get_field_index("amount"), "amount", pa.array([1, 0, 3, 100], type=pa.int64())
    )
    pq.write_table(events, source.files["events"])
    pq.write_table(events, tmp_path / "messages.parquet")
    messages = RelationalTable(
        "messages", ("id",), numeric={"other": "amount"}, event_time="at", available_at="at"
    )
    source = replace(
        source,
        tables=(*source.tables, messages),
        relationships=(
            *source.relationships,
            RelationalForeignKey("messages", "users", "messages", ("owner",)),
        ),
        files={**source.files, "messages": tmp_path / "messages.parquet"},
    )
    anchors = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["train"],
                "sample_id": ["train"],
                "cutoff": [START + timedelta(seconds=2)],
                "split": ["train"],
            }
        ),
        anchors,
    )
    with NativeInputs(
        source,
        root="users",
        history_paths={"activity": ("activity",), "messages": ("messages",)},
        anchors=anchors,
        work_dir=tmp_path / "work",
        max_events=8,
        batch_size=1,
    ) as inputs:
        with inputs.batches("train", mode="history") as batches:
            batch = next(iter(batches))
        assert batch.event_mask.sum() == 6
        assert batch.numeric_features is not None
        torch.testing.assert_close(batch.numeric_features[0, :3], torch.tensor([4 / 6, 3, 6]))


def test_equal_cutoff_requests_retain_their_partition_view_and_stored_identity(tmp_path):
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_views import SourceRows

    source = source_files(tmp_path)
    events = pq.read_table(source.files["events"]).append_column(
        "partition", pa.array(["train", "validation", "validation", "train"])
    )
    pq.write_table(events, source.files["events"])
    requested = tmp_path / "requested.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["train", "train"],
                "sample_id": ["one", "two"],
                "split": ["train", "validation"],
                "cutoff": [START + timedelta(seconds=2)] * 2,
            }
        ),
        requested,
    )
    views = {
        "train": {"events": SourceRows(partition_column="partition", partitions=("train",))},
        "validation": {
            "events": SourceRows(partition_column="partition", partitions=("train", "validation"))
        },
    }
    kwargs = dict(
        root="users",
        history_paths={"events": ("activity",)},
        max_events=8,
        batch_size=1,
        fit_view="train",
        views=views,
        split_views={"train": "train", "validation": "validation"},
    )
    saved = tmp_path / "saved.parquet"
    with NativeInputs(source, anchors=requested, work_dir=tmp_path / "work", **kwargs) as inputs:
        with pytest.raises(SchemaError, match="stored split view"):
            inputs.batches("train", mode="history", view="full")
        with inputs.batches("train", mode="history") as batches:
            first = next(iter(batches))
        with inputs.batches("validation", mode="history") as batches:
            second = next(iter(batches))
        assert first.event_mask.sum() == 1
        assert second.event_mask.sum() == 3
        assert inputs.normalized.connection.execute(
            "SELECT split,view FROM anchors ORDER BY rowid"
        ).fetchall() == [("train", "train"), ("validation", "validation")]
        changed = pq.read_table(requested).set_column(0, "id", pa.array(["heldout", "heldout"]))
        pq.write_table(changed, requested)
        inputs.save_anchors(saved)
        assert pq.read_table(saved)["id"].to_pylist() == ["train", "train"]
        assert pq.read_table(saved)["view"].to_pylist() == ["train", "validation"]
        expected = inputs.anchors_sha256
    with NativeInputs(source, anchors=saved, work_dir=tmp_path / "replay", **kwargs) as replay:
        assert replay.anchors_sha256 == expected


@pytest.mark.parametrize("alias", [False, True])
def test_equal_cutoff_conflicting_families_remain_rejected_within_one_view(tmp_path, alias):
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_views import SourceRows

    source = source_files(tmp_path)
    requested = tmp_path / "requested.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["train", "train"],
                "sample_id": ["one", "two"],
                "split": ["train", "validation"],
                "cutoff": [START + timedelta(seconds=2)] * 2,
            }
        ),
        requested,
    )
    rule = {"events": SourceRows(before_us=1704067201000000)}
    with pytest.raises(SchemaError, match="multiple splits within one view"):
        NativeInputs(
            source,
            root="users",
            history_paths={"events": ("activity",)},
            anchors=requested,
            work_dir=tmp_path / "work",
            max_events=8,
            batch_size=1,
            views={"first": rule, "alias": rule},
            fit_view="first",
            split_views={"train": "first", "validation": "alias" if alias else "first"},
        )


@pytest.mark.parametrize("saved_metadata", ["absent", "null", "matching", "view", "identity"])
def test_task_replay_preserves_and_checks_saved_source_view(tmp_path, saved_metadata):
    from tfm.config import RelationalTaskConfig
    from tfm.data.native_inputs import NativeInputs
    from tfm.data.native_views import SourceRows

    source = source_files(tmp_path)
    requested = tmp_path / "requested.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["train"],
                "sample_id": ["one"],
                "split": ["train"],
                "cutoff": [START + timedelta(seconds=2)],
            }
        ),
        requested,
    )
    kwargs = dict(
        root="users",
        history_paths={"events": ("activity",)},
        max_events=8,
        batch_size=1,
        views={"restricted": {"events": SourceRows(before_us=1704067201000000)}},
        fit_view="restricted",
        split_views={"train": "restricted"},
    )
    saved = tmp_path / "saved.parquet"
    with NativeInputs(source, anchors=requested, work_dir=tmp_path / "initial", **kwargs) as inputs:
        inputs.save_anchors(saved)
        expected = inputs.anchors_sha256
    rows = pq.read_table(saved).to_pylist()
    if saved_metadata == "absent":
        for row in rows:
            del row["view"], row["view_identity"]
    elif saved_metadata == "null":
        rows[0].update(view=None, view_identity=None)
    elif saved_metadata == "view":
        kwargs["split_views"] = {"train": "full"}
    elif saved_metadata == "identity":
        kwargs["views"] = {"restricted": {"events": SourceRows(before_us=1704067202000000)}}
    pq.write_table(pa.Table.from_pylist(rows), saved)
    task = RelationalTaskConfig(
        root="users", history_paths={"events": ("activity",)}, anchors=str(saved)
    )
    if saved_metadata in {"view", "identity"}:
        with pytest.raises(SchemaError, match="Replay anchor view differs"):
            NativeInputs(source, task=task, work_dir=tmp_path / "replay", **kwargs)
    else:
        with NativeInputs(source, task=task, work_dir=tmp_path / "replay", **kwargs) as replay:
            assert replay.anchors_sha256 == expected
            replayed = pq.read_table(replay.original_anchors).to_pylist()[0]
            assert "view" in replayed and "view_identity" in replayed
            assert replayed["view"] == "restricted"
