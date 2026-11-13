# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native anchor identity/partition and real model-facing GNN parity."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors

pytest.importorskip("diskgraph", reason="requires the optional native DiskGraph engine")


@pytest.fixture
def native_task(tmp_path, request):
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "id", {"id": api.DType("string")}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "events",
                "id",
                {
                    "id": api.DType("int64"),
                    "user": api.DType("string"),
                    "time": api.DType("timestamp", unit="s", timezone="UTC"),
                    "amount": api.DType("float64"),
                },
                time_column="time",
            ),
        ),
        relations=(api.RelationSpec("activity", "users", "id", "events", "user"),),
    )
    users = tmp_path / "users.parquet"
    events = tmp_path / "events.parquet"
    pq.write_table(pa.table({"id": ["u1", "u2", "u3"]}), users)
    pq.write_table(
        pa.table(
            {
                "id": list(range(9)),
                "user": [f"u{i // 3 + 1}" for i in range(9)],
                "time": pa.array([10, 20, 30] * 3, type=pa.timestamp("s", tz="UTC")),
                "amount": [float(i) for i in range(9)],
            }
        ),
        events,
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec("users", entity_type="user"),
        neighborhoods={
            "graph": api.NeighborhoodSpec(
                (api.GraphLayer((api.GraphSample("activity", 2, strategy="recent"),)),)
            )
        },
        targets={"label": api.ExternalTarget(dtype=api.DType("float64"))},
    )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={
                "graph": api.Input(
                    api.NeighborhoodSource("graph"),
                    api.Graph(
                        node_fields={
                            "events": api.Fields((api.Encode("amount", api.Raw()),)),
                            **(
                                {"users": api.Fields((api.Encode("id", api.Raw()),))}
                                if getattr(request, "param", None) == "raw-string"
                                else {}
                            ),
                        }
                    ),
                    None,
                )
            },
            targets={"label": api.LabelTarget("label", api.Raw())},
            batching=api.Batching(max_samples=2),
        ),
    )
    rows = pa.table(
        {
            "sid": ["a", "b", "c", "d", "e", "f", "g"],
            "entity": ["u1", "u1", "u2", "u2", "u3", "u1", "u2"],
            "cutoff": [25] * 7,
            "label": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "split": ["train", "train", "valid", "train", None, "train", "valid"],
            # These have always been legal AnchorTable metadata; RowPack does not
            # support list/struct scalars, so their exact Arrow values must survive.
            "details": [{"parts": [i, i + 1], "note": None} for i in range(7)],
        }
    )
    anchors_path = tmp_path / "anchors.parquet"
    pq.write_table(rows, anchors_path)
    table = api.AnchorTable(
        api.Parquet(str(anchors_path)),
        "entity",
        "cutoff",
        sample_id="sid",
        split="split",
        targets={"label": "label"},
        metadata=("details",),
    )
    dataset = api.Dataset(
        tables={"users": api.Parquet(str(users)), "events": api.Parquet(str(events))}
    )
    with api.prepare(
        contract, dataset, api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    ) as plan:
        with plan.materialize(table, fit_split="train") as task:
            yield plan, task, table


def test_native_task_payload_coverage_shuffle_and_bounds(native_task):
    plan, task, table = native_task
    anchors = task._anchors
    assert isinstance(anchors, NativeAnchors)
    assert not hasattr(anchors, "_columns")  # no second full anchor column store
    assert anchors.count() == 7
    assert anchors.splits() == ("train", "valid")
    assert anchors.count(frozenset({None, "valid"})) == 3
    with pytest.raises(ValueError, match="no anchors"):
        anchors.count(frozenset({"missing"}))
    requests = anchors.fetch(api.ALL_ANCHORS, [6, 0, 6])
    assert [request.sample_id for request in requests] == ["e", "a", "e"]
    assert requests[0].metadata["details"] == {"parts": [4, 5], "note": None}
    assert requests[1].entity_id == "u1" and requests[1].targets[0].value == 1.0
    with pytest.raises(IndexError):
        anchors.fetch(api.ALL_ANCHORS, [7])
    kwargs = dict(seed=3, epoch=0, shuffle=True, world_size=3)
    orders = [list(anchors.order(api.ALL_ANCHORS, rank=rank, **kwargs)) for rank in range(3)]
    assert sorted(value for order in orders for value in order) == list(range(7))
    assert orders == [
        list(anchors.order(api.ALL_ANCHORS, rank=rank, **kwargs)) for rank in range(3)
    ]


@pytest.mark.parametrize(
    "selection,expected",
    [
        (api.ALL_ANCHORS, list("abcdefg")),
        (frozenset({"train", "valid"}), list("abcdfg")),
        (frozenset({"valid", None}), list("ceg")),
    ],
)
def test_unshuffled_mixed_splits_preserve_source_order(native_task, selection, expected):
    _, task, _ = native_task
    anchors = task._anchors
    kwargs = dict(seed=0, epoch=0, shuffle=False)
    assert [row.sample_id for row in anchors.fitting_requests(selection)] == expected
    for rank in range(3):
        order = list(anchors.order(selection, rank=rank, world_size=3, **kwargs))
        assert [row.sample_id for row in anchors.fetch(selection, order)] == expected[rank::3]
        actual = list(task.batches(split=selection, rank=rank, world_size=3, output_format="model"))
        # Production refit preserves complete native stage batches; unlike
        # inspection/fitting order it does not globally mix split rows.
        groups = [
            [sid for sid in "abdf" if sid in expected],
            [sid for sid in "cg" if sid in expected],
            [sid for sid in "e" if sid in expected],
        ]
        assert list(_by_sample(actual)) == [sid for group in groups for sid in group[rank::3]]


def _by_sample(batches):
    result = {}
    for batch in batches:
        for i, (sid, padding) in enumerate(
            zip(batch.metadata["sample_ids"], batch.metadata["is_padding"].tolist(), strict=True)
        ):
            if not padding:
                result[sid] = batch.targets["label"]["values"][i].item()
    return result


def test_gnn_handoff_reuses_sampling_and_projected_features(native_task, monkeypatch):
    plan, task, table = native_task
    source = task._assembler._source
    expected = list(
        task.batches(task._anchors.fetch(frozenset({"train"}), range(4)), output_format="model")
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("GNN hot path repeated sampling or feature fetching")

    original = source._sampler

    class Guard:
        def __getattr__(self, name):
            return unexpected if name in {"sample", "fetch_features"} else getattr(original, name)

    monkeypatch.setattr(source, "_sampler", Guard())

    def no_requests(*args, **kwargs):
        raise AssertionError("native batch was converted back into requests")

    monkeypatch.setattr(task._anchors, "_requests", no_requests)
    monkeypatch.setattr(source, "build_model_ordered", no_requests)
    actual = list(task.batches(split="train", output_format="model"))
    assert _by_sample(actual) == {"a": 1.0, "b": 2.0, "d": 4.0, "f": 6.0}
    for old, new in zip(expected, actual, strict=True):
        assert old.metadata["sample_ids"] == new.metadata["sample_ids"]
        for table in old.inputs["graph"]["nodes"]:
            a, b = old.inputs["graph"]["nodes"][table], new.inputs["graph"]["nodes"][table]
            if isinstance(a["ids"], torch.Tensor):
                assert torch.equal(a["ids"], b["ids"])
            else:
                assert a["ids"] == b["ids"]
            assert torch.equal(a["batch"], b["batch"])
            for name, field in a["features"].items():
                for key, values in field.items():
                    assert torch.equal(values, b["features"][name][key])
        assert torch.equal(
            old.inputs["graph"]["edges"]["activity"]["index"],
            new.inputs["graph"]["edges"]["activity"]["index"],
        )


@pytest.mark.parametrize("world_size", [1, 3, 10])
def test_native_batch_padding_partial_drop_last_and_resume(native_task, world_size):
    _, task, _ = native_task
    batches = [
        list(
            task.batches(
                rank=rank,
                world_size=world_size,
                seed=7,
                epoch=2,
                shuffle=True,
                output_format="model",
            )
        )
        for rank in range(world_size)
    ]
    assert sorted(sid for rank in batches for sid in _by_sample(rank)) == list("abcdefg")
    assert {sum(len(batch.metadata["sample_ids"]) for batch in rank) for rank in batches} == {
        sum(-(-count // world_size) for count in (4, 2, 1))
    }
    first = task.batches(output_format="model")
    next(first)
    saved = first.state_dict()
    restored = task.batches(output_format="model")
    restored.load_state_dict(saved)
    assert [batch.metadata["sample_ids"] for batch in restored] == [
        batch.metadata["sample_ids"] for batch in first
    ]
    dropped = list(task.batches(drop_last=True, output_format="model"))
    assert sum(len(batch.metadata["sample_ids"]) for batch in dropped) == 6


@pytest.mark.parametrize("world_size", [2, 3, 10])
def test_native_single_split_padding_keeps_columnar_batch_path(
    native_task, monkeypatch, world_size
):
    from tests.test_preview import _equal

    plan, original, table = native_task
    task = plan.materialize(replace(table, split=None), fitted=original.fitted_state)
    requests = task._anchors.fetch(api.ALL_ANCHORS, range(7))
    expected = [
        list(task.batches(requests, rank=rank, world_size=world_size, output_format="model"))
        for rank in range(world_size)
    ]

    def unexpected(*args, **kwargs):
        raise AssertionError("uneven shard fell back to per-anchor requests")

    monkeypatch.setattr(task._anchors, "_requests", unexpected)
    monkeypatch.setattr(task._assembler._source, "build_model_ordered", unexpected)
    for rank in range(world_size):
        actual = list(task.batches(rank=rank, world_size=world_size, output_format="model"))
        assert len(actual) == len(expected[rank])
        for a, b in zip(expected[rank], actual, strict=True):
            _equal(a.inputs, b.inputs)
            _equal(a.targets, b.targets)
            assert torch.equal(a.metadata["is_padding"], b.metadata["is_padding"])
            assert a.metadata["request_metadata"] == b.metadata["request_metadata"]
            # Request batches name padding before splitting; native iteration
            # names it within the final batch. Real identities must match.
            for left, right, padded in zip(
                a.metadata["sample_ids"],
                b.metadata["sample_ids"],
                b.metadata["is_padding"],
                strict=True,
            ):
                if not padded:
                    assert left == right
            assert len(set(b.metadata["sample_ids"])) == len(b.metadata["sample_ids"])
    task.close()


def test_native_default_ids_remain_original_row_ordinals(native_task):
    plan, _, table = native_task
    with plan.materialize(replace(table, sample_id=None), fit_split="train") as task:
        rows = task._anchors.fetch(api.ALL_ANCHORS, range(7))
        assert [row.sample_id for row in rows] == [0, 1, 3, 5, 2, 6, 4]
        path = task._anchors.path
    assert not path.exists()
    assert plan._source._sampler is not None


def test_windows_uses_native_reader_without_symlink_privileges(native_task, monkeypatch):
    from pathlib import Path

    plan, original, table = native_task

    def forbidden(*args, **kwargs):
        pytest.fail("native task construction requires symlink privileges")

    monkeypatch.setattr(Path, "symlink_to", forbidden)
    with plan.materialize(table, fitted=original.fitted_state) as task:
        assert isinstance(task._anchors, NativeAnchors)
        assert [
            sid
            for batch in task.batches(split="train", output_format="model")
            for sid in batch.metadata["sample_ids"]
        ] == ["a", "b", "d", "f"]


@pytest.mark.parametrize("selection", [api.ALL_ANCHORS, frozenset({"train", "valid"})])
def test_cross_split_shuffle_keeps_complete_native_stage_batches(
    native_task, selection, monkeypatch
):
    _, task, table = native_task

    def forbidden(*args, **kwargs):
        raise AssertionError("refit reentered Python anchor requests")

    monkeypatch.setattr(task._anchors, "_requests", forbidden)
    rows = pq.read_table(table.source.path)
    splits = dict(zip(rows["sid"].to_pylist(), rows["split"].to_pylist(), strict=True))
    expected = set("abcdefg" if selection is api.ALL_ANCHORS else "abcdfg")
    orders = []
    for epoch in (0, 1, 0):
        batches = list(
            task.batches(split=selection, epoch=epoch, seed=19, shuffle=True, output_format="model")
        )
        ids = [sid for batch in batches for sid in batch.metadata["sample_ids"]]
        assert set(ids) == expected and len(ids) == len(expected)
        assert all(
            len({splits[sid] for sid in batch.metadata["sample_ids"]}) == 1 for batch in batches
        )
        orders.append(ids)
    assert orders[0] == orders[2]


def test_empty_native_anchors_and_cleanup(native_task, tmp_path):
    plan, task, table = native_task
    data = pq.read_table(table.source.path)
    empty = tmp_path / "empty.parquet"
    pq.write_table(data.slice(0, 0), empty)
    with plan.materialize(
        replace(table, source=api.Parquet(str(empty))), fitted=task.fitted_state
    ) as empty_task:
        assert isinstance(empty_task._anchors, NativeAnchors)
        assert empty_task.count() == 0
        assert list(empty_task.batches(output_format="model")) == []
    assert plan._source._sampler is not None


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("entity", "missing", "unknown graph entity"),
        ("cutoff", None, "cannot be NULL"),
    ],
)
def test_invalid_native_anchors_do_not_drop_rows(native_task, tmp_path, column, value, match):
    plan, task, table = native_task
    data = pq.read_table(table.source.path)
    altered = pa.array([value] + data[column].to_pylist()[1:], type=data[column].type)
    data = data.set_column(data.schema.get_field_index(column), column, altered)
    invalid = tmp_path / "invalid.parquet"
    pq.write_table(data, invalid)
    before = set(tmp_path.glob(".anchors-*"))
    with pytest.raises(ValueError, match=match):
        plan.materialize(replace(table, source=api.Parquet(str(invalid))), fitted=task.fitted_state)
    assert set(tmp_path.glob(".anchors-*")) == before


@pytest.mark.parametrize("layout", [api.Packed(), api.Padded()])
def test_sequence_native_anchors_reuse_ids_and_existing_temporal_loader(
    native_task, layout, monkeypatch
):
    from tests.test_preview import _equal

    original, _, table = native_task
    history = api.Rows(path=("activity",), time=api.TimeWindow("time", None, 0))
    sample = api.SampleSpec(
        entity=original.contract.sample.entity,
        entity_values={"count": api.Aggregate(history, "count", dtype=api.DType("int64"))},
        events={"history": api.EventSpec(history, {"amount": api.Field("amount")})},
        targets=original.contract.sample.targets,
    )
    contract = replace(
        original.contract,
        sample=sample,
        batch=api.BatchSpec(
            inputs={
                "history": api.Input(
                    api.EventSource(("history",)),
                    api.Fields((api.Encode("amount", api.Raw()),)),
                    layout,
                ),
                "entity": api.Input(
                    api.EntitySource(),
                    api.Fields((api.Encode("count", api.Raw()),)),
                    api.Separate(),
                ),
            },
            targets={"label": api.LabelTarget("label", api.Raw())},
            batching=api.Batching(max_samples=2),
        ),
    )
    with api.prepare(contract, original.dataset, original.engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            assert isinstance(task._anchors, NativeAnchors)
            requests = task._anchors.fetch(frozenset({"train"}), range(4))
            expected = list(task.batches(requests, output_format="model"))
            source = task._assembler._source
            sampler = source._sampler

            class Guard:
                def __getattr__(self, name):
                    if name == "lookup_node_ids":
                        raise AssertionError("sequence batch repeated native anchor ID lookup")
                    return getattr(sampler, name)

            monkeypatch.setattr(source, "_sampler", Guard())
            actual = list(task.batches(split="train", output_format="model"))
            for old, new in zip(expected, actual, strict=True):
                _equal(old.inputs, new.inputs)
                _equal(old.targets, new.targets)
                assert old.metadata["sample_ids"] == new.metadata["sample_ids"]
                assert old.metadata["request_metadata"] == new.metadata["request_metadata"]


@pytest.mark.parametrize("layout", [api.Packed(), api.Padded()])
@pytest.mark.parametrize("order,keep", [("ascending", "first"), ("descending", "last")])
@pytest.mark.parametrize("rank,world_size", [(0, 1), (1, 3), (9, 10)])
def test_native_sequence_iterator_retrieves_and_merges_without_python_requests(
    native_task, monkeypatch, layout, order, keep, rank, world_size
):
    from tests.test_preview import _equal

    original, _, table = native_task
    history = api.Rows(path=("activity",), time=api.TimeWindow("time", None, 0))
    sample = api.SampleSpec(
        entity=original.contract.sample.entity,
        events={
            name: api.EventSpec(history, {"amount": api.Field("amount")})
            for name in ("transactions", "visits")
        },
        targets=original.contract.sample.targets,
    )
    contract = replace(
        original.contract,
        sample=sample,
        batch=api.BatchSpec(
            inputs={
                "history": api.Input(
                    api.EventSource(
                        ("transactions", "visits"), max_events=3, order=order, keep=keep
                    ),
                    api.Fields((api.Encode("amount", api.Raw()),)),
                    layout,
                )
            },
            targets={"label": api.LabelTarget("label", api.Raw())},
            batching=api.Batching(max_samples=2),
        ),
    )
    with api.prepare(contract, original.dataset, original.engine) as plan:
        with plan.materialize(table, fit_split="train") as task:
            requests = task._anchors.fetch(frozenset({"train"}), range(4))
            expected = list(
                task.batches(requests, rank=rank, world_size=world_size, output_format="model")
            )
            source = task._assembler._source

            def unexpected(*args, **kwargs):
                raise AssertionError("history iteration crossed a per-request Python path")

            sampler = source._sampler

            class Guard:
                def __getattr__(self, name):
                    if name in {"query_temporal", "lookup_node_ids", "sample", "fetch_features"}:
                        return unexpected
                    return getattr(sampler, name)

            monkeypatch.setattr(source, "_sampler", Guard())
            monkeypatch.setattr(source, "build_model_ordered", unexpected)
            monkeypatch.setattr(source, "merge_event_streams", unexpected)
            monkeypatch.setattr(task._anchors, "_requests", unexpected)
            actual = list(
                task.batches(split="train", rank=rank, world_size=world_size, output_format="model")
            )
            assert len(actual) == len(expected)
            for a, b in zip(expected, actual, strict=True):
                _equal(a.inputs, b.inputs)
                _equal(a.targets, b.targets)
                assert a.metadata["sample_ids"] == b.metadata["sample_ids"]
                assert a.metadata["request_metadata"] == b.metadata["request_metadata"]


@pytest.mark.parametrize("native_task", ["raw-string"], indirect=True)
def test_native_model_keeps_existing_contract_validation(native_task, monkeypatch):
    _, task, _ = native_task

    def no_read(*args, **kwargs):
        pytest.fail("an invalid tensor contract must be rejected before native reading")

    monkeypatch.setattr(task._anchors, "_iter", no_read)
    with pytest.raises(ValueError, match="Raw string requires inspectable output"):
        list(task.batches(split="train", output_format="model"))


def _base_index_mappings(index):
    """Count file mappings rather than comparing RSS of shared file pages."""
    import sys
    from collections import Counter
    from pathlib import Path

    if sys.platform != "linux":
        pytest.skip("/proc mapping accounting requires Linux")
    prefix = str(Path(index).resolve() / "nodes") + "/"
    counts = Counter()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith(prefix):
            counts[tuple(parts[2:5])] += 1  # offset, device, inode
    assert counts, "expected mapped base-index files"
    return counts


def test_native_anchor_history_share_one_combined_reader(native_task):
    plan, task, _ = native_task
    from pathlib import Path

    source = plan._source
    assert task._anchors._history_engine is source._sampler
    assert any(
        name.startswith(Path(task._anchors._extension.name).name + "/")
        for name in source._node_types
    )
    assert set(_base_index_mappings(source._index_path).values()) == {1}
    for epoch in range(2):
        list(task.batches(split="train", epoch=epoch, output_format="model"))
    assert set(_base_index_mappings(source._index_path).values()) == {1}


def test_replacing_closed_anchor_task_releases_old_reader(native_task):
    plan, task, table = native_task
    fitted = task.fitted_state
    task.close()
    with plan.materialize(table, fitted=fitted) as restored:
        assert restored._anchors._history_engine is plan._source._sampler
        assert set(_base_index_mappings(plan._source._index_path).values()) == {1}
        assert sum(len(b.metadata["sample_ids"]) for b in restored.batches(split="train")) == 4


def test_three_named_neighborhoods_share_native_anchor_schedule(native_task, monkeypatch):
    plan, task, table = native_task
    sample = replace(
        plan.contract.sample,
        neighborhoods={
            name: api.NeighborhoodSpec((api.GraphLayer((api.GraphSample("activity", fanout),)),))
            for name, fanout in (("small", 1), ("medium", 2), ("large", 3))
        },
    )
    original = plan.contract.batch.inputs["graph"]
    inputs = {
        name: replace(original, source=api.NeighborhoodSource(name))
        for name in sample.neighborhoods
    }
    # Two model inputs referencing the same named neighborhood reuse its graph.
    inputs["small_copy"] = inputs["small"]
    contract = replace(
        plan.contract, sample=sample, batch=replace(plan.contract.batch, inputs=inputs)
    )
    with api.prepare(contract, plan.dataset, plan.engine) as prepared:
        with prepared.materialize(table, fit_split="train") as native:

            def forbidden(*args, **kwargs):
                raise AssertionError("multi-neighborhood iteration used request-based sampling")

            monkeypatch.setattr(native._anchors, "_requests", forbidden)
            monkeypatch.setattr(native._assembler._source, "_sample_graph_events", forbidden)
            batches = list(native.batches(split="train", output_format="model"))
            assert _by_sample(batches) == {"a": 1.0, "b": 2.0, "d": 4.0, "f": 6.0}
            for batch in batches:
                assert len(batch.inputs["small"]["nodes"]["events"]["ids"]) == batch.batch_size
                assert len(batch.inputs["medium"]["nodes"]["events"]["ids"]) == 2 * batch.batch_size
                assert torch.equal(
                    batch.inputs["small"]["edges"]["activity"]["index"],
                    batch.inputs["small_copy"]["edges"]["activity"]["index"],
                )
                for name in inputs:
                    values = batch.inputs[name]["nodes"]["events"]["features"]["amount"]["values"]
                    assert torch.isfinite(values).all()


def test_native_extension_uses_existing_base_directory_without_symlinks(native_task, monkeypatch):
    from pathlib import Path

    plan, previous, table = native_task

    def forbidden(*args, **kwargs):
        raise AssertionError("native task constructed a symlink graph view")

    monkeypatch.setattr(Path, "symlink_to", forbidden)
    with plan.materialize(table, fitted=previous.fitted_state) as task:
        extension = Path(task._anchors._extension.name)
        assert extension.parent == Path(plan._require_open()._index_path) / "train_tables"
        assert (extension / "index_metadata.json").is_file()
        assert not any(path.is_symlink() for path in extension.iterdir())
        assert _by_sample(task.batches(split="train", output_format="model")) == {
            "a": 1.0,
            "b": 2.0,
            "d": 4.0,
            "f": 6.0,
        }
    assert not extension.exists()


@pytest.mark.parametrize("neighborhood_count", [1, 2])
def test_native_graph_projection_allows_keyless_nonentity_tables(
    native_task, monkeypatch, neighborhood_count
):
    import copy

    plan, _, table = native_task
    sample = replace(
        plan.contract.sample,
        neighborhoods={
            f"n{i}": plan.contract.sample.neighborhoods["graph"] for i in range(neighborhood_count)
        },
    )
    original = plan.contract.batch.inputs["graph"]
    contract = replace(
        plan.contract,
        sample=sample,
        batch=replace(
            plan.contract.batch,
            inputs={
                name: replace(original, source=api.NeighborhoodSource(name))
                for name in sample.neighborhoods
            },
        ),
    )
    with api.prepare(contract, plan.dataset, plan.engine) as prepared:
        with prepared.materialize(table, fit_split="train") as native:
            source = native._assembler._source
            # Emulate main's keyless schema at the adapter boundary. The old
            # branch's TableSpec constructor still requires a PK; native graph
            # topology already carries dense node identity independently of it.
            definition = copy.copy(source._tables["events"])
            object.__setattr__(definition, "primary_key", None)
            monkeypatch.setitem(source._tables, "events", definition)
            actual = list(native.batches(split="train", output_format="model"))
            assert _by_sample(actual) == {"a": 1.0, "b": 2.0, "d": 4.0, "f": 6.0}
            for batch in actual:
                for name in sample.neighborhoods:
                    nodes = batch.inputs[name]["nodes"]["events"]
                    assert nodes["ids"].numel() == 2 * batch.batch_size
                    assert torch.isfinite(nodes["features"]["amount"]["values"]).all()
