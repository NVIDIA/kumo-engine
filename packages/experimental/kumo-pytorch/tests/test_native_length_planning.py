# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native pools retain complete batches and cache lengths across shuffled epochs."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from tests.test_model_history_limits import history_data as history_data
from tests.test_preview import _equal

diskgraph = pytest.importorskip("diskgraph")


@pytest.mark.parametrize("tokens", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("keep", ["first", "last"])
def test_native_planning_matches_python_and_caches_lengths(
    history_data, tmp_path, monkeypatch, tokens, packed, keep
):
    schema, dataset, _, index, _ = history_data
    rows = api.Rows(("orders",), api.TimeWindow("at", stop=0))
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            name: api.EventSpec(rows, {"amount": api.Field("amount"), "kind": api.Field("kind")})
            for name in ("a", "b")
        },
    )
    representation = api.Fields((api.Encode("amount", api.Raw()),))
    if tokens:
        representation = api.Fields(
            (api.Encode("kind", api.Vocabulary(), id_space="words"),),
            row=api.Tokens(missing="omit"),
            history=api.Flatten(9, keep=keep, separator="separator", before="bos", after="eos"),
        )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={
                "history": api.Input(
                    api.EventSource(("a", "b"), max_events=5, keep=keep),
                    representation,
                    api.Packed() if packed else api.Padded(),
                )
            },
            id_spaces={
                "words": api.IdSpace(
                    {"pad": "pad", "separator": "sep", "bos": "start", "eos": "end"}
                )
            }
            if tokens
            else {},
            batching=api.Batching(
                max_samples=3,
                max_tokens=14 if tokens else 10,
                length_input="history",
                bucket_by_length=True,
            ),
        ),
    )
    n = 37
    entities = [1, 2, 1, 3] * 10
    cutoffs = [1, 5, 3, 5, 2, 6, 0] * 6
    path = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "sid": list(range(n)),
                "entity": entities[:n],
                "cutoff": cutoffs[:n],
                "split": ["train"] * n,
            }
        ),
        path,
    )
    table = api.AnchorTable(
        api.Parquet(str(path)), "entity", "cutoff", sample_id="sid", split="split"
    )
    with api.prepare(contract, dataset, api.DiskGraphEngine(index=index, num_threads=2)) as plan:
        with plan.materialize(table, fit_split="train") as task:
            expected = {}
            requests = task._anchors.fetch(frozenset({"train"}), range(n))
            for request in requests:
                expected[request.sample_id] = next(task.batches([request], output_format="model"))

            def forbidden(*args, **kwargs):
                pytest.fail(
                    "length-aware native iteration reconstructed requests/encoded for planning"
                )

            monkeypatch.setattr(task._anchors, "_requests", forbidden)
            monkeypatch.setattr(task._assembler, "_model_lengths", forbidden)
            monkeypatch.setattr(task._assembler._source, "build_model_ordered", forbidden)

            def run(epoch):
                batches = list(
                    task.batches(
                        split="train",
                        seed=17,
                        epoch=epoch,
                        shuffle=True,
                        chunk_batches=2,
                        output_format="model",
                    )
                )
                ids = []
                for batch in batches:
                    ids.extend(batch.metadata["sample_ids"])
                    assert batch.batch_size <= 3
                    # Compare every delivered sample with the existing encoder/merger.
                    # Padding width can legitimately differ when composition changes.
                    history = batch.inputs["history"]
                    for i, sid in enumerate(batch.metadata["sample_ids"]):
                        old = expected[sid].inputs["history"]
                        if tokens:
                            # Existing model metadata gives exact sample token boundaries.
                            if packed:
                                lo, hi = history["offsets"][i : i + 2].tolist()
                                _equal(history["value_ids"][lo:hi], old["value_ids"])
                            else:
                                length = int(history["valid"][i].sum())
                                _equal(
                                    history["value_ids"][i, :length], old["value_ids"][0, :length]
                                )
                        else:
                            if packed:
                                lo, hi = history["offsets"][i : i + 2].tolist()
                                _equal(history["amount"]["values"][lo:hi], old["amount"]["values"])
                            else:
                                length = int(history["valid"][i].sum())
                                _equal(
                                    history["amount"]["values"][i, :length],
                                    old["amount"]["values"][0, :length],
                                )
                assert sorted(ids) == list(range(n))
                return ids

            first = run(0)
            assert task._anchors._last_length_planning_stats == (n, 0)
            assert run(0) == first
            assert task._anchors._last_length_planning_stats == (0, n)
            assert run(1) != first
            assert task._anchors._last_length_planning_stats == (0, n)


@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("budget", [None, 15])
def test_precomputed_text_lengths_drive_native_budget_without_planning_encoding(
    history_data, tmp_path, monkeypatch, boundary, budget
):
    from dataclasses import replace

    from kumo.pytorch.encoding import fit_encoder

    schema, dataset, _, _, _ = history_data
    order_path = dataset.tables["orders"].path
    rows = pq.read_table(order_path)
    text = ["a", "hello world", None, "", "longer repeated phrase here", "x y z", "future", "other"]
    encoder = fit_encoder("text", api.Text(vocabulary_size=20), api.DType("string"), text)
    text_config = api.Text(state=encoder.tokenizer_json)
    rows = rows.set_column(rows.schema.get_field_index("kind"), "kind", pa.array(text))
    path = tmp_path / "text-events.parquet"
    pq.write_table(rows, path)
    dataset = replace(dataset, tables={**dataset.tables, "orders": api.Parquet(str(path))})
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            "history": api.EventSpec(
                api.Rows(("orders",), api.TimeWindow("at", stop=0)), {"text": api.Field("kind")}
            )
        },
    )
    batch = api.BatchSpec(
        inputs={
            "history": api.Input(
                api.EventSource(("history",), max_events=4),
                api.Fields(
                    (api.Encode("text", text_config, id_space="words"),),
                    row=api.Tokens(missing="emit", max_tokens=4),
                    history=api.Flatten(
                        11,
                        prefer_event_boundary=boundary,
                        separator="separator",
                        before="bos",
                        after="eos",
                    ),
                ),
                api.Packed(),
            )
        },
        id_spaces={
            "words": api.IdSpace(
                {"pad": "pad", "missing": "missing", "separator": "sep", "bos": "bos", "eos": "eos"}
            )
        },
        batching=api.Batching(
            max_samples=3, max_tokens=budget, length_input="history", bucket_by_length=True
        ),
    )
    anchors_path = tmp_path / "text-anchors.parquet"
    pq.write_table(pa.table({"entity": [1, 1, 2, 3] * 6, "cutoff": [2, 6, 5, 5] * 6}), anchors_path)
    anchors = api.AnchorTable(api.Parquet(str(anchors_path)), "entity", "cutoff")
    with api.prepare(
        api.DataLoaderSpec(schema, sample, batch),
        dataset,
        api.DiskGraphEngine(index=tmp_path / "text-index", num_threads=2),
    ) as plan:
        with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
            # Both token-aware bucketing and budgeting consume #307 metadata.
            assert "orders" in task._anchors._event_costs
            assert task._anchors._event_costs["orders"]["values"].type == pa.uint16()
            assert "orders" in task.fitted_state.fit["inputs"]["statistics"]["token_lengths"]
            assert len(task._anchors._token_maps) == 1
            expected = {
                r.sample_id: next(task.batches([r], output_format="model")).inputs["history"][
                    "value_ids"
                ]
                for r in task._anchors.fetch(api.ALL_ANCHORS, range(24))
            }
            monkeypatch.setattr(
                task._assembler,
                "_model_lengths",
                lambda *a, **k: pytest.fail("planning re-encoded text"),
            )
            for epoch in (0, 1):
                seen = []
                for result in task.batches(
                    output_format="model", seed=77, epoch=epoch, shuffle=True, chunk_batches=2
                ):
                    history = result.inputs["history"]
                    if budget is not None:
                        assert len(history["value_ids"]) <= budget
                    for i, sid in enumerate(result.metadata["sample_ids"]):
                        lo, hi = history["offsets"][i : i + 2].tolist()
                        _equal(history["value_ids"][lo:hi], expected[sid])
                        seen.append(sid)
                assert sorted(seen) == list(range(24))
                assert task._anchors._last_length_planning_stats == (
                    (24, 0) if epoch == 0 else (0, 24)
                )


@pytest.mark.parametrize("nested_fields", [api.Packed(), api.Padded()])
def test_nested_native_composition_matches_existing_length_policy(
    history_data, tmp_path, nested_fields
):
    from kumo.pytorch.engines.parquet_anchors import ParquetAnchors

    schema, dataset, _, index, _ = history_data
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            "history": api.EventSpec(
                api.Rows(("orders",), api.TimeWindow("at", stop=0)), {"kind": api.Field("kind")}
            )
        },
    )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={
                "history": api.Input(
                    api.EventSource(("history",), max_events=4),
                    api.Fields(
                        (api.Encode("kind", api.Vocabulary(), id_space="tokens"),),
                        row=api.Tokens(missing="skip_event"),
                    ),
                    api.Nested(fields=nested_fields, events=api.Packed()),
                )
            },
            id_spaces={"tokens": api.IdSpace({"pad": "pad"})},
            batching=api.Batching(
                max_samples=3, max_tokens=7, length_input="history", bucket_by_length=True
            ),
        ),
    )
    path = tmp_path / "nested-anchors.parquet"
    pq.write_table(pa.table({"entity": [1, 2, 1, 3] * 7, "cutoff": [2, 5, 6, 5] * 7}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "entity", "cutoff")
    with api.prepare(contract, dataset, api.DiskGraphEngine(index=index, num_threads=2)) as plan:
        with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
            native = task._anchors
            task._anchors = ParquetAnchors(anchors)
            try:
                expected = list(task.batches(shuffle=False, chunk_batches=2, output_format="model"))
            finally:
                task._anchors.close()
                task._anchors = native
            actual = list(task.batches(shuffle=False, chunk_batches=2, output_format="model"))
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected, strict=True):
                assert a.metadata["sample_ids"] == b.metadata["sample_ids"]
                _equal(a.inputs, b.inputs)
            assert native._last_length_planning_stats == (28, 0)


def test_native_drop_last_and_cache_survive_batch_size_change(history_data, tmp_path):
    from dataclasses import replace

    schema, dataset, _, index, _ = history_data
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            "history": api.EventSpec(
                api.Rows(("orders",), api.TimeWindow("at", stop=0)), {"amount": api.Field("amount")}
            )
        },
    )
    spec = api.BatchSpec(
        {
            "history": api.Input(
                api.EventSource(("history",), max_events=4),
                api.Fields((api.Encode("amount", api.Raw()),)),
                api.Padded(),
            )
        },
        batching=api.Batching(
            max_samples=3, length_input="history", bucket_by_length=True, drop_last=True
        ),
    )
    path = tmp_path / "tail.parquet"
    pq.write_table(pa.table({"entity": [1] * 13, "cutoff": [4] * 13}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "entity", "cutoff")
    with api.prepare(
        api.DataLoaderSpec(schema, sample, spec),
        dataset,
        api.DiskGraphEngine(index=index, num_threads=2),
    ) as plan:
        with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
            first = list(task.batches(output_format="model", chunk_batches=2))
            assert [b.batch_size for b in first] == [3, 3, 3, 3]
            assert task._anchors._last_length_planning_stats == (13, 0)
            second = list(
                task.batches(output_format="model", chunk_batches=2, batch_size=2, epoch=1)
            )
            assert [b.batch_size for b in second] == [2] * 6
            assert task._anchors._last_length_planning_stats == (0, 13)
        budget = replace(
            spec, batching=api.Batching(max_samples=3, max_tokens=1, length_input="history")
        )
    with api.prepare(
        api.DataLoaderSpec(schema, sample, budget),
        dataset,
        api.DiskGraphEngine(index=index, num_threads=2),
    ) as plan:
        # Fit on a legal empty history, then refuse a longer unseen input.
        with plan.materialize(
            [api.SampleRequest("empty", 2, 4, split="train")], fit_split="train"
        ) as calibration:
            fitted = calibration.fitted_state
        with plan.materialize(anchors, fitted=fitted) as task:
            with pytest.raises(
                diskgraph.DiskgraphConfigError, match="one sample exceeds Batching.max_tokens"
            ):
                list(task.batches(output_format="model"))
            with pytest.raises(ValueError, match="world_size"):
                list(task.batches(output_format="model", world_size=2))


@pytest.mark.parametrize("missing", ["omit", "emit", "skip_event"])
def test_native_costs_include_derived_hour_and_missing_fields(history_data, tmp_path, missing):
    schema, dataset, _, _, _ = history_data
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            "history": api.EventSpec(
                api.Rows(("orders",), api.TimeWindow("at", stop=0)),
                {
                    "amount": api.Field("amount"),
                    "time": api.Field("at"),
                    "hour": api.Derived(
                        api.CalendarPart("hour"), ("time",), dtype=api.DType("int64")
                    ),
                },
            )
        },
    )
    batch = api.BatchSpec(
        inputs={
            "history": api.Input(
                api.EventSource(("history",), max_events=4),
                api.Fields(
                    (
                        api.Encode("amount", api.Buckets(bins=3), id_space="words"),
                        api.Encode("hour", api.Vocabulary(), id_space="words"),
                    ),
                    row=api.Tokens(missing=missing, max_tokens=2),
                ),
                api.Nested(fields=api.Packed(), events=api.Padded()),
            )
        },
        id_spaces={"words": api.IdSpace({"pad": "pad", "missing": "missing"})},
        batching=api.Batching(
            max_samples=3, max_tokens=16, length_input="history", bucket_by_length=True
        ),
    )
    path = tmp_path / "hour-anchors.parquet"
    pq.write_table(pa.table({"entity": [1, 2, 3] * 5, "at": [5] * 15}), path)
    with api.prepare(
        api.DataLoaderSpec(schema, sample, batch),
        dataset,
        api.DiskGraphEngine(index=tmp_path / "hour-index", num_threads=2),
    ) as plan:
        with plan.materialize(
            api.AnchorTable(api.Parquet(str(path)), "entity", "at"), fit_split=api.ALL_ANCHORS
        ) as task:
            from kumo.pytorch.engines.parquet_anchors import ParquetAnchors

            native = task._anchors
            task._anchors = ParquetAnchors(task._anchors._table)
            try:
                expected = list(task.batches(output_format="model", shuffle=False, chunk_batches=2))
            finally:
                task._anchors.close()
                task._anchors = native
            for epoch in (0, 1):
                actual = list(
                    task.batches(output_format="model", shuffle=False, epoch=epoch, chunk_batches=2)
                )
                assert len(actual) == len(expected)
                for result, reference in zip(actual, expected, strict=True):
                    assert result.metadata["sample_ids"] == reference.metadata["sample_ids"]
                    _equal(result.inputs, reference.inputs)
                assert task._anchors._last_length_planning_stats == (
                    (15, 0) if epoch == 0 else (0, 15)
                )


@pytest.mark.parametrize("planning", [False, True])
def test_fixed_width_costs_never_scan_or_allocate_per_row_lengths(
    history_data, tmp_path, monkeypatch, planning
):
    schema, dataset, _, index, _ = history_data
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            "history": api.EventSpec(
                api.Rows(("orders",), api.TimeWindow("at", stop=0)), {"kind": api.Field("kind")}
            )
        },
    )
    batch = api.BatchSpec(
        inputs={
            "history": api.Input(
                api.EventSource(("history",), max_events=4),
                api.Fields(
                    (api.Encode("kind", api.Vocabulary(), id_space="words"),),
                    row=api.Tokens(missing="emit"),
                    history=api.Flatten(8),
                ),
                api.Packed(),
            )
        },
        id_spaces={"words": api.IdSpace({"pad": "pad", "missing": "missing"})},
        batching=api.Batching(
            max_samples=3,
            max_tokens=24 if planning else None,
            length_input="history" if planning else None,
            bucket_by_length=planning,
        ),
    )
    path = tmp_path / "constant-anchors.parquet"
    pq.write_table(pa.table({"entity": [1, 2, 3], "cutoff": [5, 5, 5]}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "entity", "cutoff")
    with api.prepare(
        api.DataLoaderSpec(schema, sample, batch),
        dataset,
        api.DiskGraphEngine(index=index, num_threads=2),
    ) as plan:
        monkeypatch.setattr(
            plan._require_open(),
            "token_lengths",
            lambda *a, **k: pytest.fail("fixed costs scanned per-row tokens"),
        )
        with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
            assert task._anchors._token_maps == []
            if planning:
                assert task._anchors._event_costs["orders"]["constant"] == 1
                assert task._anchors._event_costs["orders"]["values"] is None
            else:
                assert task._anchors._event_costs == {}
            expected = {
                r.sample_id: next(task.batches([r], output_format="model"))
                for r in task._anchors.fetch(api.ALL_ANCHORS, range(3))
            }
            for actual in task.batches(output_format="model"):
                for i, sid in enumerate(actual.metadata["sample_ids"]):
                    lengths = actual.inputs["history"]["offsets"]
                    reference = expected[sid].inputs["history"]["value_ids"]
                    assert (
                        actual.inputs["history"]["value_ids"][lengths[i] : lengths[i + 1]].tolist()
                        == reference.tolist()
                    )
