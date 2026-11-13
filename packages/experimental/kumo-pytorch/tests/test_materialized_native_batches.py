# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded real-index composition check; deliberately no performance claims."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors
from tests.test_preview import _equal

pytest.importorskip("diskgraph", reason="requires the optional native DiskGraph engine")


@pytest.mark.parametrize("kind", ["gnn", "history"])
@pytest.mark.parametrize("layout", [api.Packed(), api.Padded()])
def test_materialized_values_native_anchors_and_history(tmp_path, kind, layout, monkeypatch):
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "id", {"id": api.DType("string")}, time_column=api.NON_TEMPORAL),
            *(
                api.TableSpec(
                    name,
                    "id",
                    {
                        "id": api.DType("int64"),
                        "user": api.DType("string"),
                        "time": api.DType("timestamp", unit="s", timezone="UTC"),
                        "amount": api.DType("float64"),
                    },
                    time_column="time",
                )
                for name in ("events", "web")
            ),
        ),
        relations=tuple(
            api.RelationSpec(name, "users", "id", table, "user")
            for name, table in (("activity", "events"), ("visits", "web"))
        ),
    )
    sources = {}
    values = {
        "users": pa.table({"id": ["u1", "u2", "u3"]}),
        "events": pa.table(
            {
                "id": list(range(12)),
                "user": [f"u{i // 4 + 1}" for i in range(12)],
                "time": pa.array([10, 20, 20, 30] * 3, type=pa.timestamp("s", tz="UTC")),
                "amount": [float(i) if i % 4 != 1 else None for i in range(12)],
            }
        ),
        "web": pa.table(
            {
                "id": list(range(6)),
                "user": [f"u{i // 2 + 1}" for i in range(6)],
                "time": pa.array([10, 20] * 3, type=pa.timestamp("s", tz="UTC")),
                "amount": [float(i + 30) for i in range(6)],
            }
        ),
    }
    for name, rows in values.items():
        path = tmp_path / f"{name}.parquet"
        pq.write_table(rows, path)
        sources[name] = api.Parquet(str(path))
    past = api.Rows(("activity",), time=api.TimeWindow("time", None, 0))
    recent = api.Rows(("activity",), time=api.TimeWindow("time", -12, 0))
    future = api.Rows(("activity",), time=api.TimeWindow("time", 0, 20))
    sample = api.SampleSpec(
        entity=api.EntitySpec("users", entity_type="user"),
        entity_values={
            "count": api.Aggregate(past, "count", dtype=api.DType("int64")),
            "recent_sum": api.Aggregate(recent, "sum", "amount", dtype=api.DType("float64")),
        },
        targets={
            "label": api.ExternalTarget(api.DType("float64")),
            "future": api.Aggregate(future, "count", dtype=api.DType("int64")),
        },
    )
    inputs = {
        "entity": api.Input(
            api.EntitySource(),
            api.Fields((api.Encode("count", api.Raw()), api.Encode("recent_sum", api.Raw()))),
            api.Separate(),
        ),
    }
    if kind == "gnn":
        sample = replace(
            sample,
            neighborhoods={
                "graph": api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("activity", 3, strategy="recent"),)),)
                )
            },
        )
        inputs["graph"] = api.Input(
            api.NeighborhoodSource("graph"),
            api.Graph(node_fields={"events": api.Fields((api.Encode("amount", api.Raw()),))}),
            None,
        )
    else:
        sample = replace(
            sample,
            events={
                "transactions": api.EventSpec(
                    past, {"amount": api.Field("amount")}, limit=3, order="descending"
                ),
                "visits": api.EventSpec(
                    api.Rows(("visits",), time=api.TimeWindow("time", None, 0)),
                    {"amount": api.Field("amount")},
                    limit=2,
                ),
            },
        )
        inputs["history"] = api.Input(
            api.EventSource(("transactions", "visits"), max_events=4, keep="last"),
            api.Fields((api.Encode("amount", api.Raw()),)),
            layout,
        )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs=inputs,
            targets={
                "label": api.LabelTarget("label", api.Raw()),
                "future": api.LabelTarget("future", api.Raw()),
            },
            batching=api.Batching(max_samples=128),
        ),
    )
    n = 132
    rows = pa.table(
        {
            "sid": [f"row-{i}" for i in range(n)],
            "entity": [f"u{i % 3 + 1}" for i in range(n)],
            "cutoff": [25] * n,
            "label": [float(i) for i in range(n)],
            "split": ["valid" if i % 3 == 0 else "train" for i in range(n)],
            "note": [f"meta-{i}" for i in range(n)],
        }
    )
    path = tmp_path / "anchors.parquet"
    pq.write_table(rows, path, row_group_size=31)
    anchors = api.AnchorTable(
        api.Parquet(str(path)),
        "entity",
        "cutoff",
        sample_id="sid",
        split="split",
        targets={"label": "label"},
        metadata=("note",),
    )
    with api.prepare(
        contract, api.Dataset(sources), api.DiskGraphEngine(index=tmp_path / "index", num_threads=4)
    ) as plan:
        source = plan._require_open()
        calls = {"temporal": [], "aggregate": [], "merge": 0, "prepare": 0}
        original = source._sampler
        merger = source.merge_event_streams

        def counted_merge(*args, **kwargs):
            calls["merge"] += 1
            return merger(*args, **kwargs)

        source.merge_event_streams = counted_merge

        class Guard:
            def __getattr__(self, name):
                method = getattr(original, name)
                if name == "prepare_aggregate_plan":

                    def prepare(*args):
                        calls["prepare"] += 1
                        prepared = method(*args)

                        class Prepared:
                            def execute(self, roots, cutoffs, **kwargs):
                                calls["aggregate"].append(len(roots))
                                return prepared.execute(roots, cutoffs, **kwargs)

                        return Prepared()

                    return prepare
                if name not in {"query_temporal", "query_aggregate_plan"}:
                    return method

                def call(*args, **kwargs):
                    calls["aggregate" if name == "query_aggregate_plan" else "temporal"].append(
                        len(next(iter(args[0].values())))
                    )
                    return method(*args, **kwargs)

                return call

        baseline = plan.materialize(anchors, fit_split="train")

        def no_request_build(*args, **kwargs):
            pytest.fail("native scalar materialization reconstructed SampleRequests")

        with monkeypatch.context() as guard:
            guard.setattr(source, "_sampler", Guard())
            guard.setattr(type(source), "build", no_request_build)
            cached_values = plan._materialize_values(
                anchors, tmp_path / "computed.parquet", batch_size=65
            )
        # The enriched AnchorTable carries its own binding; values= is unnecessary.
        cached = plan.materialize(cached_values.anchors, fit_split="train")
        assert cached._assembler._source is cached._anchors._source
        # The plan and its enriched task view must not retain different readers.
        assert source._sampler is cached._anchors._history_engine
        assert cached._assembler._source._materialized_reference.source._sampler is source._sampler
        assert type(cached._assembler._source) is type(source)
        assert "future" in cached_values.anchors.targets
        assert isinstance(baseline._anchors, NativeAnchors)
        assert isinstance(cached._anchors, NativeAnchors)
        for selection in (api.ALL_ANCHORS, frozenset({"train", "valid"}), "train"):
            expected_rows = n if selection != "train" else 88
            before = list(baseline.batches(split=selection, output_format="model"))
            after = list(cached.batches(split=selection, output_format="model"))
            assert sum(batch.batch_size for batch in before) == expected_rows
            assert sum(batch.batch_size for batch in after) == expected_rows
            for a, b in zip(before, after, strict=True):
                assert a.metadata["sample_ids"] == b.metadata["sample_ids"]
                _equal(a.inputs, b.inputs)
                _equal(a.targets, b.targets)
                assert a.metadata["request_metadata"] == b.metadata["request_metadata"]
        before = list(baseline.batches(split="train", output_format="model"))
        monkeypatch.setattr(cached._anchors, "_requests", no_request_build)
        monkeypatch.setattr(cached._assembler._source, "build_model_ordered", no_request_build)
        monkeypatch.setattr(
            cached._assembler._source._materialized_reference,
            "build_model_ordered",
            no_request_build,
        )
        after = list(cached.batches(split="train", output_format="model"))
        for a, b in zip(before, after, strict=True):
            _equal(a.inputs, b.inputs)
            _equal(a.targets, b.targets)
            _equal(a.metadata, b.metadata)
        assert calls["prepare"] == 1
        assert max(calls["aggregate"]) >= 64
