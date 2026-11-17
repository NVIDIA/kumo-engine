# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Global history caps bound every native component before payload decoding."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors

pytest.importorskip("diskgraph")


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("layout", [api.Padded(), api.Packed()])
@pytest.mark.parametrize("same_rows", [False, True])
def test_component_caps_preserve_global_ties_and_empty_histories(
    tmp_path, monkeypatch, keep, order, layout, same_rows
):
    sources = {}
    users = tmp_path / "users.parquet"
    pq.write_table(pa.table({"id": [1, 2]}), users)
    sources["users"] = api.Parquet(str(users))
    for component, name in enumerate(("events", "web")):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(
            pa.table(
                {
                    "id": list(range(64)),
                    "user": [1] * 64,
                    "time": pa.array(list(range(64)), type=pa.timestamp("s", tz="UTC")),
                    "value": [float(component * 1000 + i) for i in range(64)],
                }
            ),
            path,
        )
        sources[name] = api.Parquet(str(path))
    schema = api.DataSchema(
        tables=(
            api.TableSpec("users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL),
            *(
                api.TableSpec(
                    name,
                    "id",
                    {
                        "id": api.DType("int64"),
                        "user": api.DType("int64"),
                        "time": api.DType("timestamp", unit="s", timezone="UTC"),
                        "value": api.DType("float64"),
                    },
                    time_column="time",
                )
                for name in ("events", "web")
            ),
        ),
        relations=tuple(
            api.RelationSpec(name, "users", "id", name, "user") for name in ("events", "web")
        ),
    )
    sample = api.SampleSpec(
        api.EntitySpec("users"),
        events={
            name: api.EventSpec(
                api.Rows(("events" if same_rows else name,), time=api.TimeWindow("time", None, 0)),
                {"value": api.Field("value")},
            )
            for name in ("events", "web")
        },
    )
    contract = api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={
                "history": api.Input(
                    api.EventSource(("events", "web"), max_events=3, keep=keep, order=order),
                    api.Fields((api.Encode("value", api.Raw()),)),
                    layout,
                )
            },
            batching=api.Batching(max_samples=2),
        ),
    )
    path = tmp_path / "anchors.parquet"
    pq.write_table(pa.table({"id": [1, 2], "at": [100, 100]}), path)
    anchors = api.AnchorTable(api.Parquet(str(path)), "id", "at")
    observed = []
    original = NativeAnchors._iter

    def checked_iter(self, stage, **options):
        history = options.get("history")
        if history:
            plan, _ = map(json.loads, history)
            assert not plan["aggregates"]
            assert [item["limit"] for item in plan["sequences"]] == [3, 3]
        for runtime in original(self, stage, **options):
            if history:
                for sequence in runtime["history"]["sequences"]:
                    assert sequence["batch"].num_rows <= 3
                observed.append(runtime["history"]["stats"])
            yield runtime

    with api.prepare(
        contract, api.Dataset(sources), api.DiskGraphEngine(index=tmp_path / "index", num_threads=2)
    ) as plan:
        with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
            monkeypatch.setattr(NativeAnchors, "_iter", checked_iter)
            monkeypatch.setattr(
                plan._require_open(),
                "merge_event_streams",
                lambda *args: pytest.fail("native iteration must merge before returning to Python"),
            )
            batch = next(task.batches(output_format="model"))
    expected = [0.0, 1000.0, 1.0] if keep == "first" else [1062.0, 63.0, 1063.0]
    if same_rows:
        expected = [0.0, 0.0, 1.0] if keep == "first" else [62.0, 63.0, 63.0]
    if order == "descending":
        expected.reverse()
    history = batch.inputs["history"]
    actual = history["value"]["values"].tolist()
    assert actual == ([expected, [0.0] * 3] if isinstance(layout, api.Padded) else expected)
    assert observed
    # PK + clock + one payload field; only three rows per component survive.
    assert observed[0]["values_decoded"] <= 2 * 3 * 3
