# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Required multi-source semantics, including currently unsupported row filters."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from tests.native_fixtures import native_anchor_table


@pytest.mark.parametrize("filtered", [False, True], ids=["windows", "filters-and-windows"])
@pytest.mark.parametrize("keep", ["all", "first", "last"])
@pytest.mark.parametrize("order", ["ascending", "descending"])
@pytest.mark.parametrize("layout", [api.Padded(), api.Packed()], ids=["padded", "packed"])
def test_multisource_windows_filters_duplicates_and_ties(tmp_path, filtered, keep, order, layout):
    # Keyless, physically identical rows must remain separate occurrences.
    # Equal timestamps also cross sources. Source identity is an emitted value.
    rows = {
        "left": [(9, 90), (7, 70), (4, 40), (7, 70), (3, 30), (10, 100)],
        "right": [(7, 70), (7, 70), (8, 80), (6, 60), (9, 90), (10, 100)],
    }
    users = tmp_path / "users.parquet"
    pq.write_table(pa.table({"id": [1, 2]}), users)
    sources = {"users": api.Parquet(str(users))}
    tables = [
        api.TableSpec("users", "id", {"id": api.DType("int64")}, time_column=api.NON_TEMPORAL)
    ]
    events = {}
    for component, (name, values) in enumerate(rows.items()):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(
            pa.table(
                {
                    "owner": [1] * len(values),
                    "at": pa.array([t for t, _ in values], type=pa.timestamp("s", tz="UTC")),
                    "value": [v for _, v in values],
                }
            ),
            path,
        )
        sources[name] = api.Parquet(str(path))
        tables.append(
            api.TableSpec(
                name,
                None,
                {
                    "owner": api.DType("int64"),
                    "at": api.DType("timestamp", unit="s", timezone="UTC"),
                    "value": api.DType("int64"),
                },
                time_column="at",
            )
        )
        filters = (
            (api.RowFilter("value", "greater_than", 40),)
            if name == "left"
            else (api.RowFilter("value", "less_than", 80),)
        )
        events[name] = api.EventSpec(
            api.Rows(
                (name,),
                api.TimeWindow("at", -7, -1) if name == "left" else api.TimeWindow("at", -4, -2),
                filters=filters if filtered else (),
            ),
            {"value": api.Field("value"), "source": api.Constant(component, api.DType("int64"))},
        )
    spec = api.DataLoaderSpec(
        api.DataSchema(
            tuple(tables),
            tuple(api.RelationSpec(name, "users", "id", name, "owner") for name in rows),
        ),
        api.SampleSpec(api.EntitySpec("users"), events=events),
        api.BatchSpec(
            {
                "history": api.Input(
                    api.EventSource(
                        tuple(events),
                        max_events=None if keep == "all" else 3,
                        keep="first" if keep == "all" else keep,
                        order=order,
                    ),
                    api.Fields(tuple(api.Encode(name, api.Raw()) for name in ("source", "value"))),
                    layout,
                )
            }
        ),
    )
    # A native anchor table exercises the native merge path, not a Python merge.
    anchors_path = tmp_path / "anchors.parquet"
    pq.write_table(pa.table({"id": [1, 2], "at": [10, 10]}), anchors_path)
    anchors = native_anchor_table(api.Parquet(str(anchors_path)), "id", "at")
    with api.prepare(
        spec, api.Dataset(sources), api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)
    ) as plan:
        with plan.materialize(anchors, fit_split=api.ALL_ANCHORS) as task:
            batch = next(task.batches(batch_size=2, output_format="model"))
            repeated = next(task.batches(batch_size=2, output_format="model"))
    # Literal expected membership/order: time, declared source, occurrence.
    expected = (
        [(0, 70), (0, 70), (1, 70), (1, 70), (0, 90)]
        if filtered
        else [(0, 40), (0, 70), (0, 70), (1, 70), (1, 70), (1, 80), (0, 90)]
    )
    if keep != "all":
        expected = expected[:3] if keep == "first" else expected[-3:]
    if order == "descending":
        expected = expected[::-1]
    for result in (batch, repeated):
        output = result.inputs["history"]
        if isinstance(layout, api.Padded):
            assert output["valid"].tolist() == [[True] * len(expected), [False] * len(expected)]
            actual = list(
                zip(
                    output["source"]["values"][0].tolist(),
                    output["value"]["values"][0].tolist(),
                    strict=True,
                )
            )
        else:
            assert output["offsets"].tolist() == [0, len(expected), len(expected)]
            actual = list(
                zip(
                    output["source"]["values"].tolist(),
                    output["value"]["values"].tolist(),
                    strict=True,
                )
            )
        assert actual == expected
