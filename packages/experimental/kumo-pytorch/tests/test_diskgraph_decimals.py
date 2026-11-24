# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decimal transport preserves the declared type before fitting or encoding."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch import api
from kumo.pytorch.engines.diskgraph import DiskGraphSource


@pytest.mark.parametrize("precision", [38, 76])
@pytest.mark.parametrize(
    "encoding,output_format",
    [("raw", "inspectable"), ("zscore", "inspectable"), ("zscore", "model")],
)
def test_decimal_events_and_graphs(tmp_path, precision, encoding, output_format):
    pytest.importorskip("diskgraph")
    money = api.DType(f"decimal{128 if precision == 38 else 256}", precision=precision, scale=10)
    integer = api.DType("int64")
    clock = api.DType("timestamp", unit="s", timezone="UTC")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    amounts = [
        Decimal("-12.1234567890"),
        Decimal("10.0300000000"),
        None,
        Decimal("9" * (precision - 10) + ".1234567890"),
    ]
    schema = api.DataSchema(
        tables=(
            api.TableSpec("accounts", "id", {"id": integer}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "events",
                "id",
                {"id": integer, "account": integer, "time": clock, "amount": money},
                time_column="time",
            ),
        ),
        relations=(api.RelationSpec("history", "accounts", "id", "events", "account"),),
    )
    fields = api.Fields((api.Encode("amount", api.Raw() if encoding == "raw" else api.ZScore()),))
    contract = api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            api.EntitySpec("accounts"),
            events={
                "history": api.EventSpec(
                    api.Rows(("history",), api.TimeWindow("time")), {"amount": api.Field("amount")}
                )
            },
            neighborhoods={
                "history": api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("history", 10, strategy="recent"),)),)
                )
            },
        ),
        api.BatchSpec(
            inputs={
                "history": api.Input(api.EventSource(("history",)), fields, api.Padded()),
                "graph": api.Input(
                    api.NeighborhoodSource("history"),
                    api.Graph(node_fields={"events": fields}),
                    None,
                ),
            }
        ),
    )
    tables = {
        "accounts": pa.table({"id": [1]}),
        "events": pa.table(
            {
                "id": list(range(1, 5)),
                "account": [1] * 4,
                "time": pa.array(
                    [start + timedelta(seconds=i) for i in range(4)], type=clock.to_arrow()
                ),
                "amount": pa.array(amounts, type=money.to_arrow()),
            }
        ),
    }
    bindings = {}
    for name, table in tables.items():
        path = tmp_path / f"{name}.parquet"
        pq.write_table(table, path)
        bindings[name] = api.Parquet(str(path))
    request = api.SampleRequest("one", 1, start + timedelta(days=1), split="train")
    outputs = []
    for engine in (
        api.InMemoryEngine(),
        api.DiskGraphEngine(num_threads=1),
    ):
        dataset = api.Dataset(
            {name: api.InMemoryRows(table.to_pylist()) for name, table in tables.items()}
            if isinstance(engine, api.InMemoryEngine)
            else bindings
        )
        with api.prepare(contract, dataset, engine) as plan:
            task = plan.materialize((request,), fit_split="train")
            batch = next(
                task.batches(
                    batch_size=1,
                    output_format="inspectable"
                    if isinstance(engine, api.InMemoryEngine)
                    else output_format,
                )
            )
            event = batch.inputs["history"]["amount"]
            nodes = batch.inputs["graph"]["nodes"]["events"]
            graph = nodes["features"]["amount"]
            ids = nodes["ids"].tolist()
            if encoding == "raw":
                assert event["values"].type.value_type == money.to_arrow()
                assert event["values"].to_pylist() == [amounts]
                assert graph["values"].type == money.to_arrow()
                assert dict(zip(ids, graph["values"].to_pylist())) == dict(enumerate(amounts, 1))
            else:
                outputs.append((event["values"], dict(zip(ids, graph["values"].tolist()))))
                assert torch.isfinite(event["values"]).all()
                assert event["valid"].tolist() == [[True, True, False, True]]
            assert dict(zip(ids, graph["valid"].tolist())) == {1: True, 2: True, 3: False, 4: True}
    if encoding == "zscore":
        torch.testing.assert_close(outputs[0][0], outputs[1][0])
        assert outputs[0][1] == pytest.approx(outputs[1][1])


@pytest.mark.parametrize(
    "text", ["not-a-decimal", "12345678901234567890123456789.0000000000", "0.00000000001"]
)
def test_decimal_transport_rejects_invalid_or_lossy_values(text):
    source = object.__new__(DiskGraphSource)
    source._pa, source._pc = pa, pc
    source._tables = {
        "events": SimpleNamespace(
            columns={"amount": api.DType("decimal128", precision=38, scale=10)}
        )
    }
    with pytest.raises(pa.ArrowInvalid):
        source._decode_decimal_array("events", "amount", pa.array([text]))


@pytest.mark.parametrize(
    "dtype,values",
    [
        (api.DType("string"), pa.array(["10.0300000000", None])),
        (api.DType("int64"), pa.array([1, None])),
        (
            api.DType("decimal128", precision=38, scale=10),
            pa.array([Decimal("10.0300000000"), None], type=pa.decimal128(38, 10)),
        ),
    ],
)
def test_decimal_transport_leaves_other_arrays_unchanged(dtype, values):
    source = object.__new__(DiskGraphSource)
    source._pa, source._pc = pa, pc
    source._tables = {"events": SimpleNamespace(columns={"amount": dtype})}
    assert source._decode_decimal_array("events", "amount", values) is values
