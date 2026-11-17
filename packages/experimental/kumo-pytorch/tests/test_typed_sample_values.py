# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stored scalar identity and typed calculations do not depend on an encoder."""

import datetime as dt
from decimal import Decimal

import pytest

from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    CalendarPart,
    Constant,
    Derived,
    EntitySpec,
    EventSpec,
    EventTimeDelta,
    Field,
    MapValues,
    Rows,
    SampleRequest,
    SampleSpec,
    TimeWindow,
    evaluate_derived,
    sample_value_types,
)
from kumo.pytorch.contract.schema import NON_TEMPORAL, DataSchema, RelationSpec, TableSpec
from kumo.pytorch.engines.in_memory.samples import InMemorySampleBuilder


@pytest.mark.parametrize(
    "dtype,value",
    [
        (DType("int64"), 2**63 - 1),
        (DType("uint64"), 2**64 - 1),
        (DType("float64"), 1.5),
        (DType("bool"), True),
        (DType("string"), "[MISSING]"),
        (DType("bytes"), b"\x00\xff"),
        (DType("decimal128", precision=20, scale=2), Decimal("1234567890123456.78")),
        (DType("date32"), dt.date(1969, 12, 31)),
    ],
)
def test_source_values_keep_their_declared_type(dtype, value):
    schema = DataSchema(
        tables=(
            TableSpec(
                name="users",
                primary_key="id",
                columns={"id": DType("int64"), "value": dtype},
                time_column=NON_TEMPORAL,
            ),
        ),
        relations=(),
    )
    sample = SampleSpec(
        entity=EntitySpec(table="users"), entity_values={"value": Field(column="value")}
    )
    built = InMemorySampleBuilder(schema, {"users": [{"id": 1, "value": value}]}).build(
        sample, [SampleRequest(sample_id="one", entity_id=1, as_of_time=0)]
    )[0]
    field = built.history.entity.values[0]
    assert field.dtype == dtype
    assert field.value == value
    assert type(field.value) is type(value)


@pytest.mark.parametrize(
    "dtype,value",
    [
        (DType("int8"), 128),
        (DType("uint8"), -1),
        (DType("int64"), True),
        (DType("bool"), 1),
        (DType("string"), 1.5),
        (DType("decimal128", precision=3, scale=1), Decimal("1.25")),
    ],
)
def test_dtype_rejects_lossy_or_incompatible_values(dtype, value):
    with pytest.raises(ValueError, match="column.*not a valid"):
        dtype.normalize(value, path="table.column")


@pytest.mark.parametrize(
    "dtype,temporal",
    [
        (DType("timestamp", unit="s", timezone="UTC"), True),
        (DType("date32"), True),
        (DType("int64"), False),
        (DType("float64"), False),
        (DType("string"), False),
        (DType("bytes"), False),
        (DType("decimal128", precision=10, scale=2), False),
    ],
)
def test_dtype_temporal_names_the_time_types(dtype, temporal):
    assert dtype.temporal is temporal


def test_timestamp_storage_ceilings_pre_epoch_subseconds():
    dtype = DType("timestamp", unit="s", timezone="UTC")
    assert dtype.normalize(
        dt.datetime(1969, 12, 31, 23, 59, 59, 999000, tzinfo=dt.UTC)
    ) == dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


def test_event_calculations_resolve_named_prerequisites_and_use_anchor_time():
    timestamp = DType("timestamp", unit="s", timezone="UTC")
    schema = DataSchema(
        tables=(
            TableSpec(
                name="users",
                primary_key="id",
                columns={"id": DType("int64")},
                time_column=NON_TEMPORAL,
            ),
            TableSpec(
                name="events",
                primary_key="id",
                columns={
                    "id": DType("int64"),
                    "user_id": DType("int64"),
                    "at": timestamp,
                    "amount": DType("float64"),
                },
                time_column="at",
            ),
        ),
        relations=(
            RelationSpec(
                name="history",
                left_table="users",
                left_key="id",
                right_table="events",
                right_key="user_id",
            ),
        ),
    )
    sample = SampleSpec(
        entity=EntitySpec(table="users"),
        events={
            "transaction": EventSpec(
                rows=Rows(path=("history",)),
                fields={
                    "month": Derived(
                        operation=CalendarPart(part="month", timezone="UTC"),
                        inputs=("time",),
                        dtype=DType("int64"),
                    ),
                    "direction": Derived(
                        operation=MapValues(groups={"outflow": (-1,), "inflow": (0, 1)}),
                        inputs=("sign",),
                        dtype=DType("string"),
                    ),
                    "sign": Derived(operation="sign", inputs=("amount",), dtype=DType("int64")),
                    "magnitude": Derived(
                        operation="abs", inputs=("amount",), dtype=DType("float64")
                    ),
                    "amount": Field(column="amount"),
                    "time": Field(column="at"),
                    "age": EventTimeDelta(unit="seconds", dtype=DType("float64")),
                },
            )
        },
    )
    built = InMemorySampleBuilder(
        schema,
        {
            "users": [{"id": 1}],
            "events": [
                {"id": 10, "user_id": 1, "at": 40, "amount": -2.5},
                {"id": 11, "user_id": 1, "at": 100, "amount": 99.0},
            ],
        },
    ).build(sample, [SampleRequest(sample_id="one", entity_id=1, as_of_time=100)])[0]
    assert len(built.history.events) == 2
    values = {field.key: field.value for field in built.history.events[0].fields}
    assert values == {
        "amount": -2.5,
        "time": timestamp.normalize(40),
        "month": 1,
        "sign": -1,
        "direction": "outflow",
        "magnitude": 2.5,
        "age": 60.0,
    }
    assert {field.key: field.value for field in built.history.events[1].fields} == {
        "amount": 99.0,
        "time": timestamp.normalize(100),
        "month": 1,
        "sign": 1,
        "direction": "inflow",
        "magnitude": 99.0,
        "age": 0.0,
    }


def test_typed_arithmetic_checks_result_width_and_scalar_cardinality():
    spec = Derived(operation="add", inputs=("a", "a"), dtype=DType("int8"))
    with pytest.raises(ValueError, match="not a valid"):
        evaluate_derived(spec, {"a": 100}, path="sample.total")
    schema = DataSchema(
        tables=(
            TableSpec(
                name="users",
                primary_key="id",
                columns={"id": DType("int64")},
                time_column=NON_TEMPORAL,
            ),
        ),
        relations=(),
    )
    sample = SampleSpec(
        entity=EntitySpec(table="users"),
        entity_values={
            "a": Field(column="id", selection="all"),
            "total": Derived(operation="add", inputs=("a", "a"), dtype=DType("int64")),
        },
    )
    with pytest.raises(ValueError, match="scalar operands"):
        sample_value_types(schema, sample)


def test_operation_settings_are_explicit_and_validated():
    with pytest.raises(ValueError, match="disjoint"):
        MapValues(groups={"first": (1,), "second": (1,)})
    with pytest.raises(ValueError, match="timezone"):
        CalendarPart(part="month", timezone="America/New_York")
    assert Constant(value=None, dtype=DType("string")).value is None


@pytest.mark.parametrize("dtype,base", [(DType("int64"), 2**53), (DType("uint64"), 2**63)])
def test_native_integer_min_max_and_distinct_preserve_exact_values(tmp_path, dtype, base):
    pytest.importorskip("diskgraph")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kumo.pytorch import api

    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                name="users",
                primary_key="id",
                columns={"id": DType("int64")},
                time_column=api.NON_TEMPORAL,
            ),
            api.TableSpec(
                name="events",
                primary_key="id",
                columns={
                    "id": DType("int64"),
                    "user_id": DType("int64"),
                    "at": DType("timestamp", unit="s", timezone="UTC"),
                    "value": dtype,
                },
                time_column="at",
            ),
        ),
        relations=(
            api.RelationSpec(
                name="history",
                left_table="users",
                left_key="id",
                right_table="events",
                right_key="user_id",
            ),
        ),
    )
    sample = api.SampleSpec(
        entity=api.EntitySpec(table="users"),
        entity_values={
            name: api.Aggregate(
                source=api.Rows(path=("history",)),
                operation=operation,
                column="value",
                dtype=DType("int64") if operation == "count_distinct" else dtype,
            )
            for name, operation in (
                ("minimum", "min"),
                ("maximum", "max"),
                ("distinct", "count_distinct"),
                ("mode", "mode"),
            )
        },
    )
    contract = api.DataLoaderSpec(
        schema=schema,
        sample=sample,
        batch=api.BatchSpec(
            inputs={
                "values": api.Input(
                    source=api.EntitySource(),
                    representation=api.Fields(
                        fields=tuple(
                            api.Encode(field=name, encoding=api.Raw())
                            for name in sample.entity_values
                        )
                    ),
                    layout=api.Separate(),
                )
            }
        ),
    )
    rows = {
        "users": [{"id": 1}],
        "events": [
            {"id": 10, "user_id": 1, "at": 1, "value": base},
            {"id": 11, "user_id": 1, "at": 2, "value": base + 1},
        ],
    }
    sources = {}
    for table in schema.tables:
        path = tmp_path / f"{table.name}.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                rows[table.name],
                schema=pa.schema(
                    [pa.field(name, dtype.to_arrow()) for name, dtype in table.columns.items()]
                ),
            ),
            path,
        )
        sources[table.name] = api.Parquet(str(path))
    requests = (api.SampleRequest(sample_id="one", entity_id=1, as_of_time=100),)
    for dataset, engine in (
        (
            api.Dataset({name: api.InMemoryRows(value) for name, value in rows.items()}),
            api.InMemoryEngine(),
        ),
        (api.Dataset(sources), api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)),
    ):
        result = api.preview(contract, dataset, engine, requests).batch.inputs["values"]
        assert {name: value["values"].item() for name, value in result.items()} == {
            "minimum": base,
            "maximum": base + 1,
            "distinct": 2,
            "mode": base,
        }


def test_dated_occurrence_calculations_use_the_retained_occurrence():
    from kumo.pytorch.contract.sample import DatedEntityValue, dated_value_types

    schema = DataSchema(
        tables=(
            TableSpec(
                name="users",
                primary_key="id",
                columns={"id": DType("int64"), "at": DType("timestamp", unit="s", timezone="UTC")},
                time_column=NON_TEMPORAL,
            ),
        ),
        relations=(),
    )
    sample = SampleSpec(
        entity=EntitySpec(table="users"),
        dated_entity_values={
            "signup": DatedEntityValue(
                rows=Rows(time=TimeWindow(column="at")),
                field=Constant(value="signup", dtype=DType("string")),
                fields={"age": EventTimeDelta(unit="seconds", dtype=DType("float64"))},
            )
        },
    )
    built = InMemorySampleBuilder(schema, {"users": [{"id": 1, "at": 10}]}).build(
        sample, [SampleRequest(sample_id="one", entity_id=1, as_of_time=100)]
    )[0]
    occurrence = built.history.entity.dated_values[0]
    assert occurrence.value == "signup"
    assert occurrence.fields[0].value == 90.0
    assert dated_value_types(schema, sample)["signup"]["age"] == DType("float64")


def test_map_null_members_cannot_override_null_propagation():
    from kumo.pytorch.contract.sample import MapValues, _typed_map_groups

    with pytest.raises(ValueError, match="NULL operands remain NULL"):
        _typed_map_groups(MapValues(groups={1: (None,)}), DType("int64"))


def test_map_results_preserve_declared_floating_values():
    from kumo.pytorch.contract.sample import MapValues, evaluate_derived

    declaration = Derived(
        operation=MapValues(groups={float("inf"): (1,)}),
        inputs=("value",),
        dtype=DType("float64"),
    )
    assert evaluate_derived(
        declaration,
        {"value": 1},
        path="result",
        input_types={"value": DType("int64")},
    ) == float("inf")
