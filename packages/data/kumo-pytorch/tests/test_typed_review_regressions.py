# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent public-API boundary cases from the typed-contract review."""

import datetime as dt
import json
import math
from contextlib import contextmanager
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.data import cutoff_to_epoch

MODES = ("reference", "native_inspectable", "native_model")
INT = api.DType("int64")
TIME = api.DType("timestamp", unit="s", timezone="UTC")


def _list(values):
    return values.to_pylist() if hasattr(values, "to_pylist") else values.tolist()


def _root_contract(columns, values, selected, *, targets=None, batch_targets=None):
    return api.DataLoaderSpec(
        schema=api.DataSchema(
            tables=(
                api.TableSpec(
                    "rows",
                    "id",
                    {"id": INT, **columns},
                    time_column=api.NON_TEMPORAL,
                ),
            )
        ),
        sample=api.SampleSpec(
            entity=api.EntitySpec("rows"),
            entity_values=values,
            targets=targets or {},
        ),
        batch=api.BatchSpec(
            inputs={
                "x": api.Input(
                    api.EntitySource(),
                    api.Fields(fields=(api.Encode(selected, api.Raw()),)),
                    api.Separate(),
                )
            },
            targets=batch_targets or {},
        ),
    )


@contextmanager
def _prepared(tmp_path, contract, tables, mode):
    if mode == "reference":
        engine = api.InMemoryEngine()
        dataset = api.Dataset(
            tables={name: api.InMemoryRows(rows) for name, rows in tables.items()}
        )
    else:
        pytest.importorskip("diskgraph")
        directory = tmp_path / mode
        directory.mkdir()
        bindings = {}
        for table in contract.schema.tables:
            path = directory / f"{table.name}.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    tables[table.name],
                    schema=pa.schema(
                        [(name, dtype.to_arrow()) for name, dtype in table.columns.items()]
                    ),
                ),
                path,
            )
            bindings[table.name] = api.Parquet(str(path))
        dataset = api.Dataset(tables=bindings)
        engine = api.DiskGraphEngine(index=directory / "index")
    assert api.check(contract, dataset, engine) == ()
    with api.prepare(contract=contract, dataset=dataset, engine=engine) as plan:
        yield plan


def _batch(plan, requests, mode):
    task = plan.materialize(anchors=requests)
    return next(
        task.batches(
            batch_size=len(requests),
            output_format="model" if mode == "native_model" else "inspectable",
        )
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "dtype,value,member",
    (
        (api.DType("float32"), 0.1, 0.1),
        (api.DType("float64"), 1.0, 1),
        (api.DType("float64"), -0.0, 0.0),
    ),
)
def test_map_members_use_the_declared_operand_domain(tmp_path, mode, dtype, value, member):
    contract = _root_contract(
        {"v": dtype},
        {
            "v": api.Field("v"),
            "mapped": api.Derived(
                api.MapValues(groups={7: (member,)}, unknown="fallback", fallback=9),
                ("v",),
                dtype=INT,
            ),
        },
        "mapped",
    )
    with _prepared(tmp_path, contract, {"rows": [{"id": 1, "v": value}]}, mode) as plan:
        batch = _batch(plan, (api.SampleRequest("one", 1, 0),), mode)
    assert _list(batch.inputs["x"]["mapped"]["values"]) == [7]


def test_map_member_overlap_is_checked_after_float32_normalization():
    contract = _root_contract(
        {"v": api.DType("float32")},
        {
            "v": api.Field("v"),
            "mapped": api.Derived(
                api.MapValues(groups={7: (0.1,), 9: (0.10000000149011612,)}),
                ("v",),
                dtype=INT,
            ),
        },
        "mapped",
    )
    with pytest.raises(api.ValidationError, match="overlap after dtype normalization"):
        contract.validate()


@pytest.mark.parametrize("mode", MODES)
def test_aggregate_comparison_normalizes_its_float32_literal(tmp_path, mode):
    dtype = api.DType("float32")
    contract = _root_contract(
        {"v": dtype},
        {
            "same": api.Compare(
                api.Aggregate(api.Rows(), "min", "v", dtype=dtype),
                "equal",
                0.1,
                dtype=api.DType("bool"),
            )
        },
        "same",
    )
    with _prepared(tmp_path, contract, {"rows": [{"id": 1, "v": 0.1}]}, mode) as plan:
        batch = _batch(plan, (api.SampleRequest("one", 1, 0),), mode)
    assert _list(batch.inputs["x"]["same"]["values"]) == [True]


@pytest.mark.parametrize("mode", ("reference", "native_inspectable"))
@pytest.mark.parametrize("family,precision", (("decimal128", 38), ("decimal256", 76)))
@pytest.mark.parametrize(
    "operation,expected",
    (
        ("add", Decimal("0.75")),
        ("subtract", Decimal("-3.25")),
        ("multiply", Decimal("-2.5000")),
        ("abs", Decimal("1.25")),
    ),
)
def test_decimal_arithmetic_checks_actual_result_not_arrow_static_precision(
    tmp_path,
    mode,
    family,
    precision,
    operation,
    expected,
):
    operand = api.DType(family, precision=precision, scale=2)
    result = api.DType(family, precision=precision, scale=4 if operation == "multiply" else 2)
    contract = _root_contract(
        {"a": operand, "b": operand},
        {
            "a": api.Field("a"),
            "b": api.Field("b"),
            "calculated": api.Derived(
                operation,
                ("a",) if operation == "abs" else ("a", "b"),
                dtype=result,
            ),
        },
        "calculated",
    )
    rows = {"rows": [{"id": 1, "a": Decimal("-1.25"), "b": Decimal("2.00")}]}
    with _prepared(tmp_path, contract, rows, mode) as plan:
        batch = _batch(plan, (api.SampleRequest("one", 1, 0),), mode)
    assert _list(batch.inputs["x"]["calculated"]["values"]) == [expected]


@pytest.mark.parametrize("mode", MODES)
def test_missing_operand_propagates_before_nonfinite_operand_check(tmp_path, mode):
    dtype = api.DType("float64")
    contract = _root_contract(
        {"a": dtype, "b": dtype},
        {
            "a": api.Field("a"),
            "b": api.Field("b"),
            "total": api.Derived("add", ("a", "b"), dtype=dtype),
        },
        "total",
    )
    with _prepared(
        tmp_path,
        contract,
        {"rows": [{"id": 1, "a": None, "b": math.nan}]},
        mode,
    ) as plan:
        batch = _batch(plan, (api.SampleRequest("one", 1, 0),), mode)
    assert _list(batch.inputs["x"]["total"]["valid"]) == [False]


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_nonfinite_raw_constants_save_and_restore(value):
    contract = _root_contract(
        {},
        {"constant": api.Constant(value, dtype=api.DType("float64"))},
        "constant",
    )
    contract.validate()
    restored = api.DataLoaderSpec.from_json(contract.to_json())
    restored.validate()
    actual = restored.sample.entity_values["constant"].value
    assert math.isnan(actual) if math.isnan(value) else actual == value
    assert restored.fingerprint == contract.fingerprint
    assert restored.input_fingerprint == contract.input_fingerprint


def _event_schema():
    return api.DataSchema(
        tables=(
            api.TableSpec("users", "id", {"id": INT}, time_column=api.NON_TEMPORAL),
            api.TableSpec(
                "events",
                "id",
                {"id": INT, "user_id": INT, "at": TIME, "v": INT},
                time_column="at",
            ),
        ),
        relations=(api.RelationSpec("history", "users", "id", "events", "user_id"),),
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("component", ("entity", "event"))
def test_direct_all_field_keeps_occurrence_offsets_in_every_execution_path(
    tmp_path,
    mode,
    component,
):
    if component == "entity":
        contract = _root_contract({"v": INT}, {"v": api.Field("v", selection="all")}, "v")
        tables = {"rows": [{"id": 1, "v": 3}, {"id": 2, "v": None}]}
        requests = (api.SampleRequest("one", 1, 0), api.SampleRequest("two", 2, 0))
    else:
        contract = api.DataLoaderSpec(
            _event_schema(),
            api.SampleSpec(
                entity=api.EntitySpec("users"),
                events={
                    "event": api.EventSpec(
                        api.Rows(path=("history",)),
                        {"v": api.Field("v", selection="all")},
                    )
                },
            ),
            api.BatchSpec(
                inputs={
                    "x": api.Input(
                        api.EventSource(("event",)),
                        api.Fields(fields=(api.Encode("v", api.Raw()),)),
                        api.Packed(),
                    )
                }
            ),
        )
        tables = {
            "users": [{"id": 1}],
            "events": [
                {"id": 1, "user_id": 1, "at": TIME.normalize(1), "v": 3},
                {"id": 2, "user_id": 1, "at": TIME.normalize(2), "v": None},
            ],
        }
        requests = (api.SampleRequest("one", 1, 4),)
    with _prepared(tmp_path, contract, tables, mode) as plan:
        batch = _batch(plan, requests, mode)
    field = batch.inputs["x"]["v"]
    assert _list(field["values"]) == [3]
    assert _list(field["valid"]) == [True]
    assert _list(field["occurrence_offsets"]) == [0, 1, 1]


@pytest.mark.parametrize("mode", MODES)
def test_uint64_primary_key_is_not_inferred_as_int64(tmp_path, mode):
    key = 2**63 + 1
    contract = _root_contract({"id": api.DType("uint64"), "v": INT}, {"v": api.Field("v")}, "v")
    with _prepared(tmp_path, contract, {"rows": [{"id": key, "v": 3}]}, mode) as plan:
        batch = _batch(plan, (api.SampleRequest("one", key, 0),), mode)
    assert _list(batch.inputs["x"]["v"]["values"]) == [3]


@pytest.mark.parametrize("mode", MODES)
def test_qualified_entity_selector_uses_the_resolved_source_name(tmp_path, mode):
    contract = _root_contract({"v": INT}, {"v": api.Field("v")}, "entity/v")
    with _prepared(tmp_path, contract, {"rows": [{"id": 1, "v": 3}]}, mode) as plan:
        batch = _batch(plan, (api.SampleRequest("one", 1, 0),), mode)
    assert _list(batch.inputs["x"]["entity/v"]["values"]) == [3]


@pytest.mark.parametrize("engine_name", ("reference", "native"))
def test_repeated_row_target_field_refuses_before_open(tmp_path, monkeypatch, engine_name):
    contract = api.DataLoaderSpec(
        _event_schema(),
        api.SampleSpec(
            entity=api.EntitySpec("users"),
            entity_values={"id": api.Field("id")},
            targets={
                "future": api.RowTarget(
                    api.Rows(path=("history",), time=api.TimeWindow("at", start=0, stop=5)),
                    {"v": api.Field("v", selection="all")},
                )
            },
        ),
        api.BatchSpec(
            inputs={
                "x": api.Input(
                    api.EntitySource(),
                    api.Fields(fields=(api.Encode("id", api.Raw()),)),
                    api.Separate(),
                )
            },
            targets={"y": api.StructuredTarget("future", (api.Encode("v", api.Raw()),))},
        ),
    )
    if engine_name == "reference":
        engine = api.InMemoryEngine()
        dataset = api.Dataset(tables={name: api.InMemoryRows(()) for name in ("users", "events")})
    else:
        engine = api.DiskGraphEngine(index=tmp_path / "index")
        dataset = api.Dataset(
            tables={
                name: api.Parquet(str(tmp_path / f"nonexistent-{name}.parquet"))
                for name in ("users", "events")
            }
        )
    monkeypatch.setattr(engine, "open", lambda *_: pytest.fail("refused source must not open"))
    issues = api.check(contract, dataset, engine)
    assert len(issues) == 1
    assert issues[0].path == "batch.targets.y"
    assert "scalar, not repeated" in issues[0].message
    assert issues[0].suggestion
    with pytest.raises(api.ValidationError, match="scalar, not repeated"):
        api.prepare(contract=contract, dataset=dataset, engine=engine)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "target,match",
    (
        (api.TargetValue("undeclared", 1), "unknown supplied target"),
        (api.TargetValue("label", 1), r"targets.label.*bool"),
    ),
)
def test_caller_label_validation_precedes_entity_lookup(tmp_path, mode, target, match):
    contract = _root_contract(
        {"v": INT},
        {"v": api.Field("v")},
        "v",
        targets={"label": api.ExternalTarget(dtype=api.DType("bool"))},
    )
    with _prepared(tmp_path, contract, {"rows": [{"id": 1, "v": 3}]}, mode) as plan:
        # This deliberately nonexistent key must not hide the invalid label.
        with pytest.raises(ValueError, match=match):
            _batch(plan, (api.SampleRequest("one", 999, 0, targets=(target,)),), mode)


def test_table_clock_type_is_checked_before_engine_selection():
    with pytest.raises(ValueError, match="clock requires timestamp or date32"):
        api.TableSpec("events", "id", {"id": INT, "at": INT}, time_column="at")


@pytest.mark.parametrize(
    "operation,value",
    (
        ("unrecognized", 1),
        ("greater_than", None),
        ("in", "123"),
    ),
)
def test_invalid_filter_shape_is_rejected_before_source_open(operation, value):
    with pytest.raises(ValueError):
        api.RowFilter("id", operation, value)


@pytest.mark.parametrize(
    "value,expected",
    (
        (
            dt.datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=dt.UTC),
            dt.datetime(1970, 1, 1, tzinfo=dt.UTC),
        ),
        (
            dt.datetime(1970, 1, 1, 0, 0, 0, 1, tzinfo=dt.UTC),
            dt.datetime(1970, 1, 1, 0, 0, 1, tzinfo=dt.UTC),
        ),
        (
            dt.datetime(9999, 12, 31, 23, 59, 58, 1, tzinfo=dt.UTC),
            dt.datetime(9999, 12, 31, 23, 59, 59, tzinfo=dt.UTC),
        ),
    ),
)
def test_timestamp_ceil_uses_exact_integer_datetime_arithmetic(value, expected):
    assert TIME.normalize(value) == expected


@pytest.mark.parametrize(
    "unit,physical_values",
    (
        ("ms", (-1001, -1, 1, 1001)),
        ("us", (-1000001, -1, 1, 1000001)),
        ("ns", (-1000000001, -1, 1, 1000000001)),
    ),
)
def test_native_subsecond_storage_records_physical_unit_and_ceil_values(
    tmp_path,
    unit,
    physical_values,
):
    pytest.importorskip("diskgraph")
    source = tmp_path / "rows.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 4],
                "v": pa.array(physical_values, type=pa.timestamp(unit, tz="UTC")),
            }
        ),
        source,
    )
    contract = _root_contract({"v": TIME}, {"v": api.Field("v")}, "v")
    engine = api.DiskGraphEngine(index=tmp_path / "index")
    dataset = api.Dataset(tables={"rows": api.Parquet(str(source))})
    assert api.check(contract, dataset, engine) == ()
    requests = tuple(api.SampleRequest(str(index), index, 0) for index in (1, 2, 3, 4))
    with api.prepare(contract=contract, dataset=dataset, engine=engine) as plan:
        batch = _batch(plan, requests, "native_inspectable")
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
    assert _list(batch.inputs["x"]["v"]["values"]) == [
        epoch + dt.timedelta(seconds=seconds) for seconds in (-1, 0, 1, 2)
    ]
    manifest = json.loads((tmp_path / "index" / "index_manifest.json").read_text())
    assert manifest["source_types"]["rows"][str(source)]["v"] == f"timestamp[{unit}, tz=UTC]"
    assert manifest["column_types"]["rows"]["v"]["unit"] == "s"
    assert manifest["column_types"]["rows"]["v"]["timezone"] == "UTC"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("value,expected", ((63, 126), (64, None)))
def test_integer_calculation_range_error_names_its_field(tmp_path, mode, value, expected):
    dtype = api.DType("int8")
    contract = _root_contract(
        {"v": dtype},
        {"v": api.Field("v"), "total": api.Derived("add", ("v", "v"), dtype=dtype)},
        "total",
    )
    with _prepared(tmp_path, contract, {"rows": [{"id": 1, "v": value}]}, mode) as plan:
        requests = (api.SampleRequest("one", 1, 0),)
        if expected is None:
            with pytest.raises(ValueError, match="total"):
                _batch(plan, requests, mode)
        else:
            batch = _batch(plan, requests, mode)
            assert _list(batch.inputs["x"]["total"]["values"]) == [expected]


_EPOCH = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
_BASE = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


@pytest.mark.parametrize(
    "value,expected",
    (
        (dt.datetime(2024, 1, 1, 0, 0, 0, 300000, tzinfo=dt.UTC), 1704067200),
        (dt.datetime(1969, 12, 31, 23, 59, 59, 500000, tzinfo=dt.UTC), -1),
        (dt.datetime(1969, 12, 31, 23, 59, 59, tzinfo=dt.UTC), -1),
        (dt.date(1970, 1, 2), 86_400),
        (7, 7),
    ),
)
def test_cutoff_floors_while_stored_time_ceils(value, expected):
    assert cutoff_to_epoch(value) == expected


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "anchor,event,past",
    (
        (_BASE + dt.timedelta(seconds=0.3), _BASE + dt.timedelta(seconds=0.7), 0),
        (_EPOCH - dt.timedelta(seconds=0.5), _EPOCH - dt.timedelta(seconds=0.2), 0),
        (_BASE + dt.timedelta(seconds=1), _BASE + dt.timedelta(seconds=1), 1),
    ),
    ids=("same_second", "before_epoch", "whole_second"),
)
def test_fractional_anchor_never_admits_a_later_event(tmp_path, mode, anchor, event, past):
    history = api.Rows(path=("history",), time=api.TimeWindow("at", stop=0))
    future = api.Rows(path=("history",), time=api.TimeWindow("at", start=0, stop=5))
    contract = api.DataLoaderSpec(
        _event_schema(),
        api.SampleSpec(
            entity=api.EntitySpec("users"),
            entity_values={"past": api.Aggregate(history, "count", dtype=INT)},
            events={"event": api.EventSpec(history, {"v": api.Field("v")})},
            neighborhoods={
                "graph": api.NeighborhoodSpec(
                    layers=(
                        api.GraphLayer(samples=(api.GraphSample("history", 8, strategy="recent"),)),
                    )
                )
            },
            targets={"future": api.Aggregate(future, "count", dtype=INT)},
        ),
        api.BatchSpec(
            inputs={
                "x": api.Input(
                    api.EntitySource(),
                    api.Fields(fields=(api.Encode("past", api.Raw()),)),
                    api.Separate(),
                ),
                "event": api.Input(
                    api.EventSource(("event",)),
                    api.Fields(fields=(api.Encode("v", api.Raw()),)),
                    api.Packed(),
                ),
                "graph": api.Input(api.NeighborhoodSource("graph"), api.Graph(), None),
            },
            targets={"future": api.LabelTarget("future", api.Raw())},
        ),
    )
    tables = {
        "users": [{"id": 1}],
        "events": [{"id": 1, "user_id": 1, "at": event, "v": 3}],
    }
    if mode == "reference":
        engine = api.InMemoryEngine()
        dataset = api.Dataset(
            tables={name: api.InMemoryRows(rows) for name, rows in tables.items()}
        )
    else:
        pytest.importorskip("diskgraph")
        physical = {"at": pa.timestamp("us", tz="UTC")}
        bindings = {}
        for table in contract.schema.tables:
            path = tmp_path / f"{table.name}.parquet"
            schema = pa.schema(
                [
                    (name, physical.get(name, dtype.to_arrow()))
                    for name, dtype in table.columns.items()
                ]
            )
            pq.write_table(pa.Table.from_pylist(tables[table.name], schema=schema), path)
            bindings[table.name] = api.Parquet(str(path))
        dataset = api.Dataset(tables=bindings)
        engine = api.DiskGraphEngine(index=tmp_path / "index")
    assert api.check(contract, dataset, engine) == ()
    with api.prepare(contract=contract, dataset=dataset, engine=engine) as plan:
        batch = _batch(plan, (api.SampleRequest("one", 1, anchor),), mode)
    assert _list(batch.inputs["x"]["past"]["values"]) == [past]
    assert _list(batch.inputs["event"]["v"]["values"]) == [3] * past
    graph = batch.inputs["graph"]["nodes"].get("events")
    assert (len(_list(graph["ids"])) if graph else 0) == past
    assert _list(batch.targets["future"]["values"]) == [1 - past]
