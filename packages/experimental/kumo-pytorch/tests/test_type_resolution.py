# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoring inference resolves to the same saved, checked execution contract."""

from contextlib import nullcontext
from dataclasses import replace
from decimal import Decimal
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conformance.assertions import UnreadableRows, bound_case
from conformance.cases import discover_cases
from kumo.pytorch import api
from kumo.pytorch.contract.sample import reduction_dtype


def _source_contract(columns=None):
    return api.DataLoaderSpec(
        schema=api.DataSchema(
            tables=(
                api.TableSpec(
                    name="users", primary_key="id", columns=columns, time_column=api.NON_TEMPORAL
                ),
            )
        ),
        sample=api.SampleSpec(
            entity=api.EntitySpec(table="users"), entity_values={"value": api.Field(column="value")}
        ),
        batch=api.BatchSpec(
            inputs={
                "values": api.Input(
                    source=api.EntitySource(),
                    representation=api.Fields(
                        fields=(api.Encode(field="value", encoding=api.Raw()),)
                    ),
                    layout=api.Separate(),
                )
            }
        ),
    )


@pytest.mark.parametrize("columns", [None, {"id": api.DType(name="int64"), "value": None}])
@pytest.mark.parametrize("values", [[], [None], [1.5]])
def test_metadata_resolves_empty_null_and_present_columns_without_reading_rows(
    tmp_path, columns, values
):
    path = tmp_path / "users.parquet"
    pq.write_table(
        pa.table(
            {
                "id": pa.array(range(len(values)), type=pa.int64()),
                "value": pa.array(values, type=pa.float64()),
            }
        ),
        path,
    )
    dataset = api.Dataset(tables={"users": api.Parquet(path=str(path))})
    authoring = _source_contract(columns)
    with patch("pyarrow.parquet.read_table", side_effect=AssertionError("read source rows")):
        resolved = authoring.resolve(dataset=dataset)
    expected = _source_contract({"id": api.DType(name="int64"), "value": api.DType(name="float64")})
    assert resolved == expected
    assert resolved.fingerprint == expected.fingerprint
    assert resolved.input_fingerprint == expected.input_fingerprint
    assert api.DataLoaderSpec.from_json(resolved.to_json()) == expected
    with pytest.raises(ValueError, match="unresolved column types"):
        authoring.to_json()


def test_inferred_parquet_types_are_used_by_prepare_and_checked_on_reload(tmp_path):
    pytest.importorskip("diskgraph")
    path = tmp_path / "users.parquet"
    pq.write_table(pa.table({"id": [1], "value": [1.5]}), path)
    dataset = api.Dataset(tables={"users": api.Parquet(path=str(path))})
    with api.prepare(
        contract=_source_contract(),
        dataset=dataset,
        engine=api.DiskGraphEngine(index=tmp_path / "first"),
    ) as plan:
        saved = plan.contract.to_json()
        with plan.materialize(
            anchors=(api.SampleRequest(sample_id="one", entity_id=1, as_of_time=0),)
        ) as task:
            assert next(task.batches(batch_size=1)).inputs["values"]["value"][
                "values"
            ].tolist() == [1.5]
    pq.write_table(pa.table({"id": [1], "value": [2]}), path)
    with pytest.raises(ValueError, match=r"users.value.*declared.*physical"):
        api.prepare(
            contract=api.DataLoaderSpec.from_json(saved),
            dataset=dataset,
            engine=api.DiskGraphEngine(index=tmp_path / "changed"),
        )


@pytest.mark.parametrize("columns", [None, {"id": api.DType(name="int64"), "value": None}])
def test_memory_types_are_required_before_any_row_read(columns):
    dataset = api.Dataset(tables={"users": api.InMemoryRows(rows=UnreadableRows())})
    issues = api.check(
        contract=_source_contract(columns), dataset=dataset, engine=api.InMemoryEngine()
    )
    assert len(issues) == 1
    assert "declare every DType for InMemoryRows" in issues[0].message
    with pytest.raises(api.ValidationError, match="type inference requires Parquet"):
        api.prepare(
            contract=_source_contract(columns), dataset=dataset, engine=api.InMemoryEngine()
        )


def test_partial_declarations_check_explicit_types_and_every_file(tmp_path):
    pq.write_table(pa.table({"id": [1], "value": [1.5]}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"id": [2], "value": [2]}), tmp_path / "b.parquet")
    dataset = api.Dataset(tables={"users": api.Parquet(path=str(tmp_path))})
    with pytest.raises(ValueError, match=r"users.*b.parquet.*declared.*physical"):
        _source_contract().resolve(dataset=dataset)
    with pytest.raises(ValueError, match=r"users.*a.parquet.*declared.*physical"):
        _source_contract({"id": api.DType(name="string"), "value": None}).resolve(dataset=dataset)


def test_discovery_checks_all_file_columns_while_explicit_selection_can_ignore_extras(tmp_path):
    pq.write_table(pa.table({"id": [1], "value": [1.5]}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"id": [2], "value": [2.5], "extra": [3]}), tmp_path / "b.parquet")
    dataset = api.Dataset(tables={"users": api.Parquet(path=str(tmp_path))})
    with pytest.raises(ValueError, match=r"users.*b.parquet.*column names differ"):
        _source_contract().resolve(dataset=dataset)
    selected = _source_contract({"id": None, "value": None}).resolve(dataset=dataset)
    assert set(selected.schema.tables[0].columns) == {"id", "value"}


def test_metadata_inference_honors_logical_column_renames(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"physical_id": [1], "physical_value": [1.5]}), path)
    resolved = _source_contract().resolve(
        dataset=api.Dataset(
            tables={
                "users": api.Parquet(
                    path=str(path), columns={"physical_id": "id", "physical_value": "value"}
                )
            }
        )
    )
    assert resolved.schema.tables[0].columns == {
        "id": api.DType(name="int64"),
        "value": api.DType(name="float64"),
    }


def test_missing_binding_is_reported_before_unresolved_native_preflight(tmp_path):
    contract = _source_contract()
    contract = replace(
        contract,
        sample=replace(
            contract.sample,
            entity_values={
                "value": api.Aggregate(source=api.Rows(), operation="sum", column="value")
            },
        ),
    )
    dataset = api.Dataset(tables={"other": api.Parquet(path=str(tmp_path / "missing.parquet"))})
    with patch("pyarrow.parquet.read_schema", side_effect=AssertionError("read missing binding")):
        issues = api.check(
            contract=contract, dataset=dataset, engine=api.DiskGraphEngine(index=tmp_path / "index")
        )
    assert len(issues) == 2
    assert all(issue.path == "dataset" for issue in issues)


def test_inference_preserves_invalid_scope_and_duplicate_occurrence_errors():
    contract = _source_contract({"id": api.DType(name="int64"), "value": api.DType(name="int64")})
    inferred = {"count": api.Aggregate(source=api.Rows(), operation="count")}
    invalid = replace(
        contract,
        sample=replace(
            contract.sample, entity=api.EntitySpec(table="missing"), entity_values=inferred
        ),
    )
    with pytest.raises(api.ValidationError, match="unknown table 'missing'"):
        invalid.validate()
    invalid = replace(
        contract,
        sample=replace(
            contract.sample,
            entity_values=inferred,
            dated_entity_values={
                "fact": api.DatedEntityValue(
                    rows=api.Rows(),
                    field=api.Field(column="value"),
                    fields={"fact": api.Constant(value=4, dtype=api.DType(name="int64"))},
                )
            },
        ),
    )
    with pytest.raises(api.ValidationError, match="duplicates its primary value"):
        invalid.validate()


@pytest.mark.parametrize("case_id", ["C-CALC-001", "C-CALC-002", "C-HISTORY-002"])
@pytest.mark.parametrize("engine_name", ["in_memory", "diskgraph"])
def test_inferred_calculations_have_the_same_saved_contract_and_output(case_id, engine_name):
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
    case = discover_cases()[case_id]
    explicit = case.contract
    if case_id == "C-HISTORY-002":
        event = explicit.sample.events["transaction"]
        fields = dict(event.fields)
        fields["age"] = api.EventTimeDelta(unit="seconds")
        sample = replace(explicit.sample, events={"transaction": replace(event, fields=fields)})
    else:
        sample = replace(
            explicit.sample,
            entity_values={
                name: replace(value, dtype=None)
                if isinstance(value, (api.Derived, api.Aggregate))
                else value
                for name, value in explicit.sample.entity_values.items()
            },
        )
    authoring = replace(explicit, sample=sample)
    assert authoring.resolve() == explicit
    assert authoring.to_json() == explicit.to_json()
    assert authoring.fingerprint == explicit.fingerprint
    assert authoring.input_fingerprint == explicit.input_fingerprint
    with (
        bound_case(case, engine_name) as (dataset, engine),
        api.prepare(contract=authoring, dataset=dataset, engine=engine) as plan,
    ):
        assert plan.contract == explicit
        with plan.materialize(anchors=case.requests) as task:
            next(task.batches(batch_size=len(case.requests)))


def test_fixed_result_operations_and_grouped_reductions_infer_without_values():
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                name="users",
                primary_key="id",
                columns={
                    "id": api.DType(name="int64"),
                    "at": api.DType(name="timestamp", unit="s", timezone="UTC"),
                },
                time_column="at",
            ),
        )
    )
    grouped = api.GroupBy(
        source=api.Rows(), keys=("id",), values={"n": api.Reduce(operation="count")}
    )
    contract = api.DataLoaderSpec(
        schema=schema,
        sample=api.SampleSpec(
            entity=api.EntitySpec(table="users"),
            entity_values={
                "month": api.Derived(operation=api.CalendarPart(part="month"), inputs=("time",)),
                "time": api.Field(column="at"),
                "mean": api.Aggregate(source=grouped, operation="mean", column="n"),
            },
            targets={
                "label": api.Compare(
                    left=api.Aggregate(source=api.Rows(), operation="count"),
                    operation="greater_than",
                    right=0,
                )
            },
        ),
        batch=api.BatchSpec(
            inputs={
                "month": api.Input(
                    source=api.EntitySource(),
                    representation=api.Fields(
                        fields=(api.Encode(field="month", encoding=api.Raw()),)
                    ),
                    layout=api.Separate(),
                )
            }
        ),
    )
    resolved = contract.resolve()
    assert resolved.sample.entity_values["month"].dtype == api.DType(name="int64")
    assert resolved.sample.entity_values["mean"].dtype == api.DType(name="float64")
    assert resolved.sample.entity_values["mean"].source.values["n"].dtype == api.DType(name="int64")
    assert resolved.sample.targets["label"].dtype == api.DType(name="bool")
    assert resolved.sample.targets["label"].left.dtype == api.DType(name="int64")


@pytest.mark.parametrize(
    "source,result",
    [
        ("int8", "int64"),
        ("int64", "int64"),
        ("uint8", "uint64"),
        ("uint64", "uint64"),
        ("float32", "float64"),
    ],
)
def test_sum_infers_exact_integer_or_floating_result(source, result):
    columns = {"value": api.DType(name=source)}
    assert reduction_dtype(api.Reduce("sum", "value"), columns) == api.DType(name=result)
    with pytest.raises(ValueError, match="sum cannot produce"):
        reduction_dtype(api.Reduce("sum", "value", dtype=api.DType(name="bool")), columns)
    if source.startswith(("int", "uint")):
        with pytest.raises(ValueError, match="sum cannot produce"):
            reduction_dtype(api.Reduce("sum", "value", dtype=api.DType(name="float64")), columns)


@pytest.mark.parametrize("engine_name", ["in_memory", "diskgraph"])
@pytest.mark.parametrize(
    "left_type,right_type,left,right,operation,expected",
    [
        (api.DType(name="int64"), api.DType(name="float64"), 10, 0.5, "add", 10.5),
        (api.DType(name="uint64"), api.DType(name="float64"), 2**63, 0.5, "add", float(2**63)),
        (api.DType(name="int32"), api.DType(name="float32"), 16777217, -16777216.0, "add", 0.0),
        (
            api.DType(name="float32"),
            api.DType(name="float32"),
            1e38,
            10.0,
            "multiply",
            float("inf"),
        ),
        (api.DType(name="float64"), api.DType(name="float64"), 1e308, 1e308, "add", float("inf")),
        (
            api.DType(name="decimal128", precision=6, scale=1),
            api.DType(name="float64"),
            Decimal("1.5"),
            0.5,
            "add",
            2.0,
        ),
        (api.DType(name="int8"), api.DType(name="uint8"), -1, 255, "add", 254),
        pytest.param(
            api.DType(name="int64"),
            api.DType(name="uint64"),
            -1,
            2**63,
            "add",
            None,
            id="integer-operand-out-of-range",
        ),
    ],
)
def test_numeric_promotion_and_overflow_match_in_both_output_forms(
    engine_name, left_type, right_type, left, right, operation, expected
):
    if engine_name == "diskgraph":
        pytest.importorskip("diskgraph")
    case = discover_cases()["C-CALC-001"]
    table = replace(
        case.contract.schema.tables[0],
        columns={"id": api.DType(name="int64"), "a": left_type, "b": right_type},
    )
    contract = api.DataLoaderSpec(
        schema=api.DataSchema(tables=(table,)),
        sample=api.SampleSpec(
            entity=api.EntitySpec(table="users"),
            entity_values={
                "a": api.Field(column="a"),
                "b": api.Field(column="b"),
                "result": api.Derived(operation=operation, inputs=("a", "b")),
            },
        ),
        batch=api.BatchSpec(
            inputs={
                "values": api.Input(
                    source=api.EntitySource(),
                    representation=api.Fields(
                        fields=(api.Encode(field="result", encoding=api.Raw()),)
                    ),
                    layout=api.Separate(),
                )
            }
        ),
    ).resolve()
    case = replace(case, contract=contract, tables={"users": [{"id": 1, "a": left, "b": right}]})
    outcome = (
        pytest.raises(ValueError, match="not a valid|not in range")
        if expected is None
        else nullcontext()
    )
    with (
        outcome,
        bound_case(case, engine_name) as (dataset, engine),
        api.prepare(contract=contract, dataset=dataset, engine=engine) as plan,
        plan.materialize(anchors=case.requests) as task,
    ):
        formats = ("inspectable", "model") if engine_name == "diskgraph" else ("inspectable",)
        for output_format in formats:
            output = next(task.batches(batch_size=1, output_format=output_format)).inputs["values"][
                "result"
            ]
            assert output["values"].tolist() == [expected]
            assert output["valid"].tolist() == [True]
