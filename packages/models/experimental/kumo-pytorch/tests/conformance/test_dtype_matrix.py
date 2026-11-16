# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every scalar dtype/encoder pair has a checked meaning on each selected engine."""

import datetime as dt
from decimal import Decimal
from unittest.mock import patch

import pytest

from conformance.assertions import assert_projection, bound_tables
from kumo.pytorch.api import (
    NON_TEMPORAL,
    BatchSpec,
    Buckets,
    DataLoaderSpec,
    DataSchema,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    Field,
    Fields,
    Input,
    Raw,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    Text,
    ValidationError,
    Vocabulary,
    ZScore,
    check,
    prepare,
)

DTYPES = (
    *(DType(name=f"int{width}") for width in (8, 16, 32, 64)),
    *(DType(name=f"uint{width}") for width in (8, 16, 32, 64)),
    DType(name="float32"),
    DType(name="float64"),
    DType(name="decimal128", precision=8, scale=2),
    DType(name="decimal256", precision=8, scale=2),
    DType(name="bool"),
    DType(name="string"),
    DType(name="bytes"),
    DType(name="date32"),
    DType(name="timestamp", unit="s", timezone="UTC"),
)


def pytest_generate_tests(metafunc):
    if "engine_name" in metafunc.fixturenames:
        selected = metafunc.config.getoption("--conformance-engine")
        engines = ("in_memory", "diskgraph") if selected == "all" else (selected,)
        metafunc.parametrize("engine_name", engines)


def _values(dtype):
    if dtype.name == "bool":
        return [False, True]
    if dtype.name == "string":
        return ["a", "b"]
    if dtype.name == "bytes":
        return [b"a", b"b"]
    if dtype.name == "date32":
        return [dt.date(1970, 1, 2), dt.date(1970, 1, 4)]
    if dtype.name == "timestamp":
        return [dt.datetime(1970, 1, 2, tzinfo=dt.UTC), dt.datetime(1970, 1, 4, tzinfo=dt.UTC)]
    if dtype.name.startswith("decimal"):
        return [Decimal("1.00"), Decimal("3.00")]
    return [dtype.normalize(1), dtype.normalize(3)]


def _contract(dtype, encoding):
    return DataLoaderSpec(
        schema=DataSchema(
            tables=(
                TableSpec(
                    name="users",
                    primary_key="id",
                    columns={"id": DType(name="int64"), "value": dtype},
                    time_column=NON_TEMPORAL,
                ),
            ),
            relations=(),
        ),
        sample=SampleSpec(
            entity=EntitySpec(table="users"), entity_values={"value": Field(column="value")}
        ),
        batch=BatchSpec(
            inputs={
                "entity": Input(
                    source=EntitySource(),
                    representation=Fields(fields=(Encode(field="value", encoding=encoding),)),
                    layout=Separate(),
                )
            }
        ),
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda value: value.name)
@pytest.mark.parametrize("family", ("raw", "vocabulary", "zscore", "buckets", "text"))
def test_scalar_compatibility_and_real_execution(dtype, family, engine_name):
    source_values = _values(dtype)
    encodings = {
        "raw": Raw(),
        "vocabulary": Vocabulary(values=tuple(zip(source_values, (2, 3), strict=True))),
        "zscore": ZScore(),
        "buckets": Buckets(boundaries=(2.0,)),
        "text": Text(vocabulary_size=16),
    }
    contract = _contract(dtype, encodings[family])
    tables = {
        "users": [{"id": index, "value": value} for index, value in enumerate(source_values, 1)]
    }
    accepted = (
        family in {"raw", "vocabulary"}
        or family in {"zscore", "buckets"}
        and dtype.numeric
        or family == "text"
        and dtype.name == "string"
    )
    with bound_tables(contract, tables, engine_name, refusal=True) as (dataset, engine):
        with patch.object(engine, "open", side_effect=AssertionError("opened during preflight")):
            issues = check(contract, dataset, engine)
            if accepted:
                assert not issues
            else:
                assert len(issues) == 1
                assert issues[0].path == "batch.inputs.entity.representation.fields[0]"
                assert "requires" in issues[0].message
                assert issues[0].suggestion
                with pytest.raises(ValidationError) as refused:
                    prepare(contract, dataset, engine)
                assert refused.value.issues == issues
    if not accepted:
        return
    anchors = tuple(
        SampleRequest(sample_id=f"u{index}", entity_id=index, as_of_time=100) for index in (1, 2)
    )
    with (
        bound_tables(contract, tables, engine_name) as (dataset, engine),
        prepare(contract, dataset, engine) as plan,
        plan.materialize(anchors) as task,
    ):
        outputs = [next(task.batches(anchors, batch_size=2)).inputs["entity"]["value"]]
        torch_scalar = dtype.integer or dtype.name in {"float32", "float64", "bool"}
        if engine_name == "diskgraph":
            if family == "raw" and not torch_scalar:
                with (
                    patch.object(
                        plan._source,
                        "build_model",
                        side_effect=AssertionError("fetched unrepresentable raw batch"),
                    ),
                    pytest.raises(ValueError, match="Raw"),
                ):
                    next(task.batches(anchors, batch_size=2, output_format="model"))
            else:
                outputs.append(
                    next(task.batches(anchors, batch_size=2, output_format="model")).inputs[
                        "entity"
                    ]["value"]
                )
        for actual in outputs:
            if family == "text":
                assert_projection(
                    actual["piece_offsets"], {"dtype": "int64", "shape": [3], "values": [0, 1, 2]}
                )
                assert str(actual["values"].dtype) == "torch.int64"
                assert_projection(
                    actual["valid"], {"dtype": "bool", "shape": [2], "values": [True, True]}
                )
                continue
            if family == "raw":
                expected_values = source_values
                expected_dtype = dtype.name if torch_scalar else str(dtype.to_arrow())
            elif family == "vocabulary":
                expected_values, expected_dtype = [2, 3], "int64"
            elif family == "zscore":
                expected_values, expected_dtype = [-1.0, 1.0], "float32"
            else:
                expected_values, expected_dtype = [1, 2], "int64"
            assert_projection(
                actual["values"], {"dtype": expected_dtype, "shape": [2], "values": expected_values}
            )
            assert_projection(
                actual["valid"], {"dtype": "bool", "shape": [2], "values": [True, True]}
            )
