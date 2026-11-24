# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every physical file must agree with the schema before an index is built."""

import datetime as dt
import json
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from kumo.pytorch.api import (
    NON_TEMPORAL,
    BatchSpec,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DiskGraphEngine,
    DType,
    Encode,
    EntitySource,
    EntitySpec,
    Field,
    Fields,
    InMemoryEngine,
    InMemoryRows,
    Input,
    Parquet,
    Raw,
    SampleRequest,
    SampleSpec,
    Separate,
    TableSpec,
    Vocabulary,
    prepare,
)

diskgraph = pytest.importorskip("diskgraph")


def _contract(dtype):
    return DataLoaderSpec(
        schema=DataSchema(
            tables=(
                TableSpec(
                    name="rows",
                    primary_key="id",
                    columns={"id": DType(name="int64"), "value": dtype},
                    time_column=NON_TEMPORAL,
                ),
            )
        ),
        sample=SampleSpec(
            entity=EntitySpec(table="rows"), entity_values={"value": Field(column="value")}
        ),
        batch=BatchSpec(
            inputs={
                "values": Input(
                    source=EntitySource(),
                    representation=Fields(fields=(Encode(field="value", encoding=Raw()),)),
                    layout=Separate(),
                )
            }
        ),
    )


def test_a_later_file_cannot_change_the_declared_type(tmp_path):
    tables = tmp_path / "parts"
    tables.mkdir()
    pq.write_table(
        pa.table({"id": [1], "value": pa.array([1.5], type=pa.float64())}), tables / "one.parquet"
    )
    pq.write_table(pa.table({"id": [2], "value": ["not a number"]}), tables / "two.parquet")
    engine = DiskGraphEngine()
    with pytest.raises(ValueError, match=r"rows.value.*two.parquet.*declared.*physical"):
        prepare(
            contract=_contract(DType(name="float64")),
            dataset=Dataset(
                tables={
                    "rows": Parquet(path=str(tables)),
                }
            ),
            engine=engine,
        )
    assert not engine.index.exists()


def test_declared_int64_ingests_mixed_integer_and_scale_zero_decimal_files(tmp_path):
    minimum, maximum = -(2**63), 2**63 - 1
    cases = (
        (pa.int8(), -128),
        (pa.int16(), -32768),
        (pa.int32(), 2147483647),
        (pa.decimal128(38, 0), Decimal(minimum)),
        (pa.decimal256(76, 0), Decimal(maximum)),
        (pa.int32(), None),
    )
    parts = tmp_path / "parts"
    parts.mkdir()
    for row, (dtype, value) in enumerate(cases, start=1):
        key = Decimal(row) if pa.types.is_decimal(dtype) else row
        pq.write_table(
            pa.table({"id": pa.array([key], type=dtype), "value": pa.array([value], type=dtype)}),
            parts / f"{row}.parquet",
        )
    dataset = Dataset({"rows": Parquet(str(parts))})
    engine = DiskGraphEngine(num_threads=1)
    requests = tuple(SampleRequest(str(row), row, 0) for row in range(1, len(cases) + 1))
    with prepare(_contract(DType("int64")), dataset, engine) as plan:
        with plan.materialize(requests) as task:
            batch = next(task.batches(batch_size=len(cases), shuffle=False))
    field = batch.inputs["values"]["value"]
    assert field["values"].dtype == torch.int64
    assert field["values"].tolist() == [-128, -32768, 2147483647, minimum, maximum, 0]
    assert field["valid"].tolist() == [True, True, True, True, True, False]


@pytest.mark.parametrize("dtype", [pa.decimal128(38, 0), pa.decimal256(76, 0)])
@pytest.mark.parametrize("value", [-(2**63) - 1, 2**63])
def test_int64_ingest_rejects_scale_zero_decimal_overflow_in_a_later_file(tmp_path, dtype, value):
    parts = tmp_path / "parts"
    parts.mkdir()
    pq.write_table(pa.table({"id": [1], "value": [0]}), parts / "one.parquet")
    pq.write_table(
        pa.table({"id": [2], "value": pa.array([Decimal(value)], type=dtype)}),
        parts / "two.parquet",
    )
    engine = DiskGraphEngine(num_threads=1)
    with pytest.raises(diskgraph.DiskgraphSourceValidationError, match=r"rows.*value.*64-bit"):
        prepare(
            _contract(DType("int64")),
            Dataset({"rows": Parquet(str(parts))}),
            engine,
        )
    assert not engine.index.exists()


@pytest.mark.parametrize(
    "values",
    [
        pa.array([Decimal("42.0")], type=pa.decimal128(38, 1)),
        pa.array([42], type=pa.uint64()),
        pa.array([42.0], type=pa.float64()),
    ],
    ids=["scaled-decimal", "uint64", "float"],
)
def test_declared_int64_still_rejects_incompatible_physical_types(tmp_path, values):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": [1], "value": values}), path)
    with pytest.raises(ValueError, match=r"rows.value.*declared.*physical"):
        prepare(
            _contract(DType("int64")),
            Dataset({"rows": Parquet(str(path))}),
            DiskGraphEngine(),
        )


def test_untyped_all_null_source_is_not_guessed_from_a_declaration(tmp_path):
    source = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"id": [1], "value": [None]}), source)
    with pytest.raises(ValueError, match=r"rows.value.*unsupported physical type null"):
        prepare(
            contract=_contract(DType(name="float64")),
            dataset=Dataset(
                tables={
                    "rows": Parquet(path=str(source)),
                }
            ),
            engine=DiskGraphEngine(),
        )


def test_timestamp_physical_metadata_and_resolved_seconds_are_both_recorded(tmp_path):
    source = tmp_path / "rows.parquet"
    values = pa.array(
        [
            dt.datetime(1969, 12, 31, 23, 59, 59, 999_999, tzinfo=dt.UTC),
        ],
        type=pa.timestamp("us", tz="UTC"),
    )
    pq.write_table(pa.table({"id": [1], "value": values}), source)
    engine = DiskGraphEngine()
    contract = _contract(DType(name="timestamp", unit="s", timezone="UTC"))
    dataset = Dataset(tables={"rows": Parquet(path=str(source))})
    with prepare(contract=contract, dataset=dataset, engine=engine) as plan:
        task = plan.materialize(
            anchors=(SampleRequest(sample_id="one", entity_id=1, as_of_time=0),)
        )
        batch = next(task.batches(batch_size=1))
        assert batch.inputs["values"]["value"]["values"].to_pylist() == [
            dt.datetime(1970, 1, 1, tzinfo=dt.UTC),
        ]
    manifest = json.loads((engine.index / "index_manifest.json").read_text())
    assert manifest["source_types"]["rows"][str(source)]["value"] == "timestamp[us, tz=UTC]"
    assert manifest["column_types"]["rows"]["value"]["unit"] == "s"
    assert manifest["column_types"]["rows"]["value"]["timezone"] == "UTC"


def test_corrupt_schema_reports_the_table_and_file(tmp_path):
    source = tmp_path / "broken.parquet"
    source.write_bytes(b"not parquet")
    with pytest.raises(ValueError, match=r"rows.*Parquet.*broken.parquet"):
        prepare(
            contract=_contract(DType(name="float64")),
            dataset=Dataset(
                tables={
                    "rows": Parquet(path=str(source)),
                }
            ),
            engine=DiskGraphEngine(),
        )


_WIDE_ROWS = (
    {"id": 1, "label": "train", "blob": b"\x00\xff"},
    {"id": 2, "label": "véal", "blob": b""},
    {"id": 3, "label": "", "blob": b"z"},
    {"id": 4, "label": None, "blob": None},
    {"id": 5, "label": "train", "blob": b"\x00\xff"},
)


def _wide_contract(columns):
    return DataLoaderSpec(
        schema=DataSchema(
            tables=(
                TableSpec(name="rows", primary_key="id", columns=columns, time_column=NON_TEMPORAL),
            )
        ),
        sample=SampleSpec(
            entity=EntitySpec(table="rows"),
            entity_values={"label": Field(column="label"), "blob": Field(column="blob")},
        ),
        batch=BatchSpec(
            inputs={
                "values": Input(
                    source=EntitySource(),
                    representation=Fields(
                        fields=(
                            Encode(field="label", encoding=Raw()),
                            Encode(field="blob", encoding=Raw()),
                        )
                    ),
                    layout=Separate(),
                ),
                "ids": Input(
                    source=EntitySource(),
                    representation=Fields(fields=(Encode(field="label", encoding=Vocabulary()),)),
                    layout=Separate(),
                ),
            }
        ),
    )


def _batch_values(plan):
    requests = tuple(
        SampleRequest(sample_id=str(row["id"]), entity_id=row["id"], as_of_time=0)
        for row in _WIDE_ROWS
    )
    batch = next(plan.materialize(anchors=requests).batches(batch_size=len(requests)))
    return {
        name: {
            field: {key: value.tolist() for key, value in buffers.items()}
            for field, buffers in fields.items()
        }
        for name, fields in batch.inputs.items()
    }


def _write_rows(path, rows, string_type, binary_type):
    pq.write_table(
        pa.table(
            {
                "id": pa.array([row["id"] for row in rows], type=pa.int64()),
                "label": pa.array([row["label"] for row in rows], type=string_type),
                "blob": pa.array([row["blob"] for row in rows], type=binary_type),
            }
        ),
        path,
    )


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "inferred"])
@pytest.mark.parametrize("files", ["large", "mixed"])
def test_large_string_and_large_binary_parquet_match_the_reference(tmp_path, declared, files):
    # pandas 3 writes str columns as large_string ; "mixed" is a table
    # written partly by pandas 2 (string/binary) and partly by pandas 3.
    source = tmp_path / "rows"
    source.mkdir()
    if files == "large":
        _write_rows(source / "one.parquet", _WIDE_ROWS, pa.large_string(), pa.large_binary())
    else:
        _write_rows(source / "one.parquet", _WIDE_ROWS[:2], pa.string(), pa.binary())
        _write_rows(source / "two.parquet", _WIDE_ROWS[2:], pa.large_string(), pa.large_binary())
    assert [field.type for field in pq.read_schema(sorted(source.iterdir())[-1])][1:] == [
        pa.large_string(),
        pa.large_binary(),
    ]
    types = {"id": DType(name="int64"), "label": DType(name="string"), "blob": DType(name="bytes")}
    with prepare(
        contract=_wide_contract(types if declared else dict.fromkeys(types)),
        dataset=Dataset(tables={"rows": Parquet(path=str(source))}),
        engine=DiskGraphEngine(),
    ) as plan:
        actual = _batch_values(plan)
    with prepare(
        contract=_wide_contract(types),
        dataset=Dataset(tables={"rows": InMemoryRows(_WIDE_ROWS)}),
        engine=InMemoryEngine(),
    ) as plan:
        expected = _batch_values(plan)
    assert actual == expected
    valid = [True, True, True, False, True]
    assert actual["values"]["label"] == {
        "values": ["train", "véal", "", None, "train"],
        "valid": valid,
    }
    assert actual["values"]["blob"] == {
        "values": [b"\x00\xff", b"", b"z", None, b"\x00\xff"],
        "valid": valid,
    }
    ids = actual["ids"]["label"]
    assert ids["valid"] == valid
    train, veal, empty, missing, again = ids["values"]
    assert missing == Vocabulary().missing_id
    assert train == again
    assert len({train, veal, empty, missing, Vocabulary().unknown_id}) == 5
