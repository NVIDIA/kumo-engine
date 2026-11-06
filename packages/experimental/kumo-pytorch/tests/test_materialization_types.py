# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every physical file must agree with the schema before an index is built."""

import datetime as dt
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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

pytest.importorskip("diskgraph")


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
    index = tmp_path / "index"
    with pytest.raises(ValueError, match=r"rows.value.*two.parquet.*declared.*physical"):
        prepare(
            contract=_contract(DType(name="float64")),
            dataset=Dataset(
                tables={
                    "rows": Parquet(path=str(tables)),
                }
            ),
            engine=DiskGraphEngine(index=index),
        )
    assert not index.exists()


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
            engine=DiskGraphEngine(index=tmp_path / "index"),
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
    index = tmp_path / "index"
    contract = _contract(DType(name="timestamp", unit="s", timezone="UTC"))
    dataset = Dataset(tables={"rows": Parquet(path=str(source))})
    with prepare(contract=contract, dataset=dataset, engine=DiskGraphEngine(index=index)) as plan:
        task = plan.materialize(
            anchors=(SampleRequest(sample_id="one", entity_id=1, as_of_time=0),)
        )
        batch = next(task.batches(batch_size=1))
        assert batch.inputs["values"]["value"]["values"].to_pylist() == [
            dt.datetime(1970, 1, 1, tzinfo=dt.UTC),
        ]
    manifest = json.loads((index / "index_manifest.json").read_text())
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
            engine=DiskGraphEngine(index=tmp_path / "index"),
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
        engine=DiskGraphEngine(index=tmp_path / "index"),
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
