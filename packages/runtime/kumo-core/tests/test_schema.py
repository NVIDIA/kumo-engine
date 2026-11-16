# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict

import pytest
from kumo_core.types.schema import (
    FPID,
    BaseColumnConfig,
    ColumnArgs,
    IngestedTableFingerprint,
    PartialTableFingerprint,
    SqlTableFingerprint,
    TableFingerprint,
    TrainingTableFingerprint,
    ViewFingerprint,
    source_name_path_token,
    time_col_config_to_fp,
)
from kumoapi.typing import Dtype, Stype
from pydantic import TypeAdapter, ValidationError


def test_column_configs_validate_and_convert_nested_values() -> None:
    column = BaseColumnConfig(
        name="amount",
        stype="numerical",
        dtype="float32",
        metadata={"unit": "usd"},
    )

    assert column.stype is Stype.numerical
    assert column.dtype is Dtype.float32
    assert asdict(column) == {
        "name": "amount",
        "stype": Stype.numerical,
        "dtype": Dtype.float32,
        "na_values": None,
        "pre_transform": None,
        "metadata": {
            "unit": "usd"
        },
        "timestamp_unit": None,
        "multicategorical_delimiter": None,
    }

    with pytest.raises(ValidationError):
        BaseColumnConfig(name=1)
    with pytest.raises(ValidationError):
        BaseColumnConfig(name="amount", metadata=None)


def test_column_update_and_hash_preserve_existing_contract() -> None:
    destination = BaseColumnConfig(name="old")
    source = BaseColumnConfig(
        name="new",
        metadata={"format": "percent"},
        timestamp_unit="ms",
    )

    assert destination.update(source, override=False) is destination
    assert destination == BaseColumnConfig(name="old")
    assert destination.update(source) is destination
    assert destination.name == "new"
    assert destination.metadata == {"format": "percent"}
    assert destination.timestamp_unit == "ms"

    args = ColumnArgs(
        name="tags",
        stype=Stype.multicategorical,
        dtype=Dtype.string,
        multicategorical_delimiter="|",
    )
    expected_hash_input = ("tags" + str(Stype.multicategorical) +
                           str(Dtype.string) + "multicategorical_delimiter=|")
    assert hash(args) == hash(expected_hash_input)
    assert args.as_base_column_config() == BaseColumnConfig(
        name="tags",
        stype=Stype.multicategorical,
        dtype=Dtype.string,
        multicategorical_delimiter="|",
    )


def test_fpid_validates_direct_and_pydantic_inputs() -> None:
    value = "0123456789abcdef0123456789abcdef"

    assert FPID(value) == value
    assert TypeAdapter(FPID).validate_python(value) == value
    assert TypeAdapter(FPID).dump_json(FPID(value)) == f'"{value}"'.encode()
    assert FPID.is_valid(value)
    assert not FPID.is_valid("A" * 32)
    assert not FPID.is_valid("")

    with pytest.raises(ValueError, match="Please fill out this value"):
        FPID("")
    with pytest.raises(ValueError, match="32-char hex digits"):
        FPID("g" * 32)
    with pytest.raises(ValidationError):
        TypeAdapter(FPID).validate_python(1)


def test_ingested_fingerprint_validation_serialization_and_identity() -> None:
    fingerprint = IngestedTableFingerprint(
        connector_name="conn",
        source_name="src",
        cols={
            "b": "float32",
            "a": "time"
        },
        pkey=None,
        time_col={
            "name": "t",
            "timestamp_unit": "ms",
            "metadata": {
                "TS_FORMAT": "fmt"
            },
        },
        end_time_col=None,
    )

    assert fingerprint.cols == {"b": Dtype.float32, "a": Dtype.time}
    assert isinstance(fingerprint.time_col, BaseColumnConfig)
    assert time_col_config_to_fp(fingerprint.time_col) == "t:ms:fmt"
    assert fingerprint.fpid() == (
        "fpv3.conn.src_25a66342.280a01430b69748e0127b9a1871a67cf")
    assert fingerprint.ingest_path_fpid == (
        "fpv3.conn."
        "src_25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1"
        "f5d44848a26.schema_"
        "b4fe30a6ffa4760c7a151a58bc218b5e6a71ca9f56da7b88788e6898f321d8b4")

    payload = TypeAdapter(IngestedTableFingerprint).dump_json(fingerprint)
    restored = TypeAdapter(IngestedTableFingerprint).validate_json(payload)
    assert restored == fingerprint
    assert restored.fpid() == fingerprint.fpid()

    with_delimiter = IngestedTableFingerprint(**{
        **asdict(fingerprint),
        "multicategorical_delimiters": {
            "b": "|"
        },
    })
    assert with_delimiter.fpid() != fingerprint.fpid()
    assert with_delimiter.ingest_path_fpid == fingerprint.ingest_path_fpid

    wider = IngestedTableFingerprint(**{
        **asdict(fingerprint),
        "cols": {
            **fingerprint.cols,
            "c": Dtype.string,
        },
    })
    recast = IngestedTableFingerprint(**{
        **asdict(fingerprint),
        "cols": {
            **fingerprint.cols,
            "a": Dtype.string,
        },
    })
    assert wider.ingest_path_fpid != fingerprint.ingest_path_fpid
    assert recast.ingest_path_fpid != fingerprint.ingest_path_fpid


@pytest.mark.parametrize(
    ("payload", "expected_type", "expected_fpid"),
    [
        (
            {
                "_type": "partial_ingested",
                "connector_name": "conn",
                "source_name": "src",
                "cols": ["b", "a"],
                "pkey": None,
                "time_col": None,
                "end_time_col": None,
            },
            PartialTableFingerprint,
            "fpv3.conn.src_25a66342.0709abc3c986a6954ebbd3141df06e49",
        ),
        (
            {
                "_type": "view",
                "connector_name": "conn",
                "source_name": "src",
            },
            ViewFingerprint,
            "conn."
            "src_25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1"
            "f5d44848a26",
        ),
        (
            {
                "_type": "sql_table",
                "connector_name": "conn",
                "source_name": "src",
                "cols": {},
                "pkey": None,
                "time_col": None,
                "end_time_col": None,
                "data_version": "version-one",
            },
            SqlTableFingerprint,
            "fpv3.sql.conn.src",
        ),
        (
            {
                "_type": "training",
                "connector_name": "conn",
                "source_name": "src",
            },
            TrainingTableFingerprint,
            "conn."
            "src_25a6634263c1b1f6fc4697a04e2b9904ea4b042a89af59dc93ec1"
            "f5d44848a26",
        ),
    ],
)
def test_table_fingerprint_union_round_trip(
    payload: dict[str, object],
    expected_type: type,
    expected_fpid: str,
) -> None:
    adapter = TypeAdapter(TableFingerprint)

    fingerprint = adapter.validate_python(payload)
    restored = adapter.validate_json(adapter.dump_json(fingerprint))

    assert type(fingerprint) is expected_type
    assert restored == fingerprint
    assert fingerprint.fpid() == expected_fpid


def test_fingerprint_fields_that_do_not_define_identity_stay_excluded() -> None:
    common = {
        "connector_name": "conn",
        "source_name": "src",
        "cols": {
            "value": Dtype.float32
        },
        "time_col": None,
        "end_time_col": None,
    }
    first = IngestedTableFingerprint(pkey="first", **common)
    second = IngestedTableFingerprint(pkey="second", **common)
    assert first.fpid() == second.fpid()

    sql_first = SqlTableFingerprint(
        pkey=None,
        data_version="one",
        **common,
    )
    sql_second = SqlTableFingerprint(
        pkey="value",
        data_version="two",
        **common,
    )
    assert sql_first.fpid() == sql_second.fpid()
    assert source_name_path_token("src").startswith("src_")
