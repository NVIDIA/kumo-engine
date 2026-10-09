# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from statedb.db.json_storage import (
    encode_timestamp,
    payload_dict_for_storage,
    payload_json_for_storage,
)
from statedb.db.row_parsing import coerce_datetime, row_field, row_str


class _NameIndexedRow:

    def __getitem__(self, key):
        if key == "id":
            return "by-name"
        raise KeyError(key)


@dataclass
class _Payload:
    id: str
    name: str
    count: int


def test_row_field_supports_dict_name_index_and_position() -> None:
    assert row_field({"id": "dict-id"}, "id", 0) == "dict-id"
    assert row_str(_NameIndexedRow(), "id", 0) == "by-name"
    assert row_field(("tuple-id", ), "id", 0) == "tuple-id"


def test_coerce_datetime_accepts_datetime_and_iso_string() -> None:
    value = datetime(2026, 1, 1, tzinfo=timezone.utc)

    assert coerce_datetime(value) is value
    assert coerce_datetime(value.isoformat()) == value
    with pytest.raises(TypeError, match="Unsupported timestamp value"):
        coerce_datetime(None)


def test_json_storage_helpers_encode_backend_payloads() -> None:
    value = datetime(2026, 1, 1, tzinfo=timezone.utc)
    payload = payload_dict_for_storage(
        _Payload(id="row-id", name="example", count=1),
        omit_fields={"id"},
        extra_fields={"search": "example"},
    )

    assert payload == {"name": "example", "count": 1, "search": "example"}
    assert payload_json_for_storage(payload) == (
        '{"count": 1, "name": "example", "search": "example"}')
    assert encode_timestamp(SimpleNamespace(backend="sqlite"), value) == (
        value.isoformat())
    assert encode_timestamp(SimpleNamespace(backend="postgres"), value) is value
