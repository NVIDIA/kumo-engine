# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Arrow permits unaligned buffers, independent chunks, offsets and null bitmaps."""
import json
import struct

import pyarrow as pa
import pytest
from kumo.fm.native_json import encode_rows


@pytest.mark.parametrize("dtype,format,values,expected", [
    (pa.int64(), "q", [-(2**63), 0, 2**63-1], [str(-(2**63)), 0, str(2**63-1)]),
    (pa.float64(), "d", [1.25, -0.0, 3.5], [1.25, -0.0, 3.5]),
    (pa.date32(), "i", [-1, 0, 1], ["1969-12-31", "1970-01-01", "1970-01-02"]),
    (pa.timestamp("us", tz="UTC"), "q", [-1, 0, 1],
     ["1969-12-31T23:59:59.999999Z", "1970-01-01T00:00:00.000000Z", "1970-01-01T00:00:00.000001Z"]),
])
def test_unaligned_typed_arrow_buffers(dtype, format, values, expected):
    data = pa.py_buffer(b'x' + struct.pack('<' + format * len(values), *values)).slice(1)
    assert data.address % (8 if format in ("q", "d") else 4) != 0
    array = pa.Array.from_buffers(dtype, len(values), [None, data])
    for table, result in [(pa.table({"value": array}), [[v] for v in expected]),
                          (pa.table({"value": array.slice(1)}), [[v] for v in expected[1:]])]:
        assert json.loads(encode_rows(table, 1000).data) == result


@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 63, 64, 65])
def test_null_bitmaps_slices_and_independent_column_chunks(size):
    values = [None if i % 3 else i for i in range(size + 3)]
    text = [None if i % 2 else "用户😀\x00\n" for i in range(size + 3)]
    table = pa.table({
        "id": pa.chunked_array([pa.array(values[:2], pa.int64()), pa.array(values[2:], pa.int64())]),
        "text": pa.chunked_array([pa.array(text[:5], pa.string()), pa.array(text[5:], pa.string())]),
    }).slice(3, size)
    expected = [[i, t] for i, t in zip(values[3:], text[3:], strict=True)]
    body = json.dumps(expected, ensure_ascii=True, separators=(',', ':')).encode('ascii')
    assert encode_rows(table, len(body)).data == body
    with pytest.raises(ValueError, match="byte budget"):
        encode_rows(table, len(body) - 1)
