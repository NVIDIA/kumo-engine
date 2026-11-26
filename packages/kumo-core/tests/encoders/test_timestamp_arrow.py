# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
from kumo_core.encoders.tensor_mapper import TimestampTensorMapper
from kumo_core.features.time import to_datetime
from numpy.typing import NDArray
from tests.support.timestamp_reference import timestamp_reference
from torch_geometric.testing import withCUDA


@withCUDA
@pytest.mark.parametrize(
    "layout", ["plain", "no_null", "sliced", "chunked", "empty", "null"]
)
def test_timestamp_arrow_matches_pandas(
    device: torch.device,
    layout: str,
) -> None:
    """Preserve all features and input buffers across Arrow layouts and devices."""
    limits = np.iinfo(np.int64)
    values = (
        np.random.default_rng(0)
        .integers(limits.min + 1, limits.max, size=2048, dtype=np.int64)
        .tolist()
    )
    values.extend(
        [None, limits.min, limits.min + 1, limits.max, -1, 0, 1, -(10**9) - 1]
    )
    # Include leap-century boundaries and an observed federal holiday.
    dates = pa.array(["1900-03-01", "2000-02-29", "2006-01-02", "2100-03-01"]).cast(
        pa.timestamp("ns")
    )
    base = pa.concat_arrays([pa.array(values, type=pa.timestamp("ns")), dates])
    array: pa.Array | pa.ChunkedArray = base
    if layout == "no_null":
        array = base.drop_null()
    elif layout == "sliced":
        array = base.slice(1, len(base) - 2)
    elif layout == "chunked":
        array = pa.chunked_array([base[:7], base[7:]])
    elif layout == "empty":
        array = base[:0]
    elif layout == "null":
        array = pa.nulls(8, type=pa.timestamp("ns"))
    chunks = array.chunks if isinstance(array, pa.ChunkedArray) else [array]
    buffers_before = [
        [
            buffer.to_pybytes() if buffer is not None else None
            for buffer in chunk.buffers()
        ]
        for chunk in chunks
    ]
    mapper = TimestampTensorMapper()

    with patch(
        "kumo_core.encoders.tensor_mapper.to_datetime",
        side_effect=AssertionError("Native nanoseconds must bypass pandas"),
    ):
        actual = mapper(array, device=device)

    combined = array.combine_chunks() if isinstance(array, pa.ChunkedArray) else array
    expected = timestamp_reference(mapper, combined, device)

    assert torch.equal(actual, expected)
    assert actual.shape == (len(array), 16)
    assert actual.dtype == torch.int64
    assert actual.device == device
    assert buffers_before == [
        [
            buffer.to_pybytes() if buffer is not None else None
            for buffer in chunk.buffers()
        ]
        for chunk in chunks
    ]


@withCUDA
@pytest.mark.parametrize(
    ("array", "format"),
    [
        (pa.array(["05/01/2018", "invalid", None]), "%d/%m/%Y"),
        *[
            (pa.array([-1, 0, 1, None], type=pa.timestamp(unit)), None)
            for unit in ["s", "ms", "us"]
        ],
        (pa.array([0, 1, None], type=pa.timestamp("ns", tz="UTC")), None),
    ],
)
def test_timestamp_arrow_fallback(
    device: torch.device,
    array: pa.Array,
    format: str | None,
) -> None:
    """Keep parsing, units, and timezone errors on the existing pandas path."""
    mapper = TimestampTensorMapper(format=format)
    with patch(
        "kumo_core.encoders.tensor_mapper.to_datetime", wraps=to_datetime
    ) as convert:
        if pa.types.is_timestamp(array.type) and array.type.tz is not None:
            with pytest.raises(TypeError):
                mapper(array, device=device)
        else:
            actual = mapper(array, device=device)
            series = to_datetime(array.to_pandas(), format=format)
            expected = mapper(pa.Array.from_pandas(series), device=device)
            assert torch.equal(actual, expected)
        convert.assert_called_once()


@withCUDA
def test_timestamp_lookup_every_day(device: torch.device) -> None:
    """Match all supported days and the final nanosecond of each day exactly."""
    mapper = TimestampTensorMapper()
    first = pd.Timestamp("1678-01-01").value // 86_400_000_000_000
    last = pd.Timestamp.max.value // 86_400_000_000_000
    days: NDArray[np.int64] = np.arange(first, last + 1, dtype=np.int64)
    midnights: NDArray[np.int64] = days * 86_400_000_000_000
    ends: NDArray[np.int64] = midnights[:-1] + 86_400_000_000_000 - 1
    values = np.concatenate([midnights, ends, [pd.Timestamp.max.value]])
    array = pa.array(values, type=pa.timestamp("ns"))

    actual = mapper(array, device=device)
    expected = timestamp_reference(mapper, array, device)

    assert torch.equal(actual, expected)
    table = mapper._get_daily_table(device)
    assert table.shape == (len(days), 13)
    assert table.dtype == torch.int16
    assert table.device == device
    assert torch.equal(table.to(torch.int64), expected[: len(days), 2:15])


def test_timestamp_lookup_shared_and_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build once across concurrent mapper instances and skip empty inputs."""
    monkeypatch.setattr(TimestampTensorMapper, "_daily_tables", {})
    mapper = TimestampTensorMapper()
    array = pa.array([0, None], type=pa.timestamp("ns"))
    with patch.object(
        TimestampTensorMapper, "_build_daily_table", wraps=mapper._build_daily_table
    ) as build:
        assert mapper(pa.array([], type=pa.timestamp("ns"))).shape == (0, 16)
        build.assert_not_called()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(lambda _: TimestampTensorMapper()(array), range(4))
            )
        build.assert_called_once()
    assert all(torch.equal(result, results[0]) for result in results)
    assert len(mapper._daily_tables) == 1
