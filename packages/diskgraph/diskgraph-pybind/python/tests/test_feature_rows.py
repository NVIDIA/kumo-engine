# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from diskgraph import _gather_byte_rows


def test_two_source_gather_returns_independent_rows():
    bank = np.arange(20, dtype=np.uint16).reshape(5, 4)[1:]
    incoming = np.arange(8, dtype=np.uint16).reshape(2, 4) + 400
    incoming.setflags(write=False)
    indices = np.array([2, -2, 0, -1, -2, 2], dtype=np.int64)
    result = _gather_byte_rows(bank.view(np.uint8), incoming.view(np.uint8), indices)
    rows = np.frombuffer(result, dtype=np.uint16).reshape(-1, 4)
    np.testing.assert_array_equal(
        rows, [bank[2], incoming[1], bank[0], incoming[0], incoming[1], bank[2]]
    )
    rows[0].fill(0)
    np.testing.assert_array_equal(rows[-1], bank[2])
    assert (
        _gather_byte_rows(bank[:0].view(np.uint8), incoming.view(np.uint8), indices[:0])
        == b""
    )


@pytest.mark.parametrize("bad", [4, -3, np.iinfo(np.int64).min, np.iinfo(np.int64).max])
def test_gather_validates_both_source_bounds(bad):
    bank = np.zeros((4, 8), np.uint8)
    with pytest.raises(ValueError, match="out of bounds"):
        _gather_byte_rows(bank, bank[:2], np.array([bad], dtype=np.int64))


def test_gather_validates_width_rank_strides_and_dtype():
    buf = np.zeros((4, 8), np.uint8)
    row = np.zeros(4, dtype=np.int64)
    for matrices in ((buf.ravel(), buf), (buf, buf[:, :4]), (buf[:, ::2], buf[:, ::2])):
        with pytest.raises(ValueError):
            _gather_byte_rows(*matrices, row)
    with pytest.raises(ValueError, match="contiguous"):
        _gather_byte_rows(buf, buf, row[::2])
    with pytest.raises(ValueError, match="nonzero"):
        _gather_byte_rows(buf[:, :0], buf[:, :0], row)
    with pytest.raises((BufferError, TypeError)):
        _gather_byte_rows(buf, buf, row.astype(np.float64))
    with pytest.raises((BufferError, TypeError)):
        _gather_byte_rows(buf.view(np.uint16), buf.view(np.uint16), row)
