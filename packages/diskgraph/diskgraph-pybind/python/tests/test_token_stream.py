# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native flat packing: ragged parts, sample isolation, truncation and errors."""

import pyarrow as pa
import pytest
from diskgraph import DiskgraphConfigError, pack_token_stream


def column(values):
    return pa.array(values, type=pa.int64())


@pytest.mark.parametrize(
    "last,expected", [(False, [90, 10, 20, 21, 99, 91]), (True, [90, 11, 22, 91])]
)
def test_ragged_parts_and_event_boundary(last, expected):
    offsets, ids = pack_token_stream(
        column([0, 2, 3]),
        [
            (column([0, 1, 2]), column([10, 11, 12])),
            (column([0, 0, 1, 2]), column([20, 21, 22, 23])),
        ],
        6,
        last,
        True,
        99,
        True,
        90,
        91,
    )
    assert offsets.to_pylist() == [0, len(expected), len(expected) + 4]
    assert ids.to_pylist() == expected + [90, 12, 23, 91]


@pytest.mark.parametrize(
    "offsets,events,ids",
    [
        ([1, 2], [0], [10]),
        ([0, 0], [], []),
        ([0, 2, 1], [0], [10]),
        ([0, 1], [-1], [10]),
        ([0, 1], [1], [10]),
        ([0, 1], [0], []),
        ([0, 1], [0], [None]),
        ([0, None], [0], [10]),
    ],
)
def test_invalid_buffers_are_rejected(offsets, events, ids):
    with pytest.raises(DiskgraphConfigError):
        pack_token_stream(
            column(offsets),
            [(column(events), column(ids))],
            8,
            False,
            False,
            None,
            True,
            None,
            None,
        )


def test_boundary_tokens_count_toward_budget():
    with pytest.raises(DiskgraphConfigError):
        pack_token_stream(
            column([0, 1]), [(column([0]), column([10]))], 2, False, False, None, True, 90, 91
        )


@pytest.mark.parametrize("last,expected", [(False, [1, 2, 3]), (True, [3, 4, 5])])
def test_event_boundary_cuts_inside_event_when_none_fits(last, expected):
    offsets, ids = pack_token_stream(
        column([0, 1]),
        [(column([0] * 5), column([1, 2, 3, 4, 5]))],
        3,
        last,
        True,
        None,
        True,
        None,
        None,
    )
    assert offsets.to_pylist() == [0, 3]
    assert ids.to_pylist() == expected


def test_declared_event_count_is_bounded_before_allocation():
    with pytest.raises(DiskgraphConfigError, match="declared event count"):
        pack_token_stream(
            column([0, 1_000_000_000]),
            [(column([]), column([]))],
            1,
            False,
            False,
            None,
            True,
            None,
            None,
        )
