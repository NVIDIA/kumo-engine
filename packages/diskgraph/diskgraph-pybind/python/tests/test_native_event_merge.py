# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native stable event merging selects rows, without another payload materialization."""

import itertools

import diskgraph
import pyarrow as pa
import pytest


@pytest.mark.parametrize("ascending", list(itertools.product([False, True], repeat=2)))
@pytest.mark.parametrize(
    "keep_last,descending", list(itertools.product([False, True], repeat=2))
)
@pytest.mark.parametrize("maximum", [None, 1, 4, 100])
def test_merge_matches_stable_component_order_at_ties(
    ascending, keep_last, descending, maximum
):
    originals = [[[-9, 3, 3, 7], [], [-(2**63), 3]], [[-9, 3, 5], [2**63 - 1], []]]
    parts = []
    expected = [[] for _ in range(3)]
    base = 0
    for component, (events, is_ascending) in enumerate(
        zip(originals, ascending, strict=True)
    ):
        offsets, times = [0], []
        for anchor, rows in enumerate(events):
            selected = rows if is_ascending else rows[::-1]
            for position, time in enumerate(selected):
                ordinal = position if is_ascending else len(selected) - 1 - position
                expected[anchor].append(((time, component, ordinal), base + len(times)))
                times.append(time)
            offsets.append(len(times))
        parts.append(
            (
                pa.array(offsets, type=pa.uint64()),
                pa.array(times, type=pa.int64()),
                is_ascending,
            )
        )
        base += len(times)
    take, bounds = [], [0]
    for rows in expected:
        rows.sort()
        if maximum is not None:
            rows = rows[-maximum:] if keep_last else rows[:maximum]
        if descending:
            rows.reverse()
        take.extend(row for _, row in rows)
        bounds.append(len(take))
    actual_bounds, actual_take = diskgraph.merge_event_streams(
        parts, maximum, keep_last, descending
    )
    assert actual_bounds.to_pylist() == bounds
    assert actual_take.to_pylist() == take


@pytest.mark.parametrize(
    "offsets,times",
    [([0, 2], [1]), ([0, -1, 2], [1, 2]), ([0, 1], [None]), ([0, 2], [2, 1])],
)
def test_merge_refuses_invalid_clock_or_offset_contract(offsets, times):
    with pytest.raises(diskgraph.DiskgraphConfigError):
        diskgraph.merge_event_streams(
            [
                (
                    pa.array(offsets, type=pa.int64()),
                    pa.array(times, type=pa.int64()),
                    True,
                )
            ],
            None,
            True,
            False,
        )
