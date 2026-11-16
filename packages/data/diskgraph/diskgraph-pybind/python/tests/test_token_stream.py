# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native flat packing: ragged parts, sample isolation, truncation and errors."""

import bisect
import random

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



def test_declared_events_without_tokens_allocate_nothing():
    offsets, ids = pack_token_stream(
        column([0, 1_000_000_000]),
        [],
        1_000_000_000,
        False,
        False,
        None,
        True,
        None,
        None,
    )
    assert offsets.to_pylist() == [0, 0]
    assert ids.to_pylist() == []


@pytest.mark.parametrize(
    "separator,expected,boundaries",
    [(None, [7, 8], [0, 1, 2]), (99, [7, 99, 99, 99, 99] + [99] * 5 + [8], [0, 5, 11])],
)
def test_sparse_events_across_huge_samples(separator, expected, boundaries):
    offsets, ids = pack_token_stream(
        column([0, 5, 1_000_000_000]),
        [(column([999_999_999, 0]), column([8, 7]))],
        6,
        True,
        False,
        separator,
        True,
        None,
        None,
    )
    assert offsets.to_pylist() == boundaries
    assert ids.to_pylist() == expected


def test_sparse_event_fits_minimal_budget():
    offsets, ids = pack_token_stream(
        column([0, 5]), [(column([0]), column([7]))], 1, False, False, None, True, None, None
    )
    assert offsets.to_pylist() == [0, 1]
    assert ids.to_pylist() == [7]


def reference_pack(
    offsets, parts, maximum, keep_last, event_boundary, separator, between, bos, eos
):
    """Literal port of the dense algorithm the native packer must reproduce."""
    reserved = (bos is not None) + (eos is not None)
    if (
        maximum <= reserved
        or len(offsets) < 2
        or offsets[0] != 0
        or any(a >= b for a, b in zip(offsets, offsets[1:]))
    ):
        raise ValueError
    n = offsets[-1]
    lengths = [int(separator is not None)] * n
    if separator is not None and between:
        for end in offsets[1:]:
            lengths[end - 1] = 0
    for events, ids in parts:
        if len(events) != len(ids) or None in events or None in ids:
            raise ValueError
        for event in events:
            if event < 0 or event >= n:
                raise ValueError
            lengths[event] += 1
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    cursor = starts[:n]
    tokens = [0] * starts[n]
    for events, ids in parts:
        for event, token in zip(events, ids):
            tokens[cursor[event]] = token
            cursor[event] += 1
    if separator is not None:
        for event in range(n):
            if cursor[event] < starts[event + 1]:
                tokens[cursor[event]] = separator
    budget = maximum - reserved
    result, boundaries = [], [0]
    for begin, end in zip(offsets, offsets[1:]):
        window = starts[begin : end + 1]
        lo, hi = window[0], window[-1]
        if hi - lo > budget:
            if keep_last:
                limit = hi - budget
                if event_boundary:
                    candidate = window[bisect.bisect_left(window, limit)]
                    lo = limit if candidate == hi else candidate
                else:
                    lo = limit
            else:
                limit = lo + budget
                if event_boundary:
                    candidate = window[bisect.bisect_right(window, limit) - 1]
                    hi = limit if candidate == lo else candidate
                else:
                    hi = limit
        result.extend([bos] if bos is not None else [])
        result.extend(tokens[lo:hi])
        result.extend([eos] if eos is not None else [])
        boundaries.append(len(result))
    return boundaries, result


def random_case(rng):
    sizes = [rng.choice([1, 1, 2, 3, 6]) for _ in range(rng.randint(1, 4))]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    n = offsets[-1]
    empty = {event for event in range(n) if rng.random() < 0.4}
    parts = []
    for _ in range(rng.randint(0, 3)):
        events = [
            event
            for event in range(n)
            if event not in empty
            for _ in range(rng.choice([0, 1, 1, 2, 3]))
        ]
        rng.shuffle(events)
        parts.append((events, [rng.randint(0, 50) for _ in events]))
    separator = rng.choice([None, 99])
    bos, eos = rng.choice([None, 90]), rng.choice([None, 91])
    reserved = (bos is not None) + (eos is not None)
    total = sum(len(events) for events, _ in parts) + (n if separator else 0)
    maximum = rng.randint(reserved + 1, reserved + total + 3)
    if rng.random() < 0.15:
        corrupt = rng.randrange(6)
        if corrupt == 0:
            offsets[0] = 1
        elif corrupt == 1 and len(offsets) > 2:
            offsets[1] = offsets[2]
        elif corrupt == 2:
            maximum = reserved
        elif corrupt == 3:
            parts.append(([rng.choice([-1, n])], [1]))
        elif corrupt == 4:
            parts.append(([0, 0], [1]))
        else:
            parts.append(([0], [None]))
    return (
        offsets, parts, maximum, rng.random() < 0.5, rng.random() < 0.5,
        separator, rng.random() < 0.5, bos, eos,
    )


def test_sparse_packer_matches_dense_reference():
    rng = random.Random(82)
    for _ in range(4000):
        case = random_case(rng)
        offsets, parts, *rest = case
        native_args = (column(offsets), [(column(e), column(i)) for e, i in parts], *rest)
        try:
            expected = reference_pack(*case)
        except ValueError:
            with pytest.raises(DiskgraphConfigError):
                pack_token_stream(*native_args)
            continue
        got_offsets, got_ids = pack_token_stream(*native_args)
        assert (got_offsets.to_pylist(), got_ids.to_pylist()) == expected, case
