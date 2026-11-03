# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reusable native lookup must match Arrow's first-match/null semantics."""

from concurrent.futures import ThreadPoolExecutor

import diskgraph
import pyarrow as pa
import pyarrow.compute as pc
import pytest


@pytest.mark.parametrize("keys", [[], ["b", "a", "café"], [None, "a"], ["a", "a", None, None]])
@pytest.mark.parametrize("values", [[], [None, "a", "?", "café", "b"], ["b", "b", None]])
def test_reused_lookup_matches_arrow(keys, values):
    keys = pa.array(keys, type=pa.string())
    # A nonzero Arrow offset must not change positions or validity.
    values = pa.array(["prefix", *values, "suffix"], type=pa.string()).slice(1, len(values))
    lookup = diskgraph.StringLookup(keys)
    expected = pc.index_in(values, value_set=keys)
    for _ in range(3):
        assert lookup.lookup(values).equals(expected)


def test_lookup_is_safe_to_share_between_workers():
    keys = pa.array([f"value{i}" for i in range(100)])
    values = pa.array(["value99", "missing", None] * 100)
    lookup = diskgraph.StringLookup(keys)
    expected = pc.index_in(values, value_set=keys)
    with ThreadPoolExecutor(2) as pool:
        assert all(result.equals(expected) for result in pool.map(lookup.lookup, [values] * 8))


def test_lookup_rejects_non_string_arrays():
    with pytest.raises(diskgraph.DiskgraphConfigError, match="string"):
        diskgraph.StringLookup(pa.array([1, 2]))
    with pytest.raises(diskgraph.DiskgraphConfigError, match="string"):
        diskgraph.StringLookup(pa.array(["a"])).lookup(pa.array([1]))
