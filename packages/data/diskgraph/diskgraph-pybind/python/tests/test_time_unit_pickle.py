# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip pickle tests for the TimeUnit pybind enum.

The kumo path runs DG calls through @diskgraph_subprocess (a SIGKILL-able
single-worker pool), which pickles all kwargs across the process
boundary. PyO3 enums need an explicit __reduce__ to be picklable —
without it, multiprocessing fails with::

    TypeError: cannot pickle 'builtins.TimeUnit' object

This test guards the __reduce__ implementation in lib.rs so the kumo
side can pass TimeUnit values to subprocess-backed activities.
"""
import copy
import pickle

import pytest
from diskgraph import TimeUnit


_VARIANTS = [
    TimeUnit.Seconds,
    TimeUnit.Milliseconds,
    TimeUnit.Microseconds,
    TimeUnit.Nanoseconds,
    TimeUnit.Minutes,
    TimeUnit.Hours,
    TimeUnit.Days,
    TimeUnit.Weeks,
    TimeUnit.Months,
    TimeUnit.Years,
]


@pytest.mark.parametrize("unit", _VARIANTS)
def test_pickle_round_trip_preserves_variant(unit):
    blob = pickle.dumps(unit)
    restored = pickle.loads(blob)
    assert restored == unit
    # Identity holds because unpickle resolves to the class attribute,
    # which is the same singleton object.
    assert restored is unit


@pytest.mark.parametrize("unit", _VARIANTS)
def test_copy_uses_reduce(unit):
    # copy.deepcopy falls back to __reduce__ when no __copy__ is defined.
    assert copy.deepcopy(unit) is unit


def test_pickle_across_protocols():
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(TimeUnit.Nanoseconds, proto))
        assert restored is TimeUnit.Nanoseconds
