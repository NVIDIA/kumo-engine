# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``job_id`` extension on ``DiskgraphHandle``.

TT-NG only — the ``job_id`` field is unused on the legacy path.

Covers:
  * default (None) so existing call sites are unaffected
  * round-trip through JSON (Temporal activity contract)
  * frozen + hashable invariants extended to the new field
  * ``for_test`` / ``from_index_path`` classmethods unchanged
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from io_utils.json import from_json_dict, to_json_dict
from kumo.diskgraph._backend import DiskgraphHandle


def test_handle_default_job_id_is_none() -> None:
    h = DiskgraphHandle(graph_key='gk', graph_data_version='v0')
    assert h.job_id is None


def test_handle_with_job_id_is_set() -> None:
    h = DiskgraphHandle(
        graph_key='gk', graph_data_version='v0', job_id='job-123')
    assert h.job_id == 'job-123'


def test_handle_with_job_id_round_trips_through_json() -> None:
    h = DiskgraphHandle(
        graph_key='gk', graph_data_version='v0', job_id='job-123')
    payload = to_json_dict(h)
    restored = from_json_dict(payload, DiskgraphHandle)
    assert restored == h
    assert restored.job_id == 'job-123'


def test_handle_without_job_id_round_trips_through_json() -> None:
    h = DiskgraphHandle(graph_key='gk', graph_data_version='v0')
    payload = to_json_dict(h)
    restored = from_json_dict(payload, DiskgraphHandle)
    assert restored == h
    assert restored.job_id is None


def test_handle_is_frozen() -> None:
    h = DiskgraphHandle(
        graph_key='gk', graph_data_version='v0', job_id='job-123')
    with pytest.raises(FrozenInstanceError):
        h.job_id = 'different'  # type: ignore[misc]


def test_handle_with_job_id_is_hashable() -> None:
    h1 = DiskgraphHandle(
        graph_key='gk', graph_data_version='v0', job_id='job-123')
    h2 = DiskgraphHandle(
        graph_key='gk', graph_data_version='v0', job_id='job-123')
    h3 = DiskgraphHandle(
        graph_key='gk', graph_data_version='v0', job_id='job-456')
    h4 = DiskgraphHandle(graph_key='gk', graph_data_version='v0')
    # equal across construction
    assert h1 == h2
    # job_id contributes to identity
    assert h1 != h3
    assert h1 != h4
    # set semantics use hash + eq correctly
    assert {h1, h2, h3, h4} == {h1, h3, h4}


def test_for_test_classmethod_unchanged() -> None:
    h = DiskgraphHandle.for_test()
    assert h.graph_key == 'NA-test'
    assert h.graph_data_version == 'v0'
    assert h.job_id is None


def test_from_index_path_classmethod_unchanged() -> None:
    h = DiskgraphHandle.from_index_path('/mnt/localssd/dg-indexes/gk/v5')
    assert h.graph_key == 'gk'
    assert h.graph_data_version == 'v5'
    assert h.job_id is None
