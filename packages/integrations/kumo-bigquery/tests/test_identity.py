# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stamping a BigQuery object from its catalog entry, and what has no stamp."""

from datetime import datetime, timezone

import pytest
from fakes import MODIFIED, FakeBigQueryClient, FakeTable
from kumo_bigquery.identity import data_version, hour_version

FQN = "bigquery-public-data.usa_names.usa_1910_2013"
MODIFIED_VERSION = "1457746213"


def stamp(**overrides):
    return data_version(FakeBigQueryClient({FQN: FakeTable(**overrides)}), FQN)


def test_a_physical_table_is_dated_by_the_time_it_was_modified():
    """A version carrying `.0` is a different path segment and a different key."""
    assert stamp() == MODIFIED_VERSION
    assert stamp().kind == "timestamp"
    assert stamp(modified=MODIFIED.replace(
        microsecond=900000)) == (MODIFIED_VERSION)
    assert stamp(
        modified=datetime(2026, 1, 1, tzinfo=timezone.utc)) != MODIFIED_VERSION


def test_a_logical_view_falls_back_to_the_hour_window():
    assert stamp(table_type="VIEW") == hour_version()
    assert stamp(table_type="VIEW").kind == "hour_bucket"


@pytest.mark.parametrize("table_type",
                         ["MATERIALIZED_VIEW", "EXTERNAL", "SNAPSHOT"])
def test_only_the_exact_view_type_falls_back(table_type):
    """A broader gate would rebuild these hourly and hide their changes."""
    assert stamp(table_type=table_type) == MODIFIED_VERSION


def test_a_table_without_a_modification_time_is_reported_undated():
    assert stamp(modified=None) == "-1"
    assert stamp(modified=None).kind == "unknown"


def test_the_stamp_reads_the_catalog_and_scans_nothing():
    recorder = FakeBigQueryClient({FQN: FakeTable()})
    data_version(recorder, FQN)
    assert recorder.statements == []
