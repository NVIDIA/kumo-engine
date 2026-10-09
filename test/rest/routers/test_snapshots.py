# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from kumo_core.types import ColStatType

from kumo.stats.stats import Stats
from rest.routers.snapshots import SnapshotResource


@pytest.fixture(scope="session", autouse=True)
def setup_rest_test_session():
    yield


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.mark.asyncio
async def test_snapshot_col_stats_preserve_approx_metadata():
    auth = SimpleNamespace(authenticate_request_header=lambda: None, )
    resource = SnapshotResource(auth, MagicMock(), MagicMock())
    resource.get_table_snapshot = AsyncMock(return_value=SimpleNamespace(
        snapshot=SimpleNamespace(column_stats=[
            SimpleNamespace(
                column_name="status",
                stats={
                    ColStatType.COUNT.value:
                    Stats(value=10000, approx=True, data_version="v1",
                          num_entries=10000).config(),
                    ColStatType.NUM_NA.value:
                    Stats(value=0, approx=True, data_version="v1").config(),
                    ColStatType.NUM_UNIQUE.value:
                    Stats(value=2, approx=True, data_version="v1").config(),
                    ColStatType.CATEGORY_COUNTS.value:
                    Stats(value=[["active", 9000], ["inactive", 1000]],
                          approx=True, data_version="v1").config(),
                    ColStatType.INVALID_FRACTION.value:
                    Stats(value=0.0, approx=False, data_version="v1").config(),
                },
            )
        ]),
        definition=SimpleNamespace(cols=[
            SimpleNamespace(
                name="status",
                dtype="string",
                stype="categorical",
            )
        ]),
    ))

    response = await resource.get_col_stats("snapshot-id",
                                            "project_" + "1" * 32)

    assert response.statistics_available is True
    assert response.approx is True
    assert response.rows_used == 10000
    assert response.columns[0].statistics is not None
    assert response.columns[0].statistics.approx is True


@pytest.mark.asyncio
async def test_snapshot_col_stats_rehydrate_serialized_timestamp_stats():
    auth = SimpleNamespace(authenticate_request_header=lambda: None, )
    resource = SnapshotResource(auth, MagicMock(), MagicMock())
    resource.get_table_snapshot = AsyncMock(return_value=SimpleNamespace(
        snapshot=SimpleNamespace(column_stats=[
            SimpleNamespace(
                column_name="created_at",
                stats={
                    ColStatType.COUNT.value: {
                        "value": 4500,
                        "approx": False,
                        "data_version": "v1",
                        "num_entries": 4500,
                        "timestamp": None,
                    },
                    ColStatType.NUM_NA.value: {
                        "value": 0,
                        "approx": False,
                        "data_version": "v1",
                        "num_entries": 4500,
                        "timestamp": None,
                    },
                    ColStatType.NUM_UNIQUE.value: {
                        "value": 1535,
                        "approx": False,
                        "data_version": "v1",
                        "num_entries": 4500,
                        "timestamp": None,
                    },
                    ColStatType.INVALID_FRACTION.value: {
                        "value": 0.0,
                        "approx": False,
                        "data_version": "v1",
                        "num_entries": 4500,
                        "timestamp": None,
                    },
                    ColStatType.MIN.value: {
                        "value": "kumo_timestamp_1993-01-01 00:00:00",
                        "approx": False,
                        "data_version": "v1",
                        "num_entries": 4500,
                        "timestamp": None,
                    },
                    ColStatType.MAX.value: {
                        "value": "kumo_timestamp_1997-12-29 00:00:00",
                        "approx": False,
                        "data_version": "v1",
                        "num_entries": 4500,
                        "timestamp": None,
                    },
                },
            )
        ]),
        definition=SimpleNamespace(cols=[
            SimpleNamespace(
                name="created_at",
                dtype="date",
                stype="timestamp",
            )
        ]),
    ))

    response = await resource.get_col_stats("snapshot-id",
                                            "project_" + "1" * 32)

    assert response.columns[0].statistics is not None
    assert response.columns[0].statistics.summary_stats == [
        ("Range", "1,823 days"),
        ("Min", "1993-01-01"),
        ("Max", "1997-12-29"),
    ]
