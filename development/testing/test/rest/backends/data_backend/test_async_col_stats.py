# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from rest.backends.data_backend.data_backend import (
    _compute_connector_col_stats,
)


@pytest.mark.asyncio
async def test_compute_connector_col_stats_runs_off_event_loop():
    caller_thread = threading.get_ident()

    class Connector:
        def compute_col_stats(self, table_name, stats_dict, *, approx,
                              data_version):
            assert threading.get_ident() != caller_thread
            assert table_name == 'orders'
            assert stats_dict == {}
            assert approx is True
            assert data_version == '1'
            return {}

    assert await _compute_connector_col_stats.coroutine(Connector(), 'orders',
                                                        {}, '1') == {}
