# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

from rest.public.routers.data_snapshot import GraphSnapshotRouter
from rest.utils.routers import StrictQueryAPIRoute


def test_graph_snapshot_uses_strict_query_route():
    auth = mock.Mock()
    router = GraphSnapshotRouter(lambda: None, mock.AsyncMock(), auth)
    create_route = next(
        route for route in router.router.routes
        if route.path == '/graphsnapshots' and 'POST' in route.methods)
    assert isinstance(create_route, StrictQueryAPIRoute)
