# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``pool.strategy_singlehost.SinglehostPlacementStrategy``."""
import asyncio

import pytest  # pants: no-infer-dep

from pool.dataclasses import (
    PlacementRequest,
    PoolState,
    ResourceHint,
    SinglehostPlacementState,
    SnapshotActivity,
)
from pool.strategy import PlacementStrategy
from pool.strategy_singlehost import (
    SINGLEHOST_WORKER_UID,
    SinglehostPlacementStrategy,
)
from pool.task_queues import SINGLEHOST_DAEMON_TQ


def _request(activity: SnapshotActivity) -> PlacementRequest:
    return PlacementRequest(
        snapshot_id='s1',
        job_id='j1',
        workflow_id='j1',
        activity=activity,
        resource_hint=ResourceHint(),
    )


def test_satisfies_protocol():
    assert isinstance(SinglehostPlacementStrategy(), PlacementStrategy)


def test_route_returns_singlehost_worker():
    strategy = SinglehostPlacementStrategy()
    req = _request(SnapshotActivity.GEN_TT)
    decision = asyncio.run(strategy.route(req, warm_location_uid=None))
    assert decision.pool_worker_uid == SINGLEHOST_WORKER_UID


@pytest.mark.parametrize('activity', list(SnapshotActivity))
def test_provision_uses_static_singlehost_queues(activity):
    strategy = SinglehostPlacementStrategy()
    req = _request(activity)

    async def run():
        decision = await strategy.route(req, None)
        return await strategy.provision(req, decision)

    handle = asyncio.run(run())
    assert handle.task_queue == SINGLEHOST_DAEMON_TQ
    assert handle.pool_worker_uid == SINGLEHOST_WORKER_UID


def test_provision_uses_injected_deployment_scoped_queue():
    strategy = SinglehostPlacementStrategy(
        task_queue='diskgraph-daemon-tq-devserver-train-a')
    req = _request(SnapshotActivity.GEN_TT)

    async def run():
        decision = await strategy.route(req, None)
        return await strategy.provision(req, decision)

    handle = asyncio.run(run())
    assert handle.task_queue == 'diskgraph-daemon-tq-devserver-train-a'


def test_release_is_noop():
    """Static workers; nothing for the strategy to clean up."""
    strategy = SinglehostPlacementStrategy()
    req = _request(SnapshotActivity.GEN_TT)

    async def run():
        decision = await strategy.route(req, None)
        handle = await strategy.provision(req, decision)
        await strategy.release(req, handle)

    asyncio.run(run())  # should not raise


def test_reconcile_returns_singlehost_worker():
    strategy = SinglehostPlacementStrategy()
    state = PoolState(placement_state=SinglehostPlacementState())
    delta = asyncio.run(strategy.reconcile(state))
    assert delta.live_pool_worker_uids == {SINGLEHOST_WORKER_UID}
    assert delta.retired_uids == set()
    assert delta.unexpected_uids == set()
