# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side helpers for reaching the ``PoolController`` via AutoRPC."""
from typing import Protocol

from pool import timeouts
from pool.dataclasses import (
    PlacementRequest,
    PlacementResult,
    PoolStats,
    PoolStatsFilter,
    WorkerStatusReport,
)
from pool.service import PoolController
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons.auto_rpc import create_client
from temporalio_commons.auto_rpc_config import AutoRpcClientConfig


class PoolControllerClient(Protocol):
    async def report_status(self, report: WorkerStatusReport) -> None:
        ...

    async def place_snapshot(
            self, request: PlacementRequest) -> PlacementResult:
        ...

    async def release_lease(self, snapshot_id: str, job_id: str) -> None:
        ...

    async def get_pool_stats(
            self, filter: PoolStatsFilter | None = None) -> PoolStats:
        ...


def get_controller_client(config: DeployEnvConfig) -> PoolControllerClient:
    client_config = AutoRpcClientConfig.parse_from_env(PoolController, config)
    # Apply our per-method defaults (see :mod:`pool.timeouts`) where the
    # env didn't already set one. ``setdefault`` ensures
    # ``RPC_TIMEOUT_POOLCONTROLLER__<METHOD>`` env overrides win.
    for method, default in timeouts.DEFAULT_METHOD_TIMEOUTS.items():
        client_config.method_timeout_config.setdefault(method, default)
    return create_client(PoolController, client_config)
