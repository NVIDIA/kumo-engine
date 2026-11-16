# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal dispatch seam for daemon-owned snapshot operations.

``ensure_snapshot`` dispatch is workflow-owned. The controller uses this
module only for local L1 eviction: schedule the existing
``SnapshotActivities.delete_snapshot`` activity on the target daemon queue.
"""
from datetime import timedelta
from typing import Awaitable, Callable, Optional

from temporalio.common import WorkflowIDConflictPolicy

from pool.task_queues import SINGLEHOST_DAEMON_TQ, daemon_dispatch_queue
from service_framework.deploy_env_config import DeployEnvConfig
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_registry import RunSingleActivityWorkflow

ENSURE_SNAPSHOT_ACTIVITY = 'ensure_snapshot'
DELETE_SNAPSHOT_ACTIVITY = 'delete_snapshot'
DELETE_SNAPSHOT_WORKFLOW_ID_PREFIX = 'diskgraph-delete-snapshot'
DELETE_SNAPSHOT_EXECUTION_TIMEOUT = timedelta(minutes=15)


def delete_snapshot_workflow_id(snapshot_id: str,
                                pool_worker_uid: str) -> str:
    return (
        f'{DELETE_SNAPSHOT_WORKFLOW_ID_PREFIX}-'
        f'{snapshot_id}-{pool_worker_uid}')


def _task_queue_for_config(config: DeployEnvConfig) -> Callable[[str], str]:
    diskgraph = getattr(config, 'data_compute', None)
    if getattr(diskgraph, 'pool_strategy', None) == 'singlehost':
        # The statically provisioned singlehost worker still has a
        # deployment-scoped Temporal queue.  This must agree with the
        # controller placement strategy and devserver worker command.
        return lambda _pool_worker_uid: config.task_queue(SINGLEHOST_DAEMON_TQ)
    return daemon_dispatch_queue


def build_temporal_snapshot_deleter(
    config: DeployEnvConfig,
    temporal_client: Optional[TemporalClient] = None,
    task_queue_for_worker: Optional[Callable[[str], str]] = None,
) -> Callable[[str, str], Awaitable[bool]]:
    temporal = temporal_client or TemporalClient(config)
    queue_for_worker = task_queue_for_worker or _task_queue_for_config(config)

    async def delete(snapshot_id: str, pool_worker_uid: str) -> bool:
        client = await temporal.get()
        await client.execute_workflow(
            RunSingleActivityWorkflow.run,
            args=[DELETE_SNAPSHOT_ACTIVITY, [snapshot_id]],
            id=delete_snapshot_workflow_id(snapshot_id, pool_worker_uid),
            task_queue=queue_for_worker(pool_worker_uid),
            execution_timeout=DELETE_SNAPSHOT_EXECUTION_TIMEOUT,
            id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
        )
        # ``SnapshotActivities.delete_snapshot`` is intentionally idempotent
        # and returns no payload; successful completion is enough for the
        # controller to mark the local copy evicted.
        return True

    return delete
