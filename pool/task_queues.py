# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-queue naming for the diskgraph pool. Queue names are derived from
worker identity — both the controller (returning ``task_queue`` from
``place_snapshot``) and the K8s pod spawner use this module so worker
subscriptions match.
"""
from enum import Enum

from pool.storage_mode import PoolStorageMode

DAEMON_TQ_PREFIX = 'diskgraph-daemon-'
ACTIVITY_TQ_PREFIX = 'diskgraph-activity-tq-'

# Daemon and any activity-scoped pod sharing its PVC carry the same
# ``pool-worker-`` UID; ephemeral-node workers carry ``pool-eh-worker-``.
POOL_WORKER_UID_PREFIX_DURABLE = 'pool-worker-'
POOL_WORKER_UID_PREFIX_EPHNODE = 'pool-eh-worker-'

# Singlehost: one worker process polling a single queue.
# ``SinglehostPlacementStrategy`` collapses all activity classes
# (daemon ingest / gen_tt / materialize / train / predict) onto this
# queue; per-class isolation is enforced in-process (per-snapshot
# asyncio.Lock + GPU semaphore + max_concurrent_activities=1 for
# activity-scoped workers). See the spec's per-deployment table.
SINGLEHOST_DAEMON_TQ = 'diskgraph-daemon-tq'
POOL_WORKER_NAME = 'pool-worker'
# Pod identity distinguishes the selected poller from stale queue peers.
ACTIVITY_WORKER_POD_NAME_ENV = 'POOL_ACTIVITY_WORKER_POD_NAME'


class QueueKind(str, Enum):
    DAEMON = 'daemon'
    ACTIVITY = 'activity'


def daemon_dispatch_queue(pool_worker_uid: str) -> str:
    return f'{DAEMON_TQ_PREFIX}{pool_worker_uid}'


def activity_queue(job_id: str) -> str:
    return f'{ACTIVITY_TQ_PREFIX}{job_id}'


def activity_worker_name_suffix(pod_name: str) -> str:
    """WorkerEnv suffix used by one activity-scoped Kubernetes pod."""
    return f'-{pod_name}' if pod_name else ''


def activity_worker_identity(pod_name: str, instance_id: str) -> str:
    """Exact Temporal poller identity expected from ``pod_name``."""
    return (f'{POOL_WORKER_NAME}{activity_worker_name_suffix(pod_name)}-'
            f'{instance_id}')


def is_activity_scoped_pod(pool_worker_uid: str,
                           storage_mode: PoolStorageMode) -> bool:
    """True if this pod must delegate ``ensure_snapshot`` writes to a
    long-lived daemon (writer-side snapshot mutation invariant)."""
    return (storage_mode == PoolStorageMode.ACTIVITY_ON_PVC
            and pool_worker_uid.startswith(POOL_WORKER_UID_PREFIX_DURABLE))
