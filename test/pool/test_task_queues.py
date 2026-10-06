# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``pool.task_queues`` naming helpers."""
import pytest  # pants: no-infer-dep

from pool.storage_mode import PoolStorageMode
from pool.task_queues import (
    ACTIVITY_TQ_PREFIX,
    DAEMON_TQ_PREFIX,
    POOL_WORKER_UID_PREFIX_DURABLE,
    POOL_WORKER_UID_PREFIX_EPHNODE,
    SINGLEHOST_DAEMON_TQ,
    QueueKind,
    activity_queue,
    daemon_dispatch_queue,
    is_activity_scoped_pod,
)


def test_queue_kind_values():
    assert QueueKind.DAEMON.value == 'daemon'
    assert QueueKind.ACTIVITY.value == 'activity'


def test_daemon_dispatch_queue_format():
    assert (daemon_dispatch_queue('worker-abc123') ==
            'diskgraph-daemon-worker-abc123')
    assert daemon_dispatch_queue('worker-abc123').startswith(DAEMON_TQ_PREFIX)


def test_activity_queue_uses_job_id():
    assert (activity_queue('job-xyz789') == 'diskgraph-activity-tq-job-xyz789')
    assert activity_queue('job-xyz789').startswith(ACTIVITY_TQ_PREFIX)


def test_singlehost_daemon_queue_is_fixed_name():
    assert SINGLEHOST_DAEMON_TQ == 'diskgraph-daemon-tq'


def test_pool_worker_uid_prefixes_are_distinct():
    # Guard against one prefix becoming a substring of the other —
    # would silently misclassify ephemeral-node pods as activity-pods.
    assert not POOL_WORKER_UID_PREFIX_DURABLE.startswith(
        POOL_WORKER_UID_PREFIX_EPHNODE)
    assert not POOL_WORKER_UID_PREFIX_EPHNODE.startswith(
        POOL_WORKER_UID_PREFIX_DURABLE)


@pytest.mark.parametrize(
    'uid, storage_mode, expected',
    [
        ('pool-worker-nodeA', PoolStorageMode.DAEMON_ON_PVC, False),
        ('singlehost', PoolStorageMode.DAEMON_ON_PVC, False),
        # Activity-on-PVC + durable-unit prefix == activity-pod sharing daemon PVC.
        ('pool-worker-nodeA', PoolStorageMode.ACTIVITY_ON_PVC, True),
        # Activity-on-emptyDir == private storage.
        ('pool-eh-worker-instance-1', PoolStorageMode.ACTIVITY_ON_EMPTYDIR,
         False),
        # Legacy UIDs pre-dating the convention.
        ('singlehost', PoolStorageMode.ACTIVITY_ON_PVC, False),
        ('testing-worker', PoolStorageMode.ACTIVITY_ON_PVC, False),
    ])
def test_is_activity_scoped_pod(uid, storage_mode, expected):
    assert is_activity_scoped_pod(uid, storage_mode) is expected
