# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pool.metrics.

Metric objects are process-global singletons. Values are read back through the
metric objects' label children rather than scraping the HTTP endpoint.
"""
from pool import metrics
from pool.dataclasses import (
    DataplaneBackup,
    K8sPlacementState,
    PoolState,
    SnapshotActivity,
    SnapshotLocalCopyRecord,
    SnapshotLocalCopyState,
    SnapshotPlacement,
    SnapshotState,
    WorkerInventory,
    WorkerStateSnapshot,
    WorkerStatus,
)


def _gauge(metric, **labels) -> float:
    return metric.labels(**labels)._value.get()


def _counter(metric, **labels) -> float:
    return metric.labels(**labels)._value.get()


def _worker(uid: str, status: WorkerStatus) -> WorkerInventory:
    return WorkerInventory(
        pool_worker_uid=uid,
        state=WorkerStateSnapshot(ssd_total_bytes=100, ssd_used_bytes=10),
        status=status,
    )


def test_update_state_gauges_counts_every_enum_member():
    state = PoolState(
        placement_state=K8sPlacementState(),
        snapshots={
            'snap-a':
            SnapshotPlacement(worker='w1', state=SnapshotState.READY,
                              leases=['job-1', 'job-2']),
            'snap-b':
            SnapshotPlacement(worker='w1', state=SnapshotState.READY,
                              leases=['job-3']),
            'snap-c':
            SnapshotPlacement(worker='w2', state=SnapshotState.INGESTING),
        },
        dataplane_backups={
            'snap-a': DataplaneBackup(snapshot_id='snap-a',
                                      dataplane_path='/x'),
        },
        local_copies={
            'snap-a:w1':
            SnapshotLocalCopyRecord(snapshot_id='snap-a', pool_worker_uid='w1',
                                    local_state=SnapshotLocalCopyState.WARM),
            'snap-c:w2':
            SnapshotLocalCopyRecord(snapshot_id='snap-c', pool_worker_uid='w2',
                                    local_state=SnapshotLocalCopyState.EVICTED),
        },
    )
    inventory = {
        'w1': _worker('w1', WorkerStatus.READY),
        'w2': _worker('w2', WorkerStatus.DRAINING),
    }

    metrics.update_state_gauges(state, inventory, {})

    assert _gauge(metrics.POOL_SNAPSHOTS, state='READY') == 2
    assert _gauge(metrics.POOL_SNAPSHOTS, state='INGESTING') == 1
    # Absent states report 0, not a missing series.
    assert _gauge(metrics.POOL_SNAPSHOTS, state='RESERVED') == 0
    assert _gauge(metrics.POOL_SNAPSHOTS, state='RESTORING') == 0

    assert _gauge(metrics.POOL_WORKERS, status='READY') == 1
    assert _gauge(metrics.POOL_WORKERS, status='DRAINING') == 1
    assert _gauge(metrics.POOL_WORKERS, status='DEAD') == 0

    assert _gauge(metrics.POOL_LOCAL_COPIES, state='WARM') == 1
    assert _gauge(metrics.POOL_LOCAL_COPIES, state='EVICTED') == 1

    assert metrics.POOL_ACTIVE_LEASES._value.get() == 3
    assert metrics.POOL_DATAPLANE_BACKUPS._value.get() == 1


def test_update_state_gauges_recompute_drops_to_zero():
    state = PoolState(
        placement_state=K8sPlacementState(),
        snapshots={
            'snap-a': SnapshotPlacement(worker='w1', state=SnapshotState.READY),
        },
    )
    metrics.update_state_gauges(state, {}, {})
    assert _gauge(metrics.POOL_SNAPSHOTS, state='READY') == 1

    # Snapshot evicted; gauge must report 0 on the next tick (not stay at 1).
    state.snapshots.clear()
    metrics.update_state_gauges(state, {}, {})
    assert _gauge(metrics.POOL_SNAPSHOTS, state='READY') == 0


def test_inc_sweep_delete_and_error():
    base_local = _counter(metrics.POOL_LIFECYCLE_SWEEP_DELETES, type='local')
    metrics.inc_sweep_delete('local')
    metrics.inc_sweep_delete('local', count=2)
    assert _counter(metrics.POOL_LIFECYCLE_SWEEP_DELETES,
                    type='local') == base_local + 3

    # count=0 is a no-op.
    metrics.inc_sweep_delete('dataplane', count=0)
    assert _counter(metrics.POOL_LIFECYCLE_SWEEP_DELETES, type='dataplane') == 0

    base_err = _counter(metrics.POOL_CONTROLLER_ERRORS, call='place_snapshot')
    metrics.inc_error('place_snapshot')
    assert _counter(metrics.POOL_CONTROLLER_ERRORS,
                    call='place_snapshot') == base_err + 1


def test_observe_placement_records_histogram():
    metrics.observe_placement('WARM', 0.5)
    child = metrics.POOL_PLACEMENT_DURATION.labels(result='WARM')
    assert child._sum.get() == 0.5


def test_update_state_gauges_worker_capacity_and_activity():
    state = PoolState(
        placement_state=K8sPlacementState(),
        snapshots={
            'snap-a':
            SnapshotPlacement(worker='w1', state=SnapshotState.READY,
                              leases=['j1', 'j2'], lease_paths={
                                  'j1': 'durable',
                                  'j2': 'overflow'
                              }),
        },
    )
    inventory = {
        'w1':
        WorkerInventory(
            pool_worker_uid='w1',
            state=WorkerStateSnapshot(ssd_total_bytes=1000, ssd_used_bytes=400),
            status=WorkerStatus.READY, gpu_slots_total=2, gpu_slots_in_use=1),
    }
    # Active jobs are counted from the controller's lease->activity map
    # (one open lease == one running activity placement), not worker heartbeats.
    lease_activity = {
        ('snap-a', 'j1'): SnapshotActivity.GEN_TT,
        ('snap-a', 'j2'): SnapshotActivity.MATERIALIZE_TRAINING_TASK,
    }

    metrics.update_state_gauges(state, inventory, lease_activity)

    assert _gauge(metrics.POOL_WORKER_SSD_TOTAL_BYTES,
                  pool_worker_uid='w1') == 1000
    assert _gauge(metrics.POOL_WORKER_SSD_USED_BYTES,
                  pool_worker_uid='w1') == 400
    assert _gauge(metrics.POOL_WORKER_GPU_SLOTS_TOTAL,
                  pool_worker_uid='w1') == 2
    assert _gauge(metrics.POOL_WORKER_GPU_SLOTS_IN_USE,
                  pool_worker_uid='w1') == 1
    # j1 is durable -> charged to home worker w1; j2 is overflow -> "overflow".
    assert _gauge(metrics.POOL_ACTIVE_ACTIVITIES, pool_worker_uid='w1',
                  activity='GEN_TT') == 1
    assert _gauge(metrics.POOL_ACTIVE_ACTIVITIES, pool_worker_uid='overflow',
                  activity='MATERIALIZE_TRAINING_TASK') == 1
    assert _gauge(metrics.POOL_LEASES_BY_ROUTE, route='durable') == 1
    assert _gauge(metrics.POOL_LEASES_BY_ROUTE, route='overflow') == 1


def test_update_state_gauges_clears_departed_worker_series():
    w = {
        'w1':
        WorkerInventory(
            pool_worker_uid='w1',
            state=WorkerStateSnapshot(ssd_total_bytes=500, ssd_used_bytes=50),
            status=WorkerStatus.READY),
    }
    state = PoolState(placement_state=K8sPlacementState())
    metrics.update_state_gauges(state, w, {})
    assert _gauge(metrics.POOL_WORKER_SSD_TOTAL_BYTES,
                  pool_worker_uid='w1') == 500
    # Worker departs → its per-uid series must be cleared, not frozen at 500.
    metrics.update_state_gauges(state, {}, {})
    samples = {
        s.labels.get('pool_worker_uid')
        for fam in metrics.POOL_WORKER_SSD_TOTAL_BYTES.collect()
        for s in fam.samples
    }
    assert 'w1' not in samples
