# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for ``pool.dataclasses``.

We verify that every persisted dataclass survives a ``save_json`` /
``load_json`` round-trip, catching serializer regressions before they cause
a silent data-loss incident on controller restart.
"""
from dataclasses import field
from datetime import datetime, timezone
from typing import Dict, Literal

import pytest  # pants: no-infer-dep

from io_utils.dataclasses import kumo_dataclass
from io_utils.json import from_json, load_json, save_json, to_json
from pool.dataclasses import (
    ActiveJob,
    DataplaneBackup,
    K8sPlacementState,
    PlacementRequest,
    PoolState,
    PoolStats,
    PoolStatsFilter,
    PoolWorkerStats,
    ProvisionedHandle,
    PvcState,
    ResourceHint,
    RetentionMode,
    SinglehostPlacementState,
    SnapshotActivity,
    SnapshotComplianceRecord,
    SnapshotLocalCopyRecord,
    SnapshotLocalCopyState,
    SnapshotPlacement,
    SnapshotState,
    WorkerEvent,
    WorkerEventType,
    WorkerInventory,
    WorkerStateSnapshot,
    WorkerStatus,
    WorkerStatusReport,
)
from pool.storage_mode import PoolStorageMode
from pool.strategy import ReconcileDelta, RouteDecision


@kumo_dataclass
class _RollbackK8sPlacementState:
    """Schema used by the pre-PR binary during a rollback."""
    type: Literal['k8s'] = 'k8s'
    pvcs: Dict[str, PvcState] = field(default_factory=dict)


@pytest.fixture
def populated_state() -> PoolState:
    """A non-trivial state covering every nested type."""
    placement1 = SnapshotPlacement(
        worker='worker-a',
        state=SnapshotState.READY,
        size_bytes=1_024_000,
        leases=['job-1'],
    )
    placement2 = SnapshotPlacement(
        worker='worker-b',
        state=SnapshotState.RESERVED,
        size_bytes=0,
    )
    return PoolState(
        placement_state=K8sPlacementState(
            pvcs={
                'worker-a':
                PvcState(pool_worker_uid='worker-a', created_at=datetime(
                    2026, 4, 20, tzinfo=timezone.utc)),
                'worker-b':
                PvcState(pool_worker_uid='worker-b', created_at=datetime(
                    2026, 4, 21, tzinfo=timezone.utc)),
            }),
        snapshots={
            'snap-1': placement1,
            'snap-2': placement2,
        },
        dataplane_backups={
            'snap-1':
            DataplaneBackup(snapshot_id='snap-1',
                            dataplane_path='s3://b/snap-1'),
            'snap-2':
            DataplaneBackup(snapshot_id='snap-2',
                            dataplane_path='s3://b/snap-2'),
        },
        snapshot_compliance={
            'snap-1':
            SnapshotComplianceRecord(
                snapshot_id='snap-1',
                active_consumer_count=1,
                compliance_delete_at=None,
            ),
        },
        local_copies={
            'snap-1:worker-a':
            SnapshotLocalCopyRecord(
                snapshot_id='snap-1',
                pool_worker_uid='worker-a',
                local_state=SnapshotLocalCopyState.WARM,
                size_bytes=1_024_000,
            ),
        },
    )


def test_pool_state_empty_k8s_variant():
    state = PoolState(placement_state=K8sPlacementState())
    assert state.snapshots == {}
    assert state.placement_state.type == 'k8s'
    assert state.placement_state.pvcs == {}
    assert state.dataplane_backups == {}
    assert state.snapshot_compliance == {}
    assert state.local_copies == {}


def test_pool_state_empty_singlehost_variant():
    state = PoolState(placement_state=SinglehostPlacementState())
    assert state.snapshots == {}
    assert state.placement_state.type == 'singlehost'
    assert state.dataplane_backups == {}
    assert state.snapshot_compliance == {}
    assert state.local_copies == {}


def test_pool_state_round_trip_singlehost(tmp_path):
    """Singlehost variant round-trips cleanly; ``type`` discriminator
    preserved so deserialization picks ``SinglehostPlacementState``."""
    original = PoolState(placement_state=SinglehostPlacementState())
    path = str(tmp_path / 'singlehost.json')
    save_json(original, path)
    loaded = load_json(path, PoolState)
    assert loaded == original
    assert isinstance(loaded.placement_state, SinglehostPlacementState)


def test_pool_state_round_trip(tmp_path, populated_state: PoolState):
    path = str(tmp_path / 'pool_state.json')
    save_json(populated_state, path)
    loaded = load_json(path, PoolState)
    assert loaded == populated_state


def test_rollback_loads_but_drops_silent_worker_suspect_map():
    current = K8sPlacementState(
        pvcs={'worker-a': PvcState(pool_worker_uid='worker-a')},
        silent_worker_suspect_since={'worker-a': 123.0},
    )

    rollback = from_json(to_json(current), _RollbackK8sPlacementState)

    assert set(rollback.pvcs) == {'worker-a'}
    rolled_forward = from_json(to_json(rollback), K8sPlacementState)
    assert set(rolled_forward.pvcs) == {'worker-a'}
    assert rolled_forward.silent_worker_suspect_since == {}


def test_placement_request_round_trip(tmp_path):
    request = PlacementRequest(
        snapshot_id='snap-1',
        job_id='job-1',
        workflow_id='wf-1',
        activity=SnapshotActivity.TRAIN,
        resource_hint=ResourceHint(min_ssd_gib=501),
        source_size_bytes=4096,
    )
    path = str(tmp_path / 'request.json')
    save_json(request, path)
    assert load_json(path, PlacementRequest) == request


def test_block_b_strategy_types_round_trip(tmp_path):
    cases = [
        SnapshotComplianceRecord(
            snapshot_id='snap-1',
            retention_mode=RetentionMode.DEFAULT,
            active_consumer_count=1,
        ),
        SnapshotLocalCopyRecord(
            snapshot_id='snap-1',
            pool_worker_uid='worker-a',
            local_state=SnapshotLocalCopyState.WARM,
            size_bytes=2048,
        ),
        RouteDecision(
            path='durable',
            pool_worker_uid='worker-a',
            provisioned_new_worker=False,
            reason='test',
        ),
        ProvisionedHandle(
            kind='durable',
            pool_worker_uid='worker-a',
            task_queue='diskgraph-activity-tq-job-1',
        ),
        PoolStatsFilter(
            pool_worker_uids=['worker-a'],
            include_snapshots=True,
            include_leases=True,
            include_placements=True,
        ),
        PoolWorkerStats(
            pool_worker_uid='worker-a',
            health=WorkerStatus.READY,
            pvc_free_bytes=500,
            pvc_total_bytes=1000,
            gpu_slots_total=4,
            gpu_slots_in_use=1,
            active_leases=1,
            snapshot_ids=['snap-1'],
        ),
        PoolStats(
            num_nodes=1,
            num_healthy_nodes=1,
            durable_workers=[
                PoolWorkerStats(
                    pool_worker_uid='worker-a',
                    health=WorkerStatus.READY,
                    pvc_free_bytes=500,
                    pvc_total_bytes=1000,
                )
            ],
        ),
        ReconcileDelta(
            retired_uids={'worker-b'},
            unexpected_uids={'worker-a'},
            observed_pool_worker_uids={'worker-a', 'worker-b', 'worker-c'},
            live_pool_worker_uids={'worker-c'},
            orphan_snapshots={'snap-orphan': 'worker-c'},
        ),
    ]
    for i, value in enumerate(cases):
        path = str(tmp_path / f'block_b_{i}.json')
        save_json(value, path)
        assert load_json(path, type(value)) == value


def test_worker_inventory_defaults():
    inv = WorkerInventory(
        pool_worker_uid='worker-a',
        state=WorkerStateSnapshot(ssd_total_bytes=1000, ssd_used_bytes=500),
    )
    assert inv.status == WorkerStatus.READY
    assert inv.gpu_slots_total == 0
    assert inv.gpu_slots_in_use == 0
    assert inv.state.ready_snapshots == []


def test_active_job_round_trip(tmp_path):
    job = ActiveJob(job_id='job-1', job_type='train', snapshot_id='snap-1')
    path = str(tmp_path / 'j.json')
    save_json(job, path)
    assert load_json(path, ActiveJob) == job


def test_pvc_state_round_trip(tmp_path):
    pvc = PvcState(pool_worker_uid='worker-a',
                   created_at=datetime(2026, 4, 20, tzinfo=timezone.utc))
    path = str(tmp_path / 'pvc.json')
    save_json(pvc, path)
    assert load_json(path, PvcState) == pvc


def test_worker_status_report_round_trip(tmp_path):
    report = WorkerStatusReport(
        worker_name='worker-a',
        state=WorkerStateSnapshot(
            ssd_total_bytes=1000,
            ssd_used_bytes=500,
            ready_snapshots=['snap-1'],
            dataplane_total_bytes=2_000,
            dataplane_used_bytes=500,
            dataplane_same_filesystem_as_ssd=False,
        ),
        event=WorkerEvent(
            type=WorkerEventType.SNAPSHOT_READY,
            snapshot_id='snap-1',
            timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
        ),
    )
    path = str(tmp_path / 'r.json')
    save_json(report, path)
    loaded = load_json(path, WorkerStatusReport)
    assert loaded == report
    assert loaded.state.dataplane_free_bytes == 1_500


def test_dataplane_capacity_worker_state_defaults_unknown():
    state = WorkerStateSnapshot(ssd_total_bytes=1_000, ssd_used_bytes=100)

    assert state.dataplane_total_bytes == 0
    assert state.dataplane_used_bytes == 0
    assert state.dataplane_free_bytes == 0
    assert state.dataplane_same_filesystem_as_ssd is None


def test_worker_dataplane_ready_event_round_trip(tmp_path):
    event = WorkerEvent(
        type=WorkerEventType.SNAPSHOT_PERSIST_UPLOADED,
        snapshot_id='snap-1',
        dataplane_path='s3://bucket/env/snapshots/snap-1',
        size_bytes=1024,
        timestamp=datetime(2026, 4, 20, tzinfo=timezone.utc),
    )
    path = str(tmp_path / 'dataplane-ready-event.json')
    save_json(event, path)
    assert load_json(path, WorkerEvent) == event


def test_worker_status_report_without_event():
    """Periodic heartbeat: report with only state, no event."""
    report = WorkerStatusReport(
        worker_name='worker-a',
        state=WorkerStateSnapshot(ssd_total_bytes=1000, ssd_used_bytes=500),
    )
    assert report.worker_storage_mode == PoolStorageMode.DAEMON_ON_PVC
    assert report.event is None
