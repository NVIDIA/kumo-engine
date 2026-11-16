# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :mod:`pool.worker_manager.types` — dataclasses +
exception taxonomy. Pure-data tests; no mocking, no I/O.
"""
# Stub injection for laptop runs (opt-in via
# KUMO_WORKER_MANAGER_TEST_STUBS=1). Must run before
# ``import pool.*`` since pool.worker_manager.k8s imports io_utils
# transitively.
import os as _os  # noqa: E402,I100
import sys as _sys  # noqa: E402

if _os.environ.get("KUMO_WORKER_MANAGER_TEST_STUBS") == "1":
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "_stubs"))

import pytest  # noqa: E402

# yapf disagrees with isort on wrapping this import; isort wins here.
# yapf: disable
from pool.worker_manager.types import (  # noqa: E402
    ActivityPodInfo,
    ActivityPodNotFoundError,
    ApiServerThrottledError,
    ApiServerTransientError,
    InvalidPvcSpecError,
    K8sPermissionError,
    NodePoolNotFoundError,
    PoolWorkerAlreadyExistsError,
    PoolWorkerCleanupFailedError,
    PoolWorkerInfo,
    PoolWorkerManagerError,
    PoolWorkerNotFoundError,
    PoolWorkerNotReadyError,
    ProvisionTimeoutError,
    PVCBindTimeoutError,
    PvcSpec,
    QuotaExceededError,
    ResourceSpec,
)

# yapf: enable

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


def test_pvc_spec_requires_size_gib():
    spec = PvcSpec(size_gib=900)
    assert spec.size_gib == 900


def test_pvc_spec_size_gib_is_exact_tier():
    spec = PvcSpec(size_gib=1800)
    assert spec.size_gib == 1800


def test_resource_spec_default_is_cpu_only_best_effort():
    spec = ResourceSpec()
    assert spec.cpu_request is None
    assert spec.cpu_limit is None
    assert spec.memory_request is None
    assert spec.memory_limit is None
    assert spec.gpu_count == 0


def test_resource_spec_rejects_negative_gpu_count():
    with pytest.raises(ValueError, match="gpu_count must be >= 0"):
        ResourceSpec(gpu_count=-1)


def test_pool_worker_info_fields():
    info = PoolWorkerInfo(
        pool_worker_uid="abc",
        pvc_name="diskgraph-pvc-abc",
        daemon_pod_name="diskgraph-daemon-abc-xxx",
        node_name="ip-10-0-1-1.ec2.internal",
        status="Registered",
    )
    assert info.pool_worker_uid == "abc"
    assert info.pvc_name == "diskgraph-pvc-abc"
    assert info.node_name == "ip-10-0-1-1.ec2.internal"
    assert info.status == "Registered"
    assert info.created_at is None
    assert info.deletion_started_at is None


def test_pool_worker_info_node_name_optional():
    # Pre-scheduling, both daemon_pod_name and node_name are None.
    info = PoolWorkerInfo(
        pool_worker_uid="abc",
        pvc_name="diskgraph-pvc-abc",
        daemon_pod_name=None,
        node_name=None,
        status="Pending",
    )
    assert info.daemon_pod_name is None
    assert info.node_name is None
    assert info.status == "Pending"


def test_activity_pod_info_eph_has_no_pool_worker_uid():
    eph = ActivityPodInfo(
        pod_name="diskgraph-eph-activity-job-1",
        pool_worker_uid=None,
        job_id="job-1",
        node_name="ip-eph",
        status="Running",
    )
    assert eph.pool_worker_uid is None
    assert eph.job_id == "job-1"


def test_activity_pod_info_cache_has_pool_worker_uid():
    cache = ActivityPodInfo(
        pod_name="diskgraph-activity-job-2",
        pool_worker_uid="abc",
        job_id="job-2",
        node_name="ip-cache",
        status="Running",
    )
    assert cache.pool_worker_uid == "abc"


# ---------------------------------------------------------------------------
# Exception taxonomy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_cls",
    [
        ProvisionTimeoutError,
        QuotaExceededError,
        InvalidPvcSpecError,
        NodePoolNotFoundError,
        K8sPermissionError,
        PoolWorkerNotFoundError,
        PoolWorkerNotReadyError,
        PVCBindTimeoutError,
        ActivityPodNotFoundError,
        ApiServerThrottledError,
        ApiServerTransientError,
        PoolWorkerAlreadyExistsError,
        PoolWorkerCleanupFailedError,
    ],
)
def test_exceptions_inherit_from_base(exc_cls):
    """Callers (pool controller) catch by base class. Verify the
    inheritance is intact so ``except PoolWorkerManagerError`` keeps
    working as new exception types are added."""
    assert issubclass(exc_cls, PoolWorkerManagerError)
    assert issubclass(exc_cls, Exception)
    raised = pytest.raises(exc_cls, match="boom")
    with raised:
        raise exc_cls("boom")


def test_pool_worker_manager_error_is_exception():
    assert issubclass(PoolWorkerManagerError, Exception)


# Retryability markers — the AutoRPC framework's ``is_retryable``
# predicate treats ``RuntimeError`` instances as retryable, so
# transient pool errors multi-inherit from it. Deterministic ones do
# not, so the framework wraps them as ``_AutoRpcNonRetriableError``.

_RETRYABLE_POOL_ERRORS = [
    ProvisionTimeoutError,
    QuotaExceededError,
    PVCBindTimeoutError,
    ApiServerThrottledError,
    ApiServerTransientError,
    PoolWorkerCleanupFailedError,
]

_NON_RETRYABLE_POOL_ERRORS = [
    InvalidPvcSpecError,
    NodePoolNotFoundError,
    K8sPermissionError,
]


@pytest.mark.parametrize("exc_cls", _RETRYABLE_POOL_ERRORS)
def test_retryable_pool_errors_are_runtime_errors(exc_cls):
    assert issubclass(exc_cls, RuntimeError)


@pytest.mark.parametrize("exc_cls", _NON_RETRYABLE_POOL_ERRORS)
def test_non_retryable_pool_errors_are_not_runtime_errors(exc_cls):
    assert not issubclass(exc_cls, RuntimeError)
