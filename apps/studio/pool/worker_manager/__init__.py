# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public surface of the diskgraph pool-worker manager."""
from pool.worker_manager.init_canary import (
    InitCanaryValidationError,
    run_init_canary,
)
from pool.worker_manager.k8s import K8sPoolWorkerManager
from pool.worker_manager.types import (
    ActivityPodInfo,
    ActivityPodNotFoundError,
    ApiServerThrottledError,
    ApiServerTransientError,
    InvalidPvcSpecError,
    NodePoolNotFoundError,
    PodStatus,
    PoolWorkerAlreadyExistsError,
    PoolWorkerCleanupFailedError,
    PoolWorkerInfo,
    PoolWorkerManagerError,
    PoolWorkerNotFoundError,
    PoolWorkerNotReadyError,
    ProvisionTimeoutError,
    PVCBindTimeoutError,
    PvcSpec,
    SecretKeyRef,
    QuotaExceededError,
    ResourceSpec,
)

__all__ = [
    # Concrete managers
    "K8sPoolWorkerManager",
    # Init-time node-pool validation
    "run_init_canary",
    "InitCanaryValidationError",
    # Data classes
    "ActivityPodInfo",
    "PodStatus",
    "PoolWorkerInfo",
    "PvcSpec",
    "SecretKeyRef",
    "ResourceSpec",
    # Exception taxonomy
    "ActivityPodNotFoundError",
    "ApiServerThrottledError",
    "ApiServerTransientError",
    "InvalidPvcSpecError",
    "NodePoolNotFoundError",
    "PoolWorkerAlreadyExistsError",
    "PoolWorkerCleanupFailedError",
    "PoolWorkerManagerError",
    "PoolWorkerNotFoundError",
    "PoolWorkerNotReadyError",
    "ProvisionTimeoutError",
    "PVCBindTimeoutError",
    "QuotaExceededError",
]
