# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized timeout configuration for PoolController interactions.

``place_snapshot`` is long-running by design: the controller synchronously
waits for K8s daemon registration (Karpenter spin-up + ECR image pull +
pod boot + Temporal worker registration), commonly minutes on cold start.
The AutoRPC framework default ``rpc_timeout`` is 30s — far too short.
Configure it explicitly here, alongside the matching activity-side
``start_to_close`` / ``schedule_to_close`` budgets.

Three timeout layers, applied per RPC method:

1. **AutoRPC ``rpc_timeout``** — set on the ``PoolControllerClient`` via
   :data:`DEFAULT_METHOD_TIMEOUTS`. Caps the inner ``_RpcMethodWorkflow``'s
   ``schedule_to_close_timeout`` (i.e. how long the controller may run
   server-side per call). RPCs omitted here use the framework default (30s).

2. **Activity ``start_to_close_timeout``** — set on
   ``workflow.execute_activity()`` by ``pool.wf_coordinator``. Caps a
   single workflow-side attempt of the wrapper activity
   (``place_snapshot_activity`` etc.). Must accommodate the underlying
   AutoRPC ``rpc_timeout`` with margin for network / scheduler jitter.

3. **Activity ``schedule_to_close_timeout``** — total budget across
   activity retries. ``KumoActivityOutboundInterceptor`` requires this be
   set; for ``place_snapshot``'s indefinite-retry policy it acts as a
   safety net, while for release it bounds cleanup so a stuck release
   can't mask the original exit exception.

Layered: workflow ─[start_to_close, schedule_to_close]→ AutoRPC client
─[rpc_timeout]→ controller.
"""
from datetime import timedelta
from typing import Dict

from temporalio_commons.auto_rpc_config import AutoRpcTimeoutConfig

# ────────────────────────────────────────────────────────────────────
# place_snapshot
# ────────────────────────────────────────────────────────────────────
# A fresh durable placement can spend 10 minutes provisioning its daemon,
# followed by 15 minutes starting the activity pod and 2 minutes registering
# its poller. Include cleanup margin inside the RPC, then leave the wrapper
# additional room for transport and scheduler jitter. schedule_to_close is the
# outer retry-budget safety net; it does not promise two maximum-length cold
# attempts.
PLACE_SNAPSHOT_RPC_TIMEOUT = timedelta(minutes=30)
PLACE_START_TO_CLOSE = timedelta(minutes=35)
PLACE_SCHEDULE_TO_CLOSE = timedelta(minutes=45)

# ────────────────────────────────────────────────────────────────────
# release_lease
# ────────────────────────────────────────────────────────────────────
# This runs in the workflow's cleanup path. Keep retries bounded so cleanup
# cannot mask the original exit exception; the lease TTL and dangling-lease
# sweep recover a zombie if release never lands. Allow 60s for Job deletion,
# plus 30s for the controller RPC and another 30s for the wrapper activity.
ACTIVITY_JOB_DELETE_TIMEOUT_SEC = 60
RELEASE_LEASE_RPC_TIMEOUT = timedelta(seconds=ACTIVITY_JOB_DELETE_TIMEOUT_SEC +
                                      30)
RELEASE_START_TO_CLOSE = timedelta(seconds=ACTIVITY_JOB_DELETE_TIMEOUT_SEC + 60)
RELEASE_SCHEDULE_TO_CLOSE = timedelta(minutes=5)

# ────────────────────────────────────────────────────────────────────
# Pool-worker provisioning and recovery
# ────────────────────────────────────────────────────────────────────
# Cap on daemon worker provisioning/recovery. Retirement gives a newly
# created replacement pod the same window to register before treating a
# non-live K8s row as stale recovery residue.
POOL_WORKER_PROVISION_TIMEOUT_SEC = 10 * 60

# Briefly prefer capacity already being provisioned over another worker/PVC.
# 90s leaves margin inside the 30m placement RPC, including the durable
# failure path below (10m daemon + 15m activity + 2m poller + 1m cleanup).
POOL_WORKER_STARTUP_GRACE_SEC = 90

# ────────────────────────────────────────────────────────────────────
# Activity-pod provisioning (within place_snapshot)
# ────────────────────────────────────────────────────────────────────
# Cap on how long ``provision_activity_pod`` /
# ``provision_ephemeral_activity_pod`` will wait for the Job's pod to
# reach Running. Dominated by ECR image pull on a cold node:
#   - kumo-backend-cpu (~10 GB) → ~3-4 min
#   - kumo-backend-gpu (~16 GB) → ~5-7 min
# Plus Karpenter node provisioning on the eph pool (~2-3 min cold).
# The durable failure path (10m daemon + 15m pod + 2m poller + 1m cleanup)
# stays within PLACE_SNAPSHOT_RPC_TIMEOUT, so the AutoRPC budget remains the
# visible failure surface.
ACTIVITY_POD_PROVISION_TIMEOUT_SEC = 15 * 60
# Use a separate two-minute budget for worker construction and poller
# registration after the pod reaches Running.
ACTIVITY_POLLER_READY_TIMEOUT_SEC = 2 * 60

# ────────────────────────────────────────────────────────────────────
# AutoRPC client config — per-method ``rpc_timeout`` defaults that the
# PoolController client injects on top of the framework defaults.
# Operators can still override per env via
# ``RPC_TIMEOUT_POOLCONTROLLER__<METHOD>``; ``parse_from_env``
# populates ``method_timeout_config`` first, and these defaults are
# applied via ``setdefault`` so env vars win.
# ────────────────────────────────────────────────────────────────────
DEFAULT_METHOD_TIMEOUTS: Dict[str, AutoRpcTimeoutConfig] = {
    'place_snapshot': AutoRpcTimeoutConfig(
        rpc_timeout=PLACE_SNAPSHOT_RPC_TIMEOUT),
    'release_lease': AutoRpcTimeoutConfig(
        rpc_timeout=RELEASE_LEASE_RPC_TIMEOUT),
}
