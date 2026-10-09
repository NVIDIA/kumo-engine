# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K8s pod-config helpers for the (future) ``K8sPlacementStrategy``.

Computes the env-var dict that pool-worker pods need so
:func:`pool.worker_main.main` can bring up the right kind of worker.
Other parts of the manifest — image, resource requests, node selector,
volumes (PVC vs ``emptyDir``) — are the strategy's concern; this
module focuses on environment variables only.

The controller calls one of three builders depending on the placement:

* :func:`daemon_pod_env_vars` — long-lived daemon pod owning the
  per-worker PVC, runs ``ensure_snapshot`` / ``delete_snapshot``.
  Lives once per worker; subscribes to ``diskgraph-daemon-{uid}``.
* :func:`activity_pod_env_vars` (PVC-mount variant) — activity-scoped
  pod scheduled onto the daemon's node, mounts the same PVC,
  delegates ``ensure_snapshot`` writes back to the daemon (see
  diskgraph_pool_controller.md § Snapshot Mutation Invariant).
  Carries the same ``POOL_WORKER_UID`` as the daemon.
* :func:`activity_pod_env_vars` (overflow / ``emptyDir`` variant) —
  activity-scoped pod on an ephemeral node with a private
  ``emptyDir``; UID prefix differs (``pool-eh-worker-``) so
  :func:`pool.task_queues.is_activity_scoped_pod` returns False.

All three go through :func:`pool_worker_env_vars`; the per-mode
helpers above just pre-fill the right ``task_queue`` / ``storage_mode`` /
concurrency caps.
"""
import os
from typing import Dict, List

from pool.storage_mode import PoolStorageMode
from pool.task_queues import activity_queue, daemon_dispatch_queue
from service_framework.deploy_env_config import deploy_env

# Env vars forwarded from controller to pool worker pods so the
# daemon/activity's ``deploy_env()`` resolves to the same config the
# controller used. CUSTOM_S3_DATA_PLANE_BUCKET is the legacy storage path used
# when DEPLOY_ENV_CONFIG_JSON is absent.
_FORWARDED_DEPLOY_ENV_KEYS = (
    'DEPLOY_ENV_CONFIG_JSON',
    'DEPLOY_STAGE',
    'INSTANCE_ID',
    'AWS_REGION',
    'CUSTOM_S3_DATA_PLANE_BUCKET',
)

# Non-secret provider selectors required by connector I/O inside pool workers.
# Credentials are deliberately absent: connector credentials are delivered by
# the activity broker, not copied through pod environment variables.
_FORWARDED_CONNECTOR_IO_ENV_KEYS = (
    'GCP_STORAGE_TRANSFER_BUCKET',
    'GOOGLE_CLOUD_PROJECT',
)

# DEV-only: deploy_env().id_suffix prepends getpass.getuser() (which
# reads $USER) in dev mode for multi-dev disambiguation on shared
# clusters. Forward USER so daemon and controller compute matching
# id_suffixes (otherwise their AutoRPC task queue names diverge and
# report_status never lands). Prod/staging hit a different id_suffix
# branch that does not depend on USER.
_DEV_ONLY_FORWARDED_DEPLOY_ENV_KEYS = ('USER', )


def forwarded_deploy_env_vars() -> Dict[str, str]:
    """Deploy config keys forwarded from controller to worker pods."""
    keys: List[str] = list(
        (*_FORWARDED_DEPLOY_ENV_KEYS, *_FORWARDED_CONNECTOR_IO_ENV_KEYS))
    if deploy_env().deploy_mode.is_dev:
        keys.extend(_DEV_ONLY_FORWARDED_DEPLOY_ENV_KEYS)
    env = {key: os.environ[key] for key in keys if key in os.environ}
    # The controller may set DEPLOY_ENV_CONFIG_FILE (a host-local path)
    # instead of DEPLOY_ENV_CONFIG_JSON; inline the file content here
    # since the path isn't valid inside the pod.
    if 'DEPLOY_ENV_CONFIG_JSON' not in env:
        config_file = os.environ.get('DEPLOY_ENV_CONFIG_FILE')
        if config_file and os.path.isfile(config_file):
            with open(config_file) as f:
                env['DEPLOY_ENV_CONFIG_JSON'] = f.read()
    return env


def pool_worker_env_vars(
    *,
    pool_worker_uid: str,
    task_queue: str,
    storage_mode: PoolStorageMode,
    gpu_concurrency: int = 1,
    max_concurrent_activities: int = 0,
) -> Dict[str, str]:
    return {
        'POOL_WORKER_UID': pool_worker_uid,
        'POOL_TASK_QUEUE': task_queue,
        'POOL_STORAGE_MODE': storage_mode.value,
        'POOL_GPU_CONCURRENCY': str(gpu_concurrency),
        'POOL_MAX_CONCURRENT_ACTIVITIES': str(max_concurrent_activities),
    }


def daemon_pod_env_vars(
    *,
    pool_worker_uid: str,
    max_concurrent_activities: int = 0,
) -> Dict[str, str]:
    return {
        **forwarded_deploy_env_vars(),
        **pool_worker_env_vars(
            pool_worker_uid=pool_worker_uid,
            task_queue=daemon_dispatch_queue(pool_worker_uid),
            storage_mode=PoolStorageMode.DAEMON_ON_PVC,
            gpu_concurrency=1,
            max_concurrent_activities=max_concurrent_activities,
        ),
    }


def activity_pod_env_vars(
    *,
    pool_worker_uid: str,
    job_id: str,
    storage_mode: PoolStorageMode = PoolStorageMode.ACTIVITY_ON_PVC,
    gpu_concurrency: int = 1,
    require_gpu: bool = False,
) -> Dict[str, str]:
    # PVC-mount and emptyDir-overflow variants share these env vars and
    # differ only in pod-spec volume / node selector (strategy's job).
    env = {
        **forwarded_deploy_env_vars(),
        **pool_worker_env_vars(
            pool_worker_uid=pool_worker_uid,
            task_queue=activity_queue(job_id),
            storage_mode=storage_mode,
            gpu_concurrency=gpu_concurrency,
            max_concurrent_activities=1,
        ),
    }
    if require_gpu:
        env['KUMO_REQUIRE_GPU'] = 'true'
        env['KUMO_ACCELERATOR'] = 'gpu'
    return env
