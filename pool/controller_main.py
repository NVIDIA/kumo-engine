# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool controller entrypoint (``python -m pool.controller_main``).

Boots a single-replica :class:`PoolController` over AutoRPC.
"""
import asyncio
import faulthandler
import os
from pathlib import Path

from prometheus_client import start_http_server

from async_utils.loop_guards import install_loop_guards
from io_utils import logging
from pool.bootstrap import init_logging_and_tracing
from pool.service import create_pool_controller, run_configured_init_canary
from service_framework.deploy_env_config import deploy_env
from service_framework.env_var import EnvVar
from temporalio_commons.auto_rpc import (
    AutoRpcServerWorkerConfig,
    create_server_from_config,
)
from temporalio_commons.runtime_utils import init_temporal_runtime

log = logging.getLogger(__name__)

CONTROLLER_NAME = 'pool-controller'
READY_PATH = Path('/tmp/kumo-poolcontroller-ready')

# Prometheus exposition port; the kumo-delivery poolcontroller Service/Deployment
# scrapes ``:<port>/metrics``. Overridable for local runs / port conflicts.
METRICS_PORT = EnvVar(
    name='POOL_METRICS_PORT',
    type=int,
    default_value=9090,
    description='Prometheus exposition port for the pool controller.',
).parse_value()
# Bind address for the metrics server. Default ``::`` (IPv6/dual-stack) — the
# k8s pods are IPv6, so the scrape target is the pod's IPv6 address; binding the
# prometheus_client default 0.0.0.0 (IPv4-only) makes the scrape fail (up=0)
# while in-pod localhost still works. Mirrors rest/main.py's METRICS_HOST.
METRICS_HOST = EnvVar(
    name='METRICS_HOST',
    type=str,
    default_value='::',
    description='Bind address for the pool controller metrics server.',
).parse_value()

faulthandler.enable()


async def main() -> None:
    init_logging_and_tracing(CONTROLLER_NAME)
    log.info('pool.controller_main starting (pid=%d)', os.getpid())

    # The metrics listener starts before the init canary, so an HTTP probe on
    # that port would declare the controller ready while startup is blocked.
    # Keep readiness false until node-pool validation and worker construction
    # have both completed.
    READY_PATH.unlink(missing_ok=True)

    start_http_server(METRICS_PORT, addr=METRICS_HOST)
    log.info('pool metrics server listening on [%s]:%d/metrics', METRICS_HOST,
             METRICS_PORT)

    deploy_config = deploy_env()

    # Constructing the controller performs one startup reconcile with silent
    # worker ageing disabled. Run that before the canary so leaked/unexpected
    # workers from a prior crash cannot consume the capacity the canary needs.
    # Keep periodic maintenance stopped: expected persisted workers still have
    # no RPC server through which to prove health during canary validation.
    controller = await create_pool_controller(
        deploy_config,
        start_maintenance_loop=False,
    )

    # INIT-TIME node-pool validation ("canary"). Prove each configured capacity
    # before serving: existing Registered workers prove their exact persisted
    # cache tier only when their live pod/PVC still match the current envelope;
    # unproven tiers run an active canary. Failed capabilities are removed from
    # new placement, while zero available cache tiers remains a fatal startup
    # error.
    # Result-cached as an annotation on the controller's own Deployment
    # (``kumo.ai/diskgraph-pools-validated``) so a healthy pool isn't
    # re-provisioned on every restart; see ``pool.worker_manager.init_canary``.
    await run_configured_init_canary(deploy_config, controller)

    # Match ``pool.client.get_controller_client``: concurrent devservers use
    # KUMO_AUTORPC_SERVER_NAME_PREFIX to give each controller its own queue.
    server_config = AutoRpcServerWorkerConfig.parse_from_env(deploy_config)
    worker_env, worker = await create_server_from_config(
        controller,
        server_config,
    )
    # The first maintenance tick gets a boot-local heartbeat grace window.
    # Scheduling it immediately before worker.run() lets RPC polling and the
    # grace period start together, after all blocking startup work is done.
    controller._start_maintenance_loop()

    try:
        # Now that worker init is done (Temporal SDK pre-fetches build_id via
        # synchronous file I/O), start the event-loop stall watchdog. Startup
        # is once-only so steady-state stalls past this point are what we want
        # to catch.
        install_loop_guards()

        # This is startup readiness, not a steady-state Temporal health check.
        # A hard process failure makes the container unready; controlled
        # shutdown removes the sentinel below.
        READY_PATH.touch()
        log.info('pool.controller_main running')
        await worker.run()
    finally:
        READY_PATH.unlink(missing_ok=True)
        worker_env.shutdown()


if __name__ == '__main__':
    init_temporal_runtime(CONTROLLER_NAME)
    asyncio.run(main())
