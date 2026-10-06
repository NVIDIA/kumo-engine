# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Common script to deploy and run one or many Temporal Worker entities. All
workers must register a "run worker" func(coroutine) with the name of worker.

For example:

    # /module1/worker1.py
    @register_worker('worker-1')
    async def run(worker_options: WorkerOptions):
    ... # create and run the worker

    # /module2/worker2.py
    @register_worker('worker-2')
    async def run(worker_options: WorkerOptions):
    ... # create and run the worker

There are two ways to configure and run workers:
1. Create a ymal file of GlobalWorkerOptions, and pass the ymal file path via
"SETTINGS_YAML" env variable, e.g.:
    SETTINGS_YAML=/path/to/file.ymal python temporalio_commons/worker_main.py

2. pre-define worker's custom options in the GLOBAL_WORKER_OPTIONS from
worker_registry.py, then, pass comma-separated worker names through "WORKERS"
env variables to run the worker(s), e.g.:
    WORKERS=<worker-1,worker-2,...> python temporalio_commons/worker_main.py

If neither "SETTINGS_YAML" or "WORKERS" is specified, by default all workers
will be running with default or pre-defined options.
"""
import asyncio
import copy
import dataclasses
import faulthandler
import os
from os import path as osp
from pprint import pformat
from typing import Any, Callable, Dict, List, Sequence, Tuple, Type

from temporalio.worker import Worker

from async_utils.loop_guards import install_loop_guards
from common_utils.startup import log_app_startup
from io_utils import fsutil, logging
from kumo.connector import Connector
from kumo.home import get_local_home_dir
from monitoring.settings import MonitoringSettings
from rest.backends import (
    ML_PRED_MAX_NUM_ENTITIES_PER_PARTITION,
    ML_PRED_MAX_NUM_ROWS_PER_PARTITION,
)
from rest.backends.data_backend.worker import DATA_BACKEND_WORKER_NAME
from rest.backends.ml_backend.worker import (
    ML_FAST_TRAINING_ENABLED,
    ML_WORKFLOW_WORKER,
    PRED_COL_STATS,
    SLACK_URL,
)
from rest.constants import (
    DATA_BACKEND_TASK_QUEUE,
    DISABLE_REMOTE_INGESTION,
    ML_WORKFLOW_TASK_QUEUE,
)
from rest.utils import tracing
from service_framework.deploy_env_config import DeployEnvConfig, deploy_env
from service_framework.deploy_stage import DeployStage
from service_framework.env_var import EnvVar
from temporalio_commons.runtime_utils import init_temporal_runtime
from temporalio_commons.temporal_client import DeployStage
from temporalio_commons.worker_registry import (
    GLOBAL_WORKER_REGISTRY,
    WorkerEnv,
    WorkerOptions,
)

log = logging.getLogger(__name__)

# Dump the stacktrace when segfault occurs
faulthandler.enable()

HAS_LOCAL_INGESTION = EnvVar(
    DISABLE_REMOTE_INGESTION,
    bool,
    default_value=True,
).parse_value()


def _ml_backend_env_variables() -> List[EnvVar]:
    return [
        # Refer MLJobSettings dataclass for documentation on default
        # values for max entities and rows per partition.
        EnvVar(ML_PRED_MAX_NUM_ENTITIES_PER_PARTITION, int,
               default_value=10 * 10**6),
        EnvVar(ML_PRED_MAX_NUM_ROWS_PER_PARTITION, int,
               default_value=300 * 10**6),
        EnvVar(ML_FAST_TRAINING_ENABLED, bool, default_value=False),
        EnvVar(PRED_COL_STATS, bool, default_value=True),
        EnvVar(SLACK_URL, str, default_value=''),
    ]


# Code-defined defaults.
GLOBAL_WORKER_OPTIONS: Dict[str, WorkerOptions] = {
    DATA_BACKEND_WORKER_NAME:
    WorkerOptions(
        task_queue=DATA_BACKEND_TASK_QUEUE,
        # Single thread-pool executor: the compute process-pool worker was
        # retired once every data-backend activity was de-globalized and moved
        # onto the thread pool, so no max_worker_processes / multi_executors
        # config is needed here.
        max_worker_threads=20,
        env_variables=[
            EnvVar(DISABLE_REMOTE_INGESTION, bool, default_value=True),
        ],
    ),
    ML_WORKFLOW_WORKER:
    WorkerOptions(
        task_queue=ML_WORKFLOW_TASK_QUEUE,
        max_worker_threads=EnvVar('ML_WORKFLOW_MAX_WORKER_THREADS', int,
                                  default_value=10).parse_value(),
        env_variables=_ml_backend_env_variables()),

    # Add an entry here to predefine custom options for any new worker.
    #
    # 'worker-name': WorkerOptions(...),
}

# Global env variables across all workers.
_WORKERS = EnvVar(
    'WORKERS',
    str,
    required=False,
    default_value=','.join(GLOBAL_WORKER_REGISTRY.all_workers()),
)
_WORKER_ENABLE_TRACING = EnvVar(
    'WORKER_ENABLE_TRACING', bool, required=False, default_value=False,
    description='Process-level on/off switch for tracing.')
_WORKER_LOGGING_MODE = EnvVar('WORKER_LOGGING_MODE', type=logging.LoggingMode,
                              required=False,
                              default_value=logging.LoggingMode.DEBUG)
_WORKER_APP_METRICS_PORT = EnvVar(
    'WORKER_APP_METRICS_PORT',
    int,
    required=False,
    default_value=0,
    description='Optional port for prometheus_client application metrics.',
)
_WORKER_APP_METRICS_HOST = EnvVar(
    'WORKER_APP_METRICS_HOST',
    str,
    required=False,
    default_value='0.0.0.0',
    description='Bind host for prometheus_client application metrics.',
)


def _start_app_metrics_server() -> None:
    port = _WORKER_APP_METRICS_PORT.parse_value()
    if port <= 0:
        return

    from prometheus_client import start_http_server
    start_http_server(port, addr=_WORKER_APP_METRICS_HOST.parse_value())
    log.info('Started worker application metrics server on port %s', port)


# Get worker options for the given worker name. If worker name doesn't
# exist, a reasonable default options will be returned.
def _get_worker_options(worker_name: str,
                        deploy_config: DeployEnvConfig) -> WorkerOptions:
    options = copy.deepcopy(
        GLOBAL_WORKER_OPTIONS.get(worker_name, WorkerOptions()))
    # Set default task queue if not provided.
    if not options.task_queue:
        options.task_queue = worker_name + '-task-queue'
    # Update task queue based on temporal config.
    options.task_queue = deploy_config.task_queue(options.task_queue)

    return options


async def build_workers(
    deploy_config: DeployEnvConfig,
    worker_names: List[str],
    env_var_test_override: Dict[str, Any] | None = None,
    activities_override: Sequence[Callable] = [],
    workflows_override: Sequence[Type] = [],
) -> Tuple[List[Worker], List[WorkerEnv]]:
    worker_registry = GLOBAL_WORKER_REGISTRY

    worker_envs: List[WorkerEnv] = []
    workers: List[Worker] = []
    for worker_name in worker_names:
        # Make sure worker name has been registered with a RunWorkerFunc
        # via @register_worker(<worker_name>) decorator
        if not worker_registry.has_worker(worker_name):
            raise ValueError(
                f'Worker {worker_name} is not registered, '
                f'registered workers are: {worker_registry.all_workers()}')
        worker_deploy_config = dataclasses.replace(deploy_config,
                                                   client_id=worker_name)
        options = _get_worker_options(worker_name, worker_deploy_config)
        worker_env = WorkerEnv(worker_name, options, worker_deploy_config,
                               env_var_test_override=env_var_test_override,
                               activities_override=activities_override,
                               workflows_override=workflows_override)
        worker_envs.append(worker_env)
        workers.extend(await worker_registry.build_worker(worker_env))

    return (workers, worker_envs)


async def main():
    # Disable torch cuda check which produce lots of log.
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    deploy_config = deploy_env()
    deploy_mode = deploy_config.deploy_mode
    # Initialize logging based on deploy mode.
    if deploy_mode == DeployStage.PROD:
        logging_mode = (logging.LoggingMode.PRODUCTION
                        if not _WORKER_LOGGING_MODE.is_set() else
                        _WORKER_LOGGING_MODE.parse_value())
        logging_dir = osp.join(get_local_home_dir(), 'logging')
        fsutil.makedirs(logging_dir, exist_ok=True)
        logging.initialize_logging(
            mode=logging_mode,
            logging_dir=logging_dir,
            log_to_file=False,  # disable multiprocess file logging
        )
    else:
        logging.initialize_logging(
            _WORKER_LOGGING_MODE.parse_value(),
            log_to_file=False,  # disable multiprocess file logging
        )
    log_app_startup()
    _start_app_metrics_server()

    # Temporal workers may not get snowflake credentials from the environment
    Connector.disable_interactive_input()

    # TODO(siyang): Consider supporting worker name prefix or wild card.
    workers = _WORKERS.parse_value()
    worker_names = workers.split(',')

    # Initialize tracing
    monitoring_settings = MonitoringSettings.parse_from_env()
    monitoring_settings.init_tracing = _WORKER_ENABLE_TRACING.parse_value()
    if monitoring_settings.init_tracing:
        tracing.init_tracing(workers, monitoring_settings)

    (workers, worker_envs) = await build_workers(deploy_config, worker_names)

    for worker_env in worker_envs:
        log.info('Running worker env:\n%s\n', pformat(worker_env.as_dict()))

    log.info(f"Parent process PID: {os.getpid()}")

    # Now that worker init is done (Temporal SDK pre-fetches build_id via
    # synchronous file I/O, which trips the guards on cold caches), install the
    # event-loop guards. From here on any loop block is genuine and worth
    # flagging.
    install_loop_guards()

    await asyncio.gather(*[worker.run() for worker in workers])
    for worker_env in worker_envs:
        worker_env.shutdown()


if __name__ == "__main__":
    init_temporal_runtime(_WORKERS.parse_value().replace(',', '_'))
    asyncio.run(main())
