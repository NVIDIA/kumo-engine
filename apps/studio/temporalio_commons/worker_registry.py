# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-few-public-methods
import asyncio
import copy
import dataclasses
import multiprocessing
import os
from concurrent.futures import (
    Executor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from dataclasses import field
from datetime import datetime, timedelta, timezone
from threading import current_thread
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    TypeVar,
)

from kumo_core.version import get_kumo_version
from pydantic import model_validator
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.worker import (
    SharedStateManager,
    UnsandboxedWorkflowRunner,
    Worker,
)
from typing_extensions import Concatenate, ParamSpec

from async_utils.instrument import measure_blocking_code
from io_utils import kumo_dataclass, logging
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.deploy_stage import DeployStage
from service_framework.env_var import EnvVar
from temporalio_commons.process_pool_utils import (
    PROCESS_POOL_COMPATIBLE_ATTR,
    HealingProcessPoolExecutor,
    process_pool_activity_defn,
)
from temporalio_commons.temporal_client import TemporalClient
from temporalio_commons.worker_interceptors import (
    KumoActivityInterceptor,
    KumoWorkflowInterceptor,
    last_sync_activity_completion,
    last_sync_activity_heartbeat,
)

log = logging.getLogger(__name__)

T = TypeVar('T')


@kumo_dataclass
class WorkerOptions:
    """
    Options for configuring and running a Temporal Worker entity which hosts
    a set of workflows and/or activities over a task queue.
    """
    task_queue: str = ''
    max_worker_threads: Optional[int] = None
    max_worker_processes: Optional[int] = None
    multi_executors: Optional[bool] = None
    env_variables: List[EnvVar] = field(default_factory=list)

    # v1 @root_validator (post-validation) -> @model_validator(mode='after').
    @model_validator(mode='after')
    def _check_invariants(self) -> 'WorkerOptions':
        worker_threads = self.max_worker_threads
        worker_processes = self.max_worker_processes
        multi_executors = self.multi_executors
        if multi_executors:
            # Check both are configured in multi_executors mode
            if not worker_threads or not worker_processes:
                raise ValueError(
                    'Both max_worker_threads and max_worker_processes need '
                    'to be set when multi_executors is set to True.')
        else:
            # Check only 1 is configured in non-multi_executors mode
            if worker_threads and worker_processes:
                raise ValueError(
                    'WorkerOptions can only specify both max_worker_threads '
                    'and max_worker_processes when multi_executors flag is '
                    'set to True. Otherwise, only one can be specified.')
        return self


# Debug info about current status of a threadpool or process pool executor
@kumo_dataclass
class ExecutorInfo:
    max_workers: int
    pending_work_items: int

    # Number of idle thread or process workers.
    num_idle_workers: int
    # Thread pool: native id of spawned worker threads
    # Process pool: pid of spawned child worker processes
    worker_ids: Set[int] = field(default_factory=set)

    @property
    def busy(self) -> bool:
        return self.pending_work_items >= self.max_workers

    @staticmethod
    def from_threadpool(executor: ThreadPoolExecutor) -> 'ExecutorInfo':
        num_threads = len(executor._threads)
        num_idle_threads = executor._idle_semaphore._value
        items_waiting_in_queue = executor._work_queue.qsize()
        pending_work_items = (num_threads -
                              num_idle_threads) + items_waiting_in_queue
        return ExecutorInfo(
            max_workers=executor._max_workers,
            pending_work_items=pending_work_items,
            num_idle_workers=num_idle_threads,
            worker_ids={t.native_id
                        for t in executor._threads if t.native_id})

    @staticmethod
    def from_processpool(executor: ProcessPoolExecutor) -> 'ExecutorInfo':
        return ExecutorInfo(
            max_workers=executor._max_workers,  # type: ignore
            pending_work_items=len(executor._pending_work_items),
            num_idle_workers=executor._idle_worker_semaphore._value,
            worker_ids=set(pid for pid in executor._processes))


# Worker info to report (for debugging or health check purpose)
@kumo_dataclass
class WorkerInfo:
    worker_name: str
    instance_id: str
    start_time: datetime
    workflows: List[str]
    activities: List[str]
    kumo_version: Optional[str] = None

    # Information about workflow and activity task executors.
    #
    # Asyncio default executor is used for polling tasks from queues as well as
    # executing async activities.
    event_loop_default_executor_info: Optional[ExecutorInfo] = None
    # The executor that executes all sync activties on this worker.
    activity_executor_info: Optional[ExecutorInfo] = None
    # The workflow task executor that executes workflow tasks, i.e., workflow
    # code/logic and payload encoding/decoding.
    wf_task_executor_info: Optional[ExecutorInfo] = None
    # Thread native id that is currently executing WorkerInfoWorkflow task
    current_wf_task_executor_id: Optional[int] = None

    last_activity_heartbeat: Optional[datetime] = None
    last_activity_completion: Optional[datetime] = None

    @property
    def is_ec2(self) -> bool:
        return self.instance_id.startswith('i-')

    def is_stuck(self, unresponsive_threshold: timedelta) -> bool:
        if len(self.activities) == 0 or self.activity_executor_info is None:
            # This worker does not execute any synchronous activities.
            return False

        if self.activity_executor_info.busy:
            last_alive = self.start_time
            if self.last_activity_completion:
                last_alive = max(last_alive, self.last_activity_completion)
            if self.last_activity_heartbeat:
                last_alive = max(last_alive, self.last_activity_heartbeat)

            since_last_alive = datetime.now(timezone.utc) - last_alive
            # If the activity worker has no available slot, i.e.,
            # activity_executor_info.busy, and there is no activity completion
            # or activity heartbeat more than the `unresponsive_threshold`
            return since_last_alive >= unresponsive_threshold

        return False


class WorkerInfoActivity:
    def __init__(self, worker_info: WorkerInfo, *,
                 activity_executor: Optional[Executor],
                 wf_task_executor: ThreadPoolExecutor) -> None:
        self._worker_info = worker_info
        self._activity_executor = activity_executor
        self._wf_task_executor = wf_task_executor

    @activity.defn
    async def get_worker_info(self) -> WorkerInfo:
        worker_info = copy.deepcopy(self._worker_info)

        loop = asyncio.get_running_loop()
        if isinstance(loop, asyncio.BaseEventLoop):
            executor = loop._default_executor  # type: ignore
            if isinstance(executor, ThreadPoolExecutor):
                worker_info.event_loop_default_executor_info = (
                    ExecutorInfo.from_threadpool(executor))

        executor_info = None
        if self._activity_executor:
            if isinstance(self._activity_executor, ThreadPoolExecutor):
                executor_info = (ExecutorInfo.from_threadpool(
                    self._activity_executor))
            else:
                assert isinstance(self._activity_executor,
                                  HealingProcessPoolExecutor)
                executor_info = ExecutorInfo.from_processpool(
                    self._activity_executor._pool)
        worker_info.activity_executor_info = executor_info

        worker_info.wf_task_executor_info = \
            ExecutorInfo.from_threadpool(self._wf_task_executor)

        task_queue = activity.info().task_queue
        worker_info.last_activity_heartbeat = (
            last_sync_activity_heartbeat.get(task_queue))
        worker_info.last_activity_completion = (
            last_sync_activity_completion.get(task_queue))

        return worker_info


@workflow.defn
class WorkerInfoWorkflow:
    @workflow.run
    async def run(self) -> WorkerInfo:
        worker_info = await workflow.execute_local_activity_method(
            WorkerInfoActivity.get_worker_info,
            schedule_to_close_timeout=timedelta(seconds=10))
        worker_info.current_wf_task_executor_id = current_thread().native_id
        log.debug('Current worker info: %s', worker_info)

        return worker_info


# pylint: disable=too-few-public-methods
@workflow.defn
class RunSingleActivityWorkflow:
    """A private, generic single-activity workflow that executes a rpc method
    (as activity func) by the rpc method name.
    """
    @workflow.run
    async def run(
        self,
        activity_type: str,
        activity_inputs: List,
    ):
        return await workflow.execute_activity(
            activity=activity_type,
            args=activity_inputs,
            # TODO(siyang): use the original timeout config from wf history
            schedule_to_close_timeout=timedelta(hours=12),
            # This wrapper also hosts mutating RPC activities, so replaying an
            # activity here is not generally safe. Request-level callers may
            # retry by starting a new workflow, which also resolves and opens
            # a fresh connector rather than reusing a failed native session.
            retry_policy=RetryPolicy(maximum_attempts=1),
        )


# Proxy of ProcessPoolExecutor that resets the underlying process pool when
# BrokenProcessPool exception happens which indicates the old process pool
# has been broken and unusable.
class WorkerEnv:
    def __init__(
        self,
        worker_name: str,
        options: WorkerOptions,
        deploy_config: DeployEnvConfig,
        *,
        # Below are options to override worker setup for testing purpose:
        env_var_test_override: Dict[str, Any] | None = None,
        activities_override: Sequence[Callable] = [],
        workflows_override: Sequence[Type] = [],
    ) -> None:
        self._options = options
        self._deploy_config = deploy_config
        self._worker_name = worker_name
        # Sync activity executor, could be threadpool or process pool.
        self._thread_pool_executor = None
        self._process_pool_executor = None
        # Same as the default executor created by Temporal SDK, except that
        # it's created here explicitly so that we can export its runtime info.
        # Sized to 500 to match the SDK's default (since v1.11.0) — ensures
        # deadlocked threads can't starve all workflow task slots.
        self._wf_task_executor = ThreadPoolExecutor(
            max_workers=500,
            thread_name_prefix="temporal_workflow_",
        )
        self._env_var_dict: Dict[str, EnvVar] = {
            var.name: var
            for var in options.env_variables
        }

        if deploy_config.deploy_mode != DeployStage.TEST:
            # overrides are only permitted in TEST env.
            assert not env_var_test_override
            assert not activities_override
            assert not workflows_override
        self._env_var_test_override = env_var_test_override or {}
        self._activities_override = {
            activity._Definition.must_from_callable(a).name or a.__name__: a
            for a in activities_override
        }
        self._workflows_override = {
            workflow._Definition.must_from_class(c).name or c.__name__: c
            for c in workflows_override
        }
        self._maybe_init_thread_pool_executor()

    def _maybe_init_thread_pool_executor(self):
        if self._options.max_worker_threads:
            self._thread_pool_executor = ThreadPoolExecutor(
                self._options.max_worker_threads, f'{self._worker_name}-thread')

    def _maybe_init_process_pool_executor(self):
        if (self._process_pool_executor is None
                and self._options.max_worker_processes):
            env_vars = {
                k: self.get_variable_optional(k)
                for k in self._env_var_dict.keys()
            }
            env_vars.update(self._env_var_test_override)
            if (self._deploy_config.deploy_mode == DeployStage.TEST
                    and "DEPLOY_ENV_CONFIG_JSON" in os.environ):
                env_vars.setdefault(
                    "DEPLOY_ENV_CONFIG_JSON",
                    os.environ["DEPLOY_ENV_CONFIG_JSON"],
                )
            self._process_pool_executor = HealingProcessPoolExecutor(
                self._options.max_worker_processes,
                deploy_stage=self._deploy_config.deploy_mode, env_vars=env_vars)
            if self._worker_name in {'data-backend-worker'}:
                self._process_pool_executor.warm_up()

    def get_variable(self, env_var_name: str) -> Any:
        value = self.get_variable_optional(env_var_name)
        assert value is not None
        return value

    def get_variable_optional(self, env_var_name: str) -> Optional[Any]:
        return self._env_var_test_override.get(
            env_var_name, self._env_var_dict[env_var_name].parse_value())

    def as_dict(self) -> Dict:
        options_dict = dataclasses.asdict(self._options)
        del options_dict['env_variables']
        return {
            'worker': self._worker_name,
            'options': options_dict,
            'env_variables': {
                v.name: self.get_variable_optional(v.name)
                for v in self._options.env_variables
            },
            'deploy_config': self._deploy_config,
        }

    @property
    def options(self) -> WorkerOptions:
        return self._options

    @property
    def name(self) -> str:
        return self._worker_name

    @property
    def task_queue(self) -> str:
        return self._deploy_config.task_queue(self._options.task_queue)

    @property
    def deploy_config(self) -> DeployEnvConfig:
        return self._deploy_config

    @property
    def temporal_client(self) -> TemporalClient:
        return TemporalClient(self._deploy_config)

    async def build_worker(
        self,
        activities: List[Callable],
        workflows: List[Type],
        alt_task_queue: Optional[str] = None,
        worker_name_suffix: str = '',
        max_concurrent_activities: Optional[int] = None,
        use_process_executor: Optional[bool] = None,
        graceful_shutdown_timeout: Optional[timedelta] = None,
    ) -> Worker:
        log.info(f"Building worker with {len(activities)} activities and "
                 f"{len(workflows)} workflows.")
        if ':' in self.name or ':' in worker_name_suffix:
            raise ValueError(
                'Worker name cannot contain ":" as reserved separator.')
        if (use_process_executor is not None
                and not self.options.multi_executors):
            raise ValueError(
                'Cannot specify use_process_executor if multi_executors flag '
                'is not set or set to false.')

        # Apply testing override of activites/workflows
        for i, act in enumerate(activities):
            act_name = activity._Definition.must_from_callable(act).name
            if act_name in self._activities_override:
                activities[i] = self._activities_override[act_name]
                log.warning('Override activity %s with %s', act_name,
                            activities[i].__qualname__)
        for i, wf in enumerate(workflows):
            wf_name = workflow._Definition.must_from_class(wf).name
            if wf_name in self._workflows_override:
                workflows[i] = self._workflows_override[wf_name]
                log.warning('Override workflow %s with %s', wf_name,
                            workflows[i].__qualname__)

        task_queue = self.deploy_config.task_queue(
            alt_task_queue) if alt_task_queue else self.task_queue
        worker_id = self.name + worker_name_suffix + '-' + \
            self._deploy_config.instance_id

        shared_state_mgr = None
        activity_executor = self.activity_executor(use_process_executor)
        if (activity_executor
                and activity_executor is self._process_pool_executor):
            shared_state_mgr = SharedStateManager.create_from_multiprocessing(
                multiprocessing.Manager())

        activity_names = [
            activity._Definition.must_from_callable(a).name for a in activities
        ]

        worker_info = WorkerInfo(
            worker_name=self.name + worker_name_suffix,
            instance_id=self._deploy_config.instance_id,
            start_time=datetime.now(timezone.utc),
            workflows=[str(wf) for wf in workflows],
            activities=[a for a in activity_names if a is not None],
            kumo_version=get_kumo_version(),
        )

        log.debug('Building worker with info: %s', worker_info)
        worker_info_activity = WorkerInfoActivity(
            worker_info,
            activity_executor=activity_executor,
            wf_task_executor=self._wf_task_executor,
        )

        # Enable event loop blocking detection for ALL async activities.
        activity_list = []
        for activity_fn in activities:
            activity_def = activity._Definition.must_from_callable(activity_fn)
            if activity_def.is_async:
                activity_fn = measure_blocking_code(activity_def.fn,
                                                    func_name=activity_def.name)
            elif (isinstance(activity_executor, ProcessPoolExecutor)
                  and not getattr(activity_def.fn, PROCESS_POOL_COMPATIBLE_ATTR,
                                  False)):
                raise ValueError(
                    f'Acitivty fn {activity_def.name} must be defined by '
                    f'@{process_pool_activity_defn.__name__}.')

            activity_list.append(activity_fn)

        worker_kwargs: Dict[str, Any] = dict(
            task_queue=task_queue,
            activities=activity_list + [worker_info_activity.get_worker_info],
            workflows=workflows +
            [WorkerInfoWorkflow, RunSingleActivityWorkflow],
            interceptors=[KumoActivityInterceptor(),
                          KumoWorkflowInterceptor()],
            identity=worker_id,
            activity_executor=activity_executor,
            workflow_task_executor=self._wf_task_executor,
            max_concurrent_activities=max_concurrent_activities or 100,
            shared_state_manager=shared_state_mgr,
            # Reference: https://docs.temporal.io/activities#throttling
            max_heartbeat_throttle_interval=timedelta(seconds=20),
            default_heartbeat_throttle_interval=timedelta(seconds=10),
            workflow_runner=UnsandboxedWorkflowRunner(),
        )
        if graceful_shutdown_timeout is not None:
            worker_kwargs['graceful_shutdown_timeout'] = (
                graceful_shutdown_timeout)
        return Worker(await self.client(), **worker_kwargs)

    async def client(self) -> Client:
        return await self.temporal_client.get()

    def activity_executor(
        self,
        use_process_executor: Optional[bool] = None,
    ) -> Optional[Executor]:
        if use_process_executor is None or use_process_executor:
            self._maybe_init_process_pool_executor()
        if use_process_executor is None:
            return self._thread_pool_executor or self._process_pool_executor
        else:
            return self._process_pool_executor if use_process_executor \
                else self._thread_pool_executor

    def shutdown(self, *, wait: bool = True,
                 cancel_futures: bool = False) -> None:
        if self._thread_pool_executor:
            self._thread_pool_executor.shutdown(wait,
                                                cancel_futures=cancel_futures)
        if self._process_pool_executor:
            self._process_pool_executor.shutdown(wait,
                                                 cancel_futures=cancel_futures)
        self._wf_task_executor.shutdown(wait, cancel_futures=cancel_futures)


# Function signature to build and run a Temporal Worker entity from the worker
# options. Use @register_worker to register the named worker in a global
# registry of workers. For example:
#
# @register_worker('my-worker-name')
# async def run_worker(worker_env: WorkerEnv) -> Worker:
#   ...
#   return worker_env.build_worker(...)
#
# Multiple workers can be "bundled" under the a single worker env:
#
# @register_bundled_workers('my-bundled-worker-name')
# async def run_multi_workers(worker_env: WorkerEnv) -> List[Worker]:
#   worker1 = worker_env.build_worker(...)
#   worker2 = worker_env.build_worker(...)
#   return [worker1, worker2]

P = ParamSpec('P')
RunWorkerFunc = Callable[Concatenate[WorkerEnv, P], Coroutine[None, None,
                                                              Worker]]
RunMultiWorkerFunc = Callable[Concatenate[WorkerEnv, P],
                              Coroutine[None, None, List[Worker]]]


class WorkerRegistry:
    """A global registry of worker_name -> RunWorkerFunc for all workers."""
    def __init__(self) -> None:
        self._workers: Dict[str, RunWorkerFunc] = {}
        self._bundle_workers: Dict[str, RunMultiWorkerFunc] = {}

    def has_worker(self, worker_name: str) -> bool:
        """Returns whether the worker_name has been registered."""
        return (worker_name in self._workers
                or worker_name in self._bundle_workers)

    def all_workers(self) -> Set[str]:
        return set.union(set(self._workers.keys()),
                         set(self._bundle_workers.keys()))

    def register_worker(self, worker_name,
                        run_worker_func: RunWorkerFunc) -> None:
        func = self._workers.setdefault(worker_name, run_worker_func)
        if func is not run_worker_func:
            raise RuntimeError(
                f'Worker name {worker_name} is being registered by multiple '
                f'functions: {func} and {run_worker_func}')

    def register_multi_worker(self, worker_name,
                              run_worker_func: RunMultiWorkerFunc) -> None:
        func = self._bundle_workers.setdefault(worker_name, run_worker_func)
        if func is not run_worker_func:
            raise RuntimeError(
                f'Worker name {worker_name} is being registered at multiple '
                f'places: {func} and {run_worker_func}')

    async def build_worker(self, worker_env: WorkerEnv) -> List[Worker]:
        """Run the worker with provided options, return the coroutine."""
        worker_name = worker_env.name
        if worker_name in self._workers:
            return [await self._workers[worker_name](worker_env)]
        return await self._bundle_workers[worker_name](worker_env)


GLOBAL_WORKER_REGISTRY = WorkerRegistry()


# Decorator to register a RunWorkerFunc with the worker_name.
def register_worker(
        worker_name: str) -> Callable[[RunWorkerFunc], RunWorkerFunc]:
    def wrap(func: RunWorkerFunc):
        log.debug('Register %s in worker registry.', worker_name)
        GLOBAL_WORKER_REGISTRY.register_worker(worker_name, func)
        return func

    return wrap


# Decorator to register a RunMultiWorkerFunc to bundle multiple workers
# to run together under the same worker name and config.
def register_bundled_workers(
        worker_name: str) -> Callable[[RunMultiWorkerFunc], RunMultiWorkerFunc]:
    def wrap(func: RunMultiWorkerFunc):
        log.debug('Register %s in worker registry.', worker_name)
        GLOBAL_WORKER_REGISTRY.register_multi_worker(worker_name, func)
        return func

    return wrap
