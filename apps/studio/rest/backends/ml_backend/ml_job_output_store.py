# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional, Union

from io_utils import fsutil, json, logging
from kumo.home import HomeDirManager
from rest.backends.ml_backend.data_util import get_job_base_dir, get_job_dir
from rest.config.batch_prediction_job import BatchPredictionJobOutput
from rest.config.ml_backend import MaterializationTaskInfo
from rest.config.training_job import TrainingJobOutput
from statedb.config.common import (
    BatchPredictionJobID,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    JobID,
    TrainingJobID,
)
from statedb.db.ml_jobs import (
    GeneratePredTableJobOutput,
    GenerateTrainTableJobOutput,
    MLJobService,
)
from statedb.db.runtime import StateDB

log = logging.getLogger(__name__)

TRAINING_JOB_OUTPUT_FILENAME = 'training_job_output.json'
TRAINING_JOB_TEST_EXPLAIN_FILENAME = 'test_explain.json.gz'


class MLJobOutputStore:
    """Store/access state for a training/prediction workflow.

    This is a plain in-process object (no longer an AutoRPC service). It has
    two ways of reaching the ML-job StateDB:

    * **Bound** -- ``_configure_ml_job_service`` wires a long-lived
      ``MLJobService`` (e.g. the one the REST/public/worker apps already opened
      over their StateDB). All StateDB writes reuse it.
    * **Self-contained** (``self_contained=True``) -- opens one StateDB lazily
      on the first write and reuses it (and its connection pool) for every
      subsequent write; ``aclose`` closes it. Construction does no I/O, so a
      pure-ingest worker that never writes never touches StateDB. The reused
      pool is bound to the event loop of the first write, so **every
      self-contained write must run on a single event loop** -- this mode is
      driven only by the pool ``SnapshotActivities`` store, whose writes all
      run on the worker's event loop. It exists so workers without a long-lived
      in-process StateDB can still publish job state without an AutoRPC client.
    """
    def __init__(
        self,
        jobs_dir: Optional[str] = None,
        *,
        self_contained: bool = False,
    ):
        """A class for storing and accessing state related to a
        training/prediction workflow.

        Args:
          - jobs_dir: optional override for the base directory holding
            per-job immutable artifacts. Production callers leave this None so
            the dir is resolved LIVE from ``deploy_env()`` on each access via
            ``get_job_base_dir()`` -- a long-lived store (e.g. the REST/worker
            in-process MLJobBackend) must not pin the path captured at
            construction, or a later deploy-env switch (as in the multi-config
            integ harness) makes blob reads/writes target a stale root. Tests
            may pin an explicit path.
          - self_contained: when True, the store opens one StateDB lazily on
            the first write and reuses it for every subsequent write (closed by
            ``aclose``) instead of requiring a configured service. The StateDB
            path is resolved from ``DB_STORAGE_ROOT``. All writes must run on
            a single event loop (see the class docstring).
        """
        self._jobs_dir_override = jobs_dir
        self._self_contained = self_contained
        self._ml_job_service: Optional[MLJobService] = None
        # Self-contained mode only: the StateDB we opened and must close, plus
        # a lock serializing the async first-open so two concurrent first
        # writers can't each open (and leak) a pool. Bound mode leaves
        # ``_owned_state_db`` None, so ``aclose`` never touches the externally
        # configured service.
        self._owned_state_db: Optional[StateDB] = None
        self._init_lock = asyncio.Lock()

    @property
    def _jobs_dir(self) -> str:
        """Base dir for per-job artifacts, resolved live unless pinned."""
        if self._jobs_dir_override is not None:
            return self._jobs_dir_override
        return get_job_base_dir()

    def _configure_ml_job_service(self,
                                  ml_job_service: MLJobService) -> None:
        self._ml_job_service = ml_job_service

    def _test_explain_path(self, id: TrainingJobID) -> str:
        test_explain_dir = os.path.join(self._jobs_dir, 'test_explains')
        fsutil.makedirs(
            fsutil.abspath(fsutil.expanduser(test_explain_dir)),
            exist_ok=True)
        return os.path.join(test_explain_dir, id)

    @asynccontextmanager
    async def _ml_job_service_scope(self) -> AsyncIterator[MLJobService]:
        """Yield an ``MLJobService`` for a single StateDB write.

        Bound mode reuses the configured service. Self-contained mode opens one
        StateDB lazily on the first write and reuses it for every subsequent
        write; ``aclose`` closes it. The reused pool is bound to the loop of
        that first write, so all self-contained writes must run on one loop.
        """
        if self._ml_job_service is not None:
            yield self._ml_job_service
            return
        if not self._self_contained:
            raise RuntimeError(
                "MLJobOutputStore requires MLJobService to be configured "
                "before updating ML job metadata.")
        # Serialize the async first-open: two concurrent first writers would
        # otherwise both see ``None`` across the ``await`` and each open a
        # StateDB, leaking one pool. Double-check inside the lock.
        async with self._init_lock:
            if self._ml_job_service is None:
                # Lazy imports keep this module light and avoid import cycles.
                from rest.settings import get_db_storage_root
                from service_framework.deploy_env_config import deploy_env
                from statedb.db import open_state_db_from_deploy_config
                self._owned_state_db = await open_state_db_from_deploy_config(
                    deploy_env(),
                    state_root=get_db_storage_root(),
                )
                self._ml_job_service = MLJobService(self._owned_state_db)
        yield self._ml_job_service

    async def aclose(self) -> None:
        """Close the StateDB opened by self-contained mode, if any.

        No-op in bound mode (the service is owned by whoever configured it) and
        when nothing was ever opened. Idempotent, and leaves the store reusable
        -- a later write re-opens lazily.
        """
        async with self._init_lock:
            if self._owned_state_db is not None:
                await self._owned_state_db.close()
                self._owned_state_db = None
                self._ml_job_service = None

    async def save_train_table_graph_version(self, id: GenerateTrainTableJobID,
                                             graph_data_version: str):
        async with self._ml_job_service_scope() as service:
            await service.save_train_table_graph_version(
                id, graph_data_version)

    async def save_training_job_graph_version(self, id: TrainingJobID,
                                              graph_data_version: str) -> None:
        async with self._ml_job_service_scope() as service:
            await service.save_training_job_graph_version(
                id, graph_data_version)

    async def save_pred_job_graph_version(self, id: BatchPredictionJobID,
                                          graph_data_version: str) -> None:
        async with self._ml_job_service_scope() as service:
            await service.save_pred_job_graph_version(id, graph_data_version)

    async def save_training_job_output(self, id: TrainingJobID,
                                       output: TrainingJobOutput) -> None:
        if output.results:
            # Load "test_explain" if the file path is local
            if (output.results._test_explain_file_path and fsutil.is_local(
                    output.results._test_explain_file_path)):
                await output.results.load_test_explain.coroutine()

            if output.results.test_explain:
                # We need to store test_explain in a separate remote location
                # given its large size.
                await output.results.save_test_explain_to_remote.coroutine(
                    self._test_explain_path(id))

        async with self._ml_job_service_scope() as service:
            await service.save_training_job_output(id, output)

    async def save_batch_prediction_job_output(
            self, id: BatchPredictionJobID,
            output: BatchPredictionJobOutput) -> None:
        log.info(f"Saving batch prediction job output for job {id}"
                 f"output: {output}")
        async with self._ml_job_service_scope() as service:
            await service.save_batch_prediction_job_output(id, output)

    async def save_generate_train_table_job_output(
            self, id: GenerateTrainTableJobID,
            output: GenerateTrainTableJobOutput):
        log.info(f"Saving generate-train-table job output for job {id}"
                 f"output: {output}")
        async with self._ml_job_service_scope() as service:
            await service.save_generate_train_table_job_output(id, output)

    async def save_pred_table_graph_version(self,
                                            id: GeneratePredictionTableJobID,
                                            graph_data_version: str):
        async with self._ml_job_service_scope() as service:
            await service.save_pred_table_graph_version(
                id, graph_data_version)

    async def save_generate_pred_table_job_output(
            self, id: GeneratePredictionTableJobID,
            output: GeneratePredTableJobOutput):
        log.info(f"Saving generate-train-table job output for job {id}"
                 f"output: {output}")
        async with self._ml_job_service_scope() as service:
            await service.save_generate_pred_table_job_output(id, output)


# Materialization-info blob helpers ###########################################
#
# These read/write the per-job materialization-info blob directly and never
# touch the ML-job StateDB, so they are plain module-level functions rather than
# ``MLJobOutputStore`` methods -- only the ``save_*`` methods that hit the
# StateDB need an ``MLJobService`` (and hence the store). ``jobs_dir`` defaults
# to the live ``get_job_base_dir()`` (resolved from ``deploy_env()``); pass an
# explicit value only to pin the base dir (e.g. in tests).


def _materialization_info_path(id: JobID,
                               jobs_dir: Optional[str] = None) -> str:
    base = jobs_dir if jobs_dir is not None else get_job_base_dir()
    return HomeDirManager(get_job_dir(base, id)).materialization_task_info_file


def save_materialization_info(
    info: MaterializationTaskInfo,
    jobs_dir: Optional[str] = None,
) -> None:
    """Save basic statistics and debug information about the materialization
    stage of a Training or Batch Prediction Job.
    """
    dest_path = _materialization_info_path(info.id, jobs_dir)
    fsutil.makedirs(os.path.dirname(dest_path), exist_ok=True)
    json.save_json(info, dest_path, compressed=True)


def get_materialization_info(
    id: Union[TrainingJobID, BatchPredictionJobID],
    jobs_dir: Optional[str] = None,
) -> Optional[MaterializationTaskInfo]:
    """Returns what was saved by `save_materialization_info`. Returns None
    if no materialization info is available. This may indicate that the
    materialization stage is not yet complete.
    """
    dest_path = _materialization_info_path(id, jobs_dir)
    return json.load_json(dest_path, MaterializationTaskInfo,
                          compressed=True, default=None,
                          raise_on_missing=False)


def get_many_materialization_info(
    ids: Union[List[TrainingJobID], List[BatchPredictionJobID]],
    jobs_dir: Optional[str] = None,
) -> List[Optional[MaterializationTaskInfo]]:
    """Returns what was saved by `save_materialization_info` for many jobs.
    Returns None for entries with no materialization info available, which may
    indicate that the materialization stage is not yet complete.
    """
    dest_paths = [_materialization_info_path(id, jobs_dir) for id in ids]
    return json.load_many_json(
        dest_paths,
        MaterializationTaskInfo,  # type: ignore
        compressed=True,
    )  # type: ignore
