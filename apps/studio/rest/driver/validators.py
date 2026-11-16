# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from abc import abstractmethod
from collections.abc import Set
from typing import Awaitable, Callable, Dict, List, Optional, Union

from kumo_core.types.schema import FPID, VersionedSourceTableID
from kumoapi.typing import Stype

from io_utils import fsutil, json, logging
from kumo.data import get_active_cache
from kumo.data.backend_cache import AsyncDataTaskStatus, TableDataTaskResult
from kumo.data.cache.table_version_cache import TableIngestionStatus
from kumo.home import HomeDirManager
from kumo.utils import batch_ops
from kumo.validation.validation import ValidationResponse
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.ml_backend.data_util import get_job_dir
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.table_v2 import TableDefinition
from rest.deps.authentication import (
    RequestContext,
    visible_project_ids_or_none,
)
from rest.driver.graph_logic import GraphLogic
from statedb.config.common import (
    BatchPredictionJobID,
    ErrWarnCTA,
    ErrWarnDetails,
    ErrWarnResponse,
    GraphID,
    PredictiveQueryID,
    ProjectID,
    TableID,
    TableIDv2,
    TableName,
    TrainingJobID,
)
from statedb.db.services import StateServices
from statedb.utils.errors import (
    BatchPredictionDoesNotExistError,
    TableDoesNotExistError,
    TrainingJobDoesNotExistError,
)

log = logging.getLogger(__name__)

# CTA urls
TABLE_SUMMARY = "/tables/tablesummary/"
TRAIN_QUERY = "/queries/trainquery/"
GRAPH_SUMMARY = "/graphs/summary/"
EDGE_HEALTH_URL = "/graphs/edgehealth/"

# Validator timeouts in secs, for different validator types.
TABLE_VALIDATOR_TIMEOUT = 10.0
GRAPH_VALIDATOR_TIMEOUT = 11.0
PQUERY_VALIDATOR_TIMEOUT = 12.0
BATCH_PREDICTION_VALIDATOR_TIMEOUT = 10.0

ValidatorFunc = Callable[[List], Awaitable[List[List[ErrWarnDetails]]]]


class Validator:
    """
    Base class for all types of validators ( table, graph, PQ, BP)
    """
    def __init__(self, data_backend_ctx: DataBackendContext, timeout: float,
                 job_base_dir: str,
                 state_services: Optional[StateServices] = None):
        self._data_backend_ctx = data_backend_ctx
        self._state_services = state_services
        self._timeout = timeout
        self._job_base_dir = job_base_dir

    async def run_validator_with_timeout(
            self, ids: Union[List[TableID], List[GraphID],
                             List[PredictiveQueryID],
                             List[BatchPredictionJobID] | List[TrainingJobID]],
            func: ValidatorFunc) -> List[List[ErrWarnDetails]]:
        """
        A helper co-routine that runs a given validator function for a given
        ID. If the validator generates an exception ( timeout ), then it is
        caught here and an empty result is returned. This helps other validator
        co-routines to complete and successfully return their responses,
        without getting affected from this errored/timedout validator.
        """
        errs = []
        try:
            errs = await asyncio.wait_for(
                func(ids),
                timeout=self._timeout,
            )
        except Exception as e:
            log.exception(e)
        return errs

    async def run_batch_validators(
        self, ids: Union[List[TableID], List[GraphID], List[PredictiveQueryID],
                         List[BatchPredictionJobID] | List[TrainingJobID]]
    ) -> List[List[ErrWarnDetails]]:
        """
        A co-routine that runs all validator functions for all the given IDs.
        """
        tasks = []
        for func in self.get_validator_funcs():
            tasks.append(self.run_validator_with_timeout(ids, func))

        # Concatenate together all warnings for each ID. Each element in the
        # result contains the list of warnings for the corresponding input ID.
        result = []
        for _ in range(len(ids)):
            result.append([])
        for task in tasks:
            try:
                errs_lists = await task
            except Exception:
                # Skip over failed validators, but log exception
                log.exception("Failure when running validator.")
                continue

            for i, errs in enumerate(errs_lists):
                result[i].extend(errs)

        return result

    @abstractmethod
    def check_ids_exist(self, ids: List) -> None:
        pass

    async def check_ids_exist_async(self, ids: List) -> None:
        self.check_ids_exist(ids)

    @abstractmethod
    def get_validator_funcs(self) -> List[ValidatorFunc]:
        pass

    async def get_warnings(
        self, ids: Union[List[TableID], List[GraphID], List[PredictiveQueryID],
                         List[BatchPredictionJobID] | List[TrainingJobID]]
    ) -> Dict[str, ErrWarnResponse]:
        """
        Run validators and fetch warnings
        """
        # Check if ids exist ( if not, it raises appropriate
        # exception internally )
        await self.check_ids_exist_async(ids)
        results = await self.run_batch_validators(ids)
        return self.gen_warn_response(ids, results)

    def gen_warn_response(
            self, ids: Union[List[TableID], List[GraphID],
                             List[PredictiveQueryID],
                             List[BatchPredictionJobID] | List[TrainingJobID]],
            err_warn: List[List[ErrWarnDetails]]) -> Dict[str, ErrWarnResponse]:
        """
        Generates a proper REST response of type Dict[str, ErrWarnResponse]
        from the outputs of validators ( containing List[ErrWarnDetails] )
        which contains a mix of both warnings and errors in the list.
        """
        resp: Dict[str, ErrWarnResponse] = {}
        # Generate Warn response
        for id, detail_list in zip(ids, err_warn):
            w = ErrWarnResponse(errors=[], warnings=[], info_items=[])

            # detail_list is a list of ErrWarnDetails
            for detail in detail_list:
                if detail.is_error is True:
                    w.errors.append(detail)
                elif detail.id is not None:
                    w.info_items.append(detail)
                else:
                    w.warnings.append(detail)
            resp[id] = w
        return resp


def _table_id_v2_fpid(table_id: TableID) -> Optional[FPID]:
    if not isinstance(table_id, TableIDv2):
        return None
    _, _, fpid = TableIDv2.parse(table_id)
    return fpid


class TableValidator(Validator):
    """
    TableValidator class that has co-routines to check table specific
    errors/warnings.
    """
    def __init__(self, data_backend_ctx: DataBackendContext, job_base_dir: str,
                 state_services: Optional[StateServices] = None, *,
                 project_id: ProjectID):

        super().__init__(data_backend_ctx, TABLE_VALIDATOR_TIMEOUT,
                         job_base_dir, state_services)
        self._project_id = project_id

    def _require_table_service(self):
        if self._state_services is None:
            raise RuntimeError("TableValidator requires StateServices.tables.")
        return self._state_services.tables

    def check_ids_exist(self, table_ids: List[TableID]) -> None:
        """
        Checks if table ids exist in db
        """
        raise RuntimeError("TableValidator requires StateServices.tables.")

    async def check_ids_exist_async(self, table_ids: List[TableID]) -> None:
        fpid_by_table_id = await self._table_fpid_by_id(table_ids)
        fpids = [
            fpid for table_id in table_ids
            if (fpid := fpid_by_table_id[table_id]) is not None
        ]
        entries = (await self._require_table_service().get_table_entries(
            self._project_id, fpids) if fpids else {})
        for table_id in table_ids:
            fpid = fpid_by_table_id[table_id]
            if fpid is None or fpid not in entries:
                raise TableDoesNotExistError(table_id)

    async def _table_fpid_by_id(
        self,
        table_ids: List[TableID],
    ) -> Dict[TableID, Optional[FPID]]:
        table_service = self._require_table_service()
        result: Dict[TableID, Optional[FPID]] = {}
        alias_by_table_id: Dict[TableID, TableName] = {}
        for table_id in table_ids:
            fpid = _table_id_v2_fpid(table_id)
            if fpid is not None:
                result[table_id] = fpid
                continue

            table_name: Optional[TableName] = None
            if isinstance(table_id, TableIDv2):
                _, table_name, _ = TableIDv2.parse(table_id)
            if table_name is None:
                result[table_id] = None
                continue
            alias_by_table_id[table_id] = table_name
            result[table_id] = None

        if alias_by_table_id:
            entries_by_alias = await table_service.get_table_entries_by_aliases(
                self._project_id, alias_by_table_id.values())
            for table_id, table_name in alias_by_table_id.items():
                entry = entries_by_alias.get(table_name)
                result[table_id] = entry.id if entry is not None else None

        return result

    def get_validator_funcs(self) -> List[ValidatorFunc]:
        """
        Gets all the validator functions
        """
        # Update this when a new validator routine is added.
        return [self.validate_tables]

    async def _get_many_table_definitions(
        self,
        table_ids: List[TableID],
    ) -> List[TableDefinition | Exception]:
        table_service = self._require_table_service()

        result: List[TableDefinition | Exception
                     | None] = [None for _ in table_ids]
        fpid_by_table_id = await self._table_fpid_by_id(table_ids)
        fpid_by_index: Dict[int, FPID] = {
            i: fpid
            for i, table_id in enumerate(table_ids)
            if (fpid := fpid_by_table_id[table_id]) is not None
        }

        if fpid_by_index:
            entries = await table_service.get_table_entries(
                self._project_id, list(fpid_by_index.values()))
            for i, fpid in fpid_by_index.items():
                entry = entries.get(fpid)
                result[i] = (entry.config if entry is not None else
                             TableDoesNotExistError(table_ids[i]))

        return [
            item if item is not None else TableDoesNotExistError(table_id)
            for table_id, item in zip(table_ids, result)
        ]

    async def validate_tables(
            self, table_ids: List[TableID]) -> List[List[ErrWarnDetails]]:
        """Performs validations on all tables, including based on col type
        selection and based on data computed by the table ingestion pipeline
        (method does both to reuse the args retrieved from the db).
        This includes validations on the exact stats, as well as validations
        produced by the pipeline itself. Also, returns error if table ingestion
        failed.
        """
        # Get the table from the metadata db. Ignore tables that failed to
        # fetch.
        tables_and_exc = await self._get_many_table_definitions(table_ids)
        tables_or_none = batch_ops.log_errors(tables_and_exc)
        tables, nones_1 = batch_ops.remove_nones(tables_or_none)
        source_ids = [table.source_table_id for table in tables]

        # Get the most recent version of each table (including failed ingested
        # versions), skipping over uningested tables.
        cache = get_active_cache()
        success_and_failed_versions: List[Optional[VersionedSourceTableID]] = (
            await cache.table_version_cache.get_many.coroutine(
                source_ids, TableIngestionStatus.COMPLETED))

        versions, nones_2 = batch_ops.remove_nones(success_and_failed_versions)

        # Full table refresh results in data backend cache are saved under the
        # table cache key.
        refresh_results: List[Optional[TableDataTaskResult]] = \
            await cache.table_refresh_result.get_many.coroutine(
                [(v.connector_name, t.cache_key(), v.version)
                 for t, v in zip(tables, versions)],
                TableDataTaskResult)

        # Run validations on the fetched data
        results: List[Optional[List[ErrWarnDetails]]] = []
        for refresh_result in refresh_results:
            result: List[ErrWarnDetails] = []
            if refresh_result is not None:
                refresh_result = self._check_refresh_result(refresh_result)
                result.extend(refresh_result)
            results.append(result)

        # For all tables that failed, return an empty list of warnings
        results = batch_ops.insert_nones(results, nones_2)
        results = batch_ops.insert_nones(results, nones_1)
        all_results = batch_ops.replace_nones(results, [])

        # For all tables, regardless of ingestion status, check col types
        for table_args, table_result in zip(tables, all_results):
            table_result.extend(self._check_col_types(table_args))
        return all_results

    def _check_refresh_result(
            self, refresh_result: TableDataTaskResult) -> List[ErrWarnDetails]:
        result: List[ErrWarnDetails] = []
        for info_item in refresh_result.validation.info_items:
            result.append(
                ErrWarnDetails(is_error=False, title=info_item.title,
                               id=info_item.id, description=info_item.message,
                               is_dismissible=True))
        for warning in refresh_result.validation.warnings:
            result.append(
                ErrWarnDetails(is_error=False, title=warning.title,
                               description=warning.message,
                               is_dismissible=True))
        for error in refresh_result.validation.errors:
            result.append(
                ErrWarnDetails(is_error=True, title=error.title,
                               description=error.message, is_dismissible=False))

        # If table ingestion failed, but the root cause was not provided, then
        # tell the customer to contact Kumo support. This should rarely happen,
        # as whenever table ingestion fails, a human-fixable error message is
        # supposed to be returned in validation.errors.
        if (refresh_result.status == AsyncDataTaskStatus.FAILED
                and len(refresh_result.validation.errors) == 0):
            result.append(
                ErrWarnDetails(
                    is_error=True, title="Fatal error while ingesting table",
                    description=(
                        "An unexpected error was encountered while "
                        "ingesting this table. If this issue persists "
                        "after refreshing the table, please contact your "
                        "point of contact at Kumo to understand and "
                        "fix the root cause."), is_dismissible=False))

        return result

    def _check_col_types(self,
                         table_args: TableDefinition) -> List[ErrWarnDetails]:
        result: List[ErrWarnDetails] = []
        for column in table_args.cols:
            if (column.dtype and column.dtype.is_float()
                    and column.stype == Stype.ID):
                result.append(
                    ErrWarnDetails(
                        is_error=False, title="Column type precision risk",
                        description=(
                            f"Column {column.name} has source data of type "
                            " float but is being used as an ID. This may lead"
                            " to errors when joining with an ID column with "
                            " integer data."), is_dismissible=False))

        return result


class GraphValidator(Validator):
    """
    GraphValidator class that has co-routines to check graph specific
    errors/warnings.
    """
    def __init__(self, data_backend_ctx: DataBackendContext, job_base_dir: str,
                 state_services: Optional[StateServices] = None, *,
                 project_id: ProjectID):

        super().__init__(data_backend_ctx, GRAPH_VALIDATOR_TIMEOUT,
                         job_base_dir, state_services)
        self._project_id = project_id

    def _require_state_services(self):
        if self._state_services is None:
            raise RuntimeError("GraphValidator requires StateServices.")
        return self._state_services

    def check_ids_exist(self, graph_ids: List[GraphID]) -> None:
        """
        Checks if graph ids exist in db
        """
        raise RuntimeError("GraphValidator requires StateServices.")

    async def check_ids_exist_async(self, graph_ids: List[GraphID]) -> None:
        await self._get_many_graph_inputs(graph_ids)

    async def _get_many_graph_inputs(self, graph_ids: List[GraphID]) -> List:
        self._require_state_services()
        graph_logic = GraphLogic(
            self._data_backend_ctx,
            state_services=self._state_services,
            project_id=self._project_id,
        )
        graph_inputs_or_exc = await graph_logic.get_many_data_compute_inputs(
            graph_ids)
        graph_inputs = []
        for graph_id, graph_input_or_exc in zip(graph_ids, graph_inputs_or_exc):
            if isinstance(graph_input_or_exc, Exception):
                raise graph_input_or_exc
            graph_inputs.append(graph_input_or_exc)
        return graph_inputs

    def get_validator_funcs(self) -> List[ValidatorFunc]:
        """
        Gets all the validator functions
        """
        # Update this when a new validator routine is added.
        return [
            self.graph_has_errors_warnings,
            self.table_has_errors_warnings,
        ]

    async def graph_has_errors_warnings(
            self, graph_ids: List[GraphID]) -> List[List[ErrWarnDetails]]:
        graph_inputs = await self._get_many_graph_inputs(graph_ids)
        graph_keys = [g.graph_key for g in graph_inputs]
        graph_id_dict = dict(zip(graph_keys, graph_ids))
        validations = await DataBackend.get_many_graph_validations.coroutine(
            graph_keys)

        return [
            _validation_to_warn_details(validation, graph_id_dict[graph_key])
            for graph_key, validation in validations.items()
        ]

    async def table_has_errors_warnings(
            self, graph_ids: List[GraphID]) -> List[List[ErrWarnDetails]]:
        """
        Graph Validator co-routine that handles,
            Error : Graph contains table with one or more errors
            Warning: Graph contains table with one or more warnings,
                but no errors
        """
        self._require_state_services()
        table_id_set: Set[TableID] = set()
        table_ids_by_graph: List[List[TableID]] = []
        graph_inputs = await self._get_many_graph_inputs(graph_ids)
        for graph_input in graph_inputs:
            table_ids = list(graph_input.tables.keys())
            table_ids_by_graph.append(table_ids)
            table_id_set.update(table_ids)
        table_ids_dedup = list(table_id_set)

        # Create TableValidator and validate the table_ids
        tv = TableValidator(self._data_backend_ctx, self._job_base_dir,
                            self._state_services, project_id=self._project_id)
        results = await tv.run_batch_validators(table_ids_dedup)
        warnings_by_table_id = dict(zip(table_ids_dedup, results))

        # Create warnings for each graph with tables that have warnings
        result = []
        for table_ids in table_ids_by_graph:
            graph_warnings = []
            for table_id in table_ids:
                num_errors = 0
                num_warnings = 0
                warnings = warnings_by_table_id[table_id]
                for warning in warnings:
                    if warning.is_error:
                        num_errors += 1
                    else:
                        num_warnings += 1
                if num_errors > 0:
                    table_name = table_id.table_name
                    graph_warnings.append(
                        ErrWarnDetails(
                            is_error=True, title="Graph contains errors",
                            description=(
                                f'Table {table_name} in this graph has '
                                'one or more errors.'), is_dismissible=False,
                            cta=ErrWarnCTA(name="View Errors",
                                           url=f'{TABLE_SUMMARY}{table_id}')))
            result.append(graph_warnings)

        assert len(result) == len(graph_ids)
        return result


class TrainingJobValidator(Validator):
    """
    Training Job Validator class that has co-routines to check Training Job
    specific errors/warnings.
    """
    def __init__(
        self,
        data_backend_ctx: DataBackendContext,
        job_base_dir: str,
        ml_backend: MLJobBackend,
        state_services: Optional[StateServices] = None,
    ):

        super().__init__(data_backend_ctx, PQUERY_VALIDATOR_TIMEOUT,
                         job_base_dir, state_services)

        self._ml_backend = ml_backend

    @classmethod
    def from_state_services(
        cls,
        state_services: StateServices,
        data_backend_ctx: DataBackendContext,
        job_base_dir: str,
        ml_backend: MLJobBackend,
    ) -> "TrainingJobValidator":
        return cls(
            data_backend_ctx,
            job_base_dir,
            ml_backend,
            state_services=state_services,
        )

    def _require_ml_job_service(self):
        if self._state_services is None:
            raise RuntimeError("TrainingJobValidator requires StateServices")
        return self._state_services.ml_jobs

    def check_ids_exist(self, tj_ids: List[TrainingJobID]) -> None:
        """
        Checks if training job ids exist in db
        """
        # TODO(frg100): Fill this in in a synchronous way
        return

    async def check_ids_exist_async(self, tj_ids: List[TrainingJobID]) -> None:
        entries = await self._require_ml_job_service().get_training_jobs(tj_ids)
        found_ids = {entry.id for entry in entries}
        for id in tj_ids:
            if id not in found_ids:
                raise TrainingJobDoesNotExistError(id)

    def get_validator_funcs(self) -> List[ValidatorFunc]:
        """
        Gets all the validator functions
        """
        # Update this when a new validator routine is added.
        return [
            self.check_graph_has_warnings, self.check_training_job_has_warnings
        ]

    async def check_graph_has_warnings(
            self, training_job_ids: List[TrainingJobID]
    ) -> List[List[ErrWarnDetails]]:
        # TODO(frg100): Fill this in to handle old graphs as well as snapshots
        return []

    async def check_training_job_has_warnings(
            self, training_job_ids: List[TrainingJobID]
    ) -> List[List[ErrWarnDetails]]:

        paths: List[Optional[str]] = []
        for job_id in training_job_ids:
            job_dir = get_job_dir(self._job_base_dir, job_id)
            path = HomeDirManager(job_dir).validation_path(job_id)
            paths.append(path)

        result: List[List[ErrWarnDetails]] = []
        if paths:
            result = await _get_warn_details(paths)

        return result


class BatchPredictionValidator(Validator):
    """
    Batch Prediction Validator class that has co-routines to check BP specific
    errors/warnings.
    """
    def __init__(self, data_backend_ctx: DataBackendContext, job_base_dir: str,
                 ml_job_backend: MLJobBackend, ctx: RequestContext,
                 state_services: Optional[StateServices] = None):

        super().__init__(data_backend_ctx, BATCH_PREDICTION_VALIDATOR_TIMEOUT,
                         job_base_dir, state_services)

        self._ctx = ctx

    @classmethod
    def from_state_services(
        cls,
        state_services: StateServices,
        data_backend_ctx: DataBackendContext,
        job_base_dir: str,
        ml_job_backend: MLJobBackend,
        ctx: RequestContext,
    ) -> "BatchPredictionValidator":
        return cls(
            data_backend_ctx,
            job_base_dir,
            ml_job_backend,
            ctx,
            state_services=state_services,
        )

    def _require_ml_job_service(self):
        if self._state_services is None:
            raise RuntimeError(
                "BatchPredictionValidator requires StateServices")
        return self._state_services.ml_jobs

    def check_ids_exist(self, bp_ids: List[BatchPredictionJobID]) -> None:
        """
        Checks if BP ids exist in db
        """
        raise RuntimeError(
            "BatchPredictionValidator requires StateServices for batch "
            "prediction existence checks.")

    async def check_ids_exist_async(self,
                                    bp_ids: List[BatchPredictionJobID]) -> None:
        rows = await self._require_ml_job_service().get_batch_prediction_rows(
            bp_ids,
            project_ids=visible_project_ids_or_none(
                self._ctx.user_authorization),
        )
        found_ids = {row.id for row in rows}
        for id in bp_ids:
            if id not in found_ids:
                raise BatchPredictionDoesNotExistError(id)

    def get_validator_funcs(self) -> List[ValidatorFunc]:
        """
        Gets all the validator functions
        """
        # Update this when a new validator routine is added.
        return [self.check_prediction_job_has_warnings]

    async def check_prediction_job_has_warnings(
            self,
            bp_ids: List[BatchPredictionJobID]) -> List[List[ErrWarnDetails]]:

        paths: List[Optional[str]] = []
        for job_id in bp_ids:
            job_dir = get_job_dir(self._job_base_dir, job_id)
            path = HomeDirManager(job_dir).validation_path(job_id)
            paths.append(path)

        result: List[List[ErrWarnDetails]] = []

        if paths:
            result = await _get_warn_details(paths)

        return result


def _validation_to_warn_details(
        validation: ValidationResponse,
        graph_id: Optional[GraphID] = None) -> List[ErrWarnDetails]:
    errs: List[ErrWarnDetails] = []
    for error in validation.errors:
        errs.append(
            ErrWarnDetails(
                is_error=True,
                title=error.title,
                description=error.message,
                is_dismissible=False,
            ))
    for warning in validation.warnings:
        warn_detail = ErrWarnDetails(
            is_error=False,
            title=warning.title,
            description=warning.message,
            is_dismissible=True,
        )
        errs.append(warn_detail)
    for info in validation.info_items:
        info_detail = ErrWarnDetails(
            is_error=False,
            title=info.title,
            description=info.message,
            is_dismissible=False,
            id=info.id,
        )
        errs.append(info_detail)
    return errs


async def _get_warn_details(
        paths: List[Optional[str]]) -> List[List[ErrWarnDetails]]:
    files = await fsutil.get_many_if_present.coroutine(paths)
    return [
        _validation_to_warn_details(
            json.fast_from_json(
                file,
                ValidationResponse,
            )) if file is not None else [] for file in files
    ]
