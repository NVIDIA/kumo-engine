# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional

from kumo_core.types import ColStatType
from kumo_core.types.schema import SourceTableID

from io_utils import logging
from kumo.data import get_active_cache
from kumo.data.backend_cache import PredStats
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.ml_backend.ml_job_backend import MLJobBackend
from rest.config.mlops import MLOpsQueryConfig, MLOpsStats, MLOpsStatType
from rest.deps.authentication import (
    RequestContext,
    visible_project_ids_or_none,
)
from rest.utils.driver_common import (
    PREDICTIONS_SOURCE_TABLE_NAME,
    TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME,
    prediction_connector_from_job_id,
)
from statedb.config.common import (
    BatchPredictionJobID,
    JobStatus,
    TrainingJobID,
)
from statedb.db.services import StateServices

logger = logging.getLogger(__name__)
# DB doesn't use REST JobStatus enum directly but uses lower strings from
# JobStatus values:
DB_DONE_STATUS = JobStatus.DONE.value.lower()


class MLOpsLogic:
    def __init__(self, ml_job_backend: MLJobBackend,
                 data_backend_ctx: DataBackendContext,
                 state_services: Optional[StateServices] = None):
        self._ml_job_backend = ml_job_backend
        self._data_backend_ctx = data_backend_ctx
        self._state_services = state_services

    @classmethod
    def from_state_services(
        cls,
        state_services: StateServices,
        ml_job_backend: MLJobBackend,
        data_backend_ctx: DataBackendContext,
    ) -> "MLOpsLogic":
        return cls(
            ml_job_backend,
            data_backend_ctx,
            state_services=state_services,
        )

    @property
    def _ml_job_service(self):
        if self._state_services is None:
            raise RuntimeError("MLOpsLogic requires StateServices")
        return self._state_services.ml_jobs

    async def get_mlops_stats(
        self,
        query: MLOpsQueryConfig,
        ctx: RequestContext,
    ) -> Optional[MLOpsStats]:
        # add dependencies on db stats to fetch via the ml backed call:
        if MLOpsStatType.COL_PSI_SCORES in query.stat_query.stat_request:
            query.stat_query.stat_request.append(MLOpsStatType.TRAINED_JOB_ID)

        stats = await self._ml_job_backend.get_mlops_jobs_and_db_stats(query)
        if stats:
            stats = await self._add_rest_db_stats(stats, query, ctx)
            stats = await self._add_train_stats(stats, query, ctx)
            stats = await self._add_prediction_stats(stats, query, ctx)
            stats = await self._add_materialization_stats(stats, query, ctx)
        return stats

    async def _add_train_stats(self, stats: MLOpsStats, query: MLOpsQueryConfig,
                               ctx: RequestContext) -> MLOpsStats:
        if not stats.train_stats:
            return stats
        for id in stats.train_stats.keys():
            # DB doesn't use JobStatus directly but uses lower strings
            # from JobStatus values:
            if stats.job_status_dict[id] == DB_DONE_STATUS:
                if (MLOpsStatType.GRAPH_VERSION
                        in query.stat_query.stat_request):
                    version_dict = {}
                    materialization_info = \
                        await self._ml_job_backend.get_materialization_info(
                            TrainingJobID(id))
                    if materialization_info:
                        for table_info in materialization_info.table_stats:
                            version_dict[table_info.table_id] = \
                                table_info.data_version
                    stats.train_stats[id][MLOpsStatType.GRAPH_VERSION] = \
                        version_dict
        return stats

    async def _add_prediction_stats(self, stats: MLOpsStats,
                                    query: MLOpsQueryConfig,
                                    ctx: RequestContext) -> MLOpsStats:
        stat_req = query.stat_query.stat_request

        requested_prediction_data_stats = (
            MLOpsStatType.PREDICTION_DISTRIBUTION in stat_req
            or MLOpsStatType.PREDICTION_COUNT in stat_req)
        requested_transformed_prediction_data_stats = (
            MLOpsStatType.TRANSFORMED_PREDICTION_DISTRIBUTION in stat_req
            or MLOpsStatType.TRANSFORMED_PREDICTION_COUNT in stat_req)
        if not stats.bp_stats or (
                not requested_prediction_data_stats
                and not requested_transformed_prediction_data_stats):
            return stats

        pred_tables = {}
        transformed_pred_tables = {}

        for id in stats.bp_stats.keys():
            if stats.job_status_dict[id] == DB_DONE_STATUS:
                pred_tables[id] = SourceTableID(
                    connector_name=prediction_connector_from_job_id(id),
                    source_table_name=PREDICTIONS_SOURCE_TABLE_NAME,
                )
                if requested_transformed_prediction_data_stats:
                    transformed_pred_tables[id] = SourceTableID(
                        connector_name=prediction_connector_from_job_id(id),
                        source_table_name=(
                            TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME),
                    )

        cache = get_active_cache()
        pred_stats = await cache.get_many_pred_stats.coroutine(pred_tables)
        transformed_pred_stats = await cache.get_many_pred_stats.coroutine(
            transformed_pred_tables)

        # Add the raw prediction stats
        stats = self._process_bp_stats(
            stats,
            {
                BatchPredictionJobID(k): v
                for k, v in pred_stats.items()
            },
            stat_req,
            count_stat=MLOpsStatType.PREDICTION_COUNT,
            distribution_stat=MLOpsStatType.PREDICTION_DISTRIBUTION,
        )

        # Add the transformed prediction stats
        stats = self._process_bp_stats(
            stats,
            {
                BatchPredictionJobID(k): v
                for k, v in transformed_pred_stats.items()
            },
            stat_req,
            count_stat=MLOpsStatType.TRANSFORMED_PREDICTION_COUNT,
            distribution_stat=MLOpsStatType.TRANSFORMED_PREDICTION_DISTRIBUTION,
        )

        return stats

    def _process_bp_stats(
        self,
        stats: MLOpsStats,
        pred_stats: Dict[BatchPredictionJobID, PredStats],
        stat_req: List[MLOpsStatType],
        count_stat: MLOpsStatType,
        distribution_stat: MLOpsStatType,
    ):
        assert count_stat in [
            MLOpsStatType.PREDICTION_COUNT,
            MLOpsStatType.TRANSFORMED_PREDICTION_COUNT
        ]
        assert distribution_stat in [
            MLOpsStatType.PREDICTION_DISTRIBUTION,
            MLOpsStatType.TRANSFORMED_PREDICTION_DISTRIBUTION
        ]
        for id, table_pred_stats in pred_stats.items():
            stat_col_keys = table_pred_stats.col_stats.keys()
            target_col = [
                key for key in stat_col_keys
                if (key == 'TARGET_PRED' or key == 'CLASS')
            ]
            if not target_col:
                logger.warn("MLOps: No TARGET_PRED or CLASS column "
                            "found for retrieving BP stats for job "
                            f" {id}")
                continue
            target_col_stats = table_pred_stats.col_stats[target_col[0]]
            if (count_stat in stat_req
                    and ColStatType.COUNT in target_col_stats.keys()):
                stats.bp_stats[id][count_stat] = target_col_stats[
                    ColStatType.COUNT].value

            if (distribution_stat in stat_req):
                # if the column is numeric it will have quantiles, otherwise
                # use category counts (checking dtype here would require
                # loading additional data and slow down the response)
                if ColStatType.QUANTILES in target_col_stats.keys():
                    stats.bp_stats[id][distribution_stat] = target_col_stats[
                        ColStatType.QUANTILES].value
                elif ColStatType.CATEGORY_COUNTS in target_col_stats.keys():
                    stats.bp_stats[id][distribution_stat] = target_col_stats[
                        ColStatType.CATEGORY_COUNTS].value

        return stats

    async def _add_materialization_stats(self, stats: MLOpsStats,
                                         query: MLOpsQueryConfig,
                                         ctx: RequestContext) -> MLOpsStats:
        if MLOpsStatType.GRAPH_VERSION not in query.stat_query.stat_request:
            return stats

        for id in stats.bp_stats.keys():
            if stats.job_status_dict[id] == DB_DONE_STATUS:
                version_dict = {}
                materialization_info = \
                    await self._ml_job_backend.get_materialization_info(
                        BatchPredictionJobID(id))
                if materialization_info:
                    for table_info in materialization_info.table_stats:
                        version_dict[table_info.table_id] = \
                            table_info.data_version
                stats.bp_stats[id][MLOpsStatType.GRAPH_VERSION] = \
                    version_dict

        return stats

    async def _add_rest_db_stats(self, stats: MLOpsStats,
                                 query: MLOpsQueryConfig,
                                 ctx: RequestContext) -> MLOpsStats:
        # Skip lookup if the only current StateDB stat isn't requested.
        if (MLOpsStatType.PREDICTION_OUTPUTS
                not in query.stat_query.stat_request):
            return stats

        bp_ids = set([BatchPredictionJobID(id) for id in stats.bp_stats.keys()])
        if self._ml_job_service is not None:
            bp_rows = await self._ml_job_service.get_batch_prediction_rows(
                bp_ids,
                project_ids=visible_project_ids_or_none(ctx.user_authorization),
            )
        else:
            raise RuntimeError("MLOpsLogic requires StateServices.ml_jobs.")
        for bp_row in bp_rows:
            if bp_row.config:
                stats.bp_stats[bp_row.id][
                    MLOpsStatType.PREDICTION_OUTPUTS] = bp_row.config.outputs
        return stats
