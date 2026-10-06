# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
from typing import Dict, Optional, Tuple

import pandas as pd
from kumoapi.pquery import QueryType, ValidatedPredictiveQuery

from async_utils.sync import blocking_method
from io_utils import logging
from kumo.config.graph import GraphConfig
from kumo.data.generic.diskgraph_batch_pred import _resolve_lp_bp_layout
from kumo.graph import Graph
from kumo.home import get_home_dir
from kumo.pquery.generator_utils import (
    get_default_target_time_range,
    get_task_type,
)

# isort: off
from kumo.pquery.pquery_validators.pre_pred_table_validator import (
    PrePredTableValidator, )
from kumo.pquery.pquery_validators.pred_table_validator import (
    PredTableValidator, )
# isort: on

from kumo_core.types import TaskType
from kumo_core.types.conversions import DataFrame

from kumo.pquery.stats_manager import StatsManager
from kumo.pquery.utils import get_rhs_lp_table, table_name
from kumo.stats import StatsConfig
from kumo.validation import ValidationResponse
from statedb.utils.errors import UnsupportedInDiskgraphError

logger = logging.getLogger(__name__)


class PredictionTableGeneratorV2:
    r"""A class for generating batch pred data for a
    :class:`~kumo.task.Task` within :class:`~kumo.pquery.PredictiveQuery`."""

    ENTITY = 'ENTITY'  # Name of the entity column.
    TARGET = 'TARGET'  # Name of the target column.
    TIMESTAMP = 'TIMESTAMP'  # Name of the time column.

    def __init__(
        self,
        graph: Graph,
        validated_query: ValidatedPredictiveQuery,
        forecast_length: int = 1,
    ):
        self.graph = graph
        self.query = validated_query
        self.forecast_length = forecast_length

        self.task_type: TaskType = get_task_type(self.graph, self.query,
                                                 self.forecast_length).task_type
        self.list_distinct_target_is_array = False
        self.lp_bp_layout: Optional[str] = None

        self.validate_pred_table_response = ValidationResponse()
        self.stats_manager = StatsManager(self.task_type, self.query,
                                          self.graph)

        self.pred_table_validator = PredTableValidator(self.stats_manager)
        self.pre_pred_table_validator = PrePredTableValidator()

    @property
    def query_type(self) -> QueryType:
        return self.query.query_type

    @blocking_method()
    def get_default_target_time_range(
            self) -> Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timedelta]]:
        r"""Determine the total time range that can be used to generate
        targets. Unless overriden with `train_start_offset` or
        `train_end_offset`, target data will be taken from this range.

        Returns:
            Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timedelta]]: A tuple
                of min date, max date, and time delta caused by filter buffer,
                (:obj:`None` if not used or 0) respectively.
        """
        return get_default_target_time_range(self.graph, self.query)

    def get_target_table_name(self) -> str:
        r"""Gets the target table name that can be used for validating
        prediction anchor time.

        Returns:
            str: The name of the target table.
        """
        target_aggrs = self.query.get_all_target_aggregations()
        assert len(target_aggrs) > 0
        assert target_aggrs[0].time_col is not None
        return table_name(target_aggrs[0].time_col)

    def generate_pred_timestamp(self) -> pd.Timestamp:
        r"""Generates a timestamp as anchor time that can be used for
        prediction table. By default we will use the maximum timestamp in
        target table.

        Returns:
            List[pd.Timestamp]: A list of timestamps for data generation, to
                be used as input to the compiler.
        """
        _, max_date, _ = self.get_default_target_time_range()
        return max_date

    def validate_pre_pred_table_generate(
        self,
        timestamp: pd.Timestamp | None,
        lag_timestep: int | None = None,
        is_child_job: bool = False,
    ) -> ValidationResponse:
        r"""Validate any checks that can be done before the pred table
        generation, such as checking prediction anchor timestamp.

        Args:
            timestamp (pd.Timestamp, optional):
                The prediction anchor time. (default: :obj:`None`)
            lag_timestep (int, optional): The number of lag features.
                Required for forecasting problems.
        """
        validation_response = ValidationResponse()

        if self.task_type == TaskType.FORECASTING:
            assert lag_timestep is not None
            assert self.forecast_length >= 1

        if self.task_type.is_link_pred:
            target_table_name = get_rhs_lp_table(self.query, self.graph,
                                                 no_connector=True)
            validation_response.merge(
                self.pre_pred_table_validator.validate_num_rhs_rows(
                    target_table_name=target_table_name,
                    graph=self.graph,
                    is_child_job=is_child_job,
                ))

        if timestamp is not None:
            target_min_date, target_max_date, _ = (
                self.get_default_target_time_range())
            target_table_name = self.get_target_table_name()
            validation_response.merge(
                self.pre_pred_table_validator.validate_anchor_time(
                    timestamp,  # type: ignore
                    target_min_date,
                    target_table_name))
            if self.task_type == TaskType.FORECASTING:
                assert lag_timestep is not None
                timeframe = self.query.get_combined_date_offset_range(
                ).timeframe
                assert timeframe is not None
                validation_response.merge(
                    self.pre_pred_table_validator.
                    validate_anchor_time_forecasting(
                        timestamp,
                        timeframe,
                        lag_timestep,
                        target_table_name,
                        target_max_date,
                        target_min_date,
                    ))

        return validation_response

    def generate_table(
        self,
        pred_time: Optional[pd.Timestamp] = None,
        lag_timesteps: Optional[int] = None,
        use_yoy_labels: bool = False,
        job_id: Optional[str] = None,
        op_id: Optional[str] = None,
        graph_config: Optional[GraphConfig] = None,
        diskgraph_index_path: Optional[str] = None,
        target_file_path: Optional[str] = None,
        list_distinct_target_is_array: bool = False,
        output_path: Optional[str] = None,
        lp_bp_layout: Optional[str] = None,
        destination: Optional[str] = None,
    ) -> Optional[DataFrame]:
        r"""Generates prediction table data.

        For non-diskgraph backends, returns a :class:`kumo.typing.DataFrame`.
        For diskgraph, writes parquet files directly to ``output_path`` and
        returns ``None`` (no DataFrame is produced).

        Args:
            job_id: job id for the prediction job this table is for
            op_id: Unique execution id if the prediction table generation is
                being executed remotely.
            graph_config (Optional[GraphConfig]): Graph config for using the
                correct version of tables to generate prediction table.
            diskgraph_index_path (Optional[str]): Path to the diskgraph index.
            output_path (Optional[str]): Final output directory for diskgraph
                mode. When set, diskgraph writes parquet files here directly.
        """
        # Retained for callers that still send the legacy execution field.
        del op_id
        # The diskgraph path is the only supported producer; callers must
        # thread ``diskgraph_index_path`` through.
        if diskgraph_index_path is None:
            raise UnsupportedInDiskgraphError(
                "PredictionTableGeneratorV2.generate_table requires "
                "diskgraph_index_path; legacy pquery producer was retired "
                "in the DG cutover.")
        self.list_distinct_target_is_array = list_distinct_target_is_array
        lp_bp_layout = (lp_bp_layout if lp_bp_layout is not None else getattr(
            self, 'lp_bp_layout', None))
        resolved_lp_bp_layout = _resolve_lp_bp_layout(
            list_distinct_target_is_array,
            lp_bp_layout,
        )
        self.lp_bp_layout = resolved_lp_bp_layout

        pt = pred_time
        if pt is None and self.query_type == QueryType.TEMPORAL:
            pt = self.generate_pred_timestamp()

        timestamps = [int(pt.value)] if pt else []  # ns from pd.Timestamp

        if output_path is None:
            output_path = osp.join(
                destination or get_home_dir(),
                'diskgraph_batch_pred',
                'pred_table',
            )

        # Mirror of TT (train_table_generator.py): surface entity.time_col
        # so the diskgraph executor emits real per-row TIMESTAMPs for
        # static + fact-entity queries (instead of zeros), and so
        # create_time filtering kicks in for temporal queries.
        entity_table = self.graph.table(table_name(self.query.entity_column))
        entity_time_col = (entity_table.time_col.name
                           if entity_table.time_col is not None else None)

        # Bypass the service and call the TT-NG sibling directly. Lazy-import
        # TimeUnit and the sibling.
        from diskgraph import TimeUnit

        # isort: off
        from kumo.data.generic.diskgraph_batch_pred import (
            generate_batch_pred_table_via_diskgraph_ttng, )
        # isort: on
        pred_table_num_entities: int | None
        pred_table_num_entities, _ = (
            generate_batch_pred_table_via_diskgraph_ttng(
                query=self.query,
                task_type=self.task_type,
                graph=self.graph,
                timestamps=timestamps,
                forecast_length=self.forecast_length,
                lag_timesteps=lag_timesteps,
                use_yoy_labels=use_yoy_labels,
                list_distinct_target_is_array=(list_distinct_target_is_array),
                index_path=diskgraph_index_path,
                output_root=output_path,
                target_output_root=target_file_path,
                input_timestamp_unit=TimeUnit.Nanoseconds,
                target_stype=self.query.target_stype.value,
                entity_time_col=entity_time_col,
                lp_bp_layout=resolved_lp_bp_layout,
                # Names the kept predict staging dir for materialize (one-gen handoff).
                job_id=job_id,
            ))
        if job_id is not None and isinstance(pt,
                                             pd.Timestamp) and not pd.isna(pt):
            self.stats_manager.update_pred_time(job_id, pt)
        if job_id is not None and pred_table_num_entities is not None:
            self.stats_manager.update_pred_num_entities(
                job_id,
                pred_table_num_entities,
            )
        return None

    def get_batch_pred_table_stats_config(
            self, job_id: str) -> Optional[Dict[str, StatsConfig]]:
        r"""Get the prediction anchor time associated with a job ID.

        Args:
            job_id (str): The ID of the prediction job.

        Returns:
            Optional[Dict[str, StatsConfig]]: The stats config for the
                batch prediction table.
        """
        return self.stats_manager.get_batch_pred_table_stats_config(job_id)

    def get_validate_train_table_response(self) -> ValidationResponse:
        return self.validate_pred_table_response
