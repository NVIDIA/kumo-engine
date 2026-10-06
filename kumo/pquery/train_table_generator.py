# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import os.path as osp
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import kumo_core.types as core_types
from kumo_core.training.options import allow_array_targets
from kumo_core.training.splits import TrainTableTime, train_table_time
from kumo_core.training.timestamps import (
    generate_decoupled_timestamps,
    generate_legacy_timestamps,
)
from kumo_core.types.conversions import DataFrame
from kumo_core.utils.time import PSEUDO_TIME_COL_NAME
from kumoapi.pquery import QueryType, ValidatedPredictiveQuery
from kumoapi.pquery.AST import Aggregation, DateOffsetRange, Filter
from kumoapi.typing import Dtype
from pandas import DateOffset, Timedelta, Timestamp

from async_utils.sync import blocking_method
from io_utils import fsutil, logging
from kumo.config.graph import GraphConfig
from kumo.data.utils import ENTITY_INCLUDE_BIT
from kumo.diskgraph import get_service
from kumo.graph import BoolColumn, Column, ForeignKey, Graph, TimeColumn
from kumo.home import get_home_dir
from kumo.pquery.generator_utils import (
    get_default_target_time_range,
    get_task_type,
)
from kumo.pquery.pquery_validators import (
    PreTrainTableValidator,
    TrainTableValidator,
)
from kumo.pquery.stats_manager import StatsManager
from kumo.pquery.utils import col_name, get_max_train_rows, table_name
from kumo.pquery.weight_spec import (
    DEFAULT_WEIGHT_COL_NAME,
    WeightSpec,
    normalize_weight_col,
    parse_weight_spec,
)
from kumo.task.split import (
    BacktestingSplit,
    DateOffsetSplit,
    Split,
    TimeRangeSplit,
)
from kumo.task.splits import TimeSplit
from kumo.validation import ValidationException, ValidationResponse
from kumo.validation.train_table_validation import TrainingTableGenerationError
from statedb.utils.errors import (
    DiskgraphConfigError,
    DiskgraphDataError,
    DiskgraphSystemError,
    DiskgraphTargetCategoryLimitError,
    UnsupportedInDiskgraphError,
)

logger = logging.getLogger(__name__)

# Substrings (case-insensitive) emitted by the DiskGraph executor when a
# training table is materialized with no rows. The wheel has emitted three
# distinct signals across supported generations:
#   * ``execute_tt_plan returned zero rows`` -- the explicit zero-row status.
#   * ``empty training table dataset`` -- the multipart TT-NG writer emitted
#     no parquet parts.
#   * ``parquet file too small. size is 0`` -- the TT-NG streaming ttgen path
#     (``execute_tt_gen``) writes a 0-byte ``tt.parquet`` when every anchor is
#     filtered out, and the per-stage seed-parquet demux then fails reading it
#     back (see diskgraph ttgen_executor / pass_b: zero-row + computed split ->
#     Err "Parquet file too small. Size is 0 but need 8"). On the TT-NG path
#     this EOF is the deterministic empty-table signal, not a corruption; the
#     same graceful "Empty training table" validation error the legacy
#     execute_tt_plan path produced.
# Centralized so the reclassification logic and its tests stay in sync if the
# upstream messages are ever reworded.
_DISKGRAPH_ZERO_ROWS_MARKERS = (
    "diskgraph execute_tt_plan returned zero rows",
    "empty training table dataset",
    "parquet file too small. size is 0",
)


def _is_instance_of_optional_error(error: Exception, error_type: Any) -> bool:
    return error_type is not None and isinstance(error, error_type)


def _diskgraph_empty_train_table_error_message(
    error: Exception,
    *,
    entity_table: str,
    timestamp_count: int,
) -> Optional[str]:
    detail = str(error)
    detail_lower = detail.lower()
    is_diskgraph_error = (
        _is_instance_of_optional_error(error, DiskgraphConfigError)
        or _is_instance_of_optional_error(error, DiskgraphDataError)
        or _is_instance_of_optional_error(error, DiskgraphSystemError))
    if not is_diskgraph_error:
        return None
    if not any(marker in detail_lower
               for marker in _DISKGRAPH_ZERO_ROWS_MARKERS):
        return None

    anchor_context = (f"across {timestamp_count} anchor timestamps"
                      if timestamp_count > 0 else
                      "without temporal anchor timestamps")
    return (
        "DiskGraph materialized zero training rows for "
        f"entity table {entity_table!r} {anchor_context}. "
        "This can happen when the PQuery filters, "
        "target window, split boundaries, or ingested graph edges leave "
        "no valid training examples. Inspect DiskGraph ingest, edge, and "
        f"column stats for dropped rows or zero required edges. Detail: {detail}"
    )


def _raise_training_table_generation_error_for_diskgraph_empty_result(
    error: Exception,
    *,
    entity_table: str,
    timestamp_count: int,
) -> None:
    message = _diskgraph_empty_train_table_error_message(
        error,
        entity_table=entity_table,
        timestamp_count=timestamp_count,
    )
    if message is None:
        return

    raise TrainingTableGenerationError(
        title="Empty training table",
        message=message,
    ) from error


def _raise_training_table_generation_error_for_target_category_limit(
    error: Exception, ) -> None:
    """Classify DiskGraph's exact-target vocabulary limit as terminal."""
    if not _is_instance_of_optional_error(error,
                                          DiskgraphTargetCategoryLimitError):
        return

    detail = str(error)
    raise TrainingTableGenerationError(
        title="Train-table target category limit exceeded",
        message=(
            "DiskGraph could not collect the complete exact target vocabulary, "
            "so no usable train table statistics artifact was published. "
            "Retrying with unchanged data cannot succeed; reduce target "
            "cardinality, model this as link prediction when appropriate, or "
            f"contact Kumo support. Detail: {detail}"),
    ) from error


def _invoke_diskgraph_train_table_seam(
    seam_call: Callable[[], Any],
    *,
    entity_table: str,
    timestamp_count: int,
) -> Any:
    """Run a DiskGraph train-table seam call, reclassifying empty results.

    Wraps the narrowest possible call site (the ``generate_train_table*``
    invocation) so a ``DiskgraphDataError`` / ``DiskgraphSystemError``
    raised for the well-known zero-row case, or the dedicated DiskGraph target
    category limit error, is surfaced as a ``TrainingTableGenerationError``.
    All other exceptions propagate unchanged.
    """
    try:
        return seam_call()
    except Exception as e:
        _raise_training_table_generation_error_for_diskgraph_empty_result(
            e,
            entity_table=entity_table,
            timestamp_count=timestamp_count,
        )
        _raise_training_table_generation_error_for_target_category_limit(e)
        raise


def _check_diskgraph_train_table_parity(
    *,
    weight_spec: Optional[WeightSpec],
    task_type: core_types.TaskType,
    query_type: QueryType,
    split: 'Split',
) -> None:
    """Raise for DG train-table capabilities not yet wired to Rust.

    Each branch narrows on the specific trigger so unrelated DG jobs
    (node-pred regression, temporal LP, etc.) keep working. Remove
    each branch when the corresponding DG Rust operator lands.
    """
    if weight_spec is not None:
        raise UnsupportedInDiskgraphError(
            "Weighted training tables are not yet supported in "
            "DiskGraph. DiskGraphPlanCompiler has no weight parameters, "
            "so sample weights would be silently dropped. Remove this "
            "guard once DG Rust weight plumbing lands.")
    if (query_type == QueryType.STATIC and task_type.is_node_pred
            and isinstance(split, TimeSplit)):
        raise UnsupportedInDiskgraphError(
            "Static node-prediction with TimeSplit "
            "is not yet supported in DiskGraph. The Spark path filters "
            "rows to split boundaries via filter_static_split; DG "
            "would include out-of-split rows. Remove this guard once "
            "DG implements static split filtering.")


def _check_diskgraph_downsampling_parity(enabled: bool) -> None:
    """Raise if DG downsampling is requested but Rust doesn't honor it.

    ``DISKGRAPH_PRODUCER_DOWNSAMPLING_ENABLED`` is an opt-in env var.
    When not set (default), DG produces the full TT and this check is
    a no-op. When set to ``1``, downsampling is expected but is not
    wired through DiskGraphPlanCompiler to Rust today; silently
    returning the full TT would mask the mis-configuration.
    """
    if enabled:
        raise UnsupportedInDiskgraphError(
            "DISKGRAPH_PRODUCER_DOWNSAMPLING_ENABLED=1 is set, but "
            "DiskGraph downsampling is not yet wired through "
            "DiskGraphPlanCompiler to the Rust executor. Remove this "
            "guard once DG Rust downsampling lands.")


# Frequency of entity target pair for static and temporal link prediction
FREQUENCY_COL_NAME = '__kumo_internal_frequency_column__'

INTERMEDIATE_TRAIN_TABLE_DIR = 'intermediate_train_tables'
TRAIN_TABLE = 'train_table'


class TrainingTableGeneratorV2:
    r"""A data class for generating training data for a
    :class:`~kumo.task.Task` within :class:`~kumo.pquery.PredictiveQuery`."""

    ENTITY = 'ENTITY'  # Name of the entity column.
    TARGET = 'TARGET'  # Name of the target column.
    WEIGHT = DEFAULT_WEIGHT_COL_NAME  # Name of the weight column.
    TIMESTAMP = 'TIMESTAMP'  # Name of the time column.
    END_TIMESTAMP = 'END_TIMESTAMP'  # Name of the end time column.
    ASSUMPTION = 'ASSUMPTION'  # Name of the whatif result column.
    ENTITY_PRECOMPUTE = 'ENTITY_PRECOMPUTE'  # Name of precomputed entity table

    def __init__(
        self,
        graph: Graph,
        validated_query: ValidatedPredictiveQuery,
        split: Split,
        train_start_offset: Optional[int] = None,
        train_end_offset: Optional[Union[int, DateOffset]] = None,
        start_time: Optional[Union[int, str]] = None,
        end_time: Optional[Union[int, str]] = None,
        timeframe_step: Optional[int] = None,
        drop_earliest_timeframe: Optional[bool] = True,
        lag_timesteps: Optional[int] = None,
        forecast_length: int = 1,
        year_over_year: bool = False,
        use_decoupled_splits: Optional[bool] = None,
        weight_col: Optional[str] = None,
    ):
        self.graph = graph
        self.query = validated_query
        self.split = split
        self.train_start_offset = None
        if train_start_offset is not None:
            self.train_start_offset = self.get_date_offset(
                int(train_start_offset))
        if train_end_offset is not None and isinstance(train_end_offset, int):
            train_end_offset = self.get_date_offset(int(train_end_offset))
        assert (train_end_offset is None
                or isinstance(train_end_offset, DateOffset))
        self.train_end_offset = train_end_offset

        # Handle new time parameters
        self.start_time = None
        if start_time is not None:
            if start_time == '-INF':
                pass  # Keep start_time as None for -INF case
            elif isinstance(start_time, int):
                self.start_time = self.get_date_offset(int(start_time))
            else:
                assert isinstance(start_time, str)
                self.start_time = Timestamp(start_time)
                assert isinstance(self.start_time, Timestamp)

        self.end_time = None
        if end_time is not None:
            if isinstance(end_time, int):
                self.end_time = self.get_date_offset(int(end_time))
            else:
                assert isinstance(end_time, str)
                self.end_time = Timestamp(end_time)
                assert isinstance(self.end_time, Timestamp)

        self.timeframe_step = None
        if timeframe_step is not None:
            self.timeframe_step = self.get_date_offset(timeframe_step)
        self.drop_earliest_timeframe = drop_earliest_timeframe
        self.lag_timesteps = lag_timesteps
        self.forecast_length = forecast_length
        self.year_over_year = year_over_year
        self.use_decoupled_splits = use_decoupled_splits
        self.weight_col = normalize_weight_col(weight_col)

        self.task_type: core_types.TaskType = get_task_type(
            self.graph, self.query, self.forecast_length).task_type
        self.list_distinct_target_is_array = allow_array_targets()

        self._train_table_stats = None

        self.validate_train_table_response = ValidationResponse()
        self.stats_manager = StatsManager(self.task_type, self.query,
                                          self.graph)
        self.train_table_validator = TrainTableValidator(
            self.stats_manager,
            self.split,
            self.timeframe_step,
        )
        self.pre_train_table_validator = PreTrainTableValidator(
            self.split,
            self.query,
            self.graph,
            self.timeframe_step,
            self.train_start_offset,
            self.train_end_offset,
            self.start_time,
            self.end_time,
            task_type=self.task_type,
        )
        # Set by diskgraph native TT path so caller can persist producer
        # parquet directly without loading it through Spark.
        self._diskgraph_last_output_path: Optional[str] = None
        self._diskgraph_last_manifest_path: Optional[str] = None
        self._sync_list_distinct_target_is_array()

    def set_list_distinct_target_is_array(
        self,
        list_distinct_target_is_array: Optional[bool],
    ) -> bool:
        if list_distinct_target_is_array is not None:
            self.list_distinct_target_is_array = list_distinct_target_is_array
        return self._sync_list_distinct_target_is_array()

    def _sync_list_distinct_target_is_array(self) -> bool:
        list_distinct_target_is_array = bool(self.list_distinct_target_is_array)
        self.stats_manager.list_distinct_target_is_array = (
            list_distinct_target_is_array)
        self.train_table_validator.list_distinct_target_is_array = (
            list_distinct_target_is_array)
        return list_distinct_target_is_array

    def _build_diskgraph_downsampling_config(
        self,
        list_distinct_target_is_array: bool,
    ) -> dict[str, Any]:
        """Build nested downsampling config passed to diskgraph producer."""
        enabled = os.getenv('DISKGRAPH_PRODUCER_DOWNSAMPLING_ENABLED',
                            '0') == '1'
        _check_diskgraph_downsampling_parity(enabled)
        return {
            'enabled': enabled,
            'max_rows': int(get_max_train_rows()),
            'preserve_pre_downsample_tt': True,
            'is_link_prediction': bool(self.task_type.is_link_pred),
            'list_distinct_target_is_array':
            bool(list_distinct_target_is_array),
            'uses_arl_labels': bool(self.task_type.uses_arl_labels),
        }

    @property
    def query_type(self) -> QueryType:
        return self.query.query_type

    @property
    def entity_table_name(self) -> str:
        """Get the name of the entity table.

        Returns:
            str: Name of the entity table.
        """
        return table_name(self.query.entity_column)

    @blocking_method()
    def get_default_target_time_range(
            self) -> Tuple[Timestamp, Timestamp, Optional[Timedelta]]:
        r"""Determine the total time range that can be used to generate
        targets. Unless overriden with `train_start_offset` or
        `train_end_offset`, target data will be taken from this range.

        Returns:
            Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timedelta]]: A tuple
                of min date, max date, and time delta caused by filter buffer,
                (:obj:`None` if not used or 0) respectively.
        """
        return get_default_target_time_range(self.graph, self.query)

    def _timestamp_inputs(self):
        original_start, original_end, filter_buffer = (
            self.get_default_target_time_range())
        offset_range = self.query.get_combined_date_offset_range()
        assert offset_range.start_date_offset is not None
        timeframe = DateOffsetRange.date_offset_subtract(
            offset_range.end_date_offset, offset_range.start_date_offset)
        step = timeframe if self.timeframe_step is None else self.timeframe_step
        return (original_start, original_end, filter_buffer, offset_range,
                timeframe, step)

    def generate_timestamps(self) -> List[Timestamp]:
        r"""Generate legacy split-aware training-table timestamps."""
        values = self._timestamp_inputs()
        (original_start, original_end, filter_buffer, offset_range, timeframe,
         step) = values
        logger.info(
            f"For raw data, the min_date is {original_start} and the max_date "
            f"is {original_end}.")
        split_family = None
        descriptor = None
        if isinstance(self.split, DateOffsetSplit):
            split_family = "DateOffsetSplit"
            descriptor = {
                "offsets": self.split.offsets,
                "unit": self.split.unit.value,
            }
        elif isinstance(self.split, TimeRangeSplit):
            split_family = "TimeRangeSplit"
            descriptor = {"ranges": self.split.ranges}
        elif isinstance(self.split, BacktestingSplit):
            split_family = "BacktestingSplit"
            descriptor = {"timestamp": self.split.timestamp}

        def fix_boundaries() -> None:
            if self.split.flexible_boundaries:
                self.split.flexible_boundaries = False
                logger.warning(
                    'Split flexible boundaries has been set to True but '
                    'the split is used for sampling. Flexible boundaries '
                    'have been reset to False.')

        timestamps, _, short_window = generate_legacy_timestamps(
            original_start=original_start,
            original_end=original_end,
            filter_buffer=filter_buffer,
            target_timeframe=timeframe,
            target_start_offset=offset_range.start_date_offset,
            target_end_offset=offset_range.end_date_offset,
            step=step,
            split_family=split_family,
            split=descriptor,
            train_start_offset=self.train_start_offset,
            train_end_offset=self.train_end_offset,
            start_time=self.start_time,
            end_time=self.end_time,
            drop_earliest=self.drop_earliest_timeframe,
            on_fixed_boundaries=fix_boundaries,
        )
        if short_window:
            return []
        logger.info(f'Generated {len(timestamps)} timestamps:')
        for i, timestamp in enumerate(timestamps):
            logger.info(f'Timestamp {i:03d}: {timestamp}')
        return timestamps

    def generate_timestamps_v2(self) -> List[Timestamp]:
        r"""Generate decoupled training-table timestamps."""
        if (self.train_start_offset is not None
                or self.train_end_offset is not None):
            raise ValidationException(
                title="Deprecated training offsets",
                message=(
                    "train_start_offset and train_end_offset are deprecated "
                    "for generate_timestamps_v2 function. Please use "
                    "start_time and end_time parameters instead."),
            )
        values = self._timestamp_inputs()
        (original_start, original_end, filter_buffer, offset_range, timeframe,
         step) = values
        timestamps = generate_decoupled_timestamps(
            original_start=original_start,
            original_end=original_end,
            filter_buffer=filter_buffer,
            target_timeframe=timeframe,
            target_start_offset=offset_range.start_date_offset,
            step=step,
            start_time=self.start_time,
            end_time=self.end_time,
            drop_earliest=self.drop_earliest_timeframe,
        )
        logger.info(f'Generated {len(timestamps)} timestamps:')
        for i, timestamp in enumerate(timestamps):
            logger.info(f'Timestamp {i:03d}: {timestamp}')
        return timestamps

    def get_exclude_cols(self) -> Dict[str, List[str]]:
        r"""Returns the list of columns per table name to exclude from the
        :class:`~kumo.graph.Graph` in order to prevent data leakage."""
        return self.query.get_exclude_cols_dict()

    def validate(self) -> ValidationResponse:
        r"""Validates the input arguments of the :class:`SampleGenerator`
        instance. Raises ValueError for all blocking errors, and returns
        non-blocking errors as validation responses."""
        raise NotImplementedError()

    @blocking_method()
    def get_train_table_stats(self, job_id: Optional[str] = None) -> Any:
        r"""Gets important statistics of the generated
        :class:`~kumo.task.TrainingTable` in order to rate the quality of a
        :class:`PredictiveQuery`.
        Args:
            job_id: job id to retrieve the training table stats.
        """
        return self.stats_manager.get_train_table_stats(job_id)

    def get_train_table_progress(self, job_id: str) -> Any:
        r"""Gets progress and current statistics of the generated
        :class:`~kumo.task.TrainingTable`.
        Args:
            job_id: job id to retrieve the training table stats.
        """
        # TODO: It requires further refactor to make this work with PQL v2
        return None

    def get_validate_train_table_response(self) -> ValidationResponse:
        return self.validate_train_table_response

    def get_train_table_min_max_date(
        self, timestamps: List[Timestamp]
    ) -> Tuple[Optional[Timestamp], Optional[Timestamp], Optional[Timestamp]]:
        r"""Given a list of timestamps for a train table, returns the
        corresponding anchor time, min date and max date.
        """
        # `anchor_time` and `train_table_min_date` are not well defined
        # when `timestamps` is empty.
        anchor_time = None
        train_table_min_date = None
        train_table_max_date = None
        if len(timestamps) > 0:
            anchor_time = min(timestamps)
            offset_range = self.query.get_combined_date_offset_range()
            assert offset_range.start_date_offset is not None
            train_table_min_date = anchor_time + offset_range.start_date_offset
            _, train_table_max_date, _ = self.get_default_target_time_range()
        return anchor_time, train_table_min_date, train_table_max_date

    def validate_pre_train_table_generate(
            self, skip_timestamp_validation: bool = True) -> ValidationResponse:
        r"""Validate any checks that can be done before the train table
        generation, such as checking timestamps and start/end offsets for
        temporal queries.
        This is to catch basic problems that could result
        in empty or irregular train table being generated.

        Args:
            skip_tmestamp_validation (bool): If set to True, the timestamps
                will not be validated. This was added as a temporary fix, as
                timestamp validation requires fetching max/min dates from the
                databackend which can be very slow sometimes.
                (default: :obj:`True`)
        """
        validation_response = ValidationResponse()

        validation_response.merge(
            self.pre_train_table_validator.validate_forecasting_task_type(
                self.lag_timesteps, self.forecast_length))

        if not validation_response.ok:
            return validation_response

        if self.query_type == QueryType.TEMPORAL:
            validation_response.merge(
                self.pre_train_table_validator.validate_temporal_splits())

            # Add new validation for time parameter compatibility
            validation_response.merge(self.pre_train_table_validator.
                                      validate_train_time_compatibility())

            # Add new validation for time parameter values
            validation_response.merge(
                self.pre_train_table_validator.validate_train_time())

            validation_response.merge(
                self.pre_train_table_validator.validate_train_offsets(
                    self.train_start_offset, self.train_end_offset))
            if not validation_response.ok:
                return validation_response

            if not skip_timestamp_validation:
                if isinstance(self.split, TimeSplit):
                    timestamps = self.generate_timestamps_v2()
                else:
                    timestamps = self.generate_timestamps()
                _, train_table_min_date, train_table_max_date = (
                    self.get_train_table_min_max_date(timestamps))

                validation_response.merge(
                    self.pre_train_table_validator.validate_timestamps(
                        timestamps, train_table_min_date, train_table_max_date))
                if self.task_type == core_types.TaskType.FORECASTING:
                    assert self.lag_timesteps is not None
                    if isinstance(self.split, TimeSplit):
                        validation_response.merge(
                            self.pre_train_table_validator.
                            validate_timestamps_for_forecasting_v2(
                                timestamps, self.forecast_length,
                                self.lag_timesteps))
                    else:
                        validation_response.merge(
                            self.pre_train_table_validator.
                            validate_timestamps_for_forecasting(
                                timestamps, self.forecast_length,
                                self.lag_timesteps))

                validation_response.merge(
                    self.pre_train_table_validator.validate_limits(timestamps))

        return validation_response

    def validate_scale(self) -> Optional[ValidationResponse]:
        r"""Validate if the data is too large to be processed."""
        validation_response = ValidationResponse()
        if self.query_type == QueryType.TEMPORAL:
            if isinstance(self.split, TimeSplit):
                timestamps = self.generate_timestamps_v2()
            else:
                timestamps = self.generate_timestamps()
            warning = self.pre_train_table_validator.validate_limits(timestamps)
            if len(warning.warnings) > 0 or not warning.ok:
                validation_response.merge(warning)
                return validation_response
        return None

    def _get_weight_spec(self) -> Optional[WeightSpec]:
        """Parse weight_col into a WeightSpec."""
        if self.weight_col is None:
            return None
        return parse_weight_spec(self.weight_col)

    def generate_table(
        self,
        job_id: Optional[str] = None,
        op_id: Optional[str] = None,
        report_progress: Optional[Any] = None,
        intermediate_train_table_dir: Optional[str] = None,
        graph_config: Optional[GraphConfig] = None,
        diskgraph_index_path: Optional[str] = None,
        list_distinct_target_is_array: Optional[bool] = None,
        destination: Optional[str] = None,
    ) -> DataFrame:
        r"""Generates training table data as a :class:`kumo.typing.DataFrame`.
        Additionally returns a stats object denoting training table generation
        statistics.

        Args:
            job_id (str, optional): job id for the training job this table
                is for
            op_id: (str, optional): Unique execution id if the training
                table generation is being executed remotely.
            report_progress (Any): Added for backward compatibility with
                V1, ignored.
            intermediate_train_table_dir (str, optional): Name of the
                directory to materialize intermediate tables used for train
                table generation.
            graph_config(Optional[GraphConfig]): GraphConfig used to load the
                right version of tables.
        """
        # Retained for callers that still send the legacy execution field.
        del op_id
        # Honor the runtime list_distinct_target_is_array override and sync
        # it onto the stats manager / validator.
        list_distinct_target_is_array = (self.set_list_distinct_target_is_array(
            list_distinct_target_is_array))

        # Run validation before generating the table
        validation_response = self.validate_pre_train_table_generate(
            skip_timestamp_validation=False)
        if not validation_response.ok:
            raise TrainingTableGenerationError(
                title="Training table generation failed",
                message=validation_response.error_message())

        intermediate_train_table_dir = (intermediate_train_table_dir
                                        or INTERMEDIATE_TRAIN_TABLE_DIR)

        # Generate timestamps that will define timeframes for temporal queries
        logger.info('Generating timestamps')
        timestamps = []
        if self.query_type == QueryType.TEMPORAL:
            if isinstance(self.split, TimeSplit):
                timestamps = self.generate_timestamps_v2()
            else:
                timestamps = self.generate_timestamps()

        # The diskgraph path is the only supported producer; callers must
        # thread ``diskgraph_index_path`` through.
        if not diskgraph_index_path:
            raise UnsupportedInDiskgraphError(
                "TrainingTableGeneratorV2.generate_table requires "
                "diskgraph_index_path; legacy pquery producer was retired "
                "in the DG cutover.")

        _check_diskgraph_train_table_parity(
            weight_spec=self._get_weight_spec(),
            task_type=self.task_type,
            query_type=self.query_type,
            split=self.split,
        )

        # Pass raw nanosecond values from pandas Timestamps; the
        # DiskGraphPlanCompiler auto-infers the input unit and
        # normalizes to the target unit (Seconds) internally.
        timestamp_nanos = [int(ts.value) for ts in timestamps]
        entity_time_col = None
        entity_table = self.graph.table(self.entity_table_name)
        if entity_table.time_col is not None:
            entity_time_col = entity_table.time_col.name

        include_frequency = (self.task_type.is_link_pred)
        output_path = osp.join(
            destination or get_home_dir(),
            intermediate_train_table_dir,
            'diskgraph_train_table',
            'train_table.parquet',
        )

        # The intermediate output_path is a FIXED location reused by every
        # train-table generation in the process. The wheel appends this
        # run's reverse-edge dirs under ``<output_path>.artifacts/
        # train_table_edges/`` but does NOT prune dirs left by a prior
        # generation. If a previous generation used a different entity
        # table name (e.g. ``USER`` vs ``USERS``), its stale edge dir leaks
        # into this run's manifest ``reverse_edges`` map and the strict
        # edge-materialize resolver can't find a matching persist path
        # (RuntimeError: could not resolve edge persist path). Remove the
        # fixed output_path + its ``.artifacts`` sibling so each generation
        # starts from a clean slate. (Local home_dir only; the remote
        # branch below stages into a fresh tempdir.)
        if fsutil.is_local(output_path):
            for stale in (output_path, f'{output_path}.artifacts'):
                if fsutil.exists(stale):
                    fsutil.rm(stale, ignore_errors=True)

        # Diskgraph Rust writes via std::fs and cannot handle S3/GCS
        # paths.  When get_home_dir() is remote, stage in a local
        # temp directory.  Cleanup happens in generate_train_table_generic
        # after the artifacts are copied to their final location.
        self._diskgraph_staging_dir = None
        if not fsutil.is_local(output_path):
            self._diskgraph_staging_dir = tempfile.mkdtemp(
                prefix='diskgraph_tt_')
            output_path = osp.join(self._diskgraph_staging_dir,
                                   'train_table.parquet')
            logger.info(
                "Remote home_dir; staging diskgraph TT locally: %s",
                self._diskgraph_staging_dir,
            )

        # Bypass the service and call the TT-NG sibling directly. Lazy-import
        # TimeUnit and the sibling. Returns (output_path, manifest_path).
        from diskgraph import TimeUnit

        from kumo.data.generic import diskgraph_train_table

        # ``tt_id`` must be filename-safe — the wheel uses it as the
        # directory name under ``index_path/train_tables/<tt_id>/``.
        # The gen-traintable-job_id is already filename-safe and is
        # the natural per-run identifier; the diskgraph code path
        # below also enforces non-None job_id (see the
        # ``train stats cache`` assertion).
        if not job_id:
            raise RuntimeError('diskgraph TT-NG path requires non-empty job_id '
                               '(passed through to execute_tt_gen as tt_id)')
        generate_train_table = (
            diskgraph_train_table.generate_train_table_via_diskgraph_ttng)
        output_path, manifest_path = _invoke_diskgraph_train_table_seam(
            lambda: generate_train_table(
                query=self.query,
                graph=self.graph,
                timestamps=timestamp_nanos,
                task_type=self.task_type.value,
                forecast_length=self.forecast_length,
                lag_timesteps=self.lag_timesteps,
                year_over_year=self.year_over_year,
                list_distinct_target_is_array=list_distinct_target_is_array,
                include_frequency=include_frequency,
                entity_time_col=entity_time_col,
                index_path=diskgraph_index_path or '',
                output_path=output_path,
                tt_id=job_id,
                split_config=self.split.config(),
                split_family=self.split.__class__.__name__,
                downsampling=self._build_diskgraph_downsampling_config(
                    list_distinct_target_is_array=list_distinct_target_is_array,
                ),
                # No ``train_table_edges_config``: on TT-NG, DG owns
                # all edge materialization. ``execute_tt_gen`` writes
                # the edge/fkey spec into the staging
                # ``tt_ingest_spec.json``, which ``execute_tt_ingest``
                # (called from materialize) consumes to build edges.
                # kumo builds/threads no edge config here.
                input_timestamp_unit=TimeUnit.Nanoseconds,
                target_stype=self.query.target_stype.value,
            ),
            entity_table=self.entity_table_name,
            timestamp_count=len(timestamp_nanos),
        )
        self._diskgraph_last_output_path = output_path
        self._diskgraph_last_manifest_path = manifest_path

        parsed_stats = None
        manifest = get_service().validate_manifest(manifest_path)
        if isinstance(manifest, dict):
            maybe_stats = manifest.get('parsed_stats')
            if isinstance(maybe_stats, dict):
                parsed_stats = maybe_stats

        if self.query_type == QueryType.TEMPORAL and job_id is not None:
            anchor_time, train_table_min_date, train_table_max_date = (
                self.get_train_table_min_max_date(timestamps))
            assert train_table_min_date is not None
            assert train_table_max_date is not None
            assert anchor_time is not None
            self.stats_manager.update_min_max_anchor_date_cache(
                job_id, train_table_min_date, train_table_max_date, anchor_time)

        row_count = None
        if isinstance(manifest, dict):
            maybe_row_count = manifest.get('row_count')
            if isinstance(maybe_row_count, int):
                row_count = maybe_row_count
        if row_count is None and isinstance(parsed_stats, dict):
            maybe_total = parsed_stats.get('num_total_rows')
            if isinstance(maybe_total, int):
                row_count = maybe_total
        if row_count is not None and row_count <= 0:
            raise TrainingTableGenerationError(
                "Empty training table",
                "Diskgraph returned an empty training table.",
            )

        if not isinstance(parsed_stats, dict):
            raise RuntimeError(
                'diskgraph path requires parsed diskgraph stats artifact '
                'but none were available from manifest validation')
        if job_id is None:
            raise RuntimeError(
                'diskgraph path requires non-empty job_id to seed train stats cache'
            )

        missing_stats = self.stats_manager.seed_train_stats_cache_from_native_artifact(
            job_id,
            parsed_stats,
        )
        if missing_stats:
            raise RuntimeError(
                'diskgraph path requires complete diskgraph train stats payload; '
                f'missing expected stats keys: {missing_stats}')
        logger.info('Seeded train stats cache from diskgraph artifact '
                    '(native_full strict mode)')
        # Caller persists the producer parquet directly via
        # self._diskgraph_last_output_path; no Spark parquet readback.
        return None  # type: ignore[return-value]

    def validate_post_train_table_generate(self) -> ValidationResponse:
        r"""
        Validates that the generated training table is valid after collecting
        all training table statistics.
        """
        response = self.train_table_validator.\
            validate_post_train_table_generate()
        self.validate_train_table_response.merge(response)
        return response

    def get_date_offset(self, date_offset: int) -> DateOffset:
        assert self.query.target_timeframe is not None
        unit = self.query.target_timeframe.unit
        return DateOffsetRange._to_dateOffset(unit, date_offset)

    def get_target_col(self) -> Optional[Column]:
        if self.task_type.is_link_pred:
            return None
        dtype = Dtype(self.query.target_dtype.value)
        return Column(name=self.TARGET, dtype=dtype)

    def get_fkeys(self) -> List[ForeignKey]:
        r"""Foreign keys of the training table. Usually a single key pointing
        to the entity table for node predicion and 2 keys pointing to lhs
        entity table and rhs entity table for link prediction.
        """
        fkeys = [
            ForeignKey(
                name=self.ENTITY,
                dst_table=table_name(self.query.entity_column),
                dst_key=col_name(self.query.entity_column),
                dtype=self.query.entity_dtype,
            )
        ]

        if self.task_type.is_link_pred:
            # Obtain the primary key of rhs entity table
            aggr = self.query.get_final_target_aggregation()
            assert isinstance(aggr, Aggregation)
            aggregated_col_name = aggr._get_target_column_name()
            aggregated_col = self.graph.column_from_fqn(aggregated_col_name)
            assert isinstance(aggregated_col, ForeignKey)

            rhs_fkey = ForeignKey(
                name=self.TARGET,
                dst_table=aggregated_col.dst_table,
                dst_key=aggregated_col.dst_key,
                dtype=aggregated_col.dtype,
            )
            fkeys.append(rhs_fkey)

        return fkeys

    def get_cols(self) -> List[Column]:
        return []

    def get_time_col(self) -> Optional[TimeColumn]:
        entity_table_name = table_name(self.query.entity_column)
        time = train_table_time(
            self.task_type,
            self.query_type,
            list_distinct_target_is_array=bool(
                self.list_distinct_target_is_array),
            entity_table_has_time_col=self.graph.table(
                entity_table_name).has_time_col(),
        )
        if time == TrainTableTime.PSEUDO:
            return TimeColumn(PSEUDO_TIME_COL_NAME, dtype=Dtype.int64,
                              is_pseudo=True)
        if time == TrainTableTime.REAL:
            return TimeColumn(name=self.TIMESTAMP)
        return None

    def get_entity_include_bit_col(self) -> Optional[BoolColumn]:
        """Get the ENTITY_INCLUDE_BIT column if it exists.

        Returns:
            BoolColumn(optional): The ENTITY_INCLUDE_BIT column.
        """
        if self.task_type == core_types.TaskType.FORECASTING and (
                isinstance(self.query.entity_ast, Filter)
                or self.query.whatif_ast is not None):
            return BoolColumn(ENTITY_INCLUDE_BIT)
        return None

    def get_entity_table_name(self) -> str:
        """Fully qualified name of the entity table.

        Returns:
            str: The entity table name.
        """
        entity_table_name = table_name(self.query.entity_column)
        return self.graph.fully_qualified_table_name(entity_table_name)

    def get_target_table_name(self) -> str:
        r"""Fully qualified name of the target table. Right now this is only
        supported for link prediction problems.

        Raises:
            NotImplementedError: For all task types other than link prediction.

        Returns:
            str: The target table name.
        """
        if self.task_type.is_link_pred:
            aggr = self.query.get_final_target_aggregation()
            assert isinstance(aggr, Aggregation)
            aggregated_col_name = aggr._get_target_column_name()
            target_table_name = table_name(aggregated_col_name)
            return self.graph.fully_qualified_table_name(target_table_name)
        raise NotImplementedError
