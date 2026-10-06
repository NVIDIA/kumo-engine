# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType, ValidatedPredictiveQuery
from kumoapi.pquery.AST import Aggregation, DateOffsetRange, Join
from kumoapi.train import TrainingStage
from kumoapi.typing import AggregationType, TimeUnit

from io_utils import logging
from kumo.config.mixin import construct_object_string_from_config
from kumo.graph import Graph
from kumo.pquery.pquery_validators.train_table_validator import MAX_RHS_ROWS
from kumo.pquery.utils import (
    col_name,
    fqn,
    get_max_train_rows,
    get_rhs_lp_table,
    table_name,
    table_name_with_connector,
)
from kumo.pquery.weight_spec import WeightSpec, validate_weight_spec_format
from kumo.task.split import (
    ApproxDateOffsetSplit,
    BacktestingSplit,
    DateOffsetSplit,
    Split,
    TimeRangeSplit,
)
from kumo.task.splits.time_split import TimeSplit
from kumo.validation import (
    ValidationError,
    ValidationException,
    ValidationResponse,
    ValidationWarning,
)
from kumo.validation.train_table_validation import TrainingTableGenerationError
from service_framework.env_var import EnvVar

logger = logging.getLogger(__name__)

WARN_ON_NUM_TIME_INTERVALS = 100

# Guard against predictive queries that expand into an impractically large
# number of training intervals. Keep changes to this operational limit under
# review because it directly controls generated table size and runtime.
ERROR_ON_NUM_TIME_INTERVALS = EnvVar('ERROR_ON_NUM_TIME_INTERVALS', int,
                                     required=False,
                                     default_value=3000).parse_value()

TRAIN_ROWS_ERROR_LIMIT = 20_000_000_000


class PreTrainTableValidator():
    """Runs any validations that need to be run before the training
    table generation is started.

    Args:
        split (Split): `Split` used by the predictive query.
        query (ValidatedPredictiveQuery): Query used to generate the data.
        graph (Graph): Graph that the query is defined on.
        timeframe_step (int, optional): Timeframe step, if it has been used.
            (default: :obj:`None`)
        train_start_offset (int, optional): Train start offset, if it has been
             used. (default: :obj:`None`)
        train_end_offset (int, optional): Train end offset, if it has been
             used. (default: :obj:`None`)
        start_time (pd.DateOffset | pd.Timestamp, optional): Start time for the
            training data, if specified. Can be either a timestamp or date
            offset. (default: :obj:`None`)
        end_time (pd.DateOffset | pd.Timestamp, optional): End time for the
            training data, if specified. Can be either a timestamp or date
            offset. (default: :obj:`None`)
        task_type (TaskType, optional): The type of task being performed
            (e.g. REGRESSION, CLASSIFICATION, FORECASTING).
            (default: :obj:`None`)

    """
    def __init__(
        self,
        split: Split,
        query: ValidatedPredictiveQuery,
        graph: Graph,
        timeframe_step: Optional[pd.DateOffset] = None,
        train_start_offset: Optional[pd.DateOffset] = None,
        train_end_offset: Optional[pd.DateOffset] = None,
        start_time: Optional[pd.DateOffset | pd.Timestamp] = None,
        end_time: Optional[pd.DateOffset | pd.Timestamp] = None,
        task_type: TaskType | None = None,
    ):
        self.query = query
        self.split = split
        self.graph = graph
        self.timeframe_step = timeframe_step
        self.train_start_offset = train_start_offset
        self.train_end_offset = train_end_offset
        self.start_time = start_time
        self.end_time = end_time
        self.task_type = task_type

    def validate_time_range_split(self) -> ValidationResponse:
        """Validate for TimeRangeSplit, that each split (train, val, test) can
        have at least one aggregation timeframe.

        Returns:
            ValidationResponse: Summarizing errors encountered related to
            incorrect configuration of TimeRangeSplit.
        """
        response = ValidationResponse()
        assert isinstance(self.split, TimeRangeSplit)
        time_range_split: TimeRangeSplit = self.split
        aggr_range = self.query.target_ast.non_inf_date_offset_range()
        assert aggr_range is not None

        # Adding this check for future proofing but this should never happen
        # as long as Kumo APIs are used in the intended sequence since we
        # always convert TimeRangeSplit to the same granularity as target
        # aggregation in Predictive Query.
        if aggr_range.unit != time_range_split.unit:
            # We intentionally enforce matching time units to avoid assumptions
            # like 1 month = 30 days in target aggregation conversion, since
            # otherwise user can specify PREDICT SUM(x.y, 0, 1, MONTH) and we
            # do not know if we should use 30, 31, 28 or 29 days for 1 MONTH.
            response.errors.append(
                ValidationError(
                    'Time unit mismatch error',
                    'Smallest time unit used in PredictiveQuery target '
                    f'aggregation "{aggr_range.unit}" is not same as '
                    'TimeUnit in ModelPlanner split '
                    f'parameter "{time_range_split.unit}". '))

        if aggr_range.is_open:
            # Time range is large enough that it should not cause validation
            # error
            return response

        # Validating that each split definition in TimeRangeSplit has at least
        # enough time units to accommodate one target timeframe.
        start = aggr_range.start_date_offset
        aggr_window_size: pd.DateOffset = aggr_range.end_date_offset - start
        ranges = time_range_split.type_safe_ranges

        assert ranges is not None
        for time_range in ranges:
            if aggr_range.unit == TimeUnit.MONTHS:
                # We do an approximate calculation here to get an estimate
                # of number of months in each time range. This can potentially
                # lead to inaccurate results on boundaries but is a well
                # understood risk.
                split_boundary_size = ((time_range[1] - time_range[0]).days /
                                       (365.25 / 12))
                aggr_size = aggr_window_size.kwds.get('months', 0)
            else:
                seconds_per_unit = {
                    TimeUnit.SECONDS: 1,
                    TimeUnit.MINUTES: 60,
                    TimeUnit.HOURS: 60 * 60,
                    TimeUnit.DAYS: 24 * 60 * 60,
                    TimeUnit.WEEKS: 7 * 24 * 60 * 60,
                }[aggr_range.unit]
                split_boundary_size = (
                    (time_range[1] - time_range[0]).total_seconds() /
                    seconds_per_unit)
                aggr_size = aggr_window_size.kwds.get(aggr_range.unit.value, 0)
            if split_boundary_size < aggr_size:
                response.errors.append(
                    ValidationError(
                        'Incorrect TimeRangeSplit definition',
                        f'The range {time_range} is smaller than the total '
                        f'aggregation window specified {aggr_size} in the '
                        f'Predictive Query',
                    ))
        return response

    def validate_date_offset_split(self) -> ValidationResponse:
        """Validate that the val split is large enough to contain at least
        one aggregation timeframe.

        Returns:
            ValidationResponse: Summarizing errors encountered related to
            incorrect configuration of DateOffsetSplit.
        """
        response = ValidationResponse()
        assert isinstance(self.split, DateOffsetSplit)
        date_offset_split: DateOffsetSplit = self.split

        # Note: We only need to check the val split when 3 splits are specified
        # The test split and the val split when only 2 splits are specified
        # are guaranteed to be large enough
        if date_offset_split.num_splits == 3:
            split_val_time_range: int = (date_offset_split.offsets[1] -
                                         date_offset_split.offsets[0])
            assert self.query.target_timeframe is not None
            timeframe_end_offset = self.query.target_timeframe.end_date_offset
            date_offset = DateOffsetRange._to_dateOffset(
                date_offset_split.offset_unit,
                split_val_time_range,
            )
            if DateOffsetRange.date_offset_gt(
                    timeframe_end_offset,
                    date_offset,
            ):
                timeframe_end_int = list(timeframe_end_offset.kwds.values())[0]
                response.errors.append(
                    ValidationError(
                        'Split size is too small',
                        f'The size of the validation split is too small: '
                        f'{split_val_time_range} '
                        f'{date_offset_split.offset_unit.value} '
                        f'is smaller than the target timeframe: '
                        f'{timeframe_end_int} '
                        f'{self.query.target_timeframe.unit.value}'))
        return response

    def validate_temporal_splits(self) -> ValidationResponse:
        """
        Validate TimeRangeSplit and DateOffsetSplit.

        Returns:
            ValidationResponse: Summarizing errors encountered related to
                incorrect configuration of TemporalSplits. For example, if a
                split is too small compared to target aggregation or if units
                don't match between predictive query aggregation and
                TimeRangeSplit input.
        """
        if self.query.query_type != QueryType.TEMPORAL:
            raise ValueError('Incorrect invocation of '
                             'validate_temporal_splits for non temporal query')
        response = ValidationResponse()
        # Defense in depth: ModelPlan.validate already rejects this, but this
        # is a separate entry point and an uncaught user `TimeSplit` would be
        # silently dropped by the DiskGraph producer.
        if isinstance(self.split, TimeSplit):
            response.errors.append(
                ValidationError(
                    'Unsupported split',
                    'The selected split uses `TimeSplit`, which is an '
                    'internal representation and is not supported as a '
                    'user-supplied split. Use a public split family such as '
                    '`TemporalSplit(ratio=[0.8, 0.1, 0.1])`.'))
            return response
        if isinstance(self.split, TimeRangeSplit):
            return self.validate_time_range_split()
        elif isinstance(self.split, DateOffsetSplit):
            return self.validate_date_offset_split()
        return response

    def validate_weight_column_fqn(
        self,
        weight_spec: WeightSpec,
    ) -> str:
        """Validate weight_col reference and return fully qualified name.

        Args:
            weight_spec: Parsed weight column specification.

        Returns:
            Fully qualified weight column name.

        Raises:
            TrainingTableGenerationError: If weight_col is not in table.column
                form.
        """
        try:
            validate_weight_spec_format(weight_spec)
        except (ValidationException, ValueError) as exc:
            raise TrainingTableGenerationError(
                title="Invalid Weight Column",
                message=str(exc),
            ) from exc
        weight_table = table_name(weight_spec.column)
        weight_column = col_name(weight_spec.column)
        weight_table_fqn = self.graph.fully_qualified_table_name(weight_table)
        weight_col_fqn = fqn(weight_table_fqn, weight_column)
        return weight_col_fqn

    def validate_train_offsets(
        self,
        train_start_offset: Optional[pd.DateOffset],
        train_end_offset: Optional[pd.DateOffset],
    ) -> ValidationResponse:
        r"""Validation of `train_start_offset` and `train_end_offset`."""
        # TODO (vid): Move this to model planer validation
        response = ValidationResponse()
        if (train_start_offset is not None and train_end_offset is not None):
            now = datetime.datetime.now()
            if now + train_end_offset >= now + train_start_offset:
                train_start_offset_repr = (
                    DateOffsetRange.date_offset_repr(train_start_offset))
                train_end_offset_repr = (
                    DateOffsetRange.date_offset_repr(train_end_offset))
                response.errors.append(
                    ValidationError(
                        'Train start/end offsets error',
                        f'Argument train_end_offset should be smaller than '
                        f'train_start_offset, but it is not: '
                        f'train_start_offset={train_start_offset_repr}, '
                        f'train_end_offset={train_end_offset_repr}'))
        return response

    def validate_timestamps(
        self,
        timestamps: List[pd.Timestamp],
        min_date: Optional[pd.Timestamp],
        max_date: Optional[pd.Timestamp],
    ) -> ValidationResponse:
        r"""Validation of the generated timestamps.

        Args:
            timestamps (List[pd.Timestamp]): List of timestamps for which
                train table data is going to be generated.
            min_date (Optional[pd.Timestamp]): The smallest date in training
                tables that can be used to compute labels. Undefined when
                `timestamps` is empty.
            max_date (Optional[pd.Timestamp]): The largest date in training
                tables that can be used to compute labels. Undefined when
                `timestamps` is empty.

        Returns:
            ValidationResponse: Warnings and errors encountered.

        Raises:
            TrainingTableGenerationError: If the list is empty.
        """
        num_time_intervals = len(timestamps)
        if num_time_intervals == 0:
            assert min_date is None
            assert max_date is None

        response = ValidationResponse()
        suggest_msg_list = []
        if num_time_intervals > 0:
            assert min_date is not None
            assert max_date is not None
            time_duration = max_date - min_date
            time_duration_repr = DateOffsetRange.timedelta_repr(
                time_duration, TimeUnit.DAYS)
            target_tables = [
                table_name(aggr._get_target_column_name())
                for aggr in self.query.get_all_target_aggregations()
            ]
            assert len(target_tables) > 0
            target_tables = list(set(target_tables))
            if len(target_tables) == 1:
                target_tables_repr = (
                    f'target table {target_tables[0]} contains')
            else:
                target_tables_repr = (f'target tables {target_tables} contain')
            suggest_msg_list.append(
                f"The {target_tables_repr} {time_duration_repr} of data for "
                f"label generation (start: {min_date}, end: {max_date}).")

        if self.train_start_offset is not None:
            train_start_offset_repr = DateOffsetRange.date_offset_repr(
                self.train_start_offset)
            suggest_msg_list.append((f"Try increasing 'train_start_offset' "
                                     f"(got {train_start_offset_repr})."))
        if self.train_end_offset is not None:
            train_end_offset_repr = DateOffsetRange.date_offset_repr(
                self.train_end_offset)
            suggest_msg_list.append((f"Try decreasing 'train_end_offset' "
                                     f"(got {train_end_offset_repr})."))
        target_timeframe = self.query.target_timeframe
        entity_timeframe = self.query.entity_timeframe
        assert target_timeframe is not None
        if (target_timeframe.start is not None and target_timeframe.start < 0
                or entity_timeframe is not None
                and entity_timeframe.start is not None
                and entity_timeframe.start < 0):
            suggest_msg_list.append(
                "Use '-INF' as the start time of past-looking filters if you "
                "intend to apply the filter on all previous data.")

        offset_range = self.query.get_combined_date_offset_range()
        target_aggregation_start = offset_range.start_date_offset
        assert target_aggregation_start is not None
        target_aggregation_end = offset_range.end_date_offset
        target_aggregation_timeframe = (DateOffsetRange.date_offset_subtract(
            target_aggregation_end, target_aggregation_start))

        target_aggregation_start_repr = DateOffsetRange.date_offset_repr(
            target_aggregation_start)
        target_aggregation_end_repr = DateOffsetRange.date_offset_repr(
            target_aggregation_end)
        target_aggregation_timeframe_repr = (
            DateOffsetRange.date_offset_repr(target_aggregation_timeframe))

        if self.timeframe_step:
            suggest_msg_list.append(
                f"Decrease 'timeframe_step' (currently {self.timeframe_step}).")
        else:
            suggest_msg_list.append(
                f"Use a shorter target aggregation window "
                f"(currently {target_aggregation_timeframe_repr}, "
                f"start {target_aggregation_start_repr}, "
                f"end {target_aggregation_end_repr}).")

        suggest_msg_list.append("Increase available target data if possible.")
        suggest_msg = ' '.join(suggest_msg_list)
        if num_time_intervals == 0:
            response.errors.append(
                ValidationError(
                    "Empty training table",
                    f"No valid timestamps remain to generate training data. "
                    f"{suggest_msg}"))
        if num_time_intervals == 1:
            warn_msg = ("Only one timeframe remains for training, which is "
                        "typically too little to learn a generalizable model. "
                        f"{suggest_msg}")
            logger.warning(warn_msg)
            response.warnings.append(
                ValidationWarning(
                    "The number of training time intervals is too small",
                    warn_msg))

        if num_time_intervals > 0:
            if isinstance(self.split, DateOffsetSplit) or isinstance(
                    self.split, TimeRangeSplit):
                if self.split.num_splits > num_time_intervals:
                    split_repr = construct_object_string_from_config(self.split)
                    response.errors.append(
                        ValidationError(
                            title=f"The number of generated timeframes is too "
                            f"small for the split: {split_repr}.",
                            message=f"{split_repr} needs at least "
                            f"{self.split.num_splits} timeframes, but only "
                            f"{num_time_intervals} are available."))

        if isinstance(self.split, BacktestingSplit):
            if min_date is not None and min_date > timestamps[0]:
                response.warnings.append(
                    ValidationWarning(
                        title='Backtesting timestamp earlier than the data in '
                        'the database.',
                        message=f'Backtesting timestamp {timestamps[0]} is '
                        f'earlier than the earliest timestamp in the data '
                        f'range ({min_date}).'))
            if max_date is not None and max_date < timestamps[0]:
                response.errors.append(
                    ValidationError(
                        title='Backtesting timestamp later than the data in '
                        'the database.',
                        message=f'Backtesting timestamp {timestamps[0]} is '
                        f'later than the latest timestamp in the data '
                        f'range ({max_date}). No labels can be computed.'))
        if num_time_intervals > ERROR_ON_NUM_TIME_INTERVALS:
            error_msg = (
                f"Too many time intervals. Training would cover more than "
                f"{ERROR_ON_NUM_TIME_INTERVALS} (got {num_time_intervals}), "
                f"which is unsupported. Reduce intervals via "
                f"`train_start_offset` and `train_end_offset`.")
            logger.error(error_msg)
            response.errors.append(
                ValidationError(title='Large number of time intervals',
                                message=error_msg))
        elif num_time_intervals > WARN_ON_NUM_TIME_INTERVALS:
            warn_msg = (
                f"Large number of time intervals may slow training "
                f"({num_time_intervals} > {WARN_ON_NUM_TIME_INTERVALS}). "
                f"Trim via `train_start_offset` and `train_end_offset`.")
            logger.warning(warn_msg)
            response.warnings.append(
                ValidationWarning(title='Large number of time intervals',
                                  message=warn_msg))

        return response

    def validate_timestamps_for_forecasting_v2(
        self,
        timestamps: List[pd.Timestamp],
        forecast_length: int,
        lag_timesteps: int,
    ) -> ValidationResponse:
        r"""Validation of the generated timestamps for forecasting
        predictive queries. Returns a ok response if the task type is
        not forecasting. Otherwise it performs a bunch of validations.
        This is supported when _USE_DECOUPLED_SPLITS is True.
        Args:
            timestamps (List[pd.Timestamp]): List of timestamps for which
                train table data is going to be generated.
            forecast_length (int): Number of predictions to
                make in the future.
            lag_timesteps (int): Number of labels to use as features
                from the past.
        """
        response = ValidationResponse()
        assert self.task_type == TaskType.FORECASTING
        assert isinstance(self.split, TimeSplit)
        response.merge(_validate_uniform_timeframes(timestamps, self.split))

        if self.split.fractions is not None:
            response.errors.append(
                ValidationError(
                    title='TimeSplit with fractions not supported',
                    message=('TimeSplit with fractions is not supported for '
                             'forecasting tasks.')))
        if self.split.indices is not None:
            return response
        if self.split.offset_boundaries is not None:
            return response
        assert self.split.timestamp_boundaries is not None

        num_splits = self.split.num_splits
        # Forecasting requires at least
        # lag_timesteps + num_splits * forecast_length
        # timeframes because the train split needs to have enough timestamps
        # before to generate the lag features AND all `num_splits` splits
        # need to have enough data to make one forecast_length forecast.
        if len(timestamps) < lag_timesteps + num_splits * forecast_length:
            response.errors.append(
                ValidationError(
                    title='Not enough timestamps for forecasting',
                    message=f'There are not enough timestamps to generate '
                    f'forecasting data. At least '
                    f'{lag_timesteps + num_splits * forecast_length} '
                    f'timestamps are required for forecasting '
                    f'(got {len(timestamps)}). '
                    f'Try reducing `forecast_length`, `lag_timesteps` or '
                    f'increasing the amount of raw target table data.'))

        # Calculate the number of timeframes in each split
        split_names = TrainingStage.list_stages()[:num_splits]
        num_timeframes_per_split = {stage: 0 for stage in split_names}

        # Get the boundaries from the TimeSplit
        boundaries = self.split.create_timestamp_boundaries()
        assert boundaries is not None

        # For each split, count how many timestamps fall within its boundaries
        for boundary, split_name in zip(boundaries, split_names):
            start_ts, end_ts = boundary
            num_timeframes = len([
                timestamp for timestamp in timestamps
                if start_ts <= timestamp < end_ts
            ])
            num_timeframes_per_split[split_name] = num_timeframes

        for split_name, num_timeframes in num_timeframes_per_split.items():
            min_num_timeframes = (forecast_length + lag_timesteps if split_name
                                  == TrainingStage.TRAIN else forecast_length)
            if num_timeframes < min_num_timeframes:
                response.errors.append(
                    ValidationError(
                        title='Too few timeframes for forecasting',
                        message=f'The split {split_name} has too few '
                        f'timeframes for forecasting. '
                        f'Expected at least {min_num_timeframes} '
                        f'timeframes (got {num_timeframes}).'))

        return response

    def validate_timestamps_for_forecasting(
        self,
        timestamps: List[pd.Timestamp],
        forecast_length: int,
        lag_timesteps: int,
    ) -> ValidationResponse:
        r"""Validation of the generated timestamps for forecasting
        predictive queries. Returns a ok response if the task type is
        not forecasting. Otherwise it performs a bunch of validations.
        Args:
            timestamps (List[pd.Timestamp]): List of timestamps for which
                train table data is going to be generated.
            forecast_length (int): Number of predictions to
                make in the future.
            lag_timesteps (int): Number of labels to use as features
                from the past.
        """
        response = ValidationResponse()
        if self.task_type != TaskType.FORECASTING:
            return response

        response.merge(_validate_uniform_timeframes(timestamps, self.split))
        num_splits = self.split.num_splits
        # Forecasting requires at least
        # lag_timesteps + num_splits*forecast_length
        # timeframes because the train split needs to have enough timestamps
        # before to generate the lag features AND all `num_splits` splits
        # need to have enough data to make one forecast_length forecast.
        if len(timestamps) < lag_timesteps + num_splits * forecast_length:
            response.errors.append(
                ValidationError(
                    title='Not enough timestamps for forecasting',
                    message=f'There are not enough timestamps to generate '
                    f'forecasting data. At least '
                    f'{lag_timesteps + num_splits*forecast_length} '
                    f'timestamps are required for forecasting '
                    f'(got {len(timestamps)}). '
                    f'Try reducing `forecast_length`, `lag_timesteps` or '
                    f'increasing the amount of raw target table data.'))

        if (isinstance(self.split, (DateOffsetSplit, TimeRangeSplit))
                and not isinstance(self.split, ApproxDateOffsetSplit)):
            num_timeframe_split = _get_timeframes_in_split(
                self.split, timestamps)
            if (num_timeframe_split[TrainingStage.TRAIN]
                    < forecast_length + lag_timesteps):
                response.errors.append(
                    ValidationError(
                        title='Not enough timestamps in train split '
                        'for forecasting',
                        message=f'There are not enough timestamps to generate '
                        f'forecasting data in train split. At least '
                        f'{forecast_length + lag_timesteps} timestamps needed '
                        f'for forecasting '
                        f'(got {num_timeframe_split[TrainingStage.TRAIN]}).'))
            for split_name, num_timeframes in num_timeframe_split.items():
                if split_name == TrainingStage.TRAIN:
                    continue
                if num_timeframes < forecast_length:
                    response.errors.append(
                        ValidationError(
                            title='Too few timeframes in for forecasting',
                            message=f'The split {split_name} has too few '
                            f'timeframes for forecasting.'
                            f'Expected at least `forecast_length` '
                            f'timeframes (got {num_timeframes}). '))
        else:
            response.errors.append(
                ValidationError(
                    title='Invalid split type',
                    message='Forecasting tasks are only supported with '
                    f'DateOffsetSplit and TimeRangeSplit got {self.split}.'))

        return response

    def validate_forecasting_task_type(
        self,
        lag_timesteps: int | None = None,
        forecast_length: int | None = None,
    ) -> ValidationResponse:
        r"""Runs some validations if the task type is forecasting.
        Args:
            lag_timesteps (int, optional): The number of lag timesteps.
                Has to be provided if the task type is forecasting.
            forecast_length (int, optional): The number of forecast steps.

        Raises:
            ValueError: If the setup isn't valid

        Returns:
            ValidationResponse: Which is empty if the setup is valid.
                Otherwise it contains errors.
        """

        response = ValidationResponse()
        effective_forecast_length = (forecast_length
                                     if forecast_length is not None else 1)
        if (self.task_type != TaskType.FORECASTING
                and effective_forecast_length == 1):
            if lag_timesteps is not None and lag_timesteps > 0:
                response.errors.append(
                    ValidationError(
                        title='Lag timesteps not supported',
                        message='Lag timesteps are only supported for '
                        'forecasting tasks.'))
            return response
        if lag_timesteps is None or lag_timesteps < 1:
            response.errors.append(
                ValidationError(
                    title='Invalid lag timesteps',
                    message='Lag timesteps are required for forecasting tasks.')
            )
            return response

        if self.query.query_type != QueryType.TEMPORAL:
            response.errors.append(
                ValidationError(
                    title='Invalid query type', message='Forecasting tasks are '
                    'only supported for temporal queries '
                    f'(got {self.query.query_type}).'))

        if not (isinstance(self.query.target_ast, Join)
                and isinstance(self.query.target_ast.rhs_target, Aggregation)):
            response.errors.append(
                ValidationError(
                    title='Invalid target query',
                    message='Invalid target query for forecasting tasks. '
                    f'Target has to be aggregation.'
                    f"(got {self.query.target_ast})."))
        assert isinstance(self.query.target_ast, Join)
        assert isinstance(self.query.target_ast.rhs_target, Aggregation)
        rhs_target = self.query.target_ast.rhs_target
        if rhs_target.aggr not in [
                AggregationType.SUM, AggregationType.COUNT,
                AggregationType.COUNT_DISTINCT
        ]:
            response.errors.append(
                ValidationError(
                    title='Invalid aggregation type',
                    message='Invalid aggregation type for forecasting tasks. '
                    'Supported aggregations are SUM, COUNT, COUNT_DISTINCT.'))
        assert rhs_target.aggr_time_range is not None
        assert not rhs_target.aggr_time_range.is_open
        unit = rhs_target.aggr_time_range.unit
        if rhs_target.aggr_time_range.start_date_offset != pd.DateOffset(
                **{unit.value: 0}):  # type: ignore
            response.errors.append(
                ValidationError(
                    title='Invalid start offset for forecasting',
                    message='Start offset for aggregation'
                    f'should be 0 (got {rhs_target.aggr_time_range.start}).'))
        if self.timeframe_step is not None:
            target_date_range = self.query.get_combined_date_offset_range()
            assert target_date_range.end_date_offset is not None
            assert target_date_range.start_date_offset is not None
            target_timeframe = DateOffsetRange.date_offset_subtract(
                target_date_range.end_date_offset,
                target_date_range.start_date_offset)
            if self.timeframe_step != target_timeframe:
                response.errors.append(
                    ValidationError(
                        title='Invalid timeframe step for forecasting.',
                        message=f'timeframe_step({self.timeframe_step}) '
                        f'should be equal to the timeframe of '
                        f'target aggregation '
                        f'{target_date_range.end_date_offset}-'
                        f'{target_date_range.start_date_offset}.'))
        return response

    def validate_limits(
        self,
        timestamps: List[pd.Timestamp],
    ) -> ValidationResponse:
        r"""Validates any size-related limits. At the moment, it warns the
        user if the number of generated training sample could exceed
        TRAIN_ROWS_ERROR_LIMIT (set to 20B).

        Args:
            timestamps (List[pd.Timestamp]): List of timestamps.

        Returns:
            Optional[str]: Warning message if the number of generated training
                sample could exceed TRAIN_ROWS_ERROR_LIMIT (set to 20B).
        """
        validation_response = ValidationResponse()
        entity_table = table_name(self.query.entity_column)
        n_entities = self.graph.table(entity_table).num_rows(
            compute_table_stats_if_missing=True)
        expected_rows = n_entities * len(timestamps)
        max_used_rows = get_max_train_rows()
        warn_msg = None
        if expected_rows > TRAIN_ROWS_ERROR_LIMIT:
            warn_msg = (
                'Training table may be very large. DiskGraph will generate '
                f'the full physical training table (up to ~{expected_rows:,} '
                'rows after filters). Model training will consume a bounded '
                f'subset of at most {max_used_rows:,} rows. To reduce '
                'training table generation work, filter down the number '
                f'of entities (currently {n_entities:,} in table '
                f'{entity_table}) or reduce the number of timeframes '
                f'(currently {len(timestamps)}) via '
                'train_start_offset/train_end_offset.')
            validation_response.warnings.append(
                ValidationWarning(
                    title="Training table may be very large",
                    message=warn_msg,
                ))
            logger.warning(warn_msg)
        # During training table generation, the RHS table might undergo filtering.
        # However, the trainier still uses the unfiltered RHS table. This is why, here,
        # we check the unfiltered RHS table.
        if self.task_type and self.task_type.is_link_pred:
            rhs_table = get_rhs_lp_table(self.query, self.graph)
            rhs_table_name = table_name_with_connector(rhs_table)
            n_targets = self.graph.table(rhs_table_name).num_rows(
                compute_table_stats_if_missing=True)

            if n_targets > MAX_RHS_ROWS:
                error_msg = (
                    f'Link prediction target table is too large. '
                    f'{rhs_table_name} has {n_targets:,} rows, exceeding the '
                    f'{MAX_RHS_ROWS:,} row limit. Reduce RHS size before '
                    f'training.')
                logger.warning(error_msg)
                validation_response.errors.append(
                    ValidationError(
                        title="Target table is too large to be processed",
                        message=error_msg,
                    ))

        return validation_response

    def validate_train_time_compatibility(self) -> ValidationResponse:
        r"""Validates that start_time/end_time and train offsets are not used
        together.

        Note: This validation can be removed once we fully migrate from using
        train_start_offset/train_end_offset to start_time/end_time parameters.

        Returns:
            ValidationResponse: Response containing any validation errors.
        """
        validation_response = ValidationResponse()

        if ((self.start_time is not None or self.end_time is not None)
                and (self.train_start_offset is not None
                     or self.train_end_offset is not None)):
            validation_response.errors.append(
                ValidationError(
                    title="Incompatible time parameters",
                    message=("Cannot specify both start_time/end_time and "
                             "train_start_offset/train_end_offset. Use either "
                             "absolute/relative timestamps or train offset "
                             "parameters, but not both.")))

        return validation_response

    def validate_train_time(self) -> ValidationResponse:
        r"""Validates start_time and end_time parameters.

        Checks:
        1. Both must be offsets or both timestamps
        2. For timestamps: start_time must be smaller than end_time
        3. For offsets: both must be <= 0 and start_time must be < end_time

        Returns:
            ValidationResponse: Response containing any validation errors.
        """
        validation_response = ValidationResponse()

        if self.start_time is None or self.end_time is None:
            return validation_response

        # Check if both are same type (offset or timestamp)
        is_start_offset = isinstance(self.start_time, pd.DateOffset)
        is_end_offset = isinstance(self.end_time, pd.DateOffset)

        if is_start_offset != is_end_offset:
            validation_response.errors.append(
                ValidationError(
                    title="Inconsistent time parameter types",
                    message=("start_time and end_time must both be offsets "
                             "or both be timestamps, but got mixed types.")))
            return validation_response

        # Validate timestamps
        if not is_start_offset:
            assert isinstance(self.start_time, pd.Timestamp)
            assert isinstance(self.end_time, pd.Timestamp)
            if self.start_time >= self.end_time:
                validation_response.errors.append(
                    ValidationError(
                        title="Invalid timestamp range", message=(
                            f"start_time ({self.start_time}) must be smaller "
                            f"than end_time ({self.end_time})")))
        # Validate offsets
        else:
            assert isinstance(self.start_time, pd.DateOffset)
            assert isinstance(self.end_time, pd.DateOffset)
            now = pd.Timestamp.now()
            start_ts = now + self.start_time
            end_ts = now + self.end_time

            if start_ts <= now and end_ts <= now:
                if start_ts >= end_ts:
                    validation_response.errors.append(
                        ValidationError(
                            title="Invalid offset range", message=(
                                f"For relative offsets, start_time must be "
                                f"more negative than end_time, but got "
                                f"start_time={self.start_time}, "
                                f"end_time={self.end_time}")))
            else:
                validation_response.errors.append(
                    ValidationError(
                        title="Invalid offset values", message=(
                            f"Relative offsets must be <= 0 (i.e. in the "
                            f"past), but got start_time={self.start_time}, "
                            f"end_time={self.end_time}")))

        return validation_response


def _validate_uniform_timeframes(timestamps: List[pd.Timestamp],
                                 split: Split) -> ValidationResponse:
    r"""Validates that the timestamps are uniformly spaced. This
    is a requirement for forecasting tasks, where lag features
    for a given timestamp are computed from the previous timestamps,
    and non-uniform timeframes can lead to missing lag features.
    Args:
        timestamps (List[pd.Timestamp]): List of timestamps.
        split (Split): split for train table. The split effects
            generated timestamps and can cause non-uniform timeframes,
            like in the case of TimeRangeSplit.
    Returns:
        ValidationResponse: Warnings and errors encountered.
    """
    response = ValidationResponse()
    time_diffs = [
        timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))
    ]
    if len(set(time_diffs)) > 1:
        suggestion = ""
        if isinstance(split, TimeRangeSplit):
            suggestion += ("Potentially due to TimeRangeSplit configuration. "
                           "Make sure the start of each split matches the "
                           "end of the previous split.")
        response.errors.append(
            ValidationError(
                title='Non-uniform timeframes in forecasting',
                message=('Timestamps are not uniformly spaced, so lag features '
                         'will be missing. '
                         f'{suggestion.strip()}')))
    return response


def _get_timeframes_in_split(
        split: Union[DateOffsetSplit, TimeRangeSplit],
        timestamps: List[pd.Timestamp]) -> Dict[TrainingStage, int]:
    r"""Returns the number of timestamps in each split.
    Args:
        split (Union[DateOffsetSplit, TimeRangeSplit]): The split.
        timestamps (List[pd.Timestamp]): List of timestamps.
    Returns:
        Dict[str, int]: Number of timeframes in each split.
    """
    split_names = TrainingStage.list_stages()
    num_timeframes_per_split = {stage: 0 for stage in split_names}
    if len(timestamps) == 0:
        # To prevent errors below in min and max
        logger.warning(f"No timestamps found for split {split}. "
                       f"This is likely due to an empty training table.")
        return num_timeframes_per_split

    if isinstance(split, DateOffsetSplit):
        ranges = split.get_adjusted_boundaries(max(timestamps), min(timestamps))
    elif isinstance(split, TimeRangeSplit):
        ranges = split.ranges
    # For each split identify which timestamp(anchor time)
    # falls in that split.
    for range_, split_name in zip(ranges, split_names):
        num_timeframes = len([
            timestamp for timestamp in timestamps
            if range_[0] <= timestamp < range_[1]
        ])
        num_timeframes_per_split[split_name] = num_timeframes
    return num_timeframes_per_split
