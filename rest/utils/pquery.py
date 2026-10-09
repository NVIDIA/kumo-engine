# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

from kumo_core.model_plan import (
    TrainingTableGenerationPlan,
    _is_inferred,
    _is_missing,
)
from kumo_core.types import TaskType
from kumoapi.model_plan import InferredType, MissingType
from kumoapi.pquery import QueryType

from async_utils.sync import blocking_func
from io_utils import logging
from kumo.config import PredictiveQueryV2Config
from kumo.config.graph import GraphConfig
from kumo.config.model_plan_yaml import model_plan_from_yaml
from kumo.graph import Graph
from kumo.pquery.parser import parse_query_string_v2
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.validation import ValidationException
from monitoring.tracing import TraceOptions, trace

log = logging.getLogger(__name__)


@blocking_func()
def create_pquery_object(config: PredictiveQueryV2Config,
                         **kwargs) -> PredictiveQueryV2:
    return PredictiveQueryV2.from_config(
        config.with_credentials(kwargs.get('credentials', {})), **kwargs)


# `forecast_length` / `lag_timesteps` / `year_over_year` are declared valid
# only for temporal regression and forecasting queries (see the
# `valid_task_types` / `valid_query_types` metadata on
# `TrainingTableGenerationPlan`, which
# `test_guard_matches_the_plan_metadata_it_mirrors` pins this against).
# Passing them on for any other task would push `None` / `False` onto the
# parsed query config instead of leaving its values alone.
_FORECASTING_TASK_TYPES = (TaskType.REGRESSION, TaskType.FORECASTING)


def forecasting_pass_through_plan(
    task_type: TaskType,
    query_type: QueryType,
    *,
    forecast_length: Optional[int],
    lag_timesteps: Optional[int],
    year_over_year: bool,
) -> Optional[TrainingTableGenerationPlan]:
    """Builds the train-table options that carry forecasting parameters into a
    batch-prediction job's query config.

    These options exist purely to hand three values to
    :func:`make_predictive_config_with_graph`; everything else is left at its
    default so the query config keeps whatever the parser produced. Returns
    ``None`` when the task has nothing to pass through, so that callers do not
    report having supplied options they did not.
    """
    if (task_type not in _FORECASTING_TASK_TYPES
            or query_type != QueryType.TEMPORAL):
        return None
    train_options = TrainingTableGenerationPlan()
    train_options.forecast_length = forecast_length
    train_options.lag_timesteps = lag_timesteps
    train_options.year_over_year = year_over_year
    return train_options


def resolve_train_table_options(
    model_plan_yaml: Optional[str],
    train_table_options: Optional[TrainingTableGenerationPlan] = None,
) -> Tuple[TrainingTableGenerationPlan, bool]:
    """Resolves the train-table options that shape a query config.

    Callers hold either raw user-supplied model-plan YAML or, in process,
    already-built options. Returns the options plus whether any were actually
    supplied, as opposed to falling back to the defaults.
    """
    if train_table_options is not None:
        return train_table_options, True
    plan = model_plan_from_yaml(model_plan_yaml or '')
    return plan.training_table_generation, bool(model_plan_yaml)


@blocking_func()
@trace(TraceOptions(trace_params=True))
def make_predictive_config_with_graph(
    name: str,
    query_yaml: str,
    train_options: TrainingTableGenerationPlan,
    graph: Graph,
    graph_config: Optional[GraphConfig] = None,
    list_distinct_target_is_array: Optional[bool] = False,
    lp_bp_layout: Optional[str] = None,
    *,
    train_options_provided: bool,
) -> PredictiveQueryV2Config:
    """Builds the query config, letting the train-table options override what
    the query string parsed to.

    ``train_options_provided`` says whether a caller actually supplied options,
    as opposed to falling back to the defaults. Only the lag-timesteps reset
    below depends on it; getting it wrong silently leaves a forecasting query's
    ``lag_timesteps`` at the parser's value instead of clearing it, so it is
    required rather than defaulted. :func:`resolve_train_table_options` returns
    it alongside the options -- prefer that over computing it by hand.
    """
    query_config: PredictiveQueryV2Config = parse_query_string_v2(
        name, graph, query_yaml, graph_config=graph_config)
    query_config.list_distinct_target_is_array = list_distinct_target_is_array
    query_config.lp_bp_layout = lp_bp_layout
    # NOTE: Don't override the split here since we do special parsing
    plan_forecast_length = None
    if not isinstance(train_options.forecast_length, MissingType):
        plan_forecast_length = train_options.forecast_length

    if (plan_forecast_length is not None and query_config.is_forecasting
            and plan_forecast_length != query_config.forecast_length):
        raise ValidationException(
            title='Conflicting forecast_length', message=(
                f"The model plan specifies forecast_length="
                f"{plan_forecast_length}, but the query syntax specifies "
                f"FORECAST {query_config.forecast_length} TIMEFRAMES. "
                f"Either remove forecast_length from the model plan or "
                f"ensure both values match."))
    elif plan_forecast_length is not None:
        query_config.forecast_length = plan_forecast_length
    if not isinstance(train_options.lag_timesteps, MissingType):
        query_config.lag_timesteps = train_options.lag_timesteps
    elif train_options_provided and query_config.is_forecasting:
        query_config.lag_timesteps = None
    if (not isinstance(train_options.timeframe_step, MissingType)
            and not isinstance(train_options.timeframe_step, InferredType)):
        query_config.timeframe_step = train_options.timeframe_step
    if (not isinstance(train_options.train_start_offset, MissingType) and
            not isinstance(train_options.train_start_offset, InferredType)):
        query_config.train_start_offset = train_options.train_start_offset
    if (not isinstance(train_options.train_end_offset, MissingType) and
            not isinstance(train_options.train_end_offset, InferredType)):
        query_config.train_end_offset = train_options.train_end_offset
    if (not _is_missing(train_options.start_time)
            and not _is_inferred(train_options.start_time)):
        query_config.start_time = train_options.start_time
    if (not _is_missing(train_options.end_time)
            and not _is_inferred(train_options.end_time)):
        query_config.end_time = train_options.end_time
    if not isinstance(train_options.year_over_year, MissingType):
        query_config.year_over_year = train_options.year_over_year
    if train_options.weight_col is not None:
        query_config.weight_col = train_options.weight_col

    return query_config
