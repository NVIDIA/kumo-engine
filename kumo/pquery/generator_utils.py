# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import pandas as pd
from kumo_core.pquery import resolve_task_type
from kumo_core.training.timestamps import default_target_time_range
from kumo_core.types import ColStatType, TaskType, TaskTypeWithReason
from kumoapi.pquery import QueryType, ValidatedPredictiveQuery
from kumoapi.pquery.AST import Aggregation
from kumoapi.typing import AggregationType, Dtype, ProblemType, Stype

from io_utils import logging
from kumo.graph import ForeignKey, Graph, TimeColumn
# Re-exported for kumo.data.generic.diskgraph_batch_pred.
from kumo.graph.utils import map_to_api_graph_definition  # noqa: F401
from kumo.pquery.utils import col_name, table_name
from kumo.stats import is_missing_stat_value
from kumo.validation.train_table_validation import TrainingTableGenerationError

logger = logging.getLogger(__name__)


def _time_stat_to_timestamp(value) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)


def get_query_type(query: ValidatedPredictiveQuery) -> QueryType:
    if query.target_timeframe is not None:
        return QueryType.TEMPORAL
    return QueryType.STATIC


def get_task_type_from_target_column(
    target_col,
    problem_type: Optional[ProblemType] = None,
) -> TaskTypeWithReason:
    task_type, _ = resolve_task_type(
        target_is_column=True,
        target_dtype=target_col.dtype,
        target_stype=target_col.stype,
        problem_type=problem_type,
        query_type=QueryType.STATIC,
    )
    if target_col.dtype == Dtype.bool:
        reason = ('The target clause is a column in your graph '
                  'and it has a data type of Boolean.')
    elif target_col.stype == Stype.numerical:
        reason = ('The target clause is a column in your graph '
                  'and it has a Numerical semantic type.')
    elif target_col.stype == Stype.categorical:
        reason = ('The target clause is a column in your graph '
                  'and it has a Categorical semantic type.')
    elif target_col.stype == Stype.multicategorical:
        problem = ('RANK'
                   if task_type == TaskType.MULTILABEL_RANKING else 'CLASSIFY')
        reason = ('The target clause is a column in your graph, '
                  'it has a semantic type of Multicategorical, '
                  f'and the {problem} problem was selected.')
    else:
        reason = ('The target clause is a column in your graph, and '
                  'your target has ID semantic type. ')
    return TaskTypeWithReason(task_type=task_type, reason=reason)


def get_task_type_from_aggr(
    graph: Graph,
    query: ValidatedPredictiveQuery,
    aggr: Aggregation,
    forecast_length: int = 1,
) -> TaskTypeWithReason:
    aggregated_col_name = aggr._get_target_column_name()
    aggregated_col = None
    if col_name(aggregated_col_name) == '*':
        assert aggr.aggr == AggregationType.COUNT
        aggregation_target_dtype = None
        aggregation_is_wildcard = True
    else:
        aggregated_col = graph.column_from_fqn(aggregated_col_name)
        aggregation_target_dtype = aggregated_col.dtype
        aggregation_is_wildcard = False
    task_type, _ = resolve_task_type(
        target_is_column=False,
        target_dtype=query.target_dtype,
        target_stype=query.target_stype,
        problem_type=query.problem_type,
        query_type=query.query_type,
        forecast_length=forecast_length,
        aggregation_type=aggr.aggr,
        aggregation_target_dtype=aggregation_target_dtype,
        aggregation_is_wildcard=aggregation_is_wildcard,
        aggregation_target_is_foreign_key=isinstance(aggregated_col,
                                                     ForeignKey),
        rhs_node_type=(aggregated_col.dst_table if isinstance(
            aggregated_col, ForeignKey) else None),
    )
    if aggregation_is_wildcard:
        reason = (
            'The target clause is a COUNT aggregation with a wildcard column '
            'and a forecast length was specified.'
            if task_type == TaskType.FORECASTING else
            'The target clause is a numerical COUNT aggregation.')
    elif (aggr.aggr == AggregationType.LIST_DISTINCT
          and isinstance(aggregated_col, ForeignKey)):
        if task_type == TaskType.MULTILABEL_CLASSIFICATION:
            reason = (
                'The target clause is a LIST_DISTINCT aggregation on a '
                'column that acts as a foreign key and the CLASSIFY problem '
                'type was selected.')
        elif task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            reason = (
                'The target clause is a LIST_DISTINCT aggregation on a '
                'column that acts as a foreign key and the RANK problem type '
                'was selected.')
        else:
            reason = (
                'The target clause is a LIST_DISTINCT aggregation on a '
                'column that acts as a foreign key without a time window and '
                'the RANK problem type was selected.')
    elif query.target_stype in [Stype.categorical, Stype.ID]:
        reason = ('The target clause is an aggregation on '
                  'a column with a Categorical or ID type.')
    elif query.target_stype == Stype.multicategorical:
        problem = ('RANK'
                   if task_type == TaskType.MULTILABEL_RANKING else 'CLASSIFY')
        reason = ('The target clause is an aggregation on a column with a '
                  f'Multicategorical type and the {problem} problem type was '
                  'selected.')
    elif aggregated_col.dtype == Dtype.bool:
        if aggr.aggr in [
                AggregationType.COUNT,
                AggregationType.COUNT_DISTINCT,
                AggregationType.SUM,
                AggregationType.AVG,
        ]:
            suffix = (' and a forecast length was specified.'
                      if task_type == TaskType.FORECASTING else '.')
            reason = (f'The target clause is a {aggr.aggr.value} aggregation'
                      f'{suffix}')
        else:
            reason = ('The target clause is an aggregation on '
                      'a boolean column.')
    elif task_type == TaskType.FORECASTING:
        reason = ('The target clause is a numerical aggregation '
                  'and a forecast length was specified.')
    else:
        reason = (
            'The target clause is a numerical aggregation (SUM, COUNT, etc).')
    return TaskTypeWithReason(task_type=task_type, reason=reason)


def get_task_type(
    graph: Graph,
    query: ValidatedPredictiveQuery,
    forecast_length: int = 1,
) -> TaskTypeWithReason:
    r"""Returns the :class:`TaskType` of the :class:`~kumo.task.Task`."""
    assert query.target_stype not in [
        Stype.text,
        Stype.timestamp,
        Stype.sequence,
        Stype.unsupported,
    ]
    # Top node is condition or logical operation
    if (query.target_dtype == Dtype.bool
            and query.target_stype in [Stype.categorical, Stype.ID]):
        task_type, _ = resolve_task_type(
            target_is_column=False,
            target_dtype=query.target_dtype,
            target_stype=query.target_stype,
            problem_type=query.problem_type,
            query_type=query.query_type,
        )
        reason = ('The target clause is a condition or logical operation, '
                  'and is an ID or Categorical Boolean')
        return TaskTypeWithReason(task_type=task_type, reason=reason)

    # The label can now only come from an aggregation or a column.
    aggr = query.get_final_target_aggregation()
    target_col = query.get_final_target_column()

    if target_col is not None:
        assert aggr is None
        assert get_query_type(query) == QueryType.STATIC
        return get_task_type_from_target_column(target_col, query.problem_type)

    # At this stage, the final operation is guaranteed to be an aggregation
    assert aggr is not None
    return get_task_type_from_aggr(graph, query, aggr, forecast_length)


def get_default_target_time_range(
    graph: Graph, query: ValidatedPredictiveQuery
) -> Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timedelta]]:
    r"""Determine the total time range that can be used to generate
        targets. Unless overriden with `train_start_offset` or
        `train_end_offset`, target data will be taken from this range.

        Returns:
            Tuple[pd.Timestamp, pd.Timestamp, Optional[pd.Timedelta]]: A tuple
                of min date, max date, and time delta caused by filter buffer,
                (:obj:`None` if not used or 0) respectively.
        """
    target_aggrs = query.get_all_target_aggregations()
    if len(target_aggrs) == 0:
        # Should not happen
        raise ValueError('Cannot compute target time range because target '
                         'does not have any aggregations.')
    # Of all aggregations that appear in the label computation, we take the
    # latest min date and latest max date. For example, if the query is
    # COUNT(sessions.*, 0, 7) > 10 AND SUM(transactions.value, 2, 5) > 20,
    # and sessions run from 1 Mar to 10 Dec, while transactions run from
    # 1 Jan to 12 Dec, we will take the range between 1 Mar and 12 Dec.
    # This is because we assume (illustrated on the above example) that the
    # final 2 days of sessions are missing because there simply weren't
    # any sessions, but the first 2 months are missing because the data
    # isn't available.
    bounds = []
    for aggr in target_aggrs:
        assert aggr.time_col is not None
        target_table_name = table_name(aggr.time_col)
        time_col = graph.table(target_table_name).time_col
        assert isinstance(time_col, TimeColumn)

        stats = graph.table(target_table_name).compute_col_stats(
            [time_col],  # type: ignore
            [ColStatType.MIN, ColStatType.MAX],
            approx=True,
        )[time_col.name]
        _min = stats[ColStatType.MIN]
        _max = stats[ColStatType.MAX]
        assert _min is not None
        assert _max is not None
        if (is_missing_stat_value(_min.value)
                or is_missing_stat_value(_max.value)):
            # Either the table is fully empty or there was an error in exact
            # stats computation.
            raise TrainingTableGenerationError(
                title="Failed to compute target time range",
                message=("There was an error getting the time ranges for the "
                         f"{time_col.name} column in the {target_table_name} "
                         "table. Check that the table is not empty, the time "
                         "column has non-null values, and column stats are "
                         "available."),
            )
        _min = _time_stat_to_timestamp(_min.value)
        _max = _time_stat_to_timestamp(_max.value)
        bounds.append((_min, _max))
    date_range_no_filters = query.get_combined_date_offset_range()

    # We ignore -INF ranges when determining the starting offset
    entity_range_with_filters = query.entity_ast.non_inf_date_offset_range()
    target_range_with_filters = query.target_ast.non_inf_date_offset_range()
    assert target_range_with_filters is not None
    return default_target_time_range(
        bounds,
        target_range=date_range_no_filters,
        entity_filter_range=entity_range_with_filters,
        target_filter_range=target_range_with_filters,
    )
