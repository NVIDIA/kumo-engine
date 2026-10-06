# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kumo_core.train.autoregression import (
    ONE_YEAR_AGO_OFFSET_DAYS,
    PADDING_SIZE,
)
from kumo_core.types import TableStatType
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import Aggregation, ASTNode
from kumoapi.source_table import CustomSQLSourceTable, SourceTableType
from kumoapi.typing import Dtype
from pyarrow.types import is_decimal, is_floating, is_integer, is_list

from async_utils.sync import sync_to_async
from common_utils.error import PublicException
from io_utils import fsutil, logging, parquet_util
from io_utils.pyarrow import arrow_cloud_filesystem
from io_utils.retry import retry
from kumo.config.mixin import construct_object_string_from_config
from kumo.connector.connector import Connector
from kumo.graph import ForeignKey, Graph
from kumo.graph.column import BaseColumn
from kumo.validation import (
    ValidationError,
    ValidationException,
    ValidationResponse,
)

logger = logging.getLogger(__name__)


class UniqueIdGenerator():
    """
    This class is used to generate unique id for tracing purpose.
    """
    def __init__(self):
        self.node_cnt: int = 0

    def generate(self, prefix: str) -> str:
        """
        Generate <prefix><id>
        """
        self.node_cnt += 1
        return f'{prefix}{str(self.node_cnt)}'


def col_name(full_name: str) -> str:
    r""""Given a fully qualified column name in the format of
    `<table_name>`.`<col_name>`, returns the column name.

    Args:
        full_name (str): Fully qualified column name in the format of
            `<table_name>`.`<col_name>`.

    Returns:
        str: column name.
    """
    return full_name.split('.')[-1]


def table_name(full_name: str) -> str:
    r""""Given a fully qualified column name in the format of
    `<table_name>`.`<col_name>`, returns the table name.

    Args:
        full_name (str): Fully qualified column name in the format of
            `<table_name>`.`<col_name>`.

    Returns:
        str: table name.
    """
    if len(full_name.split('.')) < 2:
        raise ValidationException(
            title="Invalid column reference",
            message=(f'"{full_name}" object has no table name. '
                     "Use the format `<table>.<column>`."),
        )
    return '.'.join(full_name.split('.')[:-1])


def table_name_with_connector(connector_and_table_name: str) -> str:
    r""""Given a fully qualified table name in the format of
    `<connector_id>`.`<table_name>`, returns the table name.

    Args:
        full_name (str): Fully qualified table name in the format of
            `<connector_id>`.`<table_name>`.

    Returns:
        str: table name.
    """
    return connector_and_table_name.split('.')[-1]


def fqn(table_name: str, col_name: str) -> str:
    r"""Given a table name and column name, returns a fully qualified column
    name as `<table_name>`.`<col_name>`.

    Args:
        table_name (str): A table name.
        col_name (str): A column name.
    Returns:
        (str): Fully qualified column name in the format of
            `<table_name>`.`<col_name>`.
    """
    return f'{table_name}.{col_name}'


def get_rhs_lp_table(query: ValidatedPredictiveQuery, graph: Graph,
                     no_connector: bool = False) -> str:
    r"""Given a link prediction query, returns a fully qualified
    name `connector.table` for the RHS table."""
    aggr = query.get_final_target_aggregation()
    assert isinstance(aggr, Aggregation)
    aggregated_col_name = aggr._get_target_column_name()
    aggregated_col = graph.column_from_fqn(aggregated_col_name)
    assert isinstance(aggregated_col, ForeignKey)
    dst_table = aggregated_col.dst_table
    assert isinstance(dst_table, str)
    if no_connector:
        return dst_table.split('.')[-1]
    else:
        return dst_table


def get_rhs_lp_table_pkey(query: ValidatedPredictiveQuery, graph: Graph,
                          no_connector: bool = False) -> str:
    r"""Given a link prediction query, returns a fully qualified
    name `connector.table.pkey` or `table.pkey` for the RHS table.
    """
    aggr = query.get_final_target_aggregation()
    assert isinstance(aggr, Aggregation)
    aggregated_col_name = aggr._get_target_column_name()
    aggregated_col = graph.column_from_fqn(aggregated_col_name)
    assert isinstance(aggregated_col, ForeignKey)
    dst_table = aggregated_col.dst_table
    assert isinstance(dst_table, str)
    rhs_table = graph.table(dst_table)
    pkey = rhs_table.pkey
    assert pkey is not None
    if no_connector:
        return fqn(dst_table.split('.')[-1], pkey.name)
    else:
        return fqn(dst_table, pkey.name)


def get_max_train_rows() -> int:
    r"""Returns the max number of train rows. The behaviour depends on the
    platform.

    Returns:
        int: The max number of rows in the training table.
    """
    from kumo.pquery.pquery_validators.train_table_validator import (
        MAX_TRAIN_ROWS,
    )

    return MAX_TRAIN_ROWS


def _add_node_config_to_output(node: ASTNode, level: int,
                               output: List[str]) -> None:
    indent = "___" * level
    output.append(indent + repr(node))
    _node_config = \
        construct_object_string_from_config(node)
    output.append(indent + "__+" + _node_config)


def trace_ast_output(node: Optional[ASTNode], level: int = 1) -> List[str]:
    """
    Return a list of output string, line by line,
      with ast information

    Attributes:
        node (ASTNode): given from validated query
        level (int = 1): starting indent for first level
    """
    if not node:
        return [repr(node)]

    output = []

    def _trace(node: ASTNode, level: int, output: List[str]):
        _add_node_config_to_output(node, level, output)

        for child in node.children:
            _trace(child, level + 2, output)

    _trace(node, level, output)
    return output


def get_lag_timestamps(start_time: pd.Timestamp, num_lags: int,
                       step: pd.Timedelta) -> List[pd.Timestamp]:
    """starting from start_time, generate a list of num_lags timestamps
    with step size between each timestamp. The list goes from the most old
    to the most recent timestamp."""
    out = [start_time - step * i for i in range(1, num_lags + 1)]
    out.reverse()
    return out


def get_yoy_timestamps(
    pred_time: pd.Timestamp,
    forecast_length: int,
    window: pd.Timedelta,
) -> Tuple[List[pd.Timestamp], int]:
    """starting from start_time, generate a list of num_lags timestamps
    with step size between each timestamp. The list goes from the most old
    to the most recent timestamp."""
    yoy_length = forecast_length + 2 * PADDING_SIZE
    yoy_offset = pd.Timedelta(days=ONE_YEAR_AGO_OFFSET_DAYS)
    assert isinstance(yoy_offset, pd.Timedelta)
    yoy_timeframes = [(pred_time - yoy_offset) + (window * i) for i in range(
        -PADDING_SIZE,
        forecast_length + PADDING_SIZE,
    )]
    return yoy_timeframes, yoy_length


T = TypeVar('T')


def none_or_default(value: Optional[T], default: T) -> T:
    return value if value is not None else default


def validate_forecasting_parameters_consistency(
    training_forecast_length: Optional[int] = None,
    training_lag_timesteps: Optional[int] = None,
    training_year_over_year: Optional[bool] = None,
    prediction_forecast_length: Optional[int] = None,
    prediction_lag_timesteps: Optional[int] = None,
    prediction_year_over_year: Optional[bool] = None,
) -> ValidationResponse:
    response = ValidationResponse()

    # Make sure the parameters are the same
    training_foreacast_length = none_or_default(training_forecast_length, 1)
    prediction_foreacast_length = none_or_default(prediction_forecast_length, 1)
    if not (training_foreacast_length == prediction_foreacast_length):
        response.merge(
            ValidationResponse(errors=[
                ValidationError(
                    title='Forecast length mismatch', message=(
                        f"The forecast length in the training job "
                        f"({training_foreacast_length}) "
                        f"does not match the forecast length in the "
                        f"prediction job ({prediction_foreacast_length})"))
            ]))
    training_lag_timesteps = none_or_default(training_lag_timesteps, 0)
    prediction_lag_timesteps = none_or_default(prediction_lag_timesteps, 0)
    if not (training_lag_timesteps == prediction_lag_timesteps):
        response.merge(
            ValidationResponse(errors=[
                ValidationError(
                    title='Lag timesteps mismatch', message=(
                        f"The lag timesteps in the training job "
                        f"({training_lag_timesteps}) "
                        f"does not match the lag timesteps in the "
                        f"prediction job ({prediction_lag_timesteps})"))
            ]))

    training_year_over_year = none_or_default(training_year_over_year, False)
    prediction_year_over_year = none_or_default(prediction_year_over_year,
                                                False)
    if not (training_year_over_year == prediction_year_over_year):
        response.merge(
            ValidationResponse(errors=[
                ValidationError(
                    title='Year over year mismatch', message=(
                        f"The year over year parameter in the training job "
                        f"({training_year_over_year}) "
                        f"does not match the year over year parameter in the "
                        f"prediction job ({prediction_year_over_year})"))
            ]))
    return response


@retry(tries=9, delay=1, backoff=2, logger=logger)
def get_train_table_schema(train_table_path: str) -> pa.Schema:
    """
    Get the schema of a parquet table.

    Args:
        train_table_path (str): Path to a parquet file or partitioned parquet
            directory.

    Returns:
        pyarrow.Schema: Schema of the parquet table

    Raises:
        NotImplementedError: If the path is not local or s3
    """
    # Native TT-NG publications select one parquet file. Custom inputs may be
    # directories containing parquet parts.
    if train_table_path.endswith('parquet') and not fsutil.isdir(
            train_table_path):
        single_file = train_table_path
    else:
        entries = fsutil.listdir(train_table_path, suffix='parquet')
        single_file = next(
            (path for path in entries
             if path.endswith('parquet') and not fsutil.isdir(path)),
            None,
        )
    if single_file is None:
        raise ValidationException(
            title="Missing training table data",
            message=(f"No parquet files were found directly under "
                     f"{train_table_path}."),
        )
    if fsutil.is_ucv(single_file) or fsutil.is_adls(single_file):
        with fsutil.TemporaryDirectory() as temp_dir:
            fsutil.copy_file(single_file, temp_dir)
            dataset: pq.ParquetDataset = pq.ParquetDataset(temp_dir)
    elif fsutil.is_s3(single_file) or fsutil.is_gcs(single_file):
        # Normalize s3a first so the strip and filesystem both see s3://.
        single_file = fsutil.s3a_as_s3(single_file)
        no_prefix = single_file.removeprefix("s3://").removeprefix("gs://")
        dataset = pq.ParquetDataset(
            no_prefix,
            filesystem=arrow_cloud_filesystem(single_file),
        )
    elif fsutil.is_local(single_file):
        dataset = pq.ParquetDataset(single_file)
    else:
        raise ValidationException(
            title="Unsupported train table path",
            message=(
                "Operation only supports local, S3, and GCS paths. "
                "Use a local path, `s3://...`, or `gs://...` parquet dataset."),
        )

    return dataset.schema


def validate_weight_column(weight_col_name: str, train_table_path: str,
                           task_type,
                           list_distinct_target_is_array: bool) -> BaseColumn:
    """
    Validates a weight column exists and has the appropriate data type.

    Args:
        weight_col_name (str): Name of the weight column to validate
        train_table_path (str): Path to partionioned train table
            parquet.
        task_type: Task type. weight column needs to be float or int
            unless task_type is ranking and `list_distinct_target_is_array` is
            True,in which case weight column needs to be float or int array.
        list_distinct_target_is_array (bool): True if target is an array.

    Returns:
        BaseColumn: Weight column with the appropriate dtype

    Raises:
        ValidationException: If weight column is not found or has invalid data
            type
    """
    from kumo_core.training.options import allow_array_targets
    logger.debug(
        "validate_weight_column: task_type=%s has_array_target=%s "
        "list_distinct_target_is_array=%s allow_array_targets=%s "
        "weight_col_name=%s train_table_path=%s",
        task_type,
        getattr(task_type, 'has_array_target', None),
        list_distinct_target_is_array,
        allow_array_targets(),
        weight_col_name,
        train_table_path,
    )
    schema = get_train_table_schema(train_table_path)
    schema_dict = {field.name: field.type for field in schema}
    logger.debug(
        "validate_weight_column: schema columns=%s",
        {
            name: str(dtype)
            for name, dtype in schema_dict.items()
        },
    )
    if weight_col_name not in schema_dict:
        raise ValidationException(
            title="Invalid Weight Column",
            message=(f"Weight column '{weight_col_name}' not found in "
                     "training table."),
        )

    weight_dtype = None
    weight_col_type = schema_dict[weight_col_name]
    if task_type.has_array_target and list_distinct_target_is_array:
        if not is_list(weight_col_type):
            raise ValidationException(
                title="Invalid Weight Column",
                message=(f"Weight column '{weight_col_name}' must be an "
                         "array type since target is array type "
                         f"(got {weight_col_type})."),
            )
        weight_dtype = Dtype.floatlist
    else:
        if not (is_integer(weight_col_type) or is_floating(weight_col_type)
                or is_decimal(weight_col_type)):
            raise ValidationException(
                title="Invalid Weight Column",
                message=(
                    f"Weight column '{weight_col_name}' must be a "
                    f"numerical type (float or int) (got {weight_col_type})."),
            )
        weight_dtype = Dtype.float
    return BaseColumn(name=weight_col_name, dtype=weight_dtype)


async def validate_weight_table_num_rows(
    connector: Connector,
    source_table: SourceTableType,
    train_table_path: str,
) -> ValidationResponse:
    from rest.config.table_v2 import BigQuerySourceTable, S3SourceTable

    logger.info("Executing validate_weight_table_num_rows.")
    get_parquet_metadata = sync_to_async(parquet_util.get_parquet_metadata,
                                         executor=None)
    get_table_stats = sync_to_async(connector.get_table_stats, executor=None)

    validation = ValidationResponse()
    tt_metadata = None
    wtt_num_rows = None
    try:
        tt_metadata = await get_parquet_metadata(train_table_path)
        logger.info("Successfully fetched training table metadata.")
    except Exception as e:
        error = PublicException(
            message=("Kumo failed to fetch metadata for train table with "
                     f"path {train_table_path}. Check that the path exists and "
                     "the connector has read permissions."),
            error=e,
        ).message
        validation.errors.append(
            ValidationError(
                title="Failed to fetch original training table.",
                message=error,
            ))
        return validation

    if isinstance(source_table, BigQuerySourceTable) or isinstance(
            source_table, S3SourceTable):
        table_name = source_table.table_name
    else:
        table_name = source_table.table

    try:
        if isinstance(source_table, CustomSQLSourceTable):
            get_query_num_rows = sync_to_async(connector.get_query_num_rows,
                                               executor=None)
            wtt_num_rows = await get_query_num_rows(source_table.sql)
        else:
            wtt_stats = await get_table_stats(table_name,
                                              [TableStatType.NUM_ROWS])
            wtt_num_rows = wtt_stats[TableStatType.NUM_ROWS].value
        logger.info(f"Successfully fetched weighted training table stats. "
                    f"Number of rows: {wtt_num_rows}")
    except Exception as e:
        error = PublicException(
            message="Kumo failed to fetch num rows for weighted "
            f"train table with '{table_name}'. Check that the table exists "
            "and the connector has read permissions.",
            error=e,
        ).message
        validation.errors.append(
            ValidationError(
                title="Failed to get number of rows.",
                message=error,
            ))
        return validation

    assert tt_metadata is not None and wtt_num_rows is not None
    if tt_metadata.num_rows != wtt_num_rows:
        error = PublicException(message=(
            "Your weighted training table and original training table don't "
            "contain the same number of rows. "
            f"(Train table number of rows == {tt_metadata.num_rows}. "
            f"Weighted train table number of rows == {wtt_num_rows}.) "
            "Please double check that the joins performed to add a weight "
            "column to the train table are one-to-one and not filtering "
            "rows.")).message
        validation.errors.append(
            ValidationError(
                title="Number of rows doesn't match",
                message=error,
            ))
    else:
        logger.info("Row count validation passed: both tables have the same "
                    f"number of rows ({tt_metadata.num_rows})")
    return validation


def validate_int(const: int, min_int: int | None = None,
                 max_int: int | None = None) -> ValidationResponse:
    r"""Validate that size of :obj:`const` is within the supported
    limits.

    Args:
        const (int): Evaluated const.
        min_int (int, optional): Minimum permitted int. If :obj:`None`,
            minimum of int64 is used. (default: :obj:`None`)
        max_int (int, optional): Maximum permitted int. If :obj:`None`,
            maximum of int64 is used. (default: :obj:`None`)

    Returns:
        ValidationResponse: List of encountered errors.
    """
    if min_int is None:
        min_int = np.iinfo(np.int64).min
    if max_int is None:
        max_int = np.iinfo(np.int64).max
    response = ValidationResponse()
    if const > max_int or const < min_int:
        response.errors.append(
            ValidationError(
                title='Unsupported constant',
                message=f'Constant {const} is outside of the range of '
                f'supported integers ({min_int}, {max_int}).'))
    return response


def _allow_array_targets_for_query(query_string: Optional[str]) -> bool:
    r"""Whether ``LIST_DISTINCT`` should produce an array target for this query.

    Array targets are the link-prediction-only mode: kumopql arrays a
    ``LIST_DISTINCT`` target iff this flag is set, which is correct for a RANK
    (link-prediction) query but wrong for a CLASSIFY (multilabel) query -- there
    the target must stay multicategorical, else a non-string/non-numeric label
    (e.g. a date column) is mis-typed as an unpredictable ``ArrayDtype`` and
    rejected. So gate the flag on the query being link-prediction (RANK).

    ``query_string`` is :obj:`None` in non-query contexts (e.g. graph-only
    formatting); there we fall back to the module capability flag.
    """
    from kumo_core.training.options import allow_array_targets
    if query_string is None:
        return allow_array_targets()
    from kumoapi.typing import ProblemType
    from kumopql.parser import PQLParser
    try:
        parsed = PQLParser().to_parsed_predictive_query(query_string)
    except Exception:
        # Let the validator surface the parse error; default to the capability
        # flag so we do not mask it here.
        return allow_array_targets()
    return parsed.problem_type == ProblemType.RANK


def validator_kwargs(
    graph: Graph,
    query_string: Optional[str] = None,
) -> dict[str, ...]:
    r"""A helper method to generate keyword args for PredictiveQueryValidator.
    To avoid scattering env variables around multiple modules, they need to be
    passed as arguments. Pass ``query_string`` so array-target mode is decided
    per query (link-prediction only); omit it in graph-only contexts."""
    from kumo.graph.utils import map_to_api_graph_definition
    graph_def = map_to_api_graph_definition(graph)
    return {
        'graph': graph_def,
        'allow_array_targets': _allow_array_targets_for_query(query_string),
        'allow_timestamp_arrays': False,
    }
