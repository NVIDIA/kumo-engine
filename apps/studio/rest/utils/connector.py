# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from contextlib import contextmanager
from typing import Iterator, List

import pandas
from cachetools import TTLCache
from kumo_core.sources.preview import stringify_sample_dataframe  # noqa
from kumo_core.training.options import allow_array_targets
from kumo_core.types import ConnectorType
from kumo_core.types.schema import BaseColumnConfig
from kumoapi.typing import Dtype, Stype

from async_utils.sync import blocking_func
from io_utils import logging
from kumo.config.graph import TableMetadataConfig, TableOrViewMetadataConfig
from kumo.connector.connector import NUM_SAMPLE_ROWS, Connector
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.data.custom_sql import (
    advisory_read_only_check,
    sample_over_budget,
    subquery_safe_sql,
    validate_connector_sql_scope,
)
from kumo.graph.table import Table
from kumo.validation import ValidationResponse, ValidationWarning
from monitoring.tracing import TraceOptions
from rest.config.connector import SampleRowsResponse, SourceColumnsResource
from statedb.config.common import SourceColumn
from statedb.utils.errors import (
    ConnectorValidationError,
    CustomSqlQueryError,
    sql_query_error_message,
)

log = logging.getLogger(__name__)

# The default number of rows to return from /connector/id/sample_rows
DEFAULT_NUM_SAMPLE_ROWS = 10

# The max sample request hold in the cache and their ttl.
_MAX_SAMPLE_REQ_CACHE_ITEMS = 8
_CACHE_TTL_SECONDS = 300  # 300 seconds

# In memory cache for fetching table sample async.
_IN_PROGRESS_GET_SAMPLE_ROWS = TTLCache(_MAX_SAMPLE_REQ_CACHE_ITEMS,
                                        _CACHE_TTL_SECONDS)
_GET_SAMPLE_ROWS_CACHE = TTLCache(_MAX_SAMPLE_REQ_CACHE_ITEMS,
                                  _CACHE_TTL_SECONDS)
_GET_SAMPLE_ROWS_ERROR_CACHE = TTLCache(_MAX_SAMPLE_REQ_CACHE_ITEMS,
                                        _CACHE_TTL_SECONDS)


# TODO(dong) unify the usage of `get_sample_rows_raw` and `get_sample_rows`.
# Currently, `get_sample_rows_raw` is still used in `table_logic`.
@blocking_func(DATA_BACKEND_THREAD_POOL)
def get_sample_rows_raw(
    connector: Connector,
    source_table_name: str,
    cols: List[BaseColumnConfig],
    num_rows: int,
    skip_cache: bool = False,
) -> pandas.DataFrame:
    r"""Get a list of sample data rows from the table

        Args:
            connector (Connector): Connector where the table is stored
            source_table_name (str): Name of the table from which to fetch the
                sample
            cols: (List[BaseColumnConfig]): The columns to get sample rows for
                which should include pkey if you want stats for pkey
            num_rows: How many rows to fetch
            skip_cache: Whether to skip the cache and read from connector

        Returns:
            sample_rows: pandas dataframe of data columns + primary key if it
            exists.
        """
    log.info(
        "Getting raw sample rows (num_rows=%s skip_cache=%s)",
        num_rows,
        skip_cache,
    )
    col_names = [col.name for col in cols]
    df = connector.read_table_sample(
        table_name=source_table_name,
        pkey=None,
        cols=cols,
        num_rows=num_rows,
        skip_cache=skip_cache,
    )
    df = df.reindex(columns=col_names)
    return stringify_sample_dataframe(df)


def get_sample_rows(
    connector: Connector,
    meta_table: TableMetadataConfig | None,
    num_rows: int,
    serialize_to_json: bool,
    source_table_name: str = '',
    skip_cache: bool = False,
    response_token: str = '',
    mock_error: Exception | None = None,
    require_stats: bool = True,
) -> SampleRowsResponse:
    r"""Get a list of sample data rows from the table

        Args:
            connector (Connector): Connector where the table is stored
            source_table_name (str): Name of the table from which to fetch the
                sample
            cols (List[BaseColumnConfig]): The columns to get sample rows for
                which should include pkey if you want stats for pkey
            num_rows (int): How many rows to fetch
            source_table_name (str): The source table name used to get meta
                table if meta table is None
            skip_cache: Whether to skip the cache and read from connector
            response_token (str): A response token for async request to read
                data later.

        Returns:
            sample_rows: an array of length M, where M is the number of data
                columns + primary key if it exists. Each item in the array is
                of length N where N is the number of samples that will be
                returned. Sample rows are returned in the order of cols.
        """
    if meta_table is None:
        connector.open()
        meta_table = connector.get_table_metadata(
            table_names=[source_table_name], skip_cache=skip_cache)[0]

    ret = SampleRowsResponse()
    try:
        if mock_error:
            raise mock_error

        table = Table.from_source_config(meta_config=meta_table,
                                         name="placeholder_name",
                                         data_source=meta_table.data_source,
                                         connector=connector)

        # If `skip_cache` is True, then we always infer the stypes from the
        # stats of latest data. The samples will also be fetched from the
        # latest data so that they are always consistent.
        if require_stats is False:
            meta_table = table.infer_stypes(
                skip_cache=skip_cache,
                require_stats=require_stats,
            ).metadata_config()
        else:
            meta_table = table.infer_stypes(
                skip_cache=skip_cache, ).metadata_config()
    except Exception as e:
        error_msg = (
            "Unable to prepare sample rows from the configured data provider.")
        log.error("get_sample_rows metadata failed (error_type=%s)",
                  type(e).__name__)
        if (response_token != ''
                and response_token in _IN_PROGRESS_GET_SAMPLE_ROWS):
            _GET_SAMPLE_ROWS_ERROR_CACHE[response_token] = error_msg
            _IN_PROGRESS_GET_SAMPLE_ROWS.pop(response_token)
        return ret

    try:
        columns: List[SourceColumn] = get_table_cols(meta_table).source_cols
        base_cols_to_sample = [
            BaseColumnConfig(
                name=col.name,
                stype=col.stype,
                dtype=col.dtype,
                multicategorical_delimiter=(col.multicategorical_delimiter),
            ) for col in columns
        ]

        ret = SampleRowsResponse(columns)
        if num_rows > 0:
            rows_df = get_sample_rows_raw(
                connector=connector, source_table_name=meta_table.source_name,
                cols=base_cols_to_sample, num_rows=num_rows,
                skip_cache=skip_cache)
            if serialize_to_json:
                ret.sample_df_serialized = rows_df.to_json(orient='table')
            else:
                ret.rows = rows_df.values.tolist()

        if (response_token != ''
                and response_token in _IN_PROGRESS_GET_SAMPLE_ROWS):
            log.info(f"Cache get_sample_rows for {response_token} with "
                     f"{ret.columns} and {len(ret.rows)} rows.")
            _GET_SAMPLE_ROWS_CACHE[response_token] = ret
            _IN_PROGRESS_GET_SAMPLE_ROWS.pop(response_token)

    except Exception as e:
        # In case a failure, clean up in progress get sample rows.
        error_msg = (
            "Unable to read sample rows from the configured data provider.")
        log.error("get_sample_rows failed (error_type=%s)", type(e).__name__)
        if (response_token != ''
                and response_token in _IN_PROGRESS_GET_SAMPLE_ROWS):
            _GET_SAMPLE_ROWS_ERROR_CACHE[response_token] = error_msg
            _IN_PROGRESS_GET_SAMPLE_ROWS.pop(response_token)

    return ret


def get_table_cols(
    meta_table: TableOrViewMetadataConfig, ) -> SourceColumnsResource:
    validation = ValidationResponse()
    for col in meta_table.cols:
        if not allow_array_targets() and col.dtype == Dtype.stringlist.value:
            validation.warnings.append(
                ValidationWarning(
                    title=f"Column Type Changed: {col.name}",
                    message=(
                        f"The data type has been changed from "
                        f"{col.dtype.value} to string, and the semantic type "
                        f"has been changed from multicategorical to "
                        f"categorical because string arrays are currently "
                        f"not supported."),
                ))
            col.dtype = Dtype.string
            col.stype = Stype.categorical

    cols: List[SourceColumn] = []
    if meta_table.pkey is not None:
        assert meta_table.pkey.dtype is not None
        assert meta_table.pkey.stype is not None
        cols.append(
            SourceColumn(
                name=meta_table.pkey.name,
                stype=meta_table.pkey.stype,
                dtype=meta_table.pkey.dtype,
                is_primary=True,
                multicategorical_delimiter=(
                    meta_table.pkey.multicategorical_delimiter),
            ))

    for fkey in meta_table.fkeys:
        assert fkey.dtype is not None
        assert fkey.stype is not None
        cols.append(
            SourceColumn(
                name=fkey.name,
                stype=fkey.stype,
                dtype=fkey.dtype,
                is_primary=False,
                multicategorical_delimiter=(fkey.multicategorical_delimiter),
            ))

    for feature_col in meta_table.cols:
        assert feature_col.dtype is not None
        assert feature_col.stype is not None
        cols.append(
            SourceColumn(
                name=feature_col.name,
                stype=feature_col.stype,
                dtype=feature_col.dtype,
                is_primary=False,
                multicategorical_delimiter=(
                    feature_col.multicategorical_delimiter),
            ))

    if meta_table.time_col is not None:
        assert meta_table.time_col.dtype is not None
        assert meta_table.time_col.stype is not None
        cols.append(
            SourceColumn(
                name=meta_table.time_col.name,
                stype=meta_table.time_col.stype,
                dtype=meta_table.time_col.dtype,
                is_primary=False,
                multicategorical_delimiter=(
                    meta_table.time_col.multicategorical_delimiter),
            ))

    return SourceColumnsResource(cols, validation)


@blocking_func(DATA_BACKEND_THREAD_POOL)
def get_columns_response(
    connector: Connector,
    source_table_name: str,
    skip_cache: bool = False,
) -> SourceColumnsResource:
    connector.open()
    meta_table_configs = connector.get_table_metadata(
        table_names=[source_table_name], skip_cache=skip_cache)
    if not meta_table_configs:
        raise ValueError("Unable to fetch column data for source table "
                         f"{source_table_name}, please check if it is empty.")
    meta_table_config: TableMetadataConfig = meta_table_configs[0]
    table = Table.from_source_config(
        meta_config=meta_table_config,
        name="placeholder_name",
        data_source=meta_table_config.data_source,
        connector=connector,
    )

    # Do not require stats when we get columns response.
    table.infer_stypes(
        skip_cache=skip_cache,
        require_stats=False,
    )
    meta_table_config = table.metadata_config()
    table_cols_and_warn = get_table_cols(meta_table_config)
    return table_cols_and_warn


def _advisory_read_only(sql: str, connector_type: ConnectorType) -> None:
    """Advisory read-only pre-filter that surfaces its rejection as a public
    error.

    A raw ``ValueError`` from :func:`advisory_read_only_check` would be
    sanitized to a generic "Connector source access failed" by the connector
    activity boundary, hiding the actionable "Only read-only SELECT…" message.
    Re-raising as ``ConnectorValidationError`` (a ``PublicException`` on the
    activity's allowlist) preserves the clean message for ``/columns?query=…``
    and ``/sample?query=…``.
    """
    try:
        advisory_read_only_check(sql, connector_type)
    except ValueError as error:
        raise ConnectorValidationError(str(error)) from None


def _validate_query_scope(sql: str, connector_type: ConnectorType,
                          connector: Connector) -> None:
    try:
        validate_connector_sql_scope(sql, connector_type, connector)
    except ValueError as error:
        raise ConnectorValidationError(str(error)) from None


@contextmanager
def _warehouse_query_errors() -> Iterator[None]:
    """Convert a warehouse describe/sample failure into a public Custom SQL
    query error.

    Wrap **only** the connector's warehouse calls (``get_query_metadata`` /
    ``get_query_scan_estimate_bytes`` / ``sample_query``), which run *after*
    ``connector.open()`` — so a failure here is the *query* being rejected, not
    connector access. Re-raise it as a :class:`CustomSqlQueryError` carrying the
    warehouse's own reason (via :func:`sql_query_error_message`) instead of
    letting the connector activity boundary flatten it to the misleading
    "Connector source access failed". Local response shaping (Arrow/pandas,
    serialization) stays outside so an internal bug is never mislabeled as a
    warehouse rejection. Only the exception *type* is logged; the raw text —
    which can carry secrets on non-query paths — never reaches the log, and only
    the sanitized reason reaches the client.
    """
    try:
        yield
    except Exception as error:
        log.warning('Custom SQL query rejected by warehouse (error_type=%s)',
                    type(error).__name__)
        raise CustomSqlQueryError(sql_query_error_message(error)) from error


def _describe_query_columns(connector: Connector,
                            sql: str) -> SourceColumnsResource:
    """Column list for a Custom SQL query's result set.

    Describes the read-only ``SELECT`` (``get_query_metadata`` — no data scan)
    and resolves stypes from name/dtype heuristics only (``require_stats=False``,
    so no sample). Shared by column discovery and the sample-rows preview;
    callers enforce the read-only contract first. Returns the same
    ``SourceColumnsResource`` shape as a physical-table ``get_columns_response``.
    """
    # Only the describe touches the warehouse; the stype/shape work below is
    # local, so keep it out of the query-error wrapper.
    with _warehouse_query_errors():
        meta_table_config = connector.get_query_metadata(sql)
    table = Table.from_source_config(
        meta_config=meta_table_config,
        name="placeholder_name",
        data_source=meta_table_config.data_source,
        connector=connector,
    )
    table.infer_stypes(require_stats=False)
    return get_table_cols(table.metadata_config())


@blocking_func(DATA_BACKEND_THREAD_POOL)
def get_query_columns_response(
    connector: Connector,
    sql: str,
    connector_type: ConnectorType,
) -> SourceColumnsResource:
    """Discover a Custom SQL query's output columns without scanning data.

    Backs query-based column discovery in the table-authoring UI: apply the
    advisory read-only pre-filter, then describe the query for its output
    columns (name + dtype + heuristic stype). Authoritative stypes come from
    ``resolve_sql``'s definition-time contract, so this stays keystroke-cheap
    and never runs the Option-A sample.
    """
    _advisory_read_only(sql, connector_type)
    _validate_query_scope(sql, connector_type, connector)
    sql = subquery_safe_sql(sql, connector_type)
    # ``open`` (connect/auth) stays outside the query-error wrapper: a failure
    # here is genuine connector-access failure and must keep the "source access"
    # classification. The warehouse describe inside ``_describe_query_columns``
    # is wrapped there.
    connector.open()
    return _describe_query_columns(connector, sql)


@blocking_func(DATA_BACKEND_THREAD_POOL,
               TraceOptions(trace_params=['num_rows', 'serialize_to_json']))
def get_query_sample_rows_response(
    connector: Connector,
    sql: str,
    connector_type: ConnectorType,
    num_rows: int,
    serialize_to_json: bool = False,
) -> SampleRowsResponse:
    """Bounded preview of a Custom SQL query's result for the "Load sample"
    button.

    Apply the advisory read-only pre-filter, describe the query for the column
    list, then run the connector's bounded ``sample_query`` (LIMIT + statement
    timeout) for at most ``num_rows`` rows. Rows are stringified with nulls as
    ``None`` and returned in the same shape as the physical sample path —
    ``sample_df_serialized`` when ``serialize_to_json`` (the request-router
    default), else ``rows`` — so the response is shape-compatible with a
    physical ``get_sample_rows_response``. Explicit, button-only, and
    pre-materialization: there is no server-side sample cache.
    """
    _advisory_read_only(sql, connector_type)
    _validate_query_scope(sql, connector_type, connector)
    sql = subquery_safe_sql(sql, connector_type)
    # ``open`` (connect/auth) stays outside the query-error wrapper: a failure
    # here is genuine connector-access failure and must keep the "source access"
    # classification. Only the warehouse describe/estimate/sample calls below are
    # wrapped; local response shaping is not.
    connector.open()
    columns = _describe_query_columns(connector, sql).source_cols
    ret = SampleRowsResponse(columns)
    # Cost-gate the scan the same way the Validate-time preview does: on a
    # warehouse where LIMIT does not bound the scan (BigQuery), an over-budget
    # authoring query returns columns only (schema, no rows) rather than
    # triggering a full scan from the "Load sample" button.
    with _warehouse_query_errors():
        estimate = connector.get_query_scan_estimate_bytes(sql)
    if sample_over_budget(estimate, gated=connector.gates_query_scan):
        return ret
    if num_rows > 0:
        # Clamp to the connector-wide sample cap so a preview can never request
        # an unbounded LIMIT (mirrors the physical ``read_table_sample`` bound).
        num_rows = min(num_rows, NUM_SAMPLE_ROWS)
        col_names = [col.name for col in columns]
        with _warehouse_query_errors():
            rows_df = connector.sample_query(sql, num_rows)
        rows_df = rows_df.reindex(columns=col_names)
        rows_df = stringify_sample_dataframe(rows_df)
        if serialize_to_json:
            ret.sample_df_serialized = rows_df.to_json(orient='table')
        else:
            ret.rows = rows_df.values.tolist()
    return ret


@blocking_func(DATA_BACKEND_THREAD_POOL,
               TraceOptions(trace_params=['num_rows', 'skip_cache']))
def get_sample_rows_response(
    connector: Connector,
    source_table_name: str,
    num_rows: int,
    serialize_to_json: bool = False,
    skip_cache: bool = False,
    response_token: str = '',
    is_async: bool = False,
    mock_error: Exception | None = None,
    require_stats: bool = True,
):
    log.info(
        "Running get_sample_rows_response "
        "(num_rows=%s serialize_to_json=%s skip_cache=%s is_async=%s "
        "require_stats=%s)",
        num_rows,
        serialize_to_json,
        skip_cache,
        is_async,
        require_stats,
    )
    # For an asynchronous request with a cache key, check if the cache is still
    # in progress. If it is not, try to retrieve the cached result from the
    # cache.
    if response_token != '':
        if response_token in _IN_PROGRESS_GET_SAMPLE_ROWS:
            return SampleRowsResponse(response_token=response_token), None

        try:
            ret = _GET_SAMPLE_ROWS_CACHE.pop(response_token)
            return ret, None
        except KeyError:
            log.error('Unable to fetch cached sample data')
            if response_token in _GET_SAMPLE_ROWS_ERROR_CACHE:
                raise ValueError(
                    _GET_SAMPLE_ROWS_ERROR_CACHE.pop(response_token))
            else:
                raise ValueError(
                    "Unable to fetch sample data. Either reached the maximum "
                    f"timeout of {_CACHE_TTL_SECONDS} seconds or there is an issue "
                    "with your data source. Please try again.")

    if is_async:
        # Generate a unique cache key for the request.
        response_token = str(uuid.uuid4().hex)
        _IN_PROGRESS_GET_SAMPLE_ROWS[response_token] = 1

        # Submit the function `get_sample_rows` in a separated thread.
        # TODO(dong) timeout this thread explicitly with large timeout.
        log.info(
            "Submitting get_sample_rows to thread pool "
            "(num_rows=%s serialize_to_json=%s skip_cache=%s)",
            num_rows,
            serialize_to_json,
            skip_cache,
        )
        fut = DATA_BACKEND_THREAD_POOL.submit(get_sample_rows, connector, None,
                                              num_rows, serialize_to_json,
                                              source_table_name, skip_cache,
                                              response_token, mock_error)
        return SampleRowsResponse(response_token=response_token), fut
    else:
        connector.open()
        meta_table: TableMetadataConfig = connector.get_table_metadata(
            table_names=[source_table_name], skip_inferral=False,
            skip_cache=skip_cache)[0]
        return get_sample_rows(
            connector=connector,
            meta_table=meta_table,
            num_rows=num_rows,
            serialize_to_json=serialize_to_json,
            skip_cache=skip_cache,
            require_stats=require_stats,
        ), None
