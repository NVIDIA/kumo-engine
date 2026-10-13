# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import os
import threading
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd
from kumo_core.types import ColStatType, TableStatType
from kumo_core.types.conversions import to_dtype
from kumo_core.types.schema import (
    BaseTableFingerprint,
    IngestedTableFingerprint,
    SqlTableFingerprint,
)
from kumoapi.typing import Dtype

from common_utils import string
from common_utils.string import contains_date, contains_time
from io_utils import logging
from io_utils.fsutil.credentials import with_filesystem_credentials
from kumo.config import (
    BaseColumnConfig,
    ConfigMixin,
    TableMetadataConfig,
    TableOrViewMetadataConfig,
)
from kumo.connector import DEFAULT_DATA_VERSION
from kumo.connector.blocking import TraceOptions, blocking_method, trace
from kumo.connector.utils import (
    SAMPLING_SEED,
    cast_df_to_dtypes,
    coerce_decimal_series_for_stats,
    coerce_timestamp_stype_series_for_stats,
    name_inference_temporal_supported,
)
from kumo.data.common import DATA_BACKEND_THREAD_POOL
from kumo.metrics import Metrics
from kumo.stats import Stats, compute_col_stats_python
from kumo.utils.profiling import Timer
from statedb.utils.errors import (
    ConnectorValidationDiagnosticCode,
    ProviderBackendError,
)

# Whether to trigger automatic table data version cache cleanup
# when a connector is garbage collected. This is only set to True
# in kumo-rest container
CLEAN_TABLE_VERSION_CACHE = 'CLEAN_TABLE_VERSION_CACHE'


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


logger = logging.getLogger(__name__)


def _get_active_cache():
    # Keep the Connector ABC importable while ``kumo.data`` initializes.
    from kumo.data import get_active_cache
    return get_active_cache()


@dataclass(frozen=True)
class ExportMetrics:
    r"""Row and byte metrics returned by source preparation.

    `source_size` is always set. `num_rows` is optional:

    - Query/warehouse connectors (Snowflake COPY INTO, Databricks/BigQuery/CSV)
      get it free from the unload, so they still set it.
    - Flat Parquet copies return ``None``: its only way to get a
      source row count is an O(part-count) footer scan that does not scale
      (kumo#30173). Row-based scale limits do not consume it; emptiness is
      detected with ``is_empty`` on the local mirror instead. Its ``NUM_ROWS``
      is filled after ingest from the
      DiskGraph node counts.

    Staged-file connectors return exact metrics from the unload response or the
    files they wrote. Direct sources may return optimizer estimates for
    placement before DiskGraph replaces them with exact ingest statistics;
    ``approx`` distinguishes those estimates.

    Warehouse and CSV exports report produced Parquet bytes. Parquet exports
    retain stored input bytes, even when acquisition materializes partitions.
    Direct sources may report an estimate until ingestion completes.
    """
    num_rows: Optional[int]
    source_size: int
    approx: bool = False


def _compute_col_stats_from_sample(
    stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
    df: pd.DataFrame,
    *,
    approx: bool,
    data_version: Any,
) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
    r"""Reduce an already-bounded sample without another warehouse read."""
    out: Dict[BaseColumnConfig, Dict[ColStatType, Stats]] = {}
    for col_cfg, stats in stats_dict.items():
        if col_cfg.name not in df.columns:
            out[col_cfg] = {}
            continue
        ser = df[col_cfg.name]
        if col_cfg.dtype == Dtype.binary:
            ser = ser.map(lambda value: value.hex() if isinstance(
                value, (bytes, bytearray, memoryview)) else value)
        # A ``timestamp``-stype column read raw (cast_response=False) can
        # arrive as a bare epoch integer/float; coerce it to datetime so
        # the MIN/MAX col-stats are typed ``pd.Timestamp`` values.
        ser = coerce_timestamp_stype_series_for_stats(ser, col_cfg)
        # A warehouse decimal arrives as an object column of `decimal.Decimal`,
        # which the gate below reads as `string` and strips every numerical
        # stat from -- leaving a `numerical` column without the stats
        # `ColumnStats.from_stats_dict` requires (kumo-ai/kumo#30917).
        ser = coerce_decimal_series_for_stats(ser, col_cfg)
        # Dual-dtype gating: realized series dtype may differ from configured
        # ``col_cfg.dtype``, so filter to stats supported on both.
        try:
            ser_dtype = to_dtype(ser.dtype, ser=ser)
        except KeyError:
            # A declared dtype over a source type Kumo has no dtype for, such
            # as uint64 or float16, is read raw here.
            if not pd.api.types.is_numeric_dtype(ser.dtype):
                raise
            ser = ser.astype('float64')
            ser_dtype = Dtype.float64
        supported = [s for s in stats if s.supports(ser_dtype)]
        if not supported:
            out[col_cfg] = {}
            continue
        out[col_cfg] = compute_col_stats_python(
            ser,
            stats=supported,
            approx=approx,
            data_version=data_version,
        )
    return out


# TODO(Dulloor): rename NUM_SAMPLE_ROWS to make it more specific
# TODO(Dulloor): Benchmark connectors/datasets with the sample size
NUM_SAMPLE_ROWS = 10000


class MissingColumnError(ValueError):
    pass


class UnreadableColumnError(ValueError):
    r"""Thrown when a datatype prevents the table from being read"""
    def __init__(self, message, unreadable_column=None):
        super().__init__(message)
        self.unreadable_column = unreadable_column


class DuplicateColumnsError(ValueError):
    r"""Thrown when a table (usually a csv file) has a duplicate column name"""
    def __init__(self, message):
        super().__init__(message)


class ConnectorError(ProviderBackendError):
    """User-facing error class for connector operation failures."""
    def __init__(
        self,
        *args: Any,
        diagnostic_code: Optional[ConnectorValidationDiagnosticCode] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        #: Optional non-secret category (e.g. target-not-found vs
        #: access-denied) surfaced through the connector target-validation
        #: masking boundaries while this error keeps its detailed public
        #: message for non-masked callers. ``None`` leaves the failure
        #: uncategorized. Read via
        #: ``statedb.utils.errors.connector_validation_diagnostic_value``.
        self.diagnostic_code = diagnostic_code


class SourceTableNotFoundError(ConnectorError):
    """A configured physical source table is absent.

    File, BigQuery, and Databricks metadata probes use this serializable
    public error so its sanitized table and connector names survive activity
    boundaries. API surfaces may expose it as a typed 422 or translate it to
    their established source-table-not-found 404 contract.
    """
    template = ("Source table '{table_name}' does not exist in connector "
                "'{connector_name}'.")

    def __init__(self, table_name: str, connector_name: str,
                 related_errors: Optional[List[Exception]] = None) -> None:
        super().__init__(table_name=table_name, connector_name=connector_name,
                         related_errors=related_errors)


class SourceTableUnavailableError(ConnectorError):
    """A source table is missing or hidden by provider access controls.

    Snowflake metadata cannot safely distinguish those cases, so this
    serializable public error reports only the sanitized table and connector
    names. Connector discovery translates it to the same 404 contract as a
    definitively missing source.
    """
    template = ("Source table '{table_name}' could not be found or accessed "
                "in connector '{connector_name}'.")

    def __init__(self, table_name: str, connector_name: str,
                 related_errors: Optional[List[Exception]] = None) -> None:
        super().__init__(table_name=table_name, connector_name=connector_name,
                         related_errors=related_errors)


_IGNORABLE_SOURCE_TABLE_ERRORS = (
    FileNotFoundError,
    SourceTableNotFoundError,
    SourceTableUnavailableError,
)


class Connector(ConfigMixin, ABC):
    r"""Base class for connecting to various data sources.
    A :class:`~kumo.connector.Connector` instance will establish a connection
    upon first read, and keeps it as long as its instance stays in scope.
    There exist multiple ways to connect to a specific data source:

    1. As a **regular Python object**:

       .. code-block:: python

           connector = Connector(...)
           connector.read_table(...)  # do something.
           del connector  # closes the connection automatically.

    2. As a **decorator**:

       .. code-block:: python

           @Connector(...)
           def main(connector):
               connector.read_table(...)  # do something.

    3. As a **"with" statement**:

       .. code-block:: python

           with Connector(...) as connector:
               connector.read_table(...)  # do something.

    If a data source requires credentials for access, the credentials may be
    provided either in the constructor, as `Connector(credentials=...)`, or
    through an open call, as `connector.open(credentials=...)`, or through
    `connector.set_credentials() before connection is established.`.
    """
    def __init__(
        self,
        name: str,
        credentials: Optional[Dict] = None,
    ):
        string.validate_id(name, "Connector")
        self._name = name
        self._credentials = credentials
        self._table_metadata_cache: Dict[Tuple[str, str, str],
                                         TableOrViewMetadataConfig] = {}
        self._context = None
        self._require_explicit_credentials = False
        # Serializes open() / close() so _context can't be set by
        # one thread in the middle of another thread's open(). Reentrant
        # because close() may be invoked via weakref finalizers in the
        # same thread that is still inside open().
        self._open_lock = threading.RLock()

    @property
    def name(self) -> str:
        return self._name

    def require_explicit_credentials(self) -> None:
        """Disable deployment/environment credential fallback for this object."""
        self._require_explicit_credentials = True

    @abstractmethod
    def _get_connector_type(self) -> str:
        """
        returns the connector type - Snowflake/BigQuery/Whatever.
        """

    def set_credentials(self, credentials: Optional[Dict]) -> None:
        """
        Set the credentials before connection is established. No-op if it's
        already connected.
        """
        if not self.is_connected:
            self._credentials = credentials

    # Connecting and closing connections ######################################

    @abstractmethod
    def _open(self, credentials: Optional[Dict] = None) -> Any:
        """
        Performs authentication steps necessary to connect to a connector.
        Returns a connection context.
        """

    @abstractmethod
    def _close(self, context: Any):
        """
        Performs steps necessary to close an open connection. Accepts a
        connection context.
        """

    @property
    def is_connected(self) -> bool:
        r"""Whether the :class:`~kumo.connector.Connector` has already
        established a connection."""
        return getattr(self, '_context', None) is not None

    @property
    def context(self) -> Any:
        r"""Returns the connector context."""
        if not self.is_connected:
            self.open()
        return self._context

    def validate_permissions(self) -> None:
        pass

    @with_filesystem_credentials
    def open(self, credentials: Optional[Dict] = None) -> 'Connector':
        r"""Establishes a connection.

        Args:
            credentials (Dict, optional): Authentication credentials necessary
                to connect to the data source. If None, will default to using
                environment variables. (default: :obj:`None`)
        """
        # Fast path: already connected, no need to acquire the lock.
        if self.is_connected:
            return self
        with self._open_lock:
            # Double-check under the lock — a concurrent open() may have
            # just set _context.
            if self.is_connected:
                return self
            credentials = credentials or getattr(self, '_credentials', None)
            context = self._open(credentials=credentials)
            self._context = context
            try:
                self.validate_permissions()
            except BaseException as validation_error:
                # A context that failed post-open validation must never be
                # reused by a later open attempt. Detach it before cleanup so
                # cleanup failure cannot leave the connector looking healthy.
                self._context = None
                try:
                    self._close(context)
                except BaseException as cleanup_error:
                    logger.warning(
                        'Connector cleanup failed after open validation '
                        'failure (validation_error_type=%s '
                        'cleanup_error_type=%s)',
                        type(validation_error).__name__,
                        type(cleanup_error).__name__,
                    )
                raise
            try:
                weakref.finalize(self, clean_backend_cache, self.name)
                # Close the connection automatically when `context` is
                # about to be garbage collected. Use the local variable
                # rather than `self._context` — a concurrent close() on
                # another thread could wipe the attribute before the
                # finalizer is registered, and the race would surface
                # as AttributeError on this line.
                weakref.finalize(context, self.close)
            except TypeError:
                pass
            logger.info("Successfully connected to a data source via %s",
                        self.__class__.__name__)
        return self

    def close(self) -> 'Connector':
        r"""Closes an existing connection."""
        # Short-circuit before acquiring the lock: if `__init__` raised
        # before setting `_open_lock`, `__del__` can still call `close()`
        # on the half-built object. There is no `_context` to clean up
        # and no thread in `open()` to race against, so skip the lock.
        if not self.is_connected:
            return self
        with self._open_lock:
            if self.is_connected:
                context = self._context
                # Detach first so a failed provider close cannot poison the
                # next open or make repeated close calls reuse a broken
                # session. Provider contexts retain their own finalizers.
                self._context = None
                self._close(context)
        return self

    @classmethod
    def disable_interactive_input(cls) -> None:
        """Globally disable connectors from getting password over command line.

        Many types of connectors (such as `SnowflakeConnector`) will use the
        `input()` function to interactively get the password from the user when
        calling `Connector.open()`, if the credentials dictionary for that
        connector are not passed in as an argument, or are available in the
        environment variables.  This behavior is undesirable in the context of
        automated scripts or services, as the `input()` will hang
        indefinitely, blocking program execution. If calling
        `Connector.disable_interactive_input()` at program startup will cause
        the open() function to throw an exception when credentials are missing,
        rather than blocking indefinitely.
        """
        cls._DISABLE_INTERACTIVE_INPUT = True

    # Subclasses of connectors should check this flag before getting
    # interactive input from users, such as calling `input()`
    _DISABLE_INTERACTIVE_INPUT = False

    # Reading table metadata ##################################################

    def has_links_support(self) -> bool:
        r"""Check if connector may contain link information. Databases (such as
        Snowflake) may contain pk-fk constraints. Connector implementations
        override this method based on their capability.

        Returns:
            :obj:`True` if data in connector has link information,
            :obj:`False` otherwise.
        """
        return False

    def skip_list_table(self) -> bool:
        r"""Check if the connector should skip listing all the tables before
        fetching the table metadata for all the accessible tables.
        SQLConnectors (databricks, snowflake, bigquery) all support fetching
        table metadata for all accessible tables without explicitly specifying
        the table names. Skipping listing tables helps to avoid unnecessarily
        fetching table metadata multiple times from the warehouse.

        Returns:
            :obj:`True` if the connector support fetching table metadata
                without explicitly specifying table names,
            :obj:`False` otherwise.
        """
        return False

    @abstractmethod
    def list_tables(self, schema_name: Optional[str] = None) -> List[str]:
        """
        Default implementation for listing all accessible tables, which
        returns empty list since there's no way of discovering all the tables.
        Override this for connectors that have better capabilities.
        """

    def table_exists(self, table_name: str) -> bool:
        """Return whether one exact table is currently readable.

        This deliberately bypasses connector metadata caches. Provider
        implementations may override it with a cheaper exact lookup, but must
        never implement it by listing every table. Implementations may report
        absence by returning ``False`` or by raising a typed source-table
        error; the connector activity boundary normalizes both to ``False``.
        """
        return bool(
            self.get_table_metadata(
                table_names=[table_name],
                skip_cache=True,
                skip_inferral=True,
            ))

    def _get_table_metadata(
        self,
        table_names: Optional[List[str]] = None,
        skip_cache: bool = False,
        skip_inferral: bool = False,
        schema_name: Optional[str] = None,
        ignore_errors: bool = False,
    ) -> List[TableMetadataConfig]:
        r""" Default implementation """
        if table_names is None:
            table_names = self.list_tables()
        logger.info(f"Fetching metadata for tables: {table_names=}")
        table_metadata_list = []
        for table_name in table_names:
            try:
                metadata = self._get_table_cols_metadata(
                    table_name, skip_cache=skip_cache,
                    skip_inferral=skip_inferral)
            except _IGNORABLE_SOURCE_TABLE_ERRORS as error:
                if not ignore_errors:
                    raise
                logger.warning('Table %r failed to load (error_type=%s)',
                               table_name,
                               type(error).__name__)
                metadata = None

            if metadata:  # None if the table could not be read
                table_metadata_list.append(metadata)
        return table_metadata_list

    def _get_table_cols_metadata(
        self,
        table_name: str,
        skip_cache: bool = False,
        skip_inferral: bool = False,
    ) -> Optional[TableMetadataConfig]:
        r"""Internal method to get table metadata (columns and associated
        data types). The implementation first reads metadata from the
        connector and determines the columns for which dtype is not known.
        For those columns the implementation reads the first 100 rows of
        the table into memory and infers datatypes based on that. We do
        not distinguish if a column is a primary key or foreign key. This
        will be handled by the caller inside :meth:`infer_metadata`.
        For connector with access to table metadata, consider overriding
        this.

        Returns: Optional[TableMetadataConfig] which is None if the table file
            cannot be read, e.g. if it is an unsupported csv encoding
        """

    def _infer_cols_dtype_by_name(
        self,
        table_metadata: TableMetadataConfig,
        skip_cache: bool = False,
    ):
        # TODO(Dulloor): Do this for pkey/fkeys too
        for col in table_metadata.cols:
            # If the column has "TIME" in its name we assume that it
            # is a time column and try to cast it. If it fails, we
            # raise a warning because this may be an error on the user side.
            # Repeat for columns with "DATE" in their name.
            # Negative result: Trying to cast the first 100 rows of any string
            # column to determine if it is a date can produce false positives.
            # This happens because empty strings and numbers can all be cast
            # into datetime objects.
            if col.dtype.supports_cast_to_temporal():
                desired_dtype = None
                if contains_time(col.name):
                    desired_dtype = Dtype.time
                elif contains_date(col.name):
                    desired_dtype = Dtype.date

                from kumo.connector import CompositeFileConnector, CSVConnector
                if ((isinstance(self, CSVConnector)
                     or isinstance(self, CompositeFileConnector))
                        and (desired_dtype and col.dtype != desired_dtype)):
                    table_sample = None
                    try:
                        # This function can be called on any of the tables in
                        # the data source, even tables that are not explicitly
                        # added by the customer. But we may not have access to
                        # read all tables in a data source. This is especially
                        # true of Snowflake connectors, where supplied
                        # credentials is able to list all tables but cannot
                        # read some of them. Therefore, we must ignore
                        # if `read_table_sample` fails here.
                        # We also guarantee that the table samples are not
                        # stale so as to match latest table metadata, if
                        # `skip_cache` is True.
                        table_sample = self.read_table_sample(
                            table_metadata.source_name, skip_cache=skip_cache)
                    except (ConnectorError) as e:
                        logger.warning(
                            "Failed to read samples for table %r "
                            "(error_type=%s)", table_metadata.source_name,
                            type(e).__name__)

                    if table_sample is not None:
                        if name_inference_temporal_supported(
                                table_sample[col.name], desired_dtype):
                            col.dtype = desired_dtype
                        else:
                            logger.warning(
                                "Column %r name suggests %s but DiskGraph "
                                "could not parse enough sampled values; "
                                "keeping dtype %s.", col.name, desired_dtype,
                                col.dtype)

    def _infer_cols_dtype_by_values(
        self,
        table_metadata: TableMetadataConfig,
    ):
        # TODO(Dulloor): infer dtype based on values in the string columns
        pass

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    @with_filesystem_credentials
    def get_table_metadata(
        self,
        table_names: Optional[List[str]] = None,
        skip_cache: bool = False,
        skip_inferral: bool = False,
        schema_name: Optional[str] = None,
        ignore_errors: bool = False,
    ) -> List[TableMetadataConfig]:
        r"""Get metadata for all tables that this connector supports.

        Args:
            table_names (List[str], optional)
                Set of tables for which metadata is requested. Return metadata
                for all tables in the data source if :obj:`None`.
                (default: :obj:`None`)
            ignore_errors (bool)
                Omit sources that are missing or unavailable. Other metadata,
                provider, and programming failures still propagate.

        Returns:
            List[TableMetadataConfig]: A list of  :class:`TableMetadataConfig`
            objects with the :obj:`name` objects filled in

        Raises:
            SourceTableNotFoundError: If a requested source is absent and
                ``ignore_errors`` is :obj:`False`.
            SourceTableUnavailableError: If a requested source is missing or
                inaccessible and ``ignore_errors`` is :obj:`False`.
        """
        def _filter_missing_metadata(table_names: str):
            # return names of tables with missing metadata
            ret = []
            for table_name in table_names:
                try:
                    data_version = self.get_data_version(table_name, skip_cache)
                except _IGNORABLE_SOURCE_TABLE_ERRORS:
                    if ignore_errors:
                        continue
                    raise
                if (self.name, table_name,
                        data_version) not in self._table_metadata_cache:
                    ret.append(table_name)
            return ret

        self.open()

        if table_names is None:
            if self.skip_list_table() and skip_cache and skip_inferral:
                table_metadata = self._get_table_metadata(
                    table_names, skip_cache=skip_cache,
                    skip_inferral=skip_inferral, schema_name=schema_name,
                    ignore_errors=ignore_errors)
                if not table_metadata:
                    raise ConnectorError(
                        f"Connector '{self.name}' found no accessible tables. "
                        "Please verify that the database/schema contains tables "
                        "and that your credentials have permission to list them."
                    )
                return table_metadata
            table_names = self.list_tables(schema_name)
            if table_names is None:
                raise ConnectorError(
                    f"Connector '{self.name}' found no accessible tables. "
                    "Please verify that the database/schema contains tables "
                    "and that your credentials have permission to list them.")
            return self.get_table_metadata(
                table_names,
                skip_cache=skip_cache,
                skip_inferral=skip_inferral,
                schema_name=schema_name,
                ignore_errors=ignore_errors,
            )

        tables_to_fetch_metadata: List[str] = []
        if skip_inferral or skip_cache:
            tables_to_fetch_metadata = table_names
        else:
            # fetch metadata only if missing in the cache
            tables_to_fetch_metadata = _filter_missing_metadata(table_names)

        missing_tables_metadata: List[TableMetadataConfig] = []
        if (len(tables_to_fetch_metadata) > 0):
            try:
                missing_tables_metadata = self._get_table_metadata(
                    tables_to_fetch_metadata, skip_cache=skip_cache,
                    skip_inferral=skip_inferral, schema_name=schema_name,
                    ignore_errors=ignore_errors)
            except _IGNORABLE_SOURCE_TABLE_ERRORS:
                if not ignore_errors:
                    raise

            # If we don't want to do any further inferral of dtypes, return
            # what we have now. Since this is meant for lightweight calls to
            # get_table_metadata and is not full, we don't want to store these
            # results in the cache either
            if skip_inferral:
                return missing_tables_metadata

            readable_tables_metadata: List[TableMetadataConfig] = []
            for table_metadata in missing_tables_metadata:
                # Apply additional infer logic before caching
                self._infer_cols_dtype_by_name(table_metadata, skip_cache)
                self._infer_cols_dtype_by_values(table_metadata)

                # TODO remove this data_version call as it calls a ls
                # again after the ls in _get_table_metadata above which
                # doesn't get the data_version This will require fetching
                # data_version in _get_table_metadata above
                try:
                    data_version = self.get_data_version(
                        table_metadata.source_name, skip_cache=skip_cache)
                except _IGNORABLE_SOURCE_TABLE_ERRORS:
                    if ignore_errors:
                        continue
                    raise
                self._table_metadata_cache[(self.name,
                                            table_metadata.source_name,
                                            data_version)] = table_metadata
                readable_tables_metadata.append(table_metadata)
            missing_tables_metadata = readable_tables_metadata

        tables_metadata: List[TableMetadataConfig] = []
        missing_table_names: List[str] = []

        # Fill in from cache
        # TODO: In the future, also fill in created and modified time
        # attributes.
        for table_name in table_names:
            table_metadata = None
            if table_name in tables_to_fetch_metadata:
                # use the metadata we just got above to avoid fetching
                # data version again.
                for missing_table_metadata in missing_tables_metadata:
                    if missing_table_metadata.source_name == table_name:
                        table_metadata = missing_table_metadata
                        break
            if not table_metadata:
                # Run this even if we iterate over missing_tables_metadata
                # above as missing_tables_metadata may not have all tables
                # in tables_to_fetch_metadata
                try:
                    data_version = self.get_data_version(table_name, skip_cache)
                except _IGNORABLE_SOURCE_TABLE_ERRORS:
                    if ignore_errors:
                        continue
                    raise
                if (self.name, table_name,
                        data_version) in self._table_metadata_cache:
                    table_metadata = self._table_metadata_cache[(self.name,
                                                                 table_name,
                                                                 data_version)]
            if table_metadata:
                tables_metadata.append(copy.deepcopy(table_metadata))
            else:
                missing_table_names.append(table_name)

        if not missing_tables_metadata and len(missing_table_names) > 0:
            logger.warning(f"Some tables were not read: {missing_table_names}")

        logger.info(f"Returning metadata {tables_metadata=}")
        return tables_metadata

    # Reading raw table data ##################################################

    def compute_col_stats(
        self,
        table_name: str,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        *,
        approx: bool = False,
        num_rows: Optional[int] = None,
        data_version: Any = None,
        sql: Optional[str] = None,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        r"""Pre-ingest column-stat compute for ``table_name``.

        Hands off to the per-engine ``compute_col_stats_<engine>``
        orchestrator (which handles every :class:`ColStatType` member).

        Implementation: pull a sample via :meth:`read_table_sample` and
        reduce each column with
        :func:`kumo.stats.compute_stats_python.compute_col_stats_python`.
        All connectors share this sample-and-python path; the previous
        engine-specific pushdowns (Snowpark, Databricks) were removed in
        #28324 when the Backend ABC was deleted.

        For a Custom SQL Table the caller passes ``sql``; the sample then comes
        from executing the query (:meth:`sample_query`) instead of a physical
        read of ``table_name`` (a synthetic alias with no physical table).
        """
        from kumo.stats.stats import NUM_ROWS_FOR_APPROX_STATS

        if not stats_dict:
            return {}

        sample_n = num_rows if num_rows is not None else NUM_ROWS_FOR_APPROX_STATS
        if sql is not None:
            df = self.sample_query(sql, sample_n)
        else:
            df = self.read_table_sample(
                table_name,
                cols=list(stats_dict.keys()),
                num_rows=sample_n,
                cast_response=False,
                data_version=data_version,
            )

        return _compute_col_stats_from_sample(stats_dict, df, approx=approx,
                                              data_version=data_version)

    @staticmethod
    def _cast_unsupported_to_str(col: BaseColumnConfig) -> BaseColumnConfig:
        if col and col.dtype and col.dtype.is_unsupported():
            col_copy = copy.copy(col)
            col_copy.dtype = Dtype.string  # read raw values
            return col_copy
        return col

    @abstractmethod
    def _read_table(
        self,
        context: Any,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        r"""Internal method to read an entire table into memory."""

    @abstractmethod
    def _read_table_chunk(
        self,
        context: Any,
        table_name: str,
        num_rows: int,
        offset: int = 0,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        r"""Internal method to read first num_rows of a table into memory."""

    @abstractmethod
    def _read_table_sample(
        self,
        context: Any,
        table_name: str,
        num_rows: Optional[int],
        frac: Optional[float],
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> pd.DataFrame:
        r"""Internal method to read a sample of a table into memory."""

    @trace(TraceOptions(trace_params=True))
    @with_filesystem_credentials
    def read_table(
        self,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
        num_rows: Optional[int] = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        r"""Reads rows from a table and returns them as a
        :class:`pandas.DataFrame`.

        Args:
            table_name (str): The name of the table.
            pkey (BaseColumnConfig, optional): The primary key of the table.
                (default: :obj:`None`)
            cols (List[BaseColumnConfig], optional): The set of columns to
                return. If set to :obj:`None`, will return all columns of the
                table. (default: :obj:`None`)
            num_rows (int, optional): The number of rows to return (from the
                table head). If set to :obj:`None` or a value less than zero,
                will return all rows. (default: :obj:`None`)
            offset (int, optional): The number of rows to skip at the start of
                the table. Only applicable in conjunction with :obj:`num_rows`.
                (default: :obj:`0`)
        """
        pkey = self._cast_unsupported_to_str(BaseColumnConfig.cast(pkey))

        if cols is not None:
            cols = [
                self._cast_unsupported_to_str(BaseColumnConfig.cast(col))
                for col in cols
            ]

        self.open()

        if num_rows is None or num_rows < 0:
            with Timer(logging_prefix=f"{table_name}.read_table",
                       metric_name=Metrics.TABLE_READ_TIME,
                       metric_labels={"table_name": table_name}):
                df = self._read_table(self._context, table_name, pkey, cols)
        else:
            df = self._read_table_chunk(self._context, table_name, num_rows,
                                        offset, pkey, cols)

        return fillna(df, cols or [])

    def _cast_and_resample_table_sample(
        self,
        table_sample: pd.DataFrame,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
        num_rows: Optional[int] = None,
        frac: Optional[float] = None,
    ) -> pd.DataFrame:
        if num_rows is None or num_rows < len(table_sample):
            table_sample = table_sample.sample(n=num_rows, frac=frac,
                                               random_state=SAMPLING_SEED)

        dtypes: Dict[str, Dtype] = {}
        selected_cols: List[str] = []
        if pkey is not None:
            selected_cols.append(pkey.name)
            if pkey.dtype is not None:
                dtypes[pkey.name] = pkey.dtype

        if cols is not None:
            for col in cols:
                selected_cols.append(col.name)
                dtypes[col.name] = col.dtype

        if selected_cols:
            table_sample = table_sample[selected_cols]
            table_sample = cast_df_to_dtypes(table_sample, dtypes)

        if pkey is not None:
            table_sample = table_sample.set_index(pkey.name)

        return fillna(table_sample, cols or [])

    @trace(TraceOptions(trace_params=['table_name', 'num_rows', 'skip_cache']))
    @with_filesystem_credentials
    def read_table_sample(
        self,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
        num_rows: Optional[int] = None,
        frac: Optional[float] = None,
        skip_cache: bool = False,
        cast_response: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""Reads a sample of rows from a table and returns them as a
        :class:`pandas.DataFrame`.

        Args:
            table_name (str): The name of the table.
            pkey (BaseColumnConfig, optional): The primary key of the table.
                (default: :obj:`None`)
            cols (List[BaseColumnConfig], optional): The set of columns to
                return. If set to :obj:`None`, will return all columns of the
                table. (default: :obj:`None`)
            num_rows (int, optional). The number of rows to return. Note that
                the final number of rows is the minimum of :obj:`num_rows` and
                total number of rows.
                If set to :obj:`None`, will return 1 row.
            frac (float, optional). The fraction of rows to return. Cannot be
                used with :obj:`num_rows`.
            skip_cache (bool): Whether to skip old versions in the cache.
                Note that it will just skip cached data version if set,
                which is different from other places.
            cast_response (bool): Whether to cast the returned
                DataFrame to dtypes. Dtypes are determined by mapping the
                sampled Pandas DataFrame schema to kumo dtypes of defaulting
                to string.
        Returns: Pandas DataFrame containing the table sample rows.
        """
        pkey = self._cast_unsupported_to_str(BaseColumnConfig.cast(pkey))

        if cols is not None:
            cols = [
                self._cast_unsupported_to_str(BaseColumnConfig.cast(col))
                for col in cols
            ]

        self.open()

        if num_rows is not None and frac is not None:
            frac = None
            logger.warning(
                f"Cannot set 'num_rows' and 'frac' in a call to read a table "
                f"sample. 'frac' (set to {frac}) will be ignored.")

        if num_rows is None and frac is None:
            num_rows = NUM_SAMPLE_ROWS

        if num_rows is not None and num_rows > NUM_SAMPLE_ROWS:
            num_rows = NUM_SAMPLE_ROWS
            logger.warning(
                f"Reducing the maxinum number of rows that can be sampled "
                f"from the requested {num_rows} to the maximum "
                f"{NUM_SAMPLE_ROWS}.")

        # NOTE: we currently support two ways of obtaining the table data
        # version; either passed down or fetched from the backend cache. The
        # goal is to deprecate the second option, and only support fetching
        # from the backend cache:
        table_data_version = kwargs.get('data_version',
                                        None) or self.get_data_version(
                                            table_name, skip_cache=skip_cache)

        backend_cache = _get_active_cache()

        # Fetch table sample from source data.
        def _fetch_table_sample() -> pd.DataFrame:
            logger.info(
                f"Reading a table sample for table {table_name} directly from "
                f"source data, bypassing the backend cache. Debug info: "
                f"skip_cache={skip_cache}, data_version={table_data_version}.")

            # Always store the max number of sample rows in the cache in
            # case more samples are requested the next time this function
            # is called and we don't have enough in the cache:
            table_sample = self._read_table_sample(
                self._context,
                table_name,
                num_rows if skip_cache else NUM_SAMPLE_ROWS,
                frac,
            )

            if backend_cache is not None:
                # TODO(Dulloor): Evict samples for older versions to prevent
                # the memory usage from ballooning.
                # The save parameter is not set to True in put so samples are
                # not cached in the backend cache (non-local). This is because
                # if a customer modifies the table columns, it will diverge
                # from the samples we have stored in the backend cache.
                backend_cache.table_samples_cache.put(self.name, table_name,
                                                      table_data_version,
                                                      table_sample)
            return table_sample

        # Check if the table sample is valid.
        def _check_table_sample(table_sample) -> None:
            if table_sample is None or table_sample.empty:
                raise ConnectorError(
                    f"Failed to read any rows from table '{table_name}' "
                    f"in connector '{self._name}'. "
                    f"Please check whether table was "
                    f"accidentally dropped, renamed or emptied "
                    f"since it was first imported into Kumo and "
                    f"has valid timestamp data")
            # If the user removed a column from the underlying data source
            # this may error out:
            if cols:
                for col in cols + [pkey]:
                    if (col is not None
                            and col.name not in table_sample.columns):
                        logger.warning(
                            f"Column {col.name} could not be read from the "
                            f"table sample for table {table_name}.")
                        raise MissingColumnError(
                            f'We found a misconfiguration in the '
                            f'"{table_name}" table. If you removed'
                            f'column "{col.name}" from your data source,'
                            f'please edit your Kumo Table and remove it')

        table_sample_checked = False
        if (not backend_cache or not backend_cache.table_samples_cache.has(
                self.name, table_name, table_data_version)):
            table_sample = _fetch_table_sample()
        else:
            logger.info(
                f"Obtaining a table sample for table {table_name} from the "
                f"data backend cache, avoiding connector access. Debug info: "
                f"data_version={table_data_version}.")
            table_sample = backend_cache.table_samples_cache.get_v1(
                self.name, table_name, table_data_version)
            try:
                _check_table_sample(table_sample)
                table_sample_checked = True
            except Exception as e:
                logger.warning(
                    "Failed to read table sample for table %r from the backend "
                    "cache (error_type=%s); fetching directly from the source",
                    table_name,
                    type(e).__name__)
                table_sample = _fetch_table_sample()

        # Always ensure that table sample is valid.
        if not table_sample_checked:
            _check_table_sample(table_sample)

        if cast_response:
            table_sample = self._cast_and_resample_table_sample(
                table_sample,
                pkey,
                cols,
                num_rows,
                frac,
            )

        return table_sample

    def _get_iterator(
        self, context: Any, table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None
    ) -> Iterator[pd.DataFrame]:
        r"""Internal method to return the iterator of a table."""
        raise NotImplementedError(
            f"Get iterator is not implemented for {self.__class__.__name__}.")

    @with_filesystem_credentials
    def get_iterator(
        self,
        table_name: str,
        pkey: Optional[BaseColumnConfig] = None,
        cols: Optional[List[BaseColumnConfig]] = None,
    ) -> Iterator[pd.DataFrame]:
        r"""Returns the iterator of a full table. Each elements of the iterator
        is a chunk of the table in pd.DataFrame format. The size of the element
        is controlled by the underlying connector.

        Args:
            table_name (str): The name of the table.
            pkey (BaseColumnConfig, optional): The primary key of the table.
                (default: :obj:`None`)
            cols (List[BaseColumnConfig], optional): The set of columns to
                return. If set to :obj:`None`, will return all columns of the
                table. (default: :obj:`None`)
        """
        pkey = BaseColumnConfig.cast(pkey)
        if cols is not None:
            cols = [BaseColumnConfig.cast(col) for col in cols]

        self.open()
        return self._get_iterator(self.context, table_name, pkey, cols)

    # Writing table data ######################################################

    def _write_pandas(self, context: Any, table_name: str, df: pd.DataFrame,
                      replace: bool = False) -> None:
        r"""Write dataframe to data source."""
        raise NotImplementedError(
            f"Write df is not implemented for {self.__class__.__name__}.")

    @with_filesystem_credentials
    def write_pandas(self, table_name: str, df: pd.DataFrame,
                     replace: bool = False) -> None:
        r"""
        Write a :class:`pandas.DataFrame` to the data source.

        Args:
            table_name (str): Name of the table to write to.
            df (pd.DataFrame): Dataframe to write to the table.
            replace (bool, optional): If set to :obj:`True`, replace existing
                table contents. (default: :obj:`False`)
        """

        self.open()
        self._write_pandas(self._context, table_name, df, replace)

    # Data statistics #########################################################
    def _get_cols_invalid_fraction(
        self,
        table_name: str,
        raw_df: pd.DataFrame,
        cast_df: pd.DataFrame,
        cols: List[BaseColumnConfig],
        approx: bool,
        data_version: str = None,
    ) -> Dict[BaseColumnConfig, Stats]:
        r"""Internal method to compute fraction of invalid values for
        column(s) in a table.
        """
        cols_invalid_frac_stats = {}
        data_version = data_version or self.get_data_version(table_name)

        for col in cols:
            na_raw = raw_df[col.name].isna().sum()
            na_cast = cast_df[col.name].isna().sum()
            num_entries = len(raw_df[col.name])
            if na_cast:
                invalid_frac = (na_cast - na_raw) / num_entries
            else:
                invalid_frac = 0.0
            cols_invalid_frac_stats[col] = Stats(
                value=invalid_frac,
                approx=approx,
                data_version=data_version,
                num_entries=num_entries,
                timestamp=datetime.now().timestamp(),
            )
        return cols_invalid_frac_stats

    def _get_exact_col_stats(
        self,
        table_name: str,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        **kwargs,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        r"""Internal method to compute exact column statistics for provided
        column(s) in a table.
        """
        # NOTE: The default implementation might be an expensive op depending
        # on the table size. Connector implementations should override this
        # functionality whenever applicable.
        return self._get_col_stats(table_name=table_name, approx=False,
                                   stats_dict=stats_dict, **kwargs)

    def _get_approx_col_stats(
        self,
        table_name: str,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        **kwargs,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        r"""Internal method to compute approximate column statistics for
        provided column(s) in a table. This is done by reading the relevant
        column(s) into memory and applying standard functions. Connectors are
        not expected to override this function.
        """
        return self._get_col_stats(table_name=table_name, approx=True,
                                   stats_dict=stats_dict, **kwargs)

    @trace(TraceOptions(trace_params=['table_name', 'approx']))
    def _get_col_stats(
        self,
        table_name: str,
        approx: bool,
        stats_dict: Dict[BaseColumnConfig, List[ColStatType]],
        **kwargs,
    ) -> Dict[BaseColumnConfig, Dict[ColStatType, Stats]]:
        stats_dict = copy.deepcopy(stats_dict)
        # TODO(min) support approx col stats on the specified version of an
        # ingested table
        data_version = (kwargs.get('data_version', None)
                        or self.get_data_version(table_name))
        kwargs.pop('data_version', None)

        # compute requested stats
        if approx:
            df = self.read_table_sample(
                table_name,
                cols=stats_dict.keys(),
                data_version=data_version,
                **kwargs,
            )
        else:
            df = self.read_table(
                table_name,
                cols=stats_dict.keys(),
            )

        # `ColStatType.INVALID_FRACTION`` requires special handling because
        # to compute the stat we need both raw data samples and samples
        # after casting.
        cols_invalid_frac = []
        for col, stats in stats_dict.items():
            if ColStatType.INVALID_FRACTION in stats:
                stats_dict[col] = [
                    stat for stat in stats
                    if stat != ColStatType.INVALID_FRACTION
                ]
                cols_invalid_frac.append(col)

        out: Dict[BaseColumnConfig, Dict[ColStatType, Stats]] = {}
        for col, stats in stats_dict.items():
            out[col] = compute_col_stats_python(
                df[col.name],
                stats,
                approx=approx,
                data_version=data_version,
            )

        if len(cols_invalid_frac):
            # raw df with no casting
            if approx:
                raw_df = self.read_table_sample(table_name, **kwargs)
            else:
                raw_df = self.read_table(table_name)
            cols_invalid_frac_stats = self._get_cols_invalid_fraction(
                table_name=table_name,
                raw_df=raw_df,
                cast_df=df,
                cols=cols_invalid_frac,
                approx=approx,
                data_version=data_version,
            )
            for col, stat in cols_invalid_frac_stats.items():
                out[col][ColStatType.INVALID_FRACTION] = stat

        return out

    @abstractmethod
    def _get_table_num_rows(self, table_name: str,
                            data_version: str | None = None) -> int:
        r"""Internal method to compute number of rows in a table.

        Note: data_version is only supported for parquet connector. For other
        connectors it is a no-op.
        """

    @abstractmethod
    def _get_table_source_size(self, table_name: str,
                               data_version: str | None = None) -> int:
        r"""Internal method to compute disk usage of table at source.

        Note: data_version is only supported for parquet connector. For other
        connectors it is a no-op.
        """

    def _get_selected_table_source_size(
        self,
        table_name: str,
        cols: List[BaseColumnConfig],
    ) -> int:
        r"""Internal method to compute disk usage of selected table at source.
        """
        raise NotImplementedError(
            "Get selected table source size is not implemented for "
            f"{self.__class__.__name__}.")

    def _get_table_stats(
        self,
        table_name: str,
        table_stat_types: List[TableStatType],
        data_version: str = None,
    ) -> Dict[TableStatType, Stats]:
        r"""Internal method to compute table statistics for a whole table."""
        return_dict = {}
        # TODO(min) support table stats on the specified version of an
        # ingested table
        data_version = data_version or self.get_data_version(table_name)
        if TableStatType.NUM_ROWS in table_stat_types:
            rows_value = self._get_table_num_rows(table_name,
                                                  data_version=data_version)
            return_dict[TableStatType.NUM_ROWS] = Stats(
                value=rows_value,
                approx=False,
                data_version=data_version,
            )

        if TableStatType.SOURCE_SIZE in table_stat_types:
            source_size_value = self._get_table_source_size(
                table_name, data_version=data_version)
            return_dict[TableStatType.SOURCE_SIZE] = Stats(
                value=source_size_value,
                approx=False,
                data_version=data_version,
            )

        return return_dict

    @blocking_method(tracing_options=TraceOptions(
        trace_params=['table_name', 'table_stat_types']))
    @with_filesystem_credentials
    def get_table_stats(
        self,
        table_name: str,
        table_stat_types: List[TableStatType],
        **kwargs,
    ) -> Dict[TableStatType, Stats]:
        r"""Obtains requested statistics for the specified table.

        Args:
            table_name (str): Table name.
            table_stat_types (List[TableStatType]):  The statistics to compute.
        Returns:
            Dict[TableStatType, Stats]: A dictionary mapping the
            :obj:`TableStatType` to a :obj:`Stats` object that contains the
            computed statistics.
        """
        self.open()
        data_version = (kwargs.get('data_version', None)
                        or self.get_data_version(table_name))
        return self._get_table_stats(table_name, table_stat_types, data_version)

    # TODO: Link-level statistics

    # Data and schema versioning ##############################################

    def _get_data_version(
        self,
        table_name: str,
    ) -> str:
        r"""Fetches latest data version of the provided table.

        Args:
            table_name (str): The name of the table for which to obtain the
                data version.

        Returns:
            str: The data version of the provided table.
        """
        logger.info(f"Using default data version for table {table_name}")
        return DEFAULT_DATA_VERSION

    @blocking_method(DATA_BACKEND_THREAD_POOL,
                     tracing_options=TraceOptions(trace_params=True))
    @with_filesystem_credentials
    def get_data_version(
        self,
        table_name: str,
        skip_cache: bool = False,
        save_to_cache: bool = True,
    ) -> str:
        r"""Returns the data version of the provided table. Defaults to a
        constant; connectors with more advanced functionality should override
        this method.

        Args:
            table_name (str): The name of the table for which to obtain the
                data version.
            skip_cache (bool): Indicating whether to skip looking up table data
                version in cache and fetch data version from the source
                (default: :obj:`False`)
            save_to_cache (bool): When data version is not present in cache,
                indicate whether to save the fetched data version from
                connector into cache. (default: :obj:`True`)

        Returns:
            str: The data version of the provided table.
        """
        if not skip_cache:
            data_version = _get_active_cache().table_version_cache.get(
                self.name, table_name)
            if data_version:
                return data_version

        self.open()
        data_version = self._get_data_version(table_name)
        if save_to_cache:
            _get_active_cache().table_version_cache.save_in_memory(
                self.name, table_name, data_version)
        return data_version

    # Query pushdown ##########################################################
    #
    # Custom SQL Tables push a read-only SELECT down to the customer's
    # warehouse and ingest only its result. The three methods below form
    # that capability: describe the result schema, preview a bounded sample,
    # and estimate what the query would scan. They are only meaningful for
    # SQL warehouses, so the base rejects them and the
    # Snowflake/Databricks/BigQuery connectors override them; every other
    # connector (files, Redshift, ...) inherits the clear rejection.

    def get_query_metadata(self, sql: str) -> TableMetadataConfig:
        r"""Describe the result-set schema of a read-only SELECT without
        scanning data.

        The warehouse compiles/describes ``sql`` (Snowflake ``DESCRIBE
        RESULT`` / ``cursor.describe``, Databricks ``DESCRIBE QUERY``,
        BigQuery ``dry_run``) and returns the output columns as a single
        :class:`TableMetadataConfig` — the same config
        :meth:`get_table_metadata` produces per physical table, so the
        downstream dtype/stype inference consumes query metadata unchanged.
        A query has exactly one result set, hence a single config rather
        than the list :meth:`get_table_metadata` returns per requested
        table.

        Only the result-set columns (``cols``) come from the warehouse
        describe; the config's ``table_name`` / ``source_name`` are naming
        concerns owned by the SQL-table definition (its synthetic alias)
        and are filled in by the caller.

        Default raises; SQL warehouse connectors override.
        """
        raise NotImplementedError("get_query_metadata is not supported for "
                                  f"{self.__class__.__name__}.")

    def sample_query(self, sql: str, num_rows: int) -> pd.DataFrame:
        r"""Return a bounded preview of a read-only SELECT's result.

        Runs ``sql`` bounded to ``num_rows`` (under the connector's
        statement timeout) and returns up to that many rows as a
        :class:`pandas.DataFrame`, mirroring :meth:`read_table_sample`
        for physical tables.

        Default raises; SQL warehouse connectors override.
        """
        raise NotImplementedError(
            f"sample_query is not supported for {self.__class__.__name__}.")

    #: Whether :meth:`get_query_scan_estimate_bytes` is asked before a Custom
    #: SQL sample runs; read with the estimate by
    #: :func:`kumo_core.custom_sql.sample_over_budget`.
    gates_query_scan = False

    #: Whether this connector's destination-addressed export can read a
    #: Custom SQL query rather than a named table -- that is, whether it
    #: takes a ``sql`` argument. Declared rather than introspected:
    #: ``@blocking_method()`` wraps every export in an ``RpcFuncSync`` whose
    #: signature is ``(*args, **kwargs)``, so the real parameters are not
    #: visible to :func:`inspect.signature` and a capability cannot be read
    #: off the callable. Read by
    #: :func:`rest.backends.data_backend.activities.custom_train_table.
    #: require_custom_sql_support`.
    exports_custom_sql = False

    def get_query_scan_estimate_bytes(self, sql: str) -> Optional[int]:
        r"""Estimate the bytes a read-only SELECT would scan, without running
        it, or ``None`` when the cost is unknown.

        Cost-guards the pre-ingest stype-inference sample for a Custom SQL
        Table on warehouses where a ``LIMIT`` does not reduce the scan.
        """
        return None

    def export_table(
        self,
        source_name: str,
        table_fp: IngestedTableFingerprint,
        data_version: str,
        *,
        graph_table_count: Optional[int] = None,
    ) -> ExportMetrics:
        """Export a source table to Kumo-owned parquet for ingestion.

        ``graph_table_count`` is how many tables of one graph refresh share
        this process, so a connector that sizes pools from the machine can
        divide it.

        Default raises NotImplementedError. Subclasses may stream a read-only
        source query directly into Kumo persistence or use provider staging.
        Call download() afterwards to populate the pool-local ingest root.

        Subclasses must return a populated `ExportMetrics` describing
        the unload (both `num_rows` and `source_size` are required).
        Connectors must compute both values as part of this operation so
        callers can rely on them being present.
        """
        raise NotImplementedError(
            f"export_table is not implemented for {self.__class__.__name__}.")

    def export_query(
        self,
        sql: str,
        table_fp: SqlTableFingerprint,
        data_version: str,
    ) -> ExportMetrics:
        """Query counterpart of `export_table` for a Custom SQL Table.

        Runs the read-only SELECT in the warehouse and stages only its
        result set to the same per-table ingest location `export_table` uses
        — derived from `table_fp.ingest_path_fpid` and `data_version` — so
        `download()` is unchanged. Implementations embed the SELECT only in
        parenthesized subquery position (`... FROM (<sql>)`), project/cast to
        `cols=table_fp.cols`, and reuse `export_table`'s staging path (a
        read-only Arrow stream into Kumo-owned persistence on Snowflake or an
        internal-credentialed GCS stage on BigQuery), so no separate stage or
        volume lifecycle is created here.

        Default raises NotImplementedError; SQL warehouse connectors override.
        """
        raise NotImplementedError(
            f"export_query is not implemented for {self.__class__.__name__}.")

    def download(
        self,
        source_name: str,
        table_fp: BaseTableFingerprint,
        data_version: str,
        local_dest: str,
    ) -> None:
        """Download all files for one table to local_dest.

        Default is a no-op. Subclasses override to copy or fetch files
        and may handle concurrency internally. Per-table NUM_ROWS /
        SOURCE_SIZE come from `export_table`'s `ExportMetrics`, so
        download itself returns nothing.
        """
        return None

    # Python built-ins ########################################################

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.open()
            out = func(self, *args, **kwargs)
            self.close()
            return out

        return wrapper

    def __enter__(self) -> 'Connector':
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def __getstate__(self):
        # `_open_lock` is a process-local synchronization primitive and
        # is not picklable. Strip it from the state; `__setstate__`
        # recreates a fresh lock, which is equivalent to what a freshly
        # constructed connector would have.
        state = self.__dict__.copy()
        state.pop('_open_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_lock = threading.RLock()


def fillna(df: pd.DataFrame, cols: BaseColumnConfig) -> pd.DataFrame:
    for col in cols:
        if col.na_values is not None:
            if not isinstance(col.na_values, (list, tuple)):
                raise ValueError(f"Expected a list of 'na_values' (got "
                                 f"'{col.na_values}')")
            mapping = {
                pd.Series(v, dtype=df[col.name].dtype).values[0]: pd.NA
                for v in col.na_values
            }
            df[col.name].replace(mapping, inplace=True)

    return df


def clean_backend_cache(connector_name: str):
    r"""Clean cached table data version associated with this connector"""
    if _env_bool(CLEAN_TABLE_VERSION_CACHE):
        _get_active_cache().table_version_cache.remove(connector_name)
