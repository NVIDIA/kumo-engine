# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from datetime import datetime
from typing import Any, Dict, List, Optional

from dateutil.relativedelta import relativedelta
from kumo_core.diskgraph.validation import parse_failure_warning
from kumo_core.types import ColStatType, TableStatType
from kumo_core.types.schema import SourceTableID, VersionedSourceTableID
from kumoapi.typing import Stype

from common_utils import string
from io_utils import fsutil, logging, parquet_util
from kumo.data import MaterializationStage, get_active_cache
from kumo.data.backend_cache import BackendCacheManager
from kumo.data.cache.graph_cache import VersionedGraphKey
from kumo.data.utils import get_materialize_path
from kumo.stats import Stats, is_missing_stat_value
from kumo.stats.utils import col_stat_dict_complete
from kumo.validation import (
    ValidationError,
    ValidationResponse,
    ValidationWarning,
)
from kumo.validation.validation import ValidationException
from rest.backends.data_backend.context import DataBackendContext
from rest.backends.data_backend.utils import parse_edge_stats_result
from rest.config.data_backend import (
    ConnectorExecutionRef,
    GraphDataVersionInfo,
)
from rest.config.graph import EdgeHealth
from rest.config.graph_v2 import GraphDataComputeInput
from rest.config.table import FactTableTimeRange
from rest.config.table_v2 import TableDefinition
from rest.settings import DataScaleLimits
from rest.utils.table import ColStatsSet, get_full_time_range
from service_framework.deploy_env_config import DeployEnvConfig
from service_framework.env_var import EnvVar
from statedb.config.common import TableName

logger = logging.getLogger(__name__)

MAX_NUM_DIMENSION_PER_TABLE = EnvVar(
    "MAX_NUM_DIMENSION_PER_TABLE",
    default_value=4097,
    type=int,
    description=(
        "The number of dimension per table that we support. It is usually 1"
        " per column. For embedding column, it is the sequence length. For"
        " text column, it is a constant 300."),
)
FORCE_TABLE_REINGESTION = EnvVar(
    "FORCE_TABLE_REINGESTION",
    default_value=False,
    type=bool,
    description="Whether to force re-ingestion of tables during graph refresh",
)
_GLOVE_EMBEDDING_DIM = 300


class GraphTableStatsNotReady(RuntimeError):
    """Raised when required graph table stats are not yet readable."""


# Workflow Validators #########################################################
# Note: the validations defined here should be validations that either cannot
# be performed in the core backend code or are better performed as part of
# workflow execution. A simple example is in validating the number of tables we
# have in an input graph before ingestion is performed.

# Ingestion ###################################################################


class IngestionValidator:
    MIN_REASONABLE_ROWS = 10
    MAX_DROPPED_THRESHOLD = 0.25

    def __init__(self, deploy_config: DeployEnvConfig) -> None:
        self._limits = DataScaleLimits.parse_from_env(deploy_config)

    # TODO(Dulloor): Do we need any views specific validation?
    def pre_ingestion(self, table: TableDefinition) -> ValidationResponse:
        r"""Validates a table before ingestion. The alias-name SQL-ID check
        that v1 did here lives on `TableDBEntryV2.alias` in v2 and is
        enforced at table creation time, not at ingest.
        """
        response = ValidationResponse()
        errors = response.errors

        for col in table.cols:
            try:
                string.validate_sql_id(col.name, True)
            except ValidationException as e:
                errors.append(ValidationError("Invalid column name", str(e)))

        return response

    def post_ingestion(
        self,
        table: TableDefinition,
        data_version: str,
        existing_errors: Optional[List[ValidationError]] = None,
        check_local: bool = True,
        alias: Optional[TableName] = None,
        require_materialization_stage: bool = True,
    ) -> ValidationResponse:
        r"""Validates a table after ingestion.

        Args:
            table: Table definition.
            data_version: Data version string.
            existing_errors: Errors from the ingestion process itself. If
                provided, empty table errors will be skipped since the empty
                table is a consequence of the ingestion error, not a separate
                issue.
            check_local: When True (default), the validator additionally
                consults the LOCAL materialize path with
                ``parquet_util.is_empty(...)`` to surface the
                "Ingested table empty" recovery error (e.g. malformed CSV
                rows dropped at parse time). For deferred-download
                connectors (currently BigQuery warehouse), ``export_table``
                writes only to provider staging and the
                local materialize path is not populated until the
                graph-level ``connector.download()`` runs much later, so
                callers in those code paths must pass ``check_local=False``
                to avoid a false "Ingested table empty" error. Mirrors
                the ``get_stage(check_local=...)`` opt-out at the cache
                layer.
            alias: User-facing table name (graph alias or DB alias). When
                provided, used in validation messages instead of the
                source-side table name.
            require_materialization_stage: Whether this table is expected to
                have produced a staged-file marker. Direct DiskGraph sources
                save row and size estimates here and are read later during
                graph ingest, so they do not produce that marker.
        """
        response = ValidationResponse()
        warnings = response.warnings
        errors = response.errors

        source_table_name = table.get_source_table_name()
        connector_id = table.get_connector_id()
        fp = table.fingerprint()
        display_name = alias or source_table_name

        # ``check_local=False``: deferred-download connectors export to
        # provider staging during
        # ingest and only download to the local materialize path later at
        # graph assembly time. The stage flag (saved JSON) is the
        # authoritative record of "table was ingested"; the local-path
        # existence check is a recovery sniff that does not apply at the
        # validation step.
        materialization_stage = (
            get_active_cache().materialization_stage.get_stage(
                fp,
                data_version,
                check_local=False,
            ) if require_materialization_stage else None)
        logger.debug(f"Validating table {table=} {data_version=} \
                    {materialization_stage=}")
        # TODO: move to internal validation before user-facing validator
        if require_materialization_stage and not materialization_stage:
            errors.append(
                ValidationError(
                    "Table ingestion failed",
                    f"Table {display_name} (connector {connector_id}) failed to ingest. "
                    "Check your connector configuration and source data, then retry. "
                    "If this persists, contact Kumo support."))

            return response

        cache: BackendCacheManager = get_active_cache()
        stat_response: Dict[TableStatType, Stats] = cache.get_table_stats(
            fp,
            [TableStatType.NUM_ROWS, TableStatType.SOURCE_SIZE],
            data_version,
        )

        materialized_path = get_materialize_path(fp, data_version,
                                                 MaterializationStage.INGEST)

        # NUM_ROWS may be absent here for the file/parquet connector (no footer
        # scan, kumo#30173); it is filled later from the DiskGraph node counts.
        # Absent -> emptiness falls through to the ``is_empty`` file check
        # below.
        n_rows_stat = stat_response.get(TableStatType.NUM_ROWS)
        n_rows = n_rows_stat.value if n_rows_stat is not None else None

        # Skip empty table errors if ingestion already failed - the empty
        # table is a consequence of the ingestion error, not the root cause.
        has_ingestion_error = existing_errors and len(existing_errors) > 0

        # Deferred-download connectors export to provider staging during ingest;
        # the local ``materialized_path`` is not populated until the
        # graph-level ``connector.download()`` runs at graph assembly.
        # ``parquet_util.is_empty`` would therefore report the local
        # path as empty and emit a spurious ``"Ingested table empty"``
        # error. ``n_rows`` from the saved table stats remains the
        # authoritative empty-source signal. Mirrors the
        # ``get_stage(check_local=False)`` opt-out on the cache side.
        #
        # The file-system empty sniff is a recovery check for local
        # ingest mirrors that can catch "all rows dropped" parse
        # failures. If the local mirror is unavailable in this process
        # (e.g., DG-only ingest path writes to local SSD while the
        # deploy's persist_dir is remote), don't probe the remote
        # persistent path and override positive saved row stats with a
        # best-effort file check.
        check_file_empty = check_local and fsutil.is_local(materialized_path)
        is_empty = (parquet_util.is_empty(materialized_path)
                    if check_file_empty else False)
        if n_rows is not None and n_rows == 0 and not has_ingestion_error:
            if table.is_sql:
                # FR15: an empty Custom SQL result is a query bug. This
                # per-table check runs before the graph-level gate, so emit the
                # query-specific error here rather than the generic one.
                errors.append(
                    ValidationError(
                        "SQL table returned no rows",
                        f"The Custom SQL query for the '{display_name}' table "
                        "returned no rows. Review the query's filters and "
                        "joins, then re-run."))
            else:
                errors.append(
                    ValidationError(
                        "Table is empty",
                        f"Table {display_name} (connector {connector_id}) has no rows. "
                        "Check source data, connector configuration, and any filters or views."
                    ))
        elif check_file_empty and is_empty and not has_ingestion_error:
            if table.time_col:
                message = (
                    f'Ingested table {display_name} has no rows. '
                    f'Rows may have been dropped due to malformed data (CSV) or invalid timestamps in "Create date" column. '
                    f'Check connector, source data, and time column selection, then re-ingest.'
                )
            else:
                message = (
                    f'Ingested table {display_name} has no rows. '
                    'Check source data and connector configuration, then re-ingest.'
                )
            errors.append(ValidationError("Ingested table empty", message))
        elif n_rows is not None and 0 < n_rows < self.MIN_REASONABLE_ROWS:
            warning = ValidationWarning(
                "Table is very small",
                f"Table {display_name} (connector {connector_id}) has only {n_rows} rows. "
                "If this is unexpected, check source data and filters.")
            logger.warning(warning)
        # Log details when error happens.
        if len(errors) > 0 or len(warnings) > 0:
            logger.error(f"Validation error or warnings with "
                         f"{connector_id}."
                         f"{source_table_name} and data version "
                         f"{data_version} and path: {materialized_path}")
        response.merge(
            self._limits.validate_table(
                source_table_name,
                size_bytes=stat_response[TableStatType.SOURCE_SIZE].value))

        return response

    def _handle_maybe_empty(self, table: TableDefinition,
                            data_version: str) -> bool:
        r"""Handles the case where a table is empty after ingestion.
            Returns True if the table was empty and False otherwise.
        """
        table_fingerprint = table.fingerprint()
        materialized_path = get_materialize_path(table_fingerprint,
                                                 data_version,
                                                 MaterializationStage.INGEST)
        if parquet_util.is_empty(materialized_path):
            get_active_cache().materialization_stage.remove_stage(
                table_fingerprint, data_version)
            return True
        return False

    def validate_existing_data(self, table: TableDefinition,
                               data_version: str) -> bool:
        r"""Handles the case where existing data or table stats are invalid.
            Returns True if data and table stats are valid. False otherwise.
        """
        # Check if forced re-ingestion is enabled
        if FORCE_TABLE_REINGESTION.parse_value():
            return False
        if self._handle_maybe_empty(table, data_version):
            return False
        # Check if the table stats exist
        stats = get_active_cache().get_table_stats(
            table.fingerprint(),
            [TableStatType.NUM_ROWS, TableStatType.SOURCE_SIZE], data_version)
        if len(stats) != 2:
            return False
        return True

    def error_should_retry(self, error: ValidationError) -> bool:
        # TODO: better organize and classify errors as retryable, possibly
        # using an attribute on the actual error
        if (error.title == 'Ingested table empty'
                and 'due to invalid time column' not in error.message):
            return True
        return False


# Warn if more than 10% of any column contains invalid entries
MAX_INVALID_ROW_FRACTION = 0.1
# Warn if 100% of any column contains missing/invalid entries
MAX_BAD_ROW_FRACTION = 1
# Warn if more than 5% of the time_col contains missing/invalid entries
MAX_TIME_COL_BAD_FRACTION = 0.05
# Warn if more than 0% of the primary_key contains missing/invalid entries
MAX_PKEY_BAD_FRACTION = 0.00
# Warn if timeframe contains less than 5% of average timeframe fact count
MAX_MISSING_TIMEFRAME_FRACTION = 0.05


class ColStatsValidator:
    def post_col_stats(
        self,
        connector_name: str,
        table: TableDefinition,
        data_version: str,
        alias: Optional[TableName] = None,
        pkey_duplicate_count: Optional[int] = None,
    ) -> ValidationResponse:
        r"""Validates a table after (exact) col stats.

        Args:
            connector_name: ID of the connector hosting this table.
            table: Table definition.
            data_version: Data version string.
            alias: User-facing table name (graph alias or DB alias). When
                provided, used in validation messages instead of the
                source-side table name.
            pkey_duplicate_count: Exact rows dropped by DiskGraph for repeated
                primary keys.
        """
        response = ValidationResponse()

        bad_time_col_fraction: float = 0.0
        bad_time_col_count: int = 0
        bad_pkey_fraction: float = 0.0
        bad_pkey_count: int = 0
        dup_pkey_count: int = 0
        dup_pkey_fraction: float = 0.0
        invalid_col_names: List[str] = []
        all_bad_col_names: List[str] = []
        uniform_col_names: List[str] = []

        fp = table.fingerprint()
        table_stats = get_active_cache().get_table_stats(
            fp, [TableStatType.NUM_ROWS], data_version)
        num_source_rows = _get_table_stat(TableStatType.NUM_ROWS, table_stats)

        stat_dict = ColStatsSet.FULL.get_stats_types(table)
        # Optional diagnostic: older snapshots do not contain this counter.
        for kinds in stat_dict.values():
            kinds.append(ColStatType.NUM_PARSE_FAILURES)
        all_stat_dict = get_active_cache().get_col_stats(
            fp, stat_dict, data_version, approx=False)

        required = ColStatsSet.MINIMAL.get_stats_types(table)
        for stats in required.values():
            stats.append(ColStatType.COUNT)
        stale = sorted({
            col.name
            for col, kinds in required.items()
            for kind in kinds
            if (stat := all_stat_dict.get(col, {}).get(kind)) is not None and (
                stat.approx or str(stat.data_version) != str(data_version))
        })
        if num_source_rows and (
                not col_stat_dict_complete(all_stat_dict, required) or stale):
            response.errors.append(
                ValidationError(
                    title="Incomplete table statistics",
                    message=("The exact statistics manifest is missing or "
                             f"stale (stale={stale}). The snapshot was not "
                             "completed.")))

        display_name = alias or table.get_source_table_name()
        for col_config, col_stats in all_stat_dict.items():
            missing_fraction: float = _get_col_stat(ColStatType.NA_FRACTION,
                                                    col_stats) or 0
            num_unique: float = _get_col_stat(ColStatType.NUM_UNIQUE,
                                              col_stats) or 0
            invalid_fraction: float = _get_col_stat(
                ColStatType.INVALID_FRACTION, col_stats) or 0

            missing_or_invalid: float = missing_fraction + invalid_fraction
            # TODO: switch this to NUM_INVALID when the stat is implemented
            invalid_count = invalid_fraction * \
                (_get_col_stat(ColStatType.COUNT, col_stats) or 0)
            missing_count = _get_col_stat(ColStatType.NUM_NA, col_stats) or 0
            missing_or_invalid_count = int(missing_count + invalid_count)

            # Get column-level dropped rows stats
            col_dropped_rows = _get_col_stat(ColStatType.DROPPED_ROWS,
                                             col_stats) or 0

            # Validate time col
            if col_config.name == table.time_col:
                if col_dropped_rows:
                    missing_or_invalid_count = col_dropped_rows
                    if not num_source_rows or num_source_rows == 0:
                        # use zero to avoid warning about bad time cols,
                        # since more importantly there are no data rows
                        missing_or_invalid = 0.0
                    else:
                        missing_or_invalid = (missing_or_invalid_count /
                                              num_source_rows)

                # Check large % of missing time col values
                if missing_or_invalid >= MAX_TIME_COL_BAD_FRACTION:
                    bad_time_col_fraction = missing_or_invalid
                    bad_time_col_count = missing_or_invalid_count

            # Validate pkey entries.
            elif col_config.name == table.pkey:
                unique = _get_col_stat(ColStatType.NUM_UNIQUE, col_stats)
                count = _get_col_stat(ColStatType.COUNT, col_stats)
                # Check large % of missing pkey values
                has_bad_pkeys = (missing_or_invalid > MAX_PKEY_BAD_FRACTION)
                if has_bad_pkeys:
                    bad_pkey_fraction = missing_or_invalid
                    bad_pkey_count = missing_or_invalid_count
                if pkey_duplicate_count is not None:
                    dup_pkey_count = pkey_duplicate_count
                    dup_pkey_fraction = (dup_pkey_count /
                                         count if count else 0.0)
                elif (not has_bad_pkeys and unique is not None
                      and count is not None and count != unique):
                    dup_pkey_count = (count - unique)
                    dup_pkey_fraction = dup_pkey_count / count

            # Validate all other cols
            else:
                parse_failures = _get_col_stat(ColStatType.NUM_PARSE_FAILURES,
                                               col_stats) or 0
                if parse_failures:
                    response.warnings.append(
                        ValidationWarning(
                            title="Invalid column values",
                            message=parse_failure_warning(
                                display_name, col_config.name,
                                int(parse_failures))))
                    continue
                # All missing or invalid
                if missing_or_invalid >= MAX_BAD_ROW_FRACTION:
                    all_bad_col_names.append(col_config.name)
                # Large fraction of invalid entries.
                elif (invalid_fraction >= MAX_INVALID_ROW_FRACTION):
                    invalid_col_names.append(col_config.name)
                elif (missing_or_invalid_count == 0 and num_unique == 1):
                    uniform_col_names.append(col_config.name)

        # FR15: a Custom SQL table's rows come from a user-authored query, so
        # an empty result is almost always a query bug the user must see. Raw
        # tables keep their lenient behavior.
        if table.is_sql and not num_source_rows:
            response.errors.append(
                ValidationError(
                    title="SQL table returned no rows", message=(
                        f"The Custom SQL query for the '{display_name}' table "
                        "returned no rows. Review the query's filters and "
                        "joins, then re-run.")))

        if bad_time_col_fraction != 0.0:
            bad_row_desc = _get_bad_row_desc(
                bad_time_col_count,
                bad_time_col_fraction,
            )
            message = (
                f"{bad_row_desc} in the '{display_name}' table's "
                f"time col ({table.time_col}) are either missing or "
                f"could not be parsed. These rows are not ingested and "
                f"will not be used in Kumo. "
                "Fix by cleaning the time column or selecting the correct time column. "
                "NOTE: Column statistics on this table are calculated after dropping invalid rows."
            )
            validation_warning = ValidationWarning(
                title="Missing or invalid time column values", message=message)
            response.warnings.append(validation_warning)
            logger.warning(message)
        if bad_pkey_fraction != 0.0:
            bad_row_desc = _get_bad_row_desc(
                bad_pkey_count,
                bad_pkey_fraction,
            )
            if table.is_sql:
                # FR15: null/unparseable primary keys from a SQL result are a
                # query bug; block loudly instead of the raw-table warning.
                response.errors.append(
                    ValidationError(
                        title="Null primary keys in SQL table", message=(
                            f"{bad_row_desc}the '{display_name}' table's "
                            f"primary key column ({table.pkey}) is null or "
                            "could not be parsed. Fix the Custom SQL query so "
                            "the primary key is non-null for every row.")))
            else:
                pkey_message = (
                    f"{bad_row_desc}"
                    f"the '{display_name}' table's "
                    f"primary key column ({table.pkey}) "
                    f"is either missing or could not be parsed. "
                    "These rows will be ignored during training. "
                    "Fix by cleaning the primary key column or choosing a different primary key."
                )
                response.warnings.append(
                    ValidationWarning(
                        title="Missing or invalid primary keys",
                        message=pkey_message,
                    ))
        if dup_pkey_count != 0:
            bad_row_desc = _get_bad_row_desc(
                dup_pkey_count,
                dup_pkey_fraction,
            )
            message = (f"{bad_row_desc}"
                       f"the '{display_name}' table's "
                       f"primary key column ({table.pkey}) "
                       f"are duplicated. Consider cleaning your data by "
                       f"removing rows with duplicate primary keys for "
                       f"better model performance, or do not specify "
                       f"primary key if rows in the table are not used "
                       f"as entities. During DiskGraph ingest, one row is "
                       f"kept for each primary key value and the remaining "
                       f"duplicate rows are skipped; their features are not "
                       f"aggregated. "
                       f"NOTE: Column statistics are calculated after "
                       f"dropping invalid rows.")
            logger.warning(message)
            response.warnings.append(
                ValidationWarning(title="Duplicate primary keys",
                                  message=message))

        if len(invalid_col_names) > 0:
            invalid_message = (
                f"The following columns in the "
                f"'{display_name}' table have more than "
                f"{MAX_INVALID_ROW_FRACTION} fraction of the values "
                f"as invalid: {invalid_col_names}. These entries "
                "will be ignored during training. "
                "Consider cleaning these columns or removing them from the training data."
            )
            response.warnings.append(
                ValidationWarning(
                    title="Large number invalid entries",
                    message=invalid_message,
                ))

        _validate_feature_dimensions(table, all_stat_dict, response,
                                     display_name)

        return response


def _validate_feature_dimensions(
    table: Any,
    all_stat_dict: Dict[Any, Dict[ColStatType, Stats]],
    response: ValidationResponse,
    display_name: str,
    max_dimensions: Optional[int] = None,
):
    total_dimensions = 0
    for col in table.cols:
        if col.stype == Stype.sequence:
            col_config = table.col_to_base_config(col.name)
            assert col_config is not None
            col_stats = all_stat_dict.get(col_config)
            if not col_stats:
                response.errors.append(
                    ValidationError(
                        title="Missing embedding statistics",
                        message=f"Embedding column '{col.name}' is missing "
                        "sequence length statistics. Recompute column stats "
                        "or re-ingest the table.",
                    ))
                continue
            min_seq_len = _get_col_stat(ColStatType.SEQUENCE_MIN_LENGTH,
                                        col_stats)
            max_seq_len = _get_col_stat(ColStatType.SEQUENCE_MAX_LENGTH,
                                        col_stats)
            if min_seq_len is None or max_seq_len is None:
                response.errors.append(
                    ValidationError(
                        title="Missing embedding statistics",
                        message=f"Embedding column '{col.name}' is missing "
                        "sequence length statistics. Recompute column stats "
                        "or re-ingest the table.",
                    ))
                continue
            total_dimensions += max_seq_len
            if min_seq_len != max_seq_len:
                response.errors.append(
                    ValidationError(
                        title="Inconsistent embedding dimensions",
                        message=f"Embedding column '{col.name}' "
                        "has inconsistent "
                        f"dimensions (min: {min_seq_len}, max: {max_seq_len}). "
                        "The number of features must be the same for all rows. "
                        "Please check your source data and re-ingest.",
                    ))
        elif col.stype == Stype.text:
            total_dimensions += _GLOVE_EMBEDDING_DIM
        else:
            total_dimensions += 1

    max_dimensions = (MAX_NUM_DIMENSION_PER_TABLE.parse_value()
                      if max_dimensions is None else max_dimensions)
    if total_dimensions > max_dimensions:
        response.errors.append(
            ValidationError(
                title="Too many feature dimensions",
                message=f"Table {display_name} has "
                f"{total_dimensions} "
                "feature dimensions, exceeding the limit of "
                f"{max_dimensions}. "
                "Each column counts as 1 dimension, and `Text` columns"
                " are automatically converted to embeddings (each contributing "
                f"{_GLOVE_EMBEDDING_DIM} dimensions based on the embedding size).\n"
                f"Reduce columns or text fields (text columns "
                f"contribute {_GLOVE_EMBEDDING_DIM} dimensions each), or "
                "edit the training data to reduce dimensionality.",
            ))


def _get_col_stat(typ: ColStatType, stats: Dict[ColStatType,
                                                Stats]) -> Optional[Any]:
    if typ not in stats:
        return None
    return stats[typ].value


def _get_table_stat(typ: TableStatType, stats: Dict[TableStatType,
                                                    Stats]) -> Optional[Any]:
    if typ not in stats:
        return None
    return stats[typ].value


def _get_bad_row_desc(count: int, fraction: float) -> str:
    """
        If fraction is lower than 0.01, then show a different description.
        This is because the UI limites rendering of the error to first decimal
        place only, resulting in a weird error description like:
        0.1
        """
    if fraction < 0.01:
        return f"{count} of the rows in "
    return f"{count} ({fraction*100:0.1f}%) of the rows in "


class GraphRefreshValidator:
    # Warn if graph edge match rate is less than 20%
    # Must be kept in sync with graphEdgeThreshold in constants.js
    GRAPH_EDGE_MATCH_WARN_THRESHOLD = 20

    # Maximum allowed time range gap in months.
    MAX_ALLOWED_GAP_MONTHS = 1

    def __init__(self, deploy_config: DeployEnvConfig) -> None:
        self._limits = DataScaleLimits.parse_from_env(deploy_config)

    def post_table_ingest(
        self,
        ctx: DataBackendContext,
        versioned_table_dict: Dict[VersionedSourceTableID, TableDefinition],
        connector_specs: Optional[List[ConnectorExecutionRef]] = None,
    ) -> ValidationResponse:
        r"""Validate a graph after all tables are ingested."""
        total_size = 0
        cache = get_active_cache()
        exact_stats = cache.get_many_min_exact_stats(
            [(k, v.fingerprint()) for k, v in versioned_table_dict.items()],
            skip_col_stats=True)
        stats = []
        for (versioned_id, table), exact in zip(versioned_table_dict.items(),
                                                exact_stats):
            stats.append(exact.table_stats if exact is not None else None)
        tables_missing_stat = []
        tables_missing_source_size = []
        # FR15: Custom SQL tables whose query returned no rows (a query bug).
        sql_empty_tables: List[str] = []
        for versioned_table_id, stat in zip(list(versioned_table_dict.keys()),
                                            stats):
            if stat is None:
                tables_missing_stat.append(versioned_table_id)
                continue
            # SOURCE_SIZE is written before publishing INGEST_TABLE; if absent
            # the stats cache is not ready, so let the activity retry. NUM_ROWS
            # is not required here -- it is filled from the DiskGraph node counts
            # at graph-build time (after this validator). Connectors that count
            # for free (SQL/warehouse) may set it early, but its absence must
            # not block graph validation.
            if TableStatType.SOURCE_SIZE not in stat:
                tables_missing_source_size.append(versioned_table_id)
                continue
            num_rows_stat = stat.get(TableStatType.NUM_ROWS)
            num_rows = (int(num_rows_stat.value)
                        if num_rows_stat is not None else None)
            total_size += int(stat[TableStatType.SOURCE_SIZE].value)
            table_def = versioned_table_dict[versioned_table_id]
            if table_def.is_sql and num_rows == 0:
                sql_empty_tables.append(table_def.get_source_table_name())
        if tables_missing_stat:
            raise GraphTableStatsNotReady(
                f"Exact table stats are missing for {tables_missing_stat}. "
                "Try re-ingesting the affected tables or refreshing the graph.")
        if tables_missing_source_size:
            raise GraphTableStatsNotReady(
                f"Required table stats missing (SOURCE_SIZE) for "
                f"{tables_missing_source_size}. Try re-ingesting the table or "
                "refreshing the graph data version.")
        response = ValidationResponse()
        response.merge(
            self._limits.validate_graph(
                total_size_bytes=total_size,
                raise_on_error=False,
                config=ctx._config,
                connector_specs=connector_specs,
            ))
        for name in sql_empty_tables:
            response.errors.append(
                ValidationError(
                    title="SQL table returned no rows", message=(
                        f"The Custom SQL query for the '{name}' table returned "
                        "no rows. Review the query's filters and joins, then "
                        "re-run.")))
        return response

    def post_col_stats(
        self,
        versioned_table_dict: Dict[VersionedSourceTableID, TableDefinition],
        aliases: Optional[Dict[VersionedSourceTableID, TableName]] = None,
    ) -> ValidationResponse:
        r"""Validate a graph after all table col stats are computed;
            currently just checks time ranges of fact tables.

        Args:
            versioned_table_dict: Per-table definitions keyed by versioned
                source-table id.
            aliases: Optional graph-level aliases keyed the same way. When
                provided, used in user-facing time-gap warnings instead of
                the source-side table name.
        """
        response = ValidationResponse()
        aliases = aliases or {}

        # Get the stats for all source ids.
        source_ids = [
            SourceTableID(connector_name=v_id.connector_name,
                          source_table_name=v_id.source_table_name)
            for v_id in versioned_table_dict.keys()
        ]
        exact_stats = get_active_cache().get_many_min_exact_stats(
            [(k, v.fingerprint()) for k, v in versioned_table_dict.items()],
            skip_col_stats=False)

        # Create a map of source id to stats.
        src_to_stats_map = dict(zip(source_ids, exact_stats))

        # Get time ranges for all FACT tables.
        table_time_range: Dict[VersionedSourceTableID,
                               Optional[FactTableTimeRange]] = {}
        for tid, table in versioned_table_dict.items():
            time_range = None
            if table:
                stats = src_to_stats_map[table.source_table_id]
                if (table.time_col is not None and stats is not None):
                    time_range = get_full_time_range(table, stats)
            table_time_range[tid] = time_range
        # Validate time range for all tables in the graph.
        # Collect all time gaps, then aggregate into a single warning.
        time_gap_entries: List[Dict] = []
        # Compare each table's min(earliest) and max(latest) in this graph.
        for t1, t2 in itertools.combinations(versioned_table_dict.keys(), 2):
            tr_t1: Optional[FactTableTimeRange] = table_time_range[t1]
            tr_t2: Optional[FactTableTimeRange] = table_time_range[t2]
            # Prefer the graph-level alias when caller passes one.
            name_1 = aliases.get(
                t1) or versioned_table_dict[t1].get_source_table_name()
            name_2 = aliases.get(
                t2) or versioned_table_dict[t2].get_source_table_name()
            if (tr_t1 and tr_t2):
                # Process 'earliest' (min) time range.
                if (not is_missing_stat_value(tr_t1.min)
                        and not is_missing_stat_value(tr_t2.min)):
                    entry = self._get_time_gap_entry(name_1, name_2, tr_t1.min,
                                                     tr_t2.min, "earliest")
                    if entry:
                        time_gap_entries.append(entry)
                # Process 'latest' (max) time range.
                if (not is_missing_stat_value(tr_t1.max)
                        and not is_missing_stat_value(tr_t2.max)):
                    entry = self._get_time_gap_entry(name_1, name_2, tr_t1.max,
                                                     tr_t2.max, "latest")
                    if entry:
                        time_gap_entries.append(entry)

        if time_gap_entries:
            # Sort by largest gap first and create a single aggregated warning.
            sorted_entries = sorted(time_gap_entries, key=lambda e: e["gap"],
                                    reverse=True)
            warning = self._create_time_gap_warning(sorted_entries)
            response.warnings.append(warning)

        return response

    def post_edge_stats(
            self, input: GraphDataComputeInput,
            graph_version_info: GraphDataVersionInfo) -> ValidationResponse:
        r"""Validate edges of a graph after edge stats are computed"""
        if len(input.tables) <= 1:
            logger.info("Skipping edge stats validation for single table.")
            return ValidationResponse()

        response = ValidationResponse()
        graph_version_key = VersionedGraphKey(
            input.graph_key, graph_version_info.graph_data_version)
        edge_stats = get_active_cache().graph_cache.\
            get_graph_edge_stats(graph_version_key)
        if not edge_stats:
            raise ValidationException(
                'Graph edge stats is missing after computed.')
        edge_health_list = parse_edge_stats_result(input, edge_stats)

        # 1. scale validation
        scale_validation = self._limits.validate_graph(edge_counts=[
            e.edge_count for e in edge_health_list if e.edge_count > 0
        ])
        response.merge(scale_validation)

        # 2. FR15 SQL-table edge gates. The general low-match warning was
        # removed in #13909 as too noisy for raw tables; re-introduce it only
        # for edges touching a Custom SQL table, where a degenerate join is
        # almost always a query bug.
        self._validate_sql_edge_health(input, edge_health_list, response)

        return response

    def _validate_sql_edge_health(
        self,
        input: GraphDataComputeInput,
        edge_health_list: List[EdgeHealth],
        response: ValidationResponse,
    ) -> None:
        r"""FR15 edge gates for edges that touch a Custom SQL table: a 0-edge
        (no matching rows) is an error naming the offending SQL column(s); a
        match rate below ``GRAPH_EDGE_MATCH_WARN_THRESHOLD`` is a warning.
        Edges between raw tables only are left untouched. Runs on the
        already-computed edge stats.
        """
        sql_table_names = {
            name
            for name, tbl in input.graph_definition.tables.items() if tbl.is_sql
        }
        if not sql_table_names:
            return

        threshold = self.GRAPH_EDGE_MATCH_WARN_THRESHOLD
        for edge in edge_health_list:
            sql_cols = [
                col for col in (edge.col_a, edge.col_b)
                if col.table_name in sql_table_names
            ]
            if not sql_cols:
                continue
            named_cols = ", ".join(f"{c.table_name}.{c.column_name}"
                                   for c in sql_cols)
            pair = (f"{edge.col_a.table_name}.{edge.col_a.column_name} and "
                    f"{edge.col_b.table_name}.{edge.col_b.column_name}")
            if edge.edge_count == 0:
                response.errors.append(
                    ValidationError(
                        title="SQL table edge matches no rows", message=(
                            f"The link between {pair} produced zero matching "
                            f"rows. Check the join key(s) in the Custom SQL "
                            f"table column(s): {named_cols}.")))
                continue
            # Strictly below the configured threshold (matches the "less than"
            # semantics of the frontend graphEdgeThreshold check).
            a_pct = edge.percents.a_in_b
            b_pct = edge.percents.b_in_a
            if (a_pct is not None and b_pct is not None
                    and (a_pct < threshold or b_pct < threshold)):
                response.warnings.append(
                    ValidationWarning(
                        title="Low SQL table edge match rate", message=(
                            f"The link between {pair} has a match rate below "
                            f"{threshold}%. If this is unexpected, check the "
                            "join key(s) in the Custom SQL query.")))

    def _get_time_gap_entry(self, name_1: str, name_2: str,
                            t1_min_max: datetime, t2_min_max: datetime,
                            period: str) -> Optional[Dict]:
        """
        Detects time range gaps between two tables.
        Returns a dict with gap info if a significant gap exists, None otherwise.
        """
        if (is_missing_stat_value(t1_min_max)
                or is_missing_stat_value(t2_min_max)):
            return None
        # relativedelta(date2, date1) is positive if date2 occurs after date1
        delta = relativedelta(t2_min_max, t1_min_max)
        gap_months = (delta.years * 12 + delta.months + int(delta.days / 30))

        abs_gap = abs(gap_months)
        if abs_gap <= self.MAX_ALLOWED_GAP_MONTHS:
            return None

        # Normalize so earlier_table has the earlier date
        if gap_months > 0:
            earlier_table, later_table = name_1, name_2
            earlier_date, later_date = t1_min_max, t2_min_max
        else:
            earlier_table, later_table = name_2, name_1
            earlier_date, later_date = t2_min_max, t1_min_max

        return {
            "gap": abs_gap,
            "period": period,
            "earlier_table": earlier_table,
            "later_table": later_table,
            "earlier_date": earlier_date,
            "later_date": later_date,
        }

    def _create_time_gap_warning(self,
                                 entries: List[Dict]) -> ValidationWarning:
        """
        Creates a single aggregated warning from multiple time gap entries.
        Lists all gaps with full date details as bullet points.
        """
        header = (
            "When connected tables have large temporal gaps in their "
            "records, data may be dropped. Please review your data sources "
            "if this is unexpected:")

        bullets = []
        for entry in entries:
            earlier_date = entry["earlier_date"].date()
            later_date = entry["later_date"].date()
            if entry["period"] == "earliest":
                bullet = (f"{entry['earlier_table']} ({earlier_date} UTC) is "
                          f"{entry['gap']} months before the earliest fact in "
                          f"{entry['later_table']} ({later_date} UTC)")
            else:
                bullet = (f"{entry['later_table']} ({later_date} UTC) is "
                          f"{entry['gap']} months after the latest fact in "
                          f"{entry['earlier_table']} ({earlier_date} UTC)")
            bullets.append(f"\n- {bullet}")

        message = header + "".join(bullets)

        return ValidationWarning(
            title="Possible Data Missing",
            message=message,
        )
