# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timezone
from typing import NoReturn, Tuple

from kumo_core.sources.errors import (
    PermanentSourceError,
    ReconnectRequiredError,
    RowCountMismatchError,
    SchemaChangedError,
    SourceAccessDeniedError,
    SourceError,
    SourceNotFoundError,
    TransientSourceError,
    UnsupportedReadModeError,
)
from kumo_core.types import TableStatType
from kumo_core.version import get_kumo_version
from temporalio import activity

from common_utils.kumo_timing import kumo_timing
from io_utils import fsutil, logging
from kumo.connector.connector import ExportMetrics
from kumo.connector.file_connector import FileConnector
from kumo.data import MaterializationStage
from kumo.data.backend_cache import (
    DirectDiskgraphSourceDecision,
    TableIngestionInfo,
)
from kumo.data.custom_sql import (
    advisory_read_only_check,
    subquery_safe_sql,
    validate_connector_sql_scope,
)
from kumo.data.utils import get_materialize_path
from kumo.stats.stats import Stats
from kumo.validation import ValidationResponse
from kumo.validation.validation import ValidationError
from rest.backends.data_backend.utils import (
    _connector_type_from_spec,
    data_backend_cache,
    get_data_operation_description,
)
from rest.backends.data_backend.validation import IngestionValidator
from rest.config.data_backend import ConnectorExecutionRef
from rest.config.table_v2 import TableDefinition
from rest.utils.driver_common import get_connector_from_spec
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.table import TableDataVersion
from statedb.utils.errors import (
    ConnectorValidationError,
    IdentityResolutionUnavailableError,
    IdentityValidationError,
    TableIngestionError,
    TableIngestionOutOfMemoryError,
)
from temporalio_commons.error_utils import (
    extract_public_exception,
    is_error_non_retryable,
)

from ..connector_activity_registry import connector_authorized_activity

log = logging.getLogger(__name__)

SAFE_TABLE_INGESTION_ERROR_MESSAGE = (
    'Kumo encountered an error while ingesting the source table.')

# These errors are constructed without a provider exception or provider
# payload: the first three by the identity/connector broker, the OOM error by
# the connector export path with a fixed local message. No other
# PublicException is trusted here: generic connector exceptions historically
# classified and embedded raw warehouse messages.
_SAFE_INGESTION_PUBLIC_ERRORS = (
    ConnectorValidationError,
    IdentityResolutionUnavailableError,
    IdentityValidationError,
    TableIngestionOutOfMemoryError,
)

# Composed from the error's TYPE and its own numeric fields, never from its
# text: these types are also reached by translating a vendor exception, whose
# text must not be republished. Keyed on the exact type, so a subclass added
# later inherits no message that was not written for it.
_SAFE_SOURCE_ERROR_MESSAGES: dict[type[SourceError], str] = {
    RowCountMismatchError:
    ("Table '{table}' was read incompletely: the source returned {actual} "
     'rows but reported {expected}. Re-run the ingest; if it happens again, '
     'contact Kumo support.'),
    SourceNotFoundError:
    ("Table '{table}' does not exist in the source. Check the table name, "
     'and that the connector points at the database and schema that holds '
     'it.'),
    SchemaChangedError:
    ("The columns of table '{table}' changed while Kumo was reading it. "
     'Re-run the ingest once the schema change has finished.'),
    SourceAccessDeniedError:
    ("Kumo is not permitted to read table '{table}'. Grant the credentials "
     'this connector uses read access to the table, then re-run the ingest.'),
    ReconnectRequiredError:
    ("The connector Kumo used to read table '{table}' is no longer "
     'authorized. Reconnect it, then re-run the ingest.'),
    UnsupportedReadModeError:
    ("Kumo cannot read table '{table}' by any method this connector "
     'supports. Contact Kumo support with the table name.'),
}

# Deliberately opaque: these are the buckets an untranslatable vendor
# exception lands in, so only the vendor's text -- which can carry table
# contents, credentials or signed URLs -- would tell two of them apart.
_OPAQUE_SOURCE_ERRORS: tuple[type[SourceError], ...] = (
    SourceError,
    TransientSourceError,
    PermanentSourceError,
)


def _source_table_name(table: TableDefinition | None) -> str | None:
    try:
        return table.get_source_table_name() if table is not None else None
    except Exception:
        return None


def _safe_source_error_message(
    error: SourceError,
    table: TableDefinition | None,
) -> str | None:
    template = _SAFE_SOURCE_ERROR_MESSAGES.get(type(error))
    name = error.table or _source_table_name(table)
    if template is None or name is None:
        return None
    return template.format(**{**vars(error), 'table': name})


def _is_non_retryable(error: Exception) -> bool:
    """``is_error_non_retryable`` sees a plain ``Exception`` and calls every
    one of them retryable, so a missing grant would be retried until the
    activity ran out of attempts.
    """
    if isinstance(error, SourceError):
        return isinstance(error, PermanentSourceError)
    return is_error_non_retryable(error)


def _raise_sanitized_table_ingestion_error(
    error: Exception,
    table: TableDefinition | None = None,
) -> NoReturn:
    public_error = extract_public_exception(error)
    log.error('Table ingestion failed (error_type=%s)', type(error).__name__)
    if isinstance(public_error, _SAFE_INGESTION_PUBLIC_ERRORS):
        raise public_error from None
    message = None
    if isinstance(error, SourceError):
        message = _safe_source_error_message(error, table)
    raise TableIngestionError(
        message=message or SAFE_TABLE_INGESTION_ERROR_MESSAGE,
        non_retryable=_is_non_retryable(error),
    ) from None


@connector_authorized_activity
def ingest_table_data_activity_v2(
    connector_spec: ConnectorExecutionRef,
    table: TableDefinition,
    data_version: TableDataVersion,
    job_id: str,
    deploy_config: DeployEnvConfig,
    graph_table_count: int | None = None,
) -> Tuple[str, ValidationResponse]:
    try:
        return _ingest_table_data_activity_v2(
            connector_spec,
            table,
            data_version,
            job_id,
            deploy_config,
            graph_table_count,
        )
    except Exception as error:
        _raise_sanitized_table_ingestion_error(error, table)


def _ingest_table_data_activity_v2(
    connector_spec: ConnectorExecutionRef,
    table: TableDefinition,
    data_version: TableDataVersion,
    job_id: str,
    deploy_config: DeployEnvConfig,
    graph_table_count: int | None = None,
) -> Tuple[str, ValidationResponse]:
    """Prepare one table for graph-level DiskGraph ingestion.

    File-backed connectors stage Parquet as before. Databricks computes only
    the row and placement estimates needed before pool admission; DiskGraph
    reads the final SQL directly on the pool daemon.
    """
    source_table_name = table.get_source_table_name()
    fingerprint = table.fingerprint()

    with kumo_timing(
            "table_ingest_activity.pre_validation",
            source_table_name=source_table_name,
    ):
        validator = IngestionValidator(deploy_config)
        validation_response = ValidationResponse()

        validation_response.merge(validator.pre_ingestion(table))
    if not validation_response.ok:
        return '', validation_response

    workflow_info = activity.info()
    with kumo_timing(
            "table_ingest_activity.setup_backend",
            source_table_name=source_table_name,
    ):
        data_backend_cache().table_version_cache.save_in_memory(
            connector_spec.connector_id, source_table_name, data_version)
        connector = get_connector_from_spec(connector_spec)

    primary_error: BaseException | None = None
    try:
        direct_file_source = None
        if isinstance(connector, FileConnector):
            source_decision_cache = (
                data_backend_cache().direct_diskgraph_source_decision)
            source_decision = source_decision_cache.get(
                fingerprint,
                str(data_version),
                cls=DirectDiskgraphSourceDecision,
            )
            if source_decision is None:
                if connector.uses_direct_diskgraph_source_for(
                        source_table_name):
                    direct_file_source = connector.diskgraph_source(
                        source_name=source_table_name)
                source_decision = DirectDiskgraphSourceDecision(
                    source=direct_file_source, )
                source_decision_cache.put(
                    fingerprint.connector_name,
                    source_table_name,
                    data_version,
                    source_decision,
                    save=True,
                )
            else:
                direct_file_source = source_decision.source
            direct_diskgraph_source = direct_file_source is not None
        else:
            # Only flat cloud Parquet under an ambient identity is read
            # in place; every warehouse stages to Kumo-owned Parquet first.
            direct_diskgraph_source = False

        # Every connector that stages writes Parquet to the Kumo materialize
        # path, so a stage flag is only trustworthy alongside those files: one
        # set under a different ``data_persist_dir`` would otherwise skip
        # ``export_table`` against an empty new persist dir.
        if not direct_diskgraph_source:
            with kumo_timing(
                    "table_ingest_activity.materialization_stage_check",
                    source_table_name=source_table_name,
                    connector_type=connector.__class__.__name__,
            ):
                stage = data_backend_cache().materialization_stage.get_stage(
                    fingerprint, data_version, check_local=True)
            if stage:
                data_backend_cache().add_table_stats(
                    fingerprint, {
                        TableStatType.SOURCE_SIZE:
                        Stats(
                            value=_stored_materialization_size(
                                fingerprint, str(data_version), stage),
                            approx=False, data_version=str(data_version)),
                    })
                return get_data_operation_description(), validation_response

        start_time = datetime.now(timezone.utc)

        with kumo_timing(
                "table_ingest_activity.connector_prepare_table",
                source_table_name=source_table_name,
                connector_type=connector.__class__.__name__,
        ):
            if table.is_sql:
                # Custom SQL Table: push the read-only SELECT down to the
                # warehouse and stage only its result, keyed on the same
                # per-table ingest layout export_table uses (its fingerprint is
                # a SqlTableFingerprint). The persisted SQL is the raw user text
                # (pass-through); strip any trailing terminator/comment so it
                # embeds cleanly in the connector's subquery position
                # (`FROM (<sql>)`) — mirroring the authoring boundary
                # (rest/utils/connector.py), which sanitizes before every query
                # embedding, so a trailing ';' that passes validation/preview
                # cannot break materialization. download() is unchanged.
                connector_type = _connector_type_from_spec(connector_spec)
                advisory_read_only_check(table.source_table.sql, connector_type)
                validate_connector_sql_scope(table.source_table.sql,
                                             connector_type, connector)
                sql = subquery_safe_sql(table.source_table.sql, connector_type)
                metrics = connector.export_query(
                    sql,
                    fingerprint,
                    str(data_version),
                )
            elif direct_diskgraph_source:
                assert direct_file_source is not None
                metrics = ExportMetrics(
                    num_rows=None,
                    source_size=sum(
                        int(item['size'])
                        for item in direct_file_source['objects']),
                )
            else:
                metrics = connector.export_table(
                    source_table_name,
                    fingerprint,
                    str(data_version),
                    graph_table_count=graph_table_count,
                )

        # ``num_rows`` is ``None`` for the file/parquet connector (no footer
        # scan, kumo#30173); ``None == 0`` is False, so this fast 0-row guard
        # only fires for connectors that count for free (SQL/warehouse). The
        # file/parquet path's emptiness is caught downstream via ``is_empty``
        # on the local mirror in ``IngestionValidator.post_ingestion``.
        if metrics.num_rows == 0:
            validation_response.errors.append(
                ValidationError(
                    'Table ingestion check failed',
                    f'Table {source_table_name} has 0 rows. '
                    'This can happen when the source data was removed/changed. '
                    'Please re-ingest the table after confirming the source '
                    'data is available.'))
            return (get_data_operation_description(), validation_response)

        # Persist SOURCE_SIZE (always) and NUM_ROWS (only when the connector
        # produced it for free) so downstream consumers can read them off the
        # data backend cache. The file/parquet connector persists no NUM_ROWS;
        # its consumers tolerate the absence (row-based scale limits removed;
        # sizing/UI default or recompute lazily).
        stats_to_save = {
            TableStatType.SOURCE_SIZE:
            Stats(value=metrics.source_size, approx=metrics.approx,
                  data_version=str(data_version)),
        }
        if metrics.num_rows is not None:
            stats_to_save[TableStatType.NUM_ROWS] = Stats(
                value=metrics.num_rows, approx=metrics.approx,
                data_version=str(data_version))
        with kumo_timing(
                "table_ingest_activity.save_table_stats",
                source_table_name=source_table_name,
                num_rows=metrics.num_rows,
                source_size=metrics.source_size,
        ):
            # A reused source version must not replace exact DiskGraph stats
            # with connector estimates.
            data_backend_cache().add_table_stats(fingerprint, stats_to_save)

        info = TableIngestionInfo(
            ingest_start_time=start_time,
            ingest_end_time=datetime.now(timezone.utc),
            temporal_workflow_id=workflow_info.workflow_id,
            code_version=get_kumo_version(),
        )
        with kumo_timing(
                "table_ingest_activity.save_ingestion_info",
                source_table_name=source_table_name,
        ):
            data_backend_cache().ingestion_info.save(
                fingerprint.connector_name,
                source_table_name,
                data_version,
                info,
            )
        if not direct_diskgraph_source:
            with kumo_timing(
                    "table_ingest_activity.save_materialization_stage",
                    source_table_name=source_table_name,
            ):
                data_backend_cache().materialization_stage.put(
                    fingerprint, data_version, MaterializationStage.INGEST,
                    save=True)
        return get_data_operation_description(), validation_response
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            connector.close()
        except BaseException as cleanup_error:
            if primary_error is None:
                raise
            log.warning(
                'Connector cleanup failed while preserving primary failure '
                '(primary_error_type=%s cleanup_error_type=%s)',
                type(primary_error).__name__,
                type(cleanup_error).__name__,
            )


def _stored_materialization_size(table_fingerprint, data_version: str,
                                 stage: MaterializationStage) -> int:
    """Return the bytes in one Kumo-owned staged materialization."""
    path = get_materialize_path(table_fingerprint, data_version, stage)
    if not fsutil.isdir(path):
        raise FileNotFoundError(
            f'Staged materialization directory does not exist: {path}')
    return fsutil.dirsize(path)
