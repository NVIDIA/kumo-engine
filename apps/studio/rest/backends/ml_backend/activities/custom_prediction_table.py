# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from kumoapi.jobs import PredictionTableSource

from io_utils import fsutil, logging, parquet_util
from kumo.connector.file_connector import FileConnector
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend.activities.custom_train_table import (
    read_only_custom_sql,
    require_custom_sql_support,
)
from rest.config.connector import object_store_source_for_runtime_connector
from rest.config.data_backend import ConnectorExecutionRef
from rest.utils.driver_common import get_connector_from_spec
from statedb.utils.errors import (
    EmptyPredictionTableError,
    InvalidPredictionTableError,
)
from temporalio_commons.heartbeat_utils import auto_heartbeater

log = logging.getLogger(__name__)


def _source_label(source: PredictionTableSource) -> str:
    provider = source.storage_type.name
    return f'the configured {provider} source'


@connector_activity_registry.connector_authorized_activity
@auto_heartbeater
def stage_custom_prediction_table(
    reference: ConnectorExecutionRef,
    source: PredictionTableSource,
    destination: str,
) -> str:
    """Stage a connector-backed object-store prediction table into Kumo.

    External reads occur only under the activity-local connector credential
    scope. The durable workflow continues from the Kumo-owned staged path.
    """
    if str(reference.connector_id) != source.connector_id:
        raise ValueError(
            'Prediction source connector does not match execution reference')
    connector = get_connector_from_spec(reference)
    read_only_export = getattr(connector, 'export_table_read_only', None)
    if not isinstance(connector, FileConnector) and read_only_export is None:
        connector.close()
        raise ValueError(
            'Custom prediction source requires an object-store or warehouse '
            'connector')
    # Outside the try: `except Exception` below reports everything as an
    # invalid prediction table, which cannot say that this connector has no
    # way to read a query. Today only `DatabricksPredictionTableSource`
    # carries `sql` and only Databricks' export accepts it, so this is
    # defence -- it holds if a warehouse prediction source is ever added for
    # a connector whose exporter was never given the parameter.
    if not isinstance(connector, FileConnector):
        try:
            require_custom_sql_support(source, connector)
        except Exception:
            # Mirrors the pre-try raise above: this is outside the
            # `finally` that closes the connector, so it closes its own.
            connector.close()
            raise
    try:
        if not isinstance(connector, FileConnector):
            # Passed only when the source is a query and the exporter
            # accepts one; the combination that cannot be served is already
            # refused above.
            optional = {}
            if source.sql:
                optional['sql'] = read_only_custom_sql(source, connector)
            connector.open()
            # Pre-clean: not every warehouse export overwrites, so a stale
            # part-*.parquet could silently predict on stale rows.
            fsutil.rm(destination, ignore_errors=True)
            # Exactly one of table/sql is set; for a query the name is only
            # what staging prefixes and errors key on.
            metrics = read_only_export(
                source.table or 'custom_prediction_table', destination,
                **optional)
            if (getattr(metrics, 'num_rows', None) == 0
                    or parquet_util.is_empty(destination)):
                raise EmptyPredictionTableError(path=_source_label(source))
            return destination

        coordinate = object_store_source_for_runtime_connector(
            connector, source.source)

        with connector.filesystem_credential_scope(coordinate):
            _, file_type = connector.get_path_and_type(coordinate)
            if file_type.casefold() != 'parquet':
                raise ValueError('Custom prediction source must be Parquet')
        with connector._local_source(coordinate,
                                     transform=False) as (staged, _):
            if parquet_util.is_empty(staged.directory):
                raise EmptyPredictionTableError(path=_source_label(source))
            from kumo.connector.source_bridge import publish_source
            publish_source(staged, destination)
        return destination
    except EmptyPredictionTableError:
        raise
    except Exception as error:
        log.warning(
            'Custom prediction source staging failed (error_type=%s)',
            type(error).__name__,
        )
        raise InvalidPredictionTableError(
            path=_source_label(source),
            non_retryable=True,
        ) from None
    finally:
        connector.close()
