# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import traceback
import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest
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
from kumoapi.source_table import FileType, S3SourceTable
from kumoapi.typing import Dtype, Stype

from kumo.connector import DatabricksConnector, SnowflakeConnector
from kumo.connector.composite_file_connector import CompositeFileConnector
from kumo.connector.connector import ExportMetrics
from kumo.connector.sql_connector import SQLConnector
from kumo.data import MaterializationStage
from kumo.data.backend_cache import DirectDiskgraphSourceDecision
from kumo.validation import ValidationResponse
from rest.backends.data_backend.activities import data_backend_activities
from rest.config.data_backend import ConnectorSpec
from rest.config.table_v2 import (
    Column,
    CustomSQLSourceTable,
    DatabricksSourceTable,
    TableDefinition,
)
from service_framework.deploy_env_config import DeployEnvConfig
from statedb.config.common import ConnectorID
from statedb.utils.errors import (
    IdentityResolutionUnavailableError,
    TableIngestionError,
    TableIngestionOutOfMemoryError,
)

# Reach into sys.modules because the package re-exports the function
# under the same name as the module.
activity_module = sys.modules[
    'rest.backends.data_backend.activities.ingest_table_data_activity_v2']
ingest_table_data_activity_v2 = (activity_module.ingest_table_data_activity_v2)


def _table_and_connector_spec() -> tuple[TableDefinition, ConnectorSpec]:
    connector_id = ConnectorID(f'connector_{"0" * 32}')
    table = TableDefinition(
        cols=[],
        source_table=S3SourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': 'test_table'
            },
            file_type=FileType.PARQUET,
        ),
        pkey=None,
        time_col=None,
    )
    return table, ConnectorSpec(
        connector_id=connector_id,
        connector_config=mock.Mock(),
    )


@contextmanager
def _activity_patches():
    with mock.patch.multiple(
            activity_module,
            get_connector_from_spec=mock.DEFAULT,
            data_backend_cache=mock.DEFAULT,
            activity=mock.DEFAULT,
            IngestionValidator=mock.DEFAULT,
            get_kumo_version=mock.DEFAULT,
            _stored_materialization_size=mock.DEFAULT,
    ) as patches:
        patches['_stored_materialization_size'].return_value = 123
        with mock.patch.object(activity_module, 'setup_backend_cache',
                               create=True) as setup_backend_cache:
            patches['setup_backend_cache'] = setup_backend_cache
            yield patches


def _object_source() -> dict:
    return {
        'type':
        'object_store_parquet',
        'objects': [{
            'path': 's3://customer-bucket/root/test_table/part.parquet',
            'size': 345,
            'e_tag': 'etag',
            'version': None,
        }],
        'auth_profile':
        'ambient',
    }


def _run_file_ingest(
    *,
    decision=None,
    live_direct=None,
    cached_stage=None,
    legacy_cached_stage=None,
):
    table, connector_spec = _table_and_connector_spec()
    source = _object_source()
    connector = mock.Mock(spec=CompositeFileConnector)
    if live_direct is None:
        connector.uses_direct_diskgraph_source_for.side_effect = (
            AssertionError('frozen decision must not be reprobed'))
        connector.diskgraph_source.side_effect = AssertionError(
            'frozen manifest must not be relisted')
    else:
        connector.uses_direct_diskgraph_source_for.return_value = live_direct
        connector.diskgraph_source.return_value = source
    connector.export_table.return_value = ExportMetrics(
        num_rows=None,
        source_size=345,
    )

    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        cache = patches['data_backend_cache'].return_value
        cache.direct_diskgraph_source_decision.get.return_value = decision
        cache.materialization_stage.get_stage.return_value = cached_stage
        cache.materialization_stage.get.return_value = legacy_cached_stage
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_kumo_version'].return_value = 'test_version'
        patches['get_connector_from_spec'].return_value = connector
        _, response = ingest_table_data_activity_v2(
            connector_spec,
            table,
            'v1',
            'job_id',
            mock.Mock(spec=DeployEnvConfig),
        )

    return table, source, connector, cache, response


def test_v2_zero_rows_raises_validation_error():
    """A connector reporting num_rows=0 from its unload must surface a
    'Table ingestion check failed' ValidationError."""
    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        patches['data_backend_cache'].return_value \
            .materialization_stage.get_stage.return_value = None
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_kumo_version'].return_value = 'test_version'

        connector = mock.Mock(spec=SQLConnector)
        connector.name = 'fake'
        connector.export_table.return_value = ExportMetrics(
            num_rows=0, source_size=0)
        patches['get_connector_from_spec'].return_value = connector

        table = TableDefinition(
            cols=[],
            source_table=S3SourceTable(
                connector_id=f'connector_{"0" * 32}',
                source={
                    'mode': 'relative',
                    'path': 'test_table'
                },
                file_type=FileType.PARQUET,
            ),
            pkey=None,
            time_col=None,
        )
        connector_spec = ConnectorSpec(
            connector_id=f'connector_{"0" * 32}',
            connector_config=mock.Mock(),
        )

        _, response = ingest_table_data_activity_v2(
            connector_spec, table, 'v1', 'job_id',
            mock.Mock(spec=DeployEnvConfig))

        assert not response.ok
        assert any('0 rows' in e.message for e in response.errors)
        connector.close.assert_called_once_with()


def test_v2_none_num_rows_persists_source_size_only_no_empty_error():
    """The file/parquet connector reports ``num_rows=None`` (no O(part-count)
    footer scan, kumo#30173). The fast 0-row guard must NOT fire on ``None``,
    and only ``SOURCE_SIZE`` is persisted -- ``NUM_ROWS`` is absent, and its
    consumers tolerate that. Emptiness is caught downstream via ``is_empty``.

    ``SOURCE_SIZE`` is the size the export itself measured: the materialize
    path is remote in cloud, so walking it again is a round trip per table
    for a number already in hand.
    """
    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        cache = patches['data_backend_cache'].return_value
        cache.materialization_stage.get_stage.return_value = None
        cache.materialization_stage.get.return_value = None
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_kumo_version'].return_value = 'test_version'
        patches['_stored_materialization_size'].return_value = 456

        connector = mock.Mock(spec=SQLConnector)
        connector.name = 'fake'
        connector.export_table.return_value = ExportMetrics(
            num_rows=None, source_size=123)
        patches['get_connector_from_spec'].return_value = connector

        table = TableDefinition(
            cols=[],
            source_table=S3SourceTable(
                connector_id=f'connector_{"0" * 32}',
                source={
                    'mode': 'relative',
                    'path': 'test_table'
                },
                file_type=FileType.PARQUET,
            ),
            pkey=None,
            time_col=None,
        )
        connector_spec = ConnectorSpec(
            connector_id=f'connector_{"0" * 32}',
            connector_config=mock.Mock(),
        )

        _, response = ingest_table_data_activity_v2(
            connector_spec, table, 'v1', 'job_id',
            mock.Mock(spec=DeployEnvConfig))

        # ``None == 0`` is False: no spurious "0 rows" ingestion failure.
        assert response.ok
        assert not any('0 rows' in e.message for e in response.errors)
        # Exactly SOURCE_SIZE was persisted; NUM_ROWS was never written.
        saved_stats = cache.add_table_stats.call_args.args[1]
        assert TableStatType.SOURCE_SIZE in saved_stats
        assert saved_stats[TableStatType.SOURCE_SIZE].value == 123
        assert TableStatType.NUM_ROWS not in saved_stats
        patches['_stored_materialization_size'].assert_not_called()


def test_v2_deferred_none_num_rows_persists_export_size_only():
    """A deferred warehouse stage may omit a metadata row count.

    It must retain the completed export size without checking a nonexistent
    local mirror or treating the unknown count as an empty source.
    """
    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        cache = patches['data_backend_cache'].return_value
        cache.materialization_stage.get_stage.return_value = None
        cache.materialization_stage.get.return_value = None
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_kumo_version'].return_value = 'test_version'

        connector = mock.Mock(spec=SQLConnector)
        connector.name = 'fake'
        connector.export_table.return_value = ExportMetrics(
            num_rows=None, source_size=123)
        patches['get_connector_from_spec'].return_value = connector
        table, connector_spec = _table_and_connector_spec()

        _, response = ingest_table_data_activity_v2(
            connector_spec, table, 'v1', 'job_id',
            mock.Mock(spec=DeployEnvConfig))

        assert response.ok
        assert not any('0 rows' in error.message for error in response.errors)
        saved_stats = cache.add_table_stats.call_args.args[1]
        assert saved_stats[TableStatType.SOURCE_SIZE].value == 123
        assert TableStatType.NUM_ROWS not in saved_stats
        patches['_stored_materialization_size'].assert_not_called()


def test_provider_failure_is_absent_from_error_trace_and_logs(caplog):
    provider_secret = 'provider-token-and-stack-must-never-escape'

    class ProviderFailure(Exception):
        pass

    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_connector_from_spec'].side_effect = ProviderFailure(
            provider_secret)
        table, connector_spec = _table_and_connector_spec()

        with pytest.raises(TableIngestionError) as caught:
            ingest_table_data_activity_v2(connector_spec, table, 'v1', 'job_id',
                                          mock.Mock(spec=DeployEnvConfig))

    rendered_trace = ''.join(traceback.format_exception(caught.value))
    assert caught.value.http_code == 422
    assert caught.value.message == (
        activity_module.SAFE_TABLE_INGESTION_ERROR_MESSAGE)
    assert caught.value.__cause__ is None
    assert provider_secret not in str(caught.value)
    assert provider_secret not in rendered_trace
    assert provider_secret not in caplog.text


def test_allowlisted_identity_outage_preserves_safe_public_error(caplog):
    safe_message = 'Identity resolution is temporarily unavailable.'
    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_connector_from_spec'].side_effect = (
            IdentityResolutionUnavailableError(message=safe_message))
        table, connector_spec = _table_and_connector_spec()

        with pytest.raises(IdentityResolutionUnavailableError) as caught:
            ingest_table_data_activity_v2(connector_spec, table, 'v1', 'job_id',
                                          mock.Mock(spec=DeployEnvConfig))

    assert caught.value.http_code == 503
    assert caught.value.message == safe_message
    assert safe_message not in caplog.text


def test_oom_export_failure_preserves_specific_nonretryable_error():
    """kumo#30084: the connector's out-of-memory export error is allowlisted
    through the ingestion sanitizer, keeping its actionable message and
    non-retryable marking instead of flattening to the generic ingestion
    error (which suggests a retry that would deterministically re-OOM)."""
    oom_message = ('The worker ran out of memory while ingesting the '
                   "Databricks table 'large_embs_articles'.")
    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_connector_from_spec'].side_effect = (
            TableIngestionOutOfMemoryError(oom_message))
        table, connector_spec = _table_and_connector_spec()

        with pytest.raises(TableIngestionOutOfMemoryError) as caught:
            ingest_table_data_activity_v2(connector_spec, table, 'v1', 'job_id',
                                          mock.Mock(spec=DeployEnvConfig))

    assert caught.value.http_code == 422
    assert caught.value.non_retryable is True
    assert caught.value.message == oom_message
    assert caught.value.__cause__ is None


def test_timestamp_inference_provider_failure_is_sanitized(caplog):
    provider_secret = 'timestamp-provider-token-must-never-escape'
    table, connector_spec = _table_and_connector_spec()

    with mock.patch(
            'rest.backends.data_backend.activities.data_backend_activities.'
            'get_connector_from_spec',
            side_effect=RuntimeError(provider_secret),
    ):
        with pytest.raises(TableIngestionError) as caught:
            data_backend_activities.DataBackendActivities.update_table_timestamp_units(
                table, connector_spec, None, None)

    assert caught.value.message == (
        activity_module.SAFE_TABLE_INGESTION_ERROR_MESSAGE)
    assert provider_secret not in str(caught.value)
    assert provider_secret not in caplog.text


def test_timestamp_reconstruction_preserves_multicategorical_delimiter():
    table, connector_spec = _table_and_connector_spec()
    table = TableDefinition(
        cols=[
            Column(
                name='tags',
                stype=Stype.multicategorical,
                dtype=Dtype.string,
                multicategorical_delimiter='|',
            ),
            Column(name='event_time', stype=Stype.timestamp, dtype=Dtype.date),
        ],
        source_table=table.source_table,
        time_col='event_time',
    )
    inferred_cols = [
        SimpleNamespace(name='tags', dtype=Dtype.string, timestamp_unit=None,
                        metadata={}),
        SimpleNamespace(name='event_time', dtype=Dtype.time, timestamp_unit='s',
                        metadata={}),
    ]
    kumo_table = SimpleNamespace(
        cols=inferred_cols,
        data_cols=inferred_cols,
        _infer_dtypes_from_stypes_blocking=lambda: {'event_time': Dtype.time},
    )
    connector = mock.MagicMock()

    with mock.patch.object(data_backend_activities, 'get_connector_from_spec',
                           return_value=connector), mock.patch.object(
                               data_backend_activities,
                               'table_args_to_kumo_table',
                               return_value=kumo_table):
        updated = data_backend_activities.DataBackendActivities \
            .update_table_timestamp_units(table, connector_spec, None, None)

    updated_cols = {col.name: col for col in updated.cols}
    assert updated_cols['tags'].multicategorical_delimiter == '|'
    assert updated_cols['event_time'].timestamp_format.value == 's'
    connector.close.assert_called_once_with()


def _run_successful_v2_ingest(connector_name: str, table_connector_id: str):
    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        cache = patches['data_backend_cache'].return_value
        cache.materialization_stage.get_stage.return_value = None
        cache.materialization_stage.get.return_value = None
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_kumo_version'].return_value = 'test_version'

        connector = mock.Mock(spec=SQLConnector)
        connector.name = connector_name
        connector.export_table.return_value = ExportMetrics(
            num_rows=2, source_size=10)
        patches['get_connector_from_spec'].return_value = connector

        table = TableDefinition(
            cols=[],
            source_table=S3SourceTable(
                connector_id=table_connector_id,
                source={
                    'mode': 'relative',
                    'path': 'test_table'
                },
                file_type=FileType.PARQUET,
            ),
            pkey=None,
            time_col=None,
        )
        connector_spec = ConnectorSpec(
            connector_id=ConnectorID(table_connector_id),
            connector_config=mock.Mock())

        _, response = ingest_table_data_activity_v2(
            connector_spec, table, 'v1', 'job_id',
            mock.Mock(spec=DeployEnvConfig))
        connector.close.assert_called_once_with()

    assert response.ok
    return cache, table


def test_v2_ingest_metadata_uses_persisted_table_connector_identity():
    connector_id = f'connector_{uuid.uuid4().hex}'

    cache, table = _run_successful_v2_ingest(
        connector_name='execution_connector',
        table_connector_id=connector_id,
    )

    source_table_name = table.get_source_table_name()
    cache.ingestion_info.save.assert_called_once()
    assert cache.ingestion_info.save.call_args.args[:3] == (
        connector_id,
        source_table_name,
        'v1',
    )
    cache.materialization_stage.put.assert_called_once_with(
        table.fingerprint(), 'v1', MaterializationStage.INGEST, save=True)


def test_v2_ingest_custom_sql_dispatches_export_query():
    """A Custom SQL table pushes its SELECT to export_query (staged like a
    physical table), not the physical export_table path. The persisted raw SQL
    is made subquery-safe (trailing ';'/comment stripped) before the connector
    embeds it in `FROM (<sql>)`, so a query that passes validation cannot break
    materialization."""
    connector_id = f'connector_{"a" * 32}'
    with (
            _activity_patches() as patches,
            mock.patch.object(
                activity_module,
                'advisory_read_only_check',
                wraps=activity_module.advisory_read_only_check,
            ) as read_only_check,
            mock.patch.object(
                activity_module,
                'validate_connector_sql_scope',
                wraps=activity_module.validate_connector_sql_scope,
            ) as scope_check,
    ):
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        cache = patches['data_backend_cache'].return_value
        cache.materialization_stage.get_stage.return_value = None
        cache.materialization_stage.get.return_value = None
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_kumo_version'].return_value = 'test_version'

        connector = mock.Mock(spec=SQLConnector)
        connector.name = connector_id
        connector.database = 'DB'
        connector.schema_name = 'S'
        connector.export_query.return_value = ExportMetrics(
            num_rows=4, source_size=40)
        patches['get_connector_from_spec'].return_value = connector

        sql = 'SELECT item_key, total FROM sales WHERE total > 0'
        # Persisted SQL carries a trailing terminator + comment (valid, but
        # would break `FROM (<sql>)`); the activity must strip it to `sql`.
        raw_sql = sql + ';  -- nightly refresh'
        table = TableDefinition(
            cols=[
                Column(name='item_key', stype=Stype.ID, dtype=Dtype.string),
                Column(name='total', stype=Stype.numerical, dtype=Dtype.float),
            ],
            source_table=CustomSQLSourceTable(connector_id=connector_id,
                                              connector_type='snowflake',
                                              sql=raw_sql,
                                              table='custom_sql_abc',
                                              referenced_tables=['sales']),
            pkey='item_key',
            time_col=None,
        )
        # Production shape: connector_config is the connector's NATIVE config
        # (``.config()``), which has no ``type`` field -- the warehouse type is
        # derived from the connector class. Building it here opens no
        # connection; the activity's own resolved ``connector`` (patched above)
        # is what runs ``export_query``.
        connector_spec = ConnectorSpec(
            connector_id=ConnectorID(connector_id),
            connector_config=SnowflakeConnector(name=connector_id,
                                                account='acc',
                                                warehouse='COMPUTE',
                                                database='DB',
                                                schema_name='S').config())

        _, response = ingest_table_data_activity_v2(
            connector_spec, table, 'v1', 'job_id',
            mock.Mock(spec=DeployEnvConfig))

    assert response.ok
    read_only_check.assert_called_once_with(raw_sql, mock.ANY)
    scope_check.assert_called_once_with(raw_sql, mock.ANY, connector)
    connector.export_query.assert_called_once_with(sql, table.fingerprint(),
                                                   'v1')
    connector.export_table.assert_not_called()


@pytest.mark.parametrize('is_direct', [True, False], ids=['direct', 'staged'])
def test_v2_file_ingest_persists_decision_before_materialization(is_direct):
    cached_stage = None if is_direct else MaterializationStage.INGEST
    table, listed_source, connector, cache, response = _run_file_ingest(
        live_direct=is_direct,
        cached_stage=cached_stage,
    )
    source = listed_source if is_direct else None

    assert response.ok
    connector.uses_direct_diskgraph_source_for.assert_called_once_with(
        'test_table')
    if is_direct:
        connector.diskgraph_source.assert_called_once_with(
            source_name='test_table')
        cache.materialization_stage.get_stage.assert_not_called()
    else:
        connector.diskgraph_source.assert_not_called()
    connector.export_table.assert_not_called()
    cache.direct_diskgraph_source_decision.get.assert_called_once_with(
        table.fingerprint(),
        'v1',
        cls=DirectDiskgraphSourceDecision,
    )
    cache.direct_diskgraph_source_decision.put.assert_called_once()
    decision = cache.direct_diskgraph_source_decision.put.call_args.args[3]
    assert decision == DirectDiskgraphSourceDecision(source=source)
    assert cache.direct_diskgraph_source_decision.put.call_args.kwargs == {
        'save': True
    }
    cache.materialization_stage.put.assert_not_called()
    connector.close.assert_called_once_with()


@pytest.mark.parametrize('is_direct', [True, False], ids=['direct', 'staged'])
def test_v2_file_ingest_reuses_frozen_decision_without_reprobe(is_direct):
    source = _object_source() if is_direct else None
    decision = DirectDiskgraphSourceDecision(source=source)
    table, _, connector, cache, response = _run_file_ingest(
        decision=decision,
        cached_stage=MaterializationStage.INGEST,
    )

    assert response.ok
    cache.direct_diskgraph_source_decision.get.assert_called_once_with(
        table.fingerprint(),
        'v1',
        cls=DirectDiskgraphSourceDecision,
    )
    cache.direct_diskgraph_source_decision.put.assert_not_called()
    connector.uses_direct_diskgraph_source_for.assert_not_called()
    connector.diskgraph_source.assert_not_called()
    connector.export_table.assert_not_called()
    if source is None:
        cache.materialization_stage.get_stage.assert_called_once()
        saved_stats = cache.add_table_stats.call_args.args[1]
        assert saved_stats[TableStatType.SOURCE_SIZE].value == 123
        assert saved_stats[TableStatType.SOURCE_SIZE].approx is False
    else:
        cache.materialization_stage.get_stage.assert_not_called()
        saved_stats = cache.add_table_stats.call_args.args[1]
        assert saved_stats[TableStatType.SOURCE_SIZE].value == 345
    connector.close.assert_called_once_with()


def test_v2_staged_file_ingest_ignores_legacy_diskgraph_cache():
    table, _, connector, cache, response = _run_file_ingest(
        live_direct=False,
        cached_stage=None,
        legacy_cached_stage=MaterializationStage.INGEST,
    )

    assert response.ok
    cache.materialization_stage.get.assert_not_called()
    connector.export_table.assert_called_once_with(
        'test_table',
        table.fingerprint(),
        'v1',
        graph_table_count=None,
    )
    cache.materialization_stage.put.assert_called_once_with(
        table.fingerprint(),
        'v1',
        MaterializationStage.INGEST,
        save=True,
    )


@pytest.mark.parametrize('is_direct', [True, False], ids=['direct', 'staged'])
def test_validate_file_ingest_uses_frozen_decision_without_connector(is_direct):
    table, connector_spec = _table_and_connector_spec()
    decision = DirectDiskgraphSourceDecision(
        source=_object_source() if is_direct else None, )
    cache = mock.Mock()
    cache.direct_diskgraph_source_decision.get.return_value = decision
    validator = mock.Mock()
    validator.post_ingestion.return_value = ValidationResponse()

    with mock.patch.object(
            data_backend_activities,
            'data_backend_cache', return_value=cache), mock.patch.object(
                data_backend_activities, 'IngestionValidator',
                return_value=validator), mock.patch.object(
                    data_backend_activities, 'get_connector_from_spec',
                    side_effect=AssertionError(
                        'frozen decision must not resolve the connector')):
        response = (
            data_backend_activities.DataBackendActivities.validate_table_ingest(
                table,
                'v1',
                mock.Mock(spec=DeployEnvConfig),
                connector_spec=connector_spec,
            ))

    assert response.ok
    cache.direct_diskgraph_source_decision.get.assert_called_once_with(
        table.fingerprint(),
        'v1',
        cls=DirectDiskgraphSourceDecision,
    )
    validator.post_ingestion.assert_called_once_with(
        table,
        'v1',
        existing_errors=None,
        check_local=not is_direct,
        require_materialization_stage=not is_direct,
    )


def _databricks_connector_spec(connector_id: str) -> ConnectorSpec:
    """Production shape: the native Databricks config, opening no connection."""
    return ConnectorSpec(
        connector_id=ConnectorID(connector_id),
        connector_config=DatabricksConnector(
            name=connector_id,
            host='https://workspace.cloud.databricks.com',
            cluster_id='cluster',
            warehouse_id='warehouse',
            catalog='main',
            schema_name='kumo',
        ).config(),
    )


def _databricks_connector_mock(connector_id: str) -> mock.Mock:
    """A Databricks mock; every table of it stages through the reader."""
    connector = mock.Mock(spec=DatabricksConnector)
    connector.name = connector_id
    connector.catalog = 'main'
    connector.schema_name = 'kumo'
    return connector


def test_v2_databricks_physical_table_materializes_through_reader():
    """The real per-table rule decides here; stubbing it was what let this case
    claim the query path and skip the materialization stage it now writes.
    """
    connector_id = f'connector_{"d" * 32}'
    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        cache = patches['data_backend_cache'].return_value
        cache.materialization_stage.get_stage.return_value = None
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_kumo_version'].return_value = 'test_version'

        connector = _databricks_connector_mock(connector_id)
        connector.export_table.return_value = ExportMetrics(num_rows=12,
                                                            source_size=345)
        patches['get_connector_from_spec'].return_value = connector
        table = TableDefinition(
            cols=[
                Column(name='order_id', stype=Stype.ID, dtype=Dtype.int),
                Column(name='amount', stype=Stype.numerical, dtype=Dtype.float),
            ],
            source_table=DatabricksSourceTable(
                databricks_connector_id=connector_id,
                table='orders',
            ),
            pkey='order_id',
            time_col=None,
        )

        _, response = ingest_table_data_activity_v2(
            _databricks_connector_spec(connector_id),
            table,
            'v1',
            'job_id',
            mock.Mock(spec=DeployEnvConfig),
        )

    assert response.ok
    assert not table.is_sql
    connector.export_table.assert_called_once_with(
        'orders',
        table.fingerprint(),
        'v1',
        graph_table_count=None,
    )
    cache.materialization_stage.get_stage.assert_called_once_with(
        table.fingerprint(), 'v1', check_local=True)
    cache.materialization_stage.put.assert_called_once_with(
        table.fingerprint(), 'v1', MaterializationStage.INGEST, save=True)
    saved_stats = cache.add_table_stats.call_args.args[1]
    assert not any(stat.approx for stat in saved_stats.values())
    connector.close.assert_called_once_with()


def test_v2_databricks_custom_sql_stages_through_the_reader():
    """A Databricks query is exported like every other warehouse's: one
    warehouse execution that lands in Kumo parquet, counted off what landed,
    so its NUM_ROWS is exact rather than an optimizer estimate."""
    connector_id = f'connector_{"d" * 32}'
    with (
            _activity_patches() as patches,
            mock.patch.object(activity_module, 'advisory_read_only_check'),
            mock.patch.object(activity_module, 'validate_connector_sql_scope'),
    ):
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        cache = patches['data_backend_cache'].return_value
        cache.materialization_stage.get_stage.return_value = None
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_kumo_version'].return_value = 'test_version'

        connector = _databricks_connector_mock(connector_id)
        connector.export_query.return_value = ExportMetrics(num_rows=12,
                                                            source_size=345)
        patches['get_connector_from_spec'].return_value = connector
        sql = 'SELECT item_key, total FROM sales WHERE total > 0'
        table = TableDefinition(
            cols=[
                Column(name='item_key', stype=Stype.ID, dtype=Dtype.string),
                Column(name='total', stype=Stype.numerical, dtype=Dtype.float),
            ],
            source_table=CustomSQLSourceTable(connector_id=connector_id,
                                              connector_type='databricks',
                                              sql=sql,
                                              table='custom_sql_abc',
                                              referenced_tables=['sales']),
            pkey='item_key',
            time_col=None,
        )

        _, response = ingest_table_data_activity_v2(
            _databricks_connector_spec(connector_id),
            table,
            'v1',
            'job_id',
            mock.Mock(spec=DeployEnvConfig),
        )

    assert response.ok
    assert table.is_sql
    connector.export_query.assert_called_once_with(sql, table.fingerprint(),
                                                   'v1')
    connector.export_table.assert_not_called()
    cache.materialization_stage.put.assert_called_once_with(
        table.fingerprint(), 'v1', MaterializationStage.INGEST, save=True)
    saved_stats = cache.add_table_stats.call_args.args[1]
    assert saved_stats[TableStatType.NUM_ROWS].value == 12
    assert not any(stat.approx for stat in saved_stats.values())
    connector.close.assert_called_once_with()


@pytest.mark.parametrize('is_sql', [True, False], ids=['custom_sql', 'table'])
def test_validate_databricks_ingest_checks_staged_parquet_either_way(is_sql):
    connector_id = f'connector_{"d" * 32}'
    source_table = (CustomSQLSourceTable(
        connector_id=connector_id, connector_type='databricks',
        sql='SELECT item_key FROM sales', table='custom_sql_abc',
        referenced_tables=['sales']) if is_sql else DatabricksSourceTable(
            databricks_connector_id=connector_id, table='orders'))
    table = TableDefinition(
        cols=[Column(name='item_key', stype=Stype.ID, dtype=Dtype.string)],
        source_table=source_table,
        pkey='item_key',
        time_col=None,
    )
    connector = _databricks_connector_mock(connector_id)
    validator = mock.Mock()
    validator.post_ingestion.return_value = ValidationResponse()

    with mock.patch.object(
            data_backend_activities, 'data_backend_cache',
            return_value=mock.Mock()), mock.patch.object(
                data_backend_activities, 'IngestionValidator',
                return_value=validator), mock.patch.object(
                    data_backend_activities, 'get_connector_from_spec',
                    return_value=connector):
        response = (
            data_backend_activities.DataBackendActivities.validate_table_ingest(
                table,
                'v1',
                mock.Mock(spec=DeployEnvConfig),
                connector_spec=ConnectorSpec(
                    connector_id=ConnectorID(connector_id),
                    connector_config=mock.Mock(),
                ),
            ))

    assert response.ok
    validator.post_ingestion.assert_called_once_with(
        table,
        'v1',
        existing_errors=None,
        check_local=True,
        require_materialization_stage=True,
    )
    connector.close.assert_called_once_with()


def test_v2_ingest_metadata_keeps_connector_reference():
    connector_id = f'connector_{"f" * 32}'

    cache, table = _run_successful_v2_ingest(
        connector_name=connector_id,
        table_connector_id=connector_id,
    )

    source_table_name = table.get_source_table_name()
    assert cache.ingestion_info.save.call_args.args[:3] == (
        connector_id,
        source_table_name,
        'v1',
    )
    cache.materialization_stage.put.assert_called_once_with(
        table.fingerprint(), 'v1', MaterializationStage.INGEST, save=True)


@pytest.mark.parametrize(('parts', 'expected_size'), [
    ([], 0),
    ([b'123', b'4567'], 7),
])
def test_stored_materialization_size_uses_existing_filesystem_size(
        parts, expected_size, tmp_path, monkeypatch):
    path = tmp_path / 'materialized'
    path.mkdir()
    for index, contents in enumerate(parts):
        (path / f'part-{index}.parquet').write_bytes(contents)
    monkeypatch.setattr(activity_module, 'get_materialize_path',
                        lambda *_args: str(path))

    assert activity_module._stored_materialization_size(
        mock.sentinel.fingerprint, 'v1',
        MaterializationStage.INGEST) == expected_size


def test_stored_materialization_size_rejects_missing_directory(
        tmp_path, monkeypatch):
    path = tmp_path / 'missing-materialization'
    monkeypatch.setattr(activity_module, 'get_materialize_path',
                        lambda *_args: str(path))

    with pytest.raises(FileNotFoundError) as excinfo:
        activity_module._stored_materialization_size(
            mock.sentinel.fingerprint, 'v1', MaterializationStage.INGEST)

    assert str(path) in str(excinfo.value)


# The vendor text a translated source error carries; none of it may reach the
# user, whichever branch of the sanitizer the error takes.
VENDOR_TEXT = 'ProgrammingError sqlstate=42501 s3://bucket/x?sig=SECRET'


def _raise_from_ingest(error: Exception) -> TableIngestionError:
    table, connector_spec = _table_and_connector_spec()
    with _activity_patches() as patches:
        patches['IngestionValidator'].return_value.pre_ingestion.return_value \
            = ValidationResponse()
        patches['activity'].info.return_value.workflow_id = 'test_wf_id'
        patches['get_connector_from_spec'].side_effect = error
        with pytest.raises(TableIngestionError) as caught:
            ingest_table_data_activity_v2(connector_spec, table, 'v1', 'job_id',
                                          mock.Mock(spec=DeployEnvConfig))
    return caught.value


SAFE_SOURCE_ERROR_CASES = [
    pytest.param(
        RowCountMismatchError(expected=1000, actual=997, table='orders'),
        "Table 'orders' was read incompletely: the source returned 997 rows "
        'but reported 1000', id='row_count'),
    pytest.param(SourceNotFoundError(VENDOR_TEXT, table='orders'),
                 "Table 'orders' does not exist in the source",
                 id='source_not_found'),
    pytest.param(
        SchemaChangedError(VENDOR_TEXT, table='orders'),
        "The columns of table 'orders' changed while Kumo was "
        'reading it', id='schema_changed'),
    pytest.param(SourceAccessDeniedError(VENDOR_TEXT, table='orders'),
                 "Kumo is not permitted to read table 'orders'",
                 id='access_denied'),
    pytest.param(ReconnectRequiredError(VENDOR_TEXT, table='orders'),
                 "read table 'orders' is no longer authorized",
                 id='reconnect_required'),
    pytest.param(UnsupportedReadModeError(VENDOR_TEXT, table='orders'),
                 "Kumo cannot read table 'orders' by any method",
                 id='unsupported_mode'),
    pytest.param(
        RowCountMismatchError(expected=8, actual=3),
        "Table 'test_table' was read incompletely: the source returned 3 "
        'rows but reported 8', id='table_name_from_definition'),
]


@pytest.mark.parametrize('error, expected', SAFE_SOURCE_ERROR_CASES)
def test_self_authored_source_error_reaches_user_with_its_own_message(
        error, expected, caplog):
    """The last case carries no ``table`` of its own: the name falls back to
    the definition being ingested, so the message stays specific."""
    raised = _raise_from_ingest(error)

    assert expected in raised.message
    assert raised.message != activity_module.SAFE_TABLE_INGESTION_ERROR_MESSAGE
    assert VENDOR_TEXT not in raised.message
    assert VENDOR_TEXT not in caplog.text
    assert raised.__cause__ is None
    assert raised.http_code == 422


def test_self_authored_source_errors_do_not_share_a_message():
    templates = list(activity_module._SAFE_SOURCE_ERROR_MESSAGES.values())
    assert len(set(templates)) == len(templates)


@pytest.mark.parametrize('error', [
    PermanentSourceError(VENDOR_TEXT, table='orders'),
    TransientSourceError(VENDOR_TEXT, table='orders'),
    SourceError(VENDOR_TEXT, table='orders'),
], ids=lambda error: type(error).__name__)
def test_vendor_wrapped_source_error_stays_generic(error, caplog):
    raised = _raise_from_ingest(error)

    rendered_trace = ''.join(traceback.format_exception(raised))
    assert raised.message == activity_module.SAFE_TABLE_INGESTION_ERROR_MESSAGE
    assert VENDOR_TEXT not in rendered_trace
    assert VENDOR_TEXT not in caplog.text


@pytest.mark.parametrize(
    'error, non_retryable', [
        (TransientSourceError('throttled'), False),
        (SourceAccessDeniedError('denied', table='orders'), True),
        (PermanentSourceError('unclassified'), True),
        (RowCountMismatchError(expected=8, actual=3, table='orders'), True),
    ], ids=['transient', 'access_denied', 'permanent', 'row_count'])
def test_source_error_retryability_follows_the_taxonomy(error, non_retryable):
    assert _raise_from_ingest(error).non_retryable is non_retryable


def test_no_source_error_type_is_left_unclassified():
    """A source error added later must be given a message or declared opaque
    rather than silently landing on the generic one."""
    declared = (set(activity_module._SAFE_SOURCE_ERROR_MESSAGES)
                | set(activity_module._OPAQUE_SOURCE_ERRORS))
    pending, defined = [SourceError], set()
    while pending:
        error_type = pending.pop()
        if error_type in defined:
            continue
        defined.add(error_type)
        pending.extend(error_type.__subclasses__())

    assert {
        error_type
        for error_type in defined
        if error_type.__module__ == SourceError.__module__
    } <= declared
