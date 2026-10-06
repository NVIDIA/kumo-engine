# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for rest/utils/driver_common.py."""
import asyncio
import hashlib
import os
import os.path as osp
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kumo_core.types import ColStatType, TaskType, persisted_type_token
from kumoapi.data_source import DataSourceType
from kumoapi.source_table import (
    BigQuerySourceTable,
    DatabricksSourceTable,
    FileType,
    GCSSourceTable,
    S3RelativeSource,
    S3SourceTable,
    SnowflakeSourceTable,
)

from rest.config.data_backend import (
    ConnectorExecutionRef,
    ResolvedConnectorSpec,
)
from rest.config.table_v2 import S3SourceTable as V2S3SourceTable
from rest.config.table_v2 import TableDefinition
from rest.utils.driver_common import (
    PREDICTIONS_SOURCE_TABLE_NAME,
    TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME,
    async_connector_lifecycle,
    authorized_graph_for_activity,
    extract_source_table_type,
    fail_closed_graph_from_config,
    generate_bp_stats_workflow_args,
    hydrate_graph_config_for_activity,
)
from rest.utils.table import ColStatsSet
from statedb.config.common import ConnectorID, ProjectID

_CONNECTOR_ID = "connector_" + "1" * 32


class _LifecycleConnector:
    def __init__(self, *, open_error=None, close_error=None):
        self.open_calls = 0
        self.close_calls = 0
        self.open_error = open_error
        self.close_error = close_error

    def open(self):
        self.open_calls += 1
        if self.open_error is not None:
            raise self.open_error
        return self

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error
        return self


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'connector_family',
    ['snowflake', 'databricks', 'bigquery', 's3', 'gcs', 'adls'],
    ids=['warehouse-sf', 'warehouse-db', 'warehouse-bq', 's3', 'gcs', 'adls'],
)
async def test_async_connector_lifecycle_closes_connector_family(
        connector_family):
    connector = _LifecycleConnector()

    async with async_connector_lifecycle(connector, open_connector=True):
        assert connector_family

    assert connector.open_calls == 1
    assert connector.close_calls == 1


@pytest.mark.asyncio
async def test_async_connector_lifecycle_keeps_event_loop_responsive():
    release_open = threading.Event()
    loop_advanced = False

    class BlockingConnector(_LifecycleConnector):
        def open(self):
            self.open_calls += 1
            if not release_open.wait(timeout=2):
                raise RuntimeError('event loop did not advance')
            return self

    connector = BlockingConnector()

    async def use_connector():
        async with async_connector_lifecycle(connector, open_connector=True):
            pass

    async def advance_loop():
        nonlocal loop_advanced
        await asyncio.sleep(0)
        loop_advanced = True
        release_open.set()

    await asyncio.gather(use_connector(), advance_loop())

    assert loop_advanced
    assert connector.close_calls == 1


@pytest.mark.asyncio
async def test_async_connector_lifecycle_closes_after_body_failure():
    connector = _LifecycleConnector()
    primary_error = ValueError('primary failure')

    with pytest.raises(ValueError) as caught:
        async with async_connector_lifecycle(connector):
            raise primary_error

    assert caught.value is primary_error
    assert connector.close_calls == 1


@pytest.mark.asyncio
async def test_async_connector_lifecycle_closes_after_open_failure():
    open_error = RuntimeError('open failure')
    connector = _LifecycleConnector(open_error=open_error)

    with pytest.raises(RuntimeError) as caught:
        async with async_connector_lifecycle(connector, open_connector=True):
            pytest.fail('body must not run after open failure')

    assert caught.value is open_error
    assert connector.close_calls == 1


@pytest.mark.asyncio
async def test_async_connector_lifecycle_closes_after_cancellation():
    connector = _LifecycleConnector()
    body_started = asyncio.Event()
    never_finish = asyncio.Event()

    async def use_connector():
        async with async_connector_lifecycle(connector):
            body_started.set()
            await never_finish.wait()

    task = asyncio.create_task(use_connector())
    await body_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert connector.close_calls == 1


@pytest.mark.asyncio
async def test_async_connector_lifecycle_preserves_cancellation_when_close_fails(
        caplog):
    connector = _LifecycleConnector(
        close_error=RuntimeError('cleanup-cancellation-provider-secret'))
    body_started = asyncio.Event()
    never_finish = asyncio.Event()

    async def use_connector():
        async with async_connector_lifecycle(connector):
            body_started.set()
            await never_finish.wait()

    task = asyncio.create_task(use_connector())
    await body_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert connector.close_calls == 1
    assert 'primary_error_type=CancelledError' in caplog.text
    assert 'cleanup_error_type=RuntimeError' in caplog.text
    assert 'provider-secret' not in caplog.text


@pytest.mark.asyncio
async def test_async_connector_lifecycle_preserves_primary_cleanup_failure(
        caplog):
    primary_error = ValueError('primary-provider-secret')
    connector = _LifecycleConnector(
        close_error=RuntimeError('cleanup-provider-secret'))

    with pytest.raises(ValueError) as caught:
        async with async_connector_lifecycle(connector):
            raise primary_error

    assert caught.value is primary_error
    assert connector.close_calls == 1
    assert 'primary_error_type=ValueError' in caplog.text
    assert 'cleanup_error_type=RuntimeError' in caplog.text
    assert 'provider-secret' not in caplog.text


@pytest.mark.asyncio
async def test_async_connector_lifecycle_propagates_cleanup_only_failure():
    cleanup_error = RuntimeError('cleanup failure')
    connector = _LifecycleConnector(close_error=cleanup_error)

    with pytest.raises(RuntimeError) as caught:
        async with async_connector_lifecycle(connector):
            pass

    assert caught.value is cleanup_error


def test_authorized_graph_for_activity_closes_all_connectors(monkeypatch):
    first = _LifecycleConnector()
    second = _LifecycleConnector()
    connectors = iter([first, second])
    connector_ids = [
        ConnectorID('connector_' + '1' * 32),
        ConnectorID('connector_' + '2' * 32),
    ]
    project_id = ProjectID('project_' + '1' * 32)
    references = {
        connector_id: ConnectorExecutionRef(connector_id, project_id)
        for connector_id in connector_ids
    }
    config = SimpleNamespace(
        data_source_dict={},
        tables=[SimpleNamespace(data_source=value) for value in connector_ids],
    )
    expected_graph = object()
    from rest.utils import driver_common
    monkeypatch.setattr(driver_common, 'get_connector_from_spec',
                        lambda _reference: next(connectors))
    graph_from_config = patch.object(driver_common.Graph, 'from_config',
                                     return_value=expected_graph)

    with graph_from_config as build_graph:
        with authorized_graph_for_activity(config, references) as graph:
            assert graph is expected_graph
            hydrated = build_graph.call_args.args[0]
            assert hydrated.data_source_dict == {
                str(connector_ids[0]): first,
                str(connector_ids[1]): second,
            }

    assert first.close_calls == 1
    assert second.close_calls == 1


def test_hydrate_graph_config_closes_temporary_connector(monkeypatch):
    connector_id = ConnectorID('connector_' + '1' * 32)
    project_id = ProjectID('project_' + '1' * 32)
    reference = ConnectorExecutionRef(connector_id, project_id)
    resolved = ResolvedConnectorSpec(
        connector_id=connector_id,
        connector_config=object(),
        credentials=None,
    )
    connector = _LifecycleConnector()
    connector.config = lambda: {'public_target': 'warehouse'}
    from rest.backends.data_backend import connector_broker
    from rest.utils import driver_common
    monkeypatch.setattr(connector_broker.connector_credential_broker, 'resolve',
                        lambda _reference: resolved)
    monkeypatch.setattr(driver_common, 'get_connector_from_spec',
                        lambda _resolved: connector)
    graph_config = SimpleNamespace(data_source_dict={})

    hydrated, credentials = hydrate_graph_config_for_activity(
        graph_config, {connector_id: reference})

    assert hydrated.data_source_dict == {
        str(connector_id): {
            'public_target': 'warehouse'
        }
    }
    assert credentials == {}
    assert connector.close_calls == 1


def test_authorized_graph_for_activity_preserves_body_failure_and_closes_all(
        monkeypatch, caplog):
    primary = RuntimeError('provider-primary-secret')
    first = _LifecycleConnector(
        close_error=RuntimeError('provider-cleanup-secret'))
    second = _LifecycleConnector()
    connectors = iter([first, second])
    connector_ids = [
        ConnectorID('connector_' + '1' * 32),
        ConnectorID('connector_' + '2' * 32),
    ]
    project_id = ProjectID('project_' + '1' * 32)
    references = {
        connector_id: ConnectorExecutionRef(connector_id, project_id)
        for connector_id in connector_ids
    }
    config = SimpleNamespace(
        data_source_dict={},
        tables=[SimpleNamespace(data_source=value) for value in connector_ids],
    )
    from rest.utils import driver_common
    monkeypatch.setattr(driver_common, 'get_connector_from_spec',
                        lambda _reference: next(connectors))
    monkeypatch.setattr(driver_common.Graph, 'from_config',
                        lambda _config: object())

    with pytest.raises(RuntimeError) as caught:
        with authorized_graph_for_activity(config, references):
            raise primary

    assert caught.value is primary
    assert first.close_calls == 1
    assert second.close_calls == 1
    assert 'cleanup_error_type=RuntimeError' in caplog.text
    assert 'provider-' not in caplog.text


def test_fail_closed_graph_from_config_performs_no_provider_io(monkeypatch):
    connector = MagicMock()
    connector.open.side_effect = AssertionError('request-side provider I/O')
    connector.get_data_version.side_effect = AssertionError(
        'request-side provider I/O')
    graph = SimpleNamespace(data_source_dict={'warehouse': connector})
    from rest.utils import driver_common
    from_config = MagicMock(return_value=graph)
    monkeypatch.setattr(driver_common.Graph, 'from_config', from_config)
    config = object()

    assert fail_closed_graph_from_config(config) is graph

    from_config.assert_called_once_with(config)
    connector.require_explicit_credentials.assert_called_once_with()
    connector.open.assert_not_called()
    connector.get_data_version.assert_not_called()


class TestExtractSourceTableType:
    """Tests for extract_source_table_type function."""
    def test_extract_s3_source_table(self):
        """Test extracting S3 source table type."""
        source_table = S3SourceTable(
            connector_id=_CONNECTOR_ID,
            source={
                'mode': 'relative',
                'path': "test_table"
            },
            file_type=FileType.PARQUET,
        )
        result_table, table_name, connector_id = extract_source_table_type(
            source_table)

        assert table_name == "test_table"
        assert connector_id is not None
        assert str(connector_id) == _CONNECTOR_ID

    def test_extract_gcs_source_table(self):
        """Test extracting GCS source table type."""
        source_table = GCSSourceTable(
            connector_id=_CONNECTOR_ID,
            source_table_name="gcs_test_table",
            file_type=FileType.PARQUET,
        )
        result_table, table_name, connector_id = extract_source_table_type(
            source_table)

        assert table_name == "gcs_test_table"
        assert connector_id is not None
        assert str(connector_id) == _CONNECTOR_ID

    def test_extract_snowflake_source_table(self):
        """Test extracting Snowflake source table type."""
        source_table = SnowflakeSourceTable(
            snowflake_connector_id=_CONNECTOR_ID,
            database="test_db",
            schema_name="test_schema",
            table="test_table",
            data_source_type=DataSourceType.SNOWFLAKE,
        )
        result_table, table_name, connector_id = extract_source_table_type(
            source_table)

        assert table_name == "test_table"
        assert connector_id is not None
        assert str(connector_id) == _CONNECTOR_ID

    def test_extract_bigquery_source_table(self):
        """Test extracting BigQuery source table type."""
        source_table = BigQuerySourceTable(
            bigquery_connector_id=_CONNECTOR_ID,
            project_id="test_project",
            dataset_id="test_dataset",
            table_name="test_table",
            data_source_type=DataSourceType.BIGQUERY,
        )
        result_table, table_name, connector_id = extract_source_table_type(
            source_table)

        assert table_name == "test_table"
        assert connector_id is not None
        assert str(connector_id) == _CONNECTOR_ID

    def test_extract_databricks_source_table(self):
        """Test extracting Databricks source table type."""
        source_table = DatabricksSourceTable(
            databricks_connector_id=_CONNECTOR_ID,
            table="test_table",
            data_source_type=DataSourceType.DATABRICKS,
        )
        result_table, table_name, connector_id = extract_source_table_type(
            source_table)

        assert table_name == "test_table"
        assert connector_id is not None
        assert str(connector_id) == _CONNECTOR_ID

    def test_extract_invalid_source_table_raises_error(self):
        """Test that invalid source table type raises ValueError."""
        class InvalidSourceTable:
            pass

        with pytest.raises(ValueError, match="Invalid source table type"):
            extract_source_table_type(InvalidSourceTable())

    @pytest.mark.parametrize(
        "source_table",
        [
            S3SourceTable(connector_id=_CONNECTOR_ID, source={
                'mode': 'relative',
                'path': "t"
            }, file_type=FileType.PARQUET),
            GCSSourceTable(connector_id=_CONNECTOR_ID, source_table_name="t",
                           file_type=FileType.PARQUET),
            SnowflakeSourceTable(snowflake_connector_id=_CONNECTOR_ID,
                                 database="d", schema_name="s", table="t",
                                 data_source_type=DataSourceType.SNOWFLAKE),
            BigQuerySourceTable(bigquery_connector_id=_CONNECTOR_ID,
                                project_id="p", dataset_id="d", table_name="t",
                                data_source_type=DataSourceType.BIGQUERY),
            DatabricksSourceTable(databricks_connector_id=_CONNECTOR_ID,
                                  table="t",
                                  data_source_type=DataSourceType.DATABRICKS),
        ],
        ids=[
            "s3-explicit",
            "gcs-explicit",
            "warehouse-sf",
            "bigquery",
            "databricks",
        ],
    )
    def test_extract_source_table_connector_id_never_none(self, source_table):
        """Every final source-table shape carries an explicit connector ID."""
        _, _, connector_id = extract_source_table_type(source_table)
        assert connector_id is not None


@pytest.fixture
def _bp_home_dir():
    """Provide a temp dir mocked as a job home dir.

    Layout:
      <home>/predictions/   (parquet directory)
      <home>/transformed/   (parquet directory + metadata file)
    """
    with tempfile.TemporaryDirectory() as home:
        # raw predictions
        raw_dir = osp.join(home, 'output', PREDICTIONS_SOURCE_TABLE_NAME)
        os.makedirs(raw_dir)
        # transformed predictions
        tr_dir = osp.join(home, 'output',
                          TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME)
        os.makedirs(tr_dir)
        yield home


class TestGenerateBpStatsWorkflowArgs:
    """Tests for the v2-native `generate_bp_stats_workflow_args`.

    These tests verify the migrated function returns a `TableDefinition`
    (not a v1 table-args object) and produces byte-identical cache/workflow
    keys to what the v1 form would have produced — protecting cache hit
    continuity across the migration.
    """
    @pytest.mark.asyncio
    async def test_returns_table_definition_for_raw_predictions(
            self, _bp_home_dir):
        # `_bp_home_dir` already creates the prediction output directory,
        # so `ParquetConnector._get_data_version` works against real disk
        # (returns the dir mtime). No connector method needs to be mocked.
        with patch('rest.utils.driver_common.HomeDirManager') as MockHDM:
            hdm = MockHDM.return_value
            hdm.prediction_output_dir = osp.join(_bp_home_dir, 'output')
            hdm.internal_predictions_file = osp.join(
                hdm.prediction_output_dir, PREDICTIONS_SOURCE_TABLE_NAME)
            connector_spec, table_def, data_version, col_stats_set = (
                await generate_bp_stats_workflow_args(
                    job_id='job-abc',
                    job_home_dir=_bp_home_dir,
                    task_type=TaskType.BINARY_CLASSIFICATION,
                    is_temporal=False,
                    entity_col_name='article_id',
                ))

        assert isinstance(table_def, TableDefinition)
        # source_table is a v2 S3SourceTable wrapping the parquet output dir
        assert isinstance(table_def.source_table, V2S3SourceTable)
        assert isinstance(table_def.source_table.source, S3RelativeSource)
        assert table_def.source_table.source.path == \
            PREDICTIONS_SOURCE_TABLE_NAME
        assert table_def.source_table.connector_id == \
            connector_spec.connector_id
        # cols/pkey come from get_bp_columns_by_task_type
        assert len(table_def.cols) > 0
        assert table_def.pkey == 'ENTITY'
        assert table_def.cols[0].name == 'ENTITY'
        assert col_stats_set == ColStatsSet.BP
        # data_version is the dir mtime returned by the real connector.
        assert data_version

    @pytest.mark.asyncio
    async def test_closes_internal_prediction_connector(self, _bp_home_dir):
        connector = _LifecycleConnector()
        connector._get_data_version = SimpleNamespace(coroutine=AsyncMock(
            return_value='v0'))

        with patch('rest.utils.driver_common.HomeDirManager') as MockHDM, \
                patch(
                    'rest.utils.driver_common.get_connector_from_spec_async',
                    new=AsyncMock(return_value=connector),
                ):
            hdm = MockHDM.return_value
            hdm.prediction_output_dir = osp.join(_bp_home_dir, 'output')
            hdm.internal_predictions_file = osp.join(
                hdm.prediction_output_dir, PREDICTIONS_SOURCE_TABLE_NAME)
            await generate_bp_stats_workflow_args(
                job_id='job-abc',
                job_home_dir=_bp_home_dir,
                task_type=TaskType.BINARY_CLASSIFICATION,
                is_temporal=False,
                entity_col_name='ENTITY',
            )

        connector._get_data_version.coroutine.assert_awaited_once_with(
            PREDICTIONS_SOURCE_TABLE_NAME)
        assert connector.close_calls == 1

    @pytest.mark.asyncio
    async def test_direct_output_version_skips_missing_internal_parquet(
            self, _bp_home_dir):
        with patch('rest.utils.driver_common.HomeDirManager') as MockHDM, \
                patch(
                    'rest.utils.driver_common.get_connector_from_spec_async',
                    side_effect=AssertionError(
                        'direct output must not inspect internal parquet'),
                ):
            MockHDM.return_value.prediction_output_dir = osp.join(
                _bp_home_dir, 'missing-output')
            _, _, data_version, _ = await generate_bp_stats_workflow_args(
                job_id='job-direct',
                job_home_dir=_bp_home_dir,
                task_type=TaskType.BINARY_CLASSIFICATION,
                is_temporal=False,
                entity_col_name='ENTITY',
                data_version_override='job-direct',
            )

        assert str(data_version) == 'job-direct'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('task_type', 'column_name', 'expected_count'),
        [
            (TaskType.REGRESSION, 'TARGET_PRED', 0),
            (TaskType.MULTICLASS_CLASSIFICATION, 'SCORE', 1),
        ],
    )
    async def test_direct_stats_request_by_class(
        self,
        _bp_home_dir,
        task_type,
        column_name,
        expected_count,
    ):
        _, table_def, _, stats_set = await generate_bp_stats_workflow_args(
            job_id='job-direct',
            job_home_dir=_bp_home_dir,
            task_type=task_type,
            is_temporal=False,
            entity_col_name='ENTITY',
            data_version_override='job-direct',
        )

        request = stats_set.get_stats_types(table_def)
        stats = next(stats for column, stats in request.items()
                     if column.name == column_name)
        assert stats.count(ColStatType.SUMMARY_STATS_BY_CLASS) == expected_count

    @pytest.mark.asyncio
    async def test_stable_identity_keys_and_hardened_cache_key(
            self, _bp_home_dir):
        """Identity keys stay stable while cache paths hash the source name.

        The legacy ``CreateTableArgs`` / ``table_wf_key`` / ``table_cache_key``
        helpers were removed when the data-backend utils were narrowed to
        TableDefinition only. Workflow identity and dedupe bytes remain
        compatible, while the final cache key deliberately hashes the source
        name so arbitrary object-store paths cannot shape local directories.
        """
        from hashlib import md5

        from kumo_core.types.schema import IngestedTableFingerprint

        from rest.config import batch_prediction_job
        from rest.utils.driver_common import prediction_connector_from_job_id

        with patch('rest.utils.driver_common.HomeDirManager') as MockHDM, \
                patch.object(
                    __import__('kumo.connector',
                               fromlist=['ParquetConnector']).ParquetConnector,
                    '_get_data_version',
                    new=AsyncMock(return_value='v0')):
            hdm = MockHDM.return_value
            hdm.prediction_output_dir = osp.join(_bp_home_dir, 'output')
            hdm.internal_predictions_file = osp.join(
                hdm.prediction_output_dir, PREDICTIONS_SOURCE_TABLE_NAME)
            _, table_def, _, _ = await generate_bp_stats_workflow_args(
                job_id='job-abc',
                job_home_dir=_bp_home_dir,
                task_type=TaskType.BINARY_CLASSIFICATION,
                is_temporal=False,
                entity_col_name='ENTITY',
            )

        bp_connector_id = prediction_connector_from_job_id('job-abc')
        v1_cols = batch_prediction_job.get_bp_columns_by_task_type(
            TaskType.BINARY_CLASSIFICATION, False, 'ENTITY')
        # v1 ``table_args_to_dedupe_hash`` formula (helper removed; inlined
        # here so the byte-stability cross-check still works).
        _hasher = md5()
        _hasher.update(f'connector={bp_connector_id}'.encode())
        _hasher.update(f'source_table={PREDICTIONS_SOURCE_TABLE_NAME}'.encode())
        for _col in sorted(v1_cols):
            _hasher.update(_col.name.encode())
            # v1 spelled these with `str()`, which rendered the `kumo_core`
            # enums as `"kumo.<value>"`. That copy is gone and kumoapi's `str`
            # is the bare value, so the v1 byte sequence now has to be asked
            # for by name -- which is what the hasher under test does too.
            _hasher.update(persisted_type_token(_col.stype).encode())
            _hasher.update(persisted_type_token(_col.dtype).encode())
        _hasher.update('pkey=ENTITY'.encode())
        _hasher.update('time_col=None'.encode())
        _hasher.update('end_time_col=None'.encode())
        v1_dedupe_hash = _hasher.hexdigest()
        # v1 ``CreateTableArgs.fingerprint()`` formula (time_col/end_time_col
        # are None here, so no per-column config is attached).
        v1_fingerprint = IngestedTableFingerprint(
            connector_name=bp_connector_id,
            source_name=PREDICTIONS_SOURCE_TABLE_NAME,
            cols={c.name: c.dtype
                  for c in v1_cols},
            pkey='ENTITY',
            time_col=None,
            end_time_col=None,
        )
        v1_wf_key = '.'.join([v1_fingerprint.fpid(), v1_dedupe_hash[:8]])
        source_hash = hashlib.sha256(
            PREDICTIONS_SOURCE_TABLE_NAME.encode()).hexdigest()
        expected_cache_key = f'table_{source_hash}_{v1_dedupe_hash}'
        # Workflow-identity contracts remain byte-stable; cache paths use the
        # final hardened shape from ``TableDefinition.cache_key``.
        assert table_def._dedupe_hash() == v1_dedupe_hash
        assert table_def.wf_key() == v1_wf_key
        assert table_def.cache_key() == expected_cache_key

    @pytest.mark.asyncio
    async def test_transformed_predictions_preserve_entity_name(
            self, _bp_home_dir):
        from kumo_core.types.schema import BaseColumnConfig
        from kumoapi.typing import Dtype, Stype

        from kumo.config.graph import TableMetadataConfig

        metadata = TableMetadataConfig(
            table_name=TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME,
            source_name=TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME,
            data_source='bp',
            cols=[
                BaseColumnConfig(
                    name='article_id',
                    stype=Stype.ID,
                    dtype=Dtype.int64,
                    metadata={},
                ),
            ],
        )
        with patch('rest.utils.driver_common.HomeDirManager') as MockHDM, \
                patch(
                    'rest.utils.driver_common.load_json_async',
                    new=AsyncMock(return_value=metadata),
                ):
            hdm = MockHDM.return_value
            hdm.prediction_output_dir = osp.join(_bp_home_dir, 'output')
            hdm.internal_transformed_predictions_metadata_file = 'metadata'
            _, table_def, _, _ = await generate_bp_stats_workflow_args(
                job_id='job-abc',
                job_home_dir=_bp_home_dir,
                task_type=TaskType.BINARY_CLASSIFICATION,
                is_temporal=False,
                entity_col_name='article_id',
                use_transformed_output=True,
            )

        assert table_def.pkey == 'article_id'
        assert table_def.cols[0].name == 'article_id'

    @pytest.mark.asyncio
    async def test_pkey_none_when_entity_missing_from_transformed(
            self, _bp_home_dir):
        """When `use_transformed_output=True` and the entity column isn't
        present in the transformed-predictions metadata, pkey is None."""
        from kumo_core.types.schema import BaseColumnConfig
        from kumoapi.typing import Dtype, Stype

        from io_utils.json import to_json
        from kumo.config.graph import TableMetadataConfig

        # Write a fake transformed-predictions metadata file the function
        # will load via load_json_async. Views were removed in v2, so
        # ``ViewMetadataConfig`` no longer exists; the production code now
        # loads this file as a ``TableMetadataConfig``.
        tr_dir = osp.join(_bp_home_dir, 'output',
                          TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME)
        meta_path = osp.join(
            tr_dir, '__internal_transformed_predictions_metadata.json')
        # Metadata cols intentionally exclude 'ENTITY' so pkey -> None.
        meta = TableMetadataConfig(
            table_name=TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME,
            source_name=TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME,
            data_source='bp',
            cols=[
                BaseColumnConfig(name='SCORE', stype=Stype.numerical,
                                 dtype=Dtype.float64, metadata={}),
            ],
        )
        with open(meta_path, 'w') as f:
            f.write(to_json(meta))

        # `_bp_home_dir` already creates the transformed-predictions output
        # directory, so the connector's `_get_data_version` works against
        # real disk. Only `HomeDirManager` needs patching.
        with patch('rest.utils.driver_common.HomeDirManager') as MockHDM:
            hdm = MockHDM.return_value
            hdm.prediction_output_dir = osp.join(_bp_home_dir, 'output')
            hdm.internal_transformed_predictions_metadata_file = meta_path
            _, table_def, _, _ = await generate_bp_stats_workflow_args(
                job_id='job-abc',
                job_home_dir=_bp_home_dir,
                task_type=TaskType.BINARY_CLASSIFICATION,
                is_temporal=False,
                entity_col_name='ENTITY',
                use_transformed_output=True,
            )

        assert isinstance(table_def, TableDefinition)
        assert table_def.pkey is None
        assert isinstance(table_def.source_table.source, S3RelativeSource)
        assert table_def.source_table.source.path == \
            TRANSFORMED_PREDICTIONS_SOURCE_TABLE_NAME
