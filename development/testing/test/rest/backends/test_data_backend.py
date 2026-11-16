# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
from unittest import mock

import pytest
from botocore.exceptions import ClientError, NoCredentialsError
from kumo_core.types import ColStatType, PredictionArtifactType
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import (
    ActivityError,
    ApplicationError,
    CancelledError,
    RetryState,
    TimeoutError,
    TimeoutType,
)

from common_utils.error import PublicException
from kumo.data import get_active_cache
from kumo.stats.stats import Stats
from rest.backends.data_backend import data_backend as data_backend_module
from rest.backends.data_backend.activities import DataBackendActivities
from rest.backends.data_backend.data_backend import DataBackend
from rest.backends.data_backend.utils import get_table_data_version
from rest.config.data_backend import ConnectorExecutionRef, ConnectorSpec
from rest.config.table_v2 import TableDefinition
from statedb.config.common import ConnectorID
from statedb.config.table import TableDataVersion
from temporalio_commons.auto_rpc_exception import RpcError

_CONNECTOR_ID = ConnectorID('connector_' + '1' * 32)
_SINGLE_ACTIVITY_WORKFLOW_ID = 'validate-table:test-project:' + 'a' * 32


@pytest.mark.asyncio
async def test_embeddings_only_bp_stats_never_opens_predictions() -> None:
    job_input = mock.Mock()
    job_input.id = 'bp-job-preserved-02'
    job_input.config = mock.Mock(outputs=[
        mock.Mock(artifact_type=PredictionArtifactType.EMBEDDINGS),
    ])

    with mock.patch('rest.utils.driver_common.generate_bp_stats_workflow_args',
                    side_effect=AssertionError('must not open predictions')):
        await DataBackend.start_bp_stats_activity(mock.Mock(), job_input)


@pytest.mark.asyncio
@pytest.mark.parametrize('clock', ['seconds', 'milliseconds'])
async def test_bp_stats_reads_only_manifest_selected_parts(tmp_path,
                                                           clock) -> None:
    from types import SimpleNamespace

    from rest.utils.table import ColStatsSet

    artifact_root = tmp_path / 'predictions.parquet'
    artifact_root.mkdir()
    selected = [artifact_root / 'part-00000.parquet',
                artifact_root / 'part-00001.parquet']
    for path in selected:
        path.write_bytes(b'parquet')
    (artifact_root / 'foreign.parquet').write_bytes(b'foreign')

    job_input = mock.Mock()
    job_input.id = 'bp-job-selected-stats'
    job_input.config = mock.Mock(
        outputs=[mock.Mock(artifact_type=PredictionArtifactType.PREDICTIONS)],
        task_type=mock.sentinel.task_type,
    )
    job_input.predictive_query.get_bp_entity_col_name.return_value = 'ENTITY'
    job_input.predictive_query.query_type = mock.sentinel.query_type
    job_input.graph_version_info = SimpleNamespace(timestamp_precision=clock)

    table_def = mock.Mock()
    table_def.fingerprint.return_value = 'fingerprint'
    table_def.get_connector_id.return_value = 'connector'
    table_def.get_source_table_name.return_value = 'predictions'
    table_def.cache_key.return_value = 'cache-key'
    stats_set = mock.Mock(spec=ColStatsSet)
    data_version = TableDataVersion('selected-version')
    (tmp_path / 'cache').mkdir()
    table_dir = SimpleNamespace(
        col_stats=str(tmp_path / 'cache' / 'col_stats'),
        min_col_stats=str(tmp_path / 'cache' / 'min_col_stats'),
    )
    refresh_save = mock.Mock(coroutine=mock.AsyncMock())
    cache = mock.Mock(
        create_table_cache_dir=mock.Mock(return_value=table_dir),
        table_refresh_result=mock.Mock(save=refresh_save),
    )
    compute = mock.Mock(coroutine=mock.AsyncMock(return_value=('{}', '{}')))
    service = mock.Mock(compute_col_stats_for_parquet=compute)
    resolved = SimpleNamespace(
        root=str(artifact_root),
        files=tuple(str(path) for path in selected),
        selection_ref=mock.sentinel.selection_ref,
    )

    with mock.patch(
            'rest.backends.ml_backend.data_util.get_job_base_dir',
            return_value=str(tmp_path)), mock.patch(
                'rest.backends.ml_backend.data_util.get_job_dir',
                return_value=str(tmp_path / 'job')), mock.patch(
                    'rest.utils.driver_common.'
                    'generate_bp_stats_workflow_args',
                    new=mock.AsyncMock(return_value=(
                        mock.Mock(connector_id='connector'),
                        table_def,
                        data_version,
                        stats_set,
                    ))), mock.patch(
                        'rest.backends.data_backend.data_backend.'
                        'resolve_prediction_artifact',
                        new=SimpleNamespace(
                            coroutine=mock.AsyncMock(return_value=resolved))), \
            mock.patch(
                            'kumo.diskgraph.get_service',
                            return_value=service), mock.patch(
                                'kumo.diskgraph.service.'
                                '_to_kumo_col_specs', return_value=[]), \
            mock.patch(
                'rest.backends.data_backend.data_backend.get_active_cache',
                return_value=cache), mock.patch(
                    'rest.backends.data_backend.data_backend.'
                    '_list_parquet_files',
                    side_effect=AssertionError(
                        'selected local stats must not list the root')):
        await DataBackend.start_bp_stats_activity(mock.Mock(), job_input)

    kwargs = compute.coroutine.await_args.kwargs
    assert kwargs['parquet_paths'] == [str(path) for path in selected]
    assert str(artifact_root / 'foreign.parquet') not in kwargs['parquet_paths']
    assert kwargs['timestamp_precision'] == clock


class _UnapprovedSingleActivityError(PublicException):
    pass


@pytest.mark.asyncio
async def test_execute_single_activity_redacts_temporal_native_error(caplog):
    secret = 'password=provider-secret-value'
    temporal_client = mock.AsyncMock()
    temporal_client.execute_workflow.side_effect = WorkflowFailureError(
        cause=_activity_error(
            ApplicationError(secret, {'stack': secret}, type='ValueError')))
    temporal_client_provider = mock.Mock()
    temporal_client_provider.get = mock.AsyncMock(return_value=temporal_client)
    context = mock.Mock()
    context.temporal_client.return_value = temporal_client_provider

    async def activity_fn():
        pass

    with mock.patch.object(data_backend_module, 'uuid4') as uuid4_mock:
        uuid4_mock.return_value.hex = 'a' * 32
        with pytest.raises(RpcError) as caught:
            await DataBackend._execute_single_activity(
                context,
                activity_fn,
                [],
                'validate-table:test-project',
            )

    assert caught.value.error_type == 'InternalError'
    assert caught.value.message == (
        f'{data_backend_module._SINGLE_ACTIVITY_FAILURE_MESSAGE} '
        f'Reference: {_SINGLE_ACTIVITY_WORKFLOW_ID}.')
    assert caught.value.stack_trace == ''
    assert secret not in str(caught.value)
    assert _SINGLE_ACTIVITY_WORKFLOW_ID in caplog.text
    assert 'activity_name=activity_fn' in caplog.text
    assert 'error_type=WorkflowFailureError' in caplog.text
    assert secret not in caplog.text
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_execute_single_activity_redacts_temporal_client_setup_error():
    secret = 'temporal-transport-secret'
    temporal_client_provider = mock.Mock()
    temporal_client_provider.get = mock.AsyncMock(
        side_effect=RuntimeError(f'connection failed: {secret}'))
    context = mock.Mock()
    context.temporal_client.return_value = temporal_client_provider

    async def activity_fn():
        pass

    with mock.patch.object(data_backend_module, 'uuid4') as uuid4_mock:
        uuid4_mock.return_value.hex = 'a' * 32
        with pytest.raises(RpcError) as caught:
            await DataBackend._execute_single_activity(
                context,
                activity_fn,
                [],
                'validate-table:test-project',
            )

    assert caught.value.error_type == 'InternalError'
    assert caught.value.message.endswith(
        f'Reference: {_SINGLE_ACTIVITY_WORKFLOW_ID}.')
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_execute_single_activity_sanitizes_direct_cancellation():
    secret = 'direct-cancellation-secret'
    temporal_client = mock.AsyncMock()
    temporal_client.execute_workflow.side_effect = asyncio.CancelledError(
        secret)
    temporal_client_provider = mock.Mock()
    temporal_client_provider.get = mock.AsyncMock(return_value=temporal_client)
    context = mock.Mock()
    context.temporal_client.return_value = temporal_client_provider

    async def activity_fn():
        pass

    with mock.patch.object(data_backend_module, 'uuid4') as uuid4_mock:
        uuid4_mock.return_value.hex = 'a' * 32
        with pytest.raises(RpcError) as caught:
            await DataBackend._execute_single_activity(
                context,
                activity_fn,
                [],
                'validate-table:test-project',
            )

    assert caught.value.error_type == 'InternalError'
    assert caught.value.stack_trace == ''
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def _activity_error(cause: BaseException) -> ActivityError:
    error = ActivityError(
        'activity failed',
        scheduled_event_id=1,
        started_event_id=2,
        identity='worker',
        activity_type='test_activity',
        activity_id='activity-id',
        retry_state=RetryState.NON_RETRYABLE_FAILURE,
    )
    error.__cause__ = cause
    return error


def test_single_activity_failure_preserves_nested_public_exception():
    expected = PublicException(message='Safe table validation failure')
    application_error = ApplicationError(
        expected.message,
        expected.to_dict(),
        type=type(expected).__name__,
        non_retryable=True,
    )
    failure = WorkflowFailureError(cause=_activity_error(application_error))

    with pytest.raises(PublicException) as caught:
        data_backend_module._raise_safe_single_activity_failure(
            failure, 'test_activity', _SINGLE_ACTIVITY_WORKFLOW_ID)

    assert caught.value.message == expected.message
    assert caught.value.http_code == 422
    assert caught.value.__cause__ is None


def test_single_activity_failure_redacts_arbitrary_secret_cause():
    secret = 'token=secret-like-cause-text'
    failure = WorkflowFailureError(cause=_activity_error(
        ApplicationError(secret, {'trace': secret}, type='RuntimeError')))

    with pytest.raises(RpcError) as caught:
        data_backend_module._raise_safe_single_activity_failure(
            failure, 'test_activity', _SINGLE_ACTIVITY_WORKFLOW_ID)

    assert caught.value.error_type == 'InternalError'
    assert caught.value.message == (
        f'{data_backend_module._SINGLE_ACTIVITY_FAILURE_MESSAGE} '
        f'Reference: {_SINGLE_ACTIVITY_WORKFLOW_ID}.')
    assert caught.value.stack_trace == ''
    assert secret not in str(caught.value)


def test_unknown_pquery_validation_failure_is_redacted_with_reference():
    secret = 'api_key=unknown-model-plan-failure-secret'
    failure = WorkflowFailureError(cause=_activity_error(
        ApplicationError(secret, {'model_plan': secret}, type='RuntimeError')))

    with pytest.raises(RpcError) as caught:
        data_backend_module._raise_safe_single_activity_failure(
            failure,
            'validate_predictive_query_activity',
            _SINGLE_ACTIVITY_WORKFLOW_ID,
        )

    assert caught.value.error_type == 'InternalError'
    assert caught.value.message == (
        f'{data_backend_module._SINGLE_ACTIVITY_FAILURE_MESSAGE} '
        f'Reference: {_SINGLE_ACTIVITY_WORKFLOW_ID}.')
    assert caught.value.stack_trace == ''
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_single_activity_failure_rejects_unapproved_public_error_payload():
    secret = 'unapproved-public-provider-secret'
    unapproved = _UnapprovedSingleActivityError(message=secret)
    application_error = ApplicationError(
        unapproved.message,
        unapproved.to_dict(),
        type=type(unapproved).__name__,
        non_retryable=True,
    )
    failure = WorkflowFailureError(cause=_activity_error(application_error))

    with pytest.raises(RpcError) as caught:
        data_backend_module._raise_safe_single_activity_failure(
            failure, 'test_activity', _SINGLE_ACTIVITY_WORKFLOW_ID)

    assert caught.value.error_type == 'InternalError'
    assert caught.value.message == (
        f'{data_backend_module._SINGLE_ACTIVITY_FAILURE_MESSAGE} '
        f'Reference: {_SINGLE_ACTIVITY_WORKFLOW_ID}.')
    assert caught.value.stack_trace == ''
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_single_activity_failure_sanitizes_timeout_as_internal_error():
    secret = 'heartbeat-secret-value'
    timeout = TimeoutError(
        secret,
        type=TimeoutType.START_TO_CLOSE,
        last_heartbeat_details=(secret, ),
    )
    failure = WorkflowFailureError(cause=_activity_error(timeout))

    with pytest.raises(RpcError) as caught:
        data_backend_module._raise_safe_single_activity_failure(
            failure, 'test_activity', _SINGLE_ACTIVITY_WORKFLOW_ID)

    assert caught.value.error_type == 'InternalError'
    assert caught.value.message == (
        f'{data_backend_module._SINGLE_ACTIVITY_FAILURE_MESSAGE} '
        f'Reference: {_SINGLE_ACTIVITY_WORKFLOW_ID}.')
    assert caught.value.stack_trace == ''
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_single_activity_failure_sanitizes_cancellation_as_internal_error():
    secret = 'provider-secret-value'
    failure = WorkflowFailureError(
        cause=_activity_error(CancelledError(secret)))

    with pytest.raises(RpcError) as caught:
        data_backend_module._raise_safe_single_activity_failure(
            failure, 'test_activity', _SINGLE_ACTIVITY_WORKFLOW_ID)

    assert caught.value.error_type == 'InternalError'
    assert caught.value.stack_trace == ''
    assert secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.asyncio
async def test_get_table_data_version_closes_connector():
    connector = mock.Mock()
    connector.get_data_version.coroutine = mock.AsyncMock(
        return_value='current-version')
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())

    with mock.patch('rest.backends.data_backend.utils.get_connector_from_spec',
                    return_value=connector):
        result = await get_table_data_version(
            connector_spec,
            'source-table',
            refresh_table_version=True,
        )

    assert result == TableDataVersion('current-version')
    connector.get_data_version.coroutine.assert_awaited_once_with(
        'source-table', True, False)
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_get_table_data_version_in_activity_threads_table_def():
    """The refresh/version path forwards ``table_def`` into the activity args so
    a Custom SQL table's change-detection version folds in the SQL + referenced
    tables (rather than probing the synthetic name)."""
    ref = ConnectorExecutionRef(
        connector_id=_CONNECTOR_ID,
        project_id='project_' + '0' * 32,
        authorization_token='tok',
    )
    table_def = mock.Mock()
    with mock.patch.object(
            DataBackend,
            '_execute_single_activity',
            new=mock.AsyncMock(return_value=TableDataVersion('v-sql')),
    ) as exec_activity:
        result = await DataBackend.get_table_data_version_in_activity(
            mock.Mock(),
            ref,
            'custom_sql_abc',
            refresh_table_version=True,
            table_def=table_def,
        )

    assert result == TableDataVersion('v-sql')
    ctx_arg, activity_fn, activity_args, _wf_id = exec_activity.await_args.args
    assert activity_fn is DataBackendActivities.get_connector_table_data_version
    assert activity_args == [ref, 'custom_sql_abc', True, table_def]


def _validate_public_s3_dataset(path: str):
    try:
        return DataBackend.s3_validate(path)
    except NoCredentialsError:
        pytest.skip('AWS credentials are unavailable for public S3 validation')
    except ClientError as error:
        error_code = error.response.get('Error', {}).get('Code')
        if error_code in {
                'AccessDenied', 'ExpiredToken', 'InvalidAccessKeyId',
                'SignatureDoesNotMatch'
        }:
            pytest.skip(f'AWS access is unavailable for public S3 validation: '
                        f'{error_code}')
        raise


def test_csv_connector():
    valid_ret = _validate_public_s3_dataset(
        's3://kumo-public-datasets/netflix_integ_test_csv/')
    assert valid_ret.count_total == 3


def test_parquet_connector():
    valid_ret = _validate_public_s3_dataset(
        's3://kumo-public-datasets/test_parquet/')
    assert valid_ret.count_total == 13


def test_csv_parquet_connector():
    valid_ret = _validate_public_s3_dataset(
        's3://kumo-public-datasets/csv_parquet/')
    assert valid_ret.count_total == 2
    assert valid_ret.count_selected == 2
    assert valid_ret.count_ignored == 0


def test_large_number_files():
    valid_ret = _validate_public_s3_dataset(
        's3://kumo-public-datasets/test_1000_tables/')
    assert valid_ret.count_total == 1010
    assert valid_ret.count_selected == 200
    assert valid_ret.count_ignored == 810


@pytest.mark.asyncio
async def test_get_min_col_stats_uses_table_fingerprint_connector_for_cache(
        make_default_table_def_v2):
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('test-version')
    base_col = table.to_base_columns()[1]

    cache = get_active_cache()
    cache.add_col_stats(
        table.fingerprint(),
        {
            base_col: {
                ColStatType.COUNT:
                Stats(value=10, approx=False, data_version=data_version),
            },
        },
    )
    await cache.save_col_stats_cache.coroutine(table.fingerprint(),
                                               data_version)

    connector = mock.Mock()
    connector.name = _CONNECTOR_ID
    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector) as get_connector_from_spec:
        result = await DataBackend.get_min_col_stats(
            ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock()),
            table,
            {
                base_col: [ColStatType.COUNT, ColStatType.NUM_UNIQUE],
            },
            approx_sample_size=10,
            check_refresh_status=False,
            compute_missing_stats=False,
            table_data_version=data_version,
        )

    get_connector_from_spec.assert_not_called()
    assert result == {
        base_col: {
            ColStatType.COUNT:
            Stats(value=10, approx=False, data_version=data_version),
        },
    }


@pytest.mark.asyncio
async def test_get_min_col_stats_caches_connector_sample_fallback(
        make_default_table_def_v2):
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('sample-version')
    base_col = table.to_base_columns()[1]
    sample_stats = {
        base_col: {
            ColStatType.COUNT:
            Stats(value=10, approx=True, data_version=data_version),
            ColStatType.NUM_UNIQUE:
            Stats(value=3, approx=True, data_version=data_version),
        },
    }

    connector = mock.Mock()
    connector.compute_col_stats.return_value = sample_stats
    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector):
        result = await DataBackend.get_min_col_stats(
            ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock()),
            table,
            {
                base_col: [ColStatType.COUNT, ColStatType.NUM_UNIQUE],
            },
            approx_sample_size=10,
            table_data_version=data_version,
        )

    assert result == sample_stats
    connector.compute_col_stats.assert_called_once()
    connector.close.assert_called_once_with()

    cache = get_active_cache()
    table_fingerprint = table.fingerprint()
    assert cache.load_min_col_stats_cache(
        connector_name=table_fingerprint.connector_name,
        table_fingerprint=table_fingerprint,
        data_version=data_version,
    )
    assert cache.get_min_col_stats(
        table_fingerprint=table_fingerprint,
        stats_dict={
            base_col: [ColStatType.COUNT, ColStatType.NUM_UNIQUE],
        },
        data_version=data_version,
        approx=True,
    ) == sample_stats


@pytest.mark.asyncio
async def test_get_min_col_stats_cached_approx_requires_opt_in(
        make_default_table_def_v2):
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('cached-approx-version')
    base_col = table.to_base_columns()[1]
    sample_stats = {
        base_col: {
            ColStatType.COUNT:
            Stats(value=10, approx=True, data_version=data_version),
            ColStatType.NUM_UNIQUE:
            Stats(value=3, approx=True, data_version=data_version),
        },
    }

    cache = get_active_cache()
    cache.add_col_stats(table.fingerprint(), sample_stats)
    await cache.save_col_stats_cache.coroutine(table.fingerprint(),
                                               data_version)

    connector = mock.Mock()
    connector.name = _CONNECTOR_ID
    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector) as get_connector_from_spec:
        exact_only = await DataBackend.get_min_col_stats(
            ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock()),
            table,
            {
                base_col: [ColStatType.COUNT, ColStatType.NUM_UNIQUE],
            },
            approx_sample_size=10,
            check_refresh_status=False,
            compute_missing_stats=False,
            table_data_version=data_version,
        )
        approx_allowed = await DataBackend.get_min_col_stats(
            ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock()),
            table,
            {
                base_col: [ColStatType.COUNT, ColStatType.NUM_UNIQUE],
            },
            approx_sample_size=10,
            check_refresh_status=False,
            compute_missing_stats=False,
            allow_cached_approx_stats=True,
            table_data_version=data_version,
        )

    get_connector_from_spec.assert_not_called()
    assert exact_only == {}
    assert approx_allowed[base_col][ColStatType.COUNT].approx is True
    assert approx_allowed[base_col][ColStatType.COUNT].value == 10
    assert approx_allowed[base_col][ColStatType.NUM_UNIQUE].value == 3


@pytest.mark.asyncio
async def test_min_col_stats_cache_warmup_runs_in_background_and_dedupes(
        make_default_table_def_v2):
    data_backend_module._BACKGROUND_SAMPLE_COL_STATS_TASKS.clear()
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('background-sample-version')
    base_col = table.to_base_columns()[1]
    stats_request = {
        base_col: [ColStatType.COUNT, ColStatType.NUM_UNIQUE],
    }
    sample_stats = {
        base_col: {
            ColStatType.COUNT:
            Stats(value=10, approx=True, data_version=data_version),
            ColStatType.NUM_UNIQUE:
            Stats(value=3, approx=True, data_version=data_version),
        },
    }

    connector = mock.Mock()
    connector.compute_col_stats.return_value = sample_stats
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())

    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector):
        assert DataBackend.start_min_col_stats_cache_warmup(
            connector_spec,
            table,
            stats_request,
            approx_sample_size=10,
            table_data_version=data_version,
        )
        assert not DataBackend.start_min_col_stats_cache_warmup(
            connector_spec,
            table,
            stats_request,
            approx_sample_size=10,
            table_data_version=data_version,
        )
        tasks = list(
            data_backend_module._BACKGROUND_SAMPLE_COL_STATS_TASKS.values())
        assert len(tasks) == 1
        await asyncio.gather(*tasks)

    connector.compute_col_stats.assert_called_once()
    assert data_backend_module._BACKGROUND_SAMPLE_COL_STATS_TASKS == {}

    cache = get_active_cache()
    table_fingerprint = table.fingerprint()
    assert cache.load_min_col_stats_cache(
        connector_name=table_fingerprint.connector_name,
        table_fingerprint=table_fingerprint,
        data_version=data_version,
    )
    assert cache.get_min_col_stats(
        table_fingerprint=table_fingerprint,
        stats_dict=stats_request,
        data_version=data_version,
        approx=True,
    ) == sample_stats


@pytest.mark.asyncio
async def test_min_col_stats_cache_warmup_respects_concurrency_cap(
        make_default_table_def_v2, monkeypatch):
    data_backend_module._BACKGROUND_SAMPLE_COL_STATS_TASKS.clear()
    monkeypatch.setattr(
        data_backend_module,
        "_sample_col_stats_warmup_concurrency_limit",
        lambda: 1,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_get_min_col_stats(*_args, **_kwargs):
        started.set()
        await release.wait()
        return {}

    monkeypatch.setattr(DataBackend, "get_min_col_stats",
                        blocked_get_min_col_stats)

    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    base_col = table.to_base_columns()[1]
    stats_request = {
        base_col: [ColStatType.COUNT],
    }
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())

    assert DataBackend.start_min_col_stats_cache_warmup(
        connector_spec,
        table,
        stats_request,
        approx_sample_size=10,
        table_data_version=TableDataVersion('warmup-version-1'),
    )
    await started.wait()

    assert not DataBackend.start_min_col_stats_cache_warmup(
        connector_spec,
        table,
        stats_request,
        approx_sample_size=10,
        table_data_version=TableDataVersion('warmup-version-2'),
    )

    release.set()
    tasks = list(
        data_backend_module._BACKGROUND_SAMPLE_COL_STATS_TASKS.values())
    await asyncio.gather(*tasks)
    assert data_backend_module._BACKGROUND_SAMPLE_COL_STATS_TASKS == {}


@pytest.mark.asyncio
async def test_standalone_col_stats_skips_direct_file_connector(
        make_default_table_def_v2):
    from kumo.connector.file_connector import FileConnector

    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('test-version')
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())
    connector = mock.Mock(spec=FileConnector)
    connector.uses_direct_diskgraph_source_for.return_value = True

    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector), mock.patch(
                'rest.backends.data_backend.data_backend._list_parquet_files',
                side_effect=AssertionError('local parquet should not be read')):
        assert not await DataBackend.compute_standalone_table_col_stats(
            connector_spec, table, data_version)

    connector.uses_direct_diskgraph_source_for.assert_called_once_with(
        table.get_source_table_name())
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_standalone_col_stats_uses_frozen_direct_decision_without_connector(
        make_default_table_def_v2):
    from kumo.data.backend_cache import DirectDiskgraphSourceDecision

    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('test-version')
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())
    cache = mock.Mock()
    cache.direct_diskgraph_source_decision.get.coroutine = mock.AsyncMock(
        return_value=DirectDiskgraphSourceDecision(
            source={'type': 'object_store_parquet'}, ))

    with mock.patch(
            'rest.backends.data_backend.data_backend.get_active_cache',
            return_value=cache), mock.patch(
                'rest.backends.data_backend.data_backend.'
                'get_connector_from_spec_async', side_effect=AssertionError(
                    'frozen direct source must not resolve the connector')):
        assert not await DataBackend.compute_standalone_table_col_stats(
            connector_spec, table, data_version)

    cache.direct_diskgraph_source_decision.get.coroutine.assert_awaited_once_with(
        table.fingerprint(),
        str(data_version),
        cls=DirectDiskgraphSourceDecision,
    )


@pytest.mark.asyncio
async def test_standalone_col_stats_preserves_frozen_staged_decision(
        make_default_table_def_v2):
    from kumo.connector.file_connector import FileConnector
    from kumo.data.backend_cache import DirectDiskgraphSourceDecision

    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('test-version')
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())
    connector = mock.Mock(spec=FileConnector)
    connector.uses_direct_diskgraph_source_for.return_value = True
    cache = mock.Mock()
    cache.direct_diskgraph_source_decision.get.coroutine = mock.AsyncMock(
        return_value=DirectDiskgraphSourceDecision(source=None))

    with mock.patch('rest.backends.data_backend.data_backend.get_active_cache',
                    return_value=cache), mock.patch(
                        'rest.backends.data_backend.data_backend.'
                        'get_connector_from_spec',
                        return_value=connector), mock.patch(
                            'rest.backends.data_backend.data_backend.'
                            '_list_parquet_files',
                            return_value=[]), pytest.raises(
                                RuntimeError,
                                match='could not find ingested parquet'):
        await DataBackend.compute_standalone_table_col_stats(
            connector_spec, table, data_version)

    connector.uses_direct_diskgraph_source_for.assert_not_called()
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize('is_sql', [True, False], ids=['custom_sql', 'table'])
async def test_standalone_col_stats_scans_databricks_parquet_either_way(
        is_sql):
    """Both a Databricks table and a Databricks query stage to Kumo parquet
    now, so col stats are computed off that parquet in both cases."""
    from kumoapi.typing import Dtype, Stype

    from kumo.connector import DatabricksConnector
    from rest.config.table_v2 import (
        Column,
        CustomSQLSourceTable,
        DatabricksSourceTable,
    )

    connector_id = ConnectorID('connector_' + 'd' * 32)
    source_table = (CustomSQLSourceTable(
        connector_id=connector_id, connector_type='databricks',
        sql='SELECT item_key FROM sales', table='custom_sql_abc',
        referenced_tables=['sales']) if is_sql else DatabricksSourceTable(
            databricks_connector_id=connector_id, table='orders'))
    table = TableDefinition(
        cols=[Column(name='item_key', stype=Stype.ID, dtype=Dtype.string)],
        source_table=source_table, pkey='item_key', time_col=None)
    data_version = TableDataVersion('test-version')
    connector_spec = ConnectorSpec(connector_id, connector_config=mock.Mock())
    connector = mock.Mock(spec=DatabricksConnector)

    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector), mock.patch(
                'rest.backends.data_backend.data_backend._list_parquet_files',
                return_value=[]) as list_parquet:
        with pytest.raises(RuntimeError,
                           match='could not find ingested parquet'):
            await DataBackend.compute_standalone_table_col_stats(
                connector_spec, table, data_version)
        assert list_parquet.called

    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_direct_databricks_bp_stats_sample_stage_without_internal_copy(
        tmp_path):
    import pandas as pd
    from kumo_core.types import PredictionArtifactType
    from kumoapi.typing import Dtype, Stype

    from kumo.config import BaseColumnConfig
    from kumo.connector import DatabricksConnector
    from rest.config.artifact_export import DatabricksManagedVolumeStage
    from rest.config.prediction_output import DatabricksPredictionOutput

    connector_id = ConnectorID('connector_' + '2' * 32)
    job_input = mock.Mock()
    job_input.id = 'bp-job-direct'
    job_input.graph_version_info.timestamp_precision = 'seconds'
    job_input.config = mock.Mock(
        outputs=[
            DatabricksPredictionOutput(
                artifact_type=PredictionArtifactType.PREDICTIONS,
                connector_id=str(connector_id),
                table_name='schema__kumo__predictions',
            )
        ],
        task_type=mock.sentinel.task_type,
    )
    job_input.output_connectors = {
        connector_id: mock.sentinel.connector_reference
    }
    job_input.predictive_query.get_bp_entity_col_name.return_value = 'ENTITY'
    job_input.predictive_query.query_type = mock.sentinel.query_type

    score = BaseColumnConfig(
        name='SCORE',
        stype=Stype.numerical,
        dtype=Dtype.float64,
    )
    stats_set = mock.Mock()
    stats_set.get_stats_types.return_value = {score: [ColStatType.COUNT]}
    table_def = mock.Mock()
    table_def.fingerprint.return_value = mock.sentinel.fingerprint
    table_def.get_connector_id.return_value = 'internal-connector'
    table_def.cache_key.return_value = 'internal-cache-key'
    bp_data_version = TableDataVersion(str(job_input.id))

    connector = DatabricksConnector(
        'databricks',
        host='https://example',
        cluster_id='cluster',
        warehouse_id='warehouse',
        catalog='catalog',
        schema_name='schema',
    )
    connector.open = mock.Mock(return_value=connector)
    connector.close = mock.Mock()
    read_sample = mock.Mock()
    read_sample.coroutine = mock.AsyncMock(
        return_value=pd.DataFrame({'SCORE': [0.25, 0.75]}))
    connector.read_managed_volume_parquet_sample = read_sample

    cache_stats_path = tmp_path / 'cache-col-stats'
    cache_stats_path.write_text('{}')
    save_col_stats = mock.Mock()
    save_col_stats.coroutine = mock.AsyncMock(
        return_value=str(cache_stats_path))
    cache = mock.Mock(save_col_stats_cache=save_col_stats, )
    stage = DatabricksManagedVolumeStage(
        path=('/Volumes/catalog/schema/_kumo_staging/'
              'artifact-export/direct-job'),
        files=[
            '/Volumes/catalog/schema/_kumo_staging/'
            'artifact-export/direct-job/part.parquet'
        ],
    )
    job_home = tmp_path / 'job'

    with mock.patch(
            'rest.backends.ml_backend.data_util.get_job_base_dir',
            return_value=str(tmp_path)), mock.patch(
                'rest.backends.ml_backend.data_util.get_job_dir',
                return_value=str(job_home)), mock.patch(
                    'rest.utils.driver_common.generate_bp_stats_workflow_args',
                    new=mock.AsyncMock(return_value=(
                        mock.Mock(connector_id='internal-connector'),
                        table_def,
                        bp_data_version,
                        stats_set,
                    ))), mock.patch(
                        'rest.backends.data_backend.connector_broker.'
                        'connector_credential_broker.resolve_async',
                        new=mock.AsyncMock(
                            return_value=mock.sentinel.resolved_connector)
                    ), mock.patch(
                        'rest.backends.data_backend.data_backend.'
                        'get_connector_from_spec',
                        return_value=connector), mock.patch(
                            'rest.backends.data_backend.data_backend.'
                            'get_active_cache', return_value=cache), mock.patch(
                                'rest.backends.data_backend.data_backend.'
                                '_stage_remote_parquet_local',
                                side_effect=AssertionError(
                                    'direct stats must not copy the artifact')):
        await DataBackend.start_bp_stats_activity(
            mock.Mock(),
            job_input,
            False,
            0,
            stage,
        )

    read_sample.coroutine.assert_awaited_once_with(
        stage.path,
        [score],
        num_rows=10_000,
    )
    cache.add_col_stats.assert_called_once()
    assert (job_home / 'outputs' / 'col_stats').read_text() == '{}'
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_standalone_col_stats_local_connector_requires_parquet(
        make_default_table_def_v2):
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('test-version')
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())
    connector = mock.Mock()

    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector), mock.patch(
                'rest.backends.data_backend.data_backend._list_parquet_files',
                return_value=[]):
        with pytest.raises(RuntimeError,
                           match='could not find ingested parquet'):
            await DataBackend.compute_standalone_table_col_stats(
                connector_spec, table, data_version)
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_standalone_col_stats_writes_diskgraph_wheel_outputs(
        make_default_table_def_v2):
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('test-version')
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())
    connector = mock.Mock()

    parquet_paths = ['diskgraph-ingest/part-000.parquet']
    full_json = '{"format":"kumo_col_stats_v1","scope":"full"}'
    min_json = '{"format":"kumo_col_stats_v1","scope":"min"}'
    compute_col_stats = mock.Mock()
    compute_col_stats.coroutine = mock.AsyncMock(return_value=(full_json,
                                                               min_json))
    service = mock.Mock(compute_col_stats_for_parquet=compute_col_stats)
    cache = mock.Mock()
    cache.direct_diskgraph_source_decision.get.coroutine = mock.AsyncMock(
        return_value=None)
    table_dir = mock.Mock(col_stats='/cache/table/col_stats',
                          min_col_stats='/cache/table/min_col_stats')
    cache.create_table_cache_dir.return_value = table_dir
    kumo_column_specs = [{'name': 'col_a', 'stype': 'ID', 'dtype': 'string'}]

    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector), mock.patch(
                'kumo.data.utils.get_materialize_path',
                return_value='materialized-ingest'), mock.patch(
                    'kumo.data.utils.get_diskgraph_local_ingest_path',
                    return_value='diskgraph-ingest'), mock.patch(
                        'rest.backends.data_backend.data_backend.'
                        '_list_parquet_files',
                        side_effect=[[], parquet_paths]), mock.patch(
                            'kumo.diskgraph.get_service',
                            return_value=service), mock.patch(
                                'kumo.diskgraph.service.'
                                '_to_kumo_col_specs',
                                return_value=kumo_column_specs), mock.patch(
                                    'rest.backends.data_backend.data_backend.'
                                    'get_active_cache',
                                    return_value=cache), mock.patch(
                                        'rest.backends.data_backend.'
                                        'data_backend.fsutil.write_text'
                                    ) as write_text:
        assert await DataBackend.compute_standalone_table_col_stats(
            connector_spec, table, data_version)

    compute_col_stats.coroutine.assert_awaited_once_with(
        parquet_paths=parquet_paths,
        kumo_column_specs=kumo_column_specs,
        data_version=str(data_version),
    )
    cache.create_table_cache_dir.assert_called_once_with(
        str(table.get_connector_id()),
        table.get_source_table_name(),
        str(data_version),
        table.fingerprint(),
    )
    write_text.assert_has_calls([
        mock.call(table_dir.col_stats, full_json),
        mock.call(table_dir.min_col_stats, min_json),
    ])
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_standalone_col_stats_rejects_wide_integer_without_writing_stats(
        make_default_table_def_v2):
    from common_utils.error import PublicException

    class FakeDiskgraphSourceValidationError(Exception):
        pass

    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('test-version')
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())
    connector = mock.Mock()

    parquet_paths = ['diskgraph-ingest/part-000.parquet']
    diskgraph_error = FakeDiskgraphSourceValidationError(
        "column 'wide_int' contains a scale-zero decimal value outside "
        "Kumo's signed 64-bit integer range; cast the column to DOUBLE or "
        "STRING in Custom SQL, or keep its values within the signed 64-bit "
        "range")
    compute_col_stats = mock.Mock()
    compute_col_stats.coroutine = mock.AsyncMock(side_effect=diskgraph_error)
    service = mock.Mock(compute_col_stats_for_parquet=compute_col_stats)
    cache = mock.Mock()
    cache.direct_diskgraph_source_decision.get.coroutine = mock.AsyncMock(
        return_value=None)
    kumo_column_specs = [{
        'name': 'wide_int',
        'stype': 'numerical',
        'dtype': 'int'
    }]

    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector), mock.patch(
                'kumo.data.utils.get_materialize_path',
                return_value='materialized-ingest'), mock.patch(
                    'kumo.data.utils.get_diskgraph_local_ingest_path',
                    return_value='diskgraph-ingest'), mock.patch(
                        'rest.backends.data_backend.data_backend.'
                        '_list_parquet_files',
                        side_effect=[[], parquet_paths]), mock.patch(
                            'diskgraph.DiskgraphSourceValidationError',
                            FakeDiskgraphSourceValidationError,
                            create=True), mock.patch(
                                'kumo.diskgraph.get_service',
                                return_value=service), mock.patch(
                                    'kumo.diskgraph.service.'
                                    '_to_kumo_col_specs',
                                    return_value=kumo_column_specs), mock.patch(
                                        'rest.backends.data_backend.'
                                        'data_backend.get_active_cache',
                                        return_value=cache), mock.patch(
                                            'rest.backends.data_backend.'
                                            'data_backend.fsutil.write_text'
                                        ) as write_text:
        with pytest.raises(PublicException) as exc_info:
            await DataBackend.compute_standalone_table_col_stats(
                connector_spec, table, data_version)

    error = exc_info.value
    assert error.non_retryable
    assert f"Table '{table.get_source_table_name()}'" in error.message
    assert "column 'wide_int'" in error.message
    assert "signed 64-bit integer range" in error.message
    assert "DOUBLE or STRING in Custom SQL" in error.message
    compute_col_stats.coroutine.assert_awaited_once_with(
        parquet_paths=parquet_paths,
        kumo_column_specs=kumo_column_specs,
        data_version=str(data_version),
    )
    cache.create_table_cache_dir.assert_not_called()
    write_text.assert_not_called()
    connector.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_standalone_col_stats_stages_remote_materialize_path_for_wheel(
        make_default_table_def_v2):
    table = TableDefinition(**dataclasses.asdict(make_default_table_def_v2()))
    data_version = TableDataVersion('test-version')
    connector_spec = ConnectorSpec(_CONNECTOR_ID, connector_config=mock.Mock())
    connector = mock.Mock()

    remote_source = 's3://bucket/materialized-ingest'
    remote_paths = ['s3://bucket/materialized-ingest/part-000.parquet']
    staged_source = '/local/table_colstats/source.parquet'
    staged_paths = ['/local/table_colstats/source.parquet/part-000.parquet']
    full_json = '{"format":"kumo_col_stats_v1","scope":"full"}'
    min_json = '{"format":"kumo_col_stats_v1","scope":"min"}'
    compute_col_stats = mock.Mock()
    compute_col_stats.coroutine = mock.AsyncMock(return_value=(full_json,
                                                               min_json))
    service = mock.Mock(compute_col_stats_for_parquet=compute_col_stats)
    cache = mock.Mock()
    cache.direct_diskgraph_source_decision.get.coroutine = mock.AsyncMock(
        return_value=None)
    table_dir = mock.Mock(col_stats='/cache/table/col_stats',
                          min_col_stats='/cache/table/min_col_stats')
    cache.create_table_cache_dir.return_value = table_dir
    kumo_column_specs = [{'name': 'col_a', 'stype': 'ID', 'dtype': 'string'}]

    with mock.patch(
            'rest.backends.data_backend.data_backend.get_connector_from_spec',
            return_value=connector), mock.patch(
                'kumo.data.utils.get_materialize_path',
                return_value=remote_source), mock.patch(
                    'rest.backends.data_backend.data_backend.fsutil.is_local',
                    return_value=False), mock.patch(
                        'rest.backends.data_backend.data_backend.'
                        '_list_parquet_files',
                        side_effect=[remote_paths, staged_paths]), mock.patch(
                            'rest.backends.data_backend.data_backend.'
                            '_stage_remote_table_parquet_local',
                            new=mock.AsyncMock(
                                return_value=staged_source)) as stage_remote, \
            mock.patch('kumo.diskgraph.get_service',
                       return_value=service), mock.patch(
                           'kumo.diskgraph.service._to_kumo_col_specs',
                           return_value=kumo_column_specs), mock.patch(
                               'rest.backends.data_backend.data_backend.'
                               'get_active_cache',
                               return_value=cache), mock.patch(
                                   'rest.backends.data_backend.data_backend.'
                                   'fsutil.write_text'
                               ) as write_text, mock.patch(
                                   'rest.backends.data_backend.data_backend.'
                                   'shutil.rmtree') as rmtree:
        assert await DataBackend.compute_standalone_table_col_stats(
            connector_spec, table, data_version)

    stage_remote.assert_awaited_once_with(remote_source, table,
                                          str(data_version))
    compute_col_stats.coroutine.assert_awaited_once_with(
        parquet_paths=staged_paths,
        kumo_column_specs=kumo_column_specs,
        data_version=str(data_version),
    )
    rmtree.assert_called_once_with('/local/table_colstats', ignore_errors=True)
    write_text.assert_has_calls([
        mock.call(table_dir.col_stats, full_json),
        mock.call(table_dir.min_col_stats, min_json),
    ])
    connector.close.assert_called_once_with()
