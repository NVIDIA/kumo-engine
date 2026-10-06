# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import timedelta
from types import SimpleNamespace

import pytest
from kumo_core.types.schema import source_name_path_token
from temporalio.exceptions import ActivityError, ApplicationError

import statedb.config.diskgraph as config_mod
from kumo.data.utils import (
    get_diskgraph_snapshot_ingest_dir,
    get_diskgraph_snapshot_ingest_path,
)
from kumo.diskgraph.ingest import (
    _infer_connector_id_from_ingest_dir,
    normalize_diskgraph_source_fields,
)
from pool.dataclasses import PlacementResult, PlacementStatus
from pool.diskgraph_args import DiskgraphIngestArgs
from pool.errors import NO_SPACE_ERROR_TYPE, DiskgraphCapacityError
from rest.backends import (
    DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT_HOURS,
    DATA_BACKEND_INGEST_TIMEOUT_HOURS,
)
from rest.backends.data_backend.activities import diskgraph_ingest_activity
from service_framework._raw_deploy_config import DiskgraphConfig
from statedb.config.diskgraph import get_diskgraph_index_path_for_job


def _mock_deploy_env_with_diskgraph(data_root, local_ingest_root=None):
    """Return a mock deploy_env() whose data_compute is DiskgraphConfig."""
    cfg = type(
        'FakeDeployEnv', (), {
            'data_compute':
            DiskgraphConfig(data_root=data_root,
                            local_ingest_root=local_ingest_root),
            'is_diskgraph':
            True,
        })()
    return cfg


@pytest.mark.parametrize(
    "data_root",
    ["/tmp/dgroot", "/tmp/diskgraph-ingest"],
)
def test_get_diskgraph_index_path_root_resolution(monkeypatch, data_root):
    # get_diskgraph_index_path_for_job normalizes graph_key via
    # _graph_key_to_v2 so the consumer-side path matches the writer-side
    # path that pool/activities.py:_ensure_snapshot_local writes via
    # GraphIDv2.canonicalize. A bare display name like "g" gets mapped
    # to "NA-<md5(g)>".
    from statedb.config._graph_key_norm import _graph_key_to_v2

    mock_cfg = _mock_deploy_env_with_diskgraph(data_root)
    monkeypatch.setattr(config_mod, "deploy_env", lambda: mock_cfg)
    expected = f'{data_root}/{_graph_key_to_v2("g")}/v'
    job_input = SimpleNamespace(
        graph_data_compute_input=SimpleNamespace(graph_key="g"),
        graph_version_info=SimpleNamespace(graph_data_version="v"),
    )
    assert get_diskgraph_index_path_for_job(job_input) == expected


def test_normalize_diskgraph_source_fields_from_graphconfig_names():
    cfg = {
        'tables': [{
            'table_name': 'LOAN',
            'data_source': 'financial_local_parquet_123',
            'source_name': 'LOAN',
        }],
    }

    norm = normalize_diskgraph_source_fields(cfg, '/tmp/unused', {})
    t = norm['tables'][0]

    assert t['connector_id'] == 'financial_local_parquet_123'
    assert t['source_table_name'] == 'LOAN'
    assert t['source_table']['connector_id'] == 'financial_local_parquet_123'
    assert t['source_table']['source_table_name'] == 'LOAN'


def test_normalize_diskgraph_source_fields_preserves_explicit_top_level():
    cfg = {
        'tables': [{
            'table_name': 'LOAN',
            'connector_id': 'explicit_connector',
            'source_table_name': 'EXPLICIT_SOURCE',
            'data_source': 'fallback_connector',
            'source_name': 'FALLBACK_SOURCE',
        }],
    }

    norm = normalize_diskgraph_source_fields(cfg, '/tmp/unused', {})
    t = norm['tables'][0]

    assert t['connector_id'] == 'explicit_connector'
    assert t['source_table_name'] == 'EXPLICIT_SOURCE'
    assert t['source_table']['connector_id'] == 'explicit_connector'
    assert t['source_table']['source_table_name'] == 'EXPLICIT_SOURCE'


def test_infer_connector_id_from_ingest_dir(tmp_path):
    for schema in ('schema_aaa', 'schema_bbb'):
        ingest_dir = (tmp_path / 'ingest' /
                      ('fpv3.financial_local_parquet_777.'
                       f'{source_name_path_token("LOAN")}.{schema}') /
                      '1659426523')
        ingest_dir.mkdir(parents=True)

    inferred = _infer_connector_id_from_ingest_dir(
        str(tmp_path),
        'LOAN',
        '1659426523',
    )

    assert inferred == 'financial_local_parquet_777'


def test_normalize_diskgraph_source_fields_uses_ingest_dir_fallback(tmp_path):
    ingest_dir = (tmp_path / 'ingest' / ('fpv3.financial_local_parquet_888.'
                                         f'{source_name_path_token("LOAN")}') /
                  '1659426523')
    ingest_dir.mkdir(parents=True)

    cfg = {
        'tables': [{
            'table_name': 'LOAN',
            'connector_id': 'default',
            'source_table_name': 'LOAN',
        }],
    }

    norm = normalize_diskgraph_source_fields(
        cfg,
        str(tmp_path),
        {'LOAN': '1659426523'},
    )
    t = norm['tables'][0]

    assert t['connector_id'] == 'financial_local_parquet_888'
    assert t['source_table']['connector_id'] == 'financial_local_parquet_888'


def test_normalize_diskgraph_source_fields_coerces_graphconfig_object_fields():
    cfg = {
        'tables': [{
            'table_name':
            'LOAN',
            'pkey': {
                'name': 'LOAN_ID'
            },
            'time_col': {
                'name': 'DATE'
            },
            'end_time_col': {
                'name': 'END_DATE'
            },
            'fkeys': [{
                'name': 'ACCOUNT_ID',
                'dst_table': 'financial_local_parquet_1.ACCOUNT',
            }],
        }],
    }

    norm = normalize_diskgraph_source_fields(cfg, '/tmp/unused', {})
    t = norm['tables'][0]

    # pkey/time_col/end_time_col remain structured GraphConfig objects.
    assert t['pkey'] == {'name': 'LOAN_ID'}
    assert t['time_col'] == {'name': 'DATE'}
    assert t['end_time_col'] == {'name': 'END_DATE'}
    assert t['fkeys'][0]['dst_table'] == 'ACCOUNT'


# ---------------------------------------------------------------------------
# get_diskgraph_snapshot_ingest_dir / _path
# ---------------------------------------------------------------------------


def test_get_diskgraph_snapshot_ingest_dir_includes_snapshot_id(monkeypatch):
    mock_cfg = _mock_deploy_env_with_diskgraph(data_root='/tmp/dgroot')
    monkeypatch.setattr(config_mod, 'deploy_env', lambda: mock_cfg)

    path = get_diskgraph_snapshot_ingest_dir('NA-abc123@v1')
    assert path == '/tmp/dgroot/local-ingest/ingest/NA-abc123@v1'


def test_get_diskgraph_snapshot_ingest_dir_uses_configured_root(monkeypatch):
    mock_cfg = _mock_deploy_env_with_diskgraph(
        data_root='/tmp/dgroot', local_ingest_root='/mnt/ebs/staging')
    monkeypatch.setattr(config_mod, 'deploy_env', lambda: mock_cfg)

    path = get_diskgraph_snapshot_ingest_dir('NA-abc123@v1')
    assert path == '/mnt/ebs/staging/ingest/NA-abc123@v1'


def test_get_diskgraph_snapshot_ingest_path_nests_under_dir(monkeypatch):
    mock_cfg = _mock_deploy_env_with_diskgraph(data_root='/tmp/dgroot')
    monkeypatch.setattr(config_mod, 'deploy_env', lambda: mock_cfg)

    class _FakeFp:
        ingest_path_fpid = 'fpv3.connector.LOAN'

    path = get_diskgraph_snapshot_ingest_path('NA-abc123@v1', _FakeFp(), 'tv1')
    assert path == (
        '/tmp/dgroot/local-ingest/ingest/NA-abc123@v1/fpv3.connector.LOAN/tv1')
    # And the per-snapshot dir is the prefix of the per-table path —
    # cleanup_source_files relies on that.
    assert path.startswith(get_diskgraph_snapshot_ingest_dir('NA-abc123@v1'))


# ---------------------------------------------------------------------------
# cleanup_source_files
# ---------------------------------------------------------------------------


def test_cleanup_source_files_noop_when_absent(monkeypatch, tmp_path):
    mock_cfg = _mock_deploy_env_with_diskgraph(data_root=str(tmp_path))
    monkeypatch.setattr(config_mod, 'deploy_env', lambda: mock_cfg)
    # Sanity: nothing on disk for this snapshot yet.
    assert not os.path.exists(get_diskgraph_snapshot_ingest_dir('snap-X'))
    # Should not raise.
    diskgraph_ingest_activity.cleanup_source_files('snap-X')


def test_cleanup_source_files_rmtrees_existing_dir(monkeypatch, tmp_path):
    mock_cfg = _mock_deploy_env_with_diskgraph(data_root=str(tmp_path))
    monkeypatch.setattr(config_mod, 'deploy_env', lambda: mock_cfg)
    snapshot_dir = get_diskgraph_snapshot_ingest_dir('snap-Y')
    os.makedirs(os.path.join(snapshot_dir, 'fp', 'tv1'))
    with open(os.path.join(snapshot_dir, 'fp', 'tv1', 'data.parquet'), 'w'):
        pass
    assert os.path.isdir(snapshot_dir)

    diskgraph_ingest_activity.cleanup_source_files('snap-Y')

    assert not os.path.exists(snapshot_dir)


def test_cleanup_source_files_asserts_when_path_is_file(monkeypatch, tmp_path):
    mock_cfg = _mock_deploy_env_with_diskgraph(data_root=str(tmp_path))
    monkeypatch.setattr(config_mod, 'deploy_env', lambda: mock_cfg)
    snapshot_dir = get_diskgraph_snapshot_ingest_dir('snap-Z')
    os.makedirs(os.path.dirname(snapshot_dir))
    # Place a regular file where the directory is expected — contract violation.
    with open(snapshot_dir, 'w'):
        pass

    with pytest.raises(AssertionError, match='expected directory'):
        diskgraph_ingest_activity.cleanup_source_files('snap-Z')


# ---------------------------------------------------------------------------
# DiskgraphIngestWorkflow.run — try/finally cleanup contract
# ---------------------------------------------------------------------------


def _setup_workflow_test(monkeypatch):
    """Run the workflow body without a Temporal worker."""
    from contextlib import asynccontextmanager

    from kumo_core.diskgraph.snapshot import DiskgraphIngestResult

    import rest.backends.data_backend.workflows.diskgraph_ingest_workflow as wf_mod
    from rest.backends.data_backend import diskgraph_stats_bridge

    cleanup_source_files = diskgraph_ingest_activity.cleanup_source_files
    diskgraph_ingest_stats_activity = (
        diskgraph_stats_bridge.diskgraph_ingest_stats_activity)

    @asynccontextmanager
    async def fake_session_ctx(*args, **kwargs):
        del args
        state['session_calls'].append(kwargs)
        tier_index = min(
            len(state['session_calls']) - 1,
            len(state['placement_tiers']) - 1,
        )
        yield PlacementResult(
            snapshot_id=kwargs['snapshot_id'],
            task_queue='pool-tq-test',
            status=PlacementStatus.MISSING,
            tier_ssd_gib=state['placement_tiers'][tier_index],
        )

    monkeypatch.setattr(wf_mod, 'snapshot_placement_session_ctx',
                        fake_session_ctx)

    # Tracker for execute_activity calls. Each entry is (fn, args).
    activity_calls = []
    # Optional override: if set, raise this when ensure_snapshot is dispatched.
    state = {
        'activity_options': [],
        'ensure_snapshot_raises': None,
        'ensure_snapshot_outcomes': [],
        'placement_tiers': [900],
        'session_calls': [],
    }

    async def fake_execute_activity(activity_fn, *, args=None, **kwargs):
        activity_calls.append((activity_fn, args))
        state['activity_options'].append((activity_fn, kwargs))
        # Identify ensure_snapshot by reference (it's a method on
        # SnapshotActivities) and stats / cleanup by their imported refs.
        if state['ensure_snapshot_raises'] is not None and (getattr(
                activity_fn, '__name__', '') == 'ensure_snapshot'):
            raise state['ensure_snapshot_raises']
        if getattr(activity_fn, '__name__', '') == 'ensure_snapshot':
            if state['ensure_snapshot_outcomes']:
                outcome = state['ensure_snapshot_outcomes'].pop(0)
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome
            return DiskgraphIngestResult(
                index_path='/data/index',
                col_stats_path='/data/index/col_stats.json',
                ingest_stats_path='/data/index/ingest_stats.json',
                kumo_stats_dir='/data/index/kumo_stats',
            )
        if activity_fn is diskgraph_ingest_stats_activity:
            return {'users': 2}
        return None

    monkeypatch.setattr(wf_mod.workflow, 'execute_activity',
                        fake_execute_activity)
    monkeypatch.setattr(
        wf_mod.workflow,
        'logger',
        SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    return wf_mod, activity_calls, state, cleanup_source_files, \
        diskgraph_ingest_stats_activity


@pytest.mark.asyncio
async def test_diskgraph_ingest_workflow_retries_cancelled_activity(
        monkeypatch):
    """Cancellation is retryable and is not excluded by the ingest policy."""
    wf_mod, _calls, state, _cleanup_fn, _stats_fn = (
        _setup_workflow_test(monkeypatch))

    await wf_mod.DiskgraphIngestWorkflow().run(
        graph_definition=SimpleNamespace(col_groups=[]),
        graph_display_name='graph',
        data_persist_dir='/tmp/persist',
        table_data_versions={},
        graph_key='NA-' + 'a' * 32,
        graph_data_version='v1',
        all_graph_tables={},
        connector_specs={},
        source_size_bytes=0,
        resource_hint=None,
        source_size_bytes_by_table={},
    )

    ensure_options = next(
        options for activity_fn, options in state['activity_options']
        if getattr(activity_fn, '__name__', '') == 'ensure_snapshot')
    assert ensure_options['schedule_to_close_timeout'] == timedelta(
        hours=wf_mod.DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT_HOURS)
    assert DATA_BACKEND_INGEST_TIMEOUT_HOURS > (
        DATA_BACKEND_DISKGRAPH_WORKFLOW_TIMEOUT_HOURS)
    retry_policy = ensure_options['retry_policy']
    assert retry_policy.maximum_attempts == 3
    assert 'DiskgraphCancelledError' not in (
        retry_policy.non_retryable_error_types or [])


def _activity_error_with_type(app_type: str) -> ActivityError:
    try:
        raise ApplicationError('activity failed', type=app_type)
    except ApplicationError as app_error:
        try:
            raise ActivityError(
                message='activity failed',
                scheduled_event_id=1,
                started_event_id=1,
                identity='test',
                activity_type='ensure_snapshot',
                activity_id='ensure',
                retry_state=None,
            ) from app_error
        except ActivityError as activity_error:
            return activity_error


@pytest.mark.asyncio
async def test_diskgraph_ingest_workflow_happy_path(monkeypatch):
    """Success: ensure_snapshot → cleanup_source_files → stats, in order."""
    (wf_mod, activity_calls, _state, cleanup_fn,
     stats_fn) = _setup_workflow_test(monkeypatch)

    mock_table = type(
        'MockTable', (), {
            'get_connector_id':
            lambda self: 'conn-1',
            'get_source_table_name':
            lambda self: 'users',
            'fingerprint':
            lambda self: type('FP', (), {'ingest_path_fpid': 'fpv3.conn.users'})
            (),
        })()
    wf = wf_mod.DiskgraphIngestWorkflow()
    result = await wf.run(
        graph_definition=SimpleNamespace(col_groups=[]),
        graph_display_name='graph',
        data_persist_dir='/tmp/persist',
        table_data_versions={'users': 'tv1'},
        graph_key='NA-' + 'a' * 32,
        graph_data_version='v1',
        all_graph_tables={'users': mock_table},
        connector_specs={},
        source_size_bytes=0,
        resource_hint=None,
        source_size_bytes_by_table={},
    )

    assert result.index_path == '/data/index'
    assert result.pkey_duplicate_counts == {'users': 2}
    # Order: ensure_snapshot, cleanup_source_files, stats.
    fns = [getattr(c[0], '__name__', None) for c in activity_calls]
    assert fns == [
        'ensure_snapshot',
        'cleanup_source_files',
        'diskgraph_ingest_stats_activity',
    ], fns
    ingest_args = activity_calls[0][1][1]
    assert isinstance(ingest_args, DiskgraphIngestArgs)
    assert not hasattr(ingest_args, 'graph_config')
    assert ingest_args.connector_specs == {}
    assert ingest_args.graph_display_name == 'graph'
    # cleanup_source_files dispatched with snapshot_id.
    cleanup_call = activity_calls[1]
    assert cleanup_call[0] is cleanup_fn
    # cleanup_source_files dispatched with the canonicalized snapshot_id.
    assert len(cleanup_call[1]) == 1
    assert ('a' * 32) in cleanup_call[1][0] and 'v1' in cleanup_call[1][0]


@pytest.mark.asyncio
async def test_diskgraph_ingest_workflow_passes_no_edges_to_stats_activity(
        monkeypatch):
    (wf_mod, activity_calls, _state, _cleanup_fn,
     stats_fn) = _setup_workflow_test(monkeypatch)

    mock_table = type(
        'MockTable', (), {
            'get_connector_id':
            lambda self: 'conn-1',
            'get_source_table_name':
            lambda self: 'users',
            'fingerprint':
            lambda self: type('FP', (), {'ingest_path_fpid': 'fpv3.conn.users'})
            (),
        })()
    wf = wf_mod.DiskgraphIngestWorkflow()

    await wf.run(
        graph_definition=SimpleNamespace(col_groups=[]),
        graph_display_name='graph',
        data_persist_dir='/tmp/persist',
        table_data_versions={'users': 'tv1'},
        graph_key='NA-' + 'a' * 32,
        graph_data_version='v1',
        all_graph_tables={'users': mock_table},
        connector_specs={},
        source_size_bytes=0,
        resource_hint=None,
        source_size_bytes_by_table={},
    )

    stats_call = activity_calls[2]
    assert stats_call[0] is stats_fn
    assert stats_call[1][-2] is False
    assert stats_call[1][-1] == '/data/index/ingest_stats.json'


@pytest.mark.asyncio
async def test_diskgraph_ingest_workflow_cleanup_runs_on_failure(monkeypatch):
    """ensure_snapshot raises → cleanup STILL runs → exception propagates →
    stats does NOT run."""
    (wf_mod, activity_calls, state, cleanup_fn,
     stats_fn) = _setup_workflow_test(monkeypatch)
    state['ensure_snapshot_raises'] = RuntimeError('rust ingest OOM')

    mock_table = type(
        'MockTable', (), {
            'get_connector_id':
            lambda self: 'conn-1',
            'get_source_table_name':
            lambda self: 'users',
            'fingerprint':
            lambda self: type('FP', (), {'ingest_path_fpid': 'fpv3.conn.users'})
            (),
        })()
    wf = wf_mod.DiskgraphIngestWorkflow()

    with pytest.raises(RuntimeError, match='rust ingest OOM'):
        await wf.run(
            graph_definition=SimpleNamespace(col_groups=[]),
            graph_display_name='graph',
            data_persist_dir='/tmp/persist',
            table_data_versions={'users': 'tv1'},
            graph_key='NA-' + 'a' * 32,
            graph_data_version='v1',
            all_graph_tables={'users': mock_table},
            connector_specs={},
            source_size_bytes=0,
            resource_hint=None,
            source_size_bytes_by_table={},
        )

    fns = [getattr(c[0], '__name__', None) for c in activity_calls]
    # ensure_snapshot dispatched and raised, cleanup ran, stats never reached.
    assert fns == ['ensure_snapshot', 'cleanup_source_files'], fns
    assert activity_calls[1][0] is cleanup_fn


@pytest.mark.asyncio
async def test_diskgraph_ingest_workflow_walks_strictly_larger_ssd_tiers(
        monkeypatch):
    wf_mod, activity_calls, state, _cleanup_fn, _stats_fn = (
        _setup_workflow_test(monkeypatch))
    state['placement_tiers'] = [500, 900, 1800]
    state['ensure_snapshot_outcomes'] = [
        _activity_error_with_type(NO_SPACE_ERROR_TYPE),
        _activity_error_with_type(NO_SPACE_ERROR_TYPE),
    ]

    result = await wf_mod.DiskgraphIngestWorkflow().run(
        graph_definition=SimpleNamespace(col_groups=[]),
        graph_display_name='graph',
        data_persist_dir='/tmp/persist',
        table_data_versions={},
        graph_key='NA-' + 'a' * 32,
        graph_data_version='v1',
        all_graph_tables={},
        connector_specs={},
        source_size_bytes=0,
        resource_hint=None,
        source_size_bytes_by_table={},
    )

    assert result.index_path == '/data/index'
    assert [
        call['resource_hint'].min_ssd_gib for call in state['session_calls']
    ] == [0, 501, 901]
    function_names = [
        getattr(call[0], '__name__', None) for call in activity_calls
    ]
    assert function_names.count('ensure_snapshot') == 3
    assert function_names.count('cleanup_source_files') == 3
    assert function_names.count('diskgraph_ingest_stats_activity') == 1


@pytest.mark.asyncio
async def test_diskgraph_ingest_workflow_does_not_re_place_non_capacity_error(
        monkeypatch):
    wf_mod, _activity_calls, state, _cleanup_fn, _stats_fn = (
        _setup_workflow_test(monkeypatch))
    state['ensure_snapshot_raises'] = _activity_error_with_type(
        'DatabricksAuthenticationError')

    with pytest.raises(ActivityError):
        await wf_mod.DiskgraphIngestWorkflow().run(
            graph_definition=SimpleNamespace(col_groups=[]),
            graph_display_name='graph',
            data_persist_dir='/tmp/persist',
            table_data_versions={},
            graph_key='NA-' + 'a' * 32,
            graph_data_version='v1',
            all_graph_tables={},
            connector_specs={},
            source_size_bytes=0,
            resource_hint=None,
            source_size_bytes_by_table={},
        )

    assert len(state['session_calls']) == 1


@pytest.mark.asyncio
async def test_diskgraph_ingest_workflow_fixed_host_no_space_fails_once(
        monkeypatch):
    wf_mod, _activity_calls, state, _cleanup_fn, _stats_fn = (
        _setup_workflow_test(monkeypatch))
    state['placement_tiers'] = [0]
    state['ensure_snapshot_raises'] = _activity_error_with_type(
        NO_SPACE_ERROR_TYPE)

    with pytest.raises(DiskgraphCapacityError, match='cannot automatically'):
        await wf_mod.DiskgraphIngestWorkflow().run(
            graph_definition=SimpleNamespace(col_groups=[]),
            graph_display_name='graph',
            data_persist_dir='/tmp/persist',
            table_data_versions={},
            graph_key='NA-' + 'a' * 32,
            graph_data_version='v1',
            all_graph_tables={},
            connector_specs={},
            source_size_bytes=0,
            resource_hint=None,
            source_size_bytes_by_table={},
        )

    assert len(state['session_calls']) == 1


# ---------------------------------------------------------------------------
# DiskgraphService.download_table_files — _DOWNLOAD_SUCCESS marker idempotency
# ---------------------------------------------------------------------------


def _setup_download_test(monkeypatch, tmp_path, *, connector_raises=False,
                         pre_existing_partial=False, pre_existing_marker=False):
    """Wire DiskgraphService + a fake connector for download_table_files tests.

    Returns ``(service, table, source_paths, downloads)`` where
    ``downloads`` is a list each successful ``connector.download`` call
    appends to (so tests can count actual download invocations).
    """
    from kumo.diskgraph.service import DiskgraphService

    cfg = DiskgraphConfig(data_root=str(tmp_path))
    service = DiskgraphService(cfg)

    # Fake table / fingerprint mimicking TableDefinition's method API.
    fp = type('FP', (), {'ingest_path_fpid': 'fpv3.conn.users'})()
    table = type(
        'Table', (), {
            'get_connector_id': lambda self: 'connector-1',
            'get_source_table_name': lambda self: 'users',
            'fingerprint': lambda self: fp,
        })()

    # Per-snapshot per-table local dest, mirroring the workflow's path
    # construction so the marker logic exercises the real layout.
    local_dest = os.path.join(str(tmp_path), 'ingest', 'snap-1',
                              'fpv3.conn.users', 'tv1')
    source_paths = {'users': local_dest}

    if pre_existing_marker:
        from kumo.diskgraph.service import (
            _download_inventory_payload,
            _write_download_inventory_marker,
        )

        os.makedirs(local_dest, exist_ok=True)
        with open(os.path.join(local_dest, 'data.parquet'), 'w') as f:
            f.write('previous-good-data')
        _write_download_inventory_marker(
            os.path.join(local_dest, '_DOWNLOAD_SUCCESS'),
            _download_inventory_payload('users', table, 'tv1', local_dest),
        )
    elif pre_existing_partial:
        os.makedirs(local_dest, exist_ok=True)
        with open(os.path.join(local_dest, 'partial.parquet'), 'w') as f:
            f.write('stale-partial-data')

    downloads: list = []

    class FakeConnector:
        def download(self, source_table_name, fingerprint, data_version, dest):
            if connector_raises:
                raise RuntimeError('connector boom')
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, 'data.parquet'), 'w') as f:
                f.write('fresh-data')
            downloads.append(dest)

    monkeypatch.setattr('rest.utils.driver_common.get_connector_from_spec',
                        lambda spec: FakeConnector())

    return service, table, source_paths, downloads, local_dest


def test_download_table_files_marker_present_skips(monkeypatch, tmp_path):
    service, table, source_paths, downloads, local_dest = _setup_download_test(
        monkeypatch, tmp_path, pre_existing_marker=True)

    service.download_table_files(
        connector_specs={'connector-1': object()},
        table_data_versions={'users': 'tv1'},
        all_graph_tables={'users': table},
        source_paths=source_paths,
    )

    assert downloads == [], 'connector.download should be skipped on marker hit'
    # Pre-existing data + marker untouched.
    with open(os.path.join(local_dest, 'data.parquet')) as f:
        assert f.read() == 'previous-good-data'
    assert os.path.exists(os.path.join(local_dest, '_DOWNLOAD_SUCCESS'))


def test_download_table_files_partial_dir_rmtreed_then_redownload(
        monkeypatch, tmp_path):
    service, table, source_paths, downloads, local_dest = _setup_download_test(
        monkeypatch, tmp_path, pre_existing_partial=True)

    service.download_table_files(
        connector_specs={'connector-1': object()},
        table_data_versions={'users': 'tv1'},
        all_graph_tables={'users': table},
        source_paths=source_paths,
    )

    assert len(downloads) == 1
    assert os.path.dirname(downloads[0]) == os.path.dirname(local_dest)
    assert os.path.basename(downloads[0]).startswith('.tv1.download-')
    # Stale partial file should be gone (rmtreed before re-download).
    assert not os.path.exists(os.path.join(local_dest, 'partial.parquet'))
    # Fresh download wrote new data + marker.
    with open(os.path.join(local_dest, 'data.parquet')) as f:
        assert f.read() == 'fresh-data'
    assert os.path.exists(os.path.join(local_dest, '_DOWNLOAD_SUCCESS'))


def test_download_table_files_writes_marker_on_success(monkeypatch, tmp_path):
    service, table, source_paths, downloads, local_dest = _setup_download_test(
        monkeypatch, tmp_path)

    service.download_table_files(
        connector_specs={'connector-1': object()},
        table_data_versions={'users': 'tv1'},
        all_graph_tables={'users': table},
        source_paths=source_paths,
    )

    assert len(downloads) == 1
    assert os.path.dirname(downloads[0]) == os.path.dirname(local_dest)
    assert os.path.basename(downloads[0]).startswith('.tv1.download-')
    assert os.path.exists(os.path.join(local_dest, '_DOWNLOAD_SUCCESS'))


def test_download_table_files_no_marker_on_connector_failure(
        monkeypatch, tmp_path):
    service, table, source_paths, downloads, local_dest = _setup_download_test(
        monkeypatch, tmp_path, connector_raises=True)

    with pytest.raises(RuntimeError, match='connector boom'):
        service.download_table_files(
            connector_specs={'connector-1': object()},
            table_data_versions={'users': 'tv1'},
            all_graph_tables={'users': table},
            source_paths=source_paths,
        )

    # Marker MUST NOT exist — that's the load-bearing safety property:
    # next attempt sees no marker → re-downloads from scratch.
    assert not os.path.exists(os.path.join(local_dest, '_DOWNLOAD_SUCCESS'))
