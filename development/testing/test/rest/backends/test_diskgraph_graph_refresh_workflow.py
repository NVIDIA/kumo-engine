# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest
from kumo_core.diskgraph.snapshot import (
    DiskgraphIngestResult,
    check_diskgraph_output,
    write_diskgraph_success_marker,
)

import statedb.config.diskgraph as config_mod
from kumo.data.backend_cache import TableRefreshStage
from kumo.validation import ValidationResponse, ValidationWarning
from pool.dataclasses import ResourceHint
from rest.backends.data_backend.activities import (
    DataBackendActivities,
    GraphTableStatsResult,
)
from rest.backends.data_backend.workflows import (
    diskgraph_ingest_workflow,
    graph_refresh_workflow,
)


def test_compose_wf_id():
    wf_id = diskgraph_ingest_workflow.DiskgraphIngestWorkflow.compose_wf_id(
        'NA-abc123', 'deadbeef')
    assert wf_id == 'dg-ingest:NA-abc123@ver=deadbeef'


def test_diskgraph_ingest_workflow_declares_ingest_result():
    assert (get_type_hints(
        diskgraph_ingest_workflow.DiskgraphIngestWorkflow.run)['return']
            is DiskgraphIngestResult)


# ---------------------------------------------------------------------------
# check_diskgraph_output
# ---------------------------------------------------------------------------


def test_check_diskgraph_output_no_success(tmp_path):
    """No _SUCCESS file → returns None."""
    assert check_diskgraph_output(str(tmp_path)) is None


def test_check_diskgraph_output_valid(tmp_path):
    """Valid _SUCCESS + all files exist → returns cached result."""
    index_path = str(tmp_path)
    col_stats = tmp_path / 'col_stats.json'
    ingest_stats = tmp_path / 'ingest_stats.json'
    kumo_stats = tmp_path / 'kumo_col_stats'
    col_stats.write_text('{}')
    ingest_stats.write_text('{}')
    kumo_stats.mkdir()

    result = DiskgraphIngestResult(
        index_path=index_path,
        col_stats_path=str(col_stats),
        ingest_stats_path=str(ingest_stats),
        kumo_stats_dir=str(kumo_stats),
    )
    # Write _SUCCESS marker
    write_diskgraph_success_marker(str(tmp_path), result)

    cached = check_diskgraph_output(str(tmp_path))
    assert cached is not None
    assert cached.index_path == index_path
    assert cached.col_stats_path == str(col_stats)


def test_check_diskgraph_output_missing_file(tmp_path):
    """_SUCCESS exists but a referenced file is missing → returns None."""
    index_path = str(tmp_path)
    col_stats = tmp_path / 'col_stats.json'
    ingest_stats = tmp_path / 'ingest_stats.json'
    kumo_stats = tmp_path / 'kumo_col_stats'
    col_stats.write_text('{}')
    # ingest_stats NOT created
    kumo_stats.mkdir()

    result = DiskgraphIngestResult(
        index_path=index_path,
        col_stats_path=str(col_stats),
        ingest_stats_path=str(ingest_stats),
        kumo_stats_dir=str(kumo_stats),
    )
    write_diskgraph_success_marker(str(tmp_path), result)

    assert check_diskgraph_output(str(tmp_path)) is None


def test_check_diskgraph_output_missing_dir(tmp_path):
    """_SUCCESS exists but kumo_stats_dir is missing → returns None."""
    index_path = str(tmp_path)
    col_stats = tmp_path / 'col_stats.json'
    ingest_stats = tmp_path / 'ingest_stats.json'
    col_stats.write_text('{}')
    ingest_stats.write_text('{}')
    # kumo_stats NOT created

    result = DiskgraphIngestResult(
        index_path=index_path,
        col_stats_path=str(col_stats),
        ingest_stats_path=str(ingest_stats),
        kumo_stats_dir=str(tmp_path / 'kumo_col_stats'),
    )
    write_diskgraph_success_marker(str(tmp_path), result)

    assert check_diskgraph_output(str(tmp_path)) is None


def test_check_diskgraph_output_corrupt_success(tmp_path):
    """_SUCCESS exists but contains invalid JSON → returns None."""
    success_path = tmp_path / '_SUCCESS'
    success_path.write_text('not json')
    assert check_diskgraph_output(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# write_diskgraph_success_marker
# ---------------------------------------------------------------------------


def test_write_diskgraph_success_marker_roundtrip(tmp_path):
    """Write and read back should produce identical result."""
    result = DiskgraphIngestResult(
        index_path='/data/index',
        col_stats_path='/data/index/col_stats.json',
        ingest_stats_path='/data/index/ingest_stats.json',
        kumo_stats_dir='/data/index/kumo_col_stats',
    )
    write_diskgraph_success_marker(str(tmp_path), result)

    success_path = tmp_path / '_SUCCESS'
    assert success_path.is_file()

    data = json.loads(success_path.read_text())
    assert data['index_path'] == '/data/index'
    assert data['col_stats_path'] == '/data/index/col_stats.json'


# ---------------------------------------------------------------------------
# GraphRefreshWorkflow._run_diskgraph_colstats_pipeline
# ---------------------------------------------------------------------------


def _setup_pipeline_test(monkeypatch):
    """Common setup for _run_diskgraph_colstats_pipeline tests."""
    import rest.backends.data_backend.workflows.graph_refresh_workflow as mod
    from service_framework._raw_deploy_config import DiskgraphConfig

    mock_cfg = SimpleNamespace(
        data_compute=DiskgraphConfig(data_root='/tmp/dg'),
        is_diskgraph=True,
    )
    monkeypatch.setattr(config_mod, "deploy_env", lambda: mock_cfg)

    # get_diskgraph_index_path_for_job reads DISKGRAPH_DATA_ROOT env var
    monkeypatch.setenv("DISKGRAPH_DATA_ROOT", "/tmp/dg")

    wf = cast(Any, graph_refresh_workflow.GraphRefreshWorkflow())
    wf._ctx = object()
    wf._resource_hint = None
    mock_fp = SimpleNamespace(ingest_path_fpid="fpv3.conn.users")
    wf._input = SimpleNamespace(
        graph_key="gkey",
        tables={
            "t1": SimpleNamespace(table_name="users",
                                  fingerprint=lambda: mock_fp),
        },
        base_tables={},
        table_aliases={"t1": "users"},
        connectors={},
        graph_definition=SimpleNamespace(col_groups=[]),
        graph_display_name='graph',
    )
    wf._graph_version_info = SimpleNamespace(
        graph_data_version="v1",
        table_data_versions={"users": "tv1"},
        timestamp_precision="seconds",
    )

    monkeypatch.setattr(mod, "get_persist_dir", lambda: "/tmp/persist")

    # Track any workflow activities dispatched by this helper.
    activity_calls = []

    async def fake_execute_activity(activity_fn, *, args, **kwargs):
        activity_calls.append({
            'activity_fn': activity_fn,
            'args': args,
        })
        return {}

    monkeypatch.setattr(mod.workflow, "execute_activity", fake_execute_activity)
    monkeypatch.setattr(mod.workflow, "patched", lambda _patch_id: False)

    # Track start_child_data_wf calls
    child_data_wf_calls = []
    state = {
        'child_task_awaited':
        False,
        'child_result':
        DiskgraphIngestResult(
            index_path='/index',
            col_stats_path='/index/col_stats.json',
            ingest_stats_path='/index/ingest_stats.json',
            kumo_stats_dir='/index/kumo_col_stats',
            pkey_duplicate_counts={'users': 2},
        ),
    }

    async def fake_child_task():
        state['child_task_awaited'] = True
        return state['child_result']

    async def fake_start_child_data_wf(workflow_fn, ctx, *, wf_id, args,
                                       execution_timeout, **kwargs):
        child_data_wf_calls.append({
            'workflow_fn': workflow_fn,
            'wf_id': wf_id,
            'args': args,
            'kwargs': kwargs,
        })
        return fake_child_task()

    monkeypatch.setattr(
        mod.workflow_utils,
        "start_child_data_wf",
        fake_start_child_data_wf,
    )

    # Mock _save_diskgraph_colstat_task_results
    save_calls = []

    async def fake_save(table_data_versions, _table_validations):
        save_calls.append(table_data_versions)

    wf._save_diskgraph_colstat_task_results = fake_save

    return wf, activity_calls, child_data_wf_calls, save_calls, state


@pytest.mark.asyncio
@pytest.mark.parametrize('patched', [False, True])
async def test_child_ingest_receives_the_snapshot_clock_once_patched(
        monkeypatch, patched):
    wf, _, child_data_wf_calls, _, _ = _setup_pipeline_test(monkeypatch)
    wf._graph_version_info.timestamp_precision = 'milliseconds'
    monkeypatch.setattr(
        graph_refresh_workflow.workflow, 'patched',
        lambda patch_id: patched and patch_id == 'diskgraph-index-clock-v1')

    await wf._run_diskgraph_colstats_pipeline(source_size_bytes_by_table={},
                                              source_size_bytes=0)

    child_args = inspect.signature(
        diskgraph_ingest_workflow.DiskgraphIngestWorkflow.run).bind(
            None, *child_data_wf_calls[0]['args']).arguments
    assert child_args.get('timestamp_precision') == ('milliseconds'
                                                     if patched else None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('patched', 'expected_wf_id', 'expected_duplicate_counts'),
    [
        (False, 'dg-ingest:gkey@ver=v1', None),
        (True, 'dg-ingest:gkey@ver=v1:stats-v2', {
            'users': 2
        }),
    ],
)
async def test_run_diskgraph_colstats_pipeline_versions_child_workflow_id(
        monkeypatch, patched, expected_wf_id, expected_duplicate_counts):
    """Pipeline should start DiskgraphIngestWorkflow via start_child_data_wf."""
    wf, _activity_calls, child_data_wf_calls, save_calls, state = \
        _setup_pipeline_test(monkeypatch)
    monkeypatch.setattr(
        graph_refresh_workflow.workflow, 'patched',
        lambda patch_id: patched and patch_id == 'diskgraph-ingest-stats-v2')

    duplicate_counts = await wf._run_diskgraph_colstats_pipeline(
        source_size_bytes_by_table={},
        source_size_bytes=0,
    )

    # Should have called start_child_data_wf once
    assert len(child_data_wf_calls) == 1
    call = child_data_wf_calls[0]
    assert (call['workflow_fn'] ==
            diskgraph_ingest_workflow.DiskgraphIngestWorkflow.run)
    assert call['wf_id'] == expected_wf_id

    # Args: graph_definition, display_name, persist_dir, table_versions,
    # graph_key,
    # graph_data_version, all_tables, connector_specs, source_size_bytes,
    # resource_hint, source_size_bytes_by_table
    args = call['args']
    assert len(args) == 11
    assert args[0] is wf._input.graph_definition
    assert args[1] == 'graph'
    assert args[2] == '/tmp/persist'  # data_persist_dir
    assert args[3] == {'users': 'tv1'}  # table_data_versions
    assert args[4] == 'gkey'  # graph_key
    assert args[5] == 'v1'  # graph_data_version
    # DiskgraphIngestWorkflow passes connector_specs through placement so
    # daemon ensure_snapshot can download source files on the PVC writer.
    assert args[7] == {}  # connector_specs (empty in this fixture)
    assert args[8] == 0
    assert args[9] is None
    assert args[10] == {}

    # Should have awaited the child task
    assert state['child_task_awaited']
    assert duplicate_counts == expected_duplicate_counts

    # The parent refresh workflow validates and saves col-stat task results
    # after this helper returns.
    assert save_calls == []


@pytest.mark.asyncio
async def test_run_diskgraph_colstats_pipeline_normalizes_reused_child_result(
        monkeypatch):
    wf, _activity_calls, _child_calls, _save_calls, state = \
        _setup_pipeline_test(monkeypatch)
    monkeypatch.setattr(
        graph_refresh_workflow.workflow,
        'patched',
        lambda _patch_id: True,
    )
    state['child_result'] = {
        'index_path': '/index',
        'col_stats_path': '/index/col_stats.json',
        'ingest_stats_path': '/index/ingest_stats.json',
        'kumo_stats_dir': '/index/kumo_col_stats',
        'pkey_duplicate_counts': {
            'users': 2
        },
    }

    duplicate_counts = await wf._run_diskgraph_colstats_pipeline(
        source_size_bytes_by_table={},
        source_size_bytes=0,
    )

    assert duplicate_counts == {'users': 2}


@pytest.mark.asyncio
async def test_run_diskgraph_colstats_pipeline_forwards_resource_hint(
        monkeypatch):
    wf, _activity_calls, child_data_wf_calls, _save_calls, _state = \
        _setup_pipeline_test(monkeypatch)
    wf._resource_hint = ResourceHint(min_gpu_slots=4)

    await wf._run_diskgraph_colstats_pipeline(
        source_size_bytes_by_table={},
        source_size_bytes=0,
    )

    assert child_data_wf_calls[0]['args'][9].min_gpu_slots == 4


@pytest.mark.asyncio
async def test_run_diskgraph_colstats_pipeline_forwards_source_size_bytes(
        monkeypatch):
    wf, _activity_calls, child_data_wf_calls, _save_calls, _state = \
        _setup_pipeline_test(monkeypatch)
    await wf._run_diskgraph_colstats_pipeline(
        source_size_bytes_by_table={},
        source_size_bytes=123,
    )

    assert child_data_wf_calls[0]['args'][8] == 123


@pytest.mark.asyncio
async def test_graph_refresh_persists_diskgraph_colstats_warnings(monkeypatch):
    warning = ValidationWarning(title='Duplicate primary keys',
                                message='duplicate source keys')
    table_validation = ValidationResponse(warnings=[warning])
    table = SimpleNamespace(
        get_connector_id=lambda: 'connector',
        get_source_table_name=lambda: 'source_table',
    )
    graph_input = SimpleNamespace(
        tables={'table-id': table},
        base_tables={},
        table_aliases={'table-id': 'graph_alias'},
        connectors={},
        graph_key='graph',
        graph_definition=SimpleNamespace(
            edges=[
                SimpleNamespace(src_table='events', fkey='parent_id',
                                dst_table='events'),
                SimpleNamespace(src_table='events', fkey='user_id',
                                dst_table='users'),
            ],
            tables={'events': SimpleNamespace(pkey='event_id')},
        ),
    )
    workflow = cast(Any, graph_refresh_workflow.GraphRefreshWorkflow())
    workflow._ctx = object()
    workflow._input = graph_input
    workflow._graph_version_info = SimpleNamespace(
        graph_data_version='graph-version',
        table_data_versions={'graph_alias': 'table-version'},
    )
    events = []
    table_results = []
    graph_validations = []

    async def execute_activity(activity_fn, *, args, **_kwargs):
        if activity_fn is DataBackendActivities.validate_table_col_stats:
            events.append(('validate-table', args))
            return table_validation
        if activity_fn is DataBackendActivities.save_colstat_task_result:
            events.append(('save-colstats', args[2].stage))
            return None
        if activity_fn is DataBackendActivities.save_table_refresh_result:
            events.append(('save-table', args[2].stage))
            table_results.append(args[2])
            return None
        if activity_fn is DataBackendActivities.validate_graph_table_stats:
            events.append(('validate-graph', args[2]))
            return GraphTableStatsResult(validation=ValidationResponse())
        raise AssertionError(f'unexpected activity: {activity_fn}')

    async def save_validation(response, _graph_dir, identifier):
        events.append(('save-graph', identifier))
        graph_validations.append(response)

    monkeypatch.setattr(
        graph_refresh_workflow.workflow,
        'execute_activity',
        execute_activity,
    )
    monkeypatch.setattr(
        graph_refresh_workflow,
        'save_validation_response',
        save_validation,
    )
    monkeypatch.setattr(
        graph_refresh_workflow,
        'data_backend_cache',
        lambda: SimpleNamespace(
            graph_cache=SimpleNamespace(graph_dir=lambda *_args: '/graph'), ),
    )

    table_validations = (await
                         workflow._validate_table_col_stats_after_diskgraph(
                             {'graph_alias': 'table-version'},
                             {'graph_alias': 3},
                         ))
    await workflow._save_diskgraph_colstat_task_results(
        {'graph_alias': 'table-version'}, table_validations)
    result = await workflow._validate_graph_table_stats_and_save(
        TableRefreshStage.COMPLETED,
        extra_validation=table_validation,
    )

    assert events[0] == (
        'validate-table',
        ['connector', table, 'table-version', 3],
    )
    assert table_results[0].stage is TableRefreshStage.COMPLETED
    assert table_results[0].validation == table_validation
    assert events[-2:] == [
        ('validate-graph', TableRefreshStage.COMPLETED),
        ('save-graph', 'table_col_stats'),
    ]
    expected_warnings = [
        ('Duplicate primary keys', 'duplicate source keys'),
        ('Self-relation omitted',
         'Self-relation events.parent_id -> events.event_id is not supported '
         'and was omitted from the graph snapshot.'),
    ]
    assert [(item.title, item.message)
            for item in graph_validations[0].warnings] == expected_warnings
    assert [(item.title, item.message)
            for item in result.validation.warnings] == expected_warnings


@pytest.mark.asyncio
async def test_graph_refresh_invalid_table_stats_does_not_start_diskgraph(
        monkeypatch):
    from unittest import mock

    from kumo.validation import ValidationError
    from rest.config.data_backend import GraphDataVersionInfo, GraphVersion

    wf = cast(Any, graph_refresh_workflow.GraphRefreshWorkflow())
    invalid = ValidationResponse(errors=[
        ValidationError(title='Specific table error', message='keep me')
    ])
    monkeypatch.setattr(graph_refresh_workflow.workflow, 'execute_activity',
                        mock.AsyncMock())
    wf._update_tables_timestamp_units = mock.AsyncMock()
    wf._start_refresh_tables = mock.AsyncMock()
    wf.wait_completion_at_stage = mock.AsyncMock(
        return_value=ValidationResponse())
    wf._validate_graph_table_stats_and_save = mock.AsyncMock(
        return_value=GraphTableStatsResult(validation=invalid))
    wf._run_diskgraph_colstats_pipeline = mock.AsyncMock()
    graph_version_info = GraphDataVersionInfo(
        graph_data_version=GraphVersion('v1'), table_data_versions={})

    result, _ = await wf.run(
        object(), SimpleNamespace(), graph_version_info, job_id='job')

    assert result.validation == invalid
    wf._run_diskgraph_colstats_pipeline.assert_not_awaited()
