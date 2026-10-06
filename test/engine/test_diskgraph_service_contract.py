# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DiskgraphService.

Cover the singleton lifecycle, ``DiskgraphHandle`` semantics, the
result dataclasses, and that ``DiskgraphService``'s public methods
route to the expected wrapped functions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from diskgraph import DiskgraphSystemError, TimeUnit
from kumoapi.source_table import FileType
from kumoapi.typing import Dtype, Stype

import kumo.diskgraph.service as diskgraph_service
from io_utils.json import from_json_dict, to_json_dict
from kumo.data.backend_cache import (
    CACHE_DIR,
    BackendCacheManager,
    DirectDiskgraphSourceDecision,
)
from kumo.diskgraph import (
    DiskgraphHandle,
    DiskgraphService,
    IngestResult,
    PredTableResult,
    TrainTableResult,
)
from rest.config.table_v2 import (
    ADLSSourceTable,
    Column,
    S3SourceTable,
    TableDefinition,
)


class _FakeLocalConfig:
    data_root = '/tmp/dg-contract-test'
    local_ingest_root = None
    ingest_space_multiplier = 4


class _FakeDeployEnv:
    def __init__(self, data_compute: Any) -> None:
        self.data_compute = data_compute


@pytest.fixture
def auto_setup_kumo_backend() -> bool:
    """Opt out of conftest.py's autouse GraphEngine setup/teardown.

    These tests exercise DiskgraphService in isolation and never touch
    the global engine state. Running the default fixture is wasteful
    here, so we opt out of the autouse setup/teardown.
    """
    return False


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_service_instantiates():
    svc = DiskgraphService(_FakeLocalConfig())
    assert isinstance(svc, DiskgraphService)


# ---------------------------------------------------------------------------
# Service factory
# ---------------------------------------------------------------------------


def test_non_diskgraph_config_raises(monkeypatch):
    import service_framework.deploy_env_config as dec
    from kumo.diskgraph import get_service

    class NotDiskgraph:
        type = 'NOT_DISKGRAPH'

    monkeypatch.setattr(
        dec,
        'deploy_env',
        lambda: _FakeDeployEnv(NotDiskgraph()),
    )
    with pytest.raises(RuntimeError, match='non-DiskGraph mode'):
        get_service()


# ---------------------------------------------------------------------------
# DiskgraphHandle
# ---------------------------------------------------------------------------


def test_handle_frozen_equality():
    h1 = DiskgraphHandle(graph_key='A', graph_data_version='v1')
    h2 = DiskgraphHandle(graph_key='A', graph_data_version='v1')
    h3 = DiskgraphHandle(graph_key='A', graph_data_version='v2')
    assert h1 == h2
    assert h1 != h3
    assert hash(h1) == hash(h2)


def test_handle_is_frozen():
    h = DiskgraphHandle(graph_key='A', graph_data_version='v1')
    with pytest.raises(Exception):
        h.graph_key = 'B'


def test_handle_json_round_trip():
    """Handles must survive Temporal's JSON-based argument serialization."""
    h = DiskgraphHandle(graph_key='NA-abc', graph_data_version='deadbeef')
    d = to_json_dict(h)
    h2 = from_json_dict(d, DiskgraphHandle)
    assert h == h2


def test_handle_for_test_factory():
    h = DiskgraphHandle.for_test()
    assert h.graph_key == 'NA-test'
    assert h.graph_data_version == 'v0'
    h2 = DiskgraphHandle.for_test(graph_key='custom')
    assert h2.graph_key == 'custom'
    assert h2.graph_data_version == 'v0'


# ---------------------------------------------------------------------------
# DiskgraphService — handle resolution + delegation
# ---------------------------------------------------------------------------


def _service() -> DiskgraphService:
    return DiskgraphService(_FakeLocalConfig())


@pytest.mark.parametrize(
    'error',
    [
        FileNotFoundError('snapshot disappeared'),
        DiskgraphSystemError('Permission denied'),
    ],
)
def test_list_ready_handles_skips_inaccessible_snapshot_and_keeps_sibling(
        monkeypatch, tmp_path, error):
    import kumo_core.diskgraph.snapshot as snapshot_module

    root = tmp_path / 'diskgraph'
    inaccessible = root / 'graph' / 'inaccessible'
    ready = root / 'graph' / 'ready'
    inaccessible.mkdir(parents=True)
    ready.mkdir()

    def check_output(path):
        if path == str(inaccessible):
            raise error
        assert path == str(ready)
        return object()

    monkeypatch.setattr(snapshot_module, 'check_diskgraph_output', check_output)
    service = DiskgraphService(SimpleNamespace(data_root=str(root)))

    assert service.list_ready_handles() == [
        DiskgraphHandle(graph_key='graph', graph_data_version='ready')
    ]


def _s3_table(source_path: str = 'events') -> TableDefinition:
    return TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int)],
        source_table=S3SourceTable(
            connector_id=f'connector_{"1" * 32}',
            source={
                'mode': 'relative',
                'path': source_path,
            },
            file_type=FileType.PARQUET,
        ),
        pkey='id',
        time_col=None,
    )


def _object_source(
    *,
    size: int = 123,
    e_tag: str | None = 'etag',
    version: str | None = None,
) -> dict[str, Any]:
    return {
        'type':
        'object_store_parquet',
        'objects': [{
            'path': 's3://bucket/events/part-000.parquet',
            'size': size,
            'e_tag': e_tag,
            'version': version,
        }],
        'auth_profile':
        'ambient',
    }


def _persist_source_decision(
    tmp_path,
    table: TableDefinition,
    source=None,
    *,
    data_version: str = 'v1',
) -> None:
    cache = BackendCacheManager(str(tmp_path / 'persist' / CACHE_DIR))
    fingerprint = table.fingerprint()
    cache.direct_diskgraph_source_decision.put(
        fingerprint.connector_name,
        fingerprint.source_name,
        data_version,
        DirectDiskgraphSourceDecision(source),
        save=True,
    )


def _prepare_s3_sources(
    tmp_path,
    table: TableDefinition,
    *,
    expected_size: int = 123,
) -> dict[str, dict]:
    return _service().prepare_direct_sources(
        {str(table.get_connector_id()): object()},
        {'events_alias': table},
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={'events_alias': 'v1'},
        source_size_bytes_by_table={'events_alias': expected_size},
    )


class _DownloadConnector:
    def __init__(self, control_markers=()) -> None:
        self.calls = 0
        self.control_markers = control_markers

    def download(self, _source_name, _fingerprint, _data_version,
                 local_dest) -> None:
        self.calls += 1
        Path(local_dest, 'part-a.parquet').write_bytes(b'PAR1-a')
        Path(local_dest, 'part-b.parquet').write_bytes(b'PAR1-b')
        for marker in self.control_markers:
            Path(local_dest, marker).write_bytes(b'')


def _download_staged_source(monkeypatch, tmp_path, *, data_version='v1'):
    connector = _DownloadConnector()
    table = _s3_table()
    connector_id = str(table.get_connector_id())
    monkeypatch.setattr(
        'rest.utils.driver_common.get_connector_from_spec',
        lambda _spec: connector,
    )
    destination = str(tmp_path / 'source')
    _service().download_table_files(
        {connector_id: object()},
        {'events_alias': data_version},
        {'events_alias': table},
        {'events_alias': destination},
    )
    return connector, table, destination


def test_download_marker_reuses_exact_staged_inventory(monkeypatch, tmp_path):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    marker = Path(destination, '_DOWNLOAD_SUCCESS')
    payload = json.loads(marker.read_text())
    assert payload['schema_version'] == (
        diskgraph_service._DOWNLOAD_INVENTORY_SCHEMA)
    assert [entry['path'] for entry in payload['files']
            ] == ['part-a.parquet', 'part-b.parquet']

    _service().download_table_files(
        {str(table.get_connector_id()): object()},
        {'events_alias': 'v1'},
        {'events_alias': table},
        {'events_alias': destination},
    )
    assert connector.calls == 1


def test_download_marker_accepts_and_binds_empty_control_files(monkeypatch,
                                                               tmp_path):
    connector = _DownloadConnector(
        control_markers=('_SUCCESS', '_committed_123', '_started_123'))
    table = _s3_table()
    connector_id = str(table.get_connector_id())
    monkeypatch.setattr(
        'rest.utils.driver_common.get_connector_from_spec',
        lambda _spec: connector,
    )
    destination = str(tmp_path / 'source')

    for _ in range(2):
        _service().download_table_files(
            {connector_id: object()},
            {'events_alias': 'v1'},
            {'events_alias': table},
            {'events_alias': destination},
        )

    payload = json.loads(Path(destination, '_DOWNLOAD_SUCCESS').read_text())
    assert [entry['path'] for entry in payload['files']] == [
        '_SUCCESS', '_committed_123', '_started_123', 'part-a.parquet',
        'part-b.parquet'
    ]
    assert connector.calls == 1


def test_download_removes_stale_incomplete_attempt(monkeypatch, tmp_path):
    stale_attempt = tmp_path / '.source.download-crashed'
    stale_attempt.mkdir()
    (stale_attempt / 'part.parquet').write_bytes(b'partial')

    connector, _table, destination = _download_staged_source(
        monkeypatch, tmp_path)

    assert connector.calls == 1
    assert Path(destination, '_DOWNLOAD_SUCCESS').exists()
    assert not stale_attempt.exists()


@pytest.mark.parametrize('damage', ['missing', 'extra', 'changed', 'legacy'])
def test_download_marker_rebuilds_inconsistent_inventory(
        monkeypatch, tmp_path, damage):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    if damage == 'missing':
        Path(destination, 'part-b.parquet').unlink()
    elif damage == 'extra':
        Path(destination, 'part-extra.parquet').write_bytes(b'PAR1-extra')
    elif damage == 'changed':
        replacement = Path(destination, 'replacement')
        replacement.write_bytes(b'PAR1-z')
        os.replace(replacement, Path(destination, 'part-a.parquet'))
    else:
        Path(destination, '_DOWNLOAD_SUCCESS').write_text('')

    _service().download_table_files(
        {str(table.get_connector_id()): object()},
        {'events_alias': 'v1'},
        {'events_alias': table},
        {'events_alias': destination},
    )
    assert connector.calls == 2
    assert sorted(path.name
                  for path in Path(destination).glob('*.parquet')) == [
                      'part-a.parquet', 'part-b.parquet'
                  ]


def test_download_marker_rejects_cross_attempt_data_version(
        monkeypatch, tmp_path):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    _service().download_table_files(
        {str(table.get_connector_id()): object()},
        {'events_alias': 'v2'},
        {'events_alias': table},
        {'events_alias': destination},
    )
    assert connector.calls == 2
    payload = json.loads(Path(destination, '_DOWNLOAD_SUCCESS').read_text())
    assert payload['context']['data_version'] == 'v2'


@pytest.mark.parametrize('damage', [
    'corrupt',
    'wrong_shape',
    'oversized',
    'symlink_directory',
    'zero_byte',
])
def test_download_marker_rebuilds_bounded_or_unsafe_inventory(
        monkeypatch, tmp_path, damage):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    marker = Path(destination, '_DOWNLOAD_SUCCESS')
    if damage == 'corrupt':
        marker.write_text('{not json')
    elif damage == 'wrong_shape':
        marker.write_text('[]')
    elif damage == 'oversized':
        marker.write_bytes(
            b'x' * (diskgraph_service._MAX_DOWNLOAD_MARKER_BYTES + 1))
    elif damage == 'symlink_directory':
        external = tmp_path / 'external'
        external.mkdir()
        Path(destination, 'linked').symlink_to(external, target_is_directory=True)
    else:
        Path(destination, 'empty.parquet').write_bytes(b'')

    _service().download_table_files(
        {str(table.get_connector_id()): object()},
        {'events_alias': 'v1'},
        {'events_alias': table},
        {'events_alias': destination},
    )

    assert connector.calls == 2
    assert sorted(path.name
                  for path in Path(destination).glob('*.parquet')) == [
                      'part-a.parquet', 'part-b.parquet'
                  ]


def test_download_failure_preserves_previous_staged_copy(monkeypatch,
                                                         tmp_path):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    Path(destination, 'part-b.parquet').unlink()
    previous_marker = Path(destination, '_DOWNLOAD_SUCCESS').read_bytes()

    def fail_download(*_args, **_kwargs):
        connector.calls += 1
        raise RuntimeError('connector download failed')

    monkeypatch.setattr(connector, 'download', fail_download)
    with pytest.raises(RuntimeError, match='connector download failed'):
        _service().download_table_files(
            {str(table.get_connector_id()): object()},
            {'events_alias': 'v1'},
            {'events_alias': table},
            {'events_alias': destination},
        )

    assert Path(destination, 'part-a.parquet').read_bytes() == b'PAR1-a'
    assert not Path(destination, 'part-b.parquet').exists()
    assert Path(destination, '_DOWNLOAD_SUCCESS').read_bytes() == previous_marker
    assert not list(tmp_path.glob('.source.download-*'))


def test_interrupted_marker_publication_preserves_previous_staged_copy(
        monkeypatch, tmp_path):
    _connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    Path(destination, 'part-b.parquet').unlink()
    previous_marker = Path(destination, '_DOWNLOAD_SUCCESS').read_bytes()
    monkeypatch.setattr(
        'kumo.diskgraph.service._write_download_inventory_marker',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError('marker publication interrupted')),
    )

    with pytest.raises(RuntimeError, match='publication interrupted'):
        _service().download_table_files(
            {str(table.get_connector_id()): object()},
            {'events_alias': 'v1'},
            {'events_alias': table},
            {'events_alias': destination},
        )

    assert Path(destination, 'part-a.parquet').exists()
    assert not Path(destination, 'part-b.parquet').exists()
    assert Path(destination, '_DOWNLOAD_SUCCESS').read_bytes() == previous_marker
    assert not list(tmp_path.glob('.source.download-*'))


def test_connector_resolution_failure_happens_before_cached_copy_deletion(
        monkeypatch, tmp_path):
    _connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    Path(destination, 'part-b.parquet').unlink()
    previous_marker = Path(destination, '_DOWNLOAD_SUCCESS').read_bytes()
    monkeypatch.setattr(
        'rest.utils.driver_common.get_connector_from_spec',
        lambda _spec: (_ for _ in ()).throw(RuntimeError('connector unavailable')),
    )

    with pytest.raises(RuntimeError, match='connector unavailable'):
        _service().download_table_files(
            {str(table.get_connector_id()): object()},
            {'events_alias': 'v1'},
            {'events_alias': table},
            {'events_alias': destination},
        )

    assert Path(destination, 'part-a.parquet').exists()
    assert Path(destination, '_DOWNLOAD_SUCCESS').read_bytes() == previous_marker


def test_download_marker_replace_ack_loss_leaves_complete_marker(
        monkeypatch, tmp_path):
    marker = tmp_path / '_DOWNLOAD_SUCCESS'
    original_replace = os.replace

    def replace_then_lose_ack(source, destination):
        original_replace(source, destination)
        raise RuntimeError('replace acknowledgement lost')

    monkeypatch.setattr(
        'kumo.diskgraph.service.os.replace',
        replace_then_lose_ack,
    )
    with pytest.raises(RuntimeError, match='acknowledgement lost'):
        diskgraph_service._write_download_inventory_marker(
            str(marker), {
                'schema_version': diskgraph_service._DOWNLOAD_INVENTORY_SCHEMA,
                'context': {},
                'files': [],
            })

    assert json.loads(marker.read_text())['schema_version'] == \
        diskgraph_service._DOWNLOAD_INVENTORY_SCHEMA
    assert not list(tmp_path.glob('._DOWNLOAD_SUCCESS.*.tmp'))


def test_download_final_swap_failure_rolls_back_previous_copy(
        monkeypatch, tmp_path):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    Path(destination, 'part-b.parquet').unlink()
    previous_marker = Path(destination, '_DOWNLOAD_SUCCESS').read_bytes()
    original_replace = os.replace

    def fail_before_final_swap(source, target):
        if (target == destination
                and '.source.download-' in os.path.basename(source)):
            raise OSError('final swap failed before publication')
        return original_replace(source, target)

    monkeypatch.setattr(
        'kumo.diskgraph.service.os.replace', fail_before_final_swap)
    with pytest.raises(OSError, match='final swap failed'):
        _service().download_table_files(
            {str(table.get_connector_id()): object()},
            {'events_alias': 'v1'},
            {'events_alias': table},
            {'events_alias': destination},
        )

    assert Path(destination, 'part-a.parquet').read_bytes() == b'PAR1-a'
    assert not Path(destination, 'part-b.parquet').exists()
    assert Path(destination, '_DOWNLOAD_SUCCESS').read_bytes() == previous_marker
    assert not Path(f'{destination}.previous').exists()
    assert not list(tmp_path.glob('.source.download-*'))
    assert connector.calls == 2


def test_download_final_swap_ack_loss_accepts_exact_new_copy(
        monkeypatch, tmp_path):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    Path(destination, 'part-b.parquet').unlink()
    original_replace = os.replace
    lost_ack = False

    def replace_then_lose_final_ack(source, target):
        nonlocal lost_ack
        result = original_replace(source, target)
        if (not lost_ack and target == destination
                and '.source.download-' in os.path.basename(source)):
            lost_ack = True
            raise OSError('final swap acknowledgement lost')
        return result

    monkeypatch.setattr('kumo.diskgraph.service.os.replace',
                        replace_then_lose_final_ack)
    _service().download_table_files(
        {str(table.get_connector_id()): object()},
        {'events_alias': 'v1'},
        {'events_alias': table},
        {'events_alias': destination},
    )

    assert connector.calls == 2
    assert lost_ack
    assert sorted(path.name
                  for path in Path(destination).glob('*.parquet')) == [
                      'part-a.parquet', 'part-b.parquet'
                  ]
    assert not Path(f'{destination}.previous').exists()


def test_download_recovers_crash_after_old_cache_rename(monkeypatch,
                                                        tmp_path):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    backup = f'{destination}.previous'
    os.replace(destination, backup)

    _service().download_table_files(
        {str(table.get_connector_id()): object()},
        {'events_alias': 'v1'},
        {'events_alias': table},
        {'events_alias': destination},
    )

    assert connector.calls == 1
    assert Path(destination, '_DOWNLOAD_SUCCESS').exists()
    assert not Path(backup).exists()


def test_download_recovers_crash_after_new_copy_publication(
        monkeypatch, tmp_path):
    connector, table, destination = _download_staged_source(
        monkeypatch, tmp_path)
    Path(destination, 'part-b.parquet').unlink()
    original_remove = diskgraph_service._remove_owned_download_path
    backup = f'{destination}.previous'

    def fail_backup_cleanup(path):
        if path == backup:
            raise OSError('process stopped before backup cleanup')
        return original_remove(path)

    monkeypatch.setattr(diskgraph_service, '_remove_owned_download_path',
                        fail_backup_cleanup)
    with pytest.raises(OSError, match='before backup cleanup'):
        _service().download_table_files(
            {str(table.get_connector_id()): object()},
            {'events_alias': 'v1'},
            {'events_alias': table},
            {'events_alias': destination},
        )
    assert Path(destination, 'part-b.parquet').exists()
    assert Path(backup).exists()

    monkeypatch.setattr(diskgraph_service, '_remove_owned_download_path',
                        original_remove)
    _service().download_table_files(
        {str(table.get_connector_id()): object()},
        {'events_alias': 'v1'},
        {'events_alias': table},
        {'events_alias': destination},
    )
    assert connector.calls == 2
    assert not Path(backup).exists()
    assert Path(destination, 'part-b.parquet').exists()


def test_download_marker_rejects_file_count_before_local_inventory(
        monkeypatch, tmp_path):
    marker = tmp_path / '_DOWNLOAD_SUCCESS'
    marker.write_text(json.dumps({
        'schema_version': diskgraph_service._DOWNLOAD_INVENTORY_SCHEMA,
        'context': {},
        'files': [{}, {}],
    }))
    monkeypatch.setattr(diskgraph_service, '_MAX_DOWNLOAD_FILES', 1)
    monkeypatch.setattr(
        diskgraph_service,
        '_download_inventory_context',
        lambda *_args: {},
    )
    monkeypatch.setattr(
        diskgraph_service,
        '_local_download_inventory',
        lambda _path: pytest.fail('oversized file list must fail first'),
    )

    valid, reason = diskgraph_service._validate_download_inventory(
        str(marker), 'alias', MagicMock(), 'v1', str(tmp_path))
    assert not valid
    assert 'file-count' in reason


@pytest.mark.parametrize('unsafe_output', ['zero_byte', 'symlink_directory'])
def test_download_rejects_unsafe_connector_output_without_publication(
        monkeypatch, tmp_path, unsafe_output):
    connector = _DownloadConnector()
    table = _s3_table()
    connector_id = str(table.get_connector_id())

    def unsafe_download(_source_name, _fingerprint, _data_version,
                        local_dest):
        connector.calls += 1
        if unsafe_output == 'zero_byte':
            Path(local_dest, 'empty.parquet').write_bytes(b'')
        else:
            external = tmp_path / 'external'
            external.mkdir()
            Path(local_dest, 'linked').symlink_to(
                external, target_is_directory=True)

    monkeypatch.setattr(connector, 'download', unsafe_download)
    monkeypatch.setattr(
        'rest.utils.driver_common.get_connector_from_spec',
        lambda _spec: connector,
    )
    destination = str(tmp_path / 'source')

    with pytest.raises(RuntimeError, match='empty file|symlink directory'):
        _service().download_table_files(
            {connector_id: object()},
            {'events_alias': 'v1'},
            {'events_alias': table},
            {'events_alias': destination},
        )

    assert not Path(destination, '_DOWNLOAD_SUCCESS').exists()
    assert not list(tmp_path.glob('.source.download-*'))


def test_resolve_joins_data_root_key_version():
    svc = _service()
    h = DiskgraphHandle(graph_key='A', graph_data_version='v1')
    assert svc._resolve(h) == '/tmp/dg-contract-test/A/v1'


def test_dataplane_snapshot_path_keeps_non_adls_roots():
    h = DiskgraphHandle(graph_key='gk', graph_data_version='v10')
    assert DiskgraphService.dataplane_snapshot_path(
        h, '/dp') == ('/dp/diskgraph_snapshots/gk/v10')
    assert DiskgraphService.dataplane_snapshot_path(
        h,
        's3://bucket/root') == ('s3://bucket/root/diskgraph_snapshots/gk/v10')


def test_dataplane_snapshot_path_converts_adls_for_diskgraph():
    h = DiskgraphHandle(graph_key='gk', graph_data_version='v10')
    root = 'abfss://cont@acct.dfs.core.windows.net/dev/root'

    assert DiskgraphService.dataplane_snapshot_path(
        h, root) == ('abfss://cont@acct.dfs.core.windows.net/dev/root/'
                     'diskgraph_snapshots/gk/v10')
    assert DiskgraphService.diskgraph_dataplane_snapshot_path(
        h, root) == ('az://acct/cont/dev/root/diskgraph_snapshots/gk/v10')


def test_ingest_delegates_to_run_diskgraph_ingest(monkeypatch):
    """``ingest`` delegates to module-level ``run_diskgraph_ingest``."""
    import kumo_core.diskgraph.snapshot as sm_mod

    import kumo.diskgraph.ingest as ingest_mod

    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return sm_mod.DiskgraphIngestResult(
            index_path='/i',
            col_stats_path='/c',
            ingest_stats_path='/s',
            kumo_stats_dir='/k',
        )

    monkeypatch.setattr(ingest_mod, 'run_diskgraph_ingest', fake_run)
    svc = _service()
    h = DiskgraphHandle(graph_key='gk', graph_data_version='v10')
    heartbeat = MagicMock()
    result = svc.ingest(
        {'tables': []},
        h,
        data_persist_dir='/d',
        table_data_versions={'t': 'v'},
        dataplane_root='/dp',
        heartbeat=heartbeat,
    )
    assert captured['output_dir'] == '/tmp/dg-contract-test/gk/v10'
    assert captured['data_persist_dir'] == '/d'
    assert captured['output_dataplane_dir'] == (
        '/dp/diskgraph_snapshots/gk/v10')
    assert captured['heartbeat'] is heartbeat
    assert result.index_path == '/i'


@pytest.mark.parametrize('source', [_object_source(), None],
                         ids=['direct', 'staged'])
def test_prepare_direct_sources_reuses_frozen_decision_without_connector(
        monkeypatch, tmp_path, source):
    import rest.utils.driver_common as driver_common

    table = _s3_table()
    _persist_source_decision(tmp_path, table, source)
    monkeypatch.setattr(
        driver_common,
        'get_connector_from_spec',
        lambda _reference: pytest.fail(
            'a frozen decision must not reopen the connector'),
    )

    sources = _prepare_s3_sources(tmp_path, table)

    expected_sources = {} if source is None else {
        'events_alias': {
            'source': source,
            'source_size': 123,
        }
    }
    assert sources == expected_sources


def test_prepare_direct_sources_rejects_frozen_manifest_size_mismatch(tmp_path):
    table = _s3_table()
    _persist_source_decision(tmp_path, table, _object_source())

    with pytest.raises(RuntimeError, match='recorded 124'):
        _prepare_s3_sources(tmp_path, table, expected_size=124)


def test_prepare_direct_sources_rejects_malformed_decision(tmp_path):
    table = _s3_table()
    _persist_source_decision(
        tmp_path, table, {
            'type': 'object_store_parquet',
            'objects': 'not-a-list',
            'auth_profile': 'ambient',
        })

    with pytest.raises(RuntimeError, match='malformed'):
        _prepare_s3_sources(tmp_path, table)


def test_prepare_direct_sources_builds_object_manifest(
        monkeypatch, tmp_path):
    import rest.utils.driver_common as driver_common
    from kumo.connector.composite_file_connector import CompositeFileConnector

    connector_id = f'connector_{"2" * 32}'
    connector = CompositeFileConnector(
        connector_id,
        root_dir='abfss://events@customer.dfs.core.windows.net/root',
    )
    source = {
        'type':
        'object_store_parquet',
        'objects': [{
            'path': ('abfss://events@customer.dfs.core.windows.net/'
                     'root/events/part.parquet'),
            'size':
            123,
            'e_tag':
            'etag',
            'version':
            None,
        }],
        'auth_profile':
        'ambient',
    }
    monkeypatch.setattr(connector, 'uses_direct_diskgraph_source_for',
                        lambda name: name == 'events')
    monkeypatch.setattr(connector, 'diskgraph_source',
                        lambda *, source_name: source)
    closed = []

    def close():
        closed.append(True)
        raise RuntimeError('cleanup failure must not replace handoff')

    monkeypatch.setattr(connector, 'close', close)
    monkeypatch.setattr(driver_common, 'get_connector_from_spec',
                        lambda _reference: connector)
    table = TableDefinition(
        cols=[Column(name='id', stype=Stype.ID, dtype=Dtype.int)],
        source_table=ADLSSourceTable(
            connector_id=connector_id,
            source={
                'mode': 'relative',
                'path': 'events',
            },
            file_type=FileType.PARQUET,
        ),
        pkey='id',
        time_col=None,
    )
    sources = _service().prepare_direct_sources(
        {connector_id: object()},
        {'events_alias': table},
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={'events_alias': 'v1'},
        source_size_bytes_by_table={'events_alias': 999},
    )

    assert sources == {
        'events_alias': {
            'source': {
                **source,
                'objects': [{
                    **source['objects'][0],
                    'path':
                    'az://customer/events/root/events/part.parquet',
                }],
            },
            'source_size': 123,
        }
    }
    assert closed == [True]


def test_prepare_subprocess_environment_preserves_renewable_s3_provider(
        monkeypatch):
    import rest.utils.driver_common as driver_common
    from kumo.connector.composite_file_connector import CompositeFileConnector
    from service_framework import deploy_env_config

    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment', lambda: {})
    monkeypatch.setattr(
        deploy_env_config,
        'deploy_env',
        lambda: SimpleNamespace(data_plane_storage_root='/tmp/dataplane'),
    )
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'competing-static-key')
    monkeypatch.setenv('AWS_PROFILE', 'competing-profile')
    monkeypatch.setenv('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI',
                       '/v2/credentials/pod')
    monkeypatch.setenv('AWS_EC2_METADATA_DISABLED', 'true')

    connector = CompositeFileConnector(
        's3',
        root_dir='s3://customer-bucket/root',
    )
    monkeypatch.setattr(connector, 'uses_direct_diskgraph_source_for',
                        lambda name: name == 'events')
    monkeypatch.setattr(
        connector,
        'diskgraph_subprocess_environment',
        lambda _name: {
            'AWS_REGION': 'us-west-2',
            'AWS_DEFAULT_REGION': 'us-west-2',
        },
    )
    monkeypatch.setattr(driver_common, 'get_connector_from_spec',
                        lambda _reference: connector)
    table = SimpleNamespace(
        is_sql=False,
        get_connector_id=lambda: 's3',
        get_source_table_name=lambda: 'events',
    )

    environment = _service().prepare_subprocess_environment(
        {'s3': object()},
        {'events': table},
        source_specs={'events': object()},
        data_persist_dir='/tmp/persist',
        table_data_versions={'events': 'v1'},
    )

    assert environment['AWS_REGION'] == 'us-west-2'
    assert environment['AWS_DEFAULT_REGION'] == 'us-west-2'
    assert environment['AWS_ACCESS_KEY_ID'] is None
    assert environment['AWS_SECRET_ACCESS_KEY'] is None
    assert environment['AWS_CONTAINER_CREDENTIALS_RELATIVE_URI'] == (
        '/v2/credentials/pod')
    assert environment['AWS_EC2_METADATA_DISABLED'] is None
    assert environment['AWS_CONFIG_FILE'] == os.devnull
    assert environment['AWS_SHARED_CREDENTIALS_FILE'] == os.devnull
    assert environment['AWS_PROFILE'] is None
    assert environment['S3_ACCESS_KEY'] is None


def test_prepare_subprocess_environment_uses_exact_fresh_source_aliases(
        monkeypatch):
    import rest.utils.driver_common as driver_common
    from kumo.connector.composite_file_connector import CompositeFileConnector
    from service_framework import deploy_env_config

    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment', lambda: {})
    monkeypatch.setattr(
        deploy_env_config,
        'deploy_env',
        lambda: SimpleNamespace(data_plane_storage_root='/tmp/dataplane'),
    )
    direct = CompositeFileConnector(
        'direct',
        root_dir='s3://customer-bucket/root',
    )
    eligibility = MagicMock(
        side_effect=AssertionError('fresh sources must not be re-probed'))
    monkeypatch.setattr(direct, 'uses_direct_diskgraph_source_for', eligibility)
    environment_for = MagicMock(return_value={
        'AWS_REGION': 'us-west-2',
        'AWS_DEFAULT_REGION': 'us-west-2',
    })
    monkeypatch.setattr(direct, 'diskgraph_subprocess_environment',
                        environment_for)
    get_connector = MagicMock(return_value=direct)
    monkeypatch.setattr(driver_common, 'get_connector_from_spec', get_connector)
    direct_table = SimpleNamespace(
        is_sql=False,
        get_connector_id=lambda: 'direct',
        get_source_table_name=lambda: 'direct-events',
    )
    staged_table = SimpleNamespace(
        is_sql=False,
        get_connector_id=lambda: 'staged',
        get_source_table_name=lambda: 'staged-events',
    )

    environment = _service().prepare_subprocess_environment(
        {
            'direct': 'direct-reference',
            'staged': 'staged-reference',
        },
        {
            'direct': direct_table,
            'staged': staged_table,
        },
        source_specs={'direct': object()},
        data_persist_dir='/tmp/persist',
        table_data_versions={
            'direct': 'v1',
            'staged': 'v2',
        },
    )

    assert environment['AWS_REGION'] == 'us-west-2'
    get_connector.assert_called_once_with('direct-reference')
    environment_for.assert_called_once_with('direct-events')
    eligibility.assert_not_called()


@pytest.mark.parametrize('source,expected_direct', [
    (_object_source(), True),
    (None, False),
])
def test_prepare_subprocess_environment_reuses_frozen_retry_decision(
        monkeypatch, tmp_path, source, expected_direct):
    import rest.utils.driver_common as driver_common
    from kumo.connector.composite_file_connector import CompositeFileConnector
    from service_framework import deploy_env_config

    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment', lambda: {})
    monkeypatch.setattr(
        deploy_env_config,
        'deploy_env',
        lambda: SimpleNamespace(data_plane_storage_root='/tmp/dataplane'),
    )
    table = _s3_table()
    _persist_source_decision(tmp_path, table, source)
    connector = CompositeFileConnector(
        's3',
        root_dir='s3://customer-bucket/root',
    )
    eligibility = MagicMock(
        side_effect=AssertionError('frozen decisions must not be re-probed'))
    monkeypatch.setattr(connector, 'uses_direct_diskgraph_source_for',
                        eligibility)
    environment_for = MagicMock(return_value={
        'AWS_REGION': 'us-west-2',
        'AWS_DEFAULT_REGION': 'us-west-2',
    })
    monkeypatch.setattr(connector, 'diskgraph_subprocess_environment',
                        environment_for)
    get_connector = MagicMock(return_value=connector)
    monkeypatch.setattr(driver_common, 'get_connector_from_spec', get_connector)
    connector_id = str(table.get_connector_id())

    environment = _service().prepare_subprocess_environment(
        {connector_id: 'connector-reference'},
        {'events_alias': table},
        source_specs=None,
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={'events_alias': 'v1'},
    )

    if expected_direct:
        assert environment['AWS_REGION'] == 'us-west-2'
        get_connector.assert_called_once_with('connector-reference')
        environment_for.assert_called_once_with('events')
    else:
        get_connector.assert_not_called()
        environment_for.assert_not_called()
    eligibility.assert_not_called()


def test_direct_and_staged_s3_retry_preserves_credential_inventory_split(
        monkeypatch, tmp_path):
    import rest.utils.driver_common as driver_common
    from kumo.connector.composite_file_connector import CompositeFileConnector
    from service_framework import deploy_env_config

    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment', lambda: {})
    monkeypatch.setattr(
        deploy_env_config,
        'deploy_env',
        lambda: SimpleNamespace(data_plane_storage_root='/tmp/dataplane'),
    )
    direct_table = _s3_table('direct-events')
    staged_table = _s3_table('staged-events')
    connector_id = str(direct_table.get_connector_id())
    assert connector_id == str(staged_table.get_connector_id())
    _persist_source_decision(
        tmp_path, direct_table, _object_source(), data_version='v1')
    _persist_source_decision(
        tmp_path, staged_table, None, data_version='v2')

    connector = CompositeFileConnector(
        's3', root_dir='s3://customer-bucket/root')
    eligibility = MagicMock(
        side_effect=AssertionError('frozen decisions must not be re-probed'))
    monkeypatch.setattr(connector, 'uses_direct_diskgraph_source_for',
                        eligibility)
    environment_for = MagicMock(return_value={
        'AWS_REGION': 'us-west-2',
        'AWS_DEFAULT_REGION': 'us-west-2',
    })
    monkeypatch.setattr(connector, 'diskgraph_subprocess_environment',
                        environment_for)
    download = MagicMock(side_effect=lambda *_args: (
        Path(_args[-1], 'part-a.parquet').write_bytes(b'PAR1-a'),
        Path(_args[-1], 'part-b.parquet').write_bytes(b'PAR1-b'),
    ))
    monkeypatch.setattr(connector, 'download', download)
    get_connector = MagicMock(return_value=connector)
    monkeypatch.setattr(driver_common, 'get_connector_from_spec', get_connector)

    connector_specs = {connector_id: 'connector-reference'}
    tables = {
        'direct': direct_table,
        'staged': staged_table,
    }
    versions = {
        'direct': 'v1',
        'staged': 'v2',
    }
    staged_root = str(tmp_path / 'staged-source')
    service = _service()

    fresh_environment = service.prepare_subprocess_environment(
        connector_specs,
        tables,
        source_specs={'direct': _object_source()},
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions=versions,
    )
    service.download_table_files(
        connector_specs,
        versions,
        tables,
        {'staged': staged_root},
    )

    assert fresh_environment['AWS_REGION'] == 'us-west-2'
    environment_for.assert_called_once_with('direct-events')
    download.assert_called_once()
    marker = json.loads(
        Path(staged_root, '_DOWNLOAD_SUCCESS').read_text())
    assert marker['context']['alias'] == 'staged'
    assert marker['context']['data_version'] == 'v2'
    assert [item['path'] for item in marker['files']] == [
        'part-a.parquet',
        'part-b.parquet',
    ]

    retry_environment = service.prepare_subprocess_environment(
        connector_specs,
        tables,
        source_specs=None,
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions=versions,
    )
    service.download_table_files(
        connector_specs,
        versions,
        tables,
        {'staged': staged_root},
    )

    assert retry_environment['AWS_REGION'] == 'us-west-2'
    assert environment_for.call_count == 2
    assert download.call_count == 1
    eligibility.assert_not_called()


def test_prepare_subprocess_environment_preserves_ambient_s3_dataplane(
        monkeypatch):
    import rest.utils.driver_common as driver_common
    from service_framework import deploy_env_config

    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment', lambda: {})
    monkeypatch.setattr(
        deploy_env_config,
        'deploy_env',
        lambda: SimpleNamespace(
            data_plane_storage_root='s3://example-dataplane/root'),
    )
    monkeypatch.setattr(driver_common, 'get_connector_from_spec',
                        lambda _reference: object())
    table = SimpleNamespace(
        is_sql=False,
        get_connector_id=lambda: 'warehouse',
        get_source_table_name=lambda: 'events',
    )

    environment = _service().prepare_subprocess_environment(
        {'warehouse': object()},
        {'events': table},
        source_specs={},
        data_persist_dir='/tmp/persist',
        table_data_versions={'events': 'v1'},
    )

    assert environment == {}


def test_prepare_subprocess_environment_uses_explicit_s3_dataplane(
        monkeypatch):
    import rest.utils.driver_common as driver_common

    deployment = {
        'AWS_ACCESS_KEY_ID': 'dataplane-key',
        'AWS_SECRET_ACCESS_KEY': 'dataplane-secret',
        'AWS_ENDPOINT_URL_S3': 'https://dataplane-minio:9000',
    }
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment',
        lambda: deployment,
    )
    monkeypatch.setattr(driver_common, 'get_connector_from_spec',
                        lambda _reference: object())
    table = SimpleNamespace(
        is_sql=False,
        get_connector_id=lambda: 'warehouse',
        get_source_table_name=lambda: 'events',
    )

    environment = _service().prepare_subprocess_environment(
        {'warehouse': object()},
        {'events': table},
        source_specs={},
        data_persist_dir='/tmp/persist',
        table_data_versions={'events': 'v1'},
    )

    assert {key: environment[key] for key in deployment} == deployment
    assert environment['AWS_PROFILE'] is None


def test_prepare_subprocess_environment_rejects_distinct_s3_connections(
        monkeypatch):
    import rest.utils.driver_common as driver_common
    from kumo.connector.composite_file_connector import CompositeFileConnector

    deployment = {
        'AWS_ACCESS_KEY_ID': 'dataplane-key',
        'AWS_SECRET_ACCESS_KEY': 'dataplane-secret',
    }
    connector = CompositeFileConnector(
        's3',
        root_dir='s3://customer-bucket/root',
    )
    monkeypatch.setattr(connector, 'uses_direct_diskgraph_source_for',
                        lambda _name: True)
    monkeypatch.setattr(
        connector,
        'diskgraph_subprocess_environment',
        lambda _name: {
            'AWS_ACCESS_KEY_ID': 'connector-key',
            'AWS_SECRET_ACCESS_KEY': 'connector-secret',
        },
    )
    monkeypatch.setattr(
        'io_utils.fsutil.s3_connection.s3_subprocess_environment',
        lambda: deployment,
    )
    monkeypatch.setattr(driver_common, 'get_connector_from_spec',
                        lambda _reference: connector)
    table = SimpleNamespace(
        is_sql=False,
        get_connector_id=lambda: 's3',
        get_source_table_name=lambda: 'events',
    )

    with pytest.raises(RuntimeError, match='different connections'):
        _service().prepare_subprocess_environment(
            {'s3': object()},
            {'events': table},
            source_specs={'events': object()},
            data_persist_dir='/tmp/persist',
            table_data_versions={'events': 'v1'},
        )


def test_prepare_direct_sources_leaves_a_databricks_query_to_the_reader(
        monkeypatch, tmp_path):
    """A Databricks Custom SQL graph table is staged as Kumo Parquet before
    ingest, so it contributes no direct source and no credential profile:
    the engine is handed nothing it could open a warehouse connection with."""
    from kumoapi.data_source import ConnectorType

    import rest.utils.driver_common as driver_common
    from rest.config.table_v2 import CustomSQLSourceTable

    monkeypatch.setattr(
        driver_common, 'get_connector_from_spec',
        lambda _reference: pytest.fail(
            'a staged Databricks query must not open a connector here'))
    table = SimpleNamespace(
        cols=[SimpleNamespace(name='ID')],
        is_sql=True,
        source_table=CustomSQLSourceTable(
            connector_id='dbx',
            connector_type=ConnectorType.DATABRICKS,
            sql='SELECT ID FROM orders',
            table='orders',
        ),
        get_connector_id=lambda: 'dbx',
        get_source_table_name=lambda: 'orders',
    )

    sources = _service().prepare_direct_sources(
        {'dbx': object()},
        {'orders_alias': table},
        data_persist_dir=str(tmp_path / 'persist'),
        table_data_versions={'orders_alias': 'v1'},
        source_size_bytes_by_table={'orders_alias': 456},
    )

    assert sources == {}
    assert not list(tmp_path.glob('*.databricks.cfg'))


def test_has_resumable_ingest_requires_contiguous_encode_chain(tmp_path):
    class _Config:
        data_root = str(tmp_path)

    service = DiskgraphService(_Config())
    handle = DiskgraphHandle(graph_key='graph', graph_data_version='v1')
    marker_dir = (tmp_path / handle.graph_key / handle.graph_data_version /
                  '_tmp')
    marker_dir.mkdir(parents=True)

    assert service.has_resumable_ingest(handle) is False
    (marker_dir / '_phase_complete_encode').touch()
    assert service.has_resumable_ingest(handle) is False
    from kumo_core.diskgraph.runner import persisted_ingest_config_path
    config_path = persisted_ingest_config_path(tmp_path / handle.graph_key /
                                               handle.graph_data_version)
    config_path.write_text('{truncated')
    assert service.has_resumable_ingest(handle) is False
    config_path.write_text(json.dumps({'tables': []}))
    assert service.has_resumable_ingest(handle) is True
    (marker_dir / '_phase_complete_edges_map').touch()
    with pytest.raises(Exception, match='invalid phase marker chain'):
        service.has_resumable_ingest(handle)


def test_persisted_ingest_config_survives_early_retry_cleanup(tmp_path):
    from kumo_core.diskgraph.runner import persisted_ingest_config_path

    class _Config:
        data_root = str(tmp_path)

    service = DiskgraphService(_Config())
    handle = DiskgraphHandle(graph_key='graph', graph_data_version='v1')
    output = tmp_path / handle.graph_key / handle.graph_data_version
    config_path = persisted_ingest_config_path(output)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({'tables': []}))

    assert service.has_persisted_ingest_config(handle)
    service.cleanup_ingest_sidecars(handle, preserve_config=True)
    assert config_path.is_file()

    service.cleanup_ingest_sidecars(handle)
    assert not config_path.exists()
    assert not service.has_persisted_ingest_config(handle)


def _write_persisted_ingest_config(tmp_path, source):
    from kumo_core.diskgraph.runner import persisted_ingest_config_path

    service = DiskgraphService(
        SimpleNamespace(data_root=str(tmp_path / 'diskgraph')))
    handle = DiskgraphHandle(graph_key='graph', graph_data_version='v1')
    config_path = persisted_ingest_config_path(
        Path(service.local_snapshot_path(handle)))
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({'tables': [{'source': source}]}))
    return service, handle, config_path


@pytest.mark.parametrize('source_type', ['parquet', 'csv'])
def test_persisted_ingest_config_requires_all_local_sources(
        tmp_path, source_type, monkeypatch):
    existing = tmp_path / 'staged' / 'part-0'
    existing.parent.mkdir()
    existing.write_bytes(b'data')
    missing = tmp_path / 'staged' / 'missing'

    service, handle, config_path = _write_persisted_ingest_config(
        tmp_path,
        {
            'type': source_type,
            'paths': [str(existing)],
        },
    )
    assert service.has_persisted_ingest_config(handle)

    config_path.write_text(
        json.dumps({
            'tables': [{
                'source': {
                    'type': source_type,
                    'paths': [str(existing), str(missing)],
                },
            }],
        }))
    assert not service.has_persisted_ingest_config(handle)

    monkeypatch.setattr(
        Path,
        'exists',
        lambda _path: (_ for _ in ()).throw(OSError),
    )
    assert not service.has_persisted_ingest_config(handle)


@pytest.mark.parametrize(
    'source',
    [
        {
            'type': 'parquet',
            'paths': ['s3://bucket/object', '@remote-stage'],
        },
        {
            'type': 'object_store_parquet',
        },
    ],
)
def test_persisted_ingest_config_leaves_remote_validation_to_diskgraph(
        tmp_path, source, monkeypatch):
    service, handle, _ = _write_persisted_ingest_config(tmp_path, source)
    monkeypatch.setattr(
        Path,
        'exists',
        lambda _path: (_ for _ in ()).throw(
            AssertionError('remote sources must not be filesystem-probed')),
    )

    assert service.has_persisted_ingest_config(handle)


@pytest.mark.parametrize(
    'paths',
    [
        None,
        [],
        [None],
        ['   '],
        ['s3://[invalid'],
    ],
)
def test_persisted_ingest_config_rejects_unusable_local_paths(tmp_path, paths):
    service, handle, _ = _write_persisted_ingest_config(
        tmp_path,
        {
            'type': 'parquet',
            'paths': paths,
        },
    )

    assert not service.has_persisted_ingest_config(handle)


def test_post_encode_resume_does_not_require_cleaned_staged_sources(tmp_path):
    missing = tmp_path / 'cleaned' / 'part.parquet'
    service, handle, _ = _write_persisted_ingest_config(
        tmp_path,
        {
            'type': 'parquet',
            'paths': [str(missing)],
        },
    )
    marker_dir = Path(service.local_snapshot_path(handle)) / '_tmp'
    marker_dir.mkdir(parents=True)
    (marker_dir / '_phase_complete_encode').write_text('complete')

    assert not service.has_persisted_ingest_config(handle)
    assert service.has_resumable_ingest(handle)


def test_generate_train_table_delegates(monkeypatch):
    import kumo.data.generic.diskgraph_train_table as tt_mod

    captured: dict = {}

    def fake_gen(**kwargs):
        captured.update(kwargs)
        return ('out/path', 'manifest/path')

    monkeypatch.setattr(
        tt_mod,
        'generate_train_table_via_diskgraph',
        fake_gen,
    )
    svc = _service()
    h = DiskgraphHandle(graph_key='gk', graph_data_version='v4')
    result = svc.generate_train_table(
        h,
        query=None,
        timestamps=[],
        task_type='NP',
        forecast_length=0,
        lag_timesteps=None,
        list_distinct_target_is_array=False,
        include_frequency=False,
        entity_time_col=None,
        output_path='/tmp/out',
        split_config={},
        split_family='Temporal',
        downsampling=None,
        train_table_edges_config=None,
        input_timestamp_unit=TimeUnit.Nanoseconds,
        target_stype='numerical',
    )
    assert captured['index_path'] == '/tmp/dg-contract-test/gk/v4'
    assert result.output_path == 'out/path'
    assert result.manifest_path == 'manifest/path'


def test_generate_pred_table_delegates(monkeypatch):
    import kumo.data.generic.diskgraph_batch_pred as bp_mod

    captured: dict = {}

    def fake_gen(**kwargs):
        captured.update(kwargs)
        return (42, 7)

    monkeypatch.setattr(
        bp_mod,
        'generate_batch_pred_table_via_diskgraph',
        fake_gen,
    )
    svc = _service()
    h = DiskgraphHandle(graph_key='gk', graph_data_version='v5')
    result = svc.generate_pred_table(
        h,
        query=None,
        task_type=None,
        graph=None,
        timestamps=[],
        forecast_length=0,
        lag_timesteps=None,
        use_yoy_labels=False,
        list_distinct_target_is_array=False,
        output_root='/tmp/o',
        target_output_root=None,
        input_timestamp_unit=TimeUnit.Nanoseconds,
        target_stype='numerical',
        entity_time_col='DATE',
        lp_bp_layout='separate_lhs_rhs',
    )
    assert captured['index_path'] == '/tmp/dg-contract-test/gk/v5'
    assert captured['entity_time_col'] == 'DATE'
    assert captured['lp_bp_layout'] == 'separate_lhs_rhs'
    assert result.total_rows == 42
    assert result.total_partitions == 7


def test_cleanup_staging_removes_per_job_dirs(tmp_path):
    """``cleanup_staging`` removes the three staging subtrees for each
    job_id under ``data_root``, leaves other dirs alone.
    """
    import os

    class _Cfg:
        data_root = str(tmp_path)
        local_ingest_root = None
        ingest_space_multiplier = 4

    for subdir in ('rehydration_cache', 'edge_persist', 'pred_artifact_cache'):
        for job in ('job-1', 'job-2', 'not-me'):
            d = tmp_path / subdir / job
            d.mkdir(parents=True)
            (d / 'sentinel').write_text('x')

    svc = DiskgraphService(_Cfg())
    svc.cleanup_staging(['job-1', 'job-2'])

    for subdir in ('rehydration_cache', 'edge_persist', 'pred_artifact_cache'):
        assert not os.path.exists(tmp_path / subdir / 'job-1')
        assert not os.path.exists(tmp_path / subdir / 'job-2')
        assert (tmp_path / subdir / 'not-me').is_dir()


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


def test_result_dataclasses_constructible():
    IngestResult(index_path='/i', col_stats_path='/c', ingest_stats_path='/s',
                 kumo_stats_dir='/k')
    TrainTableResult(output_path='/o', manifest_path='/m')
    PredTableResult(total_rows=1, total_partitions=1)
