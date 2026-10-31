# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared manifest owner and local atomic-publication compatibility."""

import json
import ntpath
import posixpath
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from kumo_core.train import artifact_manifest
from kumo_core.train._artifact_storage import _LOCAL_STORAGE, _ManifestStorage
from kumo_core.types import PredictionArtifactType


@pytest.mark.parametrize('path_module', [ntpath, posixpath])
@pytest.mark.parametrize('root', [
    's3://bucket/root', 'gs://bucket/root', r's3://bucket/root\literal',
    'file:///C:/work/root',
])
def test_manifest_uri_join_preserves_keys(monkeypatch, path_module, root):
    monkeypatch.setattr(artifact_manifest, 'osp', path_module)
    assert artifact_manifest._join_path(root, '_attempts', 'part') == (
        root + '/_attempts/part')


@pytest.mark.parametrize('root', [r'C:\work\root', r'\\server\share\root'])
def test_manifest_local_join_uses_host(monkeypatch, root):
    monkeypatch.setattr(artifact_manifest, 'osp', ntpath)
    assert artifact_manifest._join_path(root, '_attempts', 'part') == (
        root + r'\_attempts\part')


@pytest.mark.parametrize('path_module', [ntpath, posixpath])
@pytest.mark.parametrize('suffix', ['root', r'root\literal'])
def test_remote_manifest_roundtrip(monkeypatch, path_module, suffix):
    monkeypatch.setattr(artifact_manifest, 'osp', path_module)
    memory = MemoryFileSystem()
    root = f'memory://manifest-{uuid4().hex}/{suffix}'
    generation = root + '/_attempts/runtime-00000/producer'
    artifact_root = generation + '/predictions.parquet'
    memory.pipe(artifact_root + '/part-00000-00000.parquet', b'data')

    def write_absent(path, text):
        if memory.exists(path):
            return False
        memory.pipe(path, text.encode())
        return True

    storage = _ManifestStorage(
        fs=SimpleNamespace(
            filesize=memory.size,
            read_binary=memory.cat_file,
            exists=memory.exists,
            listdir=lambda path: memory.ls(path, detail=False),
            write_text_if_absent=write_absent,
            rm=lambda path, ignore_errors: memory.rm(path, recursive=True),
        ),
        save_json_atomic=lambda value, path: memory.pipe(
            path, json.dumps(value).encode()),
    )
    kind = PredictionArtifactType.PREDICTIONS
    inventory = artifact_manifest.PredictionArtifactInventory(
        artifact_type=kind, root=artifact_root, file_count=1,
        schema_fingerprint='a' * 64, num_rows=1, num_cols=1, sample_rows=1,
        sample_prefix_logical_bytes=(0, 4), size_bytes=4, physical_size_bytes=4)
    shard = artifact_manifest.PredictionShardManifest(
        schema_version=artifact_manifest.PREDICTION_ARTIFACT_MANIFEST_VERSION,
        job_id='job', attempt_id='producer', graph_snapshot_id='snapshot',
        source_config_sha256='1' * 64, runtime_config_sha256='2' * 64,
        producer_abi=artifact_manifest.PREDICTION_PRODUCER_ABI,
        canonical_output_root=root, generation_root=generation,
        runtime_shard_id=0, runtime_shard_count=1, output_partition_start=0,
        output_partition_stop=1, requested_artifacts=(kind, ),
        artifacts=(inventory, ))
    ref = artifact_manifest.publish_shard_manifest(shard, _storage=storage)
    selected = artifact_manifest.publish_prediction_selection(
        [ref], job_id='job', selection_id='selection',
        selection_attempt_id='control', canonical_output_root=root,
        graph_snapshot_id='snapshot', source_config_sha256='1' * 64,
        expected_shards=1, expected_output_partitions=1, _storage=storage)
    loaded = artifact_manifest.load_prediction_selection(selected,
                                                         _storage=storage)
    # A manifest written on either host must also validate on the other.
    monkeypatch.setattr(artifact_manifest, 'osp',
                        posixpath if path_module is ntpath else ntpath)
    current = artifact_manifest.load_current_prediction_selection(
        root, _storage=storage)
    assert loaded == current
    assert current.artifacts[0].roots == (artifact_root, )
    assert selected.manifest_path == (
        root + '/_selections/selection/attempts/control/_selection_manifest.json')


@pytest.mark.parametrize('root, child', [
    ('C:/work/predictions', 'C:/work/predictions/part.parquet'),
    (r'C:\work\predictions', r'C:\work\predictions\part.parquet'),
    ('C:/work/predictions', r'C:\work\predictions\part.parquet'),
    ('C:/work/predictions', 'c:/work/predictions/part.parquet'),
    ('file:///C:/work/predictions', 'file:///c:/work/predictions/part.parquet'),
    ('file:///C:/work/predictions',
     r'file:///C:/work/predictions\part.parquet'),
    ('s3://bucket/predictions', 's3://bucket/predictions/part.parquet'),
])
def test_windows_owned_descendants(monkeypatch, root, child):
    monkeypatch.setattr(artifact_manifest, 'osp', ntpath)
    artifact_manifest.require_owned_path(child, root)


@pytest.mark.parametrize('candidate', [
    'C:/work/predictions',
    'C:/work/predictions-sibling/part.parquet',
    'C:/work/predictions/../outside/part.parquet',
    r'C:/work/predictions\..\outside/part.parquet',
    'D:/work/predictions/part.parquet',
    's3://bucket/work/predictions/part.parquet',
])
def test_windows_owned_paths_reject_escape(monkeypatch, candidate):
    monkeypatch.setattr(artifact_manifest, 'osp', ntpath)
    with pytest.raises(ValueError):
        artifact_manifest.require_owned_path(candidate, 'C:/work/predictions')


@pytest.mark.parametrize('path_module', [ntpath, posixpath])
def test_remote_owned_paths_reject_escape(monkeypatch, path_module):
    monkeypatch.setattr(artifact_manifest, 'osp', path_module)
    for candidate in ('s3://bucket/root', 's3://bucket/root-sibling/part',
                      's3://bucket/root/../outside/part',
                      's3://other/root/part', 'gs://bucket/root/part',
                      r's3://bucket/root\private.parquet'):
        with pytest.raises(ValueError):
            artifact_manifest.require_owned_path(candidate, 's3://bucket/root')


@pytest.mark.parametrize('path_module', [ntpath, posixpath])
def test_remote_literal_backslashes(monkeypatch, path_module):
    monkeypatch.setattr(artifact_manifest, 'osp', path_module)
    root = r's3://bucket/root\private'
    artifact_manifest.require_owned_path(root + '/part.parquet', root)
    with pytest.raises(ValueError):
        artifact_manifest.require_owned_path(
            's3://bucket/root/private/part.parquet', root)
    artifact_manifest.require_owned_path(
        r's3://bucket/root/part\name.parquet', 's3://bucket/root')


@pytest.mark.parametrize('path_module', [ntpath, posixpath])
@pytest.mark.parametrize('prefix', ['', 'file:///'])
def test_drive_case_preserves_directories(monkeypatch, path_module, prefix):
    monkeypatch.setattr(artifact_manifest, 'osp', path_module)
    root = prefix + 'C:/work/Predictions'
    artifact_manifest.require_owned_path(
        prefix + 'c:/work/Predictions/part.parquet', root)
    with pytest.raises(ValueError):
        artifact_manifest.require_owned_path(
            prefix + 'c:/work/predictions/part.parquet', root)


def test_local_pointer_publication_does_not_replace_existing_selection(
        tmp_path):
    path = str(tmp_path / 'pointer.json')
    assert _LOCAL_STORAGE.fs.write_text_if_absent(path, '{"selected": 1}')
    assert not _LOCAL_STORAGE.fs.write_text_if_absent(path, '{"selected": 2}')
    assert json.loads(_LOCAL_STORAGE.fs.read_binary(path)) == {'selected': 1}
    assert list(tmp_path.iterdir()) == [tmp_path / 'pointer.json']


def test_local_publication_syncs_file_and_directory(tmp_path, monkeypatch):
    import os
    import stat

    from kumo_core.train import _artifact_storage as storage
    calls = []
    original = os.fsync

    def record(fd):
        calls.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        original(fd)

    monkeypatch.setattr(os, 'fsync', record)
    storage._LOCAL_STORAGE.save_json_atomic({'test': 1},
                                            str(tmp_path / 'manifest.json'))
    assert calls == ([False, True] if hasattr(os, 'O_DIRECTORY') else [False])
    calls.clear()
    assert storage._LOCAL_STORAGE.fs.write_text_if_absent(
        str(tmp_path / 'pointer.json'), '{}')
    assert calls == ([False, True] if hasattr(os, 'O_DIRECTORY') else [False])


def test_consumer_and_full_validation_reject_same_invalid_filename(tmp_path):
    from types import SimpleNamespace

    import pytest
    from kumo_core.train.artifact_manifest import (
        _validate_selected_artifact_roots,
        selected_prediction_artifact_files,
    )
    (tmp_path / 'renamed.parquet').write_bytes(b'unchanged count')
    artifact = SimpleNamespace(roots=(str(tmp_path), ), file_count=1)
    for validate in (_validate_selected_artifact_roots,
                     selected_prediction_artifact_files):
        with pytest.raises(ValueError, match='invalid data'):
            validate(artifact)
