# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared manifest owner and local atomic-publication compatibility."""

import json
import ntpath

import pytest
from kumo_core.train import artifact_manifest
from kumo_core.train._artifact_storage import _LOCAL_STORAGE


@pytest.mark.parametrize('root, child', [
    ('C:/work/predictions', 'C:/work/predictions/part.parquet'),
    (r'C:\work\predictions', r'C:\work\predictions\part.parquet'),
    ('C:/work/predictions', r'C:\work\predictions\part.parquet'),
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


@pytest.mark.parametrize('path_module', [ntpath, artifact_manifest.osp])
def test_remote_owned_paths_reject_escape(monkeypatch, path_module):
    monkeypatch.setattr(artifact_manifest, 'osp', path_module)
    for candidate in ('s3://bucket/root', 's3://bucket/root-sibling/part',
                      's3://bucket/root/../outside/part',
                      's3://other/root/part', 'gs://bucket/root/part'):
        with pytest.raises(ValueError):
            artifact_manifest.require_owned_path(candidate, 's3://bucket/root')


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
