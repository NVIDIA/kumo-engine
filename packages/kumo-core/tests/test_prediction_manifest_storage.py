# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared manifest owner and local atomic-publication compatibility."""

import json

from kumo_core.train._artifact_storage import _LOCAL_STORAGE


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
