# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import kumo.settings
from kumo._managed_paths import _ManagedIndexPaths


def test_paths_follow_configured_root(tmp_path, monkeypatch):
    monkeypatch.setattr(kumo.settings, "_configured", tmp_path)
    paths = _ManagedIndexPaths("build-a")
    assert paths.index_dir == tmp_path / "indexes" / "build-a"
    assert paths.task_dir("task-1") == tmp_path / "tasks" / "build-a" / "task-1"
    assert paths.cache_dir == tmp_path / "cache" / "build-a"


@pytest.mark.parametrize("bad_id", ["", " ", ".", "..", "../other", "nested/task", "a\\b", "D:task"])
def test_ids_are_path_components(tmp_path, monkeypatch, bad_id):
    monkeypatch.setattr(kumo.settings, "_configured", tmp_path)
    with pytest.raises(ValueError):
        _ManagedIndexPaths(bad_id)
    with pytest.raises(ValueError):
        _ManagedIndexPaths("build").task_dir(bad_id)
