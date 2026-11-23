# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import kumo.settings
from kumo._managed_paths import _ManagedIndexPaths, _SnapshotPaths


def test_paths_use_explicit_root(tmp_path):
    paths = _ManagedIndexPaths("build-a", root=tmp_path)
    assert paths.index_dir == tmp_path / "indexes" / "build-a"
    assert paths.task_dir("task-1") == tmp_path / "tasks" / "build-a" / "task-1"
    assert paths.cache_dir == tmp_path / "cache" / "build-a"


def test_explicit_root_is_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(kumo.settings, "_configured", tmp_path / "global")
    paths = _ManagedIndexPaths("build-a", root=tmp_path / "scoped")
    monkeypatch.setattr(kumo.settings, "_configured", tmp_path / "reconfigured")
    assert paths.index_dir == tmp_path / "scoped" / "indexes" / "build-a"
    assert paths.task_dir("task-1") == tmp_path / "scoped" / "tasks" / "build-a" / "task-1"
    assert paths.cache_dir == tmp_path / "scoped" / "cache" / "build-a"


@pytest.mark.parametrize("bad_id", ["", " ", ".", "..", "../other", "nested/task", "a\\b", "D:task"])
def test_ids_are_path_components(tmp_path, bad_id):
    with pytest.raises(ValueError):
        _ManagedIndexPaths(bad_id, root=tmp_path)
    with pytest.raises(ValueError):
        _ManagedIndexPaths("build", root=tmp_path).task_dir(bad_id)


def test_snapshot_paths_share_the_configured_root(tmp_path):
    paths = _SnapshotPaths.for_root(tmp_path)
    assert paths.data_root == tmp_path / "snapshots"
    assert paths.graph_dir("graph") == tmp_path / "snapshots/graph"
    assert paths.version_dir("graph", "version") == tmp_path / "snapshots/graph/version"
    reference = paths.publication_reference("graph", "version")
    assert reference == tmp_path / "snapshots/graph/version.index.json"
    assert paths.reference_version(reference.name) == "version"


def test_snapshot_paths_preserve_a_custom_legacy_root(tmp_path):
    paths = _SnapshotPaths(tmp_path / "legacy-location")
    assert paths.version_dir("graph", "old") == tmp_path / "legacy-location/graph/old"
    assert paths.reference("graph", "old.index.json") == (
        tmp_path / "legacy-location/graph/old.index.json.index.json")
    with pytest.raises(ValueError, match="must not end with"):
        paths.publication_reference("graph", "old.index.json")


@pytest.mark.parametrize("name, version", [
    ("version.index.json", "version"),
    ("version.index.json.index.json", "version.index.json"),
    ("version", None), (".index.json", None),
    ("../other.index.json", None), ("a:b.index.json", None),
])
def test_snapshot_reference_names(name, version):
    assert _SnapshotPaths.reference_version(name) == version


@pytest.mark.parametrize("bad_id", ["", " ", ".", "..", "../other", "nested/task", "a\\b", "D:task"])
def test_snapshot_components_cannot_escape_the_root(tmp_path, bad_id):
    paths = _SnapshotPaths.for_root(tmp_path)
    with pytest.raises(ValueError):
        paths.graph_dir(bad_id)
    with pytest.raises(ValueError):
        paths.version_dir("graph", bad_id)
    with pytest.raises(ValueError):
        paths.reference("graph", bad_id)
