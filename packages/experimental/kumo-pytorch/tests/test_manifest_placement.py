# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import kumo.settings
import pytest
from kumo._managed_paths import _ManagedIndexPaths

from kumo.pytorch._storage import verify_task_manifest


@pytest.fixture
def placement(tmp_path):
    paths = _ManagedIndexPaths("build", root=tmp_path.resolve())
    paths.index_dir.mkdir(parents=True)
    paths.task_dir("task").mkdir(parents=True)
    manifest = {
        "graph": {"index_path": str(paths.index_dir)},
        "stages": {
            name: {"index_path": str(paths.task_dir("task"))} for name in ("train", "valid")
        },
    }
    return manifest, dict(paths=paths, index=paths.index_dir, task_id="task")


@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_manifest_accepts_bound_base_and_owned_task_after_reconfiguration(
    placement, snapshot, empty
):
    manifest, options = placement
    paths = options["paths"]
    if snapshot:
        # A core snapshot's producer owns its base placement, including legacy layouts.
        base = paths.root / "snapshots" / "graph-version"
        base.mkdir(parents=True)
        options["index"] = base
        manifest["graph"]["index_path"] = str(base)
    if empty:
        manifest["stages"] = {}
    kumo.settings.configure(scratch_dir=paths.root / "new-root")
    verify_task_manifest(manifest, **options)
    assert not kumo.settings.scratch_dir().exists()


@pytest.mark.parametrize(
    "destination",
    [
        "outside",
        "prefix",
        "other-build",
        "other-task",
        "cache",
        "base",
        "nested",
        "relative",
        "traversal",
    ],
)
def test_manifest_rejects_stage_outside_its_allocated_task(placement, destination):
    manifest, options = placement
    paths = options["paths"]
    task = paths.task_dir("task")
    invalid = {
        "outside": paths.root.parent / "outside",
        "prefix": Path(str(paths.root) + "-other") / "task",
        "other-build": _ManagedIndexPaths("other", root=paths.root).task_dir("task"),
        "other-task": paths.task_dir("other"),
        "cache": paths.cache_dir,
        "base": paths.index_dir,
        "nested": task / "nested",
        "relative": Path("tasks/build/task"),
        "traversal": task / ".." / task.name,
    }
    # Validate every stage, rather than just the first one.
    manifest["stages"]["valid"]["index_path"] = str(invalid[destination])
    with pytest.raises(ValueError, match=r"stages\['valid'\].index_path"):
        verify_task_manifest(manifest, **options)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("destination", ["outside", "other-base", "relative", "traversal"])
def test_manifest_rejects_unbound_graph_even_without_stages(placement, destination, empty):
    manifest, options = placement
    paths = options["paths"]
    base = paths.index_dir
    invalid = {
        "outside": paths.root.parent / "outside",
        "other-base": paths.root / "another-base",
        "relative": Path("indexes/build"),
        "traversal": base / ".." / base.name,
    }
    manifest["graph"]["index_path"] = str(invalid[destination])
    if empty:
        manifest["stages"] = {}
    with pytest.raises(ValueError, match="graph.index_path"):
        verify_task_manifest(manifest, **options)


def test_manifest_rejects_source_bound_outside_captured_root(placement):
    manifest, options = placement
    outside = options["paths"].root.parent / "outside"
    options["index"] = outside
    manifest["graph"]["index_path"] = str(outside)
    with pytest.raises(ValueError, match="captured Kumo root"):
        verify_task_manifest(manifest, **options)


@pytest.mark.parametrize("field", ["graph", "stage"])
@pytest.mark.parametrize("outside", [False, True])
def test_manifest_rejects_bound_path_redirected_by_symlink(placement, field, outside):
    manifest, options = placement
    paths = options["paths"]
    destination = (
        paths.root.parent / f"{paths.root.name}-redirected"
        if outside
        else paths.root / "redirected"
    )
    destination.mkdir()
    path = paths.index_dir if field == "graph" else paths.task_dir("task")
    path.rmdir()
    path.symlink_to(destination, target_is_directory=True)
    with pytest.raises(ValueError, match="without redirects"):
        verify_task_manifest(manifest, **options)
