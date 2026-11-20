# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from diskgraph.managed_paths import ManagedIndexPaths


def test_managed_layout_and_identity(tmp_path):
    paths = ManagedIndexPaths(tmp_path, "build-a")
    assert paths.index_dir == tmp_path / "indexes" / "build-a"
    assert paths.task_dir("task-1") == tmp_path / "tasks" / "build-a" / "task-1"
    assert paths.cache_dir == tmp_path / "cache" / "build-a"

    paths.index_dir.mkdir(parents=True)
    paths.task_dir("task-1").mkdir(parents=True)
    (paths.index_dir / "index_metadata.json").write_text(json.dumps({"index_build_id": "build-a"}))
    (paths.task_dir("task-1") / "index_metadata.json").write_text(
        json.dumps({"parent_index_build_id": "build-a"})
    )
    paths.verify_task("task-1")

    (paths.task_dir("task-1") / "index_metadata.json").write_text(
        json.dumps({"parent_index_build_id": "build-b"})
    )
    with pytest.raises(ValueError, match="parent index build ID mismatch"):
        paths.verify_task("task-1")


@pytest.mark.parametrize("bad_id", ["", " ", ".", "..", "../other", "nested/task", "a\\b"])
def test_managed_ids_are_single_components(tmp_path, bad_id):
    with pytest.raises(ValueError):
        ManagedIndexPaths(tmp_path, bad_id)
    with pytest.raises(ValueError):
        ManagedIndexPaths(tmp_path, "build").task_dir(bad_id)
