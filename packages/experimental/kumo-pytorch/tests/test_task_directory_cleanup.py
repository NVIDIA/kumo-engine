# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref
from pathlib import Path

import pytest
from kumo._managed_paths import _ManagedIndexPaths

from kumo.pytorch._storage import TaskDirectory


def test_abandoned_directory_is_collected_without_removing_other_task_data(tmp_path):
    paths = _ManagedIndexPaths("build", root=tmp_path.resolve())
    owned = TaskDirectory(paths, "abandoned")
    other = TaskDirectory(paths, "other")
    path = Path(owned.name)
    (path / "partial-ingest").write_text("incomplete")
    reference = weakref.ref(owned)
    del owned
    gc.collect()
    assert reference() is None
    assert not path.exists()
    assert Path(other.name).is_dir()
    other.cleanup()


def test_directory_cleanup_cannot_remove_replacement_or_preexisting_task(tmp_path):
    paths = _ManagedIndexPaths("build", root=tmp_path.resolve())
    first = TaskDirectory(paths, "task")
    first.cleanup()
    replacement = TaskDirectory(paths, "task")
    marker = Path(replacement.name) / "keep"
    marker.write_text("owned by replacement")
    with pytest.raises(FileExistsError):
        TaskDirectory(paths, "task")
    first.cleanup()
    del first
    gc.collect()
    assert marker.read_text() == "owned by replacement"
    replacement.cleanup()


def test_unclaimed_task_directory_is_cleaned_at_normal_exit(tmp_path):
    import subprocess
    import sys

    script = """
import sys
from pathlib import Path
from kumo._managed_paths import _ManagedIndexPaths
from kumo.pytorch._storage import TaskDirectory
directory = TaskDirectory(_ManagedIndexPaths("build", root=Path(sys.argv[1])), "unclaimed")
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True)
    assert not (tmp_path / "tasks" / "build" / "unclaimed").exists()
