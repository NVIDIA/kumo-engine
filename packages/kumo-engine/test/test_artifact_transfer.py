# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared upload completion, retry boundaries and inventory validation."""

import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest
from kumo.engine._snapshot_io import upload_files
from kumo.engine.connector import FileConnector
from kumo_core.diskgraph.sampling_artifacts import ArtifactFile
from pyarrow.fs import LocalFileSystem


def test_callback_runs_after_close_without_final_barrier(tmp_path: Path) -> None:
    """Callbacks can read each completed object before upload_files returns."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "graph").write_bytes(b"data")
    target = tmp_path / "remote"
    seen = []

    def completed(directory: str, entry: ArtifactFile) -> None:
        assert directory == str(target)
        assert (target / entry.name).read_bytes() == b"data"
        seen.append(entry)

    result = upload_files(
        FileConnector(root_dir=str(tmp_path)), source, "remote", {"graph": 4}, completed
    )

    assert tuple(seen) == result
    assert result[0].sha256 == hashlib.sha256(b"data").hexdigest()


def test_retried_upload_has_one_callback_and_renews_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient upload failure renews credentials and does not duplicate notices."""
    (tmp_path / "payload").write_bytes(b"data")
    stale = Mock(wraps=LocalFileSystem())
    stale.open_output_stream.side_effect = OSError("temporary failure")
    fresh = LocalFileSystem()
    location = Mock(
        side_effect=[
            (fresh, str(tmp_path / "remote")),
            (stale, str(tmp_path / "remote")),
            (fresh, str(tmp_path / "remote")),
        ]
    )
    monkeypatch.setattr("kumo.engine._snapshot_io._location", location)
    callback = Mock()

    result = upload_files(
        FileConnector(root_dir=str(tmp_path)),
        tmp_path,
        "remote",
        {"payload": 4},
        callback,
    )

    assert location.call_count == 3
    callback.assert_called_once_with(str(tmp_path / "remote"), result[0])
    assert (tmp_path / "remote" / "payload").read_bytes() == b"data"


def test_callback_failure_does_not_retry_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A download notification failure cannot cause a duplicate upload notice."""
    (tmp_path / "payload").write_bytes(b"data")
    store = Mock(wraps=LocalFileSystem())
    callback = Mock(side_effect=OSError("notification failed"))
    monkeypatch.setattr(
        "kumo.engine._snapshot_io._location",
        Mock(return_value=(store, str(tmp_path / "remote"))),
    )

    with pytest.raises(OSError, match="notification failed"):
        upload_files(
            FileConnector(root_dir=str(tmp_path)),
            tmp_path,
            "remote",
            {"payload": 4},
            callback,
        )

    assert store.open_output_stream.call_count == 1
    callback.assert_called_once()


def test_size_mismatch_never_notifies(tmp_path: Path) -> None:
    """Inventory sizes are authoritative rather than rediscovered during upload."""
    (tmp_path / "payload").write_bytes(b"data")
    callback = Mock()
    with pytest.raises(ValueError, match="inventory"):
        upload_files(
            FileConnector(root_dir=str(tmp_path)),
            tmp_path,
            "remote",
            {"payload": 3},
            callback,
        )
    callback.assert_not_called()


def test_empty_inventory_needs_no_storage_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty transfer does not open storage or notify consumers."""
    location, callback = Mock(), Mock()
    monkeypatch.setattr("kumo.engine._snapshot_io._location", location)
    assert (
        upload_files(
            FileConnector(root_dir=str(tmp_path)), tmp_path, "remote", {}, callback
        )
        == ()
    )
    location.assert_not_called()
    callback.assert_not_called()
