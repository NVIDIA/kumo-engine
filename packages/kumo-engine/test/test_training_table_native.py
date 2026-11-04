# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native MLOps publication, verified restoration and legacy fallback."""

from pathlib import Path
from unittest.mock import Mock

import kumo.engine as kumo
import orjson
import pytest
from kumo.engine._scratch import scoped_scratch_dir
from kumo.engine._training_table_io import load_native, store_native
from kumo_core.diskgraph.sampling_artifacts import ArtifactFile
from kumo_core.model_plan import TrainingTableGenerationPlan


def _table(root: Path) -> kumo.TrainingTable:
    staging = root / "staging"
    staging.mkdir()
    rows = staging / "tt.parquet"
    rows.mkdir()
    (rows / "part-00000.parquet").write_bytes(b"immutable row payload")
    manifest = staging / "manifest.json"
    manifest.write_bytes(orjson.dumps({"path": str(rows)}))
    return kumo.TrainingTable(
        query="PREDICT users.target FOR EACH users.id",
        train_table_path=rows,
        manifest_path=manifest,
        staging_dir=staging,
        required_stages=("train", "val", "test"),
        generation_plan=TrainingTableGenerationPlan(),
        id="tt-native",
    )


def test_one_upload_supplies_replicas_and_reload(tmp_path: Path) -> None:
    """Notifications and native reload reference the same immutable stored objects."""
    table = _table(tmp_path)
    settings = kumo.MLOps(str(tmp_path / "artifacts"))
    stored = tmp_path / "artifacts/training_tables/tt-native"
    seen: list[ArtifactFile] = []

    def uploaded(directory: str, entry: ArtifactFile) -> None:
        assert (Path(directory) / entry.name).read_bytes() == (
            table.staging_dir / entry.name
        ).read_bytes()
        seen.append(entry)

    def finished() -> None:
        assert len(seen) == 2
        assert not (stored / "kumo_training_table.json").exists()

    store_native(table, settings, uploaded, finished)
    query = Mock(query=table.query)
    with scoped_scratch_dir(tmp_path / "scratch"):
        restored = load_native(
            settings.root.connector, "training_tables/tt-native", query, Mock()
        )

    assert restored is not None
    assert restored.id == table.id
    assert restored.required_stages == table.required_stages
    assert restored.generation_plan == table.generation_plan
    assert restored.train_table_path != table.train_table_path
    assert (restored.train_table_path / "part-00000.parquet").read_bytes() == (
        table.train_table_path / "part-00000.parquet"
    ).read_bytes()
    assert orjson.loads(restored.manifest_path.read_bytes()) == {
        "path": str(restored.train_table_path)
    }
    objects = [path for path in stored.rglob("*.parquet") if path.is_file()]
    assert len(objects) == 1
    assert (stored / "_SUCCESS").is_file()
    marker = orjson.loads((stored / "kumo_training_table.json").read_bytes())
    assert {entry["name"] for entry in marker["files"]} == {e.name for e in seen}


def test_replica_failure_does_not_commit_native_table(tmp_path: Path) -> None:
    """A failed readiness barrier leaves diagnostics but no published table."""
    table = _table(tmp_path)
    settings = kumo.MLOps(str(tmp_path / "artifacts"))
    with pytest.raises(OSError, match="replica failed"):
        store_native(
            table, settings, Mock(), Mock(side_effect=OSError("replica failed"))
        )
    stored = tmp_path / "artifacts/training_tables/tt-native"
    assert not (stored / "kumo_training_table.json").exists()
    assert not (stored / "_SUCCESS").exists()
    assert sum(path.is_file() for path in stored.rglob("*.parquet")) == 1


@pytest.mark.parametrize("single_file", [False, True])
def test_legacy_export_without_native_marker_falls_back(
    tmp_path: Path,
    single_file: bool,
) -> None:
    """Existing portable MLOps exports keep their normal loader."""
    connector = kumo.FileConnector(root_dir=str(tmp_path))
    if single_file:
        (tmp_path / "legacy").write_bytes(b"portable rows")
    assert load_native(connector, "legacy", Mock(), Mock()) is None


def test_corrupt_native_rows_are_rejected(tmp_path: Path) -> None:
    """Native reload checks the persisted byte identities before exposing rows."""
    table = _table(tmp_path)
    settings = kumo.MLOps(str(tmp_path / "artifacts"))
    store_native(table, settings, Mock(), None)
    stored = tmp_path / "artifacts/training_tables/tt-native"
    next(path for path in stored.rglob("*.parquet") if path.is_file()).write_bytes(
        b"corrupt"
    )
    with (
        scoped_scratch_dir(tmp_path / "scratch"),
        pytest.raises(ValueError, match="checksum"),
    ):
        load_native(
            settings.root.connector,
            "training_tables/tt-native",
            Mock(query=table.query),
            Mock(),
        )


@pytest.mark.parametrize("field", ["file", "train_table_path", "manifest_path"])
def test_native_marker_cannot_escape_restore_root(tmp_path: Path, field: str) -> None:
    """All persisted relative names are validated before allocating restore space."""
    table = _table(tmp_path)
    settings = kumo.MLOps(str(tmp_path / "artifacts"))
    store_native(table, settings, Mock(), None)
    marker_path = (
        tmp_path / "artifacts/training_tables/tt-native/kumo_training_table.json"
    )
    marker = orjson.loads(marker_path.read_bytes())
    if field == "file":
        marker["files"][0]["name"] = "../escape"
    else:
        marker[field] = "../escape"
    marker_path.write_bytes(orjson.dumps(marker))
    with scoped_scratch_dir(tmp_path / "scratch"), pytest.raises(ValueError):
        load_native(
            settings.root.connector,
            "training_tables/tt-native",
            Mock(query=table.query),
            Mock(),
        )
    assert not (tmp_path / "scratch").exists()


def test_failed_republication_preserves_previous_native_table(tmp_path: Path) -> None:
    """A failed new generation keeps the preceding canonical marker loadable."""
    table = _table(tmp_path)
    settings = kumo.MLOps(str(tmp_path / "artifacts"))
    store_native(table, settings, Mock(), None)
    marker_path = (
        tmp_path / "artifacts/training_tables/tt-native/kumo_training_table.json"
    )
    previous = marker_path.read_bytes()
    with pytest.raises(OSError, match="replica failed"):
        store_native(
            table, settings, Mock(), Mock(side_effect=OSError("replica failed"))
        )
    assert marker_path.read_bytes() == previous
    with scoped_scratch_dir(tmp_path / "scratch"):
        restored = load_native(
            settings.root.connector,
            "training_tables/tt-native",
            Mock(query=table.query),
            Mock(),
        )
    assert restored is not None
    assert (restored.train_table_path / "part-00000.parquet").read_bytes() == (
        table.train_table_path / "part-00000.parquet"
    ).read_bytes()


@pytest.mark.parametrize("linked_root", [False, True])
def test_prepared_publication_uploads_each_file_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linked_root: bool
) -> None:
    """Overlapping directories share one inventory and commit after readiness."""
    from kumo.engine import _snapshot_io
    from kumo.engine._training_table_io import store_prepared

    source = tmp_path / "worker"
    if linked_root:
        physical = tmp_path / "worker-disk"
        physical.mkdir()
        source.symlink_to(physical, target_is_directory=True)
    extension, metadata = source / "extension", source / "fit"
    extension.mkdir(parents=True)
    metadata.mkdir()
    (extension / "graph.bin").write_bytes(b"index")
    (metadata / "task.json").write_bytes(b"{}")
    settings = kumo.MLOps(str(tmp_path / "artifacts"), store={"training_table"})
    uploaded = Mock(wraps=_snapshot_io._open_output)
    monkeypatch.setattr(_snapshot_io, "_open_output", uploaded)
    seen = []

    def notify(directory: str, entry: ArtifactFile) -> None:
        assert (Path(directory) / entry.name).read_bytes() == (
            source / entry.name
        ).read_bytes()
        seen.append(entry)

    def ready() -> None:
        assert len(seen) == 2
        assert not list((tmp_path / "artifacts").rglob("manifest.json"))

    store_prepared(source, (extension, metadata, extension), settings, notify, ready)

    assert {entry.name for entry in seen} == {
        "extension/graph.bin",
        "fit/task.json",
    }
    assert len(seen) == 2
    calls = [call.args[1] for call in uploaded.call_args_list]
    assert len(calls) == len(set(calls)) == 3
    marker_path = next((tmp_path / "artifacts").rglob("manifest.json"))
    manifest = orjson.loads(marker_path.read_bytes())
    assert manifest["source_root"] == str(source)
    assert len(manifest["files"]) == 2


@pytest.mark.parametrize("outside", [False, True])
def test_prepared_publication_rejects_invalid_directories(
    tmp_path: Path, outside: bool
) -> None:
    """Empty or unconfined inventories cannot publish files or ready markers."""
    from kumo.engine._training_table_io import store_prepared

    source = tmp_path / "worker"
    source.mkdir()
    directory = tmp_path / "outside" if outside else source / "empty"
    directory.mkdir()
    if outside:
        (directory / "file").write_bytes(b"data")
    settings = kumo.MLOps(str(tmp_path / "artifacts"), store={"training_table"})
    uploaded, ready = Mock(), Mock()

    with pytest.raises(ValueError):
        store_prepared(source, (directory,), settings, uploaded, ready)

    uploaded.assert_not_called()
    ready.assert_not_called()
    assert not (tmp_path / "artifacts").exists()
