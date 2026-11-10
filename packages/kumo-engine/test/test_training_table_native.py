# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable worker publications retain metadata without uploading native staging."""

from pathlib import Path
from unittest.mock import Mock

import orjson
import pytest
from kumo_core.model_plan import TrainingTableGenerationPlan

import kumo.engine as kumo
from kumo.engine._training_table_io import _inventory, store_prepared
from kumo.engine.cluster import (
    _ComputeWorker,
    _relocate_tree,
    _TrainingTablePublication,
)


def test_worker_skips_already_restored_table(tmp_path: Path) -> None:
    """Repeated broadcasts reuse a worker's completed local table."""
    worker = object.__new__(_ComputeWorker)
    table = kumo.TrainingTable(
        "query",
        tmp_path,
        tmp_path,
        tmp_path,
        (),
        TrainingTableGenerationPlan(),
        id="table",
    )
    worker.tables = {table.id: table}
    worker._local_training_table = Mock(return_value=table)
    worker.download_training_table(
        _TrainingTablePublication(table, "snapshot", str(tmp_path), None), Mock()
    )
    assert worker.tables["table"] is table


def test_relocation_rewrites_paths_and_preserves_binary(tmp_path: Path) -> None:
    """Restored staging uses local paths while retaining its Parquet bytes."""
    destination = tmp_path / "restored"
    destination.mkdir()
    (destination / "spec.json").write_bytes(
        orjson.dumps({"path": "/original/rows.parquet"})
    )
    (destination / "rows.parquet").write_bytes(b"parquet payload")
    _relocate_tree(Path("/original"), destination, destination)
    assert orjson.loads((destination / "spec.json").read_bytes()) == {
        "path": str(destination / "rows.parquet")
    }
    assert (destination / "rows.parquet").read_bytes() == b"parquet payload"


def test_prepared_publication_keeps_files_and_commits_after_barrier(
    tmp_path: Path,
) -> None:
    """Fit extensions use one upload and commit only after every replica is ready."""
    task = tmp_path / "task"
    task.mkdir()
    (task / "data.bin").write_bytes(b"prepared bytes")
    settings = kumo.MLOps(str(tmp_path / "artifacts"))
    calls: list[str] = []

    def uploaded(directory: str, entry: object) -> None:
        calls.append("upload")

    def finished() -> None:
        assert not list((tmp_path / "artifacts").rglob("manifest.json"))
        calls.append("barrier")

    store_prepared(tmp_path, (task,), settings, uploaded, finished)
    assert calls == ["upload", "barrier"]
    assert len(list((tmp_path / "artifacts").rglob("manifest.json"))) == 1
    assert (task / "data.bin").read_bytes() == b"prepared bytes"


def test_empty_publication_is_rejected(tmp_path: Path) -> None:
    """Empty extensions cannot be published as a completed artifact bundle."""
    with pytest.raises(ValueError, match="empty"):
        _inventory(tmp_path, (tmp_path,))


def test_portable_restore_preserves_settings_and_weights(
    tmp_path: Path,
    snapshot: kumo.GraphSnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoration retains effective settings, weights and table identity."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyarrow.fs import LocalFileSystem

    worker = object.__new__(_ComputeWorker)
    worker.root = tmp_path / "worker"
    worker.root.mkdir()
    worker.tables = {}
    worker.snapshots = {snapshot.id: snapshot}
    exported = tmp_path / "exported"
    exported.mkdir()
    (exported / "_SUCCESS").write_bytes(b"")
    pq.write_table(
        pa.table(
            {
                "ENTITY1": [1, 2, 3],
                "TARGET1": [True, False, True],
                "SPLIT": [0, 1, 2],
                "WEIGHT": [0.5, 1.0, 2.0],
            }
        ),
        exported / "part-00000.parquet",
    )
    plan = TrainingTableGenerationPlan(lag_timesteps=7, year_over_year=True)
    original_root = Path("/producer")
    original_dir = original_root / "train_tables" / "original"
    original = kumo.TrainingTable(
        "PREDICT users.label FOR EACH users.id",
        original_dir / "tt.parquet",
        original_dir / "manifest.json",
        original_dir,
        ("train", "val", "test"),
        plan,
        id="original",
    )
    generated_dir = worker.root / "generated"
    generated_dir.mkdir()
    (generated_dir / "manifest.json").write_bytes(
        orjson.dumps({"path": str(generated_dir / "tt.parquet")})
    )
    (generated_dir / "tt.parquet").write_bytes(b"staged rows")
    restored = kumo.TrainingTable(
        original.query,
        generated_dir / "tt.parquet",
        generated_dir / "manifest.json",
        generated_dir,
        original.required_stages,
        TrainingTableGenerationPlan(),
        id="temporary",
    )
    reconstruct = Mock(return_value=restored)
    monkeypatch.setattr(kumo.TrainingTable, "from_source_table", reconstruct)
    monkeypatch.setattr(
        "kumo.engine.connector.file_store",
        Mock(return_value=(LocalFileSystem(), str(exported))),
    )
    worker.download_training_table(
        _TrainingTablePublication(original, snapshot.id, str(original_root), None),
        kumo.MLOps(str(tmp_path / "store")),
    )
    supplied = reconstruct.call_args.kwargs
    assert supplied["plan"].lag_timesteps == 7
    assert supplied["plan"].year_over_year is True
    assert reconstruct.call_args.args[1].split_col == "SPLIT"
    assert reconstruct.call_args.args[1].weight_col == "WEIGHT"
    result = worker.tables[original.id]
    assert result.id == original.id
    assert result.generation_plan == plan
    assert result.required_stages == original.required_stages
    assert result.staging_dir == worker.root / "train_tables" / "original"
    assert orjson.loads(result.manifest_path.read_bytes())["path"] == str(
        result.train_table_path
    )
