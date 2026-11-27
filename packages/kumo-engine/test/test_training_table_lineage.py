# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reused row artifacts must reflect the current snapshot's source policy."""

from dataclasses import replace
from pathlib import Path
from typing import Literal
from unittest.mock import Mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.artifacts import PROVENANCE_FILE
from kumo_core.model_plan import TrainingTableGenerationPlan
from pyarrow.fs import LocalFileSystem

import kumo.engine as kumo
from kumo.engine._manifest import (
    BuildCost,
    SnapshotManifest,
    SnapshotParents,
    SourceTableEntry,
    TrainingTableManifest,
    TrainingTableParents,
    write_manifest,
)
from kumo.engine.cluster import (
    _ComputeWorker,
    _TrainingTablePublication,
    _TrainingTableRequest,
)


def _snapshot_manifest(identifier: str, record_sources: bool) -> SnapshotManifest:
    """Build source metadata with the selected recording policy for a fixture."""
    return SnapshotManifest(
        id=identifier,
        record_sources=record_sources,
        parents=SnapshotParents(
            source_tables={
                "users": SourceTableEntry(
                    connector="SNOWFLAKE",
                    location="EXAMPLE.PUBLIC.USERS" if record_sources else "users",
                    version="version-1",
                    version_kind="fingerprint",
                    location_fingerprint=None if record_sources else "location-hash",
                    normalized_sql_fingerprint="sql-hash",
                    sql="SELECT * FROM EXAMPLE.PUBLIC.USERS"
                    if record_sources
                    else None,
                    dialect="snowflake" if record_sources else None,
                    namespace=["EXAMPLE", "PUBLIC"] if record_sources else None,
                )
            }
        ),
    )


@pytest.mark.parametrize("reuse", ["cache", "restore"])
@pytest.mark.parametrize(
    "lineage",
    [
        pytest.param((False, True, True), id="redact-previous-sources"),
        pytest.param((True, True, True), id="record-current-sources"),
        pytest.param((False, False, True), id="missing-current-manifest"),
        pytest.param((False, True, False), id="missing-old-parent"),
    ],
)
def test_reused_table_refreshes_snapshot_lineage(
    reuse: Literal["cache", "restore"],
    lineage: tuple[bool, bool, bool],
    tmp_path: Path,
    snapshot: kumo.GraphSnapshot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache and restore refresh provenance without changing row bytes or identity."""
    record_sources, current_manifest_present, old_parent_present = lineage
    snapshot = replace(snapshot, record_sources=record_sources)
    current_manifest = _snapshot_manifest(snapshot.id, record_sources)
    snapshot.index_path.mkdir()
    if current_manifest_present:
        write_manifest(snapshot.index_path / PROVENANCE_FILE, current_manifest)
    worker = object.__new__(_ComputeWorker)
    worker.root = tmp_path / "worker"
    worker.root.mkdir()
    worker.snapshots = {snapshot.id: snapshot}
    worker.tables = {}
    settings = kumo.MLOps(str(tmp_path / "store"))
    producer = Path("/producer")
    original_dir = producer / "train_tables" / "original"
    table = kumo.TrainingTable(
        "PREDICT users.label FOR EACH users.id",
        original_dir / "tt.parquet",
        original_dir / "manifest.json",
        original_dir,
        ("train", "val", "test"),
        TrainingTableGenerationPlan(),
        id="original",
    )
    manifest = TrainingTableManifest(
        id=table.id,
        query=table.query,
        generation_plan={"split": "RandomSplit"},
        task_type="binary_classification",
        split_seed=123,
        build=BuildCost(
            started_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            seconds=1.0,
            disk_bytes=42,
        ),
        parents=TrainingTableParents(
            snapshot=_snapshot_manifest(snapshot.id, not record_sources)
        )
        if old_parent_present
        else None,
    )
    publication = _TrainingTablePublication(table, snapshot.id, str(producer), manifest)
    staging = worker.root / (
        "train_tables/original" if reuse == "cache" else "generated"
    )
    staging.mkdir(parents=True)
    rows = staging / "tt.parquet"
    rows.write_bytes(b"unchanged staged training rows")
    (staging / "manifest.json").write_bytes(b"{}")
    (staging / "train.parquet").write_bytes(b"unchanged split rows")
    row_bytes, row_mtime = rows.read_bytes(), rows.stat().st_mtime_ns
    write_manifest(staging / PROVENANCE_FILE, manifest)
    generate = Mock(side_effect=AssertionError("Cached rows were regenerated"))
    monkeypatch.setattr(kumo.PredictiveQuery, "generate_training_table", generate)
    monkeypatch.setattr("kumo_core.training.export.read_spec", Mock(return_value={}))
    monkeypatch.setattr(
        "kumo_core.training.export.stage_parts",
        Mock(return_value={"train": ("train.parquet",)}),
    )
    store = Mock()
    monkeypatch.setattr(kumo.TrainingTable, "_store", store)
    if reuse == "cache":
        result = worker.training_table_generate(
            _TrainingTableRequest(snapshot.id, table.query, None, publication), settings
        )
        store.assert_called_once_with(settings)
    else:
        exported = tmp_path / "exported"
        exported.mkdir()
        (exported / "_SUCCESS").write_bytes(b"")
        pq.write_table(
            pa.table({"ENTITY1": [1, 2, 3], "TARGET1": [True, False, True]}),
            exported / "part-00000.parquet",
        )
        monkeypatch.setattr(
            "kumo.engine.connector.file_store",
            Mock(return_value=(LocalFileSystem(), str(exported))),
        )
        restored = replace(
            table,
            id="temporary",
            staging_dir=staging,
            train_table_path=rows,
            manifest_path=staging / "manifest.json",
        )
        reconstruct = Mock(return_value=restored)
        monkeypatch.setattr(kumo.TrainingTable, "_from_rows", reconstruct)
        result = worker.download_training_table(publication, settings)
        reconstruct.assert_called_once()
        store.assert_not_called()
    assert result.id == table.id
    assert result.train_table_path.read_bytes() == row_bytes
    assert result.train_table_path.stat().st_mtime_ns == row_mtime
    generate.assert_not_called()
    refreshed = result.manifest
    assert refreshed is not None and refreshed.parents is not None
    parent = refreshed.parents.snapshot
    assert parent.id == snapshot.id
    assert parent.record_sources is record_sources
    if current_manifest_present:
        assert parent == current_manifest
        assert parent.parents is not None
        source = parent.parents.source_tables["users"]
        if not record_sources:
            assert source.location == "users"
            assert source.location_fingerprint == "location-hash"
            assert source.sql is None and source.namespace is None
    else:
        assert parent.reconstructed is True
        assert parent.parents is None
    assert refreshed.query == manifest.query
    assert refreshed.generation_plan == manifest.generation_plan
    assert refreshed.task_type == manifest.task_type
    assert refreshed.split_seed == manifest.split_seed
    assert refreshed.build == manifest.build
