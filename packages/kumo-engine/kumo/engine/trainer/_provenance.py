# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit record in ``provenance.json``; excluded from the bundle id."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from kumo.engine._manifest import (
    BundleParents,
    BundleProvenance,
    BundleTrainingTable,
    FitTiming,
    SourceTableEntry,
    read_manifest,
    stamp,
    write_manifest,
)
from kumo.engine.graph import GraphSnapshot
from kumo.engine.pquery.training_table import TrainingTable
from kumo.engine.trainer._spec import PROVENANCE_ROLE, listed_file


def write_provenance(
    bundle_dir: Path,
    *,
    bundle_id: Optional[str],
    train_table: TrainingTable,
    warm_start_bundle_id: Optional[str],
    started_at: datetime,
    fit_mode: Optional[str],
    snapshot: Optional[GraphSnapshot] = None,
) -> None:
    from kumo_core.artifacts import PROVENANCE_FILE, add_bundle_file

    finished_at = datetime.now(timezone.utc)
    write_manifest(
        bundle_dir / PROVENANCE_FILE,
        BundleProvenance(
            id=bundle_id,
            parents=BundleParents(training_table=train_table.tt_id,
                                  warm_start=warm_start_bundle_id),
            fit=FitTiming(
                started_at=started_at.isoformat(timespec='seconds'),
                finished_at=finished_at.isoformat(timespec='seconds'),
                seconds=round((finished_at - started_at).total_seconds(), 3),
            ),
            training_table=_training_table(train_table),
            source_tables=_source_tables(snapshot),
            fit_mode=fit_mode,
            **stamp(),
        ))
    add_bundle_file(bundle_dir, PROVENANCE_ROLE, PROVENANCE_FILE)


def _source_tables(
        snapshot: Optional[GraphSnapshot]
) -> Optional[Dict[str, SourceTableEntry]]:
    manifest = None if snapshot is None else snapshot.manifest
    if manifest is None or manifest.reconstructed:
        return None
    return manifest.parents.source_tables


def _training_table(train_table: TrainingTable) -> BundleTrainingTable:
    manifest = train_table.manifest
    return BundleTrainingTable(
        source=train_table.source_mode,
        snapshot=None if manifest is None else manifest.parents.snapshot,
        source_table=None
        if manifest is None else manifest.parents.source_table)


def record_derivation(bundle_dir: Path, parent_bundle_id: Optional[str]) -> None:
    from kumo_core.artifacts import PROVENANCE_FILE, add_bundle_file

    provenance = read_provenance(bundle_dir)
    if provenance is None:
        if listed_file(bundle_dir, PROVENANCE_ROLE) is not None:
            raise ValueError('its provenance.json is unreadable or from a '
                             'newer kumo-engine, so it is left unchanged')
        provenance = new_manifest('model_bundle', None, {},
                                  created_at=None, environment=None,
                                  scheduler=None, reconstructed=True)
    provenance['parents'] = {
        **(provenance.get('parents') or {}), 'derived_from': parent_bundle_id
    }
    write_manifest(bundle_dir / PROVENANCE_FILE, provenance)
    add_bundle_file(bundle_dir, PROVENANCE_ROLE, PROVENANCE_FILE)


def read_provenance(bundle_dir: Path) -> Optional[BundleProvenance]:
    path = listed_file(bundle_dir, PROVENANCE_ROLE)
    return None if path is None else read_manifest(path, BundleProvenance)
