# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model bundle's ``_provenance.json``; excluded from the bundle id."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from kumo.engine._manifest import (
    FitTiming,
    ModelManifest,
    ModelParents,
    embedded,
    read_manifest,
    stamp,
    stub,
    write_manifest,
)
from kumo.engine.pquery.training_table import TrainingTable
from kumo.engine.trainer._spec import PROVENANCE_ROLE, listed_file


def write_provenance(
    bundle_dir: Path,
    *,
    bundle_id: str,
    train_table: TrainingTable,
    warm_start: Optional[ModelManifest],
    started_at: datetime,
    fit_mode: Optional[str],
) -> None:
    from kumo_core.artifacts import PROVENANCE_FILE, add_bundle_file

    finished_at = datetime.now(timezone.utc)
    table = train_table.manifest
    write_manifest(
        bundle_dir / PROVENANCE_FILE,
        ModelManifest(
            id=bundle_id,
            parents=ModelParents(
                training_table=embedded(table) if table is not None else
                stub('training_table', train_table.id or train_table.tt_id),
                warm_start=None
                if warm_start is None else embedded_model(warm_start)),
            fit=FitTiming(
                started_at=started_at.isoformat(timespec='seconds'),
                finished_at=finished_at.isoformat(timespec='seconds'),
                seconds=round((finished_at - started_at).total_seconds(), 3),
            ),
            fit_mode=fit_mode,
            **stamp(),
        ))
    add_bundle_file(bundle_dir, PROVENANCE_ROLE, PROVENANCE_FILE)


def embedded_model(manifest: ModelManifest) -> ModelManifest:
    r"""``manifest`` for embedding, its own model parents cut to stubs."""
    copy = embedded(manifest)
    parents = copy.parents
    if parents is None:
        return copy

    def cut(model: Optional[ModelManifest]) -> Optional[ModelManifest]:
        return None if model is None else ModelManifest(id=model.id,
                                                        reconstructed=True)

    return replace(copy,
                   parents=replace(parents, warm_start=cut(parents.warm_start)))


def parent_model(bundle_dir: Path, bundle_id: Optional[str]
                 ) -> Optional[ModelManifest]:
    r"""The manifest of the bundle at ``bundle_dir``, or a stub."""
    manifest = read_provenance(bundle_dir)
    if manifest is not None:
        return manifest
    return None if bundle_id is None else stub('model', bundle_id)


def read_provenance(bundle_dir: Path) -> Optional[ModelManifest]:
    path = listed_file(bundle_dir, PROVENANCE_ROLE)
    return None if path is None else read_manifest(path, ModelManifest)
