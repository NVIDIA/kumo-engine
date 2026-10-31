# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model bundle's ``_provenance.json``; excluded from the bundle id."""

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kumo_core.score_counts import PROBABILITY_EDGES, Counts
from kumo_core.score_drift import drift_column
from kumo_core.types import TaskType

from kumo.engine._manifest import (
    Dependency,
    FitTiming,
    HoldoutStats,
    ModelManifest,
    ModelParents,
    TrackingRef,
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
    holdout_truncated: Optional[bool],
    fit_mode: Optional[str],
    tracking: Optional[TrackingRef] = None,
    dependencies: Optional[List[Dependency]] = None,
    holdout_counts: Optional[Dict[str, Any]] = None,
    holdout_window: Optional[Tuple[int, int, int, str]] = None,
) -> None:
    from kumo_core.artifacts import PROVENANCE_FILE, add_bundle_file

    finished_at = datetime.now(timezone.utc)
    table = train_table.manifest
    holdout = None if holdout_counts is None or fit_mode == 'refit_full' else (
        HoldoutStats(truncated=holdout_truncated,
                     rows=holdout_rows(holdout_counts), columns={},
                     counts=holdout_counts,
                     window=None if holdout_window is None else list(
                         holdout_window)))
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
            tracking=tracking,
            dependencies=dependencies,
            holdout=holdout,
            **stamp(),
        ))
    add_bundle_file(bundle_dir, PROVENANCE_ROLE, PROVENANCE_FILE)


def embedded_model(manifest: ModelManifest) -> ModelManifest:
    r"""``manifest`` as a child model embeds it: its own model parents cut
    to stubs, so a chain of retrains does not grow each manifest."""
    copy = embedded(manifest)
    parents = copy.parents
    if parents is None:
        return copy

    def cut(model: Optional[ModelManifest]) -> Optional[ModelManifest]:
        return None if model is None else ModelManifest(id=model.id,
                                                        reconstructed=True)

    return replace(copy, parents=replace(
        parents, warm_start=cut(parents.warm_start),
        derived_from=cut(parents.derived_from)))


def parent_model(bundle_dir: Path, bundle_id: Optional[str]
                 ) -> Optional[ModelManifest]:
    r"""The manifest of the bundle at ``bundle_dir``, or a stub."""
    manifest = read_provenance(bundle_dir)
    if manifest is not None:
        return manifest
    return None if bundle_id is None else ModelManifest(id=bundle_id,
                                                        reconstructed=True)


def read_provenance(bundle_dir: Path) -> Optional[ModelManifest]:
    path = listed_file(bundle_dir, PROVENANCE_ROLE)
    return None if path is None else read_manifest(path, ModelManifest)


def record_derivation(bundle_dir: Path, parent_dir: Path,
                      parent_id: Optional[str]) -> None:
    r"""Writes the manifest of ``bundle_dir``, a copy of the bundle at
    ``parent_dir`` with other scoring defaults."""
    from kumo_core.artifacts import (
        PROVENANCE_FILE,
        add_bundle_file,
        bundle_id_from_manifest,
    )

    parent = parent_model(parent_dir, parent_id)
    write_manifest(
        bundle_dir / PROVENANCE_FILE,
        ModelManifest(
            id=bundle_id_from_manifest(bundle_dir),
            parents=ModelParents(
                derived_from=None if parent is None else embedded_model(
                    parent)),
            dependencies=None if parent is None else parent.dependencies,
            holdout=None if parent is None else parent.holdout,
            **stamp(),
        ))
    add_bundle_file(bundle_dir, PROVENANCE_ROLE, PROVENANCE_FILE)


def holdout_rows(counts: Dict[str, Any]) -> int:
    r"""The rows a holdout's score ``counts`` counted."""
    holdout = Counts.from_dict(counts)
    return int(holdout.arrays[drift_column(holdout)].sum())


def holdout_counts(model: Optional[ModelManifest]) -> Optional[Counts]:
    r"""The exact score counts of ``model``'s holdout, if it records them."""
    holdout = None if model is None else model.holdout
    if holdout is None or holdout.counts is None:
        return None
    return Counts.from_dict(holdout.counts)


def count_edges(model: Optional[ModelManifest], task_type: TaskType
                ) -> Optional[Dict[str, Tuple[float, ...]]]:
    r"""The bins a prediction run of ``model`` counts its scores in:
    probabilities in hundredths, with a multilabel entity's top score; a
    recommendation's top score on its holdout's bins, if it has them; or
    the training target's bins its holdout counts; None without a
    holdout's bins."""
    if task_type in (TaskType.BINARY_CLASSIFICATION,
                     TaskType.MULTICLASS_CLASSIFICATION):
        return {'score': PROBABILITY_EDGES}
    if task_type is TaskType.MULTILABEL_CLASSIFICATION:
        return {'score': PROBABILITY_EDGES, 'top_score': PROBABILITY_EDGES}
    holdout = holdout_counts(model)
    if task_type in (TaskType.MULTILABEL_RANKING,
                     TaskType.STATIC_LINK_PREDICTION,
                     TaskType.TEMPORAL_LINK_PREDICTION):
        return {} if holdout is None or 'top_score' not in holdout.edges else {
            'top_score': holdout.edges['top_score']}
    if (task_type.has_numerical_target and holdout is not None
            and 'target' in holdout.edges):
        return {'target': holdout.edges['target']}
    return None
