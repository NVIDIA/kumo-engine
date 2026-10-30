# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Prediction-run manifest and output score statistics."""

import json
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from kumo_core.summaries import Summary
from kumo_core.types import TaskType
from kumoapi.jobs import WriteMode

from kumo.engine._manifest import (
    ArtifactStats,
    ModelManifest,
    OutputDestination,
    PredictionManifest,
    PredictionOutput,
    PredictionParents,
    PredictionTableRecord,
    RunScoring,
    RunTimings,
    SnapshotManifest,
    TrackingRef,
    embedded,
    stamp,
    stub,
)
from kumo.engine._score_stats import output_counts
from kumo.engine.trainer._provenance import count_edges, embedded_model

# Object stores locate a path themselves; other sinks' file paths are kept.
_OBJECT_STORES = frozenset({'LOCAL_FILE', 'S3', 'GCS'})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def start(*, result: Any, trainer: Any, snapshot: Any, prediction_table: Any,
          task_type: TaskType, scoring: RunScoring,
          started_at: str) -> Optional[PredictionManifest]:
    r"""The run's manifest so far; None, with a warning, if it fails."""
    try:
        return _manifest(result, trainer, snapshot, prediction_table,
                         task_type, scoring, started_at)
    except Exception as error:
        warnings.warn(f"Predicting without a prediction manifest: {error}",
                      UserWarning, stacklevel=3)
        return None


def finish(result: Any, manifest: Optional[PredictionManifest],
           output_config: Any, settings: Any, model: Any = None,
           snapshot: Any = None) -> Optional[TrackingRef]:
    r"""Export to ``output_config``, record the run (with the outputs
    written, even if the export fails), store it in the artifact root and
    track it as a run of ``model``, a `TrainingResult`, on ``snapshot``; a
    storing or tracking failure raises last under ``on_error='raise'``.
    Returns the tracked run."""
    from kumo.engine._mlops import raise_on_error
    from kumo.engine._tracking import track_prediction, tracking_ref

    records = ([] if output_config is None else output_records(
        output_config, result.id))
    written: List[PredictionOutput] = []
    error: Optional[BaseException] = None
    untracked: Optional[BaseException] = None
    unstored: Optional[BaseException] = None
    tracking: Optional[TrackingRef] = None
    try:
        if output_config is not None:
            result._export(
                output_config, None if manifest is None else replace(
                    manifest, outputs=records), records, written)
    except BaseException as raised:
        error = raised
        raise
    finally:
        if manifest is not None:
            manifest = replace(
                manifest, status='failed' if error else 'succeeded',
                outputs=None if records is None else written,
                timings=replace(manifest.timings, finished_at=_now()))
            write(result, manifest)
        if settings is not None:
            if error is None and settings.wants('prediction'):
                try:
                    manifest = store(result, manifest, settings)
                except Exception as failure:
                    unstored = failure
            untracked = track_prediction(
                manifest, settings.tracking, error or unstored, model,
                _scoring_summaries(manifest, settings, snapshot))
            tracked = getattr(settings.tracking, 'log_prediction_run', None)
            if untracked is None and tracked is not None:
                tracking = tracking_ref(settings.tracking)
    untracked = unstored or untracked
    if untracked is not None:
        raise_on_error(settings, untracked,
                       f'Trainer.predict (run {result.id}, outputs written)',
                       stacklevel=4)
    return tracking


def _scoring_summaries(manifest: Optional[PredictionManifest], settings: Any,
                       snapshot: Any) -> List[Summary]:
    r"""The scoring snapshot's graph summaries, and the run's artifacts'
    tags and datasets."""
    from kumo.engine._data_summaries import snapshot_summaries, table_details
    from kumo.engine._manifest import artifact_summaries

    if manifest is None:
        return []
    index = getattr(snapshot, 'index_path', None)
    return [
        *([] if index is None else snapshot_summaries(
            Path(index), snapshot.graph_config)),
        *artifact_summaries(manifest, 'scoring', settings,
                            None if index is None else table_details(
                                Path(index), snapshot.id)),
    ]


def store(result: Any, manifest: Optional[PredictionManifest],
          settings: Any) -> Optional[PredictionManifest]:
    r"""Copy the run's outputs and manifest to
    ``<root>/predictions/<id>/<artifact>/``; returns the manifest, which
    records where, the copies among its outputs."""
    from kumo_core.train.artifact_manifest import load_prediction_selection

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    try:
        selection = load_prediction_selection(result.selection,
                                              validate_objects=False)
        names = [item.artifact_type.value.lower()
                 for item in selection.artifacts]
        base = f'predictions/{result.id}'
        manifest = settings.locate(result._manifest_path, PredictionManifest,
                                   base) or manifest
        stored = [
            PredictionOutput(artifact=name, destination=destination_record(
                settings.root.connector,
                settings.root.location(f'{base}/{name}'))) for name in names
        ]
        if manifest is not None:
            manifest = replace(manifest,
                               outputs=[*(manifest.outputs or []), *stored])
        result._export(
            PredictionOutputConfig(settings.root.connector, output_paths={
                name: settings.root.location(f'{base}/{name}/')
                for name in names
            }, write_mode=WriteMode.OVERWRITE), manifest)
    except Exception as error:
        raise RuntimeError(f"Could not store prediction run {result.id} in "
                           f"the artifact root: {error}") from error
    if manifest is not None:
        write(result, manifest)
    return manifest


def write(result: Any, manifest: PredictionManifest) -> None:
    from kumo.engine._manifest import write_manifest

    try:
        write_manifest(result._manifest_path, manifest)
    except Exception as error:
        warnings.warn(f"Could not write the prediction manifest: {error}",
                      UserWarning, stacklevel=3)


def _manifest(result: Any, trainer: Any, snapshot: Any,
              prediction_table: Any, task_type: TaskType, scoring: RunScoring,
              started_at: str) -> PredictionManifest:
    from kumo_core.prediction import COUNTS_FILE
    from kumo_core.train.artifact_manifest import load_prediction_selection

    from kumo.engine._events import query_sha256

    selection = load_prediction_selection(result.selection,
                                          validate_objects=False)
    counts = Path(result.selection.pointer_path).parent / COUNTS_FILE
    model: Optional[ModelManifest] = trainer.result.manifest
    scoring_snapshot = getattr(snapshot, 'manifest', None)
    return _record_predictions(
        result.id, {
            item.artifact_type.value.lower():
            (item.num_rows, lambda kind=item.artifact_type: replace(
                result, artifact_type=kind).files)
            for item in selection.artifacts
        },
        model=(stub('model', trainer.result.id) if model is None else model),
        snapshot=(stub('snapshot', snapshot.id)
                  if scoring_snapshot is None else scoring_snapshot),
        task_type=task_type,
        table=PredictionTableRecord(
            query_sha256=query_sha256(prediction_table.query),
            source=prediction_table.source,
            anchors=[
                datetime.fromtimestamp(ts / 1e9, timezone.utc).isoformat(
                    timespec='seconds') for ts in prediction_table.timestamps
            ],
            rows=prediction_table.row_count,
            requested_rows=prediction_table.requested_rows,
            scored_rows=prediction_table.row_count,
            skipped_rows=prediction_table.skipped_rows,
            skipped_sample=list(prediction_table.skipped_sample),
            skip_reason=prediction_table.skip_reason,
        ), scoring=scoring, started_at=started_at,
        counts=json.loads(counts.read_text()) if counts.is_file() else None)


def _record_predictions(
    run_id: str,
    outputs: Mapping[str, Tuple[Optional[int], Callable[[], Sequence[str]]]],
    *,
    model: ModelManifest,
    snapshot: SnapshotManifest,
    task_type: TaskType,
    table: PredictionTableRecord,
    scoring: RunScoring,
    started_at: str,
    counts: Optional[Dict[str, Any]] = None,
) -> PredictionManifest:
    r"""The manifest of prediction run ``run_id`` of any ``model``:
    ``outputs`` maps each artifact (``predictions``, ``embeddings``) to its
    row count and a call listing its Parquet files; the predictions record
    ``counts``, their scores' exact `Counts`: the scorers', else counted
    from the files for ``task_type`` on ``model``'s bins (see the column
    contract in mlops.md); ``table`` holds the requested, scored and
    skipped counts."""
    stats: Dict[str, Dict[str, Any]] = {}
    for name, (rows, files) in outputs.items():
        stats[name] = {'rows': rows}
        if name == 'predictions' and counts is None:
            try:
                counted = output_counts(files(), task_type,
                                        count_edges(model, task_type))
                counts = None if counted is None else counted.to_dict()
            except Exception as error:
                warnings.warn(
                    f"Recording the prediction manifest without score "
                    f"counts: {error}", UserWarning, stacklevel=5)
        if name == 'predictions' and counts is not None:
            stats[name]['counts'] = counts
    return PredictionManifest(
        id=run_id,
        parents=PredictionParents(
            model=(model if model.reconstructed else embedded_model(model)),
            snapshot=(snapshot if snapshot.reconstructed else
                      embedded(snapshot)),
        ),
        prediction_table=table,
        scoring=scoring,
        outputs=[],
        status=None,
        timings=RunTimings(started_at=started_at, finished_at=None),
        stats={
            name: ArtifactStats.from_dict(record)
            for name, record in stats.items()
        },
        **stamp(),
    )


def output_records(output_config: Any,
                   run_id: Optional[str]) -> Optional[List[PredictionOutput]]:
    r"""Records of the destinations run ``run_id`` writes to; None, with a
    warning, if one cannot be described."""
    try:
        return [
            PredictionOutput(
                artifact=name, destination=destination_record(
                    output_config.connector, destination))
            for name, destination in output_config._targets(run_id).items()
        ]
    except Exception as error:
        warnings.warn(f"Exporting without a prediction manifest: {error}",
                      UserWarning, stacklevel=4)
        return None


def destination_record(connector: Any, destination: str) -> OutputDestination:
    from kumo.engine._provenance import source_location
    from kumo.engine.connector import is_file_destination

    kind = str(getattr(connector.source_type, 'value', connector.source_type))
    root = getattr(getattr(connector, 'config', None), 'root_dir', None)
    if '://' in destination or destination.startswith(('/', '@')):
        location = destination.rstrip('/')
    elif root is not None:
        location = (f"{root.rstrip('/')}/{destination}" if '://' in root
                    else str(Path(root, destination)))
    elif (kind not in _OBJECT_STORES
          and is_file_destination(connector, destination)):
        location = destination.rstrip('/')
    else:
        location = source_location(connector, destination)
    return OutputDestination(connector=kind, location=location)
