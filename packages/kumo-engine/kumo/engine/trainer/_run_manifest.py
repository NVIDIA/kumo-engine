# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Prediction-run manifest and output score statistics."""

import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from kumo_core.types import PredictionArtifactType, TaskType

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
    embedded,
    stamp,
    stub,
)
from kumo.engine.trainer._provenance import embedded_model

# Object stores locate a path themselves; other sinks' file paths are kept.
_OBJECT_STORES = frozenset({'LOCAL_FILE', 'S3', 'GCS', 'ADLS'})


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
           output_config: Any, settings: Any) -> None:
    r"""Export to ``output_config``, record the run (with the outputs
    written, even if the export fails), store it in the artifact root and
    track it."""
    from kumo.engine._tracking import track_prediction

    records = [] if output_config is None else output_records(output_config)
    written: List[PredictionOutput] = []
    error: Optional[BaseException] = None
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
                store(result, manifest, settings)
            track_prediction(manifest, settings.tracking, error)


def store(result: Any, manifest: Optional[PredictionManifest],
          settings: Any) -> None:
    r"""Copy the run's outputs and manifest to
    ``<root>/predictions/<id>/<artifact>/``; best-effort."""
    from kumo_core.train.artifact_manifest import load_prediction_selection

    from kumo.engine.trainer.output_config import PredictionOutputConfig

    try:
        selection = load_prediction_selection(result.selection,
                                              validate_objects=False)
        names = [item.artifact_type.value.lower()
                 for item in selection.artifacts]
        base = f'predictions/{result.id}'
        result._export(
            PredictionOutputConfig(settings.root.connector, output_paths={
                name: settings.root.location(f'{base}/{name}/')
                for name in names
            }), manifest)
    except Exception as error:
        warnings.warn(
            f"Could not store prediction run {result.id} in the artifact "
            f"root: {error}", UserWarning, stacklevel=3)


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
    from kumo_core.train.artifact_manifest import load_prediction_selection

    from kumo.engine._events import query_sha256

    selection = load_prediction_selection(result.selection,
                                          validate_objects=False)
    stats: Dict[str, Dict[str, Any]] = {}
    for item in selection.artifacts:
        stats[item.artifact_type.value.lower()] = {'rows': item.num_rows}
        if item.artifact_type is PredictionArtifactType.PREDICTIONS:
            try:
                files = replace(result, artifact_type=item.artifact_type).files
                stats['predictions'].update(output_stats(files, task_type))
            except Exception as error:
                warnings.warn(
                    f"Recording the prediction manifest without output "
                    f"score statistics: {error}", UserWarning, stacklevel=4)
    model: Optional[ModelManifest] = trainer.manifest
    scoring_snapshot = getattr(snapshot, 'manifest', None)
    return PredictionManifest(
        id=result.id,
        parents=PredictionParents(
            model=(stub('model', trainer.bundle_id)
                   if model is None else embedded_model(model)),
            snapshot=(stub('snapshot', snapshot.id) if scoring_snapshot is None
                      else embedded(scoring_snapshot)),
        ),
        prediction_table=PredictionTableRecord(
            query_sha256=query_sha256(prediction_table.query),
            source=prediction_table.source,
            anchors=[
                datetime.fromtimestamp(ts / 1e9, timezone.utc).isoformat(
                    timespec='seconds') for ts in prediction_table.timestamps
            ],
            rows=prediction_table.row_count,
        ),
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


def output_records(output_config: Any) -> Optional[List[PredictionOutput]]:
    r"""Records of ``output_config``'s destinations; None, with a warning,
    if one cannot be described."""
    try:
        return [
            PredictionOutput(
                artifact=name, destination=destination_record(
                    output_config.connector, destination))
            for name, destination in output_config.destinations().items()
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
    if root is not None:
        location = (f"{root.rstrip('/')}/{destination}" if '://' in root
                    else str(Path(root, destination)))
    elif (kind not in _OBJECT_STORES
          and is_file_destination(connector, destination)):
        location = destination.rstrip('/')
    else:
        location = source_location(connector, destination)
    return OutputDestination(connector=kind, location=location)


def output_stats(files: Sequence[str],
                 task_type: TaskType) -> Dict[str, Any]:
    from kumo.engine._score_stats import (
        SCORE_COLUMNS,
        TOP_SCORE,
        parquet_stats,
        positive_rate,
        top_score_stats,
    )

    if not files or task_type not in SCORE_COLUMNS:
        return {}
    if task_type is TaskType.BINARY_CLASSIFICATION:
        stats = parquet_stats(files, ['True_PROB', 'TARGET_PRED'])
        return {'columns': {'True_PROB': stats['True_PROB']},
                'predicted_positive_rate': positive_rate(stats['TARGET_PRED'])}
    if task_type.has_numerical_target:
        return {'columns': parquet_stats(files, ['TARGET_PRED'])}
    return {'columns': {TOP_SCORE: top_score_stats(files)}}
