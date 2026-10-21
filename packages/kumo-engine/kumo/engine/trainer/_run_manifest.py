# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Prediction-run manifest and output score statistics."""

import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from kumo_core.types import PredictionArtifactType, TaskType

MANIFEST_FILE = 'prediction-manifest.json'


def run_manifest(*, result: Any, prediction_table: Any, task_type: TaskType,
                 scoring: Dict[str, Any], source_tables: Optional[Dict],
                 started_at: str) -> Dict[str, Any]:
    from kumo_core.train.artifact_manifest import load_prediction_selection

    from kumo.engine._events import query_sha256
    from kumo.engine._manifest import new_manifest

    selection = load_prediction_selection(result.selection,
                                          validate_objects=False)
    stats = {}
    for item in selection.artifacts:
        stats[item.artifact_type.value.lower()] = {'rows': item.num_rows}
        if item.artifact_type is PredictionArtifactType.PREDICTIONS:
            try:
                files = replace(result, artifact_type=item.artifact_type).files
                stats['predictions'].update(output_stats(files, task_type))
            except Exception as error:
                warnings.warn(
                    f"Recording the prediction manifest without output "
                    f"score statistics: {error}", UserWarning, stacklevel=3)
    return new_manifest(
        'prediction_run', result.run_id, {
            'bundle': result.bundle_id,
            'snapshot': result.snapshot_id,
            'prediction_table': prediction_table.tt_id,
        },
        source_tables=source_tables,
        prediction_table={
            'query_sha256': query_sha256(prediction_table.query),
            'source': prediction_table.source,
            'anchors': [
                datetime.fromtimestamp(ts / 1e9, timezone.utc).isoformat(
                    timespec='seconds') for ts in prediction_table.timestamps
            ],
            'rows': prediction_table.row_count,
        },
        scoring=scoring,
        outputs=[],
        timings={'started_at': started_at, 'finished_at': None},
        stats=stats,
    )


def output_record(connector: Any, artifact: PredictionArtifactType,
                  destination: str) -> Dict[str, Any]:
    from kumo.engine._provenance import source_location

    kind = connector.source_type
    root = getattr(getattr(connector, 'config', None), 'root_dir', None)
    location = (str(Path(root, destination)) if root is not None else
                source_location(connector, destination))
    return {'artifact': artifact.value.lower(),
            'destination': {'connector': str(getattr(kind, 'value', kind)),
                            'location': location}}


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
