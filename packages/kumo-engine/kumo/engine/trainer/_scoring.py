# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Scoring defaults stored as ``scoring-defaults.json`` in a model bundle."""

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from kumo.engine._manifest import atomic_write
from kumo.engine.trainer._spec import MANIFEST_FILE, SCORING_DEFAULTS_ROLE

SCORING_DEFAULTS_FILE = 'scoring-defaults.json'


@dataclass(frozen=True)
class ScoringDefaults:
    """Prediction arguments a bundle applies when ``predict`` omits them."""

    binary_classification_threshold: Optional[float] = None
    num_classes_to_return: Optional[int] = None


def bundle_task_type(artifact_paths: Mapping[str, str]) -> Any:
    from kumo_core.types import TaskType

    schema = json.loads(
        Path(artifact_paths['prediction_schema']).read_text(encoding='utf-8'))
    return TaskType(schema['task_type'])


def validate_scoring_defaults(defaults: ScoringDefaults,
                              task_type: Any) -> None:
    from kumo_core.prediction_options import (
        resolve_classification_threshold,
        resolve_num_classes_to_return,
    )
    from kumo_core.types import TaskType

    threshold = defaults.binary_classification_threshold
    if threshold is not None:
        if task_type is not TaskType.BINARY_CLASSIFICATION:
            raise ValueError(
                'binary_classification_threshold applies only to binary '
                f'classification, not {task_type.value}')
        if isinstance(threshold, bool) or not isinstance(
                threshold, (int, float)):
            raise ValueError('binary_classification_threshold must be a '
                             f'number, but got {threshold!r}')
        resolve_classification_threshold(task_type, threshold)
    if defaults.num_classes_to_return is not None:
        resolve_num_classes_to_return(task_type,
                                      defaults.num_classes_to_return)


def write_scoring_defaults(bundle_dir: Path,
                           defaults: ScoringDefaults) -> None:
    from kumo_core.artifacts import add_bundle_file, remove_bundle_file

    if defaults == ScoringDefaults():
        remove_bundle_file(bundle_dir, SCORING_DEFAULTS_ROLE)
        return
    atomic_write(bundle_dir / SCORING_DEFAULTS_FILE,
                 (json.dumps(asdict(defaults), indent=2, sort_keys=True) +
                  '\n').encode('utf-8'))
    add_bundle_file(bundle_dir, SCORING_DEFAULTS_ROLE, SCORING_DEFAULTS_FILE)


def read_scoring_defaults(
        bundle_dir: Path, artifact_paths: Mapping[str,
                                                  str]) -> ScoringDefaults:
    try:
        manifest = json.loads(
            (bundle_dir / MANIFEST_FILE).read_text(encoding='utf-8'))
        relative = next((entry['relative_path']
                         for entry in manifest.get('artifacts') or []
                         if entry.get('role') == SCORING_DEFAULTS_ROLE), None)
        if relative is None:
            return ScoringDefaults()
        data = json.loads((bundle_dir / relative).read_text(encoding='utf-8'))
        defaults = ScoringDefaults(data.get('binary_classification_threshold'),
                                   data.get('num_classes_to_return'))
        validate_scoring_defaults(defaults, bundle_task_type(artifact_paths))
        return defaults
    except Exception as error:
        warnings.warn(
            f'Ignoring the scoring defaults in {bundle_dir}: {error}',
            UserWarning, stacklevel=3)
        return ScoringDefaults()
