# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Scoring defaults stored as ``scoring-defaults.json`` in a model bundle."""

import json
import warnings
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Optional

from kumo.engine.trainer._spec import SCORING_DEFAULTS_ROLE, listed_file

SCORING_DEFAULTS_FILE = 'scoring-defaults.json'


@dataclass(frozen=True)
class ScoringDefaults:
    """Prediction arguments a bundle applies when ``predict`` omits them."""

    binary_classification_threshold: Optional[float] = None
    num_classes_to_return: Optional[int] = None


def normalized_scoring_defaults(defaults: ScoringDefaults,
                                task_type: Any) -> ScoringDefaults:
    from kumo_core.prediction_options import (
        resolve_classification_threshold,
        resolve_num_classes_to_return,
    )

    threshold = defaults.binary_classification_threshold
    count = defaults.num_classes_to_return
    if threshold is not None:
        if isinstance(threshold, bool) or not isinstance(threshold, Real):
            raise ValueError('binary_classification_threshold must be a '
                             f'number, but got {threshold!r}')
        threshold = float(threshold)
        if resolve_classification_threshold(task_type, threshold) is None:
            raise ValueError(
                'binary_classification_threshold applies only to binary and '
                f'multilabel classification, not {task_type.value}')
    if isinstance(count, Integral) and not isinstance(count, bool):
        count = int(count)
    if count is not None:
        resolve_num_classes_to_return(task_type, count)
    return ScoringDefaults(threshold, count)


def write_scoring_defaults(bundle_dir: Path, defaults: ScoringDefaults) -> None:
    from kumo_core.artifacts import add_bundle_file, remove_bundle_file
    from kumo_core.io.filesystem import atomic_write_bytes

    if defaults == ScoringDefaults():
        remove_bundle_file(bundle_dir, SCORING_DEFAULTS_ROLE)
        return
    atomic_write_bytes(
        bundle_dir / SCORING_DEFAULTS_FILE,
        (json.dumps(asdict(defaults), indent=2, sort_keys=True) +
         '\n').encode('utf-8'))
    add_bundle_file(bundle_dir, SCORING_DEFAULTS_ROLE, SCORING_DEFAULTS_FILE)


def read_scoring_defaults(bundle_dir: Path,
                          artifact_paths: Mapping[str, str]) -> ScoringDefaults:
    from kumo_core.artifacts import bundle_task_type

    path = listed_file(bundle_dir, SCORING_DEFAULTS_ROLE)
    if path is None:
        return ScoringDefaults()
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return normalized_scoring_defaults(
            ScoringDefaults(data.get('binary_classification_threshold'),
                            data.get('num_classes_to_return')),
            bundle_task_type(artifact_paths))
    except Exception as error:
        warnings.warn(f'Ignoring the scoring defaults in {bundle_dir}: {error}',
                      UserWarning, stacklevel=3)
        return ScoringDefaults()
