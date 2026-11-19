# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Locate and read the Kumo files one training run leaves on disk.

Every reader returns ``None`` for a missing or unreadable optional file, so a
card whose inputs are absent is omitted instead of failing the widget.
"""

import io
import json
import logging
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from kumo_core.io import filesystem

from kumo.engine.explain._artifacts import artifact_root, resolve_run
from kumo.engine.explain.errors import InvalidRunError

logger = logging.getLogger(__name__)

REFIT_SUFFIX = '_refit_trainval'


@dataclass(frozen=True)
class RunFiles:
    """Directories of one run, in the shorthand the specification uses.

    ``bundle`` is the published model (``PUB``). ``trial`` is the evaluated
    model's trial directory: the train-plus-validation refit when one ran,
    otherwise the selected trial. ``first`` is the selected trial's first fit.
    ``table`` is the training-table directory and ``snapshot`` the graph
    snapshot. Directories that a bundle alone does not carry are ``None``.
    """

    bundle: str
    trial: Optional[str] = None
    first: Optional[str] = None
    table: Optional[str] = None
    snapshot: Optional[str] = None
    holdout_path: Optional[str] = None
    holdout_truncated: Optional[bool] = None

    @property
    def refit(self) -> bool:
        return self.trial is not None and self.trial.endswith(REFIT_SUFFIX)


def _join(base: Optional[str], *parts: str) -> Optional[str]:
    return None if base is None else posixpath.join(base, *parts)


def _present(path: Optional[str]) -> Optional[str]:
    if not isinstance(path, str):
        return None
    try:
        return path if filesystem.exists(path) else None
    except OSError:
        return None


def locate(bundle_dir: str, run_root: Optional[str] = None,
           trial_root: Optional[str] = None) -> RunFiles:
    """Resolve the run directories around a published bundle."""
    bundle = artifact_root(bundle_dir)
    if trial_root is not None:
        root = artifact_root(run_root) if run_root is not None else None
        trial = artifact_root(trial_root)
    elif run_root is None:
        return RunFiles(bundle)
    else:
        try:
            root, trial = resolve_run(run_root)
        except InvalidRunError as error:
            logger.debug('Training run not readable: %s', error)
            return RunFiles(bundle)
    first = trial.removesuffix(REFIT_SUFFIX)
    manifest = read_json(_join(root, 'task_dataset', 'runtime_manifest.json'))
    table = snapshot = None
    if isinstance(manifest, dict):
        stages, graph = manifest.get('stages'), manifest.get('graph')
        test = stages.get('test') if isinstance(stages, dict) else None
        table = _present(test.get('index_path')) if isinstance(test,
                                                               dict) else None
        snapshot = _present(graph.get('index_path')) if isinstance(
            graph, dict) else None
    return RunFiles(bundle, _present(trial), _present(first), table, snapshot)


def open_run(run_root: str) -> RunFiles:
    """Locate a retained run whose bundle is ``<run>/published-model``."""
    root = artifact_root(run_root)
    return locate(posixpath.join(root, 'published-model'), root)


def read_json(path: Optional[str]) -> Any:
    if _present(path) is None:
        return None
    assert path is not None
    try:
        with filesystem.open_file(path, 'rt', encoding='utf-8') as src:
            return json.load(src)
    except (OSError, ValueError) as error:
        logger.debug('Unreadable JSON %s: %s', path, error)
        return None


def read_parquet(path: Optional[str],
                 columns: Optional[list[str]] = None) -> Optional[pd.DataFrame]:
    if _present(path) is None:
        return None
    assert path is not None
    try:
        with filesystem.open_file(path, 'rb') as src:
            return pd.read_parquet(io.BytesIO(src.read()), columns=columns)
    except (OSError, ValueError, KeyError) as error:
        # A missing column is a missing optional input, not a widget error.
        logger.debug('Unreadable Parquet %s: %s', path, error)
        return None


def read_parquet_dir(path: Optional[str],
                     columns: list[str]) -> Optional[pd.DataFrame]:
    """Concatenate every Parquet part in a directory."""
    if _present(path) is None:
        return None
    assert path is not None
    parts = sorted(name for name in filesystem.listdir(path)
                   if name.endswith('.parquet'))
    frames = [
        read_parquet(posixpath.join(path, posixpath.basename(name)), columns)
        for name in parts
    ]
    if not frames or any(frame is None for frame in frames):
        return None
    return pd.concat(frames, ignore_index=True)


def plan(files: RunFiles) -> dict[str, Any]:
    value = read_json(
        posixpath.join(files.bundle, 'model', 'resolved-model-plan.json'))
    return value if isinstance(value, dict) else {}


def task_type(files: RunFiles) -> Optional[str]:
    schema = read_json(posixpath.join(files.bundle, 'prediction-schema.json'))
    if isinstance(schema, dict) and isinstance(schema.get('task_type'), str):
        return schema['task_type']
    return None


def entity_table(files: RunFiles) -> Optional[str]:
    """The table named in the query's ``FOR EACH``."""
    spec = read_json(posixpath.join(files.bundle, 'training-spec.json'))
    query = spec.get('predictive_query') if isinstance(spec, dict) else None
    if isinstance(query, str):
        match = re.search(r'\bFOR\s+EACH\s+([A-Za-z_][\w]*)\.', query,
                          flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def split_stats(files: RunFiles) -> Optional[dict[str, Any]]:
    value = read_json(_join(files.table, 'split_stats.json'))
    return ({
        key: item
        for key, item in value.items() if isinstance(item, dict)
    } if isinstance(value, dict) else None)


def bundle_file(files: RunFiles, role: str) -> Optional[str]:
    from kumo.engine.trainer._spec import listed_file

    try:
        bundle = Path(files.bundle)
        path = listed_file(bundle, role)
        return (str(path) if path is not None
                and path.resolve().is_relative_to(bundle.resolve()) else None)
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return None


def holdout(files: RunFiles, columns: list[str]) -> Optional[pd.DataFrame]:
    from kumo.engine.trainer._spec import HOLDOUT_ROLE

    frame = read_parquet(files.holdout_path, columns)
    if frame is not None:
        return frame
    return read_parquet(bundle_file(files, HOLDOUT_ROLE), columns)
