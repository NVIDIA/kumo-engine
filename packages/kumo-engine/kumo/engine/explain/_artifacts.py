# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read existing training artifacts without initializing training machinery."""

import csv
import math
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from kumo_core.io import filesystem
from kumo_core.io.json import load_json
from kumoapi.explain.column_analysis import ColumnAnalysisOutput, ColumnCohorts
from kumoapi.explain.common import EntityMappings
from kumoapi.explain.gradient import GraphGradientScore, TableGradientScore
from kumoapi.explain.subgraph import Subgraph
from kumoapi.task import TaskType

from kumo.engine.explain.errors import (
    ExplanationUnavailableError,
    InvalidExplanationError,
    InvalidRunError,
)
from kumo.engine.explain.types import CohortKey, SampleReference

T = TypeVar('T')


@dataclass
class _SelectedTrial:
    best_trainer: str


@dataclass
class _RefitState:
    refitted_trainval: bool


@dataclass
class _TrialState:
    # Only selection metadata is read; ignore unrelated training config fields.
    refit_cfg: _RefitState


def explanation_trial(trial_root: str | Path, *,
                      refitted_trainval: bool) -> str:
    """Use the same TEST-stage refit rule for fit results and reopened runs."""
    root = str(trial_root).rstrip('/')
    return f'{root}_refit_trainval' if refitted_trainval else root


def artifact_root(path: str | Path) -> str:
    """Keep lazy local references independent of subsequent cwd changes."""
    value = str(path)
    if filesystem.is_local(value):
        return str(filesystem.local_path(value)).rstrip('/')
    return value.rstrip('/')


def read_typed(path: str, cls: type[T]) -> T:
    if not filesystem.exists(path):
        raise ExplanationUnavailableError(f'Explanation artifact missing: {path}')
    try:
        value = load_json(path, cls)
        if value is None:
            raise InvalidExplanationError(f'Null explanation artifact: {path}')
        return value
    except (ValueError, TypeError, OSError) as error:
        raise InvalidExplanationError(
            f'Invalid explanation artifact {path}: {error}') from error


def resolve_run(run_root: str | Path) -> tuple[str, str]:
    """Resolve exactly one models/<task>/automl/best_trainer.json record.

    Selection paths are rebased onto the supplied run so copied runs remain
    usable. Only the selected direct trial under the record's AutoML directory
    is accepted; arbitrary paths and multiple task records are ambiguous.
    """
    root = artifact_root(run_root)
    models = posixpath.join(root, 'models')
    if not filesystem.exists(models):
        raise InvalidRunError(
            f'Expected a retained training run with models/: {root}. '
            'A model bundle alone does not contain training explanations.')
    records = []
    for task in filesystem.listdir(models):
        record = posixpath.join(models, posixpath.basename(task), 'automl',
                                'best_trainer.json')
        if filesystem.exists(record):
            records.append(record)
    if len(records) != 1:
        raise InvalidRunError(
            f'Expected one selected-trial record in {root}; '
            f'found {len(records)}. Supply one training run root.')
    record = records[0]
    try:
        selected = read_typed(record, _SelectedTrial).best_trainer.rstrip('/')
        automl = posixpath.dirname(record)
        trial_name = posixpath.basename(selected)
        # A selected path identifies an original trial, never a refit or parent.
        if (not trial_name.startswith('version_')
                or not trial_name.removeprefix('version_').isdigit()):
            raise InvalidRunError(f'Invalid selected trial in {record}')
        if posixpath.basename(posixpath.dirname(selected)) != 'automl':
            raise InvalidRunError(f'Invalid selected trial path in {record}')
        selected_task = posixpath.basename(posixpath.dirname(
            posixpath.dirname(selected)))
        recorded_task = posixpath.basename(posixpath.dirname(automl))
        if selected_task != recorded_task:
            raise InvalidRunError(f'Selected trial belongs to another task: {record}')
        trial = posixpath.join(automl, trial_name)
        if not filesystem.exists(trial):
            raise InvalidRunError(f'Selected trial is not retained: {trial}')
        config = posixpath.join(trial, 'config.json')
        state = read_typed(config, _TrialState)
        return root, explanation_trial(
            trial, refitted_trainval=state.refit_cfg.refitted_trainval)
    except (InvalidExplanationError, ExplanationUnavailableError) as error:
        raise InvalidRunError(f'Invalid run selection: {error}') from error


def read_gradient(path: str) -> GraphGradientScore:
    """Reconstruct shared score types from the producer's six-decimal CSV."""
    if not filesystem.exists(path):
        raise ExplanationUnavailableError(f'Gradient scores missing: {path}')
    scores = GraphGradientScore()
    seen: set[tuple[str, str]] = set()
    try:
        with filesystem.open_file(path, 'rt', encoding='utf-8', newline='') as src:
            reader = csv.DictReader(src)
            if reader.fieldnames != ['table', 'column', 'score']:
                raise ValueError('expected CSV header table,column,score')
            for row in reader:
                table, column = row['table'], row['column']
                if (not table or not column or None in row
                        or row['score'] is None):
                    raise ValueError('incomplete gradient score row')
                key = (table, column)
                if key in seen:
                    raise ValueError(f'duplicate gradient score {key}')
                seen.add(key)
                score = float(row['score'])
                _finite(score, 'gradient score')
                if table not in scores.tables:
                    scores.tables[table] = TableGradientScore(table)
                scores.tables[table].columns[column] = score
        return scores
    except (ValueError, TypeError, OSError, csv.Error) as error:
        raise InvalidExplanationError(
            f'Invalid gradient artifact {path}: {error}') from error


def _finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')


def _vector(values: list[float], length: int, name: str) -> None:
    if len(values) != length:
        raise ValueError(f'{name} length must match cohort labels')
    for value in values:
        _finite(value, name)


def validate_cohort(cohort: ColumnCohorts) -> None:
    if (not cohort.table_name or cohort.table_name == '???'
            or not cohort.col_name or cohort.col_name == '???'
            or cohort.hop < 0):
        raise ValueError('cohort requires table, nonnegative hop, and column')
    length = len(cohort.cohorts)
    if length == 0:
        raise ValueError('cohort labels must not be empty')
    _vector(cohort.populations, length, 'population shares')
    if any(value < 0 or value > 1 for value in cohort.populations):
        raise ValueError('population shares must lie between zero and one')
    if not math.isclose(sum(cohort.populations), 1, abs_tol=1e-5):
        raise ValueError('population shares must sum to one')
    _finite(cohort.prediction_score, 'prediction score')
    if cohort.target_score is not None:
        _finite(cohort.target_score, 'target score')
    predictions, targets = cohort.predictions, cohort.targets
    if isinstance(predictions, dict):
        if not predictions or any(not name for name in predictions):
            raise ValueError('classification predictions require class labels')
        for values in predictions.values():
            _vector(values, length, 'predictions')
        if targets is not None:
            if not isinstance(targets, dict) or targets.keys() != predictions.keys():
                raise ValueError('target classes must match prediction classes')
            for values in targets.values():
                _vector(values, length, 'targets')
    else:
        _vector(predictions, length, 'predictions')
        if targets is not None:
            if isinstance(targets, dict):
                raise ValueError('regression targets must be a vector')
            _vector(targets, length, 'targets')


def read_cohorts(path: str) -> ColumnAnalysisOutput:
    result = read_typed(path, ColumnAnalysisOutput)
    seen: set[CohortKey] = set()
    try:
        for cohort in result.cohorts:
            key = (cohort.table_name, cohort.hop, cohort.col_name)
            if key in seen:
                raise ValueError(f'duplicate cohort key {key}')
            seen.add(key)
            validate_cohort(cohort)
            multiclass = result.task_type in {
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
            }
            if multiclass != isinstance(cohort.predictions, dict):
                raise ValueError('prediction representation does not match task type')
        return result
    except ValueError as error:
        raise InvalidExplanationError(
            f'Invalid cohort artifact {path}: {error}') from error


def _artifact_id(value: str) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError('sample artifact ID must be a nonnegative integer string')
    return value


def sample_paths(root: str) -> dict[str, str]:
    if not filesystem.exists(root):
        return {}
    paths = {}
    for path in filesystem.listdir(root):
        name = posixpath.basename(path)
        if name.endswith('.json') and name[:-5].isascii() and name[:-5].isdigit():
            paths[name[:-5]] = posixpath.join(root, name)
    return paths


def list_samples(root: str) -> tuple[SampleReference, ...]:
    paths = sample_paths(posixpath.join(root, 'subgraph_json'))
    mapping_path = posixpath.join(root, 'strategy_mappings.json')
    mappings = (read_typed(mapping_path, EntityMappings)
                if filesystem.exists(mapping_path) else EntityMappings())
    categories: dict[str, list[str]] = {}
    originals: dict[str, dict[str, str | int]] = {}
    try:
        for category, ids in mappings.mappings.items():
            original = (mappings.orig_mappings.get(category)
                        if mappings.orig_mappings is not None else None)
            if original is not None and len(original) != len(ids):
                raise ValueError('original entity IDs must align with artifact IDs')
            for index, artifact in enumerate(ids):
                _artifact_id(artifact)
                categories.setdefault(artifact, []).append(category)
                if original is not None:
                    originals.setdefault(artifact, {})[category] = original[index]
        return tuple(
            SampleReference(
                artifact_id=artifact,
                path=paths.get(artifact, posixpath.join(
                    root, 'subgraph_json', f'{artifact}.json')),
                available=artifact in paths,
                categories=tuple(categories.get(artifact, [])),
                original_ids=originals.get(artifact, {}),
            ) for artifact in sorted(paths.keys() | categories.keys(), key=int))
    except ValueError as error:
        raise InvalidExplanationError(
            f'Invalid selection metadata {mapping_path}: {error}') from error


def read_sample(root: str, artifact_id: str) -> Subgraph:
    _artifact_id(artifact_id)
    path = posixpath.join(root, 'subgraph_json', f'{artifact_id}.json')
    graph = read_typed(path, Subgraph)
    try:
        store = graph.subgraph_store
        if store is not None:
            nodes = store.nodes
            if store.root_id not in nodes:
                raise ValueError('root node is missing')
            if any(key != node.id for key, node in nodes.items()):
                raise ValueError('node IDs do not match dictionary keys')
            for source, relations in store.edges.items():
                if source not in nodes:
                    raise ValueError('edge source node is missing')
                for destinations in relations.values():
                    if any(target not in nodes for target in destinations):
                        raise ValueError('edge destination node is missing')
            for ids in store.tables.values():
                if any(node not in nodes for node in ids):
                    raise ValueError('table references a missing node')
        if graph.feature_store is not None:
            for node_id, columns in graph.feature_store.features.items():
                if store is not None and node_id not in store.nodes:
                    raise ValueError('features reference a missing node')
                for feature in columns.values():
                    if feature.score is not None:
                        _finite(feature.score, 'feature score')
        return graph
    except ValueError as error:
        raise InvalidExplanationError(
            f'Invalid sampled graph {path}: {error}') from error
