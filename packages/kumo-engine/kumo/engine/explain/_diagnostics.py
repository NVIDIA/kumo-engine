# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded local snapshots of retained training diagnostics."""

import csv
import io
import json
import math
import re
import stat
from pathlib import Path
from typing import Any, TypedDict

from kumo_core.explain.schemas import EntityOutput
from kumo_core.io.json import from_json_dict, to_json_dict
from kumoapi.jobs import AutoTrainerProgress, TrainerProgress
from kumoapi.task import TaskType

from kumo.engine.explain.results import TrainingExplanations

LIMITS = {
    'artifact_bytes': 1024 * 1024,
    'input_bytes': 8 * 1024 * 1024,
    'sample_bytes': 2 * 1024 * 1024,
    'payload_bytes': 4 * 1024 * 1024,
    'samples': 12,
    'catalog': 100,
    'directory_entries': 1000,
    'trials': 64,
    'epochs': 2000,
    'gradient_rows': 2000,
    'cohorts': 200,
    'json_items': 50000,
    'depth': 24,
    'string': 8192,
}


class TrainingPayload(TypedDict):
    schema_version: int
    provenance: dict[str, Any]
    context: dict[str, Any]
    limits: dict[str, int]
    metrics: dict[str, Any]
    history: dict[str, Any]
    trials: dict[str, Any]
    gradient: dict[str, Any]
    cohorts: dict[str, Any]
    samples: dict[str, Any]


class _Limited(ValueError):
    pass


def _encoded(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(',', ':'),
                      ensure_ascii=True).encode('utf-8')


def _check_tree(value: Any) -> None:
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > LIMITS['json_items'] or depth > LIMITS['depth']:
            raise _Limited('JSON structure limit')
        if isinstance(item, str) and len(item) > LIMITS['string']:
            raise _Limited('String length limit')
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError('Nonfinite artifact value')
        if type(item) is int and abs(item) > 2**53 - 1:
            raise ValueError('Integer exceeds exact notebook numeric range')
        if isinstance(item, dict):
            pending.extend(
                (part, depth + 1) for pair in item.items() for part in pair)
        elif isinstance(item, list):
            pending.extend((part, depth + 1) for part in item)


class _Guard:
    def __init__(self, root: str):
        if '://' in root:
            raise ValueError('Widget snapshots require a local retained run')
        try:
            self.root = Path(root).resolve(strict=True)
        except RuntimeError as error:
            raise ValueError('Invalid retained run location') from error
        self.remaining = LIMITS['input_bytes']

    def path(self, path: str | Path) -> Path:
        try:
            resolved = Path(path).resolve()
        except RuntimeError as error:
            raise ValueError('Invalid artifact location') from error
        if not resolved.is_relative_to(self.root):
            raise ValueError('Artifact escapes retained run')
        return resolved

    def read(self, path: str | Path) -> bytes:
        source = self.path(path)
        info = source.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Artifact is not a regular file')
        bound = min(LIMITS['artifact_bytes'], self.remaining)
        if info.st_size > bound:
            raise _Limited('Artifact byte limit')
        with source.open('rb') as stream:
            data = stream.read(bound + 1)
        if len(data) > bound:
            raise _Limited('Artifact byte limit')
        self.remaining -= len(data)
        return data

    def json(self, path: str | Path) -> dict:
        try:
            value = json.loads(self.read(path))
        except RecursionError as error:
            raise _Limited('JSON nesting limit') from error
        _check_tree(value)
        if not isinstance(value, dict):
            raise ValueError('Expected an artifact object')
        return value

    def entries(self, path: Path) -> list[Path]:
        directory = self.path(path)
        if not directory.exists():
            return []
        entries = []
        for entry in directory.iterdir():
            if len(entries) >= LIMITS['directory_entries']:
                raise _Limited('Directory entry limit; coverage is unknown')
            entries.append(entry)
        return entries


def _failure(section: dict, error: Exception) -> None:
    section['status'] = ('limited' if isinstance(error, _Limited) else 'missing'
                         if isinstance(error, FileNotFoundError) else 'invalid')
    section['reason'] = (
        'Artifact byte or count limit' if isinstance(error, _Limited) else
        'Artifact not retained' if isinstance(error, FileNotFoundError) else
        'Artifact cannot be read' if isinstance(
            error, OSError) else 'Artifact contents or location are invalid')


def _progress_rows(raw: dict, trial: str, fit: str) -> list[dict]:
    metrics = raw.get('metrics', {})
    if not isinstance(metrics, dict):
        raise ValueError('Epoch metrics must be an object')
    if len(metrics) > LIMITS['epochs']:
        raise _Limited('Epoch count limit')
    if any(not re.fullmatch(r'0|[1-9][0-9]{0,8}', key) for key in metrics):
        raise ValueError('Invalid recorded epoch key')
    # Reuse the producer schema while preserving the original epoch keys.
    typed = to_json_dict(from_json_dict(raw, TrainerProgress))
    _check_tree(typed)
    return [
        dict(trial=trial, fit=fit, epoch=epoch, **values)
        for epoch, values in typed['metrics'].items()
    ]


def _read_training(guard: _Guard, trial: Path,
                   payload: TrainingPayload) -> tuple[str | None, dict]:
    original_name = trial.name.removesuffix('_refit_trainval')
    if not re.fullmatch(r'version_[0-9]{1,9}', original_name):
        raise ValueError('Invalid selected trial name')
    original = trial.with_name(original_name)
    relative = original.relative_to(guard.root).parts
    if (len(relative) != 4 or relative[0] != 'models'
            or relative[2] != 'automl'):
        raise ValueError('Trial must be a direct models/task/automl child')
    provenance = payload['provenance']
    provenance.update(selected_trial=original_name,
                      explanation_trial=trial.name, refitted_trainval=trial
                      != original)
    try:
        config = guard.json(original / 'config.json')
        refit = config['refit_cfg']['refitted_trainval']
        if type(refit) is not bool or refit != (trial != original):
            raise ValueError('TEST fit does not match recorded refit selection')
        selected = guard.json(original.parent / 'best_trainer.json')
        if Path(selected['best_trainer']).name != original_name:
            raise ValueError('Facade trial does not match recorded selection')
        provenance['selection_verified'] = True
        tune = config.get('tune_metric')
        if isinstance(tune, str) and tune:
            provenance['selection_metric'] = f'val_{tune}'
            try:
                from kumo_core.evaluation.suite import MetricStore
                provenance['selection_direction'] = MetricStore.tune_mode(tune)
            except (ImportError, ValueError, KeyError):
                provenance['selection_direction'] = None
    except (OSError, ValueError, TypeError, KeyError):
        provenance['selection_verified'] = False
        provenance.update(selected_trial=None, refitted_trainval=None)
        provenance[
            'selection_reason'] = 'Selection metadata is missing or invalid'

    for filename, keys in (('training-spec.json', {
            'predictive_query': 'query'
    }), ('manifests/task.json', {
            'task_type': 'task_type'
    })):
        try:
            raw = guard.json(guard.root / 'published-model' / filename)
            for source, destination in keys.items():
                if isinstance(raw.get(source), str):
                    payload['context'][destination] = raw[source]
        except (OSError, ValueError, TypeError):
            pass

    producer_notes = {}
    try:
        if provenance['selection_verified'] and payload['context'].get(
                'task_type') in ('static_link_prediction',
                                 'temporal_link_prediction'):
            for path in dict.fromkeys((original, trial)):
                recorded = config if path == original else guard.json(
                    path / 'config.json')
                plan = guard.json(path / 'model_plan.json')
                if (recorded['module_kwargs']['module'] != 'two_tower'
                        or plan['model_architecture']['module'] != 'two_tower'):
                    raise ValueError('Unverified two_tower module')
            for name, rule in (
                ('gradient', 'Gradient explanations exclude this module.'),
                ('cohorts', 'Column cohorts require a node-prediction module.'),
                ('samples',
                 'Selected-example collection excludes this module.')):
                producer_notes[name] = (
                    'Recorded two-tower link prediction. ' + rule +
                    ' Eligibility alone does not explain missing artifacts.')
    except (OSError, ValueError, TypeError, KeyError):
        pass

    section = payload['metrics']
    try:
        if not provenance['selection_verified']:
            raise ValueError('Metric fit identity cannot be verified')
        values = guard.json(guard.root / 'published-model/metrics.json')
        rows = []
        sources = {}
        for name, value in values.items():
            if value is not None and (type(value) not in (int, float)):
                raise ValueError('Metrics must be numeric or null')
            split = next((key for key in ('train', 'val', 'test')
                          if name.startswith(f'{key}_')), 'unknown')
            fit = ('refit_trainval' if name.startswith('test_refit_') else
                   'original' if split != 'unknown' else 'unknown')
            prefix = 'test_refit_' if fit == 'refit_trainval' else f'{split}_'
            display_name = name.removeprefix(prefix)
            path = (
                (trial if fit == 'refit_trainval' else original) /
                ('test_result.csv' if split == 'test' else 'train_result.json'))
            if path not in sources:
                try:
                    if split == 'test':
                        records = list(
                            csv.reader(
                                io.StringIO(guard.read(path).decode('utf-8'))))
                        if (len(records) != 2
                                or len(records[0]) != len(records[1])
                                or len(set(records[0])) != len(records[0])):
                            raise ValueError('Invalid evaluation metrics CSV')
                        sources[path] = dict(zip(*records))
                    else:
                        sources[path] = guard.json(path)
                except (OSError, ValueError, TypeError, csv.Error):
                    sources[path] = {}
            recorded = sources[path].get('test_' + display_name if split ==
                                         'test' else name)
            origin = 'Unknown metric origin'
            if type(value) in (int, float):
                if split in ('train', 'val') and type(recorded) in (
                        int, float) and recorded == value:
                    origin = 'Last recorded training metrics (epoch unverified)'
                elif split == 'test' and recorded == f'{value:.6f}' and (
                        fit == 'original' or trial != original):
                    origin = 'Selected-checkpoint evaluation'
            rows.append(
                dict(name=display_name, recorded_name=name, value=value,
                     split=split, fit=fit, origin=origin))
        section.update(status='available', reason=None, rows=rows,
                       source='published-model/metrics.json')
    except (OSError, ValueError, TypeError) as error:
        _failure(section, error)

    section = payload['history']
    history_errors = []
    for path in dict.fromkeys((original, trial)):
        try:
            fit = 'original' if path == original else 'refit_trainval'
            raw = guard.json(path / 'train_progress.json')
            section['rows'].extend(_progress_rows(raw, path.name, fit))
        except (OSError, ValueError, TypeError) as error:
            history_errors.append(error)
    if section['rows']:
        section.update(status='partial' if history_errors else 'available',
                       reason=None)
        if history_errors:
            section['reason'] = 'One fit history is missing or invalid'
    elif history_errors:
        _failure(section, history_errors[0])
    else:
        section.update(status='available', reason=None)

    section = payload['trials']
    try:
        raw = guard.json(original.parent / 'auto_train_progress.json')
        trials = raw.get('trial_progress', {})
        if not isinstance(trials, dict):
            raise ValueError('Trial progress must be an object')
        if len(trials) > LIMITS['trials']:
            raise _Limited('Trial count limit')
        for key, progress in trials.items():
            if not re.fullmatch(r'[0-9]{1,9}', key):
                raise ValueError('Invalid recorded trial key')
            _progress_rows(progress, f'version_{key}', 'original')
        typed = to_json_dict(from_json_dict(raw, AutoTrainerProgress))
        _check_tree(typed)
        trials = typed['trial_progress']
        section.update(
            status='available', reason=None, total=raw.get('total_trials', 0),
            completed=raw.get('completed_trials', 0), rows=[
                dict(
                    trial=f'version_{key}', fit='original',
                    selected=(f'version_{key}' == original_name if
                              provenance['selection_verified'] else None), **{
                                  k: v
                                  for k, v in value.items() if k != 'metrics'
                              }) for key, value in trials.items()
            ])
        for row, progress in zip(section['rows'], trials.values()):
            comparison = progress['final_validation_metrics']
            source = 'final_validation_metrics'
            epoch = None
            if comparison is None:
                metrics = progress['metrics']
                epoch = max(metrics, key=int) if metrics else None
                comparison = metrics[epoch][
                    'validation_metrics'] if epoch else {}
                source = 'last_recorded_epoch' if epoch is not None else 'unavailable'
            row.update(comparison_metrics=comparison, comparison_source=source,
                       comparison_epoch=epoch)
    except (OSError, ValueError, TypeError) as error:
        _failure(section, error)
    return payload['context'].get('task_type'), producer_notes


def _read_sample_output(guard: _Guard, root: Path, artifact_id: str,
                        body: dict) -> dict:
    result = dict(status='unknown',
                  reason='Sample or output task type unavailable')
    try:
        raw = guard.json(root / 'output' / f'{artifact_id}.json')
        task = raw.get('task_type')
        graph_task = (body.get('subgraph_store') or {}).get('task_type')
        if task is None or graph_task is None:
            return result
        if task != graph_task:
            raise ValueError('Output task does not match sampled graph')
        TaskType(task)
        result['task_type'] = task
        if task not in ('regression', 'binary_classification'):
            result.update(status='unsupported',
                          reason='Output display is unsupported for this task')
            return result
        for key in ('prediction', 'target'):
            value = raw.get(key)
            if value is not None and type(value) not in (int, float):
                raise ValueError('Expected a numeric scalar or null')
            if task == 'binary_classification' and value is not None and (
                    not 0 <= value <= 1
                    or key == 'target' and value not in (0, 1)):
                raise ValueError('Invalid binary probability or target')
        typed = to_json_dict(from_json_dict(raw, EntityOutput))
        result.update(status='available', reason=None,
                      prediction=typed['prediction'], target=typed['target'])
        absent = [
            name for name in ('prediction', 'target') if typed[name] is None
        ]
        if absent:
            result.update(status='partial',
                          reason='Not recorded: ' + ', '.join(absent))
    except (OSError, ValueError, TypeError) as error:
        _failure(result, error)
    return result


def _read_samples(guard: _Guard, trial: Path,
                  explanations: TrainingExplanations, section: dict,
                  max_samples: int) -> None:
    root = trial / 'subgraph_collector/test'
    mapping = root / 'strategy_mappings.json'
    if not guard.path(
            root /
            'subgraph_json').exists() and not guard.path(mapping).exists():
        raise FileNotFoundError('Sample artifacts not retained')
    paths = guard.entries(root / 'subgraph_json')
    for path in paths:
        if path.suffix == '.json':
            if len(path.stem) > 32:
                raise _Limited('Collector ID length limit')
            guard.path(path)
    if guard.path(mapping).exists():
        raw = guard.json(mapping)
        for ids in raw.get('mappings', {}).values():
            if not isinstance(ids, list):
                raise ValueError('Selection IDs must be lists')
            if len(ids) > LIMITS['directory_entries']:
                raise _Limited('Selection reference count limit')
            if any(not isinstance(key, str) or len(key) > 32 for key in ids):
                raise ValueError('Invalid collector ID')
    references = explanations.samples()
    listed = references[:LIMITS['catalog']]
    coverage = section['coverage']
    coverage.update(references_total=len(references),
                    references_listed=len(listed),
                    available=sum(ref.available for ref in references),
                    missing=sum(not ref.available for ref in references),
                    catalog_excluded=len(references) - len(listed))
    remaining = LIMITS['sample_bytes']
    for reference in listed:
        row = dict(artifact_id=reference.artifact_id,
                   categories=list(reference.categories),
                   original_ids=reference.original_ids,
                   available=reference.available, status='missing',
                   reason='Graph file missing')
        section['references'].append(row)
        if not reference.available:
            continue
        if len(section['bodies']) >= max_samples:
            row.update(status='limited', reason='Sample count limit')
            coverage['excluded'] += 1
            continue
        try:
            path = guard.path(reference.path)
            if path.stat().st_size > min(remaining, LIMITS['artifact_bytes']):
                raise _Limited('Sample byte limit')
            guard.json(path)
            body = to_json_dict(explanations.sample(reference.artifact_id))
            _check_tree(body)
            size = len(_encoded(body))
            if size > remaining:
                raise _Limited('Serialized sample byte limit')
            if body.get('subgraph_store') is None and body.get(
                    'feature_store') is None:
                raise ValueError('Sample has neither structure nor features')
            remaining -= size
            section['bodies'][reference.artifact_id] = body
            row.update(status='loaded', reason=None)
            coverage['loaded'] += 1
        except (OSError, ValueError, TypeError, KeyError,
                AttributeError) as error:
            _failure(row, error)
            coverage['excluded' if isinstance(error, _Limited
                                              ) else 'malformed'] += 1
    section['status'] = ('partial'
                         if coverage['malformed'] or coverage['missing'] else
                         'limited' if coverage['excluded']
                         or coverage['catalog_excluded'] else 'available')
    section['reason'] = None
    for artifact_id, body in section['bodies'].items():
        section['outputs'][artifact_id] = _read_sample_output(
            guard, root, artifact_id, body)


def build_training_payload(explanations: TrainingExplanations, *,
                           max_samples: int = 12) -> TrainingPayload:
    """Preload complete samples from a local run for either notebook transport."""
    if type(max_samples
            ) is not int or not 0 <= max_samples <= LIMITS['samples']:
        raise ValueError(
            f'max_samples must be between 0 and {LIMITS["samples"]}')
    payload: TrainingPayload = dict(
        schema_version=1, limits=dict(LIMITS),
        context=dict(run=Path(explanations.run_root).name[:512] if explanations.
                     run_root else None), provenance=dict(
                         stage='test',
                         generation_failure='Explanation generation failed'
                         if explanations._failure else None),
        **{
            name: dict(status='missing', reason=explanations._reason, rows=[])
            for name in ('metrics', 'history', 'trials', 'gradient', 'cohorts')
        }, samples=dict(
            status='missing', reason=explanations._reason, references=[],
            bodies={}, outputs={},
            coverage=dict(references_total=None, references_listed=0,
                          available=None, loaded=0, missing=None, malformed=0,
                          excluded=0, catalog_excluded=0)))
    if explanations.run_root is None or explanations.trial_root is None:
        for name in ('metrics', 'history', 'trials', 'gradient', 'cohorts',
                     'samples'):
            payload[name]['status'] = 'unavailable'
        return payload
    try:
        guard = _Guard(explanations.run_root)
        trial = guard.path(explanations.trial_root)
        producer_task, producer_notes = _read_training(guard, trial, payload)
    except (OSError, ValueError, TypeError, KeyError) as error:
        for name in ('metrics', 'history', 'trials', 'gradient', 'cohorts',
                     'samples'):
            _failure(payload[name], error)
        return payload

    section = payload['gradient']
    try:
        text = guard.read(trial / 'explain_test/scores.csv').decode('utf-8')
        for index, row in enumerate(csv.reader(io.StringIO(text))):
            if index > LIMITS['gradient_rows']:
                raise _Limited('Gradient row count limit')
            _check_tree(row)
        scores = explanations.gradient_scores()
        rows = [
            dict(table=table.name, column=column, score=score)
            for table in scores.tables.values()
            for column, score in table.columns.items()
        ]
        if any(row['score'] < 0 for row in rows):
            raise ValueError(
                'Negative score is unsupported by magnitude display')
        section.update(status='available', reason=None, rows=rows)
    except (OSError, ValueError, TypeError, AttributeError, csv.Error) as error:
        _failure(section, error)

    section = payload['cohorts']
    try:
        raw = guard.json(trial / 'column_analysis/test/column_analysis.json')
        if len(raw.get('cohorts', [])) > LIMITS['cohorts']:
            raise _Limited('Cohort count limit')
        result = to_json_dict(explanations.column_analysis())
        section.update(status='available', reason=None,
                       task_type=result['task_type'], rows=result['cohorts'])
        payload['provenance']['task_type'] = result['task_type']
        payload['context']['task_type'] = result['task_type']
    except (OSError, ValueError, TypeError, AttributeError) as error:
        _failure(section, error)

    try:
        _read_samples(guard, trial, explanations, payload['samples'],
                      max_samples)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        _failure(payload['samples'], error)
    recorded_tasks = [payload['context'].get('task_type')
                      ] + [(body.get('subgraph_store') or {}).get('task_type')
                           for body in payload['samples']['bodies'].values()]
    if all(task in (None, producer_task) for task in recorded_tasks):
        for name, note in producer_notes.items():
            payload[name]['producer_note'] = note
    for name in ('gradient', 'cohorts', 'samples'):
        section = payload[name]
        if explanations._failure and section['status'] == 'available':
            section.update(status='partial',
                           reason='Explanation generation failed')
    if len(_encoded(payload)) > LIMITS['payload_bytes']:
        for name in ('samples', 'cohorts', 'history', 'trials', 'gradient',
                     'metrics'):
            section = payload[name]
            if name == 'samples':
                section['bodies'] = {}
                section['outputs'] = {}
                for reference in section['references']:
                    if reference['status'] == 'loaded':
                        reference.update(status='limited',
                                         reason='Payload byte limit')
                section['coverage']['excluded'] += section['coverage']['loaded']
                section['coverage']['loaded'] = 0
            else:
                section['rows'] = []
            section.update(status='limited', reason='Payload byte limit')
            if len(_encoded(payload)) <= LIMITS['payload_bytes']:
                break
        if len(_encoded(payload)) > LIMITS['payload_bytes']:
            payload['samples']['references'] = []
            payload['samples']['coverage']['references_listed'] = 0
    return payload
