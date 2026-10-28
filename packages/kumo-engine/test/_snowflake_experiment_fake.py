# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A recording stand-in for `snowflake.ml.experiment.ExperimentTracking` that
keeps its observable rules (singleton, one active run, LIKE-matched resume)."""

import os
import re
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

from kumo_core.sources.sql import resolve_snowflake_identifier as _identifier

STATUSES = ('FAILED', 'FINISHED')


def _like(pattern: str, name: str) -> bool:
    regex = ''.join('.*' if c == '%' else '.' if c == '_' else re.escape(c)
                    for c in pattern)
    return re.fullmatch(regex, name, re.IGNORECASE | re.DOTALL) is not None


class FakeRun:
    def __init__(self, tracking: 'FakeExperimentTracking', experiment: str,
                 name: str) -> None:
        self._tracking = tracking
        self.experiment_name = experiment
        self.name = name

    def __enter__(self) -> 'FakeRun':
        self._tracking.calls.append(('run.__enter__', self.name))
        self._tracking.lineage_stack.append((self.experiment_name, self.name))
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._tracking.calls.append(('run.__exit__', self.name))
        self._tracking.lineage_stack.pop()
        if self._tracking._run is self:
            self._tracking.end_run(
                status='FAILED' if exc_type is not None else None)


class FakeExperimentTracking:
    _instance: Optional['FakeExperimentTracking'] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> 'FakeExperimentTracking':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, session: Any, *, database_name: Optional[str] = None,
                 schema_name: Optional[str] = None,
                 capture_source_info: bool = True, model_registry: Any = None,
                 warnings_as_errors: bool = False) -> None:
        self.capture_source_info = capture_source_info
        self.warnings_as_errors = warnings_as_errors
        if hasattr(self, '_initialized'):
            self.calls.append(('__init__', database_name, schema_name))
            return
        self._session = session
        self.database = _identifier(database_name
                                    or session.get_current_database())
        self.schema = _identifier(schema_name or session.get_current_schema()
                                  or 'PUBLIC')
        self.calls: List[Tuple[Any, ...]] = [('__init__', database_name,
                                              schema_name)]
        self.fail_on: set = set()
        self.experiment: Optional[str] = None
        self._run: Optional[FakeRun] = None
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.lineage_stack: List[Tuple[str, str]] = []
        self._sql_client = types.SimpleNamespace(
            show_runs_in_experiment=self._show_runs)
        self._initialized = True

    def _show_runs(self, *, experiment_name: str,
                   like: Optional[str] = None) -> List[Dict[str, Any]]:
        self._record('show_runs', experiment_name, like)
        return self._matching(experiment_name, like)

    def _matching(self, experiment: str,
                  like: Optional[str]) -> List[Dict[str, Any]]:
        return [{'name': name} for name, run in self.runs.items()
                if run['experiment'] == experiment
                and (like is None or _like(like, name))]

    def _record(self, *call: Any) -> None:
        self.calls.append(call)
        if call[0] in self.fail_on:
            raise RuntimeError(f'fake Snowflake error in {call[0]}')

    def set_experiment(self, experiment_name: str,
                       database_name: Optional[str] = None,
                       schema_name: Optional[str] = None) -> Any:
        self._record('set_experiment', experiment_name, database_name,
                     schema_name)
        if database_name is not None and schema_name is None:
            schema_name = 'PUBLIC'
        location = (_identifier(database_name or self.database),
                    _identifier(schema_name or self.schema))
        name = _identifier(experiment_name)
        if (name, *location) != (self.experiment, self.database, self.schema):
            self._run = None
        self.experiment = name
        self.database, self.schema = location
        return types.SimpleNamespace(name=name)

    def start_run(self, run_name: Optional[str] = None) -> FakeRun:
        self._record('start_run', run_name)
        if self._run is not None:
            raise RuntimeError('A run is already active. Please end the '
                               'current run before starting a new one.')
        if self.experiment is None:
            self.experiment = 'DEFAULT'
        matches = self._matching(self.experiment,
                                 run_name) if run_name else []
        if matches:
            name = matches[0]['name']
            if self.runs[name]['status'] != 'RUNNING':
                raise RuntimeError(f'Run {run_name} exists but cannot be '
                                   f'resumed as it is no longer running.')
        else:
            name = _identifier(run_name or f'RUN_{len(self.runs) + 1}')
            self.runs[name] = {'experiment': self.experiment,
                               'status': 'RUNNING', 'params': {},
                               'metrics': [], 'artifacts': {}}
        self._run = FakeRun(self, self.experiment, name)
        return self._run

    def end_run(self, run_name: Optional[str] = None,
                status: Optional[str] = None) -> None:
        self._record('end_run', run_name, status)
        if self.experiment is None:
            raise RuntimeError('No experiment set.')
        if self._run is None:
            raise RuntimeError('No run is active.')
        if status is not None and status not in STATUSES:
            raise ValueError(f"Unsupported status value: '{status}'.")
        self.runs[self._run.name]['status'] = status or 'FINISHED'
        self._run = None

    def _active(self) -> Dict[str, Any]:
        if self._run is None:
            self.start_run()
        return self.runs[self._run.name]

    def log_params(self, params: Dict[str, Any]) -> None:
        self._record('log_params', dict(params))
        self._active()['params'].update(
            {key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        self._record('log_metrics', dict(metrics), step)
        self._active()['metrics'].extend(
            (key, float(value), step) for key, value in metrics.items())

    def log_artifact(self, local_path: str,
                     artifact_path: Optional[str] = None) -> None:
        self._record('log_artifact', local_path, artifact_path)
        with open(local_path, encoding='utf-8') as file:
            self._active()['artifacts'][os.path.basename(local_path)] = (
                file.read())


class FakeSession:
    def __init__(self, database: Optional[str] = 'DB',
                 schema: Optional[str] = 'PUBLIC') -> None:
        self.database, self.schema = database, schema
        self.statements: List[str] = []

    def sql(self, query: str) -> Any:
        self.statements.append(query)
        return types.SimpleNamespace(collect=lambda: [])

    def get_current_database(self) -> Optional[str]:
        return self.database

    def get_current_schema(self) -> Optional[str]:
        return self.schema


def install(monkeypatch: Any) -> type:
    """Makes `import snowflake.ml.experiment` return a fresh fake."""
    FakeExperimentTracking._instance = None
    module = types.ModuleType('snowflake.ml.experiment')
    module.ExperimentTracking = FakeExperimentTracking
    for name in ('snowflake', 'snowflake.ml'):
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'snowflake.ml.experiment', module)
    return FakeExperimentTracking
