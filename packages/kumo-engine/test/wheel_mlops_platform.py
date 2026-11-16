#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared lifecycle inside a Snowflake notebook."""

import json
import math
import os
import re
import subprocess
import sys
from functools import cache, cached_property
from pathlib import Path
from uuid import uuid4

NOTEBOOK_CONTEXT = 'KUMO_MLOPS_NOTEBOOK_CONTEXT'
NOTEBOOK_ATTEMPT = 'KUMO_MLOPS_NOTEBOOK_ATTEMPT'
IDENTITY_SQL = 'SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE()'
FIELDS = {
    'platform',
    'artifact_root',
    'experiment',
    'model_prefix',
    'identity',
    'database',
    'schema',
    'warehouse',
    'role',
    'accelerator',
    'execution_context',
    'environment',
    'denied_artifact_root',
}
COMMON = {
    'platform', 'artifact_root', 'experiment', 'model_prefix', 'accelerator',
    'execution_context', 'environment'
}


class PlatformUnavailable(ImportError):
    """The selected platform cannot execute in this environment."""


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def access_error(error):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, 'response', {})
        if (getattr(error, 'sqlstate', None) in ('42501', '28000')
                or getattr(error, 'status_code', None) in (401, 403)
                or getattr(response, 'status_code', None) in (401, 403)):
            return True
        error = error.__cause__ or error.__context__
    return False


def denied_access(error):
    """Require denial evidence, not the filesystem wrapper exception's name."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, 'response', None)
        if (getattr(error, 'sqlstate', None) == '42501'
                or getattr(error, 'status_code', None) == 403
                or getattr(response, 'status_code', None) == 403):
            return True
        error = error.__cause__ or error.__context__
    return False


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',
                                                      value):
        raise ValueError(f'Expected an unquoted test identifier, got {value!r}')
    return f'"{value.upper()}"'


def validate(config):
    """Validate references only; never load credentials or construct a client."""
    if not isinstance(config, dict) or set(config) - FIELDS:
        raise ValueError('Profile must contain only the documented fields')
    missing = COMMON - set(config)
    if missing or not all(
            isinstance(value, str) and value.strip()
            for value in config.values()):
        raise ValueError(f'Missing or empty profile fields: {sorted(missing)}')
    if config['platform'] != 'snowflake':
        raise ValueError('Only the Snowflake notebook profile is supported')
    if config['accelerator'] != 'cpu':
        raise ValueError('Snowflake notebook validation requires CPU')
    root = config['artifact_root']
    if any(part in ('..', '.')
           for part in root.split('/')) or '?' in root or '#' in root:
        raise ValueError(
            'Artifact roots cannot contain traversal or URL parameters')
    if not re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,23}', config['model_prefix']):
        raise ValueError('Use a short alphanumeric model prefix')
    for name in ('database', 'schema', 'warehouse', 'experiment',
                 'model_prefix'):
        identifier(config.get(name))
    if config.get('role'):
        identifier(config['role'])
    if (config.get('identity') != 'notebook'
            or config['execution_context'] != 'snowflake_notebook'
            or config['environment'] != 'managed'):
        raise ValueError(
            'Snowflake requires the managed notebook execution path')
    stage = r'@[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*/[A-Za-z0-9_/-]+'
    if not re.fullmatch(stage, root):
        raise ValueError(
            'Use an explicit @DATABASE.SCHEMA.STAGE/test-prefix root')
    denied = config.get('denied_artifact_root')
    if denied:
        other = dict(config, artifact_root=denied)
        other.pop('denied_artifact_root')
        validate(other)
        if denied.rstrip('/') == root.rstrip('/'):
            raise ValueError(
                'The denied root must differ from the writable test root')
        if denied.split('/', 1)[0].upper() == root.split('/', 1)[0].upper():
            raise ValueError(
                'The denied root must use a separate Snowflake stage')
    return dict(config)


def prepare(root, filename, attempt=None):
    config = validate(json.loads(filename.read_text()))
    if attempt is None:
        attempt = os.environ.get(NOTEBOOK_ATTEMPT)
    if attempt is not None and (not isinstance(attempt, str)
                                or not re.fullmatch('[a-f0-9]{12}', attempt)):
        raise ValueError('Invalid notebook lifecycle attempt')
    state = dict(config=config, attempt=attempt or uuid4().hex[:12])
    (root / 'platform.json').write_text(json.dumps(state, indent=2) + '\n')


@cache
def backend(root):
    path = root / 'platform.json'
    if not path.is_file():
        return None
    state = json.loads(path.read_text())
    config = validate(state['config'])
    if not re.fullmatch('[a-f0-9]{12}', state['attempt']):
        raise ValueError('Invalid attempt identifier')
    return Snowflake(config, state['attempt'])


class Snowflake:
    def __init__(self, config, attempt):
        self.config = config
        self.attempt = attempt
        self.experiment = (config['experiment'] + '_' + attempt).upper()
        self.model = (config['model_prefix'] + '_' + attempt).upper()
        self.artifact_root = config['artifact_root'].rstrip('/') + '/' + attempt

    @property
    def accelerator(self):
        return self.config['accelerator']

    def placement(self):
        try:
            launch = json.loads(os.environ.get(NOTEBOOK_CONTEXT, '{}'))
        except ValueError as error:
            raise PlatformUnavailable(
                'Invalid notebook launch context') from error
        if (not isinstance(launch, dict)
                or not str(launch.get('session', '')).isdecimal()
                or launch.get('notebook_extension') is not True
                or launch.get('workspace')
                != os.environ.get('SNOWFLAKE_WORKSPACE_ID')
                or not isinstance(launch.get('identity'), list)
                or len(launch['identity']) != 3):
            raise PlatformUnavailable(
                'Launch Snowflake validation from the notebook')
        from kumo_snowflake.spcs import SPCS_TOKEN_PATH
        if (not Path(SPCS_TOKEN_PATH).is_file()
                or not os.environ.get('SNOWFLAKE_ACCOUNT')
                or not os.environ.get('SNOWFLAKE_HOST')):
            raise PlatformUnavailable(
                'Notebook Container Runtime must expose fresh-process authentication'
            )
        return dict(
            context='snowflake_notebook', notebook=launch,
            identity='mounted runtime token; verified against notebook')

    def negative(self):
        import kumo.engine as kumo

        denied = self.config.get('denied_artifact_root')
        if not denied:
            raise PlatformUnavailable(
                'Configure denied_artifact_root for a real provider permission test'
            )
        try:
            kumo.MLOps(
                (self.connector, denied.rstrip('/') + '/' + self.attempt),
                on_error='raise')
        except PermissionError as error:
            if denied_access(error):
                return dict(denied_artifact_access='passed',
                            diagnostic=str(error))
            raise PlatformUnavailable(
                'Probe failed without permission-denial evidence') from error
        raise AssertionError('The denied artifact root was writable')

    @staticmethod
    def history_api():
        from snowflake.ml.experiment import ExperimentTracking

        if not callable(getattr(ExperimentTracking, 'get_metric_history',
                                None)):
            raise PlatformUnavailable(
                'Snowflake experiment SDK requires get_metric_history()')
        return ExperimentTracking

    @cached_property
    def connector(self):
        import kumo.engine as kumo

        kwargs = {name: self.config[name] for name in ('warehouse', 'database')}
        kwargs['schema_name'] = self.config['schema']
        if self.config.get('role'):
            kwargs['role'] = self.config['role']
        launch = self.placement()['notebook']
        connector = kumo.SnowflakeConnector.from_spcs_env(**kwargs)
        observed = list(connector.session().sql(IDENTITY_SQL).collect()[0])
        if observed != launch['identity']:
            raise PlatformUnavailable(
                'Worker account/user/role differs from the notebook session')
        return connector

    def settings(self, register):
        import kumo.engine as kumo
        from kumo.engine.integrations.snowflake import SnowflakeTracking

        self.placement()
        self.history_api()
        return kumo.MLOps(
            (self.connector, self.artifact_root),
            tracking=SnowflakeTracking(self.connector,
                                       experiment_name=self.experiment,
                                       database=self.config['database'],
                                       schema_name=self.config['schema']),
            registry=kumo.ModelRegistryLocation(self.model)
            if register else None, on_error='raise')

    def rows(self, session, sql):
        return [{
            str(k).lower(): v
            for k, v in row.as_dict().items()
        } for row in session.sql(sql).collect()]

    def record(self, cfg, ref):
        require(ref is not None, 'Expected a Snowflake tracking run')
        session = cfg.tracking.session
        experiment = '.'.join(
            identifier(name)
            for name in (self.config['database'], self.config['schema'],
                         self.experiment))
        run = identifier(ref.run_id)
        rows = self.rows(session, f'SHOW RUNS IN EXPERIMENT {experiment}')
        selected = [row for row in rows if row['name'] == ref.run_id]
        require(
            len(selected) == 1, 'Snowflake tracking run missing or duplicated')
        metadata = selected[0]['metadata']
        metadata = json.loads(metadata) if isinstance(metadata,
                                                      str) else metadata
        require(
            metadata.get('status') == 'FINISHED',
            'Snowflake tracking run did not finish')
        parameters = self.rows(
            session,
            f'SHOW RUN PARAMETERS IN EXPERIMENT {experiment} RUN {run}')
        points = self.rows(
            session, f'SHOW RUN METRICS IN EXPERIMENT {experiment} RUN {run}')
        tracking = self.history_api()(session,
                                      database_name=self.config['database'],
                                      schema_name=self.config['schema'])
        tracking.set_experiment(self.experiment,
                                database_name=self.config['database'],
                                schema_name=self.config['schema'])
        full_history = [{
            str(k).lower(): v
            for k, v in row.as_dict().items()
        } for row in tracking.get_metric_history(run_name=ref.run_id).collect()]
        history = {}
        metrics = {point['name']: float(point['value']) for point in points}
        for point in sorted(full_history, key=lambda p: int(p['step'])):
            name, value = point['name'], float(point['value'])
            history.setdefault(name, []).append((int(point['step']), value))
        return dict(params={p['name']: str(p['value'])
                            for p in parameters}, metrics=metrics,
                    history=history, run=ref.run_id, experiment=experiment,
                    url=None)

    def registered(self, cfg, version, bundle):
        from snowflake.ml.registry import Registry

        registry = Registry(cfg.tracking.session,
                            database_name=self.config['database'],
                            schema_name=self.config['schema'])
        metrics = registry.get_model(self.model).version(version).show_metrics()
        require(
            metrics.get('kumo_bundle_id') == bundle,
            'Snowflake registry version refers to another bundle')

    def set_alias(self, cfg, version):
        import kumo.engine as kumo

        self.connector.sink_backend.set_model_alias(
            self.connector,
            kumo.ModelRegistryLocation(self.model, version=version), 'champion')

    def alias(self, cfg):
        from snowflake.ml.registry import Registry

        registry = Registry(cfg.tracking.session,
                            database_name=self.config['database'],
                            schema_name=self.config['schema'])
        return str(
            registry.get_model(self.model).version('CHAMPION').version_name)


def fit_record(profile, cfg, result):
    record = profile.record(cfg, result.tracking)
    for name in ('train/loss', 'val/auroc'):
        points = record['history'].get(name, [])
        require(
            len({step
                 for step, _ in points}) >= 2
            and all(math.isfinite(value) for _, value in points),
            f'Missing multi-epoch metric history: {name}')
    require(record['params'].get('kumo.bundle_id') == result.id,
            'Fit tracking refers to another bundle')
    profile.registered(cfg, result.registered_model.version, result.id)
    return record


def prediction_record(profile, cfg, run, expected):
    record = profile.record(cfg, run.tracking)
    require(
        record['metrics'].get('pred/rows') == run.scored_rows
        and all(math.isfinite(value) for value in record['metrics'].values()),
        'Platform prediction tracking lost its row metrics')
    require(
        all(
            record['params'].get(key) == value for key, value in {
                'kumo.model_name': profile.model,
                'kumo.model_version': expected['version'],
                'kumo.bundle_id': expected['bundle'],
                'kumo.prediction_id': run.id,
            }.items()),
        'Platform prediction tracking records the wrong identity')
    return record


def drift_record(profile, cfg, run, drift):
    record = profile.record(cfg, drift.tracking)
    require(
        record['params'].get('kumo.prediction_id') == run.id
        and record['params'].get('kumo.bundle_id') == run.bundle_id
        and record['params'].get('kumo.drift_previous')
        == drift.previous_prediction_id,
        'Platform drift tracking records another prediction/model/reference')
    for name in ('psi', 'ks', 'wasserstein'):
        expected = getattr(drift, 'previous_' + name)
        require(
            expected is not None and math.isclose(
                record['metrics'].get(f'drift/{name}_vs_prev', math.nan),
                expected, rel_tol=0, abs_tol=1e-6),
            'Platform drift metric differs from the previous-prediction result')
    return record


def run_notebook(*, config, wheel, expected_version, output_dir,
                 python_executable=None, phase=None, attempt=None):
    """Launch fresh lifecycle processes from a Snowflake Container Runtime notebook."""
    validate(json.loads(Path(config).read_text()))
    if phase not in (None, 'negative'):
        raise ValueError('The standalone notebook phase must be negative')
    from IPython import get_ipython
    from snowflake.snowpark.context import get_active_session

    shell = get_ipython()
    extensions = getattr(getattr(shell, 'extension_manager', None), 'loaded',
                         ())
    if 'snowflake_notebook_utils.ipython_execution_context_monitor_extension' not in extensions:
        raise PlatformUnavailable(
            'Run in a Snowflake Workspace notebook kernel')
    session = get_active_session()
    session_id = str(session.sql('SELECT CURRENT_SESSION()').collect()[0][0])
    if not session_id.isdecimal():
        raise PlatformUnavailable('Cannot identify the active notebook session')
    launch = dict(session=session_id,
                  workspace=os.environ.get('SNOWFLAKE_WORKSPACE_ID'),
                  notebook_extension=True,
                  identity=list(session.sql(IDENTITY_SQL).collect()[0]))
    environment = dict(os.environ)
    environment['MPLBACKEND'] = 'Agg'
    environment[NOTEBOOK_CONTEXT] = json.dumps(launch)
    if attempt is not None:
        if not isinstance(attempt, str) or not re.fullmatch(
                '[a-f0-9]{12}', attempt):
            raise ValueError('Invalid notebook lifecycle attempt')
        environment[NOTEBOOK_ATTEMPT] = attempt
    else:
        environment.pop(NOTEBOOK_ATTEMPT, None)
    helper = Path(__file__).with_name('wheel_mlops_lifecycle.py')
    command = [
        str(python_executable or sys.executable), '-I',
        str(helper), '--execute-platform', '--platform-config',
        str(Path(config).resolve()), '--wheel',
        str(Path(wheel).resolve()), '--expected-version', expected_version,
        '--output-dir',
        str(Path(output_dir).resolve())
    ]
    if phase == 'negative':
        Path(output_dir).mkdir(parents=True, exist_ok=False)
        prepare(Path(output_dir), Path(config), attempt=attempt)
        command += ['--phase', phase]
    completed = subprocess.run(command, env=environment, check=False,
                               timeout=300 if phase else None)
    summary = Path(output_dir) / ('negative.json' if phase else 'summary.json')
    return dict(
        returncode=completed.returncode,
        summary=json.loads(summary.read_text()) if summary.is_file() else None)
