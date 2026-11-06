#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan or execute the shared lifecycle against a named platform profile."""

import argparse
import errno
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from functools import cache, cached_property
from pathlib import Path
from uuid import uuid4

PLATFORMS = ('snowflake', 'databricks', 'sagemaker')
NOTEBOOK_CONTEXT = 'KUMO_MLOPS_NOTEBOOK_CONTEXT'
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
    'catalog',
    'cluster_id',
    'warehouse_id',
    'tracking_arn',
    'role_arn',
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
        code = response.get('Error', {}).get('Code') if isinstance(
            response, dict) else None
        if (getattr(error, 'sqlstate', None) in ('42501', '28000')
                or getattr(error, 'status_code', None) in (401, 403)
                or getattr(response, 'status_code', None) in (401, 403)
                or getattr(error, 'error_code', None)
                in ('PERMISSION_DENIED', 'UNAUTHENTICATED') or code
                in ('AccessDenied', 'AccessDeniedException', 'ExpiredToken',
                    'InvalidClientTokenId', 'UnrecognizedClientException')
                or type(error).__name__
                in ('NoCredentialsError', 'PartialCredentialsError',
                    'ProfileNotFound', 'NoRegionError', 'Unauthenticated',
                    'PermissionDenied')):
            return True
        error = error.__cause__ or error.__context__
    return False


def denied_access(error, *, provider_only=False):
    """Require denial evidence, not the filesystem wrapper exception's name."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        response = getattr(error, 'response', None)
        code = response.get('Error', {}).get('Code') if isinstance(
            response, dict) else None
        if (getattr(error, 'sqlstate', None) == '42501'
                or getattr(error, 'status_code', None) == 403
                or getattr(response, 'status_code', None) == 403 or
            (not provider_only and
             (getattr(error, 'errno', None) in (errno.EACCES, errno.EPERM)
              or code in ('AccessDenied', 'AccessDeniedException')
              or type(error).__name__ == 'PermissionDenied'))):
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
    platform = config['platform']
    if platform not in PLATFORMS:
        raise ValueError('Unknown platform')
    if config['accelerator'] not in ('cpu', 'gpu'):
        raise ValueError('accelerator must be cpu or gpu')
    if config['environment'] not in ('venv', 'managed'):
        raise ValueError('environment must be venv or managed')
    if config['execution_context'] not in ('client', platform,
                                           'snowflake_notebook'):
        raise ValueError(
            'execution_context must be client or the selected platform')
    root = config['artifact_root']
    if any(part in ('..', '.')
           for part in root.split('/')) or '?' in root or '#' in root:
        raise ValueError(
            'Artifact roots cannot contain traversal or URL parameters')
    if not re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,23}', config['model_prefix']):
        raise ValueError('Use a short alphanumeric model prefix')
    if platform == 'snowflake':
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
    elif platform == 'databricks':
        if config['execution_context'] == 'snowflake_notebook':
            raise ValueError('Notebook context is only for Snowflake')
        for name in ('catalog', 'schema'):
            identifier(config.get(name))
        if bool(config.get('cluster_id')) == bool(config.get('warehouse_id')):
            raise ValueError(
                'Databricks requires exactly one cluster_id or warehouse_id')
        if config.get('identity') not in ('ambient', 'service_principal'):
            raise ValueError(
                'Databricks identity must be ambient or service_principal')
        if not root.startswith(
                f'/Volumes/{config["catalog"]}/{config["schema"]}/') or len(
                    root.split('/')) < 6:
            raise ValueError(
                'Use a volume and test prefix in the selected catalog/schema')
        if not config['experiment'].startswith('/'):
            raise ValueError(
                'Use a workspace experiment path with an existing parent')
    else:
        if config['execution_context'] == 'snowflake_notebook':
            raise ValueError('Notebook context is only for Snowflake')
        if not re.fullmatch(r's3://[a-z0-9][a-z0-9.-]+/[A-Za-z0-9_/-]+', root):
            raise ValueError('Use an S3 bucket and test prefix')
        if not re.fullmatch(
                r'arn:aws[a-z-]*:sagemaker:[a-z0-9-]+:[0-9]{12}:(mlflow-tracking-server|mlflow-app)/[A-Za-z0-9-]+',
                config.get('tracking_arn', '')):
            raise ValueError('Use a managed MLflow server or app ARN')
        if '_' in config['model_prefix']:
            raise ValueError(
                'SageMaker model prefixes cannot contain underscores')
    denied = config.get('denied_artifact_root')
    if denied:
        other = dict(config, artifact_root=denied)
        other.pop('denied_artifact_root')
        validate(other)
        if denied.rstrip('/') == root.rstrip('/'):
            raise ValueError(
                'The denied root must differ from the writable test root')
        if (platform == 'snowflake' and denied.split('/', 1)[0].upper()
                == root.split('/', 1)[0].upper()):
            raise ValueError(
                'The denied root must use a separate Snowflake stage')
    return dict(config)


def plan(config):
    config = validate(config)
    return dict(
        platform=config['platform'],
        execution_context=config['execution_context'], configuration=config,
        phases=['negative', 'fit', 'score', 'evaluate'], mutations=[
            'Store artifacts under a new attempt prefix',
            'Create a new experiment and model with two versions',
            'Move champion only on that new test model'
        ], retention='Keep test resources and evidence; no automatic deletion',
        ui_review='Required separately after API execution')


def prepare(root, filename):
    config = validate(json.loads(filename.read_text()))
    state = dict(config=config, attempt=uuid4().hex[:12])
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
    cls = Snowflake if config['platform'] == 'snowflake' else Mlflow
    return cls(config, state['attempt'])


class Backend:
    def __init__(self, config, attempt):
        self.config = config
        self.attempt = attempt
        suffix = '_' if config['platform'] == 'snowflake' else '-'
        self.experiment = config['experiment'] + suffix + attempt
        self.model = config['model_prefix'] + suffix + attempt
        if config['platform'] == 'snowflake':
            self.experiment, self.model = self.experiment.upper(
            ), self.model.upper()
        if config['platform'] == 'databricks':
            self.model = f'{config["catalog"]}.{config["schema"]}.{self.model}'
        self.artifact_root = config['artifact_root'].rstrip('/') + '/' + attempt

    @property
    def accelerator(self):
        return self.config['accelerator']

    def placement(self):
        context = self.config['execution_context']
        if context == 'client':
            return dict(context='client',
                        native_platform_qualification='blocked')
        if context == 'snowflake_notebook':
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
                context=context, notebook=launch,
                identity='mounted runtime token; verified against notebook')
        if context == 'databricks':
            version = os.environ.get('DATABRICKS_RUNTIME_VERSION')
            if not version:
                raise PlatformUnavailable(
                    'Run inside the selected Databricks runtime')
            from databricks.sdk import WorkspaceClient
            user = WorkspaceClient().current_user.me().user_name
            expected = os.environ.get('KUMO_MLOPS_EXPECTED_DATABRICKS_USER')
            if expected and user != expected:
                raise PlatformUnavailable(
                    'Fresh Databricks process resolved another identity')
            return dict(context='databricks', runtime=version, user=user)
        processing = Path('/opt/ml/config/processingjobconfig.json')
        if processing.is_file():
            record = json.loads(processing.read_text())
            return dict(context='sagemaker', job=record['ProcessingJobName'],
                        arn=record['ProcessingJobArn'])
        training = os.environ.get('TRAINING_JOB_NAME')
        if training and Path(
                '/opt/ml/input/config/resourceconfig.json').is_file():
            return dict(context='sagemaker', job=training)
        raise PlatformUnavailable(
            'Run inside the selected SageMaker training/processing job')

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
            if denied_access(
                    error,
                    provider_only=self.config['platform'] == 'snowflake'):
                return dict(denied_artifact_access='passed',
                            diagnostic=str(error))
            raise PlatformUnavailable(
                'Probe failed without permission-denial evidence') from error
        raise AssertionError('The denied artifact root was writable')


class Mlflow(Backend):
    @cached_property
    def connector(self):
        import kumo.engine as kumo

        if self.config['platform'] == 'databricks':
            identity = None
            if self.config['identity'] == 'service_principal':
                names = ('DATABRICKS_HOST', 'DATABRICKS_CLIENT_ID',
                         'DATABRICKS_CLIENT_SECRET')
                missing = [name for name in names if not os.environ.get(name)]
                if missing:
                    raise PlatformUnavailable(
                        f'Missing credential environment names: {missing}')
                identity = kumo.DatabricksIdentity(
                    host=os.environ[names[0]], client_id=os.environ[names[1]],
                    client_secret=os.environ[names[2]])
            return kumo.DatabricksConnector(
                catalog=self.config['catalog'],
                schema_name=self.config['schema'], identity=identity,
                cluster_id=self.config.get('cluster_id'),
                warehouse_id=self.config.get('warehouse_id'))
        identity = (kumo.S3Identity(role_arn=self.config['role_arn'])
                    if self.config.get('role_arn') else None)
        return kumo.S3Connector(identity=identity)

    def settings(self, register):
        import mlflow

        import kumo.engine as kumo
        from kumo.engine.integrations.mlflow import (
            MlflowTracking,
            SageMakerMLflow,
        )

        self.placement()
        if self.config['platform'] == 'databricks':
            target = self.connector
            mlflow.set_tracking_uri('databricks')
            mlflow.set_registry_uri('databricks-uc')
        else:
            identity = (kumo.S3Identity(role_arn=self.config['role_arn'])
                        if self.config.get('role_arn') else None)
            target = SageMakerMLflow(self.config['tracking_arn'],
                                     identity=identity)
            mlflow.set_tracking_uri(self.config['tracking_arn'])
            mlflow.set_registry_uri(self.config['tracking_arn'])
        return kumo.MLOps((self.connector, self.artifact_root),
                          tracking=MlflowTracking(self.experiment,
                                                  target=target,
                                                  log_system_metrics=False),
                          registry=kumo.ModelRegistryLocation(self.model)
                          if register else None, on_error='raise')

    def record(self, cfg, ref):
        require(ref is not None, 'Expected a tracked platform run')
        with cfg.tracking.client() as client:
            run = client.get_run(ref.run_id)
            history = {
                name: [(int(p.step), float(p.value))
                       for p in client.get_metric_history(ref.run_id, name)]
                for name in ('train/loss', 'val/auroc')
            }
        require(run.info.status == 'FINISHED',
                'Platform tracking run did not finish')
        return dict(params=run.data.tags, metrics=run.data.metrics,
                    history=history, run=ref.run_id, experiment=self.experiment,
                    experiment_id=run.info.experiment_id, url=ref.url)

    def registered(self, cfg, version, bundle):
        with cfg.tracking.client() as client:
            model = client.get_model_version(self.model, version)
        require(
            model.tags.get('kumo_bundle_id') == bundle,
            'Registry version refers to another bundle')

    def set_alias(self, cfg, version):
        with cfg.tracking.client() as client:
            client.set_registered_model_alias(self.model, 'champion', version)

    def alias(self, cfg):
        with cfg.tracking.client() as client:
            return str(
                client.get_model_version_by_alias(self.model,
                                                  'champion').version)


class Snowflake(Backend):
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
                 python_executable=None, phase=None):
    """Launch fresh lifecycle processes from a Snowflake Container Runtime notebook."""
    profile = validate(json.loads(Path(config).read_text()))
    if profile['platform'] != 'snowflake':
        raise ValueError('This notebook entrypoint is for Snowflake')
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
        prepare(Path(output_dir), Path(config))
        command += ['--phase', phase]
    completed = subprocess.run(command, env=environment, check=False)
    summary = Path(output_dir) / ('negative.json' if phase else 'summary.json')
    return dict(
        returncode=completed.returncode,
        summary=json.loads(summary.read_text()) if summary.is_file() else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--expected-version')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    proposal = plan(json.loads(args.config.read_text()))
    if not args.execute:
        print(json.dumps(proposal, indent=2))
        return 0
    if proposal['platform'] == 'snowflake':
        parser.error(
            'Snowflake execution requires run_notebook() in the supplied notebook'
        )
    if not all((args.output_dir, args.wheel, args.expected_version)):
        parser.error(
            '--execute requires --output-dir, --wheel and --expected-version')
    path = Path(__file__).with_name('wheel_mlops_lifecycle.py')
    spec = importlib.util.spec_from_file_location('mlops_lifecycle', path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load the lifecycle helper at {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main([
        '--execute-platform', '--platform-config',
        str(args.config), '--output-dir',
        str(args.output_dir), '--wheel',
        str(args.wheel), '--expected-version', args.expected_version
    ])


if __name__ == '__main__':
    raise SystemExit(main())
