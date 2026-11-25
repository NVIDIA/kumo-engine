# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared small lifecycle with the Databricks notebook package environment."""


def execute(config, dbutils):
    import hashlib
    import importlib.util
    import json
    import os
    import shutil
    import subprocess
    import sys
    import tempfile
    import traceback
    from pathlib import Path
    from uuid import uuid4

    runtime_token = ''
    root = Path(tempfile.mkdtemp(prefix='kumo-databricks-lifecycle-'))
    payload, evidence, lifecycle = root / 'payload', root / 'evidence', root / 'lifecycle'
    payload.mkdir()
    evidence.mkdir()
    report = dict(
        status='failed', attempt=config['attempt'],
        source_commit=config['source_commit'],
        wheel_sha256=config['candidate']['wheel_sha256'], identity=None,
        denied_storage=config.get('denied_storage'),
        execution_id=uuid4().hex[:12],
        runtime=os.environ.get('DATABRICKS_RUNTIME_VERSION'),
        notebook_path=config['notebook_path'], payload=config['assets'],
        test_sources=config['test_sources'], phases={}, required_gaps={
            'native_ui':
            dict(status='blocked',
                 reason='PAT does not authenticate the native browser'),
            'nonzero_drift_sensitivity':
            dict(status='not implemented',
                 reason='Nonzero drift sensitivity is not tested'),
            'interrupted_remote_writer':
            dict(status='not implemented',
                 reason='Active remote writer interruption is not exercised'),
            'exported_epoch_history':
            dict(status='not implemented',
                 reason='Full epoch history is not available in model exports'),
        })
    try:
        from databricks.sdk import WorkspaceClient

        assert os.environ.get('KUMO_DATABRICKS_TEST_PAT') is None
        context = dbutils.notebook.entry_point.getDbutils().notebook(
        ).getContext()
        assert context.notebookPath().get() == config['notebook_path']
        assert os.environ.get('DATABRICKS_RUNTIME_VERSION')
        workspace = WorkspaceClient()
        user = workspace.current_user.me().user_name
        assert user == config['expected_user'], 'Notebook identity differs from the configured test principal'
        authorization = workspace.config.authenticate()['Authorization']
        assert authorization.startswith('Bearer ')
        runtime_token = authorization.removeprefix('Bearer ')
        report['identity'] = user
        env = dict(os.environ, DATABRICKS_HOST=workspace.config.host,
                   DATABRICKS_TOKEN=runtime_token, DATABRICKS_AUTH_TYPE='pat',
                   KUMO_MLOPS_EXPECTED_DATABRICKS_USER=user, MPLBACKEND='Agg',
                   MLFLOW_HTTP_REQUEST_TIMEOUT='300',
                   MLFLOW_HTTP_REQUEST_MAX_RETRIES='2')
        for name, asset in config['assets'].items():
            target = payload / name
            shutil.copyfile(asset['path'], target)
            assert target.stat().st_size == asset['size']
            assert hashlib.sha256(
                target.read_bytes()).hexdigest() == asset['sha256']
        helper_spec = importlib.util.spec_from_file_location(
            'lifecycle_helpers', payload / 'wheel_mlops_lifecycle.py')
        helpers = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helpers)
        wheel = payload / config['candidate']['wheel_file']
        # Databricks adds notebook-scoped packages to the kernel's sys.path.
        # Carry its search order into the fresh lifecycle workers.
        env['PYTHONPATH'] = os.pathsep.join(path for path in sys.path
                                            if os.path.isabs(path))
        report['environment'] = helpers.check_install(
            wheel, config['candidate']['expected_version'], managed=True)
        import transformers

        import kumo.pytorch

        report['native_imports'] = dict(pytorch=kumo.pytorch.__name__,
                                        transformers=transformers.__version__)
        with (evidence / 'native-pip-check.log').open('w') as log:
            check = subprocess.run([sys.executable, '-P', '-m', 'pip', 'check'],
                                   env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=120)
            report['pip_check_returncode'] = check.returncode
            check.check_returncode()
        profile = dict(platform='databricks', catalog='kumo_test_catalogue',
                       schema='kumo_test', cluster_id=config['cluster_id'],
                       artifact_root=config['artifact_root'],
                       experiment=config['experiment'],
                       model_prefix='KUMO_TEST_MLOPS', identity='ambient',
                       accelerator='cpu', execution_context='databricks',
                       environment='managed')
        if config.get('denied_artifact_root'):
            profile['denied_artifact_root'] = config['denied_artifact_root']
        profile_path = root / 'profile.json'
        profile_path.write_text(json.dumps(profile))
        command = [
            sys.executable, '-P',
            str(payload / 'wheel_mlops_lifecycle.py'), '--execute-platform',
            '--platform-config',
            str(profile_path), '--attempt', config['attempt'], '--wheel',
            str(wheel), '--expected-version',
            config['candidate']['expected_version'], '--output-dir',
            str(lifecycle), '--phase-timeout', '600'
        ]
        with (evidence / 'lifecycle.log').open('w') as log:
            helpers.execute(command, root, log, 3000, env=env)
        summary = json.loads((lifecycle / 'summary.json').read_text())
        report['lifecycle_status'] = summary['status']
        report['phases'] = summary['phases']
        report['engine_checks'] = (
            'passed' if report['lifecycle_status'] in ('passed', 'blocked') and
            set(report['phases']) == {'fit', 'score', 'evaluate', 'negative'}
            and all(p['status'] == 'passed'
                    for p in report['phases'].values()) else 'incomplete')
        report['status'] = (
            'failed' if report['lifecycle_status'] == 'failed' else
            'blocked' if report['engine_checks'] == 'passed' or any(
                p['status'] == 'blocked'
                for p in report['phases'].values()) else 'failed')
    except Exception:
        report['error'] = traceback.format_exc()
        if runtime_token:
            report['error'] = report['error'].replace(runtime_token,
                                                      '[REDACTED]')
    finally:
        for pattern in ('*.json', '*.log'):
            for path in lifecycle.glob(pattern):
                shutil.copyfile(path, evidence / path.name)
        (evidence / 'lifecycle-result.json').write_text(
            json.dumps(report, indent=2))
        destination = Path(config['output'])
        destination.mkdir(parents=True, exist_ok=False)
        inventory = {}
        for path in evidence.iterdir():
            if path.is_file():
                text = path.read_text(errors='replace')
                if runtime_token:
                    text = text.replace(runtime_token, '[REDACTED]')
                target = destination / path.name
                target.write_text(text)
                inventory[path.name] = hashlib.sha256(
                    target.read_bytes()).hexdigest()
        receipt = dict(attempt=config['attempt'], status=report['status'],
                       execution_id=report['execution_id'],
                       source_commit=config['source_commit'],
                       output=config['output'], files=inventory)
        print('KUMO_DATABRICKS_LIFECYCLE_RECEIPT=' + json.dumps(receipt),
              flush=True)
