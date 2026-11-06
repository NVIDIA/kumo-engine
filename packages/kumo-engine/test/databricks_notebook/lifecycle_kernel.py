# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared small lifecycle in isolated processes inside a real notebook."""


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
    lifecycle.mkdir()
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
        assert user == 'kumo-databricks@nvidia.com', 'Use the dedicated test identity'
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
        python = root / 'venv/bin/python'
        with (evidence / 'venv.log').open('w') as log:
            subprocess.run([
                sys.executable, '-m', 'venv', '--without-pip',
                str(root / 'venv')
            ], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        with (evidence / 'install.log').open('w') as log:
            install = [
                sys.executable, '-I', '-m', 'pip', '--isolated', '--python',
                str(python), 'install', '--index-url',
                'https://pypi.org/simple', 'pip',
                str(wheel) + '[databricks,mlflow]'
            ]
            if helpers.execute(install, root, log, 900):
                raise RuntimeError(
                    'Clean environment installation failed; see install.log')
        with (evidence / 'pip-check.log').open('w') as log:
            subprocess.run(
                [str(python), '-I', '-m', 'pip', '--isolated', 'check'],
                check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        report['pip_check_returncode'] = 0
        profile = dict(platform='databricks', catalog='kumo_test_catalogue',
                       schema='kumo_test', cluster_id=config['cluster_id'],
                       artifact_root=config['artifact_root'],
                       experiment=config['experiment'],
                       model_prefix='KUMO_TEST_MLOPS', identity='ambient',
                       accelerator='cpu', execution_context='databricks',
                       environment='managed')
        if config.get('denied_artifact_root'):
            profile['denied_artifact_root'] = config['denied_artifact_root']
        (lifecycle / 'platform.json').write_text(
            json.dumps(dict(config=profile, attempt=config['attempt'])))
        for phase in ('fit', 'score', 'evaluate', 'negative'):
            command = [
                str(python), '-I',
                str(payload / 'wheel_mlops_lifecycle.py'), '--execute-platform',
                '--wheel',
                str(wheel), '--expected-version',
                config['candidate']['expected_version'], '--output-dir',
                str(lifecycle), '--phase', phase
            ]
            with (evidence / (phase + '.log')).open('w') as log:
                returncode = helpers.execute(command, lifecycle, log, 900,
                                             env=env)
            receipt = lifecycle / (phase + '.json')
            assert receipt.is_file(), phase + ' produced no receipt'
            result = json.loads(receipt.read_text())
            result['returncode'] = returncode
            if returncode and result['status'] == 'passed':
                result.update(
                    status='failed',
                    error='Worker exited nonzero after a passed receipt')
            report['phases'][phase] = result
            if returncode and phase != 'negative':
                break
            scratch = lifecycle / ('scratch-' + phase)
            if scratch.exists():
                shutil.rmtree(scratch)
            assert not scratch.exists()
            if phase == 'fit':
                for path in lifecycle.glob('data-*'):
                    shutil.rmtree(path)
                assert not list(lifecycle.glob('data-*'))
        report['engine_checks'] = (
            'passed' if
            set(report['phases']) == {'fit', 'score', 'evaluate', 'negative'}
            and all(p['status'] == 'passed'
                    for p in report['phases'].values()) else 'incomplete')
        report['status'] = ('blocked' if report['engine_checks'] == 'passed'
                            or any(p['status'] == 'blocked'
                                   for p in report['phases'].values()) else
                            'failed')
    except Exception:
        report['error'] = traceback.format_exc()
        if runtime_token:
            report['error'] = report['error'].replace(runtime_token,
                                                      '[REDACTED]')
    finally:
        for path in lifecycle.glob('*.json'):
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
