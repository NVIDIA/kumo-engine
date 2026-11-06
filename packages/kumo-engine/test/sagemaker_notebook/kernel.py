# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the shared checks in the selected wheel's actual notebook kernel."""


def execute(config):
    import hashlib
    import importlib.util
    import json
    import shutil
    import sys
    import tarfile
    import traceback
    from pathlib import Path
    from uuid import uuid4

    import boto3
    from IPython.display import display

    root = Path('/tmp/kumo-sagemaker-' + config['attempt'])
    payload, evidence = root / 'payload', root / 'evidence'

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, payload / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    report = dict(
        status='failed', execution_id=uuid4().hex[:12],
        attempt=config['attempt'], source_commit=config['source_commit'],
        wheel_sha256=config['wheel_sha256'],
        test_sources=config['test_sources'], payload=config['assets'],
        output_prefix=config['output_prefix'], phases={}, required_gaps={
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
    session = boto3.Session(region_name=config['region'])
    credentials = session.get_credentials().get_frozen_credentials()
    secrets = [
        v.encode() for v in (credentials.access_key, credentials.secret_key,
                             credentials.token) if v
    ]
    lifecycle = root / 'lifecycle'
    lifecycle.mkdir()
    try:
        assert 'include-system-site-packages = false' in (
            root / 'venv/pyvenv.cfg').read_text()
        assert Path(sys.executable).parent == root / 'venv/bin'
        for name, asset in config['assets'].items():
            path = payload / name
            assert hashlib.sha256(
                path.read_bytes()).hexdigest() == asset['sha256']
        runtime = load('sagemaker_runtime', 'sagemaker_runtime.py')
        context = runtime.discover(config, session)
        report['runtime'] = runtime.qualify(context, session, kernel=True)
        (payload / 'runtime-context.json').write_text(json.dumps(context))
        installation = json.loads((evidence / 'installation.json').read_text())
        assert installation['pip_check_returncode'] == 0
        assert installation['wheel_sha256'] == config['wheel_sha256']
        report['installation'] = installation
        helpers = load('lifecycle_helpers', 'wheel_mlops_lifecycle.py')
        wheel = payload / config['candidate']['wheel_file']
        report['environment'] = helpers.check_install(
            wheel, config['candidate']['expected_version'])
        sys.path.insert(0, str(payload))
        if config['scenario'] in ('graph', 'all'):
            fixture = root / 'fixture'
            fixture.mkdir()
            with tarfile.open(payload / 'pr172-inputs-v1.tar.gz') as archive:
                for item in archive.getmembers():
                    name = Path(item.name)
                    if (name.is_absolute() or '..' in name.parts
                            or not (item.isfile() or item.isdir())):
                        raise ValueError('Unsafe fixture archive')
                archive.extractall(fixture, filter='data')
            graph = load('graph_fixture', 'graph_fixture.py')
            result = graph.exercise(root / 'graph', fixture, display=display)
            result.update(runtime=dict(report['runtime'], notebook=True),
                          execution_id=report['execution_id'])
            report['graph'] = result
            for path in (root / 'graph').iterdir():
                if path.is_file():
                    shutil.copyfile(path, evidence / path.name)
            (evidence / 'graph-result.json').write_text(
                json.dumps(result, indent=2))
        if config['scenario'] in ('engine', 'all'):
            profile = dict(
                platform='sagemaker', artifact_root=config['artifact_root'],
                denied_artifact_root=config['denied_artifact_root'],
                tracking_arn=config['tracking_arn'],
                experiment='kumo-test-mlops', model_prefix='kumo-test-mlops',
                accelerator='cpu', execution_context='sagemaker',
                environment='venv',
                notebook_context=str(payload / 'runtime-context.json'))
            (lifecycle / 'platform.json').write_text(
                json.dumps(dict(config=profile, attempt=config['attempt'])))
            for phase in ('negative', 'fit', 'score', 'evaluate'):
                command = [
                    sys.executable, '-I',
                    str(payload / 'wheel_mlops_lifecycle.py'),
                    '--execute-platform', '--wheel',
                    str(wheel), '--expected-version',
                    config['candidate']['expected_version'], '--output-dir',
                    str(lifecycle), '--phase', phase
                ]
                with (evidence / (phase + '.log')).open('w') as log:
                    code = helpers.execute(command, lifecycle, log, 900)
                path = lifecycle / (phase + '.json')
                result = json.loads(path.read_text())
                result['returncode'] = code
                if code and result['status'] == 'passed':
                    result.update(status='failed',
                                  error='Nonzero worker exit after receipt')
                report['phases'][phase] = result
                path.write_text(json.dumps(result, indent=2))
                if code or result['status'] != 'passed':
                    break
                scratch = lifecycle / ('scratch-' + phase)
                if scratch.exists():
                    shutil.rmtree(scratch)
                if phase == 'fit':
                    for directory in lifecycle.glob('data-*'):
                        shutil.rmtree(directory)
        phases = report['phases']
        graph_ok = (config['scenario'] == 'engine'
                    or report.get('graph', {}).get('status') == 'passed')
        engine_ok = (config['scenario'] == 'graph'
                     or set(phases) == {'fit', 'score', 'evaluate', 'negative'}
                     and all(v['status'] == 'passed' and v['returncode'] == 0
                             for v in phases.values()))
        report['status'] = 'passed' if graph_ok and engine_ok else 'failed'
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        for path in lifecycle.glob('*.json'):
            shutil.copyfile(path, evidence / path.name)
        (evidence / 'result.json').write_text(json.dumps(report, indent=2))
        inventory = {}
        s3 = session.client('s3')
        for path in evidence.iterdir():
            if path.is_file():
                content = path.read_bytes()
                if path.suffix in ('.json', '.log', '.html', '.svg', '.md'):
                    for secret in secrets:
                        content = content.replace(secret, b'[REDACTED]')
                s3.put_object(Bucket=config['bucket'],
                              Key=config['output_prefix'] + '/' + path.name,
                              Body=content, IfNoneMatch='*')
                inventory[path.name] = hashlib.sha256(content).hexdigest()
        print(
            'KUMO_SAGEMAKER_RECEIPT=' + json.dumps(
                dict(attempt=config['attempt'],
                     source_commit=config['source_commit'],
                     wheel_sha256=config['wheel_sha256'],
                     output_prefix=config['output_prefix'],
                     execution_id=report['execution_id'], files=inventory)),
            flush=True)
