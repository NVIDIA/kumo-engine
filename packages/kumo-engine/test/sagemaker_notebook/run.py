# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared lifecycle and graph fixture in an actual SageMaker Notebook Job."""
import argparse
import ast
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

from candidate import candidate_path
from client import Client
from config import configuration
from evidence import collect, digest, save, save_probe_cell

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SHARED = HERE.parent / 'snowflake_notebook'
SOURCES = {
    'wheel_mlops_lifecycle.py': HERE.parent / 'wheel_mlops_lifecycle.py',
    'wheel_mlops_platform.py': HERE.parent / 'wheel_mlops_platform.py',
    'wheel_gpu_smoke.py': HERE.parent / 'wheel_gpu_smoke.py',
    'sagemaker_runtime.py': HERE / 'runtime.py',
    'sagemaker_budget.py': HERE / 'budget.py',
    'graph_fixture.py': SHARED / 'graph_fixture.py',
    'kernel-requirements.txt': HERE / 'kernel-requirements.txt',
}


def committed_sources():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                     text=True).strip()
    paths = sorted(
        set(SOURCES.values()) | set(HERE.glob('*.py')) | set(HERE.glob('*.in'))
        | set(HERE.glob('*.txt')) | set(HERE.glob('*.cjs'))
        | {
            HERE / 'browser.cjs', HERE / 'requirements.txt', SHARED /
            'graph_ui.cjs', SHARED / 'verification.cjs', HERE.parent /
            'platform_candidate.py', SHARED / 'browser/package.json', SHARED /
            'browser/package-lock.json', HERE.parent / 'pr172/inputs.json'
        })
    contents, hashes = {}, {}
    for path in paths:
        relative = path.relative_to(REPO).as_posix()
        content = subprocess.check_output(
            ['git', 'show', commit + ':' + relative], cwd=REPO)
        if path.read_bytes() != content:
            raise ValueError('Commit the exact test inputs before execution: ' +
                             relative)
        contents[relative], hashes[relative] = content, digest(content)
    return commit, contents, hashes


def notebook(spec, kernel):
    code = 'CONFIG = ' + repr(spec) + '\n' + kernel + '\nexecute(CONFIG)'
    return dict(
        nbformat=4, nbformat_minor=5, metadata=dict(
            kumo_test_attempt=spec['attempt'],
            kernelspec=dict(name=spec['kernel'], language='python',
                            display_name='Kumo test candidate')),
        cells=[
            save_probe_cell(),
            dict(cell_type='code', id=spec['attempt'], metadata={},
                 execution_count=None, outputs=[], source=code.splitlines(True))
        ])


def initialization(spec, bootstrap):
    return ('#!/bin/bash\nset -euo pipefail\npython3 - <<\'KUMO_TEST_SETUP\'\n'
            'CONFIG = ' + repr(spec) + '\n' + bootstrap +
            '\nprepare(CONFIG)\nKUMO_TEST_SETUP\n')


def output_notebook(s3, config, job, spec, root):
    prefix = spec['job_prefix'] + '/'
    paginator = s3.get_paginator('list_objects_v2')
    found = []
    for page in paginator.paginate(Bucket=config['bucket'], Prefix=prefix):
        for item in page.get('Contents', []):
            key = item['Key']
            if (job['TrainingJobName'] in key and
                (key.endswith('.ipynb') or key.endswith('/output.tar.gz')
                 or key.endswith('/model.tar.gz'))):
                if item['Size'] > 64 << 20:
                    raise ValueError('Notebook output exceeds evidence budget')
                found.append(key)
    candidates = []
    for key in found:
        body = s3.get_object(Bucket=config['bucket'], Key=key)['Body']
        try:
            content = body.read((64 << 20) + 1)
        finally:
            body.close()
        if len(content) > 64 << 20:
            raise ValueError('Notebook output exceeds evidence budget')
        if key.endswith('.ipynb'):
            candidates.append(content)
        else:
            with tarfile.open(fileobj=io.BytesIO(content),
                              mode='r|gz') as archive:
                expanded = 0
                for member in archive:
                    expanded += member.size
                    if expanded > 128 << 20:
                        raise ValueError(
                            'Notebook archive exceeds evidence budget')
                    if member.isfile() and member.name.endswith('.ipynb'):
                        if member.size > 32 << 20:
                            raise ValueError(
                                'Notebook output exceeds evidence budget')
                        candidates.append(archive.extractfile(member).read())
    executed = []
    for content in candidates:
        record = json.loads(content)
        if (record.get('metadata',
                       {}).get('kumo_test_attempt') == spec['attempt']
                and any(c.get('outputs') for c in record['cells'])):
            executed.append(content)
    unique = {digest(content): content for content in executed}
    if len(unique) != 1:
        raise ValueError(
            'Expected one actual executed notebook for this attempt')
    content = next(iter(unique.values()))
    (root / 'executed.ipynb').write_bytes(content)
    return json.loads(content)


def verdict(report, spec, job, pipeline):
    expected_pipeline = (f'arn:aws:sagemaker:{spec["region"]}:{spec["account"]}:'
                         f'pipeline/{spec["pipeline_name"]}')
    if (pipeline.get('PipelineExecutionStatus') != 'Succeeded'
            or pipeline.get('PipelineArn') != expected_pipeline):
        raise ValueError('Notebook pipeline did not succeed for this attempt')
    if (report['status'] != 'passed'
            or report.get('installation', {}).get('pip_check_returncode') != 0
            or report.get('installation', {}).get('wheel_sha256') != spec['wheel_sha256']
            or report.get('environment', {}).get('sha256') != spec['wheel_sha256']
            or report.get('runtime', {}).get('job_arn') != job['TrainingJobArn']
            or report['runtime'].get('execution_role') != spec['execution_role']
            or report['runtime'].get('context') != 'sagemaker_notebook'
            or job['TrainingJobStatus'] != 'Completed'):
        raise ValueError(
            'Notebook lacks successful installed-product runtime evidence')
    if spec['scenario'] in ('engine', 'all'):
        phases = report.get('phases', {})
        if set(phases) != {'negative', 'fit', 'score', 'evaluate'}:
            raise ValueError('Lifecycle did not execute all four phases')
        for name, phase in phases.items():
            runtime = phase.get('platform', {}).get('placement', {})
            if (phase.get('status') != 'passed'
                    or phase.get('returncode') != 0 or phase.get(
                        'environment', {}).get('sha256') != spec['wheel_sha256']
                    or runtime.get('context') != 'sagemaker_notebook'
                    or runtime.get('job_arn') != job['TrainingJobArn']
                    or runtime.get('execution_role') != spec['execution_role']):
                raise ValueError(
                    'Lifecycle phase lacks actual execution evidence: ' + name)
    if spec['scenario'] in ('graph', 'all') and report.get('graph', {}).get('status') != 'passed':
        raise ValueError('Notebook graph checks did not pass')
    return dict(engine='passed' if report['phases'] else 'not selected',
                notebook_export='not verified', native_studio='not verified',
                native_mlflow='not verified', full_qualification='incomplete',
                remaining_requirements=report.get('required_gaps', {}))


def execute(args, root):
    commit, frozen, hashes = committed_sources()

    def source(path):
        return frozen[path.relative_to(REPO).as_posix()]

    if args.candidate is None:
        raise ValueError('Supply --candidate or SAGEMAKER_TEST_CANDIDATE from candidate.py')
    candidate = json.loads(args.candidate.read_text())
    sys.path.insert(0, str(HERE.parent))
    from platform_candidate import verify
    verify(REPO, candidate, commit)
    cfg = configuration()
    client = Client(cfg)
    admission = client.preflight()
    version = admission.get('tracking_version')
    if isinstance(version, str) and re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+',
                                                 version):
        print('Managed MLflow version:', version)
    if not args.engine_checks_only:
        from config import DOMAIN_ID
        from studio import Studio
        admission['studio'] = Studio(client, DOMAIN_ID).preflight()
    save(root / 'admission.json', admission)
    candidate = candidate['candidate']
    fixture = json.loads(source(HERE.parent / 'pr172/inputs.json'))
    assets = {}
    for name, path in SOURCES.items():
        assets[name] = client.asset(name, digest(source(path)), path)
    assets[candidate['wheel_file']] = client.asset(candidate['wheel_file'],
                                                   candidate['wheel_sha256'],
                                                   args.wheel)
    assets['pr172-inputs-v1.tar.gz'] = client.asset('pr172-inputs-v1.tar.gz',
                                                    fixture['sha256'],
                                                    args.fixture)
    attempt = uuid.uuid4().hex[:12]
    spec = dict(
        cfg, attempt=attempt, candidate=candidate, assets=assets,
        source_commit=commit, test_sources=hashes, scenario=args.scenario,
        wheel_sha256=candidate['wheel_sha256'],
        pipeline_name='kumo-test-notebook-' + attempt,
        notebook_name='kumo-test-' + attempt + '.ipynb',
        output_prefix=cfg['root'] + '/runs/' + attempt + '/outputs',
        job_prefix=cfg['root'] + '/notebook-jobs/' + attempt,
        job_receipt_key=cfg['root'] + '/runs/' + attempt + '/job-identity.json')
    save(root / 'scenario.json', spec)
    from cleanup import AttemptResources
    with AttemptResources(client, spec, root) as resources:
        resources.prepare_tracking()
        submitted = notebook(spec, source(HERE / 'kernel.py').decode())
        input_path = root / spec['notebook_name']
        save(input_path, submitted)
        init_path = root / 'initialize.sh'
        init_path.write_text(
            initialization(spec,
                           source(HERE / 'bootstrap.py').decode()))
        job = client.submit(input_path, init_path, spec, root, args.timeout)
        save(root / 'job.json', job)
        executed = output_notebook(client.s3, cfg, job, spec, root)
        report = collect(client.s3, cfg['bucket'], executed, submitted, spec, root)
        report['verdict'] = verdict(
            report, spec, job, json.loads((root / 'pipeline-execution.json').read_text()))
        return verify_browser(args, root, client, spec, report)


def verify_browser(args, root, client, spec, report):
    import nbformat
    from browser import verify
    from nbconvert import HTMLExporter

    save(root / 'execution-result.json', report)
    executed = json.loads((root / 'executed.ipynb').read_text())
    document = nbformat.reads(json.dumps(executed), as_version=4)
    html, _ = HTMLExporter(template_name='lab',
                           embed_images=True).from_notebook_node(document)
    (root / 'notebook.html').write_text(html)
    completed = ['execution']
    if spec['scenario'] != 'engine' or not args.engine_checks_only:
        ui = verify(client, spec, root, native=not args.engine_checks_only)
        if (spec['scenario'] in ('engine', 'all') and not args.engine_checks_only
                and (not isinstance(ui.get('native_mlflow'), dict)
                     or ui['native_mlflow'].get('status') != 'passed')):
            raise ValueError('Managed MLflow UI verification was not completed')
        if (not args.engine_checks_only
                and (not isinstance(ui.get('native_studio'), dict)
                     or ui['native_studio'].get('status') != 'passed')):
            raise ValueError('Native Studio notebook verification was not completed')
        report['browser'] = ui
        report['verdict'].update(notebook_export=ui['notebook_export'],
                                 native_mlflow=ui['native_mlflow'],
                                 native_studio=ui.get('native_studio', 'not selected'))
        if spec['scenario'] in ('graph', 'all'):
            completed.append('saved export')
        if isinstance(ui.get('native_mlflow'), dict) and ui['native_mlflow'].get('status') == 'passed':
            completed.append('managed MLflow UI')
        if isinstance(ui.get('native_studio'), dict) and ui['native_studio'].get('status') == 'passed':
            completed.append('native Studio notebook UI')
    report['verification_scope'] = ', '.join(completed)
    return report


def verify_existing(args, root):
    """Validate retained execution evidence before any browser/AWS access."""
    previous = args.ui_run.resolve()
    spec = json.loads((previous / 'scenario.json').read_text())
    if (not re.fullmatch(r'[0-9a-f]{40}', args.producer_commit or '')
            or spec.get('source_commit') != args.producer_commit):
        raise ValueError('Retained execution must match its GitHub producer commit')
    cfg = configuration()
    for key in ('account', 'region', 'bucket', 'root', 'execution_role',
                'pipeline_role', 'tracking_arn', 'tracking_server', 'domain_name',
                'user_profile', 'space', 'image', 'instance', 'kernel'):
        if spec.get(key) != cfg[key]:
            raise ValueError('Retained execution differs from dedicated test configuration: ' + key)
    selected = getattr(args, 'candidate', candidate_path())
    if selected is None:
        raise ValueError('Supply the original runtime candidate manifest for UI replay')
    candidate = json.loads(selected.read_text())['candidate']
    if spec.get('candidate') != candidate or spec.get('wheel_sha256') != candidate['wheel_sha256']:
        raise ValueError('Retained execution uses another candidate')
    submitted_path = previous / spec['notebook_name']
    if (not re.fullmatch(r'kumo-test-[0-9a-f]{12}\.ipynb', spec['notebook_name'])
            or spec['notebook_name'] != 'kumo-test-' + spec['attempt'] + '.ipynb'):
        raise ValueError('Invalid retained notebook identity')
    submitted = json.loads(submitted_path.read_text())
    executed = json.loads((previous / 'executed.ipynb').read_text())
    from evidence import receipt
    record = receipt(executed, submitted, spec)
    outputs = previous / 'outputs'
    for name, expected in record['files'].items():
        if digest((outputs / name).read_bytes()) != expected:
            raise ValueError('Retained evidence checksum mismatch: ' + name)
    report = json.loads((outputs / 'result.json').read_text())
    for key in ('attempt', 'source_commit', 'wheel_sha256', 'output_prefix', 'execution_id'):
        if report.get(key) != record.get(key):
            raise ValueError('Retained report identifies another execution: ' + key)
    if report.get('test_sources') != spec['test_sources'] or report.get('payload') != spec['assets']:
        raise ValueError('Retained report differs from submitted sources/assets')
    job = json.loads((previous / 'job.json').read_text())
    pipeline = json.loads((previous / 'pipeline-execution.json').read_text())
    report['verdict'] = verdict(report, spec, job, pipeline)
    (root / 'outputs').mkdir()
    for name in record['files']:
        shutil.copyfile(outputs / name, root / 'outputs' / name)
    for name in ('scenario.json', 'executed.ipynb', 'job.json', 'pipeline-execution.json'):
        shutil.copyfile(previous / name, root / name)
    shutil.copyfile(submitted_path, root / spec['notebook_name'])
    verifier_commit, _, verifier_hashes = committed_sources()
    save(root / 'verification.json', dict(
        producer_commit=args.producer_commit, verifier_commit=verifier_commit,
        verifier_sources=verifier_hashes, training_submitted=False))
    return verify_browser(args, root, Client(cfg), spec, report)


def browser_diagnostics(root):
    """Expose only source-authored check names and messages in public job logs."""
    sources = [
        HERE / name
        for name in ('browser.cjs', 'studio_browser.cjs', 'assertions.cjs')
    ]
    sources += [SHARED / 'graph_ui.cjs']
    literals = set()
    for source in sources:
        # Never interpolate browser/provider data into a diagnostic message.
        for match in re.finditer(r"(['\"])(?:\\.|(?!\1)[^\\])*?\1",
                                 source.read_text()):
            try:
                value = ast.literal_eval(match.group())
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, str):
                literals.add(value)
    reports = {}
    for name in ('browser', 'studio'):
        for filename in ('result.json', 'progress.json'):
            path = root / name / filename
            if not path.is_file():
                continue
            try:
                report = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            if not isinstance(report, dict):
                continue
            safe = {}
            if report.get('status') in ('passed', 'failed'):
                safe['status'] = report['status']
            for key in ('stage', 'assertion', 'error'):
                value = report.get(key)
                if not isinstance(value, str):
                    continue
                if value in literals and len(value) <= 180:
                    safe[key] = value
                elif key == 'stage' and re.fullmatch(
                        r'mlflow-(?:fit|score(?:-drift-evaluate)?)-[0-5]|mlflow-version-[12]',
                        value):
                    safe[key] = value
                elif key == 'assertion':
                    prefix = value.split(':', 1)[0] + ':'
                    if (prefix in literals and re.fullmatch(
                            r'(?:metric-row|history-[a-z-]+):', prefix)):
                        safe[key] = prefix + '[fixture value]'
            for key in ('notebook_export', 'native_mlflow'):
                nested = report.get(key)
                if isinstance(
                        nested,
                        dict) and nested.get('status') in ('passed', 'failed'):
                    safe[key] = nested['status']
            errors = report.get('browser_errors')
            if isinstance(errors, list):
                safe['browser_error_count'] = len(errors)
                safe['browser_errors'] = []
                for error in errors[:20]:
                    if not isinstance(error, dict):
                        continue
                    item = {}
                    for field, allowed in {
                            'category': {'graphics', 'react', 'frontend'},
                            'name': {
                                'Error', 'TypeError', 'RangeError',
                                'ReferenceError', 'SyntaxError'
                            }
                    }.items():
                        value = error.get(field)
                        if isinstance(value, str) and value in allowed:
                            item[field] = value
                    stage = error.get('stage')
                    if isinstance(stage, str) and stage in literals:
                        item['stage'] = stage
                    signatures = error.get('signatures')
                    if isinstance(signatures, list):
                        item['signatures'] = sorted(
                            set(
                                value for value in signatures
                                if isinstance(value, str) and value in {
                                    'dynamic_import', 'chunk_load',
                                    'script_load', 'resource_fetch',
                                    'resize_observer', 'invalid_hook', 'kernel',
                                    'widget', 'comm', 'disposed',
                                    'disposed_model', 'disposed_cache',
                                    'cancellation', 'websocket', 'session',
                                    'extension', 'schema', 'module',
                                    'connection', 'missing', 'forbidden'
                                }))
                    if item:
                        safe['browser_errors'].append(item)
            reports[name] = safe
            break
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument(
        '--engine-checks-only', action='store_true',
        help='Run execution checks without claiming full UI qualification')
    parser.add_argument('--scenario', choices=('engine', 'graph', 'all'),
                        default='all')
    parser.add_argument('--ui-run', type=Path, help='Verify retained execution without submitting training')
    parser.add_argument('--producer-commit', help='Producer SHA verified from GitHub run metadata')
    parser.add_argument('--candidate', type=Path, default=candidate_path())
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=4500)
    args = parser.parse_args(argv)
    if bool(args.ui_run) != bool(args.producer_commit):
        parser.error('--ui-run requires its verified --producer-commit')
    if args.ui_run and (args.wheel or args.fixture):
        parser.error('--ui-run cannot submit a new wheel or fixture')
    if args.timeout <= 0:
        parser.error('Timeout must be positive')
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(status='planned', configuration=configuration(),
                  scenario=args.scenario)
    try:
        if args.execute:
            report = verify_existing(args, root) if args.ui_run else execute(args, root)
    except Exception as error:
        retained = root / 'execution-result.json'
        if retained.is_file():
            report = json.loads(retained.read_text())
        report.update(status='failed', error=str(error))
        frames = []
        frame = error.__traceback__
        while frame is not None:
            source = Path(frame.tb_frame.f_code.co_filename).resolve()
            if source.parent == HERE:
                frames.append(f'{source.name}:{frame.tb_lineno}')
            frame = frame.tb_next
        print('Failure sites:', ' > '.join(frames))
        print('Browser checks:',
              json.dumps(browser_diagnostics(root), sort_keys=True))
    save(root / 'summary.json', report)
    print('Result:', report['status'], 'Summary:', root / 'summary.json')
    return 0 if report['status'] in ('planned', 'passed') else 1


if __name__ == '__main__':
    sys.exit(main())
