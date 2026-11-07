# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shared lifecycle and graph fixture in an actual SageMaker Notebook Job."""
import argparse
import io
import json
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

from client import Client
from config import configuration
from evidence import collect, digest, save

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SHARED = HERE.parent / 'snowflake_notebook'
SOURCES = {
    'wheel_mlops_lifecycle.py': HERE.parent / 'wheel_mlops_lifecycle.py',
    'wheel_mlops_platform.py': HERE.parent / 'wheel_mlops_platform.py',
    'wheel_gpu_smoke.py': HERE.parent / 'wheel_gpu_smoke.py',
    'sagemaker_runtime.py': HERE / 'runtime.py',
    'graph_fixture.py': SHARED / 'graph_fixture.py',
}


def committed_sources():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                     text=True).strip()
    paths = sorted(
        set(SOURCES.values()) | set(HERE.glob('*.py')) | set(HERE.glob('*.cjs'))
        | {
            HERE / 'candidate.json', HERE / 'browser.cjs', HERE /
            'apply_access.sh', HERE / 'requirements.txt', SHARED /
            'graph_ui.cjs', SHARED / 'browser/package.json', SHARED /
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


def verdict(report, spec, job):
    if (report['status'] != 'passed'
            or report.get('installation', {}).get('pip_check_returncode') != 0
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
                    or runtime.get('job_arn') != job['TrainingJobArn']
                    or runtime.get('execution_role') != spec['execution_role']):
                raise ValueError(
                    'Lifecycle phase lacks actual execution evidence: ' + name)
    return dict(engine='passed' if report['phases'] else 'not selected',
                notebook_export='not verified', native_studio='not verified',
                native_mlflow='not verified', full_qualification='incomplete',
                remaining_requirements=report.get('required_gaps', {}))


def execute(args, root):
    commit, frozen, hashes = committed_sources()

    def source(path):
        return frozen[path.relative_to(REPO).as_posix()]

    cfg = configuration()
    client = Client(cfg)
    admission = client.preflight()
    save(root / 'admission.json', admission)
    candidate = json.loads(source(HERE / 'candidate.json'))['candidate']
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
    report['verdict'] = verdict(report, spec, job)
    import nbformat
    from nbconvert import HTMLExporter

    document = nbformat.reads(json.dumps(executed), as_version=4)
    html, _ = HTMLExporter(template_name='lab',
                           embed_images=True).from_notebook_node(document)
    (root / 'notebook.html').write_text(html)
    from browser import verify

    if args.scenario != 'engine' or not args.engine_checks_only:
        ui = verify(client, spec, root, native=not args.engine_checks_only)
        report['browser'] = ui
        report['verdict'].update(notebook_export=ui['notebook_export'],
                                 native_mlflow=ui['native_mlflow'])
    if not args.engine_checks_only:
        missing = (
            'Notebook execution completed; required Studio and MLflow browser '
            'verification has not completed. See verdict for uncovered checks.')
        report.update(status='blocked', error=missing)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument(
        '--engine-checks-only', action='store_true',
        help='Run execution checks without claiming full UI qualification')
    parser.add_argument('--scenario', choices=('engine', 'graph', 'all'),
                        default='all')
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=4500)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error('Timeout must be positive')
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(status='planned', configuration=configuration(),
                  scenario=args.scenario)
    try:
        if args.execute:
            report = execute(args, root)
    except Exception as error:
        report = dict(status='failed', error=str(error))
    save(root / 'summary.json', report)
    print('Result:', report['status'], 'Summary:', root / 'summary.json')
    return 0 if report['status'] in ('planned', 'passed') else 1


if __name__ == '__main__':
    sys.exit(main())
