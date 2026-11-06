# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the same small lifecycle as Snowflake in an actual Databricks notebook."""
import argparse
import json
import re
import subprocess
import sys
import uuid
from pathlib import Path

from client import Client, configuration
from run import (
    HERE,
    SHARED,
    asset,
    digest,
    notebook_export,
    save,
    submit_and_wait,
    successful_task,
)


def notebook(spec, kernel_source=None):
    if kernel_source is None:
        kernel_source = (HERE / 'lifecycle_kernel.py').read_text()
    code = 'CONFIG = ' + repr(
        spec) + '\n' + kernel_source + '\nexecute(CONFIG, dbutils)'
    codes = [
        "import platform; assert platform.machine() == 'x86_64', platform.machine()",
        code
    ]
    return dict(
        nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
            name='python3', language='python', display_name='Python 3')),
        cells=[
            dict(cell_type='code', id=spec['attempt'] + '-' + str(i),
                 metadata={}, execution_count=None, outputs=[],
                 source=c.splitlines(True)) for i, c in enumerate(codes)
        ])


def collect(client, run, spec, submitted, root):
    model = notebook_export(client, successful_task(run), root)
    if ([''.join(c['command']) for c in model['commands']
         ] != [''.join(c['source']) for c in submitted['cells']]
            or any(c['state'] != 'finished' for c in model['commands'])):
        raise ValueError(
            'Lifecycle exported cells differ from submitted execution')
    receipts = []
    for command in model['commands']:
        for result in (command.get('results') or {}).get('data', []):
            if result.get('type') == 'ansi':
                for line in result['data'].splitlines():
                    if line.startswith('KUMO_DATABRICKS_LIFECYCLE_RECEIPT='):
                        receipts.append(json.loads(line.split('=', 1)[1]))
    if len(receipts) != 1:
        raise ValueError('Expected one actual lifecycle receipt')
    receipt = receipts[0]
    if (receipt.get('attempt') != spec['attempt']
            or receipt.get('source_commit') != spec['source_commit']
            or receipt.get('output') != spec['output'] or
            not re.fullmatch(r'[0-9a-f]{12}', receipt.get('execution_id', ''))
            or 'lifecycle-result.json' not in receipt.get('files', {})):
        raise ValueError('Lifecycle receipt does not identify this execution')
    output = root / 'outputs'
    output.mkdir()
    for name, expected in receipt['files'].items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or not re.fullmatch(
                r'[0-9a-f]{64}', expected):
            raise ValueError('Invalid lifecycle evidence inventory')
        content = client.api('/api/2.0/fs/files' + spec['output'] + '/' + name,
                             binary=True)
        if digest(content) != expected:
            raise ValueError('Lifecycle output checksum mismatch: ' + name)
        (output / name).write_bytes(content)
    report = json.loads((output / 'lifecycle-result.json').read_text())
    if (report.get('attempt') != spec['attempt']
            or report.get('source_commit') != spec['source_commit']
            or report.get('wheel_sha256') != spec['candidate']['wheel_sha256']
            or report.get('payload') != spec['assets']
            or report.get('test_sources') != spec['test_sources']
            or report.get('notebook_path') != spec['notebook_path']
            or report.get('identity') != 'kumo-databricks@nvidia.com'
            or report.get('execution_id') != receipt['execution_id']):
        raise ValueError(
            'Lifecycle stored report identity differs from execution')
    save(root / 'receipt.json', receipt)
    return report


def engine_verdict(report):
    expected = {'fit', 'score', 'evaluate', 'negative'}
    phases = report.get('phases', {})
    states = {
        name: phases.get(name, {}).get('status', 'blocked')
        for name in expected
    }
    for name, phase in phases.items():
        if phase.get('status') == 'passed':
            if phase.get('returncode') != 0:
                raise ValueError(
                    'Passed lifecycle receipt has a nonzero or missing worker exit code: '
                    + name)
            if (phase.get('environment', {}).get('sha256')
                    != report['wheel_sha256'] or phase.get('platform', {}).get(
                        'placement', {}).get('context') != 'databricks'
                    or phase.get('platform', {}).get(
                        'placement', {}).get('user') != report['identity']):
                raise ValueError(
                    'Passed lifecycle phase lacks installed-wheel runtime/identity evidence: '
                    + name)
    positive = all(states[n] == 'passed' for n in ('fit', 'score', 'evaluate'))
    complete = set(phases) == expected and all(s == 'passed'
                                               for s in states.values())
    if (positive or complete) and report.get('pip_check_returncode') != 0:
        raise ValueError('Lifecycle dependency admission did not pass')
    return dict(positive_lifecycle='passed' if positive else 'incomplete',
                engine_checks='passed' if complete else 'incomplete',
                phases=states, full_qualification='incomplete',
                remaining_requirements=report.get('required_gaps', {}))


def execute(args, root):
    repo = HERE.parents[3]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                     text=True).strip()
    paths = [
        HERE / n for n in ('lifecycle_kernel.py', 'lifecycle_notebook.py',
                           'client.py', 'run.py')
    ]
    paths += [
        HERE.parent / n
        for n in ('wheel_mlops_lifecycle.py', 'wheel_mlops_platform.py',
                  'wheel_gpu_smoke.py')
    ]
    paths.append(SHARED / 'candidate.json')
    committed, sources = {}, {}
    for path in paths:
        relative = path.relative_to(repo).as_posix()
        content = subprocess.check_output(
            ['git', 'show', commit + ':' + relative], cwd=repo)
        if path.read_bytes() != content:
            raise ValueError(
                'Commit the exact lifecycle test inputs before execution: ' +
                relative)
        committed[path.name] = content
        sources[relative] = digest(content)
    config = configuration(args.cluster_id)
    client = Client(config)
    candidate = json.loads(committed['candidate.json'])['candidate']
    assets = {}
    assets[candidate['wheel_file']], _ = asset(client, config['volume'],
                                               candidate['wheel_file'],
                                               candidate['wheel_sha256'],
                                               args.wheel)
    for name in ('wheel_mlops_lifecycle.py', 'wheel_mlops_platform.py',
                 'wheel_gpu_smoke.py'):
        local = HERE.parent / name
        assets[name], _ = asset(client, config['volume'], name,
                                digest(committed[name]), local)
    attempt = uuid.uuid4().hex[:12]
    folder = '/Shared/kumo_test_catalogue/KUMO_TEST/lifecycle'
    spec = dict(
        attempt=attempt, candidate=candidate, assets=assets,
        source_commit=commit, test_sources=sources,
        cluster_id=config['cluster_id'],
        notebook_path=folder + '/egu-lifecycle-' + attempt,
        experiment=folder + '/KUMO_TEST_MLOPS',
        artifact_root=config['volume'].removesuffix('/notebook-ui') +
        '/lifecycle',
        output=config['volume'] + '/lifecycle/runs/' + attempt + '/outputs')
    save(root / 'scenario.json', spec)
    submitted = notebook(spec, committed['lifecycle_kernel.py'].decode())
    save(root / 'notebook.ipynb', submitted)
    client.api('/api/2.0/workspace/mkdirs', 'POST', dict(path=folder))
    run = submit_and_wait(client, config, spec, submitted, root, args.timeout,
                          task_timeout=3600)
    report = collect(client, run, spec, submitted, root)
    report.update(github_or_local_notebook_run=run['run_id'],
                  verdict=engine_verdict(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--engine-checks-only', action='store_true',
                        help='Require engine phases; retain UI gaps')
    parser.add_argument('--cluster-id', required=True,
                        type=lambda value: configuration(value)['cluster_id'])
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=int, default=4200)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('Timeout must be positive')
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    report = dict(
        status='planned',
        scope='Shared platform lifecycle in an isolated notebook environment')
    try:
        if args.execute:
            report = execute(args, root)
    except Exception as error:
        report = dict(status='failed', error=str(error))
    save(root / 'summary.json', report)
    print('Result:', report['status'], 'Summary:', root / 'summary.json')
    if args.engine_checks_only and report.get(
            'verdict', {}).get('engine_checks') == 'passed':
        return 0
    return 0 if report['status'] in ('passed', 'planned') else 1


if __name__ == '__main__':
    sys.exit(main())
