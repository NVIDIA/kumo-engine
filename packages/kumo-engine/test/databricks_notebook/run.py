# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Snowflake's shared graph scenario in a Databricks notebook using a PAT."""
import argparse
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import time
import urllib.parse
import uuid
import zipfile
from pathlib import Path

from client import Client, configuration

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / 'snowflake_notebook'
CANDIDATE = HERE / 'candidate.json'
FILES = {
    'graph.svg', 'graph.pdf', 'graph.md', 'default.html', 'sampled.html',
    'graph-result.json'
}
MODULES = ('graph.py', 'table.py', '_widget.py', '_visualization.py',
           '_visualization_viewer.py', '_visualization_pdf.py',
           '_visualization_layout.py')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def asset(client, volume, name, expected, local=None):
    remote = volume + '/payload-cache/' + expected + '/' + name
    try:
        content = client.api('/api/2.0/fs/files' + remote, binary=True)
    except RuntimeError as error:
        if str(error) != 'Databricks API HTTP 404' or local is None:
            raise
        content = Path(local).read_bytes()
        if Path(local).name != name or digest(content) != expected:
            raise ValueError('Local asset differs from pinned input: ' + name)
        client.api('/api/2.0/fs/directories' + remote.rsplit('/', 1)[0], 'PUT')
        client.api('/api/2.0/fs/files' + remote + '?overwrite=false', 'PUT',
                   raw=content)
    if digest(content) != expected:
        raise ValueError('Cached asset checksum mismatch: ' + name)
    return dict(path=remote, sha256=expected, size=len(content)), content


def notebook(config):
    wheel = config['assets'][config['candidate']['wheel_file']]['path']
    codes = [
        "import platform; assert platform.machine() == 'x86_64', platform.machine()",
        '%pip install "' + wheel + '[databricks]" "pypdf==6.21.4"',
        'dbutils.library.restartPython()',
        'CONFIG = ' + repr(config) + '\n' + (HERE / 'kernel.py').read_text() +
        '\nexecute(CONFIG, dbutils)',
    ]
    return dict(
        nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
            name='python3', language='python', display_name='Python 3')),
        cells=[
            dict(cell_type='code', id=config['attempt'] + '-' + str(i),
                 metadata={}, execution_count=None, outputs=[],
                 source=code.splitlines(True)) for i, code in enumerate(codes)
        ])


def task_request(config, path, attempt, client, timeout=1800):
    task = dict(task_key='graph_notebook',
                notebook_task=dict(notebook_path=path, source='WORKSPACE'),
                timeout_seconds=timeout, max_retries=0)
    payload = dict(run_name='kumo-test-notebook-ui-' + attempt,
                   timeout_seconds=timeout, tasks=[task])
    if config['cluster_id']:
        cluster = client.api('/api/2.0/clusters/get?cluster_id=' +
                             config['cluster_id'])
        if (not cluster.get('cluster_name', '').startswith('KUMO_TEST_') or
                cluster.get('node_type_id') not in ('i3.xlarge', 'i3.2xlarge')
                or cluster.get('num_workers') != 0
                or cluster.get('state') != 'RUNNING'):
            raise ValueError(
                'Use a running dedicated single-node x86 KUMO_TEST_ cluster')
        task['existing_cluster_id'] = config['cluster_id']
    else:
        task['environment_key'] = 'graph'
        payload['environments'] = [
            dict(environment_key='graph', spec=dict(environment_version='2',
                                                    dependencies=[]))
        ]
    return payload


def successful_task(run):
    if run['state'].get('result_state') != 'SUCCESS':
        raise ValueError('Notebook job did not succeed; inspect run.json')
    tasks = [
        task for task in run.get('tasks', [])
        if task['task_key'] == 'graph_notebook'
        and task['state'].get('result_state') == 'SUCCESS'
    ]
    if len(tasks) != 1:
        raise ValueError('Expected one successful notebook task attempt')
    return tasks[0]['run_id']


def notebook_export(client, task, root, timeout=120):
    deadline = time.monotonic() + timeout
    attempts = 0
    while True:
        export = client.api('/api/2.2/jobs/runs/export?run_id=' + str(task) +
                            '&views_to_export=ALL')
        save(root / 'export.json', export)
        views = [v for v in export.get('views', []) if v['type'] == 'NOTEBOOK']
        if len(views) != 1:
            raise ValueError('Expected the actual notebook HTML export')
        html = views[0]['content']
        (root / 'notebook.html').write_text(html)
        match = re.search(r"var __DATABRICKS_NOTEBOOK_MODEL = '([^']+)'", html)
        if not match:
            raise ValueError('Databricks export model is unavailable')
        model = json.loads(
            urllib.parse.unquote(base64.b64decode(match[1]).decode()))
        save(root / 'notebook-model.json', model)
        attempts += 1
        pending = [
            c.get('position') for c in model['commands']
            if c.get('resultDbfsStatus') == 'WRITING_TO_DBFS'
            and c.get('results') is None
        ]
        save(root / 'export-status.json',
             dict(attempts=attempts, pending_cells=pending))
        if not pending:
            return model
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'Databricks notebook cell outputs are still being persisted')
        print('Waiting for persisted notebook outputs', flush=True)
        time.sleep(5)


def collect(client, run, config, root, submitted):
    task = successful_task(run)
    model = notebook_export(client, task, root)
    actual = [''.join(c['command']) for c in model['commands']]
    submitted = [''.join(c['source']) for c in submitted['cells']]
    if actual != submitted or any(c['state'] != 'finished'
                                  for c in model['commands']):
        raise ValueError(
            'Exported notebook cells differ from completed submitted cells')
    receipts = []
    for command in model['commands']:
        for result in (command.get('results') or {}).get('data', []):
            if result.get('type') == 'ansi':
                for line in result['data'].splitlines():
                    if line.startswith('KUMO_DATABRICKS_GRAPH_RECEIPT='):
                        receipts.append(json.loads(line.split('=', 1)[1]))
    if len(receipts) != 1:
        raise ValueError('Expected exactly one graph execution receipt')
    receipt = receipts[0]
    if (receipt.get('status') != 'passed'
            or receipt.get('attempt') != config['attempt']
            or receipt.get('output') != config['output']
            or set(receipt.get('files', {})) != FILES or
            not re.fullmatch(r'[0-9a-f]{12}', receipt.get('execution_id', ''))):
        raise ValueError('Notebook receipt identity or inventory mismatch')
    output = root / 'outputs'
    output.mkdir()
    for name, expected in receipt['files'].items():
        content = client.api(
            '/api/2.0/fs/files' + config['output'] + '/' + name, binary=True)
        if digest(content) != expected:
            raise ValueError('Notebook evidence checksum mismatch: ' + name)
        (output / name).write_bytes(content)
    stored = json.loads((output / 'graph-result.json').read_text())
    if (stored.get('status') != 'passed'
            or stored.get('attempt') != config['attempt']
            or stored.get('execution_id') != receipt['execution_id']
            or stored.get('wheel_sha256') != config['candidate']['wheel_sha256']
            or stored.get('source_commit') != config['source_commit']
            or stored.get('payload') != config['assets'] or stored.get(
                'runtime', {}).get('notebook_path') != config['notebook_path']):
        raise ValueError('Stored graph report differs from submitted scenario')
    save(root / 'receipt.json', receipt)
    return stored


def submit_and_wait(client, config, spec, content, root, timeout,
                    task_timeout=1800):
    path = spec['notebook_path']
    client.api('/api/2.0/workspace/mkdirs', 'POST',
               dict(path=config['workspace_dir']))
    client.api(
        '/api/2.0/workspace/import', 'POST',
        dict(path=path, format='JUPYTER',
             content=base64.b64encode(json.dumps(content).encode()).decode(),
             overwrite=False))
    payload = task_request(config, path, spec['attempt'], client, task_timeout)
    save(root / 'submission.json', payload)
    run_id = client.api('/api/2.2/jobs/runs/submit', 'POST', payload)['run_id']
    save(root / 'submitted-run.json', dict(run_id=run_id))
    print('Notebook submitted:', run_id, flush=True)
    deadline = time.monotonic() + timeout
    try:
        while True:
            run = client.api('/api/2.2/jobs/runs/get?run_id=' + str(run_id))
            save(root / 'run.json', run)
            if run['state']['life_cycle_state'] in ('TERMINATED', 'SKIPPED',
                                                    'INTERNAL_ERROR'):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Notebook deadline exceeded')
            print('Notebook:', run['state']['life_cycle_state'], flush=True)
            time.sleep(20)
    except BaseException:
        try:
            client.api('/api/2.2/jobs/runs/cancel', 'POST', dict(run_id=run_id))
        except RuntimeError:
            pass
        raise
    return run


def execute(args, root):
    config = configuration(args.cluster_id)
    client = Client(config)
    candidate = json.loads(CANDIDATE.read_text())['candidate']
    fixture = json.loads((HERE.parent / 'pr172/inputs.json').read_text())
    assets = {}
    wheel = None
    for name, expected, local in [
        (candidate['wheel_file'], candidate['wheel_sha256'], args.wheel),
        ('pr172-inputs-v1.tar.gz', fixture['sha256'], args.fixture),
        ('graph_fixture.py', digest((SHARED / 'graph_fixture.py').read_bytes()),
         SHARED / 'graph_fixture.py'),
        ('wheel_gpu_smoke.py',
         digest((HERE.parent / 'wheel_gpu_smoke.py').read_bytes()),
         HERE.parent / 'wheel_gpu_smoke.py')
    ]:
        assets[name], content = asset(client, config['volume'], name, expected,
                                      local)
        if name == candidate['wheel_file']:
            wheel = content
    repo = HERE.parents[3]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo,
                                     text=True).strip()
    modules = {}
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        for name in MODULES:
            relative = 'packages/kumo-engine/kumo/engine/' + name
            content = subprocess.check_output(
                ['git', 'show', commit + ':' + relative], cwd=repo)
            if content != archive.read('kumo/engine/' + name):
                raise ValueError(
                    'Candidate graph module differs from test revision: ' +
                    name)
            modules[name] = digest(content)
    attempt = uuid.uuid4().hex[:12]
    path = config['workspace_dir'] + '/egu-graph-' + attempt
    spec = dict(attempt=attempt, assets=assets, candidate=candidate,
                output=config['volume'] + '/runs/' + attempt + '/outputs',
                source_commit=commit, modules=modules, notebook_path=path)
    save(root / 'scenario.json', spec)
    content = notebook(spec)
    save(root / 'notebook.ipynb', content)
    run = submit_and_wait(client, config, spec, content, root, args.timeout)
    run_id = run['run_id']
    report = collect(client, run, spec, root, content)
    subprocess.run(
        [args.node, str(HERE / 'verify_export.cjs'),
         str(root)], check=True, timeout=300)
    return dict(
        status='passed', scope='Notebook graph and actual Jobs HTML export',
        run_id=run_id, execution_id=report['execution_id'],
        fixture_sha256=fixture['sha256'],
        wheel_sha256=candidate['wheel_sha256'], source_commit=commit,
        ui=json.loads((root / 'ui-result.json').read_text()), limitations=[
            'No live workspace-editor or native registry UI coverage',
            'No training lifecycle or full dependency compatibility qualification',
            'Serverless CPU architecture can vary; ARM attempts fail before installation'
        ])


def positive_seconds(value):
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError('Timeout must be positive')
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument(
        '--wheel', type=Path,
        help='Seed the pinned candidate if absent from the test volume')
    parser.add_argument('--fixture', type=Path,
                        help='Seed the unchanged An PR172 fixture if absent')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--timeout', type=positive_seconds, default=2400)
    parser.add_argument('--node', default='node')
    parser.add_argument(
        '--cluster-id', type=lambda value: configuration(value)['cluster_id'],
        help='Optional running dedicated KUMO_TEST_ x86 cluster')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    summary = dict(status='planned', credentials='KUMO_DATABRICKS_TEST_PAT',
                   resources=configuration(args.cluster_id))
    try:
        if args.execute:
            summary = execute(args, root)
    except Exception as error:
        summary = dict(status='failed', error=str(error))
    save(root / 'summary.json', summary)
    print('Result:', summary['status'], 'Summary:', root / 'summary.json')
    return 1 if summary['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())
