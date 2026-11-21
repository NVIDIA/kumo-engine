# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a clean product environment before the actual notebook starts."""
import hashlib
import importlib.util
import json
import platform
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

STAGING_SECONDS = 300


@contextmanager
def staging_limit(seconds):
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('Initialization already has an execution timer')

    def expired(*_):
        raise TimeoutError('Notebook asset download budget exhausted')

    if seconds <= 0:
        expired()
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def prepare(config, session=None):
    if session is None:
        import boto3
        session = boto3.Session(region_name=config['region'])
    if (platform.machine() != 'x86_64' or sys.version_info < (3, 11)
            or platform.libc_ver()[0] != 'glibc'
            or tuple(map(int,
                         platform.libc_ver()[1].split('.'))) < (2, 34)):
        raise ValueError(
            'Candidate requires x86_64, Python >=3.11 and glibc >=2.34')
    identity = session.client('sts').get_caller_identity()
    expected = config['execution_role'].rsplit('/', 1)[-1]
    if (identity['Account'] != config['account'] or identity['Arn'].split(
            ':assumed-role/', 1)[-1].split('/')[0] != expected):
        raise ValueError('Initialization resolved another execution identity')
    root = Path('/tmp/kumo-sagemaker-' + config['attempt'])
    root.mkdir(mode=0o700, exist_ok=False)
    work_started = time.monotonic()
    payload, evidence = root / 'payload', root / 'evidence'
    payload.mkdir()
    evidence.mkdir()
    from botocore.config import Config

    # Bound staging separately so it cannot consume the work/evidence reserve.
    download_deadline = work_started + STAGING_SECONDS
    s3 = session.client(
        's3', config=Config(connect_timeout=10, read_timeout=30,
                            retries={'total_max_attempts': 1}))

    def download_remaining():
        seconds = download_deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError('Notebook asset download budget exhausted')
        return seconds

    with staging_limit(download_remaining()):
        for name, asset in config['assets'].items():
            target = payload / name
            download_remaining()
            body = s3.get_object(Bucket=asset['bucket'],
                                 Key=asset['key'])['Body']
            try:
                body.set_socket_timeout(min(30, download_remaining()))
                with target.open('wb') as stream:
                    while True:
                        chunk = body.read(8 << 20)
                        download_remaining()
                        if not chunk:
                            break
                        stream.write(chunk)
            finally:
                body.close()
            if (target.stat().st_size != asset['size'] or hashlib.sha256(
                    target.read_bytes()).hexdigest() != asset['sha256']):
                raise ValueError('Staged asset differs from source: ' + name)
    helper = payload / 'wheel_mlops_lifecycle.py'
    spec = importlib.util.spec_from_file_location('lifecycle_helpers', helper)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    budget_spec = importlib.util.spec_from_file_location(
        'sagemaker_budget', payload / 'sagemaker_budget.py')
    budget = importlib.util.module_from_spec(budget_spec)
    budget_spec.loader.exec_module(budget)
    (root / 'execution-deadline.json').write_text(
        json.dumps(work_started + budget.WORK_SECONDS))
    python = root / 'venv/bin/python'
    with (evidence / 'venv.log').open('w') as log:
        subprocess.run(
            [sys.executable, '-m', 'venv', '--without-pip',
             str(root / 'venv')], stdout=log, stderr=subprocess.STDOUT,
            timeout=budget.remaining(root, 120), check=True)
    for name, requirements in [
        ('kernel-install',
         ['--require-hashes', '-r',
          str(payload / 'kernel-requirements.txt')]),
        ('product-install',
         [str(payload / config['candidate']['wheel_file']) + '[sagemaker]']),
    ]:
        if name == 'product-install' and config.get('workload') is not None:
            requirements += [
                str(payload / item) for item in config['workload']['wheels']
            ]
        if name == 'product-install' and config.get('accelerator') == 'gpu':
            tools_spec = importlib.util.spec_from_file_location(
                'workload_tools', payload / 'wheel_mlops_workload.py')
            tools = importlib.util.module_from_spec(tools_spec)
            tools_spec.loader.exec_module(tools)
            requirements += tools.pytorch_install_args('gpu')
        with (evidence / (name + '.log')).open('w') as log:
            command = [
                sys.executable, '-I', '-m', 'pip', '--isolated', '--python',
                str(python), 'install', '--index-url',
                'https://pypi.org/simple', *requirements
            ]
            if helpers.execute(command, root, log, budget.remaining(root,
                                                                    1200)):
                raise RuntimeError(name + ' failed')
    with (evidence / 'pip-check.log').open('w') as log:
        subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', 'check'],
                       check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=budget.remaining(root, 120))
    subprocess.run([
        str(python), '-m', 'ipykernel', 'install', '--user', '--name',
        config['kernel'], '--display-name', 'Kumo test candidate'
    ], check=True, timeout=budget.remaining(root, 120))
    (evidence / 'installation.json').write_text(
        json.dumps(
            dict(
                pip_check_returncode=0, interpreter=str(python),
                identity=identity['Arn'],
                version=config['candidate']['expected_version'],
                wheel_sha256=config['candidate']['wheel_sha256'],
                workload_wheels={
                    name: config['assets'][name]['sha256']
                    for name in config.get('workload', {}).get('wheels', [])
                }), indent=2))
