# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the nightly CPU-to-SSM connection without training a model."""

import argparse
import email
import hashlib
import json
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REGION = 'us-east-2'
INSTANCE = 'i-06dd0e3c558780731'
BUCKET = 'kumo-unit-test'
PREFIX = 'diskgraph-profiles/engine-nightly/ssm-integration'
DOCUMENT = 'AWS-RunShellScript'
DOCUMENT_VERSION = '1'

# The fixed probe optionally installs the exact CI wheel, without fitting models.
REMOTE = r'''set -eu
exec runuser -u ubuntu -- env -i HOME=/home/ubuntu PATH=/usr/local/bin:/usr/bin:/bin CUDA_VISIBLE_DEVICES=0 SSM_RunId="$SSM_RunId" SSM_RequestSha256="$SSM_RequestSha256" python3 - <<'PY'
import csv, hashlib, json, os, pathlib, subprocess, tempfile
instance = 'i-06dd0e3c558780731'
bucket = 'kumo-unit-test'
prefix = 'diskgraph-profiles/engine-nightly/ssm-integration/' + os.environ['SSM_RunId']
def aws(*args, **kwargs):
    return subprocess.run(['aws', '--region', 'us-west-2', *args],
        check=True, capture_output=True, timeout=45, **kwargs).stdout
assert pathlib.Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip() == instance
raw = aws('s3', 'cp', 's3://' + bucket + '/' + prefix + '/request.json', '-', '--only-show-errors')
assert hashlib.sha256(raw).hexdigest() == os.environ['SSM_RequestSha256'], 'Request checksum differs'
request = json.loads(raw)
assert request['run_id'] == os.environ['SSM_RunId'] and request['instance_id'] == instance
assert request['mode'] == 'integration-only' and request['physical_fits'] == 0
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
    '--format=csv,noheader,nounits'], text=True, timeout=15)
gpus = [dict(index=int(r[0]), name=r[1].strip(), memory_mib=int(r[2]), driver=r[3].strip())
    for r in csv.reader(rows.splitlines())]
assert len(gpus) == 2 and all('RTX PRO 6000' in g['name'] for g in gpus), 'Unexpected GPU profile'
ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
disk = os.statvfs('/opt/dlami/nvme')
assert disk.f_blocks * disk.f_frsize >= 3 * 2**40, 'Expected the selected local NVMe capacity'
result = dict(request, status='INTEGRATION_PASS', gpu_work_started=False,
    cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], physical_gpus=gpus,
    system_memory_bytes=ram, nvme_total_bytes=disk.f_blocks * disk.f_frsize,
    nvme_available_bytes=disk.f_bavail * disk.f_frsize,
    request_sha256=hashlib.sha256(raw).hexdigest(),
    scope='SSM and small-file transport only; no model validation')
if request.get('wheel'):
    spec = request['wheel']
    assert pathlib.Path(spec['filename']).name == spec['filename']
    with tempfile.TemporaryDirectory(prefix='kumo-engine-ci-', dir='/opt/dlami/nvme') as directory:
        stage = pathlib.Path(directory)
        wheel = stage / spec['filename']
        aws('s3', 'cp', 's3://' + bucket + '/' + spec['key'], str(wheel), '--only-show-errors')
        assert hashlib.sha256(wheel.read_bytes()).hexdigest() == spec['sha256'], 'Wheel checksum differs'
        subprocess.run(['python3', '-m', 'venv', str(stage / 'env')], check=True, timeout=60)
        python = str(stage / 'env/bin/python')
        log = stage / 'install.log'
        with log.open('w') as stream:
            try:
                subprocess.run([python, '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check',
                    'install', '--index-url', 'https://pypi.org/simple', '--no-cache-dir',
                    '--progress-bar', 'off', str(wheel)], check=True, timeout=900,
                    stdout=stream, stderr=subprocess.STDOUT)
                subprocess.run([python, '-I', '-m', 'pip', '--isolated', 'check'],
                    check=True, timeout=60, stdout=stream, stderr=subprocess.STDOUT)
            finally:
                aws('s3', 'cp', str(log), 's3://' + bucket + '/' + prefix + '/wheel-install.log', '--only-show-errors')
        code = """import importlib.metadata, json, pathlib, sys
import kumo.engine as engine
import diskgraph
from diskgraph import _c
root = pathlib.Path(sys.prefix).resolve()
assert sys.prefix != sys.base_prefix
paths = dict(engine_import=str(pathlib.Path(engine.__file__).resolve()),
             diskgraph_import=str(pathlib.Path(diskgraph.__file__).resolve()))
assert all(pathlib.Path(p).is_relative_to(root) for p in paths.values())
print(json.dumps(dict(status='PASS', version=importlib.metadata.version('kumo-engine'),
    engine_version=engine.__version__, isolated=True, **paths)))"""
        installed = subprocess.check_output([python, '-I', '-c', code], cwd=stage, text=True, timeout=90)
        result['installation'] = json.loads(installed)
        assert result['installation']['version'] == spec['version'] == result['installation']['engine_version']
        result['scope'] = 'CI wheel build, S3 transfer, isolated installation and imports; zero model fits'
encoded = json.dumps(result, indent=2).encode()
aws('s3', 'cp', '-', 's3://' + bucket + '/' + prefix + '/result.json', '--only-show-errors', input=encoded)
print(encoded.decode())
PY'''


def aws(*args, region=REGION):
    result = subprocess.run(
        ['aws', '--region', region, *args, '--output', 'json'],
        capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout) if result.stdout.strip() else {}


def wait_for_command(command_id, output, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            result = aws('ssm', 'get-command-invocation', '--command-id',
                         command_id, '--instance-id', INSTANCE)
        except RuntimeError as error:
            if 'InvocationDoesNotExist' not in str(error):
                raise
        else:
            (output / 'ssm-invocation.json').write_text(
                json.dumps(result, indent=2))
            status = result['Status']
            if status == 'Success' and result['ResponseCode'] == 0:
                return result
            if status not in ('Pending', 'InProgress', 'Delayed'):
                raise RuntimeError(
                    f'SSM invocation failed: {status}, exit {result["ResponseCode"]}'
                )
        time.sleep(3)
    raise TimeoutError(
        'Nightly SSM integration did not finish before its deadline')


def validate_result(request, result):
    for key in ('run_id', 'source_sha', 'instance_id', 'nonce'):
        if result.get(key) != request[key]:
            raise ValueError(f'Integration result identity differs: {key}')
    if (result.get('status') != 'INTEGRATION_PASS'
            or result.get('physical_fits') != 0
            or result.get('gpu_work_started') is not False
            or result.get('cuda_visible_devices') != '0'):
        raise ValueError(
            'Expected a completed integration-only result with zero model work')
    if request.get('wheel'):
        install = result.get('installation') or {}
        if (result.get('wheel') != request['wheel']
                or install.get('status') != 'PASS'
                or install.get('version') != request['wheel']['version']
                or install.get('engine_version') != request['wheel']['version']
                or install.get('isolated') is not True):
            raise ValueError(
                'Expected the exact CI wheel installed and imported in isolation'
            )


def wheel_identity(path, key):
    with zipfile.ZipFile(path) as archive:
        records = [
            name for name in archive.namelist()
            if name.endswith('.dist-info/METADATA')
        ]
        if len(records) != 1:
            raise ValueError('Expected one wheel metadata record')
        metadata = email.message_from_bytes(archive.read(records[0]))
    if metadata['Name'] != 'kumo-engine' or not metadata['Version']:
        raise ValueError('Expected the built kumo-engine wheel')
    return dict(filename=path.name, version=metadata['Version'], key=key,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def run(args):
    if not re.fullmatch(r'[0-9]{1,20}-[0-9]{1,5}', args.run_id):
        raise ValueError('Invalid GitHub run ID and attempt')
    if not re.fullmatch(r'[a-f0-9]{40}', args.source_sha):
        raise ValueError('Expected the exact source commit SHA')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    request = dict(run_id=args.run_id, source_sha=args.source_sha,
                   instance_id=INSTANCE, nonce=secrets.token_hex(16),
                   mode='integration-only', physical_fits=0)
    key = f'{PREFIX}/{args.run_id}'
    wheel = getattr(args, 'wheel', None)
    if wheel is not None:
        request['wheel'] = wheel_identity(wheel, key + '/' + wheel.name)
        aws('s3api', 'put-object',
            '--bucket', BUCKET, '--key', request['wheel']['key'], '--body',
            str(wheel), '--server-side-encryption', 'AES256',
            region='us-west-2')
    path = output / 'request.json'
    path.write_text(json.dumps(request, indent=2))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    aws('s3api', 'put-object', '--bucket', BUCKET, '--key',
        key + '/request.json', '--body', str(path), '--server-side-encryption',
        'AES256', region='us-west-2')
    script = (f'export SSM_RunId={shlex.quote(args.run_id)}\n'
              f'export SSM_RequestSha256={shlex.quote(checksum)}\n{REMOTE}')
    command = aws(
        'ssm', 'send-command', '--instance-ids', INSTANCE, '--document-name',
        DOCUMENT, '--document-version', DOCUMENT_VERSION, '--parameters',
        json.dumps(
            dict(commands=[script],
                 executionTimeout=['1200' if wheel is not None else '150'])),
        '--timeout-seconds', '60', '--comment',
        'Nightly integration only; zero model fits')
    command_id = command['Command']['CommandId']
    (output / 'command-id.txt').write_text(command_id + '\n')
    try:
        wait_for_command(command_id, output, args.timeout)
        result_path = output / 'result.json'
        aws('s3api', 'get-object', '--bucket', BUCKET, '--key',
            key + '/result.json', str(result_path), region='us-west-2')
        result = json.loads(result_path.read_text())
        validate_result(request, result)
        if wheel is not None:
            aws('s3api', 'get-object', '--bucket', BUCKET,
                '--key', key + '/wheel-install.log',
                str(output / 'wheel-install.log'), region='us-west-2')
        if result.get('request_sha256') != checksum:
            raise ValueError(
                'Remote receipt does not match the staged request bytes')
    except BaseException:
        if wheel is not None:
            try:
                aws('s3api', 'get-object', '--bucket', BUCKET,
                    '--key', key + '/wheel-install.log',
                    str(output / 'wheel-install.log'), region='us-west-2')
            except RuntimeError as error:
                print(f'Could not collect wheel installation log: {error}',
                      file=sys.stderr)
        try:
            aws('ssm', 'cancel-command', '--command-id', command_id,
                '--instance-ids', INSTANCE)
        except RuntimeError as error:
            print(f'Could not cancel SSM command {command_id}: {error}',
                  file=sys.stderr)
        raise
    print(
        f'Nightly integration PASS: {command_id}; zero fits; GPU work not started'
    )


def interrupt(_signum, _frame):
    raise KeyboardInterrupt('Nightly integration controller was cancelled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    probe = commands.add_parser('run')
    probe.add_argument('--run-id', required=True)
    probe.add_argument('--source-sha', required=True)
    probe.add_argument('--output-dir', type=Path, required=True)
    probe.add_argument('--timeout', type=int, default=240)
    probe.add_argument('--wheel', type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    run(args)


if __name__ == '__main__':
    main()
