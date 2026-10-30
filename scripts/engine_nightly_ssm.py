# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the nightly CPU-to-SSM connection without training a model."""

import argparse
import hashlib
import json
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REGION = 'us-east-2'
INSTANCE = 'i-06dd0e3c558780731'
BUCKET = 'kumo-unit-test'
PREFIX = 'diskgraph-profiles/engine-nightly/ssm-integration'
DOCUMENT = 'AWS-RunShellScript'
DOCUMENT_VERSION = '1'

# This fixed probe reads data only. It never downloads executable code.
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
    path = output / 'request.json'
    path.write_text(json.dumps(request, indent=2))
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    key = f'{PREFIX}/{args.run_id}'
    aws('s3api', 'put-object', '--bucket', BUCKET, '--key',
        key + '/request.json', '--body', str(path), '--server-side-encryption',
        'AES256', region='us-west-2')
    script = (f'export SSM_RunId={shlex.quote(args.run_id)}\n'
              f'export SSM_RequestSha256={shlex.quote(checksum)}\n{REMOTE}')
    command = aws('ssm', 'send-command', '--instance-ids', INSTANCE,
                  '--document-name', DOCUMENT, '--document-version',
                  DOCUMENT_VERSION, '--parameters',
                  json.dumps(dict(commands=[script], executionTimeout=['150'])),
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
        if result.get('request_sha256') != checksum:
            raise ValueError(
                'Remote receipt does not match the staged request bytes')
    except BaseException:
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
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    run(args)


if __name__ == '__main__':
    main()
