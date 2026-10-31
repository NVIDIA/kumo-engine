#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer the CPU-built wheel and control one fixed nightly VM through SSM."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'nightly_transport',
    Path(__file__).with_name('engine_nightly_ssm.py'))
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)
PREFIX = transport.PREFIX + '/nightly'
ROLE = 'arn:aws:iam::926922431314:role/gha-kumo-workflow-tests'
CODE = {
    'nightly_check.py': ROOT / 'packages/kumo-engine/scripts/nightly_check.py',
    'wheel_gpu_smoke.py':
    ROOT / 'packages/kumo-engine/test/wheel_gpu_smoke.py',
    'nightly_host.py': ROOT / 'packages/kumo-engine/scripts/nightly_host.py'
}
ACTIVE = ('Pending', 'InProgress', 'Delayed')

# systemd bounds the entire unprivileged process group independently of CI.
BOOTSTRAP = r'''import hashlib,json,os,subprocess,tarfile,tempfile
from pathlib import Path
instance='i-06dd0e3c558780731'
assert Path('/sys/devices/virtual/dmi/id/board_asset_tag').read_text().strip()==instance
prefix='s3://kumo-unit-test/diskgraph-profiles/engine-nightly/ssm-integration/nightly/'+os.environ['NIGHTLY_RUN_ID']
def get(name,path):
 subprocess.run(['aws','--region','us-west-2','s3','cp',prefix+'/'+name,str(path),'--only-show-errors'],check=True,timeout=120)
base=Path('/opt/dlami/nvme/engine-nightly');base.mkdir(exist_ok=True)
stage=Path(tempfile.mkdtemp(prefix=os.environ['NIGHTLY_RUN_ID']+'-',dir=base))
request=stage/'request.json';get('request.json',request)
assert hashlib.sha256(request.read_bytes()).hexdigest()==os.environ['NIGHTLY_REQUEST_SHA256']
spec=json.loads(request.read_text())
assert spec['mode']=='nightly' and spec['run_id']==os.environ['NIGHTLY_RUN_ID'] and spec['instance_id']==instance
archive=stage/'payload.tar.gz';get('payload.tar.gz',archive)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==spec['payload']['sha256']
with tarfile.open(archive) as source:
 members=source.getmembers()
 assert len(members)==3 and {m.name for m in members}=={'nightly_check.py','wheel_gpu_smoke.py','nightly_host.py'}
 assert all(m.isfile() for m in members)
 source.extractall(stage,filter='data')
for name,checksum in spec['payload']['files'].items():
 assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==checksum
os.chdir(stage)
os.execv('/usr/bin/python3',['/usr/bin/python3','-I',str(stage/'nightly_host.py'),str(request)])
'''


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack_payload(files, output):
    if set(files) != set(CODE) or any(
            Path(name).name != name or path.is_symlink() or not path.is_file()
            for name, path in files.items()):
        raise ValueError('Expected the three trusted nightly source files')
    with tarfile.open(output, 'w:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname=name, recursive=False)
    return dict(sha256=digest(output),
                files={
                    name: digest(path)
                    for name, path in files.items()
                })


def cancel_marker(request, checksum):
    return dict(run_id=request['run_id'],
                nonce=request['nonce'],
                request_sha256=checksum,
                cancel=True)


def refresh_credentials():
    """Renew only this controller's short-lived role through GitHub OIDC."""
    url = os.environ['ACTIONS_ID_TOKEN_REQUEST_URL']
    url += ('&' if '?' in url else '?') + urllib.parse.urlencode(
        {'audience': 'sts.amazonaws.com'})
    request = urllib.request.Request(
        url,
        headers={
            'Authorization':
            'Bearer ' + os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']
        })
    with urllib.request.urlopen(request, timeout=30) as response:
        token = json.load(response)['value']
    with tempfile.TemporaryDirectory(prefix='nightly-oidc-') as directory:
        path = Path(directory) / 'token'
        path.touch(mode=0o600)
        path.write_text(token)
        result = subprocess.run([
            'aws', '--region', transport.REGION, 'sts',
            'assume-role-with-web-identity', '--role-arn', ROLE,
            '--role-session-name', 'engine-nightly-refresh',
            '--web-identity-token', 'file://' + str(path),
            '--duration-seconds', '3600', '--output', 'json'
        ],
                                capture_output=True,
                                text=True,
                                timeout=60)
        if result.returncode:
            raise RuntimeError(
                'Nightly controller could not renew its AWS role')
        credentials = json.loads(result.stdout)['Credentials']
    os.environ.update(AWS_ACCESS_KEY_ID=credentials['AccessKeyId'],
                      AWS_SECRET_ACCESS_KEY=credentials['SecretAccessKey'],
                      AWS_SESSION_TOKEN=credentials['SessionToken'])


def aws(*args, **kwargs):
    try:
        return transport.aws(*args, **kwargs)
    except RuntimeError as error:
        if not any(code in str(error)
                   for code in ('ExpiredToken', 'RequestExpired')):
            raise
        refresh_credentials()
        return transport.aws(*args, **kwargs)


def validate_nightly_result(request, receipt, report):
    for key in ('run_id', 'source_sha', 'instance_id', 'nonce', 'wheel',
                'payload', 'inputs'):
        if key not in request or receipt.get(key) != request[key]:
            raise ValueError(f'Nightly result identity differs: {key}')
    if (receipt.get('status') != 'PASS' or receipt.get('run_exit') != 0
            or receipt.get('compare_exit') != 0
            or not receipt.get('processes_ended')
            or receipt.get('cuda_visible_devices') != '0'
            or report.get('format') != 'kumo-engine-nightly-result-v1'
            or report.get('status') != 'MEASURED'
            or report.get('completed_runs') != 16
            or report.get('expected_runs') != 16
            or report.get('input_manifest_sha256')
            != request['inputs']['input_manifest_sha256']
            or report.get('runner_sha256')
            != request['payload']['files']['nightly_check.py']
            or report.get('helper_sha256')
            != request['payload']['files']['wheel_gpu_smoke.py']
            or report.get('runtime', {}).get('wheel_sha256')
            != request['wheel']['sha256']):
        raise ValueError(
            'Expected a completed, bound sixteen-case nightly result')
    if (report.get('reference_only') is not request['reference_only']
            or report.get('reference_sha256')
            != request['inputs']['reference_sha256'] or
            report['runtime'].get('version') != request['wheel']['version']):
        raise ValueError(
            'Nightly mode, reference or installed version differs')
    spec = importlib.util.spec_from_file_location('nightly_validation',
                                                  CODE['nightly_check.py'])
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    try:
        nightly.hardware_profile(report['runtime'])
        nightly.check_results(report['runs'])
        for row in report['runs']:
            nightly.validate_case_measurements(row)
            nightly.validate_quality_measurement(row['task'], row['score'])
            nightly.check_training(row['work'],
                                   row['selected_checkpoint_steps'])
            work = row['work']
            if work.get('restored_tensors', 0) <= 0 or work.get(
                    'reload_checks', 0) < 1:
                raise ValueError('Exact restored-state proof is missing')
            for key in ('fit_ledger', 'core_fit_ledger'):
                ledger = work.get(key, [])
                if (len(ledger) != 1 or ledger[0].get('status') != 'PASS'
                        or ledger[0].get('steps', 0) <= 0
                        or ledger[0].get('cuda_batches', 0) <= 0):
                    raise ValueError(
                        'Physical fit completion proof is missing')
        for row in report['runs']:
            nightly.validate_training_window(
                row['work'].get('training_window', {}))
    except (KeyError, TypeError) as error:
        raise ValueError(
            'Required nightly measurements are missing') from error


def wait_for_command(command_id, output, deadline):
    while time.monotonic() < deadline:
        try:
            result = aws('ssm', 'get-command-invocation', '--command-id',
                         command_id, '--instance-id', transport.INSTANCE)
        except RuntimeError as error:
            if 'InvocationDoesNotExist' not in str(error):
                raise
        else:
            (output / 'ssm-invocation.json').write_text(
                json.dumps(result, indent=2))
            if result['Status'] not in ACTIVE:
                return result
        time.sleep(5)
    raise TimeoutError(
        'The nightly VM did not finish before the controller deadline')


def remote_command(run_id, checksum):
    return shlex.join([
        'systemd-run', '--unit=engine-nightly-' + run_id, '--wait',
        '--collect', '--pipe', '--property=User=ubuntu',
        '--property=Group=ubuntu', '--property=RuntimeMaxSec=11100',
        '--property=TimeoutStopSec=30', '--property=KillMode=control-group',
        '/usr/bin/env', '-i', 'HOME=/home/ubuntu',
        'PATH=/usr/local/bin:/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES=0',
        'NIGHTLY_RUN_ID=' + run_id, 'NIGHTLY_REQUEST_SHA256=' + checksum,
        '/usr/bin/python3', '-I', '-c', BOOTSTRAP
    ])


def run(args):
    if not re.fullmatch(r'[0-9]{1,20}-[0-9]{1,5}',
                        args.run_id) or not re.fullmatch(
                            r'[a-f0-9]{40}', args.source_sha):
        raise ValueError(
            'Expected a GitHub run ID, attempt and exact harness commit')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prefix = f'{PREFIX}/{args.run_id}'
    wheel = transport.wheel_identity(args.wheel,
                                     prefix + '/' + args.wheel.name)
    build = json.loads(args.build_receipt.read_text())
    if (build['wheel_sha256'] != wheel['sha256']
            or build['version'] != wheel['version']
            or build['harness'] != args.source_sha or args.reference_only and
        (wheel['version'] != '1.0.0' or wheel['sha256'] !=
         '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e')):
        raise ValueError('The selected build receipt and wheel differ')
    request = dict(mode='nightly',
                   run_id=args.run_id,
                   source_sha=args.source_sha,
                   instance_id=transport.INSTANCE,
                   nonce=secrets.token_hex(16),
                   reference_only=args.reference_only,
                   wheel=wheel,
                   build=build,
                   inputs=json.loads(args.input_spec.read_text()),
                   payload=pack_payload(CODE, output / 'payload.tar.gz'))
    path = output / 'request.json'
    path.write_text(json.dumps(request, indent=2))
    checksum = digest(path)
    for name, source in [(wheel['filename'], args.wheel),
                         ('payload.tar.gz', output / 'payload.tar.gz'),
                         ('request.json', path)]:
        aws('s3api',
            'put-object',
            '--bucket',
            transport.BUCKET,
            '--key',
            prefix + '/' + name,
            '--body',
            str(source),
            '--server-side-encryption',
            'AES256',
            region='us-west-2')
    command_id = None
    try:
        command = aws(
            'ssm', 'send-command', '--instance-ids', transport.INSTANCE,
            '--document-name', transport.DOCUMENT, '--document-version',
            transport.DOCUMENT_VERSION, '--parameters',
            json.dumps(
                dict(commands=[remote_command(args.run_id, checksum)],
                     executionTimeout=['11200'])), '--timeout-seconds', '60',
            '--comment', 'Nightly sixteen-case model suite')
        command_id = command['Command']['CommandId']
        (output / 'command-id.txt').write_text(command_id + '\n')
        command_path = output / 'command.json'
        command_path.write_text(json.dumps(dict(CommandId=command_id)))
        aws('s3api',
            'put-object',
            '--bucket',
            transport.BUCKET,
            '--key',
            prefix + '/command.json',
            '--body',
            str(command_path),
            '--server-side-encryption',
            'AES256',
            region='us-west-2')
        invocation = wait_for_command(command_id, output,
                                      time.monotonic() + 11300)
        if invocation['Status'] != 'Success' or invocation['ResponseCode'] != 0:
            raise RuntimeError(
                f'Nightly VM failed: {invocation["Status"]}, exit {invocation["ResponseCode"]}'
            )
    except BaseException:
        try:
            stage_cancel(request, checksum, output)
            confirm_stopped(request, checksum, command_id, output)
        except (RuntimeError, TimeoutError):
            print(
                'Remote termination is not confirmed. The VM deadline and host lock remain in force.',
                file=sys.stderr)
        raise
    finally:
        for name in ('receipt.json', 'portable.tar.gz'):
            try:
                aws('s3api',
                    'get-object',
                    '--bucket',
                    transport.BUCKET,
                    '--key',
                    prefix + '/' + name,
                    str(output / name),
                    region='us-west-2')
            except RuntimeError as error:
                print(f'Could not collect {name}: {error}', file=sys.stderr)
    receipt = json.loads((output / 'receipt.json').read_text())
    if receipt.get('request_sha256') != checksum or digest(
            output / 'portable.tar.gz') != receipt.get('portable_sha256'):
        raise ValueError(
            'The remote request or portable-result checksum differs')
    with tarfile.open(output / 'portable.tar.gz') as archive:
        archive.extractall(output / 'results', filter='data')
    report = json.loads(
        (output / 'results/result/nightly-result.json').read_text())
    validate_nightly_result(request, receipt, report)
    print(f'Nightly PASS: {command_id}; sixteen completed primary fits')


def stage_cancel(request, checksum, output):
    marker = output / 'cancel.json'
    marker.write_text(json.dumps(cancel_marker(request, checksum)))
    aws('s3api',
        'put-object',
        '--bucket',
        transport.BUCKET,
        '--key',
        f'{PREFIX}/{request["run_id"]}/cancel.json',
        '--body',
        str(marker),
        '--server-side-encryption',
        'AES256',
        region='us-west-2')


def confirm_stopped(request, checksum, command_id, output):
    if command_id is not None:
        invocation = wait_for_command(command_id, output,
                                      time.monotonic() + 120)
        if invocation['Status'] in ACTIVE:
            raise RuntimeError('Remote termination is not confirmed')
        return dict(status='TERMINATED',
                    processes_ended=True,
                    command_id=command_id)
    end = time.monotonic() + 120
    path = output / 'receipt.json'
    while time.monotonic() < end:
        try:
            aws('s3api',
                'get-object',
                '--bucket',
                transport.BUCKET,
                '--key',
                f'{PREFIX}/{request["run_id"]}/receipt.json',
                str(path),
                region='us-west-2')
        except RuntimeError as error:
            if not any(code in str(error) for code in ('NoSuchKey', '404')):
                raise
        else:
            receipt = json.loads(path.read_text())
            if (all(
                    receipt.get(key) == request[key]
                    for key in ('run_id', 'source_sha', 'nonce',
                                'instance_id'))
                    and receipt.get('request_sha256') == checksum
                    and receipt.get('processes_ended') is True
                    and receipt.get('status') in ('PASS', 'FAIL')):
                return dict(status='TERMINATED',
                            processes_ended=True,
                            command_id=None,
                            evidence='Bound VM receipt')
        time.sleep(3)
    raise TimeoutError(
        'Remote termination is not confirmed. Preserve the VM lock and deadline.'
    )


def cancel_run(args):
    if not re.fullmatch(r'[0-9]{1,20}-[0-9]{1,5}', args.run_id):
        raise ValueError('Invalid GitHub run ID and attempt')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    prefix = f'{PREFIX}/{args.run_id}'
    path = output / 'request.json'
    try:
        aws('s3api',
            'get-object',
            '--bucket',
            transport.BUCKET,
            '--key',
            prefix + '/request.json',
            str(path),
            region='us-west-2')
    except RuntimeError as error:
        if any(code in str(error) for code in ('NoSuchKey', '404')):
            result = dict(status='NO_REQUEST_STAGED',
                          model_dispatch_not_possible=True)
            (output / 'cleanup.json').write_text(json.dumps(result))
            return result
        raise
    request = json.loads(path.read_text())
    if (request.get('run_id') != args.run_id
            or request.get('source_sha') != args.source_sha
            or request.get('instance_id') != transport.INSTANCE
            or request.get('mode') != 'nightly' or not request.get('nonce')):
        raise ValueError('Cancellation request identity differs')
    command = output / 'command.json'
    try:
        aws('s3api',
            'get-object',
            '--bucket',
            transport.BUCKET,
            '--key',
            prefix + '/command.json',
            str(command),
            region='us-west-2')
    except RuntimeError as error:
        if not any(code in str(error) for code in ('NoSuchKey', '404')):
            raise
        command_id = None
    else:
        command_id = json.loads(command.read_text())['CommandId']
    checksum = digest(path)
    stage_cancel(request, checksum, output)
    result = confirm_stopped(request, checksum, command_id, output)
    (output / 'cleanup.json').write_text(json.dumps(result, indent=2))
    return result


def interrupt(*_):
    raise KeyboardInterrupt('The nightly controller received termination')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--build-receipt', type=Path)
    parser.add_argument('--input-spec', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--reference-only', action='store_true')
    parser.add_argument('--cancel-only', action='store_true')
    signal.signal(signal.SIGTERM, interrupt)
    args = parser.parse_args()
    started = time.monotonic()
    existed = args.output_dir.exists()
    status = 'FAIL'
    try:
        if args.cancel_only:
            cancel_run(args)
        else:
            if not all((args.wheel, args.build_receipt, args.input_spec)):
                parser.error(
                    'Running models requires --wheel, --build-receipt and --input-spec'
                )
            run(args)
        status = 'PASS'
    finally:
        if not existed and args.output_dir.is_dir():
            (args.output_dir / 'controller-job-time.json').write_text(
                json.dumps(
                    dict(status=status,
                         seconds=time.monotonic() - started,
                         scope='cancel controller CLI'
                         if args.cancel_only else 'nightly controller CLI')))


if __name__ == '__main__':
    main()
