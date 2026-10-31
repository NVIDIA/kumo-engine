#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install and run nightly in one unprivileged, bounded job on the fixed VM."""
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

BASE = Path('/opt/dlami/nvme/engine-nightly')
BUCKET = 'kumo-unit-test'
PREFIX = 'diskgraph-profiles/engine-nightly/ssm-integration/nightly'
INSTANCE = 'i-06dd0e3c558780731'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def host_lock(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream.fileno()


def group_alive(group):
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            state, _, pgroup, *_ = path.read_text().rsplit(') ', 1)[1].split()
            if int(pgroup) == group and state != 'Z':
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def run_process(command,
                log_path,
                deadline,
                cancelled,
                *,
                env=None,
                cwd=None,
                lock_fd=None):
    with Path(log_path).open('a') as log:
        child = subprocess.Popen(command,
                                 stdout=log,
                                 stderr=subprocess.STDOUT,
                                 start_new_session=True,
                                 env=env,
                                 cwd=cwd,
                                 pass_fds=() if lock_fd is None else
                                 (lock_fd, ))
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Nightly reached its VM-side deadline')
                if cancelled():
                    raise InterruptedError(
                        'Nightly received its bound cancellation marker')
                time.sleep(.1)
            if child.returncode:
                raise subprocess.CalledProcessError(child.returncode, command)
        finally:
            # Keep the host lock until this whole process group stops.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if group_alive(child.pid):
                    try:
                        os.killpg(child.pid, sig)
                    except ProcessLookupError:
                        pass
                    end = time.monotonic() + (5
                                              if sig == signal.SIGTERM else 1)
                    while group_alive(child.pid) and time.monotonic() < end:
                        time.sleep(.05)
            child.wait()
            while group_alive(child.pid):
                time.sleep(.1)


def portable_archive(root, output):
    skipped = {
        'work', 'published-model', 'prediction-inputs', 'prediction-outputs',
        'env', 'tools', 'inputs', 'cache', 'home'
    }
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(root.rglob('*')):
            relative = path.relative_to(root)
            if (path.is_file() and not path.is_symlink()
                    and not skipped.intersection(relative.parts)
                    and path.suffix in ('.json', '.log', '.txt')
                    and not path.name.endswith('-ranked.json')):
                archive.add(path, arcname=str(relative), recursive=False)


def capacity(output):
    disk = shutil.disk_usage(BASE)
    ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    values = dict(
        status='INCOMPLETE',
        cpu_count=os.cpu_count(),
        system_memory_bytes=ram,
        nvme_total_bytes=disk.total,
        nvme_available_bytes=disk.free,
        available_memory_bytes=next(
            int(line.split()[1]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith('MemAvailable:')))
    try:
        if (os.geteuid() == 0 or sys.version_info[:2] != (3, 12)
                or Path('/sys/devices/virtual/dmi/id/board_asset_tag'
                        ).read_text().strip() != INSTANCE
                or os.environ.get('CUDA_VISIBLE_DEVICES') != '0'):
            raise ValueError(
                'Use Python3.12 as the unprivileged user on the selected VM with GPU0'
            )
        rows = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
            '--format=csv,noheader,nounits'
        ],
                                       text=True,
                                       timeout=15)
        gpus = [
            dict(index=int(r[0]),
                 model=r[1].strip(),
                 memory_mib=int(r[2]),
                 driver=r[3].strip()) for r in csv.reader(rows.splitlines())
        ]
        values['physical_gpus'] = gpus
        if len(gpus) != 2 or any('RTX PRO 6000' not in gpu['model']
                                 for gpu in gpus):
            raise ValueError(
                'The selected VM requires its two physical RTX PRO 6000 GPUs')
        processes = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'
        ],
                                            text=True,
                                            timeout=15)
        if processes.strip():
            raise ValueError(
                'The selected GPU host is busy. Preserve the other workload.')
        if os.cpu_count() != 48 or ram < 480 * 2**30 or disk.total < 3 * 2**40:
            raise ValueError('The selected g7e.12xlarge capacity differs: ' +
                             json.dumps(values))
        values['status'] = 'MEASURED'
        return values
    finally:
        output.write_text(json.dumps(values, indent=2) + '\n')


def main(request_path):
    job = request_path.parent.resolve()
    raw = request_path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != os.environ['NIGHTLY_REQUEST_SHA256']:
        raise ValueError('The staged request checksum differs')
    request = json.loads(raw)
    if (request['instance_id'] != INSTANCE or request['mode'] != 'nightly'
            or not re.fullmatch(r'[0-9]{1,20}-[0-9]{1,5}', request['run_id'])):
        raise ValueError('The request targets another workload or host')
    prefix = f's3://{BUCKET}/{PREFIX}/{request["run_id"]}'
    receipt = dict(request,
                   status='FAIL',
                   request_sha256=checksum,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   remote_directory=str(job),
                   run_exit=None,
                   compare_exit=None,
                   processes_ended=False)
    started = time.monotonic()
    deadline = started + 10800
    last_check = [0.]

    def aws(*args):
        return subprocess.run(['aws', '--region', 'us-west-2', *args],
                              capture_output=True,
                              text=True,
                              timeout=30)

    def cancelled():
        now = time.monotonic()
        if now - last_check[0] < 10:
            return False
        last_check[0] = now
        response = aws('s3', 'cp', prefix + '/cancel.json', '-',
                       '--only-show-errors')
        if response.returncode:
            if any(code in response.stderr for code in ('404', 'NoSuchKey')):
                return False
            raise RuntimeError(
                'Could not read the nightly cancellation marker: ' +
                response.stderr)
        marker = json.loads(response.stdout)
        return marker == dict(run_id=request['run_id'],
                              nonce=request['nonce'],
                              request_sha256=checksum,
                              cancel=True)

    def execute(command, log='setup.log', **kwargs):
        return run_process(command,
                           job / log,
                           deadline,
                           cancelled,
                           cwd=job,
                           lock_fd=lock[0],
                           **kwargs)

    def download(bucket, key, target, expected):
        execute([
            'aws', '--region', 'us-west-2', 's3', 'cp', f's3://{bucket}/{key}',
            str(target), '--only-show-errors'
        ])
        if digest(target) != expected:
            raise ValueError('Downloaded artifact checksum differs: ' +
                             target.name)

    lock = [None]
    temporary = None
    try:
        with host_lock(BASE / 'host.lock') as lock[0]:
            receipt['capacity'] = capacity(job / 'capacity.json')
            inputs_spec = request['inputs']
            legacy = Path(
                '/opt/dlami/nvme/pr137-local-validation-20261030/reference/engine-nightly/inputs'
            )
            if (legacy / 'manifest.json').is_file() and digest(
                    legacy /
                    'manifest.json') == inputs_spec['input_manifest_sha256']:
                inputs = legacy
            else:
                archive = job / 'inputs.tar'
                download(inputs_spec['bucket'], inputs_spec['key'], archive,
                         inputs_spec['sha256'])
                inputs = job / 'inputs'
                with tarfile.open(archive) as source:
                    source.extractall(inputs, filter='data')
            reference = job / 'reference.json'
            download(inputs_spec['bucket'], inputs_spec['reference_key'],
                     reference, inputs_spec['reference_sha256'])
            wheel = job / request['wheel']['filename']
            if wheel.name != request['wheel']['filename']:
                raise ValueError('Invalid wheel filename')
            download(BUCKET, request['wheel']['key'], wheel,
                     request['wheel']['sha256'])
            for name, expected in request['payload']['files'].items():
                if Path(name).name != name or digest(job / name) != expected:
                    raise ValueError('The staged source checksum differs')
            execute([sys.executable, '-m', 'venv', str(job / 'tools')])
            tools_python = str(job / 'tools/bin/python')
            execute([
                tools_python, '-I', '-m', 'pip', '--isolated', 'install',
                '--index-url', 'https://pypi.org/simple', 'uv==0.11.28'
            ])
            execute([sys.executable, '-m', 'venv', str(job / 'env')])
            python = str(job / 'env/bin/python')
            install = [
                str(job / 'tools/bin/uv'), '--no-config', 'pip', 'install',
                '--python', python, '--cache-dir',
                str(job / 'cache'), '--default-index',
                'https://pypi.org/simple'
            ]
            if request['reference_only']:
                manifest_path = inputs / 'manifest.json'
                if digest(
                        manifest_path) != inputs_spec['input_manifest_sha256']:
                    raise ValueError(
                        'Retained input manifest checksum differs')
                manifest = json.loads(manifest_path.read_text())
                relative = manifest['reference_dependencies']
                profile_path = (inputs / relative).resolve()
                if not profile_path.is_relative_to(inputs.resolve()) or digest(
                        profile_path) != manifest['files'][relative]:
                    raise ValueError('Frozen dependency profile differs')
                versions = json.loads(profile_path.read_text())
                if (not versions or not isinstance(versions, dict)
                        or 'kumo-engine' in versions
                        or any(not re.fullmatch(r'[a-z0-9][a-z0-9._-]*', name)
                               or not re.fullmatch(r'[0-9][0-9A-Za-z.!+_-]*',
                                                   version)
                               for name, version in versions.items())):
                    raise ValueError('Invalid frozen package-version profile')
                requirements = job / 'requirements.txt'
                requirements.write_text(''.join(
                    f'{name}=={version}\n'
                    for name, version in sorted(versions.items())))
                execute([*install, '-r', str(requirements)])
            else:
                execute([*install, str(wheel), 'boto3>=1.34,<2'])
            result = job / 'result'
            setup = result / 'setup'
            setup.mkdir(parents=True)
            (setup / 'build.json').write_text(
                json.dumps(request['build'], indent=2))
            (setup / 'capacity.json').write_text(
                json.dumps(receipt['capacity'], indent=2))
            execute([
                python, '-I', '-m', 'pip', '--isolated', 'install',
                '--no-index', '--no-deps', '--force-reinstall', '--no-compile',
                '--report',
                str(setup / 'install.json'),
                str(wheel)
            ])
            execute([python, '-I', '-m', 'pip', '--isolated', 'check'])
            code = '''import hashlib,importlib.metadata,json,runpy,sys
from pathlib import Path
wheel,setup,harness=sys.argv[1:]
helper=runpy.run_path(harness)
count=helper['verify_installed_files'](Path(wheel),importlib.metadata.distribution('kumo-engine'))
report=json.loads((Path(setup)/'install.json').read_text())
entries=[r for r in report['install'] if r['metadata']['name'].replace('_','-').lower()=='kumo-engine']
sha=hashlib.sha256(Path(wheel).read_bytes()).hexdigest()
assert len(entries)==1 and entries[0]['download_info']['archive_info']['hashes']['sha256']==sha
(Path(setup)/'wheel.json').write_text(json.dumps(dict(filename=Path(wheel).name,sha256=sha,version=importlib.metadata.version('kumo-engine'),installed_files=count)))'''
            execute([
                python, '-I', '-c', code,
                str(wheel),
                str(setup),
                str(job / 'nightly_check.py')
            ])
            shutil.copyfile(reference, result / 'reference.json')
            (job / 'home').mkdir()
            temporary = tempfile.TemporaryDirectory(prefix='kn-', dir='/tmp')
            environment = dict(HOME=str(job / 'home'),
                               PATH='/usr/local/bin:/usr/bin:/bin',
                               TMPDIR=temporary.name,
                               CUDA_VISIBLE_DEVICES='0',
                               OMP_NUM_THREADS='4',
                               MKL_NUM_THREADS='4',
                               OPENBLAS_NUM_THREADS='4',
                               PYTHONDONTWRITEBYTECODE='1',
                               GITHUB_SHA=request['source_sha'],
                               AWS_EC2_METADATA_DISABLED='true',
                               AWS_SHARED_CREDENTIALS_FILE='/dev/null',
                               AWS_CONFIG_FILE='/dev/null',
                               PIP_CONFIG_FILE='/dev/null')
            flags = [
                '--input-sha256', inputs_spec['input_manifest_sha256'],
                '--reference',
                str(reference), '--reference-sha256',
                inputs_spec['reference_sha256']
            ]
            if request['reference_only']:
                flags.append('--reference-only')
            execute([
                python, '-I',
                str(job / 'nightly_check.py'), 'run', '--input-root',
                str(inputs), '--output-dir',
                str(result), '--expected-version', request['wheel']['version'],
                *flags
            ],
                    log='run.log',
                    env=environment)
            receipt['run_exit'] = 0
            execute([
                python, '-I',
                str(job / 'nightly_check.py'), 'compare', '--input-root',
                str(result / 'input-metadata'), '--results-dir',
                str(result), '--output-dir',
                str(job / 'summary'), *flags
            ],
                    log='compare.log',
                    env=environment)
            receipt.update(compare_exit=0, status='PASS')
    except BaseException as error:
        receipt.update(error=f'{type(error).__name__}: {error}',
                       traceback=traceback.format_exc())
    finally:
        if temporary is not None:
            temporary.cleanup()
        receipt.update(processes_ended=True,
                       seconds=time.monotonic() - started)
        archive = job / 'portable.tar.gz'
        portable_archive(job, archive)
        receipt['portable_sha256'] = digest(archive)
        (job / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        for name in ('portable.tar.gz', 'receipt.json'):
            response = aws('s3', 'cp', str(job / name), prefix + '/' + name,
                           '--only-show-errors')
            if response.returncode:
                raise RuntimeError('Could not retain nightly evidence: ' +
                                   response.stderr)
    if receipt['status'] != 'PASS':
        raise SystemExit(receipt.get('error', 'Nightly failed'))


def interrupt(*_):
    raise KeyboardInterrupt('Nightly host received termination')


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt)
    main(Path(sys.argv[1]))
