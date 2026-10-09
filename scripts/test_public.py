# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and run the selected public test groups without cloud credentials."""

import argparse
import contextlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from test_public_connectors import SUITES, select_test_file

ROOT = Path(__file__).resolve().parents[1]

GROUPS = {
    **SUITES,
    'hosted-sdk': ('packages/kumo-sdk/test/connector/local',),
    'local-sdk': ('packages/kumo-engine/test/connector',),
    'mixed-files': ('test/connector/local', 'test/io_utils/fsutil/local'),
    'pandas': ('test/graph/local',),
    'minio': ('test/io_utils/fsutil/minio',),
    'shared-sql': ('packages/kumo-core/tests/sources/sql',
                   'packages/kumo-snowflake/tests/native'),
}


@contextlib.contextmanager
def report_file(path):
    """Replace host reports atomically, without following an existing link."""
    stream = tempfile.NamedTemporaryFile(mode='w', dir=path.parent,
                                         delete=False)
    try:
        with stream:
            yield stream
    finally:
        os.replace(stream.name, path)


def run_group(group, image, reports, compose, tls, run_id,
              test_file=None, keyword=None):
    name = f'{run_id}-{group}'
    paths = (test_file,) if test_file else GROUPS[group]
    row = {'paths': paths, 'exit_code': 1}
    xml = reports / f'{group}.xml'
    with tempfile.TemporaryDirectory(prefix='public-test-output-') as directory, \
            report_file(reports / f'{group}.log') as log:
        # Only this child is shared. Its private parent excludes other host users.
        output = Path(directory) / 'output'
        output.mkdir(mode=0o777)
        output.chmod(0o777)
        try:
            network = 'none'
            if group == 'minio':
                subprocess.run(compose + ['build', 'minio'], check=True,
                               timeout=900, stdout=log,
                               stderr=subprocess.STDOUT)
                subprocess.run([
                    'openssl',
                    'req',
                    '-x509',
                    '-nodes',
                    '-newkey',
                    'rsa:2048',
                    '-days',
                    '1',
                    '-subj',
                    '/CN=localhost',
                    '-addext',
                    'subjectAltName=DNS:localhost,DNS:minio,IP:127.0.0.1',
                    '-keyout',
                    f'{tls}/private.key',
                    '-out',
                    f'{tls}/public.crt',
                ], check=True, stdout=log, stderr=subprocess.STDOUT)
                Path(tls, 'public.crt').chmod(0o644)
                subprocess.run(compose + ['up', '-d', 'minio'], check=True,
                               timeout=180, stdout=log,
                               stderr=subprocess.STDOUT)
                subprocess.run(compose + ['run', '--rm', 'init'], check=True,
                               timeout=120, stdout=log,
                               stderr=subprocess.STDOUT)
                container = subprocess.check_output(
                    compose + ['ps', '-q', 'minio'], text=True).strip()
                network = f'container:{container}'
            command = [
                'docker',
                'run',
                '--rm',
                '--cpus',
                '1',
                '--name',
                name,
                '--network',
                network,
                '--cap-drop',
                'ALL',
                '--security-opt',
                'no-new-privileges',
                '--tmpfs',
                '/tmp:rw,nosuid,nodev,size=2g',
                '-v',
                f'{output}:/reports',
            ]
            if group == 'minio':
                command += [
                    '-v', f'{tls}:/tls:ro', '-e',
                    'KUMO_MINIO_TEST_ENDPOINT=https://127.0.0.1:9000', '-e',
                    'KUMO_MINIO_TEST_CA=/tls/public.crt'
                ]
            if group in SUITES:
                command += [
                    '--read-only', image, 'python', '-I',
                    'scripts/test_public_connectors.py', '--suite', group,
                    '--junitxml', f'/reports/{group}.xml'
                ]
            else:
                command += [
                    image, 'python', '-m', 'pytest', '-vv', '-ra', '-o',
                    'addopts=', '--import-mode=importlib', '-o',
                    'cache_dir=/tmp/pytest_cache', '-o',
                    'faulthandler_timeout=120', '--durations=10',
                    '--rootdir=/src',
                    f'--junitxml=/reports/{group}.xml'
                ]
            if group in ('hosted-sdk', 'local-sdk'):
                package = 'kumo-sdk' if group == 'hosted-sdk' else 'kumo-engine'
                command += [
                    '-c', f'packages/{package}/pyproject.toml',
                    f'--confcutdir=packages/{package}'
                ]
            elif group not in SUITES:
                # Load this group's fixtures without unrelated hosted/model setup.
                config = ('packages/kumo-core/pyproject.toml' if group
                          == 'shared-sql' else 'tests/public_setup/pytest.ini')
                command += ['-c', config, f'--confcutdir={GROUPS[group][0]}']
            if group not in SUITES:
                command += [f'--ignore={path}/live' for path in GROUPS[group]]
                command += paths
            elif test_file:
                command += ['--test-file', test_file]
            if keyword is not None:
                command += ['-k', keyword]
            row['command'] = command
            print(f'Running {group}', flush=True)
            result = subprocess.run(command, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1200)
            row['exit_code'] = result.returncode
        except (OSError, subprocess.SubprocessError) as error:
            row.update(exit_code=1, error=str(error))
        finally:
            # A timed-out Docker CLI can leave its container running.
            cleanup_commands = [(['docker', 'rm', '-f', name], 60)]
            if group == 'minio':
                cleanup_commands.append((compose + ['down', '-v'], 120))
            for command, timeout in cleanup_commands:
                try:
                    cleanup = subprocess.run(command, stdout=log,
                                             stderr=subprocess.STDOUT,
                                             timeout=timeout)
                    if command == compose + ['down', '-v'
                                             ] and cleanup.returncode:
                        row.update(exit_code=1,
                                   cleanup_error=cleanup.returncode)
                except (OSError, subprocess.SubprocessError) as error:
                    row.update(exit_code=1, cleanup_error=str(error))
        try:
            # The container is gone. Accept only its regular XML file, never links.
            descriptor = os.open(output / xml.name,
                                 os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor) as source:
                if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                    raise ValueError('Test report must be a regular file')
                suites = ET.parse(source).getroot().findall('testsuite')
                row.update({
                    key: sum(int(s.attrib.get(key, 0)) for s in suites)
                    for key in ('tests', 'failures', 'errors', 'skipped')
                })
                row['passed'] = (row['tests'] - row['failures'] -
                                 row['errors'] - row['skipped'])
                source.seek(0)
                with report_file(xml) as destination:
                    shutil.copyfileobj(source, destination)
            if not row['passed'] or row['failures'] or row['errors']:
                row['exit_code'] = 1
        except (OSError, ValueError, ET.ParseError) as error:
            row.update(exit_code=1, report_error=str(error))
    return row


def build_wheel(destination):
    """Build the native prerequisite from this checkout, using public tools."""
    settings = dict(
        line.split('=', 1) for line in subprocess.check_output(
            ['./scripts/diskgraph', 'wheel-env'], text=True).splitlines())
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        output = Path(directory) / 'wheel'
        command = [
            'docker', 'buildx', 'build', '--file',
            'tests/public_setup/Dockerfile.diskgraph', '--output',
            f'type=local,dest={output}'
        ]
        for key, value in settings.items():
            command += ['--build-arg', f'{key}={value}']
        subprocess.run(command + ['.'], check=True)
        # The local exporter retains old files. Replace only after success.
        if destination.exists():
            shutil.rmtree(destination)
        output.rename(destination)


def build_image(profile, image, wheel):
    dockerfile = ('tests/public/Dockerfile' if profile == 'connectors' else
                  'tests/public_setup/Dockerfile')
    command = ['docker', 'build', '--file', dockerfile, '--tag', image]
    if profile != 'connectors':
        command += ['--target', profile]
    if profile == 'full':
        command += ['--build-context', f'wheel={wheel}']
    subprocess.run(command + ['.'], check=True)


def image_environment(image):
    command = [
        'docker', 'run', '--rm', '--network', 'none', '--read-only', image,
        'python', '-I', '-c', 'import importlib.metadata as m, json, platform; '
        'print(json.dumps({"python": platform.python_version(), '
        '"architecture": platform.machine(), '
        '"packages": {d.metadata["Name"]: d.version for d in m.distributions()}}))'
    ]
    environment = json.loads(subprocess.check_output(command, text=True))
    environment['image_id'] = subprocess.check_output(
        ['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
        text=True).strip()
    return environment


def interrupted(signum, frame):
    # Let run_group's finally block remove the test container and MinIO.
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=(*GROUPS, 'connectors', 'all'),
                        default='all')
    parser.add_argument('--reports', type=Path, default=Path('public-reports'))
    parser.add_argument('--test-file', type=Path, help='one repository-relative test file')
    parser.add_argument('-k', help='pytest name expression for one group')
    wheel_options = parser.add_mutually_exclusive_group()
    wheel_options.add_argument(
        '--wheel-dir', type=Path,
        help='reuse a DiskGraph wheel built from this checkout')
    wheel_options.add_argument(
        '--build-wheel', action='store_true',
        help='only build DiskGraph into REPORTS/wheel for CI')
    args = parser.parse_args()
    if (args.test_file or args.k is not None) and (
            args.group not in GROUPS or args.build_wheel):
        parser.error('--test-file and -k require one --group without --build-wheel')
    test_file = select_test_file(
        parser, args.test_file, GROUPS.get(args.group, ()),
        ('live', 'native') if args.group in SUITES else ('live',))
    if test_file:
        test_file = str(test_file.relative_to(ROOT))
    reports = args.reports.absolute()
    os.chdir(ROOT)
    reports.mkdir(mode=0o700, parents=True, exist_ok=True)
    if reports.is_symlink() or reports.stat().st_mode & 0o022:
        parser.error(
            '--reports must be a real directory writable only by its owner; choose a new directory'
        )
    selected = (tuple(GROUPS) if args.group == 'all' else
                tuple(SUITES) if args.group == 'connectors' else (args.group, ))
    if not args.build_wheel:
        # A setup failure must not leave successful reports from an earlier run.
        (reports / 'results.json').unlink(missing_ok=True)
        for group in selected:
            for suffix in ('xml', 'log'):
                (reports / f'{group}.{suffix}').unlink(missing_ok=True)
    needs_native = args.build_wheel or any(group not in (*SUITES, 'hosted-sdk')
                                           for group in selected)
    if needs_native:
        architecture = subprocess.check_output(
            ['docker', 'info', '--format', '{{.Architecture}}'],
            text=True).strip()
        if architecture not in ('x86_64', 'amd64'):
            parser.error(
                'The native test setup requires an x86_64 Docker host. '
                'Use --group connectors for the ARM64 setup.')
    wheel = args.wheel_dir.resolve() if args.wheel_dir else reports / 'wheel'
    if needs_native and (args.build_wheel or not args.wheel_dir):
        build_wheel(wheel)
    if args.build_wheel:
        return 0
    if needs_native and len(list(wheel.glob('diskgraph-*.whl'))) != 1:
        parser.error(
            '--wheel-dir must contain exactly one matching DiskGraph wheel')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       text=True).strip()
    run_id = f'public-tests-{uuid.uuid4().hex[:12]}'
    compose = [
        'docker', 'compose', '-p', run_id, '-f',
        'tests/public_setup/compose.yml'
    ]
    results = {
        'revision':
        revision,
        'dirty':
        bool(
            subprocess.check_output(['git', 'status', '--porcelain'],
                                    text=True).strip()),
        'environments': {},
        'groups': {},
    }
    images = {}
    build_errors = {}
    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        with tempfile.TemporaryDirectory() as tls:
            os.environ['PROOF_TLS_DIR'] = tls
            os.environ['PUBLIC_TEST_RUN_ID'] = run_id
            Path(tls).chmod(0o755)
            for group in selected:
                profile = ('connectors' if group in SUITES else
                           'sdk' if group == 'hosted-sdk' else 'full')
                if profile not in images:
                    image = f'{run_id}-{profile}'
                    images[profile] = image
                    try:
                        build_image(profile, image, wheel)
                        results['environments'][profile] = image_environment(
                            image)
                    except (OSError, ValueError,
                            subprocess.SubprocessError) as error:
                        build_errors[profile] = str(error)
                try:
                    row = ({
                        'exit_code': 1,
                        'build_error': build_errors[profile]
                    } if profile in build_errors else run_group(
                        group, images[profile], reports, compose, tls, run_id,
                        test_file, args.k))
                except (OSError, ValueError,
                        subprocess.SubprocessError) as error:
                    row = {'exit_code': 1, 'error': str(error)}
                row['environment'] = profile
                results['groups'][group] = row
                with report_file(reports / 'results.json') as summary:
                    summary.write(json.dumps(results, indent=2) + '\n')
                print(f'{group}: {json.dumps(row)}', flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        # Keep Docker's build cache; remove only image tags from this run.
        if 'minio' in selected:
            images['minio'] = f'{run_id}-minio'
        for image in images.values():
            subprocess.run(['docker', 'image', 'rm', image], check=False)
    return int(
        any(row['exit_code'] or not row.get('passed')
            for row in results['groups'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
