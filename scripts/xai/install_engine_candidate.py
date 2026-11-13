"""Install a standalone Engine wheel in a new, isolated test environment.

This is an exact-byte probe installer, not the stress harness fat-wheel path.
Run on an admitted dedicated test host; it provisions no cloud resources.
Dependencies must be provided as a hashed requirements file and local wheelhouse.
"""
import argparse
import email
import hashlib
import json
import re
import subprocess
import sys
import venv
import zipfile
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def install(args):
    for path, expected in ((args.wheel, args.wheel_sha256),
                           (args.requirements, args.requirements_sha256)):
        if not re.fullmatch('[a-f0-9]{64}',
                            expected) or digest(path) != expected:
            raise ValueError('Candidate input checksum mismatch')
    if not re.fullmatch('[a-f0-9]{40}', args.source_revision):
        raise ValueError('Source revision must be a full commit SHA')
    if not args.wheel.name.startswith('kumo_engine-'):
        raise ValueError('Expected standalone kumo_engine wheel')
    with zipfile.ZipFile(args.wheel) as archive:
        metadata_file, = [
            name for name in archive.namelist()
            if name.endswith('.dist-info/METADATA')
        ]
        metadata = email.message_from_bytes(archive.read(metadata_file))
        suffix = metadata['Version'].partition('+')[2]
        if not re.fullmatch(
                '[a-f0-9]{7,40}',
                suffix) or not args.source_revision.startswith(suffix):
            raise ValueError(
                'Wheel version does not identify the requested source revision')
    if args.venv.exists():
        raise ValueError('Refusing to reuse an existing environment')
    if sys.version_info[:2] != (3, 12):
        raise ValueError(
            'Run installer with the candidate Python 3.12 interpreter')
    args.venv.mkdir(parents=True, exist_ok=False)
    venv.EnvBuilder(with_pip=True).create(args.venv)
    python = args.venv / 'bin/python'
    command = [str(python), '-I', '-m', 'pip', '--isolated']
    subprocess.run(
        command + [
            'install', '--no-index', '--find-links',
            str(args.wheelhouse), '--require-hashes', '-r',
            str(args.requirements)
        ], check=True)
    subprocess.run(
        command + ['install', '--no-index', '--no-deps',
                   str(args.wheel)], check=True)
    subprocess.run(command + ['check'], check=True)
    verification = r'''
import hashlib, importlib, json, sys, zipfile
from pathlib import Path
wheel = Path(sys.argv[1])
modules = {}
with zipfile.ZipFile(wheel) as archive:
    for name in ('kumo.engine', 'kumo_core', 'kumo_core.explain.walk_attribution'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(Path(sys.prefix).resolve()):
            raise RuntimeError('Import escaped isolated environment')
        member = name.replace('.', '/')
        member += '/__init__.py' if hasattr(module, '__path__') else '.py'
        expected = archive.read(member)
        if expected != path.read_bytes():
            raise RuntimeError('Installed bytes differ from candidate wheel')
        modules[name] = {'path': str(path), 'sha256': hashlib.sha256(expected).hexdigest()}
print(json.dumps(modules))
'''
    modules = json.loads(
        subprocess.check_output(
            [str(python), '-I', '-c', verification,
             str(args.wheel)], text=True))
    frozen = subprocess.check_output(command + ['freeze', '--all'], text=True)
    (args.venv / 'installed-requirements.txt').write_text(frozen)
    receipt = {
        'schema': 'engine-candidate-install/v1',
        'source_revision': args.source_revision,
        'wheel_version': metadata['Version'],
        'wheel_sha256': args.wheel_sha256,
        'requirements_sha256': args.requirements_sha256,
        'installer_sha256': digest(__file__),
        'modules': modules,
        'python': sys.version,
        'scope': 'installation only; no workload or GPU qualification'
    }
    (args.venv / 'candidate-install.json').write_text(
        json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wheel', 'requirements', 'wheelhouse', 'venv'):
        parser.add_argument('--' + name, required=True,
                            type=lambda p: Path(p).absolute())
    for name in ('wheel-sha256', 'requirements-sha256', 'source-revision'):
        parser.add_argument('--' + name, required=True)
    install(parser.parse_args())
