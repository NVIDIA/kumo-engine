# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build one clean product revision and bind its exact wheel bytes."""
import email
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def build(product, output):
    product, output = Path(product).resolve(), Path(output).resolve()

    def git(*args):
        return subprocess.check_output(
            ['git', '-C', str(product), *args], text=True).strip()

    if git('status', '--porcelain'):
        raise ValueError('Build the clean selected product tree')
    revision, tree = git('rev-parse', 'HEAD', 'HEAD^{tree}').splitlines()
    expected = (product / 'VERSION').read_text().strip() + '+' + git(
        'rev-parse', '--short', 'HEAD')
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([
        str(product / 'packages/kumo-engine/scripts/build_wheel.sh'),
        '--output-dir',
        str(output), '--kumo-repo',
        str(product), '--version', expected
    ], cwd=product, check=True)
    if git('status', '--porcelain') or git(
            'rev-parse', 'HEAD',
            'HEAD^{tree}').splitlines() != [revision, tree]:
        raise ValueError('Product source changed during the build')
    [wheel] = output.glob('*.whl')
    with zipfile.ZipFile(wheel) as archive:
        [member] = [
            n for n in archive.namelist() if n.endswith('.dist-info/METADATA')
        ]
        metadata = email.message_from_bytes(archive.read(member))
    if metadata['Name'].replace(
            '_', '-') != 'kumo-engine' or metadata['Version'] != expected:
        raise ValueError(
            'Wheel does not identify the selected product revision')
    checksum = hashlib.sha256()
    with wheel.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    result = dict(revision=revision, tree=tree, version=expected,
                  wheel_sha256=checksum.hexdigest(),
                  harness=os.environ.get('GITHUB_SHA'))
    (output / 'build.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    build(*sys.argv[1:])
