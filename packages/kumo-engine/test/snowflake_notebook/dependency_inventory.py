# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inventory an existing dependency archive without extracting its members."""
import argparse
import hashlib
import json
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def inventory(archive, plan):
    wheels = {}
    with tarfile.open(archive, 'r:*') as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if not name.endswith('.whl') or name in wheels:
                raise ValueError(
                    'Dependency archive must contain unique wheel files')
            digest = hashlib.sha256()
            with bundle.extractfile(
                    member) as source, tempfile.TemporaryFile() as local:
                while data := source.read(1024 * 1024):
                    digest.update(data)
                    local.write(data)
                local.seek(0)
                with zipfile.ZipFile(local) as wheel:
                    metadata = [
                        path for path in wheel.namelist()
                        if path.endswith('.dist-info/METADATA')
                        and len(Path(path).parts) == 2
                    ]
                    if len(metadata) != 1:
                        raise ValueError(
                            'Wheel must contain exactly one distribution metadata file'
                        )
                    record = BytesParser().parsebytes(wheel.read(metadata[0]))
                if not record['Name'] or not record['Version'] or record[
                        'Name'] == 'kumo-engine':
                    raise ValueError(
                        'Dependency metadata is incomplete or includes the candidate'
                    )
            wheels[name] = dict(sha256=digest.hexdigest(), size=member.size,
                                name=record['Name'], version=record['Version'])
    if not wheels or not set(plan['local_wheels']) <= wheels.keys():
        raise ValueError(
            'Dependency archive is empty or misses an authored local wheel')
    with Path(archive).open('rb') as stream:
        archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(plan, archive_sha256=archive_hash,
                wheels=dict(sorted(wheels.items())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--plan', type=Path,
                        default=root / 'dependency-plan.json')
    parser.add_argument('--output', type=Path,
                        default=root / 'dependencies.json')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = inventory(args.archive, json.loads(args.plan.read_text()))
    if args.check:
        if result != json.loads(args.output.read_text()):
            raise SystemExit('Dependency inventory differs from the archive')
    else:
        args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
