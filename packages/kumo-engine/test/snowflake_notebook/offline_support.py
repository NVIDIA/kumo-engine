# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify and unpack the flat, reviewed offline wheel set."""

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unpack(payload, destination):
    payload, destination = Path(payload), Path(destination)
    manifest = json.loads((payload / 'wheelhouse-manifest.json').read_text())
    archive = payload / 'wheelhouse.tar'
    if sha(archive) != manifest['archive_sha256']:
        raise ValueError('Wheel archive checksum mismatch')
    expected = manifest['wheels']
    if destination.exists():
        verify(destination, expected)
        return manifest
    with tempfile.TemporaryDirectory(
        prefix=destination.name + '-pending-', dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, mode='r:') as bundle:
            members = bundle.getmembers()
            if len(members) != len(expected) or {m.name for m in members} != set(
                expected
            ):
                raise ValueError('Wheel archive file set mismatch')
            for member in members:
                if (
                    not member.isfile()
                    or Path(member.name).name != member.name
                    or not member.name.endswith('.whl')
                    or member.size != expected[member.name]['size']
                ):
                    raise ValueError('Invalid wheel archive entry: ' + member.name)
            for member in members:
                target = staging / member.name
                with bundle.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)
                if sha(target) != expected[member.name]['sha256']:
                    raise ValueError(
                        'Dependency wheel checksum mismatch: ' + member.name
                    )
        staging.rename(destination)
    return manifest


def verify(destination, expected):
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError('Existing wheelhouse must be a real directory')
    if {p.name for p in destination.iterdir()} != set(expected):
        raise ValueError('Existing wheelhouse file set mismatch')
    for name, metadata in expected.items():
        target = destination / name
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != metadata['size']
            or sha(target) != metadata['sha256']
        ):
            raise ValueError('Existing wheelhouse checksum mismatch: ' + name)
