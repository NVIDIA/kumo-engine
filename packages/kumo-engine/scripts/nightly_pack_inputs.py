#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Split an authenticated input tar stream into seven transport archives."""

import hashlib
import importlib.util
import io
import json
import os
import re
import tarfile
import tempfile
from contextlib import ExitStack
from pathlib import Path

CHUNK = 1024 * 1024


class DigestReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()
        self.size = 0
        self.recent = b''

    def read(self, size):
        value = self.source.read(size)
        self.digest.update(value)
        self.size += len(value)
        self.recent = (self.recent + value)[-CHUNK - tarfile.BLOCKSIZE:]
        return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate input manifest key')
        result[key] = value
    return result


def chunks(source, size):
    while size:
        value = source.read(min(size, CHUNK))
        if not value:
            raise ValueError('Truncated input payload')
        size -= len(value)
        yield value


def pack_workloads(sourcefileobj, outputroot, manifest_bytes,
                   expected_archive_sha256):
    """Use a fresh output directory and preserve the parent manifest identity."""
    root = Path(outputroot)
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    if not re.fullmatch('[0-9a-f]{64}', expected_archive_sha256):
        raise ValueError('Use the authenticated source archive SHA256')
    manifest = json.loads(manifest_bytes, object_pairs_hook=unique_object)
    if (manifest.get('format') != 'kumo-engine-nightly-inputs-v2'
            or manifest.get('scale') not in ('100M', '1B')
            or not isinstance(manifest.get('files'), dict)
            or 'manifest.json' in manifest['files']):
        raise ValueError('Use the complete frozen parent manifest')
    spec = importlib.util.spec_from_file_location(
        'nightly_pack_job',
        Path(__file__).with_name('nightly_job.py'))
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)
    selected = {
        family: job.workload_files(manifest, family)
        for family in job.WORKLOADS
    }
    routes = {
        name: [family for family, names in selected.items() if name in names]
        for name in manifest['files']
    }
    if any(not families for families in routes.values()):
        raise ValueError('Every parent payload must belong to a workload')
    parent_sha = hashlib.sha256(manifest_bytes).hexdigest()
    expected = {'manifest.json': parent_sha, **manifest['files']}
    source = DigestReader(sourcefileobj)
    root.parent.mkdir(parents=True, exist_ok=True)
    receipts = {}
    with tempfile.TemporaryDirectory(prefix='.nightly-pack-',
                                     dir=root.parent) as temporary:
        with ExitStack() as stack:
            outputs = {
                family:
                stack.enter_context(
                    open(Path(temporary) / (family + '.tar'), 'xb'))
                for family in job.WORKLOADS
            }
            digests = {family: hashlib.sha256() for family in job.WORKLOADS}
            sizes = dict.fromkeys(job.WORKLOADS, 0)
            counts = dict.fromkeys(job.WORKLOADS, 0)

            def emit(families, value):
                for family in families:
                    if outputs[family].write(value) != len(value):
                        raise OSError('Incomplete workload archive write')
                    digests[family].update(value)
                    sizes[family] += len(value)

            def member(families, name, size, values):
                header = tarfile.TarInfo(name)
                header.size, header.mode, header.mtime = size, 0o444, 0
                emit(families, header.tobuf(format=tarfile.PAX_FORMAT))
                digest = hashlib.sha256()
                for value in values:
                    digest.update(value)
                    emit(families, value)
                emit(families, b'\0' * (-size % tarfile.BLOCKSIZE))
                for family in families:
                    counts[family] += 1
                return digest.hexdigest()

            member(job.WORKLOADS, 'manifest.json', len(manifest_bytes),
                   chunks(io.BytesIO(manifest_bytes), len(manifest_bytes)))
            seen = set()
            with tarfile.open(fileobj=source, mode='r|',
                              bufsize=CHUNK) as archive:
                for item in archive:
                    name = item.name
                    job.input_path(name)
                    if (item.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                            or item.sparse is not None or item.size < 0):
                        raise ValueError('Input members must be regular files')
                    if name not in expected or name in seen:
                        raise ValueError('Duplicate or undeclared input member')
                    seen.add(name)
                    payload = archive.extractfile(item)
                    if name == 'manifest.json':
                        if item.size != len(manifest_bytes):
                            raise ValueError('Source manifest bytes differ')
                        offset = 0
                        for value in chunks(payload, item.size):
                            if value != manifest_bytes[offset:offset +
                                                       len(value)]:
                                raise ValueError('Source manifest bytes differ')
                            offset += len(value)
                    elif member(routes[name], name, item.size,
                                chunks(payload, item.size)) != expected[name]:
                        raise ValueError('Frozen payload checksum differs')
                position = archive.fileobj.tell() - tarfile.BLOCKSIZE
                offset = position - (source.size - len(source.recent))
                if (offset < 0
                        or source.recent[offset:offset + tarfile.BLOCKSIZE]
                        != b'\0' * tarfile.BLOCKSIZE):
                    raise ValueError('Invalid input tar end header')
                tail_size = 0
                while True:
                    value = archive.fileobj.read(CHUNK)
                    if not value:
                        break
                    tail_size += len(value)
                    if value.strip(b'\0'):
                        raise ValueError('Nonzero data follows the input tar')
                if tail_size < tarfile.BLOCKSIZE or source.size % tarfile.BLOCKSIZE:
                    raise ValueError('Truncated input tar end or padding')
            if seen != set(expected):
                raise ValueError('Missing frozen input members')
            if source.digest.hexdigest() != expected_archive_sha256:
                raise ValueError('Source archive checksum differs')
            for family in job.WORKLOADS:
                padding = 2 * tarfile.BLOCKSIZE
                padding += -(sizes[family] + padding) % tarfile.RECORDSIZE
                emit([family], b'\0' * padding)
                outputs[family].flush()
                os.fsync(outputs[family].fileno())
                receipts[family] = dict(
                    workload=family, scale=manifest['scale'],
                    archive_path=str(root / (family + '.tar')),
                    archive_sha256=digests[family].hexdigest(),
                    archive_bytes=sizes[family], files=counts[family],
                    manifest_sha256=parent_sha,
                    parent_archive_sha256=expected_archive_sha256)
        if root.exists() or root.is_symlink():
            raise FileExistsError(root)
        os.rename(temporary, root)
    return receipts
