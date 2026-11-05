# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download one case's checksum-bound raw inputs without generating answers."""

import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def stage(root, manifest, family, *, input_path):
    inventory_path = manifest['source_objects'][family]
    input_path(inventory_path)
    root = Path(root)
    inventory = root / inventory_path
    if (inventory.is_symlink() or not inventory.is_file()
            or hashlib.sha256(inventory.read_bytes()).hexdigest()
            != manifest['files'][inventory_path]):
        raise ValueError('Source inventory checksum differs')
    entries = json.loads(inventory.read_text())['objects']
    required = {
        name
        for graph in manifest['graphs'].values() if graph['profile'] == family
        for source in graph['sources'].values() for name in manifest['files']
        if name == source or name.startswith(source + '/')
    }
    if (not required or not isinstance(entries, list)
            or len(entries) != len(required)
            or {item['path']
                for item in entries} != required):
        raise ValueError('Source inventory does not match the selected graph')
    for item in entries:
        path = input_path(item['path'])
        if (not item['path'].startswith('raw/')
                or item['sha256'] != manifest['files'][item['path']]
                or type(item['bytes']) is not int or item['bytes'] < 1
                or item.get('version_id')
                or any(not isinstance(item.get(key), str) or not item[key]
                       for key in ('bucket', 'region', 'key'))):
            raise ValueError('Raw source identity differs from the manifest')
        input_path(item['key'])
        target = root
        for part in path.parts:
            target = target / part
            if target.is_symlink():
                raise ValueError('Source paths must not contain symbolic links')
    total = sum(item['bytes'] for item in entries)
    if total > shutil.disk_usage(root).free:
        raise ValueError('Insufficient free disk to stage raw inputs')

    import boto3

    clients = {
        region: boto3.client('s3', region_name=region)
        for region in {item['region']
                       for item in entries}
    }

    def transfer(item):
        target = root / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError('Raw source staging requires a fresh path')
        response = clients[item['region']].get_object(Bucket=item['bucket'],
                                                      Key=item['key'])
        temporary = None
        try:
            checksum, size = hashlib.sha256(), 0
            with response['Body'] as source, tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > item['bytes']:
                        raise ValueError(
                            'Raw source length exceeds the manifest')
                    checksum.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if checksum.hexdigest() != item['sha256'] or size != item['bytes']:
                raise ValueError('Raw source length or checksum differs')
            temporary.chmod(0o444)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(transfer, entries))
    return dict(files=len(entries), bytes=total)
