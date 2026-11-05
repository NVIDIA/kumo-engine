# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require local instance SSD storage before a large Engine case starts."""

import argparse
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path


def require_local_ssd(path):
    """Inspect the mounted ancestor without creating or modifying scratch."""
    ancestor = Path(path).resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError('Scratch must be below an existing directory')

    def query(command):
        try:
            return json.loads(
                subprocess.run(command, check=True, capture_output=True,
                               text=True).stdout)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            raise ValueError(
                'Cannot verify local NVMe scratch backing') from error

    mounts = query([
        'findmnt', '--json', '--output', 'SOURCE,TARGET', '--target',
        str(ancestor)
    ])['filesystems']
    if len(mounts) != 1:
        raise ValueError('Scratch needs one unambiguous backing mount')
    mount = mounts[0]
    source = mount['source'].split('[', 1)[0]
    devices = query([
        'lsblk', '--json', '--inverse', '--paths', '--output',
        'NAME,TYPE,MODEL', '--', source
    ])['blockdevices']
    physical = set()

    def inspect(device):
        if device['type'] == 'disk':
            if (device.get('model')
                    or '').strip() != ('Amazon EC2 NVMe Instance Storage'):
                raise ValueError('Scratch requires local NVMe instance storage')
            physical.add(device['name'])
        elif not device.get('children'):
            raise ValueError('Scratch backing has an unresolved physical disk')
        for parent in device.get('children', []):
            inspect(parent)

    for device in devices:
        inspect(device)
    if not physical:
        raise ValueError('Scratch backing has no verified physical disk')
    return dict(mount_source=mount['source'], mount_target=mount['target'],
                physical_devices=sorted(physical),
                available_bytes=shutil.disk_usage(ancestor).free)


def require_frozen_inputs(path):
    """Reject mutable input trees before omitting the final byte scan."""
    uid = os.geteuid()
    if uid == 0:
        raise ValueError('Frozen input execution requires an unprivileged user')
    root = Path(path)
    if root != root.resolve() or not root.is_dir():
        raise ValueError('Frozen inputs require a canonical directory')
    if not os.statvfs(root).f_flag & os.ST_RDONLY:
        raise ValueError('Frozen inputs require a read-only mount')
    files = 0
    for item in (root, *root.rglob('*')):
        info = item.lstat()
        if (not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))
                or info.st_uid == uid or info.st_mode & 0o222):
            raise ValueError(
                'Frozen inputs need trusted ownership and no write permissions')
        files += stat.S_ISREG(info.st_mode)
    for parent in root.parents:
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid == uid
                or info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX):
            raise ValueError('Frozen input ancestors must prevent replacement')
    return dict(path=str(root), worker_uid=uid, files=files, read_only=True,
                permissions='trusted-owner-no-writes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    print(json.dumps(require_local_ssd(parser.parse_args().scratch)))
