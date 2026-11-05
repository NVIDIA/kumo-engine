# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Require local instance SSD storage before a large Engine case starts."""

import argparse
import json
import shutil
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    print(json.dumps(require_local_ssd(parser.parse_args().scratch)))
