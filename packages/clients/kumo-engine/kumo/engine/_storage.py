# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detects network storage behind the scratch directory (Linux only)."""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

_NETWORK_FILESYSTEMS = frozenset({
    '9p', 'afs', 'ceph', 'cifs', 'glusterfs', 'lustre', 'nfs', 'nfs4',
    'smb3', 'smbfs'
})

_NETWORK_BLOCK_MODELS = (
    ('Amazon Elastic Block Store', 'Amazon EBS'),
    ('PersistentDisk', 'a Google Persistent Disk'),
)

_MOUNTINFO = Path('/proc/self/mountinfo')
_SYS_BLOCK = Path('/sys/dev/block')


def _unescape(field: bytes) -> str:
    # mountinfo octal-escapes whitespace; other bytes are raw.
    return os.fsdecode(
        re.sub(rb'\\([0-7]{3})', lambda m: bytes([int(m[1], 8)]), field))


def _mount_of(path: Path, mountinfo: Path) -> Optional[Tuple[str, str, str]]:
    try:
        lines = mountinfo.read_bytes().splitlines()
    except OSError:
        return None
    best: Optional[Tuple[str, str, str]] = None
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index(b'-')
            device, point = fields[2].decode(), _unescape(fields[4])
            fstype = fields[separator + 1].decode()
        except (IndexError, ValueError):
            continue
        if (path == Path(point) or Path(point) in path.parents) and (
                best is None or len(point) >= len(best[0])):
            best = (point, fstype, device)
    return best


def _block_models(device: str, sys_block: Path) -> List[str]:
    try:
        node = (sys_block / device).resolve(strict=True)
    except OSError:
        return []
    if (node / 'partition').exists():
        node = node.parent
    slaves = node / 'slaves'
    if slaves.is_dir() and any(slaves.iterdir()):
        models: List[str] = []
        for slave in sorted(slaves.iterdir()):
            try:
                number = (slave / 'dev').read_text().strip()
            except OSError:
                continue
            models.extend(_block_models(number, sys_block))
        return models
    try:
        return [(node / 'device' / 'model').read_text().strip()]
    except OSError:
        return []


def network_storage(path: Path) -> Optional[str]:
    mount = _mount_of(Path(os.path.realpath(path)), _MOUNTINFO)
    if mount is None:
        return None
    point, fstype, device = mount
    if fstype in _NETWORK_FILESYSTEMS or fstype.startswith('fuse'):
        return f'a network filesystem ({fstype} at {point})'
    # One network disk behind an LVM/RAID volume is enough.
    models = _block_models(device, _SYS_BLOCK)
    for marker, name in _NETWORK_BLOCK_MODELS:
        if any(marker in model for model in models):
            return name if len(models) == 1 else (
                f'a volume of {len(models)} disks that includes {name}')
    return None
