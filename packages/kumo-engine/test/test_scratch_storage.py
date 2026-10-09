# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import warnings
from pathlib import Path

import pytest

import kumo.engine._storage as storage_module
from kumo.engine import FileConnector, Graph, Table
from kumo.engine._storage import network_storage


def _mountinfo(monkeypatch, tmp_path: Path, *mounts: str) -> None:
    path = tmp_path / 'mountinfo'
    monkeypatch.setattr(storage_module, '_MOUNTINFO', path)
    path.write_bytes(os.fsencode(''.join(
        f'{i} 1 {device} / {point} rw - {fstype} src rw\n'
        for i, (point, fstype, device) in enumerate(
            mount.split() for mount in mounts))))


def _disk(sys_block: Path, number: str, name: str, model: str) -> Path:
    node = sys_block.parent / 'devices' / name
    (node / 'device').mkdir(parents=True)
    (node / 'device' / 'model').write_text(f'{model}\n')
    (node / 'dev').write_text(f'{number}\n')
    (sys_block / number).symlink_to(node)
    return node


@pytest.fixture
def sys_block(tmp_path, monkeypatch):
    path = tmp_path / 'sys' / 'block'
    monkeypatch.setattr(storage_module, '_SYS_BLOCK', path)
    path.mkdir(parents=True)
    ebs = _disk(path, '259:0', 'nvme0n1', 'Amazon Elastic Block Store')
    (ebs / 'nvme0n1p1').mkdir()
    (ebs / 'nvme0n1p1' / 'partition').write_text('1\n')
    (path / '259:1').symlink_to(ebs / 'nvme0n1p1')
    _disk(path, '259:2', 'nvme1n1', 'Amazon EC2 NVMe Instance Storage')
    _disk(path, '259:3', 'nvme2n1', 'Amazon EC2 NVMe Instance Storage')
    raid = path.parent / 'devices' / 'md0'
    (raid / 'slaves').mkdir(parents=True)
    for name in ('nvme1n1', 'nvme2n1'):
        (raid / 'slaves' / name).symlink_to(path.parent / 'devices' / name)
    (path / '9:0').symlink_to(raid)
    # Databricks-style LVM over NVMe + EBS.
    _disk(path, '259:4', 'nvme3n1', 'Amazon Elastic Block Store')
    lvm = path.parent / 'devices' / 'dm-0'
    (lvm / 'slaves').mkdir(parents=True)
    for name in ('nvme1n1', 'nvme3n1'):
        (lvm / 'slaves' / name).symlink_to(path.parent / 'devices' / name)
    (path / '253:0').symlink_to(lvm)
    return path


def test_network_storage(monkeypatch, tmp_path, sys_block):
    _mountinfo(
        monkeypatch, tmp_path,
        '/ ext4 259:1',  # partition of an EBS volume
        '/local_disk0 xfs 9:0',  # RAID over instance-store NVMe
        '/local_disk0/dbfs fuse 0:50',
        '/mnt/nfs nfs4 0:51',
        '/mnt/lvm xfs 253:0',
        r'/mnt/données\040x fuse 0:52',
        os.fsdecode(b'/mnt/\xff nfs 0:53'),  # not UTF-8
    )

    def storage(path):
        return network_storage(Path(path))

    assert storage('/root/.cache/kumo-engine') == 'Amazon EBS'
    assert storage('/local_disk0/kumo-engine') is None
    assert storage('/local_disk0/dbfs/x') == (
        'a network filesystem (fuse at /local_disk0/dbfs)')
    assert storage('/mnt/nfs') == 'a network filesystem (nfs4 at /mnt/nfs)'
    assert storage('/mnt/lvm/kumo-engine') == (
        'a volume of 2 disks that includes Amazon EBS')
    assert storage('/mnt/données x/kumo') == (
        'a network filesystem (fuse at /mnt/données x)')
    assert storage(os.fsdecode(b'/mnt/\xff/kumo')) == os.fsdecode(
        b'a network filesystem (nfs at /mnt/\xff)')
    monkeypatch.setattr(storage_module, '_MOUNTINFO', tmp_path / 'missing')
    assert storage('/x') is None


def test_snapshot_warns_once_on_ebs_scratch(monkeypatch, tmp_path, sys_block):
    scratch = tmp_path / 'scratch'
    monkeypatch.setattr('kumo.engine._scratch._root', scratch)
    _mountinfo(monkeypatch, tmp_path, '/ ext4 259:2',
               f'{os.path.realpath(tmp_path)} ext4 259:1')
    sources = tmp_path / 'sources'
    sources.mkdir()
    (sources / 'users.csv').write_text('id,name\n1,Alice\n2,Bob\n')
    graph = Graph(tables=[Table.from_source_table(
        FileConnector(root_dir=str(sources))['users'], primary_key='id')])

    with pytest.warns(UserWarning, match=f'{scratch} is on Amazon EBS'):
        graph.snapshot()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        graph.snapshot()
