# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject scratch on EBS even when its device name starts with NVMe."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parents[1] / 'scripts/nightly_host.py'


def host():
    assert SOURCE.is_file(), 'The local SSD admission check is missing'
    spec = importlib.util.spec_from_file_location('nightly_host_tested', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def storage(monkeypatch, module, mount, source, tree):
    def run(command, **kwargs):
        if command[0] == 'findmnt':
            assert command[-1] == str(mount)
            result = {'filesystems': [dict(source=source, target=str(mount))]}
        else:
            assert command[0] == 'lsblk' and '--inverse' in command
            assert command[-1] == source.split('[', 1)[0]
            result = {'blockdevices': tree}
        return SimpleNamespace(stdout=json.dumps(result))

    monkeypatch.setattr(module.subprocess, 'run', run)


def test_striped_scratch_uses_both_local_ssds(monkeypatch, tmp_path):
    module = host()
    storage(monkeypatch, module, tmp_path, '/dev/mapper/vg-lv[/scratch]', [{
        'name':
        '/dev/mapper/vg-lv',
        'type':
        'lvm',
        'model':
        None,
        'children': [{
            'name':
            '/dev/md0',
            'type':
            'raid0',
            'model':
            None,
            'children': [{
                'name': '/dev/nvme1n1',
                'type': 'disk',
                'model': 'Amazon EC2 NVMe Instance Storage'
            }, {
                'name': '/dev/nvme2n1',
                'type': 'disk',
                'model': 'Amazon EC2 NVMe Instance Storage'
            }]
        }]
    }])
    result = module.require_local_ssd(tmp_path / 'new-case' / 'graph')
    assert result['mount_target'] == str(tmp_path)
    assert result['physical_devices'] == ['/dev/nvme1n1', '/dev/nvme2n1']
    assert result['available_bytes'] > 0
    assert not (tmp_path / 'new-case').exists()


@pytest.mark.parametrize('models', [
    ['Amazon Elastic Block Store'],
    ['Amazon EC2 NVMe Instance Storage', 'Amazon Elastic Block Store'],
    ['Unknown SSD'],
    [None],
])
def test_ebs_or_unknown_backing_disk_is_rejected(monkeypatch, tmp_path, models):
    module = host()
    disks = [
        dict(name=f'/dev/nvme{index}n1', type='disk', model=model)
        for index, model in enumerate(models)
    ]
    storage(monkeypatch, module, tmp_path, '/dev/mapper/vg-lv', [
        dict(name='/dev/mapper/vg-lv', type='lvm', model=None, children=disks)
    ])
    with pytest.raises(ValueError, match='local NVMe instance storage'):
        module.require_local_ssd(tmp_path / 'case')


@pytest.mark.parametrize('has_ssd', [False, True])
def test_unresolved_backing_branch_is_rejected(monkeypatch, tmp_path, has_ssd):
    module = host()
    devices = [dict(name='/dev/loop0', type='loop', model=None)]
    if has_ssd:
        devices.append(
            dict(name='/dev/nvme1n1', type='disk',
                 model='Amazon EC2 NVMe Instance Storage'))
    storage(monkeypatch, module, tmp_path, '/dev/loop0', devices)
    with pytest.raises(ValueError, match='physical disk'):
        module.require_local_ssd(tmp_path)
