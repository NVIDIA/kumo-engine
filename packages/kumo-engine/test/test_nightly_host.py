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


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    import os
    import stat

    module = host()
    root = tmp_path / 'inputs'
    (root / 'rows').mkdir(parents=True)
    (root / 'rows/part.parquet').write_bytes(b'Pinned source bytes')
    overrides = {}

    class FrozenPath(type(tmp_path)):
        def lstat(self):
            original = super().lstat()
            mode = (stat.S_IFDIR |
                    (0o555 if self.is_relative_to(root) else 0o755)
                    ) if stat.S_ISDIR(original.st_mode) else original.st_mode
            if stat.S_ISREG(original.st_mode):
                mode = stat.S_IFREG | 0o444
            values = list(original)
            values[0], values[4] = overrides.get(str(self), (mode, 0))
            return os.stat_result(values)

        def open(self, *args, **kwargs):
            raise AssertionError('Frozen input check read file contents')

    monkeypatch.setattr(module, 'Path', FrozenPath)
    monkeypatch.setattr(
        module, 'os',
        SimpleNamespace(geteuid=lambda: 1001, ST_RDONLY=1,
                        statvfs=lambda path: SimpleNamespace(f_flag=1)),
        raising=False)
    return module, root, overrides


def test_frozen_inputs_are_verified_without_opening_payloads(frozen_inputs):
    module, root, _ = frozen_inputs
    assert hasattr(module,
                   'require_frozen_inputs'), 'Frozen input admission is missing'
    result = module.require_frozen_inputs(root)
    assert result['files'] == 1
    assert result['worker_uid'] == 1001


@pytest.mark.parametrize('case', [
    'root-user', 'writable-mount', 'file-owner', 'directory-owner',
    'file-write', 'directory-write', 'ancestor-owner', 'ancestor-write',
    'symlink', 'special-file'
])
def test_mutable_input_handoffs_are_rejected(frozen_inputs, case):
    import stat

    module, root, overrides = frozen_inputs
    assert hasattr(module,
                   'require_frozen_inputs'), 'Frozen input admission is missing'
    if case == 'root-user':
        module.os.geteuid = lambda: 0
    elif case == 'writable-mount':
        module.os.statvfs = lambda path: SimpleNamespace(f_flag=0)
    elif case == 'file-owner':
        overrides[str(root / 'rows/part.parquet')] = (stat.S_IFREG | 0o444,
                                                      1001)
    elif case == 'directory-owner':
        overrides[str(root / 'rows')] = (stat.S_IFDIR | 0o555, 1001)
    elif case == 'file-write':
        overrides[str(root / 'rows/part.parquet')] = (stat.S_IFREG | 0o644, 0)
    elif case == 'directory-write':
        overrides[str(root / 'rows')] = (stat.S_IFDIR | 0o755, 0)
    elif case == 'ancestor-owner':
        overrides[str(root.parent)] = (stat.S_IFDIR | 0o755, 1001)
    elif case == 'ancestor-write':
        overrides[str(root.parent)] = (stat.S_IFDIR | 0o777, 0)
    elif case == 'symlink':
        (root / 'linked').symlink_to(root / 'rows/part.parquet')
    else:
        overrides[str(root / 'rows/part.parquet')] = (stat.S_IFIFO | 0o444, 0)
    with pytest.raises(ValueError):
        module.require_frozen_inputs(root)


def test_trusted_child_under_root_owned_sticky_parent_is_allowed(frozen_inputs):
    import stat

    module, root, overrides = frozen_inputs
    assert hasattr(module,
                   'require_frozen_inputs'), 'Frozen input admission is missing'
    overrides[str(root.parent)] = (stat.S_IFDIR | stat.S_ISVTX | 0o777, 0)
    assert module.require_frozen_inputs(root)['files'] == 1
