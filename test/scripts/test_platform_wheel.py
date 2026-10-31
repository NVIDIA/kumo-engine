# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import hashlib
import io
import runpy
import struct
from base64 import urlsafe_b64encode
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] /
          'packages/kumo-engine/scripts/verify_platform_wheel.py')
VERIFY = runpy.run_path(str(SCRIPT))['verify_wheel']


def elf(machine=183):
    data = bytearray(64)
    data[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<H', data, 18, machine)
    return bytes(data)


def pe(dependency='python3.dll'):
    data = bytearray(1024)
    data[:2] = b'MZ'
    struct.pack_into('<I', data, 0x3c, 64)
    data[64:68] = b'PE\0\0'
    struct.pack_into('<HH', data, 68, 0x8664, 1)
    struct.pack_into('<H', data, 84, 240)
    struct.pack_into('<H', data, 88, 0x20b)
    struct.pack_into('<I', data, 196, 16)
    struct.pack_into('<II', data, 208, 4096, 40)
    struct.pack_into('<IIII', data, 336, 512, 4096, 512, 512)
    struct.pack_into('<I', data, 524, 4160)
    encoded = dependency.encode() + b'\0'
    data[576:576 + len(encoded)] = encoded
    return bytes(data)


def wheel(tmp_path, platform='manylinux_2_34_aarch64', updates=None,
          missing=None, corrupt=False):
    tag = 'cp311-abi3-' + platform
    info = 'kumo_engine-1.0.dist-info'
    windows = platform == 'win_amd64'
    files = {name + '/__init__.py': b'# package' for name in
             ('kumo/engine', 'kumo_core', 'kumoapi', 'kumopql', 'diskgraph')}
    files.update({
        info + '/METADATA': b'Name: kumo-engine\nRequires-Python: >=3.11\n'
        b'License-Expression: Apache-2.0\n',
        info + '/WHEEL': ('Tag: ' + tag + '\n').encode(),
        info + '/licenses/LICENSE': b'license',
        info + '/licenses/THIRD-PARTY.txt': b'notices',
        'kumo_bundle_manifest.json': b'{}',
        'diskgraph/_c.pyd' if windows else 'diskgraph/_c.abi3.so':
        pe() if windows else elf(),
        'diskgraph/bin/ingest.exe' if windows else 'diskgraph/bin/ingest':
        pe('kernel32.dll') if windows else elf(),
    })
    files.update(updates or {})
    if missing:
        del files[missing]
    record = io.StringIO()
    writer = csv.writer(record)
    for name, data in files.items():
        digest = urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=')
        writer.writerow((name, 'sha256=' + digest.decode(), len(data)))
    writer.writerow((info + '/RECORD', '', ''))
    files[info + '/RECORD'] = record.getvalue().encode()
    if corrupt:
        files['kumo/engine/__init__.py'] = b'changed'
    path = tmp_path / ('kumo_engine-1.0-' + tag + '.whl')
    with ZipFile(path, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


@pytest.mark.parametrize('platform', [
    'manylinux_2_34_aarch64', 'win_amd64', 'macosx_14_0_arm64'])
def test_complete_wheel(tmp_path, platform):
    result = VERIFY(wheel(tmp_path, platform), platform)
    expected = ['python3.dll'] if platform == 'win_amd64' else []
    assert result['runtime_dlls'] == expected


@pytest.mark.parametrize('member', [
    'diskgraph/_c.abi3.so', 'diskgraph/bin/ingest'])
def test_wrong_elf_architecture(tmp_path, member):
    with pytest.raises(ValueError, match='ELF architecture'):
        VERIFY(wheel(tmp_path, updates={member: elf(62)}),
               'manylinux_2_34_aarch64')


@pytest.mark.parametrize('dependency,bundled,error', [
    ('python311.dll', False, 'versioned Python'),
    ('python311.dll', True, 'versioned Python'),
    ('private.dll', False, 'unbundled non-system'),
])
def test_invalid_native_dependency(tmp_path, dependency, bundled, error):
    updates = {'diskgraph/_c.pyd': pe(dependency)}
    if bundled:
        updates['diskgraph/' + dependency] = pe('kernel32.dll')
    with pytest.raises(ValueError, match=error):
        VERIFY(wheel(tmp_path, 'win_amd64', updates), 'win_amd64')


def test_misplaced_bundled_dependency(tmp_path):
    updates = {'diskgraph/_c.pyd': pe('private.dll'),
               'elsewhere/private.dll': pe('kernel32.dll')}
    with pytest.raises(ValueError, match='not beside'):
        VERIFY(wheel(tmp_path, 'win_amd64', updates), 'win_amd64')


def test_windows_performance_library(tmp_path):
    updates = {'diskgraph/bin/ingest.exe': pe('pdh.dll')}
    result = VERIFY(wheel(tmp_path, 'win_amd64', updates), 'win_amd64')
    assert result['system_dlls'] == ['pdh.dll']
    assert result['runtime_dlls'] == ['python3.dll']


def test_unknown_dependency_diagnostics(tmp_path):
    updates = {'diskgraph/_c.pyd': pe('private.dll'),
               'diskgraph/bin/ingest.exe': pe('pdh.dll')}
    with pytest.raises(ValueError, match='unbundled non-system DLL private.dll') as exc:
        VERIFY(wheel(tmp_path, 'win_amd64', updates), 'win_amd64')
    message = str(exc.value)
    assert '"diskgraph/_c.pyd": ["private.dll"]' in message
    assert '"diskgraph/bin/ingest.exe": ["pdh.dll"]' in message
    assert str(tmp_path) not in message


@pytest.mark.parametrize('missing', [
    'diskgraph/bin/ingest',
    'kumo_engine-1.0.dist-info/licenses/THIRD-PARTY.txt'])
def test_missing_payload(tmp_path, missing):
    with pytest.raises(ValueError, match='Missing wheel payload'):
        VERIFY(wheel(tmp_path, missing=missing), 'manylinux_2_34_aarch64')


def test_modified_record(tmp_path):
    with pytest.raises(ValueError, match='RECORD mismatch'):
        VERIFY(wheel(tmp_path, corrupt=True), 'manylinux_2_34_aarch64')


def test_mismatched_tag(tmp_path):
    with pytest.raises(ValueError, match='metadata tag'):
        VERIFY(wheel(tmp_path, updates={
            'kumo_engine-1.0.dist-info/WHEEL': b'Tag: cp312-cp312-win_amd64\n'
        }), 'manylinux_2_34_aarch64')
