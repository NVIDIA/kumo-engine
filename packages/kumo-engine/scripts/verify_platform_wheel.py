#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit an assembled wheel before installing it in a clean runtime.

Only the standard library is used: this runs before Kumo and its filesystem
abstraction are installed. Windows runtime imports are reported separately
from bundled DLLs; runtime execution is still required to qualify a wheel.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import struct
import zipfile
from base64 import urlsafe_b64encode
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

SYSTEM_DLLS = frozenset({
    'advapi32.dll', 'bcrypt.dll', 'bcryptprimitives.dll', 'cabinet.dll',
    'cfgmgr32.dll', 'comctl32.dll', 'comdlg32.dll', 'crypt32.dll',
    'dbghelp.dll', 'dnsapi.dll', 'gdi32.dll', 'iphlpapi.dll', 'kernel32.dll',
    'msvcrt.dll', 'ncrypt.dll', 'netapi32.dll', 'ntdll.dll', 'ole32.dll',
    'oleaut32.dll', 'pdh.dll', 'powrprof.dll', 'propsys.dll', 'psapi.dll',
    'rpcrt4.dll',
    'secur32.dll', 'setupapi.dll', 'shell32.dll', 'shlwapi.dll', 'ucrtbase.dll',
    'user32.dll', 'userenv.dll', 'version.dll', 'winhttp.dll', 'winmm.dll',
    'wintrust.dll', 'ws2_32.dll', 'wtsapi32.dll',
})
RUNTIME_DLLS = frozenset({'python3.dll', 'vcruntime140.dll',
                         'vcruntime140_1.dll', 'msvcp140.dll'})


def verify_elf(data: bytes, platform_tag: str) -> None:
    """Require the declared platform's little-endian 64-bit ELF machine."""
    machine = 183 if platform_tag.endswith('_aarch64') else 62
    if (len(data) < 64 or data[:6] != b'\x7fELF\x02\x01'
            or struct.unpack_from('<H', data, 18)[0] != machine):
        raise ValueError('ELF architecture differs from wheel platform')


def pe_imports(data: bytes) -> set[str]:
    """Read normal and delayed DLL imports from an AMD64 PE32+ image."""
    if data[:2] != b'MZ':
        raise ValueError('Windows native member is not a PE image')
    pe = struct.unpack_from('<I', data, 0x3c)[0]
    if data[pe:pe + 4] != b'PE\0\0':
        raise ValueError('Invalid PE signature')
    machine, sections = struct.unpack_from('<HH', data, pe + 4)
    optional_size = struct.unpack_from('<H', data, pe + 20)[0]
    optional = pe + 24
    if machine != 0x8664 or struct.unpack_from('<H', data, optional)[0] != 0x20b:
        raise ValueError('Windows wheel requires AMD64 PE32+ binaries')
    section_start = optional + optional_size
    ranges = []
    for index in range(sections):
        size, rva, raw_size, offset = struct.unpack_from(
            '<IIII', data, section_start + index * 40 + 8)
        ranges.append((rva, max(size, raw_size), offset, raw_size))

    def locate(rva: int) -> int:
        for start, size, offset, raw_size in ranges:
            if start <= rva < start + size and rva - start < raw_size:
                return offset + rva - start
        raise ValueError(f'Unmapped PE address: {rva}')

    def name_at(rva: int) -> str:
        offset = locate(rva)
        end = data.index(b'\0', offset, offset + 512)
        name = data[offset:end].decode('ascii').lower()
        if '/' in name or '\\' in name or not name.endswith('.dll'):
            raise ValueError('Invalid PE DLL import name')
        return name

    imports: set[str] = set()
    directory_count = struct.unpack_from('<I', data, optional + 108)[0]
    for directory, width, name_index in ((1, 20, 3), (13, 32, 1)):
        if directory_count <= directory:
            continue
        address, size = struct.unpack_from('<II', data,
                                           optional + 112 + directory * 8)
        if not address:
            continue
        offset = locate(address)
        terminated = False
        for position in range(offset, offset + size, width):
            fields = struct.unpack_from('<' + 'I' * (width // 4), data, position)
            if not any(fields):
                terminated = True
                break
            if directory == 13 and fields[0] != 1:
                raise ValueError('Unsupported delayed-import address format')
            imports.add(name_at(fields[name_index]))
        if not terminated:
            raise ValueError('Unterminated PE import directory')
    return imports


def verify_wheel(path: Path, platform_tag: str) -> dict[str, list[str]]:
    expected_tag = f'cp311-abi3-{platform_tag}'
    if not path.name.endswith('-' + expected_tag + '.whl'):
        raise ValueError('Wheel filename does not match the expected ABI/platform')
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel members')
        for name in names:
            member = PurePosixPath(name)
            if member.is_absolute() or '..' in member.parts or '\\' in name:
                raise ValueError('Unsafe wheel member path')
        metadata_paths = [n for n in names if n.endswith('.dist-info/METADATA')]
        if len(metadata_paths) != 1:
            raise ValueError('Expected one merged distribution metadata record')
        dist_info = metadata_paths[0].rsplit('/', 1)[0]
        metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
        if metadata['Name'] != 'kumo-engine':
            raise ValueError('Unexpected distribution name')
        if not metadata['Requires-Python'] or not metadata['License-Expression']:
            raise ValueError('Missing Python requirement or license expression')
        wheel_metadata = BytesParser().parsebytes(
            wheel.read(dist_info + '/WHEEL'))
        if wheel_metadata.get_all('Tag') != [expected_tag]:
            raise ValueError('Wheel metadata tag differs from its filename')
        required = {
            'kumo/engine/__init__.py', 'kumo_core/__init__.py',
            'kumoapi/__init__.py', 'kumopql/__init__.py',
            'diskgraph/__init__.py', 'kumo_bundle_manifest.json',
            dist_info + '/licenses/LICENSE',
            dist_info + '/licenses/THIRD-PARTY.txt',
        }
        windows = platform_tag == 'win_amd64'
        required.add('diskgraph/bin/ingest.exe' if windows
                     else 'diskgraph/bin/ingest')
        missing = required - set(names)
        if missing:
            raise ValueError(f'Missing wheel payload: {sorted(missing)}')
        extension = [n for n in names if n.startswith('diskgraph/_c') and
                     n.endswith('.pyd' if windows else '.so')]
        if len(extension) != 1:
            raise ValueError('Expected one DiskGraph native extension')
        for name in required:
            if not wheel.read(name):
                raise ValueError(f'Empty required wheel member: {name}')
        record = dist_info + '/RECORD'
        recorded: set[str] = set()
        for name, digest, size in csv.reader(io.StringIO(
                wheel.read(record).decode())):
            recorded.add(name)
            if name == record:
                continue
            data = wheel.read(name)
            actual = urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=')
            if digest != 'sha256=' + actual.decode() or size != str(len(data)):
                raise ValueError(f'Wheel RECORD mismatch: {name}')
        if recorded != set(names):
            raise ValueError('Wheel RECORD does not cover all members')
        runtime: set[str] = set()
        system: set[str] = set()
        bundled = {PurePosixPath(n).name.lower() for n in names
                   if n.lower().endswith('.dll')}
        if platform_tag.startswith('manylinux_'):
            for name in extension + ['diskgraph/bin/ingest']:
                verify_elf(wheel.read(name), platform_tag)
        if windows:
            native = extension + ['diskgraph/bin/ingest.exe'] + [
                n for n in names if n.lower().endswith('.dll')]
            imports = {name: sorted(pe_imports(wheel.read(name)))
                       for name in native}
            for name in native:
                for dependency in imports[name]:
                    if (re.fullmatch(r'python\d+(?:_d)?\.dll', dependency)
                            and dependency != 'python3.dll'):
                        raise ValueError('abi3 wheel imports versioned Python DLL')
                    if dependency in bundled:
                        location = str(PurePosixPath(name).parent / dependency)
                        if location not in {n.lower() for n in names}:
                            raise ValueError('Bundled DLL is not beside its importer')
                        continue
                    if dependency in SYSTEM_DLLS or dependency.startswith(
                            ('api-ms-win-', 'ext-ms-win-')):
                        system.add(dependency)
                    elif dependency in RUNTIME_DLLS:
                        runtime.add(dependency)
                    else:
                        raise ValueError(
                            f'{name} imports unbundled non-system DLL {dependency}; '
                            f'native imports: {json.dumps(imports, sort_keys=True)}')
        return {'runtime_dlls': sorted(runtime), 'system_dlls': sorted(system),
                'bundled_dlls': sorted(bundled)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('--platform-tag', required=True)
    args = parser.parse_args()
    print(json.dumps(verify_wheel(args.wheel, args.platform_tag), sort_keys=True))


if __name__ == '__main__':
    main()
