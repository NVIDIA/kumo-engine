#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record installed-wheel evidence before executing the CPU smoke.

This standalone audit uses stdlib filesystem operations so it can reject
checkout imports before loading Kumo's own filesystem abstraction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

MODULES = ('kumo.engine', 'kumo_core', 'kumoapi', 'kumopql', 'diskgraph._c')


def check_origin(origin: str | None, prefix: Path) -> str:
    if origin is None:
        raise ValueError('Required module has no installed file origin')
    path = Path(origin).resolve()
    if not path.is_relative_to(prefix.resolve()):
        raise ValueError(f'Module originates outside the virtual environment: {path}')
    return str(path)


def check_platform(expected: str, system: str, machine: str, release: str) -> None:
    arm = machine.lower() in {'aarch64', 'arm64'}
    amd = machine.lower() in {'amd64', 'x86_64'}
    wsl = 'microsoft' in release.lower()
    valid = {
        'windows-native': system == 'Windows' and amd,
        'windows-wsl': system == 'Linux' and amd and wsl,
        'linux-arm64': system == 'Linux' and arm and not wsl,
    }
    if not valid[expected]:
        raise ValueError(f'Unexpected runtime: {system}/{machine}/{release}')


def check_install(source: str | None, wheel: Path) -> None:
    if source is None:
        raise ValueError('Installed distribution lacks direct-wheel provenance')
    recorded = urlsplit(json.loads(source)['url'])
    if (recorded.scheme != 'file' or recorded.netloc
            or recorded.query or recorded.fragment):
        raise ValueError('Installed distribution is not from a local wheel')
    installed_path = Path(url2pathname(recorded.path)).resolve()
    if os.path.normcase(str(installed_path)) != os.path.normcase(str(wheel.resolve())):
        raise ValueError('Installed distribution differs from the downloaded wheel')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', required=True, type=Path)
    parser.add_argument('--platform', required=True,
                        choices=('windows-native', 'windows-wsl', 'linux-arm64'))
    parser.add_argument('--python-version', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix:
        raise ValueError('Qualification requires a fresh virtual environment')
    if os.environ.get('PYTHONPATH') or os.environ.get('DISKGRAPH_INGEST_BIN'):
        raise ValueError('PYTHONPATH and DISKGRAPH_INGEST_BIN must be unset')
    actual_python = f'{sys.version_info.major}.{sys.version_info.minor}'
    if actual_python != args.python_version:
        raise ValueError('Python version differs from the qualification matrix')
    check_platform(args.platform, platform.system(), platform.machine(),
                   platform.release())
    origins = {name: check_origin(importlib.import_module(name).__file__,
                                  Path(sys.prefix)) for name in MODULES}
    diskgraph = importlib.import_module('diskgraph')
    package = Path(check_origin(diskgraph.__file__, Path(sys.prefix))).parent
    ingest = package / 'bin' / ('ingest.exe' if sys.platform == 'win32' else 'ingest')
    if not ingest.is_file() or not os.access(ingest, os.X_OK):
        raise ValueError('Installed wheel ingestion executable is missing')
    faiss_present = importlib.util.find_spec('faiss') is not None
    if args.platform != 'windows-wsl' and faiss_present:
        raise ValueError('Bare Windows/ARM wheel qualification must omit Faiss')
    import torch
    with args.wheel.open('rb') as wheel_file:
        wheel_sha256 = hashlib.file_digest(wheel_file, 'sha256').hexdigest()
    check_install(importlib.metadata.distribution('kumo-engine').read_text(
        'direct_url.json'), args.wheel)
    report = {
        'commit': args.commit, 'wheel': args.wheel.name,
        'wheel_sha256': wheel_sha256,
        'platform': args.platform, 'system': platform.platform(),
        'python': sys.version, 'executable': sys.executable,
        'origins': origins, 'ingest': str(ingest), 'faiss_present': faiss_present,
        'torch': torch.__version__, 'torch_cuda_build': torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
