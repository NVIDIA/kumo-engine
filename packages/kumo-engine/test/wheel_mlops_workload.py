# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standard-library admission shared by managed customer-model notebooks."""
import ast
import hashlib
import re
import zipfile
from pathlib import PurePosixPath

CUDA_TORCH = 'torch==2.9.1+cu128'
CUDA_INDEX = 'https://download.pytorch.org/whl/cu128'


def pytorch_install_args(accelerator):
    if accelerator not in ('cpu', 'gpu'):
        raise ValueError('Select a CPU or GPU workload environment')
    return [CUDA_TORCH, '--extra-index-url', CUDA_INDEX] if accelerator == 'gpu' else []


def digest(content):
    return hashlib.sha256(content).hexdigest()


def installed_wheel_files(wheel):
    """Map ZIP members to their site-packages destinations before comparing."""
    files = {}
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            parts = PurePosixPath(entry.filename).parts
            if (not parts or parts[0] == '/' or '..' in parts
                    or '\\' in entry.filename):
                raise ValueError('Unsafe workload wheel path: ' +
                                 entry.filename)
            if parts[0].endswith('.data'):
                if len(parts) < 3 or parts[1] not in ('purelib', 'platlib'):
                    raise ValueError(
                        'Workload wheels must install into site-packages: ' +
                        entry.filename)
                parts = parts[2:]
            target = '/'.join(parts)
            checksum = digest(archive.read(entry))
            if target in files and files[target] != checksum:
                raise ValueError('Conflicting installed wheel path: ' + target)
            files[target] = checksum
    return files


def check_workload_wheels(engine_wheel, extra_wheels):
    """Reject dependency wheels that replace admitted Engine/dependency bytes."""
    installed = installed_wheel_files(engine_wheel)
    for wheel in extra_wheels:
        if not re.fullmatch(r'[A-Za-z0-9_.+\-]+\.whl', wheel.name):
            raise ValueError('Expected a regular workload wheel filename')
        files = installed_wheel_files(wheel)
        for target in installed.keys() & files.keys():
            if installed[target] != files[target]:
                raise ValueError(
                    'Workload wheel replaces candidate bytes or another dependency: '
                    + target)
        installed.update(files)


def workload_phases(source):
    """Inspect the literal phase declaration without importing customer code."""
    declarations = []
    for statement in ast.parse(source).body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
        else:
            continue
        if any(
                isinstance(target, ast.Name) and target.id == 'PHASES'
                for target in targets):
            declarations.append(statement.value)
    try:
        phases = ast.literal_eval(
            declarations[0]) if len(declarations) == 1 else None
    except (ValueError, TypeError, SyntaxError):
        phases = None
    if (not isinstance(phases, (tuple, list)) or not phases
            or any(not isinstance(name, str)
                   or not re.fullmatch('[a-z][a-z0-9_-]*', name)
                   for name in phases) or len(set(phases)) != len(phases)
            or set(phases) & {'summary', 'platform', 'platform-prerequisite'}):
        raise ValueError(
            'Notebook workloads require one top-level literal PHASES list/tuple '
            'of unique, unreserved lowercase phase names')
    return list(phases)
