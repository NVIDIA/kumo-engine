# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legal metadata and payload checks for the assembled kumo-engine wheel."""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
MERGE = REPO_ROOT / 'packages/kumo-engine/scripts/build/merge_wheels.py'


def _write_component_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(
            'kumo_engine-1.0.dist-info/METADATA',
            'Metadata-Version: 2.1\nName: kumo-engine\nVersion: 1.0\n'
            'Requires-Python: >=3.11\n',
        )
        archive.writestr('kumo/engine/__init__.py', '')


@pytest.mark.parametrize("in_process", [False, True])
def test_finished_wheel_contains_exact_legal_payloads(
        tmp_path: Path, in_process: bool) -> None:
    """Preserve identical metadata and license payloads through either entry point."""
    wheels = tmp_path / 'wheels'
    output = tmp_path / 'output'
    wheels.mkdir()
    component = wheels / 'kumo_engine-1.0-py3-none-any.whl'
    _write_component_wheel(component)
    components = tmp_path / 'components.tsv'
    components.write_text(f'{component.name}\tkumo-first-party\tfixture\n',
                          encoding='utf-8')
    license_path = tmp_path / 'LICENSE'
    notice_path = tmp_path / 'THIRD-PARTY.txt'
    license_expression = 'Apache-2.0 AND MIT'
    license_path.write_bytes(b'first-party license\n')
    notice_path.write_bytes(b'third-party notices\n')
    if in_process:
        merger = runpy.run_path(str(MERGE))
        options = merger['MergeOptions'](
            wheels_dir=wheels, components_tsv=components, out_dir=output,
            version='1.0', platform_tag='manylinux_2_34_x86_64',
            notice=notice_path, license=license_path,
            license_expression=license_expression,
        )
        wheel = merger['merge_wheels'](options)
    else:
        result = subprocess.run(
            [
                sys.executable,
                str(MERGE),
                str(wheels),
                str(components),
                str(output),
                '--version',
                '1.0',
                '--platform-tag',
                'manylinux_2_34_x86_64',
                '--notice',
                str(notice_path),
                '--license',
                str(license_path),
                '--license-expression',
                license_expression,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        wheel = Path(result.stdout.strip())

    with zipfile.ZipFile(wheel) as archive:
        metadata_member = next(name for name in archive.namelist()
                               if name.endswith('.dist-info/METADATA'))
        dist_info = metadata_member.rsplit('/', 1)[0]
        metadata = Parser().parsestr(
            archive.read(metadata_member).decode('utf-8'))
        assert metadata['Metadata-Version'] == '2.4'
        assert metadata['License-Expression'] == license_expression
        assert metadata.get_all('License-File') == [
            'LICENSE', 'THIRD-PARTY.txt'
        ]
        assert archive.read(
            f'{dist_info}/licenses/LICENSE') == license_path.read_bytes()
        assert archive.read(f'{dist_info}/licenses/THIRD-PARTY.txt'
                            ) == notice_path.read_bytes()
        bundle = json.loads(archive.read('kumo_bundle_manifest.json'))
        assert bundle['components'][0]['classification'] == 'kumo-first-party'


def test_merge_api_rejects_missing_engine_without_exiting(tmp_path: Path) -> None:
    """Report invalid bundle inputs as Python exceptions for in-process callers."""
    merger = runpy.run_path(str(MERGE))
    components = tmp_path / 'components.tsv'
    components.write_text('')
    options = merger['MergeOptions'](
        wheels_dir=tmp_path, components_tsv=components, out_dir=tmp_path,
        version='1.0', platform_tag='linux_x86_64',
        notice=tmp_path / 'notice', license=tmp_path / 'license',
        license_expression='Apache-2.0',
    )

    with pytest.raises(ValueError, match='no kumo-engine component'):
        merger['merge_wheels'](options)
