# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legal metadata and payload checks for the assembled fat wheel."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from email.parser import Parser
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
SCRIPT = REPO_ROOT / 'packages/kumo-engine/scripts/build_fat_wheel.sh'


def _merge_source() -> str:
    script = SCRIPT.read_text(encoding='utf-8')
    merge = script.split('cat > "${MERGE_PY}" <<\'PYEOF\'\n', 1)[1]
    return merge.split('\nPYEOF\n', 1)[0]


def _write_component_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(
            'kumo_fixture-1.0.dist-info/METADATA',
            'Metadata-Version: 2.1\nName: kumo-fixture\nVersion: 1.0\n',
        )
        archive.writestr('kumo_fixture/__init__.py', '')


def test_finished_wheel_contains_exact_legal_payloads(tmp_path: Path) -> None:
    wheels = tmp_path / 'wheels'
    output = tmp_path / 'output'
    wheels.mkdir()
    component = wheels / 'kumo_fixture-1.0-py3-none-any.whl'
    _write_component_wheel(component)
    components = tmp_path / 'components.tsv'
    components.write_text(f'{component.name}\tkumo-first-party\tfixture\n',
                          encoding='utf-8')
    license_path = tmp_path / 'LICENSE'
    notice_path = tmp_path / 'THIRD-PARTY.txt'
    kumo_api_license_path = tmp_path / 'KUMO-API-LICENSE'
    license_expression = 'Apache-2.0 AND MIT'
    license_path.write_bytes(b'first-party license\n')
    notice_path.write_bytes(b'third-party notices\n')
    kumo_api_license_path.write_bytes(b'kumo-api MIT license\n')
    merger = tmp_path / 'merge.py'
    merger.write_text(_merge_source(), encoding='utf-8')

    result = subprocess.run(
        [
            sys.executable,
            str(merger),
            str(wheels),
            str(components),
            str(output),
            'kumo-fat-bundle',
            '1.0',
            'py3-none-any',
            'cp310',
            '3.10',
            '0',
            str(notice_path),
            str(license_path),
            str(kumo_api_license_path),
            license_expression,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
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
            'LICENSE', 'THIRD-PARTY.txt', 'KUMO-API-LICENSE'
        ]
        assert archive.read(
            f'{dist_info}/licenses/LICENSE') == license_path.read_bytes()
        assert archive.read(f'{dist_info}/licenses/THIRD-PARTY.txt'
                            ) == notice_path.read_bytes()
        assert archive.read(f'{dist_info}/licenses/KUMO-API-LICENSE'
                            ) == kumo_api_license_path.read_bytes()
        bundle = json.loads(archive.read('kumo_bundle_manifest.json'))
        assert bundle['components'][0]['classification'] == 'kumo-first-party'
