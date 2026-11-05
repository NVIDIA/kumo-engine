# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse pinned test-stage assets, or seed verified local copies."""
import hashlib
import json
from pathlib import Path

ASSETS = json.loads(Path(__file__).with_name('graph-assets.json').read_text())
PDF = {name: ASSETS['pdf'][name] for name in ('name', 'sha256', 'size')}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root, key_file, fixture_archive=None, pdf_wheel=None):
    import launch

    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    fixture = json.loads(
        Path(__file__).parent.parent.joinpath('pr172/inputs.json').read_text())
    if ASSETS['fixture']['sha256'] != fixture['sha256']:
        raise ValueError('Staged fixture differs from An shared input manifest')
    local = {'fixture': fixture_archive, 'pdf': pdf_wheel}
    for name, asset in ASSETS.items():
        launch.validate_remote_asset(asset['name'], asset)
        if local[name] and digest(local[name]) != asset['sha256']:
            raise ValueError('Local test asset checksum mismatch: ' + name)
    with launch.connect(key_file) as connection, connection.cursor() as cursor:
        assets = {}
        for name, asset in ASSETS.items():
            if local[name]:
                if Path(local[name]).name != asset['name']:
                    raise ValueError('Use the pinned test asset filename: ' +
                                     asset['name'])
                assets[name] = launch.ensure_cached_file(
                    cursor, Path(local[name]), asset['sha256'], root)
            else:
                assets[name] = launch.existing_remote_asset(
                    cursor, asset['name'], asset)
    (root / 'assets.json').write_text(json.dumps(assets, indent=2) + '\n')
    return assets
