# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build notebook payloads from this checkout and content-pinned staged assets."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import launch
from resources import admit_candidate
from resources import candidate_path as default_candidate_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, candidate_path):
    source = Path(__file__).resolve().parent
    test = source.parent
    candidate = json.loads(candidate_path.read_text())
    identity = candidate['candidate']
    assets = candidate['assets']
    for name, asset in assets.items():
        launch.validate_remote_asset(name, asset)
    if set(assets) != {'wheelhouse.tar', identity['wheel_file']}:
        raise ValueError('Candidate must pin the wheel and dependency archive')
    if assets[identity['wheel_file']]['sha256'] != identity['wheel_sha256']:
        raise ValueError('Candidate wheel hash differs from asset')
    dependencies = json.loads((source / 'dependencies.json').read_text())
    if dependencies['archive_sha256'] != assets['wheelhouse.tar']['sha256']:
        raise ValueError('Dependency manifest differs from archive')
    cell = (source / 'cell.py').read_text()
    compile(cell, 'snowflake-notebook-cell', 'exec')
    profile = json.loads((test / 'mlops_platforms/snowflake.json').read_text())
    target = dict(
        database=launch.DATABASE, schema=launch.SCHEMA,
        warehouse=launch.WAREHOUSE, role=launch.ROLE,
        artifact_root=launch.STAGE + '/mlops-validation', denied_artifact_root=(
            '@KUMO_TEST_DB.KUMO_TEST.KUMO_TEST_DENIED_STAGE/mlops-validation'))
    profile.update(target)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = dict(candidate=identity, remote_assets=assets,
                    profile_target=target, modes={})
    for mode in ('preflight', 'lifecycle', 'negative'):
        folder = output / mode
        folder.mkdir(mode=0o700)
        for name in ('wheel_mlops_lifecycle.py', 'wheel_mlops_platform.py',
                     'wheel_gpu_smoke.py'):
            shutil.copyfile(test / name, folder / name)
        shutil.copyfile(source / 'offline_support.py',
                        folder / 'offline_support.py')
        shutil.copyfile(source / 'dependencies.json',
                        folder / 'wheelhouse-manifest.json')
        (folder / 'lifecycle-config.json'
         ).write_text(json.dumps(dict(identity, mode=mode)) + '\n')
        (folder /
         'snowflake.json').write_text(json.dumps(profile, indent=2) + '\n')
        notebook = dict(
            nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(
                name='python3', language='python', display_name='Python 3')),
            cells=[
                dict(cell_type='code', id='kumo-installed-lifecycle',
                     metadata={}, execution_count=None, outputs=[],
                     source=cell.splitlines(True))
            ])
        (folder /
         'lifecycle.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
        manifest['modes'][mode] = {
            file.name: digest(file)
            for file in folder.iterdir()
        }
        manifest['modes'][mode].update({
            name: asset['sha256']
            for name, asset in assets.items()
        })
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    candidate = admit_candidate(path=args.candidate or default_candidate_path())
    prepare(args.output_dir, candidate)
    print('Prepared:', args.output_dir)


if __name__ == '__main__':
    main()
