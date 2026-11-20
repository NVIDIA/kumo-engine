# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build notebook payloads from this checkout and content-pinned staged assets."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import launch
from resources import admit_candidate
from resources import candidate_path as default_candidate_path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, candidate_path, workload=None, workload_helpers=None):
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
    dependencies = candidate['dependencies']
    if dependencies['archive_sha256'] != assets['wheelhouse.tar']['sha256']:
        raise ValueError('Dependency manifest differs from archive')
    custom = None
    custom_sources = []
    custom_contents = {}
    extra_wheels = candidate.get('workload_wheels', [])
    if extra_wheels != dependencies.get('workload_wheels', []):
        raise ValueError(
            'Candidate workload wheels differ from dependency inventory')
    if workload is not None:
        custom_sources = [workload, *(workload_helpers or [])]
        if any(path.is_symlink() or not path.is_file()
               for path in custom_sources):
            raise ValueError('Workload sources must be regular files')
        phases = launch.workload_tools().workload_phases(workload.read_bytes())
        custom = dict(file=workload.name,
                      helpers=[path.name
                               for path in custom_sources[1:]], phases=phases,
                      sha256=digest(workload), wheels=extra_wheels)
        names = launch.workload_files(custom)
        reserved = {
            'lifecycle.ipynb', 'lifecycle-config.json', 'snowflake.json',
            'wheel_mlops_lifecycle.py', 'wheel_mlops_platform.py',
            'wheel_gpu_smoke.py', 'wheel_mlops_workload.py',
            'offline_support.py'
        }
        if names & reserved:
            raise ValueError('Workload source collides with a notebook helper')
        repo = Path(__file__).resolve().parents[4]
        for path in custom_sources:
            relative = path.resolve().relative_to(repo).as_posix()
            committed = subprocess.check_output(
                ['git', 'show', 'HEAD:' + relative], cwd=repo)
            if path.read_bytes() != committed:
                raise ValueError(
                    'Commit the exact workload source before preparing: ' +
                    relative)
            custom_contents[path.name] = committed
        custom['sha256'] = hashlib.sha256(
            custom_contents[workload.name]).hexdigest()
        custom['phases'] = launch.workload_tools().workload_phases(
            custom_contents[workload.name])
    elif workload_helpers or extra_wheels:
        raise ValueError('Workload helpers/dependencies require --workload')
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
    if custom is not None:
        manifest['workload'] = custom
    for mode in ('preflight', 'lifecycle', 'negative'):
        folder = output / mode
        folder.mkdir(mode=0o700)
        for name in ('wheel_mlops_lifecycle.py', 'wheel_mlops_platform.py',
                     'wheel_gpu_smoke.py', 'wheel_mlops_workload.py'):
            shutil.copyfile(test / name, folder / name)
        for name, content in custom_contents.items():
            (folder / name).write_bytes(content)
        shutil.copyfile(source / 'offline_support.py',
                        folder / 'offline_support.py')
        (folder / 'wheelhouse-manifest.json'
         ).write_text(json.dumps(dependencies, indent=2) + '\n')
        config = dict(identity, mode=mode)
        if custom is not None:
            config['workload'] = custom
        (folder / 'lifecycle-config.json').write_text(json.dumps(config) + '\n')
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
    parser.add_argument('--workload', type=Path,
                        help='Committed native customer workload module')
    parser.add_argument(
        '--workload-helper', type=Path, action='append',
        help='Committed sibling Python helper; repeat as needed')
    args = parser.parse_args()
    candidate = admit_candidate(path=args.candidate or default_candidate_path())
    prepare(args.output_dir, candidate, args.workload, args.workload_helper)
    print('Prepared:', args.output_dir)


if __name__ == '__main__':
    main()
