# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'packages/kumo-engine/scripts/build_fat_wheel.sh'


@pytest.mark.parametrize(('args', 'valid', 'providers'), [
    ([], True, None),
    (['--extra-components', 'kumo-databricks'], True, ['kumo-databricks']),
    (['--extra-components', ' kumo-databricks, kumo-s3 '
      ], True, ['kumo-databricks', 'kumo-s3']),
    (['--extra-components', 'kumo-missing'], False, []),
    (['--extra-components', 'kumo-sdk'], False, []),
])
def test_fat_wheel_builds_first_party_packages_from_monorepo(
        tmp_path, args, valid, providers):
    repo = tmp_path / 'repo'
    for name in [
            'kumo-api', 'kumo-core', 'kumo-pql', 'kumo-engine', 'kumo-sdk',
            'kumo-databricks', 'kumo-snowflake', 'kumo-s3', 'kumo-gcs',
            'kumo-bigquery', 'diskgraph'
    ]:
        package = repo / 'packages' / name
        package.mkdir(parents=True)
        (package / 'pyproject.toml').touch()
    diskgraph = repo / 'packages/diskgraph'
    (diskgraph / 'diskgraph-pybind').mkdir()
    (diskgraph / 'diskgraph-pybind/Cargo.toml').touch()
    (repo / 'LICENSE').touch()
    (repo / 'packages/kumo-api/LICENSE').touch()
    (repo / 'THIRD-PARTY.txt').touch()
    glove = repo / 'tools/glove-6b-50d/build_wheel.sh'
    glove.parent.mkdir(parents=True)
    glove.write_text('#!/bin/bash\nmkdir -p "$1"\n'
                     'touch "$1/kumo_glove_6b_50d-1-py3-none-any.whl"\n')
    glove.chmod(0o755)
    calls = tmp_path / 'calls.jsonl'
    cargo = tmp_path / 'cargo'
    cargo.touch(mode=0o755)
    python = tmp_path / 'python'
    python.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
from zipfile import ZipFile
args = sys.argv[1:]
if args == ['-m', 'pip', '--version']:
    sys.exit(0)
with open(os.environ['BUILD_CALLS'], 'a') as output:
    output.write(json.dumps(args) + '\\n')
source = Path(args[-1])
if source.name == 'diskgraph':
    sys.exit(73)
assert args[:3] == ['-m', 'pip', 'wheel'], args
wheel_dir = Path(args[args.index('-w') + 1])
(wheel_dir / (source.name.replace('-', '_') + '-1-py3-none-any.whl')).touch()
''')
    python.chmod(0o755)
    result = subprocess.run(
        ['bash', str(SCRIPT), '--no-zig', *args],
        env=dict(os.environ, KUMO_REPO=str(repo), PYTHON=str(python),
                 PYTHON_TAG='', EXTRA_COMPONENTS='', BUILD_CALLS=str(calls),
                 THIRD_PARTY_NOTICE=str(repo / 'THIRD-PARTY.txt'),
                 TMPDIR=str(tmp_path), PATH=f'{tmp_path}:{os.environ["PATH"]}'),
        capture_output=True,
        text=True,
    )
    if not valid:
        assert result.returncode == 1
        assert 'extra component:' in result.stderr
        assert not calls.exists()
        return
    assert result.returncode == 73, result.stderr
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    sources = [call[-1] for call in recorded]
    expected = ['kumo-api', 'kumo-core', 'kumo-engine', 'kumo-pql']
    expected += providers if providers is not None else [
        'kumo-bigquery', 'kumo-databricks', 'kumo-gcs', 'kumo-s3',
        'kumo-snowflake'
    ]
    assert [Path(source) for source in sources] == [
        repo / 'packages' / name for name in sorted(expected) + ['diskgraph']
    ]


@pytest.mark.parametrize('providers', [
    ['kumo-databricks', 'kumo-s3', 'kumo-snowflake'],
    ['kumo-databricks'],
    ['kumo-databricks', 'kumo-s3'],
])
def test_selected_components_requirements_reach_wheel_metadata(
        tmp_path, providers):
    merger = tmp_path / 'merge.py'
    merger.write_text(SCRIPT.read_text().split(
        "cat > \"${MERGE_PY}\" <<'PYEOF'\n", 1)[1].split('\nPYEOF', 1)[0])
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    dependencies = {
        'kumo-core': [
            'requests>=2.31', 'torch>=2.7.0,<3',
            'faiss-gpu==1.14.3; extra == "faiss-gpu"'
        ],
        'kumo-engine': [
            'kumo-core[faiss-gpu]', 'kumo-databricks; extra == "databricks"',
            'kumo-s3; extra == "s3"', 'kumo-snowflake; extra == "snowflake"'
        ],
        'kumo-databricks': ['databricks-sdk'],
        'kumo-s3': ['boto3'],
        'kumo-snowflake': ['snowflake-snowpark-python; extra == "sdk"'],
    }
    components = tmp_path / 'components.tsv'
    rows = []
    for name in ['kumo-core', 'kumo-engine', *providers]:
        stem = name.replace('-', '_')
        component = wheels / f'{stem}-1.0-py3-none-any.whl'
        with zipfile.ZipFile(component, 'w') as archive:
            archive.writestr(stem + '/__init__.py', '')
            archive.writestr(
                f'{stem}-1.0.dist-info/METADATA',
                f'Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n' +
                ''.join(f'Requires-Dist: {r}\n' for r in dependencies[name]))
        rows.append(f'{component.name}\tkumo-first-party\ttest fixture\n')
    components.write_text(''.join(rows))
    notice = tmp_path / 'THIRD-PARTY.txt'
    notice.write_text('Test notice\n')
    license_file = tmp_path / 'LICENSE'
    license_file.write_text('Test license\n')
    kumo_api_license = tmp_path / 'KUMO-API-LICENSE'
    kumo_api_license.write_text('Test API license\n')
    subprocess.run([
        sys.executable,
        str(merger),
        str(wheels),
        str(components),
        str(tmp_path / 'dist'),
        'kumo-fat-bundle',
        '1.0',
        'cp312-cp312-linux_x86_64',
        'cp312',
        '3.12',
        '0',
        str(notice),
        str(license_file),
        str(kumo_api_license),
        'Apache-2.0 AND MIT',
    ], check=True, capture_output=True, text=True)
    with zipfile.ZipFile(next((tmp_path / 'dist').glob('*.whl'))) as archive:
        metadata = BytesParser().parsebytes(
            archive.read('kumo_fat_bundle-1.0.dist-info/METADATA'))
        manifest = json.loads(archive.read('kumo_bundle_manifest.json'))
    requirements = set(metadata.get_all('Requires-Dist'))
    expected = {
        'torch>=2.7.0,<3', 'faiss-gpu==1.14.3', 'requests>=2.31',
        'databricks-sdk'
    }
    if 'kumo-s3' in providers:
        expected.add('boto3')
    if 'kumo-snowflake' in providers:
        expected.add('snowflake-snowpark-python')
    assert requirements == expected
    assert set(manifest['requires_at_install']) == requirements


@pytest.mark.parametrize('providers', [
    ['kumo_databricks'],
    ['kumo_databricks', 'kumo_s3'],
    [
        'kumo_databricks', 'kumo_s3', 'kumo_snowflake', 'kumo_bigquery',
        'kumo_gcs'
    ],
])
def test_fat_wheel_verifies_selected_providers(tmp_path, providers):
    smoke = tmp_path / 'verify.py'
    smoke.write_text(SCRIPT.read_text().split(
        "cat > \"${WORK}/verify.py\" <<'PYEOF'\n", 1)[1].split('\nPYEOF', 1)[0])
    # Supply the shared runtime and real entry-point metadata for selection.
    modules = {
        'diskgraph/__init__.py':
        '',
        'diskgraph/bin/ingest':
        '',
        'kumo_core/__init__.py':
        '',
        'kumo_core/embedding_models.py':
        'class EmbeddingModelStore:\n'
        '    def get(self, name): return self\n'
        '    def vectors(self): return []\n',
        'kumoapi.py':
        '',
        'kumo/engine/__init__.py':
        'Graph = Table = Trainer = None\n',
        'torch.py':
        '__version__ = "test"\n',
        'torch_geometric.py':
        '__version__ = "test"\n',
    }
    modules.update({
        name + '.py': f'print("{name}")\nProvider = object\n'
        for name in providers
    })
    modules['test_bundle-1.0.dist-info/METADATA'] = (
        'Metadata-Version: 2.1\nName: test-bundle\nVersion: 1.0\n')
    modules['test_bundle-1.0.dist-info/entry_points.txt'] = (
        '[kumoai_studio.sources]\n' + ''.join(f'{name} = {name}:Provider\n'
                                              for name in providers))
    for name, source in modules.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (tmp_path / 'diskgraph/bin/ingest').chmod(0o755)
    result = subprocess.run(
        [sys.executable, '-S', str(smoke), 'test-bundle'],
        env=dict(os.environ, PYTHONPATH=''),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert set(providers) <= set(result.stdout.splitlines())


def test_fat_wheel_uses_torch_fallbacks_without_native_extensions(tmp_path):
    """Run the real assembly script with local stand-ins for component builds."""
    repo = tmp_path / 'repo'
    for name in ['kumo-core', 'kumo-engine', 'diskgraph']:
        package = repo / 'packages' / name
        package.mkdir(parents=True)
        (package / 'pyproject.toml').touch()
    (repo / 'packages/diskgraph/diskgraph-pybind').mkdir()
    (repo / 'packages/diskgraph/diskgraph-pybind/Cargo.toml').touch()
    (repo / 'LICENSE').write_text('Apache-2.0\n')
    api = repo / 'packages/kumo-api'
    api.mkdir()
    (api / 'LICENSE').write_text('MIT\n')
    (repo / 'THIRD-PARTY.txt').touch()
    generator = repo / 'scripts/gen_third_party.py'
    generator.parent.mkdir()
    generator.write_text("print('Apache-2.0 AND MIT')\n")
    calls = tmp_path / 'calls.jsonl'
    python = tmp_path / 'python'
    python.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
from zipfile import ZipFile
args = sys.argv[1:]
if args == ['-m', 'pip', '--version']:
    sys.exit(0)
if args[:3] != ['-m', 'pip', 'wheel'] and args[:3] != ['-m', 'pip', 'download']:
    os.execv(sys.executable, [sys.executable, *args])
with open(os.environ['BUILD_CALLS'], 'a') as output:
    output.write(json.dumps(args) + '\\n')
directory = Path(args[args.index('-w' if args[2] == 'wheel' else '-d') + 1])
for source in (args[-1:] if args[2] == 'wheel' else args[-2:]):
    name = Path(source).name.split('==')[0].replace('-', '_')
    with ZipFile(directory / (name + '-1-py3-none-any.whl'), 'w') as archive:
        metadata = 'Metadata-Version: 2.1\\nName: ' + name + '\\nVersion: 1\\n'
        if name == 'kumo_core':
            metadata += 'Requires-Dist: torch>=2.7,<3\\n'
        archive.writestr(name + '-1.dist-info/METADATA', metadata)
        archive.writestr(name + '/__init__.py', '')
''')
    python.chmod(0o755)
    glove = repo / 'tools/glove-6b-50d/build_wheel.sh'
    glove.parent.mkdir(parents=True)
    glove.write_text(f'''#!/bin/bash
mkdir -p "$1"
"{python}" -m pip wheel --no-deps -q -w "$1" kumo-glove-6b-50d
''')
    glove.chmod(0o755)
    for command in ['cargo', 'curl']:
        executable = tmp_path / command
        executable.write_text('#!/bin/bash\nexit 97\n')
        executable.chmod(0o755)
    output = tmp_path / 'dist'
    result = subprocess.run(
        [
            'bash',
            str(SCRIPT), '--no-zig', '--version', '1', '--output-dir',
            str(output)
        ],
        env=dict(os.environ, KUMO_REPO=str(repo), PYTHON=str(python),
                 BUILD_CALLS=str(calls), TMPDIR=str(tmp_path),
                 THIRD_PARTY_NOTICE=str(repo / 'THIRD-PARTY.txt'),
                 PATH=f'{tmp_path}:{os.environ["PATH"]}'),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    downloads = [
        json.loads(line) for line in calls.read_text().splitlines()
        if json.loads(line)[2] == 'download'
    ]
    assert not downloads
    with ZipFile(next(output.glob('*.whl'))) as archive:
        assert 'pyg_lib/__init__.py' not in archive.namelist()
        assert 'torch_scatter/__init__.py' not in archive.namelist()
        metadata_name = next(name for name in archive.namelist()
                             if name.endswith('.dist-info/METADATA'))
        metadata = archive.read(metadata_name).decode()
        assert 'Requires-Dist: torch>=2.7,<3' in metadata
        assert 'Requires-Dist: pyg-lib' not in metadata
        assert 'Requires-Dist: torch-scatter' not in metadata
