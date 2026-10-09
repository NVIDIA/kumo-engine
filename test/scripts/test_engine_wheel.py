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
SCRIPTS = REPO_ROOT / 'packages/kumo-engine/scripts'
SCRIPT = SCRIPTS / 'build_wheel.sh'
MERGE = SCRIPTS / 'merge_wheels.py'


@pytest.mark.parametrize(('args', 'valid', 'providers'), [
    ([], True, None),
    (['--extra-components', 'kumo-databricks'], True, ['kumo-databricks']),
    (['--extra-components', ' kumo-databricks, kumo-s3 '
      ], True, ['kumo-databricks', 'kumo-s3']),
    (['--extra-components', 'kumo-missing'], False, []),
    (['--extra-components', 'kumo-sdk'], False, []),
    (['--extra-components', 'kumo-api'], False, []),
])
def test_wheel_builds_first_party_packages_from_monorepo(
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
    (repo / 'THIRD-PARTY.txt').touch()
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
        ['bash', str(SCRIPT), '--no-zig', '--version', '1', *args],
        env=dict(os.environ, KUMO_REPO=str(repo), PYTHON=str(python),
                 EXTRA_COMPONENTS='', BUILD_CALLS=str(calls),
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
    # kumo-api is a PyPI dependency, never bundled.
    expected = ['kumo-core', 'kumo-engine', 'kumo-pql']
    expected += providers if providers is not None else [
        'kumo-bigquery', 'kumo-databricks', 'kumo-gcs', 'kumo-s3',
        'kumo-snowflake'
    ]
    assert [Path(source) for source in sources] == [
        repo / 'packages' / name for name in sorted(expected) + ['diskgraph']
    ]


def _wheel(directory, name, requires=(), extras=(), tag='py3-none-any',
           python='>=3.11', files=None):
    """Write a component wheel with just enough metadata to merge."""
    stem = name.replace('-', '_')
    path = directory / f'{stem}-1.0-{tag}.whl'
    metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n'
    metadata += f'Requires-Python: {python}\n'
    metadata += ''.join(f'Provides-Extra: {e}\n' for e in extras)
    metadata += ''.join(f'Requires-Dist: {r}\n' for r in requires)
    metadata += '\nREADME body\n'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(f'{stem}/__init__.py', '')
        for filename, content in (files or {}).items():
            archive.writestr(filename, content)
        archive.writestr(f'{stem}-1.0.dist-info/METADATA', metadata)
    return path


ENGINE_EXTRAS = ['snowflake', 'databricks', 's3', 'text', 'test']
COMPONENTS = {
    'kumo-engine': {
        'requires': [
            'pandas', 'kumo-core[gpu]>=1.0', 'pytest; extra == "test"',
            'kumo-snowflake[sdk]>=1.0; extra == "snowflake"',
            'kumo-databricks; extra == "databricks"', 'kumo-s3; extra == "s3"',
            'kumo-glove-6b-50d==1.0.0; extra == "text"'
        ],
        'extras':
        ENGINE_EXTRAS,
    },
    'kumo-core': {
        'requires': [
            'requests>=2.31', 'torch>=2.7.0,<3', 'diskgraph>=0.1.0',
            'faiss-gpu==1.14.3; extra == "gpu"',
            'faiss-cpu==1.14.2; extra == "cpu"'
        ],
        'extras': ['gpu', 'cpu'],
    },
    'diskgraph': {
        'tag': 'cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64',
    },
    'kumo-snowflake': {
        'requires': [
            'kumo-core==1.0', 'snowflake-connector-python>=4.3',
            'snowflake-snowpark-python; extra == "sdk"',
            'unrelated; extra == "test"'
        ],
        'extras': ['sdk', 'test'],
    },
    'kumo-databricks': {
        'requires': ['kumo-core==1.0', 'databricks-sdk'],
    },
    'kumo-s3': {
        'requires': ['kumo-core==1.0', 'boto3'],
    },
}


def _merge(tmp_path, names, *args):
    wheels = tmp_path / 'wheels'
    wheels.mkdir(exist_ok=True)
    rows = []
    for name in names:
        spec = COMPONENTS.get(name, {})
        path = _wheel(wheels, name, **spec)
        rows.append(f'{path.name}\tkumo-first-party\ttest fixture\n')
    components = tmp_path / 'components.tsv'
    components.write_text(''.join(rows))
    notice = tmp_path / 'THIRD-PARTY.txt'
    notice.write_text('Test notice\n')
    license_file = tmp_path / 'LICENSE'
    license_file.write_text('Test license\n')
    return subprocess.run([
        sys.executable,
        str(MERGE),
        str(wheels),
        str(components),
        str(tmp_path / 'dist'),
        '--version',
        '1.0',
        '--platform-tag',
        'manylinux_2_34_x86_64',
        '--notice',
        str(notice),
        '--license',
        str(license_file),
        '--license-expression',
        'Apache-2.0 AND MIT',
        *args,
    ], capture_output=True, text=True)


def _read_wheel(tmp_path):
    with zipfile.ZipFile(next((tmp_path / 'dist').glob('*.whl'))) as archive:
        metadata = BytesParser().parsebytes(
            archive.read('kumo_engine-1.0.dist-info/METADATA'))
        manifest = json.loads(archive.read('kumo_bundle_manifest.json'))
        names = archive.namelist()
    return metadata, manifest, names


@pytest.mark.parametrize('providers', [
    ['kumo-databricks', 'kumo-s3', 'kumo-snowflake'],
    ['kumo-databricks'],
])
def test_provider_dependencies_only_come_with_their_extra(tmp_path, providers):
    result = _merge(tmp_path,
                    ['kumo-engine', 'kumo-core', 'diskgraph', *providers])
    assert result.returncode == 0, result.stderr
    metadata, manifest, names = _read_wheel(tmp_path)
    requirements = set(metadata.get_all('Requires-Dist'))

    # Base install: engine + core closure, no provider dependency at all.
    base = {'pandas', 'requests>=2.31', 'torch>=2.7.0,<3', 'faiss-gpu==1.14.3'}
    assert {r for r in requirements if 'extra ==' not in r} == base
    assert set(manifest['requires']) == base

    # Provider code ships regardless of the extras...
    for provider in providers:
        assert provider.replace('-', '_') + '/__init__.py' in names
    # ...but its dependencies are conditional on the extra.
    expected_extras = {
        'kumo-databricks': {'databricks-sdk; extra == "databricks"'},
        'kumo-s3': {'boto3; extra == "s3"'},
        'kumo-snowflake': {
            'snowflake-connector-python>=4.3; extra == "snowflake"',
            'snowflake-snowpark-python; extra == "snowflake"',
        },
    }
    assert requirements - base == set().union(
        {'kumo-glove-6b-50d==1.0.0; extra == "text"'},
        *(expected_extras[p] for p in providers))
    # `test` is a development extra; `text` points at the PyPI glove package.
    provided = set(metadata.get_all('Provides-Extra'))
    assert provided == {'text', *(p.removeprefix('kumo-') for p in providers)}
    # Extras for providers that are not bundled would point at a package that
    # is not on any index; the only first-party requirement left is glove.
    assert {r
            for r in requirements
            if 'kumo-' in r} == {'kumo-glove-6b-50d==1.0.0; extra == "text"'}
    assert metadata['Requires-Python'] == '>=3.11'
    assert metadata['Name'] == 'kumo-engine'
    assert metadata['Metadata-Version'] == '2.4'


def test_glove_stays_a_pypi_dependency_of_the_text_extra(tmp_path):
    result = _merge(tmp_path, ['kumo-engine', 'kumo-core', 'diskgraph'])
    assert result.returncode == 0, result.stderr
    metadata, _, names = _read_wheel(tmp_path)
    assert 'text' in metadata.get_all('Provides-Extra')
    assert ('kumo-glove-6b-50d==1.0.0; extra == "text"'
            in metadata.get_all('Requires-Dist'))
    assert 'kumo_glove_6b_50d/__init__.py' not in names


@pytest.mark.parametrize('name', ['kumo-api', 'kumo-glove-6b-50d'])
def test_published_packages_are_never_bundled(tmp_path, name):
    result = _merge(tmp_path, ['kumo-engine', 'kumo-core', 'diskgraph', name])
    assert result.returncode != 0
    assert 'must not be bundled' in result.stderr
    assert not (tmp_path / 'dist').exists()


def test_engine_version_provider_entry_point_reaches_the_wheel(tmp_path):
    entry = ('[kumo_core.version_providers]\n'
             'kumo-engine = kumo_engine_version:installed_version\n')
    COMPONENTS['kumo-engine'] = dict(
        COMPONENTS['kumo-engine'],
        files={'kumo_engine-1.0.dist-info/entry_points.txt': entry})
    try:
        result = _merge(tmp_path, ['kumo-engine', 'kumo-core', 'diskgraph'])
    finally:
        del COMPONENTS['kumo-engine']['files']
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(next((tmp_path / 'dist').glob('*.whl'))) as archive:
        points = archive.read('kumo_engine-1.0.dist-info/entry_points.txt')
    assert b'kumo-engine = kumo_engine_version:installed_version' in points


def test_wheel_tag_is_one_abi3_wheel_for_all_pythons(tmp_path):
    result = _merge(tmp_path, ['kumo-engine', 'kumo-core', 'diskgraph'])
    assert result.returncode == 0, result.stderr
    wheel = next((tmp_path / 'dist').glob('*.whl'))
    assert wheel.name == (
        'kumo_engine-1.0-cp311-abi3-manylinux_2_34_x86_64.whl')


def test_native_component_for_newer_python_is_refused(tmp_path):
    COMPONENTS['diskgraph-new'] = {'tag': 'cp312-cp312-linux_x86_64'}
    try:
        result = _merge(
            tmp_path,
            ['kumo-engine', 'kumo-core', 'diskgraph', 'diskgraph-new'])
    finally:
        del COMPONENTS['diskgraph-new']
    assert result.returncode != 0
    assert 'diskgraph-new' in result.stderr


def test_bundled_package_without_an_extra_is_refused(tmp_path):
    COMPONENTS['kumo-orphan'] = {'requires': ['orphan-dependency']}
    try:
        result = _merge(
            tmp_path, ['kumo-engine', 'kumo-core', 'diskgraph', 'kumo-orphan'])
    finally:
        del COMPONENTS['kumo-orphan']
    assert result.returncode != 0
    assert 'kumo-orphan' in result.stderr
    assert not (tmp_path / 'dist').exists()


def test_wheel_verify_script_checks_base_install(tmp_path):
    smoke = tmp_path / 'verify.py'
    smoke.write_text(SCRIPT.read_text().split(
        "cat > \"${WORK}/verify.py\" <<'PYEOF'\n", 1)[1].split('\nPYEOF', 1)[0])
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
        'kumo_core/version.py':
        'def get_kumo_version(): return "1.0"\n',
        'kumoapi.py':
        '',
        'kumo/engine/__init__.py':
        '__version__ = "1.0"\n'
        'FileConnector = SnowflakeConnector = Graph = Table = Trainer = None\n',
        'torch.py':
        '__version__ = "test"\n',
        'torch_geometric.py':
        '__version__ = "test"\n',
        'test_bundle-1.0.dist-info/METADATA':
        'Metadata-Version: 2.1\nName: test-bundle\nVersion: 1.0\n'
        'Provides-Extra: snowflake\n',
        'test_bundle-1.0.dist-info/entry_points.txt':
        '[kumoai_studio.sources]\nsnowflake = kumo_snowflake.backend:X\n',
    }
    for name, source in modules.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    (tmp_path / 'diskgraph/bin/ingest').chmod(0o755)

    def run():
        return subprocess.run(
            [sys.executable, '-S',
             str(smoke), 'test-bundle'],
            env=dict(os.environ, PYTHONPATH=''),
            capture_output=True,
            text=True,
        )

    result = run()
    assert result.returncode == 0, result.stderr
    assert 'wheel OK 1.0' in result.stdout

    # A base install that nevertheless carries the Snowflake client is
    # exactly what the check exists to catch.
    (tmp_path / 'snowflake').mkdir()
    (tmp_path / 'snowflake/__init__.py').write_text('')
    (tmp_path / 'snowflake/connector.py').write_text('')
    result = run()
    assert result.returncode != 0
    assert 'without the snowflake extra' in result.stderr


@pytest.mark.parametrize(
    ('args', 'version'),
    [
        (['--version', '1'], '1'),
        ([], '2.5.0+unknown'),  # VERSION + commit (not a git repo here)
        (['--release'], '2.5.0'),
    ])
def test_wheel_uses_torch_fallbacks_without_native_extensions(
        tmp_path, args, version):
    """Run the real assembly script with local stand-ins for component builds."""
    repo = tmp_path / 'repo'
    for name in ['kumo-core', 'kumo-engine', 'diskgraph']:
        package = repo / 'packages' / name
        package.mkdir(parents=True)
        (package / 'pyproject.toml').touch()
    (repo / 'VERSION').write_text('2.5.0\n')
    (repo / 'packages/diskgraph/diskgraph-pybind').mkdir()
    (repo / 'packages/diskgraph/diskgraph-pybind/Cargo.toml').touch()
    (repo / 'LICENSE').write_text('Apache-2.0\n')
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
        if name == 'kumo_engine':
            metadata += ('Requires-Python: >=3.11\\nProvides-Extra: text\\n'
                         'Requires-Dist: kumo-core\\n'
                         'Requires-Dist: kumo-glove-6b-50d; extra == "text"\\n')
        if name == 'kumo_core':
            metadata += ('Requires-Dist: torch>=2.7,<3\\n'
                         'Requires-Dist: diskgraph\\n'
                         'Requires-Dist: torch-geometric\\n')
        archive.writestr(name + '-1.dist-info/METADATA', metadata)
        archive.writestr(name + '/__init__.py', '')
''')
    python.chmod(0o755)
    for command in ['cargo', 'curl']:
        executable = tmp_path / command
        executable.write_text('#!/bin/bash\nexit 97\n')
        executable.chmod(0o755)
    output = tmp_path / 'dist'
    result = subprocess.run(
        ['bash',
         str(SCRIPT), '--no-zig', *args, '--output-dir',
         str(output)],
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
        assert 'License-Expression: Apache-2.0 AND MIT' in metadata
        assert (f'kumo_engine-{version}.dist-info/licenses/LICENSE'
                in archive.namelist())
        assert not any('KUMO-API' in n for n in archive.namelist())
        assert 'Requires-Dist: kumo-glove-6b-50d; extra == "text"' in metadata
        assert 'Requires-Dist: pyg-lib' not in metadata
        assert 'Requires-Dist: torch-scatter' not in metadata
    assert next(output.glob('*.whl')).name == (
        f'kumo_engine-{version}-py3-none-any.whl')
