# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import runpy
import subprocess
import sys
import types
import zipfile
from email.parser import BytesParser
from pathlib import Path
from typing import cast
from zipfile import ZipFile

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / 'packages/kumo-engine/scripts'
SCRIPT = SCRIPTS / 'build_wheel.sh'
MERGE = SCRIPTS / 'merge_wheels.py'


@pytest.mark.parametrize('legacy_selection', ['', 'kumo-databricks'])
def test_wheel_builds_all_first_party_packages_from_monorepo(
        tmp_path, legacy_selection):
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
        ['bash', str(SCRIPT), '--no-zig', '--version', '1'],
        env=dict(os.environ, KUMO_REPO=str(repo), PYTHON=str(python),
                 EXTRA_COMPONENTS=legacy_selection, BUILD_CALLS=str(calls),
                 THIRD_PARTY_NOTICE=str(repo / 'THIRD-PARTY.txt'),
                 TMPDIR=str(tmp_path), PATH=f'{tmp_path}:{os.environ["PATH"]}'),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 73, result.stderr
    recorded = [json.loads(line) for line in calls.read_text().splitlines()]
    sources = [call[-1] for call in recorded]
    expected = ['kumo-api', 'kumo-core', 'kumo-engine', 'kumo-pql']
    expected += [
        'kumo-bigquery', 'kumo-databricks', 'kumo-gcs', 'kumo-s3',
        'kumo-snowflake'
    ]
    assert [Path(source) for source in sources] == [
        repo / 'packages' / name for name in sorted(expected) + ['diskgraph']
    ]


def test_removed_component_selection_flag_is_rejected(tmp_path):
    result = subprocess.run(
        ['bash', str(SCRIPT), '--extra-components', 'kumo-databricks'],
        env=dict(os.environ, KUMO_REPO=str(tmp_path)),
        capture_output=True, text=True)
    assert result.returncode == 2
    assert "unknown argument '--extra-components'" in result.stderr


@pytest.mark.parametrize('target', [
    '10.15', '11', '12.7', '13', '13.9', 'invalid', '14.invalid',
])
def test_macos_deployment_target_rejected_before_build(tmp_path, target):
    uname = tmp_path / 'uname'
    uname.write_text('#!/bin/sh\necho Darwin\n')
    uname.chmod(0o755)
    marker = tmp_path / 'python-invoked'
    python = tmp_path / 'python'
    python.write_text(
        f'#!{sys.executable}\nfrom pathlib import Path\n'
        f'Path({str(marker)!r}).touch()\n')
    python.chmod(0o755)

    result = subprocess.run(
        ['bash', str(SCRIPT)],
        env=dict(os.environ, MACOSX_DEPLOYMENT_TARGET=target, USE_ZIG='0',
                 PYTHON=str(python), PATH=f'{tmp_path}:{os.environ["PATH"]}'),
        capture_output=True, text=True)

    assert result.returncode == 1
    assert 'MACOSX_DEPLOYMENT_TARGET >= 14.0' in result.stderr
    assert not marker.exists()


def _wheel(directory, name, requires=(), extras=(), tag='py3-none-any',
           python='>=3.11', files=None, headers=''):
    """Write a component wheel with just enough metadata to merge."""
    stem = name.replace('-', '_')
    path = directory / f'{stem}-1.0-{tag}.whl'
    metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n'
    metadata += headers
    if python is not None:
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


CORE_COMPONENTS = ['kumo-engine', 'kumo-api', 'kumo-pql', 'kumo-core',
                   'diskgraph']
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
            'kumo-api>=3', 'kumo-pql>=0.1', 'torch-geometric>=2.8.1,<3',
            'faiss-gpu==1.14.3; extra == "gpu"',
            'faiss-cpu==1.14.2; extra == "cpu"'
        ],
        'extras': ['gpu', 'cpu'],
    },
    'kumo-api': {
        'requires': ['pydantic>=2.7'],
        'files': {'kumoapi/__init__.py': ''},
    },
    'kumo-pql': {
        'requires': ['kumo-api>=3', 'antlr4-python3-runtime==4.9.3'],
        'files': {'kumopql/__init__.py': ''},
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


def test_merged_module_reports_bundle_version(tmp_path, monkeypatch):
    monkeypatch.setitem(COMPONENTS, 'kumo-engine', {
        **COMPONENTS['kumo-engine'],
        'files': {'kumo/engine/_version.py': '__version__ = "0.1"\n'},
    })
    result = _merge(tmp_path, list(COMPONENTS))
    assert result.returncode == 0, result.stderr
    with ZipFile(next((tmp_path / 'dist').glob('*.whl'))) as archive:
        assert archive.read('kumo/engine/_version.py') == (
            b"__version__ = '1.0'\n")


@pytest.mark.parametrize(('platform', 'expected'), [
    ('linux', {'faiss-gpu'}),
    ('darwin', {'faiss-cpu'}),
    ('win32', set()),
])
def test_engine_faiss_platform_selection_survives_merge(
        tmp_path, monkeypatch, platform, expected):
    from packaging.requirements import Requirement

    setup_kwargs = {}
    monkeypatch.setitem(sys.modules, 'setuptools', types.SimpleNamespace(
        setup=lambda **kwargs: setup_kwargs.update(kwargs)))
    runpy.run_path(str(REPO_ROOT / 'packages/kumo-engine/setup.py'))
    engine = COMPONENTS['kumo-engine']
    provider_requirements = [
        requirement for requirement in engine['requires']
        if 'extra ==' in requirement
    ]
    monkeypatch.setitem(COMPONENTS, 'kumo-engine', {
        **engine,
        'requires': setup_kwargs['install_requires'] + provider_requirements,
    })
    result = _merge(tmp_path, list(COMPONENTS))
    assert result.returncode == 0, result.stderr
    metadata, manifest, _ = _read_wheel(tmp_path)
    faiss = [
        Requirement(value) for value in metadata.get_all('Requires-Dist')
        if value.startswith('faiss-')
    ]
    assert {requirement.name for requirement in faiss} == {
        'faiss-cpu', 'faiss-gpu'
    }
    assert all(str(requirement.specifier) == '==1.14.3'
               and requirement.marker is not None for requirement in faiss)
    selected = {
        requirement.name for requirement in faiss
        if requirement.marker.evaluate({'sys_platform': platform, 'extra': ''})
    }
    assert selected == expected
    assert {str(requirement) for requirement in faiss} <= set(
        manifest['requires'])


@pytest.mark.parametrize('providers', [
    ['kumo-databricks', 'kumo-s3', 'kumo-snowflake'],
    ['kumo-databricks'],
])
def test_provider_dependencies_only_come_with_their_extra(tmp_path, providers):
    result = _merge(tmp_path,
                    [*CORE_COMPONENTS, *providers])
    assert result.returncode == 0, result.stderr
    metadata, manifest, names = _read_wheel(tmp_path)
    requirements = set(metadata.get_all('Requires-Dist'))

    # Base install: engine + core closure, no provider dependency at all.
    base = {'pandas', 'requests>=2.31', 'torch>=2.7.0,<3', 'faiss-gpu==1.14.3',
            'pydantic>=2.7', 'antlr4-python3-runtime==4.9.3',
            'torch-geometric>=2.8.1,<3'}
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
    assert metadata.get_payload().rstrip() == 'README body'


def test_core_packages_and_shared_license_are_bundled(tmp_path):
    result = _merge(tmp_path, CORE_COMPONENTS)
    assert result.returncode == 0, result.stderr
    metadata, manifest, names = _read_wheel(tmp_path)
    assert {'kumoapi/__init__.py', 'kumopql/__init__.py',
            'kumo_core/__init__.py'} <= set(names)
    assert {component['name'] for component in manifest['components']} == set(
        CORE_COMPONENTS)
    assert set(metadata.get_all('License-File')) == {'LICENSE', 'THIRD-PARTY.txt'}
    with ZipFile(next((tmp_path / 'dist').glob('*.whl'))) as archive:
        assert archive.read('kumo_engine-1.0.dist-info/licenses/LICENSE'
                            ) == b'Test license\n'
        assert not any('KUMO-API-LICENSE' in name for name in archive.namelist())
    requirements = metadata.get_all('Requires-Dist')
    for name in ['kumo-api', 'kumo-pql', 'kumo-core']:
        assert not any(requirement.startswith(name)
                       for requirement in requirements)


def test_engine_project_metadata_passes_through(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    headers = ('Author: NVIDIA Corporation\n'
               'Classifier: Development Status :: 5 - Production/Stable\n'
               'Project-URL: issues, https://example.com/issues\n')
    component = cast(dict[str, object], COMPONENTS['kumo-engine'])
    monkeypatch.setitem(COMPONENTS, 'kumo-engine',
                        dict(component, headers=headers))

    result = _merge(tmp_path, CORE_COMPONENTS)

    assert result.returncode == 0, result.stderr
    metadata, _, _ = _read_wheel(tmp_path)
    assert metadata['Author'] == 'NVIDIA Corporation'
    assert metadata['Classifier'] == (
        'Development Status :: 5 - Production/Stable')
    assert metadata['Project-URL'] == 'issues, https://example.com/issues'


def test_glove_stays_a_pypi_dependency_of_the_text_extra(tmp_path):
    result = _merge(tmp_path, CORE_COMPONENTS)
    assert result.returncode == 0, result.stderr
    metadata, _, names = _read_wheel(tmp_path)
    assert 'text' in metadata.get_all('Provides-Extra')
    assert ('kumo-glove-6b-50d==1.0.0; extra == "text"'
            in metadata.get_all('Requires-Dist'))
    assert 'kumo_glove_6b_50d/__init__.py' not in names


@pytest.mark.parametrize('name', ['kumo-glove-6b-50d'])
def test_published_packages_are_never_bundled(tmp_path, name):
    result = _merge(tmp_path, [*CORE_COMPONENTS, name])
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
        result = _merge(tmp_path, CORE_COMPONENTS)
    finally:
        del COMPONENTS['kumo-engine']['files']
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(next((tmp_path / 'dist').glob('*.whl'))) as archive:
        points = archive.read('kumo_engine-1.0.dist-info/entry_points.txt')
    assert b'kumo-engine = kumo_engine_version:installed_version' in points


def test_wheel_tag_is_one_abi3_wheel_for_all_pythons(tmp_path):
    result = _merge(tmp_path, CORE_COMPONENTS)
    assert result.returncode == 0, result.stderr
    wheel = next((tmp_path / 'dist').glob('*.whl'))
    assert wheel.name == (
        'kumo_engine-1.0-cp311-abi3-manylinux_2_34_x86_64.whl')


@pytest.mark.parametrize(('core_python', 'pql_python', 'allowed', 'excluded'), [
    ('>=3.12', '>=3.11', ['3.12', '3.14'], ['3.11']),
    ('>=3.11.4,<3.14', '!=3.12.*', ['3.11.4', '3.13'],
     ['3.11.3', '3.12.1', '3.14']),
    ('~=3.12.1', '<3.14', ['3.12.1', '3.12.9'], ['3.12', '3.13']),
    ('==3.13.*', '>=3.11', ['3.13', '3.13.5'], ['3.12', '3.14']),
    ('>=3.11', None, ['3.11', '3.14'], ['3.10']),
])
def test_merged_python_requirement_intersects_all_components(
        tmp_path, monkeypatch, core_python, pql_python, allowed, excluded):
    from packaging.specifiers import SpecifierSet

    for name, python in [('kumo-core', core_python), ('kumo-pql', pql_python)]:
        monkeypatch.setitem(COMPONENTS, name,
                            dict(COMPONENTS[name], python=python))
    result = _merge(tmp_path, CORE_COMPONENTS)
    assert result.returncode == 0, result.stderr
    metadata, manifest, _ = _read_wheel(tmp_path)
    requirement = SpecifierSet(metadata['Requires-Python'])
    assert all(version in requirement for version in allowed)
    assert all(version not in requirement for version in excluded)
    assert manifest['bundle']['requires_python'] == str(requirement)
    recorded = {component['name']: component['requires_python']
                for component in manifest['components']}
    assert recorded['kumo-core'] == core_python
    assert recorded['kumo-pql'] == pql_python


@pytest.mark.parametrize(('component_python', 'expected'), [
    ('>=3.10', '>=3.11'),
    ('>=3.12', '>=3.12'),
    ('>=3.11.4,<3.14,!=3.12.*', '!=3.12.*,<3.14,>=3.11.4'),
])
def test_merged_python_requirement_removes_weaker_minimums(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        component_python: str, expected: str) -> None:
    """Keep the strongest minimum and all upper bounds and exclusions."""
    component = cast(dict[str, object], COMPONENTS['kumo-core'])
    monkeypatch.setitem(COMPONENTS, 'kumo-core',
                        dict(component, python=component_python))

    result = _merge(tmp_path, CORE_COMPONENTS)

    assert result.returncode == 0, result.stderr
    metadata, manifest, _ = _read_wheel(tmp_path)
    assert metadata['Requires-Python'] == expected
    assert manifest['bundle']['requires_python'] == expected
    recorded = {component['name']: component['requires_python']
                for component in manifest['components']}
    assert recorded['kumo-core'] == component_python


@pytest.mark.parametrize(('core_python', 'pql_python', 'error'), [
    ('>=3.13', '<3.13', 'no common stable CPython 3 version'),
    ('==3.12.*', '!=3.12.*', 'no common stable CPython 3 version'),
    ('>=3.11.1', '<3.11.1', 'no common stable CPython 3 version'),
    ('invalid', '>=3.11', 'kumo-core has invalid Requires-Python'),
])
def test_incompatible_python_requirements_fail_before_writing_wheel(
        tmp_path, monkeypatch, core_python, pql_python, error):
    for name, python in [('kumo-core', core_python), ('kumo-pql', pql_python)]:
        monkeypatch.setitem(COMPONENTS, name,
                            dict(COMPONENTS[name], python=python))
    result = _merge(tmp_path, CORE_COMPONENTS)
    assert result.returncode != 0
    assert error in result.stderr
    assert not (tmp_path / 'dist').exists()


def test_native_abi_floor_uses_combined_python_minimum(tmp_path, monkeypatch):
    monkeypatch.setitem(COMPONENTS, 'kumo-core',
                        dict(COMPONENTS['kumo-core'], python='>=3.12'))
    monkeypatch.setitem(COMPONENTS, 'diskgraph',
                        {'tag': 'cp312-abi3-manylinux_2_34_x86_64'})
    result = _merge(tmp_path, CORE_COMPONENTS)
    assert result.returncode == 0, result.stderr
    assert next((tmp_path / 'dist').glob('*.whl')).name == (
        'kumo_engine-1.0-cp312-abi3-manylinux_2_34_x86_64.whl')


def test_native_component_for_newer_python_is_refused(tmp_path):
    COMPONENTS['diskgraph-new'] = {'tag': 'cp312-cp312-linux_x86_64'}
    try:
        result = _merge(
            tmp_path,
            [*CORE_COMPONENTS, 'diskgraph-new'])
    finally:
        del COMPONENTS['diskgraph-new']
    assert result.returncode != 0
    assert 'diskgraph-new' in result.stderr


def test_bundled_package_without_an_extra_is_refused(tmp_path):
    COMPONENTS['kumo-orphan'] = {'requires': ['orphan-dependency']}
    try:
        result = _merge(
            tmp_path, [*CORE_COMPONENTS, 'kumo-orphan'])
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
@pytest.mark.parametrize('host_os,target', [
    ('Linux', ''), ('Darwin', ''), ('Darwin', '14'), ('Darwin', '15.0'),
])
def test_wheel_uses_torch_fallbacks_without_native_extensions(
        tmp_path, args, version, host_os, target):
    """Run the real assembly script with local stand-ins for component builds."""
    deployment_target = target or '14.0'
    major, minor, *_ = deployment_target.split('.') + ['0']
    native_tag = f'cp311-abi3-macosx_{major}_{minor}_arm64'
    repo = tmp_path / 'repo'
    for name in CORE_COMPONENTS:
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
    if name == 'diskgraph' and os.environ['TEST_HOST_OS'] == 'Darwin':
        assert os.environ['DISKGRAPH_BUILD_ZIG'] == '0'
        assert os.environ['MACOSX_DEPLOYMENT_TARGET'] == {deployment_target!r}
    tag = ({native_tag!r} if name == 'diskgraph'
           and os.environ['TEST_HOST_OS'] == 'Darwin' else 'py3-none-any')
    with ZipFile(directory / (name + '-1-' + tag + '.whl'), 'w') as archive:
        metadata = 'Metadata-Version: 2.1\\nName: ' + name + '\\nVersion: 1\\n'
        if name == 'kumo_engine':
            metadata += ('Requires-Python: >=3.11\\nProvides-Extra: text\\n'
                         'Requires-Dist: kumo-core\\n'
                         'Requires-Dist: kumo-glove-6b-50d; extra == "text"\\n')
        if name == 'kumo_core':
            metadata += ('Requires-Dist: torch>=2.7,<3\\n'
                         'Requires-Dist: diskgraph\\n'
                         'Requires-Dist: torch-geometric>=2.8.1,<3\\n'
                         'Requires-Dist: kumo-api\\n'
                         'Requires-Dist: kumo-pql\\n')
        if name == 'kumo_pql':
            metadata += 'Requires-Dist: kumo-api\\n'
        archive.writestr(name + '-1.dist-info/METADATA', metadata)
        archive.writestr(name + '/__init__.py', '')
''')
    python.chmod(0o755)
    uname = tmp_path / 'uname'
    arch = 'arm64' if host_os == 'Darwin' else 'x86_64'
    uname.write_text(f'#!/bin/sh\ncase "$1" in\n'
                     f'-s) echo {host_os} ;;\n-m) echo {arch} ;;\nesac\n')
    uname.chmod(0o755)
    for command in ['cargo', 'curl']:
        executable = tmp_path / command
        executable.write_text('#!/bin/bash\nexit 97\n')
        executable.chmod(0o755)
    output = tmp_path / 'dist'
    result = subprocess.run(
        ['bash',
         str(SCRIPT), *(['--no-zig'] if host_os == 'Linux' else []),
         *args, '--output-dir',
         str(output)],
        env=dict(os.environ, KUMO_REPO=str(repo), PYTHON=str(python),
                 BUILD_CALLS=str(calls), TMPDIR=str(tmp_path),
                 TEST_HOST_OS=host_os, USE_ZIG='', PLATFORM_TAG='',
                 MACOSX_DEPLOYMENT_TARGET=target,
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
    if host_os == 'Darwin':
        assert next(output.glob('*.whl')).name.endswith(f'-{native_tag}.whl')
    with ZipFile(next(output.glob('*.whl'))) as archive:
        assert not any(name.startswith('torch_geometric/')
                       for name in archive.namelist())
        assert 'pyg_lib/__init__.py' not in archive.namelist()
        assert 'torch_scatter/__init__.py' not in archive.namelist()
        metadata_name = next(name for name in archive.namelist()
                             if name.endswith('.dist-info/METADATA'))
        metadata = archive.read(metadata_name).decode()
        assert 'Requires-Dist: torch>=2.7,<3' in metadata
        assert 'Requires-Dist: torch-geometric>=2.8.1,<3' in metadata
        assert 'License-Expression: Apache-2.0 AND MIT' in metadata
        assert (f'kumo_engine-{version}.dist-info/licenses/LICENSE'
                in archive.namelist())
        assert not any('KUMO-API-LICENSE' in name
                       for name in archive.namelist())
        for component in ['kumo_api', 'kumo_pql', 'kumo_core']:
            assert component + '/__init__.py' in archive.namelist()
        for component in ['kumo-api', 'kumo-pql', 'kumo-core']:
            assert 'Requires-Dist: ' + component not in metadata
        assert 'Requires-Dist: kumo-glove-6b-50d; extra == "text"' in metadata
        assert 'Requires-Dist: pyg-lib' not in metadata
        assert 'Requires-Dist: torch-scatter' not in metadata
    expected_tag = native_tag if host_os == 'Darwin' else 'py3-none-any'
    assert next(output.glob('*.whl')).name == (
        f'kumo_engine-{version}-{expected_tag}.whl')
