# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a real Workspace notebook; all large assets stay on the test stage."""
import base64
import hashlib
import json
import re
from pathlib import Path

from resources import admit_candidate, candidate_path

MODULES = {
    'graph.py', 'table.py', '_widget.py', '_visualization.py',
    '_visualization_viewer.py', '_visualization_pdf.py',
    '_visualization_layout.py'
}


def validate_source(commit, hashes):
    if (not re.fullmatch(r'[0-9a-f]{40}', commit) or set(hashes) != MODULES
            or not all(
                re.fullmatch(r'[0-9a-f]{64}', value)
                for value in hashes.values())):
        raise ValueError(
            'Complete reviewed graph module hashes and commit required')


def collect(receipt, spec, root, key_file):
    """Download only the identified execution's checksummed test evidence."""
    import shutil
    from xml.etree import ElementTree

    import launch

    execution = receipt.get('execution_id', '')
    base = '@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE/mlops-validation/graph-notebook/'
    stage = base + spec['attempt'] + '/results/' + execution
    names = {
        'graph.svg', 'graph.pdf', 'graph.md', 'default.html', 'sampled.html',
        'graph-result.json'
    }
    if (not re.fullmatch(r'[0-9a-f]{12}', execution)
            or receipt.get('output_stage') != stage
            or set(receipt.get('uploaded', {})) != names or not all(
                re.fullmatch(r'[0-9a-f]{64}', digest)
                for digest in receipt['uploaded'].values())):
        raise ValueError('Invalid graph evidence inventory or non-test stage')
    for suffix in ('svg', 'pdf', 'md'):
        if receipt['uploaded']['graph.' +
                               suffix] != receipt['exports'][suffix]['sha256']:
            raise ValueError('Uploaded graph differs from verified export')
    root = Path(root)
    root.mkdir(mode=0o700, exist_ok=False)
    with launch.connect(key_file) as connection, connection.cursor() as cursor:
        for name, expected in receipt['uploaded'].items():
            incoming = root / (name + '.download')
            incoming.mkdir()
            downloaded = launch.download_file(cursor, stage + '/' + name,
                                              incoming)
            if launch.sha(downloaded) != expected:
                raise ValueError(
                    'Downloaded graph evidence checksum mismatch: ' + name)
            shutil.move(str(downloaded), root / name)
            incoming.rmdir()
    stored = json.loads((root / 'graph-result.json').read_text())
    expected_report = {
        name: value
        for name, value in receipt.items()
        if name not in ('output_stage', 'uploaded')
    }
    if stored != expected_report:
        raise ValueError(
            'Visible notebook result differs from retained receipt')
    tree = ElementTree.fromstring((root / 'graph.svg').read_bytes())
    tables = {
        node.get('data-kumo-table')
        for node in tree.iter() if node.get('data-kumo-table')
    }
    if tables != set(receipt['schema']['tables']):
        raise ValueError('Downloaded graph lost fixture tables')
    if not (root / 'graph.pdf').read_bytes().startswith(
            b'%PDF-') or 'flowchart LR' not in (root / 'graph.md').read_text():
        raise ValueError('Downloaded graph export is invalid')
    for suffix in ('svg', 'pdf', 'md'):
        raw = (root / ('graph.' + suffix)).read_bytes()
        if any(value.encode() in raw for value in receipt['canaries']):
            raise ValueError('Downloaded diagram contains sample values')
    return dict(status='passed', stage=stage, files=receipt['uploaded'])


def load_config(path):
    value = json.loads(Path(path).read_text())
    validate_source(value['source']['commit'], value['source']['modules'])
    if not re.fullmatch(r'[0-9a-f]{12}', value['attempt']):
        raise ValueError('Invalid notebook attempt')
    notebook = Path(value['path'])
    if not notebook.is_absolute():
        if notebook.name != value['path']:
            raise ValueError('Notebook reference must be a bundle filename')
        notebook = Path(path).resolve().parent / notebook
    elif not notebook.exists():
        notebook = Path(path).resolve().parent / notebook.name
    if (notebook.name != 'kumo-test-graph-' + value['attempt'] + '.ipynb'
            or notebook.is_symlink() or hashlib.sha256(
                notebook.read_bytes()).hexdigest() != value['sha256']):
        raise ValueError('Prepared notebook checksum/identity mismatch')
    candidate = json.loads(candidate_path().read_text())['candidate']
    if value['candidate'] != candidate or value.get(
            'action', 'create') not in ('create', 'run', 'verify', 'cleanup'):
        raise ValueError('Invalid notebook candidate or action')
    expected_fixture = json.loads(
        Path(__file__).parent.parent.joinpath(
            'pr172/inputs.json').read_text())['sha256']
    if value['fixture_sha256'] != expected_fixture or set(
            value['test_helper_sha256']) != {
                'graph_fixture.py', 'wheel_gpu_smoke.py'
            }:
        raise ValueError(
            'Notebook must identify the shared graph fixture and test helpers')
    if not all(
            re.fullmatch(r'[0-9a-f]{64}', digest)
            for digest in value['test_helper_sha256'].values()):
        raise ValueError('Invalid test helper hashes')
    return {
        key: value[key]
        for key in ('path', 'attempt', 'source', 'candidate', 'sha256',
                    'fixture_sha256', 'test_helper_sha256')
    } | {
        'path': str(notebook.resolve()),
        'action': value.get('action', 'create')
    }


def prepare(destination, pdf_asset, source_commit, module_hashes,
            fixture_asset=None):
    validate_source(source_commit, module_hashes)
    from launch import cache_location_allowed
    source = Path(__file__).parent
    fixture_manifest = json.loads(
        (source.parent / 'pr172/inputs.json').read_text())
    if not fixture_asset or fixture_asset.get(
            'sha256') != fixture_manifest['sha256']:
        raise ValueError(
            'shared fixture asset must match the shared PR172 manifest')
    for asset in (pdf_asset, fixture_asset):
        if (not re.fullmatch(r'[0-9a-f]{64}', asset['sha256'])
                or type(asset['size']) is not int or asset['size'] <= 0 or
                not cache_location_allowed(asset['location'], asset['name'])):
            raise ValueError(
                'Asset must be content-addressed on the dedicated test stage')
    from graph_assets import PDF
    if any(pdf_asset[key] != PDF[key] for key in ('name', 'sha256', 'size')):
        raise ValueError('Use the pinned PDF test dependency')
    from resources import require
    target = require()
    identity = [
        target[key] for key in ('account_locator', 'browser_user', 'role',
                                'warehouse', 'database', 'schema')
    ]
    candidate = json.loads(candidate_path().read_text())
    payload = {
        'candidate.json':
        candidate,
        'wheelhouse-manifest.json':
        candidate['dependencies'],
        'pdf-asset.json':
        pdf_asset,
        'fixture-asset.json':
        fixture_asset,
        'fixture-manifest.json':
        fixture_manifest,
        'source.json':
        dict(commit=source_commit, modules=module_hashes),
        'identity.json':
        identity,
    }
    files = {name: json.dumps(value) for name, value in payload.items()}
    for name in ('offline_support.py', 'graph_fixture.py'):
        files[name] = (source / name).read_text()
    for name in ('wheel_gpu_smoke.py', 'wheel_mlops_lifecycle.py'):
        files[name] = (source.parent / name).read_text()
    encoded = base64.b64encode(json.dumps(files).encode()).decode()
    bootstrap = '''import base64, hashlib, json, os, subprocess, sys
from pathlib import Path
from snowflake.snowpark.context import get_active_session
ROOT = Path('/tmp/kumo-graph-notebook-ATTEMPT')
ROOT.mkdir(mode=0o700, exist_ok=True)
from uuid import uuid4
setup_execution = uuid4().hex[:12]
print('KUMO_GRAPH_INSTALL_STARTED=' + setup_execution, flush=True)
files = json.loads(base64.b64decode('PAYLOAD'))
for name, content in files.items():
    (ROOT / name).write_text(content)
session = get_active_session()
session.sql('USE WAREHOUSE KUMO_TEST_WH').collect()
session.sql('USE DATABASE KUMO_TEST_DB').collect()
session.sql('USE SCHEMA KUMO_TEST_DB.KUMO_TEST').collect()
identity = session.sql('SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()').collect()[0]
assert list(identity) == json.loads((ROOT / 'identity.json').read_text()), 'Notebook identity differs from protected test configuration'
candidate = json.loads((ROOT / 'candidate.json').read_text())
assets = dict(candidate['assets'])
pdf = json.loads((ROOT / 'pdf-asset.json').read_text())
assets[pdf['name']] = pdf
fixture = json.loads((ROOT / 'fixture-asset.json').read_text())
assets[fixture['name']] = fixture
for name, asset in assets.items():
    target = ROOT / name
    if not target.exists() or hashlib.file_digest(target.open('rb'), 'sha256').hexdigest() != asset['sha256']:
        session.file.get(asset['location'], str(ROOT))
    with target.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == asset['sha256'], name
    assert target.stat().st_size == asset['size'], name
import tarfile
fixture_root = ROOT / 'fixtures'
fixture_root.mkdir(exist_ok=True)
assert fixture['sha256'] == json.loads((ROOT / 'fixture-manifest.json').read_text())['sha256']
with tarfile.open(ROOT / fixture['name']) as archive:
    for entry in archive.getmembers():
        assert (entry.isfile() or entry.isdir()) and not Path(entry.name).is_absolute() and '..' not in Path(entry.name).parts
    archive.extractall(fixture_root, filter='data')
sys.path.insert(0, str(ROOT))
from offline_support import unpack
wheelhouse = ROOT / 'wheelhouse'
if not wheelhouse.exists():
    unpack(ROOT, wheelhouse)
wheel = ROOT / candidate['candidate']['wheel_file']
command = [sys.executable, '-m', 'pip', '--isolated', 'install', '--no-index', '--find-links', str(wheelhouse), str(wheel) + '[snowflake]', str(ROOT / pdf['name'])]
with (ROOT / 'install.log').open('w') as log:
    installed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
if installed.returncode:
    print((ROOT / 'install.log').read_text()[-12000:])
    raise RuntimeError('Notebook package installation failed')
import shutil
shutil.rmtree(wheelhouse)
(ROOT / 'wheelhouse.tar').unlink()
print('KUMO_GRAPH_INSTALL_COMPLETE=' + setup_execution, flush=True)
'''.replace('PAYLOAD', encoded)
    execute = '''from uuid import uuid4
execution = uuid4().hex[:12]
print('KUMO_GRAPH_EXECUTION_STARTED=' + execution, flush=True)
assert 'setup_execution' not in globals(), 'Notebook kernel retained setup state after reset'
import hashlib, importlib.metadata, json, os, sys
from pathlib import Path
from IPython import get_ipython
from IPython.display import display
from snowflake.snowpark.context import get_active_session
ROOT = Path('/tmp/kumo-graph-notebook-ATTEMPT')
session = get_active_session()
session.sql('USE WAREHOUSE KUMO_TEST_WH').collect()
session.sql('USE DATABASE KUMO_TEST_DB').collect()
session.sql('USE SCHEMA KUMO_TEST_DB.KUMO_TEST').collect()
identity = session.sql('SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA()').collect()[0]
assert list(identity) == json.loads((ROOT / 'identity.json').read_text()), 'Notebook identity differs from protected test configuration'
sys.path.insert(0, str(ROOT))
import kumo.engine
from kumo.engine import _widget
assert _widget.is_snowflake_workspace(), 'Not an actual Snowflake Workspace kernel'
source = json.loads((ROOT / 'source.json').read_text())
for name, expected in source['modules'].items():
    actual = Path(kumo.engine.__file__).parent / name
    with actual.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
candidate = json.loads((ROOT / 'candidate.json').read_text())
assert importlib.metadata.version('kumo-engine') == candidate['candidate']['expected_version']
from graph_fixture import exercise, retain
RESULTS = ROOT / ('results-' + execution)
from wheel_mlops_lifecycle import check_install
installation = check_install(ROOT / candidate['candidate']['wheel_file'], candidate['candidate']['expected_version'], managed=True)
fixture_manifest = json.loads((ROOT / 'fixture-manifest.json').read_text())
fixture_asset = json.loads((ROOT / 'fixture-asset.json').read_text())
with (ROOT / fixture_asset['name']).open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == fixture_manifest['sha256']
report = exercise(RESULTS, ROOT / 'fixtures', display=display)
report['installation'] = installation
report['execution_id'] = execution
report['fixture_sha256'] = fixture_manifest['sha256']
report['test_helper_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ['graph_fixture.py', 'wheel_gpu_smoke.py']}
report['identity'] = dict(zip(['account', 'user', 'role', 'warehouse', 'database', 'schema'], identity))
report['runtime'] = dict(python=sys.version, source=source, wheel=candidate['candidate'], workspace=True, extensions=sorted(get_ipython().extension_manager.loaded))
(RESULTS / 'graph-result.json').write_text(json.dumps(report, indent=2))
report['output_stage'] = '@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE/mlops-validation/graph-notebook/ATTEMPT/results/' + execution
report['uploaded'] = retain(session, RESULTS, report['output_stage'])
print('KUMO_GRAPH_RESULT=' + json.dumps(report))
'''
    attempt = Path(destination).stem.removeprefix('kumo-test-graph-')
    if not re.fullmatch(r'[0-9a-f]{12}', attempt):
        raise ValueError('Use a twelve-digit hexadecimal attempt identifier')
    cells = [
        dict(id=attempt + '-' + str(index), cell_type='code',
             execution_count=None, metadata={}, outputs=[],
             source=code.replace('ATTEMPT', attempt).splitlines(keepends=True))
        for index, code in enumerate((bootstrap, execute))
    ]
    notebook = dict(
        nbformat=4, nbformat_minor=5, metadata={
            'kernelspec':
            dict(display_name='Python 3', language='python', name='python3')
        }, cells=cells)
    for cell in cells:
        compile(''.join(cell['source']), '<graph-notebook>', 'exec')
    Path(destination).write_text(json.dumps(notebook, indent=2) + '\n')
    return dict(
        path=Path(destination).name, attempt=attempt,
        source=dict(commit=source_commit,
                    modules=module_hashes), candidate=candidate['candidate'],
        fixture_sha256=fixture_manifest['sha256'], test_helper_sha256={
            name: hashlib.sha256(files[name].encode()).hexdigest()
            for name in ['graph_fixture.py', 'wheel_gpu_smoke.py']
        }, sha256=hashlib.sha256(Path(destination).read_bytes()).hexdigest())


def main():
    import argparse
    import subprocess
    import tempfile
    from uuid import uuid4

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--source-ref', default='HEAD')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--key-file', type=Path)
    parser.add_argument('--ui-credentials', type=Path)
    parser.add_argument('--fixture-archive', type=Path)
    parser.add_argument('--pdf-wheel', type=Path)
    parser.add_argument(
        '--verify-notebook', type=Path,
        help='Only inspect outputs of an existing graph notebook')
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[4]
    commit = subprocess.check_output(
        ['git', 'rev-parse', '--verify', args.source_ref + '^{commit}'],
        cwd=repository, text=True).strip()
    hashes = {
        name:
        hashlib.sha256(
            subprocess.check_output([
                'git', 'show',
                commit + ':packages/kumo-engine/kumo/engine/' + name
            ], cwd=repository)).hexdigest()
        for name in MODULES
    }
    root = args.output_dir or Path(
        tempfile.mkdtemp(prefix='kumo-snowflake-graph-'))
    if args.output_dir:
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
    if not args.execute:
        plan = dict(
            status='planned', training_submitted=False,
            source=dict(commit=commit, modules=hashes),
            resources=dict(pool='KUMO_TEST_POOL', warehouse='KUMO_TEST_WH',
                           database='KUMO_TEST_DB',
                           schema='KUMO_TEST'), fixture=json.loads(
                               Path(__file__).parent.parent.joinpath(
                                   'pr172/inputs.json').read_text()))
        (root / 'summary.json').write_text(json.dumps(plan, indent=2) + '\n')
        print('Plan only; no credentials, downloads or cloud execution:',
              root / 'summary.json')
        return 0
    try:
        if args.verify_notebook:
            config = load_config(args.verify_notebook)
            config['action'] = 'verify'
            import shutil
            notebook = root / Path(config['path']).name
            shutil.copyfile(config['path'], notebook)
            config['path'] = notebook.name
        else:
            admit_candidate(commit)
            from graph_assets import prepare as assets
            staged = assets(root / 'assets', args.key_file,
                            args.fixture_archive, args.pdf_wheel)
            config = prepare(
                root / ('kumo-test-graph-' + uuid4().hex[:12] + '.ipynb'),
                staged['pdf'], commit, hashes, staged['fixture'])
        configuration = root / 'notebook.json'
        configuration.write_text(json.dumps(config, indent=2) + '\n')
        import ui
        report = ui.run(output=root / 'ui', execute=True,
                        credential_file=args.ui_credentials,
                        notebook=configuration, key_file=args.key_file)
    except Exception as error:
        report = dict(
            status='failed' if isinstance(error,
                                          (ValueError,
                                           AssertionError)) else 'blocked',
            phase='preparation', training_submitted=False,
            reason=type(error).__name__ + ': ' + str(error))
        print(report['reason'], flush=True)
    (root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Graph result:', report['status'], 'Summary:', root / 'summary.json',
          flush=True)
    return report['status'] != 'passed'


if __name__ == '__main__':
    raise SystemExit(main())
