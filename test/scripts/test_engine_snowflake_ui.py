# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UI mode isolation and receipt binding; live qualification requires Snowsight."""
import ast
import hashlib
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

DIRECTORY = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/test/snowflake_notebook'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    return importlib.import_module('launch'), importlib.import_module('ui')


@pytest.mark.parametrize('ui_flag', [['--ui'], ['--ui', 'true'], ['--ui=true']])
def test_ui_execute_never_prepares_or_connects_for_training(
        modules, monkeypatch, tmp_path, ui_flag):
    launch, ui = modules
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Notebook or SQL path invoked by UI-only mode')

    monkeypatch.setattr(launch, 'prepare_payload', forbidden)
    monkeypatch.setattr(launch, 'connect', forbidden)
    monkeypatch.setattr(launch, 'run_pipeline', forbidden)
    monkeypatch.setattr(
        ui, 'run',
        lambda **kwargs: calls.append(kwargs) or {'status': 'blocked'})
    assert launch.main(['--execute', *ui_flag, '--ui-run', str(tmp_path)]) == 1
    assert len(calls) == 1
    assert calls[0]['execute'] is True
    assert calls[0]['source'] == tmp_path


@pytest.mark.parametrize(
    'mode', [['--negative-only'], ['--check-history', '/tmp/source'],
             ['--show-result', '/tmp/source']])
def test_ui_rejects_mixed_modes_before_work(modules, mode):
    launch, _ = modules
    with pytest.raises(SystemExit) as error:
        launch.main(['--ui', *mode])
    assert error.value.code == 2


@pytest.fixture
def receipt(tmp_path):
    folder = tmp_path / 'lifecycle/download'
    folder.mkdir(parents=True)
    manifest = dict(
        wheel_sha256='a' * 64, artifacts={}, logical_files={},
        identity=dict(account='SVB14499', database='KUMO_TEST_DB',
                      schema='KUMO_TEST'))
    model = 'EGU_MLOPS_AAAAAAAAAAAA'
    experiment = f'"KUMO_TEST_DB"."KUMO_TEST"."{model}"'
    prefix = f'https://app.snowflake.com/_deeplink/#/experiments/databases/KUMO_TEST_DB/schemas/KUMO_TEST/experiments/{model}/runs/'

    def write(logical, value):
        name = hashlib.sha256(logical.encode()).hexdigest() + '.json'
        data = value.encode() if isinstance(
            value, str) else json.dumps(value).encode()
        (folder / name).write_bytes(data)
        manifest['artifacts'][name] = hashlib.sha256(data).hexdigest()
        manifest['logical_files'][name] = logical
        (folder / 'lifecycle-result.json').write_text(json.dumps(manifest))

    for phase, count in [('fit', 2), ('score', 4), ('evaluate', 4),
                         ('negative', 0)]:
        refs = []
        for index in range(count):
            run = phase.upper() + '_' + str(index)
            refs.append(
                dict(run=run, experiment=experiment,
                     params={'kumo.model_name':
                             model}, metrics={'test/auroc': 0.75},
                     history={'test/auroc': [[0, 0.75]]}))
        result = dict(references=refs)
        if phase == 'fit':
            result.update(
                model=model, versions=[
                    dict(version='V1', bundle='b' * 64),
                    dict(version='V2', bundle='c' * 64)
                ])
        write(
            'lifecycle/' + phase + '.json',
            dict(status='passed', environment={'sha256': 'a' * 64},
                 result=result))
        if phase != 'negative':
            write('lifecycle/' + phase + '.log',
                  '\n'.join(prefix + ref['run'] for ref in refs))
    return tmp_path, folder, manifest, write


def test_ui_expectations_bind_exact_runs_candidate_and_checksums(
        modules, receipt):
    _, ui = modules
    source, _, _, _ = receipt
    result = ui.expectations(source)
    assert len(result['runs']) == 10
    assert result['candidate_sha256'] == 'a' * 64
    assert result['versions'][1]['name'] == 'V2'
    assert len(result['source_checksums']) == 7


def test_tampered_or_missing_source_cannot_qualify_ui(modules, receipt):
    _, ui = modules
    source, folder, manifest, _ = receipt
    name = next(name for name, logical in manifest['logical_files'].items()
                if logical == 'lifecycle/fit.json')
    (folder / name).write_text('{}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        ui.expectations(source)


def test_missing_reference_is_rejected_even_with_updated_checksum(
        modules, receipt):
    _, ui = modules
    source, folder, manifest, write = receipt
    name = next(name for name, logical in manifest['logical_files'].items()
                if logical == 'lifecycle/score.json')
    phase = json.loads((folder / name).read_text())
    phase['result']['references'].pop()
    write('lifecycle/score.json', phase)
    with pytest.raises(ValueError, match='Missing tracking references'):
        ui.expectations(source)


def test_credentials_with_public_permissions_are_blocked(modules, tmp_path):
    _, ui = modules
    file = tmp_path / 'credentials.json'
    file.write_text('{}')
    file.chmod(0o644)
    with pytest.raises(ui.UIBlocked, match='owner-only'):
        ui.credentials(file)


def test_totp_matches_rfc6238_vectors():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = '''const {totp}=require(process.argv[1]);
const vectors=[[59,'94287082'],[1111111109,'07081804'],[1111111111,'14050471'],[1234567890,'89005924'],[2000000000,'69279037'],[20000000000,'65353130']];
for(const [seconds,code] of vectors){if(totp('GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ',seconds*1000,8)!==code)throw Error('RFC vector mismatch');}'''
    subprocess.run([node, '-e', program, str(DIRECTORY / 'ui.cjs')], check=True)


def test_displayed_metric_rounding_rejects_wrong_or_nonfinite_values():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = '''const {metricValue}=require(process.argv[1]);
metricValue('0.55952',0.5595238208770752);
metricValue('191.85858',191.858582);
metricValue('1',1);
for(const [shown,expected] of [['0.56',0.5595238208770752],['NaN',1],['Infinity',1],['',0],['0',NaN]]){
 let rejected=false;try{metricValue(shown,expected);}catch{rejected=true;}
 if(!rejected)throw Error('Incorrect rendered metric accepted');
}'''
    subprocess.run([node, '-e', program, str(DIRECTORY / 'ui.cjs')], check=True)


def graph_modules(monkeypatch):
    monkeypatch.syspath_prepend(str(DIRECTORY))
    return importlib.import_module('graph_notebook'), importlib.import_module(
        'graph_fixture')


@pytest.mark.parametrize('missing', [
    'graph.py', 'table.py', '_widget.py', '_visualization.py',
    '_visualization_viewer.py', '_visualization_pdf.py',
    '_visualization_layout.py'
])
def test_graph_notebook_rejects_incomplete_source_coverage(
        monkeypatch, tmp_path, missing):
    notebook, _ = graph_modules(monkeypatch)
    hashes = {
        name: 'a' * 64
        for name in [
            'graph.py', 'table.py', '_widget.py', '_visualization.py',
            '_visualization_viewer.py', '_visualization_pdf.py',
            '_visualization_layout.py'
        ] if name != missing
    }
    with pytest.raises(ValueError, match='Complete reviewed'):
        notebook.prepare(tmp_path / 'egu-graph-abc.ipynb', {}, 'b' * 40, hashes)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('status', ['ERROR', 'SKIPPED', 'COLLISION'])
def test_graph_evidence_requires_actual_upload(monkeypatch, tmp_path, status):
    _, fixture = graph_modules(monkeypatch)
    (tmp_path / 'graph.svg').write_text('<svg/>')
    session = SimpleNamespace(file=SimpleNamespace(
        put=lambda *a, **kw: [SimpleNamespace(status=status)]))
    with pytest.raises(RuntimeError, match='Evidence upload failed'):
        fixture.retain(session, tmp_path, '@test')


def test_graph_evidence_rejects_missing_exports(monkeypatch, tmp_path):
    _, fixture = graph_modules(monkeypatch)
    (tmp_path / 'graph.svg').write_text('<svg/>')
    session = SimpleNamespace(file=SimpleNamespace(
        put=lambda *a, **kw: [SimpleNamespace(status='UPLOADED')]))
    with pytest.raises(RuntimeError, match='Incomplete graph evidence'):
        fixture.retain(session, tmp_path, '@test')


@pytest.fixture
def graph_bundle(monkeypatch, tmp_path):
    notebook, _ = graph_modules(monkeypatch)
    assets = json.loads((DIRECTORY / 'graph-assets.json').read_text())
    root = tmp_path / 'original'
    root.mkdir()
    config = notebook.prepare(root / 'egu-graph-0123456789ab.ipynb',
                              assets['pdf'], 'a' * 40,
                              {name: 'b' * 64
                               for name in notebook.MODULES}, assets['fixture'])
    (root / 'notebook.json').write_text(json.dumps(config))
    return notebook, root, config


def test_graph_notebook_bundle_is_relocatable_and_checksum_bound(
        graph_bundle, tmp_path):
    notebook, root, config = graph_bundle
    moved = tmp_path / 'downloaded-artifact'
    shutil.move(root, moved)
    actual = notebook.load_config(moved / 'notebook.json')
    assert actual['path'] == str(moved / config['path'])
    (moved / config['path']).write_text('{}')
    with pytest.raises(ValueError, match='checksum/identity'):
        notebook.load_config(moved / 'notebook.json')


def test_generated_graph_notebook_has_valid_cells_and_compiles(graph_bundle):
    _, root, config = graph_bundle
    data = json.loads((root / config['path']).read_text())
    assert len(data['cells']) == 2
    assert len({cell['id'] for cell in data['cells']}) == 2
    for cell in data['cells']:
        compile(''.join(cell['source']), '<graph-notebook>', 'exec')
    nbformat = pytest.importorskip('nbformat')
    nbformat.validate(nbformat.from_dict(data))


@pytest.mark.parametrize('location', [
    '@PRODUCTION.PUBLIC.DATA/file.whl',
    '@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE/../../other/file.whl'
])
def test_graph_assets_cannot_reference_other_stages(monkeypatch, tmp_path,
                                                    location):
    notebook, _ = graph_modules(monkeypatch)
    assets = json.loads((DIRECTORY / 'graph-assets.json').read_text())
    assets['pdf']['location'] = location
    with pytest.raises(ValueError, match='dedicated test stage'):
        notebook.prepare(tmp_path / 'egu-graph-0123456789ab.ipynb',
                         assets['pdf'], 'a' * 40,
                         {name: 'b' * 64
                          for name in notebook.MODULES}, assets['fixture'])
    assert not list(tmp_path.iterdir())


def test_graph_plan_uses_no_credentials_or_cloud(modules, monkeypatch,
                                                 tmp_path):
    _, ui = modules
    import graph_assets
    import graph_notebook

    def forbidden(*args, **kwargs):
        raise AssertionError(
            'Plan-only test contacted the cloud or loaded credentials')

    monkeypatch.setattr(graph_assets, 'prepare', forbidden)
    monkeypatch.setattr(ui, 'credentials', forbidden)
    monkeypatch.setattr(ui, 'run', forbidden)
    monkeypatch.setattr(
        'sys.argv',
        ['graph_notebook.py', '--output-dir',
         str(tmp_path / 'plan')])
    assert graph_notebook.main() == 0
    assert json.loads(
        (tmp_path / 'plan/summary.json').read_text())['status'] == 'planned'


def test_runner_browser_secret_uses_same_validation_as_local_file(
        modules, monkeypatch):
    _, ui = modules
    value = dict(
        account_url='https://app.snowflake.com/lfwgwbp/kumo_us_west_test_5/',
        user='KUMO_SNOWFLAKE_UI_TEST', password='synthetic-password',
        totp_secret='GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ')
    monkeypatch.setenv('SNOWFLAKE_UI_CREDENTIALS', json.dumps(value))
    assert ui.credentials() == value
    value['user'] = 'NON_TEST_USER'
    monkeypatch.setenv('SNOWFLAKE_UI_CREDENTIALS', json.dumps(value))
    with pytest.raises(ui.UIBlocked, match='dedicated browser'):
        ui.credentials()


def test_graph_collection_rejects_wrong_stage_before_authentication(
        monkeypatch, tmp_path):
    notebook, _ = graph_modules(monkeypatch)
    import launch

    def forbidden(*args, **kwargs):
        raise AssertionError(
            'Invalid evidence must be rejected before authentication')

    monkeypatch.setattr(launch, 'connect', forbidden)
    with pytest.raises(ValueError, match='inventory or non-test stage'):
        notebook.collect(
            dict(execution_id='a' * 12,
                 output_stage='@PROD.PUBLIC.STAGE', uploaded={}),
            dict(attempt='b' * 12), tmp_path / 'evidence', None)


def test_compute_wait_is_read_only_and_rejects_another_pool(
        modules, monkeypatch):
    launch, _ = modules
    calls = []
    states = iter(['STARTING', 'ACTIVE'])

    class Cursor:
        description = [(name, )
                       for name in ('name', 'instance_family', 'state')]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, statement):
            calls.append(statement)

        def fetchall(self):
            return [('KUMO_TEST_POOL', 'GEN_X64_G2_8', next(states))]

    class Connection(Cursor):
        def cursor(self):
            return Cursor()

    monkeypatch.setattr(launch, 'connect', lambda key: Connection())
    monkeypatch.setattr(launch.time, 'sleep', lambda seconds: None)
    assert launch.wait_for_compute(None, 30) == dict(pool='KUMO_TEST_POOL',
                                                     state='ACTIVE')
    assert calls == ['DESCRIBE COMPUTE POOL KUMO_TEST_POOL'] * 2
    monkeypatch.setattr(
        Cursor, 'fetchall',
        lambda self: [('SHARED_POOL', 'GEN_X64_G2_8', 'ACTIVE')])
    with pytest.raises(ValueError, match='identity differs'):
        launch.wait_for_compute(None, 30)


@pytest.mark.parametrize(
    'mode', [['--execute'], ['--ui'], ['--check-history', '/tmp/result']])
def test_compute_wait_cannot_launch_execution(modules, mode):
    launch, _ = modules
    with pytest.raises(SystemExit) as error:
        launch.main(['--wait-compute', *mode])
    assert error.value.code == 2


def test_graph_collection_rejects_empty_retained_receipt(monkeypatch, tmp_path):
    from contextlib import nullcontext

    notebook, _ = graph_modules(monkeypatch)
    import launch

    contents = {'graph-result.json': b'{}'}
    for name in ('graph.svg', 'graph.pdf', 'graph.md', 'default.html',
                 'sampled.html'):
        contents[name] = b'retained-file'
    uploaded = {
        name: hashlib.sha256(value).hexdigest()
        for name, value in contents.items()
    }
    stage = ('@KUMO_TEST_DB.KUMO_TEST.TEST_STAGE/mlops-validation/'
             'graph-notebook/' + 'b' * 12 + '/results/' + 'a' * 12)
    receipt = dict(
        status='passed', execution_id='a' * 12, output_stage=stage,
        uploaded=uploaded, exports={
            suffix: dict(sha256=uploaded['graph.' + suffix])
            for suffix in ('svg', 'pdf', 'md')
        })

    def download(cursor, source, target):
        destination = target / source.rsplit('/', 1)[1]
        destination.write_bytes(contents[destination.name])
        return destination

    connection = SimpleNamespace(cursor=lambda: nullcontext(None))
    monkeypatch.setattr(launch, 'connect', lambda key: nullcontext(connection))
    monkeypatch.setattr(launch, 'download_file', download)
    with pytest.raises(ValueError, match='differs from retained receipt'):
        notebook.collect(receipt, dict(attempt='b' * 12), tmp_path / 'exports',
                         None)


@pytest.mark.parametrize('event,branch,repository,mode,graph,engine', [
    ('push', 'egu/mlops-platform-integration', 'NVIDIA/kumo-engine', '', True,
     False),
    ('push', 'main', 'NVIDIA/kumo-engine', '', False, False),
    ('push', 'master', 'NVIDIA/kumo-engine', '', False, False),
    ('push', 'egu/another-branch', 'NVIDIA/kumo-engine', '', False, False),
    ('push', 'egu/mlops-platform-integration', 'kumo-ai/kumo', '', False,
     False),
    ('workflow_dispatch', 'egu/mlops-platform-integration',
     'NVIDIA/kumo-engine', 'graph-ui', True, False),
    ('workflow_dispatch', 'egu/mlops-platform-integration',
     'NVIDIA/kumo-engine', 'engine', False, True),
    ('workflow_dispatch', 'main', 'NVIDIA/kumo-engine', 'graph-ui', False,
     False),
    ('pull_request', 'egu/mlops-platform-integration', 'NVIDIA/kumo-engine',
     'graph-ui', False, False),
])
def test_trial_triggers_isolate_graph_ui_from_training(event, branch,
                                                       repository, mode, graph,
                                                       engine):
    workflow = yaml.safe_load(DIRECTORY.parents[3].joinpath(
        '.github/workflows/kumo-engine-snowflake-notebook.yml').read_text())
    assert workflow[True]['push']['branches'] == [
        'egu/mlops-platform-integration'
    ]
    context = dict(
        github=SimpleNamespace(repository=repository,
                               ref='refs/heads/' + branch, event_name=event),
        inputs=SimpleNamespace(mode=mode))
    for name, expected in [('notebook-ui', graph),
                           ('engine-lifecycle', engine)]:
        job = workflow['jobs'][name]
        condition = job['if'].replace('&&', ' and ').replace('||', ' or ')
        assert eval(condition, {'__builtins__': {}}, context) is expected
        assert job['environment'] == 'snowflake-mlops-test'


def test_graph_execution_acknowledges_before_imports_or_sql(
        graph_bundle, capsys):
    _, root, config = graph_bundle
    cell = json.loads((root / config['path']).read_text())['cells'][1]
    code = ast.parse(''.join(cell['source']))
    namespace = {}
    exec(
        compile(ast.Module(body=code.body[:3], type_ignores=[]),
                '<graph-cell-acknowledgment>', 'exec'), namespace)
    assert capsys.readouterr(
    ).out == 'KUMO_GRAPH_EXECUTION_STARTED=' + namespace['execution'] + '\n'
    assert len(namespace['execution']) == 12
    int(namespace['execution'], 16)


def test_graph_cell_rejects_setup_state_retained_after_restart(
        graph_bundle, capsys):
    _, root, config = graph_bundle
    cell = json.loads((root / config['path']).read_text())['cells'][1]
    code = ast.parse(''.join(cell['source']))
    prefix = compile(ast.Module(body=code.body[:4], type_ignores=[]),
                     '<graph-cell-restart-guard>', 'exec')
    with pytest.raises(AssertionError,
                       match='retained setup state after restart'):
        exec(prefix, dict(setup_execution='previous-kernel'))
    assert capsys.readouterr().out.startswith('KUMO_GRAPH_EXECUTION_STARTED=')
