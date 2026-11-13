# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UI mode isolation and receipt binding; live qualification requires Snowsight."""
import hashlib
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.fixture
def receipt(tmp_path):
    folder = tmp_path / 'lifecycle/download'
    folder.mkdir(parents=True)
    manifest = dict(
        wheel_sha256='a' * 64, artifacts={}, logical_files={},
        identity=dict(account='TEST_LOCATOR', database='KUMO_TEST_DB',
                      schema='KUMO_TEST'))
    model = 'KUMO_TEST_MLOPS_AAAAAAAAAAAA'
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


def test_tampered_or_missing_source_cannot_qualify_ui(modules, receipt):
    _, ui = modules
    source, folder, manifest, _ = receipt
    name = next(name for name, logical in manifest['logical_files'].items()
                if logical == 'lifecycle/fit.json')
    (folder / name).write_text('{}')
    with pytest.raises(ValueError, match='checksum mismatch'):
        ui.expectations(source)


def test_credentials_with_public_permissions_are_blocked(
        modules, tmp_path, monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
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


@pytest.fixture
def graph_bundle(monkeypatch, tmp_path):
    notebook, _ = graph_modules(monkeypatch)
    import resources
    monkeypatch.setenv(
        'SNOWFLAKE_TEST_CANDIDATE',
        str(Path(__file__).parent / 'data/snowflake-candidate.json'))
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG',
                       json.dumps(resources.expected()))
    assets = json.loads((DIRECTORY / 'graph-assets.json').read_text())
    root = tmp_path / 'original'
    root.mkdir()
    config = notebook.prepare(root / 'kumo-test-graph-0123456789ab.ipynb',
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
    value = dict(account_url='https://app.snowflake.com/account/test_account/',
                 user='KUMO_SNOWFLAKE_UI_TEST', password='synthetic-password',
                 totp_secret='GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ')
    monkeypatch.setenv('SNOWFLAKE_UI_CREDENTIALS', json.dumps(value))
    import resources
    monkeypatch.setenv('SNOWFLAKE_TEST_CONFIG',
                       json.dumps(resources.expected()))
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
    'mode', [['--execute'], ['--ui'], ['--show-result', '/tmp/result']])
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


@pytest.mark.parametrize('action', ['verify', 'unknown'])
def test_failed_read_only_inspection_never_stops_a_kernel(action):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is unavailable')
    program = """
const assert = require('node:assert/strict');
const {cleanupCreatedNotebook} = require(process.argv[1]);
const forbidden = () => {throw new Error('Read-only cleanup touched the browser')};
cleanupCreatedNotebook(new Proxy({}, {get:forbidden}), {notebook:{action:process.argv[2]}, notebook_creation_started:true}, forbidden)
  .then(result => assert.equal(result.status, 'not_run'))
  .catch(error => {console.error(error);process.exitCode=1});
"""
    subprocess.run(
        [node, '-e', program,
         str(DIRECTORY / 'notebook_ui.cjs'), action], check=True)


@pytest.mark.parametrize('phase', ['before', 'after'])
def test_capacity_warning_cannot_pass_as_connected(phase):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js is unavailable')
    program = """
const assert=require('node:assert/strict');
const {connectionOutcome}=require(process.argv[1]);
let checks=0;
const memory={isVisible:async()=>process.argv[2]==='before'||++checks>1,
              waitFor:()=>new Promise(()=>{})};
const frame={getByRole:(role,options)=>{
  if(options.name==='High memory usage')return memory;
  if(options.name==='Connection status')return{filter:()=>({waitFor:async()=>{}})};
  if(role==='alertdialog')return{waitFor:()=>new Promise(()=>{})};
  throw new Error('Unexpected UI action or forced connection');
}};
connectionOutcome(frame,1000).then(()=>{throw new Error('Memory warning incorrectly passed')})
 .catch(error=>{assert.equal(error.name,'NotebookBlocked');assert.match(error.message,/80% memory/)});
"""
    subprocess.run(
        [node, '-e', program,
         str(DIRECTORY / 'notebook_ui.cjs'), phase], check=True)


def test_browser_evidence_requires_current_test_identity_without_auth_fields():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = r"""
const assert = require('node:assert/strict');
const {captureRoute,captureAllowed,safeError}=require(process.argv[1]);
const account='https://app.snowflake.com/org/test/';
const config={auth:{account_url:account,user:'TEST_USER'},expected:{model:'TEST_MODEL',runs:[]}};
const clean=account+'#/experiments/TEST_MODEL';
for(const url of [account+'?session_token=private','https://app.snowflake.com/org/test_other/',account+'#/login',account+'#/auth',account+'#/%6cogin',account+'#/experiments?code=private','https://app.snowflake.com/org/other/','https://example.com/org/test/','http://app.snowflake.com/org/test/','invalid'])
 assert.equal(captureRoute(url,account),false,url);
assert.equal(captureRoute(clean,account),true);
const prepared={attempt:'abcdef012345',path:'/tmp/kumo-test-graph-abcdef012345.ipynb'};
for(const cell of ['abcdef012345-0','abcdef012345-1']){
 assert.equal(captureRoute(account+'#/workspaces/test?cell='+cell,account,prepared),true);
 assert.equal(captureRoute(account+'#/workspaces/test?cell='+cell,account),false);
}
for(const suffix of ['?cell=abcdef012345-2','?cell=aaaaaaaaaaaa-0','?cell=abcdef012345-0&token=private','?cell=abcdef012345-0&cell=abcdef012345-1','?cell=abcdef012345-0?token=private'])
 assert.equal(captureRoute(account+'#/workspaces/test'+suffix,account,prepared),false);
assert.equal(captureRoute(account+'#/workspaces/%3Ftoken?cell=abcdef012345-0',account,prepared),false);
assert.equal(captureRoute(account+'#/workspaces/test?cell=abcdef012345-0',account,{...prepared,path:'foreign.ipynb'}),false);
for (const cell of ['0','123','9ed66216-576e-4d7e-8dc4-3901a82af5a4'])
 assert.equal(captureRoute(account+'#/workspaces/test?cell='+cell,account),true);
for (const suffix of ['?cell=session_token','?cell=0&session_token=private','?cell=0&cell=1','?token=private','?cell=','?cell=0?token=private'])
 assert.equal(captureRoute(account+'#/workspaces/test'+suffix,account),false);
assert.equal(captureRoute(account+'#/login?cell=0',account),false);
assert.equal(captureRoute(account+'#/workspaces/%3Ftoken?cell=0',account),false);
assert.equal(captureRoute('https://app.snowflake.com/_deeplink/#/experiments/TEST_MODEL',account),true);
const state={url:clean,menu:true,password:false,otp:false,identity:true,heading:'TEST_MODEL'};
const page={url:()=>state.url,frames:()=>[page],
 locator:()=>({count:async()=>state.password?1:0}),
 getByRole:(role,options)=>({first(){return this},isVisible:async()=>role==='button'?state.menu:role==='textbox'?state.otp:state.identity&&options.name===state.heading})};
(async()=>{
 assert.equal(await captureAllowed(page,config,true),true);
 state.heading='Models / TEST_MODEL';assert.equal(await captureAllowed(page,config,true),true);
 state.heading='Models / OTHER_MODEL';assert.equal(await captureAllowed(page,config,true),false);
 assert.equal(await captureAllowed(page,config,true,false),true);
 state.heading='TEST_MODEL';
 assert.equal(await captureAllowed(page,config,false),false);
 assert.equal(await captureAllowed(page,config,false,false),false);
 for(const key of ['password','otp']){
  state[key]=true;assert.equal(await captureAllowed(page,config,true),false,key);
  assert.equal(await captureAllowed(page,config,true,false),false,key);state[key]=false;
 }
 for(const key of ['menu','identity']){
  state[key]=false;assert.equal(await captureAllowed(page,config,true),false,key);state[key]=true;
 }
 state.url=account+'#/login';assert.equal(await captureAllowed(page,config,true),false);
 const malicious=new Error('https://app.snowflake.com/login?token=private session-value');
 malicious.name='UntrustedPrivateError';
 const result=JSON.stringify(safeError(malicious));
 for(const secret of ['private','session-value','UntrustedPrivateError'])assert.ok(!result.includes(secret));
})().catch(error=>{console.error(error);process.exitCode=1});
"""
    subprocess.run([node, '-e', program, str(DIRECTORY / 'ui.cjs')], check=True)


def test_session_expiry_does_not_save_login_pages_or_browser_errors(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    module = tmp_path / 'fake-playwright'
    module.mkdir()
    (module / 'package.json').write_text('{"version":"test"}')
    (module / 'index.js').write_text(r'''
let navigations=0;
const locator={fill:async()=>{},click:async()=>{},waitFor:async()=>{},first(){return this},isVisible:async()=>false};
const page={
 setDefaultTimeout(){},setDefaultNavigationTimeout(){},waitForTimeout:async()=>{},waitForURL:async()=>{},
 locator:()=>locator,getByRole:()=>locator,
 getByText:()=>({...locator,waitFor:async()=>{throw Error('not rejected')}}),
 goto:async()=>{if(++navigations===2)throw Error('redirect https://app.snowflake.com/login?token=UNEXPECTED_REDIRECT_CREDENTIAL')},
 url:()=> 'https://app.snowflake.com/org/test/#/login',
 screenshot:async()=>{throw Error('Authentication screenshot must not be attempted')},
 frames:()=>{throw Error('Authentication frame dump must not be attempted')}
};
exports.chromium={launch:async()=>({version:()=> 'test',newContext:async()=>({newPage:async()=>page}),close:async()=>{}})};
''')
    output = tmp_path / 'evidence'
    output.mkdir()
    program = r'''
const fs=require('node:fs'),assert=require('node:assert/strict');
const {run}=require(process.argv[1]);
run({output:process.argv[2],playwright_module:process.argv[3],timeout:10,
 auth:{account_url:'https://app.snowflake.com/org/test/',user:'TEST_USER',password:'SYNTHETIC_PASSWORD',totp_secret:'GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ'},
 expected:{model:'TEST_MODEL',runs:[{url:'https://app.snowflake.com/org/test/#/experiments/MODEL/runs/RUN'}]}})
 .then(result=>{
  assert.equal(result.authentication,'passed');assert.equal(result.status,'failed');
  for(const file of fs.readdirSync(process.argv[2])){
   assert.ok(!file.endsWith('.png')&&!file.endsWith('.aria.txt'));
   const text=fs.readFileSync(process.argv[2]+'/'+file,'utf8');
   for(const value of ['UNEXPECTED_REDIRECT_CREDENTIAL','SYNTHETIC_PASSWORD','GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ'])assert.ok(!text.includes(value));
  }
 }).catch(error=>{console.error(error);process.exitCode=1});
'''
    subprocess.run([
        node, '-e', program,
        str(DIRECTORY / 'ui.cjs'),
        str(output),
        str(module)
    ], check=True)


def test_graph_capture_discards_reauthentication_overlays(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = r'''
const fs=require('node:fs'),assert=require('node:assert/strict');
const {captureElement}=require(process.argv[1]);
const output=process.argv[2];
let authenticated=false,overlay=false,shots=0;
const target={
 screenshot:async({path})=>{shots++;fs.writeFileSync(path,overlay?'SYNTHETIC_AUTH_OVERLAY':'safe graph');if(overlay)authenticated=false},
 ariaSnapshot:async()=>overlay?'SYNTHETIC_AUTH_OVERLAY':'safe schema'
};
(async()=>{
 await assert.rejects(captureElement(target,output,'blocked',async()=>authenticated));
 assert.equal(shots,0);
 authenticated=true;overlay=true;
 await assert.rejects(captureElement(target,output,'expired',async()=>authenticated),/Authentication changed/);
 assert.deepEqual(fs.readdirSync(output),[]);
 authenticated=true;overlay=false;
 await captureElement(target,output,'verified',async()=>authenticated);
 assert.equal(fs.readFileSync(output+'/verified.png','utf8'),'safe graph');
 assert.equal(fs.readFileSync(output+'/verified.aria.txt','utf8'),'safe schema');
 // Saved HTML callers use the default guard and have no browser credentials.
 await captureElement(target,output,'offline');
})().catch(error=>{console.error(error);process.exitCode=1});
'''
    subprocess.run(
        [node, '-e', program,
         str(DIRECTORY / 'graph_ui.cjs'),
         str(tmp_path)], check=True)


def test_cleanup_rejects_modified_notebook_before_browser(
        graph_bundle, monkeypatch):
    _, root, config = graph_bundle
    import cleanup
    (root / config['path']).write_text('{}')
    monkeypatch.setattr(cleanup.ui, 'run',
                        lambda **kw: pytest.fail('browser started'))
    with pytest.raises(ValueError, match='checksum/identity'):
        cleanup.cleanup(root, execute=True)


def test_cleanup_preserves_stage_without_confirmed_kernel_cleanup(
        graph_bundle, monkeypatch):
    _, root, _ = graph_bundle
    import cleanup
    spec = json.loads((root / 'notebook.json').read_text())
    (root / 'ui').mkdir()
    (root / 'ui/notebook-creation-started.json').write_text(
        json.dumps(dict(attempt=spec['attempt'], sha256=spec['sha256'])))
    monkeypatch.setattr(cleanup.ui, 'run', lambda **kw: dict(status='failed'))
    monkeypatch.setattr(cleanup.launch, 'connect',
                        lambda *a: pytest.fail('stage cleanup started'))
    with pytest.raises(RuntimeError, match='preserving stage'):
        cleanup.cleanup(root, execute=True)


def test_cleanup_read_only_and_missing_attempts_do_not_connect(
        graph_bundle, monkeypatch, tmp_path):
    _, root, config = graph_bundle
    import cleanup
    monkeypatch.setattr(cleanup.ui, 'run',
                        lambda **kw: pytest.fail('browser started'))
    assert cleanup.cleanup(tmp_path / 'absent',
                           execute=True)['status'] == 'not_run'
    assert cleanup.cleanup(root)['status'] == 'planned'
    config['action'] = 'verify'
    (root / 'notebook.json').write_text(json.dumps(config))
    assert cleanup.cleanup(root, execute=True)['status'] == 'not_run'


def test_cleanup_does_not_claim_attempt_without_durable_creation_marker(
        graph_bundle, monkeypatch):
    _, root, _ = graph_bundle
    import cleanup
    monkeypatch.setattr(cleanup.ui, 'run',
                        lambda **kw: pytest.fail('browser started'))
    monkeypatch.setattr(cleanup.launch, 'connect',
                        lambda *a: pytest.fail('stage removed'))
    assert cleanup.cleanup(root, execute=True)['status'] == 'not_run'


def test_cleanup_uses_confirmed_original_deletion_before_stage_remove(
        graph_bundle, monkeypatch):
    _, root, spec = graph_bundle
    from contextlib import nullcontext

    import cleanup
    (root / 'ui').mkdir()
    (root / 'ui/notebook-creation-started.json').write_text(
        json.dumps(dict(attempt=spec['attempt'], sha256=spec['sha256'])))
    (root / 'ui/browser-result.json').write_text(
        json.dumps(
            dict(notebook_cleanup=dict(status='passed'),
                 exports=dict(status='passed'))))
    monkeypatch.setattr(cleanup.ui, 'run',
                        lambda **kw: pytest.fail('browser started'))
    commands = []
    cursor = SimpleNamespace(execute=commands.append)
    connection = SimpleNamespace(cursor=lambda: nullcontext(cursor))
    monkeypatch.setattr(cleanup.launch, 'connect',
                        lambda *a: nullcontext(connection))
    assert cleanup.cleanup(root, execute=True)['status'] == 'passed'
    assert commands == [
        'REMOVE ' + cleanup.launch.STAGE + '/mlops-validation/graph-notebook/' +
        spec['attempt'] + '/'
    ]


def test_delayed_workspace_tree_cannot_report_notebook_absent():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = '''const {cleanupCreatedNotebook}=require(process.argv[1]);
const locator = {isVisible:async()=>false,click:async()=>{},waitFor:async()=>{},first(){return this;}};
const file = {count:async()=>0,waitFor:async()=>{throw Error('tree not loaded');}};
const frame = {getByRole:(role)=>role==='treeitem'?file:locator};
const page = {getByRole:(role)=>role==='link'?{...locator,isVisible:async()=>true}:locator,
 frameLocator:()=>frame,addLocatorHandler:async()=>{}};
const config={notebook:{action:'cleanup',path:'kumo-test-graph-aaaaaaaaaaaa.ipynb'},notebook_creation_started:true};
cleanupCreatedNotebook(page,config,()=>{}).then(()=>{throw Error('False absence accepted');},e=>{if(e.message!=='tree not loaded')throw e;});'''
    subprocess.run([node, '-e', program,
                    str(DIRECTORY / 'notebook_ui.cjs')], check=True)


@pytest.mark.parametrize(
    'active,persists,wrong_service,expected_stop,expected_delete', [
        (False, False, False, False, True),
        (True, False, False, True, True),
        (True, True, False, True, False),
        (True, False, True, False, False),
    ])
def test_cleanup_checks_kernel_action_even_when_connection_says_connect(
        active, persists, wrong_service, expected_stop, expected_delete):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = r'''const {cleanupCreatedNotebook}=require(process.argv[1]);
const options=JSON.parse(process.argv[2]);
let active=options.active, stopped=false, deleted=false, loaded=false, reads=0;
const base={async waitFor(){},async click(){},async isVisible(){return false;},
  first(){return this;},filter(){return this;},async press(){}};
const connection={...base,async innerText(){return 'Connect';}};
const shutdown={...base,async isEnabled(){if(!loaded)throw Error('Unloaded kernel state');reads++;return active;},
  async click(){stopped=true;active=options.persists;}};
const panel={...base,async innerText(){return 'Compute pool: KUMO_TEST_POOL (GEN_X64_G2_8) Connected service '+(options.wrong_service?'FOREIGN_SERVICE':'KUMO_TEST_NOTEBOOK_SERVICE');},
  getByText(text){return text==='Loading...'?{async waitFor(){loaded=true;}}:base;},
  getByRole(){return shutdown;}};
const deletion={...base,getByRole(){return {...base,async click(){
  if(active||reads<2)throw Error('Deletion before terminal kernel proof');deleted=true;}};}};
const file={...base,async dblclick(){}};
const dropdown={...base,async click(){loaded=false;}};
const frame={getByRole(role,opts={}){
 if(role==='treeitem')return file;
 if(opts.name==='Connection status')return connection;
 if(opts.name==='Service List dropdown button')return dropdown;
 if(opts.name==='Notebook service management')return panel;
 if(role==='dialog'&&!opts.name)return {filter(){return deletion;}};
 if(role==='dialog'&&opts.name instanceof RegExp)return base;
 if(role==='button'&&String(opts.name).includes('warehouse size'))return {...base,async isVisible(){return true;}};
 return base;
}};
const page={getByRole(role){return role==='link'?{...base,async isVisible(){return true;}}:base;},
 frameLocator(){return frame;},async addLocatorHandler(){}};
const config={notebook:{action:'cleanup',path:'kumo-test-graph-aaaaaaaaaaaa.ipynb'},
 notebook_creation_started:true,target:{notebook_service:'KUMO_TEST_NOTEBOOK_SERVICE'}};
(async()=>{
 let rejected=false;
 try{await cleanupCreatedNotebook(page,config,()=>{});}catch{rejected=true;}
 if(stopped!==options.expected_stop||deleted!==options.expected_delete||rejected===options.expected_delete)
   throw Error(JSON.stringify({stopped,deleted,rejected,reads}));
})();'''
    subprocess.run([
        node, '-e', program,
        str(DIRECTORY / 'notebook_ui.cjs'),
        json.dumps(
            dict(active=active, persists=persists, wrong_service=wrong_service,
                 expected_stop=expected_stop, expected_delete=expected_delete))
    ], check=True)


def test_public_diagnostics_preserve_primary_failure_without_private_values():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = r'''const assert=require('node:assert/strict');
const {failureDiagnostics,retainFailure}=require(process.argv[1]);
const error=new Error('PRIVATE account token=SECRET');error.name='TimeoutError';
const report={};
retainFailure(report,'original safe failure',error,'notebook_visualization','dedicated_notebook_service_selected');
retainFailure(report,'later cleanup failure',new Error('SECRET'),'notebook_cleanup','attempt_notebook_deleted');
assert.equal(report.reason,'original safe failure');
assert.deepEqual({failure_code:report.failure_code,failure_phase:report.failure_phase,failure_step:report.failure_step},
 {failure_code:'timeout',failure_phase:'notebook_visualization',failure_step:'dedicated_notebook_service_selected'});
const unknown=failureDiagnostics(new Error('SECRET'),'https://private/?token=SECRET','PRIVATE');
assert.deepEqual(unknown,{failure_code:'browser_error'});
assert.deepEqual(failureDiagnostics(error,'run_details:PRIVATE','run_verified'),
 {failure_code:'timeout',failure_phase:'run_details',failure_step:'run_verified'});
for(const value of [JSON.stringify(report),JSON.stringify(unknown)])
 for(const secret of ['PRIVATE','SECRET','token='])assert.ok(!value.includes(secret));'''
    subprocess.run([node, '-e', program, str(DIRECTORY / 'ui.cjs')], check=True)


@pytest.mark.parametrize('scenario,expected_status,node_calls,removed', [
    ('cleanup_only', 'passed', 2, True),
    ('verification_failure', 'failed', 2, True),
    ('collection_failure', 'failed', 2, False),
    ('browser_close_failure', 'failed', 1, False),
    ('fresh_cleanup_failure', 'failed', 2, False),
    ('verify_action', 'failed', 1, False),
    ('missing_marker', 'failed', 1, False),
    ('wrong_marker', 'failed', 1, False),
    ('unexpected_exit', 'failed', 1, False),
    ('missing_receipt', 'failed', 2, False),
    ('unknown_verification', 'failed', 2, True),
])
def test_fresh_cleanup_is_bounded_and_cannot_hide_verification_failure(
        graph_bundle, monkeypatch, tmp_path, scenario, expected_status,
        node_calls, removed):
    from contextlib import nullcontext

    _, prepared, spec = graph_bundle
    import graph_notebook
    import launch
    import resources
    from cleanup import cleanup

    import ui

    configuration = prepared / 'notebook.json'
    if scenario == 'verify_action':
        spec['action'] = 'verify'
        configuration.write_text(json.dumps(spec))
    target = resources.require()
    monkeypatch.setattr(
        ui, 'credentials', lambda path: dict(account_url=target['account_url'],
                                             user=target['browser_user']))
    monkeypatch.setattr(ui.shutil, 'which', lambda name: 'node')
    monkeypatch.setenv('KUMO_PLAYWRIGHT_MODULE', '/synthetic/playwright')
    monkeypatch.setattr(ui.subprocess, 'check_output',
                        lambda *a, **kw: 'a' * 40)
    commands = []
    cursor = SimpleNamespace(execute=commands.append, fetchall=lambda: [])
    connection = SimpleNamespace(cursor=lambda: nullcontext(cursor))
    monkeypatch.setattr(launch, 'connect', lambda key: nullcontext(connection))

    def collect(*args):
        if scenario == 'collection_failure':
            raise ValueError('Synthetic export checksum failure')
        return {'status': 'passed'}

    monkeypatch.setattr(graph_notebook, 'collect', collect)
    calls = []

    def browser(command, **kwargs):
        config = json.loads(kwargs['input'])
        calls.append(config['notebook']['action'])
        output = Path(config['output'])
        if config['notebook']['action'] == 'cleanup':
            failed = scenario == 'fresh_cleanup_failure'
            result = dict(
                status='failed' if failed else 'passed',
                verification_status='failed' if failed else 'passed',
                browser_cleanup={'status': 'passed'},
                notebook_cleanup={'status': 'failed' if failed else 'passed'},
                notebook_creation_started=False)
            returncode = 1 if failed else 0
        else:
            result = dict(
                status='failed', verification_status='passed',
                notebook_creation_started=True,
                browser_cleanup={'status': 'passed'}, notebook_cleanup={
                    'status': 'failed',
                    'type': 'TimeoutError'
                }, reason='Initial cleanup failed', failure_code='timeout',
                failure_phase='notebook_cleanup',
                failure_step='workspace_opened')
            if scenario == 'verification_failure':
                result.update(verification_status='failed',
                              reason='PRIMARY ASSERTION',
                              failure_phase='notebook_visualization')
            elif scenario == 'browser_close_failure':
                result['browser_cleanup']['status'] = 'failed'
            elif scenario == 'unknown_verification':
                result.pop('verification_status')
            marker = dict(attempt=spec['attempt'], sha256=spec['sha256'])
            if scenario == 'wrong_marker':
                marker['attempt'] = 'f' * 12
            if scenario != 'missing_marker':
                (output / 'notebook-creation-started.json').write_text(
                    json.dumps(marker))
            if scenario != 'missing_receipt':
                (output / 'notebook-receipt.json').write_text('{}')
            returncode = 2 if scenario == 'unexpected_exit' else 1
        (output / 'browser-result.json').write_text(json.dumps(result))
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(ui.subprocess, 'run', browser)
    output = tmp_path / 'separate-ui-output'
    result = ui.run(output=output, notebook=configuration, execute=True)
    assert result['status'] == expected_status
    assert len(calls) == node_calls
    assert all(action == 'cleanup' for action in calls[1:])
    assert any(command.startswith('REMOVE ') for command in commands) is removed
    assert json.loads(
        (output / 'browser-result.json').read_text())['status'] == 'failed'
    if scenario == 'verification_failure':
        assert result['reason'] == 'PRIMARY ASSERTION'
    if expected_status == 'passed':
        assert result['initial_cleanup_failure'][
            'reason'] == 'Initial cleanup failed'
        proof = Path(result['cleanup_retry']['evidence'])
        assert json.loads(
            proof.read_text())['browser_cleanup']['status'] == 'passed'
        # The always() step must consume final cleanup proof, not reopen a deleted file.
        cleanup(prepared, notebook=configuration, ui_output=output,
                execute=True)
        assert len(calls) == node_calls
    elif scenario == 'collection_failure':
        assert 'collection_error' in result
        retained = cleanup(prepared, notebook=configuration, ui_output=output,
                           execute=True)
        assert retained['stage_cleanup']['status'] == 'blocked'
        assert len(calls) == node_calls
        assert not any(command.startswith('REMOVE ') for command in commands)


@pytest.mark.parametrize('eventually_loaded', [True, False])
def test_saved_graph_images_retry_decode_within_deadline(
        eventually_loaded, tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = r'''
const assert = require('node:assert/strict');
const {checkSaved} = require(process.argv[1]);
const loaded = process.argv[3] === 'true';
let clock = 0, waits = 0;
Date.now = () => clock;
const details = {tables:['test'],rows:[{table:'test',column:'id',tip:''}],edges:[]};
const image = {isVisible:async()=>true,
 evaluate:async()=>loaded && waits >= 2 ? details : null,
 screenshot:async()=>{}};
const frame = {locator:()=>({count:async()=>0}),
 getByRole:()=>({count:async()=>2,nth:()=>image})};
const page = {frames:()=>[frame],waitForTimeout:async(ms)=>{waits++;clock+=ms}};
const receipt = {schema:{tables:{test:{cols:[{name:'id'}]}},edges:[]}};
(async()=>{
 if(loaded){
  const result=await checkSaved(page,receipt,process.argv[2]);
  assert.equal(result.mode,'static_fallback');
  assert.equal(result.checks.length,2);
  assert.equal(waits,2);
 } else {
  await assert.rejects(checkSaved(page,receipt,process.argv[2]),
   /Saved graph outputs are missing/);
  assert.equal(clock,60000);
 }
})().catch(error=>{console.error(error);process.exitCode=1});'''
    subprocess.run([
        node, '-e', program,
        str(DIRECTORY / 'graph_ui.cjs'),
        str(tmp_path),
        json.dumps(eventually_loaded)
    ], check=True)


def test_browser_assertions_preserve_trusted_details_only():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required for the browser helper')
    program = r'''
const assert=require('node:assert/strict');
const path=require('node:path');
const {safeError,failureDiagnostics}=require(process.argv[1]);
const {check}=require(path.join(path.dirname(process.argv[1]),'graph_ui.cjs'));
(async()=>{
 try {await check(null,{status:'failed'},'unused');assert.fail('assertion accepted');}
 catch(error){
  assert.deepEqual(safeError(error),{type:'VerificationError',detail:'Kernel graph checks did not pass in a Workspace'});
  assert.equal(failureDiagnostics(error,'notebook_visualization').failure_code,'assertion_failed');
 }
 const row={count:async()=>0};
 const svg={locator:selector=>selector==='.kumo-card-name'?{filter:()=>({waitFor:async()=>{}})}:row};
 const region={waitFor:async()=>{},boundingBox:async()=>({width:200,height:200}),locator:()=>svg};
 const graph={getByRole:()=>region};
 const frame={frameElement:async()=>({evaluate:async()=>{}}),locator:()=>({count:async()=>2,nth:()=>graph})};
 const receipt={status:'passed',runtime:{workspace:true},default_reads:{backend_accesses:0,sample_calls:0},
 opt_in_reads:{sample_calls:1},canaries:Array(6).fill('x'),schema:{tables:{PRIVATE:{cols:[{name:'SECRET'}]}}}};
 try{await check({frames:()=>[frame]},receipt,'unused');assert.fail('schema fault accepted');}
 catch(error){
  assert.equal(safeError(error).detail,'Missing visible schema row');
  assert.ok(!JSON.stringify(safeError(error)).includes('PRIVATE'));
  assert.ok(!JSON.stringify(safeError(error)).includes('SECRET'));
 }
 const foreign=new Error('private-password');foreign.name='VerificationError';
 assert.ok(!JSON.stringify(safeError(foreign)).includes('private-password'));
})().catch(error=>{console.error(error);process.exitCode=1});'''
    subprocess.run([node, '-e', program, str(DIRECTORY / 'ui.cjs')], check=True)


@pytest.mark.parametrize('repr_fails', [False, True])
def test_graph_assertion_failure_retains_partial_report(monkeypatch, tmp_path,
                                                        repr_fails):
    import sys
    from types import ModuleType

    _, fixture = graph_modules(monkeypatch)
    preview = ModuleType('kumo_core.sources.preview')
    preview.sample_dataframe = lambda value: value
    monkeypatch.setitem(sys.modules, 'kumo_core.sources.preview', preview)
    monkeypatch.setitem(sys.modules, 'pypdf', SimpleNamespace(PdfReader=None))

    class Connector:
        @property
        def source_backend(self):
            return backend

    class Backend:
        def get_sample(self):
            raise AssertionError('No data read expected')

    backend = Backend()
    connector = Connector()

    def render_html():
        if repr_fails:
            raise RuntimeError('PRIVATE render failure')
        return '<div>partial graph</div>'

    graph = SimpleNamespace(
        tables={'test': SimpleNamespace(connector=connector)},
        visualize=lambda **kw: SimpleNamespace(_repr_html_=render_html))
    monkeypatch.setitem(
        sys.modules, 'wheel_gpu_smoke',
        SimpleNamespace(build_graph=lambda *a, **kw: (graph, None)))
    engine = SimpleNamespace(FileConnector=Connector)
    monkeypatch.setitem(sys.modules, 'kumo', SimpleNamespace(engine=engine))
    monkeypatch.setitem(sys.modules, 'kumo.engine', engine)
    (tmp_path / 'graphs.json').write_text(
        json.dumps({'adtech': {
            'tables': {},
            'edges': []
        }}))

    def failed_display(widget):
        raise AssertionError('PRIVATE sample value')

    output = tmp_path / 'outputs'
    with pytest.raises(AssertionError, match='PRIVATE sample value'):
        fixture.exercise(output, tmp_path, display=failed_display)
    report = json.loads((output / 'graph-result.json').read_text())
    assert report['status'] == 'failed'
    assert report['failure'] == 'AssertionError'
    assert 'PRIVATE' not in json.dumps(report)
    if repr_fails:
        assert report['output_failures'] == {'default': 'RuntimeError'}
        assert not (output / 'default.html').exists()
    else:
        assert (output /
                'default.html').read_text() == '<div>partial graph</div>'
    assert not (output / 'sampled.html').exists()


def test_graph_serialization_failure_does_not_leave_a_passed_receipt(
        monkeypatch, tmp_path):
    _, fixture = graph_modules(monkeypatch)

    def invalid_html():
        raise ValueError('PRIVATE widget data')

    report = {'status': 'passed'}
    with pytest.raises(ValueError, match='PRIVATE widget data'):
        fixture.save_outputs(tmp_path, report, [
            ('default', SimpleNamespace(_repr_html_=invalid_html)),
            ('sampled', SimpleNamespace(_repr_html_=lambda: '<div>saved</div>'))
        ])
    retained = json.loads((tmp_path / 'graph-result.json').read_text())
    assert retained == {
        'status': 'failed',
        'failure': 'ValueError',
        'output_failures': {
            'default': 'ValueError'
        }
    }
    assert (tmp_path / 'sampled.html').read_text() == '<div>saved</div>'


def test_upload_click_failure_handles_pending_filechooser(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required')
    program = r'''
const assert=require('node:assert/strict');
const {uploadChooser}=require(process.argv[1]);
(async()=>{
 const unhandled=[];
 process.on('unhandledRejection',error=>unhandled.push(error));
 const page={waitForEvent:()=>new Promise((resolve,reject)=>setImmediate(()=>reject(new Error('chooser timed out'))))};
 const failure=new Error('upload menu click failed');
 await assert.rejects(uploadChooser(page,{click:async()=>{throw failure;}}),error=>error===failure);
 await new Promise(resolve=>setTimeout(resolve,25));
 assert.deepEqual(unhandled,[]);
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    subprocess.run([node, '-e', program,
                    str(DIRECTORY / 'notebook_ui.cjs')], check=True)


def test_metric_ui_check_requires_finite_nonempty_metrics():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node.js required')
    program = r'''
const assert=require('node:assert/strict');
const {runMetrics}=require(process.argv[1]);
for(const value of [undefined,{}, {'pred/rows':NaN}])
 assert.throws(()=>runMetrics(value),/finite run metrics/);
assert.deepEqual(runMetrics({'pred/rows':60}),[['pred/rows',60]]);
'''
    subprocess.run([node, '-e', program, str(DIRECTORY / 'ui.cjs')], check=True)


def test_blocked_graph_prerequisite_preserves_blocked_status(
        graph_bundle, monkeypatch, tmp_path):
    from contextlib import nullcontext

    import launch
    import resources

    import ui

    _, prepared, _ = graph_bundle
    target = resources.require()
    monkeypatch.setattr(
        ui, 'credentials', lambda path: dict(account_url=target['account_url'],
                                             user=target['browser_user']))
    monkeypatch.setattr(ui.shutil, 'which', lambda name: 'node')
    monkeypatch.setenv('KUMO_PLAYWRIGHT_MODULE', '/synthetic/playwright')
    monkeypatch.setattr(ui.subprocess, 'check_output', lambda *a, **k: 'a' * 40)
    cursor = SimpleNamespace(execute=lambda command: None, fetchall=lambda: [])
    monkeypatch.setattr(
        launch, 'connect', lambda key: nullcontext(
            SimpleNamespace(cursor=lambda: nullcontext(cursor))))

    def browser(command, **kwargs):
        config = json.loads(kwargs['input'])
        (Path(config['output']) / 'browser-result.json').write_text(
            json.dumps(
                dict(status='blocked', reason='Notebook capacity unavailable',
                     notebook_creation_started=False)))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(ui.subprocess, 'run', browser)
    report = ui.run(notebook=prepared / 'notebook.json',
                    output=tmp_path / 'blocked', execute=True)
    assert report['status'] == 'blocked'
    assert report['reason'] == 'Notebook capacity unavailable'
