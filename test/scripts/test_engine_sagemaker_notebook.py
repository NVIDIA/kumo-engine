# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate notebook evidence and the actual SDK launch request without AWS calls."""
import copy
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

HERE = (Path(__file__).resolve().parents[2] /
        'packages/kumo-engine/test/sagemaker_notebook')


@pytest.fixture
def modules(monkeypatch):
    names = ('config', 'evidence', 'client', 'run', 'policies',
             'prepare_access', 'expected_ui')
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(HERE))
    result = SimpleNamespace(
        **{name: importlib.import_module(name)
           for name in names})
    yield result
    for name in names:
        sys.modules.pop(name, None)


@pytest.fixture
def execution(modules):
    spec = dict(attempt='123456abcdef', source_commit='a' * 40,
                wheel_sha256='b' * 64, output_prefix='test/outputs',
                test_sources={'kernel.py': 'c' * 64}, assets={})
    report = dict(spec, status='passed', execution_id='abcdef123456',
                  payload={})
    content = json.dumps(report).encode()
    record = {
        key: spec[key]
        for key in ('attempt', 'source_commit', 'wheel_sha256', 'output_prefix')
    }
    record.update(execution_id=report['execution_id'],
                  files={'result.json': modules.evidence.digest(content)})
    submitted = dict(
        cells=[dict(cell_type='code', source=['print("receipt")'])])
    executed = copy.deepcopy(submitted)
    executed['cells'][0].update(
        execution_count=1, outputs=[
            dict(output_type='stream',
                 text=modules.evidence.MARKER + json.dumps(record) + '\n')
        ])
    return SimpleNamespace(spec=spec, report=report, content=content,
                           record=record, submitted=submitted,
                           executed=executed)


def test_collect_binds_saved_output_to_executed_receipt(modules, execution,
                                                        tmp_path):
    s3 = Mock()
    s3.get_object.return_value = {'Body': io.BytesIO(execution.content)}
    result = modules.evidence.collect(s3, 'test-bucket', execution.executed,
                                      execution.submitted, execution.spec,
                                      tmp_path)
    assert result == execution.report
    s3.get_object.assert_called_once_with(Bucket='test-bucket',
                                          Key='test/outputs/result.json')


@pytest.mark.parametrize('change', [
    'source', 'unexecuted', 'cell-error', 'duplicate', 'attempt', 'unsafe-file'
])
def test_rejects_notebook_evidence_substitution(modules, execution, change):
    cell = execution.executed['cells'][0]
    if change == 'source':
        cell['source'] = ['print("different")']
    elif change == 'unexecuted':
        cell['execution_count'] = None
    elif change == 'cell-error':
        cell['outputs'].append(dict(output_type='error', ename='Error'))
    elif change == 'duplicate':
        cell['outputs'] *= 2
    else:
        if change == 'attempt':
            execution.record['attempt'] = 'another'
        else:
            execution.record['files']['../escape'] = 'a' * 64
        cell['outputs'][0]['text'] = (modules.evidence.MARKER +
                                      json.dumps(execution.record))
    with pytest.raises(ValueError):
        modules.evidence.receipt(execution.executed, execution.submitted,
                                 execution.spec)


def test_rejects_modified_s3_report(modules, execution, tmp_path):
    s3 = Mock()
    s3.get_object.return_value = {'Body': io.BytesIO(b'{}')}
    with pytest.raises(ValueError, match='checksum'):
        modules.evidence.collect(s3, 'bucket', execution.executed,
                                 execution.submitted, execution.spec, tmp_path)


def test_failed_pipeline_retains_job_for_diagnostics(modules, tmp_path):
    client = modules.client.Client.__new__(modules.client.Client)
    client.sm = Mock()
    name = 'kumo-test-notebook-123456abcdef'
    arn = 'arn:aws:sagemaker:us-west-2:926922431314:training-job/' + name
    job = dict(TrainingJobArn=arn, TrainingJobStatus='Failed',
               FailureReason='Notebook initialization failed')
    client.sm.describe_training_job.return_value = job
    execution = Mock()
    execution.describe.return_value = dict(
        PipelineExecutionStatus='Failed',
        PipelineArn='arn:aws:sagemaker:us-west-2:926922431314:pipeline/' + name)
    execution.list_steps.return_value = [
        dict(Metadata=dict(TrainingJob=dict(Arn=arn)))
    ]
    assert client.wait(execution, tmp_path, 1) == job
    assert (tmp_path / 'pipeline-steps.json').is_file()
    execution.stop.assert_not_called()


def test_cleanup_failure_preserves_original_error(modules, tmp_path):
    client = modules.client.Client.__new__(modules.client.Client)
    execution = Mock()
    execution.describe.side_effect = [
        ValueError('original'), RuntimeError('cleanup')
    ]
    with pytest.raises(ValueError, match='original'):
        client.wait(execution, tmp_path, 1)


def test_aliased_payload_is_uploaded_once_then_verified_from_cache(
        modules, tmp_path):
    import base64

    from botocore.exceptions import ClientError

    client = modules.client.Client.__new__(modules.client.Client)
    client.config = modules.config.configuration()
    client.s3 = Mock()
    content = b'actual runtime helper bytes'
    original = tmp_path / 'runtime.py'
    original.write_bytes(content)
    sha = modules.evidence.digest(content)
    checksum = base64.b64encode(bytes.fromhex(sha)).decode()
    head = dict(ContentLength=len(content), ChecksumSHA256=checksum)
    missing = ClientError({'Error': {'Code': '404'}}, 'HeadObject')
    client.s3.head_object.side_effect = [missing, head, head]
    first = client.asset('sagemaker_runtime.py', sha, original)
    second = client.asset('sagemaker_runtime.py', sha)
    assert first == second
    assert first['key'].endswith('/' + sha + '/sagemaker_runtime.py')
    assert client.s3.put_object.call_count == 1
    assert client.s3.put_object.call_args.kwargs['Body'] == content
    client.s3.get_object.assert_not_called()


def test_wrong_bytes_cannot_seed_payload_cache(modules, tmp_path):
    from botocore.exceptions import ClientError

    client = modules.client.Client.__new__(modules.client.Client)
    client.config = modules.config.configuration()
    client.s3 = Mock()
    client.s3.head_object.side_effect = ClientError({'Error': {
        'Code': '404'
    }}, 'HeadObject')
    path = tmp_path / 'runtime.py'
    path.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='differs from pinned input'):
        client.asset('sagemaker_runtime.py', 'a' * 64, path)
    client.s3.put_object.assert_not_called()


def test_candidate_wheel_local_version_is_a_safe_cache_name(modules):
    import base64

    client = modules.client.Client.__new__(modules.client.Client)
    client.config = modules.config.configuration()
    client.s3 = Mock()
    sha = 'a' * 64
    client.s3.head_object.return_value = dict(
        ContentLength=1,
        ChecksumSHA256=base64.b64encode(bytes.fromhex(sha)).decode())
    candidate = json.loads((HERE / 'candidate.json').read_text())['candidate']
    assert '+' in candidate['wheel_file']
    assert client.asset(candidate['wheel_file'], sha)['sha256'] == sha
    client.s3.put_object.assert_not_called()


@pytest.mark.parametrize(
    'name',
    ['..', '../wheel.whl', '/wheel.whl', 'wheel?token=x', 'wheel\n.whl'])
def test_unsafe_cache_names_are_rejected_before_aws(modules, name):
    client = modules.client.Client.__new__(modules.client.Client)
    client.config = modules.config.configuration()
    client.s3 = Mock()
    with pytest.raises(ValueError, match='Invalid content-addressed asset'):
        client.asset(name, 'a' * 64)
    client.s3.head_object.assert_not_called()


def test_sdk_compiles_actual_scoped_notebook_training_job(
        modules, monkeypatch, tmp_path):
    boto3 = pytest.importorskip('boto3')
    pytest.importorskip('sagemaker')
    from sagemaker.workflow.notebook_job_step import NotebookJobStep
    from sagemaker.workflow.pipeline import Pipeline

    upload = Mock()
    monkeypatch.setattr(NotebookJobStep, '_upload_job_files', upload)
    monkeypatch.setattr(Pipeline, 'upsert', Mock())
    monkeypatch.setattr(Pipeline, 'start',
                        Mock(return_value=SimpleNamespace(arn='test')))
    config = modules.config.configuration()
    client = modules.client.Client.__new__(modules.client.Client)
    client.config = config
    client.session = boto3.Session(aws_access_key_id='offline',
                                   aws_secret_access_key='offline',
                                   region_name=config['region'])
    client.sm = client.session.client('sagemaker')
    client.wait = Mock(return_value={})
    name = 'kumo-test-notebook-123456abcdef'
    notebook = tmp_path / 'test.ipynb'
    notebook.write_text('{}')
    init = tmp_path / 'init.sh'
    init.write_text('true\n')
    spec = dict(pipeline_name=name,
                job_prefix=config['root'] + '/notebook-jobs/123456abcdef',
                attempt='123456abcdef', source_commit='a' * 40)
    client.submit(notebook, init, spec, tmp_path, 1)
    definition = json.loads((tmp_path / 'pipeline-definition.json').read_text())
    request = definition['Steps'][0]['Arguments']
    assert request['TrainingJobName'] == name
    assert request['RoleArn'] == config['execution_role']
    assert request['AlgorithmSpecification']['TrainingImage'] == config['image']
    assert request['Environment']['SM_KERNEL_NAME'] == config['kernel']
    assert request['Environment']['SM_INPUT_NOTEBOOK_NAME'] == notebook.name
    assert request['Environment']['SM_JOB_DEF_VERSION'] == '1.0'
    assert request['InputDataConfig'][0]['DataSource']['S3DataSource'][
        'S3Uri'].startswith(f's3://{config["bucket"]}/{spec["job_prefix"]}/')
    assert request['ResourceConfig']['InstanceType'] == 'ml.m5.2xlarge'
    assert request['RetryStrategy']['MaximumRetryAttempts'] == 0
    upload.assert_called_once()


def test_policy_plan_reuses_existing_roles_without_replacing_trust(modules):
    plan = modules.policies.plan()
    assert set(plan['roles']) == {'RUNNER', 'EXECUTION'}
    assert plan['roles']['RUNNER']['arn'].endswith('/gha-kumo-workflow-tests')
    assert plan['roles']['EXECUTION']['arn'].endswith(
        '/kumo-mlops-validation-sagemaker')
    assert all('trust' not in value for value in plan['roles'].values())
    condition = plan['runner_trust_addition']['Condition']['StringEquals']
    assert condition['token.actions.githubusercontent.com:sub'].endswith(
        ':environment:sagemaker-mlops-test')


@pytest.mark.parametrize(
    'case', ['normal', 'already-added', 'wrong-account', 'trust-changed'])
def test_portable_admin_handoff_preserves_trust_and_fails_closed(
        modules, tmp_path, monkeypatch, case):
    if not shutil.which('jq'):
        pytest.skip('Portable admin handoff needs jq')
    bundle = tmp_path / 'portable access bundle'
    monkeypatch.setattr(sys, 'argv',
                        ['prepare_access.py', '--output-dir',
                         str(bundle)])
    modules.prepare_access.main()
    assert bundle.with_suffix('.zip').is_file()
    assert not list(bundle.glob('*.py'))
    original = dict(
        Version='2012-10-17', Statement=[
            dict(Effect='Allow', Principal=dict(Service='existing.example'),
                 Action='sts:AssumeRole')
        ])
    addition = json.loads((bundle / 'runner-trust-addition.json').read_text())
    if case == 'already-added':
        original['Statement'].append(addition)
    (tmp_path / 'trust.json').write_text(json.dumps(original))
    fake_bin = tmp_path / 'fake-bin'
    fake_bin.mkdir()
    stub = fake_bin / 'aws'
    stub.write_text('#!' + sys.executable + '\n' + '''import json, os, sys
from pathlib import Path
root = Path(os.environ['HANDOFF_TEST_ROOT'])
case = os.environ['HANDOFF_TEST_CASE']
args = sys.argv[1:]
with (root / 'calls.jsonl').open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
if args[:2] == ['sts', 'get-caller-identity']:
    print('000000000000' if case == 'wrong-account' else '926922431314')
elif args[:2] == ['iam', 'get-role']:
    trust = json.loads((root / 'trust.json').read_text())
    marker = root / 'read-once'
    if case == 'trust-changed' and marker.exists():
        trust['Statement'].append({'Sid': 'ConcurrentChange'})
    marker.touch()
    print(json.dumps(trust))
elif args[:2] == ['iam', 'update-assume-role-policy']:
    filename = args[args.index('--policy-document') + 1]
    (root / 'applied-trust.json').write_bytes(Path(filename.removeprefix('file://')).read_bytes())
elif args[:2] == ['iam', 'put-role-policy']:
    pass
else:
    raise SystemExit('Unexpected AWS operation')
''')
    stub.chmod(0o700)
    environment = dict(os.environ,
                       PATH=str(fake_bin) + os.pathsep + os.environ['PATH'],
                       HANDOFF_TEST_ROOT=str(tmp_path), HANDOFF_TEST_CASE=case)
    result = subprocess.run(['sh', str(bundle / 'apply.sh')], cwd=tmp_path,
                            env=environment, capture_output=True, text=True)
    calls = [
        json.loads(line)
        for line in (tmp_path / 'calls.jsonl').read_text().splitlines()
    ]
    mutations = [
        args for args in calls if args[0] == 'iam' and args[1] != 'get-role'
    ]
    if case in ('wrong-account', 'trust-changed'):
        assert result.returncode != 0
        assert not mutations
    else:
        assert result.returncode == 0, result.stderr
        applied = json.loads((tmp_path / 'applied-trust.json').read_text())
        assert applied['Statement'].count(addition) == 1
        assert applied['Statement'][0] == original['Statement'][0]
        assert [args[1] for args in mutations] == [
            'update-assume-role-policy', 'put-role-policy', 'put-role-policy'
        ]


@pytest.fixture
def lifecycle_receipts(tmp_path):
    counter = iter(range(1, 11))

    def reference():
        number = next(counter)
        return dict(
            run=f'{number:032x}', status='FINISHED', run_name=f'run-{number}',
            experiment_id='1',
            params={'kumo.prediction_id':
                    f'prediction-{number}'}, metrics={'metric': 0.5}, history={
                        'train/loss': [[0, 0.8], [1, 0.6]],
                        'val/auroc': [[0, 0.5], [1, 0.7]]
                    })

    fit = dict(
        model='kumo-test-model', versions=[
            dict(version=str(v), bundle=f'bundle-{v}', reference=reference())
            for v in (1, 2)
        ])

    def prediction():
        record = reference()
        return dict(prediction=record['params']['kumo.prediction_id'],
                    reference=record)

    score = dict(explicit=[prediction() for _ in range(2)],
                 aliases=[prediction() for _ in range(2)])
    evaluate = dict(monitoring=[])
    for alias in score['aliases']:
        record = copy.deepcopy(alias['reference'])
        record['metrics'].update({
            'drift/psi_vs_prev': 0.0,
            'realized/auroc': 0.75
        })
        evaluate['monitoring'].append(
            dict(prediction=alias['prediction'], reference=record,
                 drift_reference=copy.deepcopy(record)))
    phases = dict(fit=fit, score=score, evaluate=evaluate)
    for name, result in phases.items():
        (tmp_path / (name + '.json')).write_text(
            json.dumps(dict(status='passed', returncode=0, result=result)))
    return tmp_path


def test_browser_expectations_include_entire_lifecycle(modules,
                                                       lifecycle_receipts):
    result = modules.expected_ui.expectations(lifecycle_receipts)
    assert [run['roles']
            for run in result['runs']] == [['fit'], ['fit'], ['score'],
                                           ['score'],
                                           ['score', 'drift', 'evaluate'],
                                           ['score', 'drift', 'evaluate']]
    assert len({run['reference']['run'] for run in result['runs']}) == 6
    assert result['runs'][-1]['reference']['metrics']['realized/auroc'] == 0.75


def test_recorded_mlflow_runs_receive_monitoring_updates(modules, tmp_path):
    path = Path(__file__).parent / 'fixtures/mlflow_lifecycle_receipts.json'
    recorded = json.loads(path.read_text())
    for name, phase in recorded['phases'].items():
        (tmp_path / (name + '.json')).write_text(json.dumps(phase))
    result = modules.expected_ui.expectations(tmp_path)
    assert len(result['runs']) == 6
    for run in result['runs'][-2:]:
        assert run['roles'] == ['score', 'drift', 'evaluate']
        assert run['reference']['metrics']['pred/rows'] == 60
        assert run['reference']['metrics']['drift/psi_vs_prev'] == 0
        assert 0 <= run['reference']['metrics']['realized/auroc'] <= 1


@pytest.mark.parametrize('change', ['prediction-lineage', 'lost-score-metric'])
def test_monitoring_cannot_substitute_or_erase_prediction_evidence(
        modules, lifecycle_receipts, change):
    path = lifecycle_receipts / 'evaluate.json'
    phase = json.loads(path.read_text())
    reference = phase['result']['monitoring'][0]['reference']
    if change == 'prediction-lineage':
        reference['params']['kumo.prediction_id'] = 'another-prediction'
    else:
        reference['metrics'].pop('metric')
    path.write_text(json.dumps(phase))
    with pytest.raises(ValueError):
        modules.expected_ui.expectations(lifecycle_receipts)


def test_browser_history_assertions_bind_both_axes_and_numeric_tooltip():
    node = shutil.which('node')
    if not node:
        pytest.skip('Browser assertion regression needs Node.js')
    script = '''
const assert = require('node:assert/strict');
const { numericLabels, historyPoint, targetPosition, observedCoordinates, historyCsv,
  VerificationError, verificationMessage, captureAllowed } = require(process.argv[1]);
assert.deepEqual(numericLabels(['0', ' -0.25 ', '1.5e-3', '.75', 'NaN', '', 'step 1']),
  [0, -0.25, 0.0015, 0.75]);
historyPoint({step: 1, value: 0.75}, 1, 0.75, ['0.75']);
assert.throws(() => historyPoint({step: 1, value: 0.8}, 1, 1, ['1', '0.8']));
assert.throws(() => historyPoint({step: 2, value: 0.75}, 1, 0.75, ['0.75']));
assert.throws(() => historyPoint({step: 1, value: 0.75}, 1, 0.75, ['0.8']));
const geometry = { x: [{value:1,pixel:100},{value:2,pixel:200}],
  y: [{value:0.5,pixel:300},{value:0.75,pixel:200}], origin:{x:151,y:260} };
assert.deepEqual(targetPosition(geometry,1.5,0.6),{x:150,y:260});
historyPoint(observedCoordinates(geometry),1.5,0.6,['0.6']);
assert.throws(() => historyPoint(observedCoordinates({...geometry,origin:{x:154,y:260}}),1.5,0.6,['0.6']));
const csv = 'run_id,key,value,step,timestamp\\nr,train/loss,0.75,1,0\\nr,train/loss,0.6,2,1';
assert.deepEqual(historyCsv(csv,'r','train/loss',[[1,0.75],[2,0.6]]),[[1,0.75],[2,0.6]]);
assert.throws(() => historyCsv(csv,'wrong','train/loss',[[1,0.75],[2,0.6]]));
assert.throws(() => historyCsv(csv,'r','val/auroc',[[1,0.75],[2,0.6]]));
assert.throws(() => historyCsv(csv,'r','train/loss',[[1,0.75]]));
assert.throws(() => historyCsv(csv.replace('0.6,2','0.7,2'),'r','train/loss',[[1,0.75],[2,0.6]]));
assert.throws(() => historyCsv(csv.replace('0.6,2',',2'),'r','train/loss',[[1,0.75],[2,0]]));
assert.equal(verificationMessage(new VerificationError('Metric value mismatch')), 'Metric value mismatch');
assert(!verificationMessage(new Error('https://example.invalid/?token=private')).includes('private'));
const owned = 'https://mlflow.example/#/experiments/1/runs/x';
assert(captureAllowed(owned,owned,true));
assert(!captureAllowed(owned,owned,false));
assert(!captureAllowed('https://login.example/',owned,true));
assert(!captureAllowed('https://mlflow.example/?token=private#/experiments/1/runs/x',owned,true));
assert(!captureAllowed('https://mlflow.example/#/models/x',owned,true));
assert(!captureAllowed('invalid URL',owned,true));
'''
    result = subprocess.run(
        [node, '-e', script, str(HERE / 'assertions.cjs')], capture_output=True,
        text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', [
    'missing-alias', 'duplicate-run', 'failed-worker', 'unfinished-run',
    'nan-metric', 'missing-epochs', 'same-version'
])
def test_partial_or_invalid_lifecycle_cannot_qualify_ui(modules,
                                                        lifecycle_receipts,
                                                        change):
    filename = lifecycle_receipts / ('score.json' if change == 'missing-alias'
                                     else 'fit.json')
    phase = json.loads(filename.read_text())
    if change == 'missing-alias':
        phase['result']['aliases'].pop()
    elif change == 'failed-worker':
        phase['returncode'] = 1
    elif change == 'same-version':
        phase['result']['versions'][1]['version'] = phase['result']['versions'][
            0]['version']
    else:
        reference = phase['result']['versions'][0]['reference']
        if change == 'duplicate-run':
            reference['run'] = phase['result']['versions'][1]['reference'][
                'run']
        elif change == 'unfinished-run':
            reference['status'] = 'RUNNING'
        elif change == 'nan-metric':
            reference['metrics']['metric'] = float('nan')
        elif change == 'missing-epochs':
            reference['history']['train/loss'] = [[0, 0.8]]
    filename.write_text(json.dumps(phase))
    with pytest.raises(ValueError):
        modules.expected_ui.expectations(lifecycle_receipts)


@pytest.mark.parametrize('dirty', [
    'sagemaker_notebook/browser.cjs', 'snowflake_notebook/graph_ui.cjs',
    'snowflake_notebook/browser/package-lock.json'
])
def test_dirty_browser_sources_cannot_reuse_committed_provenance(
        modules, tmp_path, monkeypatch, dirty):
    root = tmp_path / 'repository'
    here = root / 'test/sagemaker_notebook'
    shared = root / 'test/snowflake_notebook'
    files = [
        'sagemaker_notebook/run.py', 'sagemaker_notebook/candidate.json',
        'sagemaker_notebook/browser.cjs', 'sagemaker_notebook/apply_access.sh',
        'sagemaker_notebook/requirements.txt', 'pr172/inputs.json',
        'snowflake_notebook/graph_ui.cjs',
        'snowflake_notebook/browser/package.json',
        'snowflake_notebook/browser/package-lock.json'
    ]
    frozen = {}
    for name in files:
        filename = root / 'test' / name
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(b'committed input\n')
        frozen['test/' + name] = filename.read_bytes()
    monkeypatch.setattr(modules.run, 'REPO', root)
    monkeypatch.setattr(modules.run, 'HERE', here)
    monkeypatch.setattr(modules.run, 'SHARED', shared)
    monkeypatch.setattr(modules.run, 'SOURCES', {})

    def git_read(command, **kwargs):
        if command == ['git', 'rev-parse', 'HEAD']:
            return 'a' * 40 + '\n'
        assert command[:2] == ['git', 'show']
        return frozen[command[2].split(':', 1)[1]]

    monkeypatch.setattr(modules.run.subprocess, 'check_output', git_read)
    assert modules.run.committed_sources()[0] == 'a' * 40
    (root / 'test' / dirty).write_bytes(b'dirty browser verifier\n')
    with pytest.raises(ValueError, match='Commit the exact test inputs'):
        modules.run.committed_sources()
