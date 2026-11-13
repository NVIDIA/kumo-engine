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
    monkeypatch.setenv('SAGEMAKER_TEST_DOMAIN_ID', 'd-test123')
    monkeypatch.setenv('SAGEMAKER_TEST_CANDIDATE', str(
        Path(__file__).parent / 'data/sagemaker-candidate.json'))
    names = ('config', 'evidence', 'runtime', 'client', 'run', 'policies',
             'prepare_access', 'expected_ui', 'studio', 'browser', 'cleanup')
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
    assert client.wait(execution, tmp_path, 1, {}) == job
    assert (tmp_path / 'pipeline-steps.json').is_file()
    execution.stop.assert_not_called()


def test_cleanup_failure_preserves_original_error(modules, tmp_path):
    client = modules.client.Client.__new__(modules.client.Client)
    execution = Mock()
    execution.describe.side_effect = [
        ValueError('original'), RuntimeError('cleanup')
    ]
    with pytest.raises(ValueError, match='original'):
        client.wait(execution, tmp_path, 1, {})


@pytest.fixture
def actual_job_reference(modules):
    config = modules.config.configuration()
    attempt = '123456abcdef'
    pipeline = 'kumo-test-notebook-' + attempt
    config.update(
        attempt=attempt, source_commit='a' * 40, pipeline_name=pipeline,
        job_receipt_key=config['root'] + '/runs/' + attempt +
        '/job-identity.json')
    base = 'arn:aws:sagemaker:us-west-2:926922431314:'
    receipt = dict(
        attempt=attempt, source_commit=config['source_commit'],
        job_arn=base + 'training-job/' + pipeline + '-actual',
        execution_arn=base + 'pipeline/' + pipeline + '/execution/actual123')
    return config, receipt


def test_kernel_discovers_only_its_controller_receipt(modules,
                                                      actual_job_reference):
    config, receipt = actual_job_reference
    session = Mock()
    session.client.return_value.get_object.return_value = {
        'Body': io.BytesIO(json.dumps(receipt).encode())
    }
    assert modules.runtime.discover(config,
                                    session)['job_arn'] == receipt['job_arn']
    session.client.assert_called_once_with('s3')
    session.client.return_value.get_object.assert_called_once_with(
        Bucket=config['bucket'], Key=config['job_receipt_key'])


@pytest.mark.parametrize(
    'field,value',
    [('attempt', 'another'), ('source_commit', 'b' * 40),
     ('job_arn',
      'arn:aws:sagemaker:us-west-2:000000000000:training-job/unrelated'),
     ('execution_arn',
      'arn:aws:sagemaker:us-west-2:926922431314:pipeline/other/execution/x'),
     ('execution_arn', ''), ('job_arn', None),
     ('secret', 'unrecognised field')])
def test_wrong_controller_job_receipt_is_rejected(modules, actual_job_reference,
                                                  field, value):
    config, receipt = actual_job_reference
    receipt[field] = value
    with pytest.raises(ValueError):
        modules.runtime.receipt_reference(config, receipt)


def test_missing_job_receipt_wait_is_bounded(modules, actual_job_reference,
                                             monkeypatch):
    from botocore.exceptions import ClientError

    config, _ = actual_job_reference
    session = Mock()
    session.client.return_value.get_object.side_effect = ClientError(
        {'Error': {
            'Code': 'NoSuchKey'
        }}, 'GetObject')
    times = iter([0, 91])
    monkeypatch.setattr(modules.runtime.time, 'monotonic', lambda: next(times))
    with pytest.raises(TimeoutError, match='did not publish'):
        modules.runtime.discover(config, session)


@pytest.mark.parametrize('collision', [False, True])
def test_controller_publishes_actual_job_once_and_rejects_conflicts(
        modules, actual_job_reference, tmp_path, collision):
    from botocore.exceptions import ClientError

    config, receipt = actual_job_reference
    client = modules.client.Client.__new__(modules.client.Client)
    client.config = config
    client.s3, client.sm = Mock(), Mock()
    client.sm.describe_training_job.return_value = dict(
        TrainingJobArn=receipt['job_arn'], RoleArn=config['execution_role'])
    client.sm.list_tags.return_value = {
        'Tags': [
            dict(Key='KumoTestAttempt', Value=config['attempt']),
            dict(Key='KumoTestSource', Value=config['source_commit'])
        ]
    }
    client.s3.put_object.return_value = {'ETag': 'actual'}
    execution = SimpleNamespace(arn=receipt['execution_arn'])
    if collision:
        client.s3.put_object.side_effect = ClientError(
            {'Error': {
                'Code': 'PreconditionFailed'
            }}, 'PutObject')
        client.s3.get_object.return_value = {
            'Body': io.BytesIO(b'other execution')
        }
        with pytest.raises(ValueError, match='another execution'):
            client.publish_job(execution, receipt['job_arn'], config, tmp_path)
    else:
        client.publish_job(execution, receipt['job_arn'], config, tmp_path)
        stored = client.s3.put_object.call_args.kwargs
        assert stored['IfNoneMatch'] == '*'
        assert json.loads(stored['Body']) == receipt
        assert (tmp_path / 'job-identity.json').read_bytes() == stored['Body']
        assert json.loads(
            (tmp_path / 'training-job-live.json'
             ).read_text())['TrainingJobArn'] == receipt['job_arn']


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
    candidate = json.loads((Path(__file__).parent / 'data/sagemaker-candidate.json').read_text())['candidate']
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
    assert request['ResourceConfig']['InstanceCount'] == 1
    assert request['StoppingCondition']['MaxRuntimeInSeconds'] == 3600
    assert 'RetryStrategy' not in request
    upload.assert_called_once()


def test_policy_plan_separates_launch_and_runtime_roles(modules):
    plan = modules.policies.plan()
    assert set(plan['roles']) == {'RUNNER', 'PIPELINE', 'EXECUTION'}
    assert len({r['arn'] for r in plan['roles'].values()}) == 3
    runner = plan['roles']['RUNNER']
    assert runner['arn'].endswith('/kumo-engine-sagemaker-test-runner')
    trusts = runner['trust']['Statement']
    assert len(trusts) == 1
    condition = trusts[0]['Condition']['StringEquals']
    assert condition['token.actions.githubusercontent.com:aud'] == 'sts.amazonaws.com'
    assert condition['token.actions.githubusercontent.com:sub'] == (
        'repo:NVIDIA@1728152/kumo-engine@1402052651:environment:sagemaker-mlops-test')
    execution = plan['roles']['EXECUTION']['permissions']['Statement']
    assert not any('iam:PassRole' in s['Action'] or
                   'sagemaker:CreateTrainingJob' in s['Action'] for s in execution)
    pipeline = plan['roles']['PIPELINE']['permissions']['Statement']
    create = next(s for s in pipeline if s['Action'] == ['sagemaker:CreateTrainingJob'])
    assert create['Condition'] == {
        'ForAllValues:StringEquals': {
            'sagemaker:InstanceTypes': ['ml.m5.2xlarge']
        },
        'NumericLessThanEquals': {
            'sagemaker:MaxRuntimeInSeconds': 3600
        },
        'Null': {
            'sagemaker:InstanceTypes': 'false',
            'sagemaker:MaxRuntimeInSeconds': 'false'
        },
    }
    passrole = next(s for s in pipeline if s['Action'] == ['iam:PassRole'])
    assert passrole['Resource'] == [plan['roles']['EXECUTION']['arn']]
    assert all('sagemaker:InstanceTypes' not in json.dumps(s.get('Condition', {}))
               for s in pipeline if s is not create)


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
    path = Path(__file__).parent / 'data/mlflow_lifecycle_receipts.json'
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
  VerificationError, verificationMessage, captureAllowed, frontendFailure, safeSnapshot, apiFailure } = require(process.argv[1]);
assert.deepEqual(numericLabels(['0', ' -0.25 ', '1.5e-3', '.75', 'NaN', '', 'step 1']),
  [0, -0.25, 0.0015, 0.75]);
assert.equal(frontendFailure({name:'TypeError',message:"Cannot read properties of null (reading 'getContext')"},'history').message,
  "Cannot read properties of null (reading 'getContext')");
assert.equal(frontendFailure({name:'Error',message:'Failed: https://secret.example/?token=PRIVATE'},'auth').message,undefined);
assert.equal(frontendFailure({name:'Error',message:'WebGL context failed https://secret.example/?token=PRIVATE'},'history').category,'graphics');
const cancelled=frontendFailure({name:'Error',message:'Kernel session disposed: canceled future https://secret.example/?token=PRIVATE'},'save-native-notebook');
assert.deepEqual(cancelled.signatures,['kernel','disposed','cancellation','session']);
assert(!JSON.stringify(cancelled).includes('PRIVATE'));
assert.deepEqual(frontendFailure({name:'Error',message:'Model is disposed'},'editing').signatures,['disposed','disposed_model']);
assert.deepEqual(frontendFailure({name:'Error',message:'This cache has already been disposed'},'editing').signatures,['disposed','disposed_cache']);
assert.deepEqual(frontendFailure({name:'Error',message:'Failed to fetch dynamically imported module: https://secret.example/?token=PRIVATE'},'history').signatures,['dynamic_import','resource_fetch','module']);
const secretUrl='https://mlflow.example/auth?token=PRIVATE&code=OTHER';
const sanitized=safeSnapshot(`- link: ${secretUrl}\n- url: /auth?token=PRIVATE\n- encoded: ${encodeURIComponent(secretUrl)}`,[secretUrl]);
assert.ok(!sanitized.includes('PRIVATE') && !sanitized.includes('OTHER'));
assert.ok(sanitized.includes('[authentication URL]'));
const locatorLog=safeSnapshot(`locator.dblclick: Timeout\nCall log: waiting for locator('.jp-FileBrowser')\n navigated to ${secretUrl}\n password=PRIVATE`,[secretUrl]).slice(0,1500);
assert(!locatorLog.includes('PRIVATE') && !locatorLog.includes('OTHER'));
assert(locatorLog.includes('jp-FileBrowser'));
const apiError=apiFailure('https://mlflow.example/ajax-api/2.0/mlflow/metrics/get-history?token=PRIVATE','https://mlflow.example/',500,
 {error_code:'INTERNAL_ERROR',message:`Failed ${secretUrl}`},[secretUrl]);
assert.equal(apiError.endpoint,'metrics/get-history');
assert.equal(apiError.error_code,'INTERNAL_ERROR');
assert.equal(apiFailure('https://mlflow.example/ajax-api/2.0/mlflow/runs/get','https://mlflow.example/',403,
 {Message:'Forbidden'}).message,'Forbidden');
assert(!JSON.stringify(apiError).includes('PRIVATE'));
assert.equal(apiFailure('https://evil.example/ajax-api/2.0/mlflow/metrics/get-history','https://mlflow.example/',500,{}),undefined);
assert.equal(apiFailure('https://mlflow.example/auth','https://mlflow.example/',500,{}),undefined);
assert.equal(apiFailure('https://mlflow.example/ajax-api/2.0/mlflow/metrics/get-history','https://mlflow.example/',200,{}),undefined);

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
const graphVerification = require(require('node:path').join(require('node:path').dirname(process.argv[1]), '../snowflake_notebook/verification.cjs'));
assert.equal(verificationMessage(new graphVerification.VerificationError('Graph mismatch')), 'Graph mismatch');
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
    'snowflake_notebook/browser/package-lock.json',
    'snowflake_notebook/verification.cjs', 'platform_candidate.py'
])
def test_dirty_browser_sources_cannot_reuse_committed_provenance(
        modules, tmp_path, monkeypatch, dirty):
    root = tmp_path / 'repository'
    here = root / 'test/sagemaker_notebook'
    shared = root / 'test/snowflake_notebook'
    files = [
        'sagemaker_notebook/run.py', 'sagemaker_notebook/browser.cjs',
        'sagemaker_notebook/apply_access.sh',
        'sagemaker_notebook/requirements.txt', 'pr172/inputs.json',
        'snowflake_notebook/graph_ui.cjs',
        'snowflake_notebook/verification.cjs', 'platform_candidate.py',
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


def test_studio_setup_requests_match_aws_service_shapes(modules):
    boto3 = pytest.importorskip('boto3')
    from botocore.validate import validate_parameters
    client = boto3.client('sagemaker', region_name='us-west-2',
                          aws_access_key_id='test', aws_secret_access_key='test')
    plan = modules.policies.studio_plan('vpc-989a8fe0', ['subnet-5101710c'])
    for name, operation in [('domain', 'CreateDomain'),
                            ('profile', 'CreateUserProfile'),
                            ('space', 'CreateSpace')]:
        request = json.loads(json.dumps(plan[name]).replace(
            plan['placeholder'], 'd-test123'))
        shape = client.meta.service_model.operation_model(operation).input_shape
        validate_parameters(request, shape)
    with pytest.raises(ValueError, match='distinct'):
        modules.policies.studio_plan('vpc-989a8fe0',
                                     ['subnet-5101710c', 'subnet-5101710c'])


@pytest.mark.parametrize('role', ['RUNNER', 'PIPELINE', 'EXECUTION', 'MLFLOW'])
def test_bucket_listing_is_limited_to_test_prefix(modules, role):
    import fnmatch

    policy = modules.policies.permissions(role)
    listings = [s for s in policy['Statement'] if 's3:ListBucket' in s['Action']]
    assert listings
    for statement in listings:
        assert statement['Action'] == ['s3:ListBucket']
        pattern = statement['Condition']['StringLike']['s3:prefix']
        assert pattern == 'mlops-validation/sagemaker/*'
        assert fnmatch.fnmatchcase('mlops-validation/sagemaker/', pattern)
        assert fnmatch.fnmatchcase('mlops-validation/sagemaker/payload/file', pattern)
        for prefix in ['', 'mlops-validation/', 'mlops-validation/sagemaker',
                       'mlops-validation/sagemaker-denied/', 'other/']:
            assert not fnmatch.fnmatchcase(prefix, pattern)
    locations = [s for s in policy['Statement'] if 's3:GetBucketLocation' in s['Action']]
    assert all(s['Action'] == ['s3:GetBucketLocation'] and 'Condition' not in s
               for s in locations)


def test_admin_script_needs_no_mode_flag(modules, tmp_path):
    script = modules.prepare_access.standalone(tmp_path, modules.policies.plan())
    text = script.read_text()
    assert '--studio' not in text.split('KUMO_ACCESS_CONFIG')[0]
    result = subprocess.run([shutil.which('sh'), str(script), '--studio'],
                            capture_output=True, text=True,
                            env=dict(os.environ, PATH='/nonexistent'))
    assert result.returncode != 0
    assert 'Usage: sh' in result.stderr


@pytest.fixture
def retained_execution(modules, tmp_path):
    source = tmp_path / 'retained'
    source.mkdir()
    cfg = modules.config.configuration()
    candidate = json.loads((Path(__file__).parent / 'data/sagemaker-candidate.json').read_text())['candidate']
    spec = dict(cfg, attempt='123456abcdef', source_commit='a' * 40,
                scenario='all', pipeline_name='kumo-test-notebook-123456abcdef',
                notebook_name='kumo-test-123456abcdef.ipynb',
                output_prefix=cfg['root'] + '/outputs/123456abcdef',
                candidate=candidate, wheel_sha256=candidate['wheel_sha256'],
                assets={}, test_sources={})
    job = dict(
        TrainingJobArn='arn:aws:sagemaker:us-west-2:926922431314:training-job/'
        + spec['pipeline_name'], TrainingJobStatus='Completed')
    runtime = dict(context='sagemaker_notebook', job_arn=job['TrainingJobArn'],
                   execution_role=cfg['execution_role'])
    phases = {
        name:
        dict(status='passed', returncode=0,
             environment=dict(sha256=spec['wheel_sha256']),
             platform=dict(placement=runtime))
        for name in ['fit', 'score', 'evaluate', 'negative']
    }
    report = dict(
        status='passed', attempt=spec['attempt'],
        source_commit=spec['source_commit'], wheel_sha256=spec['wheel_sha256'],
        output_prefix=spec['output_prefix'], execution_id='abcdef123456',
        payload=spec['assets'], test_sources=spec['test_sources'],
        runtime=runtime, installation=dict(pip_check_returncode=0,
                                           wheel_sha256=spec['wheel_sha256']),
        environment=dict(sha256=spec['wheel_sha256']),
        graph=dict(status='passed'), phases=phases, required_gaps={})
    (source / 'outputs').mkdir()
    content = json.dumps(report).encode()
    (source / 'outputs/result.json').write_bytes(content)
    record = {
        key: report[key]
        for key in [
            'attempt', 'source_commit', 'wheel_sha256', 'output_prefix',
            'execution_id'
        ]
    }
    record['files'] = {'result.json': modules.evidence.digest(content)}
    submitted = dict(
        cells=[dict(cell_type='code', source=['print("retained")'])])
    executed = copy.deepcopy(submitted)
    executed['cells'][0].update(
        execution_count=1, outputs=[
            dict(output_type='stream',
                 text=modules.evidence.MARKER + json.dumps(record) + '\n')
        ])
    pipeline = dict(
        PipelineExecutionStatus='Succeeded',
        PipelineArn='arn:aws:sagemaker:us-west-2:926922431314:pipeline/' +
        spec['pipeline_name'])
    for name, value in [('scenario.json', spec), ('job.json', job),
                        ('executed.ipynb', executed),
                        (spec['notebook_name'], submitted),
                        ('pipeline-execution.json', pipeline)]:
        (source / name).write_text(json.dumps(value))
    return SimpleNamespace(source=source, spec=spec, report=report, job=job,
                           pipeline=pipeline)


@pytest.mark.parametrize('state',
                         ['Failed', 'Stopped', 'Stopping', 'Executing'])
def test_completed_job_cannot_qualify_failed_pipeline(modules,
                                                      retained_execution,
                                                      state):
    e = retained_execution
    e.pipeline['PipelineExecutionStatus'] = state
    with pytest.raises(ValueError, match='pipeline did not succeed'):
        modules.run.verdict(e.report, e.spec, e.job, e.pipeline)


def test_ui_replay_never_submits_or_uploads_training(modules,
                                                     retained_execution,
                                                     monkeypatch, tmp_path):
    e = retained_execution
    client = Mock()
    client.submit.side_effect = AssertionError('UI replay submitted training')
    client.asset.side_effect = AssertionError(
        'UI replay uploaded training inputs')
    monkeypatch.setattr(modules.run, 'Client', lambda cfg: client)
    monkeypatch.setattr(modules.run, 'committed_sources', lambda:
                        ('b' * 40, {}, {}))

    def browser(args, root, actual_client, spec, report):
        assert actual_client is client
        assert (root / 'outputs/result.json').read_bytes() == (
            e.source / 'outputs/result.json').read_bytes()
        assert report['verdict']['engine'] == 'passed'
        return report

    monkeypatch.setattr(modules.run, 'verify_browser', browser)
    output = tmp_path / 'ui'
    output.mkdir()
    args = SimpleNamespace(ui_run=e.source,
                           producer_commit=e.spec['source_commit'])
    report = modules.run.verify_existing(args, output)
    assert report['status'] == 'passed'
    assert json.loads(
        (output /
         'verification.json').read_text())['training_submitted'] is False
    client.submit.assert_not_called()
    client.asset.assert_not_called()


@pytest.mark.parametrize(
    'change', ['checksum', 'producer', 'pipeline', 'candidate', 'role'])
def test_ui_replay_rejects_bad_evidence_before_aws(modules, retained_execution,
                                                   monkeypatch, tmp_path,
                                                   change):
    e = retained_execution
    args = SimpleNamespace(ui_run=e.source,
                           producer_commit=e.spec['source_commit'])
    if change == 'checksum':
        (e.source / 'outputs/result.json').write_text('{}')
    elif change == 'producer':
        args.producer_commit = 'b' * 40
    elif change == 'pipeline':
        e.pipeline['PipelineExecutionStatus'] = 'Stopped'
        (e.source / 'pipeline-execution.json').write_text(json.dumps(
            e.pipeline))
    else:
        e.spec['wheel_sha256' if change ==
               'candidate' else 'execution_role'] = 'wrong'
        (e.source / 'scenario.json').write_text(json.dumps(e.spec))
    monkeypatch.setattr(modules.run, 'Client',
                        Mock(side_effect=AssertionError('AWS accessed')))
    output = tmp_path / 'ui'
    output.mkdir()
    with pytest.raises(ValueError):
        modules.run.verify_existing(args, output)
    modules.run.Client.assert_not_called()


@pytest.mark.parametrize('case', ['failed-native', 'failed-studio', 'graph-only'])
def test_browser_verdict_requires_selected_checks_and_reports_actual_scope(
        modules, retained_execution, monkeypatch, case):
    e = retained_execution
    monkeypatch.setitem(sys.modules, 'nbformat', SimpleNamespace(reads=lambda *a, **k: {}))
    exporter = SimpleNamespace(from_notebook_node=lambda document: ('<html/>', {}))
    monkeypatch.setitem(sys.modules, 'nbconvert', SimpleNamespace(HTMLExporter=lambda **k: exporter))
    ui = dict(status='passed', notebook_export=dict(status='passed'),
              native_mlflow=dict(status='failed') if case == 'failed-native' else 'not selected',
              native_studio=dict(status='failed' if case == 'failed-studio' else 'passed'))
    monkeypatch.setitem(sys.modules, 'browser', SimpleNamespace(verify=lambda *a, **k: ui))
    if case in ('graph-only', 'failed-studio'):
        e.spec['scenario'] = 'graph'
    e.report['verdict'] = modules.run.verdict(e.report, e.spec, e.job, e.pipeline)
    args = SimpleNamespace(engine_checks_only=False)
    if case == 'failed-native':
        with pytest.raises(ValueError, match='MLflow UI verification was not completed'):
            modules.run.verify_browser(args, e.source, Mock(), e.spec, e.report)
    elif case == 'failed-studio':
        with pytest.raises(ValueError, match='Native Studio notebook verification was not completed'):
            modules.run.verify_browser(args, e.source, Mock(), e.spec, e.report)
    else:
        result = modules.run.verify_browser(args, e.source, Mock(), e.spec, e.report)
        assert result['verification_scope'] == 'execution, saved export, native Studio notebook UI'
        assert result['verdict']['full_qualification'] == 'incomplete'


@pytest.fixture
def studio_resources(modules):
    cfg = modules.config.configuration()
    client = Mock(config=cfg)
    studio = modules.studio.Studio(client, modules.config.DOMAIN_ID)
    leases = {}

    def get_lease(**kwargs):
        from botocore.exceptions import ClientError
        if not leases:
            raise ClientError({'Error': {'Code': 'NoSuchKey'}}, 'GetObject')
        return dict(Body=io.BytesIO(leases['body']), ETag='lease-etag')

    def put_lease(**kwargs):
        leases['body'] = kwargs['Body']

    client.s3.get_object.side_effect = get_lease
    client.s3.put_object.side_effect = put_lease
    client.s3.delete_object.side_effect = lambda **kwargs: leases.clear()
    sm = client.sm
    sm.list_tags.return_value = dict(Tags=[dict(
        Key='KumoTestSuite', Value='kumo-engine-sagemaker-studio')])
    sm.describe_domain.return_value = dict(
        DomainArn=studio.arn('domain', studio.domain_id),
        DomainName=cfg['domain_name'], Status='InService', AuthMode='IAM',
        DefaultUserSettings=dict(ExecutionRole=cfg['execution_role']),
        DefaultSpaceSettings=dict(ExecutionRole=cfg['execution_role']))
    sm.describe_user_profile.return_value = dict(
        UserProfileArn=studio.arn('user-profile', studio.domain_id + '/' + cfg['user_profile']),
        Status='InService', UserSettings=dict(ExecutionRole=cfg['execution_role']))
    sm.describe_space.return_value = dict(
        SpaceArn=studio.arn('space', studio.domain_id + '/' + cfg['space']), Status='InService',
        Url='https://space-example.studio.us-west-2.sagemaker.aws/jupyterlab/default',
        OwnershipSettings=dict(OwnerUserProfileName=cfg['user_profile']),
        SpaceSharingSettings=dict(SharingType='Private'),
        SpaceSettings=dict(AppType='JupyterLab',
                           JupyterLabAppSettings=dict(DefaultResourceSpec=dict(InstanceType=cfg['instance'])),
                           SpaceStorageSettings=dict(EbsStorageSettings=dict(EbsVolumeSizeInGb=100))))
    app = dict(studio.target, AppArn=studio.arn('app', studio.domain_id + '/' + cfg['space'] + '/JupyterLab/default'),
               Status='InService', ResourceSpec=dict(InstanceType=cfg['instance'],
                                                     SageMakerImageArn=modules.config.STUDIO_IMAGE))
    sm.describe_app.return_value = app
    return studio, sm, app


@pytest.mark.parametrize('change', ['foreign-domain', 'wrong-role', 'public', 'owner',
                                    'instance', 'foreign-tag', 'foreign-app', 'space-url', 'app-image'])
def test_studio_preflight_rejects_other_resources_before_mutation(studio_resources, change):
    studio, sm, app = studio_resources
    if change == 'foreign-domain':
        sm.describe_domain.return_value['DomainName'] = 'production'
    elif change == 'wrong-role':
        sm.describe_user_profile.return_value['UserSettings']['ExecutionRole'] = 'other-role'
    elif change == 'public':
        sm.describe_space.return_value['SpaceSharingSettings']['SharingType'] = 'Shared'
    elif change == 'owner':
        sm.describe_space.return_value['OwnershipSettings']['OwnerUserProfileName'] = 'other-user'
    elif change == 'instance':
        app['ResourceSpec']['InstanceType'] = 'ml.p5.48xlarge'
    elif change == 'space-url':
        sm.describe_space.return_value['Url'] = 'https://evil.example/jupyterlab/default'
    elif change == 'app-image':
        app['ResourceSpec']['SageMakerImageArn'] = 'other-image'
    elif change == 'foreign-tag':
        sm.list_tags.return_value = dict(Tags=[])
    else:
        app['AppArn'] += '-other'
    with pytest.raises(ValueError):
        studio.start()
    sm.create_app.assert_not_called()
    sm.delete_app.assert_not_called()
    sm.create_presigned_domain_url.assert_not_called()


def test_studio_reuses_and_stops_suite_owned_app(studio_resources):
    studio, sm, app = studio_resources
    assert studio.start()['created'] is False
    sm.describe_app.side_effect = [app, dict(app, Status='Deleted')]
    assert studio.close() == dict(status='deleted', created=False)
    sm.create_app.assert_not_called()
    sm.delete_app.assert_called_once_with(**studio.target)
    sm.create_presigned_domain_url.return_value = dict(AuthorizedUrl='memory-only')
    assert studio.url() == 'memory-only'
    sm.create_presigned_domain_url.assert_called_once_with(
        DomainId=studio.domain_id, UserProfileName=studio.config['user_profile'],
        SpaceName=studio.config['space'], LandingUri='app:JupyterLab:/',
        ExpiresInSeconds=300, SessionExpirationDurationInSeconds=1800)


def test_studio_deletes_only_the_app_created_by_this_invocation(modules, studio_resources):
    studio, sm, app = studio_resources
    deleted = dict(app, Status='Deleted')
    sm.describe_app.side_effect = [deleted, deleted, app, app, deleted]
    sm.create_app.return_value = dict(AppArn=app['AppArn'])
    sm.list_tags.return_value['Tags'].append(dict(Key='KumoTestInvocation', Value=studio.invocation))
    assert studio.start()['created'] is True
    request = sm.create_app.call_args.kwargs
    assert request['ResourceSpec']['SageMakerImageArn'] == modules.config.STUDIO_IMAGE
    assert studio.close() == dict(status='deleted', created=True)
    sm.delete_app.assert_called_once_with(**studio.target)


def test_studio_refuses_to_delete_an_app_without_suite_ownership(studio_resources):
    studio, sm, app = studio_resources
    studio.created = True
    studio.acquire()
    sm.list_tags.return_value = dict(Tags=[])
    with pytest.raises(ValueError, match='not owned by this test suite'):
        studio.close()
    sm.delete_app.assert_not_called()


def test_studio_browser_requires_exact_aws_space_origin_and_app_path():
    node = shutil.which('node')
    if not node:
        pytest.skip('Browser boundary regression needs Node.js')
    result = subprocess.run([node, '-e', '''
const assert = require('node:assert/strict');
const Module = require('node:module');
const originalLoad = Module._load;
Module._load = function(request, ...args) {
 if (request.includes('playwright')) throw new Error('Offline URL guards must not load Playwright');
 return originalLoad.call(this, request, ...args);
};
const {safePage} = require(process.argv[1]);
const expected='https://space-example.studio.us-west-2.sagemaker.aws/jupyterlab/default';
assert(safePage(expected+'/lab',expected));
for (const url of ['https://space-other.studio.us-west-2.sagemaker.aws/jupyterlab/default',
 'https://space-example.studio.us-west-2.sagemaker.aws/auth',
 expected+'?token=PRIVATE', 'https://evil-d-example.studio.sagemaker.aws/',
 'https://d-example.studio.sagemaker.aws.evil.example/',
 'https://d-example.studio.sagemaker.aws/?token=private',
 'https://user:password@d-example.studio.sagemaker.aws/', 'http://d-example.studio.sagemaker.aws/']) {
 assert(!safePage(url,expected));
}
''', str(HERE / 'studio_browser.cjs')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_native_studio_receipt_reports_cleanup_failures_and_omits_auth_url(
        modules, execution, monkeypatch, tmp_path, cleanup_fails):
    e = execution
    e.spec.update(notebook_name='kumo-test.ipynb', scenario='engine')
    (tmp_path / 'executed.ipynb').write_text(json.dumps(e.executed))
    (tmp_path / e.spec['notebook_name']).write_text(json.dumps(e.submitted))
    studio = Mock(invocation='c' * 32, space_url='https://space-example.studio.us-west-2.sagemaker.aws/jupyterlab/default')
    studio.start.return_value = dict(status='InService', created=True)
    studio.url.return_value = 'https://test.example/?token=PRIVATE'
    if cleanup_fails:
        studio.close.side_effect = RuntimeError('cleanup failed')
    else:
        studio.close.return_value = dict(status='deleted')
    monkeypatch.setattr(modules.studio, 'Studio', lambda *a: studio)

    def browser(*args, **kwargs):
        request = json.loads(kwargs['input'])
        assert request['receipt'] == e.record
        assert request['url'] == studio.url.return_value
        output = tmp_path / 'studio'
        output.mkdir()
        (output / 'result.json').write_text('{"status":"passed"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(modules.studio.subprocess, 'run', browser)
    if cleanup_fails:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            modules.studio.verify_studio(Mock(), e.spec, tmp_path)
    else:
        assert modules.studio.verify_studio(Mock(), e.spec, tmp_path)['status'] == 'passed'
    report = (tmp_path / 'studio-app.json').read_text()
    assert 'PRIVATE' not in report
    assert json.loads(report)['status'] == ('failed' if cleanup_fails else 'passed')
    studio.close.assert_called_once()


@pytest.mark.parametrize('export_passed', [False, True])
def test_native_studio_collects_independent_evidence_after_export_failure(
        modules, monkeypatch, tmp_path, export_passed):
    def browser(*args, **kwargs):
        root = tmp_path / 'browser'
        root.mkdir()
        report = dict(status='passed' if export_passed else 'failed',
                      notebook_export=dict(status='passed' if export_passed else 'failed'))
        (root / 'result.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=0 if export_passed else 1)

    monkeypatch.setattr(modules.browser.subprocess, 'run', browser)
    native = Mock(return_value=dict(status='passed'))
    monkeypatch.setattr(modules.studio, 'verify_studio', native)
    if export_passed:
        assert modules.browser.verify(Mock(), dict(scenario='graph'), tmp_path, native=True)['status'] == 'passed'
    else:
        with pytest.raises(RuntimeError, match='browser verification failed'):
            modules.browser.verify(Mock(), dict(scenario='graph'), tmp_path, native=True)
    native.assert_called_once()
    report = json.loads((tmp_path / 'browser/result.json').read_text())
    assert report['native_studio']['status'] == 'passed'
    assert report['status'] == ('passed' if export_passed else 'failed')


def test_runner_mlflow_ui_addition_is_readonly_and_dedicated(modules):
    policy = modules.policies.plan()['mlflow_ui_permissions']
    assert len(policy['Statement']) == 1
    statement = policy['Statement'][0]
    assert statement['Effect'] == 'Allow'
    assert statement['Resource'] == [modules.config.configuration()['tracking_arn']]
    actions = statement['Action']
    assert {'sagemaker-mlflow:GetRun', 'sagemaker-mlflow:GetExperiment',
            'sagemaker-mlflow:GetMetricHistory'} <= set(actions)
    assert all(action.startswith(('sagemaker-mlflow:Get', 'sagemaker-mlflow:List',
                                  'sagemaker-mlflow:Search')) and '*' not in action
               for action in actions)


def test_launcher_initializes_mlflow_auth_before_first_request(
        modules, tmp_path, monkeypatch):
    tracking_arn = modules.config.configuration()['tracking_arn']
    state = {'tracking': 'file:///tmp/mlruns', 'registry': None}
    fake = Mock()
    fake.set_tracking_uri.side_effect = lambda uri: state.update(tracking=uri)
    fake.set_registry_uri.side_effect = lambda uri: state.update(registry=uri)

    def create_client(**kwargs):
        assert state == {'tracking': tracking_arn, 'registry': tracking_arn}
        return SimpleNamespace(get_experiment_by_name=lambda name: state['tracking'])

    fake.MlflowClient.side_effect = create_client
    monkeypatch.setitem(sys.modules, 'mlflow', fake)
    spec = dict(modules.config.configuration(), attempt='123456abcdef',
                pipeline_name='kumo-test-notebook-123456abcdef',
                source_commit='a' * 40, scenario='engine')
    resources = modules.cleanup.AttemptResources(Mock(), spec, tmp_path)
    client = resources.mlflow()
    assert client.get_experiment_by_name('unused') == tracking_arn
    assert resources.mlflow() is client
    fake.MlflowClient.assert_called_once_with(
        tracking_uri=tracking_arn, registry_uri=tracking_arn)


def test_attempt_cleanup_preserves_cache_and_only_deletes_owned_prefixes(modules, tmp_path):
    spec = dict(modules.config.configuration(), attempt='123456abcdef',
                pipeline_name='kumo-test-notebook-123456abcdef',
                source_commit='a' * 40, scenario='graph')
    from botocore.exceptions import ClientError
    client = Mock()
    client.sm.describe_pipeline.side_effect = ClientError(
        {'Error': {'Code': 'ResourceNotFound'}}, 'DescribePipeline')
    client.s3.list_objects_v2.return_value = {}
    with modules.cleanup.AttemptResources(client, spec, tmp_path):
        pass
    prefixes = [c.kwargs['Prefix'] for c in client.s3.list_objects_v2.call_args_list]
    assert set(prefixes) == {
        spec['root'] + '/runs/123456abcdef/',
        spec['root'] + '/notebook-jobs/123456abcdef/',
        spec['root'] + '/artifacts/123456abcdef/',
    }
    assert json.loads((tmp_path / 'cleanup.json').read_text())['status'] == 'passed'
    client.sm.delete_pipeline.assert_not_called()


def test_cleanup_cannot_acquire_existing_attempt_or_delete_foreign_pipeline(modules, tmp_path):
    from botocore.exceptions import ClientError
    spec = dict(modules.config.configuration(), attempt='123456abcdef',
                pipeline_name='kumo-test-notebook-123456abcdef',
                source_commit='a' * 40, scenario='graph')
    client = Mock()
    client.s3.list_objects_v2.return_value = {'Contents': [{'Key': 'existing'}]}
    with pytest.raises(ValueError, match='already exists'):
        with modules.cleanup.AttemptResources(client, spec, tmp_path):
            pass
    client.sm.delete_pipeline.assert_not_called()
    client.s3.delete_objects.assert_not_called()
    client.s3.list_objects_v2.return_value = {}
    client.sm.describe_pipeline.side_effect = [
        ClientError({'Error': {'Code': 'ResourceNotFound'}}, 'DescribePipeline'),
        dict(PipelineArn='foreign', RoleArn=spec['pipeline_role'])]
    client.sm.list_tags.return_value = {'Tags': []}
    with pytest.raises(ValueError, match='ownership changed'):
        with modules.cleanup.AttemptResources(client, spec, tmp_path):
            pass
    client.sm.delete_pipeline.assert_not_called()
    client.s3.delete_objects.assert_not_called()


def test_cleanup_times_out_without_deleting_active_pipeline(modules, tmp_path, monkeypatch):
    from botocore.exceptions import ClientError
    spec = dict(modules.config.configuration(), attempt='123456abcdef',
                pipeline_name='kumo-test-notebook-123456abcdef',
                source_commit='a' * 40, scenario='graph')
    client = Mock()
    client.s3.list_objects_v2.return_value = {}
    arn = f'arn:aws:sagemaker:{spec["region"]}:{spec["account"]}:pipeline/{spec["pipeline_name"]}'
    client.sm.describe_pipeline.side_effect = [
        ClientError({'Error': {'Code': 'ResourceNotFound'}}, 'DescribePipeline'),
        dict(PipelineArn=arn, RoleArn=spec['pipeline_role'])]
    client.sm.list_tags.return_value = dict(Tags=[
        dict(Key='KumoTestSuite', Value='sagemaker-platform'),
        dict(Key='KumoTestAttempt', Value=spec['attempt']),
        dict(Key='KumoTestSource', Value=spec['source_commit'])])
    client.sm.list_pipeline_executions.return_value = dict(
        PipelineExecutionSummaries=[dict(PipelineExecutionStatus='Executing',
                                         PipelineExecutionArn=arn + '/execution/test')])
    client.sm.list_pipeline_execution_steps.return_value = {}
    monkeypatch.setattr(modules.cleanup.time, 'monotonic', Mock(side_effect=[0, 601]))
    with pytest.raises(TimeoutError, match='did not stop within its deadline'):
        with modules.cleanup.AttemptResources(client, spec, tmp_path):
            pass
    client.sm.delete_pipeline.assert_not_called()
    client.s3.delete_objects.assert_not_called()


@pytest.mark.parametrize('dispatch',
                         ['main', 'egu/release-v1.1-platform-tests'])
@pytest.mark.parametrize('prevent_self_review', [False, True])
@pytest.mark.parametrize('invalid', [None, 'admin-bypass', 'reviewers', 'branches'])
def test_sagemaker_environment_keeps_main_independent_and_trial_self_approval(
        modules, monkeypatch, prevent_self_review, invalid, dispatch):
    import urllib.request
    from importlib.util import module_from_spec, spec_from_file_location
    rule = dict(type='required_reviewers', prevent_self_review=prevent_self_review,
                reviewers=[dict(type='Team', reviewer=dict(id=19956283))])
    environment = dict(name='sagemaker-mlops-test',
                       can_admins_bypass=False, protection_rules=[rule],
                       deployment_branch_policy=dict(protected_branches=False,
                                                     custom_branch_policies=True))
    policies = dict(
        total_count=2, branch_policies=[
            dict(type='branch', name='main'),
            dict(type='branch', name='egu/release-v1.1-platform-tests')
        ])
    if dispatch == 'egu/release-v1.1-platform-tests':
        policies['branch_policies'].append(dict(
            type='branch', name='egu/sagemaker-platform-validation'))
        policies['total_count'] = 3
    if invalid == 'admin-bypass':
        environment['can_admins_bypass'] = True
    if invalid == 'reviewers':
        rule['reviewers'] = []
    if invalid == 'branches':
        policies['branch_policies'][1]['name'] = 'feature-*'
    responses = iter([environment, policies])
    requested = []

    def get(request, **kwargs):
        requested.append(request.full_url)
        return io.BytesIO(json.dumps(next(responses)).encode())

    monkeypatch.setattr(urllib.request, 'urlopen', get)
    for name, value in dict(GITHUB_TOKEN='test-token', GITHUB_REF_NAME=dispatch,
                            GITHUB_REF='refs/heads/' + dispatch,
                            GITHUB_EVENT_NAME='workflow_dispatch',
                            GITHUB_REPOSITORY='NVIDIA/kumo-engine').items():
        monkeypatch.setenv(name, value)
    spec = spec_from_file_location('sagemaker_environment', HERE / 'check_environment.py')
    guard = module_from_spec(spec)
    spec.loader.exec_module(guard)
    if invalid is None and (prevent_self_review or dispatch != 'main'):
        guard.main()
    else:
        with pytest.raises(ValueError):
            guard.main()
    assert all('/environments/sagemaker-mlops-test' in url for url in requested)


@pytest.mark.parametrize('foreign_artifacts', [False, True])
def test_cleanup_removes_only_the_attempt_experiment_artifacts(
        modules, monkeypatch, tmp_path, foreign_artifacts):
    from botocore.exceptions import ClientError

    class MissingModel(Exception):
        error_code = 'RESOURCE_DOES_NOT_EXIST'

    monkeypatch.setitem(sys.modules, 'mlflow.exceptions',
                        SimpleNamespace(MlflowException=MissingModel))
    spec = dict(modules.config.configuration(), attempt='123456abcdef',
                pipeline_name='kumo-test-notebook-123456abcdef',
                source_commit='a' * 40, scenario='engine')
    client, tracking = Mock(), Mock()
    client.s3.list_objects_v2.return_value = {}
    client.sm.describe_pipeline.side_effect = ClientError(
        {'Error': {'Code': 'ResourceNotFound'}}, 'DescribePipeline')
    experiment = SimpleNamespace(
        experiment_id='42', name='kumo-test-mlops-123456abcdef',
        tags={'kumo.test.attempt': spec['attempt'],
              'kumo.test.source': spec['source_commit']},
        artifact_location=('s3://other-bucket/private/42' if foreign_artifacts else
                           spec['tracking_artifact_root'] + '/42'))
    tracking.get_experiment_by_name.side_effect = [None, experiment]
    model = SimpleNamespace(name=experiment.name, creation_timestamp=1, tags=experiment.tags)
    tracking.create_experiment.return_value = '42'
    tracking.create_registered_model.return_value = model
    tracking.get_registered_model.side_effect = [MissingModel(), model]
    resources = modules.cleanup.AttemptResources(client, spec, tmp_path, tracking)
    if foreign_artifacts:
        with pytest.raises(ValueError, match='unexpected artifact location'):
            with resources:
                resources.prepare_tracking()
        tracking.delete_experiment.assert_not_called()
    else:
        with resources:
            resources.prepare_tracking()
        tracking.delete_experiment.assert_called_once_with('42')
        assert any(call.kwargs['Prefix'].endswith('/mlflow-artifacts-final/42/')
                   for call in client.s3.list_objects_v2.call_args_list)
    tracking.delete_registered_model.assert_called_once_with('kumo-test-mlops-123456abcdef')


def test_pipeline_permissions_cover_pipeline_and_execution_arns(modules):
    import fnmatch
    policies = modules.policies.permissions('RUNNER')['Statement']
    statement = next(s for s in policies if 'sagemaker:CreatePipeline' in s['Action'])
    base = 'arn:aws:sagemaker:us-west-2:926922431314:pipeline/'
    for arn in [base + 'kumo-test-notebook-123456abcdef',
                base + 'kumo-test-notebook-123456abcdef/execution/test']:
        assert any(fnmatch.fnmatchcase(arn, p) for p in statement['Resource'])
    assert not any(fnmatch.fnmatchcase(base + 'production', p)
                   for p in statement['Resource'])


def test_studio_lease_refuses_overlapping_invocations(studio_resources):
    import time
    studio, sm, app = studio_resources
    studio.s3.get_object.side_effect = lambda **kw: dict(
        Body=io.BytesIO(json.dumps(dict(invocation='other', expires=time.time() + 3600)).encode()),
        ETag='other-etag')
    with pytest.raises(ValueError, match='in use by another invocation'):
        studio.start()
    assert studio.close()['status'] == 'no lease acquired'
    sm.create_app.assert_not_called()
    sm.delete_app.assert_not_called()
    studio.s3.put_object.assert_not_called()


def test_studio_lease_loss_cannot_stop_another_invocation(studio_resources):
    import time
    studio, sm, app = studio_resources
    studio.acquire()
    studio.s3.get_object.side_effect = lambda **kw: dict(
        Body=io.BytesIO(json.dumps(dict(invocation='other', expires=time.time() + 3600)).encode()),
        ETag='other-etag')
    with pytest.raises(ValueError, match='lease ownership changed'):
        studio.close()
    sm.delete_app.assert_not_called()
    studio.s3.delete_object.assert_not_called()


def test_cleanup_waits_for_pipeline_and_its_job_to_stop(modules, tmp_path, monkeypatch):
    spec = dict(modules.config.configuration(), attempt='123456abcdef',
                pipeline_name='kumo-test-notebook-123456abcdef',
                source_commit='a' * 40, scenario='graph')
    client = Mock()
    resource = modules.cleanup.AttemptResources(client, spec, tmp_path)
    pipeline = f'arn:aws:sagemaker:{spec["region"]}:{spec["account"]}:pipeline/{spec["pipeline_name"]}'
    job = f'arn:aws:sagemaker:{spec["region"]}:{spec["account"]}:training-job/{spec["pipeline_name"]}-job'
    client.sm.list_pipeline_executions.side_effect = [
        dict(PipelineExecutionSummaries=[dict(PipelineExecutionStatus=state,
                                              PipelineExecutionArn=pipeline + '/execution/test')])
        for state in ('Executing', 'Stopped')]
    client.sm.list_pipeline_execution_steps.return_value = dict(
        PipelineExecutionSteps=[dict(Metadata=dict(TrainingJob=dict(Arn=job)))])
    client.sm.describe_training_job.side_effect = [
        dict(TrainingJobArn=job, RoleArn=spec['execution_role'], TrainingJobStatus=state)
        for state in ('Stopping', 'Stopped')]
    sleep = Mock()
    monkeypatch.setattr(modules.cleanup.time, 'sleep', sleep)
    resource.wait_terminal()
    sleep.assert_called_once_with(10)
    client.sm.stop_pipeline_execution.assert_called_once_with(
        PipelineExecutionArn=pipeline + '/execution/test')


def test_cleanup_refuses_changed_registered_model_identity(modules, tmp_path, monkeypatch):
    from botocore.exceptions import ClientError

    class MissingModel(Exception):
        error_code = 'RESOURCE_DOES_NOT_EXIST'

    monkeypatch.setitem(sys.modules, 'mlflow.exceptions',
                        SimpleNamespace(MlflowException=MissingModel))
    spec = dict(modules.config.configuration(), attempt='123456abcdef',
                pipeline_name='kumo-test-notebook-123456abcdef',
                source_commit='a' * 40, scenario='engine')
    client, tracking = Mock(), Mock()
    client.s3.list_objects_v2.return_value = {}
    client.sm.describe_pipeline.side_effect = ClientError(
        {'Error': {'Code': 'ResourceNotFound'}}, 'DescribePipeline')
    tracking.get_experiment_by_name.return_value = None
    model = SimpleNamespace(name='kumo-test-mlops-123456abcdef', creation_timestamp=1,
                            tags={'kumo.test.attempt': 'other', 'kumo.test.source': 'a' * 40})
    tracking.create_experiment.return_value = '42'
    tracking.create_registered_model.return_value = model
    tracking.get_registered_model.side_effect = [MissingModel(), model]
    resources = modules.cleanup.AttemptResources(client, spec, tmp_path, tracking)
    with pytest.raises(ValueError, match='model ownership changed'):
        with resources:
            resources.prepare_tracking()
    tracking.delete_registered_model.assert_not_called()
    tracking.delete_experiment.assert_not_called()
    client.s3.delete_objects.assert_not_called()


def test_missing_cache_object_403_requires_an_empty_scoped_listing(modules, tmp_path):
    import base64

    from botocore.exceptions import ClientError
    client = modules.client.Client.__new__(modules.client.Client)
    client.config, client.s3 = modules.config.configuration(), Mock()
    content = b'cached payload'
    sha = modules.evidence.digest(content)
    local = tmp_path / 'runtime.py'
    local.write_bytes(content)
    denied = ClientError({'Error': {'Code': '403'}}, 'HeadObject')
    head = dict(ContentLength=len(content),
                ChecksumSHA256=base64.b64encode(bytes.fromhex(sha)).decode())
    client.s3.head_object.side_effect = [denied, head]
    client.s3.list_objects_v2.return_value = {}
    result = client.asset('runtime.py', sha, local)
    client.s3.list_objects_v2.assert_called_once_with(
        Bucket=client.config['bucket'], Prefix=result['key'], MaxKeys=1)
    assert client.s3.put_object.call_count == 1
    client.s3.head_object.side_effect = denied
    client.s3.list_objects_v2.return_value = dict(Contents=[dict(Key=result['key'])])
    with pytest.raises(ClientError):
        client.asset('runtime.py', sha, local)
    assert client.s3.put_object.call_count == 1


@pytest.mark.parametrize('code,expected', [
    ('ACCESS_DENIED', True), ('NO_SUCH_BUCKET', False),
    ('NETWORK_CONNECTION', False), ('EXPIRED_TOKEN', False),
])
def test_arrow_s3_permission_probe_requires_provider_denial(code, expected):
    spec = importlib.util.spec_from_file_location(
        'sagemaker_platform_probe', HERE.parent / 'wheel_mlops_platform.py')
    platform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(platform)
    error = OSError(
        "When initiating multiple part upload for key 'test/.kumo-write-check' "
        "in bucket 'kumo-test-dataplane-bucket': AWS Error " + code +
        ' during CreateMultipartUpload operation: provider diagnostic')
    wrapped = PermissionError('MLOps cannot write under artifact_root')
    wrapped.__cause__ = error
    assert platform.denied_access(wrapped, provider_only=True,
                                  provider='sagemaker') is expected
    assert not platform.denied_access(wrapped, provider_only=True,
                                      provider='snowflake')
    assert not platform.denied_access(PermissionError('local spool denied'),
                                      provider_only=True)


@pytest.mark.parametrize('state', [
    'Created', 'Started', 'Updated', 'MaintenanceComplete',
    'MaintenanceInProgress', 'Stopped'
])
def test_tracking_server_admission_requires_healthy_active_state(
        modules, state):
    cfg = modules.config.configuration()
    client = modules.client.Client.__new__(modules.client.Client)
    client.config = cfg
    client.identity = {'Account': cfg['account'], 'Arn': 'test-principal'}
    client.s3 = Mock()
    client.sm = Mock()
    client.s3.get_bucket_location.return_value = {
        'LocationConstraint': cfg['region']
    }
    client.s3.get_object.return_value = {'Body': io.BytesIO(b'sentinel')}
    client.sm.describe_mlflow_tracking_server.return_value = dict(
        TrackingServerArn=cfg['tracking_arn'], RoleArn=cfg['tracking_role'],
        ArtifactStoreUri=cfg['tracking_artifact_root'],
        TrackingServerStatus=state, IsActive='Active', MlflowVersion='3.5.0')
    if state in ('Stopped', 'MaintenanceInProgress'):
        with pytest.raises(ValueError, match='not ready'):
            client.preflight()
    else:
        assert client.preflight()['tracking_arn'] == cfg['tracking_arn']


@pytest.mark.parametrize('code', ['403', 'AccessDenied', '404', 'NoSuchKey'])
def test_runtime_waits_for_prefix_scoped_receipt_publication(
        modules, actual_job_reference, monkeypatch, code):
    from botocore.exceptions import ClientError
    config, receipt = actual_job_reference
    session = Mock()
    session.client.return_value.get_object.side_effect = [
        ClientError({'Error': {
            'Code': code
        }}, 'GetObject'), {
            'Body': io.BytesIO(json.dumps(receipt).encode())
        }
    ]
    monkeypatch.setattr(modules.runtime.time, 'sleep', lambda _: None)
    assert modules.runtime.discover(config,
                                    session)['job_arn'] == receipt['job_arn']
    assert session.client.return_value.get_object.call_count == 2


def test_untrusted_ref_rejected_before_environment_lookup(modules, monkeypatch):
    import urllib.request
    path = HERE / 'check_environment.py'
    spec = importlib.util.spec_from_file_location('sagemaker_guard_untrusted',
                                                  path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setenv('GITHUB_REPOSITORY', 'NVIDIA/kumo-engine')
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'workflow_dispatch')
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/untrusted')
    monkeypatch.setenv('GITHUB_REF_NAME', 'untrusted')
    monkeypatch.setattr(
        urllib.request, 'urlopen',
        lambda *args: pytest.fail('Untrusted ref contacted API'))
    with pytest.raises(SystemExit):
        guard.main()


def test_kernel_budget_interrupts_hung_work_and_leaves_evidence_time(tmp_path):
    source = HERE / 'budget.py'
    script = '''
import importlib.util,json,signal,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('budget',sys.argv[1])
budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)
root=Path(sys.argv[2])
(root/'execution-deadline.json').write_text(json.dumps(time.monotonic()+0.1))
previous=signal.getsignal(signal.SIGALRM)
try:
    with budget.limit(root):
        time.sleep(10)
except TimeoutError:
    pass
else:
    raise AssertionError('Hung work escaped the work budget')
finally:
    (root/'evidence.txt').write_text('published after work timeout')
assert signal.getitimer(signal.ITIMER_REAL)==(0.0,0.0)
assert signal.getsignal(signal.SIGALRM)==previous
'''
    result = subprocess.run(
        [sys.executable, '-I', '-c', script,
         str(source),
         str(tmp_path)], capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    assert (tmp_path /
            'evidence.txt').read_text() == 'published after work timeout'


def test_browser_diagnostics_publish_checks_without_runtime_data(
        modules, tmp_path):
    browser = tmp_path / 'browser'
    browser.mkdir()
    secret = 'https://private.example/?token=secret-value'
    (browser / 'result.json').write_text(
        json.dumps(
            dict(status='failed', stage='notebook-export',
                 error='Browser emitted an uncaught error', url=secret,
                 assertion='metric-row:' + secret)))
    studio = tmp_path / 'studio'
    studio.mkdir()
    (studio / 'result.json').write_text(
        json.dumps(
            dict(status=secret, stage=secret, error=secret, assertion=secret)))
    result = modules.run.browser_diagnostics(tmp_path)
    assert result == dict(
        browser=dict(status='failed', stage='notebook-export',
                     error='Browser emitted an uncaught error',
                     assertion='metric-row:[fixture value]'), studio={})
    assert secret not in json.dumps(result)


def test_browser_diagnostics_preserve_progress_without_result(
        modules, tmp_path):
    studio = tmp_path / 'studio'
    studio.mkdir()
    (studio / 'progress.json').write_text(
        json.dumps(dict(stage='native-file-open')))
    assert modules.run.browser_diagnostics(tmp_path) == dict(studio=dict(
        stage='native-file-open'))


def test_browser_diagnostics_publish_only_fixed_frontend_categories(
        modules, tmp_path):
    browser = tmp_path / 'browser'
    browser.mkdir()
    secret = 'https://private.example/?token=secret-value'
    (browser / 'result.json').write_text(
        json.dumps(
            dict(
                status='failed', stage='mlflow-score-drift-evaluate-5',
                native_mlflow=dict(status='passed', url=secret),
                notebook_export=dict(status=secret), browser_errors=[
                    dict(
                        category='graphics', name='TypeError',
                        stage='native-save-probe-reopened', message=secret,
                        signatures=[
                            'resize_observer', 'disposed_model', secret
                        ]),
                    dict(category=secret, name=secret, stage=secret,
                         message=secret)
                ])))
    result = modules.run.browser_diagnostics(tmp_path)
    assert result == dict(browser=dict(
        status='failed', stage='mlflow-score-drift-evaluate-5',
        native_mlflow='passed', browser_error_count=2, browser_errors=[
            dict(category='graphics', name='TypeError',
                 stage='native-save-probe-reopened',
                 signatures=['disposed_model', 'resize_observer'])
        ]))
    assert secret not in json.dumps(result)


def test_producer_reserves_markdown_edit_without_changing_code_execution(
        modules):
    import nbformat

    spec = dict(attempt='123456abcdef', kernel='kumo-test-engine')
    document = modules.run.notebook(spec, 'def execute(config): pass')
    nbformat.validate(nbformat.from_dict(document))
    markdown = [
        cell for cell in document['cells'] if cell['cell_type'] == 'markdown'
    ]
    code = [cell for cell in document['cells'] if cell['cell_type'] == 'code']
    assert len(markdown) == len(code) == 1
    assert document['cells'][0] is markdown[0]
    assert markdown[0]['id'] == 'studio-save-probe'
    assert markdown[0]['source'] == ['Studio save probe.']
    assert 'KUMO_SAVE_PROBE=' not in json.dumps(document)
    assert code[0]['id'] == spec['attempt']
    assert code[0]['execution_count'] is None and code[0]['outputs'] == []
    assert ''.join(code[0]['source']).endswith('execute(CONFIG)')


@pytest.mark.parametrize('reserved', [False, True, 'other-markdown'])
def test_studio_input_keeps_original_code_outputs_and_receipt(
        modules, execution, tmp_path, reserved):
    executed = copy.deepcopy(execution.executed)
    if reserved is True:
        executed['cells'].insert(0, modules.evidence.save_probe_cell())
    elif reserved == 'other-markdown':
        executed['cells'].insert(
            0,
            dict(cell_type='markdown', id='old-header',
                 metadata=dict(editable=False),
                 source=['Original producer header']))
    original = tmp_path / 'executed.ipynb'
    original.write_text(json.dumps(executed))
    before = original.read_bytes()
    upload = modules.studio.prepare_studio_input(tmp_path, executed)
    prepared = json.loads(upload.read_text())
    assert prepared['cells'][0] == modules.evidence.save_probe_cell()
    assert [cell for cell in prepared['cells']
            if cell['cell_type'] == 'code'] == execution.executed['cells']
    assert modules.evidence.receipt(prepared, execution.submitted,
                                    execution.spec) == execution.record
    assert original.read_bytes() == before
    if reserved is True:
        assert upload == original
    else:
        assert upload != original
        assert modules.evidence.digest(
            upload.read_bytes()) != modules.evidence.digest(before)
