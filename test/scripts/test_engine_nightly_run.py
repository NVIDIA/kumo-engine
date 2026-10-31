# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject old, incomplete or mismatched evidence at the nightly controller."""
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[2] / 'scripts/engine_nightly_run.py'


def controller():
    assert SOURCE.is_file(), 'The full nightly controller is missing'
    spec = importlib.util.spec_from_file_location('nightly_run', SOURCE)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_historical_reference_and_zero_fit_audit_cannot_pass_as_a_new_run():
    run = controller()
    request = dict(run_id='123-1',
                   source_sha='a' * 40,
                   nonce='nonce',
                   instance_id='i-06dd0e3c558780731')
    for receipt in [
            dict(status='AUDIT_COMPLETE', new_physical_fits=0),
            dict(format='kumo-engine-nightly-reference-v1', cases=[{}] * 16),
            dict(request, status='RUNNING', physical_fits=16)
    ]:
        with pytest.raises(ValueError):
            run.validate_nightly_result(request, receipt, {})


def test_cancel_marker_binds_to_the_exact_request_and_has_no_credentials():
    run = controller()
    request = dict(run_id='123-1', nonce='nonce', source_sha='a' * 40)
    marker = run.cancel_marker(request, 'b' * 64)
    assert marker == dict(run_id='123-1',
                          nonce='nonce',
                          request_sha256='b' * 64,
                          cancel=True)


def test_payload_keeps_only_the_trusted_nightly_code(tmp_path):
    import hashlib
    import tarfile

    run = controller()
    source = tmp_path / 'source'
    source.mkdir()
    files = {}
    for name in ('nightly_check.py', 'wheel_gpu_smoke.py', 'nightly_host.py'):
        path = source / name
        path.write_text('# trusted source\n')
        files[name] = path
    output = tmp_path / 'payload.tar.gz'
    description = run.pack_payload(files, output)
    assert description['sha256'] == hashlib.sha256(
        output.read_bytes()).hexdigest()
    with tarfile.open(output) as archive:
        assert set(archive.getnames()) == set(files)
        assert all(member.isfile() for member in archive.getmembers())
    assert description['files'] == {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in files.items()
    }


def measured_fixture():
    """Protocol data only. This fixture is never GPU evidence."""
    import copy
    request = dict(
        run_id='123-1',
        source_sha='a' * 40,
        nonce='nonce',
        instance_id='i-06dd0e3c558780731',
        reference_only=False,
        wheel=dict(sha256='b' * 64, version='1.0.0.dev0+aaa'),
        payload=dict(files={
            'nightly_check.py': 'c' * 64,
            'wheel_gpu_smoke.py': 'd' * 64
        }),
        inputs=dict(input_manifest_sha256='e' * 64, reference_sha256='f' * 64))
    receipt = dict(request,
                   status='PASS',
                   run_exit=0,
                   compare_exit=0,
                   processes_ended=True,
                   cuda_visible_devices='0')
    report = dict(format='kumo-engine-nightly-result-v1',
                  status='MEASURED',
                  completed_runs=16,
                  expected_runs=16,
                  reference_only=False,
                  reference_sha256='f' * 64,
                  input_manifest_sha256='e' * 64,
                  runner_sha256='c' * 64,
                  helper_sha256='d' * 64,
                  runtime=dict(wheel_sha256='b' * 64,
                               version='1.0.0.dev0+aaa',
                               gpu='RTX PRO 6000',
                               gpu_count=1,
                               architecture='x86_64'),
                  runs=[])

    def metric(scope):
        return dict(format='kumo-nightly-resources-v1',
                    scope=scope,
                    samples=2,
                    sample_interval_seconds=.2,
                    cpu_read_errors=0,
                    cuda_read_errors=0,
                    elapsed_seconds=1.,
                    cpu_self_seconds=1.,
                    cpu_tree_pss_sampled_peak_bytes=1,
                    cpu_tree_rss_sampled_peak_bytes=1,
                    cuda_device_used_sampled_peak_bytes=1,
                    torch_allocated_sampled_peak_bytes=1,
                    torch_reserved_sampled_peak_bytes=1)

    work = dict(
        core_fit_attempts=1,
        fits=1,
        completed_fits=1,
        training_batches=200,
        optimizer_steps=200,
        training_epochs=[dict(batches=20, batch_sizes=[64] * 20)] * 10,
        restored_tensors=1,
        reload_checks=1,
        core_fit_ledger=[dict(status='PASS', steps=200, cuda_batches=200)],
        fit_ledger=[dict(status='PASS', steps=200, cuda_batches=200)])
    for task in [
            'binary', 'multiclass', 'regression', 'multilabel',
            'multilabel_ranking', 'forecasting', 'static_link', 'temporal_link'
    ]:
        for arm in ['normal', 'control']:
            operations = {
                name: metric(name)
                for name in [
                    'training', 'predict', 'repeat', 'partial',
                    'reloaded_prediction'
                ]
            }
            report['runs'].append(
                dict(
                    task=task,
                    arm=arm,
                    seed=0,
                    fit_id=task + arm,
                    status='MEASURED',
                    physical_fits=1,
                    checkpoint_sha256='1' * 64,
                    score=.5,
                    prediction_batches={
                        name: 1
                        for name in ['predict', 'repeat', 'partial', 'reload']
                    },
                    measurements=metric('complete_case'),
                    operation_measurements=operations,
                    timings={name: 1.
                             for name in operations},
                    work=copy.deepcopy(work),
                    selected_checkpoint_steps=200))
    return request, receipt, report


def test_current_result_needs_bound_mode_version_reference_and_restored_state(
):
    import copy
    run = controller()
    request, receipt, report = measured_fixture()
    run.validate_nightly_result(request, receipt, report)
    damaged = []
    for key, value in [('reference_only', True),
                       ('reference_sha256', '0' * 64)]:
        value_report = copy.deepcopy(report)
        value_report[key] = value
        damaged.append(value_report)
    value_report = copy.deepcopy(report)
    value_report['runtime']['version'] = 'other'
    damaged.append(value_report)
    value_report = copy.deepcopy(report)
    value_report['runs'][0]['work']['restored_tensors'] = 0
    damaged.append(value_report)
    value_report = copy.deepcopy(report)
    value_report['runs'][0]['work']['fit_ledger'] = []
    damaged.append(value_report)
    value_report = copy.deepcopy(report)
    del value_report['runs'][0]['measurements']
    damaged.append(value_report)
    for candidate in damaged:
        with pytest.raises(ValueError):
            run.validate_nightly_result(request, receipt, candidate)


def test_cancellation_transport_failure_preserves_the_original_failure(
        tmp_path, monkeypatch):
    import argparse
    import hashlib
    import json
    import zipfile
    run = controller()
    wheel = tmp_path / 'kumo_engine-1.2.3-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('kumo_engine-1.2.3.dist-info/METADATA',
                         'Name: kumo-engine\nVersion: 1.2.3\n')
    build = tmp_path / 'build.json'
    build.write_text(
        json.dumps(
            dict(version='1.2.3',
                 wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                 harness='a' * 40)))
    inputs = tmp_path / 'inputs.json'
    inputs.write_text('{}')
    args = argparse.Namespace(run_id='123-1',
                              source_sha='a' * 40,
                              wheel=wheel,
                              build_receipt=build,
                              input_spec=inputs,
                              reference_only=False,
                              output_dir=tmp_path / 'output')

    def invoke(*args, **kwargs):
        if args[:2] == ('ssm', 'send-command'):
            return {'Command': {'CommandId': 'command'}}
        if 'cancel.json' in str(args):
            raise RuntimeError('Cancellation upload denied')
        return {}

    monkeypatch.setattr(run, 'aws', invoke)

    def failed(*args):
        raise RuntimeError('Original VM failure')

    monkeypatch.setattr(run, 'wait_for_command', failed)
    with pytest.raises(RuntimeError, match='Original VM failure'):
        run.run(args)


def test_expired_aws_session_renews_once_and_retries_the_same_operation(
        monkeypatch):
    run = controller()
    calls = []

    def operation(*args, **kwargs):
        calls.append((args, kwargs))
        if len(calls) == 1:
            raise RuntimeError('ExpiredToken')
        return {'Status': 'Success'}

    renew = []
    monkeypatch.setattr(run.transport, 'aws', operation)
    monkeypatch.setattr(run, 'refresh_credentials', lambda: renew.append(True))
    assert run.aws('ssm', 'get-command-invocation', region='us-east-2') == {
        'Status': 'Success'
    }
    assert calls == [(('ssm', 'get-command-invocation'), {
        'region': 'us-east-2'
    })] * 2
    assert renew == [True]


def test_non_authentication_failure_does_not_renew_permissions(monkeypatch):
    run = controller()

    def denied(*args, **kwargs):
        raise RuntimeError('AccessDenied')

    monkeypatch.setattr(run.transport, 'aws', denied)
    monkeypatch.setattr(run, 'refresh_credentials',
                        lambda: pytest.fail('Unexpected role renewal'))
    with pytest.raises(RuntimeError, match='AccessDenied'):
        run.aws('ssm', 'send-command')


def test_cleanup_job_cancels_only_the_bound_run_and_confirms_the_same_command(
        tmp_path, monkeypatch):
    import argparse
    import json
    run = controller()
    assert hasattr(
        run, 'cancel_run'), 'Cancellation needs a separate cleanup entry point'
    request = dict(mode='nightly',
                   run_id='123-1',
                   source_sha='a' * 40,
                   nonce='original',
                   instance_id='i-06dd0e3c558780731')
    markers = []

    def invoke(*args, **kwargs):
        if args[:2] == ('s3api', 'get-object'):
            key = args[args.index('--key') + 1]
            value = request if key.endswith('/request.json') else {
                'CommandId': 'command'
            }
            Path(args[-1]).write_text(json.dumps(value))
        elif args[:2] == ('s3api', 'put-object'):
            markers.append(
                json.loads(Path(args[args.index('--body') + 1]).read_text()))
        return {}

    monkeypatch.setattr(run, 'aws', invoke)
    commands = []

    def wait(command, *args):
        commands.append(command)
        return dict(Status='Cancelled', ResponseCode=-1)

    monkeypatch.setattr(run, 'wait_for_command', wait)
    args = argparse.Namespace(run_id='123-1',
                              source_sha='a' * 40,
                              output_dir=tmp_path / 'cleanup')
    result = run.cancel_run(args)
    assert result['processes_ended'] is True and commands == ['command']
    assert markers == [
        dict(run_id='123-1',
             nonce='original',
             cancel=True,
             request_sha256=run.digest(args.output_dir / 'request.json'))
    ]


def test_controller_records_total_cli_time_even_when_the_remote_call_fails(
        tmp_path, monkeypatch):
    import json
    import time
    run = controller()
    output = tmp_path / 'timing'
    monkeypatch.setattr(run.sys, 'argv', [
        'controller', '--run-id', '123-1', '--source-sha', 'a' * 40,
        '--output-dir',
        str(output), '--cancel-only'
    ])
    monkeypatch.setattr(run.signal, 'signal', lambda *args: None)

    def failed(args):
        args.output_dir.mkdir()
        time.sleep(.01)
        raise RuntimeError('Remote call failed')

    monkeypatch.setattr(run, 'cancel_run', failed)
    with pytest.raises(RuntimeError, match='Remote call failed'):
        run.main()
    timing = json.loads((output / 'controller-job-time.json').read_text())
    assert timing['status'] == 'FAIL' and timing['seconds'] >= .01
    assert timing['scope'] == 'cancel controller CLI'


def test_unknown_command_id_needs_a_terminal_receipt_bound_to_the_same_request(
        tmp_path, monkeypatch):
    import json
    run = controller()
    request = dict(run_id='123-1',
                   source_sha='a' * 40,
                   nonce='original',
                   instance_id=run.transport.INSTANCE)
    good = dict(request,
                status='FAIL',
                processes_ended=True,
                request_sha256='b' * 64)
    responses = [dict(good, nonce='stale'), dict(good, status='RUNNING'), good]

    def invoke(*args, **kwargs):
        Path(args[-1]).write_text(json.dumps(responses.pop(0)))
        return {}

    monkeypatch.setattr(run, 'aws', invoke)
    monkeypatch.setattr(run.time, 'sleep', lambda *_: None)
    result = run.confirm_stopped(request, 'b' * 64, None, tmp_path)
    assert result['processes_ended'] is True and responses == []
    assert result['evidence'] == 'Bound VM receipt'
