# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Receipts must fail closed and process supervision must not leak workers."""

import json
import os
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest

from scripts import author_tools, benchmark_pair


@pytest.fixture
def plan(tmp_path):
    manifest = tmp_path / 'fixture.json'
    manifest.write_text('{"seed": 7, "rows": 4}\n')
    result = {
        'schema_version': 1,
        'workload_id': 'sum-four-v1',
        'scope': 'Four integer sums; only complete-command timing.',
        'context': {
            'seed': 7,
            'workers': 1,
            'cache_state': 'uncontrolled',
            'device': 'cpu'
        },
        'timeout_seconds': 5,
        'pairs': 2,
        'warmups': 0,
        'inputs': [{
            'path': str(manifest)
        }],
        'correctness_expect': {
            'sum': 10,
            'cases': 1
        }
    }
    for side in benchmark_pair.SIDES:
        cwd = tmp_path / side
        cwd.mkdir()
        script = cwd / 'work.py'
        script.write_text(
            'import json, pathlib, sys\n'
            'answer = sum([1, 2, 3, 4])\n'
            'assert answer == 10\n'
            'if len(sys.argv) > 1:\n'
            '    pathlib.Path(sys.argv[1]).write_text(json.dumps({"sum":answer,"cases":1}))\n'
            'print(answer)\n')
        result[side] = {
            'cwd': str(cwd),
            'python': sys.executable,
            'argv': ['{python}', 'work.py'],
            'correctness_argv': ['{python}', 'work.py', '{result}'],
            'artifacts': [str(script)]
        }
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(result))
    return path, result


def test_paired_order_raw_logs_and_scoped_summary(plan, tmp_path):
    path, _ = plan
    output = tmp_path / 'report'
    assert benchmark_pair.main([str(path), '--output', str(output)]) == 0
    result = json.loads((output / 'results.json').read_text())
    assert result['status'] == 'ok'
    measured = [r for r in result['runs'] if r['phase'] == 'measure']
    assert [r['side'] for r in measured
            ] == ['baseline', 'candidate', 'candidate', 'baseline']
    assert all((output / r['log']).is_file() for r in result['runs'])
    assert len(result['summary']['paired_ratios']) == 2
    assert result['variants']['baseline']['correctness_receipt'] == {
        'sum': 10,
        'cases': 1
    }
    summary = (output / 'summary.md').read_text()
    assert 'complete command wall time' in summary
    assert 'No automatic performance gate' in summary
    assert 'not stable performance evidence' in summary


@pytest.mark.parametrize('mutate', [
    lambda p: p.update(timeout_seconds=0),
    lambda p: p.update(pairs=True),
    lambda p: p.update(warmups=-1),
    lambda p: p.update(inputs=[]),
    lambda p: p.update(correctness_expect={}),
    lambda p: p.update(context=[]),
    lambda p: p['context'].pop('cache_state'),
    lambda p: p['baseline'].update(argv='python work.py'),
    lambda p: p['baseline'].update(correctness_argv=['{python}', 'work.py']),
    lambda p: p['baseline'].update(
        env={'AWS_SECRET_ACCESS_KEY': 'never-record-this'}),
    lambda p: p['inputs'][0].update(sha256='0' * 64),
    lambda p: p.update(candidate=[]),
])
def test_invalid_plan_cannot_start(plan, tmp_path, mutate):
    path, data = plan
    mutate(data)
    path.write_text(json.dumps(data))
    output = tmp_path / 'invalid-report'
    with pytest.raises(SystemExit) as error:
        benchmark_pair.main([str(path), '--output', str(output)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize('payload',
                         ['[]', '{"schema_version": NaN}', '{', 'true'])
def test_malformed_json_is_a_cli_error(tmp_path, payload):
    path = tmp_path / 'invalid.json'
    path.write_text(payload)
    with pytest.raises(SystemExit) as error:
        benchmark_pair.main([str(path), '--output', str(tmp_path / 'report')])
    assert error.value.code == 2


@pytest.mark.parametrize(
    'change', ['nonzero', 'wrong-answer', 'missing-receipt', 'type-mismatch'])
def test_checker_failure_has_no_timing_summary(plan, tmp_path, change):
    path, data = plan
    if change == 'nonzero':
        data['candidate']['correctness_argv'] = [
            '{python}', '-c', 'raise SystemExit(7)', '{result}'
        ]
    elif change == 'wrong-answer':
        data['correctness_expect']['sum'] = 11
    elif change == 'type-mismatch':
        data['correctness_expect']['cases'] = True
    else:
        data['candidate']['correctness_argv'] = [
            '{python}', '-c', 'pass', '{result}'
        ]
    path.write_text(json.dumps(data))
    output = tmp_path / 'failed-report'
    assert benchmark_pair.main([str(path), '--output', str(output)]) == 1
    result = json.loads((output / 'results.json').read_text())
    assert result['status'] == 'failed'
    assert 'summary' not in result
    assert not any(r['phase'] == 'measure' for r in result['runs'])


def test_timing_failure_invalidates_partial_samples(plan, tmp_path):
    path, data = plan
    data['candidate']['argv'] = ['{python}', '-c', 'raise SystemExit(8)']
    path.write_text(json.dumps(data))
    output = tmp_path / 'failed-report'
    assert benchmark_pair.main([str(path), '--output', str(output)]) == 1
    result = json.loads((output / 'results.json').read_text())
    assert result['status'] == 'failed'
    assert 'summary' not in result
    assert any(r['exit_code'] == 8 for r in result['runs'])


def test_refuse_existing_or_in_worktree_output(plan, tmp_path):
    path, data = plan
    sentinel = tmp_path / 'existing'
    sentinel.mkdir()
    (sentinel / 'keep').write_text('retained')
    for output in [sentinel, Path(data['baseline']['cwd']) / 'nested-report']:
        with pytest.raises(SystemExit):
            benchmark_pair.main([str(path), '--output', str(output)])
    assert (sentinel / 'keep').read_text() == 'retained'


def test_symlink_venv_python_keeps_its_environment(plan, tmp_path):
    path, data = plan
    environment = tmp_path / 'venv'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    data['baseline']['python'] = str(environment / 'bin/python')
    path.write_text(json.dumps(data))
    prepared = benchmark_pair.read_plan(path)
    assert prepared['baseline']['python'] == str(environment / 'bin/python')
    output = tmp_path / 'venv-report'
    assert benchmark_pair.benchmark(prepared, output) == 0
    probe = json.loads((output / 'interpreter-0-baseline.log').read_text())
    assert probe['prefix'] == str(environment)
    assert probe['prefix'] != probe['base_prefix']


def test_changed_input_invalidates_run(plan, tmp_path):
    path, data = plan
    fixture = data['inputs'][0]['path']
    data['candidate']['argv'] = [
        '{python}', '-c',
        'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("changed")',
        fixture
    ]
    path.write_text(json.dumps(data))
    output = tmp_path / 'changed-report'
    assert benchmark_pair.main([str(path), '--output', str(output)]) == 1
    result = json.loads((output / 'results.json').read_text())
    assert 'changed during' in result['error']
    assert 'summary' not in result


def test_timeout_stops_descendant_process(tmp_path):
    pid_file = tmp_path / 'pid'
    command = [
        sys.executable, '-c', 'import pathlib, subprocess, sys, time; '
        'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
        'pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)',
        str(pid_file)
    ]
    row = author_tools.run_process(command, tmp_path, tmp_path / 'timeout.log',
                                   .3)
    assert row['status'] == 'timeout'
    pid = int(pid_file.read_text())
    # Linux can briefly retain a killed orphan as a zombie; it cannot execute.
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state = Path(f'/proc/{pid}/stat')
        if not state.exists() or state.read_text().split()[2] == 'Z':
            break
        time.sleep(.02)
    else:
        os.kill(pid, 9)
        pytest.fail('Timeout leaked a running descendant')


def test_successful_parent_cannot_leave_background_work(tmp_path):
    command = [
        sys.executable, '-c',
        'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])'
    ]
    result = author_tools.run_process(command, tmp_path,
                                      tmp_path / 'background.log', 3)
    assert result['status'] == 'failed'
    assert 'background' in result['error']


def test_normal_multiprocessing_cleanup_is_not_a_leaked_worker(tmp_path):
    for index in range(3):
        result = author_tools.run_process([
            sys.executable, '-c', 'from multiprocessing import shared_memory; '
            'block=shared_memory.SharedMemory(create=True,size=8); block.close()'
        ], tmp_path, tmp_path / f'shared-{index}.log', 5)
        assert result['status'] == 'ok', result


def test_output_cannot_be_a_sibling_inside_the_actual_git_root(plan, tmp_path):
    path, data = plan
    root = Path(data['baseline']['cwd'])
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    nested = root / 'package'
    nested.mkdir()
    data['baseline']['cwd'] = str(nested)
    path.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        benchmark_pair.main([str(path), '--output', str(root / 'reports')])


def test_sigterm_cancels_cli_workers_and_keeps_failed_receipt(plan, tmp_path):
    import signal
    path, data = plan
    pid_file = tmp_path / 'terminated-worker-pid'
    data['timeout_seconds'] = 30
    data['baseline']['argv'] = [
        '{python}', '-c',
        'import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text(str(__import__("os").getpid())); time.sleep(60)',
        str(pid_file)
    ]
    path.write_text(json.dumps(data))
    output = tmp_path / 'terminated-report'
    cli = Path(benchmark_pair.__file__)
    process = subprocess.Popen(
        [sys.executable,
         str(cli), str(path), '--output',
         str(output)])
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert pid_file.is_file()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 1
        assert json.loads(
            (output / 'results.json').read_text())['status'] == 'failed'
        assert not author_tools.group_running(int(pid_file.read_text()))
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_tokenizer_parallelism_is_a_supported_environment_knob(plan):
    path, data = plan
    data['baseline']['env'] = {'TOKENIZERS_PARALLELISM': 'false'}
    path.write_text(json.dumps(data))
    assert benchmark_pair.read_plan(path)['baseline']['env'] == {
        'TOKENIZERS_PARALLELISM': 'false'
    }
