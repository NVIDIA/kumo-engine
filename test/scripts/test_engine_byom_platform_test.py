# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared customer-model evidence must prove every requested device and version."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'packages/kumo-engine/test/byom_platform_test.py'
SPEC = importlib.util.spec_from_file_location('byom_platform_test', SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def phases(platform, accelerator):
    devices = ('cuda',) if accelerator == 'gpu' else ('cpu',)
    versions = [dict(version=str(i), run='run-' + str(i), device=device)
                for i, device in enumerate(d for d in devices for _ in range(2))]
    receipt = dict(status='passed', returncode=0,
                   workload=dict(sha256=hashlib.sha256(runner.workload(platform).read_bytes()).hexdigest()),
                   result=dict(versions=versions))
    return dict((phase, copy.deepcopy(receipt)) for phase in runner.PHASES)


@pytest.mark.parametrize('platform', runner.PLATFORMS)
@pytest.mark.parametrize('accelerator', runner.ACCELERATORS)
def test_every_platform_uses_the_same_device_and_version_contract(monkeypatch, platform, accelerator):
    monkeypatch.setattr(runner, 'phase_receipts', lambda *_: phases(platform, accelerator))
    result = runner.verify(platform, accelerator, Path('/unused'))
    expected = dict(gpu='passed') if accelerator == 'gpu' else dict(cpu='passed')
    assert result['status'] == 'passed'
    assert result['devices'] == expected


@pytest.mark.parametrize('fault', ['cpu-only', 'missing-phase', 'default-kumo',
                                   'wrong-source', 'wrong-version', 'duplicates', 'nonzero-exit'])
def test_gpu_evidence_cannot_pass_with_missing_or_substituted_work(monkeypatch, fault):
    evidence = phases('sagemaker', 'gpu')
    if fault == 'cpu-only':
        for phase in ('train', 'reload'):
            for version in evidence[phase]['result']['versions']:
                version['device'] = 'cpu'
    elif fault == 'missing-phase':
        evidence.pop('reload')
    elif fault == 'default-kumo':
        evidence = {'fit': evidence['train'], 'score': evidence['reload']}
    elif fault == 'wrong-source':
        evidence['train']['workload']['sha256'] = '0' * 64
    elif fault == 'wrong-version':
        evidence['reload']['result']['versions'][0]['run'] = 'another-run'
    elif fault == 'duplicates':
        evidence['reload']['result']['versions'][1] = evidence['reload']['result']['versions'][0]
    elif fault == 'nonzero-exit':
        evidence['train']['returncode'] = 1
    monkeypatch.setattr(runner, 'phase_receipts', lambda *_: evidence)
    with pytest.raises(ValueError):
        runner.verify('sagemaker', 'gpu', Path('/unused'))


def test_default_plan_contains_all_six_cases_and_no_cloud_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'a' * 40)
    args = SimpleNamespace(platform='all', accelerator='all', execute=False,
                           verify_existing=False, wheel=tmp_path / 'engine.whl',
                           workload_wheel=[tmp_path / 'diskgraph.whl', tmp_path / 'kumo_pytorch.whl'],
                           candidate=None, cpu_cluster_id=None, gpu_cluster_id=None,
                           output_dir=tmp_path / 'matrix')
    result = runner.run(args)
    assert set(result['cases']) == {p + '-' + a for p in runner.PLATFORMS for a in runner.ACCELERATORS}
    assert result['status'] == 'planned'
    for name in result['cases']:
        plan = json.loads((args.output_dir / name / 'plan.json').read_text())
        command = plan[1] if name.startswith('snowflake') else plan[0]
        assert str(runner.workload(name.rsplit('-', 1)[0])) in command
        assert '--accelerator' in command
        assert command[command.index('--accelerator') + 1] == name.rsplit('-', 1)[1]
    snowflake = json.loads((args.output_dir / 'snowflake-gpu/plan.json').read_text())
    assert 'V2.9-GPU-PY3.12' in snowflake[2]


def test_missing_gpu_cluster_fails_before_any_case_is_submitted(tmp_path):
    args = SimpleNamespace(platform='all', accelerator='all', execute=True,
                           verify_existing=False, wheel=tmp_path / 'engine.whl',
                           workload_wheel=[tmp_path / 'diskgraph.whl', tmp_path / 'kumo_pytorch.whl'],
                           cpu_cluster_id='0101-010101-test', gpu_cluster_id=None)
    with pytest.raises(ValueError, match='gpu Databricks cluster'):
        runner.run(args)


def test_interrupted_matrix_never_reports_unrun_cases_as_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'a' * 40)
    monkeypatch.setattr(runner.subprocess, 'call', lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    args = SimpleNamespace(platform='all', accelerator='all', execute=True,
                           verify_existing=False, wheel=tmp_path / 'engine.whl',
                           workload_wheel=[tmp_path / 'diskgraph.whl', tmp_path / 'kumo_pytorch.whl'],
                           candidate=tmp_path / 'candidate.json',
                           cpu_cluster_id='0101-010101-test', gpu_cluster_id='0202-020202-test',
                           output_dir=tmp_path / 'matrix')
    with pytest.raises(KeyboardInterrupt):
        runner.run(args)
    report = json.loads((args.output_dir / 'summary.json').read_text())
    assert report['status'] == 'running'
    assert len(report['cases']) == 6
    assert all(case['status'] == 'planned' for case in report['cases'].values())


def test_distinct_gpu_runs_cannot_reuse_one_registry_version(monkeypatch):
    evidence = phases('sagemaker', 'gpu')
    for phase in ('train', 'reload'):
        for index, version in enumerate(evidence[phase]['result']['versions']):
            version.update(version='one-version', run='run-' + str(index))
    monkeypatch.setattr(runner, 'phase_receipts', lambda *_: evidence)
    with pytest.raises(ValueError, match='duplicates'):
        runner.verify('sagemaker', 'gpu', Path('/unused'))


@pytest.mark.parametrize('fault', ['parent', 'parent-error', 'cleanup', 'missing-cleanup', 'verdict'])
def test_passed_phases_cannot_override_failed_attempt_or_cleanup(tmp_path, fault):
    report = dict(status='blocked', verdict=dict(engine_checks='passed'),
                  phases=phases('databricks', 'cpu'))
    cleanup = dict(status='passed')
    if fault == 'parent':
        report['status'] = 'failed'
    elif fault == 'parent-error':
        report['error'] = 'Attempt cleanup failed'
    elif fault == 'cleanup':
        cleanup['status'] = 'failed'
    elif fault == 'verdict':
        report['verdict']['engine_checks'] = 'incomplete'
    (tmp_path / 'summary.json').write_text(json.dumps(report))
    if fault != 'missing-cleanup':
        (tmp_path / 'cleanup.json').write_text(json.dumps(cleanup))
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.verify('databricks', 'cpu', tmp_path)


@pytest.mark.parametrize('platform,accelerator,count', [
    ('all', 'all', 6), ('all', 'cpu', 3), ('sagemaker', 'gpu', 1),
])
def test_actions_configuration_selects_every_requested_case(tmp_path, monkeypatch, platform, accelerator, count):
    import sys

    import yaml

    workflow = yaml.safe_load((ROOT / '.github/workflows/kumo-engine-byom-platforms.yml').read_text())
    step = workflow['jobs']['configuration']['steps'][1]
    script = step['run'].split("python3 - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    calls = []
    monkeypatch.setitem(sys.modules, 'check_environment', SimpleNamespace(main=lambda **kw: calls.append(kw)))
    for name, value in dict(PLATFORM=platform, ACCELERATOR=accelerator,
                            CPU_CLUSTER='0101-010101-cpu', GPU_CLUSTER='0202-020202-gpu',
                            GITHUB_OUTPUT=str(tmp_path / 'output')).items():
        monkeypatch.setenv(name, value)
    exec(compile(script, '<matrix-configuration>', 'exec'), {})
    matrix = json.loads((tmp_path / 'output').read_text().split('=', 1)[1])
    assert len(matrix['include']) == count
    expected_platforms = runner.PLATFORMS if platform == 'all' else (platform,)
    expected_accelerators = runner.ACCELERATORS if accelerator == 'all' else (accelerator,)
    assert {(c['platform'], c['accelerator']) for c in matrix['include']} == {
        (p, a) for p in expected_platforms for a in expected_accelerators}
    assert {c['environment'] for c in calls} == {p + '-mlops-test' for p in expected_platforms}


@pytest.mark.parametrize('fault', ['checksum', 'exit', 'missing-exit'])
def test_snowflake_phase_evidence_requires_receipt_hashes_and_worker_exits(tmp_path, fault):
    execution = tmp_path
    stage = execution / 'lifecycle'
    download = stage / 'download'
    download.mkdir(parents=True)
    body = phases('snowflake', 'gpu')
    summary = dict(status='blocked', phases=copy.deepcopy(body))
    if fault == 'exit':
        summary['phases']['reload']['returncode'] = 7
    elif fault == 'missing-exit':
        summary['phases']['reload'].pop('returncode')
    documents = {'summary.json': summary, **{p + '.json': value for p, value in body.items()}}
    names, hashes = {}, {}
    for index, (logical, value) in enumerate(documents.items()):
        name = 'evidence-' + str(index) + '.json'
        data = json.dumps(value).encode()
        (download / name).write_bytes(data)
        names[name] = 'lifecycle/' + logical
        hashes[name] = hashlib.sha256(data).hexdigest()
    if fault == 'checksum':
        hashes['evidence-0.json'] = '0' * 64
    (execution / 'summary.json').write_text(json.dumps(
        dict(status='blocked', stages=dict(preflight=dict(status='passed')))))
    (stage / 'report.json').write_text(json.dumps(dict(
        status='blocked', cleanup=dict(status='passed'),
        notebook_receipt=dict(logical_files=names, artifacts=hashes))))
    with pytest.raises(ValueError, match='checksum|exit code'):
        runner.verify('snowflake', 'gpu', execution)


def test_reusable_matrix_receives_only_declared_platform_credentials():
    import yaml

    keys = {'KUMO_DATABRICKS_TEST_HOST', 'KUMO_DATABRICKS_TEST_PAT',
            'KUMO_DATABRICKS_TEST_USER', 'SNOWFLAKE_TEST_CONFIG',
            'SNOWFLAKE_PRIVATE_KEY', 'SNOWFLAKE_PRIVATE_KEY_PASSPHRASE'}
    caller = yaml.safe_load((ROOT / '.github/workflows/kumo-engine-sagemaker-notebook.yml').read_text())
    workflow = yaml.safe_load((ROOT / '.github/workflows/kumo-engine-byom-platforms.yml').read_text())
    declared = workflow[True]['workflow_call']['secrets']
    forwarded = caller['jobs']['byom-platforms']['secrets']
    assert set(declared) == keys
    assert forwarded == {key: '${{ secrets.' + key + ' }}' for key in keys}
    assert workflow['jobs']['lifecycle']['environment'] == '${{ matrix.environment }}'
