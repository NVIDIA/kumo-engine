# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prove that the reload qualification harness rejects each failure mode.

These tests need no training. The harness itself runs against a wheel.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

_PATH = Path(__file__).with_name('desktop_reload_qualification.py')
_SPEC = importlib.util.spec_from_file_location('desktop_reload_qualification',
                                               _PATH)
drq = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = drq
_SPEC.loader.exec_module(drq)

pytestmark = pytest.mark.skipif(sys.platform == 'win32',
                                reason='The harness is POSIX only.')

ENTITIES = [0, 2, 4]
CLASSES = list(range(6))
ANCHOR = '2024-04-22 00:00:00'
PRODUCER = dict(entities=ENTITIES, classes=[], anchor=ANCHOR,
                time_column='TIMESTAMP')


def producer(case):
    classes = CLASSES if case.endswith('link') else []
    anchor = None if case == 'static_link' else ANCHOR
    return {**PRODUCER, 'classes': classes, 'anchor': anchor}


def frame(case, score=lambda entity, rank: 1.0 - rank / 10):
    """A correct output of the shared task checks for one case."""
    anchor = {} if case == 'static_link' else dict(
        ANCHOR_TIMESTAMP=pd.Timestamp(ANCHOR))
    if case == 'regression':
        return pd.DataFrame([
            dict(ENTITY=e, TARGET_PRED=e / 10, **anchor) for e in ENTITIES
        ])
    if case == 'binary':
        return pd.DataFrame([
            dict(ENTITY=e, False_PROB=1 - p, True_PROB=p, TARGET_PRED=p > .5,
                 **anchor) for e, p in zip(ENTITIES, (.2, .7, .9))
        ])
    return pd.DataFrame([
        dict(ENTITY=e, CLASS=(e + rank) % len(CLASSES), SCORE=score(e, rank),
             **anchor) for e in ENTITIES for rank in range(len(CLASSES))
    ])


FRAME_CASES = ['binary', 'regression', 'static_link', 'temporal_link']


@pytest.mark.parametrize('case', FRAME_CASES)
def test_identical_predictions_pass(case):
    result = drq.compare_frames(case, frame(case), frame(case),
                                producer=producer(case), atol=1e-4)
    assert result['max_abs_difference'] == 0 and result['decisions_equal']


@pytest.mark.parametrize('case', FRAME_CASES)
def test_every_tampered_prediction_fails(case):
    changes = drq.tampers(case, frame(case), 1e-4)
    expected = {
        'prediction_score_shift', 'prediction_entity_missing',
        'prediction_entity_duplicated'
    }
    if case != 'static_link':
        expected.add('timestamp_shifted')
    if case.endswith('link'):
        expected |= {'class_replaced', 'class_dropped', 'rank_swapped'}
    if case == 'binary':
        expected.add('decision_flipped')
    assert set(changes) == expected
    for name, change in changes.items():
        with pytest.raises(drq.QualificationError):
            drq.compare_frames(case, frame(case), change(frame(case)),
                               producer=producer(case), atol=1e-4)


def test_rank_change_allowed_only_inside_a_tie():
    tied = frame('static_link', score=lambda entity, rank: 1.0)
    swapped = drq.tampers('static_link', tied, 1e-4)['rank_swapped']
    drq.compare_frames('static_link', tied, swapped(tied.copy()),
                       producer=producer('static_link'), atol=1e-4)


def test_cases_are_the_shared_gpu_smoke_tasks():
    assert drq.CASES == tuple(drq.tasks.TASKS) and len(drq.CASES) == 8
    assert {drq.ARTIFACT_CASE, drq.FAISS_CASE} <= set(drq.CASES)
    assert set(drq.RANKING_CASES) <= set(drq.CASES)


def test_changed_or_missing_artifact_fails(tmp_path):
    (tmp_path / 'model').mkdir()
    weights = tmp_path / 'model' / 'weights.bin'
    weights.write_bytes(b'abc')
    hashes = drq.tree_hashes(tmp_path / 'model')
    weights.write_bytes(b'abd')
    with pytest.raises(drq.QualificationError, match='changed'):
        drq.require_same_tree(hashes, tmp_path / 'model', 'model')
    weights.unlink()
    with pytest.raises(drq.QualificationError):
        drq.require_same_tree(hashes, tmp_path / 'model', 'model')
    with pytest.raises(drq.QualificationError, match='missing'):
        drq.tree_hashes(tmp_path / 'absent')


def test_durable_directory_must_be_outside_cleanup_roots(tmp_path):
    with pytest.raises(drq.QualificationError, match='overlaps'):
        drq.require_disjoint(tmp_path / 'scratch' / 'durable',
                             [tmp_path / 'scratch'])
    drq.require_disjoint(tmp_path / 'durable', [tmp_path / 'scratch'])


def run(tmp_path, code, timeout=30):
    return drq.run_worker([sys.executable, '-c', code], cwd=tmp_path, env=None,
                          timeout=timeout, log_path=tmp_path / 'log')


def test_worker_exit_code_and_timeout_are_reported(tmp_path):
    assert run(tmp_path, 'import sys; sys.exit(3)')['exit_code'] == 3
    with pytest.raises(drq.QualificationError, match='exceeded'):
        run(tmp_path, 'import time; time.sleep(30)', timeout=1)


def test_worker_must_not_leave_descendants(tmp_path, monkeypatch):
    monkeypatch.setattr(drq, 'DESCENDANT_GRACE', 1.0)
    code = ('import subprocess, sys; subprocess.Popen([sys.executable, "-c",'
            ' "import time; time.sleep(30)"])')
    with pytest.raises(drq.QualificationError,
                       match=r'descendant processes running: \[.+\]'):
        run(tmp_path, code)
    # A helper that exits within the grace period is not a leak.
    code = ('import subprocess, sys; subprocess.Popen([sys.executable, "-c",'
            ' "import time; time.sleep(0.3)"])')
    assert run(tmp_path, code)['exit_code'] == 0


def receipt(tmp_path, **changes):
    body = dict(
        schema=drq.RECEIPT_SCHEMA, phase='consumer', case='regression',
        device='cpu', status='PASS', system='Linux',
        work=dict(device_types=['cpu'], fits=0, training_batches=0,
                  optimizer_steps=0, prediction_batches=2, retrieval={}))
    body.update(changes)
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(body))
    return path


def test_missing_skipped_or_failed_receipt_fails(tmp_path):
    kwargs = dict(phase='consumer', case='regression', device='cpu')
    with pytest.raises(drq.QualificationError, match='missing'):
        drq.read_receipt(tmp_path / 'absent.json', **kwargs)
    for status in ('SKIP', 'FAIL', 'RUNNING'):
        with pytest.raises(drq.QualificationError, match='status'):
            drq.read_receipt(receipt(tmp_path, status=status), **kwargs)
    with pytest.raises(drq.QualificationError, match='device'):
        drq.read_receipt(receipt(tmp_path, device='mps'), **kwargs)
    assert drq.read_receipt(receipt(tmp_path), **kwargs)['status'] == 'PASS'


def test_consumer_training_fails(tmp_path):
    body = json.loads(receipt(tmp_path).read_text())
    body['work']['optimizer_steps'] = 1
    with pytest.raises(drq.QualificationError, match='trained'):
        drq.require_device_work(body, phase='consumer', device='cpu')


@pytest.mark.parametrize('device', ['gpu', 'mps'])
def test_cpu_fallback_does_not_pass_a_gpu_case(tmp_path, device):
    body = json.loads(receipt(tmp_path, device=device).read_text())
    with pytest.raises(drq.QualificationError, match='ran on'):
        drq.require_device_work(body, phase='consumer', device=device)


def test_metal_case_requires_metal_search_and_apple_gpu(tmp_path):
    body = json.loads(
        receipt(tmp_path, device='mps', system='Darwin',
                case='static_link').read_text())
    body['work']['retrieval'] = {'predict': {'torch-bruteforce': 3}}
    with pytest.raises(drq.QualificationError, match='faiss-metal'):
        drq.require_retrieval(body, phase='consumer', case='static_link',
                              device='mps')
    body['work']['retrieval'] = {'predict': {'faiss-metal': 3}}
    body['metal'] = dict(mps_available=True, faiss_metal=True, faiss_num_gpus=0,
                         mps_fallback='0')
    with pytest.raises(drq.QualificationError, match='Apple GPU'):
        drq.require_retrieval(body, phase='consumer', case='static_link',
                              device='mps')
    body['metal'].update(faiss_num_gpus=1)
    drq.require_retrieval(body, phase='consumer', case='static_link',
                          device='mps')


def test_wheel_identity_is_parsed_from_the_file_name():
    assert drq.wheel_version(
        'kumo_engine-1.1.0+8b4942f-cp311-abi3-manylinux_2_34_aarch64.whl'
    ) == '1.1.0+8b4942f'
    with pytest.raises(drq.QualificationError):
        drq.wheel_version('other-1.0-py3-none-any.whl')


def write_summary(root, label, *, attempt=1, commit='abc', **changes):
    name, device, python = label.rsplit('-', 2)
    body = dict(
        label=label, commit=commit, attempt=attempt, device=device,
        python_version=python.removeprefix('py') + '.0', atol=drq.REQUIRED_ATOL,
        wheel=dict(installed_files_verified=10), cases={
            case:
            dict(status='PASS', negative_checks=[dict(name='x', detected=True)])
            for case in drq.CASES
        })
    body.update(changes)
    folder = root / f'{label}-{attempt}'
    folder.mkdir()
    (folder / 'summary.json').write_text(json.dumps(body))


def statuses(report, label):
    return {r['status'] for r in report['rows'] if r['label'] == label}


def test_report_marks_missing_configurations_unverified(tmp_path):
    label = 'linux-arm64-cpu-py3.12'
    write_summary(tmp_path, label)
    report = drq.build_report(tmp_path, commit='abc')
    rows = {(r['label'], r['case']): r for r in report['rows']}
    assert report['status'] == 'OPEN'
    assert rows[(label, 'regression')]['status'] == 'PASS'
    assert rows[('macos-15-mps-py3.12',
                 'static_link')]['status'] == ('UNVERIFIED')
    assert ('macos-14-mps-py3.12', 'static_link') not in rows
    assert len(rows) == len(drq.REQUIRED_TARGETS) * len(drq.PYTHONS) * len(
        drq.CASES)
    assert statuses(drq.build_report(tmp_path, commit='def'), label) == {'FAIL'}
    drq.write_report(report, tmp_path / 'out')
    assert 'UNVERIFIED' in (tmp_path / 'out' / 'matrix.md').read_text()


def test_report_uses_the_newest_attempt(tmp_path):
    label = 'linux-arm64-cpu-py3.12'
    write_summary(tmp_path, label, attempt=1,
                  cases={case: dict(status='FAIL')
                         for case in drq.CASES})
    write_summary(tmp_path, label, attempt=2)
    assert statuses(drq.build_report(tmp_path, commit='abc'), label) == {'PASS'}


@pytest.mark.parametrize('changes', [
    dict(atol=1.0),
    dict(device='gpu'),
    dict(python_version='3.13.1'),
    dict(wheel={}),
    dict(cases={case: dict(status='PASS')
                for case in drq.CASES}),
    dict(
        cases={
            case:
            dict(status='PASS',
                 negative_checks=[dict(name='x', detected=False)])
            for case in drq.CASES
        }),
])
def test_report_rejects_weakened_evidence(tmp_path, changes):
    label = 'linux-arm64-cpu-py3.12'
    write_summary(tmp_path, label, **changes)
    assert statuses(drq.build_report(tmp_path, commit='abc'), label) == {'FAIL'}


def test_installed_files_must_match_the_wheel(tmp_path, monkeypatch):
    import base64
    import hashlib
    import zipfile

    site = tmp_path / 'site'
    (site / 'pkg').mkdir(parents=True)
    (site / 'pkg' / 'mod.py').write_bytes(b'x = 1\n')
    digest = base64.urlsafe_b64encode(
        hashlib.sha256(b'x = 1\n').digest()).rstrip(b'=').decode()
    wheel = tmp_path / 'kumo_engine-1.0-cp311-abi3-linux_aarch64.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('pkg/mod.py', 'x = 1\n')
        archive.writestr(
            'kumo_engine-1.0.dist-info/RECORD',
            f'pkg/mod.py,sha256={digest},6\n'
            'kumo_engine-1.0.dist-info/RECORD,,\n')
    monkeypatch.setattr(drq.sysconfig, 'get_paths',
                        lambda: dict(purelib=str(site), platlib=str(site)))
    assert drq.verify_installed_wheel(wheel) == 1
    (site / 'pkg' / 'mod.py').write_bytes(b'x = 2\n')
    with pytest.raises(drq.QualificationError, match='differs'):
        drq.verify_installed_wheel(wheel)
    (site / 'pkg' / 'mod.py').unlink()
    with pytest.raises(drq.QualificationError, match='not installed'):
        drq.verify_installed_wheel(wheel)


def test_durable_directory_must_hold_only_the_exports(tmp_path):
    for name in drq.DURABLE_FILES:
        (tmp_path / name).write_bytes(b'')
    for name in ('model', 'snapshot'):
        (tmp_path / name).mkdir()
    drq.require_durable_layout(tmp_path)
    (tmp_path / 'extra.parquet').write_bytes(b'')
    with pytest.raises(drq.QualificationError, match='holds'):
        drq.require_durable_layout(tmp_path)


def test_every_product_rejection_has_an_expected_error():
    for name, pattern in drq.EXPECTED_ERRORS.items():
        assert pattern and name
    assert re.search(
        drq.EXPECTED_ERRORS['unavailable_device_mps'],
        'MisconfigurationException: `MPSAccelerator` can not run on your '
        'system since the accelerator is not available.')
    assert not re.search(drq.EXPECTED_ERRORS['faiss_missing'],
                         'ImportError: No module named torch')


def test_required_matrix_matches_the_workflow():
    workflow = Path(__file__).resolve().parents[3] / (
        '.github/workflows/kumo-engine-platform-smoke.yml')
    if not workflow.is_file():
        pytest.skip('Runs from the source checkout only.')
    yaml = pytest.importorskip('yaml')
    matrix = yaml.safe_load(
        workflow.read_text())['jobs']['reload']['strategy']['matrix']
    assert tuple(matrix['python']) == drq.PYTHONS
    assert tuple((t['name'], t['device'])
                 for t in matrix['target']) == drq.HOSTED_TARGETS
    assert not set(drq.EXTERNAL_TARGETS) & set(drq.HOSTED_TARGETS)


def test_report_merges_external_evidence(tmp_path):
    hosted, external = tmp_path / 'hosted', tmp_path / 'external'
    hosted.mkdir()
    external.mkdir()
    write_summary(hosted, 'linux-arm64-cpu-py3.12')
    write_summary(external, 'macos-15-mps-py3.12')
    report = drq.build_report([hosted, external], commit='abc')
    assert statuses(report, 'macos-15-mps-py3.12') == {'PASS'}
    assert statuses(report, 'linux-arm64-cpu-py3.12') == {'PASS'}


def test_macos_14_row_needs_the_clear_mps_error(tmp_path):
    label = 'macos-14-cpu-py3.12'
    write_summary(tmp_path, label)
    report = drq.build_report(tmp_path, commit='abc')
    rows = {r['case']: r for r in report['rows'] if r['label'] == label}
    assert rows[drq.ARTIFACT_CASE]['status'] == 'FAIL'
    assert 'unavailable_device_mps_old_macos' in rows[drq.ARTIFACT_CASE]['reason']
    assert rows['static_link']['status'] == 'PASS'

    check = dict(name='unavailable_device_mps_old_macos', detected=True)
    cases = {
        case: dict(status='PASS', negative_checks=[check])
        for case in drq.CASES
    }
    write_summary(tmp_path, label, attempt=2, cases=cases)
    assert statuses(drq.build_report(tmp_path, commit='abc'), label) == {'PASS'}


@pytest.mark.parametrize('system,release,mps,expected', [
    ('Linux', '', False, 'mps'),
    ('Darwin', '14.7.1', True, 'mps_old_macos'),
    ('Darwin', '15.0', True, None),
    ('Darwin', '26.1', True, None),
    ('Darwin', '14.7.1', False, 'mps_old_macos'),
    ('Darwin', '15.0', False, 'mps'),
    ('Darwin', '', True, None),
])
def test_unavailable_device_follows_the_macos_floor(monkeypatch, system,
                                                    release, mps, expected):
    torch = pytest.importorskip('torch')
    monkeypatch.setattr(drq.platform, 'system', lambda: system)
    monkeypatch.setattr(drq.platform, 'mac_ver', lambda: (release,
                                                          ('', '', ''), ''))
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    assert drq.unavailable_device() == expected


def test_old_macos_mps_error_names_the_floor():
    assert re.search(
        drq.EXPECTED_ERRORS['unavailable_device_mps_old_macos'],
        "UnsupportedAcceleratorError: accelerator='mps' uses Apple MPS, which "
        'requires macOS 15 or later, but this host runs macOS 14.7.1. Use '
        "accelerator='cpu' on this host.")
