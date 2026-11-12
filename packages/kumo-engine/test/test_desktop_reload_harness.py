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
CATALOG = list(range(8))
ANCHORS = {0: '2024-04-22', 2: '2024-04-15', 4: '2024-04-22'}


def link_frame(temporal=False):
    rows = []
    for entity in ENTITIES:
        for rank in range(drq.TOP_K):
            row = dict(ENTITY=entity, CLASS=(entity + rank) % len(CATALOG),
                       SCORE=1.0 - rank / 10, RANK=rank)
            if temporal:
                row['ANCHOR_TIMESTAMP'] = pd.Timestamp(ANCHORS[entity])
            rows.append(row)
    return pd.DataFrame(rows)


def numeric_frame():
    return pd.DataFrame({'ENTITY': ENTITIES, 'TARGET_PRED': [.1, .2, .3]})


def validate(case, frame):
    drq.validate_frame(
        case, frame, entities=ENTITIES,
        catalog=None if case == 'numerical' else CATALOG, anchors={
            str(k): v
            for k, v in ANCHORS.items()
        } if case == 'temporal_link' else None)


@pytest.mark.parametrize('case', drq.CASES)
def test_identical_predictions_pass(case):
    frame = (numeric_frame() if case == 'numerical' else link_frame(
        case == 'temporal_link'))
    validate(case, frame)
    result = drq.compare_predictions(case, frame, frame.copy(), atol=1e-4)
    assert result['exact'] and result['rows'] == len(frame)


def _swap_rank(frame):
    first = frame.index[(frame.ENTITY == 0) & (frame.RANK == 0)][0]
    last = frame.index[(frame.ENTITY == 0) & (frame.RANK == drq.TOP_K - 1)][0]
    frame.loc[[first, last], 'CLASS'] = frame.loc[[last, first],
                                                  'CLASS'].to_numpy()
    return frame


LINK_TAMPERS = {
    'score':
    lambda f: f.assign(SCORE=f.SCORE + 1e-2),
    'missing_entity':
    lambda f: f[f.ENTITY != 0],
    'foreign_entity':
    lambda f: f.assign(ENTITY=f.ENTITY.replace(0, 99)),
    'duplicate_row':
    lambda f: pd.concat([f, f.iloc[:1]]),
    'item_replaced':
    lambda f: f.assign(CLASS=f.CLASS.where(f.index != 0, 7)),
    'rank_swapped':
    _swap_rank,
    'top_k_truncated':
    lambda f: f[f.RANK != drq.TOP_K - 1],
    'item_outside_catalog':
    lambda f: f.assign(CLASS=f.CLASS.where(f.index != 0, 99)),
}


@pytest.mark.parametrize('case', ['static_link', 'temporal_link'])
@pytest.mark.parametrize('name', sorted(LINK_TAMPERS))
def test_wrong_link_prediction_fails(case, name):
    baseline = link_frame(case == 'temporal_link')
    after = LINK_TAMPERS[name](baseline.copy())
    with pytest.raises(drq.QualificationError):
        validate(case, after)
        drq.compare_predictions(case, baseline, after, atol=1e-4)


def test_wrong_timestamp_fails():
    baseline = link_frame(temporal=True)
    after = baseline.assign(ANCHOR_TIMESTAMP=baseline.ANCHOR_TIMESTAMP +
                            pd.Timedelta(days=1))
    with pytest.raises(drq.QualificationError, match='timestamps'):
        validate('temporal_link', after)
    with pytest.raises(drq.QualificationError, match='coordinates'):
        drq.compare_predictions('temporal_link', baseline, after, atol=1e-4)


def test_k_must_be_smaller_than_catalog():
    with pytest.raises(drq.QualificationError, match='smaller'):
        drq.validate_frame('static_link', link_frame(), entities=ENTITIES,
                           catalog=CATALOG[:drq.TOP_K])


def test_rank_change_allowed_only_inside_a_tie():
    baseline = link_frame().assign(SCORE=1.0)
    drq.compare_predictions('static_link', baseline,
                            _swap_rank(baseline.copy()), atol=1e-4)


@pytest.mark.parametrize('name', ['shift', 'missing', 'nan'])
def test_wrong_numerical_prediction_fails(name):
    baseline = numeric_frame()
    after = {
        'shift': baseline.assign(TARGET_PRED=baseline.TARGET_PRED + 1e-3),
        'missing': baseline.iloc[1:],
        'nan': baseline.assign(TARGET_PRED=float('nan')),
    }[name]
    with pytest.raises(drq.QualificationError):
        validate('numerical', after)
        drq.compare_predictions('numerical', baseline, after, atol=1e-4)


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


def test_worker_must_not_leave_descendants(tmp_path):
    code = ('import subprocess, sys; subprocess.Popen([sys.executable, "-c",'
            ' "import time; time.sleep(30)"])')
    with pytest.raises(drq.QualificationError, match='descendant'):
        run(tmp_path, code)


def receipt(tmp_path, **changes):
    body = dict(
        schema=drq.RECEIPT_SCHEMA, phase='consumer', case='numerical',
        device='cpu', status='PASS', system='Linux',
        work=dict(device_types=['cpu'], fits=0, training_batches=0,
                  optimizer_steps=0, prediction_batches=2, retrieval={}))
    body.update(changes)
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps(body))
    return path


def test_missing_skipped_or_failed_receipt_fails(tmp_path):
    kwargs = dict(phase='consumer', case='numerical', device='cpu')
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
    assert rows[(label, 'numerical')]['status'] == 'PASS'
    assert rows[('macos-15-mps-py3.12',
                 'static_link')]['status'] == ('UNVERIFIED')
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
                 for t in matrix['target']) == drq.REQUIRED_TARGETS
