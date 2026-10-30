# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep native source links out of portable evidence without weakening inventory."""

import hashlib
import json
import os
import runpy
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve(
).parents[2] / 'packages/kumo-engine/scripts/weekly_check.py'


def api():
    return runpy.run_path(str(SCRIPT))


def native_link(root):
    table = root / 'worker/scratch/prediction_tables' / ('a' *
                                                         32) / 'source-table'
    target = table / 'source/predictions.parquet/predictions.parquet'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'PAR1 frozen native prediction inputs')
    link = table / 'parts/part-00000.parquet'
    link.parent.mkdir()
    link.symlink_to('../source/predictions.parquet/predictions.parquet')
    return link, target


def test_materializes_only_native_internal_regular_file_bytes_and_records_proof(
        tmp_path):
    root = tmp_path / 'case'
    link, target = native_link(root)
    with pytest.raises(ValueError, match='symbolic'):
        api()['inventory'](root)
    receipt = api()['materialize_prediction_links'](root)
    assert receipt['status'] == 'PASS'
    assert len(receipt['links']) == 1
    assert receipt['links'][0]['sha256'] == hashlib.sha256(
        target.read_bytes()).hexdigest()
    assert receipt['links'][0][
        'link_target'] == '../source/predictions.parquet/predictions.parquet'
    assert not link.is_symlink()
    assert link.read_bytes() == target.read_bytes()
    assert 'prediction-link-portability.json' in api()['inventory'](root)


@pytest.mark.parametrize(
    'kind',
    ['external', 'directory', 'broken', 'cycle', 'escape', 'unexpected'])
def test_rejects_unsafe_links_before_materializing_anything(tmp_path, kind):
    root = tmp_path / 'case'
    link, target = native_link(root)
    original = os.readlink(link)
    bad = link.with_name('part-00001.parquet')
    if kind == 'unexpected':
        bad = root / 'unrelated-link'
        bad.symlink_to(target)
    elif kind == 'external':
        outside = tmp_path / 'outside.parquet'
        outside.write_bytes(b'external')
        target.unlink()
        target.symlink_to(outside)
        bad.symlink_to(original)
    elif kind == 'directory':
        target.unlink()
        target.mkdir()
        bad.symlink_to(original)
    elif kind == 'broken':
        target.unlink()
        bad.symlink_to(original)
    elif kind == 'cycle':
        target.unlink()
        target.symlink_to(link)
        bad.symlink_to(original)
    else:
        bad.symlink_to('../../../../../../outside.parquet')
    with pytest.raises(ValueError):
        api()['materialize_prediction_links'](root)
    assert link.is_symlink() and os.readlink(link) == original


def test_changed_target_bytes_fail_before_link_replacement(tmp_path):
    root = tmp_path / 'case'
    link, target = native_link(root)
    frozen = api()['prediction_source_links'](root)
    target.write_bytes(b'changed native input')
    with pytest.raises(ValueError, match='changed'):
        api()['materialize_prediction_links'](root, expected=frozen)
    assert link.is_symlink()


def test_portable_copy_preserves_original_failure_and_exact_reviewed_links(
        tmp_path):
    source = tmp_path / 'original'
    link, target = native_link(source)
    (source / 'weekly-case.json').write_text('{"status":"FAIL"}')
    expected = api()['prediction_source_links'](source)
    destination = tmp_path / 'portable'
    receipt = api()['copy_portable_case'](source, destination, expected)
    assert receipt['status'] == 'PASS'
    assert link.is_symlink()
    assert (source / 'weekly-case.json').read_text() == '{"status":"FAIL"}'
    assert (destination /
            link.relative_to(source)).read_bytes() == target.read_bytes()
    assert not (destination / link.relative_to(source)).is_symlink()
    api()['inventory'](destination)
    assert receipt['source_unchanged'] is True


def test_unreviewed_link_or_recursive_destination_prevents_copy(tmp_path):
    source = tmp_path / 'original'
    native_link(source)
    with pytest.raises(ValueError):
        api()['copy_portable_case'](source, tmp_path / 'copy', [])
    with pytest.raises(ValueError):
        api()['copy_portable_case'](source, source / 'recursive',
                                    api()['prediction_source_links'](source))
    assert not (source / 'recursive').exists()


def test_evidence_root_itself_cannot_redirect_to_another_tree(tmp_path):
    native_link(tmp_path / 'original')
    redirected = tmp_path / 'redirected'
    redirected.symlink_to(tmp_path / 'original', target_is_directory=True)
    with pytest.raises(ValueError):
        api()['materialize_prediction_links'](redirected)


def test_auxiliary_prediction_source_uses_the_same_bounded_rule(tmp_path):
    root = tmp_path / 'auxiliary'
    link, _ = native_link(root)
    (root / 'worker').rename(root / 'move')
    (root / 'move/scratch').rename(root / 'scratch')
    (root / 'move').rmdir()
    assert api()['materialize_prediction_links'](root)['materialized'] == 1
    assert not (root / link.relative_to(root / 'worker')).is_symlink()


def test_exact_released_custom_source_fixture_becomes_portable_without_source_edits(
        tmp_path):
    fixture = Path(__file__).with_name('weekly_prediction_source_fixture')
    proof = json.loads((fixture / 'proof.json').read_text())
    assert proof[
        'wheel_sha256'] == '1bc15ecb7696bce216b05aa883d6897d2b8ce4342c6c2f9d62b7cc28b314539e'
    assert proof['all_participating_python_sources_match_release'] is True
    source = tmp_path / 'released'
    shutil.copytree(fixture / 'worker', source / 'worker', symlinks=True)
    marker = source / 'worker/scratch/prediction_tables/released-source-fixture'
    marker.rename(marker.with_name('a' * 32))
    expected = api()['prediction_source_links'](source)
    assert len(expected) == 1 and expected[0]['bytes'] == 832
    assert expected[0]['sha256'] == proof['source']['sha256']
    with pytest.raises(ValueError, match='symbolic'):
        api()['inventory'](source)
    api()['copy_portable_case'](source, tmp_path / 'portable', expected)
    api()['inventory'](tmp_path / 'portable')
    assert (source / expected[0]['path']).is_symlink()


def test_target_changing_during_byte_copy_fails_without_replacing_link(
        tmp_path, monkeypatch):
    root = tmp_path / 'case'
    link, target = native_link(root)
    module = api()
    original_copy = shutil.copyfileobj

    def change_after_copy(source, destination):
        original_copy(source, destination)
        target.write_bytes(b'changed while copying')

    monkeypatch.setattr(module['shutil'], 'copyfileobj', change_after_copy)
    with pytest.raises(ValueError, match='changed'):
        module['materialize_prediction_links'](root)
    assert link.is_symlink()
    assert json.loads(
        (root /
         'prediction-link-portability.json').read_text())['status'] == 'FAIL'


def test_unreviewed_managed_path_drift_rejects_before_copy(tmp_path):
    source = tmp_path / 'source'
    native_link(source)
    expected = api()['prediction_source_links'](source)
    original = source / 'worker/scratch/prediction_tables' / ('a' * 32)
    original.rename(original.with_name('b' * 32))
    destination = tmp_path / 'portable'
    with pytest.raises(ValueError, match='changed'):
        api()['copy_portable_case'](source, destination, expected)
    assert not destination.exists()


def recovery_fixture(tmp_path):
    helpers = runpy.run_path(
        str(Path(__file__).with_name('test_engine_weekly_check.py')))
    spec = helpers['ready_case'](tmp_path)
    runtime = tmp_path / 'original-runtime'
    for relative in ('scripts/weekly_check.py', 'scripts/pr_check.py',
                     'test/wheel_gpu_learning.py',
                     'test/wheel_measurements.py'):
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCRIPT.parents[1] / relative, target)
        spec['inputs'][str(target)] = hashlib.sha256(
            target.read_bytes()).hexdigest()
    validator_path = runtime / 'scripts/weekly_check.py'
    validator = dict(path=str(validator_path),
                     sha256=spec['inputs'][str(validator_path)])
    source = tmp_path / 'failed-case'
    report_path = helpers['evidence_tree'](source, spec)
    link, target = native_link(source)
    (source /
     'worker/scratch/model_bundle/model/checkpoint').mkdir(parents=True)
    checkpoint = hashlib.sha256(
        (source / 'model/checkpoint').read_bytes()).hexdigest()
    result = json.loads(
        (source / 'result.json').read_text().replace('a' * 64, checkpoint))
    (source / 'result.json').write_text(json.dumps(result))
    (source / 'worker/result.json').write_text(json.dumps(result))
    failed = json.loads(report_path.read_text())
    failed.update(status='FAIL',
                  error='ValueError: Evidence cannot contain symbolic links')
    failed.pop('result')
    report_path.write_text(json.dumps(failed))
    audit = dict(
        source_case=str(source), fit_id=result['fit_id'],
        selected_checkpoint_sha256=checkpoint, links=[
            dict(path=link.relative_to(source).as_posix(),
                 link_target=os.readlink(link),
                 target_bytes=target.stat().st_size,
                 target_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        ])
    audit_path = tmp_path / 'link-audit.json'
    audit_path.write_text(json.dumps(audit))
    return source, validator, dict(
        path=str(audit_path),
        sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest())


def test_same_fit_recovery_uses_original_strict_validator_and_preserves_failure(
        tmp_path):
    source, validator, audit = recovery_fixture(tmp_path)
    original_failure = (source / 'weekly-case.json').read_bytes()
    recovery = runpy.run_path(str(SCRIPT.with_name('weekly_recover.py')))
    report = recovery['recover'](source, tmp_path / 'recovered', audit,
                                 validator)
    assert report['status'] == 'MEASURED'
    assert report['physical_fits'] == 1
    assert report['identity'] == json.loads(original_failure)['identity']
    assert (source / 'weekly-case.json').read_bytes() == original_failure
    assert (tmp_path / 'recovered/source-collection-failure.json'
            ).read_bytes() == original_failure
    proof = json.loads((tmp_path / 'recovered-recovery.json').read_text())
    assert proof['status'] == 'PASS' and proof['new_fits'] == proof[
        'new_predictions'] == 0
    assert proof['original_validator']['sha256'] == validator['sha256']


@pytest.mark.parametrize(
    'changed', ['validator', 'sibling', 'missing_sibling', 'worker', 'audit'])
def test_same_fit_recovery_refuses_foreign_or_incomplete_evidence_before_copy(
        tmp_path, changed):
    source, validator, audit = recovery_fixture(tmp_path)
    if changed == 'validator':
        validator['sha256'] = 'f' * 64
    elif changed == 'sibling':
        helper = Path(
            validator['path']).parents[1] / 'test/wheel_measurements.py'
        helper.write_text(helper.read_text() + '\n# changed sibling\n')
    elif changed == 'missing_sibling':
        (Path(validator['path']).parents[1] /
         'test/wheel_measurements.py').unlink()
    elif changed == 'worker':
        (source / 'worker/result.json').write_text('{}')
    else:
        data = json.loads(Path(audit['path']).read_text())
        data['links'][0]['target_bytes'] += 1
        Path(audit['path']).write_text(json.dumps(data))
        audit['sha256'] = hashlib.sha256(Path(
            audit['path']).read_bytes()).hexdigest()
    recovery = runpy.run_path(str(SCRIPT.with_name('weekly_recover.py')))
    with pytest.raises(ValueError):
        recovery['recover'](source, tmp_path / 'recovered', audit, validator)
    assert not (tmp_path / 'recovered').exists()
