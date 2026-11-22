# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test discovery, safe defaults and evidence receipts without model dependencies."""

import json
import venv

import pytest

from scripts import author_checks


@pytest.fixture
def profile(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'check.py').write_text('print("check completed")\n')
    item = {
        'description': 'A lightweight local check',
        'cwd': '.',
        'argv': ['{python}', 'check.py'],
        'tools': [],
        'modules': [],
        'parameters': {},
        'timeout_seconds': 5,
        'limits': 'A test fixture',
        'guide': 'test guide'
    }
    monkeypatch.setattr(author_checks, 'ROOT', root)
    monkeypatch.setattr(author_checks, 'load_catalog',
                        lambda: {'fixture': item})
    return root, item


def test_list_show_and_dry_run_create_nothing(profile, tmp_path, capsys):
    root, _ = profile
    assert author_checks.main(['list']) == 0
    assert author_checks.main(['show', 'fixture']) == 0
    output = tmp_path / 'reports'
    assert author_checks.main(
        ['run', 'fixture', '--dry-run', '--reports',
         str(output)]) == 0
    assert not output.exists()
    assert sorted(p.name for p in root.iterdir()) == ['check.py']
    assert 'check.py' in capsys.readouterr().out


def test_missing_dependencies_doctor_is_actionable(profile, capsys):
    _, item = profile
    item['modules'] = ['surely_missing_author_tools_dependency_xyz.child']
    assert author_checks.main(['doctor', 'fixture']) == 1
    output = json.loads(capsys.readouterr().out)
    assert any(not row['ok'] for row in output['checks'])


def test_existing_profile_propagates_exit_status_and_logs(profile, tmp_path):
    root, _ = profile
    output = tmp_path / 'success'
    assert author_checks.main(['run', 'fixture', '--reports', str(output)]) == 0
    receipt = json.loads((output / 'results.json').read_text())
    assert receipt['status'] == 'ok'
    assert 'check completed' in (output / 'run.log').read_text()
    (root / 'check.py').write_text('raise SystemExit(9)\n')
    output = tmp_path / 'failure'
    assert author_checks.main(['run', 'fixture', '--reports', str(output)]) == 1
    assert json.loads(
        (output / 'results.json').read_text())['run']['exit_code'] == 9


def test_run_requires_explicit_new_reports(profile, tmp_path):
    with pytest.raises(SystemExit):
        author_checks.main(['run', 'fixture'])
    with pytest.raises(SystemExit):
        author_checks.main(['run', 'fixture', '--reports', str(tmp_path)])


@pytest.mark.parametrize(
    'value', ['results=s3://bucket/data', 'other=/tmp/none', 'results='])
def test_reject_remote_unknown_or_empty_parameters(profile, value):
    _, item = profile
    item['parameters'] = {'results': 'new-dir'}
    with pytest.raises(ValueError):
        author_checks.parameters(item, [value])


def test_scale_profiles_require_budget_before_running(profile, tmp_path):
    _, item = profile
    item['disk_budget_required'] = True
    output = tmp_path / 'no-budget'
    with pytest.raises(SystemExit):
        author_checks.main(['run', 'fixture', '--reports', str(output)])
    assert not output.exists()
    assert author_checks.main([
        'run', 'fixture', '--reports',
        str(output), '--disk-budget-gib', '.001'
    ]) == 0


def test_disk_budget_checks_scratch_filesystem(profile, tmp_path, monkeypatch):
    _, item = profile
    item['parameters'] = {'scratch': 'new-dir'}
    item['disk_budget_required'] = True

    class FullDisk:
        free = 0

    monkeypatch.setattr(author_checks.shutil, 'disk_usage',
                        lambda path: FullDisk())
    with pytest.raises(SystemExit):
        author_checks.main([
            'run', 'fixture', '--reports',
            str(tmp_path / 'reports'), '--param',
            f'scratch={tmp_path / "scratch"}', '--disk-budget-gib', '1'
        ])
    assert not (tmp_path / 'scratch').exists()
    assert not (tmp_path / 'reports').exists()


def test_selected_virtual_environment_is_preserved(profile, tmp_path):
    root, _ = profile
    environment = tmp_path / 'venv'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    (root / 'check.py').write_text('import sys; print(sys.prefix)\n')
    output = tmp_path / 'venv-reports'
    assert author_checks.main([
        'run', 'fixture', '--python',
        str(environment / 'bin/python'), '--reports',
        str(output)
    ]) == 0
    assert (output / 'run.log').read_text().strip() == str(environment)


def test_catalog_paths_exist_and_profiles_do_not_provision():
    profiles = author_checks.load_catalog()
    for profile in profiles.values():
        assert (author_checks.ROOT / profile['cwd']).is_dir()
        assert (author_checks.ROOT / profile['guide']).is_file()
        command = ' '.join(profile['argv'])
        assert not any(
            x in command
            for x in ['sudo', 'mkfs', '--drop-all', 'benchmark_ingest.sh'])
        assert profile['timeout_seconds'] > 0
