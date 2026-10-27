# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import subprocess

import pytest
import yaml

from io_utils import fsutil
from tools.ci_test_profiles import GCP_AWS_MODULE, permits_target, profile_for

WORKFLOWS = (
    ('.github/workflows/testing.yml', 'pytest', 'general'),
    ('.github/workflows/testing_snowflake.yml', 'pytest', 'snowflake'),
    ('.github/workflows/testing_gcs.yml', 'pytest-gcs', 'gcp'),
)


@pytest.fixture
def workflows():
    return {
        profile: yaml.safe_load(fsutil.read_text(path))
        for path, _, profile in WORKFLOWS
    }


def test_existing_check_names_are_stable(workflows):
    assert workflows['general']['name'] == 'Testing and Coverage'
    assert workflows['snowflake']['name'] == 'Snowflake Testing and Coverage'
    assert workflows['gcp']['name'] == 'GCS Testing'
    assert workflows['general']['jobs']['pytest']['name'] == (
        'pytest (${{ matrix.test_dir }})')
    assert workflows['snowflake']['jobs']['pytest']['name'] == (
        'all-snowflake-tests (${{ matrix.test_name }})')
    gcp = workflows['gcp']['jobs']['pytest-gcs']
    assert gcp['name'] == 'gcs-e2e-tests (${{ matrix.test_dir }})'
    labels = [row['test_dir'] for row in gcp['strategy']['matrix']['include']]
    for existing in (
            'test/rest/integ/test_gcs_e2e.py',
            'test/rest/integ/test_bigquery_diskgraph_e2e.py',
            'test/rest/integ/test_bigquery_ingestion_diskgraph_e2e.py'):
        assert labels.count(existing) == 1


def test_gcp_rows_cover_registry_once(workflows):
    gcp = workflows['gcp']['jobs']['pytest-gcs']
    migrated = workflows['general']['jobs']['pytest-gcp']
    rows = (gcp['strategy']['matrix']['include']
            + migrated['strategy']['matrix']['include'])
    assert len(rows) == 7
    selected = []
    for row in rows:
        profile = profile_for(row['profile'])
        assert row['aws'] == ('aws' in profile.capabilities)
        for target in shlex.split(row['test_dir']):
            assert target in profile.targets
            assert permits_target(profile, target)
            selected.append(target)
        assert row['restricted_identities'] == (
            row['test_dir'] == 'test/connector/test_bigquery_connector.py')
    assert len(selected) == len(set(selected))
    assert set(selected) == set(profile_for('gcp').targets) | {GCP_AWS_MODULE}


def test_google_auth_cannot_block_other_providers(workflows):
    for name in ('general', 'snowflake'):
        job = workflows[name]['jobs']['pytest']
        assert job['continue-on-error'] is True
        assert job['env']['KUMO_RUN_GCP_TESTS'] == '0'
        assert not any(step.get('uses', '').startswith('google-github-actions/')
                       for step in job['steps'])
        commands = [step['run'] for step in job['steps']
                    if '-m pytest' in step.get('run', '')]
        assert len(commands) == 2
        for command in commands:
            assert '-m coverage run -m pytest' in command
            assert f'-p tools.pytest_test_profiles --test-profile {name}' in command
            assert 'tools.ci_test_profiles run' not in command
        assert 'gcp' not in str(job.get('needs', []))
    assert 'kumo-core-gpu-ci' not in str(workflows)


def test_selected_gcp_auth_failure_is_not_masked(workflows):
    job = workflows['gcp']['jobs']['pytest-gcs']
    assert not job.get('continue-on-error', False)
    assert job['strategy']['fail-fast'] is False
    steps = job['steps']
    setup = next(i for i, step in enumerate(steps)
                 if step.get('uses') == './.github/actions/setup')
    aws = next(i for i, step in enumerate(steps)
               if step.get('uses', '').startswith('aws-actions/'))
    google = next(i for i, step in enumerate(steps)
                  if step.get('uses', '').startswith('google-github-actions/'))
    tests = next(i for i, step in enumerate(steps)
                 if 'tools.ci_test_profiles run' in step.get('run', ''))
    assert setup < aws < google < tests
    assert steps[aws]['if'] == "steps.test-impact.outputs.run == 'true' && matrix.aws"
    for index in (setup, google, tests):
        assert steps[index]['if'] == "steps.test-impact.outputs.run == 'true'"
        assert not steps[index].get('continue-on-error', False)
    assert 'SNOWFLAKE' not in str(job)
    assert 'GOOGLE_CREDENTIALS_STR_JSON' not in str(job)
    assert not any('upload-artifact' in step.get('uses', '') for step in steps)


def test_oidc_permission_is_scoped_to_cloud_jobs(workflows):
    for _, job_id, profile in WORKFLOWS:
        workflow = workflows[profile]
        assert workflow['permissions'] == {'contents': 'read'}
        assert workflow['jobs'][job_id]['permissions']['id-token'] == 'write'
        events = workflow.get('on', workflow.get(True))
        assert 'paths-ignore' not in events['push']
        assert events['pull_request'] is None


@pytest.mark.parametrize('changed,missing,expected', [
    ('docs/development/testing.md', False, 'false'),
    ('tools/ci_test_profiles.py', False, 'true'),
    ('uv.lock', False, 'true'),
    ('docs/executable.py', False, 'true'),
    ('ui/application.ts', False, 'true'),
    ('test/path with spaces.py', False, 'true'),
    ('docs/development/testing.md', True, 'true'),
])
def test_workflow_impact_gate(workflows, tmp_path, changed, missing, expected):
    scripts = [
        next(step['run'] for step in workflows[profile]['jobs'][job]['steps']
             if step.get('id') == 'test-impact')
        for _, job, profile in WORKFLOWS
    ]
    assert len(set(scripts)) == 1
    output = str(tmp_path / 'output')
    fsutil.write_text(output, '')
    environment = {
        'PATH': '/usr/bin:/bin', 'PYTHONPATH': os.getcwd(),
        'GITHUB_OUTPUT': output, 'RUNNER_TEMP': str(tmp_path),
        'BASE_SHA': 'base', 'HEAD_SHA': 'head', 'CHANGED_PATH': changed,
    }
    git = ('git() { return 1; }\n' if missing else
           'git() { printf "%s\\0" "$CHANGED_PATH"; }\n')
    result = subprocess.run(['/bin/bash', '-e', '-c', git + scripts[0]],
                            env=environment, check=False, capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    assert fsutil.read_text(output).strip() == f'run={expected}'


def test_missing_coverage_has_no_upload_error(workflows):
    step = next(step for step in workflows['general']['jobs']['pytest']['steps']
                if step['name'] == 'Upload coverage artifact')
    assert "env.ARTIFACT_NAME_SAFE != ''" in step['if']
    assert "hashFiles(env.ARTIFACT_NAME_SAFE) != ''" in step['if']


def test_migrated_coverage_joins_existing_report(workflows):
    jobs = workflows['general']['jobs']
    job = jobs['pytest-gcp']
    assert job['permissions'] == {'contents': 'read', 'id-token': 'write'}
    assert not job.get('continue-on-error', False)
    assert job['strategy']['fail-fast'] is False
    assert job['needs'] == ['diskgraph-wheel']
    rows = job['strategy']['matrix']['include']
    assert len(rows) == 3
    assert all(row['profile'] == 'gcp' and not row['aws'] for row in rows)
    prefix = job['env']['COVERAGE_PREFIX']
    assert prefix == '.coverage-${{ github.run_id }}-gcp-${{ strategy.job-index }}'
    steps = job['steps']
    assert not any(step.get('uses', '').startswith('aws-actions/') for step in steps)
    setup = next(i for i, step in enumerate(steps)
                 if step.get('uses') == './.github/actions/setup')
    auth = next(i for i, step in enumerate(steps)
                if step.get('uses', '').startswith('google-github-actions/'))
    execute = next(i for i, step in enumerate(steps)
                   if 'tools.ci_test_profiles run' in step.get('run', ''))
    assert setup < auth < execute
    assert '--coverage-file "$COVERAGE_PREFIX"' in steps[execute]['run']
    for index in (setup, auth, execute):
        assert steps[index]['if'] == (
            "env.FORCE_RUN_TESTS == 'true' || steps.test-impact.outputs.run == 'true'")
        assert not steps[index].get('continue-on-error', False)
    upload = next(step for step in steps
                  if step.get('name') == 'Upload GCP coverage artifact')
    assert upload['with']['path'] == '${{ env.COVERAGE_PREFIX }}*'
    assert upload['with']['include-hidden-files'] is True
    assert 'hashFiles' in upload['if']
    coverage = jobs['coverage']
    assert coverage['needs'] == ['pytest', 'pytest-gcp']
    assert "needs.pytest-gcp.result == 'success'" in coverage['if']
    download = next(step for step in coverage['steps']
                    if step.get('name') == 'Download coverage data')
    assert download['with']['pattern'] == '.coverage-${{ github.run_id }}-*'
