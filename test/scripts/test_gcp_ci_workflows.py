# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import yaml

from io_utils import fsutil


def workflow(name):
    return yaml.safe_load(fsutil.read_text(f'.github/workflows/{name}.yml'))


def evaluate(expression, repository, event, source_repository, target):
    expression = expression.removeprefix('${{').removesuffix('}}').strip()
    expression = expression.replace('&&', 'and').replace('||', 'or')
    github = SimpleNamespace(
        repository=repository, event_name=event,
        event=SimpleNamespace(pull_request=SimpleNamespace(
            head=SimpleNamespace(repo=SimpleNamespace(
                full_name=source_repository)))))
    return eval(expression, {'__builtins__': {}}, {
        'github': github, 'matrix': SimpleNamespace(test_dir=target),
        'contains': lambda value, substring: substring in value,
    })


@pytest.mark.parametrize('repository', ['NVIDIA/kumo-engine', 'other/repo'])
@pytest.mark.parametrize('event,source', [
    ('pull_request', 'NVIDIA/kumo-engine'), ('pull_request', 'other/fork'),
    ('push', ''), ('merge_group', ''), ('workflow_dispatch', ''),
])
@pytest.mark.parametrize('target,gcp', [
    ('test/io_utils/', True), ('test/connector/', True),
    ('--splits 2 --group 1 test/rest/test_*.py', True),
    ('test/data/', False),
])
def test_general_auth_routing(repository, event, source, target, gcp):
    job = workflow('testing')['jobs']['pytest']
    eligible = (repository == 'NVIDIA/kumo-engine' and gcp
                and (event != 'pull_request' or source == repository))
    enabled = evaluate(job['env']['KUMO_RUN_GCP_TESTS'], repository,
                       event, source, target)
    environment = evaluate(job['environment']['name'], repository,
                           event, source, target)

    assert enabled == ('1' if eligible else '0')
    assert environment == ('gha-kumo-workflow-tests' if eligible
                           else 'azure-adls-tests')


def test_snowflake_has_no_google():
    job = workflow('testing_snowflake')['jobs']['pytest']
    assert not any('google-github-actions/auth' in step.get('uses', '')
                   for step in job['steps'])
    assert job['name'] == 'all-snowflake-tests'


def test_gcs_runs_every_change():
    document = workflow('testing_gcs')
    # YAML 1.1 interprets the workflow key "on" as True.
    triggers = document[True]
    assert 'pull_request' in triggers
    assert 'paths-ignore' not in triggers['push']
    job = document['jobs']['pytest-gcs']
    assert not job.get('continue-on-error', False)
    assert job['name'] == 'gcs-e2e-tests'
    steps = job['steps']
    assert not any('changed-files' in step.get('uses', '') for step in steps)
    run = next(step for step in steps if step['name'] == 'Run E2E tests')
    assert 'if' not in run
