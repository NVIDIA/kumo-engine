# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select CI cases and describe their resource and process requirements."""

import json
import os
import re
from pathlib import Path

import pytest

GROUPS = ('public', 'aws', 'aws-public-api', 'gcp', 'gcp-e2e', 'azure',
          'snowflake', 'databricks')
PROFILE_ENV = {
    'diskgraph': {},
    'gcp': {
        'IS_GCP_DEPLOY': '1'
    },
    'databricks-pat': {
        'DATABRICKS_CLIENT_ID': '',
        'DATABRICKS_CLIENT_SECRET': ''
    },
    'databricks-oauth': {
        'DATABRICKS_TOKEN': ''
    },
}
GATES = {'gcp-e2e': 'CI_RUN_GCP_E2E', 'snowflake-e2e': 'CI_RUN_SNOWFLAKE_E2E'}


def execution_profile(*profiles):
    """Attach process requirements to a fixture without resolving it."""
    def decorate(func):
        func.__kumo_ci_profiles__ = profiles
        return func

    return decorate


def annotated_functions(item):
    yield item.obj
    # Fixture marks do not propagate to tests. Keep pytest-internal inspection
    # here, and read metadata without resolving any fixture.
    for name in item.fixturenames:
        for definition in item._fixtureinfo.name2fixturedefs.get(name, ()):
            yield definition.func


def resources_for_item(item):
    resources = set()
    for func in annotated_functions(item):
        resources.update(getattr(func, '__kumo_test_resources__', ()))
    for marker in item.iter_markers('skipif'):
        if marker.kwargs.get('kumo_resource'):
            resources.add(marker.kwargs['kumo_resource'])
    for marker in item.iter_markers('private_resource'):
        resources.update(marker.args)
    unknown = resources - {'aws', 'gcp', 'azure', 'snowflake', 'databricks'}
    if unknown:
        raise pytest.UsageError(
            f'Unknown resources for {item.nodeid}: {sorted(unknown)}')
    return resources


def profiles_for_item(item):
    profiles = set()
    for func in annotated_functions(item):
        profiles.update(getattr(func, '__kumo_ci_profiles__', ()))
    for marker in item.iter_markers('ci_profile'):
        profiles.update(marker.args)
    if profiles - PROFILE_ENV.keys():
        raise pytest.UsageError(
            f'Unknown CI profiles for {item.nodeid}: {sorted(profiles)}')
    if {'databricks-pat', 'databricks-oauth'} <= profiles:
        raise pytest.UsageError(
            f'Conflicting Databricks auth profiles: {item.nodeid}')
    return sorted(profiles)


def group_for_item(item):
    resources = resources_for_item(item)
    if not resources:
        return 'public'
    provider = next(p
                    for p in ('snowflake', 'databricks', 'gcp', 'azure', 'aws')
                    if p in resources)
    marker = item.get_closest_marker('ci_group')
    group = marker.args[0] if marker else provider
    if group not in GROUPS or group.split('-', 1)[0] != provider:
        raise pytest.UsageError(f'Invalid CI group {group} for {item.nodeid}')
    return group


def pytest_addoption(parser):
    parser.addoption('--resource-group', choices=GROUPS, default=None,
                     help='Select public tests or one private CI group')
    parser.addoption(
        '--resource-manifest', type=Path,
        help='Write selected node IDs, resources and process profiles')
    parser.addoption(
        '--private-discovery', action='store_true',
        help='Apply private CI provider change gates during collection')
    parser.addoption('--require-resource-credentials', action='store_true',
                     help='Fail private CI on missing resource credentials')


def pytest_configure(config):
    for marker in ('private_resource(provider): uses a live private resource',
                   'ci_profile(name): requires a declared process environment',
                   'ci_group(name): assigns resource cases to a CI workload',
                   'ci_gate(name): follows a provider E2E changed-file gate'):
        config.addinivalue_line('markers', marker)
    config._resource_credential_skips = []


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    group = config.getoption('--resource-group')
    if group is None:
        return
    selected, deselected = [], []
    for item in items:
        enabled = True
        if config.getoption('--private-discovery'):
            for marker in item.iter_markers('ci_gate'):
                gate = marker.args[0]
                if gate not in GATES:
                    raise pytest.UsageError(f'Unknown CI gate: {gate}')
                enabled &= os.environ.get(GATES[gate]) != 'false'
        (selected if enabled and group_for_item(item) == group else
         deselected).append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)


def pytest_collection_finish(session):
    destination = session.config.getoption('--resource-manifest')
    if destination:
        manifest = {}
        for item in session.items:
            profiles = profiles_for_item(item)
            manifest[item.nodeid] = {
                'group':
                group_for_item(item),
                'resources':
                sorted(resources_for_item(item)),
                'profiles':
                profiles,
                'isolated_module':
                item.nodeid.split('::', 1)[0]
                if 'diskgraph' in profiles else '',
            }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2) + '\n')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if report.when == 'call':
        reporter = item.config.pluginmanager.get_plugin('terminalreporter')
        if reporter:
            for line in report.capstdout.splitlines():
                timing = re.match(
                    r'^CI_PHASE public-api [A-Za-z0-9_]+ elapsed_seconds=\d+\.\d{3}',
                    line)
                if timing:
                    reporter.write_line(timing.group(0))
    if (item.config.getoption('--require-resource-credentials')
            and report.skipped and group_for_item(item) != 'public'):
        reason = str(report.longrepr).lower()
        if any(fragment in reason
               for fragment in ('credential', 'not configured',
                                'access is unavailable')):
            item.config._resource_credential_skips.append(item.nodeid)


def pytest_sessionfinish(session, exitstatus):
    missing = session.config._resource_credential_skips
    if missing:
        reporter = session.config.pluginmanager.get_plugin('terminalreporter')
        if reporter:
            reporter.write_sep(
                '=', 'Private CI skipped tests for missing credentials',
                red=True)
            for nodeid in missing:
                reporter.write_line(nodeid)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
