# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select credential profiles before provider imports and fixture setup."""

import json
import os
from pathlib import Path

import pytest

from tools.ci_test_profiles import (
    GCP_MIXED_MODULE,
    GCP_MODULES,
    GCP_NODES,
    MANIFEST_PREFIX,
    TestProfile,
    owns_node,
    permits_target,
    profile_for,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption('--test-profile', default=None,
                     help='Select an isolated CI credential profile')
    parser.addoption('--test-profile-manifest', action='store_true',
                     help='Emit selected node IDs after collection')


def _profile(name: str) -> TestProfile:
    try:
        return profile_for(name)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def pytest_load_initial_conftests(early_config: pytest.Config,
                                  parser: pytest.Parser,
                                  args: list[str]) -> None:
    options = parser.parse_known_args(args)
    if options.test_profile is None:
        return
    profile = _profile(options.test_profile)
    os.environ['KUMO_TEST_PROFILE'] = profile.id
    for target in options.file_or_dir:
        filename, *nodes = target.split('::')
        path = Path(filename)
        if not path.is_absolute():
            path = early_config.invocation_params.dir / path
        relative = _relative(path, early_config.rootpath)
        nodeid = '::'.join([relative, *nodes])
        if not permits_target(profile, nodeid):
            raise pytest.UsageError(
                f'{target} is incompatible with test profile {profile.id}')


def pytest_configure(config: pytest.Config) -> None:
    for marker in (
            'requires_credentials(*capabilities): explicit live providers',
            'bigquery_credentials(service_account_key): BigQuery prerequisite',
            'gcs_credentials: GCS prerequisite',
    ):
        config.addinivalue_line('markers', marker)
    name = config.getoption('test_profile')
    if name is not None:
        _profile(name)


def pytest_ignore_collect(collection_path: Path,
                          config: pytest.Config) -> bool | None:
    name = config.getoption('test_profile')
    if name is None:
        return None
    relative = _relative(collection_path, config.rootpath)
    if not permits_target(_profile(name), relative):
        return True
    return None


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    name = config.getoption('test_profile')
    if name is None:
        return
    profile = _profile(name)
    selected = []
    deselected = []
    for item in items:
        relative = _relative(item.path, config.rootpath)
        capabilities = {
            capability
            for marker in item.iter_markers('requires_credentials')
            for capability in marker.args
        }
        base_nodeid = item.nodeid.split('[', 1)[0]
        if (item.get_closest_marker('gcs_credentials') is not None
                or item.get_closest_marker('bigquery_credentials') is not None):
            capabilities.add('gcp')
        if ('gcp' in capabilities and relative not in GCP_MODULES
                and relative != GCP_MIXED_MODULE
                and base_nodeid not in GCP_NODES):
            raise pytest.UsageError(
                f'{item.nodeid} requires GCP but is not registered in '
                'the test profile collection targets')
        if owns_node(profile, relative, base_nodeid, frozenset(capabilities)):
            missing = capabilities - profile.capabilities
            if missing:
                raise pytest.UsageError(
                    f'{item.nodeid} requires undeclared profile capabilities: '
                    f'{", ".join(sorted(missing))}')
            selected.append(item)
        else:
            deselected.append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def pytest_collection_finish(session: pytest.Session) -> None:
    if not session.config.getoption('test_profile_manifest'):
        return
    profile = session.config.getoption('test_profile')
    if profile is None:
        if not session.config.getoption('collectonly'):
            raise pytest.UsageError(
                'Unprofiled manifests require --collect-only')
        profile = 'unprofiled'
    payload = json.dumps({
        'profile': profile,
        'nodeids': [item.nodeid for item in session.items],
    })
    reporter = session.config.pluginmanager.get_plugin('terminalreporter')
    if reporter is None:
        raise pytest.UsageError('Profile manifests require the terminal reporter')
    reporter.write_line(MANIFEST_PREFIX + payload)
