# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for conservative multi-devserver lifecycle primitives."""

import fcntl
import json
import os

import pytest

from rest.devserver.lifecycle import (
    DevserverLifecycleError,
    DevserverLifecycleLockError,
    StaleManifestStatus,
    advisory_lock,
    coordinator_locks,
    inspect_stale_manifest,
    new_manifest,
    read_manifest,
    write_manifest_atomic,
)


def _manifest(phase='planned'):
    return new_manifest(
        {
            'train-a': {
                'resources': {
                    'scope': {
                        'owner_token': 'owner-a',
                        'unit': 'kumo-devserver-train-a',
                    },
                },
            },
        },
        phase=phase,
        owner_token='owner-a')


def test_manifest_round_trip_is_atomic_json(tmp_path):
    path = tmp_path / 'lifecycle.json'
    manifest = _manifest('launching')

    write_manifest_atomic(path, manifest)

    assert read_manifest(path) == manifest
    assert json.loads(path.read_text())['phase'] == 'launching'
    assert not list(tmp_path.glob('*.tmp'))


def test_manifest_rejects_unowned_resources_and_missing_parent(tmp_path):
    with pytest.raises(DevserverLifecycleError, match='owner token'):
        new_manifest({
            'train-a': {
                'resources': {
                    'scope': {
                        'owner_token': 'somebody-else',
                    },
                },
            },
        }, owner_token='owner-a')

    with pytest.raises(DevserverLifecycleError, match='does not exist'):
        write_manifest_atomic(tmp_path / 'missing' / 'manifest.json',
                              _manifest())


def test_manifest_rejects_invalid_phase_when_read(tmp_path):
    path = tmp_path / 'lifecycle.json'
    path.write_text(json.dumps({
        'version': 1,
        'phase': 'deleting',
        'owner_token': 'owner-a',
        'children': {},
    }))

    with pytest.raises(DevserverLifecycleError, match='phase'):
        read_manifest(path)


def test_advisory_lock_and_coordinator_child_locks(tmp_path):
    lock_path = tmp_path / 'one.lock'
    with advisory_lock(lock_path):
        second_descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(second_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(second_descriptor)

    with coordinator_locks(tmp_path, ['train-b', 'train-a']):
        assert (tmp_path / 'coordinator.lock').is_file()
        assert (tmp_path / 'child-train-a.lock').is_file()
        assert (tmp_path / 'child-train-b.lock').is_file()

    with pytest.raises(DevserverLifecycleError, match='duplicate'):
        with coordinator_locks(tmp_path, ['train-a', 'train-a']):
            pass
    with pytest.raises(DevserverLifecycleError, match='normalized safe IDs'):
        with coordinator_locks(tmp_path, ['../not-a-child']):
            pass


def test_lock_reports_contention(tmp_path):
    path = tmp_path / 'contention.lock'
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DevserverLifecycleLockError, match='already held'):
            with advisory_lock(path):
                pass
    finally:
        os.close(descriptor)


def test_stale_manifest_is_ambiguous_without_full_owner_verification():
    manifest = _manifest()

    missing = inspect_stale_manifest(manifest)
    false = inspect_stale_manifest(manifest,
                                   lambda *_: False)
    raised = inspect_stale_manifest(
        manifest, lambda *_: (_ for _ in ()).throw(RuntimeError('unreachable')))

    assert missing.status is StaleManifestStatus.AMBIGUOUS
    assert false.status is StaleManifestStatus.AMBIGUOUS
    assert raised.status is StaleManifestStatus.AMBIGUOUS
    assert not missing.verified_resources


def test_stale_manifest_requires_all_resources_to_verify():
    manifest = new_manifest(
        {
            'train-a': {
                'resources': {
                    'scope': {'owner_token': 'owner-a'},
                    'port-block': {'owner_token': 'owner-a'},
                },
            },
        },
        owner_token='owner-a')

    inspection = inspect_stale_manifest(
        manifest,
        lambda resource_id, resource, owner_token: (
            resource['owner_token'] == owner_token and resource_id !=
            'train-a/port-block'))
    verified = inspect_stale_manifest(
        manifest,
        lambda resource_id, resource, owner_token: resource['owner_token'] ==
        owner_token)

    assert inspection.status is StaleManifestStatus.AMBIGUOUS
    assert inspection.verified_resources == ('train-a/scope', )
    assert verified.status is StaleManifestStatus.VERIFIED
    assert verified.verified_resources == ('train-a/scope', 'train-a/port-block')
