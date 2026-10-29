# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
from types import SimpleNamespace

import pytest

import kumo.engine as kumo
from kumo.engine._manifest import (
    SnapshotManifest,
    SnapshotParents,
    embedded,
    stub,
)
from kumo.engine._mlops import resolve


def test_mlops_needs_an_artifact_root_it_can_write_files_to(tmp_path):
    with pytest.raises(TypeError):
        kumo.MLOps()
    with pytest.raises(ValueError, match='artifact_root'):
        kumo.MLOps(artifact_root='')
    table_only = SimpleNamespace(sink_backend=SimpleNamespace(
        holds_files=lambda c, p: False, filesystem=None))
    with pytest.raises(ValueError, match='file location'):
        kumo.MLOps(artifact_root=(table_only, 'DB.SCHEMA'))
    with pytest.raises(ValueError, match='needs its connector'):
        kumo.MLOps(artifact_root='@DB.SCHEMA.STAGE/kumo')
    with pytest.raises(ValueError, match='Unknown store kinds'):
        kumo.MLOps(artifact_root=str(tmp_path), store={'models'})
    with pytest.raises(ValueError, match='on_error'):
        kumo.MLOps(artifact_root=str(tmp_path), on_error='ignore')


def test_the_tracker_registers_and_its_registry_checks_the_name(tmp_path):
    def check_model_location(connector, location):
        if '-' in location.model_name:
            raise ValueError('bad name')

    tracker = SimpleNamespace(connector=SimpleNamespace(
        sink_backend=SimpleNamespace(
            check_model_location=check_model_location)))
    good = kumo.ModelRegistryLocation('USERS_CHURN', alias='candidate')
    kumo.MLOps(str(tmp_path), tracking=tracker, registry=good)
    with pytest.raises(ValueError, match='bad name'):
        kumo.MLOps(str(tmp_path), tracking=tracker,
                   registry=kumo.ModelRegistryLocation('users-churn'))
    with pytest.raises(ValueError, match='model alias'):
        kumo.MLOps(str(tmp_path), tracking=tracker,
                   registry=kumo.ModelRegistryLocation('USERS_CHURN',
                                                       alias='LATEST'))
    with pytest.raises(ValueError, match='registers through the tracker'):
        kumo.MLOps(str(tmp_path), registry=good)
    with pytest.raises(TypeError, match='ModelRegistryLocation'):
        kumo.MLOps(str(tmp_path), tracking=tracker,
                   registry=kumo.ModelOutputConfig(tracker.connector,
                                                   location=good))


def test_a_root_is_a_uri_or_a_connector_and_path(tmp_path):
    uri = kumo.MLOps(artifact_root=str(tmp_path))
    assert uri.root.location('models/m') == 'models/m'
    assert uri.root.uri('models/m') == f'{tmp_path}/models/m'
    assert isinstance(uri.root.connector, kumo.FileConnector)
    assert uri.kinds == {'snapshot', 'training_table', 'model'}

    connector = kumo.FileConnector(root_dir=str(tmp_path))
    pair = kumo.MLOps(artifact_root=(connector, 'volume/kumo/'),
                      store={'model'})
    assert pair.root.connector is connector
    assert pair.root.location('models/m') == 'volume/kumo/models/m'
    assert pair.wants('model') and not pair.wants('snapshot')
    assert not kumo.MLOps(artifact_root=str(tmp_path), store=False).kinds
    assert uri.snapshot_uri('ab12@cd34') == f'{tmp_path}/snapshots/ab12/cd34'
    assert kumo.MLOps('s3://bucket/kumo').snapshot_uri(
        SimpleNamespace(id='ab12@cd34')) == 's3://bucket/kumo/snapshots/ab12/cd34'


def test_a_connector_root_outside_its_root_dir_fails_when_built():
    data = kumo.S3Connector(root_dir='s3://bucket/data')
    assert kumo.MLOps(artifact_root=(data, 's3://bucket/data/kumo'),
                      store=False).root
    with pytest.raises(ValueError, match=r"'s3://bucket/kumo' is outside "
                       r"this S3Connector's root_dir s3://bucket/data .*"
                       r"S3Connector\(root_dir='s3://bucket/kumo'"):
        kumo.MLOps(artifact_root=(data, 's3://bucket/kumo'), store=False)


def test_a_root_that_cannot_be_written_fails_when_built(tmp_path):
    if os.geteuid() == 0:
        pytest.skip('root writes through a read-only directory')
    read_only = tmp_path / 'read-only'
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        with pytest.raises(PermissionError, match='cannot write under '
                           'artifact_root'):
            kumo.MLOps(artifact_root=str(read_only / 'kumo'))
        assert kumo.MLOps(artifact_root=str(read_only), store=False)
    finally:
        read_only.chmod(0o700)
    kumo.MLOps(artifact_root=str(tmp_path / 'kumo'))
    assert not (tmp_path / 'kumo').exists()


def test_a_call_argument_beats_the_nearest_block(tmp_path):
    outer = kumo.MLOps(artifact_root=str(tmp_path / 'outer'))
    inner = kumo.MLOps(artifact_root=str(tmp_path / 'inner'))
    call = kumo.MLOps(artifact_root=str(tmp_path / 'call'))
    assert resolve() is None
    seen = []
    with kumo.mlops(outer):
        with kumo.mlops(inner):
            assert resolve() is inner
            assert resolve(call) is call
        assert resolve() is outer
        thread = threading.Thread(target=lambda: seen.append(resolve()))
        thread.start()
        thread.join()

        async def task():
            return resolve()

        assert asyncio.run(task()) is outer
    assert seen == [None] and resolve() is None


def test_stubs_stand_in_for_parents_without_a_manifest():
    snapshot = stub('snapshot', 'abc@0123')
    assert isinstance(snapshot, SnapshotManifest)
    assert snapshot.to_dict() == {
        'kind': 'snapshot',
        'schema_version': 1,
        'id': 'abc@0123',
        'created_at': None,
        'environment': None,
        'reconstructed': True,
        'parents': None,
    }
    full = SnapshotManifest(id='abc@0123',
                            parents=SnapshotParents(source_tables={}))
    assert embedded(full) == full
