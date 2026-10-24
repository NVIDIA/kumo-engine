# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
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
    with pytest.raises(ValueError, match='Unknown export kinds'):
        kumo.MLOps(artifact_root=str(tmp_path), export={'models'})


def test_a_root_is_a_uri_or_a_connector_and_path(tmp_path):
    uri = kumo.MLOps(artifact_root=str(tmp_path))
    assert uri.root.location('models/m') == 'models/m'
    assert uri.root.uri('models/m') == f'{tmp_path}/models/m'
    assert isinstance(uri.root.connector, kumo.FileConnector)
    assert uri.exports == {'snapshot', 'training_table', 'model',
                           'prediction'}

    connector = kumo.FileConnector(root_dir=str(tmp_path))
    pair = kumo.MLOps(artifact_root=(connector, 'volume/kumo/'),
                      export={'model'})
    assert pair.root.connector is connector
    assert pair.root.location('models/m') == 'volume/kumo/models/m'
    assert pair.wants('model') and not pair.wants('snapshot')
    assert not kumo.MLOps(artifact_root=str(tmp_path), export=False).exports


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
