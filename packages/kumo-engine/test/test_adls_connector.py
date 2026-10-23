# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kumo_core.sources.object_store_access import ObjectStoreAccess
from kumoapi.data_source import AzureCloud, DataSourceType
from kumoapi.jobs import PredictionArtifactType, WriteMode
from kumoapi.model_plan import ModelPlan
from kumoapi.source_table import ADLSRelativeSource, ADLSSourceTable
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem

from kumo.engine import ADLSConnector, ADLSIdentity, Table
from kumo.engine._portable import _source_table_name, portable_graph_config

ROOT = 'abfss://curated@companylake.dfs.core.windows.net/orders/'


@pytest.fixture
def storage(monkeypatch, tmp_path):
    directory = tmp_path / 'storage'
    target = directory / 'curated@companylake.dfs.core.windows.net/orders'
    target.mkdir(parents=True)
    pq.write_table(pa.table({
        'id': [1, 2],
        'value': [10., 20.]
    }), target / 'events.parquet')
    filesystem = SubTreeFileSystem(str(directory), LocalFileSystem())

    def identity(path):
        with filesystem.open_input_file(path) as stream:
            return (hashlib.sha256(stream.read()).hexdigest(), )

    monkeypatch.setattr(
        'kumo_adls.backend.source_access',
        lambda *args: ObjectStoreAccess(filesystem, object_identity=identity))
    return directory


@pytest.mark.parametrize(
    'root, prefix', [(ROOT, 'orders'), (ROOT.rstrip('/'), 'orders'),
                     (ROOT.removesuffix('orders/'), ''),
                     (ROOT.removesuffix('/orders/'), ''),
                     (ROOT.replace('orders/', 'space%20name/'), 'space name')])
def test_root_parsing_without_connecting(root, prefix):
    connector = ADLSConnector(root_dir=root)
    assert connector.prefix == prefix
    assert connector.filesystem == 'curated'
    assert connector.storage_account == 'companylake'
    assert connector.source_type == DataSourceType.ADLS


@pytest.mark.parametrize('root', [
    'abfs://curated@companylake.dfs.core.windows.net/orders/',
    ROOT + '../outside', ROOT + '?token=secret',
    ROOT.replace('orders/', 'orders//')
])
def test_invalid_roots_fail_locally(root):
    with pytest.raises(ValueError):
        ADLSConnector(root_dir=root)


def test_workload_identity_is_registered_without_credentials():
    identity = ADLSIdentity(tenant_id='tenant', client_id='client',
                            cloud=AzureCloud.PUBLIC)
    connector = ADLSConnector(root_dir=ROOT, identity=identity)
    assert connector.identity is identity
    assert connector.identity_id == identity.slug
    assert identity._credentials is None
    assert 'tenant' not in repr(identity)
    assert connector.source_backend._identity is identity
    assert connector.sink_backend._identity is identity
    assert connector.source_backend is connector.source_backend
    assert connector.sink_backend is connector.sink_backend


def test_table_inference_portable_graph_and_ingest(storage, tmp_path):
    connector = ADLSConnector(root_dir=ROOT)
    assert connector.table_names() == ['events']
    source = connector.source_table_for('events')
    assert isinstance(source, ADLSSourceTable)
    assert isinstance(source.source, ADLSRelativeSource)
    assert _source_table_name(source).startswith('adls.')
    assert len(_source_table_name(source).split('.')) == 3
    table = Table.from_source_table(connector['events'],
                                    primary_key='id').infer_metadata()
    graph = portable_graph_config({'events': table}, [], name='test')
    assert graph['tables'][0]['source_table_name'].startswith('adls.')
    ingest = connector.source_backend.prepare_ingest(connector, 'events',
                                                     str(tmp_path / 'ingest'))
    assert pq.read_table(ingest.source['paths']).num_rows == 2
    connector.sink_backend.write_table(connector, 'scores',
                                       pd.DataFrame({'score': [0.5]}))
    assert pq.read_table(
        storage /
        'curated@companylake.dfs.core.windows.net/orders/scores.parquet'
    ).num_rows == 1


def test_rootless_graph_coordinate_contains_exact_source(storage):
    connector = ADLSConnector()
    source = connector.source_table_for(ROOT + 'events.parquet')
    coordinate = _source_table_name(source)
    assert 'companylake' in coordinate
    assert len(coordinate.split('.')) == 3


@pytest.mark.parametrize('case,error,match', [
    ('rootless', NotImplementedError, 'source-only'),
    ('client', NotImplementedError, 'match the selected workload identity'),
    ('tenant', NotImplementedError, 'match the selected workload identity'),
    ('token', NotImplementedError, 'projected token'),
    ('authority', ValueError, 'AZURE_AUTHORITY_HOST'),
    ('cloud', ValueError, 'selected identity cloud'),
    ('append', NotImplementedError, 'OVERWRITE'),
    ('outside', ValueError, 'outside the connector root'),
    ('encoded_uri', ValueError, 'must not be a URI, query, or fragment'),
])
def test_invalid_prediction_output_fails_before_scoring(monkeypatch, tmp_path,
                                                        snapshot, case, error,
                                                        match):
    from kumo_core.types import TaskType

    from kumo.engine import PredictionOutputConfig, PredictionTable, Trainer

    monkeypatch.setenv('AZURE_TENANT_ID', 'tenant')
    monkeypatch.setenv('AZURE_CLIENT_ID', 'client')
    monkeypatch.setenv('AZURE_FEDERATED_TOKEN_FILE', '/projected/token')
    monkeypatch.setenv('AZURE_AUTHORITY_HOST',
                       'https://login.microsoftonline.com')
    if case in ('client', 'tenant'):
        monkeypatch.setenv('AZURE_' + case.upper() + '_ID', 'different')
    elif case == 'token':
        monkeypatch.delenv('AZURE_FEDERATED_TOKEN_FILE')
    elif case == 'authority':
        monkeypatch.setenv('AZURE_AUTHORITY_HOST',
                           'https://login.microsoftonline.us')
    identity = ADLSIdentity(
        tenant_id='tenant', client_id='client', cloud=AzureCloud.US_GOVERNMENT
        if case == 'cloud' else AzureCloud.PUBLIC)
    connector = ADLSConnector(root_dir=None if case == 'rootless' else ROOT,
                              identity=identity)
    destination = 'predictions/'
    if case == 'outside':
        destination = ROOT.replace('orders/', 'outside/')
    elif case == 'encoded_uri':
        destination = ROOT + 'scores%23review.parquet'
    output = PredictionOutputConfig(
        connector, artifact_type=PredictionArtifactType.PREDICTIONS,
        table_name=destination, write_mode=WriteMode.APPEND
        if case == 'append' else WriteMode.OVERWRITE)
    trainer = Trainer(ModelPlan(), accelerator='cpu')
    monkeypatch.setattr(trainer, '_training_result',
                        SimpleNamespace(artifact_paths={}))
    trainer._training_query = 'trained query'
    monkeypatch.setattr(Trainer, 'scratch_dir', property(lambda _: tmp_path))
    # Isolate output validation from query parsing and model execution.
    monkeypatch.setattr(
        'kumo_core.prediction_options.prediction_query_contract', lambda *args:
        (None, TaskType.REGRESSION.value))
    monkeypatch.setattr('kumo_core.pquery.normalize_graph', lambda _: None)
    monkeypatch.setattr(
        'kumo_core.pquery.parse_and_validate_query',
        lambda *args: SimpleNamespace(query=SimpleNamespace(top_k=None)))
    work = Mock(side_effect=AssertionError('prediction work started'))
    monkeypatch.setattr(
        'kumo.engine.trainer.trainer.build_generated_training_task_dataset',
        work)
    monkeypatch.setattr('kumo_core.prediction.materialize_prediction_task',
                        work)
    monkeypatch.setattr('kumo_core.prediction.score_prediction', work)
    azure = Mock(side_effect=AssertionError('Azure access started'))
    monkeypatch.setattr('kumo_adls.backend.source_access', azure)
    monkeypatch.setattr('kumo_adls.sink.filesystem_for', azure)
    table = cast(PredictionTable, SimpleNamespace(query='prediction query'))

    with pytest.raises(error, match=match):
        trainer.predict(table, snapshot=snapshot, output_config=output)

    work.assert_not_called()
    azure.assert_not_called()
    assert not (tmp_path / 'runs').exists()
