# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest

from kumoai_studio.backend import (
    CloudBackend,
    RestSourceBackend,
    cloud_backend,
)
from kumoai_studio.backend.base import (
    ArtifactExportBackend,
    GraphBackend,
    PQueryBackend,
    PredictionBackend,
    PredictionTableBackend,
    TableBackend,
    TrainingBackend,
    TrainTableBackend,
)
from kumoai_studio.client.graph import GraphAPI
from kumoai_studio.client.jobs import (
    ArtifactExportJobAPI,
    BatchPredictionJobAPI,
    GeneratePredictionTableJobAPI,
    GenerateTrainTableJobAPI,
    TrainingJobAPI,
)
from kumoai_studio.client.pquery import PQueryAPI
from kumoai_studio.client.table import TableAPI


def test_default_backend_is_cloud():
    backend = cloud_backend()
    assert isinstance(backend, CloudBackend)
    assert backend.name == 'cloud'


def test_default_backend_is_shared():
    assert cloud_backend() is cloud_backend()


def test_cloud_source_area_is_the_rest_reader():
    r"""Unlike the other areas, source needs an adapter: the client API is
    keyed by connector ID, while the area receives the connector itself.
    """
    assert isinstance(CloudBackend().source, RestSourceBackend)


def test_cloud_backend_areas_resolve_to_client_apis():
    client = mock.MagicMock()
    backend = CloudBackend()

    with mock.patch('kumoai_studio.global_state') as state:
        state.client = client
        assert backend.table is client.table_api
        assert backend.graph is client.graph_api
        assert backend.pquery is client.pquery_api
        assert backend.train_table is client.generate_train_table_job_api
        assert backend.training is client.training_job_api
        assert (backend.prediction_table
                is client.generate_prediction_table_job_api)
        assert backend.prediction is client.batch_prediction_job_api
        assert backend.artifact_export is client.artifact_export_api


def test_cloud_backend_does_not_cache_the_client():
    r"""Clients are thread-local and SPCS tokens are refreshed between calls,
    so every area access must re-resolve the client.
    """
    backend = CloudBackend()

    with mock.patch('kumoai_studio.global_state') as state:
        state.client = mock.MagicMock()
        first = backend.graph
        state.client = mock.MagicMock()
        second = backend.graph

    assert first is not second


@pytest.mark.parametrize(
    'api,protocol',
    [
        (TableAPI, TableBackend),
        (GraphAPI, GraphBackend),
        (PQueryAPI, PQueryBackend),
        (GenerateTrainTableJobAPI, TrainTableBackend),
        (TrainingJobAPI, TrainingBackend),
        (GeneratePredictionTableJobAPI, PredictionTableBackend),
        (BatchPredictionJobAPI, PredictionBackend),
        (ArtifactExportJobAPI, ArtifactExportBackend),
    ],
)
def test_client_apis_implement_their_protocol(api, protocol, protocol_members):
    r"""The cloud path is the reference implementation of the contract, so
    every operation a protocol names must exist on the API it maps to.
    """
    instance = api(mock.MagicMock())
    missing = [
        attr for attr in protocol_members(protocol)
        if not hasattr(instance, attr)
    ]
    assert not missing, f'{api.__name__} is missing {missing}'


def test_relational_objects_use_the_active_backend(mock_backend):
    r"""A graph reaches the installed backend rather than the cloud client."""
    from kumoai_studio.graph import Graph

    graph = object.__new__(Graph)
    graph._tables = {}
    graph._edges = []
    graph._kumo_project_id = 'project-owned'
    graph._graph_snapshot_id = None
    mock_backend.graph.create_graph.return_value = 'graph-abc'

    assert graph.save(skip_validation=True) == 'graph-abc'
    mock_backend.graph.create_graph.assert_called_once()
    assert mock_backend.graph.create_graph.call_args.kwargs[
        'project_id'] == 'project-owned'
