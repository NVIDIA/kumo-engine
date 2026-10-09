# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import pytest
from kumoapi.common import JobStatus
from kumoapi.jobs import FileTrainingTableOutputConfig
from kumoapi.source_table import FileType, S3RelativeSource, S3SourceTable

from kumoai_studio.artifact_export import (
    ArtifactExportJob,
    TrainingTableExportConfig,
)
from kumoai_studio.connector import (
    ADLSConnector,
    GCSConnector,
    S3Connector,
    SourceTable,
)
from kumoai_studio.pquery.training_table import TrainingTable


@pytest.mark.parametrize("connector_type",
                         [S3Connector, GCSConnector, ADLSConnector])
@pytest.mark.parametrize(
    'request_id',
    [None, 'd4af0106-44dc-4c42-aa53-d622b49161bd'],
)
def test_export_uses_durable_project_connector_reference(
        connector_type, request_id, disable_artifact_export_polling):
    connector = object.__new__(connector_type)
    connector._id = "connector-project-owned"
    connector._kumo_project_id = "project-owned"
    connector.root_dir = "unused-by-export"
    if isinstance(connector, S3Connector):
        connector.any_uri = False
    output = TrainingTableExportConfig(
        output_types={"training_table"},
        output_connector=connector,
        output_table_name="exports/training",
    )
    table = object.__new__(TrainingTable)
    table.job_id = "generate-train-table-job"
    table._kumo_project_id = "project-owned"

    with mock.patch("kumoai_studio.global_state",
                    new=mock.MagicMock()) as state, mock.patch(
                        'kumoai_studio.pquery.training_table.uuid.uuid4',
                        return_value='8f14e45f-ea5e-4a40-a79b-3f55a4df7c4d',
                    ) as uuid4:
        state.client.artifact_export_api.create.return_value = "export-job"

        if request_id is None:
            result = table.export(output)
        else:
            result = table.export(output, request_id=request_id)

    assert isinstance(result, ArtifactExportJob)
    request = state.client.artifact_export_api.create.call_args.args[0]
    expected_request_id = ('8f14e45f-ea5e-4a40-a79b-3f55a4df7c4d'
                           if request_id is None else request_id)
    assert request.request_id == expected_request_id
    if request_id is None:
        uuid4.assert_called_once_with()
    else:
        uuid4.assert_not_called()
    expected = FileTrainingTableOutputConfig(
        connector_id="connector-project-owned",
        path="exports/training",
    )
    assert request.training_table_output == expected
    state.client.artifact_export_api.create.assert_called_once_with(
        request, project_id='project-owned')


@pytest.mark.parametrize(
    ('connector_type', 'root_dir', 'full_path'),
    [
        (S3Connector, 's3://bucket/root', 's3://bucket/root/deep/training'),
        (GCSConnector, 'gs://bucket/root', 'gs://bucket/root/deep/training'),
        (
            ADLSConnector,
            'abfss://curated@lake.dfs.core.windows.net/root',
            ('abfss://curated@lake.dfs.core.windows.net/root/'
             'deep/training'),
        ),
    ],
)
def test_rooted_object_store_training_export_normalizes_full_uri_to_relative(
        connector_type, root_dir, full_path, disable_artifact_export_polling):
    connector = object.__new__(connector_type)
    connector._id = 'connector-output'
    connector._kumo_project_id = 'project-owned'
    connector.any_uri = False
    connector.root_dir = root_dir
    output = TrainingTableExportConfig(
        output_types={'training_table'},
        output_connector=connector,
        output_table_name=full_path,
    )
    table = object.__new__(TrainingTable)
    table.job_id = 'generate-train-table-job'
    table._kumo_project_id = 'project-owned'

    with mock.patch('kumoai_studio.global_state',
                    new=mock.MagicMock()) as state:
        state.client.artifact_export_api.create.return_value = 'export-job'
        table.export(output)

    request = state.client.artifact_export_api.create.call_args.args[0]
    assert request.training_table_output == FileTrainingTableOutputConfig(
        connector_id='connector-output', path='deep/training')


@pytest.mark.parametrize(('connector_type', 'output_name', 'destination'), [
    (S3Connector, 's3://output-bucket/deep/training',
     's3://output-bucket/deep/training'),
    (GCSConnector, 'gs://output-bucket/deep/training',
     'gs://output-bucket/deep/training'),
    (ADLSConnector, 'abfss://out@lake.dfs.core.windows.net/deep/training',
     'abfss://out@lake.dfs.core.windows.net/deep/training'),
])
def test_rootless_object_store_training_export_uses_full_path(
        connector_type, output_name, destination,
        disable_artifact_export_polling):
    connector = object.__new__(connector_type)
    connector._id = 'connector-any-uri'
    connector._kumo_project_id = 'project-owned'
    connector.any_uri = True
    connector.root_dir = None
    output = TrainingTableExportConfig(
        output_types={'training_table'},
        output_connector=connector,
        output_table_name=output_name,
    )
    table = object.__new__(TrainingTable)
    table.job_id = 'generate-train-table-job'
    table._kumo_project_id = 'project-owned'
    with mock.patch('kumoai_studio.global_state') as state:
        state.client.artifact_export_api.create.return_value = 'export-job'
        table.export(output)

    request = state.client.artifact_export_api.create.call_args.args[0]
    assert request.training_table_output.path == destination


@pytest.mark.parametrize('connector_type',
                         [S3Connector, GCSConnector, ADLSConnector])
def test_rootless_object_store_training_export_requires_full_path(
        connector_type):
    connector = object.__new__(connector_type)
    connector._id = 'connector-any-uri'
    connector._kumo_project_id = 'project-owned'
    connector.any_uri = True
    connector.root_dir = None
    output = TrainingTableExportConfig(
        output_types={'training_table'},
        output_connector=connector,
        output_table_name='relative/training',
    )
    table = object.__new__(TrainingTable)
    table.job_id = 'generate-train-table-job'
    table._kumo_project_id = 'project-owned'
    state = mock.MagicMock()

    with mock.patch('kumoai_studio.global_state', state), \
            pytest.raises(ValueError, match='URI must use'):
        table.export(output)

    state.client.artifact_export_api.create.assert_not_called()


def test_training_export_rejects_cross_project_connector_before_request():
    connector = object.__new__(S3Connector)
    connector._id = 'connector-s3'
    connector._kumo_project_id = 'project-other'
    connector.any_uri = False
    output = TrainingTableExportConfig(
        output_types={'training_table'},
        output_connector=connector,
        output_table_name='training',
    )
    table = object.__new__(TrainingTable)
    table.job_id = 'job'
    table._kumo_project_id = 'project-owned'
    state = mock.MagicMock()

    with mock.patch('kumoai_studio.global_state', state):
        with pytest.raises(ValueError, match='project-other.*project-owned'):
            table.export(output)

    state.client.artifact_export_api.create.assert_not_called()


def test_custom_training_table_validation_keeps_owning_project(mock_backend):
    table = object.__new__(TrainingTable)
    table.job_id = 'generate-train-table-job'
    table._kumo_project_id = 'project-owned'
    source = mock.MagicMock()
    modification = mock.MagicMock()
    mock_backend.train_table.validate_custom_train_table.return_value.ok = True

    table.validate_custom_table(source, modification,
                                extensive_validation=True)

    mock_backend.train_table \
        .validate_custom_train_table.assert_called_once_with(
            table.job_id,
            source,
            modification,
            True,
            project_id='project-owned',
        )


def test_custom_training_table_validation_formats_server_error(mock_backend):
    table = object.__new__(TrainingTable)
    table.job_id = 'generate-train-table-job'
    table._kumo_project_id = 'project-owned'
    response = mock.MagicMock(ok=False)
    response.error_message.return_value = (
        'Required column not found. Source table is missing ENTITY1.')
    mock_backend.train_table \
        .validate_custom_train_table.return_value = response

    with pytest.raises(ValueError) as error:
        table.validate_custom_table(mock.MagicMock(), mock.MagicMock())

    assert str(error.value) == (
        'Invalid weighted train table: Required column not found. '
        'Source table is missing ENTITY1.')
    response.error_message.assert_called_once_with()


def test_a_supplied_table_from_another_project_is_refused_before_lookup():
    from kumoai_studio.pquery.training_table import _api_source_table

    connector = object.__new__(S3Connector)
    connector._id = 'connector-s3'
    connector._kumo_project_id = 'project-other'
    source = SourceTable('custom/training.parquet', connector)
    to_api_source_table = mock.MagicMock()
    cast(Any, source)._to_api_source_table = to_api_source_table

    with pytest.raises(ValueError, match='project-other.*project-owned'):
        _api_source_table(source, 'project-owned')

    to_api_source_table.assert_not_called()


def test_custom_s3_training_table_preserves_connector_id():
    connector = object.__new__(S3Connector)
    connector._id = "connector-project-owned"
    connector._kumo_project_id = "project-owned"
    connector.root_dir = "s3://bucket/prefix"
    source = mock.MagicMock(spec=SourceTable)
    source.connector = connector
    source.name = "training"
    source._to_api_source_table.return_value = S3SourceTable(
        connector_id=connector.id,
        source=S3RelativeSource(path="training"),
        file_type=FileType.PARQUET,
    )
    from kumoai_studio.pquery.training_table import _api_source_table

    result = _api_source_table(source, connector.kumo_project_id)

    # The resolver answers the union a request carries, so the family is
    # asserted before the fields only this member has are read.
    assert isinstance(result, S3SourceTable)
    assert result.connector_id == connector.id
    assert isinstance(result.source, S3RelativeSource)
    assert result.source.path == "training"
    assert not hasattr(result, "s3_path")


@pytest.mark.parametrize('use_sql', [False, True])
def test_databricks_custom_training_table_preserves_source(use_sql):
    from kumoapi.source_table import (
        CustomSQLSourceTable,
        DatabricksSourceTable,
    )

    from kumoai_studio import DatabricksConnector, SnowflakeConnector

    connector = object.__new__(DatabricksConnector)
    connector._id = 'connector-databricks'
    connector._kumo_project_id = 'project-owned'
    if use_sql:
        source = connector.sql(
            'SELECT ENTITY1, TARGET1, sample_weight FROM weighted_train')
        expected = CustomSQLSourceTable(
            connector_id=connector.id,
            sql=source.sql,
            table=source.name,
        )
    else:
        expected = DatabricksSourceTable(
            databricks_connector_id=connector.id,
            table='weighted_train',
        )
        source = SourceTable(
            'weighted_train',
            connector,
            _api_source_table=expected,
        )
    from kumoai_studio.pquery.training_table import _api_source_table

    assert _api_source_table(source, 'project-owned') == expected

    if use_sql:
        snowflake = object.__new__(SnowflakeConnector)
        snowflake._id = 'connector-snowflake'
        snowflake._kumo_project_id = 'project-owned'
        with pytest.raises(ValueError, match='only for Databricks'):
            _api_source_table(
                snowflake.sql('SELECT ENTITY1, TARGET1, sample_weight '
                              'FROM weighted_train'), 'project-owned')


def test_generated_training_table_uses_server_selected_safe_references(
        mock_backend):
    table = object.__new__(TrainingTable)
    table.job_id = 'generate-train-table-job'
    table._kumo_project_id = 'project-owned'
    mock_backend.train_table.get_table_data.return_value = [
        'https://safe-reference/part-0',
        'https://safe-reference/part-1',
    ]

    assert table.data_urls() == [
        'https://safe-reference/part-0',
        'https://safe-reference/part-1',
    ]

    mock_backend.train_table.get_table_data \
        .assert_called_once_with(
            'generate-train-table-job', project_id='project-owned')


# --- reconstructing a `TrainingTable` recovers its custom table -----------


def _resource(status, custom_train_table=None, used=None):
    r"""A job resource, as either backend's `get` returns one.

    ``used`` defaults to agreeing with the spec, which is what a record
    written by a backend that persists both looks like. Passing it
    explicitly is how a record that disagrees with itself is built.
    """
    return SimpleNamespace(
        job_status_report=SimpleNamespace(status=status),
        config=SimpleNamespace(custom_train_table=custom_train_table),
        custom_train_table_used=(custom_train_table is not None
                                 if used is None else used),
    )


def test_a_registered_jobs_custom_table_round_trips_through_the_job_id(
        mock_backend):
    r"""``TrainingTable(job_id)`` reads the job record, not an in-memory
    attribute -- the job record is the only thing that survives a fresh
    process, so this is what makes a registered job's custom table visible
    to a caller who only has its ID, not the object `from_source_table`
    itself returned.
    """
    from kumoapi.jobs import CustomTrainingTable
    from kumoapi.source_table import S3RelativeSource, S3SourceTable
    from kumoapi.train import TrainingTableSpec

    source = S3SourceTable(connector_id='connector-owned',
                           source=S3RelativeSource(path='labels'),
                           file_type=FileType.PARQUET)
    spec = TrainingTableSpec(weight_col='weight')
    custom = CustomTrainingTable(source_table=source, table_mod_spec=spec)
    mock_backend.train_table.get.return_value = _resource(
        JobStatus.DONE, custom_train_table=custom)

    table = TrainingTable('gen-traintable-job-registered',
                          project_id='project-owned')

    assert table._custom_train_table == custom
    mock_backend.train_table.get.assert_called_once_with(
        'gen-traintable-job-registered', project_id='project-owned')


def test_a_generated_jobs_custom_table_is_none(mock_backend):
    r"""A job `create()` generated carries no `custom_train_table` on its
    record, so reconstructing it recovers ``None`` -- not whatever the
    in-memory object that first returned it happened to carry.
    """
    mock_backend.train_table.get.return_value = _resource(JobStatus.DONE)

    table = TrainingTable('gen-traintable-job-plain',
                          project_id='project-owned')

    assert table._custom_train_table is None


def test_an_unanswerable_job_record_refuses_rather_than_guessing(mock_backend):
    r"""A record that says it registered a table but cannot produce the spec
    must not be read as "it had none" -- that trains on generated rows in
    exactly the case this exists to prevent. It raises instead, naming the
    job.

    This is the shape a deployment that records *that* a table was
    registered without recording *what* was registered returns, which is
    what the cloud did for every `from_source_table` call before the job
    config carried the spec.
    """
    mock_backend.train_table.get.return_value = _resource(
        JobStatus.DONE, custom_train_table=None, used=True)

    with pytest.raises(ValueError,
                       match="'unanswerable-job'.*registered a supplied"):
        TrainingTable('unanswerable-job', project_id='project-owned')


def test_a_local_file_source_needs_no_connector_and_no_project():
    from kumoapi.source_table import LocalFileSourceTable

    from kumoai_studio.pquery.training_table import _api_source_table

    source = LocalFileSourceTable(path='/tmp/weighted.parquet')

    assert _api_source_table(source, 'project-owned') is source


def test_the_cloud_backend_says_where_a_custom_table_has_to_live():
    from kumoai_studio.client.jobs import GenerateTrainTableJobAPI

    api = GenerateTrainTableJobAPI(mock.MagicMock())

    with pytest.raises(NotImplementedError, match='reads the rows from the '
                       'connector'):
        api.stage_custom_table(mock.MagicMock(), mock.MagicMock(), 'pq')


def _ok(payload):
    r"""A response the client's own parser accepts."""
    import json as _json

    response = mock.MagicMock()
    response.ok = True
    response.text = _json.dumps(payload)
    response.json.return_value = payload
    return response


def test_the_cloud_mints_a_job_for_a_table_it_did_not_generate():
    r"""The rows ride on the request the platform mints a job from.

    Registering is the create endpoint carrying a table instead of asking for
    one, so the ID is the server's exactly as it is for a generated table --
    which is what makes a supplied table nameable on the cloud at all.
    """
    from kumoapi.jobs import GenerateTrainTableRequest
    from kumoapi.model_plan import TrainingTableGenerationPlan
    from kumoapi.source_table import FileType, S3RelativeSource, S3SourceTable
    from kumoapi.train import TrainingTableSpec

    from kumoai_studio.client.jobs import GenerateTrainTableJobAPI

    transport = mock.MagicMock()
    transport._post.return_value = _ok({'id': 'gen-traintable-job-minted'})
    api = GenerateTrainTableJobAPI(transport)

    source = S3SourceTable(connector_id='connector-owned',
                           source=S3RelativeSource(path='labels'),
                           file_type=FileType.PARQUET)
    plan = TrainingTableGenerationPlan()
    spec = TrainingTableSpec(weight_col='weight')

    job_id = api.register_custom_table(
        GenerateTrainTableRequest({}, 'pq', plan, None), source, spec,
        project_id='project-owned')

    assert job_id == 'gen-traintable-job-minted'
    endpoint, = transport._post.call_args.args
    assert endpoint == '/gentraintable_jobs'
    body = transport._post.call_args.kwargs['json']
    assert transport._post.call_args.kwargs['params'] == {
        'project_id': 'project-owned'
    }
    # The plan still travels, because it is what the run reads its split out
    # of, and the rows travel beside it rather than instead of it.
    assert body['pquery_id'] == 'pq'
    assert body['plan'] is not None
    assert body['custom_train_table']['table_mod_spec']['weight_col'] == (
        'weight')
    assert body['custom_train_table']['source_table']['connector_id'] == (
        'connector-owned')


# --- the split a supplied table resolves ------------------------------------


def test_a_named_split_column_becomes_the_plan_split():
    from kumoai_studio.pquery.training_table import _plan_with_split

    plan = _plan_with_split(None, 'SPLIT')

    assert plan.split == ("ManualSplit(col='SPLIT', train_value=0, "
                          "val_value=1, test_value=2)")


def test_a_named_split_column_keeps_the_rest_of_the_plan():
    from kumoapi.model_plan import TrainingTableGenerationPlan

    from kumoai_studio.pquery.training_table import _plan_with_split

    plan = _plan_with_split(
        TrainingTableGenerationPlan(split='RandomSplit([0.8, 0.1, 0.1])',
                                    timeframe_step=7), 'SPLIT')

    assert plan.split.startswith('ManualSplit(')
    assert plan.timeframe_step == 7


def test_a_plan_split_is_used_when_no_column_is_named():
    from kumoapi.model_plan import TrainingTableGenerationPlan

    from kumoai_studio.pquery.training_table import _plan_with_split

    plan = TrainingTableGenerationPlan(split='RandomSplit([0.8, 0.1, 0.1])')

    assert _plan_with_split(plan, None) is plan


@pytest.mark.parametrize('pinned', [False, True])
def test_an_unresolved_split_is_refused_rather_than_defaulted(pinned):
    r"""A plan whose split is `inferred` is as unresolved as no plan at all:
    `_split_options` would substitute `_default_split` for both.
    """
    from kumoapi.model_plan import TrainingTableGenerationPlan

    from kumoai_studio.pquery.training_table import _plan_with_split

    supplied = TrainingTableGenerationPlan() if pinned else None

    with pytest.raises(ValueError, match='needs a split, and this one names'):
        _plan_with_split(supplied, None)


def test_the_split_column_is_the_specifications_alone(monkeypatch):
    r"""It used to fall back to a deployment-wide environment variable, which
    the cloud read *instead of* the specification -- so a table registered
    with ``split_col`` had its own column silently ignored there. The scope
    is per table on both sides now, which is the scope DiskGraph names it at
    (``split_contract.py`` takes a ``split_column`` parameter).
    """
    from kumoai_studio.pquery.training_table import _plan_with_split

    monkeypatch.setenv('KUMO_CUSTOM_TT_MANUAL_SPLIT_COLUMN', '__kumo_parity')

    assert "col='SPLIT'" in _plan_with_split(None, 'SPLIT').split
    assert "col='fold'" in _plan_with_split(None, 'fold').split
    # The environment names nothing any more, even when it is set.
    with pytest.raises(ValueError, match='needs a split'):
        _plan_with_split(None, None)


def test_training_export_config_preserves_positional_arguments():
    connector = object.__new__(S3Connector)
    config = TrainingTableExportConfig({'training_table'}, connector,
                                       'exports')
    assert config.output_connector is connector
    assert config.output_table_name == 'exports'


@pytest.mark.parametrize('missing', ['output_connector', 'output_table_name'])
def test_training_export_config_requires_destination(missing):
    arguments: dict[str, Any] = {
        'output_types': {'training_table'},
        'output_connector': object.__new__(S3Connector),
        'output_table_name': 'exports',
    }
    del arguments[missing]
    with pytest.raises(ValueError, match=f'{missing} is required'):
        TrainingTableExportConfig(**arguments)
