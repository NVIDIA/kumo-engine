# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import os
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import kumoapi.data_source as data_source_api
import kumoapi.jobs as jobs_api
import pytest
from kumo_core.model_plan import ModelPlan, TrainingTableGenerationPlan
from kumo_core.types import PredictionArtifactType, TaskType
from kumo_core.types.schema import FPID
from kumoapi.jobs import CustomTrainingTable as ApiCustomTrainingTable
from kumoapi.jobs import ErrorDetail, ErrorType
from kumoapi.model_plan import PredictionTableGenerationPlan
from kumoapi.pquery import QueryType
from kumoapi.source_table import ADLSURISource
from kumoapi.source_table import ConnectorType as SourceConnectorType
from kumoapi.source_table import (
    CustomSQLSourceTable,
    DatabricksSourceTable,
    FileType,
    GCSURISource,
    S3RelativeSource,
    S3SourceTable,
)
from kumoapi.train import TrainingTableSpec
from kumoapi.typing import Dtype, Stype

from io_utils.fsutil.impl_local import LocalPresignedHelper
from io_utils.fsutil.uri import LocalURI
from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph import Column, ForeignKey, Graph, Table
from kumo.train.artifact_manifest import (
    PREDICTION_ARTIFACT_MANIFEST_VERSION,
    PREDICTION_PRODUCER_ABI,
    PredictionSelectionManifest,
    SelectedPredictionArtifact,
)
from kumo.validation import ValidationException
from kumo.validation.validation import ValidationError, ValidationResponse
# Keep this import stable across yapf and isort.
# yapf: disable
from rest.backends.ml_backend.activities.materialize_prediction_task import (
    _configure_prediction_output_fields,
)
# yapf: enable
from rest.config.batch_prediction_job import (
    BatchPredictionJobConfig,
    BatchPredictionJobOutput,
    PaginationArgs,
)
from rest.config.data_backend import ConnectorExecutionRef, ConnectorSpec
from rest.config.graph_v2 import GraphDefinition
from rest.config.predictive_query import InsertFilterOverridesRequest
from rest.driver.error_constants import (
    BATCH_PREDICTION_JOB_NOT_FOUND_MESSAGE,
    BATCH_PREDICTION_JOBS_DELETE_MESSAGE,
    PREDICTION_ARTIFACT_TYPE_NOT_FOUND_MESSAGE,
    PREDICTION_TABLE_JOB_FAILED_TITLE,
    TRAINING_JOB_INCOMPLETE_TITLE,
)
from rest.driver.pquery_api_driver import ForecastingParams, PQueryApiDriver
from rest.driver.pquery_logic import PQueryLogic
from rest.utils.routers import ValidationHTTPException
from statedb.config.common import (
    BatchPredictionJobID,
    ConnectorID,
    ErrWarnDetails,
    GeneratePredictionTableJobID,
    GenerateTrainTableJobID,
    GraphIDv2,
    JobEventLogEntry,
    JobFailureInfo,
    JobSource,
    JobStatus,
    JobType,
    PredictiveQueryID,
    ProjectID,
    TrainingJobID,
)
from statedb.config.job_tag import (
    TAG_KEY_PARENT_JOB,
    TAG_KEY_PQUERY_ID,
    TAG_KEY_PQUERY_NAME,
    JobTagKey,
    JobTags,
    JobTagValue,
)
from statedb.config.predictive_query import PQueryDefinition
from statedb.config.timestamp import (
    ResourceTimestamps,
    get_current_timestamps,
    now_plus_seconds_utc,
)
from statedb.db.ml_jobs import (
    GenerateTrainTableJobConfig,
    GenerateTrainTableJobDBEntry,
    GenerateTrainTableJobOutput,
)
from statedb.ml.jobs import BatchPredictionJobDBEntry
from statedb.rest.connector_authorization import CONNECTOR_GRAPH_ACTIONS
from statedb.utils.errors import (
    BatchPredictionDoesNotExistError,
    ProjectNotFoundError,
    TrainingJobDoesNotExistError,
)

_TEST_PROJECT_ID = ProjectID("project_" + "1" * 32)
_OTHER_PROJECT_ID = ProjectID("project_" + "2" * 32)


def _project_service():
    return SimpleNamespace(get_project=AsyncMock(return_value=SimpleNamespace(
        id=_TEST_PROJECT_ID)))


def _request_ctx(*project_ids: ProjectID):
    visible = project_ids or (_TEST_PROJECT_ID, )
    return SimpleNamespace(
        user_authorization=SimpleNamespace(
            is_admin=False,
            project_ids=tuple(str(project_id) for project_id in visible),
        ),
        is_customer=True,
    )


def _pquery_driver(**overrides) -> PQueryApiDriver:
    """Build a driver with the neutral dependencies used by unit tests."""
    state_services = overrides.pop("state_services", None)
    kwargs = {
        "ml_job_backend": None,
        "data_backend_ctx": None,
        "allow_local_batch_pred_paths": True,
        "data_scale_limits": None,
        "pquery_settings": None,
        "table_driver": None,
    }
    kwargs.update(overrides)
    driver = PQueryApiDriver(**kwargs)
    if state_services is not None:
        driver._configure_state_services(state_services)
    return driver


_ARTIFACT_TYPES = (
    PredictionArtifactType.PREDICTIONS,
    PredictionArtifactType.EMBEDDINGS,
)


@pytest.mark.parametrize(
    ('source', 'config', 'expected_path'),
    [
        (
            jobs_api.S3PredictionTableSource(
                connector_id=str(ConnectorID.generate()),
                source=S3RelativeSource(path='prediction.parquet'),
            ),
            data_source_api.S3ConnectorResourceConfig(
                target=data_source_api.S3RootedConnectorTarget(
                    bucket='customer-data', prefix='root'), ),
            'prediction.parquet',
        ),
        (
            jobs_api.GCSPredictionTableSource(
                connector_id=str(ConnectorID.generate()),
                source=GCSURISource(
                    uri='gs://customer-data/root/prediction.parquet'),
            ),
            data_source_api.GCSConnectorResourceConfig(
                target=data_source_api.GCSRootedConnectorTarget(
                    bucket='customer-data', prefix='root'), ),
            'prediction.parquet',
        ),
        (
            jobs_api.ADLSPredictionTableSource(
                connector_id=str(ConnectorID.generate()),
                source=ADLSURISource(
                    uri=('abfss://filesystem@account.dfs.core.windows.net/'
                         'root/prediction.parquet')),
            ),
            data_source_api.ADLSConnectorResourceConfig(
                target=data_source_api.ADLSRootedConnectorTarget(
                    storage_account='account',
                    filesystem='filesystem',
                    prefix='root',
                ), ),
            'prediction.parquet',
        ),
        (
            jobs_api.DatabricksPredictionTableSource(
                connector_id=str(ConnectorID.generate()),
                table='prediction_input',
            ),
            data_source_api.DatabricksConnectorResourceConfig(
                catalog='catalog',
                schema_name='schema',
                warehouse_id='warehouse',
            ),
            None,
        ),
    ],
    ids=['s3', 'gcs', 'adls', 'databricks'],
)
@pytest.mark.asyncio
async def test_custom_prediction_source_authorizes_then_canonicalizes(
        source, config, expected_path):
    connector_id = ConnectorID(source.connector_id)
    reference = ConnectorExecutionRef(connector_id=connector_id,
                                      project_id=_TEST_PROJECT_ID)
    connectors = SimpleNamespace(
        get_for_project=AsyncMock(return_value=SimpleNamespace(config=config)),
        execution_ref_for_project=AsyncMock(return_value=reference),
    )
    driver = _pquery_driver(state_services=SimpleNamespace(
        connectors=connectors))

    normalized, actual_reference = (
        await driver._authorize_custom_prediction_table_source(
            source, _TEST_PROJECT_ID))

    connectors.get_for_project.assert_awaited_once_with(_TEST_PROJECT_ID,
                                                        connector_id, None)
    if expected_path is None:
        assert normalized is source
    else:
        assert normalized.source.path == expected_path
    assert actual_reference == reference


def _artifact_method(driver, artifact_type):
    if artifact_type == PredictionArtifactType.PREDICTIONS:
        return driver.get_presigned_prediction_urls
    return driver.get_presigned_embedding_urls


def _artifact_partitions(response, artifact_type):
    if artifact_type == PredictionArtifactType.PREDICTIONS:
        return response.prediction_partitions
    return response.embedding_partitions


def _configure_batch_prediction_artifact(driver, artifact_type):
    output = SimpleNamespace(artifact_type=artifact_type)
    job = SimpleNamespace(config=SimpleNamespace(outputs=[output]))
    driver.get_batch_prediction_job = AsyncMock(return_value=job)


def _configure_implicit_batch_prediction_artifact(driver):
    job = SimpleNamespace(config=SimpleNamespace(outputs=[]))
    driver.get_batch_prediction_job = AsyncMock(return_value=job)


def _prediction_selection(output_root, artifact_type, files):
    selection_root = os.path.join(output_root, '_selections', 'selection-1',
                                  'attempts', 'temporal-00001')
    artifact_root = os.path.dirname(files[0])
    return PredictionSelectionManifest(
        schema_version=PREDICTION_ARTIFACT_MANIFEST_VERSION,
        producer_abi=PREDICTION_PRODUCER_ABI,
        job_id='test-bp-outputs',
        selection_id='selection-1',
        selection_attempt_id='temporal-00001',
        graph_snapshot_id='graph-snapshot-1',
        source_config_sha256='a' * 64,
        canonical_output_root=output_root,
        selection_root=selection_root,
        shard_manifest_sha256s=('b' * 64, ),
        artifacts=(SelectedPredictionArtifact(
            artifact_type=artifact_type,
            roots=(artifact_root, ),
            file_count=len(files),
            schema_fingerprint='c' * 64,
            num_rows=len(files),
            num_cols=1,
            size_bytes=len(files),
            physical_size_bytes=5 * len(files),
        ), ),
    )


def _selected_pointer_exists(path):
    return str(path).endswith('/_selected_artifacts.json') or os.path.exists(
        path)


@pytest.fixture(autouse=True)
def auto_cleanup_per_func():
    yield


@pytest.fixture
def setup_rest_test_session() -> None:
    return None


def _assert_s3_url_starts_with(url: str, expected_path: str) -> bool:
    """
    Helper function to check if a URL starts with the expected S3 path,
    allowing for both legacy (s3.amazonaws.com) and region-specific
    (s3.<region>.amazonaws.com) formats.
    """
    # Pattern for legacy format: s3.amazonaws.com
    legacy_pattern = (f"https://kumo-public-datasets\\.s3\\.amazonaws\\.com/"
                      f"{re.escape(expected_path)}")
    # Pattern for region-specific format: s3.<region>.amazonaws.com
    regional_pattern = (
        f"https://kumo-public-datasets\\.s3\\.[a-z0-9-]+\\.amazonaws\\.com/"
        f"{re.escape(expected_path)}")

    return bool(
        re.match(legacy_pattern, url) or re.match(regional_pattern, url))


def _get_hm_graph() -> Graph:
    """Helper function to generate a test graph with customers and
    transactions tables."""
    return Graph(
        data_source_dict={
            's3':
            ParquetConnector(
                name='s3',
                root_dir='/tmp/data',
                suffix='parquet',
            )
        },
        tables=[
            Table(
                table_name='customers',
                data_source='s3',
                cols=[
                    Column(
                        name='id',
                        stype=Stype.ID,
                        dtype=Dtype.int32,
                    ),
                    Column(
                        name='active',
                        stype=Stype.categorical,
                        dtype=Dtype.int32,
                    ),
                ],
                pkey='id',
            ),
            Table(
                table_name='transactions',
                data_source='s3',
                cols=[
                    Column(
                        name='t_dat',
                        stype=Stype.timestamp,
                        dtype=Dtype.time,
                    ),
                    Column(
                        name='customer_id',
                        stype=Stype.ID,
                        dtype=Dtype.int32,
                    ),
                    Column(
                        name='article_id',
                        stype=Stype.ID,
                        dtype=Dtype.int32,
                    ),
                    Column(
                        name='sales_channel',
                        stype=Stype.categorical,
                        dtype=Dtype.int16,
                    ),
                    Column(
                        name='price',
                        stype=Stype.numerical,
                        dtype=Dtype.float32,
                    ),
                ],
                time_col='t_dat',
                fkeys=[
                    ForeignKey(
                        dst_table='customers',
                        name='customer_id',
                        dst_key='id',
                    ),
                    ForeignKey(
                        dst_table='articles',
                        name='article_id',
                        dst_key='article_id',
                    ),
                ],
            ),
            Table(
                table_name='articles',
                data_source='s3',
                cols=[
                    Column(
                        name='article_id',
                        stype=Stype.ID,
                        dtype=Dtype.int32,
                    ),
                    Column(
                        name='product_type',
                        stype=Stype.categorical,
                        dtype=Dtype.int32,
                    ),
                ],
                pkey='article_id',
            ),
        ],
    )


_REGRESSION_QUERY = ('PREDICT SUM(transactions.price WHERE '
                     'transactions.sales_channel=1, 0, 30) FOR EACH '
                     'customers.id WHERE customers.active = 1')
_LINK_PREDICTION_QUERY = ('PREDICT LIST_DISTINCT(transactions.article_id WHERE '
                          '(transactions.sales_channel = 1) AND '
                          '(articles.product_type = 272), 0, 90, days) '
                          'RANK TOP 10 FOR EACH customers.id '
                          'WHERE COUNT(transactions.*, -90, -15) > 0')


def _filter_override_driver(query_string: str, task_type: TaskType):
    driver = _pquery_driver()
    training_job = SimpleNamespace(
        pquery_def_fpid=FPID("d4ebace23c174078b9db6edda1c7eb0a"),
        project_id=_TEST_PROJECT_ID,
    )
    driver.get_training_job = AsyncMock(return_value=training_job)
    driver.get_parsed_pq_v2_by_fpid = AsyncMock(return_value=SimpleNamespace(
        config=SimpleNamespace(
            graph=_get_hm_graph(),
            query_string=Mock(return_value=query_string),
        ),
        task_type=task_type,
    ))
    return driver


class _MLJobTagReadService:
    def __init__(self, tags_by_job, batch_prediction_rows=None,
                 training_rows=None):
        self.tags_by_job = tags_by_job
        self.batch_prediction_rows = batch_prediction_rows or {}
        self.training_rows = training_rows or {}
        self.calls = []
        self.list_calls = []
        self.delete_calls = []
        self.apply_calls = []
        self.batch_prediction_row_calls = []
        self.batch_prediction_exists_calls = []
        self.current_batch_prediction_calls = []
        self.batch_prediction_project_scope_calls = []

    async def list_training_job_rows(
        self,
        *,
        job_ids,
        project_ids,
        sort_created_at="asc",
        limit=None,
    ):
        rows = [
            self.training_rows[job_id] for job_id in job_ids
            if job_id in self.training_rows and (
                project_ids is None
                or self.training_rows[job_id].project_id in project_ids)
        ]
        rows.sort(
            key=lambda row: row.timestamps.created_at,
            reverse=sort_created_at == "desc",
        )
        return rows[:limit] if limit is not None else rows

    async def get_job_tags(self, job_type, job_id):
        self.calls.append((job_type, job_id))
        return self.tags_by_job.get((job_type, job_id))

    async def list_job_tags(self, job_type, tag_filters, *, job_ids=None):
        if job_ids is not None:
            job_ids = set(job_ids)
            self.list_calls.append((job_type, job_ids))
            return {
                job_id: tags
                for (stored_type, job_id), tags in self.tags_by_job.items()
                if stored_type == job_type and job_id in job_ids
            }
        self.list_calls.append((job_type, tag_filters))
        return {
            job_id: tags
            for (stored_type, job_id), tags in self.tags_by_job.items()
            if stored_type == job_type and tags.match(tag_filters)
        }

    async def delete_job_tag(self, job_type, job_id, tag):
        self.delete_calls.append((job_type, job_id, tag))
        job_tags = self.tags_by_job.get((job_type, job_id))
        if job_tags is None or tag not in job_tags.tags:
            return False
        job_tags.tags.pop(tag)
        return True

    async def delete_job_tags(self, job_type, job_id):
        self.delete_calls.append((job_type, job_id, None))
        return self.tags_by_job.pop((job_type, job_id), None) is not None

    async def apply_job_tags(self, job_type, job_id, tags):
        self.apply_calls.append((job_type, job_id, dict(tags)))
        existing = self.tags_by_job.get((job_type, job_id))
        if existing is None:
            existing = JobTags(
                timestamps=get_current_timestamps(),
                tags={},
            )
            self.tags_by_job[(job_type, job_id)] = existing
        existing.tags.update(tags)
        return True

    async def get_batch_prediction_rows(self, ids, *, project_ids):
        self.batch_prediction_project_scope_calls.append(
            ("get", None if project_ids is None else set(project_ids)))
        self.batch_prediction_row_calls.append(set(ids))
        return [
            self.batch_prediction_rows[id] for id in ids
            if id in self.batch_prediction_rows and (
                project_ids is None
                or self.batch_prediction_rows[id].project_id in project_ids)
        ]

    async def list_batch_prediction_rows(
        self,
        *,
        project_ids,
        model_id=None,
        pquery_id=None,
        job_ids=None,
        status=None,
        sort_created_at="asc",
        limit=None,
        offset=0,
        match_pquery_def_fpid=True,
    ):
        self.batch_prediction_project_scope_calls.append(
            ("list", None if project_ids is None else set(project_ids)))
        rows = [
            row for row in self.batch_prediction_rows.values()
            if (project_ids is None or row.project_id in project_ids)
            and self._batch_prediction_row_matches(
                row,
                model_id=model_id,
                pquery_id=pquery_id,
                job_ids=job_ids,
                status=status,
                match_pquery_def_fpid=match_pquery_def_fpid,
            )
        ]
        rows.sort(
            key=lambda row: row.timestamps.created_at,
            reverse=sort_created_at == "desc",
        )
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        self.batch_prediction_row_calls.append({row.id for row in rows})
        return rows

    async def count_batch_prediction_rows(
        self,
        *,
        project_ids,
        model_id=None,
        pquery_id=None,
        job_ids=None,
        status=None,
        match_pquery_def_fpid=True,
    ):
        self.batch_prediction_project_scope_calls.append(
            ("count", None if project_ids is None else set(project_ids)))
        return len([
            row for row in self.batch_prediction_rows.values()
            if (project_ids is None or row.project_id in project_ids)
            and self._batch_prediction_row_matches(
                row,
                model_id=model_id,
                pquery_id=pquery_id,
                job_ids=job_ids,
                status=status,
                match_pquery_def_fpid=match_pquery_def_fpid,
            )
        ])

    async def has_batch_prediction_rows(
        self,
        *,
        project_ids,
        model_id=None,
        pquery_id=None,
        job_ids=None,
        status=None,
        match_pquery_def_fpid=True,
    ):
        self.batch_prediction_exists_calls.append({
            "project_ids":
            None if project_ids is None else set(project_ids),
            "model_id":
            model_id,
            "pquery_id":
            pquery_id,
            "job_ids":
            None if job_ids is None else set(job_ids),
            "status":
            status,
            "match_pquery_def_fpid":
            match_pquery_def_fpid,
        })
        return any((project_ids is None or row.project_id in project_ids)
                   and self._batch_prediction_row_matches(
                       row,
                       model_id=model_id,
                       pquery_id=pquery_id,
                       job_ids=job_ids,
                       status=status,
                       match_pquery_def_fpid=match_pquery_def_fpid,
                   ) for row in self.batch_prediction_rows.values())

    async def get_current_batch_prediction_job_ids(self, pquery_ids, *,
                                                   project_ids):
        self.batch_prediction_project_scope_calls.append(
            ("current", None if project_ids is None else set(project_ids)))
        self.current_batch_prediction_calls.append(list(pquery_ids))
        result = {}
        for pquery_id in set(pquery_ids):
            rows = [
                row for row in self.batch_prediction_rows.values()
                if (project_ids is None or row.project_id in project_ids)
                and self._batch_prediction_row_matches(
                    row,
                    model_id=None,
                    pquery_id=pquery_id,
                    job_ids=None,
                    status=None,
                    match_pquery_def_fpid=False,
                )
            ]
            if rows:
                result[pquery_id] = max(
                    rows,
                    key=lambda row: row.timestamps.created_at,
                ).id
        return result

    @staticmethod
    def _batch_prediction_row_matches(
        row,
        *,
        model_id,
        pquery_id,
        job_ids,
        status,
        match_pquery_def_fpid,
    ):
        config = row.config
        if config is None:
            return False
        pquery_matches = pquery_id is None or config.predictive_query_id == pquery_id
        if not pquery_matches and match_pquery_def_fpid:
            _, fpid = PredictiveQueryID.parse(pquery_id)
            pquery_matches = (fpid is not None and
                              PredictiveQueryID.compose(fpid=fpid) == pquery_id
                              and config.pquery_def_fpid == fpid)
        return ((model_id is None or config.training_job_id == model_id)
                and pquery_matches and (job_ids is None or row.id in job_ids)
                and (status is None or row.execution.status == status))

    async def list_job_ids_by_type(self, job_type):
        if job_type == JobType.BATCH_PREDICTION_JOB:
            return list(self.batch_prediction_rows)
        return [
            job_id for stored_type, job_id in self.tags_by_job
            if stored_type == job_type
        ]


class _PQueryDefinitionReadService:
    def __init__(self, definitions, aliases=None):
        self.definitions = definitions
        self.aliases = aliases or {}
        self.alias_calls = []
        self.add_calls = []
        self.definition_calls = []
        self.definition_by_id_calls = []
        self.definitions_by_id_calls = []
        self.definitions_calls = []
        self.list_definition_calls = 0
        self.list_alias_calls = 0
        self.required_calls = []
        self.set_alias_calls = []
        self.delete_alias_calls = []
        self.delete_definition_calls = []

    async def get_alias(self, name):
        self.alias_calls.append(name)
        return self.aliases.get(name)

    async def resolve_fpids(self, pquery_ids):
        return {
            pquery_id: fpid
            for pquery_id in pquery_ids
            if (fpid := self.aliases.get(str(pquery_id))) is not None
        }

    async def list_aliases(self, name_pattern=None):
        self.list_alias_calls += 1
        if name_pattern is None:
            return list(self.aliases)
        return [
            name for name in self.aliases
            if re.compile(name_pattern).search(name)
        ]

    async def get_definition(self, fpid):
        self.definition_calls.append(fpid)
        return self.definitions.get(fpid)

    async def get_definitions(self, fpids):
        fpids = set(fpids)
        self.definitions_calls.append(fpids)
        return {
            fpid: self.definitions[fpid]
            for fpid in fpids if fpid in self.definitions
        }

    async def list_definitions(self):
        self.list_definition_calls += 1
        return list(self.definitions.values())

    async def get_definition_by_id(self, pquery_id):
        self.definition_by_id_calls.append(pquery_id)
        name, fpid = PredictiveQueryID.parse(pquery_id)
        if fpid is None and name is not None:
            fpid = self.aliases.get(str(name))
        return self.definitions.get(fpid) if fpid is not None else None

    async def get_definitions_by_ids(self, pquery_ids):
        pquery_ids = list(pquery_ids)
        self.definitions_by_id_calls.append(pquery_ids)
        definitions = {}
        for pquery_id in pquery_ids:
            name, fpid = PredictiveQueryID.parse(pquery_id)
            if fpid is None and name is not None:
                fpid = self.aliases.get(str(name))
            if fpid in self.definitions:
                definitions[pquery_id] = self.definitions[fpid]
        return definitions

    async def get_definition_required(self, fpid):
        self.required_calls.append(fpid)
        return self.definitions[fpid]

    async def add_definition(self, definition):
        self.add_calls.append(definition)
        self.definitions[definition.fpid] = definition
        return definition.fpid

    async def set_alias(self, name, fpid):
        self.set_alias_calls.append((name, fpid))
        self.aliases[name] = fpid

    async def delete_alias(self, name):
        self.delete_alias_calls.append(name)
        self.aliases.pop(name, None)

    async def delete_definition(self, fpid):
        self.delete_definition_calls.append(fpid)
        return self.definitions.pop(fpid, None) is not None


class _GraphDefinitionReadService:
    def __init__(self, definitions):
        self.definitions = definitions
        self.definition_calls = []
        self.definitions_calls = []

    async def get_graph_definition(self, project_id, fpid):
        self.definition_calls.append((project_id, fpid))
        return self.definitions.get(fpid)

    async def get_graph_definitions(self, project_id, fpids):
        fpids = set(fpids)
        self.definitions_calls.append((project_id, fpids))
        return {
            fpid: self.definitions[fpid]
            for fpid in fpids if fpid in self.definitions
        }


class _GraphDefinitionDeleteService:
    def __init__(self):
        self.get_entry_calls = []
        self.decrement_or_delete_calls = []

    async def get_graph_entry(self, project_id, fpid):
        self.get_entry_calls.append((project_id, fpid))
        return SimpleNamespace(id=fpid, project_id=project_id)

    async def decrement_or_delete_graph_definition(self, project_id, fpid):
        self.decrement_or_delete_calls.append((project_id, fpid))
        return True


class _ConnectorSpecReadService:
    def __init__(self, *, project_upload=False):
        self.calls = []
        self.project_upload = project_upload

    async def source_table_connector_for_project(self, project_id, connector_id,
                                                 principal):
        self.calls.append(("source", project_id, connector_id, principal))
        if self.project_upload:
            return None
        return SimpleNamespace()

    async def execution_ref_for_project(self, principal, project_id,
                                        connector_id, actions, *,
                                        expected_type=None):
        self.calls.append(("execution", principal, project_id, connector_id,
                           actions, expected_type))
        return ConnectorExecutionRef(project_id=project_id,
                                     connector_id=connector_id)


class _MaterializationInfo:
    def __init__(self):
        self.cleared = False

    def clear_private_fields(self):
        self.cleared = True


def _batch_prediction_row(
    *,
    job_id: BatchPredictionJobID,
    pquery_fpid: FPID,
    training_job_id: TrainingJobID,
) -> BatchPredictionJobDBEntry:
    now = get_current_timestamps()
    return BatchPredictionJobDBEntry(
        id=job_id,
        timestamps=now,
        config=BatchPredictionJobConfig(
            predictive_query_id=PredictiveQueryID.public(pquery_fpid),
            training_job_id=training_job_id,
            task_type=TaskType.REGRESSION,
            generate_pred_table_job_id=GeneratePredictionTableJobID.
            make_random_id(),
            graph_id=GraphIDv2.public(FPID("11111111111111111111111111111111")),
            pquery_def_fpid=pquery_fpid,
        ),
        execution=BatchPredictionJobOutput(
            status=JobStatus.QUEUED,
            log=JobEventLogEntry.not_started(now.created_at),
        ),
        description="batch prediction scoring",
        batch_prediction_workflow_id="workflow_1",
        source=JobSource.SDK,
        user="user@example.com",
        project_id=_TEST_PROJECT_ID,
        kumo_version="dev",
    )


@pytest.mark.asyncio
async def test_pquery_job_tag_reads_use_state_service_when_configured():
    training_job_id = TrainingJobID.make_random_id()
    pquery_id = PredictiveQueryID("riskpquery")
    job_tags = JobTags(
        timestamps=get_current_timestamps(), tags={
            TAG_KEY_PQUERY_NAME:
            JobTagValue.of_pquery_name(pquery_id),
            TAG_KEY_PQUERY_ID:
            JobTagValue.of_pquery_id(FPID("0123456789abcdef0123456789abcdef")),
        })
    tag_service = _MLJobTagReadService({
        (JobType.TRAINING_JOB, training_job_id):
        job_tags,
    })
    training_job = SimpleNamespace(
        id=training_job_id,
        pquery_id=None,
        tags=None,
        project_id=_TEST_PROJECT_ID,
    )
    ml_job_backend = SimpleNamespace(
        get_training_job=AsyncMock(return_value=training_job),
        get_job_tags_if_exists=AsyncMock(
            side_effect=AssertionError("backend tag scalar read used")),
        list_job_tags=AsyncMock(
            side_effect=AssertionError("backend tag list read used")),
    )
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=tag_service,
            projects=_project_service(),
        ))

    readback = await pquery.get_training_job(
        ctx=SimpleNamespace(user_authorization=object()),
        id=training_job_id,
    )

    assert readback.pquery_id == pquery_id
    assert readback.tags == job_tags.tags
    assert await pquery.get_training_job_pquery_id(training_job_id) == pquery_id
    assert await pquery.get_job_tags(JobType.TRAINING_JOB,
                                     training_job_id) == job_tags
    assert await pquery.list_job_tags(
        JobType.TRAINING_JOB,
        {TAG_KEY_PQUERY_NAME: pquery_id},
    ) == {
        training_job_id: job_tags
    }
    with pytest.raises(KeyError):
        await pquery.get_job_tags(
            JobType.TRAINING_JOB,
            TrainingJobID.make_random_id(),
        )
    assert tag_service.calls[:3] == [
        (JobType.TRAINING_JOB, training_job_id),
        (JobType.TRAINING_JOB, training_job_id),
        (JobType.TRAINING_JOB, training_job_id),
    ]
    assert len(tag_service.calls) == 4
    assert tag_service.calls[3][0] == JobType.TRAINING_JOB
    assert tag_service.calls[3][1] != training_job_id
    assert tag_service.list_calls == [
        (JobType.TRAINING_JOB, {
            TAG_KEY_PQUERY_NAME: pquery_id
        }),
    ]


@pytest.mark.asyncio
async def test_generate_pred_table_lookup_requires_exact_visible_project():
    job_id = GeneratePredictionTableJobID.make_random_id()
    backend = SimpleNamespace(get_generate_pred_table_jobs=AsyncMock(
        return_value=[]))
    project_service = _project_service()
    pquery = _pquery_driver(
        ml_job_backend=backend, state_services=SimpleNamespace(
            ml_jobs=SimpleNamespace(),
            projects=project_service,
        ))
    ctx = _request_ctx(_TEST_PROJECT_ID)

    with pytest.raises(KeyError, match="not found"):
        await pquery.get_generate_pred_table_job(ctx, job_id, _TEST_PROJECT_ID)

    project_service.get_project.assert_awaited_once_with(
        _TEST_PROJECT_ID, ctx.user_authorization)
    backend.get_generate_pred_table_jobs.assert_awaited_once_with(
        {job_id}, project_ids={_TEST_PROJECT_ID})


def test_pquery_configures_internal_graph_driver_state_services():
    pquery = _pquery_driver(ml_job_backend=SimpleNamespace(), )
    state_services = SimpleNamespace(
        ml_jobs=SimpleNamespace(),
        projects=_project_service(),
    )

    pquery._configure_state_services(state_services)

    assert pquery._state_services is state_services
    assert pquery._graph_driver._state_services is state_services


@pytest.mark.asyncio
async def test_graph_data_compute_input_uses_state_services_without_opening_dao(
):
    graph_id = GraphIDv2.compose(fpid=FPID("11111111111111111111111111111111"))
    graph_data_compute_input = Mock()
    graph_service = SimpleNamespace(get_graph_data_compute_input=AsyncMock(
        return_value=graph_data_compute_input))
    state_services = SimpleNamespace(graphs=graph_service)
    pquery = _pquery_driver(ml_job_backend=SimpleNamespace(), )
    pquery._configure_state_services(state_services)

    with patch("rest.driver.pquery_api_driver.GraphLogic") as graph_logic_cls:
        result = await pquery._get_graph_data_compute_input(
            graph_id, _TEST_PROJECT_ID)

    assert result is graph_data_compute_input
    graph_service.get_graph_data_compute_input.assert_awaited_once_with(
        graph_id,
        project_id=_TEST_PROJECT_ID,
        user_authorization=None,
    )
    graph_logic_cls.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('project_upload', 'path', 'source_attr', 'expected_type'),
    [
        (False, 'custom_train', 'table', data_source_api.ConnectorType.S3),
        (True, 'custom_train.parquet', 'table_name', None),
    ],
)
async def test_custom_train_table_conversion_uses_project_execution_ref(
    project_upload,
    path,
    source_attr,
    expected_type,
):
    connector_id = ConnectorID("connector_" + "c" * 32)
    connector_service = _ConnectorSpecReadService(project_upload=project_upload)
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            connectors=connector_service,
            projects=_project_service(),
        ))

    result = await pquery._from_custom_tt(
        _TEST_PROJECT_ID,
        ApiCustomTrainingTable(
            source_table=S3SourceTable(
                connector_id=str(connector_id),
                source={
                    'mode': 'relative',
                    'path': path
                },
                file_type=FileType.PARQUET,
            ),
            table_mod_spec=TrainingTableSpec(
                weight_col="weight", timestamp_unit="milliseconds"),
            validated=True,
        ),
    )

    assert result is not None
    assert result.timestamp_unit == "milliseconds"
    assert result.connector_spec == ConnectorExecutionRef(
        project_id=_TEST_PROJECT_ID,
        connector_id=connector_id,
    )
    assert getattr(result.source_table, source_attr) == path
    assert result.weight_col_name == "weight"
    assert result.validated is True
    assert connector_service.calls == [
        ("source", _TEST_PROJECT_ID, connector_id, None),
        (
            "execution",
            None,
            _TEST_PROJECT_ID,
            connector_id,
            CONNECTOR_GRAPH_ACTIONS,
            expected_type,
        ),
    ]


@pytest.mark.asyncio
async def test_a_custom_table_naming_no_weight_column_converts_without_one():
    """A supplied table may add nothing at all.

    The weight column is optional on the record a training run carries, so a
    specification naming none converts rather than tripping an assertion --
    which is what makes "here is my complete table, unmodified" expressible
    on the cloud.
    """
    connector_id = ConnectorID("connector_" + "e" * 32)
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            connectors=_ConnectorSpecReadService(),
            projects=_project_service(),
        ))

    result = await pquery._from_custom_tt(
        _TEST_PROJECT_ID,
        ApiCustomTrainingTable(
            source_table=S3SourceTable(
                connector_id=str(connector_id),
                source={
                    'mode': 'relative',
                    'path': 'unmodified'
                },
                file_type=FileType.PARQUET,
            ),
            table_mod_spec=TrainingTableSpec(),
        ),
    )

    assert result is not None
    assert result.weight_col_name is None


@pytest.mark.asyncio
async def test_a_registered_table_converts_like_any_other_custom_one():
    """Registering resolves the rows through the same conversion."""
    connector_id = ConnectorID("connector_" + "f" * 32)
    connector_service = _ConnectorSpecReadService()
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            connectors=connector_service,
            projects=_project_service(),
        ))

    result = await pquery._registered_custom_tt(
        _TEST_PROJECT_ID,
        ApiCustomTrainingTable(
            source_table=S3SourceTable(
                connector_id=str(connector_id),
                source={
                    'mode': 'relative',
                    'path': 'labels'
                },
                file_type=FileType.PARQUET,
            ),
            table_mod_spec=TrainingTableSpec(weight_col='weight'),
        ),
    )

    assert result.weight_col_name == 'weight'
    # The connector is reached for through a project-scoped execution ref,
    # exactly as the training path reaches for it.
    assert result.connector_spec == ConnectorExecutionRef(
        project_id=_TEST_PROJECT_ID,
        connector_id=connector_id,
    )


@pytest.mark.asyncio
async def test_a_databricks_table_registers_like_any_other_custom_one():
    """The direct, read-in-place Databricks path this used to require a
    generated table to project onto is retired (kumo#30914): a Databricks
    source now stages through the same connector export path
    (``ingest_custom_train_table`` -> ``export_table_read_only``) as any
    other warehouse, so registering one converts exactly like an S3 or
    Snowflake table rather than being refused.
    """
    connector_id = ConnectorID("connector_" + "a" * 32)
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            connectors=_ConnectorSpecReadService(),
            projects=_project_service(),
        ))

    result = await pquery._registered_custom_tt(
        _TEST_PROJECT_ID,
        ApiCustomTrainingTable(
            source_table=DatabricksSourceTable(
                databricks_connector_id=str(connector_id),
                table='custom_train',
            ),
            table_mod_spec=TrainingTableSpec(weight_col='weight'),
        ),
    )

    assert result.weight_col_name == 'weight'
    assert result.connector_spec == ConnectorExecutionRef(
        project_id=_TEST_PROJECT_ID,
        connector_id=connector_id,
    )
    assert isinstance(result.source_table, DatabricksSourceTable)


@pytest.mark.asyncio
async def test_a_custom_sql_table_registers_against_a_databricks_connector():
    """Custom SQL registration is not refused by ``_registered_custom_tt``
    itself -- only ``_from_custom_tt``'s own Databricks-only restriction
    still applies, exactly as it does for a table supplied against a
    generated job. The exporter that actually reads the query
    (``DatabricksConnector.export_table_read_only``, gated on
    ``exports_custom_sql``) is the same one either path stages through.
    """
    connector_id = ConnectorID("connector_" + "d" * 32)
    reference = ConnectorExecutionRef(
        project_id=_TEST_PROJECT_ID,
        connector_id=connector_id,
    )
    connectors = SimpleNamespace(
        execution_ref_for_project=AsyncMock(return_value=reference),
        get_for_project=AsyncMock(return_value=SimpleNamespace(
            config=SimpleNamespace(
                type=data_source_api.ConnectorType.DATABRICKS))),
    )
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(),
        state_services=SimpleNamespace(
            connectors=connectors,
            projects=_project_service(),
        ),
    )

    result = await pquery._registered_custom_tt(
        _TEST_PROJECT_ID,
        ApiCustomTrainingTable(
            source_table=CustomSQLSourceTable(
                connector_id=str(connector_id),
                sql='SELECT entity, target, weight FROM orders',
                table='custom_train',
            ),
            table_mod_spec=TrainingTableSpec(weight_col='weight'),
        ),
    )

    assert result.source_table.connector_type == (
        SourceConnectorType.DATABRICKS)
    assert result.source_table.referenced_tables == ['orders']


@pytest.mark.asyncio
async def test_custom_train_sql_overwrites_client_routing_and_lineage():
    connector_id = ConnectorID("connector_" + "d" * 32)
    reference = ConnectorExecutionRef(
        project_id=_TEST_PROJECT_ID,
        connector_id=connector_id,
    )
    connectors = SimpleNamespace(
        execution_ref_for_project=AsyncMock(return_value=reference),
        get_for_project=AsyncMock(return_value=SimpleNamespace(
            config=SimpleNamespace(
                type=data_source_api.ConnectorType.DATABRICKS))),
    )
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(),
        state_services=SimpleNamespace(
            connectors=connectors,
            projects=_project_service(),
        ),
    )

    result = await pquery._from_custom_tt(
        _TEST_PROJECT_ID,
        ApiCustomTrainingTable(
            source_table=CustomSQLSourceTable(
                connector_id=str(connector_id),
                sql='SELECT entity, target, weight FROM orders',
                table='custom_train',
                connector_type=SourceConnectorType.SNOWFLAKE,
                referenced_tables=['spoofed_table'],
            ),
            table_mod_spec=TrainingTableSpec(weight_col='weight'),
            validated=True,
        ),
    )

    assert result is not None
    assert result.source_table.connector_type == (
        SourceConnectorType.DATABRICKS)
    assert result.source_table.referenced_tables == ['orders']
    connectors.get_for_project.assert_awaited_once_with(_TEST_PROJECT_ID,
                                                        connector_id, None)

    connectors.get_for_project.return_value.config.type = (
        data_source_api.ConnectorType.SNOWFLAKE)
    with pytest.raises(ValueError, match='only for Databricks'):
        await pquery._from_custom_tt(
            _TEST_PROJECT_ID,
            ApiCustomTrainingTable(
                source_table=CustomSQLSourceTable(
                    connector_id=str(connector_id),
                    sql='SELECT entity, target, weight FROM orders',
                    table='custom_train',
                ),
                table_mod_spec=TrainingTableSpec(weight_col='weight'),
                validated=True,
            ),
        )


@pytest.mark.asyncio
async def test_validate_custom_train_table_dispatches_project_execution_ref():
    connector_id = ConnectorID("connector_" + "c" * 32)
    connector_service = _ConnectorSpecReadService()
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            connectors=connector_service,
            projects=_project_service(),
        ))
    with patch(
            "rest.driver.pquery_api_driver.DataBackend._execute_single_activity",
            new=AsyncMock(return_value=ValidationResponse()),
    ) as execute_activity:
        response = await pquery.validate_custom_train_table(
            S3SourceTable(
                connector_id=str(connector_id),
                source={
                    'mode': 'relative',
                    'path': "custom_train"
                },
                file_type=FileType.PARQUET,
            ),
            weight_col="weight",
            split_col=None,
            orig_train_table_path="s3://bucket/original_train.parquet",
            extensive_validation=False,
            project_id=_TEST_PROJECT_ID,
        )

    assert response.errors == []
    assert connector_service.calls == [
        ("source", _TEST_PROJECT_ID, connector_id, None),
        (
            "execution",
            None,
            _TEST_PROJECT_ID,
            connector_id,
            CONNECTOR_GRAPH_ACTIONS,
            data_source_api.ConnectorType.S3,
        ),
    ]
    activity_inputs = execute_activity.await_args.args[2]
    assert activity_inputs[0] == ConnectorExecutionRef(
        project_id=_TEST_PROJECT_ID,
        connector_id=connector_id,
    )
    assert all(not isinstance(value, ConnectorSpec)
               for value in activity_inputs)


@pytest.mark.asyncio
async def test_create_predictive_query_v2_writes_definition_with_state_service(
):
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("11111111111111111111111111111111"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService({})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))
    pquery._graph_driver.create_full_graph_v2 = AsyncMock()
    resource = SimpleNamespace(
        name="riskpquery",
        query_string=definition.query_str,
        graph=GraphDefinition(tables={}, col_groups=[]),
    )

    with patch.object(
            PQueryLogic,
            "build_pquery_def_v2",
            new_callable=AsyncMock,
            return_value=definition,
    ) as build_definition:
        pq_id = await pquery.create_predictive_query_v2(resource,
                                                        _TEST_PROJECT_ID)

    assert pq_id == PredictiveQueryID.public(definition.fpid)
    pquery._graph_driver.create_full_graph_v2.assert_awaited_once_with(
        GraphDefinition(tables={}, col_groups=[]),
        project_id=_TEST_PROJECT_ID,
        skip_validation=True,
    )
    build_definition.assert_awaited_once()
    assert pquery_service.add_calls == [definition]
    assert pquery_service.set_alias_calls == [("riskpquery", definition.fpid)]


@pytest.mark.asyncio
async def test_validate_predictive_query_v2_checks_unique_id_with_state_service(
):
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("11111111111111111111111111111111"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))
    request = SimpleNamespace(
        pquery_resource=SimpleNamespace(
            name="riskpquery",
            query_string=definition.query_str,
            graph=GraphDefinition(tables={}, col_groups=[]),
            desc="",
            list_distinct_target_is_array=None,
            lp_bp_layout=None,
        ),
        model_plan=None,
    )

    response = await pquery.validate_predictive_query_v2(
        request, _TEST_PROJECT_ID)

    assert not response.ok
    assert "already exists" in response.error_message()
    assert pquery_service.definition_by_id_calls == [
        PredictiveQueryID("riskpquery"),
    ]


@pytest.mark.asyncio
async def test_get_predictive_query_v2_uses_state_services_without_opening_dao(
):
    graph_fpid = FPID("11111111111111111111111111111111")
    graph_def = GraphDefinition(tables={}, col_groups=[])
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    graph_def = GraphDefinition(tables={}, col_groups=[])
    graph_service = _GraphDefinitionReadService({graph_fpid: graph_def})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            graphs=graph_service,
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    result = await pquery.get_predictive_query_v2(
        PredictiveQueryID("riskpquery"), _TEST_PROJECT_ID)

    assert result.name == "riskpquery"
    assert result.query_string == definition.query_str
    assert result.graph == graph_def
    assert pquery_service.definition_by_id_calls == [
        PredictiveQueryID("riskpquery")
    ]
    assert graph_service.definition_calls == [(_TEST_PROJECT_ID, graph_fpid)]


@pytest.mark.asyncio
async def test_list_predictive_queries_v2_uses_state_services_without_opening_dao(
):
    graph_fpid = FPID("11111111111111111111111111111111")
    graph_def = GraphDefinition(tables={}, col_groups=[])
    risk_definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    churn_definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="churnpquery",
        query_str="PREDICT COUNT(c.*) FOR EACH c.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService(
        {
            risk_definition.fpid: risk_definition,
            churn_definition.fpid: churn_definition,
        },
        aliases={
            "riskpquery": risk_definition.fpid,
            "churnpquery": churn_definition.fpid,
        },
    )
    graph_service = _GraphDefinitionReadService({graph_fpid: graph_def})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            graphs=graph_service,
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    results = await pquery.list_predictive_queries_v2(_TEST_PROJECT_ID, "^risk")

    assert [result.name for result in results] == ["riskpquery"]
    assert results[0].query_string == risk_definition.query_str
    assert results[0].graph == graph_def
    assert pquery_service.list_alias_calls == 1
    assert pquery_service.list_definition_calls == 0
    assert pquery_service.definition_by_id_calls == []
    assert pquery_service.definitions_by_id_calls == [[
        PredictiveQueryID("riskpquery")
    ]]
    assert graph_service.definition_calls == []
    assert graph_service.definitions_calls == [(_TEST_PROJECT_ID, {graph_fpid})]


@pytest.mark.asyncio
async def test_get_many_predictive_queries_v2_uses_state_services_without_opening_dao(
):
    risk_definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("11111111111111111111111111111111"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    churn_definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="churnpquery",
        query_str="PREDICT COUNT(c.*) FOR EACH c.user_id",
        graph_fpid=FPID("22222222222222222222222222222222"),
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService({
        risk_definition.fpid:
        risk_definition,
        churn_definition.fpid:
        churn_definition,
    })
    pquery = _pquery_driver(ml_job_backend=SimpleNamespace(), )
    training_job_id = TrainingJobID.make_random_id()
    batch_prediction_job_id = BatchPredictionJobID.make_random_id()
    risk_id = PredictiveQueryID.public(risk_definition.fpid)
    churn_id = PredictiveQueryID.public(churn_definition.fpid)
    jobs = {
        training_job_id:
        SimpleNamespace(
            project_id=_TEST_PROJECT_ID,
            pquery_def_fpid=risk_definition.fpid,
        ),
        batch_prediction_job_id:
        SimpleNamespace(
            project_id=_TEST_PROJECT_ID,
            config=SimpleNamespace(predictive_query_id=churn_id),
        ),
    }
    ml_job_service = SimpleNamespace(
        get_job_if_exists_generic=AsyncMock(
            side_effect=lambda job_id: jobs.get(job_id)), )
    pquery._configure_state_services(
        SimpleNamespace(
            ml_jobs=ml_job_service,
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    result = await pquery.get_many_predictive_queries_v2(
        {
            training_job_id: risk_id,
            batch_prediction_job_id: churn_id,
        }, _TEST_PROJECT_ID)

    assert result == {
        training_job_id: (risk_definition.query_str, TaskType.REGRESSION),
        batch_prediction_job_id:
        (churn_definition.query_str, TaskType.TEMPORAL_LINK_PREDICTION),
    }
    assert pquery_service.definition_calls == []
    assert pquery_service.definitions_calls == [{
        risk_definition.fpid,
        churn_definition.fpid,
    }]
    assert ml_job_service.get_job_if_exists_generic.await_count == 2


@pytest.mark.asyncio
async def test_pquery_definitions_by_ids_resolves_name_only_ids_in_bulk():
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("11111111111111111111111111111111"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    result = await pquery._pquery_definitions_by_ids([
        PredictiveQueryID("riskpquery"),
        PredictiveQueryID("riskpquery"),
        PredictiveQueryID("missing"),
    ])

    assert result == {
        PredictiveQueryID("riskpquery"): definition,
        PredictiveQueryID("missing"): None,
    }
    assert pquery_service.definition_by_id_calls == []
    assert pquery_service.definitions_by_id_calls == [[
        PredictiveQueryID("riskpquery"),
        PredictiveQueryID("missing"),
    ]]


@pytest.mark.asyncio
async def test_resolve_pquery_for_launch_uses_state_service_when_configured():
    fpid = FPID("0123456789abcdef0123456789abcdef")
    graph_fpid = FPID("11111111111111111111111111111111")
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService(
        {fpid: definition},
        aliases={"riskpquery": fpid},
    )
    graph_def = GraphDefinition(tables={}, col_groups=[])
    graph_service = _GraphDefinitionReadService({graph_fpid: graph_def})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            graphs=graph_service,
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    resolved_fpid, resource = await pquery._resolve_pquery_for_launch(
        "riskpquery", _TEST_PROJECT_ID)

    assert resolved_fpid == fpid
    assert resource.name == "riskpquery"
    assert resource.query_string == definition.query_str
    assert resource.graph == graph_def
    assert pquery_service.alias_calls == ["riskpquery"]
    assert pquery_service.definition_calls == [fpid]
    assert graph_service.definition_calls == [(_TEST_PROJECT_ID, graph_fpid)]


@pytest.mark.asyncio
async def test_parsed_pq_by_fpid_uses_state_service_when_configured():
    fpid = FPID("0123456789abcdef0123456789abcdef")
    graph_fpid = FPID("11111111111111111111111111111111")
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService({fpid: definition})
    graph_def = GraphDefinition(tables={}, col_groups=[])
    graph_service = _GraphDefinitionReadService({graph_fpid: graph_def})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            graphs=graph_service,
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))
    parsed = SimpleNamespace(task_type=TaskType.REGRESSION)

    with patch("rest.driver.pquery_api_driver.PQueryLogic") as pquery_logic_cls:
        pquery_logic_cls.return_value.parse_predictive_query_v2 = AsyncMock(
            return_value=(parsed, ValidationResponse()))

        assert await pquery.get_parsed_pq_v2_by_fpid(
            fpid, _TEST_PROJECT_ID, skip_validation=True) is parsed

    assert pquery_service.definition_calls == [fpid]
    assert graph_service.definition_calls == [(_TEST_PROJECT_ID, graph_fpid)]
    pquery_logic_cls.return_value.parse_predictive_query_v2.assert_awaited_once(
    )
    assert (pquery_logic_cls.return_value.parse_predictive_query_v2.await_args.
            kwargs["skip_validation"] is True)


@pytest.mark.asyncio
async def test_get_pquery_metadata_uses_stored_state_service_metadata_without_opening_dao(
):
    graph_fpid = FPID("11111111111111111111111111111111")
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    graph_def = GraphDefinition(tables={}, col_groups=[])
    graph_service = _GraphDefinitionReadService({graph_fpid: graph_def})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            graphs=graph_service,
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    assert await pquery.get_pquery_metadata(PredictiveQueryID("riskpquery"),
                                            _TEST_PROJECT_ID) == (
                                                TaskType.REGRESSION,
                                                QueryType.TEMPORAL,
                                            )

    assert pquery_service.definition_by_id_calls == [
        PredictiveQueryID("riskpquery")
    ]
    assert pquery_service.alias_calls == []
    assert pquery_service.definition_calls == []
    assert graph_service.definition_calls == [(_TEST_PROJECT_ID, graph_fpid)]


@pytest.mark.asyncio
async def test_snapshot_override_uses_snapshot_info(monkeypatch):
    graph_fpid = FPID("a" * 32)
    graph_id = GraphIDv2.compose(fpid=graph_fpid)
    version_info = SimpleNamespace(graph_data_version="version-one")
    pquery = _pquery_driver(ml_job_backend=None, )
    get_snapshot_info = AsyncMock(return_value=(graph_id, version_info))
    state_services = SimpleNamespace(projects=_project_service())
    pquery._configure_state_services(state_services)
    monkeypatch.setattr(
        "rest.driver.pquery_api_driver.get_graph_snapshot_info",
        get_snapshot_info,
    )
    pq_resource = SimpleNamespace(
        graph=SimpleNamespace(fpid=graph_fpid),
        name="pquery-one",
    )

    result = await pquery._handle_snapshot_override_v2(
        pq_resource,
        "snapshot-one",
    )

    assert result == (graph_fpid, version_info)
    get_snapshot_info.assert_awaited_once_with("snapshot-one")


@pytest.mark.asyncio
async def test_snapshot_override_falls_back_to_pquery_graph_when_omitted(
        monkeypatch):
    # Regression for #30136: an omitted graph_snapshot_id must resolve to the
    # pquery's graph without a snapshot lookup, so BP honors the optional field.
    graph_fpid = FPID("b" * 32)
    pquery = _pquery_driver(ml_job_backend=None)
    pquery._configure_state_services(
        SimpleNamespace(projects=_project_service()))
    get_snapshot_info = AsyncMock()
    monkeypatch.setattr(
        "rest.driver.pquery_api_driver.get_graph_snapshot_info",
        get_snapshot_info,
    )
    pq_resource = SimpleNamespace(
        graph=SimpleNamespace(fpid=graph_fpid),
        name="pquery-one",
    )

    result = await pquery._handle_snapshot_override_v2(pq_resource, None)

    assert result == (graph_fpid, None)
    get_snapshot_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_fork_pquery_def_for_graph_uses_state_service_when_configured():
    fpid = FPID("0123456789abcdef0123456789abcdef")
    new_graph_fpid = FPID("11111111111111111111111111111111")
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("22222222222222222222222222222222"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService({fpid: definition})
    graph_def = GraphDefinition(tables={}, col_groups=[])
    graph_service = _GraphDefinitionReadService({new_graph_fpid: graph_def})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            graphs=graph_service,
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    forked_fpid, resource = await pquery._fork_pquery_def_for_graph(
        fpid,
        SimpleNamespace(
            name=definition.pquery_name,
            query_string=definition.query_str,
            desc="",
            runtime_options=None,
        ),
        new_graph_fpid,
        _TEST_PROJECT_ID,
    )

    assert pquery_service.definition_calls == [fpid]
    assert len(pquery_service.add_calls) == 1
    assert pquery_service.add_calls[0].graph_fpid == new_graph_fpid
    assert forked_fpid == pquery_service.add_calls[0].fpid
    assert resource.name == definition.pquery_name
    assert resource.query_string == definition.query_str
    assert resource.graph == graph_def
    assert graph_service.definition_calls == [(_TEST_PROJECT_ID, new_graph_fpid)
                                              ]


@pytest.mark.asyncio
async def test_get_pq_and_graph_v2_uses_state_services_without_opening_dao():
    graph_def = GraphDefinition(tables={}, col_groups=[])
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_def.fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    graph_service = _GraphDefinitionReadService(
        {definition.graph_fpid: graph_def})
    graph_compute_input = SimpleNamespace(graph_fpid=definition.graph_fpid)
    graph_service.get_graph_data_compute_input = AsyncMock(
        return_value=graph_compute_input)
    state_services = SimpleNamespace(
        graphs=graph_service,
        ml_jobs=SimpleNamespace(),
        pqueries=pquery_service,
        projects=_project_service(),
    )
    pquery = _pquery_driver(ml_job_backend=SimpleNamespace(), )
    pquery._configure_state_services(state_services)

    pq_id, pq_resource, fpid, graph_version_info, data_compute_input = (
        await pquery._get_pq_and_graph_v2("riskpquery", _TEST_PROJECT_ID, None))

    assert pq_id == PredictiveQueryID.compose(fpid=definition.fpid)
    assert pq_resource.name == "riskpquery"
    assert pq_resource.graph == graph_def
    assert fpid == definition.fpid
    assert graph_version_info is None
    assert data_compute_input == graph_compute_input
    assert pquery_service.alias_calls == ["riskpquery"]
    assert pquery_service.definition_calls == [definition.fpid]
    assert graph_service.definition_calls == [(_TEST_PROJECT_ID,
                                               definition.graph_fpid)]
    graph_service.get_graph_data_compute_input.assert_awaited_once_with(
        GraphIDv2.compose(fpid=definition.graph_fpid),
        project_id=_TEST_PROJECT_ID,
        user_authorization=None,
    )


@pytest.mark.asyncio
async def test_current_training_job_uses_state_services_when_configured():
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("11111111111111111111111111111111"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    fpid = definition.fpid
    older_training_job_id = TrainingJobID.make_random_id()
    newer_training_job_id = TrainingJobID.make_random_id()
    foreign_training_job_id = TrainingJobID.make_random_id()
    pquery_service = _PQueryDefinitionReadService(
        {fpid: definition},
        aliases={"riskpquery": fpid},
    )
    older_time = now_plus_seconds_utc(1)
    newer_time = now_plus_seconds_utc(2)
    older_timestamps = ResourceTimestamps(created_at=older_time,
                                          updated_at=older_time)
    newer_timestamps = ResourceTimestamps(created_at=newer_time,
                                          updated_at=newer_time)
    tag_service = _MLJobTagReadService(
        {
            (JobType.TRAINING_JOB, older_training_job_id):
            JobTags(
                timestamps=get_current_timestamps(),
                tags={TAG_KEY_PQUERY_ID: JobTagValue.of_pquery_id(fpid)},
            ),
            (JobType.TRAINING_JOB, newer_training_job_id):
            JobTags(
                timestamps=get_current_timestamps(),
                tags={TAG_KEY_PQUERY_ID: JobTagValue.of_pquery_id(fpid)},
            ),
            (JobType.TRAINING_JOB, foreign_training_job_id):
            JobTags(
                timestamps=get_current_timestamps(),
                tags={TAG_KEY_PQUERY_ID: JobTagValue.of_pquery_id(fpid)},
            ),
        }, training_rows={
            older_training_job_id:
            SimpleNamespace(
                id=older_training_job_id,
                project_id=_TEST_PROJECT_ID,
                timestamps=older_timestamps,
            ),
            newer_training_job_id:
            SimpleNamespace(
                id=newer_training_job_id,
                project_id=_TEST_PROJECT_ID,
                timestamps=newer_timestamps,
            ),
            foreign_training_job_id:
            SimpleNamespace(
                id=foreign_training_job_id,
                project_id=_OTHER_PROJECT_ID,
                timestamps=ResourceTimestamps(
                    created_at=now_plus_seconds_utc(3),
                    updated_at=now_plus_seconds_utc(3),
                ),
            ),
        })
    pquery = _pquery_driver(ml_job_backend=SimpleNamespace(), )
    project_service = _project_service()
    pquery._configure_state_services(
        SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=pquery_service,
            projects=project_service,
        ))

    assert await pquery.get_current_training_job_id(
        _request_ctx(_TEST_PROJECT_ID, _OTHER_PROJECT_ID),
        PredictiveQueryID("riskpquery"),
        _TEST_PROJECT_ID,
    ) == newer_training_job_id
    project_service.get_project.assert_awaited_once()
    assert pquery_service.definition_by_id_calls == [
        PredictiveQueryID("riskpquery")
    ]
    assert tag_service.list_calls == [(
        JobType.TRAINING_JOB,
        {
            str(TAG_KEY_PQUERY_ID): str(JobTagValue.of_pquery_id(fpid))
        },
    )]


@pytest.mark.asyncio
async def test_current_training_job_rejects_hidden_project_before_pquery_lookup(
):
    project_service = SimpleNamespace(get_project=AsyncMock(
        side_effect=ProjectNotFoundError("Project not found")))
    pquery_service = _PQueryDefinitionReadService({})
    tag_service = _MLJobTagReadService({})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=pquery_service,
            projects=project_service,
        ))

    with pytest.raises(ProjectNotFoundError, match="not found"):
        await pquery.get_current_training_job_id(
            _request_ctx(_TEST_PROJECT_ID),
            PredictiveQueryID("riskpquery"),
            _OTHER_PROJECT_ID,
        )

    assert pquery_service.definition_by_id_calls == []
    assert tag_service.list_calls == []


@pytest.mark.asyncio
async def test_cancel_batch_prediction_job_validates_pquery_with_state_service(
):
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("11111111111111111111111111111111"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_id = PredictiveQueryID("riskpquery")
    bp_job_id = BatchPredictionJobID("batch-prediction-job-123")
    training_job_id = TrainingJobID("training-job-123")
    bp_row = _batch_prediction_row(
        job_id=bp_job_id,
        pquery_fpid=definition.fpid,
        training_job_id=training_job_id,
    )
    bp_row.config.predictive_query_id = pquery_id
    tag_service = _MLJobTagReadService(
        {},
        batch_prediction_rows={bp_job_id: bp_row},
    )
    ml_job_backend = SimpleNamespace(cancel_batch_prediction_job=AsyncMock(
        return_value=True))
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    assert await pquery.cancel_batch_prediction_job(
        _request_ctx(),
        pquery_id,
        bp_job_id,
    )
    assert pquery_service.definition_by_id_calls == [pquery_id]
    assert tag_service.batch_prediction_row_calls == [{bp_job_id}]
    ml_job_backend.cancel_batch_prediction_job.assert_awaited_once_with(
        bp_job_id)


@pytest.mark.asyncio
async def test_current_batch_prediction_job_uses_state_services_without_opening_dao(
):
    pquery_id = PredictiveQueryID("riskpquery")
    pquery_fpid = FPID("11111111111111111111111111111111")
    training_job_id = TrainingJobID("training-job-123")
    older_bp_job_id = BatchPredictionJobID("batch-prediction-job-123")
    newer_bp_job_id = BatchPredictionJobID("batch-prediction-job-456")
    older_row = _batch_prediction_row(
        job_id=older_bp_job_id,
        pquery_fpid=pquery_fpid,
        training_job_id=training_job_id,
    )
    newer_row = _batch_prediction_row(
        job_id=newer_bp_job_id,
        pquery_fpid=pquery_fpid,
        training_job_id=training_job_id,
    )
    newer = now_plus_seconds_utc(1)
    newer_row.timestamps = ResourceTimestamps(
        created_at=newer,
        updated_at=newer,
    )
    older_row.config.predictive_query_id = pquery_id
    newer_row.config.predictive_query_id = pquery_id
    tag_service = _MLJobTagReadService(
        {
            (JobType.BATCH_PREDICTION_JOB, older_bp_job_id):
            JobTags(
                timestamps=older_row.timestamps,
                tags={
                    TAG_KEY_PQUERY_NAME: JobTagValue.of_pquery_name(pquery_id)
                },
            ),
            (JobType.BATCH_PREDICTION_JOB, newer_bp_job_id):
            JobTags(
                timestamps=newer_row.timestamps,
                tags={
                    TAG_KEY_PQUERY_NAME: JobTagValue.of_pquery_name(pquery_id)
                },
            ),
        },
        batch_prediction_rows={
            older_bp_job_id: older_row,
            newer_bp_job_id: newer_row,
        },
    )
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=_PQueryDefinitionReadService({}),
        ))

    assert await pquery.get_current_batch_prediction_job_id(
        _request_ctx(), pquery_id) == newer_bp_job_id
    assert tag_service.current_batch_prediction_calls == [[pquery_id]]
    assert tag_service.list_calls == []
    assert tag_service.batch_prediction_row_calls == []


@pytest.mark.asyncio
async def test_delete_predictive_query_uses_state_services_when_configured():
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("11111111111111111111111111111111"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_id = PredictiveQueryID("riskpquery")
    training_job_id = TrainingJobID("training-job-123")
    done_training_job_id = TrainingJobID("training-job-456")
    ctx = _request_ctx()
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    graph_service = _GraphDefinitionDeleteService()
    tag_service = _MLJobTagReadService(
        {
            (JobType.TRAINING_JOB, training_job_id):
            JobTags(
                timestamps=get_current_timestamps(),
                tags={
                    TAG_KEY_PQUERY_ID: JobTagValue.of_pquery_id(
                        definition.fpid),
                },
            ),
            (JobType.TRAINING_JOB, done_training_job_id):
            JobTags(
                timestamps=get_current_timestamps(),
                tags={
                    TAG_KEY_PQUERY_ID: JobTagValue.of_pquery_id(
                        definition.fpid),
                },
            ),
        }, training_rows={
            training_job_id:
            SimpleNamespace(
                id=training_job_id,
                project_id=_TEST_PROJECT_ID,
                timestamps=ResourceTimestamps(
                    created_at=now_plus_seconds_utc(1),
                    updated_at=now_plus_seconds_utc(1),
                ),
            ),
            done_training_job_id:
            SimpleNamespace(
                id=done_training_job_id,
                project_id=_TEST_PROJECT_ID,
                timestamps=ResourceTimestamps(
                    created_at=now_plus_seconds_utc(2),
                    updated_at=now_plus_seconds_utc(2),
                ),
            ),
        })
    ml_job_backend = SimpleNamespace(
        get_training_job=AsyncMock(
            side_effect=AssertionError("scalar training job lookup used")),
        get_training_jobs=AsyncMock(
            return_value={
                training_job_id:
                SimpleNamespace(execution=SimpleNamespace(
                    status=JobStatus.RUNNING)),
                done_training_job_id:
                SimpleNamespace(execution=SimpleNamespace(
                    status=JobStatus.DONE)),
            }),
        cancel_training_job=AsyncMock(return_value=True),
        get_batch_prediction_jobs=AsyncMock(return_value={}),
    )
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=pquery_service,
            graphs=graph_service,
            projects=_project_service(),
        ))

    await pquery.delete_predictive_query(ctx, pquery_id, _TEST_PROJECT_ID)

    assert pquery_service.definition_by_id_calls == [pquery_id]
    assert pquery_service.delete_alias_calls == ["riskpquery"]
    assert pquery_service.delete_definition_calls == [definition.fpid]
    assert definition.fpid not in pquery_service.definitions
    assert graph_service.get_entry_calls == [(_TEST_PROJECT_ID,
                                              definition.graph_fpid)]
    assert graph_service.decrement_or_delete_calls == [(_TEST_PROJECT_ID,
                                                        definition.graph_fpid)]
    ml_job_backend.get_training_jobs.assert_awaited_once_with(
        ctx, {training_job_id, done_training_job_id})
    ml_job_backend.get_training_job.assert_not_awaited()
    ml_job_backend.cancel_training_job.assert_awaited_once_with(training_job_id)
    assert tag_service.list_calls == [
        (
            JobType.TRAINING_JOB,
            {
                str(TAG_KEY_PQUERY_ID):
                str(JobTagValue.of_pquery_id(definition.fpid))
            },
        ),
    ]
    assert tag_service.batch_prediction_exists_calls == [{
        "project_ids": {_TEST_PROJECT_ID},
        "model_id":
        None,
        "pquery_id":
        pquery_id,
        "job_ids":
        None,
        "status":
        None,
        "match_pquery_def_fpid":
        False,
    }]
    assert tag_service.batch_prediction_row_calls == []


@pytest.mark.asyncio
async def test_delete_predictive_query_checks_batch_predictions_by_pquery():
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=FPID("11111111111111111111111111111111"),
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    pquery_id = PredictiveQueryID("riskpquery")
    bp_job_id = BatchPredictionJobID("batch-prediction-job-123")
    bp_row = _batch_prediction_row(
        job_id=bp_job_id,
        pquery_fpid=definition.fpid,
        training_job_id=TrainingJobID("training-job-123"),
    )
    bp_row.config.predictive_query_id = pquery_id
    tag_service = _MLJobTagReadService(
        {},
        batch_prediction_rows={bp_job_id: bp_row},
    )
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=pquery_service,
            graphs=_GraphDefinitionDeleteService(),
            projects=_project_service(),
        ))
    pquery.list_batch_prediction_jobs = AsyncMock(
        side_effect=AssertionError("broad batch prediction listing used"))

    with pytest.raises(ValidationHTTPException) as exc_info:
        await pquery.delete_predictive_query(
            _request_ctx(_TEST_PROJECT_ID, _OTHER_PROJECT_ID),
            pquery_id,
            _TEST_PROJECT_ID,
        )

    assert exc_info.value.detail == BATCH_PREDICTION_JOBS_DELETE_MESSAGE
    assert tag_service.batch_prediction_exists_calls == [{
        "project_ids": {_TEST_PROJECT_ID},
        "model_id":
        None,
        "pquery_id":
        pquery_id,
        "job_ids":
        None,
        "status":
        None,
        "match_pquery_def_fpid":
        False,
    }]
    assert tag_service.batch_prediction_row_calls == []
    pquery.list_batch_prediction_jobs.assert_not_awaited()
    assert pquery_service.delete_alias_calls == []
    assert pquery_service.delete_definition_calls == []


@pytest.mark.asyncio
async def test_pquery_job_tag_writes_use_ml_backend_when_state_service_configured(
):
    training_job_id = TrainingJobID.make_random_id()
    project_id = ProjectID.generate()
    existing_tags = JobTags(
        timestamps=get_current_timestamps(),
        tags={TAG_KEY_PQUERY_NAME: JobTagValue("riskpquery")},
    )
    tag_service = _MLJobTagReadService({
        (JobType.TRAINING_JOB, training_job_id):
        existing_tags,
    })

    async def update_job_tags(job_type, job_id, tags_arg, *, project_id):
        if job_id != training_job_id:
            raise KeyError(f"{job_type} id={job_id} is not found in DB")
        return True

    ml_job_backend = SimpleNamespace(
        update_job_tags=AsyncMock(side_effect=update_job_tags), )
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=_PQueryDefinitionReadService({}),
        ))
    tags = {JobTagKey("phase"): JobTagValue("one")}

    assert await pquery.update_job_tags(
        JobType.TRAINING_JOB,
        training_job_id,
        tags,
        project_id=project_id,
    )
    with pytest.raises(KeyError):
        await pquery.update_job_tags(
            JobType.TRAINING_JOB,
            TrainingJobID.make_random_id(),
            tags,
            project_id=project_id,
        )

    ml_job_backend.update_job_tags.assert_any_await(
        JobType.TRAINING_JOB,
        training_job_id,
        tags,
        project_id=project_id,
    )


@pytest.mark.asyncio
async def test_training_job_tag_update_uses_state_service_without_opening_dao():
    training_job_id = TrainingJobID.make_random_id()
    pquery_fpid = FPID("0123456789abcdef0123456789abcdef")
    tag_service = _MLJobTagReadService({})
    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(), state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=_PQueryDefinitionReadService({}),
        ))
    request = SimpleNamespace(
        job_tags={"phase": "one"},
        warm_start_job_id=None,
        train_table_job_id=None,
    )

    await pquery._update_train_job_tags(
        request,
        training_job_id,
        "riskpquery",
        PredictiveQueryID.compose(fpid=pquery_fpid),
    )

    train_table_job_id = GenerateTrainTableJobID.as_child_of(training_job_id)
    assert [(job_type, job_id)
            for job_type, job_id, _ in tag_service.apply_calls] == [
                (JobType.TRAINING_JOB, training_job_id),
                (JobType.GENERATE_TRAIN_TABLE_JOB, train_table_job_id),
            ]
    training_tags = tag_service.tags_by_job[(JobType.TRAINING_JOB,
                                             training_job_id)].tags
    child_tags = tag_service.tags_by_job[(JobType.GENERATE_TRAIN_TABLE_JOB,
                                          train_table_job_id)].tags
    assert training_tags[TAG_KEY_PQUERY_ID] == JobTagValue.of_pquery_id(
        pquery_fpid)
    assert training_tags[TAG_KEY_PQUERY_NAME] == JobTagValue.of_pquery_name(
        "riskpquery")
    assert child_tags[TAG_KEY_PARENT_JOB] == JobTagValue(training_job_id)


@pytest.mark.asyncio
async def test_pquery_job_tag_delete_uses_state_service_when_configured():
    training_job_id = TrainingJobID.make_random_id()
    project_id = ProjectID.generate()
    custom_tag = JobTagKey("phase")
    existing_tags = JobTags(
        timestamps=get_current_timestamps(),
        tags={custom_tag: JobTagValue("one")},
    )
    tag_service = _MLJobTagReadService({
        (JobType.TRAINING_JOB, training_job_id):
        existing_tags,
    })

    async def delete_job_tag(job_type, job_id, tag, *, project_id):
        return await tag_service.delete_job_tag(job_type, job_id, tag)

    pquery = _pquery_driver(
        ml_job_backend=SimpleNamespace(
            delete_job_tag=AsyncMock(side_effect=delete_job_tag), ),
        state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=_PQueryDefinitionReadService({}),
        ))

    assert await pquery.remove_job_tag(
        JobType.TRAINING_JOB,
        training_job_id,
        custom_tag,
        project_id=project_id,
    )
    assert not await pquery.remove_job_tag(
        JobType.TRAINING_JOB,
        training_job_id,
        custom_tag,
        project_id=project_id,
    )
    assert tag_service.delete_calls == [
        (JobType.TRAINING_JOB, training_job_id, custom_tag),
        (JobType.TRAINING_JOB, training_job_id, custom_tag),
    ]


@pytest.mark.asyncio
async def test_batch_pred_materialization_info_uses_state_service_when_configured(
):
    bp_job_id = BatchPredictionJobID.make_random_id()
    info = _MaterializationInfo()
    ml_job_backend = SimpleNamespace(
        _has_batch_prediction_job=AsyncMock(return_value=True),
        get_materialization_info=AsyncMock(return_value=info),
    )
    pquery = _pquery_driver(ml_job_backend=ml_job_backend, )
    pquery._configure_state_services(SimpleNamespace(ml_jobs=SimpleNamespace()))

    result = await pquery.get_batch_pred_materialization_info(
        ctx=None,
        id=bp_job_id,
    )

    assert result is info
    assert info.cleared is True
    ml_job_backend._has_batch_prediction_job.assert_awaited_once_with(
        None, bp_job_id)
    ml_job_backend.get_materialization_info.assert_awaited_once_with(bp_job_id)


@pytest.mark.asyncio
async def test_batch_pred_materialization_info_raises_not_found_without_compat_dao(
):
    bp_job_id = BatchPredictionJobID.make_random_id()
    ml_job_backend = SimpleNamespace(
        _has_batch_prediction_job=AsyncMock(return_value=False),
        get_materialization_info=AsyncMock(),
    )
    pquery = _pquery_driver(ml_job_backend=ml_job_backend, )
    pquery._configure_state_services(SimpleNamespace(ml_jobs=SimpleNamespace()))

    with pytest.raises(BatchPredictionDoesNotExistError):
        await pquery.get_batch_pred_materialization_info(
            ctx=None,
            id=bp_job_id,
        )
    ml_job_backend._has_batch_prediction_job.assert_awaited_once_with(
        None, bp_job_id)
    ml_job_backend.get_materialization_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_prediction_readbacks_use_state_service_when_configured():
    bp_job_id = BatchPredictionJobID.make_random_id()
    training_job_id = TrainingJobID.make_random_id()
    pquery_fpid = FPID("0123456789abcdef0123456789abcdef")
    row = _batch_prediction_row(
        job_id=bp_job_id,
        pquery_fpid=pquery_fpid,
        training_job_id=training_job_id,
    )
    job_tags = JobTags(
        timestamps=get_current_timestamps(),
        tags={
            TAG_KEY_PQUERY_ID: JobTagValue.of_pquery_id(pquery_fpid),
        },
    )
    tag_service = _MLJobTagReadService(
        {(JobType.BATCH_PREDICTION_JOB, bp_job_id): job_tags},
        batch_prediction_rows={bp_job_id: row},
    )
    ml_job_backend = SimpleNamespace(
        get_batch_prediction_jobs=AsyncMock(
            return_value={bp_job_id: row.execution}), )
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=_PQueryDefinitionReadService({}),
            projects=_project_service(),
        ))
    ctx = _request_ctx()

    direct = await pquery.get_batch_prediction_job(ctx, bp_job_id)
    listed = await pquery.list_batch_prediction_jobs(
        ctx,
        pquery_id=PredictiveQueryID.public(pquery_fpid),
    )

    assert direct is not None
    assert direct.id == bp_job_id
    assert direct.description == "batch prediction scoring"
    assert direct.batch_prediction_workflow_id == "workflow_1"
    assert direct.tags == job_tags.tags
    assert direct.predictive_query_link.id == PredictiveQueryID.compose(
        fpid=pquery_fpid)
    assert [job.id for job in listed] == [bp_job_id]
    assert listed[0].tags == job_tags.tags
    assert tag_service.calls == [
        (JobType.BATCH_PREDICTION_JOB, bp_job_id),
    ]
    assert tag_service.list_calls == [(JobType.BATCH_PREDICTION_JOB,
                                       {bp_job_id})]
    assert tag_service.batch_prediction_row_calls == [
        {bp_job_id},
        {bp_job_id},
    ]
    ml_job_backend.get_batch_prediction_jobs.assert_any_await(
        ctx, {bp_job_id}, None, None)
    ml_job_backend.get_batch_prediction_jobs.assert_any_await(
        ctx, {bp_job_id}, None, 500)


@pytest.mark.asyncio
@pytest.mark.parametrize(("is_admin", "project_ids", "expected_scope"), [
    (False, (_TEST_PROJECT_ID, ), {_TEST_PROJECT_ID}),
    (True, (_TEST_PROJECT_ID, ), None),
    (False, (), set()),
])
async def test_batch_prediction_list_propagates_explicit_request_scope(
        is_admin, project_ids, expected_scope):
    service = _MLJobTagReadService({})
    backend = SimpleNamespace(get_batch_prediction_jobs=AsyncMock(
        return_value={}))
    driver = _pquery_driver(ml_job_backend=backend, )
    driver._configure_state_services(
        SimpleNamespace(
            ml_jobs=service,
            pqueries=_PQueryDefinitionReadService({}),
        ))
    ctx = SimpleNamespace(user_authorization=SimpleNamespace(
        is_admin=is_admin,
        project_ids=tuple(str(id) for id in project_ids),
    ))

    assert await driver.list_batch_prediction_jobs(ctx) == []
    assert service.batch_prediction_project_scope_calls == [("list",
                                                             expected_scope)]


@pytest.mark.asyncio
async def test_batch_prediction_readback_bulk_fetches_pquery_definitions():
    risk_bp_job_id = BatchPredictionJobID.make_random_id()
    churn_bp_job_id = BatchPredictionJobID.make_random_id()
    training_job_id = TrainingJobID.make_random_id()
    graph_fpid = FPID("11111111111111111111111111111111")
    risk_definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    churn_definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="churnpquery",
        query_str="PREDICT COUNT(c.*) FOR EACH c.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    risk_row = _batch_prediction_row(
        job_id=risk_bp_job_id,
        pquery_fpid=risk_definition.fpid,
        training_job_id=training_job_id,
    )
    risk_row.config.predictive_query_id = PredictiveQueryID.compose(
        "riskpquery", risk_definition.fpid)
    risk_row.config.pquery_def_fpid = None
    churn_row = _batch_prediction_row(
        job_id=churn_bp_job_id,
        pquery_fpid=churn_definition.fpid,
        training_job_id=training_job_id,
    )
    churn_row.config.predictive_query_id = PredictiveQueryID.compose(
        "churnpquery", churn_definition.fpid)
    churn_row.config.pquery_def_fpid = None
    pquery_service = _PQueryDefinitionReadService({
        risk_definition.fpid:
        risk_definition,
        churn_definition.fpid:
        churn_definition,
    })
    pquery_service.get_definition_by_id = AsyncMock(
        side_effect=AssertionError("scalar pquery lookup used"))
    ml_job_backend = SimpleNamespace(get_batch_prediction_jobs=AsyncMock(
        return_value={
            risk_bp_job_id: risk_row.execution,
            churn_bp_job_id: churn_row.execution,
        }))
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=SimpleNamespace(),
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    responses = await pquery._batch_prediction_responses_from_state_rows(
        None,
        [risk_row, churn_row],
    )

    assert [response.predictive_query_link.name
            for response in responses] == ["riskpquery", "churnpquery"]
    assert pquery_service.definitions_calls == []
    assert pquery_service.definitions_by_id_calls == [[
        PredictiveQueryID.compose("riskpquery", risk_definition.fpid),
        PredictiveQueryID.compose("churnpquery", churn_definition.fpid),
    ]]
    pquery_service.get_definition_by_id.assert_not_awaited()


@pytest.mark.asyncio
async def test_paginated_batch_prediction_readback_uses_state_service_without_dao(
):
    older_bp_job_id = BatchPredictionJobID.make_random_id()
    newer_bp_job_id = BatchPredictionJobID.make_random_id()
    training_job_id = TrainingJobID.make_random_id()
    pquery_fpid = FPID("0123456789abcdef0123456789abcdef")
    older_row = _batch_prediction_row(
        job_id=older_bp_job_id,
        pquery_fpid=pquery_fpid,
        training_job_id=training_job_id,
    )
    newer_row = _batch_prediction_row(
        job_id=newer_bp_job_id,
        pquery_fpid=pquery_fpid,
        training_job_id=training_job_id,
    )
    older_row.timestamps = ResourceTimestamps(
        created_at=now_plus_seconds_utc(1),
        updated_at=now_plus_seconds_utc(1),
    )
    newer_row.timestamps = ResourceTimestamps(
        created_at=now_plus_seconds_utc(2),
        updated_at=now_plus_seconds_utc(2),
    )
    tag_service = _MLJobTagReadService(
        {},
        batch_prediction_rows={
            older_bp_job_id: older_row,
            newer_bp_job_id: newer_row,
        },
    )
    ml_job_backend = SimpleNamespace(
        get_batch_prediction_jobs=AsyncMock(
            return_value={newer_bp_job_id: newer_row.execution}), )
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=_PQueryDefinitionReadService({}),
        ))

    ctx = _request_ctx()
    response = await pquery.list_paginated_batch_prediction_jobs(
        ctx,
        PaginationArgs(page=1, page_size=1),
    )

    assert [job.id for job in response.jobs] == [newer_bp_job_id]
    assert response.current_page == 1
    assert response.number_of_pages == 3
    assert response.total_number_of_jobs == 2
    assert tag_service.batch_prediction_row_calls == [{newer_bp_job_id}]
    ml_job_backend.get_batch_prediction_jobs.assert_awaited_once_with(
        ctx, {newer_bp_job_id}, None, 1)


@pytest.mark.asyncio
async def test_delete_batch_prediction_uses_state_services_without_opening_dao(
):
    bp_job_id = BatchPredictionJobID.make_random_id()
    training_job_id = TrainingJobID.make_random_id()
    row = _batch_prediction_row(
        job_id=bp_job_id,
        pquery_fpid=FPID("0123456789abcdef0123456789abcdef"),
        training_job_id=training_job_id,
    )
    tag_service = _MLJobTagReadService(
        {},
        batch_prediction_rows={bp_job_id: row},
    )
    tag_service.delete_batch_prediction_metadata = AsyncMock(return_value=True)
    tag_service.delete_job_tags = AsyncMock(return_value=True)
    ml_job_backend = SimpleNamespace(
        delete_batch_prediction=AsyncMock(return_value=True), )
    pquery = _pquery_driver(ml_job_backend=ml_job_backend, )
    pquery._configure_state_services(SimpleNamespace(ml_jobs=tag_service))
    ctx = _request_ctx()

    assert await pquery.delete_batch_prediction(ctx, bp_job_id)

    assert tag_service.batch_prediction_row_calls == []
    tag_service.delete_batch_prediction_metadata.assert_awaited_once_with(
        bp_job_id)
    tag_service.delete_job_tags.assert_awaited_once_with(
        JobType.BATCH_PREDICTION_JOB, bp_job_id)
    ml_job_backend.delete_batch_prediction.assert_awaited_once_with(
        ctx, bp_job_id)


@pytest.mark.asyncio
async def test_delete_training_job_uses_state_service_without_opening_dao():
    training_job_id = TrainingJobID.make_random_id()
    tag_service = _MLJobTagReadService({
        (JobType.TRAINING_JOB, training_job_id):
        JobTags(timestamps=get_current_timestamps(), tags={}),
    })
    ml_job_backend = SimpleNamespace(
        delete_training_job=AsyncMock(return_value=True), )
    pquery = _pquery_driver(ml_job_backend=ml_job_backend, )
    pquery._configure_state_services(SimpleNamespace(ml_jobs=tag_service))
    ctx = SimpleNamespace()

    await pquery.delete_training_job(ctx, PredictiveQueryID("riskpquery"),
                                     training_job_id)

    assert tag_service.delete_calls == [
        (JobType.TRAINING_JOB, training_job_id, None),
    ]
    ml_job_backend.delete_training_job.assert_awaited_once_with(
        ctx, training_job_id)


@pytest.mark.asyncio
async def test_batch_prediction_readback_resolves_named_pquery_link_with_state_service(
):
    bp_job_id = BatchPredictionJobID.make_random_id()
    training_job_id = TrainingJobID.make_random_id()
    pquery_id = PredictiveQueryID("riskpquery")
    graph_fpid = FPID("11111111111111111111111111111111")
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    now = get_current_timestamps()
    row = BatchPredictionJobDBEntry(
        id=bp_job_id,
        timestamps=now,
        config=BatchPredictionJobConfig(
            predictive_query_id=pquery_id,
            training_job_id=training_job_id,
            task_type=TaskType.REGRESSION,
            graph_id=GraphIDv2.public(graph_fpid),
        ),
        execution=BatchPredictionJobOutput(
            status=JobStatus.RUNNING,
            log=JobEventLogEntry.starting(),
        ),
        description="named batch prediction",
        batch_prediction_workflow_id="workflow_1",
        source=JobSource.UI,
        user="user@example.com",
        project_id=_TEST_PROJECT_ID,
        kumo_version="dev",
    )
    job_tags = JobTags(
        timestamps=get_current_timestamps(),
        tags={TAG_KEY_PQUERY_NAME: JobTagValue.of_pquery_name(pquery_id)},
    )
    tag_service = _MLJobTagReadService(
        {(JobType.BATCH_PREDICTION_JOB, bp_job_id): job_tags},
        batch_prediction_rows={bp_job_id: row},
    )
    pquery_service = _PQueryDefinitionReadService(
        {definition.fpid: definition},
        aliases={"riskpquery": definition.fpid},
    )
    ml_job_backend = SimpleNamespace(
        get_batch_prediction_jobs=AsyncMock(
            return_value={bp_job_id: row.execution}), )
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=tag_service,
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    result = await pquery.get_batch_prediction_job(_request_ctx(), bp_job_id)

    assert result is not None
    assert result.predictive_query_link.name == "riskpquery"
    assert result.predictive_query_link.id == PredictiveQueryID.compose(
        "riskpquery", definition.fpid)
    assert result.graph_link.id == GraphIDv2.compose(fpid=graph_fpid)
    assert pquery_service.definitions_by_id_calls == [[pquery_id]]
    assert pquery_service.definition_by_id_calls == []
    assert tag_service.batch_prediction_row_calls == [{bp_job_id}]


@pytest.mark.asyncio
async def test_training_to_sdk_reads_pquery_definition_from_state_service():
    training_job_id = TrainingJobID.make_random_id()
    pquery_fpid = FPID("0123456789abcdef0123456789abcdef")
    graph_fpid = FPID("11111111111111111111111111111111")
    definition = PQueryDefinition(
        timestamps=get_current_timestamps(),
        pquery_name="riskpquery",
        query_str="PREDICT COUNT(t.*) FOR EACH t.user_id",
        graph_fpid=graph_fpid,
        task_type=TaskType.REGRESSION,
        query_type=QueryType.TEMPORAL,
    )
    training_job = SimpleNamespace(
        project_id=_TEST_PROJECT_ID,
        pquery_def_fpid=pquery_fpid,
        advanced_options=ModelPlan(),
        generate_train_table_job_id=GenerateTrainTableJobID.make_random_id(),
    )
    ml_job_backend = SimpleNamespace(
        get_training_job=AsyncMock(return_value=training_job),
        get_encoders_for_job=AsyncMock(return_value={}),
    )
    pquery_service = _PQueryDefinitionReadService({pquery_fpid: definition})
    pquery = _pquery_driver(
        ml_job_backend=ml_job_backend, state_services=SimpleNamespace(
            ml_jobs=_MLJobTagReadService({}),
            pqueries=pquery_service,
            projects=_project_service(),
        ))

    ctx = SimpleNamespace(user_authorization=None)
    with patch(
            "rest.driver.pquery_api_driver.GraphLogic.get_graph_metadata_for_sdk",
            new_callable=AsyncMock,
            return_value=([], [], []),
    ) as graph_metadata, patch(
            "rest.driver.pquery_api_driver.generate_conn_graph_or_table_sdk_script",
            return_value="graph-script\n",
    ) as generate_graph_script, patch(
            "rest.driver.pquery_api_driver.generate_training_job_sdk_script",
            return_value="training-script\n",
    ), patch(
            "rest.driver.pquery_api_driver.make_stype_and_dtype_sdk_compat",
            side_effect=lambda script: script,
    ):
        script = await pquery.training_to_sdk(training_job_id, ctx)

    assert script == "graph-script\ntraining-script\n"
    assert pquery_service.definitions_calls == [{pquery_fpid}]
    graph_metadata.assert_awaited_once_with(graph_fpid)
    assert generate_graph_script.call_args.kwargs["project_id"] == (
        _TEST_PROJECT_ID)
    ml_job_backend.get_training_job.assert_awaited_once_with(
        ctx, training_job_id, strip_expensive_fields=True)


def test_generate_holdout_presigned_url():
    best_train_dir = (
        's3://kumo-public-datasets/jobs/test_trainingjob/models/test/automl/'
        'version_0')
    test_df_path = os.path.join(best_train_dir, 'test_df.parquet')
    get_test_df = AsyncMock(return_value=test_df_path)
    mock_helper = Mock()
    mock_helper.generate_download_url.coroutine = AsyncMock(
        return_value=('https://kumo-public-datasets.s3.amazonaws.com/'
                      'jobs/test_trainingjob/models/test/automl/version_0/'
                      'test_df.parquet?sig=test'))

    pquery = _pquery_driver(ml_job_backend=None, )

    with (
            patch.object(PQueryApiDriver, '_get_test_df',
                         SimpleNamespace(coroutine=get_test_df)),
            patch('rest.driver.pquery_api_driver.get_job_base_dir',
                  return_value='s3://kumo-public-datasets/jobs'),
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=mock_helper),
    ):
        ret = asyncio.run(
            pquery.generate_holdout_presigned_url('test_trainingjob'))

    assert _assert_s3_url_starts_with(
        ret.presigned_url,
        'jobs/test_trainingjob/models/test/automl/version_0/test_df.parquet?')
    get_test_df.assert_awaited_once_with(
        's3://kumo-public-datasets/jobs/test_trainingjob')
    mock_helper.generate_download_url.coroutine.assert_awaited_once_with(
        test_df_path)


@pytest.mark.parametrize('refit_enabled', [False, True])
def test_generate_holdout_presigned_url_local_path(refit_enabled):
    """Local paths fall back to as_url (no presigned helper available)."""
    pquery = _pquery_driver(ml_job_backend=None, )
    best_train_dir = (
        '/kumo-public-datasets/jobs/test_trainingjob/models/test/automl/'
        'version_0')

    mock_home = Mock()
    mock_home.first_task_name.return_value = 'test'
    mock_home.get_best_trainer_dir.return_value = best_train_dir
    mock_trainer_config = Mock()
    mock_trainer_config.refit_cfg.refitted_trainval = refit_enabled

    with (
            patch('rest.driver.pquery_api_driver.HomeDirManager',
                  return_value=mock_home),
            patch('rest.driver.pquery_api_driver.load_json',
                  return_value=mock_trainer_config),
            patch('rest.driver.pquery_api_driver.fsutil.exists',
                  return_value=True),
    ):
        result = asyncio.run(
            pquery.generate_holdout_presigned_url('test_trainingjob'))
        # Local path: helper is None → falls back to as_url (path as-is)
        if refit_enabled:
            expected_dir = best_train_dir.rstrip('/') + '_refit_trainval'
        else:
            expected_dir = best_train_dir
        expected_path = os.path.join(expected_dir, 'test_df.parquet')
        assert result.presigned_url == expected_path


def _create_mock_ml_job_backend(file_storage_path: str):
    """Create a mock ml_job_backend that returns correct job home dirs."""
    mock_backend = Mock()

    async def mock_get_job_home_dir(job_id: str) -> str:
        return f"{file_storage_path}/jobs/{job_id}"

    mock_backend.get_job_home_dir = mock_get_job_home_dir
    return mock_backend


def test_generate_bp_presigned_url(tmp_path):
    """BP download with real files + mocked helper: exercises HomeDirManager
    resolution and single-file presign path."""
    job_dir = tmp_path / "test-bp-job"
    predictions_dir = job_dir / "prediction_output" / "predictions.parquet"
    predictions_dir.mkdir(parents=True)
    out_file = predictions_dir / "part-00000-00000.parquet"
    out_file.write_text("dummy")

    mock_backend = Mock()

    async def mock_get_job_home_dir(job_id: str) -> str:
        return str(job_dir)

    mock_backend.get_job_home_dir = mock_get_job_home_dir

    mock_helper = Mock()
    mock_helper.generate_download_url.return_value = 'https://presigned/part-0'

    pquery = _pquery_driver(ml_job_backend=mock_backend, )

    output_root = str(job_dir / 'prediction_output')
    selection = _prediction_selection(output_root,
                                      PredictionArtifactType.PREDICTIONS,
                                      [str(out_file)])
    with (
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection) as load_selection,
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=mock_helper),
    ):
        ret = asyncio.run(pquery.generate_bp_presigned_url('test-bp-job'))

    assert ret.presigned_url == 'https://presigned/part-0'
    load_selection.assert_called_once_with(output_root)
    mock_helper.generate_download_url.assert_called_once_with(str(out_file))


def test_generate_bp_presigned_url_local_raises(tmp_path):
    """Local paths raise ValueError (no presigned helper)."""
    file_storage_path = str(tmp_path)
    job_dir = tmp_path / "jobs" / "test-bp-job"
    predictions_dir = job_dir / "prediction_output" / "predictions.parquet"
    predictions_dir.mkdir(parents=True)
    (predictions_dir / "part-00000-00000.parquet").write_text("dummy")

    pquery = _pquery_driver(
        ml_job_backend=_create_mock_ml_job_backend(file_storage_path), )

    output_root = str(job_dir / 'prediction_output')
    out_file = str(predictions_dir / 'part-00000-00000.parquet')
    selection = _prediction_selection(output_root,
                                      PredictionArtifactType.PREDICTIONS,
                                      [out_file])
    with (
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection),
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
    ):
        with pytest.raises(ValueError, match="Download not supported for"):
            asyncio.run(pquery.generate_bp_presigned_url('test-bp-job'))


@pytest.mark.parametrize(('helper', 'expected'), [
    (LocalPresignedHelper(LocalURI(path='/tmp/predictions')), False),
    (Mock(), True),
    (None, False),
])
def test_supports_batch_prediction_download(helper, expected):
    mock_backend = Mock()
    mock_backend.get_job_home_dir = AsyncMock(return_value='/jobs/test-bp-job')
    pquery = _pquery_driver(ml_job_backend=mock_backend)

    output_root = '/jobs/test-bp-job/prediction_output'
    selection = _prediction_selection(
        output_root,
        PredictionArtifactType.PREDICTIONS,
        [output_root + '/_selections/selection-1/predictions.parquet/part'],
    )
    with (
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection) as load_selection,
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=helper),
    ):
        actual = asyncio.run(
            pquery.supports_batch_prediction_download('test-bp-job'))

    assert actual is expected
    load_selection.assert_called_once_with(output_root)


def test_get_job_download_capability_never_validates_part_objects():
    mock_backend = Mock()
    mock_backend.get_job_home_dir = AsyncMock(return_value='/jobs/test-bp-job')
    pquery = _pquery_driver(ml_job_backend=mock_backend)
    output_root = '/jobs/test-bp-job/prediction_output'
    selection = _prediction_selection(
        output_root,
        PredictionArtifactType.PREDICTIONS,
        [
            output_root + '/_attempts/runtime-00000/winner/'
            'predictions.parquet/part-00000-00000.parquet'
        ],
    )

    with (
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection) as metadata_load,
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.fsutil.listdir',
                side_effect=AssertionError(
                    'GET capability must not probe selected roots')) as listdir,
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=Mock()),
    ):
        assert asyncio.run(
            pquery.supports_batch_prediction_download('test-bp-job')) is True

    metadata_load.assert_called_once_with(output_root)
    listdir.assert_not_called()


def test_generate_bp_presigned_url_s3_single_file():
    """S3 path with single output file: presigns the file directly."""
    s3_job_dir = 's3://bucket/jobs/test-bp-job'
    s3_predictions = s3_job_dir + '/prediction_output/predictions.parquet'
    s3_file = s3_predictions + '/part-00000-00000.parquet'

    mock_backend = Mock()
    mock_backend.get_job_home_dir = AsyncMock(return_value=s3_job_dir)

    mock_home = Mock()
    mock_home.prediction_output_dir = s3_job_dir + '/prediction_output'

    mock_helper = Mock()
    mock_helper.generate_download_url.return_value = 'https://presigned/part-0'

    pquery = _pquery_driver(
        ml_job_backend=mock_backend,
        allow_local_batch_pred_paths=False,
    )

    selection = _prediction_selection(
        mock_home.prediction_output_dir,
        PredictionArtifactType.PREDICTIONS,
        [s3_file],
    )
    with (
            patch('rest.driver.pquery_api_driver.HomeDirManager',
                  return_value=mock_home),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection),
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.fsutil.listdir',
                return_value=[s3_file]),
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=mock_helper),
    ):
        ret = asyncio.run(pquery.generate_bp_presigned_url('test-bp-job'))

    assert ret.presigned_url == 'https://presigned/part-0'
    mock_helper.generate_download_url.assert_called_once_with(s3_file)


def test_generate_bp_presigned_url_s3_multi_file():
    """S3 path with multiple output files: zips first, then presigns the zip."""
    s3_job_dir = 's3://bucket/jobs/test-bp-job'
    s3_predictions = s3_job_dir + '/prediction_output/predictions.parquet'
    archive_name = hashlib.sha256(b'test-bp-job').hexdigest() + '.zip'
    s3_zip = f's3://bucket/root/tmp_archives/{archive_name}'

    mock_backend = Mock()
    mock_backend.get_job_home_dir = AsyncMock(return_value=s3_job_dir)

    mock_home = Mock()
    mock_home.prediction_output_dir = s3_job_dir + '/prediction_output'

    mock_helper = Mock()
    mock_helper.generate_download_url.return_value = 'https://presigned/output.zip'

    pquery = _pquery_driver(
        ml_job_backend=mock_backend,
        allow_local_batch_pred_paths=False,
    )

    files = [
        s3_predictions + '/part-00000-00000.parquet',
        s3_predictions + '/part-00001-00000.parquet'
    ]
    selection = _prediction_selection(
        mock_home.prediction_output_dir,
        PredictionArtifactType.PREDICTIONS,
        files,
    )
    with (
            patch('rest.driver.pquery_api_driver.HomeDirManager',
                  return_value=mock_home),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection),
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.fsutil.listdir',
                return_value=files),
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=mock_helper),
            patch('rest.driver.pquery_api_driver.fsutil.zip_files',
                  return_value=s3_zip) as zip_files,
            patch(
                'rest.driver.pquery_api_driver.deploy_env',
                return_value=SimpleNamespace(
                    data_plane_storage_root='s3://bucket/root')),
    ):
        ret = asyncio.run(pquery.generate_bp_presigned_url('test-bp-job'))

    assert ret.presigned_url == 'https://presigned/output.zip'
    mock_helper.generate_download_url.assert_called_once_with(s3_zip)
    zip_files.assert_called_once_with(files, output_path=s3_zip)


def test_generate_bp_presigned_url_unsupported_backend():
    """Non-S3/GCS path (e.g. ADLS) raises ValueError."""
    job_dir = 'abfss://container@account.dfs.core.windows.net/jobs/test-bp-job'
    predictions = job_dir + '/prediction_output/predictions.parquet'

    mock_backend = Mock()
    mock_backend.get_job_home_dir = AsyncMock(return_value=job_dir)

    mock_home = Mock()
    mock_home.prediction_output_dir = job_dir + '/prediction_output'

    pquery = _pquery_driver(
        ml_job_backend=mock_backend,
        allow_local_batch_pred_paths=False,
    )

    selection = _prediction_selection(
        mock_home.prediction_output_dir,
        PredictionArtifactType.PREDICTIONS,
        [predictions + '/part-00000-00000.parquet'],
    )
    with (
            patch('rest.driver.pquery_api_driver.HomeDirManager',
                  return_value=mock_home),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection),
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.fsutil.listdir',
                return_value=[predictions + '/part-00000-00000.parquet']),
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=None),
            patch('rest.driver.pquery_api_driver.fsutil.filesize',
                  return_value=1),
    ):
        with pytest.raises(ValueError, match="Download not supported for"):
            asyncio.run(pquery.generate_bp_presigned_url('test-bp-job'))


@pytest.mark.parametrize('artifact_type', _ARTIFACT_TYPES)
def test_get_presigned_artifact_urls_for_s3(artifact_type):
    """S3 paths: helper.generate_download_url is called for each partition."""
    s3_job_dir = 's3://bucket/jobs/test-bp-outputs'

    mock_backend = Mock()
    mock_backend.get_job_home_dir = AsyncMock(return_value=s3_job_dir)

    pquery = _pquery_driver(
        ml_job_backend=mock_backend,
        allow_local_batch_pred_paths=False,
    )

    _configure_batch_prediction_artifact(pquery, artifact_type)
    method = _artifact_method(pquery, artifact_type)

    mock_helper = Mock()
    mock_helper.generate_download_url.side_effect = (
        lambda p: f'https://presigned/{p}')

    output_root = s3_job_dir + '/prediction_output'
    artifact_root = (output_root + '/' + artifact_type.output_file_name)
    s3_partitions = [
        artifact_root + '/part-00000-00000.parquet',
        artifact_root + '/part-00001-00000.parquet'
    ]
    selection = _prediction_selection(output_root, artifact_type, s3_partitions)

    with (
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection),
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.fsutil.listdir',
                return_value=s3_partitions),
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=mock_helper),
    ):
        ret = asyncio.run(method(None, 'test-bp-outputs'))

    assert ret is not None
    partitions = _artifact_partitions(ret, artifact_type)
    assert len(partitions) == 2
    assert all(p.startswith('https://presigned/') for p in partitions)
    assert mock_helper.generate_download_url.call_count == 2


@pytest.mark.parametrize('artifact_type', _ARTIFACT_TYPES)
def test_get_presigned_artifact_urls_as_url_fallback(artifact_type):
    """Non-presignable path: falls back to customer_facing_path for each partition."""
    job_dir = '/local/jobs/test-bp-outputs'

    mock_backend = Mock()
    mock_backend.get_job_home_dir = AsyncMock(return_value=job_dir)

    pquery = _pquery_driver(ml_job_backend=mock_backend, )

    _configure_batch_prediction_artifact(pquery, artifact_type)
    method = _artifact_method(pquery, artifact_type)

    output_root = job_dir + '/prediction_output'
    artifact_root = output_root + '/' + artifact_type.output_file_name
    partitions = [
        artifact_root + '/part-00000-00000.parquet',
        artifact_root + '/part-00001-00000.parquet'
    ]
    selection = _prediction_selection(output_root, artifact_type, partitions)

    with (
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection),
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
            patch(
                'rest.backends.ml_backend.prediction_artifacts.fsutil.listdir',
                return_value=partitions),
            patch('rest.driver.pquery_api_driver.fsutil.get_presigned_helper',
                  return_value=None),
            patch('rest.driver.pquery_api_driver.fsutil.customer_facing_path',
                  side_effect=lambda p: p),
    ):
        ret = asyncio.run(method(None, 'test-bp-outputs'))

    assert ret is not None
    assert _artifact_partitions(ret, artifact_type) == partitions


@pytest.mark.parametrize(
    ('artifact_type', 'partition_count'),
    [
        (PredictionArtifactType.PREDICTIONS, 1),
        (PredictionArtifactType.EMBEDDINGS, 2),
    ],
)
def test_get_presigned_artifact_urls_for_implicit_local_output(
        artifact_type, partition_count, tmp_path):
    """Empty outputs store requested artifacts only in Kumo's data plane."""
    job_dir = tmp_path / 'test-bp-outputs'
    artifact_dir = (job_dir / 'prediction_output' /
                    f'{artifact_type.value.lower()}.parquet')
    artifact_dir.mkdir(parents=True)
    expected = []
    for part in range(partition_count):
        path = artifact_dir / f'part-{part:05d}-00000.parquet'
        path.write_text('dummy')
        expected.append(str(path))

    mock_backend = Mock()
    mock_backend.get_job_home_dir = AsyncMock(return_value=str(job_dir))
    pquery = _pquery_driver(ml_job_backend=mock_backend)
    _configure_implicit_batch_prediction_artifact(pquery)

    selection = _prediction_selection(str(job_dir / 'prediction_output'),
                                      artifact_type, expected)
    with (
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection),
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
    ):
        response = asyncio.run(
            _artifact_method(pquery, artifact_type)(None, 'test-bp-outputs'))

    assert response is not None
    assert _artifact_partitions(response, artifact_type) == expected


@pytest.mark.parametrize('artifact_type', _ARTIFACT_TYPES)
def test_get_presigned_artifact_urls_bp_job_not_found(artifact_type):
    pquery = _pquery_driver(ml_job_backend=None, )

    pquery.get_batch_prediction_job = AsyncMock(return_value=None)
    job_id = 'test-bp-outputs'

    method = _artifact_method(pquery, artifact_type)

    with pytest.raises(
            ValueError, match=re.escape(
                BATCH_PREDICTION_JOB_NOT_FOUND_MESSAGE.format(job_id))):
        asyncio.run(method(None, job_id))


@pytest.mark.parametrize('artifact_type', _ARTIFACT_TYPES)
def test_get_presigned_artifact_urls_artifact_type_not_found(artifact_type):
    pquery = _pquery_driver(ml_job_backend=None, )

    other_type = next(value for value in _ARTIFACT_TYPES
                      if value != artifact_type)
    _configure_batch_prediction_artifact(pquery, other_type)
    method = _artifact_method(pquery, artifact_type)

    with pytest.raises(
            ValueError, match=re.escape(
                PREDICTION_ARTIFACT_TYPE_NOT_FOUND_MESSAGE.format(
                    artifact_type))):
        asyncio.run(method(None, 'test-bp-outputs'))


@pytest.mark.parametrize('artifact_type', _ARTIFACT_TYPES)
def test_get_presigned_artifact_urls_file_not_found(artifact_type, tmp_path):
    file_storage_path = str(tmp_path)
    pquery = _pquery_driver(
        ml_job_backend=_create_mock_ml_job_backend(file_storage_path), )

    _configure_batch_prediction_artifact(pquery, artifact_type)
    method = _artifact_method(pquery, artifact_type)

    ret = asyncio.run(method(None, 'test-bp-outputs'))

    assert ret is None


@pytest.mark.parametrize('artifact_type', _ARTIFACT_TYPES)
def test_get_presigned_artifact_urls_for_local_path(artifact_type, tmp_path):
    """Test that local paths return file paths directly (for testing/devserver)."""
    # Create dummy output files in tmp directory
    job_dir = tmp_path / "test-bp-outputs"
    output_dir = job_dir / "prediction_output"
    output_dir.mkdir(parents=True)

    # Create predictions or embeddings parquet directory with partitions
    artifact_dir = output_dir / f"{artifact_type.value.lower()}.parquet"

    artifact_dir.mkdir(parents=True)
    (artifact_dir / "part-00000-00000.parquet").write_text("dummy")
    (artifact_dir / "part-00001-00000.parquet").write_text("dummy")

    mock_backend = Mock()

    async def mock_get_job_home_dir(job_id: str) -> str:
        return str(job_dir)

    mock_backend.get_job_home_dir = mock_get_job_home_dir

    pquery = _pquery_driver(ml_job_backend=mock_backend, )

    _configure_batch_prediction_artifact(pquery, artifact_type)
    method = _artifact_method(pquery, artifact_type)

    # Local paths should return file paths directly (not presigned URLs)
    files = [
        str(artifact_dir / 'part-00000-00000.parquet'),
        str(artifact_dir / 'part-00001-00000.parquet')
    ]
    selection = _prediction_selection(str(output_dir), artifact_type, files)
    with (
            patch(
                'rest.backends.ml_backend.prediction_artifacts.load_current_prediction_selection_metadata',
                return_value=selection),
            patch('rest.backends.ml_backend.prediction_artifacts.fsutil.exists',
                  side_effect=_selected_pointer_exists),
    ):
        ret = asyncio.run(method(None, 'test-bp-outputs'))
    assert ret is not None

    partitions = _artifact_partitions(ret, artifact_type)
    assert partitions is not None
    assert len(partitions) == 2
    assert all(p.startswith(str(tmp_path)) for p in partitions)


@pytest.mark.asyncio
async def test_get_error_details():
    pquery = _pquery_driver(ml_job_backend=None, )

    training_job_id = 'trainingjob-ueiu'
    with (patch(
            "rest.driver.pquery_api_driver._get_warn_details",
            new_callable=AsyncMock,
    ) as mock_get_warn_details):
        mock_get_warn_details.return_value = [[
            ErrWarnDetails(is_error=True, title='Training Error',
                           description='OOM in GPU', is_dismissible=False)
        ]]
        result = await pquery.get_errors_for_job(TrainingJobID(training_job_id))

        mock_get_warn_details.assert_awaited_once()
        assert len(result.items) == 1
        err: ErrorDetail = result.items[0]
        err.type == ErrorType.ERROR
        err.description = 'OOM in GPU'
        err.title = 'Training Error'


def _make_mock_ml_job_backend(job_entry=None):
    """Helper to create a mock MLJobBackend with a job entry."""
    mock_backend = Mock()
    # Mock the RPC method that returns job execution details
    execution = job_entry.execution if job_entry is not None else None
    mock_backend.get_generate_table_job_execution = AsyncMock(
        return_value=execution)
    return mock_backend


def _make_gen_train_table_job_entry(
    job_id,
    validation=None,
    failure_info=None,
    log_detail=None,
    log_stage_name="Failed",
):
    """Helper to create a generate-train-table job entry for testing."""
    from datetime import datetime
    now = datetime.now()
    return GenerateTrainTableJobDBEntry(
        id=job_id,
        project_id=_TEST_PROJECT_ID,
        config=Mock(spec=GenerateTrainTableJobConfig),
        execution=GenerateTrainTableJobOutput(
            status=JobStatus.FAILED,
            log=JobEventLogEntry(
                stage_name=log_stage_name,
                last_updated_at=now,
                detail=log_detail,
            ),
            validation=validation,
            failure_info=failure_info,
        ),
        timestamps=ResourceTimestamps(created_at=now, updated_at=now),
    )


@pytest.mark.asyncio
async def test_get_errors_for_gen_train_table_job_from_ml_db_validation():
    """Test that get_errors_for_job reads StateDB validation first."""
    job_id = GenerateTrainTableJobID('gen-traintable-job-test1')
    validation = ValidationResponse(errors=[
        ValidationError(title='Validation Error', message='Missing column X')
    ])
    job_entry = _make_gen_train_table_job_entry(job_id, validation=validation)

    pquery = _pquery_driver(
        ml_job_backend=_make_mock_ml_job_backend(job_entry), )

    result = await pquery.get_errors_for_job(job_id)

    assert len(result.items) == 1
    err = result.items[0]
    assert err.type == ErrorType.ERROR
    assert err.title == 'Validation Error'
    assert err.description == 'Missing column X'


@pytest.mark.asyncio
async def test_get_errors_for_gen_train_table_job_from_failure_info():
    """Test fallback to failure_info when validation is empty."""
    job_id = GenerateTrainTableJobID('gen-traintable-job-test2')
    failure_info = JobFailureInfo(
        message='Job failed due to OOM',
        stacktrace='...',
    )
    job_entry = _make_gen_train_table_job_entry(
        job_id,
        validation=None,
        failure_info=failure_info,
    )

    pquery = _pquery_driver(
        ml_job_backend=_make_mock_ml_job_backend(job_entry), )

    # Mock _get_job_dir to return None (no ValidationResponse file)
    with patch.object(pquery, '_get_job_dir', new_callable=AsyncMock) as mock:
        mock.return_value = None
        result = await pquery.get_errors_for_job(job_id)

    assert len(result.items) == 1
    err = result.items[0]
    assert err.type == ErrorType.ERROR
    assert err.title is None
    assert err.description == 'Job failed due to OOM'


@pytest.mark.asyncio
async def test_get_errors_for_gen_train_table_job_from_log_detail():
    """Test fallback to log.detail when validation and failure_info are empty."""
    job_id = GenerateTrainTableJobID('gen-traintable-job-test3')
    job_entry = _make_gen_train_table_job_entry(
        job_id,
        validation=None,
        failure_info=None,
        log_detail='Something went wrong during execution',
        log_stage_name='Execution Failed',
    )

    pquery = _pquery_driver(
        ml_job_backend=_make_mock_ml_job_backend(job_entry), )

    # Mock _get_job_dir to return None (no ValidationResponse file)
    with patch.object(pquery, '_get_job_dir', new_callable=AsyncMock) as mock:
        mock.return_value = None
        result = await pquery.get_errors_for_job(job_id)

    assert len(result.items) == 1
    err = result.items[0]
    assert err.type == ErrorType.ERROR
    assert err.title is None
    assert err.description == 'Something went wrong during execution'


@pytest.mark.asyncio
async def test_get_errors_for_gen_train_table_job_not_found():
    """Test that missing StateDB jobs return no errors."""
    job_id = GenerateTrainTableJobID('gen-traintable-job-nonexistent')

    pquery = _pquery_driver(
        ml_job_backend=_make_mock_ml_job_backend(),  # Job not found
    )

    result = await pquery.get_errors_for_job(job_id)

    assert len(result.items) == 0


@pytest.mark.asyncio
async def test_get_allowed_bp_overrides():
    pquery_api_driver = _filter_override_driver(_REGRESSION_QUERY,
                                                TaskType.REGRESSION)

    training_job_id = TrainingJobID.make_random_id()
    response = (await pquery_api_driver.get_allowed_bp_query_overrides_v2(
        training_job_id=training_job_id,
        request_context=None,
    ))
    assert response is not None
    assert response.text.startswith('PREDICT SUM(transactions.price WHERE')
    assert response.editable_fields is not None
    assert response.editable_fields[0].field_name == 'Entity filter'
    assert response.editable_fields[0].text == 'customers.active = 1'
    assert response.editable_fields[0].position == (103, 130)


@pytest.mark.asyncio
async def test_get_allowed_bp_overrides_training_job_not_found():
    pquery_api_driver = _pquery_driver(ml_job_backend=None, )

    # Mock get_training_job to return None
    pquery_api_driver.get_training_job = AsyncMock(return_value=None)

    training_job_id = TrainingJobID.make_random_id()

    with pytest.raises(TrainingJobDoesNotExistError):
        await pquery_api_driver.get_allowed_bp_query_overrides_v2(
            training_job_id=training_job_id,
            request_context=None,
        )


@pytest.mark.asyncio
async def test_insert_filter_overrides_v2():
    pquery_api_driver = _filter_override_driver(_REGRESSION_QUERY,
                                                TaskType.REGRESSION)

    training_job_id = TrainingJobID.make_random_id()
    overrides = InsertFilterOverridesRequest(
        entity_filter='customers.active = 2',
        target_filter='',
    )

    response = await pquery_api_driver.insert_filter_overrides_v2(
        training_job_id=training_job_id,
        overrides=overrides,
        request_context=None,
    )

    assert response is not None
    assert response.text.startswith('PREDICT SUM(transactions.price WHERE')
    # Verify the entity filter was updated
    assert 'customers.active = 2' in response.text
    # Verify positions were updated
    assert response.editable_fields[0].field_name == 'Entity filter'
    assert response.editable_fields[0].text == 'customers.active = 2'
    # Position should be updated to reflect new filter text
    assert response.editable_fields[0].position == (103, 130)


@pytest.mark.asyncio
async def test_insert_filter_overrides_v2_link_prediction():
    pquery_api_driver = _filter_override_driver(
        _LINK_PREDICTION_QUERY, TaskType.TEMPORAL_LINK_PREDICTION)

    training_job_id = TrainingJobID.make_random_id()
    overrides = InsertFilterOverridesRequest(
        entity_filter='COUNT(transactions.*, -180, -30) > 5',
        target_filter='articles.product_type = 272',
    )

    response = await pquery_api_driver.insert_filter_overrides_v2(
        training_job_id=training_job_id,
        overrides=overrides,
        request_context=None,
    )

    assert response is not None
    assert response.text.startswith(
        'PREDICT LIST_DISTINCT(transactions.article_id WHERE')
    # Verify the entity filter was updated
    assert 'COUNT(transactions.*, -180, -30) > 5' in response.text
    entity_filter_field = next(f for f in response.editable_fields
                               if f.field_name == 'Entity filter')
    assert entity_filter_field.text == 'COUNT(transactions.*, -180, -30) > 5'
    assert entity_filter_field.position == (166, 209)

    # Find the target filter field
    target_filter_field = next(f for f in response.editable_fields
                               if f.field_name == 'Target filter')
    assert target_filter_field.text == 'articles.product_type = 272'
    assert target_filter_field.position == (45, 118)


@pytest.mark.asyncio
async def test_insert_filter_overrides_v2_link_prediction_incorrect_target():
    pquery_api_driver = _filter_override_driver(
        _LINK_PREDICTION_QUERY, TaskType.TEMPORAL_LINK_PREDICTION)

    training_job_id = TrainingJobID.make_random_id()
    overrides = InsertFilterOverridesRequest(
        entity_filter='COUNT(transactions.*, -180, -30) > 5',
        target_filter='transactions.sales_channel = 2',
    )

    response = await pquery_api_driver.insert_filter_overrides_v2(
        training_job_id=training_job_id,
        overrides=overrides,
        request_context=None,
    )

    assert response is not None
    assert response.validation_response is not None
    assert response.validation_response.ok is False
    assert len(response.validation_response.errors) == 1
    assert response.validation_response.errors[
        0].title == 'Invalid target filter'
    assert ('Table transactions appears in the target filter'
            in response.validation_response.errors[0].message)


def test_create_ordered_job_tags():
    pquery_api_driver = _pquery_driver(ml_job_backend=None, )

    # Test inputs
    request_tags = {"custom_tag1": "value1", "custom_tag2": "value2"}
    pquery_name = "test_pquery"
    pquery_def_fpid = FPID("0123456789abcdef0123456789abcdef")

    # Call the method
    result = pquery_api_driver.create_ordered_job_tags(
        request_tags=request_tags, pquery_name=pquery_name,
        pquery_def_fpid=pquery_def_fpid)

    # Convert result keys to list to verify order
    keys = list(result.keys())

    # Verify custom tags come first
    assert keys[0] == "custom_tag1"
    assert keys[1] == "custom_tag2"

    # Verify standard tags come after
    assert keys[2] == TAG_KEY_PQUERY_NAME
    assert keys[3] == TAG_KEY_PQUERY_ID

    # Verify all values are correct
    assert result[JobTagKey("custom_tag1")] == JobTagValue("value1")
    assert result[JobTagKey("custom_tag2")] == JobTagValue("value2")
    assert result[TAG_KEY_PQUERY_NAME] == JobTagValue.of_pquery_name(
        pquery_name)
    assert result[TAG_KEY_PQUERY_ID] == JobTagValue.of_pquery_id(
        pquery_def_fpid)


# ---------------------------------------------------------------------------
# ForecastingParams tests
# ---------------------------------------------------------------------------


class TestForecastingParams:
    """Tests for the ForecastingParams dataclass."""
    def test_from_plan_with_concrete_values(self):
        plan = TrainingTableGenerationPlan()
        plan.lag_timesteps = 7
        plan.year_over_year = True
        plan.forecast_length = 14

        params = ForecastingParams.from_plan(plan)

        assert params.lag_timesteps == 7
        assert params.year_over_year is True
        assert params.forecast_length == 14
        assert params.is_forecasting is True

    def test_from_plan_with_default_values(self):
        plan = TrainingTableGenerationPlan()
        # Defaults are MissingType (shown as ???)

        params = ForecastingParams.from_plan(plan)

        assert params.lag_timesteps is None
        assert params.year_over_year is False
        assert params.forecast_length is None
        assert params.is_forecasting is False

    def test_from_plan_partial_values(self):
        plan = TrainingTableGenerationPlan()
        plan.lag_timesteps = 5
        # year_over_year and forecast_length remain inferred

        params = ForecastingParams.from_plan(plan)

        assert params.lag_timesteps == 5
        assert params.year_over_year is False
        assert params.forecast_length is None
        assert params.is_forecasting is False

    def test_default(self):
        params = ForecastingParams.default()

        assert params.lag_timesteps is None
        assert params.year_over_year is False
        assert params.forecast_length is None
        assert params.is_forecasting is False

    def test_is_forecasting_true(self):
        params = ForecastingParams(lag_timesteps=7, year_over_year=False,
                                   forecast_length=14)
        assert params.is_forecasting is True

    def test_is_forecasting_false(self):
        params = ForecastingParams(lag_timesteps=7, year_over_year=True,
                                   forecast_length=None)
        assert params.is_forecasting is False

    def test_is_forecasting_false_for_single_step(self):
        params = ForecastingParams(lag_timesteps=7, year_over_year=True,
                                   forecast_length=1)
        assert params.is_forecasting is False

    def test_frozen(self):
        params = ForecastingParams(lag_timesteps=7, year_over_year=False,
                                   forecast_length=14)
        with pytest.raises(AttributeError):
            params.forecast_length = 21

    # has_any tests ---------------------------------------------------------

    def test_has_any_all_defaults(self):
        params = ForecastingParams.default()
        assert params.has_any is False

    def test_has_any_with_year_over_year(self):
        params = ForecastingParams(lag_timesteps=None, year_over_year=True,
                                   forecast_length=None)
        assert params.has_any is True

    def test_has_any_with_lag_timesteps(self):
        params = ForecastingParams(lag_timesteps=0, year_over_year=False,
                                   forecast_length=None)
        assert params.has_any is True

    def test_has_any_with_forecast_length(self):
        params = ForecastingParams(lag_timesteps=None, year_over_year=False,
                                   forecast_length=4)
        assert params.has_any is True

    # from_plan with PredictionTableGenerationPlan --------------------------

    def test_from_prediction_plan_with_concrete_values(self):
        plan = PredictionTableGenerationPlan()
        plan.lag_timesteps = 3
        plan.year_over_year = True
        plan.forecast_length = 7

        params = ForecastingParams.from_plan(plan)

        assert params.lag_timesteps == 3
        assert params.year_over_year is True
        assert params.forecast_length == 7

    def test_from_prediction_plan_with_default_values(self):
        plan = PredictionTableGenerationPlan()

        params = ForecastingParams.from_plan(plan)

        assert params.lag_timesteps is None
        assert params.year_over_year is False
        assert params.forecast_length is None
        assert params.has_any is False


# ---------------------------------------------------------------------------
# _get_forecasting_params tests
# ---------------------------------------------------------------------------


class TestGetForecastingParams:
    """Tests for PQueryApiDriver._get_forecasting_params."""
    @pytest.mark.asyncio
    async def test_returns_default_when_no_training_job(self):
        """When training_job_id is None, return default (non-forecasting)."""
        pquery = _pquery_driver(ml_job_backend=None, )

        result = await pquery._get_forecasting_params(None)

        assert result == ForecastingParams.default()

    @pytest.mark.asyncio
    async def test_extracts_from_gen_tt_job(self):
        """When gen-TT job exists, extract params from its plan."""
        plan = TrainingTableGenerationPlan()
        plan.lag_timesteps = 7
        plan.year_over_year = True
        plan.forecast_length = 14

        gen_tt_entry = Mock()
        gen_tt_entry.config.plan = plan

        mock_backend = Mock()
        mock_backend.get_training_gen_tt_job = AsyncMock(
            return_value=gen_tt_entry)

        pquery = _pquery_driver(ml_job_backend=mock_backend, )

        result = await pquery._get_forecasting_params(TrainingJobID('train-123')
                                                      )

        assert result.lag_timesteps == 7
        assert result.year_over_year is True
        assert result.forecast_length == 14
        assert result.is_forecasting is True
        mock_backend.get_training_gen_tt_job.assert_awaited_once_with(
            TrainingJobID('train-123'))

    @pytest.mark.asyncio
    async def test_falls_back_to_training_plan(self):
        """When gen-TT job is None, fall back to training plan."""
        plan = TrainingTableGenerationPlan()
        plan.lag_timesteps = 3
        plan.forecast_length = 10

        mock_backend = Mock()
        mock_backend.get_training_gen_tt_job = AsyncMock(return_value=None)
        mock_backend.get_training_job_training_table_plan = AsyncMock(
            return_value=plan)

        pquery = _pquery_driver(ml_job_backend=mock_backend, )

        result = await pquery._get_forecasting_params(TrainingJobID('train-456')
                                                      )

        assert result.lag_timesteps == 3
        assert result.forecast_length == 10
        assert result.is_forecasting is True
        mock_backend.get_training_job_training_table_plan \
            .assert_awaited_once_with(TrainingJobID('train-456'))

    @pytest.mark.asyncio
    async def test_returns_default_when_neither_source_available(self):
        """When both gen-TT and training plan are None, return default."""
        mock_backend = Mock()
        mock_backend.get_training_gen_tt_job = AsyncMock(return_value=None)
        mock_backend.get_training_job_training_table_plan = AsyncMock(
            return_value=None)

        pquery = _pquery_driver(ml_job_backend=mock_backend, )

        result = await pquery._get_forecasting_params(TrainingJobID('train-789')
                                                      )

        assert result == ForecastingParams.default()
        assert result.is_forecasting is False

    @pytest.mark.asyncio
    async def test_raises_on_gen_tt_job_corruption(self):
        """When gen-TT job ID exists but entry is missing, raise error."""
        mock_backend = Mock()
        mock_backend.get_training_gen_tt_job = AsyncMock(
            side_effect=ValidationException(
                title="Internal Data Error",
                message="Training table generation job... is missing"))

        pquery = _pquery_driver(ml_job_backend=mock_backend, )

        with pytest.raises(ValidationException) as exc_info:
            await pquery._get_forecasting_params(
                TrainingJobID('train-corrupted'))

        assert exc_info.value._title == "Internal Data Error"
        assert "is missing" in exc_info.value._message


@pytest.mark.parametrize(
    ('parsed_task_type', 'forecasting_params', 'expected_task_type'),
    [
        pytest.param(
            TaskType.REGRESSION,
            ForecastingParams(lag_timesteps=12, year_over_year=True,
                              forecast_length=12),
            TaskType.FORECASTING,
            id='forecast-trained-regression',
        ),
        pytest.param(
            TaskType.REGRESSION,
            ForecastingParams(lag_timesteps=12, year_over_year=True,
                              forecast_length=1),
            TaskType.REGRESSION,
            id='single-step-regression',
        ),
        pytest.param(
            TaskType.REGRESSION,
            ForecastingParams(lag_timesteps=12, year_over_year=False,
                              forecast_length=None),
            TaskType.REGRESSION,
            id='lag-only-regression',
        ),
        pytest.param(
            TaskType.REGRESSION,
            ForecastingParams(lag_timesteps=None, year_over_year=True,
                              forecast_length=None),
            TaskType.REGRESSION,
            id='year-over-year-only-regression',
        ),
        pytest.param(
            TaskType.FORECASTING,
            ForecastingParams.default(),
            TaskType.FORECASTING,
            id='explicit-forecast-query',
        ),
    ],
)
@pytest.mark.asyncio
async def test_bp_promotes_forecast_trained_regression_query_task_type(
        monkeypatch, parsed_task_type, forecasting_params, expected_task_type):
    """An external prediction table retains the trained model's task type."""
    training_job_id = TrainingJobID('trainingjob-forecast')
    pred_table_job_id = GeneratePredictionTableJobID('gen-predtable-job-done')
    pquery_fpid = FPID('a' * 32)
    graph_fpid = FPID('b' * 32)
    pquery_id = PredictiveQueryID.public(pquery_fpid)

    driver = _pquery_driver()
    driver._get_forecasting_params = AsyncMock(return_value=forecasting_params)

    pq_resource = SimpleNamespace(
        graph=SimpleNamespace(fpid=graph_fpid),
        query_string='PREDICT SUM(target.amount) FOR customer.id = 0',
        name='forecast',
        desc='',
        list_distinct_target_is_array=False,
        lp_bp_layout=None,
    )
    parsed_pq = SimpleNamespace(
        task_type=parsed_task_type,
        query_type=QueryType.TEMPORAL,
        predictive_query=SimpleNamespace(
            validated_predictive_query=SimpleNamespace(top_k=None)),
    )
    pquery_logic = SimpleNamespace(parse_predictive_query_v2=AsyncMock(
        return_value=(parsed_pq, None)))
    persisted = SimpleNamespace(config=None)

    async def create_job(_ctx, request, _pq_resource, _pq_def_fpid, task_type,
                         _query_type, model_training_job_id, **_kwargs):
        persisted.config = request.to_config(
            predictive_query_id=pquery_id,
            training_job_id=model_training_job_id,
            graph_id=GraphIDv2.public(graph_fpid),
            task_type=task_type,
            filter_override_args=None,
            effective_num_classes_to_return=None,
            effective_threshold=None,
        )
        return BatchPredictionJobID('bp-job-forecast')

    bp_logic = SimpleNamespace(
        validate_batch_prediction_args_v2=AsyncMock(
            return_value=ValidationResponse()),
        create_batch_prediction_job_v2=AsyncMock(side_effect=create_job),
    )
    driver._authorize_custom_prediction_table_source = AsyncMock(
        return_value=(None, None))
    driver.get_generate_pred_table_job = AsyncMock(return_value=SimpleNamespace(
        config=SimpleNamespace(pquery_id=pquery_id),
        job_status_report=SimpleNamespace(status=JobStatus.DONE),
    ))
    driver._resolve_pquery_for_launch = AsyncMock(return_value=(pquery_fpid,
                                                                pq_resource))
    driver._handle_snapshot_override_v2 = AsyncMock(return_value=(graph_fpid,
                                                                  None))
    driver.validate_batch_prediction_inputs_v2 = AsyncMock(
        return_value=ValidationResponse())
    driver._link_graph_to_project = AsyncMock()
    driver._pquery_logic = Mock(return_value=pquery_logic)
    driver._batch_prediction_logic = Mock(return_value=bp_logic)
    driver._apply_job_tags = AsyncMock()
    monkeypatch.setattr(
        'rest.driver.pquery_api_driver.PQueryValidationArgs',
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    request = jobs_api.BatchPredictionRequest(
        job_tags={},
        model_training_job_id=str(training_job_id),
        predict_options=jobs_api.BatchPredictionOptions(),
        outputs=[
            jobs_api.LocalPredictionOutput(
                artifact_type=PredictionArtifactType.PREDICTIONS)
        ],
        pred_table_job_id=str(pred_table_job_id),
    )
    job_id, validation = await driver.start_batch_prediction_job_v2(
        ctx=SimpleNamespace(), request=request, project_id=_TEST_PROJECT_ID)

    assert job_id == BatchPredictionJobID('bp-job-forecast')
    assert validation.ok
    assert persisted.config.task_type == expected_task_type
    assert bp_logic.validate_batch_prediction_args_v2.await_args.args[3] == (
        expected_task_type)
    assert bp_logic.create_batch_prediction_job_v2.await_args.args[4] == (
        expected_task_type)

    prediction_config = SimpleNamespace(
        ignore_time_col=None,
        pred_time_col_name=None,
        start_time=None,
        output_extra_fields=None,
    )
    _configure_prediction_output_fields(
        prediction_config,
        SimpleNamespace(start_time=None, config=persisted.config),
    )
    expected_ignore_time_col = expected_task_type != TaskType.FORECASTING
    assert prediction_config.ignore_time_col == expected_ignore_time_col


def _make_validate_bp_inputs_driver(pred_table_status: JobStatus):
    """Build a PQueryApiDriver whose ``validate_batch_prediction_inputs_v2``
    pre-guard helper calls are stubbed so execution reaches the gen-pred-table
    status guard. The training and prediction graphs are made identical so the
    graph-compatibility checks pass trivially.
    """
    graph = SimpleNamespace(tables={}, col_groups=[])
    training_pquery = SimpleNamespace(graph=graph, query_string="q")
    graph_def = SimpleNamespace(graph=graph)
    training_job = SimpleNamespace(
        execution=SimpleNamespace(status=JobStatus.DONE),
        generate_train_table_job_id="gen-tt-job-1",
        project_id=_TEST_PROJECT_ID,
    )
    train_table_job = SimpleNamespace(config=SimpleNamespace(plan=None))
    pred_table_job = SimpleNamespace(
        job_status_report=SimpleNamespace(status=pred_table_status),
        config=SimpleNamespace(plan=None),
    )

    pquery = _pquery_driver(ml_job_backend=SimpleNamespace(), )
    pquery.get_training_job = AsyncMock(return_value=training_job)
    pquery.get_training_job_pquery_id_v2 = AsyncMock(
        return_value=PredictiveQueryID("pquerytest"))
    pquery.get_predictive_query_v2 = AsyncMock(return_value=training_pquery)
    pquery.get_generate_pred_table_job = AsyncMock(return_value=pred_table_job)
    pquery.get_training_table_generation_job = AsyncMock(
        return_value=train_table_job)
    pquery._graph_driver.get_graph_v2 = AsyncMock(return_value=(graph_def, None,
                                                                None))
    return pquery


async def _validate_bp_inputs_for_status(pred_table_status: JobStatus):
    pquery = _make_validate_bp_inputs_driver(pred_table_status)
    return await pquery.validate_batch_prediction_inputs_v2(
        ctx=None,
        training_job_id=TrainingJobID.make_random_id(),
        pred_table_job_id=GeneratePredictionTableJobID.make_random_id(),
        graph_id=GraphIDv2.compose(fpid=FPID("a" * 32)),
        predictive_query_id=None,
    )


@pytest.mark.asyncio
async def test_validate_bp_inputs_pred_table_job_status_guard():
    # A gen-pred-table job in a failed terminal state can never yield a usable
    # prediction table, so BP submit must reject it.
    failed = await _validate_bp_inputs_for_status(JobStatus.FAILED)
    assert not failed.ok
    failed_titles = [error.title for error in failed.errors]
    assert PREDICTION_TABLE_JOB_FAILED_TITLE in failed_titles
    # The guard must not surface a raw graph-snapshot 422 or the unrelated
    # training-job-incomplete error.
    assert TRAINING_JOB_INCOMPLETE_TITLE not in failed_titles

    # A still-running gen-pred-table job is valid at submit: the BP workflow
    # waits for it to finish before materializing. The guard must not reject it.
    running = await _validate_bp_inputs_for_status(JobStatus.RUNNING)
    assert running.ok
    assert PREDICTION_TABLE_JOB_FAILED_TITLE not in [
        error.title for error in running.errors
    ]
