# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import pytest
from kumoapi.model_plan import NumNeighborsConfig

import kumoai_studio as kumo
from kumoai_studio import Dtype, Stype
from kumoai_studio.codegen.generate import generate_code
from kumoai_studio.connector import S3Connector
from kumoai_studio.graph import Graph, Table
from kumoai_studio.pquery import PredictiveQuery
from kumoai_studio.pquery.training_table import TrainingTableJob
from kumoai_studio.testing.codegen import _assert_train_job_equal
from kumoai_studio.trainer import Trainer, TrainingJob

logger = logging.getLogger(__name__)

# Global variable for S3 bucket used in tests
TEST_S3_URI = "s3://example-bucket/UI2SDKtests/"
# TEST_S3_URI = "s3://kumo-public-datasets/codegentests/"


class E2EJobManager:
    def __init__(self):
        self.setup_jobs = {}
        self.generated_jobs = {}

    def track_setup_job(self, job):
        self.setup_jobs[job.id] = job
        logger.info(f"Tracking setup job: {job.id}")

    def track_generated_jobs(self, exec_namespace):
        for var_name, obj in exec_namespace.items():
            if isinstance(obj, (TrainingJob, TrainingTableJob)):
                self.generated_jobs[obj.id] = obj
                logger.info(f"Tracking generated job: {var_name} -> {obj.id}")

    def cleanup_setup_jobs(self):
        self._cleanup_jobs(self.setup_jobs, "setup")

    def cleanup_generated_jobs(self):
        self._cleanup_jobs(self.generated_jobs, "generated")

    def cleanup_all_jobs(self):
        self.cleanup_setup_jobs()
        self.cleanup_generated_jobs()

    def _cleanup_jobs(self, job_dict, job_type):
        for job_id, job in job_dict.items():
            try:
                if hasattr(job, 'cancel'):
                    job.cancel()
                    logger.info(f"Cancelled {job_type} job: {job_id}")
                else:
                    logger.warning(f"Job {job_id} has no cancel method")
            except Exception as e:
                logger.error(f"Failed to cancel {job_type} job {job_id}: {e}")
        job_dict.clear()


_job_manager = E2EJobManager()

# Global variable for S3 bucket used in tests
TEST_S3_URI = "s3://kumo-public-datasets/codegentests/"

# Global configs updated by setup_test_objects
TRAIN_TABLE_JOB_CONFIG: dict[str, Any] = {
    'id': None,
    'plan': None,
}
TRAIN_JOB_CONFIG: dict[str, Any] = {
    'id': None,
    'plan': None,
}


def setup_test_objects():
    """Create test objects for integration testing."""
    s3_uri = TEST_S3_URI
    available_tables = ["articles", "transactions", "customers"]
    pk_time_map = {
        "articles": ("article_id", None),
        "transactions": (None, "t_dat"),
        "customers": ("customer_id", None),
    }

    s3_connector = S3Connector(name='codegen_source', root_dir=s3_uri)

    graph_tables = {}
    for table_name in available_tables:
        table = Table.from_source_table(
            source_table=s3_connector[table_name],
            primary_key=pk_time_map[table_name][0],
            time_column=pk_time_map[table_name][1],
        )

        if table_name == "articles":
            for i, col in enumerate(table.columns):
                if col.dtype == Dtype.int64 and col.name != "article_id":
                    table.columns[i].dtype = Dtype.float64
                    break
        if table_name == "customers":
            assert table.primary_key is not None
            for i, col in enumerate(table.columns):
                if (col.stype == Stype.ID and col.dtype == "string"
                        and col.name != table.primary_key.name):
                    table.columns[i].stype = Stype.categorical
                    break
        table.save(f"codegen_test_{table_name}")
        graph_tables[f"codegen_test_{table_name}"] = table

    graph_edges = [
        dict(src_table="codegen_test_transactions", fkey="customer_id",
             dst_table="codegen_test_customers"),
        dict(src_table="codegen_test_transactions", fkey="article_id",
             dst_table="codegen_test_articles")
    ]

    graph = Graph(tables=graph_tables, edges=graph_edges)
    graph.validate()
    graph.save("codegen_test_graph")

    # PredictiveQuery
    query_string = """PREDICT COUNT(codegen_test_transactions.*
WHERE codegen_test_transactions.price > 0.1, 0, 5, days)
                            FOR EACH codegen_test_customers.customer_id"""  # noqa
    pquery = PredictiveQuery(graph=graph, query=query_string)
    pquery.save("codegen_test_pquery")

    # TrainingTablePlan
    train_table_plan = pquery.suggest_training_table_plan()
    train_table_plan.split = 'TemporalSplit(ratio=[0.8, 0.1, 0.1])'

    #  TrainingTableJob
    train_table_job = pquery.generate_training_table(train_table_plan,
                                                     non_blocking=True)
    # ModelPlan
    model_plan = pquery.suggest_model_plan()
    num_neighbors = model_plan.neighbor_sampling.num_neighbors[0]
    assert isinstance(num_neighbors, NumNeighborsConfig)
    num_neighbors[0].hop1.default = 64

    # TrainingJob
    trainer = Trainer(model_plan)
    train_job = trainer.fit(graph=graph, train_table=train_table_job,
                            non_blocking=True)

    TRAIN_TABLE_JOB_CONFIG['id'] = train_table_job.id
    TRAIN_TABLE_JOB_CONFIG['plan'] = train_table_plan
    TRAIN_JOB_CONFIG['id'] = train_job.id
    TRAIN_JOB_CONFIG['plan'] = model_plan

    _job_manager.track_setup_job(train_table_job)
    _job_manager.track_setup_job(train_job)


@pytest.fixture(scope="module", autouse=True)
def init_kumo_with_cleanup(request):
    import os

    kumo_endpoint = os.getenv("KUMO_API_ENDPOINT")
    kumo_api_key = os.getenv("KUMO_API_KEY")

    if not kumo_endpoint or not kumo_api_key:
        raise RuntimeError(
            "KUMO_API_ENDPOINT and KUMO_API_KEY are required for integration "
            "tests")
    kumo.init(url=kumo_endpoint, api_key=kumo_api_key)

    setup_test_objects()
    request.addfinalizer(_job_manager.cleanup_setup_jobs)


class TestTrainingE2E:
    """Comprehensive end-to-end training pipeline integration tests.

    These tests validate the entire codegen system by testing TrainingJob
    generation, which exercises all handlers and their dependency resolution:

    TrainingJob codegen should generate code for:
    - TrainingJob (using suggest_model_plan -> trainer.fit pattern)
    - TrainingTableJob dependency (using suggest_training_table_plan ->
      generate_training_table pattern)
    - PredictiveQuery dependency
    - Graph dependency
    - Table dependencies (articles, transactions, customers)
    - S3Connector dependency
    """
    @pytest.mark.skip(
        reason="Skipping integration test, next PR will enable it")
    def test_training_job_comprehensive_e2e(self, request):
        assert TRAIN_JOB_CONFIG['id'] is not None
        input_spec = {
            "id": TRAIN_JOB_CONFIG['id'],
            "entity_class": "TrainingJob",
        }

        original_train_job = TrainingJob(TRAIN_JOB_CONFIG['id'])
        code = generate_code(input_spec=input_spec)

        logger.debug(f"generated code: {code}")
        assert "S3Connector" in code
        assert "Table.from_source_table" in code
        assert "Graph(" in code
        assert "PredictiveQuery(" in code
        assert "suggest_training_table_plan" in code
        assert "suggest_model_plan" in code
        assert "trainer.fit(" in code

        exec_namespace = {}
        exec(code, exec_namespace)

        _job_manager.track_generated_jobs(exec_namespace)
        request.addfinalizer(_job_manager.cleanup_generated_jobs)

        recreated_train_job = None
        for key, value in exec_namespace.items():
            if isinstance(value, TrainingJob):
                recreated_train_job = value
                break

        if recreated_train_job is None:
            raise ValueError("No training job found in exec_namespace")

        _assert_train_job_equal(recreated_train_job, original_train_job)
        logger.info("training job codegen test passed")
