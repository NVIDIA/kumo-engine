# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from tempfile import TemporaryDirectory

from kumo.config.explain import ColumnContributionSummary
from rest.config.training_job import (
    Metric,
    TestExplanation,
    TrainingJobResults,
)


def test_save_load_test_plain():
    test_explain = TestExplanation(
        contribution_summary=ColumnContributionSummary(
            global_column_contributions=[]))
    result = TrainingJobResults(
        total_training_time_sec=1,
        total_iterations=1,
        test_explain=test_explain,
    )

    with TemporaryDirectory() as dir:
        test_explain_file_path = os.path.join(dir, 'test_explain.json.gz')
        result.save_test_explain_to_remote(test_explain_file_path)

        assert result.test_explain is None
        assert result._test_explain_file_path == test_explain_file_path
        assert result.load_test_explain() == test_explain


def test_with_metrics():
    result = TrainingJobResults(total_training_time_sec=1, total_iterations=1,
                                _test_explain_file_path=__file__).with_metrics(
                                    [
                                        Metric("test_val", 1),
                                        Metric("train_val", 2),
                                        Metric("val_val", 3),
                                    ])
    assert result.test_metrics == [Metric("val", 1)]
    assert result.train_metrics == [Metric("val", 2)]
    assert result.val_metrics == [Metric("val", 3)]


def test_a_local_file_custom_training_table_is_refused_at_the_boundary():
    import pytest
    from kumoapi.jobs import CustomTrainingTable
    from kumoapi.source_table import LocalFileSourceTable, SnowflakeSourceTable
    from kumoapi.train import TrainingTableSpec

    from rest.config.training_job_v2 import TrainingJobInternalRequest

    spec = TrainingTableSpec(weight_col='weight')

    with pytest.raises(Exception, match='calling process'):
        TrainingJobInternalRequest(
            pquery_id='pq', custom_train_table=CustomTrainingTable(
                source_table=LocalFileSourceTable(path='/tmp/rows.parquet'),
                table_mod_spec=spec))

    request = TrainingJobInternalRequest(
        pquery_id='pq', custom_train_table=CustomTrainingTable(
            source_table=SnowflakeSourceTable(snowflake_connector_id='c',
                                              database='D', schema_name='S',
                                              table='t'), table_mod_spec=spec))

    assert request.custom_train_table is not None


def test_a_local_file_is_refused_by_name_on_the_driver_route():
    """The drivers reach ``extract_source_table_type`` directly
    (``pquery_api_driver.py:2209``, ``:2357``), not through the request model.
    """
    import pytest
    from kumoapi.source_table import LocalFileSourceTable

    from rest.utils.driver_common import extract_source_table_type

    with pytest.raises(ValueError, match='calling process'):
        extract_source_table_type(
            LocalFileSourceTable(path='/tmp/rows.parquet'))
