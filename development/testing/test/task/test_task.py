# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os.path as osp
import shutil
from dataclasses import asdict
from test.fixtures.dg_indexed_graph import materialize_task_dg
from unittest.mock import patch

from kumo_core.types import TaskType
from kumoapi.encoder import MultiCategorical

from kumo.config.task import TimePeriod
from kumo.connector.pandas_connector import PandasConnector
from kumo.connector.parquet_connector import ParquetConnector
from kumo.home import get_home_dir
from kumo.pquery.predictive_query import PredictiveQueryV2
from kumo.task.split import RandomSplit
from kumo.task.task import Task
from kumo.testing.pquery import get_validated_query

TEST_GRAPH_NAME = 'FINANCIAL'
TEST_TASK_NAME = 'kumo_test_task'
TEST_LINK_PREDICTION_TASK_NAME = 'TEST_LINK_PREDICTION_TASK'


def teardown_function():
    folder = osp.join(get_home_dir(), 'graphs')
    shutil.rmtree(folder, ignore_errors=True)

    folder = osp.join(get_home_dir(), 'tasks')
    shutil.rmtree(folder, ignore_errors=True)


def test_save_task_pandas_connector(create_task, tmp_path):
    task = create_task('SYNTHETIC', 'multiclass_classification', temporal=True)
    task_name = task.name

    # Current fixtures use PandasConnector for creating a task:
    assert isinstance(task.train_table.connector, PandasConnector)

    # Save/load round-trip through an explicit temp dir (no dependence on the
    # process-global home dir).
    save_dir = str(tmp_path)

    # Test saving task before materialization - to test training table
    # PandasConnector persistence before materialization:
    task.save(destination=save_dir)

    # Load the task again:
    loaded_task = Task.load(task_name, save_dir)

    assert not task.graph.has_data
    assert not task.train_table.has_data
    assert isinstance(loaded_task.train_table.connector, ParquetConnector)
    assert asdict(task.config()) == asdict(loaded_task.config())


def test_multilabel_ranking(create_task):
    task: Task = create_task('SYNTHETIC', 'multilabel_ranking', temporal=True)
    assert task.task_type == TaskType.MULTILABEL_RANKING
    task.infer_metadata()
    task.infer_encoders()
    task.validate(raise_on_error=True)

    assert task.task_type.is_multilabel
    assert len(task.get_default_metrics()) == 12
    assert isinstance(task.train_table.target_col.encoder, MultiCategorical)


@patch('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', new=True)
def test_task_link_prediction(dg_indexed_static_link_pred_graph, tmp_path):
    graph, index_path = dg_indexed_static_link_pred_graph(
        num_entities=3, set_entity_time_col=False, set_trans_time_col=False)
    graph.infer_metadata()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_query = get_validated_query(graph, query)
    pquery = PredictiveQueryV2(
        name=TEST_LINK_PREDICTION_TASK_NAME,
        graph=graph,
        validated_predictive_query=validated_query,
        split=RandomSplit([0.8, 0.1, 0.1]),
    )

    # materialize_task_dg threads an explicit destination (tmp_path); it does
    # not touch the process-global home dir.
    task = materialize_task_dg(pquery, index_path, tmp_path)

    assert task.train_table.has_data
    assert task.task_type == TaskType.STATIC_LINK_PREDICTION
    assert not task.is_temporal

    fkeys = task.train_table.fkeys
    assert [fk.name for fk in fkeys] == ['ENTITY', 'TARGET']
    entity_fk, target_fk = fkeys
    assert entity_fk.dst_table.split('.')[-1] == 'USERS'
    assert entity_fk.dst_key == 'USER_ID'
    assert target_fk.dst_table.split('.')[-1] == 'PRODUCTS'
    assert target_fk.dst_key == 'PRODUCT_ID'

    assert task.train_table.time_col is None

    # This test pins config persistence: Task.load is config-only (no data
    # reload), which also sidesteps the static-LP DG-cutover gap (a full data
    # reload raises pandas KeyError reindexing the DG-native fkey-cols
    # artifact).
    #
    # Save/load round-trip through an explicit temp dir (no dependence on the
    # process-global home dir).
    save_dir = str(tmp_path / 'saved_task')
    task.save(destination=save_dir)
    new_task = Task.load(TEST_LINK_PREDICTION_TASK_NAME, save_dir)
    assert asdict(new_task.config()) == asdict(task.config())
    assert new_task.task_type == TaskType.STATIC_LINK_PREDICTION
    assert [fk.name
            for fk in new_task.train_table.fkeys] == ['ENTITY', 'TARGET']


def test_task_is_temporal(create_task):
    task = create_task(TEST_GRAPH_NAME, 'multiclass_classification',
                       temporal=True)
    assert task.is_temporal

    task.train_table.set_time_col(None)
    assert not task.is_temporal

    task.graph['ACCOUNT'].set_time_col(None)
    task.graph['LOAN'].set_time_col(None)
    task.graph['TRANS'].set_time_col(None)
    assert not task.is_temporal


def test_parse_datetime_str_clamps_out_of_range():
    # Serialized horizon bounds can land outside Python's datetime year range
    # (1-9999). A degenerate year-0 horizon clamps to ``datetime.min``; a
    # far-future sentinel (year 9918, valid datetime but overflows pandas) is
    # preserved; in-range ISO and non-ISO strings parse normally.
    from datetime import datetime

    from kumo.config.task import _parse_datetime_str

    assert _parse_datetime_str('0000-09-10T00:00:00') == datetime.min
    assert _parse_datetime_str('9918-07-23T00:00:00') == datetime(9918, 7, 23)
    assert _parse_datetime_str('2020-01-01T00:00:00') == datetime(2020, 1, 1)
    assert _parse_datetime_str('2020-01-01') == datetime(2020, 1, 1)
    assert _parse_datetime_str('Jan 5 2021') == datetime(2021, 1, 5)
    assert _parse_datetime_str('10000-01-01T00:00:00') == datetime.max


def test_time_period_roundtrips_degenerate_horizon():
    # Reproduces the deserialization failure: a TimePeriod whose serialized
    # ``end`` is the degenerate year-0 sentinel must round-trip through
    # pydantic-v2 validation (TypeAdapter.validate_python) instead of raising
    # OutOfBoundsDatetime. Year-0 was already degenerate, so it clamps to
    # ``datetime.min``; the far-future sentinel is preserved.
    from datetime import datetime

    from io_utils.json import from_json_dict, to_json_dict

    period = TimePeriod(start=datetime(2020, 1, 1), end=datetime(2020, 6, 1),
                        is_pseudo=False)
    data = to_json_dict(period)

    data['end'] = '0000-09-10T00:00:00'
    assert from_json_dict(data, TimePeriod).end == datetime.min

    data['end'] = '9918-07-23T00:00:00'
    assert from_json_dict(data, TimePeriod).end == datetime(9918, 7, 23)
