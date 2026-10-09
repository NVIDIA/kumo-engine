# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import os.path as osp
import pathlib
import shutil
from types import SimpleNamespace
from typing import List, Optional

import pandas as pd
import pytest
from kumo_core.types import TaskType
from kumoapi.encoder import Datetime, Index, MaxLogNumerical
from kumoapi.typing import TimeUnit
from kumopql.parser import PQLParser
from kumopql.validator import PredictiveQueryValidator

from io_utils import fsutil
from kumo.connector.parquet_connector import ParquetConnector
from kumo.graph.graph import Graph
from kumo.home import get_home_dir
from kumo.pquery.predictive_query import PQuerySettings, PredictiveQueryV2
from kumo.pquery.utils import validator_kwargs
from kumo.task.split import DateOffsetSplit, RandomLinkSplit, TimeRangeSplit
from kumo.task.splits.time_split import TimeSplit
from kumo.testing.pquery import get_validated_query
from kumo.validation import ValidationException

TEST_TASK_NAME = 'SUM_AMOUNT_class'
TEST_TASK_OUTPUT_DIR = osp.join(osp.dirname(__file__), ".test_files")


def setup_function():
    fsutil.makedirs(TEST_TASK_OUTPUT_DIR, exist_ok=True)


def teardown_function():
    # Saved queries:
    folder = osp.join(get_home_dir(), 'queries', TEST_TASK_NAME)
    shutil.rmtree(folder, ignore_errors=True)

    # Saved tasks configs and graph configs
    folder = osp.join(get_home_dir(), 'tasks', TEST_TASK_NAME)
    shutil.rmtree(folder, ignore_errors=True)
    folder = osp.join(get_home_dir(), 'graphs', TEST_TASK_NAME)
    shutil.rmtree(folder, ignore_errors=True)

    # Saved tasks:
    shutil.rmtree(TEST_TASK_OUTPUT_DIR, ignore_errors=True)


@pytest.mark.parametrize(
    'query,expected_task_type,expected_reason',
    [
        (
            'PREDICT USERS.AGE FOR EACH USERS.USER_ID',
            TaskType.REGRESSION,
            'The target clause is a column in your graph and it has a '
            'Numerical semantic type.',
        ),
        (
            'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) '
            'FOR EACH USERS.USER_ID',
            TaskType.REGRESSION,
            'The target clause is a numerical aggregation (SUM, COUNT, etc).',
        ),
        (
            'PREDICT COUNT(TRANSACTIONS.*, 0, 2) FOR EACH USERS.USER_ID',
            TaskType.REGRESSION,
            'The target clause is a numerical COUNT aggregation.',
        ),
        (
            'PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 2) CLASSIFY '
            'FOR EACH USERS.USER_ID',
            TaskType.MULTILABEL_CLASSIFICATION,
            'The target clause is a LIST_DISTINCT aggregation on a column '
            'that acts as a foreign key and the CLASSIFY problem type was '
            'selected.',
        ),
        (
            'PREDICT LIST_DISTINCT(TRANSACTIONS.BANK_ID, 0, 2) RANK TOP 10 '
            'FOR EACH USERS.USER_ID',
            TaskType.TEMPORAL_LINK_PREDICTION,
            'The target clause is a LIST_DISTINCT aggregation on a column '
            'that acts as a foreign key and the RANK problem type was '
            'selected.',
        ),
        (
            'PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) > 30 '
            'FOR EACH USERS.USER_ID',
            TaskType.BINARY_CLASSIFICATION,
            'The target clause is a condition or logical operation, and is '
            'an ID or Categorical Boolean',
        ),
    ],
)
def test_task_type_reason_contract(
    query,
    expected_task_type,
    expected_reason,
    create_node_pred_graph,
    tempdir,
):
    from kumo.pquery.generator_utils import get_task_type

    graph = create_node_pred_graph('test_graph', tempdir.name)
    result = get_task_type(graph, get_validated_query(graph, query))

    assert result.task_type == expected_task_type
    assert result.reason == expected_reason


@pytest.fixture(scope='function')
def basic_test_graph(tmp_path: pathlib.Path) -> Graph:
    temp_path: str = tmp_path.resolve().as_posix()
    entity_df = pd.DataFrame({'USER_ID': range(2), 'AGE': [17, 25]})
    dates = pd.Series(pd.date_range(start='1/1/2018', end='1/30/2018'))
    dates = pd.concat([dates, dates], axis=0, ignore_index=True)
    types = ['A', 'B'] * 10 + ['A', 'C'] * 10 + ['B', 'C'] * 10
    target_df = pd.DataFrame({
        'TRANS_ID': range(len(dates)),
        'USER_ID': [0] * 30 + [1] * 30,
        'AMOUNT': [1.0] * len(dates),
        'DATE': dates,
        'TYPE': types
    })
    graph = Graph('test')

    def persist_dfs(dfs: List[pd.DataFrame], names: List[str]):
        for i, df in enumerate(dfs):
            path = osp.join(temp_path, 'pandas_converted', names[i])
            os.makedirs(path)
            df.to_parquet(osp.join(path, 'data.parquet'))

    persist_dfs([entity_df, target_df], ['USERS', 'TRANS'])

    graph = Graph()
    data_source = ParquetConnector("pandas_converted",
                                   osp.join(temp_path, 'pandas_converted'))
    graph.connect(data_source)

    graph.add('USERS', pkey='USER_ID')
    graph.add('TRANS', pkey='TRANS_ID')
    graph.link('TRANS', 'USER_ID', 'USERS', 'USER_ID')
    graph.infer_metadata()
    # graph.infer_encoders()
    graph['USERS', 'AGE'].encoder = MaxLogNumerical()
    graph['TRANS', 'AMOUNT'].encoder = Index(min_occ=2)
    graph['TRANS', 'DATE'].encoder = Datetime()
    graph['TRANS', 'TYPE'].encoder = Index(min_occ=2)
    graph['TRANS'].set_time_col('DATE')
    return graph


@pytest.mark.parametrize('query, expected_errors', [
    ('PREDICT USERS.USER_ID FOR EACH USERS.USER_ID', ['row 1, column 8']),
    ('PREDICT USERS.USER_ID > 0 FOR EACH USERS.USER_ID', ['row 1, column 8']),
    ('PREDICT (USERS.USER_ID > 0) OR (USERS.USER_ID = 0) '
     'FOR EACH USERS.USER_ID WHERE USERS.USER_ID > 1',
     ['row 1, column 32', 'row 1, column 9']),
])
def test_predictive_query_no_predict_entity(
    basic_test_graph: Graph,
    query: str,
    expected_errors: List[str],
):
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(basic_test_graph))
    parsed_query = parser.to_parsed_predictive_query(query)
    _, response = validator.validate_predictive_query(parsed_query)
    assert len(response.errors) == len(expected_errors)
    # Order the errors by message in order to compare with expected errors
    response.errors.sort(key=lambda x: x.message)
    for error, expected_error in zip(response.errors, expected_errors):
        assert error.title == 'Predicting entity primary key column'
        assert expected_error in error.message


@pytest.mark.parametrize('split', [
    DateOffsetSplit(offsets=[-7, 0]),
    TimeRangeSplit([
        (pd.Timestamp('1990-01-01'), pd.Timestamp('1995-01-01')),
        (pd.Timestamp('1995-01-01'), pd.Timestamp('1995-06-01')),
        (pd.Timestamp('1995-06-01'), pd.Timestamp('1996-01-01')),
    ])
])
def test_predictive_query_set_split(split, basic_test_graph, tempdir):
    graph = copy.deepcopy(basic_test_graph)
    query = 'PREDICT COUNT(TRANS.*, 2, 7, hours) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)

    # node classification
    pquery = PredictiveQueryV2(
        name='PRED_TYPE',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        timeframe_step=10,
        split=split,
    )
    assert pquery.split.unit == TimeUnit.HOURS
    assert pquery.split.timeframe == 5
    assert pquery.split.anchor_start_offset == 2
    assert pquery.split.timeframe_step == 10
    assert not pquery.split.flexible_boundaries


@pytest.mark.parametrize(
    'task_type,expected_flexible_boundaries',
    [
        (TaskType.REGRESSION, False),
        (TaskType.FORECASTING, True),
    ],
)
def test_default_date_offset_split_preserves_flexible_boundaries(
        basic_test_graph, task_type, expected_flexible_boundaries):
    query = 'PREDICT COUNT(TRANS.*, 2, 7, hours) FOR EACH USERS.USER_ID'
    pquery = PredictiveQueryV2(
        name='default_split_flexible_boundaries',
        graph=basic_test_graph,
        validated_predictive_query=get_validated_query(basic_test_graph, query),
    )

    split = pquery._get_default_split(task_type)

    assert isinstance(split, DateOffsetSplit)
    assert split.flexible_boundaries is expected_flexible_boundaries


def test_col_not_in_graph(basic_test_graph):
    parser = PQLParser()
    validator = PredictiveQueryValidator(**validator_kwargs(basic_test_graph))
    # All columns used in the query must be in the graph.
    query = 'PREDICT TRANS.TYPE FOR EACH USERS.USERSS_ID'
    parsed_query = parser.to_parsed_predictive_query(query)
    _, response = validator.validate_predictive_query(parsed_query)
    assert not response.ok
    assert "'USERSS_ID' does not exist in table 'USERS'" in response.errors[
        0].message

    query = 'PREDICT TRANS.TYPEE FOR EACH TRANS.TRANS_ID'
    parsed_query = parser.to_parsed_predictive_query(query)
    _, response = validator.validate_predictive_query(parsed_query)
    assert not response.ok
    assert "'TYPEE' does not exist in table 'TRANS'" in response.errors[
        0].message

    query = 'PREDICT TRANS.TYPE FOR EACH TRANS.TRANS_ID WHERE USERS.AGESS > 21'
    parsed_query = parser.to_parsed_predictive_query(query)
    _, response = validator.validate_predictive_query(parsed_query)
    assert not response.ok
    assert "Column 'AGESS' does not exist in table 'USERS'" in response.errors[
        0].message

    query = (
        'PREDICT TRANS.TYPE FOR EACH TRANS.TRANS_ID WHERE TRANS.TYPEE = "A"')
    parsed_query = parser.to_parsed_predictive_query(query)
    _, response = validator.validate_predictive_query(parsed_query)
    assert not response.ok
    assert "Column 'TYPEE' does not exist" in response.errors[0].message


def test_predictive_query_serialization(basic_test_graph, tempdir):
    graph = copy.deepcopy(basic_test_graph)
    query = 'PREDICT COUNT(TRANS.*, 0, 3) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)

    # Test whether serialization works properly when `train_end_offset`
    # is `pd.DateOffset` type.
    pquery = PredictiveQueryV2(
        name='test_query_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        train_start_offset=60,
        train_end_offset=pd.DateOffset(days=29, hours=48),
    )
    pquery_load = PredictiveQueryV2.load(pquery.save())
    assert pquery_load == pquery

    # Test whether serialization works properly for general cases
    pquery = PredictiveQueryV2(
        name='test_query_v2',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        train_start_offset=60,
        train_end_offset=30,
    )
    pquery_load = PredictiveQueryV2.load(pquery.save())
    assert pquery_load == pquery


def test_predictive_query_train_end_offset(basic_test_graph, tempdir):
    graph = copy.deepcopy(basic_test_graph)
    query = 'PREDICT COUNT(TRANS.*, 2, 7, hours) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)

    # Test whether `train_end_offset` with `pd.DateOffset` type works
    pquery = PredictiveQueryV2(
        name='test_query_v1',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        train_start_offset=72,
        train_end_offset=pd.DateOffset(days=2),
    )
    validate_response = pquery.get_validate_response()
    assert validate_response.ok
    table_generator = pquery.table_generator
    assert table_generator.train_start_offset == pd.DateOffset(hours=72)
    assert table_generator.train_end_offset == pd.DateOffset(days=2)

    # Test when `train_end_offset` with integer type works
    pquery = PredictiveQueryV2(
        name='test_query_v2',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        train_start_offset=72,
        train_end_offset=24,
    )
    validate_response = pquery.get_validate_response()
    assert validate_response.ok
    table_generator = pquery.table_generator
    assert table_generator.train_start_offset == pd.DateOffset(hours=72)
    assert table_generator.train_end_offset == pd.DateOffset(hours=24)

    # Test whether the existing train end offset validation works as expected
    # when `train_end_offset` is in `pd.DateOffset` type
    pquery = PredictiveQueryV2(
        name='test_query_3',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        train_start_offset=72,
        train_end_offset=pd.DateOffset(days=4),
    )
    validate_response = pquery.get_validate_response()
    assert not validate_response.ok
    assert validate_response.errors[0].title == 'Train start/end offsets error'


def test_forecasting_aggregation_unit_validation(basic_test_graph, tempdir):
    graph = copy.deepcopy(basic_test_graph)
    query = 'PREDICT COUNT(TRANS.*, 0, 1, months) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)

    with pytest.raises(ValidationException,
                       match="Forecasting does not support month"):
        PredictiveQueryV2(
            name='test_query_v1',
            graph=graph,
            validated_predictive_query=validated_predictive_query,
            train_start_offset=72,
            train_end_offset=pd.DateOffset(days=2),
            forecast_length=2,
            lag_timesteps=2,
        )


def test_time_split_parameter_inferral(create_node_pred_graph, tempdir):
    """Test that _set_time_split correctly infers parameters from temporal
    queries.
    """
    # Create a basic TimeSplit with only boundaries defined
    split = TimeSplit(
        offset_boundaries=[(-180, -120), (-120, -60), (-60, 0)],
        strict_split_boundaries=True,
        unit=TimeUnit.DAYS,
    )

    # Create graph with test data
    graph = create_node_pred_graph(
        'test_graph',
        tempdir.name,
        add_na_booleans=True,
        add_dangling_fkeys=True,
    )

    # Create a simple temporal query
    query = 'PREDICT COUNT(TRANSACTIONS.*, 30, 60) FOR EACH USERS.USER_ID'
    validated_query = get_validated_query(graph, query)
    # Create the predictive query object with time parameters
    pquery = PredictiveQueryV2(
        name='test_time_query',
        graph=graph,
        validated_predictive_query=validated_query,
        split=split,
        start_time='1990-01-01',
        end_time='1990-01-04',
    )

    # Verify parameters were correctly inferred
    split = pquery.split
    assert split.unit == TimeUnit.DAYS
    assert split.timeframe_size == 30
    assert split.anchor_start_offset == 30
    assert split.offset_boundaries == [(-180, -120), (-120, -60), (-60, 0)]
    assert split.strict_split_boundaries is True


def test_predictive_query_time_range_validation(basic_test_graph, tempdir):
    graph = copy.deepcopy(basic_test_graph)
    query = 'PREDICT COUNT(TRANS.*, 2, 7, hours) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)

    # Test validation when using both sets of parameters
    pquery = PredictiveQueryV2(
        name='test_query_time_range_conflict',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        train_start_offset=72,
        train_end_offset=24,
        start_time='2018-01-01',
        end_time='2018-01-30',
    )
    validate_response = pquery.get_validate_response()
    assert not validate_response.ok
    assert validate_response.errors[0].title == 'Incompatible time parameters'

    # Test validation when end_time is before start_time
    pquery = PredictiveQueryV2(
        name='test_query_time_range_invalid',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        start_time='2018-01-30',
        end_time='2018-01-01',
    )
    validate_response = pquery.get_validate_response()
    assert not validate_response.ok
    assert validate_response.errors[0].title == 'Invalid timestamp range'


def test_get_default_split_with_decoupled_splits(basic_test_graph):
    """Test that get_default_split returns the correct split when
    _USE_DECOUPLED_SPLITS is True.
    """
    # Set _USE_DECOUPLED_SPLITS to True using monkeypatch
    # monkeypatch.setattr('rest.settings._USE_DECOUPLED_SPLITS', True)

    # Create a simple temporal query
    query = 'PREDICT COUNT(TRANS.*, 2, 7, hours) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(basic_test_graph, query)

    # Create the predictive query object without using PQuerySettings
    # This avoids any import issues with PQuerySettings
    settings = PQuerySettings()
    settings.use_decoupled_splits = True

    pquery = PredictiveQueryV2(
        name='test_default_split',
        graph=basic_test_graph,
        validated_predictive_query=validated_predictive_query,
        pquery_settings=settings,
    )

    # Get the default split
    split = pquery.get_default_split()

    # Verify the split is a TimeSplit with correct parameters
    assert isinstance(split, TimeSplit)
    assert split.unit == TimeUnit.HOURS
    assert split.timeframe_size == 5  # 7 - 2 = 5
    assert split.anchor_start_offset == 2


def test_default_split_for_forecasting(basic_test_graph):
    """Test that get_default_split for a forecasting task returns a TimeSplit
    with correct indices based on forecast_length.
    """
    # Create a simple forecasting query
    query = 'PREDICT SUM(TRANS.AMOUNT, 2, 7, hours) FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(basic_test_graph, query)

    # Create the predictive query object with forecast_length=3
    pquery_settings = PQuerySettings()
    pquery_settings.use_decoupled_splits = True
    pquery = PredictiveQueryV2(
        name='test_forecasting_split',
        graph=basic_test_graph,
        validated_predictive_query=validated_predictive_query,
        forecast_length=3,
        pquery_settings=pquery_settings,
    )

    # Force task type to be forecasting
    pquery._task_type = TaskType.FORECASTING

    # Get the default split
    split = pquery.get_default_split()

    # Verify the split is a TimeSplit with correct indices based on
    # forecast_length
    assert isinstance(split, TimeSplit)
    assert split.unit == TimeUnit.HOURS
    assert split.timeframe_size == 5  # 7 - 2 = 5
    assert split.anchor_start_offset == 2

    # Most importantly, check that indices are correctly set based on
    # forecast_length
    assert hasattr(split, 'indices')
    expected_indices = [('-INF', -6), (-6, -3), (-3, 0)]  # -2*3, -1*3, 0
    assert split.indices == expected_indices


def _static_link_pred_pquery(
    create_lp_graph,
    tempdir,
    *,
    list_distinct_target_is_array: Optional[bool],
    lp_bp_layout: Optional[str] = None,
    split=None,
) -> PredictiveQueryV2:
    graph = create_lp_graph(root_dir=tempdir.name, num_entities=3,
                            key_name_match=False, set_entity_time_col=False,
                            set_trans_time_col=False)
    graph['PRODUCTS'].remove_time_col()
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
             'FOR EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)
    kwargs = {}
    if list_distinct_target_is_array is not None:
        kwargs['list_distinct_target_is_array'] = (
            list_distinct_target_is_array)
    if split is not None:
        kwargs['split'] = split
    return PredictiveQueryV2(
        name=f'static_lp_shape_{list_distinct_target_is_array}',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        lp_bp_layout=lp_bp_layout,
        **kwargs,
    )


def test_link_pred_shape_generator_defaults_preserve_direct_call_contract():
    from inspect import signature

    from kumo.pquery import batch_pred_table_generator
    from kumo.pquery.train_table_generator import TrainingTableGeneratorV2

    train_default = signature(
        TrainingTableGeneratorV2.generate_table,
    ).parameters['list_distinct_target_is_array'].default
    pred_default = signature(
        batch_pred_table_generator.PredictionTableGeneratorV2.generate_table,
    ).parameters['list_distinct_target_is_array'].default

    assert train_default is None
    assert pred_default is False


def test_train_generator_omitted_shape_preserves_resolved_default():
    from kumo.pquery.train_table_generator import TrainingTableGeneratorV2

    generator = TrainingTableGeneratorV2.__new__(TrainingTableGeneratorV2)
    generator.list_distinct_target_is_array = True
    generator.stats_manager = SimpleNamespace(
        list_distinct_target_is_array=None)
    generator.train_table_validator = SimpleNamespace(
        list_distinct_target_is_array=None)

    assert generator.set_list_distinct_target_is_array(None) is True
    assert generator.list_distinct_target_is_array is True
    assert generator.stats_manager.list_distinct_target_is_array is True
    assert generator.train_table_validator.list_distinct_target_is_array is True

    assert generator.set_list_distinct_target_is_array(False) is False
    assert generator.list_distinct_target_is_array is False
    assert generator.stats_manager.list_distinct_target_is_array is False
    assert generator.train_table_validator.list_distinct_target_is_array is False


def test_static_link_pred_explicit_array_shape_uses_array_split(
    create_lp_graph,
    tempdir,
    monkeypatch,
):
    monkeypatch.setattr('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', True)

    pquery = _static_link_pred_pquery(
        create_lp_graph,
        tempdir,
        list_distinct_target_is_array=True,
    )

    assert pquery.list_distinct_target_is_array is True
    assert isinstance(pquery.get_default_split(), RandomLinkSplit)
    pquery.table_generator.stats_manager._train_table_stats = object()
    task = pquery.create_task(
        train_table_path=osp.join(tempdir.name, 'train_table'))
    assert task.list_distinct_target_is_array is True
    assert isinstance(task.train_table.split, RandomLinkSplit)
    assert task.train_table.time_col is None


def test_native_bundle_preserves_logical_name_with_physical_tt_source(
    create_lp_graph,
    tempdir,
    monkeypatch,
):
    monkeypatch.setattr('kumo.pquery.AST.ast_node.ALLOW_ARRAY_TARGETS', True)
    pquery = _static_link_pred_pquery(
        create_lp_graph,
        tempdir,
        list_distinct_target_is_array=True,
    )
    pquery.table_generator.stats_manager._train_table_stats = object()
    physical_path = osp.join(tempdir.name, 'immutable', 'bundle', 'tt.parquet')

    task = pquery.create_task(train_table_path=physical_path)

    assert task.train_table.table_name == 'train_table'
    assert task.train_table.source_name == 'tt'
    assert task.train_table.connector._get_path('tt') == physical_path


def test_static_link_pred_tt_ng_forces_array_and_link_split(
    create_lp_graph,
    tempdir,
):
    """Array-mode is the only LP mode, so a static-LP query with omitted shape
    defaults list_distinct_target_is_array=True and the default split is
    RandomLinkSplit (per-link gen) -- which is what writes the static past-visit
    exclude column, so the #158 ``exclude_target_keys not found`` crash cannot
    occur."""
    pquery = _static_link_pred_pquery(
        create_lp_graph,
        tempdir,
        list_distinct_target_is_array=None,
    )

    assert pquery.list_distinct_target_is_array is True
    assert isinstance(pquery.get_default_split(), RandomLinkSplit)


def test_static_link_pred_tt_ng_forces_array_on_deserialized_false(
    create_lp_graph,
    tempdir,
):
    """A saved/deserialized config carries an EXPLICIT
    list_distinct_target_is_array=False (not None). That False must still be
    forced to array-mode for LP (the only LP mode); otherwise the
    RandomSplit->RandomLinkSplit force never fires and static-LP silently emits
    no FK-free lp_resolution (the #158 regression)."""
    pquery = _static_link_pred_pquery(
        create_lp_graph,
        tempdir,
        list_distinct_target_is_array=False,
    )

    assert pquery.list_distinct_target_is_array is True
    assert isinstance(pquery.get_default_split(), RandomLinkSplit)


def test_default_split_for_temporal_link_pred(create_lp_graph, tempdir):
    """Test that get_default_split returns the correct TimeSplit for temporal
    link prediction tasks.
    """
    # Create a basic graph for link prediction
    graph = create_lp_graph(root_dir=tempdir.name, num_entities=3,
                            key_name_match=False)

    # Create a temporal link prediction query
    query = ('PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK, 3, 7) RANK TOP 2 FOR '
             'EACH USERS.USER_ID')
    validated_predictive_query = get_validated_query(graph, query)

    # Create the predictive query with decoupled splits enabled
    settings = PQuerySettings()
    settings.use_decoupled_splits = True

    # Case 1: timeframe_step is not set
    pquery = PredictiveQueryV2(
        name='test_temporal_link_pred_split',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        pquery_settings=settings,
    )
    assert pquery.entity_table_names == ['USERS', 'PRODUCTS']

    # Get the default split
    split = pquery.get_default_split()

    assert isinstance(split, TimeSplit)
    assert split.unit == TimeUnit.DAYS
    assert split.timeframe_size == 4  # 7 - 3 = 4
    assert split.anchor_start_offset == 3

    # Verify offset boundaries are correctly set
    assert split.offset_boundaries is not None
    assert len(split.offset_boundaries) == 3
    assert split.offset_boundaries == [('-INF', -14), (-14, -7), (-7, 0)]

    # Case 2: timeframe_step is set
    pquery = PredictiveQueryV2(
        name='test_temporal_link_pred_split',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        pquery_settings=settings,
        timeframe_step=10,
    )

    # Get the default split
    split = pquery.get_default_split()

    assert isinstance(split, TimeSplit)
    assert split.unit == TimeUnit.DAYS
    assert split.timeframe_size == 4  # 7 - 3 = 4
    assert split.anchor_start_offset == 3

    # Verify offset boundaries are correctly set
    assert split.offset_boundaries is not None
    assert len(split.offset_boundaries) == 3
    assert split.offset_boundaries == [('-INF', -17), (-17, -7), (-7, 0)]


def test_default_split_for_static_query(create_static_node_pred_graph):
    """Test that get_default_split_v2 returns the correct split for a static
    query.
    """
    # Create a graph with static data
    graph = create_static_node_pred_graph()

    # Create a simple static query
    query = 'PREDICT USERS.PAYMENTS FOR EACH USERS.USER_ID'
    validated_predictive_query = get_validated_query(graph, query)

    # Create the predictive query object with decoupled splits enabled
    settings = PQuerySettings()
    settings.use_decoupled_splits = True

    pquery = PredictiveQueryV2(
        name='test_static_default_split',
        graph=graph,
        validated_predictive_query=validated_predictive_query,
        pquery_settings=settings,
    )

    # Get the default split
    split = pquery.get_default_split()

    # Verify the split is a TimeSplit with fractions
    assert isinstance(split, TimeSplit)
    assert hasattr(split, 'fractions')
    assert split.fractions == [0.8, 0.1, 0.1]


class TestParseQueryStringV2ForecastBootstrap:
    """Tests for parse_query_string_v2 FORECAST syntax and bootstrap
    lag_timesteps behavior."""
    def test_forecast_syntax_sets_forecast_length(
        self,
        create_node_pred_graph,
        tempdir,
    ):
        """FORECAST 3 TIMEFRAMES should set forecast_length=3."""
        from kumo.pquery.parser import parse_query_string_v2

        graph = create_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH USERS.USER_ID")

        config = parse_query_string_v2('test', graph, query)

        assert config.forecast_length == 3

    def test_forecast_syntax_sets_bootstrap_lag_timesteps(
            self, create_node_pred_graph, tempdir):
        """FORECAST syntax should set lag_timesteps=1 as bootstrap default
        so the pquery can be constructed before model plan is consulted."""
        from kumo.pquery.parser import parse_query_string_v2

        graph = create_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH USERS.USER_ID")

        config = parse_query_string_v2('test', graph, query)

        assert config.lag_timesteps == 1

    def test_no_forecast_syntax_leaves_defaults(
        self,
        create_node_pred_graph,
        tempdir,
    ):
        """Without FORECAST clause, forecast_length and lag_timesteps
        should be None."""
        from kumo.pquery.parser import parse_query_string_v2

        graph = create_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) "
                 "FOR EACH USERS.USER_ID")

        config = parse_query_string_v2('test', graph, query)

        assert config.forecast_length is None
        assert config.lag_timesteps is None

    def test_forecast_syntax_model_plan_and_task_type(
        self,
        create_node_pred_graph,
        tempdir,
    ):
        """End-to-end: FORECAST keyword in PQL → forecast_length set in
        model plan, task type is FORECASTING, and defaults are correct."""
        from kumo_core.model_plan import ModelPlan
        from kumoapi.model_plan import RunMode

        from kumo.pquery.generator_utils import get_query_type, get_task_type
        from kumo.pquery.parser import parse_query_string_v2

        graph = create_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(TRANSACTIONS.AMOUNT, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH USERS.USER_ID")

        # Step 1: Parser extracts forecast_length from FORECAST syntax
        config = parse_query_string_v2('test', graph, query)
        assert config.forecast_length == 3

        # Step 2: Task type resolves to FORECASTING
        task_type_result = get_task_type(
            graph,
            config.validated_predictive_query,
            forecast_length=config.forecast_length,
        )
        assert task_type_result.task_type == TaskType.FORECASTING

        # Step 3: Model plan from_default has correct forecasting defaults
        query_type = get_query_type(config.validated_predictive_query)
        model_plan = ModelPlan.from_default(
            task_type=task_type_result.task_type,
            query_type=query_type,
            has_train_table_time_col=True,
            run_mode=RunMode.NORMAL,
        )
        ttg = model_plan.training_table_generation
        assert ttg.lag_timesteps == 1
        assert ttg.year_over_year is False

        # Step 4: forecast_length is populated on model plan from query
        if config.is_forecasting:
            ttg.forecast_length = config.forecast_length
        assert ttg.forecast_length == 3


class TestParseQueryStringV2SyntaxError:
    """A syntactically malformed PQL is user input, so parse_query_string_v2
    must raise the allowlisted ValidationException (which the data-backend
    surfaces) instead of the parser's bare ValueError (which the data-backend
    allowlist masks as a generic "contact Kumo support" 500). Regression guard
    for finding 018."""
    @pytest.mark.parametrize('bad_query', [
        'PREDICT FOR EACH',
        'this is not a query',
        'PREDICT COUNT( FOR EACH USERS.USER_ID',
    ])
    def test_syntax_error_raises_validation_exception_not_value_error(
        self,
        create_node_pred_graph,
        tempdir,
        bad_query,
    ):
        from kumo.pquery.parser import parse_query_string_v2

        graph = create_node_pred_graph('test_graph', tempdir.name)
        with pytest.raises(ValidationException) as exc_info:
            parse_query_string_v2('test', graph, bad_query)

        # Anti-regression: pre-fix the parser's bare ValueError propagates
        # uncaught (ValidationException is a PublicException, not a subclass of
        # ValueError), so pytest.raises(ValidationException) would not catch it.
        assert not isinstance(exc_info.value, ValueError)
        # The parser's actionable diagnostic is preserved, not discarded.
        assert 'Encountered the following issues during parsing' in str(
            exc_info.value)


@pytest.mark.parametrize('split_expression', [
    'RandomSplit([0.6, 0.3, 0.1], seed=17)',
    'TemporalSplit([0.6, 0.3, 0.1])',
])
def test_explicit_static_link_split_preserves_hosted_normalization(
        create_lp_graph, tempdir, split_expression):
    from kumo.task.splits.parsing import get_split

    pquery = _static_link_pred_pquery(create_lp_graph, tempdir,
                                      list_distinct_target_is_array=True,
                                      split=get_split(split_expression))
    assert type(pquery.split) is RandomLinkSplit
    assert pquery.split.ratio == [0.6, 0.3, 0.1]
    assert not hasattr(pquery.split, 'seed')


@pytest.mark.parametrize('entity_time, query, accepted', [
    (True, 'PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
     'FOR EACH USERS.USER_ID', False),
    (False, 'PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK) RANK TOP 2 '
     'FOR EACH USERS.USER_ID', True),
    (True, 'PREDICT LIST_DISTINCT(TRANS.PRODUCT_ID_FK, 0, 7) RANK TOP 2 '
     'FOR EACH USERS.USER_ID', True),
])
def test_hosted_query_validation_owns_the_static_link_time_rule(
        create_lp_graph, tempdir, entity_time, query, accepted):
    from kumo.pquery.parser import parse_query_string_v2

    graph = create_lp_graph(root_dir=tempdir.name, num_entities=3,
                            key_name_match=False,
                            set_entity_time_col=entity_time,
                            set_trans_time_col=', 0, 7)' in query)
    graph['PRODUCTS'].remove_time_col()
    if accepted:
        parse_query_string_v2('test', graph, query)
        return
    with pytest.raises(ValidationException, match='USERS.DATE'):
        parse_query_string_v2('test', graph, query)
