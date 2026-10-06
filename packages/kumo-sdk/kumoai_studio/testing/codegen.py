# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

from kumoai_studio.pquery import PredictiveQuery, TrainingTableJob


def assert_code_equal(code1: str, code2: str):
    """Assert code equality ignoring whitespace."""
    assert ast.dump(ast.parse(code1)) == ast.dump(ast.parse(code2))


def _assert_table_equal(table1, table2):
    """Assert table equality."""
    assert (table1.primary_key is None
            and table2.primary_key is None) or (table1.primary_key
                                                == table2.primary_key)

    assert table1.source_name == table2.source_name
    assert table1.time_column == table2.time_column
    assert table1.end_time_column == table2.end_time_column

    table1_cols = {col.name: col for col in table1.columns}
    table2_cols = {col.name: col for col in table2.columns}
    assert len(table1_cols) == len(table2_cols)
    for col_name, col1 in table1_cols.items():
        col2 = table2_cols[col_name]
        assert _compare_columns(col1, col2)


def _assert_graph_equal(graph1, graph2):
    """Assert graph equality."""
    assert len(graph1.tables) == len(graph2.tables)
    for table_name in graph1.tables:
        assert table_name in graph2.tables
        _assert_table_equal(graph1.tables[table_name],
                            graph2.tables[table_name])

    assert len(graph1.edges) == len(graph2.edges)
    for edge in graph1.edges:
        assert edge in graph2.edges


def _assert_pquery_equal(pquery1, pquery2):
    """Assert predictive query equality."""
    _assert_graph_equal(pquery1.graph, pquery2.graph)
    assert pquery1.query == pquery2.query


def _compare_columns(col1, col2):
    """Compare column equality."""
    return (col1.name == col2.name and col1.dtype == col2.dtype
            and col1.stype == col2.stype)


def _assert_object_equal(obj1, obj2):
    """Assert object equality."""
    obj1_dict = obj1.__dict__
    obj2_dict = obj2.__dict__
    for key in obj1_dict:
        assert key in obj2_dict
        assert obj1_dict[key] == obj2_dict[key]
    for key in obj2_dict:
        assert key in obj1_dict
        assert obj2_dict[key] == obj1_dict[key]
    assert len(obj1_dict) == len(obj2_dict)


def _assert_train_table_job_equal(train_table_job1, train_table_job2):
    """Assert training table job equality."""
    job1_config = train_table_job1.load_config()
    job2_config = train_table_job2.load_config()
    job1_pquery_id = job1_config.pquery_id
    job2_pquery_id = job2_config.pquery_id
    job1_pquery = PredictiveQuery.load(job1_pquery_id)
    job2_pquery = PredictiveQuery.load(job2_pquery_id)
    job1_plan = job1_config.plan
    job2_plan = job2_config.plan
    _assert_pquery_equal(job1_pquery, job2_pquery)
    _assert_object_equal(job1_plan, job2_plan)


def _assert_train_job_equal(train_job1, train_job2):
    """Assert training job equality."""
    job1_config = train_job1.load_config()
    job2_config = train_job2.load_config()
    job1_train_table_job_id = job1_config.train_table_job_id
    job2_train_table_job_id = job2_config.train_table_job_id
    job1_train_table_job = TrainingTableJob(job1_train_table_job_id)
    job2_train_table_job = TrainingTableJob(job2_train_table_job_id)
    _assert_train_table_job_equal(job1_train_table_job, job2_train_table_job)

    job1_model_plan = job1_config.model_plan
    job2_model_plan = job2_config.model_plan
    _assert_object_equal(job1_model_plan, job2_model_plan)


class MockSDKObject:
    """Mock SDK object for testing."""
    def __init__(self, name: str = "test", parents=None, imports=None):
        self.name = name
        self.parents = parents or []
        self.imports = imports or []

    def __hash__(self):
        return id(self)
