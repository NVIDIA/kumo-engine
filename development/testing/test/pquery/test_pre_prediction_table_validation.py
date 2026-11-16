# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pandas as pd
import pytest

# isort: off
from kumo.pquery.pquery_validators.pre_pred_table_validator import (
    PrePredTableValidator)
from kumo.pquery.pquery_validators.train_table_validator import MAX_RHS_ROWS
# isort: on


def test_validate_anchor_date_is_valid():
    validator = PrePredTableValidator()
    anchor_date = pd.Timestamp('2020-01-01')
    min_date = pd.Timestamp('2010-01-01')
    assert (isinstance(anchor_date, pd.Timestamp)
            and isinstance(min_date, pd.Timestamp))
    response = validator.validate_anchor_time(anchor_date, min_date,
                                              'target_table')
    assert response.ok


def test_validate_anchor_date_is_early():
    validator = PrePredTableValidator()
    anchor_date = pd.Timestamp('2000-01-01')
    min_date = pd.Timestamp('2010-01-01')
    assert (isinstance(anchor_date, pd.Timestamp)
            and isinstance(min_date, pd.Timestamp))
    response = validator.validate_anchor_time(anchor_date, min_date,
                                              'target_table')
    assert not response.ok
    assert (response.errors[0].title == 'Invalid Prediction Anchor Time')


@pytest.mark.parametrize('kwargs', [
    dict(
        anchor_date=pd.Timestamp('2020-01-15'),
        timeframe_step=pd.Timedelta(days=1),
        lag_timesteps=10,
        max_date=pd.Timestamp('2020-01-15'),
        min_date=pd.Timestamp('2020-01-01'),
        warning_message=None,
    ),
    dict(
        anchor_date=pd.Timestamp('2020-01-08'),
        timeframe_step=pd.Timedelta(days=1),
        lag_timesteps=10,
        max_date=pd.Timestamp('2020-01-15'),
        min_date=pd.Timestamp('2020-01-01'),
        warning_message='Older lag data is missing',
    ),
    dict(
        anchor_date=pd.Timestamp('2020-01-16'),
        timeframe_step=pd.Timedelta(days=2),
        lag_timesteps=3,
        max_date=pd.Timestamp('2020-01-15'),
        min_date=pd.Timestamp('2020-01-01'),
        warning_message=None,
    ),
    dict(
        anchor_date=pd.Timestamp('2020-01-19'),
        timeframe_step=pd.Timedelta(days=2),
        lag_timesteps=3,
        max_date=pd.Timestamp('2020-01-15'),
        min_date=pd.Timestamp('2020-01-01'),
        warning_message='Latest lag data is not present',
    ),
])
def test_validate_anchor_time_for_forecasting(kwargs):
    validator = PrePredTableValidator()
    response = validator.validate_anchor_time_forecasting(
        anchor_date=kwargs['anchor_date'],
        timeframe_step=kwargs['timeframe_step'],
        lag_timesteps=kwargs['lag_timesteps'],
        max_date=kwargs['max_date'],
        min_date=kwargs['min_date'],
        target_table_name='target_table',
    )
    if kwargs['warning_message'] is None:
        assert response.ok
    else:
        assert kwargs['warning_message'] in response.errors[0].title


def _mock_graph_with_table(table_name, num_rows):
    """Create a mock Graph where graph.table(table_name) returns a Table-like
    object with the given num_rows."""
    mock_table = MagicMock(spec=['num_rows'])
    mock_table.num_rows.return_value = num_rows
    # Make isinstance(mock_table, Table) return True:
    mock_table.__class__ = __import__(
        'kumo.graph.table', fromlist=['Table']).Table
    mock_graph = MagicMock()
    mock_graph.table.return_value = mock_table
    return mock_graph


@pytest.mark.parametrize(
    'num_rows,is_child_job,expect_ok,expect_errors,expect_warnings', [
        (1_000_000, False, True, 0, 0),       # under limit
        (MAX_RHS_ROWS + 1, False, True, 0, 1),  # over limit, warning only
        (MAX_RHS_ROWS + 1, True, False, 1, 0),  # over limit, child job → error
    ])
def test_validate_num_rhs_rows(num_rows, is_child_job, expect_ok,
                               expect_errors, expect_warnings):
    validator = PrePredTableValidator()
    graph = _mock_graph_with_table('templates', num_rows=num_rows)
    response = validator.validate_num_rhs_rows(
        target_table_name='templates', graph=graph,
        is_child_job=is_child_job)
    assert response.ok == expect_ok
    assert len(response.errors) == expect_errors
    assert len(response.warnings) == expect_warnings
