# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.types import TaskType
from kumoapi.pquery import QueryType

from kumo.validation import ValidationException
from rest.utils.pquery import (
    _FORECASTING_TASK_TYPES,
    forecasting_pass_through_plan,
    make_predictive_config_with_graph,
    resolve_train_table_options,
)


def _make_model_plan_yaml(**overrides):
    """Build a minimal model plan YAML with forecast-related fields."""
    lines = ["training_table_generation:"]
    for key, value in overrides.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _make_config(name, query, yaml_repr, graph):
    """Mirror the production callers: resolve the model plan YAML into train
    table options, then hand those plus the were-any-supplied flag to the
    config builder."""
    train_options, train_options_provided = resolve_train_table_options(
        yaml_repr)
    return make_predictive_config_with_graph(
        name, query, train_options, graph,
        train_options_provided=train_options_provided)


class TestForecastLengthConflictDetection:
    """Tests for conflict detection between PQL FORECAST syntax and
    model plan forecast_length in make_predictive_config_with_graph."""
    def test_conflict_raises_when_values_differ(self,
                                                create_tiny_node_pred_graph,
                                                tempdir):
        """PQL says FORECAST 3, model plan says forecast_length: 5 → error."""
        graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(actions.amount, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH entity.entity_id")
        yaml = _make_model_plan_yaml(forecast_length=5)

        with pytest.raises(ValidationException) as exc_info:
            _make_config('test', query, yaml, graph)

        assert 'Conflicting forecast_length' in exc_info.value._title
        assert 'forecast_length=5' in exc_info.value._message
        assert 'FORECAST 3 TIMEFRAMES' in exc_info.value._message

    def test_no_conflict_when_values_match(self, create_tiny_node_pred_graph,
                                           tempdir):
        """PQL says FORECAST 3, model plan says forecast_length: 3 → ok."""
        graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(actions.amount, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH entity.entity_id")
        yaml = _make_model_plan_yaml(forecast_length=3)

        config = _make_config('test', query, yaml, graph)

        assert config.forecast_length == 3

    def test_forecast_length_from_plan_when_no_syntax(
            self, create_tiny_node_pred_graph, tempdir):
        """No FORECAST clause, model plan has forecast_length: 3 → use plan."""
        graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(actions.amount, 0, 2) "
                 "FOR EACH entity.entity_id")
        yaml = _make_model_plan_yaml(forecast_length=3)

        config = _make_config('test', query, yaml, graph)

        assert config.forecast_length == 3

    def test_forecast_length_from_syntax_when_no_plan(
            self, create_tiny_node_pred_graph, tempdir):
        """FORECAST 3 in syntax, no forecast_length in plan → use syntax."""
        graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(actions.amount, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH entity.entity_id")

        config = _make_config('test', query, '', graph)

        assert config.forecast_length == 3


class TestLagTimestepsReset:
    """Tests for lag_timesteps reset logic when model plan doesn't
    specify lag_timesteps for a forecasting query."""
    def test_reset_to_none_when_plan_missing_lag_timesteps(
            self, create_tiny_node_pred_graph, tempdir):
        """User submitted model plan without lag_timesteps for FORECAST
        query → lag_timesteps reset to None (validator catches it)."""
        graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(actions.amount, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH entity.entity_id")
        # Model plan with forecast_length but no lag_timesteps
        yaml = _make_model_plan_yaml(forecast_length=3)

        config = _make_config('test', query, yaml, graph)

        assert config.lag_timesteps is None

    def test_lag_timesteps_preserved_when_plan_specifies_it(
            self, create_tiny_node_pred_graph, tempdir):
        """User submitted model plan with lag_timesteps=5 → preserved."""
        graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(actions.amount, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH entity.entity_id")
        yaml = _make_model_plan_yaml(forecast_length=3, lag_timesteps=5)

        config = _make_config('test', query, yaml, graph)

        assert config.lag_timesteps == 5

    def test_bootstrap_keeps_lag_timesteps(self, create_tiny_node_pred_graph,
                                           tempdir):
        """Empty model plan (bootstrap/autopopulation) → lag_timesteps
        stays at 1 from parser, not reset to None."""
        graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(actions.amount, 0, 2) "
                 "FORECAST 3 TIMEFRAMES "
                 "FOR EACH entity.entity_id")

        config = _make_config('test', query, '', graph)

        assert config.lag_timesteps == 1

    def test_no_reset_for_non_forecast_query(self, create_tiny_node_pred_graph,
                                             tempdir):
        """Normal regression query with model plan → lag_timesteps
        stays None (no reset logic triggered)."""
        graph = create_tiny_node_pred_graph('test_graph', tempdir.name)
        query = ("PREDICT SUM(actions.amount, 0, 2) "
                 "FOR EACH entity.entity_id")
        yaml = _make_model_plan_yaml()

        config = _make_config('test', query, yaml, graph)

        assert config.lag_timesteps is None


class TestForecastingPassThroughPlan:
    """The BP pass-through plan exists only to carry three forecasting values
    into the query config, and must stay silent for tasks where those values
    do not apply."""

    _PARAMS = dict(forecast_length=3, lag_timesteps=5, year_over_year=True)

    @pytest.mark.parametrize('task_type',
                             [TaskType.FORECASTING, TaskType.REGRESSION])
    def test_carries_params_for_temporal_forecasting_tasks(self, task_type):
        train_options = forecasting_pass_through_plan(task_type,
                                                      QueryType.TEMPORAL,
                                                      **self._PARAMS)

        assert train_options.forecast_length == 3
        assert train_options.lag_timesteps == 5
        assert train_options.year_over_year is True

    @pytest.mark.parametrize(('task_type', 'query_type'), [
        (TaskType.BINARY_CLASSIFICATION, QueryType.TEMPORAL),
        (TaskType.TEMPORAL_LINK_PREDICTION, QueryType.TEMPORAL),
        (TaskType.REGRESSION, QueryType.STATIC),
    ])
    def test_returns_none_for_tasks_with_nothing_to_pass_through(
            self, task_type, query_type):
        """Passing these on would push None/False onto the parsed query
        config. Returning None rather than empty options also keeps the
        caller from reporting that it supplied options when it did not."""
        assert forecasting_pass_through_plan(task_type, query_type,
                                             **self._PARAMS) is None

    @pytest.mark.parametrize(('task_type', 'query_type'), [
        (TaskType.FORECASTING, QueryType.TEMPORAL),
        (TaskType.BINARY_CLASSIFICATION, QueryType.TEMPORAL),
    ])
    def test_provided_flag_follows_whether_options_were_built(
            self, task_type, query_type):
        train_options = forecasting_pass_through_plan(task_type, query_type,
                                                      **self._PARAMS)
        _resolved, provided = resolve_train_table_options('', train_options)

        assert provided is (train_options is not None)

    def test_guard_matches_the_plan_metadata_it_mirrors(self):
        """The task/query guard duplicates `TrainingTableGenerationPlan`
        metadata. Pin the guard itself against that metadata, so widening
        either side fails here instead of silently dropping the params."""
        gated = {'forecast_length', 'lag_timesteps', 'year_over_year'}
        seen = set()
        for name, _value, metadata in TrainingTableGenerationPlan().items():
            if name not in gated:
                continue
            seen.add(name)
            assert set(metadata.valid_task_types) == set(
                _FORECASTING_TASK_TYPES), name
            assert set(metadata.valid_query_types) == {QueryType.TEMPORAL}, name
        assert seen == gated
