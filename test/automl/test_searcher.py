# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import random
from dataclasses import fields
from unittest.mock import Mock

import fsspec
import pytest
from kumo_core.automl.optuna_searcher import NSGAIISearcher, TPESearcher
from kumo_core.automl.random_searcher import RandomSearcher
from kumo_core.io import filesystem
from kumo_core.io.json import to_json_dict
from kumo_core.model_plan import (
    GNNModelPlan,
    GraphTransformerModelPlan,
    ModelPlan,
)
from kumo_core.types import TaskType
from kumoapi.model_plan import (
    AggregationType,
    HopConfig,
    MissingType,
    NumNeighborsConfig,
    PlanMixin,
    PositionalEncodingType,
    RunMode,
)
from kumoapi.pquery import QueryType


@pytest.mark.parametrize(
    'SearcherCls, kwargs',
    [
        (TPESearcher, dict(n_startup_trials=2)),
        (NSGAIISearcher, dict(population_size=4)),
        (RandomSearcher, dict()),
    ],
)
def test_searcher_default(SearcherCls, kwargs):
    space = ModelPlan.from_default(
        TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )

    searcher = SearcherCls(space, **kwargs)
    if not isinstance(searcher, RandomSearcher):
        assert (searcher.study.sampler.__class__.__name__ ==
                SearcherCls.__name__.replace('Searcher', 'Sampler'))

    instance = searcher.suggest()
    assert isinstance(instance, ModelPlan)

    for field in fields(instance):
        for key, value, metadata in getattr(instance, field.name).items():
            values = getattr(getattr(space, field.name), key)

            if metadata.tunable and values != MissingType.VALUE:
                assert isinstance(value, list)
                assert len(value) == 1

                # test nested space
                if isinstance(value[0], PlanMixin):
                    for subkey, subvalue, submetadata in value[0].items():
                        candidates = [
                            getattr(v, subkey) for v in values
                            if hasattr(v, subkey)
                        ]
                        candidates = [x for li in candidates for x in li]

                        if (submetadata.tunable
                                and subvalue != MissingType.VALUE):
                            assert isinstance(subvalue, list)
                            assert len(subvalue) == 1
                            assert subvalue[0] in candidates
                else:
                    assert value[0] in values
            else:
                assert value == values


@pytest.mark.parametrize('SearcherCls', [TPESearcher, RandomSearcher])
def test_unique_trials(SearcherCls):
    space = ModelPlan.from_default(
        TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    searcher = SearcherCls(space)

    for _ in range(20):
        cfg = searcher.suggest()
        searcher.update(cfg, [random.random()])
    unique_cfgs = set(
        json.dumps(to_json_dict(cfg), sort_keys=True)
        for cfg, _ in searcher.state)
    assert len(unique_cfgs) == 20


def test_analyze_single_trial_skips_parameter_importance(tmp_path):
    space = ModelPlan.from_default(
        TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    searcher = TPESearcher(space, n_startup_trials=2)
    instance = searcher.suggest()
    searcher.update(instance, [0.5])

    searcher.analyze(str(tmp_path))

    assert (tmp_path / "optimization_history.png").is_file()
    assert not (tmp_path / "param_importances.png").exists()


@pytest.mark.parametrize('scheme', ['local', 'memory', 'abfss'])
def test_analyze_with_unhashable_categorical_value(tmp_path, monkeypatch,
                                                   scheme):
    # `NumNeighborsConfig` instances are unhashable, so Optuna's plotting
    # code (which uses trial params as dict keys) must be given a
    # hashable stand-in rather than the raw config objects.
    space = ModelPlan.from_default(
        TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    space.neighbor_sampling.num_neighbors = [
        NumNeighborsConfig(hop1=HopConfig(default=10)),
        NumNeighborsConfig(hop1=HopConfig(default=20)),
    ]

    # Seeded so `num_neighbors` actually varies across the two trials below
    # -- see the seeding note in test_analyze_with_unhashable_nested_
    # categorical_value above.
    searcher = TPESearcher(space, n_startup_trials=2, seed=1)
    for _ in range(2):
        instance = searcher.suggest()
        searcher.update(instance, [random.random()])

    choices = space.neighbor_sampling.num_neighbors
    choice_labels = tuple(f"{index}: {choice}"
                          for index, choice in enumerate(choices))
    distribution = searcher.study.trials[0].distributions["num_neighbors"]
    assert distribution.choices == choice_labels

    monkeypatch.chdir(tmp_path)
    path = str(tmp_path) if scheme == 'local' else f'{scheme}://{tmp_path.name}'
    if scheme == 'abfss':
        original = fsspec.get_filesystem_class

        def get_filesystem_class(protocol):
            if protocol == 'abfss':
                raise ValueError('Protocol not known: abfss')
            return original(protocol)

        monkeypatch.setattr(fsspec, 'get_filesystem_class',
                            get_filesystem_class)
        monkeypatch.setattr(filesystem, '_fsspec_fallback',
                            lambda uri: uri.replace('abfss://', 'memory://', 1))

    searcher.analyze(path)

    for artifact in ('optimization_history.png', 'slice/num_neighbors.png'):
        with filesystem.open_file(f'{path}/{artifact}', 'rb') as source:
            assert source.read(8) == b'\x89PNG\r\n\x1a\n'
    assert not (tmp_path / f'{scheme}:').exists()


def test_sample_value_with_later_unhashable_choice():
    space = ModelPlan.from_default(
        TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    searcher = TPESearcher(space, n_startup_trials=2)
    searcher.current_trial = Mock()
    choices = [1, [2, 3]]
    choice_labels = ["0: 1", "1: [2, 3]"]
    searcher.current_trial.suggest_categorical.return_value = choice_labels[1]

    assert searcher.sample_value("mixed", choices) == [2, 3]
    searcher.current_trial.suggest_categorical.assert_called_once_with(
        "mixed", choice_labels)


def test_sample_value_uses_unique_labels_for_duplicate_choices():
    space = ModelPlan.from_default(
        TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    searcher = TPESearcher(space, n_startup_trials=2)
    searcher.current_trial = Mock()
    choices = [[1], [1]]
    choice_labels = ["0: [1]", "1: [1]"]
    searcher.current_trial.suggest_categorical.return_value = choice_labels[1]

    assert searcher.sample_value("duplicate", choices) is choices[1]
    searcher.current_trial.suggest_categorical.assert_called_once_with(
        "duplicate", choice_labels)


def test_analyze_with_unhashable_nested_categorical_value(tmp_path):
    # `positional_encodings` values are plain `list`s, which are also
    # unhashable and can't be made hashable the way a dataclass could.
    space = ModelPlan.from_default(
        TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    space.model_architecture.channels = MissingType.VALUE
    space.model_architecture.aggregation = MissingType.VALUE
    space.model_architecture.model = [
        GraphTransformerModelPlan(
            channels=[16, 32],
            num_layers=[1, 2],
            num_heads=[4, 8],
            dropout=[0.1, 0.2],
            positional_encodings=[
                [PositionalEncodingType.HOP, PositionalEncodingType.RANDOM_GNN],
                [],  # no positional encoding
            ],
        ),
    ]

    # Seeded so `positional_encodings` actually varies across the two
    # trials below -- analyze() skips the slice plot for a param that
    # sampled the same value in every trial (nothing to show a
    # relationship for), so an unseeded run could flakily collapse to a
    # single value and fail this assertion by chance.
    searcher = TPESearcher(space, n_startup_trials=2, seed=0)
    for _ in range(2):
        instance = searcher.suggest()
        searcher.update(instance, [random.random()])

    searcher.analyze(str(tmp_path))

    assert (tmp_path / "optimization_history.png").is_file()
    assert (tmp_path / "slice" / "model0_positional_encodings.png").is_file()


@pytest.fixture()
def nested_model_plan() -> ModelPlan:
    model_plan = ModelPlan.from_default(
        task_type=TaskType.BINARY_CLASSIFICATION,
        query_type=QueryType.TEMPORAL,
        has_train_table_time_col=True,
        run_mode=RunMode.FAST,
    )
    # unsets default values for attributes outside model_architecture.model
    model_plan.model_architecture.channels = MissingType.VALUE
    model_plan.model_architecture.aggregation = MissingType.VALUE

    model_plan.model_architecture.model = [
        GNNModelPlan(
            channels=[64, 128],
            aggregation=[[AggregationType.MEAN]],
        ),
        GraphTransformerModelPlan(
            channels=[16, 32],
            num_layers=[1, 2],
            num_heads=[4, 8],
            dropout=[0.1, 0.2],
            positional_encodings=[
                [PositionalEncodingType.HOP, PositionalEncodingType.RANDOM_GNN],
                [],  # no positional encoding
            ],
        ),
    ]

    return model_plan


@pytest.mark.parametrize(
    'SearcherCls, kwargs',
    [
        (TPESearcher, dict(n_startup_trials=2)),
        (NSGAIISearcher, dict(population_size=4)),
        (RandomSearcher, dict()),
    ],
)
def test_searcher_nested_space_all_values(SearcherCls, kwargs,
                                          nested_model_plan: ModelPlan) -> None:
    searcher = SearcherCls(nested_model_plan, **kwargs)

    instance = searcher.suggest()
    assert isinstance(instance, ModelPlan)

    for field in fields(instance):
        section = getattr(instance, field.name)
        for key, value, metadata in section.items():
            values = getattr(getattr(nested_model_plan, field.name), key)

            if metadata.tunable and value != MissingType.VALUE:
                assert isinstance(value, list)
                assert len(value) == 1

                # test nested space
                if isinstance(value[0], PlanMixin):
                    for _, subvalue, submetadata in value[0].items():
                        if (submetadata.tunable
                                and subvalue != MissingType.VALUE):
                            assert isinstance(subvalue, list)
                            assert len(subvalue) == 1
            else:
                assert value == values


@pytest.mark.parametrize(
    'SearcherCls, kwargs',
    [
        (TPESearcher, dict(n_startup_trials=2)),
        (NSGAIISearcher, dict(population_size=4)),
        (RandomSearcher, dict()),
    ],
)
def test_searcher_nested_space_model_architecture(nested_model_plan: ModelPlan,
                                                  SearcherCls, kwargs):
    gnn_options = {
        k: v
        for k, v, _ in nested_model_plan.model_architecture.model[0].items()
    }
    gt_options = {
        k: v
        for k, v, _ in nested_model_plan.model_architecture.model[1].items()
    }

    searcher = SearcherCls(nested_model_plan, **kwargs)

    for _ in range(20):
        instance = searcher.suggest()
        assert isinstance(instance, ModelPlan)

        candidates = gnn_options if instance.model_architecture.model[
            0].name == 'GNN' else gt_options
        for key, value, metadata in instance.model_architecture.items():
            original_values = getattr(nested_model_plan.model_architecture, key)

            if metadata.tunable and value != MissingType.VALUE:
                assert isinstance(value, list)
                assert len(value) == 1

                # test nested space
                if isinstance(value[0], PlanMixin):
                    for subkey, subvalue, submetadata in value[0].items():
                        if (submetadata.tunable
                                and subvalue != MissingType.VALUE):
                            assert isinstance(subvalue, list)
                            assert len(subvalue) == 1
                            assert subvalue[0] in candidates[subkey]
            else:
                assert value == original_values

        searcher.update(instance, [random.random()])
