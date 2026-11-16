# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import kumo_core.training.explain_selection as select_mod
import pytest
import torch
from kumo_core.training.explain_selection import (
    ClassificationSelectionStrategy,
    LinkPredictionSelectionStrategy,
    RegressionSelectionStrategy,
    select_classification,
    select_link_prediction,
    select_regression,
)
from kumo_core.types import TaskType
from kumoapi.train import TrainingStage
from torch import Tensor


@pytest.mark.parametrize(
    'strategy',
    ClassificationSelectionStrategy.get_strategies(
        TaskType.BINARY_CLASSIFICATION, TrainingStage.PRED))
def test_select_binary_classification_pred(strategy):
    prob = torch.tensor([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.2, 0.8],
        [0.8, 0.2],
    ])
    nodes = torch.arange(prob.size(0))
    time = torch.tensor([1, 10, 2, 5, 8, 9, 7, 6, 3, 0])

    out, idx = select_classification(
        nodes=nodes,
        topk=4,
        strategy=strategy,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target=None,
        prob=prob,
        time=time,
    )
    assert isinstance(out, Tensor)
    assert isinstance(idx, Tensor)
    assert out.unique().numel() == 4
    assert out.min() >= 0 and out.max() < nodes.size(0)
    assert idx.unique().numel() == 4
    assert idx.min() >= 0 and idx.max() < nodes.size(0)


def test_select_unsure_clamps_topk_to_available_nodes():
    """UNSURE must clamp ``topk`` to the number of available probabilities.
    When the (test) set is smaller than ``topk`` an unclamped ``topk`` raised
    "selected index k out of range"; clamp to ``min(topk, num_nodes)``."""
    prob = torch.tensor([[0.9, 0.1], [0.4, 0.6], [0.5, 0.5]])
    nodes = torch.arange(prob.size(0))
    time = torch.tensor([1, 2, 0])

    out, idx = select_classification(
        nodes=nodes,
        topk=10,  # larger than the 3 available nodes
        strategy=ClassificationSelectionStrategy.UNSURE,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target=None,
        prob=prob,
        time=time,
    )
    # Returns at most the available nodes instead of raising.
    assert out.numel() == nodes.size(0)
    assert idx.min() >= 0 and idx.max() < nodes.size(0)


@pytest.mark.parametrize(
    'strategy',
    ClassificationSelectionStrategy.get_strategies(
        TaskType.BINARY_CLASSIFICATION))
def test_select_binary_classification_eval(strategy):
    target = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0, 0, 1]).view(-1, 1)
    prob = torch.tensor([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.2, 0.8],
        [0.2, 0.8],
        [0.8, 0.2],
    ])
    nodes = torch.arange(target.size(0)).flip(0)
    time = torch.tensor([1, 10, 2, 5, 8, 9, 7, 6, 3, 0])

    out, idx = select_classification(
        nodes=nodes,
        topk=4,
        strategy=strategy,
        task_type=TaskType.BINARY_CLASSIFICATION,
        target=target,
        prob=prob,
        time=time,
    )
    assert isinstance(out, Tensor)
    assert isinstance(idx, Tensor)
    assert out.unique().numel() == 4
    assert idx.unique().numel() == 4
    assert out.min() >= 0 and out.max() < nodes.size(0)
    assert idx.min() >= 0 and idx.max() < nodes.size(0)

    if strategy == ClassificationSelectionStrategy.UNSURE:
        assert out.equal(torch.tensor([4, 3, 7, 2]))
        assert idx.equal(torch.tensor([5, 6, 2, 7]))
    elif strategy == ClassificationSelectionStrategy.TRUE_POSITIVE:
        assert out.equal(torch.tensor([6, 8, 4, 3]))
        assert idx.equal(torch.tensor([3, 1, 5, 6]))
    elif strategy == ClassificationSelectionStrategy.TRUE_NEGATIVE:
        assert out.equal(torch.tensor([9, 5, 7, 2]))
        assert idx.equal(torch.tensor([0, 4, 2, 7]))
    elif strategy == ClassificationSelectionStrategy.FALSE_POSITIVE:
        assert out.equal(torch.tensor([1, 2, 7, 5]))
        assert idx.equal(torch.tensor([8, 7, 2, 4]))
    elif strategy == ClassificationSelectionStrategy.FALSE_NEGATIVE:
        assert out.equal(torch.tensor([0, 4, 3, 8]))
        assert idx.equal(torch.tensor([9, 5, 6, 1]))
    elif strategy == ClassificationSelectionStrategy.OLD_ENTITY:
        assert out.equal(torch.tensor([0, 9, 7, 1]))
        assert idx.equal(torch.tensor([9, 0, 2, 8]))
    elif strategy == ClassificationSelectionStrategy.RECENT_ENTITY:
        assert out.equal(torch.tensor([8, 4, 5, 3]))
        assert idx.equal(torch.tensor([1, 5, 4, 6]))


@pytest.mark.parametrize(
    'strategy',
    ClassificationSelectionStrategy.get_strategies(
        TaskType.MULTICLASS_CLASSIFICATION))
def test_select_multiclass_classification(strategy):
    target = torch.tensor([0, 1, 3, 1, 1, 1, 1, 1]).view(-1, 1)
    prob = torch.tensor([
        [0.70, 0.10, 0.10, 0.10],
        [0.10, 0.80, 0.05, 0.05],
        [0.30, 0.20, 0.25, 0.25],
        [0.10, 0.30, 0.40, 0.20],
        [0.50, 0.20, 0.10, 0.20],
        [0.05, 0.05, 0.70, 0.20],
        [0.20, 0.30, 0.25, 0.25],
        [0.20, 0.60, 0.10, 0.10],
    ])
    nodes = torch.arange(target.size(0)).flip(0)
    time = torch.tensor([1, 10, 2, 5, 8, 9, 7, 6])

    out, idx = select_classification(
        nodes=nodes,
        topk=3,
        strategy=strategy,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        target=target,
        prob=prob,
        time=time,
    )
    assert isinstance(out, Tensor)
    assert isinstance(idx, Tensor)
    assert out.unique().numel() == 3
    assert out.min() >= 0 and out.max() < nodes.size(0)
    assert idx.unique().numel() == 3
    assert idx.min() >= 0 and idx.max() < nodes.size(0)

    if strategy == ClassificationSelectionStrategy.UNSURE:
        assert out.equal(torch.tensor([1, 5, 4]))
        assert idx.equal(torch.tensor([6, 2, 3]))
    elif strategy == ClassificationSelectionStrategy.TRUE_POSITIVE:
        assert out.equal(torch.tensor([6, 7, 0]))
        assert idx.equal(torch.tensor([1, 0, 7]))
    elif strategy == ClassificationSelectionStrategy.FALSE_POSITIVE:
        assert out.equal(torch.tensor([4, 5, 3]))
        assert idx.equal(torch.tensor([3, 2, 4]))
    elif strategy == ClassificationSelectionStrategy.OLD_ENTITY:
        assert out.equal(torch.tensor([7, 5, 4]))
        assert idx.equal(torch.tensor([0, 2, 3]))
    elif strategy == ClassificationSelectionStrategy.RECENT_ENTITY:
        assert out.equal(torch.tensor([6, 2, 3]))
        assert idx.equal(torch.tensor([1, 5, 4]))


@pytest.mark.parametrize(
    'strategy',
    ClassificationSelectionStrategy.get_strategies(
        TaskType.MULTILABEL_CLASSIFICATION))
def test_select_multilabel_classification(strategy):
    target = torch.tensor([
        [1., 1., 0.],
        [1., 1., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
    ])
    prob = torch.tensor([
        [0.7, 0.6, 0.9],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.7],
        [0.1, 0.3, 0.4],
        [0.5, 0.8, 0.1],
    ])
    nodes = torch.arange(target.size(0)).flip(0)
    time = torch.tensor([1, 10, 2, 5, 8])

    out, idx = select_classification(
        nodes=nodes,
        topk=2,
        strategy=strategy,
        task_type=TaskType.MULTILABEL_CLASSIFICATION,
        target=target,
        prob=prob,
        time=time,
    )
    assert isinstance(out, Tensor)
    assert isinstance(idx, Tensor)
    assert out.unique().numel() == 2
    assert out.min() >= 0 and out.max() < nodes.size(0)
    assert idx.unique().numel() == 2
    assert idx.min() >= 0 and idx.max() < nodes.size(0)

    if strategy == ClassificationSelectionStrategy.UNSURE:
        assert out.equal(torch.tensor([1, 0]))
        assert idx.equal(torch.tensor([3, 4]))
    elif strategy == ClassificationSelectionStrategy.TRUE_POSITIVE:
        assert out.equal(torch.tensor([3, 0]))
        assert idx.equal(torch.tensor([1, 4]))
    elif strategy == ClassificationSelectionStrategy.TRUE_NEGATIVE:
        assert out.equal(torch.tensor([3, 0]))
        assert idx.equal(torch.tensor([1, 4]))
    elif strategy == ClassificationSelectionStrategy.FALSE_POSITIVE:
        assert out.equal(torch.tensor([4, 2]))
        assert idx.equal(torch.tensor([0, 2]))
    elif strategy == ClassificationSelectionStrategy.FALSE_NEGATIVE:
        assert out.equal(torch.tensor([3, 2]))
        assert idx.equal(torch.tensor([1, 2]))
    elif strategy == ClassificationSelectionStrategy.OLD_ENTITY:
        assert out.equal(torch.tensor([4, 2]))
        assert idx.equal(torch.tensor([0, 2]))
    elif strategy == ClassificationSelectionStrategy.RECENT_ENTITY:
        assert out.equal(torch.tensor([3, 0]))
        assert idx.equal(torch.tensor([1, 4]))


@pytest.mark.parametrize(
    'strategy', RegressionSelectionStrategy.get_strategies(TrainingStage.PRED))
def test_select_regression_pred(strategy):
    pred = torch.tensor([110.0, 1.0, -0.2, -100.0, 10.0, -0.001]).view(-1, 1)
    nodes = torch.arange(pred.size(0))
    time = torch.tensor([1, 10, 2, 5, 8, 9])

    out, idx = select_regression(
        nodes=nodes,
        topk=3,
        strategy=strategy,
        target=None,
        pred=pred,
        time=time,
    )
    assert isinstance(out, Tensor)
    assert isinstance(idx, Tensor)
    assert out.unique().numel() == 3
    assert idx.unique().numel() == 3
    assert out.min() >= 0 and out.max() < nodes.size(0)
    assert idx.min() >= 0 and idx.max() < nodes.size(0)


@pytest.mark.parametrize('strategy', list(RegressionSelectionStrategy))
def test_select_regression_eval(strategy):
    target = torch.tensor([100.0, 1.0, 0.1, 100.0, 10.0, 0.0]).view(-1, 1)
    pred = torch.tensor([110.0, 1.0, -0.2, -100.0, 10.0, -0.001]).view(-1, 1)
    nodes = torch.arange(target.size(0)).flip(0)
    time = torch.tensor([1, 10, 2, 5, 8, 9])

    out, idx = select_regression(
        nodes=nodes,
        topk=3,
        strategy=strategy,
        target=target,
        pred=pred,
        time=time,
    )
    assert isinstance(out, Tensor)
    assert isinstance(idx, Tensor)
    assert out.unique().numel() == 3
    assert out.min() >= 0 and out.max() < nodes.size(0)
    assert idx.unique().numel() == 3
    assert idx.min() >= 0 and idx.max() < nodes.size(0)

    if strategy == RegressionSelectionStrategy.SMALL_ERROR:
        assert out.equal(torch.tensor([4, 1, 0]))
        assert idx.equal(torch.tensor([1, 4, 5]))
    elif strategy == RegressionSelectionStrategy.LARGE_ERROR:
        assert out.equal(torch.tensor([2, 5, 3]))
        assert idx.equal(torch.tensor([3, 0, 2]))
    elif strategy == RegressionSelectionStrategy.SMALL_PERCENT_ERROR:
        assert out.equal(torch.tensor([4, 1, 5]))
        assert idx.equal(torch.tensor([1, 4, 0]))
    elif strategy == RegressionSelectionStrategy.LARGE_PERCENT_ERROR:
        assert out.equal(torch.tensor([0, 3, 2]))
        assert idx.equal(torch.tensor([5, 2, 3]))
    elif strategy == RegressionSelectionStrategy.OLD_ENTITY:
        assert out.equal(torch.tensor([5, 3, 2]))
        assert idx.equal(torch.tensor([0, 2, 3]))
    elif strategy == RegressionSelectionStrategy.RECENT_ENTITY:
        assert out.equal(torch.tensor([4, 0, 1]))
        assert idx.equal(torch.tensor([1, 5, 4]))


@pytest.mark.parametrize(
    'strategy',
    LinkPredictionSelectionStrategy.get_strategies(TrainingStage.PRED))
def test_select_link_prediction_pred(strategy):
    time = torch.tensor([1, 10, 2, 5, 8])
    nodes = torch.arange(time.size(0))

    out, idx = select_link_prediction(
        nodes=nodes,
        topk=3,
        strategy=strategy,
        pred_isin_mat=None,
        time=time,
    )
    assert isinstance(out, Tensor)
    assert isinstance(idx, Tensor)
    assert out.unique().numel() == 3
    assert idx.unique().numel() == 3
    assert out.min() >= 0 and out.max() < nodes.size(0)
    assert idx.min() >= 0 and idx.max() < nodes.size(0)
    assert idx.equal(torch.tensor([0, 1, 2]))


@pytest.mark.parametrize('strategy', list(LinkPredictionSelectionStrategy))
def test_select_link_prediction_eval(strategy):
    pred_isin_mat = torch.tensor([
        [False, False, True],
        [False, False, False],
        [False, True, False],
        [True, False, True],
        [True, True, True],
    ])
    nodes = torch.arange(pred_isin_mat.size(0))
    time = torch.tensor([1, 10, 2, 5, 8])

    out, idx = select_link_prediction(
        nodes=nodes,
        topk=3,
        strategy=strategy,
        pred_isin_mat=pred_isin_mat,
        time=time,
    )
    assert isinstance(out, Tensor)
    assert isinstance(idx, Tensor)
    assert out.unique().numel() == 3
    assert idx.unique().numel() == 3
    assert out.min() >= 0 and out.max() < nodes.size(0)
    assert idx.min() >= 0 and idx.max() < nodes.size(0)

    if strategy == LinkPredictionSelectionStrategy.LOW_PRECISION:
        assert (out.equal(torch.tensor([1, 2, 0]))
                or out.equal(torch.tensor([1, 0, 2])))
        assert out.equal(idx)
    elif strategy == LinkPredictionSelectionStrategy.HIGH_PRECISION:
        assert (out.equal(torch.tensor([4, 3, 2]))
                or out.equal(torch.tensor([4, 3, 1])))
        assert out.equal(idx)
    elif strategy == LinkPredictionSelectionStrategy.OLD_ENTITY:
        assert out.equal(torch.tensor([0, 2, 3]))
        assert idx.equal(torch.tensor([0, 2, 3]))
    elif strategy == LinkPredictionSelectionStrategy.RECENT_ENTITY:
        assert out.equal(torch.tensor([1, 4, 3]))
        assert idx.equal(torch.tensor([1, 4, 3]))


def test_select_input_nodes_link_prediction_eval_uses_input_id(monkeypatch):
    class FeatureStore:
        entity_table_names = ['User']
        time_col_dict = {}

        def __getitem__(self, key):
            table_name, value_name, ids = key
            assert table_name == 'User'
            assert value_name == 'pkey'
            return ids + 1000

    monkeypatch.setattr(select_mod, 'EVAL_NUM_SELECTION_NODES', 1)
    input_nodes = torch.tensor([7, 7, 9])
    input_id = torch.tensor([100, 101, 102])
    outputs = {
        'label': torch.tensor([
            [False, False],
            [True, True],
            [False, True],
        ])
    }

    seed_nodes, mappings = select_mod.select_input_nodes(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        feature_store=FeatureStore(),
        input_nodes=input_nodes,
        input_id=input_id,
        outputs=outputs,
        stage=TrainingStage.TEST,
    )

    assert set(seed_nodes.tolist()).issubset({100, 101, 102})
    assert 101 in seed_nodes
    assert 7 not in seed_nodes
    assert mappings.mappings['low_precision'] == ['0']
    assert mappings.mappings['high_precision'] == ['1']
    assert mappings.orig_mappings['high_precision'] == [1007]


def test_select_input_nodes_pred_uses_input_id_when_available(monkeypatch):
    class FeatureStore:
        entity_table_names = ['User']
        time_col_dict = {}

        def __getitem__(self, key):
            table_name, value_name, ids = key
            assert table_name == 'User'
            assert value_name == 'pkey'
            return ids + 1000

    monkeypatch.setattr(select_mod, 'PRED_NUM_SELECTION_NODES', 2)
    input_nodes = torch.tensor([7, 7, 9])
    input_id = torch.tensor([100, 101, 102])
    outputs = {
        'pred': torch.tensor([1, 0, 1]),
        'prob': torch.tensor([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ]),
    }

    seed_nodes, mappings = select_mod.select_input_nodes(
        task_type=TaskType.BINARY_CLASSIFICATION,
        feature_store=FeatureStore(),
        input_nodes=input_nodes,
        input_id=input_id,
        outputs=outputs,
        stage=TrainingStage.PRED,
    )

    assert torch.equal(seed_nodes, torch.tensor([100, 101]))
    assert mappings.mappings['first'] == ['0', '1']
    assert mappings.orig_mappings['first'] == [1007, 1007]


def test_select_input_nodes_link_prediction_pred_uses_input_id(monkeypatch):
    class FeatureStore:
        entity_table_names = ['User']
        time_col_dict = {}

        def __getitem__(self, key):
            table_name, value_name, ids = key
            assert table_name == 'User'
            assert value_name == 'pkey'
            return ids + 1000

    monkeypatch.setattr(select_mod, 'PRED_NUM_SELECTION_NODES', 2)
    input_nodes = torch.tensor([7, 7, 9])
    input_id = torch.tensor([100, 101, 102])
    outputs = {
        'target': torch.tensor([[2, 3], [4, 5], [6, 7]]),
        'score': torch.ones(3, 2),
    }

    seed_nodes, mappings = select_mod.select_input_nodes(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        feature_store=FeatureStore(),
        input_nodes=input_nodes,
        input_id=input_id,
        outputs=outputs,
        stage=TrainingStage.PRED,
    )

    assert torch.equal(seed_nodes, torch.tensor([100, 101]))
    assert mappings.mappings['first'] == ['0', '1']
    assert mappings.orig_mappings['first'] == [1007, 1007]
