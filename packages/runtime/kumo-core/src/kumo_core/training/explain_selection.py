# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hosted selection policy shared by local and hosted subgraph collectors."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch
from kumo_core.types import TaskType
from kumoapi.explain import EntityMappings
from kumoapi.train import TrainingStage
from torch import Tensor

if TYPE_CHECKING:
    from kumo_core.features.stores import FeatureStore

EVAL_NUM_SELECTION_NODES = 50
PRED_NUM_SELECTION_NODES = 1000


class ClassificationSelectionStrategy(Enum):
    FIRST = 'first'
    RANDOM = 'random'
    UNSURE = 'unsure'
    TRUE_POSITIVE = 'true_positive'
    TRUE_NEGATIVE = 'true_negative'
    FALSE_POSITIVE = 'false_positive'
    FALSE_NEGATIVE = 'false_negative'
    OLD_ENTITY = 'old_entity'
    RECENT_ENTITY = 'recent_entity'

    @staticmethod
    def get_strategies(
        task_type: TaskType, stage: TrainingStage | None = None
    ) -> list['ClassificationSelectionStrategy']:
        assert task_type.is_classification

        if stage == TrainingStage.PRED:
            return [ClassificationSelectionStrategy.FIRST]

        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            # For multi-class classification, we select true positives
            # (model prediction is correct) and false positives
            # (model prediction is incorrect).
            # True negatives and false negatives do not have a meaning.
            strategies = list(
                set(ClassificationSelectionStrategy) - {
                    ClassificationSelectionStrategy.TRUE_NEGATIVE,
                    ClassificationSelectionStrategy.FALSE_NEGATIVE,
                })
        else:
            strategies = list(ClassificationSelectionStrategy)
        if ClassificationSelectionStrategy.FIRST in strategies:
            strategies.remove(ClassificationSelectionStrategy.FIRST)
        return strategies


class RegressionSelectionStrategy(Enum):
    FIRST = 'first'
    RANDOM = 'random'
    SMALL_ERROR = 'small_error'
    LARGE_ERROR = 'large_error'
    SMALL_PERCENT_ERROR = 'small_percent_error'
    LARGE_PERCENT_ERROR = 'large_percent_error'
    OLD_ENTITY = 'old_entity'
    RECENT_ENTITY = 'recent_entity'

    @staticmethod
    def get_strategies(
        stage: TrainingStage | None = None
    ) -> list['RegressionSelectionStrategy']:
        if stage == TrainingStage.PRED:
            return [RegressionSelectionStrategy.FIRST]
        strategies = list(RegressionSelectionStrategy)
        strategies.remove(RegressionSelectionStrategy.FIRST)
        return strategies


class LinkPredictionSelectionStrategy(Enum):
    FIRST = 'first'
    RANDOM = 'random'
    LOW_PRECISION = 'low_precision'
    HIGH_PRECISION = 'high_precision'
    OLD_ENTITY = 'old_entity'
    RECENT_ENTITY = 'recent_entity'

    @staticmethod
    def get_strategies(
        stage: TrainingStage | None = None
    ) -> list['LinkPredictionSelectionStrategy']:
        if stage == TrainingStage.PRED:
            return [LinkPredictionSelectionStrategy.FIRST]
        strategies = list(LinkPredictionSelectionStrategy)
        strategies.remove(LinkPredictionSelectionStrategy.FIRST)
        return strategies


def select_classification(
    nodes: Tensor,
    topk: int,
    strategy: ClassificationSelectionStrategy,
    task_type: TaskType,
    target: Tensor | None,
    prob: Tensor,
    time: Tensor | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    r"""Selects the `obj`:topk: nodes from :obj:`nodes` and
    returns selected indices according to a selection strategy
    for classification tasks.
    """
    # nodes shape: [num_nodes]
    # target shape: [num_nodes, 1] or [num_nodes, num_classes] or None
    # prob shape: [num_nodes, num_classes]
    # time shape: [num_nodes]
    assert task_type.is_classification

    topk = min(topk, nodes.numel())

    if strategy in {
            ClassificationSelectionStrategy.OLD_ENTITY,
            ClassificationSelectionStrategy.RECENT_ENTITY,
    }:
        if time is None:
            return None, None

        assert time.size(0) == nodes.size(0)
        largest = True
        if strategy == ClassificationSelectionStrategy.OLD_ENTITY:
            largest = False
        index = time.topk(topk, largest=largest)[1]
        return nodes[index], index

    if strategy == ClassificationSelectionStrategy.FIRST:
        return nodes[:topk], torch.arange(topk)

    if strategy == ClassificationSelectionStrategy.RANDOM:
        perm = torch.randperm(nodes.size(0), device=nodes.device)
        return nodes[perm][:topk], perm[:topk]

    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        if strategy == ClassificationSelectionStrategy.UNSURE:
            # Return the top-k nodes with the lowest variance:
            index = prob.var(dim=-1).topk(k=topk, largest=False)[1]
            return nodes[index], index

        assert target is not None
        if strategy == ClassificationSelectionStrategy.TRUE_POSITIVE:
            # Return the top-k nodes with the highest correct prediction:
            mask = target == prob.argmax(dim=-1, keepdim=True)
            prob = prob.gather(1, target)[mask]
            topk = min(topk, prob.numel())
            index = prob.view(-1).topk(k=topk, largest=True)[1]
            index = mask.view(-1).nonzero().view(-1)[index]
            return nodes[index], index

        if strategy == ClassificationSelectionStrategy.FALSE_POSITIVE:
            # Return the top-k nodes with the lowest incorrect prediction:
            mask = target != prob.argmax(dim=-1, keepdim=True)
            prob = prob.gather(1, target)[mask]
            topk = min(topk, prob.numel())
            index = prob.view(-1).topk(k=topk, largest=True)[1]
            index = mask.view(-1).nonzero().view(-1)[index]
            return nodes[index], index

        raise NotImplementedError

    # Convert outputs into flattened binary probabilities:
    multi_index: Tensor | None = None
    if task_type == TaskType.BINARY_CLASSIFICATION:
        prob = prob[:, 1]
    elif task_type == TaskType.MULTILABEL_CLASSIFICATION:
        multi_index = torch.arange(prob.size(0))
        multi_index = multi_index.repeat_interleave(prob.size(1))
        nodes = nodes[multi_index]
    else:
        raise NotImplementedError

    prob = prob.view(-1)

    if strategy == ClassificationSelectionStrategy.UNSURE:
        diff = (prob - prob.median()).abs()
        # Clamp k to the number of available probabilities — when the
        # (test) set is smaller than ``topk``, an unclamped topk raises
        # "selected index k out of range". Mirrors the clamp the
        # TRUE_POSITIVE branch below already applies via ``min(topk, ...)``.
        index = diff.topk(k=min(topk, diff.numel()), largest=False)[1]
        if task_type != TaskType.MULTILABEL_CLASSIFICATION:
            return nodes[index], index
        assert multi_index is not None
        return nodes[index], multi_index[index]

    assert target is not None
    target = target.view(-1)
    if strategy == ClassificationSelectionStrategy.TRUE_POSITIVE:
        mask = target > 0
        num_nodes = int(mask.sum())
        index = prob[mask].topk(k=min(topk, num_nodes), largest=True)[1]
        if task_type != TaskType.MULTILABEL_CLASSIFICATION:
            return nodes[mask][index], torch.arange(mask.numel())[mask][index]
        assert multi_index is not None
        return nodes[mask][index], multi_index[mask][index]

    if strategy == ClassificationSelectionStrategy.FALSE_NEGATIVE:
        mask = target > 0
        num_nodes = int(mask.sum())
        index = prob[mask].topk(k=min(topk, num_nodes), largest=False)[1]
        if task_type != TaskType.MULTILABEL_CLASSIFICATION:
            return nodes[mask][index], torch.arange(mask.numel())[mask][index]
        assert multi_index is not None
        return nodes[mask][index], multi_index[mask][index]

    if strategy == ClassificationSelectionStrategy.TRUE_NEGATIVE:
        mask = target == 0
        num_nodes = int(mask.sum())
        index = prob[mask].topk(k=min(topk, num_nodes), largest=False)[1]
        if task_type != TaskType.MULTILABEL_CLASSIFICATION:
            return nodes[mask][index], torch.arange(mask.numel())[mask][index]
        assert multi_index is not None
        return nodes[mask][index], multi_index[mask][index]

    if strategy == ClassificationSelectionStrategy.FALSE_POSITIVE:
        mask = target == 0
        num_nodes = int(mask.sum())
        index = prob[mask].topk(k=min(topk, num_nodes), largest=True)[1]
        if task_type != TaskType.MULTILABEL_CLASSIFICATION:
            return nodes[mask][index], torch.arange(mask.numel())[mask][index]
        assert multi_index is not None
        return nodes[mask][index], multi_index[mask][index]

    raise ValueError(f"Selection strategy {strategy.value} not supported")


def select_regression(
    nodes: Tensor,
    topk: int,
    strategy: RegressionSelectionStrategy,
    target: Tensor | None,
    pred: Tensor,
    time: Tensor | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    r"""Selects the `obj`:topk: nodes from :obj:`nodes` and
    returns selected indices according to a selection strategy
    for regression tasks.
    """
    # nodes shape: [num_nodes]
    # target shape: [num_nodes, 1] or None
    # pred shape: [num_nodes, 1]
    # time shape: [num_nodes]
    topk = min(topk, nodes.numel())

    if strategy in {
            RegressionSelectionStrategy.OLD_ENTITY,
            RegressionSelectionStrategy.RECENT_ENTITY,
    }:
        if time is None:
            return None, None

        assert time.size(0) == nodes.size(0)
        largest = True
        if strategy == RegressionSelectionStrategy.OLD_ENTITY:
            largest = False
        index = time.topk(topk, largest=largest)[1]
        return nodes[index], index

    if strategy == RegressionSelectionStrategy.FIRST:
        return nodes[:topk], torch.arange(nodes.numel())[:topk]

    if strategy == RegressionSelectionStrategy.RANDOM:
        perm = torch.randperm(nodes.size(0), device=nodes.device)
        return nodes[perm][:topk], perm[:topk]

    assert target is not None
    target = target.view(-1)
    pred = pred.view(-1)
    error = (target - pred).abs()

    if strategy in {
            RegressionSelectionStrategy.SMALL_PERCENT_ERROR,
            RegressionSelectionStrategy.LARGE_PERCENT_ERROR,
    }:
        error = error / (target.abs() + 1e-8)

    if strategy in {
            RegressionSelectionStrategy.SMALL_ERROR,
            RegressionSelectionStrategy.SMALL_PERCENT_ERROR,
    }:
        largest = False
    else:
        largest = True

    index = error.topk(k=topk, largest=largest)[1]
    return nodes[index], index


def select_link_prediction(
    nodes: Tensor,
    topk: int,
    strategy: LinkPredictionSelectionStrategy,
    pred_isin_mat: Tensor | None,
    time: Tensor | None = None,
) -> tuple[Tensor | None, Tensor | None]:
    r"""Selects the `obj`:topk: nodes from :obj:`nodes` according to a
    selection strategy for link prediction tasks.
    """
    # nodes shape: [num_nodes]
    # pred_isin_mat shape: [num_nodes, k] or None
    # time shape: [num_nodes]

    topk = min(topk, nodes.size(0))

    if strategy in {
            LinkPredictionSelectionStrategy.OLD_ENTITY,
            LinkPredictionSelectionStrategy.RECENT_ENTITY,
    }:
        if time is None:
            return None, None

        assert time.size(0) == nodes.size(0)
        largest = True
        if strategy == LinkPredictionSelectionStrategy.OLD_ENTITY:
            largest = False
        index = time.topk(topk, largest=largest)[1]
        return nodes[index], index

    if strategy == LinkPredictionSelectionStrategy.FIRST:
        return nodes[:topk], torch.arange(nodes.numel())[:topk]

    if strategy == LinkPredictionSelectionStrategy.RANDOM:
        perm = torch.randperm(nodes.size(0), device=nodes.device)
        return nodes[perm][:topk], perm[:topk]

    assert pred_isin_mat is not None
    precision = pred_isin_mat.sum(dim=-1) / pred_isin_mat.size(1)

    if strategy == LinkPredictionSelectionStrategy.LOW_PRECISION:
        largest = False
    else:
        largest = True

    index = precision.topk(k=topk, largest=largest)[1]
    return nodes[index], index


def select_input_nodes(
    task_type: TaskType,
    feature_store: FeatureStore,
    input_nodes: Tensor,
    input_id: Tensor | None,
    outputs: dict[str, Tensor],
    stage: TrainingStage,
) -> tuple[Tensor, EntityMappings]:
    r"""Return selected nodes and corresponding mappings.

    For node prediction task, due to potential duplicated :obj:`input_nodes`,
    it returns :obj:`input_id` instead.

    Args:
        task_type: The task type.
        feature_store: The feature store.
        input_nodes: The :obj:`input_nodes` from the batch.
        input_id: The :obj:`input_id` from the batch.
        outputs: The dictionary of model outputs including prediction
            probabilities and labels etc.
        stage: Stage to run the explanation.

        Returns:
            A tuple where the first element is tensor of selected nodes
            from :obj:`input_id` or :obj:`input_nodes`. During evaluation
            stage for node prediction tasks, unique nodes in ascending order
            are selected from :obj:`input_id` to avoid removing duplicated
            nodes but with different timestamps. Otherwise, nodes are selected
            from :obj:`input_nodes`. The second element is the mappings of
            selected nodes, either from :obj:`input_id` or :obj:`input_nodes`,
            to the orders fed into the explainer.
    """
    nodes = []
    n_ids = []
    strategies = []

    entity_table_name = feature_store.entity_table_names[0]
    input_time: Tensor | None = None
    if entity_table_name in feature_store.time_col_dict.keys():
        with feature_store.safe_batch_mode(batch_size=50_000):
            input_time = feature_store[entity_table_name, 'time', input_nodes]

    topk = EVAL_NUM_SELECTION_NODES
    if stage == TrainingStage.PRED:
        topk = PRED_NUM_SELECTION_NODES
    if task_type.is_classification:
        for strategy in ClassificationSelectionStrategy.get_strategies(
                task_type, stage):
            target = outputs.get('target')
            pred = outputs['pred']
            prob = outputs.get('prob')
            assert isinstance(pred, Tensor)
            assert isinstance(prob, Tensor)
            selected_nodes, index = select_classification(
                nodes=input_nodes,
                topk=topk,
                strategy=strategy,
                task_type=task_type,
                target=target,
                prob=prob,
                time=input_time,
            )
            if selected_nodes is not None:
                nodes.append(selected_nodes)
                strategies.append(strategy)
                if input_id is not None:
                    assert index is not None
                    n_ids.append(input_id[index])
    elif task_type == TaskType.REGRESSION:
        for strategy in RegressionSelectionStrategy.get_strategies(stage):
            target = outputs.get('target')
            pred = outputs['pred']
            assert isinstance(pred, Tensor)
            selected_nodes, index = select_regression(
                nodes=input_nodes,
                topk=topk,
                strategy=strategy,
                target=target,
                pred=pred,
                time=input_time,
            )
            if selected_nodes is not None:
                nodes.append(selected_nodes)
                strategies.append(strategy)
                if input_id is not None:
                    assert index is not None
                    n_ids.append(input_id[index])
    elif task_type.is_link_pred:
        for strategy in LinkPredictionSelectionStrategy.get_strategies(stage):
            selected_nodes, index = select_link_prediction(
                nodes=input_nodes,
                topk=topk,
                strategy=strategy,
                pred_isin_mat=outputs.get('label'),
                time=input_time,
            )
            if selected_nodes is not None:
                nodes.append(selected_nodes)
                strategies.append(strategy)
                if input_id is not None:
                    assert index is not None
                    n_ids.append(input_id[index])
    else:
        raise ValueError(f'Unsupported task type {task_type}')

    mappings = {}
    orig_mappings = {}
    nodes_cat: Tensor | None = None
    n_ids_cat: Tensor | None = None
    if len(n_ids) == 0:
        # Return unique nodes in ascending order:
        nodes_cat = torch.cat(nodes, dim=0).unique(sorted=True)
        assert nodes_cat is not None
        node_mappings = {
            int(node_id): str(i)
            for i, node_id in enumerate(nodes_cat)
        }
    else:
        n_ids_cat = torch.cat(n_ids, dim=0).unique(sorted=True)
        assert isinstance(n_ids_cat, Tensor)
        node_mappings = {
            int(node_id): str(i)
            for i, node_id in enumerate(n_ids_cat)
        }

    for i, strategy in enumerate(strategies):
        strategy_nodes: Tensor = nodes[i].cpu()
        seed_entities = \
            feature_store[entity_table_name, 'pkey', strategy_nodes].tolist()
        if len(n_ids) > 0:
            input_ids = n_ids[i].cpu()
            assert input_ids is not None
            mappings[strategy.value] = [
                node_mappings[int(input_id)] for input_id in input_ids
            ]
        else:
            mappings[strategy.value] = [
                node_mappings[int(node_id)] for node_id in strategy_nodes
            ]
        orig_mappings[strategy.value] = seed_entities

    strategy_mappings = EntityMappings(
        mappings=mappings,
        orig_mappings=orig_mappings,
    )
    if len(n_ids) != 0:
        assert n_ids_cat is not None
        return n_ids_cat, strategy_mappings
    assert nodes_cat is not None
    return nodes_cat, strategy_mappings
