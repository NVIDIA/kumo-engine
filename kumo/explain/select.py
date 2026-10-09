# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
from kumo_core.training.explain_selection import (  # noqa: F401
    EVAL_NUM_SELECTION_NODES,
    PRED_NUM_SELECTION_NODES,
    ClassificationSelectionStrategy,
    LinkPredictionSelectionStrategy,
    RegressionSelectionStrategy,
    select_classification,
    select_input_nodes,
    select_link_prediction,
    select_regression,
)
from kumoapi.train import TrainingStage
from torch import Tensor
from torch.utils.data import DataLoader


def get_explain_node_inputs(
    datamodule: DiskgraphLightningDataModule,
    stage: TrainingStage,
) -> tuple[Tensor | None, Tensor | None]:
    r"""Get input nodes and ids for entity-level XAI.

    Args:
        datamodule (DiskgraphLightningDataModule): The datamodule.
        stage (TrainingStage): The evaluation or prediction stage.

    Returns:
        A tuple of tensors where the first item is mainly from
        the :obj:`input_nodes` and the second item is :obj:`input_id`
        for node prediction tasks in evaluation stage. Otherwise
        this will be :obj:`None`.
    """
    input_dict = get_data_dict(datamodule, stage)
    # Keys are populated by ``DLDM.stage_inputs``.
    input_nodes = input_dict.get('input_nodes')
    input_id = input_dict.get('input_id')
    return input_nodes, input_id


def get_data_dict(
    datamodule: DiskgraphLightningDataModule,
    stage: TrainingStage,
) -> dict[str, Any]:
    r"""Get the datamodule input data in dictionary."""
    input_type, input_nodes, input_time, input_id = (datamodule.stage_inputs(
        stage.value))
    return dict(input_type=input_type, input_nodes=input_nodes,
                input_time=input_time, input_id=input_id)


def get_eval_dataloader(datamodule: DiskgraphLightningDataModule,
                        stage: TrainingStage, seed_nodes: Tensor) -> DataLoader:
    """Get evaluation dataloader for entity-level XAI.

    Args:
        datamodule (DiskgraphLightningDataModule): Datamodule.
        stage (TrainingStage): Training stage, either val or test.
        seed_nodes (Tensor): Seed input nodes in ascending order.
            For node prediction task during evaluation stage,
            :obj:`input_id` is used instead of the :obj:`input_nodes`
            to avoid duplications.
    """
    # ``seed_nodes`` are TT-NG stage-table ids selected from the captured
    # eval prepass, so the datamodule can replay only those rows in the wheel.
    return datamodule.explain_dataloader(stage.value, seed_nodes)
