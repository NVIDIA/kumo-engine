# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os.path as osp
from collections import defaultdict
from typing import Any, Literal

import torch
from kumoapi.explain import GraphGradientScore
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import scatter

from kumo_core.explain.base import Explainer, continue_on_error
from kumo_core.io.csv import save_csv
from kumo_core.io.json import save_json


class GradientExplainer(Explainer):
    r"""Gradient-based explanation."""
    supported_task_types = {
        TaskType.BINARY_CLASSIFICATION,
        TaskType.MULTICLASS_CLASSIFICATION,
        TaskType.MULTILABEL_CLASSIFICATION,
        TaskType.REGRESSION,
        TaskType.STATIC_LINK_PREDICTION,
        TaskType.TEMPORAL_LINK_PREDICTION,
        TaskType.FORECASTING,
    }
    requires_grad: bool = True

    def __init__(
        self,
        task_type: TaskType,
        root_dir: str,
        formats: list[Literal['json', 'csv']] | None = None,
        normalize: bool = True,
    ) -> None:
        super().__init__(task_type, root_dir)
        self.formats = formats or ['json']
        self.normalize = normalize

    def explain_start(
        self,
        module: torch.nn.Module,
        datamodule: Any,
        stage: TrainingStage,
    ) -> None:
        self._table_score_dict: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float))
        self._table_count_dict: dict[str, dict[str, int]] = defaultdict(  #
            lambda: defaultdict(int))

    def explain_pre_step(
        self,
        batch: HeteroData,
        module: torch.nn.Module,
        datamodule: Any,
        stage: TrainingStage,
        batch_idx: int,
    ) -> None:
        if module.training:
            raise RuntimeError(f"Execution of '{self.__class__.__name__}' is "
                               f"only applicable during model inference")

    @continue_on_error
    def explain_post_step(
        self,
        batch: HeteroData,
        module: torch.nn.Module,
        datamodule: Any,
        step_output: dict[str, Tensor],
        stage: TrainingStage,
        batch_idx: int,
    ) -> None:
        table_grad_dict = self.get_grad(  #
            self.task_type, batch.tf_dict, module, step_output)

        # Aggregate gradients into column-level scores ########################

        # NOTE Aggregating row-level scores into column-level scores is tricky
        # since we can't assume they are evenly distributed across nodes.
        # Imagine that only the latest fact(s) of a table have an impact on
        # model performance: Intuitively, we would expect only the latest facts
        # have large gradients, while the remaining ones have a gradient close
        # to zero. As such, simply averaging the scores across the full
        # neighborhood would heavily underestimate a column's contribution.
        # As such, we utilize a weighted aggregation to be able to still rank
        # columns high, even if only a few of their nodes have a high score.
        # Furthermore, we apply a two-stage aggregation: We first aggregate
        # scores within examples, and then average scores across examples:

        for table_name, grad_dict in table_grad_dict.items():
            grad = torch.stack(list(grad_dict.values()), dim=1)

            # (1) Weighted aggregation of gradients per example:
            batch_vec = batch[table_name].batch
            accum_grad = scatter(grad, batch_vec, reduce='sum')
            weight = grad / accum_grad[batch_vec]

            # Handle division by zero (in case `accum_grad==0`):
            weight[weight.isnan() | weight.isinf()] = 1.0
            grad = scatter(grad * weight, batch_vec, reduce='sum')

            # (2) Mean aggregation of gradients across examples:
            for i, col_name in enumerate(grad_dict.keys()):
                col_grad = float(grad[:, i].sum())
                self._table_score_dict[table_name][col_name] += col_grad
                self._table_count_dict[table_name][col_name] += grad.size(0)

    @continue_on_error
    def explain_end(
        self,
        module: torch.nn.Module,
        datamodule: Any,
        outputs: dict[str, Tensor],
        stage: TrainingStage,
    ) -> None:
        # Save column-level gradient score ####################################

        out = GraphGradientScore()

        for table_name, score_dict in self._table_score_dict.items():
            for col_name, score in score_dict.items():
                count = self._table_count_dict[table_name][col_name]
                score = score / count if count > 0 else 0.0
                out[table_name][col_name] = score

        if self.normalize:
            out = out.normalize()

        for format in self.formats:
            path = osp.join(self.root_dir, f'scores.{format}')
            if format == 'json':
                save_json(out, path)
            elif format == 'csv':
                save_csv(out.to_pandas(), path, overwrite=True)
