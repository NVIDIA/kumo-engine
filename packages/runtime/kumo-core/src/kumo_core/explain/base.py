# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import os
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from kumoapi.task import TaskType
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData
from torch_geometric.utils import one_hot

import logging
from kumo_core.encoders.tensor_mapper import CategoricalTensorMapper, MultiCategoricalTensorMapper, TensorMapper
from kumo_core.encoders.target.base import TargetEncoder

logger = logging.getLogger(__name__)


def continue_on_error(func: Callable) -> Callable:
    r"""A decorator to not fail training/prediction in case explainability
    fails.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Fail in tests, unless explicitly turned on or off:
            in_test = 'PYTEST_CURRENT_TEST' in os.environ
            raise_on_error = os.getenv('KUMO_EXPLAINER_RAISE_ON_ERROR')
            continue_on_error = os.getenv('KUMO_EXPLAINER_CONTINUE_ON_ERROR')
            if raise_on_error == 'true':
                raise e
            if in_test and continue_on_error != 'true':
                raise e
            logger.exception(f"Explainer failed in '{func.__qualname__}' "
                             f"with error message: {e}")

    return wrapper


class Explainer:
    r"""Base class for implementing step-wise explainers.

    Args:
        task: The task for which explanations are computed.
        root_dir: The directory to write explanations to.
    """
    supported_task_types: set[TaskType]
    requires_grad: bool = False

    def __init__(
        self,
        task_type: TaskType,
        root_dir: str,
    ) -> None:
        if task_type not in self.supported_task_types:
            raise NotImplementedError(
                f"The task type '{task_type}' is not yet supported in "
                f"'{self.__class__.__name__}'")

        self.task_type = task_type
        self.root_dir = root_dir

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

    def explain_start(
        self,
        module: torch.nn.Module,
        datamodule: Any,
        stage: TrainingStage,
    ) -> None:
        r"""Callback before model starts ``TrainingStage``."""

    def explain_pre_step(
        self,
        batch: HeteroData,
        module: torch.nn.Module,
        datamodule: Any,
        stage: TrainingStage,
        batch_idx: int,
    ) -> None:
        r"""Callback before each model step in ``TrainingStage``."""

    def explain_post_step(
        self,
        batch: HeteroData,
        module: torch.nn.Module,
        datamodule: Any,
        step_output: dict[str, Tensor],
        stage: TrainingStage,
        batch_idx: int,
    ) -> None:
        r"""Callback after each model step in ``TrainingStage``."""

    def explain_end(
        self,
        module: torch.nn.Module,
        datamodule: Any,
        outputs: dict[str, Tensor],
        stage: TrainingStage,
    ) -> None:
        r"""Callback after model has finished ``TrainingStage``."""

    def transform_prediction(
        self,
        stage: TrainingStage,
        outputs: dict[str, Tensor],
        num_nodes: int | None = None,
        max_classes: int | None = None,
        target_encoder: TargetEncoder | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        r"""Transforms predictions and ground truth (if available) to tensors
        with proper and original scale.
        """
        target: Tensor | None = None
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            if outputs.get('prob', None) is not None:
                prob = outputs['prob'][:, 1]
            else:
                prob = outputs['logits'].sigmoid()

            if num_nodes is not None:
                prob = prob[:num_nodes]

            if stage != TrainingStage.PRED:
                target = outputs['target'].to(prob.dtype).view(-1)
                if num_nodes is not None:
                    target = target[:num_nodes]

            return prob, target

        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            if outputs.get('prob', None) is not None:
                prob = outputs['prob']
            else:
                prob = outputs['logits'].softmax(dim=-1)

            if num_nodes is not None:
                prob = prob[:num_nodes]

            if stage != TrainingStage.PRED:
                target = outputs['target'].view(-1)
                if num_nodes is not None:
                    target = target[:num_nodes]
                target = one_hot(target, prob.size(1), dtype=prob.dtype)
                if max_classes is not None:
                    assert target is not None
                    target = target[:, :max_classes]

            if max_classes is not None:
                prob = prob[:, :max_classes]

            return prob, target

        if self.task_type == TaskType.MULTILABEL_CLASSIFICATION:
            if outputs.get('prob', None) is not None:
                prob = outputs['prob']
            else:
                prob = outputs['logits'].sigmoid()

            if num_nodes is not None:
                prob = prob[:num_nodes]
            if max_classes is not None:
                prob = prob[:, :max_classes]

            if stage != TrainingStage.PRED:
                target = outputs['target'].to(prob.dtype)
                if num_nodes is not None:
                    target = target[:num_nodes]
                if max_classes is not None:
                    target = target[:, :max_classes]

            return prob, target

        elif self.task_type == TaskType.REGRESSION:
            assert target_encoder is not None
            if 'pred' in outputs:
                pred = outputs['pred'].view(-1)
            else:
                pred = outputs['logits'].view(-1)

            if num_nodes is not None:
                pred = pred[:num_nodes]

            pred = target_encoder.decode_tensor(pred)

            if stage != TrainingStage.PRED:
                target = outputs['target'].view(-1)
                if num_nodes is not None:
                    target = target[:num_nodes]
                target = target_encoder.decode_tensor(target)

            return pred, target

        raise NotImplementedError(
            f"The task type '{self.task_type}' is not yet supported "
            f"in 'transform_prediction'")

    def to_human_readable_prediction(
        self,
        outputs: Tensor,
        tensor_mapper: TensorMapper | None = None,
    ) -> list[float] | dict[str, list[float]]:
        r"""Converts outputs to human-readable Python built-in types."""
        if self.task_type in {
                TaskType.MULTICLASS_CLASSIFICATION,
                TaskType.MULTILABEL_CLASSIFICATION,
        }:
            assert isinstance(
                tensor_mapper,
                CategoricalTensorMapper | MultiCategoricalTensorMapper,
            )
            return {
                str(tensor_mapper.mapping[i]): outputs[:, i].tolist()
                for i in range(outputs.size(1))
            }

        return outputs.view(-1).tolist()

    @staticmethod
    @torch.enable_grad()
    def get_grad(
        task_type: TaskType,
        tf_dict: dict[str, TensorFrame],
        module: torch.nn.Module,
        step_output: dict[str, Tensor],
    ) -> dict[str, dict[str, Tensor]]:

        if task_type in {
                TaskType.BINARY_CLASSIFICATION,
                TaskType.MULTICLASS_CLASSIFICATION,
        }:
            step_output['logits'].max(dim=1)[0].sum().backward()
        elif 'logits' in step_output:
            step_output['logits'].sum().backward()
        else:
            step_output['score'].sum().backward()

        grad_dict: dict[str, dict[str, Tensor]] = defaultdict(dict)
        for table_name, tf in tf_dict.items():
            # Non-sampled node types will never receive gradients:
            if tf.num_rows == 0:
                continue

            assert isinstance(module.encoder.enc_dict, torch.nn.ModuleDict)
            tf_encoder = module.encoder.enc_dict[table_name]
            for stype, encoder in tf_encoder.stype_encoder_dict.items():
                # Do not raise error when receiving gradients, e.g., in case
                # the number of GNN layers is less than sampled hops.
                grad = encoder.get_grad_wrt_input(raise_error=False)
                if grad is None:
                    continue

                col_names = tf.col_names_dict[stype]
                for i, col_name in enumerate(col_names):
                    grad_dict[table_name][col_name] = grad[:, i]

            rel_time_encoder = tf_encoder.rel_time_encoder
            if rel_time_encoder is not None:
                grad = rel_time_encoder.get_grad_wrt_input(raise_error=False)
                if grad is not None:
                    time_cols = tf.col_names_dict.get(Stype.timestamp) or []
                    if (tf_encoder.time_col is not None
                            and tf_encoder.time_col not in time_cols):
                        time_cols = [tf_encoder.time_col] + time_cols
                    for i, time_col in enumerate(time_cols):
                        if time_col not in grad_dict[table_name]:
                            grad_dict[table_name][time_col] = grad[:, i]
                        else:
                            grad_dict[table_name][time_col] += grad[:, i]
        module.zero_grad()

        return grad_dict
