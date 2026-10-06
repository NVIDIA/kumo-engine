# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from typing import Any

import torch
from kumo_core.train.base_module import (
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_INTERVAL,
    DEFAULT_OPTIMIZER,
)
from kumo_core.train.base_node_module import BaseNodeLightningModule
from kumo_core.train.loss import DistributionLoss, MultiQuantileLoss
from kumo_core.train.weighting import build_binary_class_weight
from kumo_core.types import TaskType, TrainModuleOutput
from kumoapi.model_plan import WeightMode
from kumoapi.train import TrainingStage
from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.modeling.gnn.transformer import GraphTransformer
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType

logger = logging.getLogger(__name__)


class NodeGNNLightningModule(BaseNodeLightningModule):
    r"""A PyTorch Lightning module that handles the execution of GNN-based
    node-level training, validation, testing and prediction routines.

    Args:
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The message-passing model.
        head (torch.nn.Module): The model head.
        train_table_name (str): The name of the training table.
        entity_table_name (str): The name of the entity table.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        model: torch.nn.Module,
        head: torch.nn.Module,
        task_type: TaskType,
        train_table_name: str,
        entity_table_name: str,
        # `BaseNodeLightningModule` arguments:
        loss: str | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        target_encoder: TargetEncoder | None = None,
        metrics: list[str] | None = None,
        optimizer: str = DEFAULT_OPTIMIZER,
        optimizer_kwargs: dict[str, Any] | None = None,
        majority_sampling_weights: dict[bool, float] | None = None,
        weight_mode: WeightMode | None = None,
        lr_scheduler: str | None = DEFAULT_LR_SCHEDULER,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_interval: str = DEFAULT_LR_SCHEDULER_INTERVAL,
        max_epochs_for_lr_scheduler: int = 100,
        enable_compile: bool = False,
    ) -> None:
        super().__init__(
            task_type=task_type,
            loss=loss,
            loss_kwargs=loss_kwargs,
            target_encoder=target_encoder,
            metrics=metrics,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_interval=lr_scheduler_interval,
            max_epochs_for_lr_scheduler=max_epochs_for_lr_scheduler,
            enable_compile=enable_compile,
        )

        self.encoder = encoder
        self.model = model
        self.head = head
        self.train_table_name = train_table_name
        self.entity_table_name = entity_table_name
        self.majority_sampling_weights = majority_sampling_weights
        self.weight_mode = weight_mode
        # Emit this info-level message once to avoid per-step log noise.
        self._has_logged_fused_loss_weights = False

        if self.model.channels != self.encoder.out_channels:
            logger.warning("The model output channels do not match the "
                           "encoder output channels. The entity features "
                           "are not enhanced by training table features.")

    # Model Execution #########################################################

    def forward(
        self,
        tf_dict: dict[NodeType, TensorFrame],
        edge_index_dict: dict[EdgeType, Tensor],
        time_dict: dict[NodeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
        seed_time: Tensor | None = None,
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
        return_embeddings: bool = False,
    ) -> Tensor:
        x_dict = self.encoder(tf_dict, batch_dict, time_dict, seed_time)
        x_train_table = x_dict[self.train_table_name]
        batch_size = x_train_table.size(0)
        del x_dict[self.train_table_name]

        if isinstance(self.model, GraphTransformer):
            # x shape: [batch_size, seq_len, channels]
            if self.train_table_name in num_sampled_nodes_dict:
                del num_sampled_nodes_dict[self.train_table_name]
            x = self.model(
                x_dict,
                edge_index_dict,
                batch_dict=batch_dict,
                batch_size=batch_size,
                entity_table_name=self.entity_table_name,
                num_sampled_nodes_dict=num_sampled_nodes_dict,
            )
            x = x[:, 0]  # target node is in the first position of every batch

        else:
            x_dict = self.model(
                x_dict,
                edge_index_dict,
                num_sampled_nodes_dict,
                num_sampled_edges_dict,
            )
            x = x_dict[self.entity_table_name][:batch_size]

        if return_embeddings:
            # TODO: Return embeddings of other node types as well
            return x

        return self.head(x, seed_time)

    def common_step(
        self,
        batch: HeteroData,
        return_embeddings: bool = False,
    ) -> Tensor:
        return self(
            batch.tf_dict,
            batch.collect('edge_index', allow_empty=True),
            batch.collect('time', allow_empty=True),
            batch.batch_dict,
            seed_time=batch[self.train_table_name].get('time'),
            num_sampled_nodes_dict=batch.collect('num_sampled_nodes', True),
            num_sampled_edges_dict=batch.collect('num_sampled_edges', True),
        )

    def get_embedding(self, batch, entity_name, side=None):
        return self(
            batch.tf_dict,
            batch.collect('edge_index', allow_empty=True),
            batch.collect('time', allow_empty=True),
            batch.batch_dict,
            seed_time=batch[entity_name].get('seed_time'),
            num_sampled_nodes_dict=batch.collect('num_sampled_nodes',
                                                 allow_empty=True),
            num_sampled_edges_dict=batch.collect('num_sampled_edges',
                                                 allow_empty=True),
            return_embeddings=True,
        )

    # Training Callbacks ######################################################

    @staticmethod
    def _output_dtype(
        out: Tensor | dict[str, Tensor],
        fallback: Tensor,
    ) -> torch.dtype:
        if isinstance(out, Tensor):
            return out.dtype
        for value in out.values():
            if isinstance(value, Tensor):
                return value.dtype
        return fallback.dtype

    def train_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        out = self.common_step(batch)
        assert self.target_encoder is not None
        y = self.target_encoder.encode_tensor(batch[self.train_table_name].y)
        out_dtype = self._output_dtype(out, y)

        weight: Tensor | None = None
        if self.weight_mode is not None and self.weight_mode.use_weighted_loss:
            row_weight: Tensor | None = None
            if 'weight' in batch[self.train_table_name]:
                row_weight = batch[self.train_table_name].weight.view(-1)

            class_weight: Tensor | None = None
            if self.majority_sampling_weights is not None:
                class_weight = build_binary_class_weight(
                    target=y,
                    class_weight_map=self.majority_sampling_weights,
                    out_dtype=out_dtype,
                )

            fused_weight: Tensor | None = None
            if row_weight is not None:
                fused_weight = row_weight.to(dtype=out_dtype).view(-1)
            if class_weight is not None:
                if fused_weight is not None:
                    fused_weight = fused_weight * class_weight
                else:
                    fused_weight = class_weight
            if fused_weight is not None:
                weight = fused_weight.view(-1, 1)

            if (row_weight is not None and class_weight is not None
                    and not self._has_logged_fused_loss_weights):
                logger.info("Using fused loss weights from weight_col and "
                            "majority_sampling.")
                self._has_logged_fused_loss_weights = True

        loss = self.loss(out, y, weight=weight)

        loss_name = TrainingStage.TRAIN.add_stage_prefix('loss')

        self.log_learning_rate()
        self.log_epoch(loss_name, loss, batch_size=y.size(0))

        if isinstance(self.loss, DistributionLoss):
            out = self.loss.distribution.to_prediction(out)  # type: ignore
        elif isinstance(self.loss, MultiQuantileLoss):
            out = self.loss.to_prediction(out)

        self.update_metrics(TrainingStage.TRAIN, out, y)

        return loss

    # Evaluation Callbacks ####################################################

    def evaluation_step(
        self,
        batch: HeteroData,
        stage: TrainingStage,
        batch_idx: int,
    ) -> TrainModuleOutput:
        out = self.common_step(batch)
        batch_size = batch[self.entity_table_name].batch_size
        pkey = batch[self.entity_table_name].pkey[:batch_size]
        time = batch[self.train_table_name].get('time')
        assert self.target_encoder is not None
        y = self.target_encoder.encode_tensor(batch[self.train_table_name].y)
        loss = self.loss(out, y, weight=None)
        loss_name = stage.add_stage_prefix('loss')
        self.log_epoch(loss_name, loss, batch_size=y.size(0))

        distribution = None
        if isinstance(self.loss, DistributionLoss):
            distribution = out
            out = self.loss.to_prediction(out)  # type: ignore

        if not self.disable_eval_metrics:
            self.update_metrics(
                stage,
                self.loss.to_prediction(out) if isinstance(
                    self.loss, MultiQuantileLoss) else out, y)

        return dict(pkey=pkey, time=time, logits=out, target=y,
                    prob=distribution)

    # Prediction Callbacks ####################################################

    def predict_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:
        return_embeddings = getattr(self, '_kumo_return_predict_embeddings',
                                    False)
        embedding = None
        if return_embeddings:
            embedding = self(
                batch.tf_dict,
                batch.collect('edge_index', allow_empty=True),
                batch.collect('time', allow_empty=True),
                batch.batch_dict,
                seed_time=batch[self.train_table_name].get('time'),
                num_sampled_nodes_dict=batch.collect('num_sampled_nodes', True),
                num_sampled_edges_dict=batch.collect('num_sampled_edges', True),
                return_embeddings=True,
            )
            out = self.head(embedding, batch[self.train_table_name].get('time'))
        else:
            out = self.common_step(batch)
        batch_size = batch[self.entity_table_name].batch_size
        pkey = batch[self.entity_table_name].pkey[:batch_size]
        time = batch[self.train_table_name].get('time')

        result = dict(pkey=pkey, time=time, logits=out)
        if embedding is not None:
            result['embedding'] = embedding.detach().cpu()
        return result

    # Embedding Callbacks #####################################################

    def _get_register_embedding_hook_fn(self) -> Callable | None:
        return self.head.register_forward_pre_hook

    def embedding_step(
        self,
        stage: TrainingStage,
        module: torch.nn.Module,
        inputs: tuple[Tensor, ...],
    ) -> dict[str, Tensor]:
        return {self.entity_table_name: inputs[0]}
