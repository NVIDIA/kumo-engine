# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from typing import Any, Literal

import kumo_core.modeling.context as modeling_context
import numpy as np
import torch
import torch.nn.functional as F
from kumo_core.train.base_link_module import (
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_INTERVAL,
    DEFAULT_OPTIMIZER,
    BaseLinkLightningModule,
)
from kumo_core.types import TaskType
from kumoapi.model_plan import WeightMode
from kumoapi.train import TrainingStage
from kumo_core.modeling.heads.link_prediction import LinkPredHead
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric import is_torch_instance
from torch_geometric.data import HeteroData
from torch_geometric.metrics.link_pred import LinkPredMetricData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import to_dense_batch

logger = logging.getLogger(__name__)


class IDLinkGNNLightningModule(BaseLinkLightningModule):
    r"""A PyTorch Lightning module that handles the execution of GNN-based
    link-level training, validation, testing and prediction routines.
    :class:`IDLinkGNNLightningModule` learns pair representations from a graph
    by a source-specific message passing process, where every node learns a
    representation conditioned on the source node.
    The underlying implementation is inspired by the `"Neural Bellman-Ford
    Networks: A General Graph Neural Network Framework for Link Prediction"
    <https://arxiv.org/abs/2106.06935>`_ paper.

    Args:
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The GNN model.
        head (LinkPredHead): The model head.
        train_table_name (str): The name of the training table.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        model: torch.nn.Module,
        head: LinkPredHead,
        train_table_name: str,
        # `BaseLinkLightningModule` arguments:
        entity_table_names: tuple[str, str],
        metrics: list[str] | None = None,
        optimizer: str = DEFAULT_OPTIMIZER,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: str | None = DEFAULT_LR_SCHEDULER,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_interval: str = DEFAULT_LR_SCHEDULER_INTERVAL,
        max_epochs_for_lr_scheduler: int = 100,
        enable_compile: bool = False,
        distance_measure: Literal['dot_product', 'cosine'] = 'dot_product',
        weight_mode: WeightMode | None = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            model=model,
            head=head,
            task_type=TaskType.TEMPORAL_LINK_PREDICTION,
            entity_table_names=entity_table_names,
            metrics=metrics,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_interval=lr_scheduler_interval,
            max_epochs_for_lr_scheduler=max_epochs_for_lr_scheduler,
            enable_compile=enable_compile,
            distance_measure=distance_measure,
            weight_mode=weight_mode,
        )

        self.train_table_name = train_table_name

        self.identity_awareness = torch.nn.Embedding(1, model.channels)

    # Model Execution #########################################################

    def forward(
        self,
        tf_dict: dict[NodeType, TensorFrame],
        edge_index_dict: dict[EdgeType, Tensor],
        time_dict: dict[NodeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
        seed_time: Tensor | None,
        batch_size: int,
    ) -> Tensor:
        # TODO (matthias) The train table shouldn't be part of the mini-batch.
        tf_dict = copy.copy(tf_dict)
        batch_dict = copy.copy(batch_dict)
        tf_dict.pop(self.train_table_name, None)
        batch_dict.pop(self.train_table_name, None)
        x_dict = self.encoder(tf_dict, batch_dict, time_dict, seed_time)

        # Add identity-awareness to LHS seed nodes:
        x = x_dict[self.lhs_name].clone()
        x[:batch_size] += self.identity_awareness.weight
        x_dict[self.lhs_name] = x

        x_dict = self.model(x_dict, edge_index_dict)

        lhs_embedding = x_dict[self.lhs_name][:batch_size]
        lhs_batch = batch_dict[self.rhs_name]

        if self.distance_measure == 'cosine':
            lhs_embedding[lhs_batch] = F.normalize(lhs_embedding[lhs_batch],
                                                   dim=1)
            x_dict[self.rhs_name] = F.normalize(x_dict[self.rhs_name], dim=1)
        id_out = self.head.mlp_head(x_dict[self.rhs_name])
        emb_out = (lhs_embedding[lhs_batch] *
                   x_dict[self.rhs_name]).sum(dim=-1).view(-1, 1)

        return id_out + emb_out

    def common_step(self, batch: HeteroData) -> tuple[Tensor, Tensor, Tensor]:
        logits = self(
            batch.tf_dict,
            batch.collect('edge_index', allow_empty=True),
            batch.collect('time', allow_empty=True),
            batch.batch_dict,
            seed_time=batch[self.lhs_name].get('seed_time'),
            batch_size=batch[self.lhs_name].batch_size,
        )

        rhs_index = batch[self.rhs_name].n_id
        lhs_batch = batch[self.rhs_name].batch

        return logits, rhs_index, lhs_batch

    def get_rhs_pred_index_mat(
        self,
        logits: Tensor,
        rhs_pred_index: Tensor,
        lhs_pred_batch: Tensor,
        lhs_y_batch: Tensor,
    ):
        k = self.metric_store.get_metric_collection(TrainingStage.TRAIN).max_k

        # Sort logits batch-wise according to `lhs_pred_batch`:
        perm = np.lexsort((  # TODO (matthias) CPU-computation :(
            -logits.view(-1).detach().cpu().numpy(),
            lhs_pred_batch.cpu().numpy(),
        ))
        perm = torch.from_numpy(perm).to(logits.device)

        # TODO (matthias) Fill short rows with the most popular RHS nodes.
        #
        # Pad empty slots with ``num_rhs_nodes`` (out-of-range), NOT -1:
        # LinkPredMetricData encodes (row, rhs_id) as row*max_index+rhs_id, so a
        # -1 pad collides with a real edge and inflates recall/MAP.
        rhs_pred_index_mat, _ = to_dense_batch(
            rhs_pred_index[perm],
            lhs_pred_batch[perm],
            max_num_nodes=k,
            batch_size=int(lhs_y_batch.max()) + 1,
            fill_value=self.num_rhs_nodes,
        )

        return rhs_pred_index_mat

    # Training Callbacks ######################################################

    def train_step(self, batch: HeteroData, batch_idx: int) -> Tensor:

        logits, rhs_pred_index, lhs_pred_batch = self.common_step(batch)
        rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(batch)

        # For now, we use `torch.isin()` to search for overlaps between
        # ground-truth RHS node indices and sampled RHS node indices.
        # TODO (matthias) Make more efficient via sparse matrix arithmetics.
        index = rhs_pred_index + lhs_pred_batch * self.num_rhs_nodes
        y_index = rhs_y_index + lhs_y_batch * self.num_rhs_nodes
        mask = torch.isin(index, y_index)
        y = mask.to(logits.dtype)

        if edge_weight is None or self.weight_mode != WeightMode.WEIGHTED_LOSS:
            loss = F.binary_cross_entropy_with_logits(logits.view(-1),
                                                      y.view(-1))
        else:
            edge_weight_full = edge_weight.new_ones(len(logits))
            edge_weight_full[mask] = edge_weight[torch.isin(y_index, index)]
            loss = F.binary_cross_entropy_with_logits(logits.view(-1),
                                                      y.view(-1),
                                                      weight=edge_weight_full)
        loss_name = TrainingStage.TRAIN.add_stage_prefix('loss')

        self.log_learning_rate()
        self.log_epoch(loss_name, loss, batch_size=y.size(0))
        rhs_pred_index_mat = self.get_rhs_pred_index_mat(
            logits, rhs_pred_index, lhs_pred_batch, lhs_y_batch)
        self.update_metrics(TrainingStage.TRAIN, rhs_pred_index_mat,
                            (lhs_y_batch, rhs_y_index),
                            edge_label_weight=edge_weight)
        return loss

    # Evaluation Callbacks ####################################################

    def evaluation_step(
        self,
        batch: HeteroData,
        stage: TrainingStage,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:

        logits, rhs_pred_index, lhs_pred_batch = self.common_step(batch)
        rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(batch)

        # TODO (matthias) Make more efficient via sparse matrix arithmetics.
        index = rhs_pred_index + lhs_pred_batch * self.num_rhs_nodes
        y_index = rhs_y_index + lhs_y_batch * self.num_rhs_nodes
        y = torch.isin(index, y_index).to(logits.dtype)

        loss = F.binary_cross_entropy_with_logits(logits.view(-1), y.view(-1))
        loss_name = stage.add_stage_prefix('loss')

        self.log_epoch(loss_name, loss, batch_size=y.size(0))

        if not self.disable_eval_metrics:
            rhs_pred_index_mat = self.get_rhs_pred_index_mat(
                logits, rhs_pred_index, lhs_pred_batch, lhs_y_batch)
            self.update_metrics(stage, rhs_pred_index_mat,
                                (lhs_y_batch, rhs_y_index),
                                edge_label_weight=edge_weight)

        if modeling_context.is_explaining() or self.eval_top_k > 0:
            eval_top_k = min(self.eval_top_k, self.num_rhs_nodes)

            batch_size = batch[self.lhs_name].batch_size
            lhs_pkey = batch[self.lhs_name].pkey[:batch_size]
            time = batch[self.lhs_name].seed_time
            lhs_index = batch[self.lhs_name].n_id[:batch_size]

            # Sort logits batch-wise according to `lhs_pred_batch`:
            logits_np = -logits.view(-1).detach().cpu().numpy()
            perm = np.lexsort((logits_np, lhs_pred_batch.cpu().numpy()))
            perm = torch.from_numpy(perm).to(logits.device)

            # TODO (matthias) In case a LHS node has less than `k` connections
            # to RHS nodes, we ideally fill up the remaining slots with the
            # most popular RHS nodes.
            target, _ = to_dense_batch(
                rhs_pred_index[perm],
                lhs_pred_batch[perm],
                max_num_nodes=eval_top_k,
                batch_size=batch[self.lhs_name].batch_size,
                fill_value=-1,
            )

            score, _ = to_dense_batch(
                logits[perm].sigmoid().view(-1),
                lhs_pred_batch[perm],
                max_num_nodes=eval_top_k,
                batch_size=batch[self.lhs_name].batch_size,
                fill_value=0.,
            )
            # Keep -1 ("no prediction") in the returned target, but pad a copy
            # with num_rhs_nodes for the metric — -1 would alias a real edge in
            # LinkPredMetricData (see get_rhs_pred_index_mat).
            metric_target = target.masked_fill(target == -1, self.num_rhs_nodes)
            data = LinkPredMetricData(metric_target, (lhs_y_batch, rhs_y_index))
            label = data.pred_rel_mat
            return dict(pkey=lhs_pkey, id=lhs_index, time=time, target=target,
                        score=score, label=label)

        return {}

    # Prediction Callbacks ####################################################

    def predict_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:

        assert self.pred_top_k is not None
        pred_top_k = min(self.pred_top_k, self.num_rhs_nodes)
        logits, rhs_index, lhs_batch = self.common_step(batch)

        batch_size = batch[self.lhs_name].batch_size
        lhs_pkey = batch[self.lhs_name].pkey[:batch_size]
        time = batch[self.lhs_name].seed_time
        lhs_index = batch[self.lhs_name].n_id[:batch_size]

        lhs_index = batch[self.lhs_name].n_id
        lhs_index = lhs_index[:batch[self.lhs_name].batch_size].cpu()

        # Detach logits in prediction as explainer may turn on gradients
        logits_np = -logits.view(-1).detach().cpu().numpy()
        # Sort logits batch-wise according to `lhs_batch`:
        perm = np.lexsort((logits_np, lhs_batch.cpu().numpy()))
        perm = torch.from_numpy(perm).to(logits.device)

        # TODO (matthias) In case a LHS node has less than `k` connections to
        # RHS nodes, we ideally fill up the remaining slots with the most
        # popular RHS nodes.
        target, _ = to_dense_batch(
            rhs_index[perm],
            lhs_batch[perm],
            max_num_nodes=pred_top_k,
            batch_size=batch[self.lhs_name].batch_size,
            fill_value=-1,
        )

        score, _ = to_dense_batch(
            logits[perm].sigmoid().view(-1),
            lhs_batch[perm],
            max_num_nodes=pred_top_k,
            batch_size=batch[self.lhs_name].batch_size,
            fill_value=0.,
        )

        return dict(pkey=lhs_pkey, id=lhs_index, time=time, target=target,
                    score=score)

    def validate_head(self):
        assert is_torch_instance(self.head, LinkPredHead)
        assert not self.head.normalize_vectors
        assert self.head.use_mlp_head
        assert not self.head.use_rhs_projector

    @property
    def supports_rhs_node_filtering(self):
        # We do not currently support RHS node filtering in ID-GNN.
        # TODO: Implement this when needed.
        return False
