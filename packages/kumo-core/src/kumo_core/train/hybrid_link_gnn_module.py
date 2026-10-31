# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from typing import Any, Literal

import kumo_core.modeling.context as modeling_context
import pandas as pd
import torch
import torch.nn.functional as F
from kumo_core.train.base_module import (
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_INTERVAL,
    DEFAULT_OPTIMIZER,
)
from kumo_core.types import TaskType
from kumoapi.model_plan import RHSEmbeddingMode, WeightMode
from kumoapi.train import TrainingStage
from kumo_core.modeling.heads.link_prediction import LinkPredHead
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric import is_torch_instance
from torch_geometric.data import HeteroData
from torch_geometric.metrics.link_pred import LinkPredMetricData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils import lexsort, to_dense_batch
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from torch_geometric.utils.map import map_index

from .shallow_rhs_link_gnn_module import ShallowRHSLinkGNNLightningModule

logger = logging.getLogger(__name__)


# TODO (matthias) Drop inheritance pattern.
class HybridLinkGNNLightningModule(ShallowRHSLinkGNNLightningModule):
    r"""A PyTorch Lightning module that handles the execution of GNN-based
    link-level training, validation, testing and prediction routines.
    :class:`HybridLinkGNNLightningModule` combines
    :class:`IDLinkGNNLightningModule`
    and :class:`ShallowRHSLinkGNNLightningModule`.
    Args:
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The GNN model.
        head (LinkPredHead): The model head.
        train_table_name (str): The name of the training table.
        entity_table_names ((str, str)): The names of the entity tables.
        num_rhs_nodes (int): number of rhs nodes.
        rhs_emb_init_std (float): std for rhs embedding initialization
        emb_l2_reg_coeff (float): L2 regularization on the final
            embeddings. (default: :obj:`0.0`)
        emb_loss_coeff (float): The coefficient of embedding loss applied to
            train Hybrid GNN. If a user cares about embedding-based prediction,
            the recommended value is 0.2. (default: :obj:`0.`)
        rhs_sample_size (int, optional): The number of rhs nodes to sample
            during training. If set to None, full rhs nodes will be used
            for loss computation. (default: :obj:`None`)
        rhs_emb_mode (RHSEmbeddingMode): How to model rhs features.
            :obj:`RHSEmbeddingMode.LOOKUP`, :obj:`RHSEmbeddingMode.FEATURE`,
            :obj:`RHSEmbeddingMode.SHALLOW_FEATURE`, or
            :obj:`RHSEmbeddingMode.FUSION`.
            (default: :obj:`RHSEmbeddingMode.FUSION`)
        eval_new_visit (bool): Whether to focus on evaluation on new visits.
            (default: :obj:`False`)
        distance_measure: The measure method of comparing semantic similarity.
            If `'dot_product'`, all output vectors are normalized.
        loss: The loss function to use.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        model: torch.nn.Module,
        head: LinkPredHead,
        train_table_name: str,
        entity_table_names: tuple[str, str],
        num_rhs_nodes: int,
        # Important to initialize with small values so that ID-GNN
        # prediction is respected at the beginning of training.
        rhs_emb_init_std: float = 0.01,
        emb_l2_reg_coeff: float = 0.0,
        emb_loss_coeff: float = 0.,
        rhs_sample_size: int | None = None,
        rhs_emb_mode: RHSEmbeddingMode | str = RHSEmbeddingMode.FUSION,
        rhs_feat_bytes_threshold: int | None = None,
        eval_new_visit: bool = False,
        # lower dimensionality to support fast inner product
        rhs_index_map: pd.DataFrame | None = None,
        distance_measure: Literal['dot_product', 'cosine'] = 'dot_product',
        loss: Literal[
            'cross_entropy_loss',
            'binary_cross_entropy_loss',
        ] = 'cross_entropy_loss',
        # `BaseLightningModule` arguments:
        metrics: list[str] | None = None,
        optimizer: str = DEFAULT_OPTIMIZER,
        optimizer_kwargs: dict[str, Any] | None = None,
        lookup_optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: str | None = DEFAULT_LR_SCHEDULER,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_interval: str = DEFAULT_LR_SCHEDULER_INTERVAL,
        max_epochs_for_lr_scheduler: int = 100,
        enable_compile: bool = False,
        weight_mode: WeightMode | None = None,
    ) -> None:
        hidden_dim: int = model.channels  # type: ignore
        super().__init__(
            encoder=encoder,
            model=model,
            head=head,
            task_type=TaskType.TEMPORAL_LINK_PREDICTION,
            train_table_name=train_table_name,
            entity_table_names=entity_table_names,
            num_rhs_nodes=num_rhs_nodes,
            rhs_emb_init_std=rhs_emb_init_std,
            emb_l2_reg_coeff=emb_l2_reg_coeff,
            rhs_sample_size=rhs_sample_size,
            rhs_emb_mode=rhs_emb_mode,
            rhs_feat_bytes_threshold=rhs_feat_bytes_threshold,
            eval_new_visit=eval_new_visit,
            rhs_index_map=rhs_index_map,
            metrics=metrics,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lookup_optimizer_kwargs=lookup_optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_interval=lr_scheduler_interval,
            max_epochs_for_lr_scheduler=max_epochs_for_lr_scheduler,
            enable_compile=enable_compile,
            distance_measure=distance_measure,
            loss=loss,
            weight_mode=weight_mode,
        )

        # Learning the importance between the predictions of ID-GNN
        # and emb-GNN
        self.emb_loss_coeff = emb_loss_coeff
        self.lin_offset_idgnn = torch.nn.Linear(hidden_dim, 1)
        self.lin_offset_embgnn = torch.nn.Linear(hidden_dim, 1)

    def _maybe_compile(self) -> None:
        if not self.enable_compile:
            return

        is_sparse = getattr(self.rhs_embedding.lookup_embedding, 'sparse',
                            False)
        self.rhs_embedding = torch.compile(
            self.rhs_embedding,
            disable=is_sparse,
        )
        self.compute_loss = torch.compile(self.compute_loss, dynamic=True)
        self.postprocess_step = torch.compile(
            self.postprocess_step,
            dynamic=True,
        )
        self.sampled_postprocess_step = torch.compile(
            self.sampled_postprocess_step,
            dynamic=True,
        )

    # Model Execution #########################################################

    def forward(
        self,
        tf_dict: dict[NodeType, TensorFrame],
        node_id_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        time_dict: dict[NodeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
        seed_time: Tensor | None,
        batch_size: int,
        # Always `None` for HybridLinkGNNLightningModule:
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Returns lhs_embedding, rhs_gnn_embedding, rhs_idgnn_index,
        # lhs_idgnn_batch
        # Note: Returned lhs_embedding is unprojected.
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

        # Add rhs-awareness of lookup features:
        if self.rhs_embedding.lookup_embedding is not None:
            assert self.rhs_projector is not None
            n_id = node_id_dict[self.rhs_name]
            emb = self.rhs_projector(self.rhs_embedding.lookup_embedding(n_id))
            x_dict[self.rhs_name] = x_dict[self.rhs_name] + emb

        x_dict = self.model(x_dict, edge_index_dict)

        lhs_embedding = x_dict[self.lhs_name][:batch_size]
        rhs_gnn_embedding = x_dict[self.rhs_name]

        rhs_idgnn_index = node_id_dict[self.rhs_name]
        lhs_idgnn_batch = batch_dict[self.rhs_name]

        return (lhs_embedding, rhs_gnn_embedding, rhs_idgnn_index,
                lhs_idgnn_batch)

    def postprocess_step(
        self,
        lhs_embedding: Tensor,
        rhs_gnn_embedding: Tensor,
        rhs_idgnn_index: Tensor,
        lhs_idgnn_batch: Tensor,
        rhs_emb: Tensor,
        valid_rhs_node_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        # Calculate embgnn logits
        lhs_embedding_projected = self.lhs_projector(lhs_embedding)
        embgnn_logits = lhs_embedding_projected @ rhs_emb.t()

        rhs_idgnn_mask = None
        if valid_rhs_node_idx is not None:
            # Only keep the valid rhs_idgnn_index
            rhs_idgnn_index, rhs_idgnn_mask = map_index(
                rhs_idgnn_index,
                valid_rhs_node_idx,
                max_index=self.num_rhs_nodes,
                inclusive=False,
            )

        # Model the importance of embedding-GNN prediction for each lhs node
        embgnn_offset_logits = self.lin_offset_embgnn(lhs_embedding).flatten()
        embgnn_logits += embgnn_offset_logits.view(-1, 1)

        if rhs_idgnn_mask is not None:
            rhs_gnn_embedding = rhs_gnn_embedding[rhs_idgnn_mask]
            lhs_idgnn_batch = lhs_idgnn_batch[rhs_idgnn_mask]
        if self.distance_measure == 'cosine':
            rhs_gnn_embedding = F.normalize(rhs_gnn_embedding, dim=1)
            lhs_embedding = F.normalize(lhs_embedding, dim=1)

        # Calculate idgnn logits
        idgnn_logits = self.head.mlp_head(rhs_gnn_embedding).flatten()
        idgnn_logits = idgnn_logits + (lhs_embedding[lhs_idgnn_batch] *
                                       rhs_gnn_embedding).sum(dim=-1).flatten()
        # Model the importance of ID-GNN prediction for each lhs node
        idgnn_offset_logits = self.lin_offset_idgnn(lhs_embedding).flatten()
        idgnn_logits = idgnn_logits + idgnn_offset_logits[lhs_idgnn_batch]

        embgnn_logits[lhs_idgnn_batch, rhs_idgnn_index] = idgnn_logits
        return embgnn_logits, lhs_embedding_projected

    def sampled_postprocess_step(
        self,
        lhs_embedding: Tensor,
        rhs_gnn_embedding: Tensor,
        rhs_idgnn_index: Tensor,
        lhs_idgnn_batch: Tensor,
        rhs_y_index: Tensor,
        lhs_y_batch: Tensor,
        edge_weight: Tensor | None,
        valid_rhs_node_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor,
               Tensor, Tensor]:
        r"""Computes sampled logits and re-maps :obj:`rhs_y_index` and
        :obj:`lhs_y_batch` to valid range. Also returns the projected
        lhs embeddings for L2 regularization."""
        # Compute prediction over randomly sampled rhs node indices.
        # Used for scalable training.
        assert self.rhs_sample_size is not None

        # Sample a set of negative logits for the whole batch
        rnd = torch.rand(self.num_rhs_nodes, device=self.device)
        if valid_rhs_node_idx is not None:
            assert valid_rhs_node_idx.numel() > self.rhs_sample_size
            rnd[valid_rhs_node_idx] += 1.

        # Prioritize rhs_idgnn_index to be sampled.
        rnd[rhs_idgnn_index] = 3.
        # Ensure that we always sample positives.
        rnd[rhs_y_index] = 4.
        rhs_index = rnd.topk(self.rhs_sample_size, sorted=True).indices

        inclusive = rhs_y_index.numel() <= self.rhs_sample_size
        rhs_y_index, mask = map_index(
            rhs_y_index,
            rhs_index,
            max_index=self.num_rhs_nodes,
            inclusive=inclusive,
        )
        lhs_y_batch = lhs_y_batch if inclusive else lhs_y_batch[mask]
        if edge_weight is not None and not inclusive:
            edge_weight = edge_weight[mask]

        rhs_emb = self.rhs_embedding(rhs_index)

        # Calculate embgnn logits
        lhs_embedding_projected = self.lhs_projector(lhs_embedding)

        embgnn_logits = lhs_embedding_projected @ rhs_emb.t()
        # Model the importance of embedding-GNN prediction for each lhs node
        embgnn_offset_logits = self.lin_offset_embgnn(lhs_embedding).flatten()
        embgnn_logits += embgnn_offset_logits.view(-1, 1)

        # Calculate idgnn logits
        inclusive = (rhs_y_index.numel() + rhs_idgnn_index.numel()
                     <= self.rhs_sample_size)
        rhs_idgnn_index, mask = map_index(
            rhs_idgnn_index,
            rhs_index,
            max_index=self.num_rhs_nodes,
            inclusive=inclusive,
        )
        if not inclusive:
            lhs_idgnn_batch = lhs_idgnn_batch[mask]
            rhs_gnn_embedding = rhs_gnn_embedding[mask]
        if self.distance_measure == 'cosine':
            rhs_gnn_embedding = F.normalize(rhs_gnn_embedding, dim=1)
            lhs_embedding = F.normalize(lhs_embedding, dim=1)

        idgnn_logits = self.head.mlp_head(rhs_gnn_embedding).flatten()
        idgnn_logits = idgnn_logits + (lhs_embedding[lhs_idgnn_batch] *
                                       rhs_gnn_embedding).sum(dim=-1).flatten()
        # Model the importance of ID-GNN prediction for each lhs node
        idgnn_offset_logits = self.lin_offset_idgnn(lhs_embedding).flatten()
        idgnn_logits = idgnn_logits + idgnn_offset_logits[lhs_idgnn_batch]

        embgnn_logits[lhs_idgnn_batch, rhs_idgnn_index] = idgnn_logits
        return (embgnn_logits, rhs_y_index, lhs_y_batch, edge_weight, rhs_emb,
                lhs_embedding_projected, rhs_index)

    def compute_loss(
        self,
        lhs_emb: Tensor,
        rhs_emb: Tensor,
        logits: Tensor,
        rhs_y_index: Tensor,
        lhs_y_batch: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:

        emb_logits = None
        if self.emb_loss_coeff > 0:
            # If a user cares about embedding-based prediction, we additionally
            # compute the loss over the embedding inner-product logits to
            # ensure embedding-based predictions are of high quality.
            emb_logits = lhs_emb @ rhs_emb.t()

        if self.loss == 'cross_entropy_loss':
            edge_label_index = torch.stack([lhs_y_batch, rhs_y_index], dim=0)
            loss = sparse_cross_entropy(logits, edge_label_index, edge_weight)
            if emb_logits is not None:
                emb_loss = sparse_cross_entropy(  #
                    emb_logits, edge_label_index, edge_weight)
                loss = loss + self.emb_loss_coeff * emb_loss
        else:
            assert self.loss == 'binary_cross_entropy_loss'
            target = torch.zeros_like(logits)
            target[lhs_y_batch, rhs_y_index] = 1.0
            if edge_weight is not None:
                weight = torch.ones_like(logits)
                weight[lhs_y_batch, rhs_y_index] = edge_weight.abs()
            else:
                weight = None
            loss = F.binary_cross_entropy_with_logits(logits, target, weight)
            if emb_logits is not None:
                emb_loss = F.binary_cross_entropy_with_logits(  #
                    emb_logits, target, weight)
                loss = loss + self.emb_loss_coeff * emb_loss

        if self.emb_l2_reg_coeff > 0:
            l2_reg_loss = (lhs_emb.pow(2).sum(dim=1).mean() +
                           rhs_emb.pow(2).sum(dim=1).mean())
            loss = loss + self.emb_l2_reg_coeff * l2_reg_loss

        return loss

    def postprocess_eval_step(
        self,
        lhs_emb: Tensor,
        rhs_gnn_emb: Tensor,
        rhs_index: Tensor,
        lhs_batch: Tensor,
        k: int,
        exclude_rhs_index: Tensor | None = None,
        exclude_lhs_batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:

        emb_score, emb_index = self.mips_index.search(
            self.lhs_projector(lhs_emb),
            k=k,
            exclude=(exclude_lhs_batch, exclude_rhs_index)
            if exclude_lhs_batch is not None else None,
        )
        emb_score = emb_score + self.lin_offset_embgnn(lhs_emb).view(-1, 1)

        if exclude_lhs_batch is not None and exclude_rhs_index is not None:
            # Only keep not excluded RHS GNN indices:
            index1 = lhs_batch * self.num_rhs_nodes + rhs_index
            index2 = exclude_lhs_batch * self.num_rhs_nodes + exclude_rhs_index
            rhs_mask = ~torch.isin(index1, index2)

            lhs_batch = lhs_batch[rhs_mask]
            rhs_index = rhs_index[rhs_mask]
            rhs_gnn_emb = rhs_gnn_emb[rhs_mask]

        if self.valid_rhs_node_idx is not None:
            # Only keep the valid RHS GNN indices:
            rhs_index, rhs_mask = map_index(
                rhs_index,
                self.valid_rhs_node_idx,
                max_index=self.num_rhs_nodes,
                inclusive=False,
            )
            if rhs_mask is not None:
                lhs_batch = lhs_batch[rhs_mask]
                rhs_gnn_emb = rhs_gnn_emb[rhs_mask]

        if self.distance_measure == 'cosine':
            lhs_emb = F.normalize(lhs_emb, dim=1)
            rhs_gnn_emb = F.normalize(rhs_gnn_emb, dim=1)

        id_score = self.head.mlp_head(rhs_gnn_emb).view(-1)
        id_score = id_score + (lhs_emb[lhs_batch] *
                               rhs_gnn_emb).sum(-1).view(-1)

        perm = lexsort([-id_score, lhs_batch])
        id_score = id_score[perm]
        rhs_index = rhs_index[perm]
        lhs_batch = lhs_batch[perm]

        id_score, _ = to_dense_batch(
            id_score,
            lhs_batch,
            fill_value=float('-inf'),
            max_num_nodes=emb_index.size(1),
            batch_size=emb_index.size(0),
        )
        id_index, _ = to_dense_batch(
            rhs_index,
            lhs_batch,
            fill_value=0,
            max_num_nodes=emb_index.size(1),
            batch_size=emb_index.size(0),
        )

        score = torch.cat([id_score, emb_score], dim=-1)
        index = torch.cat([id_index, emb_index], dim=-1)

        # First, sort by score:
        score, perm = score.sort(dim=-1, descending=True)
        index = index.gather(-1, perm)

        # Stable sort by index to preserve score ranking within same indices:
        index, perm = index.sort(dim=-1, stable=True)
        score = score.gather(-1, perm)

        # Only maintain top-score per index:
        dup_mask = torch.zeros_like(index, dtype=torch.bool)
        dup_mask[:, 1:] = index[:, 1:] == index[:, :-1]
        score = score.masked_fill(dup_mask, float('-inf'))

        # Final sort to maintop top-k global scores:
        score, perm = score.sort(dim=-1, descending=True)
        index = index.gather(-1, perm)

        score = score[:, :emb_score.size(1)].contiguous()
        index = index[:, :emb_score.size(1)].contiguous()

        return score, index

    # Training Callbacks ######################################################

    def train_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        out = self.common_step(batch)  # NOTE `lhs_emb` is unprojected.

        rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(batch)

        # Obtain the valid RHS nodes to train against:
        valid_rhs_node_idx = self._get_valid_rhs_idx_for_batch(batch)

        # No need for sampled softmax if set of valid RHS nodes is small:
        rhs_sample_size = self.rhs_sample_size
        if (rhs_sample_size is not None and valid_rhs_node_idx is not None
                and valid_rhs_node_idx.numel() <= rhs_sample_size):
            rhs_sample_size = None

        if rhs_sample_size is None:
            rhs_emb = self.rhs_embedding(valid_rhs_node_idx)
            logits, lhs_emb_projected = self.postprocess_step(
                *out, rhs_emb=rhs_emb, valid_rhs_node_idx=valid_rhs_node_idx)

            # The RHS embeddings have been filtered to its valid set.
            # Therefore, we need to map `rhs_y_index` to the filtered set:
            if valid_rhs_node_idx is not None:
                rhs_y_index, mask = map_index(
                    rhs_y_index,
                    valid_rhs_node_idx,
                    max_index=self.num_rhs_nodes,
                    inclusive=False,
                )
                lhs_y_batch = lhs_y_batch[mask]
                if edge_weight is not None:
                    edge_weight = edge_weight[mask]
        else:
            (  # Wow, this is ugly...
                logits,
                rhs_y_index,
                lhs_y_batch,
                edge_weight,
                rhs_emb,
                lhs_emb_projected,
                valid_rhs_node_idx,
            ) = self.sampled_postprocess_step(
                *out,
                rhs_y_index=rhs_y_index,
                lhs_y_batch=lhs_y_batch,
                edge_weight=edge_weight,
                valid_rhs_node_idx=valid_rhs_node_idx,
            )
            assert rhs_y_index is not None
            assert lhs_y_batch is not None

        loss = self.compute_loss(
            lhs_emb_projected, rhs_emb, logits, rhs_y_index, lhs_y_batch,
            edge_weight
            if self.weight_mode == WeightMode.WEIGHTED_LOSS else None)
        loss_name = TrainingStage.TRAIN.add_stage_prefix('loss')

        self.log_learning_rate()
        self.log_epoch(loss_name, loss, batch_size=logits.size(0))
        _, rhs_pred_index_mat = self.get_rhs_pred_index_mat(logits, None)

        # In order to compute metrics, we map back predictions and ground-truth
        # to their original node IDs:
        if valid_rhs_node_idx is not None:
            rhs_pred_index_mat = valid_rhs_node_idx[rhs_pred_index_mat]
            rhs_y_index = valid_rhs_node_idx[rhs_y_index]

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

        lhs_emb, rhs_gnn_emb, rhs_index, lhs_batch = self.common_step(batch)

        rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(batch)

        exclude_rhs_index: Tensor | None = None
        exclude_lhs_batch: Tensor | None = None
        if self.eval_new_visit:
            (rhs_y_index, lhs_y_batch, exclude_rhs_index,
             exclude_lhs_batch) = self.get_new_and_past_visit_ground_truth(
                 batch, rhs_y_index, lhs_y_batch)

        if stage == TrainingStage.TEST and not self.disable_eval_metrics:
            assert self.locality_score is not None
            self.locality_score.update(
                edge_label_index=(lhs_y_batch, rhs_y_index),
                rhs_node=batch[self.rhs_name].n_id,
                rhs_batch=batch[self.rhs_name].batch,
            )

        # The RHS embeddings have been filtered to its valid set.
        # Therefore, we need to map `rhs_y_index` to the filtered set:
        if self.valid_rhs_node_idx is not None:
            rhs_y_index, mask = map_index(
                rhs_y_index,
                self.valid_rhs_node_idx,
                max_index=self.num_rhs_nodes,
                inclusive=False,
            )
            lhs_y_batch = lhs_y_batch[mask]
            if edge_weight is not None:
                edge_weight = edge_weight[mask]

            if exclude_rhs_index is not None:
                exclude_rhs_index, mask = map_index(
                    exclude_rhs_index,
                    self.valid_rhs_node_idx,
                    max_index=self.num_rhs_nodes,
                    inclusive=False,
                )
                assert exclude_lhs_batch is not None
                exclude_lhs_batch = exclude_lhs_batch[mask]

        k = self.metric_store.get_metric_collection(TrainingStage.TRAIN).max_k
        if self.eval_top_k > 0:
            k = max(k, self.eval_top_k)

        score, index = self.postprocess_eval_step(
            lhs_emb,
            rhs_gnn_emb,
            rhs_index,
            lhs_batch,
            k=k,
            exclude_rhs_index=exclude_rhs_index,
            exclude_lhs_batch=exclude_lhs_batch,
        )

        if not self.disable_eval_metrics:
            self.update_metrics(
                stage,
                self.valid_rhs_node_idx[index]
                if self.valid_rhs_node_idx is not None else index,
                (
                    lhs_y_batch,
                    self.valid_rhs_node_idx[rhs_y_index]
                    if self.valid_rhs_node_idx is not None else rhs_y_index,
                ),
                edge_label_weight=edge_weight,
            )

        if modeling_context.is_explaining() or self.eval_top_k > 0:
            batch_size = batch[self.lhs_name].batch_size
            lhs_pkey = batch[self.lhs_name].pkey[:batch_size]
            time = batch[self.lhs_name].seed_time
            lhs_index = batch[self.lhs_name].n_id[:batch_size]

            score = score[:, :self.eval_top_k].contiguous()
            index = index[:, :self.eval_top_k].contiguous()

            data = LinkPredMetricData(index, (lhs_y_batch, rhs_y_index))
            label = data.pred_rel_mat
            if self.valid_rhs_node_idx is not None:
                index = self.valid_rhs_node_idx[index]

            return dict(pkey=lhs_pkey, id=lhs_index, time=time, target=index,
                        score=score, label=label)

        return self.holdout_top_scores(stage, score)

    # Prediction Callbacks ####################################################

    def predict_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:
        lhs_emb, rhs_gnn_emb, rhs_index, lhs_batch = self.common_step(batch)
        return_embeddings = getattr(self, '_kumo_return_predict_embeddings',
                                    False)
        projected_lhs: Tensor | None = None
        hook = None
        if return_embeddings:

            def capture_lhs(module, inputs, output) -> None:
                nonlocal projected_lhs
                projected_lhs = output

            hook = self.lhs_projector.register_forward_hook(capture_lhs)

        batch_size = batch[self.lhs_name].batch_size
        lhs_pkey = batch[self.lhs_name].pkey[:batch_size]
        time = batch[self.lhs_name].seed_time
        lhs_index = batch[self.lhs_name].n_id[:batch_size]

        try:
            # Avoid top-k computation if only interested in embeddings:
            if self.pred_top_k is None or self.pred_top_k == 0:
                self.lhs_projector(lhs_emb)
                result: dict[str, Tensor | None] = dict(pkey=lhs_pkey,
                                                        id=lhs_index, time=time)
            else:
                score, index = self.postprocess_eval_step(
                    lhs_emb,
                    rhs_gnn_emb,
                    rhs_index,
                    lhs_batch,
                    k=self.pred_top_k,
                )
                # Map back to the original RHS node indices
                if self.valid_rhs_node_idx is not None:
                    index = self.valid_rhs_node_idx[index]
                result = dict(pkey=lhs_pkey, id=lhs_index, time=time,
                              target=index, score=score)
        finally:
            if hook is not None:
                hook.remove()
        if return_embeddings:
            if projected_lhs is None:
                raise RuntimeError(
                    'Hybrid prediction did not produce requested embeddings')
            result['lhs_embedding'] = projected_lhs.detach().cpu()

        return result

    def validate_head(self):
        assert is_torch_instance(self.head, LinkPredHead)
        assert not self.head.normalize_vectors
        assert self.head.use_mlp_head
        assert not self.head.use_rhs_projector

    def get_embedding(self, batch, entity_name, side=None):
        if side == 'lhs':
            batch_size = batch[entity_name].batch_size
            out = self(
                batch.tf_dict,
                batch.n_id_dict,
                batch.collect('edge_index', allow_empty=True),
                batch.collect('time', allow_empty=True),
                batch.batch_dict,
                seed_time=batch[entity_name].get('seed_time'),
                batch_size=batch_size,
                num_sampled_nodes_dict=batch.collect(
                    'num_sampled_nodes',
                    allow_empty=True,
                ),
                num_sampled_edges_dict=batch.collect(
                    'num_sampled_edges',
                    allow_empty=True,
                ),
            )
            return self.lhs_projector(out[0])
        return super().get_embedding(batch, entity_name, side)
