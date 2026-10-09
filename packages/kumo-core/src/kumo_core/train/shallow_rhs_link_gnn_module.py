# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from collections.abc import Callable
from typing import Any, Literal

import kumo_core.modeling.context as modeling_context
import pandas as pd
import torch
import torch.nn.functional as F
from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
from kumo_core.train.base_link_module import (
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_INTERVAL,
    DEFAULT_OPTIMIZER,
    BaseLinkLightningModule,
)
from kumo_core.train.mapping import TTNG_RHS_PRIMARY_KEY_INDEX_NAME
from kumo_core.types import TaskType
from kumoapi.model_plan import RHSEmbeddingMode, WeightMode
from kumoapi.train import TrainingStage
from kumo_core.evaluation.locality import LocalityScore
from kumo_core.modeling.heads.link_prediction import LinkPredHead
from kumo_core.modeling.retrieval.rhs_embedding import RHSEmbedding
from kumo_core.sampling.edge_exclusion import drop_edge_index, drop_edge_label_index
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric import is_torch_instance
from torch_geometric.data import HeteroData
from torch_geometric.metrics.link_pred import LinkPredMetricData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.cross_entropy import sparse_cross_entropy
from torch_geometric.utils.map import map_index
from typing_extensions import Self

logger = logging.getLogger(__name__)

MAX_RHS_SAMPLE_SIZE = 500_000


class Projector(torch.nn.Module):
    def __init__(self, projector: torch.nn.Module, normalize: bool):
        super().__init__()
        self.projector = projector
        self.normalize = normalize

    def forward(self, x: Tensor) -> Tensor:
        out = self.projector(x)
        if self.normalize:
            out = F.normalize(out, dim=1)
        return out


class ShallowRHSLinkGNNLightningModule(BaseLinkLightningModule):
    r"""A PyTorch Lightning module that handles the execution of GNN-based
    link-level training, validation, testing and prediction routines.
    :class:`ShallowRHSLinkGNNLightningModule` learns a shallow embedding
    (embedding that only depends on the information of the node itself) for
    each RHS node and produces an embedding for each LHS node.
    Furthermore, it adds shallow embedding for occurrences of rhs nodes in the
    LHS subgraph for query awareness while continuing to produce inner product
    compatible embeddings.

    This module provides several options to model shallow embeddings.
    - lookup: Use lookup-based embedding (distinct for each node)
    - feature: Use feature-based embedding
    - fusion: Add lookup-based embedding and feature-based embedding together

    Args:
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The GNN model.
        head (LinkPredHead): The model head. Currently not used.
        task_type (TaskType): The task type.
        train_table_name (str): The name of the training table.
        entity_table_names ((str, str)): The names of the entity tables.
        num_rhs_nodes (int): The number of rhs nodes.
        rhs_emb_init_std (float): std for rhs embedding initialization.
            (default: :obj:`1.0`)
        emb_l2_reg_coeff (float): L2 regularization on the final
            embeddings. (default: :obj:`0.0`)
        rhs_sample_size (int, optional): The number of rhs nodes to sample
            during training. If set to :obj:`None`, all rhs nodes will
            be used for loss computation. (default: :obj:`None`)
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
        target_fraction: How many ground-truth links shall be used for
            supervision.
        overlap_ratio: The overlap ratio between message passing edges
            and supervision edges.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        model: torch.nn.Module,
        head: LinkPredHead,
        task_type: TaskType,
        train_table_name: str,
        entity_table_names: tuple[str, str],
        num_rhs_nodes: int,
        rhs_emb_init_std: float = 1.0,
        emb_l2_reg_coeff: float = 0.0,
        rhs_sample_size: int | None = None,
        rhs_emb_mode: RHSEmbeddingMode | str = RHSEmbeddingMode.FUSION,
        rhs_feat_bytes_threshold: int | None = None,
        eval_new_visit: bool = False,
        rhs_index_map: pd.DataFrame | None = None,  # for backcompat...
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
        target_fraction: float = 1.0,
        overlap_ratio: float = 1.0,
        target_to_entity_paths: list[list[EdgeType]] | None = None,
    ) -> None:
        super().__init__(
            encoder=encoder,
            model=model,
            head=head,
            task_type=task_type,
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

        self.lookup_optimizer_kwargs = lookup_optimizer_kwargs

        self.locality_score = None
        if task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            self.locality_score = LocalityScore()

        # We will use a more complex RHS encoder in case we are solely
        # operating on RHS features (without lookup embedding). As such, we
        # lower the threshold of RHS nodes for sampled softmax computation.
        rhs_emb_mode = RHSEmbeddingMode(rhs_emb_mode)
        if rhs_emb_mode == RHSEmbeddingMode.FEATURE:
            if rhs_sample_size is None:
                if num_rhs_nodes > MAX_RHS_SAMPLE_SIZE:
                    rhs_sample_size = MAX_RHS_SAMPLE_SIZE
        else:
            if rhs_sample_size is None:
                if num_rhs_nodes > 2 * MAX_RHS_SAMPLE_SIZE:
                    rhs_sample_size = MAX_RHS_SAMPLE_SIZE

        if rhs_sample_size is not None:
            if rhs_sample_size > MAX_RHS_SAMPLE_SIZE:
                rhs_sample_size = MAX_RHS_SAMPLE_SIZE

            if rhs_sample_size >= num_rhs_nodes:
                rhs_sample_size = None

        if rhs_sample_size is None:
            logger.info(f"Using full softmax over {num_rhs_nodes} RHS nodes.")
        else:
            logger.info(f"Using sampled softmax over {rhs_sample_size} nodes "
                        f"out of {num_rhs_nodes} RHS nodes.")

        if rhs_sample_size is None:
            # If we train against full RHS nodes, we get zero benefits when
            # storing RHS features on CPU.
            rhs_feat_bytes_threshold = None

        self.train_table_name = train_table_name
        self.num_rhs_nodes = num_rhs_nodes
        self.rhs_emb_init_std = rhs_emb_init_std
        self.emb_l2_reg_coeff = emb_l2_reg_coeff
        self.rhs_sample_size = rhs_sample_size
        self.rhs_emb_mode = rhs_emb_mode
        self.eval_new_visit = eval_new_visit
        self.loss = loss
        self.target_fraction = target_fraction
        self.overlap_ratio = overlap_ratio
        self.target_to_entity_paths = target_to_entity_paths

        self.lhs_projector = Projector(projector=self.head.lhs_projector,
                                       normalize=distance_measure == 'cosine')
        self.identity_awareness = torch.nn.Embedding(1, model.channels)

        # For projecting RHS lookup embeddings to the input of the GNN:
        self.rhs_projector: torch.nn.Linear | None = None
        if self.rhs_emb_mode.use_rhs_lookup:  # type: ignore
            self.rhs_projector = torch.nn.Linear(
                self.head.embedding_dim,  # type: ignore
                self.model.channels,  # type: ignore
            )

        self.rhs_embedding = RHSEmbedding(
            emb_mode=self.rhs_emb_mode,
            num_nodes=self.num_rhs_nodes,
            embedding_dim=self.head.embedding_dim,  # type: ignore
            encoder_dict=self.encoder.table_encoder_dict[self.rhs_name],
            sparse=self.rhs_sample_size is not None,
            init_std=self.rhs_emb_init_std,
            index_map=rhs_index_map,
            distance_measure=distance_measure,
            rhs_feat_bytes_threshold=rhs_feat_bytes_threshold,
        )
        self._maybe_compile()

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
        self.sampled_postprocess_step = torch.compile(
            self.sampled_postprocess_step,
            dynamic=True,
        )

    def _get_valid_rhs_idx_for_batch(self, batch: HeteroData) -> Tensor | None:
        if self.task_type == TaskType.STATIC_LINK_PREDICTION:
            return self._canonical_rhs_node_ids(
                batch[self.lhs_name].n_id.device)

        seed_time = batch[self.lhs_name].seed_time
        return self._get_valid_rhs_idx_for_seed_time(
            seed_time=int(seed_time[0]),  # Only contains shared seed times.
            device=seed_time.device,
        )

    def _get_valid_rhs_idx_for_seed_time(
        self,
        seed_time: int,
        device: torch.device | None = None,
    ) -> Tensor | None:
        if self.task_type == TaskType.STATIC_LINK_PREDICTION:
            return self._canonical_rhs_node_ids(device)

        if self.lp_train_table.valid_rhs_idx_dict is None:
            return self._canonical_rhs_node_ids(device)

        valid = self.lp_train_table.get_valid_rhs_idx(seed_time).to(device)
        return self._filter_to_canonical_rhs_node_ids(valid, device)

    @property
    def datamodule(self) -> DiskgraphLightningDataModule:
        r"""The ``DiskgraphLightningDataModule`` (TT-NG) backing this module."""
        datamodule = getattr(self.trainer, 'datamodule')
        assert isinstance(datamodule, DiskgraphLightningDataModule)
        return datamodule

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
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
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

        # Add rhs-awareness of lookup features:
        if self.rhs_embedding.lookup_embedding is not None:
            assert self.rhs_projector is not None
            n_id = node_id_dict[self.rhs_name]

            emb = self.rhs_projector(self.rhs_embedding.lookup_embedding(n_id))
            x_dict[self.rhs_name] = x_dict[self.rhs_name] + emb

        x_dict = self.model(
            x_dict,
            edge_index_dict,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        )

        # Return LHS embeddings:
        return self.lhs_projector(x_dict[self.lhs_name][:batch_size])

    def common_step(self, batch: HeteroData) -> Any:
        # Use `seed_time` as input only for temporal tasks:
        seed_time: Tensor | None = None
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            seed_time = batch[self.lhs_name].seed_time

        return self(
            batch.tf_dict,
            batch.n_id_dict,
            batch.collect('edge_index', allow_empty=True),
            batch.collect('time', allow_empty=True),
            batch.batch_dict,
            seed_time=seed_time,
            batch_size=batch[self.lhs_name].batch_size,
            num_sampled_nodes_dict=batch.collect('num_sampled_nodes', True),
            num_sampled_edges_dict=batch.collect('num_sampled_edges', True),
        )

    def sampled_postprocess_step(
        self,
        lhs_emb: Tensor,
        rhs_y_index: Tensor,
        lhs_y_batch: Tensor,
        edge_weight: Tensor | None,
        valid_rhs_node_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        r"""Computes sampled logits and re-maps :obj:`rhs_y_index`,
        :obj:`lhs_y_batch`, and :obj:`edge_weight` to valid range."""
        assert self.rhs_sample_size is not None

        # Sample a set of negative logits for the whole batch:
        rnd = torch.rand(self.num_rhs_nodes, device=self.device)
        if valid_rhs_node_idx is not None:
            assert valid_rhs_node_idx.numel() > self.rhs_sample_size
            rnd[valid_rhs_node_idx] += 1.

        rnd[rhs_y_index] = 3.  # Ensure that we always sample positives.
        rhs_index = rnd.topk(self.rhs_sample_size, sorted=True).indices

        # Compute sampled logits of shape `[batch_size, rhs_sample_size]`:
        rhs_emb = self.rhs_embedding(rhs_index)
        logits = lhs_emb @ rhs_emb.t()

        # Remap `rhs_y_index` and `lhs_y_batch` to valid range. We use a
        # slightly faster code path here in case it is guaranteed that all
        # positives are included in the sampled set. We approximate that via
        # `rhs_y_index` count (worst-case of no-repeating ground-truth edges).
        inclusive = rhs_y_index.numel() <= self.rhs_sample_size
        rhs_y_index, mask = map_index(
            rhs_y_index,
            rhs_index,
            max_index=self.num_rhs_nodes,
            inclusive=inclusive,
        )
        # `map_index` returns a mask for any indices in `rhs_y_index` that
        # are not included in `rhs_index`. We use that to filter
        # `lhs_y_batch` correspondingly.
        lhs_y_batch = lhs_y_batch if inclusive else lhs_y_batch[mask]
        if edge_weight is not None:
            edge_weight = edge_weight if inclusive else edge_weight[mask]

        return (logits, rhs_y_index, lhs_y_batch, rhs_emb, edge_weight,
                rhs_index)

    def compute_loss(
        self,
        lhs_emb: Tensor,
        rhs_emb: Tensor,
        logits: Tensor,
        rhs_y_index: Tensor,
        lhs_y_batch: Tensor,
        edge_weight: Tensor | None,
    ) -> Tensor:
        if self.loss == 'cross_entropy_loss':
            edge_label_index = torch.stack([lhs_y_batch, rhs_y_index], dim=0)
            loss = sparse_cross_entropy(logits, edge_label_index, edge_weight)
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

        if self.emb_l2_reg_coeff > 0:
            l2_reg_loss = (lhs_emb.pow(2).sum(dim=1).mean() +
                           rhs_emb.pow(2).sum(dim=1).mean())
            loss = loss + self.emb_l2_reg_coeff * l2_reg_loss

        return loss

    # Training Callbacks ######################################################

    def on_fit_start(self):
        # Pre-fetch RHS features. Only feature/fusion modes have features (and
        # expose feature_store on TT-NG); guarding is a no-op for lookup/legacy.
        self.rhs_embedding.filter(self._canonical_rhs_node_ids(self.device))
        if self.rhs_emb_mode.use_rhs_feature:
            self.rhs_embedding.fetch_feats(
                self.datamodule.feature_store,
                self.rhs_name,
            )

    def train_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        # Collect ground-truth links:
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(batch)
        else:
            assert self.task_type == TaskType.STATIC_LINK_PREDICTION
            rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(
                batch, stage=TrainingStage.TRAIN)

        # Adjust static LP training (memorization, induction, expansion).
        if self.task_type == TaskType.STATIC_LINK_PREDICTION:
            lhs_y_batch, rhs_y_index, mask = drop_edge_label_index(
                (lhs_y_batch, rhs_y_index),
                p=1.0 - self.target_fraction,
            )
            if mask is not None and edge_weight is not None:
                edge_weight = edge_weight[mask]

            if self.overlap_ratio != 1.0:
                assert self.target_to_entity_paths is not None
                fact, lhs_rel, _ = self.target_to_entity_paths[0][0]
                _, rhs_rel, _ = self.target_to_entity_paths[1][0]
                fact = fact.rsplit('.', maxsplit=1)[-1]

                batch = drop_edge_index(
                    batch,
                    fact_to_lhs_type=(
                        fact,
                        lhs_rel,
                        self.entity_table_names[0],
                    ),
                    rhs_to_fact_type=(
                        self.entity_table_names[1],
                        f'rev_{rhs_rel}',
                        fact,
                    ),
                    edge_label_index=(lhs_y_batch, rhs_y_index),
                    p=1.0 - self.overlap_ratio,
                    num_rhs_nodes=self.num_rhs_nodes,
                )

        lhs_emb = self.common_step(batch)

        # Obtain the valid RHS nodes to train against:
        valid_rhs_node_idx = self._get_valid_rhs_idx_for_batch(batch)

        # No need for sampled softmax if set of valid RHS nodes is small:
        rhs_sample_size = self.rhs_sample_size
        if (rhs_sample_size is not None and valid_rhs_node_idx is not None
                and valid_rhs_node_idx.numel() <= rhs_sample_size):
            rhs_sample_size = None

        if rhs_sample_size is None:
            rhs_emb = self.rhs_embedding(valid_rhs_node_idx)
            logits = lhs_emb @ rhs_emb.t()

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
                rhs_emb,
                edge_weight,
                valid_rhs_node_idx,
            ) = self.sampled_postprocess_step(
                lhs_emb,
                rhs_y_index,
                lhs_y_batch,
                edge_weight,
                valid_rhs_node_idx,
            )

        loss = self.compute_loss(
            lhs_emb, rhs_emb, logits, rhs_y_index, lhs_y_batch, edge_weight
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

    def on_evaluation_start(self, stage: TrainingStage):
        super().on_evaluation_start(stage)

        self.valid_rhs_node_idx = self._canonical_rhs_node_ids(self.device)
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            if stage == TrainingStage.VAL:
                stage_time = self.datamodule.val_time
            else:
                assert stage == TrainingStage.TEST
                stage_time = self.datamodule.test_time
            # ``val_time`` / ``test_time`` may be ``None`` on TT-NG; then we
            # can't apply the temporal filter, so retain the canonical set.
            if stage_time is not None:
                seed_time = int(stage_time[0])
                self.valid_rhs_node_idx = (
                    self._get_valid_rhs_idx_for_seed_time(
                        seed_time, self.device))

        self.rhs_embedding.filter(self.valid_rhs_node_idx)
        # Only feature/fusion modes (re)fetch features; lookup builds the MIPS
        # index from the learned table and must not touch feature_store.
        if self.rhs_emb_mode.use_rhs_feature:
            self.rhs_embedding.fetch_feats(
                self.datamodule.feature_store,
                self.rhs_name,
            )

        self._mips_index = self.rhs_embedding.get_mips_index()

    def evaluation_step(
        self,
        batch: HeteroData,
        stage: TrainingStage,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:

        lhs_emb = self.common_step(batch)

        exclude_rhs_index: Tensor | None = None
        exclude_lhs_batch: Tensor | None = None

        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(batch)

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

        else:
            assert self.task_type == TaskType.STATIC_LINK_PREDICTION
            assert not self.eval_new_visit

            rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(
                batch, stage=stage)
            (exclude_rhs_index,
             exclude_lhs_batch) = self.get_tt_ng_exclude_ground_truth(batch)
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

        score, index = self.get_rhs_pred_index_mat(None, lhs_emb,
                                                   exclude_rhs_index,
                                                   exclude_lhs_batch)

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
            time: Tensor | None = None
            if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
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

        return {}

    def evaluation_epoch_end(self, *args, **kwargs):
        super().evaluation_epoch_end(*args, **kwargs)

        if (self.locality_score is not None
                and self.locality_score._update_count > 0):
            score = self.locality_score.compute()
            self.locality_score.reset()
            self.log_epoch('test_locality_score', score, prog_bar=False)

        self.valid_rhs_node_idx = None
        self.rhs_embedding.filter(self._canonical_rhs_node_ids(self.device))
        if self.rhs_emb_mode.use_rhs_feature:
            self.rhs_embedding.fetch_feats(
                self.datamodule.feature_store,
                self.rhs_name,
            )
        self.clear_mips_index()

    # Prediction Callbacks ####################################################

    def on_predict_start(self, *args, **kwargs):
        super().on_predict_start(*args, **kwargs)

        # TT-NG temporal LP: filter predict candidates to RHS valid at the
        # prediction anchor (matches eval). No-op on static.
        self.valid_rhs_node_idx = self.tt_ng_valid_rhs_for_stage(
            TrainingStage.PRED)
        self.valid_rhs_node_idx = self._filter_to_canonical_rhs_node_ids(
            self.valid_rhs_node_idx, self.device)
        self.rhs_embedding.filter(self.valid_rhs_node_idx)
        # Pass feature_store only for feature/fusion (lookup builds the MIPS
        # index from the learned table).
        if self.rhs_emb_mode.use_rhs_feature:
            self._mips_index = self.rhs_embedding.get_mips_index(
                feature_store=self.datamodule.feature_store,
                table_name=self.rhs_name,
            )
        else:
            self._mips_index = self.rhs_embedding.get_mips_index()

    def predict_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:

        lhs_emb = self.common_step(batch)
        return_embeddings = getattr(self, '_kumo_return_predict_embeddings',
                                    False)

        batch_size = batch[self.lhs_name].batch_size
        lhs_pkey = batch[self.lhs_name].pkey[:batch_size]
        time: Tensor | None = None
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            time = batch[self.lhs_name].seed_time
        lhs_index = batch[self.lhs_name].n_id[:batch[self.lhs_name].batch_size]

        # Avoid top-k computation if only interested in embeddings:
        if self.pred_top_k is None or self.pred_top_k == 0:
            result = dict(pkey=lhs_pkey, id=lhs_index, time=time)
            if return_embeddings:
                result['lhs_embedding'] = lhs_emb.detach().cpu()
            return result

        # NOTE No exclusion + top-k prediction for now since we don't
        # necessarily have access to the underlying training table anymore:
        pred_top_k = min(self.pred_top_k, self.mips_index.num_keys)
        score, target = self.mips_index.search(lhs_emb, k=pred_top_k)

        if modeling_context.is_explaining():
            # Recompute scores with proper gradients:
            rhs_emb = self.mips_index._key[target.view(-1)]
            rhs_emb = rhs_emb.view(*target.size(), -1)
            lhs_emb = lhs_emb.unsqueeze(1).repeat(1, target.size(1), 1)
            score = (lhs_emb * rhs_emb).sum(dim=-1)

        # Map back to the original RHS node indices
        if self.valid_rhs_node_idx is not None:
            target = self.valid_rhs_node_idx[target]

        result = dict(pkey=lhs_pkey, id=lhs_index, time=time, target=target,
                      score=score)
        if return_embeddings:
            result['lhs_embedding'] = lhs_emb.detach().cpu()
        return result

    def on_predict_end(self, *args, **kwargs):
        super().on_predict_end(*args, **kwargs)

        self.valid_rhs_node_idx = None
        self.rhs_embedding.filter(None)
        self.clear_mips_index()

    # Embedding Callbacks #####################################################

    def _get_register_embedding_hook_fn(self) -> Callable | None:
        # We hook to the output of the lhs projector, which contains the
        # projected LHS embeddings used for the inner product search.
        return self.lhs_projector.register_forward_hook

    def embedding_step(
        self,
        stage: TrainingStage,
        module: torch.nn.Module,
        inputs: tuple[Tensor, ...],
        outputs: Tensor,
    ) -> dict[str, Tensor]:
        return {self.lhs_emb_name: outputs}

    @torch.no_grad()
    def embedding_epoch_end(
        self,
        stage: TrainingStage,
        outputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        # Attach RHS embeddings:
        outputs[self.rhs_emb_name] = self.mips_index._key.cpu()

        return outputs

    # Callbacks ###############################################################

    def set_rhs_index_map(self, index_map: pd.DataFrame):
        self.rhs_index_map = index_map
        self._require_ttng_raw_rhs_index = (
            index_map.index.name == TTNG_RHS_PRIMARY_KEY_INDEX_NAME)

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ):
        out = super().load_state_dict(state_dict, strict)

        if hasattr(self, 'rhs_index_map'):
            lookup = self.rhs_embedding.lookup_embedding
            if (getattr(self, '_require_ttng_raw_rhs_index', False)
                    and lookup is not None
                    and (lookup.index_map is None or lookup.index_map.index.name
                         != TTNG_RHS_PRIMARY_KEY_INDEX_NAME)):
                raise RuntimeError(
                    'This TT-NG RHS lookup checkpoint predates raw-primary-key '
                    'catalog mappings and cannot be safely remapped to a new '
                    'DiskGraph snapshot. Retrain the model before batch '
                    'prediction.')
            self.rhs_embedding.order_by(self.rhs_index_map, default_init="mean")
            del self.rhs_index_map
            if hasattr(self, '_require_ttng_raw_rhs_index'):
                del self._require_ttng_raw_rhs_index

        return out

    def validate_head(self):
        assert is_torch_instance(self.head, LinkPredHead)
        assert not self.head.normalize_vectors
        assert not self.head.use_mlp_head
        assert not self.head.use_rhs_projector

    def get_embedding(self, batch, entity_name, side=None):
        if side == 'lhs':
            batch_size = batch[entity_name].batch_size
            return self(
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
        elif side == 'rhs':
            batch_size = batch[entity_name].batch_size
            rhs_node_ids = batch[entity_name].n_id[:batch_size]
            return self.rhs_embedding(rhs_node_ids)
        raise ValueError(
            f"side must be 'lhs' or 'rhs' for link modules, got {side!r}")

    def to(self, *args, **kwargs) -> Self:
        # Explicitly call `to` on the RHS embedding to move caches to the
        # device.
        self.rhs_embedding.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def cpu(self) -> Self:
        # Explicitly call `to` on the RHS embedding to move caches to the
        # device.
        self.rhs_embedding.cpu()
        return super().cpu()

    def cuda(self, *args, **kwargs) -> Self:
        # Explicitly call `to` on the RHS embedding to move caches to the
        # device.
        self.rhs_embedding.cuda(*args, **kwargs)
        return super().cuda(*args, **kwargs)
