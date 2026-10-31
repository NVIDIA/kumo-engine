# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from typing import Any, Literal, Protocol, runtime_checkable

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
from kumo_core.modeling.heads.link_prediction import LinkPredHeadOutput
from kumo_core.modeling.retrieval.mips import require_faiss
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric import is_torch_instance
from torch_geometric.data import HeteroData
from torch_geometric.metrics.link_pred import LinkPredMetricData
from torch_geometric.typing import EdgeType, NodeType
from torch_geometric.utils.map import map_index

logger = logging.getLogger(__name__)


@runtime_checkable
class _RHSDataModule(Protocol):
    r"""Structural type for the LP datamodule used by
    :meth:`LinkGNNLightningModule.init_mips_structure`; duck-typed so
    ``DiskgraphLightningDataModule`` qualifies.
    ``rhs_predict_dataloader`` is consumed but not asserted (attached
    dynamically by the legacy predict path).
    """
    def rhs_val_dataloader(self) -> Any:
        ...

    def rhs_test_dataloader(self) -> Any:
        ...


class LinkGNNLightningModule(BaseLinkLightningModule):
    r"""A PyTorch Lightning module that handles the execution of GNN-based
    link-level training, validation, testing and prediction routines.

    Args:
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The GNN model.
        head (torch.nn.Module): The model head.
        task_type (TaskType): The task type.
        train_table_name (str): The name of the training table.
        entity_table_names ((str, str)): The names of the entity tables.
        emb_l2_reg_coeff (float): L2 regularization on the final
            embeddings. (default: :obj:`0.0`)
        loss_fn (str, optional): The loss function to use. One of
            :obj:`"margin"`, :obj:`"bpr"`, or :obj:`"sampled_softmax"`.
            (default: :obj:`"margin"`)
        margin (float, optional): The margin to use in case
            :obj:`loss_fn = "margin"`. (default: :obj:`1.0`)
        sampled_softmax_factor (float, optional): The value to add to
            negative scores in case :obj:`loss_fn = "sampled_softmax"`.
            (default: :obj:`0.0`)
        eval_new_visit (bool): Whether to focus on evaluation on new visits.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        model: torch.nn.Module,
        head: LinkPredHead,
        task_type: TaskType,
        train_table_name: str,
        entity_table_names: tuple[str, str],
        # TODO Add a config class for link prediction loss functions.
        emb_l2_reg_coeff: float = 0.0,
        loss_fn: str = 'margin',
        margin: float = 1.0,
        sampled_softmax_factor: float = 0.0,
        eval_new_visit: bool = False,
        # `BaseLightningModule` arguments:
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

        self.lhs_identity = torch.nn.Identity()  # for getting lhs embeddings
        self.train_table_name = train_table_name
        self.entity_table_names = entity_table_names
        self.emb_l2_reg_coeff = emb_l2_reg_coeff
        self.eval_new_visit = eval_new_visit
        self.loss_fn = loss_fn
        self.margin = margin
        self.sampled_softmax_factor = sampled_softmax_factor
        self.top_k = 0

    @property
    def datamodule(self) -> "_RHSDataModule":
        r"""The LP datamodule driving MIPS eval; duck-typed (see
        :class:`_RHSDataModule`)."""
        datamodule = getattr(self.trainer, 'datamodule')
        # Require only the VAL/TEST RHS hooks; rhs_predict_dataloader is attached
        # dynamically by the legacy predict path.
        for _hook in ('rhs_val_dataloader', 'rhs_test_dataloader'):
            if not callable(getattr(datamodule, _hook, None)):
                raise TypeError(
                    f"{type(self).__name__}.datamodule expects an RHS-capable "
                    f"link-prediction datamodule exposing {_hook}(); got "
                    f"{type(datamodule).__name__}. "
                    "DiskgraphLightningDataModule satisfies this contract.")
        return datamodule

    @torch.no_grad()
    def init_mips_structure(self, stage: TrainingStage):
        r"""Link-level prediction function to initialize a MIPS data structure
        on the RHS embedding nodes to quickly find the nearest neighbors from
        LHS embeddings. This function is executed at the beginning of
        validation and testing (see :meth:`on_validation_start` and
        :meth:`on_test_start`) for link prediction tasks that require retrieval
        metrics.

        Args:
            stage (TrainingStage): The stage for which the MIPS structure is
                computed. VAL, TEST or PRED
        """

        require_faiss(self.device)
        # TODO Cache these dataloaders.
        if stage == TrainingStage.VAL:
            loader = self.datamodule.rhs_val_dataloader()
        elif stage == TrainingStage.TEST:
            loader = self.datamodule.rhs_test_dataloader()
        elif stage == TrainingStage.PRED:
            loader = self.datamodule.rhs_predict_dataloader()
        else:
            raise ValueError(
                f'{stage} stage is invalid for init_mips_structure()')

        outs = []
        for batch in loader:
            # Fixed RHS seed time (set by rhs_dataloader), not the train-table time.
            batch = batch.to(self.device)
            seed_time: Tensor | None = None
            if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                seed_time = batch[self.rhs_name].seed_time
            batch_size = batch[self.rhs_name].batch_size
            out_dict = self(
                batch.tf_dict,
                batch.collect('edge_index', allow_empty=True),
                batch.collect('time', allow_empty=True),
                batch.batch_dict,
                seed_time=seed_time,
            )
            out = out_dict[self.rhs_name][:batch_size]
            out = self.head.rhs_forward(out)
            outs.append(out)

        emb = torch.cat(outs, dim=0)
        # TT-NG temporal LP: the rhs_*_dataloader yields the full catalog in
        # node-id order, so restrict the MIPS index to RHS valid at the seed
        # time (set in on_evaluation_start).
        if self.valid_rhs_node_idx is not None:
            emb = emb.index_select(0, self.valid_rhs_node_idx.to(emb.device))
        self.init_mips_index(emb)

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
    ) -> dict[NodeType, Tensor]:
        x_dict = self.encoder(tf_dict, batch_dict, time_dict, seed_time)
        x_dict.pop(self.train_table_name, None)

        x_dict = self.model(
            x_dict,
            edge_index_dict,
            num_sampled_nodes_dict,
            num_sampled_edges_dict,
        )
        if self.distance_measure == 'cosine':
            x_dict[self.lhs_name] = F.normalize(x_dict[self.lhs_name], dim=1)
            x_dict[self.rhs_name] = F.normalize(x_dict[self.rhs_name], dim=1)

        return x_dict

    def common_step(self, batch: HeteroData) -> Any:
        # Use seed_time as input feature only for temporal prediction tasks.
        # for static prediction tasks, we still have seed_time, but it is only
        # used for sampling neighbors:
        seed_time: Tensor | None = None
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            seed_time = batch[self.train_table_name].time
        x_dict = self(
            batch.tf_dict,
            batch.collect('edge_index', allow_empty=True),
            batch.collect('time', allow_empty=True),
            batch.batch_dict,
            seed_time=seed_time,
            num_sampled_nodes_dict=batch.collect('num_sampled_nodes', True),
            num_sampled_edges_dict=batch.collect('num_sampled_edges', True),
        )

        # if at eval/prediction stage, return here.
        if self.is_mips_index_initialized:
            return x_dict

        # otherwise, prepare everything needed for training loss computation.
        lhs_index = batch[self.lhs_name].src_index
        rhs_pos_index = batch[self.rhs_name].dst_pos_index
        rhs_neg_index = batch[self.rhs_name].dst_neg_index

        out = self.head(
            x_dict[self.lhs_name][lhs_index],
            x_dict[self.rhs_name][rhs_pos_index],
            x_dict[self.rhs_name][rhs_neg_index]
            if rhs_neg_index is not None else None,
            # We can safely share negatives if the task is static LP.
            share_negatives=self.task_type == TaskType.STATIC_LINK_PREDICTION,
        )
        return out

    # Training Callbacks ######################################################

    def margin_loss(
        self,
        pos_score: Tensor,
        neg_score: Tensor,
        neg_mask: Tensor | None = None,
    ) -> Tensor:
        loss = (neg_score - pos_score + self.margin).clamp(min=0)
        return self._reduce_neg_loss(loss, neg_mask)

    def sampled_softmax_loss(
        self,
        pos_logits: Tensor,
        neg_logits: Tensor,
        neg_mask: Tensor | None = None,
    ) -> Tensor:
        neg_logits = neg_logits + self.sampled_softmax_factor
        if neg_mask is not None:
            # Masked negatives -> exp(-inf) == 0 in logsumexp (as if absent).
            neg_logits = neg_logits.masked_fill(~neg_mask, float('-inf'))
        logits = torch.cat([pos_logits, neg_logits], dim=-1)
        loss = -pos_logits + torch.logsumexp(logits, dim=-1)
        # pos_logits is finite and in logits, so an all-masked row reduces to 0;
        # no NaN guard needed.
        return loss.mean()

    def bpr_loss(
        self,
        pos_score: Tensor,
        neg_score: Tensor,
        neg_mask: Tensor | None = None,
    ) -> Tensor:
        # The default beta = 1 leads to the BPR loss. See:
        # https://github.com/kuandeng/LightGCN/blob/master/LightGCN.py#L348
        loss = F.softplus(neg_score - pos_score)
        return self._reduce_neg_loss(loss, neg_mask)

    def _reduce_neg_loss(
        self,
        loss: Tensor,
        neg_mask: Tensor | None,
    ) -> Tensor:
        r"""Reduce a ``[B, M]`` per-negative loss to a scalar. With ``neg_mask``
        (bool ``[B, M]``, True == real negative), masked negatives are excluded
        via a per-anchor masked mean; all-masked anchors contribute ``0``.
        """
        if neg_mask is None:
            return loss.mean()
        neg_mask = neg_mask.to(loss.dtype)
        denom = neg_mask.sum(dim=-1)
        per_anchor = (loss * neg_mask).sum(dim=-1)
        # Guard all-masked rows (0 real negatives) against div-by-0.
        safe_denom = denom.clamp(min=1.0)
        per_anchor = torch.where(
            denom > 0,
            per_anchor / safe_denom,
            torch.zeros_like(per_anchor),
        )
        return per_anchor.mean()

    def l2_reg_loss(
        self,
        lhs_x: Tensor,
        rhs_pos_x: Tensor,
        rhs_neg_x: Tensor | None,
    ) -> Tensor:

        l2_reg_loss = (lhs_x**2).sum(dim=1).mean()
        l2_reg_loss = l2_reg_loss + (rhs_pos_x**2).sum(dim=1).mean()
        if rhs_neg_x is not None:
            l2_reg_loss = l2_reg_loss + (rhs_neg_x**2).sum(dim=1).mean()
        return l2_reg_loss

    def loss(
        self,
        out: LinkPredHeadOutput,
        neg_mask: Tensor | None = None,
    ) -> Tensor:
        # [batch_size, 1]
        pos_score = out.pos_score.view(-1, 1)
        # [batch_size, neg_amount] for temporal link prediction.
        # [batch_size, neg_amount * batch_size] for static link prediction.
        neg_score = out.neg_score
        # `neg_mask` (True == real negative) comes only from the DG two-tower
        # per-anchor path; None elsewhere.
        if self.loss_fn == 'margin':
            loss = self.margin_loss(pos_score, neg_score, neg_mask)
        elif self.loss_fn == 'sampled_softmax':
            loss = self.sampled_softmax_loss(pos_score, neg_score, neg_mask)
        elif self.loss_fn == 'bpr':
            loss = self.bpr_loss(pos_score, neg_score, neg_mask)
        else:
            raise ValueError(f"Unknown link prediction loss '{self.loss_fn}'")

        # No need for L2 regularization if embeddings are already normalized.
        if not self.head.normalize_vectors:
            l2_reg_loss = self.l2_reg_loss(out.lhs_x, out.rhs_pos_x,
                                           out.rhs_neg_x)
            if self.emb_l2_reg_coeff > 0:
                loss = loss + self.emb_l2_reg_coeff * l2_reg_loss

        return loss

    def train_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        loss_name = TrainingStage.TRAIN.add_stage_prefix('loss')
        out = self.common_step(batch)
        # `[B, M]` bool mask (True == real negative) from the DG two-tower path;
        # None for legacy / shared-negative paths.
        neg_mask: Tensor | None = batch[self.rhs_name].get('dst_neg_mask')
        # Static LP pools negatives into a shared set ([B, B*M]), so the [B, M]
        # mask no longer aligns; it's all-true anyway (exhaustion is temporal),
        # so drop it and use the plain mean. Only temporal honors the mask.
        if self.task_type == TaskType.STATIC_LINK_PREDICTION:
            neg_mask = None
        loss = self.loss(out, neg_mask=neg_mask)
        batch_size = out.pos_score.size(0)
        self.log_learning_rate()
        self.log_epoch(loss_name, loss, batch_size=batch_size)

        return loss

    # Evaluation Callbacks ####################################################

    def on_evaluation_start(self, stage: TrainingStage):
        super().on_evaluation_start(stage)
        # TT-NG: the NodeRange RHS loader yields the full catalog, so set the
        # temporal valid-RHS set here; init_mips_structure then restricts the
        # index to it.
        self.valid_rhs_node_idx = self.tt_ng_valid_rhs_for_stage(stage)
        self.init_mips_structure(stage)
        assert self.is_mips_index_ready

    def evaluation_step(
        self,
        batch: HeteroData,
        stage: TrainingStage,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:
        out = self.common_step(batch)
        lhs_emb = out[self.lhs_name]
        lhs_emb = lhs_emb[:batch[self.lhs_name].batch_size]
        lhs_emb = self.head.lhs_forward(lhs_emb)
        exclude_rhs_index: Tensor | None = None
        exclude_lhs_batch: Tensor | None = None

        # TT-NG: skip past-visit exclusion for temporal LP (it queries
        # lp_train_table, which DiskgraphValidRHS doesn't implement) and rank
        # against the full RHS catalog. Deferred TT-NG parity item.
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(batch)
        else:
            assert self.task_type == TaskType.STATIC_LINK_PREDICTION
            rhs_y_index, lhs_y_batch, edge_weight = self.get_ground_truth(
                batch, stage=stage)
            # TT-NG: the adapter pre-resolved the prior-stage exclude set
            # onto the batch (global RHS space); remapped to valid space below.
            (exclude_rhs_index,
             exclude_lhs_batch) = self.get_tt_ng_exclude_ground_truth(batch)

        # TT-NG: the MIPS index covers only the temporally-valid RHS set, so it
        # returns valid-set-space positions while rhs_y_index is global. Map the
        # ground truth into valid-set space, then preds + y back to global before
        # update_metrics.
        valid_rhs = self.valid_rhs_node_idx
        if valid_rhs is not None:
            rhs_y_index, keep = map_index(rhs_y_index, valid_rhs,
                                          max_index=self.num_rhs_nodes,
                                          inclusive=False)
            lhs_y_batch = lhs_y_batch[keep]
            if edge_weight is not None:
                edge_weight = edge_weight[keep]
            # The exclude set is also in global RHS space — remap it the same way
            # (mips_index.search consumes exclude in valid-set space), dropping
            # excludes outside valid_rhs via the keep mask.
            if exclude_rhs_index is not None:
                exclude_rhs_index, ex_keep = map_index(
                    exclude_rhs_index, valid_rhs, max_index=self.num_rhs_nodes,
                    inclusive=False)
                assert exclude_lhs_batch is not None
                exclude_lhs_batch = exclude_lhs_batch[ex_keep]

        score = None
        if not self.disable_eval_metrics:
            score, rhs_pred_index_mat = self.get_rhs_pred_index_mat(
                None, lhs_emb, exclude_rhs_index, exclude_lhs_batch)
            self.update_metrics(
                stage, valid_rhs[rhs_pred_index_mat]
                if valid_rhs is not None else rhs_pred_index_mat,
                (lhs_y_batch, valid_rhs[rhs_y_index] if valid_rhs is not None
                 else rhs_y_index), edge_label_weight=edge_weight)

        if self.eval_top_k > 0:
            eval_top_k = min(self.eval_top_k, self.mips_index.num_keys)
            batch_size = batch[self.lhs_name].batch_size
            lhs_pkey = batch[self.lhs_name].pkey[:batch_size]
            lhs_index = batch[self.lhs_name].n_id[:batch_size]
            time: Tensor | None = None
            if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
                time = batch[self.train_table_name].time
            score, target = self.mips_index.search(lhs_emb, k=eval_top_k)
            # `target` is valid-set space (matched against rhs_y_index here),
            # then mapped back to global node-ids for the returned predictions.
            data = LinkPredMetricData(target, (lhs_y_batch, rhs_y_index))
            label = data.pred_rel_mat
            if valid_rhs is not None:
                target = valid_rhs[target]
            return dict(pkey=lhs_pkey, id=lhs_index, time=time, target=target,
                        score=score, label=label)

        return self.holdout_top_scores(stage, score)

    def evaluation_epoch_end(
        self,
        stage: TrainingStage,
        step_outputs: dict[str, list[Tensor]],
    ):
        super().evaluation_epoch_end(stage, step_outputs)
        self.clear_mips_index()

    # Prediction Callbacks ####################################################

    def on_predict_start(self):
        super().on_predict_start()
        # TT-NG temporal LP: restrict predict candidates to RHS valid at the
        # prediction anchor time (matches eval). None on static -> full catalog.
        self.valid_rhs_node_idx = self.tt_ng_valid_rhs_for_stage(
            TrainingStage.PRED)
        self.init_mips_structure(TrainingStage.PRED)
        assert self.is_mips_index_ready

    def on_predict_end(self):
        super().on_predict_end()
        self.clear_mips_index()

    def predict_step(
        self,
        batch: HeteroData,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:
        out = self.common_step(batch)

        lhs = out[self.lhs_name]
        batch_size = batch[self.lhs_name].batch_size
        global_ind = batch[self.lhs_name].n_id[:batch_size]
        lhs_pkey = batch[self.lhs_name].pkey[:batch_size]
        time: Tensor | None = None
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            time = batch[self.train_table_name].time
        lhs = lhs[:global_ind.numel()]
        lhs = self.head.lhs_forward(lhs)
        lhs = self.lhs_identity(lhs)  # do nothing. for getting lhs embeddings
        return_embeddings = getattr(self, '_kumo_return_predict_embeddings',
                                    False)

        # If the user is only interested in embeddings, avoid expensive
        # computation of top-k predictions:
        if self.pred_top_k is None or self.pred_top_k == 0:
            result = dict(pkey=lhs_pkey, id=global_ind, time=time)
            if return_embeddings:
                result['lhs_embedding'] = lhs.detach().cpu()
            return result

        # NOTE (Weihua): Exclusion mechanism for BP is removed at PR:6553
        pred_top_k = min(self.pred_top_k, self.mips_index.num_keys)
        score, rhs_pred_index_mat = self.mips_index.search(lhs, pred_top_k)

        # Map back to the original RHS node indices
        if self.valid_rhs_node_idx is not None:
            rhs_pred_index_mat = self.valid_rhs_node_idx[rhs_pred_index_mat]

        result = dict(pkey=lhs_pkey, id=global_ind, time=time,
                      target=rhs_pred_index_mat, score=score)
        if return_embeddings:
            result['lhs_embedding'] = lhs.detach().cpu()
        return result

    # Embedding Callbacks #####################################################

    def _get_register_embedding_hook_fn(self) -> Callable | None:
        # We hook to the output of the lhs identity, which contains the LHS
        # embeddings (post applying self.head.lhs_forward) used for the nearest
        # neighbor search.
        return self.lhs_identity.register_forward_hook

    def embedding_step(
        self,
        stage: TrainingStage,
        module: torch.nn.Module,
        inputs: tuple[Tensor, ...],
        outputs: Tensor,
    ) -> dict[str, Tensor]:
        return {self.lhs_emb_name: outputs}

    def embedding_epoch_end(
        self,
        stage: TrainingStage,
        outputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        # Attach RHS embeddings:
        outputs[self.rhs_emb_name] = self.mips_index._key.cpu()

        return outputs

    def validate_head(self):
        assert is_torch_instance(self.head, LinkPredHead)
        assert not self.head.use_mlp_head
        assert self.head.use_rhs_projector

    def get_embedding(self, batch, entity_name, side=None):
        batch_size = batch[entity_name].batch_size
        x_dict = self(
            batch.tf_dict,
            batch.collect('edge_index', allow_empty=True),
            batch.collect('time', allow_empty=True),
            batch.batch_dict,
            seed_time=batch[entity_name].get('seed_time'),
            num_sampled_nodes_dict=batch.collect('num_sampled_nodes',
                                                 allow_empty=True),
            num_sampled_edges_dict=batch.collect('num_sampled_edges',
                                                 allow_empty=True),
        )
        raw = x_dict[entity_name][:batch_size]
        if side == 'lhs':
            return self.head.lhs_forward(raw)
        elif side == 'rhs':
            return self.head.rhs_forward(raw)
