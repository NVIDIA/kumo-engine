# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import torch
from kumo_core.train.base_module import (
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_INTERVAL,
    DEFAULT_OPTIMIZER,
    BaseLightningModule,
)
from kumo_core.types import TaskType
from kumoapi.model_plan import WeightMode
from kumoapi.train import TrainingStage
from kumo_core.modeling.retrieval.mips import MIPSFaiss, MIPSStructure
from kumo_core.modeling.heads.link_prediction import LinkPredHead
from torch import Tensor
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    from kumo_core.diskgraph.lightning_data import DiskgraphValidRHS

logger = logging.getLogger(__name__)

# One-time guard so the array-LP empty-supervision warning isn't per-batch spam.
_WARNED_ARRAY_EMPTY_COO = False


class BaseLinkLightningModule(BaseLightningModule, ABC):
    r"""A common base class for all link prediction modules.

    Args:
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The GNN model.
        head (LinkPredHead): The model head.
        entity_table_names ((str, str)): The names of the entity tables.
    """
    def __init__(
        self,
        encoder: torch.nn.Module,
        model: torch.nn.Module,
        head: LinkPredHead,
        task_type: TaskType,
        entity_table_names: tuple[str, str],
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
        assert task_type in {
            TaskType.TEMPORAL_LINK_PREDICTION,
            TaskType.STATIC_LINK_PREDICTION,
        }
        assert weight_mode is None or weight_mode == WeightMode.WEIGHTED_LOSS

        super().__init__(
            task_type=task_type,
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
        self._valid_rhs_node_idx = None
        self.distance_measure = distance_measure
        self.weight_mode = weight_mode
        self.validate_head()

        if (not isinstance(entity_table_names, (list, tuple))
                or len(entity_table_names) != 2):
            raise ValueError(f"Expected two entity tables for link prediction "
                             f"task (got '{entity_table_names}')")

        self.entity_table_names = entity_table_names

    # General properties ######################################################

    @property
    def lhs_name(self) -> str:
        r"""Returns the LHS entity table name."""
        return self.entity_table_names[0]

    @property
    def rhs_name(self) -> str:
        r"""Returns the RHS entity table name."""
        return self.entity_table_names[1]

    @property
    def lhs_emb_name(self) -> str:
        r"""Returns a unique LHS entity table name for referencing embeddings
        in case LHS and RHS tables are identical."""
        if self.entity_table_names[0] == self.entity_table_names[1]:
            return f'{self.lhs_name}_SRC'
        return self.lhs_name

    @property
    def rhs_emb_name(self) -> str:
        r"""Returns a unique RHS entity table name for referencing embeddings
        in case LHS and RHS tables are identical."""
        if self.entity_table_names[0] == self.entity_table_names[1]:
            return f'{self.rhs_name}_DST'
        return self.rhs_name

    @property
    def lp_train_table(self) -> "DiskgraphValidRHS":
        r"""Returns the DiskGraph RHS validity helper attached to the module."""
        if hasattr(self, '_lp_train_table'):
            return self._lp_train_table  # type: ignore
        return self.trainer.datamodule.lp_train_table  # type: ignore

    @property
    def num_rhs_nodes(self) -> int:
        r"""Returns the number of RHS nodes."""
        if hasattr(self, '_num_rhs_nodes'):
            return self._num_rhs_nodes
        return self.lp_train_table.num_rhs_nodes

    @num_rhs_nodes.setter
    def num_rhs_nodes(self, value: int):
        self._num_rhs_nodes = value

    @property
    def valid_rhs_node_idx(self) -> Tensor | None:
        r"""Returns the RHS node indices to train or make prediction over.
        If :obj:`None`, we will predict against the entire RHS nodes.
        Otherwise, we will predict against RHS nodes specified via
        :obj:`valid_rhs_node_idx`"""
        return self._valid_rhs_node_idx

    @property
    def num_valid_rhs_nodes(self) -> int:
        if self.valid_rhs_node_idx is None:
            return self.num_rhs_nodes
        return len(self.valid_rhs_node_idx)

    @valid_rhs_node_idx.setter
    def valid_rhs_node_idx(self, value: Tensor | None):
        if value is not None and not self.supports_rhs_node_filtering:
            raise RuntimeError(f"{self.__class__.__name__} does not "
                               f"support RHS node filtering.")
        self._valid_rhs_node_idx = value

    @property
    def supports_rhs_node_filtering(self) -> bool:
        r"""Whether to support RHS node filtering based on
        :obj:`valid_rhs_node_idx`"""
        return True

    def on_predict_start(self):
        super().on_predict_start()
        if self._valid_rhs_node_idx is not None:
            self._valid_rhs_node_idx = self._valid_rhs_node_idx.to(self.device)

    def tt_ng_valid_rhs_for_stage(self, stage: "Any") -> "Tensor | None":
        """TT-NG temporal LP: RHS node-ids valid at ``stage``'s seed time, or
        ``None`` (static LP / no temporal info -> rank the full catalog).
        Shared by eval AND predict across LP families so both filter
        consistently."""
        from kumoapi.train import TrainingStage

        canonical = self._canonical_rhs_node_ids(self.device)
        if self.task_type != TaskType.TEMPORAL_LINK_PREDICTION:
            return canonical
        dm = self.datamodule
        if stage == TrainingStage.VAL:
            stage_time = dm.val_time
        elif stage == TrainingStage.TEST:
            stage_time = dm.test_time
        else:  # PRED
            stage_time = dm.predict_time
        if stage_time is None or self.lp_train_table.valid_rhs_idx_dict is None:
            return canonical
        valid = self.lp_train_table.get_valid_rhs_idx(int(stage_time[0])).to(
            self.device)
        return self._filter_to_canonical_rhs_node_ids(valid, self.device)

    def _canonical_rhs_node_ids(
            self, device: torch.device | None = None) -> Tensor | None:
        canonical = getattr(self, 'canonical_rhs_node_idx', None)
        if canonical is None or device is None or canonical.device == device:
            return canonical
        return canonical.to(device)

    def _filter_to_canonical_rhs_node_ids(
        self,
        valid: Tensor | None,
        device: torch.device | None = None,
    ) -> Tensor | None:
        canonical = self._canonical_rhs_node_ids(device)
        if valid is None or canonical is None:
            return canonical if valid is None else valid
        positions = torch.searchsorted(canonical, valid)
        in_range = positions < canonical.numel()
        if canonical.numel():
            probe = positions.clamp_max(canonical.numel() - 1)
            in_range &= canonical[probe] == valid
        return valid[in_range]

    # Top-K prediction ########################################################

    def init_mips_index(self, rhs_emb: Tensor):
        r"""Initializes MIPS index for the given RHS embeddings. Useful for
        fast top-:math:`k` MIPS of RHS nodes."""
        if self.is_mips_index_initialized:
            raise ValueError("MIPS index already initialized. Please detach "
                             "it first by running `module.clear_mips_index()`")

        self._mips_index = MIPSFaiss(rhs_emb)

    @property
    def is_mips_index_initialized(self) -> bool:
        return hasattr(self, '_mips_index')

    @property
    def is_mips_index_ready(self) -> bool:
        r"""Judge whether MIPS index is ready or not."""
        return (self.is_mips_index_initialized
                and (self.mips_index.num_keys == self.num_valid_rhs_nodes))

    @property
    def mips_index(self) -> MIPSStructure:
        r"""Returns the MIPS index for fast top-math:`k` querying of RHS
        nodes."""
        if not self.is_mips_index_initialized:
            raise ValueError("MIPS index not initialized yet. Please specify "
                             "it via `module.init_mips_index(...)`")
        return self._mips_index

    def clear_mips_index(self):
        r"""Removes the given MIPS index."""
        if self.is_mips_index_initialized:
            del self._mips_index

    def garbage_collect(self):
        r"""Force garbage collect on unreferenced tensors."""
        if 'PYTEST_CURRENT_TEST' in os.environ:
            # No need for explicit garbage collect in tests since there do not
            # exist large memory requirements and it slows down tests by a bit:
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @staticmethod
    def holdout_top_scores(stage: TrainingStage,
                           score: Tensor | None) -> dict[str, Tensor]:
        r"""Each test entity's top score, for the holdout's counts."""
        if stage != TrainingStage.TEST or score is None:
            return {}
        return {'top_score': score.max(dim=-1).values}

    def get_rhs_pred_index_mat(
        self,
        logits: Tensor | None,
        lhs_emb: Tensor | None,
        exclude_rhs_index: Tensor | None = None,
        exclude_lhs_batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if logits is None and lhs_emb is None:
            raise ValueError("Either 'logits' or 'lhs_emb' must be provided "
                             "for computing metrics")

        k = self.metric_store.get_metric_collection(TrainingStage.TRAIN).max_k
        if logits is not None:
            assert exclude_rhs_index is None
            assert exclude_lhs_batch is None
            score, index = logits.topk(min(k, logits.size(1)), dim=-1)
        else:
            assert lhs_emb is not None
            score, index = self.mips_index.search(
                lhs_emb,
                k=k,
                exclude=(exclude_lhs_batch, exclude_rhs_index)
                if exclude_lhs_batch is not None else None,
            )

        return score, index

    def get_ground_truth(
        self,
        batch: HeteroData,
        stage: TrainingStage | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        r"""Given :obj:`batch` object, returns their RHS node indices.

        Args:
            batch (HeteroData): The batch object.
            stage (TrainingStage, optional): Specifying train/val/test stages.
                (default: :obj:`None`)

        Returns:
            rhs_index (torch.Tensor): The RHS node indices.
            lhs_batch (torch.Tensor): Assigns each output RHS node to the
                corresponding LHS node query entry.
            edge_weights (torch.Tensor, optional): The edge weights.
                Returns :class:`Tensor` object if edge weights exist.
                Otherwise, returns :obj:`None`.
        """
        batch_size = batch[self.lhs_name].batch_size
        lhs_index = batch[self.lhs_name].n_id[:batch_size]
        lhs_block = batch[self.lhs_name]
        # Ragged multi-positive COO ([2, ΣP]) from the adapter; preferred
        # over the diagonal y/arange(B). Use only when non-empty — an empty
        # COO would feed the loss zero positives (silent train_loss=0.0).
        edge_label_index = getattr(lhs_block, 'edge_label_index', None)
        if (edge_label_index is not None and edge_label_index.numel() > 0):
            rhs_index = edge_label_index[1]
            lhs_batch = edge_label_index[0]
            edge_weights = getattr(lhs_block, 'edge_weight', None)
            return rhs_index, lhs_batch, edge_weights
        # Empty COO: resolver emitted no positives. Warn once, fall to ``.y``.
        global _WARNED_ARRAY_EMPTY_COO
        if (edge_label_index is not None and edge_label_index.numel() == 0
                and not _WARNED_ARRAY_EMPTY_COO):
            _WARNED_ARRAY_EMPTY_COO = True
            logger.warning("array LP supervision empty (resolver produced no "
                           "positives?) — falling back to single-positive .y; "
                           "check lp_resolution wiring.")
        y = getattr(lhs_block, 'y', None)
        if y is not None:
            rhs_index = y.long().view(-1)
            lhs_batch = torch.arange(batch_size, device=rhs_index.device)
            edge_weights = getattr(lhs_block, 'weight', None)
            # Drop rows whose TARGET1 PK didn't resolve (target_mask True ==
            # unresolved); mask rather than fail to stay robust.
            target_mask = getattr(lhs_block, 'target_mask', None)
            if target_mask is not None:
                tm = target_mask.to(rhs_index.device).view(-1).bool()
                valid = ~tm
                rhs_index = rhs_index[valid]
                lhs_batch = lhs_batch[valid]
                if edge_weights is not None:
                    edge_weights = edge_weights[valid]
            return rhs_index, lhs_batch, edge_weights
        # No COO and no ``.y``: the legacy lp_train_table.query() fall-through
        # is invalid on TT-NG, so return empty supervision and continue.
        empty = torch.empty(0, dtype=torch.long, device=lhs_index.device)
        return empty, empty, None

    def get_new_and_past_visit_ground_truth(
        self,
        batch: HeteroData,
        rhs_index: Tensor,
        lhs_batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Given the ground truth, return the new-visit ground-truth together
        with excluding ground-truth.

        Args:
            batch (HeteroData): The heterogeneous mini-batch.
            rhs_index (Tensor): rhs_index of ground-truth
            lhs_batch (Tensor): lhs_batch of ground-truth

        Returns:
            rhs_index (Tensor): rhs_index of new-visit ground-truth
            lhs_batch (Tensor): lhs_batch of new-visit ground-truth
            exclude_rhs_index (Tensor): rhs_index of past-visit ground-truth
            exclude_lhs_batch (Tensor): lhs_batch of past-visit ground-truth
        """
        batch_size = batch[self.lhs_name].batch_size
        # Get past visit links
        (exclude_rhs_index,
         exclude_lhs_batch) = self.get_past_visit_ground_truth(batch)
        # Exclude past visits from the ground-truth
        isin = torch.isin(rhs_index * batch_size + lhs_batch,
                          exclude_rhs_index * batch_size + exclude_lhs_batch)
        rhs_index = rhs_index[~isin]
        lhs_batch = lhs_batch[~isin]
        return rhs_index, lhs_batch, exclude_rhs_index, exclude_lhs_batch

    def get_past_visit_ground_truth(
        self,
        batch: HeteroData,
        stage: TrainingStage | None = None,
    ) -> tuple[Tensor, Tensor]:
        # Exclusive of now.
        lhs_index = batch[self.lhs_name].n_id[:batch[self.lhs_name].batch_size]
        if self.task_type == TaskType.TEMPORAL_LINK_PREDICTION:
            rhs_index, lhs_batch, _ = self.lp_train_table.query_until(
                lhs_index, time=batch[self.lhs_name].get('seed_time') - 1)
        else:
            assert stage is not None
            if stage == TrainingStage.VAL:
                prior_stage = TrainingStage.TRAIN
            elif stage == TrainingStage.TEST:
                prior_stage = TrainingStage.VAL
            elif stage == TrainingStage.PRED:
                prior_stage = TrainingStage.TEST
            else:
                raise RuntimeError(f"No prior stage for '{stage.value}'")

            rhs_index, lhs_batch, _ = self.lp_train_table.query_stage_until(
                lhs_index, stage=prior_stage)
        return rhs_index, lhs_batch

    def get_tt_ng_exclude_ground_truth(
        self,
        batch: HeteroData,
    ) -> tuple[Tensor | None, Tensor | None]:
        """TT-NG static-eval past-visit exclude set, read off the batch.

        The adapter pre-resolves the prior-stage targets into the ragged
        ``exclude_(lhs_batch, rhs_index)`` fields on the LHS block (global RHS
        space). Returns ``(None, None)`` when absent (no excludes / non-static).
        """
        lhs_block = batch[self.lhs_name]
        exclude_rhs_index = getattr(lhs_block, 'exclude_rhs_index', None)
        exclude_lhs_batch = getattr(lhs_block, 'exclude_lhs_batch', None)
        return exclude_rhs_index, exclude_lhs_batch

    @abstractmethod
    def validate_head(self):
        r"""Validates the configuration of the link prediction head."""
