# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any

import pandas as pd
import torch
from kumo_core.train.autoregression import (
    AUTOREGRESSION_COL,
    FUTURE_LABELS_COL,
    YOY_COL,
)
from kumo_core.train.base_module import (
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_INTERVAL,
    DEFAULT_OPTIMIZER,
)
from kumo_core.train.base_node_module import BaseNodeLightningModule
from kumo_core.train.loss import DistributionLoss, MultiQuantileLoss
from kumo_core.types import TaskType, TrainModuleOutput
from kumoapi.train import TrainingStage
from kumoapi.typing import Stype
from kumo_core.encoders.time import TimeEncoder
from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.types.features import NANOSECONDS_IN_DAY
from kumo_core.training.supervision.forecasting import ForecastingTrainTable
from kumo_core.modeling.gnn.transformer import GraphTransformer
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame import stype as torch_stype
from torch_frame.data import MultiEmbeddingTensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType, NodeType


class ForecastingGNNLightningModule(BaseNodeLightningModule):
    r"""A PyTorch Lightning module that handles the execution of GNN-based
    node-level forecasting training, validation, testing and prediction
    routines.

    :class:`ForecastingGNNLightningModule` works by requesting past and future
    sequences from the :class:`ForecastingTrainTable`, which are used as
    autoregressive input features and target labels, respectively.

    During training, given a random node and seed time of the training split,
    we compute a node representation given a GNN and its autoregressive input
    features, and then either:

    1) Predict the whole sequence at once (Sequence-MLP Head)
    2) Predict the whole sequence autoregressively (Auto-Regrssive Head)
    3) Predict sampled future timestamps implicitly (Implicit Head)

    During evaluation, given a random node and the its start time of the
    validation/test split, we compute node representations once, and then use
    them to predict the whole sequence at once (independent of model head).

    During prediction, we rely on autoregressive input features being generated
    directly from Predictive Query, as we do not have access to the training
    table during prediction (as it might be also out-dated).
    This leads to some change in logic, since the autoregressive input features
    are no longer coming from :class:`ForecastingTrainTable`, but are instead
    part of the input mini-batch in :class:`HeteroData`.

    Args:
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The message-passing model.
        head (torch.nn.Module): The model head.
        train_table_name (str): The name of the training table.
        entity_table_name (str): The name of the entity table.
        arl_length (int): The number of past targets to use as autoregressive
            input features.
        forecast_length (int): The forecasting length, *i.e* the sequence
            length to predict.
    """
    HOLIDAY_LOSS_FACTOR: float = 10.0

    def __init__(
        self,
        encoder: torch.nn.Module,
        model: torch.nn.Module,
        head: torch.nn.Module,
        train_table_name: str,
        entity_table_name: str,
        arl_length: int,
        forecast_length: int,
        unix_step_size: int,
        # `BaseNodeLightningModule` arguments:
        target_encoder: TargetEncoder,
        past_encoder: torch.nn.Module | None = None,
        year_over_year_encoder: torch.nn.Module | None = None,
        loss: str | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        metrics: list[str] | None = None,
        optimizer: str = DEFAULT_OPTIMIZER,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: str | None = DEFAULT_LR_SCHEDULER,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_interval: str = DEFAULT_LR_SCHEDULER_INTERVAL,
        max_epochs_for_lr_scheduler: int = 100,
        enable_compile: bool = False,
    ) -> None:
        super().__init__(
            task_type=TaskType.FORECASTING,
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
        self.time_encoder = TimeEncoder(unix_step_size)
        self.past_encoder = past_encoder
        self.year_over_year_encoder = year_over_year_encoder
        self.model = model
        self.head = head
        self.train_table_name = train_table_name
        self.entity_table_name = entity_table_name
        self.arl_length = arl_length
        self.forecast_length = forecast_length
        self.unix_step_size = unix_step_size
        # Set by DiskgraphNestedTrainer for TT-NG task-dataset batches.
        self.use_precomputed_forecasting_sequences = False

    def set_index_map(self, index_map: pd.DataFrame):
        self.index_map = index_map

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ):
        out = super().load_state_dict(state_dict, strict)

        if hasattr(self, 'index_map'):
            self.head.order_by(self.index_map)
            del self.index_map

        return out

    # Training Table Access ###################################################

    @property
    def train_table(self) -> ForecastingTrainTable:
        r"""Returns the :class:`ForecastingTrainTable` from the module."""
        if hasattr(self, '_fc_train_table'):
            return self._fc_train_table  # type: ignore
        return self.trainer.datamodule.train_table  # type: ignore

    def get_future_targets(self, batch: HeteroData) -> Tensor:
        r"""Given a :class:`HeteroData` mini-batch, returns :obj:`num_targets`
        many future targets for every seed node with the target encoder's
        normalization."""
        if self.use_precomputed_forecasting_sequences:
            future = self._sequence_from_batch(
                batch,
                FUTURE_LABELS_COL,
                self.forecast_length,
            )
            if future is not None:
                return self.target_encoder.encode_tensor(future)  # type: ignore

        batch_size = batch[self.entity_table_name].batch_size
        # Query the past targets from the train table:
        future = self.train_table.query_future(
            node_id=batch[self.entity_table_name].n_id[:batch_size],
            seed_time=batch[self.entity_table_name].seed_time[:batch_size],
            length=self.forecast_length,
        )
        # Normalize and return the future targets:
        return self.target_encoder.encode_tensor(future)  # type: ignore

    def get_past_targets(
        self,
        batch: HeteroData,
    ) -> Tensor:
        r"""Given a :class:`HeteroData` mini-batch, returns :obj:`arl_length`
        many past targets for every seed node with the target encoder's
        normalization."""
        if self.use_precomputed_forecasting_sequences:
            past = self._sequence_from_batch(
                batch,
                AUTOREGRESSION_COL,
                self.arl_length,
            )
            if past is not None:
                return self.target_encoder.encode_tensor(past)  # type: ignore

        batch_size = batch[self.entity_table_name].batch_size
        # Query the past targets from the train table:
        past = self.train_table.query_past(
            node_id=batch[self.entity_table_name].n_id[:batch_size],
            seed_time=batch[self.entity_table_name].seed_time[:batch_size],
            length=self.arl_length,
        )
        # Normalize and return the past targets:
        return self.target_encoder.encode_tensor(past)  # type: ignore

    def get_past_year_targets(
        self,
        batch: HeteroData,
    ) -> Tensor:
        r"""Given a :class:`HeteroData` mini-batch, returns :obj:`num_targets`
        many past targets of the previous year for every seed node."""
        assert self.year_over_year_encoder is not None
        padding_size = self.year_over_year_encoder.kernel_size * 2
        if self.use_precomputed_forecasting_sequences:
            past = self._sequence_from_batch(
                batch,
                YOY_COL,
                self.forecast_length + 2 * padding_size,
            )
            if past is not None:
                past = self.target_encoder.encode_tensor(past)  # type: ignore
                past = past.unsqueeze(-1)
                seed_time = batch[self.train_table_name].time[:past.size(0)]
                yoy_time = seed_time - NANOSECONDS_IN_DAY * 364
                time_enc = self.time_encoder(yoy_time, self.forecast_length,
                                             padding_size)
                past = torch.cat([past, time_enc], dim=-1)
                return self.year_over_year_encoder(past)

        batch_size = batch[self.entity_table_name].batch_size
        past, time = self.train_table.query_past_year(
            node_id=batch[self.entity_table_name].n_id[:batch_size],
            seed_time=batch[self.entity_table_name].seed_time[:batch_size],
            length=self.forecast_length,
            padding_size=padding_size,
        )

        past = self.target_encoder.encode_tensor(past)  # type: ignore
        past = past.unsqueeze(-1)
        time_enc = self.time_encoder(time, self.forecast_length, padding_size)

        past = torch.cat([past, time_enc], dim=-1)
        return self.year_over_year_encoder(past)

    def _sequence_from_batch(
        self,
        batch: HeteroData,
        col_name: str,
        length: int,
    ) -> Tensor | None:
        block = batch[self.train_table_name]
        if 'tf' not in block:
            return None
        tf = block.tf
        return self._sequence_from_tensor_frame(tf, col_name, length)

    def _sequence_from_tensor_frame(
        self,
        tf: TensorFrame,
        col_name: str,
        length: int,
    ) -> Tensor | None:
        for stype_key in (Stype.sequence.value, torch_stype.sequence_numerical):
            col_names = tf.col_names_dict.get(stype_key, [])
            if col_name not in col_names:
                continue
            feat = tf.feat_dict[stype_key]
            index = col_names.index(col_name)
            assert isinstance(feat, MultiEmbeddingTensor)
            values = feat[:, index].values.nan_to_num()
            assert isinstance(values, Tensor)
            assert values.size(1) == length
            return values
        return None

    def _batch_seed_node_ids_and_times(
        self,
        batch: HeteroData,
    ) -> tuple[Tensor, Tensor]:
        batch_size = batch[self.entity_table_name].batch_size
        node_ids = batch[self.entity_table_name].n_id[:batch_size]
        seed_time = batch[self.train_table_name].time[:batch_size]
        return node_ids, seed_time

    def _raw_future_targets(self, batch: HeteroData) -> Tensor | None:
        if self.use_precomputed_forecasting_sequences:
            return self._sequence_from_batch(
                batch,
                FUTURE_LABELS_COL,
                self.forecast_length,
            )
        node_ids, seed_time = self._batch_seed_node_ids_and_times(batch)
        return self.train_table.query_future(
            node_id=node_ids,
            seed_time=seed_time,
            length=self.forecast_length,
        )

    def _raw_past_targets(self, batch: HeteroData) -> Tensor | None:
        if self.arl_length == 0:
            _, seed_time = self._batch_seed_node_ids_and_times(batch)
            return torch.empty(
                seed_time.size(0),
                0,
                device=seed_time.device,
                dtype=torch.float32,
            )
        if self.use_precomputed_forecasting_sequences:
            return self._sequence_from_batch(
                batch,
                AUTOREGRESSION_COL,
                self.arl_length,
            )
        node_ids, seed_time = self._batch_seed_node_ids_and_times(batch)
        return self.train_table.query_past(
            node_id=node_ids,
            seed_time=seed_time,
            length=self.arl_length,
        )

    def _raw_past_year_targets(
        self,
        batch: HeteroData,
    ) -> tuple[Tensor | None, Tensor | None]:
        if self.year_over_year_encoder is None:
            return None, None

        padding_size = self.year_over_year_encoder.kernel_size * 2
        length = self.forecast_length + 2 * padding_size
        if self.use_precomputed_forecasting_sequences:
            past = self._sequence_from_batch(batch, YOY_COL, length)
            if past is None:
                return None, None
            seed_time = batch[self.train_table_name].time[:past.size(0)]
            yoy_time = seed_time - NANOSECONDS_IN_DAY * 364
            return past, yoy_time

        node_ids, seed_time = self._batch_seed_node_ids_and_times(batch)
        return self.train_table.query_past_year(
            node_id=node_ids,
            seed_time=seed_time,
            length=self.forecast_length,
            padding_size=padding_size,
        )

    def _dump_tensor(self, value: Tensor | None) -> Tensor | None:
        if value is None:
            return None
        return value.detach().cpu()

    def _batch_dump_extra_payload(
        self,
        batch: HeteroData,
        stage: str,
    ) -> dict[str, Any] | None:
        del stage
        if not isinstance(batch, HeteroData):
            return None
        if self.use_precomputed_forecasting_sequences:
            return None

        with torch.no_grad():
            payload: dict[str, Tensor | None] = {
                FUTURE_LABELS_COL:
                self._dump_tensor(self._raw_future_targets(batch)),
                AUTOREGRESSION_COL:
                self._dump_tensor(self._raw_past_targets(batch)),
            }

            past_year_raw, past_year_time = self._raw_past_year_targets(batch)
            if past_year_raw is not None:
                payload[YOY_COL] = self._dump_tensor(past_year_raw)
            if past_year_time is not None:
                payload['year_over_year_time'] = self._dump_tensor(
                    past_year_time)

        return {'forecasting': payload} if payload else None

    # Model Execution #########################################################

    def forward(
        self,
        tf_dict: dict[NodeType, TensorFrame],
        edge_index_dict: dict[EdgeType, Tensor],
        time_dict: dict[NodeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
        node_id: Tensor,
        seed_time: Tensor,
        past: Tensor | None = None,
        past_year: Tensor | None = None,
        num_sampled_nodes_dict: dict[NodeType, list[int]] | None = None,
        num_sampled_edges_dict: dict[EdgeType, list[int]] | None = None,
    ) -> Tensor:

        # In case of `trainer.predict()`, we expect that autoregressive
        # features are part of the input mini-batch.
        if past is None and self.arl_length > 0:
            tf = tf_dict[self.train_table_name]
            past = self._sequence_from_tensor_frame(tf, AUTOREGRESSION_COL,
                                                    self.arl_length)
            assert past is not None
            # Normalize past targets according to target encoder:
            past = self.target_encoder.encode_tensor(past)  # type: ignore
        elif past is None and self.arl_length == 0:
            past = torch.empty(seed_time.size(0), 0, device=seed_time.device)
        assert past is not None

        if self.year_over_year_encoder is not None and past_year is None:
            tf = tf_dict[self.train_table_name]
            padding_size = self.year_over_year_encoder.kernel_size * 2
            past_year = self._sequence_from_tensor_frame(
                tf, YOY_COL, self.forecast_length + 2 * padding_size)
            assert past_year is not None
            assert past_year.size(1) == self.forecast_length + 2 * padding_size
            # Normalize past targets according to target encoder:
            past_year = self.target_encoder.encode_tensor(  # type: ignore
                past_year)
            past_year = past_year.unsqueeze(-1)  # type: ignore
            yoy_time = seed_time - NANOSECONDS_IN_DAY * 364
            time_enc = self.time_encoder(yoy_time, self.forecast_length,
                                         padding_size)
            past_year = torch.cat([past_year, time_enc], dim=-1)
            past_year = self.year_over_year_encoder(past_year)

        if self.past_encoder is not None:
            past = self.past_encoder(past)
            assert isinstance(past, Tensor)

        tf_dict.pop(self.train_table_name, None)
        batch_dict.pop(self.train_table_name, None)

        x_dict = self.encoder(tf_dict, batch_dict, time_dict, seed_time)
        batch_size = seed_time.numel()

        if isinstance(self.model, GraphTransformer):
            # x shape: [batch_size, seq_len, channels]
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

        return self.head(
            x,
            length=self.forecast_length,
            past=past,
            past_year=past_year,
            pred_time=seed_time,
            node_id=node_id[:batch_size],
        )

    def common_step(self, batch: HeteroData, predict: bool = False) -> Tensor:
        past_year = None
        if self.year_over_year_encoder is not None and not predict:
            past_year = self.get_past_year_targets(batch)

        return self(
            batch.tf_dict,
            batch.collect('edge_index', allow_empty=True),
            batch.collect('time', allow_empty=True),
            batch.batch_dict,
            node_id=batch[self.entity_table_name].n_id,
            seed_time=batch[self.train_table_name].time,
            past=self.get_past_targets(batch) if not predict else None,
            past_year=past_year,
            num_sampled_nodes_dict=batch.collect('num_sampled_nodes', True),
            num_sampled_edges_dict=batch.collect('num_sampled_edges', True),
        )

    def get_train_weight(self, time: Tensor) -> Tensor | None:
        r"""Weigh the error during holidays by a factor of
        `HOLIDAY_LOSS_FACTOR`. Only computed in case forecast step sizes are
        smaller than or equal to 1 day."""
        if self.unix_step_size > NANOSECONDS_IN_DAY:
            return None

        mask = self.time_encoder.is_holiday(time, self.forecast_length)
        return torch.where(mask, self.HOLIDAY_LOSS_FACTOR, 1.0)

    # Training Callbacks ######################################################

    def train_step(self, batch: HeteroData, batch_idx: int) -> Tensor:
        out: Tensor | dict[str, Tensor] = self.common_step(batch)
        if isinstance(out, Tensor):
            out = out.squeeze(-1)
        else:
            assert isinstance(out, dict)
            out = {key: value.squeeze(-1) for key, value in out.items()}

        assert self.target_encoder is not None
        y = self.get_future_targets(batch)

        weight = self.get_train_weight(batch[self.train_table_name].time)
        loss = self.loss(out, y, weight=weight)
        loss_name = TrainingStage.TRAIN.add_stage_prefix('loss')

        self.log_learning_rate()
        self.log_epoch(loss_name, loss, batch_size=y.size(0))

        if isinstance(self.loss, DistributionLoss):
            out = self.loss.distribution.to_prediction(out)  # type: ignore
        elif isinstance(self.loss, MultiQuantileLoss):
            out = self.loss.to_prediction(out)

        assert isinstance(out, Tensor)
        self.update_metrics(TrainingStage.TRAIN, out, y)

        return loss

    # Evaluation Callbacks ####################################################

    def evaluation_step(
        self,
        batch: HeteroData,
        stage: TrainingStage,
        batch_idx: int,
    ) -> TrainModuleOutput:
        out: Tensor | dict[str, Tensor] = self.common_step(batch)
        batch_size = batch[self.entity_table_name].batch_size
        pkey = batch[self.entity_table_name].pkey[:batch_size]
        time = batch[self.train_table_name].time
        if isinstance(out, Tensor):
            out = out.squeeze(-1)
        else:
            assert isinstance(out, dict)
            out = {key: value.squeeze(-1) for key, value in out.items()}

        assert self.target_encoder is not None
        y = self.get_future_targets(batch)

        weight = self.get_train_weight(batch[self.train_table_name].time)
        loss = self.loss(out, y, weight=weight)
        loss_name = stage.add_stage_prefix('loss')

        self.log_epoch(loss_name, loss, batch_size=y.size(0))

        distribution = None
        if isinstance(self.loss, DistributionLoss):
            distribution = out
            out = self.loss.to_prediction(out)  # type: ignore

        assert isinstance(out, Tensor)
        self.update_metrics(
            stage,
            self.loss.to_prediction(out)
            if isinstance(self.loss, MultiQuantileLoss) else out, y)

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
        embedding: Tensor | None = None
        handle = None
        if return_embeddings:
            register_fn = self._get_register_embedding_hook_fn()
            assert register_fn is not None

            def capture_embedding(module, inputs) -> None:
                nonlocal embedding
                embedding = inputs[0]

            handle = register_fn(capture_embedding)
        try:
            out: Tensor | dict[str, Tensor] = self.common_step(
                batch, predict=True)
        finally:
            if handle is not None:
                handle.remove()
        batch_size = batch[self.entity_table_name].batch_size
        pkey = batch[self.entity_table_name].pkey[:batch_size]
        time = batch[self.train_table_name].time
        if isinstance(out, Tensor):
            out = out.squeeze(-1)
        else:
            assert isinstance(out, dict)
            out = {key: value.squeeze(-1) for key, value in out.items()}
        result = dict(pkey=pkey, time=time, logits=out)
        if return_embeddings:
            if embedding is None:
                raise RuntimeError(
                    'Forecasting prediction did not produce requested '
                    'embeddings')
            result['embedding'] = embedding.detach().cpu()
        return result  # type: ignore

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

    # Helpers #################################################################

    def get_future_unix(self, seed_time: Tensor) -> Tensor:
        # Given seed_time and forecast_length and unix_step_size, computes
        # an array of unix times for the forecasting horizon
        unix_time = seed_time.view(-1, 1).expand(-1, self.forecast_length)
        offsets = torch.arange(self.forecast_length) * self.unix_step_size
        offsets = offsets.to(device=unix_time.device)
        return unix_time + offsets.view(1, -1)
