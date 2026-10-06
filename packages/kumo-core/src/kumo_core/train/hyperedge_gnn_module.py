# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from kumo_core.train.base_node_module import (
    DEFAULT_LR_SCHEDULER,
    DEFAULT_LR_SCHEDULER_INTERVAL,
    DEFAULT_OPTIMIZER,
    BaseNodeLightningModule,
)
from kumo_core.types import TaskType
from kumoapi.train import TrainingStage
from kumo_core.features.stores.feature import FeatureStore
from kumo_core.features.stores.graph import GraphStore
from kumo_core.modeling.gnn.hetero import HeteroGNN
from kumo_core.encoders.tensor_frame_encoder import HeteroTensorFrameEncoder
from kumo_core.encoders.target.base import TargetEncoder
from kumo_core.modeling.heads.node_prediction import NodePredHead
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric.data import HeteroData
from torch_geometric.data.lightning.datamodule import LightningData
from torch_geometric.loader import NodeLoader, ZipLoader
from torch_geometric.sampler import BaseSampler
from torch_geometric.typing import EdgeType, NodeType


class HyperedgeLightningData(LightningData):
    r"""A custom PyTorch Lightning data module for hyperedge-level predictions.
    Provides training, validation, testing and prediction data loaders that
    return separate subgraphs for each node in a hyperedge.

    Args:
        data (Tuple[:class:`FeatureStore`, :class:`GraphStore`]): The feature
            store and graph store to operate on.
        node_sampler (BaseSampler): The node sampler to use.
        input_train_fkeys (List[Tuple[str, torch.Tensor]): The node types and
            foreign keys of training hyperedges.
        input_train_time (torch.Tensor, optional): The timestamp of training
            hyperedges. (default: :obj:`None`)
        input_val_fkeys (List[Tuple[str, torch.Tensor], optional): The node
            types and foreign keys of validation hyperedges.
            (default: :obj:`None`)
        input_val_time (torch.Tensor, optional): The timestamp of validation
            hyperedges. (default: :obj:`None`)
        input_test_fkeys (List[Tuple[str, torch.Tensor], optional): The node
            types and foreign keys of test hyperedges. (default: :obj:`None`)
        input_test_time (torch.Tensor, optional): The timestamp of test
            hyperedges. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of :class:`LightningData`.
    """
    def __init__(
        self,
        data: tuple[FeatureStore, GraphStore],
        graph_sampler: BaseSampler,
        input_train_fkeys: list[tuple[NodeType, Tensor]],
        input_train_time: Tensor | None = None,
        input_val_fkeys: list[tuple[NodeType, Tensor]] | None = None,
        input_val_time: Tensor | None = None,
        input_test_fkeys: list[tuple[NodeType, Tensor]] | None = None,
        input_test_time: Tensor | None = None,
        **kwargs,
    ):
        # TODO (matthias) Measure training throughput and consider
        # optimizations of `ZipLoader` in the future since we are basically
        # doing sequential sampling and feature fetching here.
        super().__init__(
            data,
            graph_sampler=graph_sampler,
            has_val=input_val_fkeys is not None,
            has_test=input_test_fkeys is not None,
            **kwargs,
        )

        self.input_train_fkeys = input_train_fkeys
        self.input_train_time = input_train_time
        self.input_train_id: Tensor | None = None

        self.input_val_fkeys = input_val_fkeys
        self.input_val_time = input_val_time
        self.input_val_id: Tensor | None = None

        self.input_test_fkeys = input_test_fkeys
        self.input_test_time = input_test_time
        self.input_test_id: Tensor | None = None

    def dataloader(
        self,
        input_fkeys: list[tuple[str, Tensor]],
        input_time: Tensor | None = None,
        input_id: Tensor | None = None,
        **kwargs,
    ) -> ZipLoader:

        # ZipLoader will take care of worker handling, so it is safe to disable
        # it for individual node loaders:
        individual_kwargs = copy.copy(kwargs)
        individual_kwargs['num_workers'] = 0
        individual_kwargs['persistent_workers'] = False

        loaders: list[NodeLoader] = []
        for input_fkey in input_fkeys:
            loader = NodeLoader(
                self.data,
                node_sampler=self.graph_sampler,
                input_nodes=input_fkey,
                input_time=input_time,
                input_id=input_id,
                **individual_kwargs,
            )
            loaders.append(loader)

        # Remove keyword arguments that are not present in `ZipLoader`:
        kwargs.pop('transform_sampler_output', None)
        kwargs.pop('custom_cls', None)

        return ZipLoader(loaders, **kwargs)

    def train_dataloader(self) -> ZipLoader:
        return self.dataloader(
            self.input_train_fkeys,
            self.input_train_time,
            self.input_train_id,
            shuffle=self.train_shuffle,
            **self.loader_kwargs,
        )

    def val_dataloader(self) -> ZipLoader:
        assert self.input_val_fkeys is not None
        return self.dataloader(
            self.input_val_fkeys,
            self.input_val_time,
            self.input_val_id,
            shuffle=False,
            **self.eval_loader_kwargs,
        )

    def test_dataloader(self) -> ZipLoader:
        assert self.input_test_fkeys is not None
        return self.dataloader(
            self.input_test_fkeys,
            self.input_test_time,
            self.input_test_id,
            shuffle=False,
            **self.eval_loader_kwargs,
        )


class HyperedgeGNNLightningModule(BaseNodeLightningModule):
    r"""A PyTorch Lightning module that handles the execution of GNN-based
    training, validation, testing and prediction routines in case multiple
    foreign keys exist in the training table, i.e., when making a
    hyperedge-level predictions.

    Args:
        encoder (torch.nn.Module): The encoder model.
        model (torch.nn.Module): The GNN model.
        head (torch.nn.Module): The model head.
        train_table_name (str): The name of the training table.
        entity_table_names (List[str]): The name of the entity tables.
        with_identity_awareness (bool, optional): If set to :obj:`False`, will
            not add a node coloring to distinguish sampled nodes which are also
            present in the hyperedge. (default: :obj:`True`)
    """
    def __init__(
        self,
        encoder: HeteroTensorFrameEncoder,
        model: HeteroGNN,
        head: NodePredHead,
        task_type: TaskType,
        train_table_name: str,
        entity_table_names: list[str],
        with_identity_awareness: bool = True,
        # `BaseNodeLightningModule` arguments:
        loss: str | None = None,
        loss_kwargs: dict[str, Any] | None = None,
        target_encoder: TargetEncoder | None = None,
        metrics: list[str] | None = None,
        optimizer: str = DEFAULT_OPTIMIZER,
        optimizer_kwargs: dict[str, Any] | None = None,
        lr_scheduler: str | None = DEFAULT_LR_SCHEDULER,
        lr_scheduler_kwargs: dict[str, Any] | None = None,
        lr_scheduler_interval: str = DEFAULT_LR_SCHEDULER_INTERVAL,
        max_epochs_for_lr_scheduler: int = 100,
        enable_compile: bool = False,
    ) -> None:
        # TODO (matthias) Consider using separate models for each end point.
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

        if len(entity_table_names) < 2:
            raise ValueError(f"Found less than two entity tables in "
                             f"'{self.__class__.__name__}'")

        self.encoder = encoder
        self.model = model
        self.head = head
        self.train_table_name = train_table_name
        self.entity_table_names = entity_table_names

        # Coloring added to all entity nodes that belong to the same hyperedge:
        if with_identity_awareness:
            self.identity_awareness = torch.nn.Embedding(
                num_embeddings=len(entity_table_names),
                embedding_dim=model.channels,
            )
        else:
            self.identity_awareness = None

    # Model Execution #########################################################

    def forward(
        self,
        batch_size: int,
        entity_table_name: int,
        tf_dict: dict[NodeType, TensorFrame],
        node_id_dict: dict[NodeType, Tensor],
        edge_index_dict: dict[EdgeType, Tensor],
        time_dict: dict[NodeType, Tensor],
        batch_dict: dict[NodeType, Tensor],
        fkeys: Tensor,
        seed_time: Tensor | None = None,
    ) -> Tensor:
        # We do not need to encode training table features as they are handled
        # separately in `common_step`. As such, shallow-copy and remove:
        tf_dict = copy.copy(tf_dict)
        batch_dict = copy.copy(batch_dict)
        tf_dict.pop(self.train_table_name, None)
        batch_dict.pop(self.train_table_name, None)

        x_dict = self.encoder(tf_dict, batch_dict, time_dict, seed_time)

        # Add identity-awareness (see #4981 for more information):
        if self.identity_awareness is not None:
            for i, node_type in enumerate(self.entity_table_names):
                fkey = fkeys[:, i]
                mask = fkey[batch_dict[node_type]] == node_id_dict[node_type]

                x = x_dict[node_type]
                color = x[mask] + self.identity_awareness.weight[i]
                x_dict[node_type] = x.masked_scatter(mask.view(-1, 1), color)

        x_dict = self.model(x_dict, edge_index_dict)

        return x_dict[entity_table_name][:batch_size]

    def common_step(self, batches: tuple[HeteroData, ...]) -> Tensor:
        assert len(batches) == len(self.entity_table_names)

        # Obtain node embeddings for every foreign key in the training table:
        outs: list[Tensor] = []
        for entity_table_name, batch in zip(self.entity_table_names, batches):
            out = self(
                batch[entity_table_name].batch_size,
                entity_table_name,
                batch.tf_dict,
                batch.n_id_dict,
                batch.collect('edge_index', allow_empty=True),
                batch.collect('time', allow_empty=True),
                batch.batch_dict,
                fkeys=batch[self.train_table_name].fkey,
                seed_time=batch[self.train_table_name].get('time'),
            )
            outs.append(out)

        # Obtain training table features through the encoder:
        train_table = self.train_table_name
        x_train_table = self.encoder(
            tf_dict={train_table: batches[0].tf_dict[train_table]},
            batch_dict={train_table: batches[0][train_table].batch},
        )[train_table]
        outs.append(x_train_table)

        # Concatenate/sum up all embeddings for the final prediction:
        seed_time = batches[0][self.train_table_name].get('time')
        return self.head(sum(outs), seed_time)

    # Training Callbacks ######################################################

    def train_step(
        self,
        batches: tuple[HeteroData, ...],
        batch_idx: int,
    ) -> Tensor:
        out = self.common_step(batches)
        y = batches[0][self.train_table_name].y
        assert self.target_encoder is not None
        y = self.target_encoder.encode_tensor(y)

        loss = self.loss(out, y)
        loss_name = TrainingStage.TRAIN.add_stage_prefix('loss')

        self.log_learning_rate()
        self.log_epoch(loss_name, loss, batch_size=y.size(0))
        self.update_metrics(TrainingStage.TRAIN, out, y)

        return loss

    # Evaluation Callbacks ####################################################

    def evaluation_step(
        self,
        batches: tuple[HeteroData, ...],
        stage: TrainingStage,
        batch_idx: int,
    ) -> dict[str, Tensor | None]:
        out = self.common_step(batches)
        y = batches[0][self.train_table_name].y
        batch_size = batches[0][self.entity_table_names[0]].batch_size
        pkeys = [
            batches[i][entity_table_name].pkey[:batch_size]
            for i, entity_table_name in enumerate(self.entity_table_names)
        ]
        pkey = np.stack(pkeys, axis=1)
        time = batches[0][self.train_table_name].get('time')
        assert self.target_encoder is not None
        y = self.target_encoder.encode_tensor(y)

        loss = self.loss(out, y)
        loss_name = stage.add_stage_prefix('loss')

        self.log_epoch(loss_name, loss, batch_size=y.size(0))
        self.update_metrics(stage, out, y)

        return dict(pkey=pkey, time=time, logits=out, target=y)  # type: ignore

    # Prediction Callbacks ####################################################

    def predict_step(
        self,
        batches: tuple[HeteroData, ...],
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
            out = self.common_step(batches)
        finally:
            if handle is not None:
                handle.remove()
        batch_size = batches[0][self.entity_table_names[0]].batch_size
        pkeys = [
            batches[i][entity_table_name].pkey[:batch_size]
            for i, entity_table_name in enumerate(self.entity_table_names)
        ]
        pkey = np.stack(pkeys, axis=1)
        time = batches[0][self.train_table_name].get('time')

        result = dict(pkey=pkey, time=time, logits=out)
        if return_embeddings:
            if embedding is None:
                raise RuntimeError(
                    'Hyperedge prediction did not produce requested '
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
        return {'__'.join(self.entity_table_names): inputs[0]}
