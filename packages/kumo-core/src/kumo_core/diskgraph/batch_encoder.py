# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serializable actor-side configuration for the existing Python tensor encoder."""

from dataclasses import dataclass

from diskgraph.runtime_batch import RuntimeBatch
from torch_geometric.data import HeteroData

from kumo_core.diskgraph.runtime_adapter import (
    FeatureSchema,
    MapperBank,
    runtime_batch_to_hetero,
)
from kumo_core.encoders.target.base import TargetEncoder


@dataclass(frozen=True)
class RuntimeBatchEncoder:
    """Encode training graphs without serializing a DataModule or Ray connection.

    Args:
        train_table_name: Task namespace for seeds and targets.
        mapper_bank: Fitted Python column mappers, read-only during sampling.
        feature_schema: Task graph feature schema.
        task_type: Target layout selected by the task.
        seed_node_type: Graph namespace of seed nodes.
        pk_col_dict: Primary-key columns used by the adapter.
        time_col_dict: Timestamp columns used by the adapter.
        target_encoder: Fitted target transformation, when applicable.
        index_np_time_unit: Unit of native index timestamps.
        bidirectional_output: Include reverse edges in the encoded graph.
    """

    train_table_name: str
    mapper_bank: MapperBank
    feature_schema: FeatureSchema
    task_type: str | None
    seed_node_type: str | None
    pk_col_dict: dict[str, str]
    time_col_dict: dict[str, str]
    target_encoder: TargetEncoder | None
    index_np_time_unit: str
    bidirectional_output: bool

    def __call__(self, batch: RuntimeBatch) -> HeteroData:
        """Apply the same tensor conversion as local training.

        Args:
            batch: Typed native sample, including raw feature columns.

        Returns:
            Model-ready CPU graph, without evaluation-only primary-key objects.
        """
        return runtime_batch_to_hetero(
            batch,
            train_table_name=self.train_table_name,
            mapper_bank=self.mapper_bank,
            feature_schema=self.feature_schema,
            task_type=self.task_type,
            seed_node_type=self.seed_node_type,
            pk_col_dict=self.pk_col_dict,
            time_col_dict=self.time_col_dict,
            target_encoder=self.target_encoder,
            index_np_time_unit=self.index_np_time_unit,
            bidirectional_output=self.bidirectional_output,
            attach_pkeys=False,
        )[0]
