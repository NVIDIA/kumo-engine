# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explainability capability boundary for the numerical trainer."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Protocol

from .base import Explainer, continue_on_error
from .column_analysis import ColumnAnalysis
from .gradient_explainer import GradientExplainer
from .subgraph_collector import SubgraphCollector

if TYPE_CHECKING:
    from kumo_core.diskgraph.lightning_data import DiskgraphLightningDataModule
    from torch import Tensor


class ExplainRuntime(Protocol):
    prediction_selection_nodes: int
    max_batches: int
    max_facts: int
    mapping_file_name: str

    def get_prediction_inputs(
        self, datamodule: DiskgraphLightningDataModule, limit: int,
    ) -> tuple[Tensor, Tensor]:
        ...

    def prediction_rows(
        self, datamodule: DiskgraphLightningDataModule, rows: Tensor,
    ) -> AbstractContextManager[None]:
        ...

    def get_explain_node_inputs(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def select_input_nodes(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def get_eval_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        ...


__all__ = [
    "ColumnAnalysis",
    "Explainer",
    "ExplainRuntime",
    "GradientExplainer",
    "SubgraphCollector",
    "continue_on_error",
]
