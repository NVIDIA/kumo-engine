# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monolith implementation of the portable explainability capability."""

from typing import Any

from kumo.explain.select import (
    PRED_NUM_SELECTION_NODES,
    get_eval_dataloader,
    get_explain_node_inputs,
    select_input_nodes,
)
from kumo.explain.utils import (
    MAPPING_FILE_NAME,
    MAX_NUM_BATCHES,
    MAX_NUM_FACTS,
)


class MonolithExplainRuntime:
    prediction_selection_nodes = PRED_NUM_SELECTION_NODES
    max_batches = MAX_NUM_BATCHES
    max_facts = MAX_NUM_FACTS
    mapping_file_name = MAPPING_FILE_NAME

    def get_explain_node_inputs(self, *args: Any, **kwargs: Any) -> Any:
        return get_explain_node_inputs(*args, **kwargs)

    def select_input_nodes(self, *args: Any, **kwargs: Any) -> Any:
        return select_input_nodes(*args, **kwargs)

    def get_eval_dataloader(self, *args: Any, **kwargs: Any) -> Any:
        return get_eval_dataloader(*args, **kwargs)
