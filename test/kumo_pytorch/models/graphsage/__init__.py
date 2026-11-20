# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RelBench GraphSAGE with an Kumo graph-batch adapter."""

from .adapter import GraphBatch, KumoGraphAdapter
from .config import GraphSAGEConfig, NodeFeatureSpec
from .model import GraphSAGEEncoding, GraphSAGEModel, GraphSAGEOutput

__all__ = [
    "GraphBatch", "KumoGraphAdapter", "GraphSAGEConfig", "NodeFeatureSpec",
    "GraphSAGEEncoding", "GraphSAGEModel", "GraphSAGEOutput",
]
