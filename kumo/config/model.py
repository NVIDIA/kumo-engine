# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from kumoapi.encoder import EncoderType

from io_utils import kumo_dataclass


@kumo_dataclass
class ModelEncoderConfig:
    table_encoder_dict: Optional[Dict[str, Dict[str, EncoderType]]] = None
    out_channels: Optional[Any] = '${model.channels}'


@kumo_dataclass
class ModelConfig:
    channels: int
    metadata: Optional[List[Any]] = None  # {task.data.metadata()}


@kumo_dataclass
class ModelHeadConfig:
    in_channels: Optional[Any] = '${model.channels}'
    out_channels: Optional[int] = None


@kumo_dataclass
class LinkPredHeadConfig:
    in_channels: Optional[Any] = '${model.channels}'
    use_rhs_projector: bool = False
    use_mlp_head: bool = False
    embedding_dim: int = 32
    normalize_vectors: bool = False


@kumo_dataclass
class ForecastingHeadConfig:
    pass
