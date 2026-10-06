# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import cast

import pandas as pd
import pyarrow as pa
import torch
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor

from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.types.features import TensorData


def combine_feats(
        feats_dict: dict[Stype, list[TensorData]]) -> dict[Stype, TensorData]:
    """Combine per-column tensors into stype-wise tensors for TensorFrame."""
    feat_dict: dict[Stype, TensorData] = {}

    for stype, feats in feats_dict.items():
        if len(feats) == 0:
            continue

        if isinstance(feats[0], MultiNestedTensor):
            feat_dict[stype] = MultiNestedTensor.cat(feats, dim=1)
        elif isinstance(feats[0], MultiEmbeddingTensor):
            feat_dict[stype] = MultiEmbeddingTensor.cat(feats, dim=1)
        elif Stype(stype) in {Stype.numerical, Stype.categorical}:
            assert isinstance(feats[0], Tensor)
            # [num_rows, num_cols]
            if len(feats) == 1:
                feat_dict[stype] = feats[0]
            else:
                feat_dict[stype] = torch.cat(cast(list[Tensor], feats), 1)
        else:
            assert isinstance(feats[0], Tensor)
            # [num_rows, num_cols, num_features]
            if len(feats) == 1:
                feat_dict[stype] = feats[0].unsqueeze(1)
            else:
                feat_dict[stype] = torch.stack(cast(list[Tensor], feats), 1)

    return feat_dict


def get_tf(
    df: pd.DataFrame | pa.RecordBatch | pa.Table,
    mapper_dict: dict[str, TensorMapper],
) -> TensorFrame:

    feats_dict: dict[Stype, list[TensorData]] = defaultdict(list)
    col_names_dict: dict[Stype, list[str]] = defaultdict(list)

    is_arrow = isinstance(df, (pa.RecordBatch, pa.Table))

    for col_name, mapper in mapper_dict.items():
        if mapper.stype == Stype.unsupported:
            continue

        col = df.column(col_name) if is_arrow else df[col_name]
        feats_dict[mapper.stype].append(mapper(col))
        col_names_dict[mapper.stype].append(col_name)

    feat_dict = combine_feats(feats_dict)
    num_rows = df.num_rows if is_arrow else len(df)
    return TensorFrame(feat_dict, col_names_dict, num_rows=num_rows)
