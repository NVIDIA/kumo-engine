# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prediction identifier mapping owned by the portable training package."""

from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from torch import Tensor

NODE_INDEX_COLUMN = "arange"
TTNG_RHS_PRIMARY_KEY_INDEX_NAME = "__ttng_rhs_primary_key__"


def from_index_to_raw(
    node_ids: Tensor,
    index_map: pd.DataFrame,
) -> list[Any]:
    """Map contiguous model node identifiers back to raw primary keys."""

    primary_key = index_map.index.name or "index"
    indexed = index_map.reset_index().set_index(NODE_INDEX_COLUMN)
    values = pa.Array.from_pandas(indexed[primary_key].values)
    indices = pa.array(node_ids.cpu().numpy())
    raw = values.take(indices).to_pylist()
    return [
        value.item() if isinstance(value, np.generic) else value
        for value in raw
    ]
