# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent pandas reference for sequence encoding parity tests."""

import numpy as np
import pandas as pd
import torch
from kumo_core.encoders.tensor_mapper import SequenceTensorMapper
from torch_frame.data import MultiEmbeddingTensor


def sequence_pandas_reference(
    mapper: SequenceTensorMapper,
    ser: pd.Series,
    *,
    device: torch.device | None = None,
) -> MultiEmbeddingTensor:
    """Encode synthetic sequences with the previous pandas implementation.

    Args:
        mapper: Sequence settings and backward-format state.
        ser: Synthetic rows to encode.
        device: Destination tensor device.

    Returns:
        The reference padded or truncated sequence tensor.
    """
    if len(ser) == 0:
        return MultiEmbeddingTensor(
            num_rows=0,
            num_cols=1,
            values=torch.empty(0, mapper.max_length, device=device),
            offset=torch.tensor([0, mapper.max_length], device=device),
        )

    # Support for both string-based and list-like input data:
    if mapper._is_string_dtype is None and len(ser) > 0:
        index = ser.first_valid_index()
        mapper._is_string_dtype = index is not None and not pd.api.types.is_list_like(
            ser[index]
        )

    ser = mapper.to_list_like(ser)
    dtype = str(torch.get_default_dtype()).split(".")[-1]

    def pad_with_nan(row: list[float] | None) -> np.ndarray | list[float]:
        if row is None or len(row) == 0:
            return np.full(mapper.max_length, np.nan, dtype=dtype)

        if len(row) < mapper.max_length:
            return np.concatenate(
                [
                    np.array(row, dtype=dtype),
                    np.full(mapper.max_length - len(row), np.nan, dtype=dtype),
                ]
            )

        return row[: mapper.max_length]

    tensor = torch.from_numpy(np.stack(ser.apply(pad_with_nan)))  # type: ignore

    return MultiEmbeddingTensor(
        num_rows=tensor.size(0),
        num_cols=1,
        values=tensor.to(device, torch.get_default_dtype()),
        offset=torch.tensor([0, mapper.max_length], device=device),
    )
