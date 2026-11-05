# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent pandas references for Arrow-only mapper parity tests."""

import numpy as np
import pandas as pd
import torch
from kumo_core.encoders.index_mapping import Mapper
from kumo_core.encoders.tensor_mapper import (
    MultiCategoricalTensorMapper,
    SequenceTensorMapper,
)
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor


def _multi_categorical_reference(
    self: MultiCategoricalTensorMapper,
    ser: pd.Series,
    *,
    device: torch.device | None = None,
) -> MultiNestedTensor:
    """Handle split/explode cases outside the native Arrow path."""
    # Support for both string-based and list-like input data:
    if self._is_string_dtype is None and len(ser) > 0:
        index = ser.first_valid_index()
        self._is_string_dtype = index is not None and not pd.api.types.is_list_like(
            ser[index]
        )

    rowptr, index = Mapper(self.mapping).map_multi(ser, self.sep)
    return MultiNestedTensor(
        num_rows=rowptr.numel() - 1,
        num_cols=1,
        values=index.to(device),
        offset=rowptr.to(device),
    )


def _sequence_reference(
    self: SequenceTensorMapper,
    ser: pd.Series,
    *,
    device: torch.device | None = None,
) -> MultiEmbeddingTensor:
    """Parse and pad inputs outside the native Arrow list path."""
    if len(ser) == 0:
        return MultiEmbeddingTensor(
            num_rows=0,
            num_cols=1,
            values=torch.empty(0, self.max_length, device=device),
            offset=torch.tensor([0, self.max_length], device=device),
        )

    # Support for both string-based and list-like input data:
    if self._is_string_dtype is None and len(ser) > 0:
        index = ser.first_valid_index()
        self._is_string_dtype = index is not None and not pd.api.types.is_list_like(
            ser[index]
        )

    ser = self.to_list_like(ser)
    dtype = str(torch.get_default_dtype()).split(".")[-1]

    def pad_with_nan(row: list[float] | None) -> np.ndarray | list[float]:
        if row is None or len(row) == 0:
            return np.full(self.max_length, np.nan, dtype=dtype)

        if len(row) < self.max_length:
            return np.concatenate(
                [
                    np.array(row, dtype=dtype),
                    np.full(self.max_length - len(row), np.nan, dtype=dtype),
                ]
            )

        return row[: self.max_length]

    tensor = torch.from_numpy(np.stack(ser.apply(pad_with_nan)))  # type: ignore

    return MultiEmbeddingTensor(
        num_rows=tensor.size(0),
        num_cols=1,
        values=tensor.to(device, torch.get_default_dtype()),
        offset=torch.tensor([0, self.max_length], device=device),
    )
