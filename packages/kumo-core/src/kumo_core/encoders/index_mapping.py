# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.index import index2ptr


def _is_string_like(data: pd.Series | pd.Index) -> bool:
    r"""Returns whether ``data`` holds string values.

    Tolerant of missing values: an object/extension column of strings that
    also contains ``NaN`` / ``None`` is still string-like. ``is_string_dtype``
    alone inspects the actual values and reports ``False`` for such columns.
    """
    if pd.api.types.is_string_dtype(data):
        return True
    if (pd.api.types.is_object_dtype(data)
            or pd.api.types.is_extension_array_dtype(data)):
        return pd.api.types.infer_dtype(data,
                                        skipna=True) in ('string', 'empty')
    return False


class Mapper:
    r"""Creates a mapper between values from ``data`` to integer IDs.

    Args:
        data: A sequence of values.
    """
    def __init__(self, data: Any) -> None:
        self._map = pd.CategoricalDtype(categories=data, ordered=True)
        self._index = pd.Index(data)
        categories = self._map.categories
        if (categories is data and type(self._index) is type(categories)
                and self._index.dtype == categories.dtype):
            # Reuse the lookup cache populated by category validation.
            self._index = categories
        self._length = len(data)

    def map(self, data: Any, strict: bool = False) -> Tensor:
        r"""Maps categorical values in ``data`` to integer IDs based on the
        pre-defined mapping.

        Args:
            data: The data to map. ``pd.Series`` is the canonical input; a
                ``np.ndarray`` is also accepted and routed through a faster
                path that skips Series/CategoricalDtype construction.
            strict: If set to ``True``, will raise an error if data types are
                incompatible. Only honored on the ``pd.Series`` path; the
                numpy fast path always behaves as if ``strict=False``.
        """
        if isinstance(data, np.ndarray):
            # Fast path: hash-lookup the values directly. Missing values
            # become ``-1`` (same convention as ``CategoricalDtype.astype``
            # → ``cat.codes``).
            positions = self._index.get_indexer(data)
            return torch.from_numpy(positions).to(torch.long)

        if not isinstance(data, pd.Series):
            data = pd.Series(data=data)

        if strict:
            error_msg = "Encountered incompatible data types."
            suggestion_msg = ("Please ensure both source and destination data "
                              "types belong to a compatible type family.")

            if (pd.api.types.is_numeric_dtype(self._map.categories)
                    and not pd.api.types.is_numeric_dtype(data)):

                if data.name is not None:
                    raise RuntimeError(
                        f"{error_msg} '{data.name}' cannot be mapped to "
                        f"numeric data type. {suggestion_msg}")
                else:
                    raise RuntimeError(f"{error_msg} {suggestion_msg}")

            if (_is_string_like(self._map.categories)
                    and not _is_string_like(data)):

                if data.name is not None:
                    raise RuntimeError(
                        f"{error_msg} '{data.name}' cannot be mapped to "
                        f"string data type. {suggestion_msg}")
                else:
                    raise RuntimeError(f"{error_msg} {suggestion_msg}")

        ser = data.astype(self._map)
        return torch.from_numpy(ser.cat.codes.to_numpy().copy()).to(torch.long)

    def map_multi(self, data: Any, sep: str | None) -> tuple[Tensor, Tensor]:
        r"""Maps multi-categorical values in ``data`` to integer IDs in CSR
        ``(rowptr, index)`` format based on the pre-defined mapping.
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data=data)

        index = data.first_valid_index()

        if len(data) == 0:
            index = torch.empty(0, dtype=torch.long)
            rowptr = torch.tensor([0], dtype=torch.long)
        elif index is not None and pd.api.types.is_list_like(data[index]):
            data = data.reset_index(drop=True)
            # Filter empty lists:
            mask = data.apply(lambda x: not isinstance(x, list) or len(x) > 0)
            masked_data = data[mask]
            exploded_data = masked_data.explode()
            index = self.map(exploded_data)
            row = torch.from_numpy(exploded_data.index.to_numpy())
            mask = torch.from_numpy((exploded_data != '').to_numpy())
            index, row = index[mask], row[mask]
            rowptr = index2ptr(row, size=len(data))
        elif sep is None:
            index = self.map(data)
            row = torch.arange(index.numel())
            mask = torch.from_numpy((data != '').to_numpy())
            index, row = index[mask], row[mask]
            rowptr = index2ptr(row, size=len(data))
        else:
            data = data.fillna('__None__').astype(str)
            data = data.reset_index(drop=True).str.split(sep)
            exploded_data = data.explode()
            index = self.map(exploded_data)
            row = torch.from_numpy(exploded_data.index.to_numpy())
            mask = torch.from_numpy((exploded_data != '').to_numpy())
            index, row = index[mask], row[mask]
            rowptr = index2ptr(row, size=len(data))

        return rowptr, index

    def tokenize(self, data: Any) -> tuple[Tensor, Tensor]:
        r"""Tokenize sentences in ``data`` to integer IDS in CSR
        ``(rowptr, index)`` format based on the pre-defined mapping.
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data=data)

        if len(data) == 0:
            index = torch.empty(0, dtype=torch.long)
            rowptr = torch.tensor([0], dtype=torch.long)
        else:
            from kumo_core.encoders.text import normalize_text

            data = normalize_text(data)
            exploded_data = data.reset_index(drop=True).explode()
            index = self.map(exploded_data)
            row = torch.from_numpy(exploded_data.index.to_numpy())
            mask = index >= 0
            row, index = row[mask], index[mask]
            rowptr = index2ptr(row, size=len(data))

        return rowptr, index

    def __len__(self) -> int:
        return self._length
