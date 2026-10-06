# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from kumoapi.encoder import (
    Datetime,
    Encoder,
    Hash,
    Image,
    Index,
    MaxLogNumerical,
    MinLogNumerical,
    MultiCategorical,
    Null,
    Numerical,
    NumericalList,
    TextEmbedding,
)
from kumoapi.typing import ColStatType, Stype
from torch import Tensor
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor
from torch_geometric.index import index2ptr, ptr2index

from kumo_core.types.features import SECONDS_IN_DAY
from kumo_core.embedding_models import TokenEmbeddingModel, resolve_encoder
from kumo_core.encoders.index_mapping import Mapper
from kumo_core.features.time import TimestampHelper, to_datetime

T = TypeVar('T', Tensor, MultiNestedTensor, MultiEmbeddingTensor)


class ClosestHoliday(NamedTuple):
    days_to_holiday: Tensor
    to_holiday_name: Tensor
    days_from_holiday: Tensor
    from_holiday_name: Tensor


class TensorMapper(Generic[T], ABC):
    r"""A base class to handle the conversion from raw input data into a
    compact tensor representation, *i.e.*, the identity for numerical values,
    indices for categorical values, etc.
    """
    stype: Stype

    def __call__(
        self,
        ser: pd.Series | np.ndarray | pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> T:
        if isinstance(ser, (pa.Array, pa.ChunkedArray)):
            return self._from_arrow(ser, device=device)
        return self.forward(ser, device=device)

    def _from_arrow(
        self,
        arr: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> T:
        r"""Maps a native :class:`pyarrow.Array` into a compact tensor
        representation.

        The base implementation converts to pandas and defers to
        :meth:`forward`, so *every* mapper accepts Arrow input out of the box.
        Hot stypes (numerical / categorical / multicategorical / sequence)
        override this to build tensors directly from Arrow buffers, no pandas.
        """
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        return self.forward(arr.to_pandas(), device=device)

    @abstractmethod
    def forward(
        self,
        ser: pd.Series,
        *,
        device: torch.device | None = None,
    ) -> T:
        r"""Maps raw input data into a compact tensor representation."""

    @abstractmethod
    def backward(self, tensor: T) -> pd.Series:
        r"""Maps a compact tensor representation back into its raw input data.

        The reverse operation of :meth:`forward`.
        """

    @abstractmethod
    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> T:
        r"""Maps data serialized with :class:`PreMatMapper` into a compact
        tensor representation.
        """

    @classmethod
    def from_stype(cls, stype: Stype, ser: pd.Series) -> TensorMapper:
        r"""Creates a tensor mapper for a given :class:`pandas.Series` and
        semantic type.
        """
        import kumo_core.features.stats

        if stype == Stype.unsupported:
            return UnsupportedTensorMapper()

        if stype == Stype.image:
            return ImageTensorMapper()

        if stype == Stype.numerical:
            return NumericalTensorMapper()

        if stype in {Stype.categorical, Stype.ID}:
            category_counts = kumo_core.features.stats.category_count(ser)
            return CategoricalTensorMapper(category_counts=category_counts)

        if stype == Stype.multicategorical:
            sep = kumo_core.features.stats.sep_multi(ser)
            ser = ser.astype(str).split(sep) if sep is not None else ser
            ser = ser.explode()
            category_counts = kumo_core.features.stats.category_count(ser)
            return MultiCategoricalTensorMapper(
                category_counts=category_counts,
                sep=sep,
            )

        if stype == Stype.sequence:
            return SequenceTensorMapper(
                min_length=kumo_core.features.stats.sequence_min_length(ser),
                max_length=kumo_core.features.stats.sequence_max_length(ser),
            )

        if stype == Stype.timestamp:
            return TimestampTensorMapper()

        if stype == Stype.text:
            return TextTensorMapper()

        raise ValueError(f"Unsupported semantic type '{stype}'")

    @classmethod
    def from_encoder(cls, encoder: Encoder) -> TensorMapper:
        r"""Creates a tensor mapper from an encoder."""
        if isinstance(encoder, Null):
            return UnsupportedTensorMapper()

        if isinstance(encoder, Numerical | MaxLogNumerical | MinLogNumerical):
            return NumericalTensorMapper()

        if isinstance(encoder, Index | Hash):
            category_counts = encoder._stats[ColStatType.CATEGORY_COUNTS]
            return CategoricalTensorMapper(category_counts=category_counts)

        if isinstance(encoder, MultiCategorical):
            category_counts = encoder._stats[ColStatType.MULTI_CATEGORY_COUNTS]
            return MultiCategoricalTensorMapper(
                category_counts=category_counts,
                sep=encoder._stats[ColStatType.MULTI_CATEGORIES_SEPARATOR],
            )

        if isinstance(encoder, NumericalList):
            return SequenceTensorMapper(
                min_length=encoder._stats[ColStatType.SEQUENCE_MIN_LENGTH],
                max_length=encoder._stats[ColStatType.SEQUENCE_MAX_LENGTH],
            )

        if isinstance(encoder, Datetime):
            return TimestampTensorMapper()

        if isinstance(encoder, TextEmbedding):
            return TextTensorMapper(resolve_encoder(encoder))

        if isinstance(encoder, Image):
            return ImageTensorMapper()

        raise ValueError(f"Unsupported encoder '{encoder}'")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class UnsupportedTensorMapper(TensorMapper[Tensor]):
    r"""Handles the conversion from unsupported values to tensors."""
    stype: Stype = Stype.unsupported

    def forward(
        self,
        ser: pd.Series,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        return torch.empty((len(ser), 0), device=device)

    def backward(self, tensor: Tensor) -> pd.Series:
        assert isinstance(tensor, Tensor)
        raise RuntimeError(f"'{self.__class__.__name__}' cannot map empty "
                           f"tensor representation back to raw inputs")

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        return torch.empty((len(array), 0), device=device)


class ImageTensorMapper(TensorMapper[Tensor]):
    r"""Handles the no-op conversion for image columns."""
    stype: Stype = Stype.image

    def forward(
        self,
        ser: pd.Series,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        return torch.empty((len(ser), 0), device=device)

    def backward(self, tensor: Tensor) -> pd.Series:
        assert isinstance(tensor, Tensor)
        raise RuntimeError(f"'{self.__class__.__name__}' cannot map empty "
                           f"tensor representation back to raw inputs")

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        return torch.empty((len(array), 0), device=device)


class NumericalTensorMapper(TensorMapper[Tensor]):
    r"""Handles the conversion from numerical values to tensors."""
    stype: Stype = Stype.numerical

    def forward(
        self,
        ser: pd.Series | np.ndarray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        dtype = str(torch.get_default_dtype()).split('.')[-1]
        if isinstance(ser, np.ndarray):
            # NumPy fast path: skip pandas. Timedelta inputs use the pandas
            # path below so they can be normalized to seconds.
            value = torch.from_numpy(ser.astype(dtype))
            return value.to(device).view(-1, 1)

        # Convert PyTorch default dtype to numpy format:
        if pd.api.types.is_timedelta64_dtype(ser):
            ser = ser.dt.total_seconds()

        value = torch.from_numpy(ser.values.astype(dtype))
        return value.to(device).view(-1, 1)

    def backward(self, tensor: Tensor) -> pd.Series:
        if tensor.dim() == 2 and tensor.size(1) > 1:
            arr = tensor.detach().cpu().numpy()
            return pd.Series([row for row in arr])
        return pd.Series(tensor.view(-1).detach().cpu().numpy())

    def _from_arrow(
        self,
        arr: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        # Defer durations to ``forward``, which normalizes to seconds; a raw
        # ``astype(float)`` would keep nanoseconds and diverge from pandas.
        if pa.types.is_duration(arr.type):
            return self.forward(arr.to_pandas(), device=device)
        dtype = str(torch.get_default_dtype()).split('.')[-1]
        value = torch.from_numpy(
            arr.to_numpy(zero_copy_only=False).astype(dtype))
        return value.to(device).view(-1, 1)

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        # TODO (vid) recent versions of pyarrow support DLPack direct
        # conversions from PyArrow to Torch without numpy/pandas intermediates.

        # Convert PyTorch default dtype to numpy format.
        # We cast dtype in numpy to turn an immutable array into a mutable one.
        dtype = str(torch.get_default_dtype()).split('.')[-1]
        return torch.from_numpy(
            array.to_pandas().to_numpy().astype(dtype)).to(device).view(-1, 1)


class CategoricalTensorMapper(TensorMapper[Tensor]):
    r"""Handles the conversion from categorical values to consecutive indices.

    Args:
        category_counts: Category counts to create a mapping from categorical
            values to integers.
    """
    stype: Stype = Stype.categorical

    def __init__(self, category_counts: list[list[Any]]) -> None:
        category_counts = _sort_category_counts(category_counts)
        self.mapping = _create_mapping_from_category_counts(category_counts)
        self.mapper = Mapper(self.mapping)

    def forward(
        self,
        ser: pd.Series | np.ndarray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        if isinstance(ser, np.ndarray):
            # NumPy goes straight through ``Mapper.map``'s pd.Index hash
            # lookup without Series construction.
            # ``get_indexer`` already returns ``-1`` for unknown / missing
            # values, matching the all-null branch's behavior, so no
            # separate guard is needed.
            index = self.mapper.map(ser)
            return index.to(device).view(-1, 1)

        if ser.isnull().all():  # Avoid errors due to poor typing.
            return torch.full((len(ser), 1), fill_value=-1, dtype=torch.long,
                              device=device)

        index = self.mapper.map(ser)
        return index.to(device).view(-1, 1)

    def backward(self, tensor: Tensor) -> pd.Series:
        if len(self.mapping) == 0:
            return pd.Series([None] * tensor.size(0), dtype=self.mapping.dtype)

        index = tensor.view(-1).cpu().numpy()
        ser = pd.Series(self.mapping[index].to_numpy(),
                        dtype=self.mapping.dtype)
        ser[index < 0] = None
        return ser

    def _from_arrow(
        self,
        arr: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()

        if pa.types.is_dictionary(arr.type):
            # Dictionary columns store integer codes into a small dictionary of
            # unique values; hash only the K unique values, then gather.
            dict_codes = self.mapping.get_indexer(
                arr.dictionary.to_pandas()).astype(np.int64)
            indices = arr.indices
            null_mask = None
            if indices.null_count > 0:
                null_mask = indices.is_null().to_numpy(zero_copy_only=False)
                indices = indices.fill_null(0)
            codes = dict_codes[indices.to_numpy(zero_copy_only=False)]
            if null_mask is not None:
                codes[null_mask] = -1
            index = torch.from_numpy(codes).to(torch.long)
            return index.to(device).view(-1, 1)

        if arr.null_count == len(arr):  # Avoid errors due to poor typing.
            return torch.full((len(arr), 1), fill_value=-1, dtype=torch.long,
                              device=device)

        index = self.mapper.map(arr.to_numpy(zero_copy_only=False))
        return index.to(device).view(-1, 1)

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        # See comments in Numerical post_forward on casting and mutability
        return torch.from_numpy(array.to_pandas().to_numpy().astype(
            np.int64)).to(device).view(-1, 1)


class MultiCategoricalTensorMapper(TensorMapper[MultiNestedTensor]):
    r"""Handles the conversion from multi-categorical values to consecutive
    indices.

    Args:
        category_counts: Category counts to create a mapping from categorical
            values to integers.
        sep: The separator.
    """
    stype: Stype = Stype.multicategorical

    def __init__(
        self,
        category_counts: list[list[Any]],
        sep: str | None,
    ) -> None:
        category_counts = _sort_category_counts(category_counts)
        self.mapping = _create_mapping_from_category_counts(category_counts)
        self.mapper = Mapper(self.mapping)
        self.sep = sep
        self._is_string_dtype: bool | None = None

    def forward(
        self,
        ser: pd.Series | np.ndarray,
        *,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        # No numpy fast path for multi-categorical yet — the explode/split
        # pipeline is pandas-shaped. Wrap so the rest of the function sees
        # the canonical Series without requiring a full DataFrame round-trip.
        if isinstance(ser, np.ndarray):
            ser = pd.Series(ser, copy=False)

        # Support for both string-based and list-like input data:
        if self._is_string_dtype is None and len(ser) > 0:
            index = ser.first_valid_index()
            self._is_string_dtype = (index is not None and
                                     not pd.api.types.is_list_like(ser[index]))

        rowptr, index = self.mapper.map_multi(ser, self.sep)
        return MultiNestedTensor(
            num_rows=rowptr.numel() - 1,
            num_cols=1,
            values=index.to(device),
            offset=rowptr.to(device),
        )

    def backward(self, tensor: MultiNestedTensor) -> pd.Series:
        assert tensor.num_cols == 1

        if tensor.num_rows == 0:
            dtype = str if self._is_string_dtype else object
            return pd.Series([], dtype=dtype)

        # Only maintain valid indices:
        row = ptr2index(tensor.offset, output_size=tensor.values.numel())
        mask = tensor.values >= 0
        row, index = row[mask], tensor.values[mask]
        rowptr = index2ptr(row, size=tensor.num_rows)

        values = self.mapping[index.cpu().numpy()].to_numpy()
        ser = pd.Series(np.split(values, rowptr[1:-1].cpu().numpy()))

        if self._is_string_dtype is False:
            return ser
        elif self.sep is not None:
            return ser.map(self.sep.join)
        else:
            return ser.map(lambda x: x[0] if len(x) > 0 else '')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sep={self.sep})'

    def _from_arrow(
        self,
        arr: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()

        # An Arrow ``list<int>`` column is already the ``(rowptr, index)`` CSR
        # layout of ``MultiNestedTensor``: map the flat values and reuse the
        # offsets, skipping the explode / index2ptr machinery.
        is_list = isinstance(arr, (pa.ListArray, pa.LargeListArray))
        if is_list and pa.types.is_integer(arr.type.value_type):
            offsets_np = arr.offsets.to_numpy(zero_copy_only=False)
            values = arr.values
            # Defer to the pandas fallback on: list-level nulls; sliced offsets
            # (``arr.values`` ignores the slice, so require the full span); and
            # empty lists, where ``map_multi`` emits a spurious -1 we replicate
            # for backwards compatibility.
            no_empty_rows = bool((offsets_np[1:] > offsets_np[:-1]).all())
            if (arr.null_count == 0 and offsets_np[0] == 0
                    and offsets_np[-1] == len(values) and no_empty_rows):
                self._is_string_dtype = False
                if values.null_count == 0:
                    index = self.mapper.map(
                        values.to_numpy(zero_copy_only=False))
                else:
                    index = self.mapper.map(values.to_pandas())
                offsets = torch.from_numpy(offsets_np.astype(np.int64))
                return MultiNestedTensor(
                    num_rows=len(arr),
                    num_cols=1,
                    values=index.to(device),
                    offset=offsets.to(device),
                )

        # Fallback (string lists, nested nulls, sliced offsets, ...).
        return self.forward(arr.to_pandas(), device=device)

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        if isinstance(array, pa.ListArray):
            offsets = torch.from_numpy(array.offsets.to_numpy().astype(
                np.int64)).to(device)
            values = torch.from_numpy(array.values.to_numpy().astype(
                np.int64)).to(device)
        else:
            assert len(array) == 0
            offsets = torch.tensor([0], device=device, dtype=torch.int64)
            values = torch.tensor([], device=device, dtype=torch.int64)
        return MultiNestedTensor(num_cols=1, num_rows=len(array),
                                 offset=offsets, values=values)


def _row_path(
    seq_len_np: np.ndarray,
    row_len: int,
    max_length: int,
    spans: bool,
) -> str:
    # 'dense': every row is max_length long; 'uniform': rows are all
    # row_len or empty; 'ragged': anything else.
    if not spans or len(seq_len_np) == 0:
        return 'ragged'
    full = seq_len_np == row_len
    if not (full | (seq_len_np == 0)).all():
        return 'ragged'
    return 'dense' if row_len == max_length and full.all() else 'uniform'


class SequenceTensorMapper(TensorMapper[MultiEmbeddingTensor]):
    r"""Handles the conversion from sequence/embedding-like values.

    Args:
        min_length: The minimum length of a sequence.
        max_length: The maximum length of a sequence.
    """
    stype: Stype = Stype.sequence

    def __init__(self, min_length: int, max_length: int) -> None:
        super().__init__()
        self.min_length = min_length
        self.max_length = max_length
        self._is_string_dtype: bool | None = None

    @staticmethod
    def to_list_like(ser: pd.Series) -> pd.Series:
        index = ser.first_valid_index()
        if index is not None and not pd.api.types.is_list_like(ser[index]):
            # Convert PyTorch default dtype to numpy format:
            dtype = str(torch.get_default_dtype()).split('.')[-1]
            return ser.map(lambda row: np.fromstring(
                row.strip('[]'),
                sep=',',
                dtype=dtype,
            ) if row is not None else None)
        return ser

    def forward(
        self,
        ser: pd.Series | np.ndarray,
        *,
        device: torch.device | None = None,
    ) -> MultiEmbeddingTensor:
        if isinstance(ser, np.ndarray):
            ser = pd.Series(ser, copy=False)

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
            self._is_string_dtype = (index is not None and
                                     not pd.api.types.is_list_like(ser[index]))

        ser = self.to_list_like(ser)
        dtype = str(torch.get_default_dtype()).split('.')[-1]

        def pad_with_nan(row: list[float] | None) -> np.ndarray | list[float]:
            if row is None or len(row) == 0:
                return np.full(self.max_length, np.nan, dtype=dtype)

            if len(row) < self.max_length:
                return np.concatenate([
                    np.array(row, dtype=dtype),
                    np.full(self.max_length - len(row), np.nan, dtype=dtype)
                ])

            return row[:self.max_length]

        tensor = torch.from_numpy(np.stack(
            ser.apply(pad_with_nan)))  # type: ignore

        return MultiEmbeddingTensor(
            num_rows=tensor.size(0),
            num_cols=1,
            values=tensor.to(device, torch.get_default_dtype()),
            offset=torch.tensor([0, self.max_length], device=device),
        )

    def backward(self, tensor: MultiEmbeddingTensor) -> pd.Series:
        assert tensor.num_cols == 1
        ser = pd.Series(tensor.values.tolist(), dtype=object)
        if self._is_string_dtype:
            ser = ser.map(lambda row: str(row))
        return ser

    def _from_arrow(
        self,
        arr: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> MultiEmbeddingTensor:
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()

        is_list = isinstance(arr, (pa.ListArray, pa.LargeListArray))
        if is_list and len(arr) > 0:
            offsets_np = arr.offsets.to_numpy(zero_copy_only=False)
            values_arr = arr.values
            # Defer sliced offsets to the pandas fallback (``arr.values``
            # ignores the slice, so require the full span); empty/null rows
            # already map to all-NaN rows in both paths.
            if offsets_np[0] == 0 and offsets_np[-1] == len(values_arr):
                self._is_string_dtype = False
                return self._pad_or_truncate(
                    values_arr.to_numpy(zero_copy_only=False), offsets_np,
                    device)

        if (isinstance(arr, pa.FixedSizeListArray) and len(arr) > 0
                and arr.null_count == 0):
            width = arr.type.list_size
            self._is_string_dtype = False
            return self._pad_or_truncate(
                arr.flatten().to_numpy(zero_copy_only=False),
                np.arange(len(arr) + 1, dtype=np.int64) * width, device)

        # Fallback (string-encoded "[1,2,3]" sequences, sliced offsets, ...).
        return self.forward(arr.to_pandas(), device=device)

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> MultiEmbeddingTensor:
        r"""Pad the possibly uneven array of floats with NaNs and turn it into
        a tensor.
        """
        if len(array) == 0:
            return MultiEmbeddingTensor(
                num_rows=0,
                num_cols=1,
                values=torch.empty(0, self.max_length, device=device),
                offset=torch.tensor([0, self.max_length], device=device),
            )
        assert isinstance(array, pa.ListArray)
        return self._pad_or_truncate(array.values.to_numpy(),
                                     array.offsets.to_numpy(), device)

    def _pad_or_truncate(
        self,
        values_np: np.ndarray,
        offsets_np: np.ndarray,
        device: torch.device | None,
    ) -> MultiEmbeddingTensor:
        L = self.max_length
        seq_len_np = np.diff(offsets_np)
        n_rows = len(seq_len_np)
        row_len = int(seq_len_np.max()) if n_rows else 0
        spans = bool(offsets_np[0] == 0 and offsets_np[-1] == len(values_np))
        path = _row_path(seq_len_np, row_len, L, spans)
        full = seq_len_np == row_len
        dense = path == 'dense'
        uniform = path != 'ragged'
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=".*non-writable.*")
            # Never alias Arrow memory.
            values = torch.from_numpy(values_np).to(
                device, torch.get_default_dtype(), copy=dense)

        if dense:
            result = values.view(n_rows, L)
        elif uniform:
            k = min(row_len, L)
            rows = values.view(int(full.sum()), row_len)[:, :k]
            result = torch.full((n_rows, L), float('NaN'), device=device)
            result[torch.from_numpy(full).to(device), :k] = rows
        else:
            seq_len = torch.from_numpy(seq_len_np.astype(np.int64)).to(device)
            drop_count = (seq_len - L).clamp(min=0)
            seq_len = seq_len.clamp(max=L)
            values = values[_tiled_mask(seq_len, drop_count)]
            result = torch.full((n_rows * L, ), float('NaN'), device=device)
            result[_tiled_mask(seq_len, L - seq_len)] = values

        return MultiEmbeddingTensor(
            num_rows=n_rows,
            num_cols=1,
            values=result.view(n_rows, L),
            offset=torch.tensor([0, L], device=device),
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(min_length={self.min_length}, '
                f'max_length={self.max_length})')


class TimestampTensorMapper(TensorMapper[Tensor]):
    r"""Handles the conversion from timestamp values to a tensor.

    Args:
        format: The ``strftime`` to parse time in case input values are given
            as string. If :obj:`None`, will try to infer the correct format.
    """
    stype: Stype = Stype.timestamp

    index: dict[str, int] = {
        'minute': 0,
        'hour': 1,
        'year': 2,
        'day_of_week': 3,
        'day_of_month': 4,
        'day_of_year': 5,
        'days_per_month': 6,  # used for normalisation
        'days_per_year': 7,  # used for normalisation
        'weeks_per_year': 8,  # used for normalisation
        'month': 9,
        'week_of_year': 10,
        'days_to_holiday': 11,
        'days_from_holiday': 12,
        'to_holiday_name': 13,
        'from_holiday_name': 14,
        'unix': 15,
    }
    # NOTE Boundary is for slicing the output for normalization.
    index_boundary: int = 6

    num_holiday_names: int = 11

    def __init__(self, format: str | None = None) -> None:
        self.format = format
        self.helper = TimestampHelper.instance()

    def forward(
        self,
        ser: pd.Series | np.ndarray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        if isinstance(ser, np.ndarray):
            ser = pd.Series(ser, copy=False)
        ser = to_datetime(ser, format=self.format)

        unix = torch.from_numpy(ser.astype('int64').to_numpy()).to(device)
        na_mask = unix == torch.iinfo(torch.int64).min
        unix[na_mask] = 0
        min_date = pd.Timestamp(f'{pd.Timestamp.min.year+1}-01-01').value
        unix = unix.clamp(min=min_date)

        unix = unix // 10**9  # Convert to seconds.

        year_offset = self.helper.year_offset.to(unix.device)
        year_index = torch.searchsorted(
            year_offset,
            unix,
            side='right',
        ) - 1
        year = year_index + pd.Timestamp.min.year + 1

        is_leap_year = (year % 4 == 0) & ((year % 100 != 0) |
                                          (year % 400 == 0))

        day_of_year = unix - year_offset[year_index]
        day_of_year = day_of_year // (60 * 60 * 24)
        days_per_year = torch.where(is_leap_year, 366, 365)

        cumsum_common_days_per_month = (
            self.helper.cumsum_common_days_per_month.to(unix.device))
        cumsum_leap_days_per_month = (
            self.helper.cumsum_leap_days_per_month.to(unix.device))

        month = torch.searchsorted(
            cumsum_common_days_per_month[1:],
            day_of_year,
            side='right',
        )
        month[is_leap_year] = torch.searchsorted(
            cumsum_leap_days_per_month[1:],
            day_of_year[is_leap_year],
            side='right',
        )

        minute = (unix // 60) % 60
        hour = (unix // 3600) % 24

        # Janurary 1st, 1970 is a Thursday, so offset by 3 days such that
        # Monday gets label 0 and Sunday gets label 6:
        day_of_week = (unix // 86400 + 3) % 7

        day_of_month = day_of_year - cumsum_common_days_per_month[month]
        day_of_month[is_leap_year] = (
            day_of_year[is_leap_year] -
            cumsum_leap_days_per_month[month[is_leap_year]])

        common_days_per_month = (self.helper.common_days_per_month.to(
            unix.device))
        leap_days_per_month = self.helper.leap_days_per_month.to(unix.device)

        days_per_month = common_days_per_month[month]
        days_per_month[is_leap_year] = leap_days_per_month[month[is_leap_year]]

        week_of_year = day_of_year // 7  # Approximated.
        weeks_per_year = (days_per_year / 7).ceil().to(torch.long)

        closest_holiday = self.get_closest_holiday(day=unix // SECONDS_IN_DAY)

        tensor = torch.stack(
            [
                minute,
                hour,
                year,
                day_of_week,
                day_of_month,
                day_of_year,
                days_per_month,  # used for normalisation
                days_per_year,  # used for normalisation
                weeks_per_year,  # unused
                month,  # unused
                week_of_year,  # unused
                closest_holiday.days_to_holiday,
                closest_holiday.days_from_holiday,
                closest_holiday.to_holiday_name,
                closest_holiday.from_holiday_name,
                unix,  # Used for rel_time computation
            ],
            dim=-1)

        tensor[na_mask] = -1

        return tensor

    def get_closest_holiday(self, day: Tensor) -> ClosestHoliday:
        r"""Returns the number of days and name of the closest previous and
        upcoming holidays.
        """
        holidays = self.helper.holidays.to(day.device)
        holiday_names = self.helper.holiday_names.to(day.device)

        to_index = torch.searchsorted(holidays, day, side='left')
        to_index = to_index.clamp(max=len(holidays) - 1)
        from_index = torch.searchsorted(holidays, day, side='right') - 1
        from_index = from_index.clamp(min=0)

        return ClosestHoliday(
            days_to_holiday=holidays[to_index] - day,
            to_holiday_name=holiday_names[to_index],
            days_from_holiday=day - holidays[from_index],
            from_holiday_name=holiday_names[from_index],
        )

    def backward(self, tensor: Tensor) -> pd.Series:
        data = tensor[:, torch.tensor([2, 9, 4, 1, 0], device=tensor.device)]
        return pd.Series([
            pd.Timestamp(
                year=year,
                month=month + 1,
                day=day + 1,
                hour=hour,
                minute=minute,
            ) if year >= 0 else None
            for year, month, day, hour, minute in data.tolist()
        ], dtype='datetime64[ns]')

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        out = torch.from_numpy(array.values.to_numpy()).to(device)
        return out.view((len(array), len(self.index)))


class TextTensorMapper(TensorMapper[MultiNestedTensor]):
    r"""Handles the conversion from text to word tokens."""
    stype: Stype = Stype.text

    def __init__(self, model: TokenEmbeddingModel | None = None) -> None:
        self._model = model

    @property
    def model(self) -> TokenEmbeddingModel:
        if self._model is None:
            self._model = resolve_encoder(TextEmbedding())
        return self._model

    def forward(
        self,
        ser: pd.Series | np.ndarray,
        *,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        if isinstance(ser, np.ndarray):
            ser = pd.Series(ser, copy=False)
        rowptr, index = (torch.as_tensor(x, dtype=torch.long)
                         for x in self.model.tokenize(ser))
        if rowptr.numel() != len(ser) + 1:
            raise ValueError(
                f"{self.model!r}.tokenize() returned {rowptr.numel()} row "
                f"pointers for {len(ser)} texts; it must return "
                f"len(texts) + 1.")
        return MultiNestedTensor(
            num_rows=rowptr.numel() - 1,
            num_cols=1,
            values=index.to(device),
            offset=rowptr.to(device),
        )

    def backward(self, tensor: MultiNestedTensor) -> pd.Series:
        assert tensor.num_cols == 1

        if tensor.num_rows == 0:
            return pd.Series([], dtype=str)

        values = self.model.vocabulary()[
            tensor.values.cpu().numpy()].to_numpy()
        ser = pd.Series(np.split(values, tensor.offset[1:-1].cpu().numpy()))
        ser = ser.map(lambda x: ' '.join(x) if len(x) > 0 else None)
        return ser

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        # If the default `Null` encoder was manually overriden to
        # TextEmbedding, PMM didn't take place. In that case, we run a
        # regular `forward`.
        if isinstance(array, pa.StringArray):
            return self.forward(array.to_pandas(), device=device)
        if isinstance(array, pa.ListArray):
            offsets = torch.from_numpy(array.offsets.to_numpy().astype(
                np.int64)).to(device)
            values = torch.from_numpy(array.values.to_numpy().astype(
                np.int64)).to(device)
        else:
            assert len(array) == 0
            offsets = torch.tensor([0], device=device, dtype=torch.int64)
            values = torch.tensor([], device=device, dtype=torch.int64)
        return MultiNestedTensor(num_cols=1, num_rows=len(array),
                                 offset=offsets, values=values)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


def _sort_category_counts(category_counts: list[list[Any]]) -> list[list[Any]]:
    r"""Sort the category counts in descending order."""
    return sorted(category_counts, key=lambda x: x[1], reverse=True)


def _create_mapping_from_category_counts(
        category_counts: list[list[Any]]) -> pd.Index:
    # When mapping contains boolean values, pd.Index directly assigns object
    # data-type. Explicitly check for this case and correct the dtype.
    mapping = [cc[0] for cc in category_counts]
    if len(mapping) > 0 and isinstance(mapping[0], bool):
        return pd.Index(mapping, dtype=pd.BooleanDtype())
    return pd.Index(mapping)


def _tiled_mask(
    pos_count: Tensor,
    neg_count: Tensor,
) -> Tensor:
    r"""Creates a mask where first pos_count[0] values are :obj:`True`,
    next neg_count[0] values :obj:`False`, next pos_count[1] values
    :obj:`True`, etc.
    """
    device = pos_count.device
    assert pos_count.device == neg_count.device
    assert len(pos_count) == len(neg_count)
    # Organise them as array [p0, n0, p1, n1, ...]
    tiles = torch.stack((pos_count, neg_count)).t().flatten()
    mask = torch.BoolTensor([True, False]).to(device).repeat(len(pos_count))
    return torch.repeat_interleave(mask, tiles)


STYPE_TO_TENSOR_MAPPER = {
    Stype.unsupported: UnsupportedTensorMapper,
    Stype.image: ImageTensorMapper,
    Stype.numerical: NumericalTensorMapper,
    Stype.categorical: CategoricalTensorMapper,
    Stype.ID: CategoricalTensorMapper,
    Stype.multicategorical: MultiCategoricalTensorMapper,
    Stype.sequence: SequenceTensorMapper,
    Stype.timestamp: TimestampTensorMapper,
    Stype.text: TextTensorMapper,
}
