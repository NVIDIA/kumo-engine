# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from typing import Any, ClassVar, Generic, NamedTuple, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import torch
from dateutil.parser import parse as parse_datetime
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

from kumo_core.embedding_models import TokenEmbeddingModel, resolve_encoder
from kumo_core.features.time import TimestampHelper
from kumo_core.types.features import SECONDS_IN_DAY

T = TypeVar("T", Tensor, MultiNestedTensor, MultiEmbeddingTensor)


def _arrow_chunks(array: pa.Array | pa.ChunkedArray) -> list[pa.Array]:
    """Return original chunks without concatenating their buffers."""
    return array.chunks if isinstance(array, pa.ChunkedArray) else [array]


def _copy_numpy(destination: Tensor, values: np.ndarray) -> None:
    """Copy a NumPy view into owned tensor storage, converting during the copy."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-writable.*")
        destination.copy_(torch.from_numpy(values))


def _canonical_categories(array: pa.Array) -> pa.Array:
    """Give signed floating zeros a single hash representation."""
    if pa.types.is_floating(array.type):
        return pc.if_else(pc.equal(array, 0), pa.scalar(0, array.type), array)
    return array


def _category_codes(array: pa.Array, vocabulary: pa.Array) -> np.ndarray:
    """Look up Arrow categories without coercing strings into numeric values."""
    if pa.types.is_dictionary(array.type):
        codes = pa.array(_category_codes(array.dictionary, vocabulary))
        return pc.take(codes, array.indices).fill_null(-1).to_numpy()
    if not len(vocabulary) or array.null_count == len(array):
        return np.full(len(array), -1, dtype=np.int64)
    array = _canonical_categories(array)
    try:
        codes = pc.index_in(array, value_set=vocabulary)
    except (pa.ArrowTypeError, pa.ArrowNotImplementedError):
        return np.full(len(array), -1, dtype=np.int64)
    return codes.fill_null(-1).to_numpy()


def _is_list(array: pa.Array | pa.ChunkedArray) -> bool:
    """Recognize variable- and fixed-width Arrow list columns."""
    value_type = array.type
    if pa.types.is_dictionary(value_type):
        value_type = value_type.value_type
    return (
        pa.types.is_list(value_type)
        or pa.types.is_large_list(value_type)
        or pa.types.is_fixed_size_list(value_type)
    )


def _list_values_offsets(array: pa.Array) -> tuple[pa.Array, np.ndarray]:
    """Flatten visible list rows and normalize offsets, discarding null slots."""
    if isinstance(array, (pa.ListArray, pa.LargeListArray)) and not array.null_count:
        offsets = array.offsets.to_numpy().astype(np.int64)
        start = int(offsets[0])
        values = array.values.slice(start, int(offsets[-1]) - start)
        offsets -= start
        return pc.dictionary_decode(values), offsets
    lengths = pc.fill_null(pc.list_value_length(array), 0).to_numpy()
    offsets = np.empty(len(array) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return pc.dictionary_decode(pc.list_flatten(array)), offsets


def _nested_from_arrow(
    array: pa.Array, device: torch.device | None
) -> MultiNestedTensor:
    """Decode visible serialized integer lists into an owned nested tensor."""
    if pa.types.is_null(array.type):
        array = pa.array([None] * len(array), type=pa.list_(pa.int64()))
    values, offsets = _list_values_offsets(array)
    indices = pc.cast(values, pa.int64()).fill_null(-1).to_numpy().copy()
    return MultiNestedTensor(
        num_cols=1,
        num_rows=len(array),
        offset=torch.from_numpy(offsets).to(device),
        values=torch.from_numpy(indices).to(device),
    )


def _timestamp_nanoseconds(array: pa.Array, format: str | None) -> np.ndarray:
    """Normalize typed timestamps or parsed text to nanoseconds."""
    array = pc.dictionary_decode(array)
    if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
        if format is not None and "%f" not in format:
            array = pc.strptime(array, format=format, unit="ns", error_is_null=True)
        elif format is not None:
            array = pa.array(
                [_parse_timestamp(value, format) for value in array.to_pylist()],
                type=pa.int64(),
            ).cast(pa.timestamp("ns"))
        else:
            first = next((value.as_py() for value in array if value.is_valid), "")
            tz = "UTC" if re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", first) else None
            try:
                array = pc.cast(array, pa.timestamp("ns", tz=tz))
            except pa.ArrowInvalid:
                parsed = [
                    _parse_timestamp(value, format) for value in array.to_pylist()
                ]
                array = pa.array(parsed, type=pa.int64()).cast(pa.timestamp("ns"))
    elif pa.types.is_null(array.type) or pa.types.is_time(array.type):
        return np.full(len(array), np.iinfo(np.int64).min, dtype=np.int64)
    elif pa.types.is_floating(array.type):
        array = pc.if_else(
            pc.is_finite(array), pc.cast(array, pa.int64(), safe=False), None
        )
    else:
        array = pc.cast(array, pa.timestamp("ns"))
    return pc.cast(array, pa.int64()).fill_null(np.iinfo(np.int64).min).to_numpy()


def _parse_timestamp(value: str | None, format: str | None) -> int | None:
    """Parse irregular strings without losing fractional nanoseconds."""
    if not value:
        return None
    if format is None:
        for arrow_type in (pa.timestamp("ns"), pa.timestamp("ns", tz="UTC")):
            try:
                return pa.scalar(value).cast(arrow_type).cast(pa.int64()).as_py()
            except pa.ArrowInvalid:
                pass
    remainder = 0
    text = value
    fraction = re.search(r"([.,])(\d{6})(\d{1,3})(?=\D|$)", text)
    if fraction is not None:
        remainder = int(fraction[3].ljust(3, "0"))
        text = text[: fraction.start(3)] + text[fraction.end(3) :]
    try:
        parsed = (
            datetime.strptime(text, format)
            if format is not None
            else parse_datetime(text, default=datetime(1, 1, 1))
        )
        timestamp = pa.scalar(parsed)
        return timestamp.cast(pa.timestamp("ns")).cast(pa.int64()).as_py() + remainder
    except (ValueError, OverflowError, pa.ArrowInvalid):
        return None


class ClosestHoliday(NamedTuple):
    days_to_holiday: Tensor
    to_holiday_name: Tensor
    days_from_holiday: Tensor
    from_holiday_name: Tensor


class TensorMapper(ABC, Generic[T]):
    r"""A base class to handle the conversion from raw input data into a
    compact tensor representation, *i.e.*, the identity for numerical values,
    indices for categorical values, etc.
    """

    stype: Stype

    def __call__(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> T:
        """Encode Arrow values chunk by chunk into owned destination storage.

        Args:
            array: Raw column values, retaining their Arrow chunk boundaries.
            device: Destination device, or the default device when omitted.

        Returns:
            The column's compact tensor representation.

        Raises:
            TypeError: If the input is not an Arrow array or chunked array.
        """
        if not isinstance(array, (pa.Array, pa.ChunkedArray)):
            raise TypeError(
                "TensorMapper input must be an Arrow array or chunked array"
            )
        return self.forward(array, device=device)

    @abstractmethod
    def forward(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> T:
        """Encode an Arrow array into a compact tensor representation.

        Args:
            array: Raw column values in Arrow chunks.
            device: Destination device, or the default device when omitted.

        Returns:
            The column's compact tensor representation.
        """

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
        return f"{self.__class__.__name__}()"


class UnsupportedTensorMapper(TensorMapper[Tensor]):
    r"""Handles the conversion from unsupported values to tensors."""

    stype: Stype = Stype.unsupported

    def forward(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        """Return an empty feature tensor with the Arrow input's row count."""
        return torch.empty((len(array), 0), device=device)

    def backward(self, tensor: Tensor) -> pd.Series:
        assert isinstance(tensor, Tensor)
        raise RuntimeError(
            f"'{self.__class__.__name__}' cannot map empty "
            f"tensor representation back to raw inputs"
        )

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
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        """Return an empty feature tensor with the Arrow input's row count."""
        return torch.empty((len(array), 0), device=device)

    def backward(self, tensor: Tensor) -> pd.Series:
        assert isinstance(tensor, Tensor)
        raise RuntimeError(
            f"'{self.__class__.__name__}' cannot map empty "
            f"tensor representation back to raw inputs"
        )

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        return torch.empty((len(array), 0), device=device)


class NumericalTensorMapper(TensorMapper[Tensor]):
    r"""Handles the conversion from numerical values to tensors."""

    stype: Stype = Stype.numerical

    def backward(self, tensor: Tensor) -> pd.Series:
        if tensor.dim() == 2 and tensor.size(1) > 1:
            arr = tensor.detach().cpu().numpy()
            return pd.Series([row for row in arr])
        return pd.Series(tensor.view(-1).detach().cpu().numpy())

    def forward(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        """Encode numerical Arrow values, normalizing durations to seconds."""
        result = torch.empty((len(array), 1), device=device)
        row = 0
        for chunk in _arrow_chunks(array):
            chunk = pc.dictionary_decode(chunk)
            duration = pa.types.is_duration(chunk.type)
            values = (pc.cast(chunk, pa.int64()) if duration else chunk).to_numpy(
                zero_copy_only=False
            )
            if values.dtype.kind == "O":
                values = values.astype(str(torch.get_default_dtype()).split(".")[-1])
            destination = result[row : row + len(chunk), 0]
            if duration:
                scale = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
                if destination.device.type == "cpu":
                    np.divide(
                        values,
                        scale[chunk.type.unit],
                        out=destination.numpy(),
                        casting="unsafe",
                    )
                else:
                    _copy_numpy(destination, values / scale[chunk.type.unit])
            else:
                _copy_numpy(destination, values)
            row += len(chunk)
        return result

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        dtype = str(torch.get_default_dtype()).split(".")[-1]
        return (
            torch.from_numpy(array.to_numpy(zero_copy_only=False).astype(dtype))
            .to(device)
            .view(-1, 1)
        )


class CategoricalTensorMapper(TensorMapper[Tensor]):
    r"""Handles the conversion from categorical values to consecutive indices.

    Args:
        category_counts: Category counts to create a mapping from categorical
            values to integers.
    """

    stype: Stype = Stype.categorical

    def __init__(self, category_counts: Sequence[Sequence[Any]]) -> None:
        category_counts = _sort_category_counts(category_counts)
        self.mapping = _create_mapping_from_category_counts(category_counts)
        self._vocabulary = _canonical_categories(
            pa.array([value for value, _ in category_counts])
        )

    def backward(self, tensor: Tensor) -> pd.Series:
        if len(self.mapping) == 0:
            return pd.Series([None] * tensor.size(0), dtype=self.mapping.dtype)

        index = tensor.view(-1).cpu().numpy()
        ser = pd.Series(self.mapping[index].to_numpy(), dtype=self.mapping.dtype)
        ser[index < 0] = None
        return ser

    def forward(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        """Map Arrow categories to cached vocabulary indices; unknowns become -1."""
        result = torch.empty((len(array), 1), dtype=torch.long, device=device)
        row = 0
        for chunk in _arrow_chunks(array):
            _copy_numpy(
                result[row : row + len(chunk), 0],
                _category_codes(chunk, self._vocabulary),
            )
            row += len(chunk)
        return result

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        return (
            torch.from_numpy(array.to_numpy(zero_copy_only=False).astype(np.int64))
            .to(device)
            .view(-1, 1)
        )


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
        category_counts: Sequence[Sequence[Any]],
        sep: str | None,
    ) -> None:
        category_counts = _sort_category_counts(category_counts)
        self.mapping = _create_mapping_from_category_counts(category_counts)
        self._vocabulary = _canonical_categories(
            pa.array([value for value, _ in category_counts])
        )
        self.sep = sep
        self._is_string_dtype: bool | None = None

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
            return ser.map(lambda x: x[0] if len(x) > 0 else "")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sep={self.sep})"

    def forward(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        """Encode Arrow lists and strings without constructing pandas objects."""
        if self._is_string_dtype is None and len(array):
            self._is_string_dtype = not _is_list(array) and array.null_count < len(
                array
            )
        chunks = [self._prepare_chunk(chunk) for chunk in _arrow_chunks(array)]
        encoded_values = torch.empty(
            sum(len(values) for values, _ in chunks), dtype=torch.long, device=device
        )
        encoded_offsets = torch.empty(len(array) + 1, dtype=torch.long, device=device)
        row = value_start = 0
        encoded_offsets[0] = 0
        for values, offsets in chunks:
            _copy_numpy(
                encoded_values[value_start : value_start + len(values)],
                _category_codes(values, self._vocabulary),
            )
            destination = encoded_offsets[row + 1 : row + len(offsets)]
            _copy_numpy(destination, offsets[1:])
            destination.add_(value_start)
            row += len(offsets) - 1
            value_start += len(values)
        return MultiNestedTensor(
            num_rows=len(array),
            num_cols=1,
            values=encoded_values,
            offset=encoded_offsets,
        )

    def _prepare_chunk(self, array: pa.Array) -> tuple[pa.Array, np.ndarray]:
        """Prepare category values and row offsets before sizing the destination."""
        array = pc.dictionary_decode(array)
        is_list = _is_list(array)
        if is_list:
            if isinstance(array, pa.FixedSizeListArray):
                array = pc.cast(array, pa.list_(array.type.value_type))
            # Existing models expect one -1 token for both empty and null lists.
            missing_rows = pc.fill_null(pc.equal(pc.list_value_length(array), 0), True)
            if pc.any(missing_rows).as_py():
                array = pc.if_else(
                    missing_rows, pa.scalar([None], type=array.type), array
                )
            values, offsets = _list_values_offsets(array)
        elif self.sep is not None:
            strings = pc.fill_null(pc.cast(array, pa.string()), "__None__")
            split = pc.split_pattern if len(self.sep) == 1 else pc.split_pattern_regex
            values, offsets = _list_values_offsets(split(strings, pattern=self.sep))
        else:
            values = array
            offsets = np.arange(len(array) + 1, dtype=np.int64)

        if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
            keep = pc.fill_null(pc.not_equal(values, ""), True)
            keep_np = keep.to_numpy(zero_copy_only=False)
            if not keep_np.all():
                prefix = np.empty(len(values) + 1, dtype=np.int64)
                prefix[0] = 0
                np.cumsum(keep_np, out=prefix[1:])
                offsets = prefix[offsets]
                values = pc.filter(values, keep)
        return values, offsets

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        return _nested_from_arrow(array, device)


def _row_path(
    seq_len_np: np.ndarray,
    row_len: int,
    max_length: int,
    spans: bool,
) -> str:
    # 'dense': every row is max_length long; 'uniform': rows are all
    # row_len or empty; 'ragged': anything else.
    if not spans or len(seq_len_np) == 0:
        return "ragged"
    full = seq_len_np == row_len
    if not (full | (seq_len_np == 0)).all():
        return "ragged"
    return "dense" if row_len == max_length and full.all() else "uniform"


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
            dtype = str(torch.get_default_dtype()).split(".")[-1]
            return ser.map(
                lambda row: (
                    np.fromstring(
                        row.strip("[]"),
                        sep=",",
                        dtype=dtype,
                    )
                    if row is not None
                    else None
                )
            )
        return ser

    def backward(self, tensor: MultiEmbeddingTensor) -> pd.Series:
        assert tensor.num_cols == 1
        ser = pd.Series(tensor.values.tolist(), dtype=object)
        if self._is_string_dtype:
            ser = ser.map(lambda row: str(row))
        return ser

    def forward(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> MultiEmbeddingTensor:
        """Encode Arrow lists and strings without constructing pandas objects."""
        if self._is_string_dtype is None and len(array):
            self._is_string_dtype = not _is_list(array) and array.null_count < len(
                array
            )
        result = torch.empty((len(array), self.max_length), device=device)
        row = 0
        for chunk in _arrow_chunks(array):
            self._encode_chunk(chunk, result[row : row + len(chunk)])
            row += len(chunk)
        return MultiEmbeddingTensor(
            num_rows=len(array),
            num_cols=1,
            values=result,
            offset=torch.tensor([0, self.max_length], device=device),
        )

    def _encode_chunk(self, array: pa.Array, destination: Tensor) -> None:
        """Encode sequence rows directly into a slice of the destination."""
        array = pc.dictionary_decode(array)
        if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
            dtype = str(torch.get_default_dtype()).split(".")[-1]
            rows = [
                np.fromstring(row.strip("[]"), sep=",", dtype=dtype)
                if row is not None
                else None
                for row in array.to_pylist()
            ]
            array = pa.array(rows, type=pa.list_(pa.from_numpy_dtype(dtype)))
        if pa.types.is_null(array.type):
            array = pa.array([None] * len(array), type=pa.list_(pa.float64()))
        values, offsets = _list_values_offsets(array)
        self._write_sequences(
            values.to_numpy(zero_copy_only=False), offsets, destination
        )

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> MultiEmbeddingTensor:
        r"""Pad the possibly uneven array of floats with NaNs and turn it into
        a tensor.
        """
        if pa.types.is_null(array.type):
            array = pa.array([None] * len(array), type=pa.list_(pa.float64()))
        values, offsets = _list_values_offsets(array)
        return self._pad_or_truncate(
            values.to_numpy(zero_copy_only=False), offsets, device
        )

    def _pad_or_truncate(
        self,
        values_np: np.ndarray,
        offsets_np: np.ndarray,
        device: torch.device | None,
    ) -> MultiEmbeddingTensor:
        result = torch.empty((len(offsets_np) - 1, self.max_length), device=device)
        self._write_sequences(values_np, offsets_np, result)
        return MultiEmbeddingTensor(
            num_rows=len(offsets_np) - 1,
            num_cols=1,
            values=result,
            offset=torch.tensor([0, self.max_length], device=device),
        )

    def _write_sequences(
        self, values_np: np.ndarray, offsets_np: np.ndarray, result: Tensor
    ) -> None:
        """Pad or truncate Arrow value views into the final sequence storage."""
        device = result.device
        if values_np.dtype.kind == "O":
            dtype = str(torch.get_default_dtype()).split(".")[-1]
            values_np = values_np.astype(dtype)
        L = self.max_length
        seq_len_np = np.diff(offsets_np)
        n_rows = len(seq_len_np)
        row_len = int(seq_len_np.max()) if n_rows else 0
        spans = bool(offsets_np[0] == 0 and offsets_np[-1] == len(values_np))
        path = _row_path(seq_len_np, row_len, L, spans)
        full = seq_len_np == row_len
        dense = path == "dense"
        uniform = path != "ragged"
        if dense:
            _copy_numpy(result, values_np.reshape(n_rows, L))
        elif uniform:
            result.fill_(float("NaN"))
            k = min(row_len, L)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*non-writable.*")
                rows = torch.from_numpy(values_np).view(int(full.sum()), row_len)
            result[torch.from_numpy(full).to(device), :k] = rows[:, :k].to(
                device, result.dtype
            )
        else:
            result.fill_(float("NaN"))
            seq_len = torch.from_numpy(seq_len_np.astype(np.int64)).to(device)
            drop_count = (seq_len - L).clamp(min=0)
            seq_len = seq_len.clamp(max=L)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*non-writable.*")
                values = torch.from_numpy(values_np).to(device, result.dtype)
            result.view(-1)[_tiled_mask(seq_len, L - seq_len)] = values[
                _tiled_mask(seq_len, drop_count)
            ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(min_length={self.min_length}, "
            f"max_length={self.max_length})"
        )


class TimestampTensorMapper(TensorMapper[Tensor]):
    r"""Handles the conversion from timestamp values to a tensor.

    Args:
        format: The ``strftime`` to parse time in case input values are given
            as string. If :obj:`None`, will try to infer the correct format.
    """

    stype: Stype = Stype.timestamp

    index: ClassVar[dict[str, int]] = {
        "minute": 0,
        "hour": 1,
        "year": 2,
        "day_of_week": 3,
        "day_of_month": 4,
        "day_of_year": 5,
        "days_per_month": 6,  # used for normalisation
        "days_per_year": 7,  # used for normalisation
        "weeks_per_year": 8,  # used for normalisation
        "month": 9,
        "week_of_year": 10,
        "days_to_holiday": 11,
        "days_from_holiday": 12,
        "to_holiday_name": 13,
        "from_holiday_name": 14,
        "unix": 15,
    }
    # NOTE Boundary is for slicing the output for normalization.
    index_boundary: int = 6

    num_holiday_names: int = 11

    def __init__(self, format: str | None = None) -> None:
        self.format = format
        self.helper = TimestampHelper.instance()

    def forward(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> Tensor:
        """Encode Arrow timestamps as calendar, holiday, and UNIX features."""
        result = torch.empty(
            (len(array), len(self.index)), dtype=torch.long, device=device
        )
        row = 0
        for chunk in _arrow_chunks(array):
            self._encode_chunk(chunk, result[row : row + len(chunk)])
            row += len(chunk)
        return result

    def _encode_chunk(self, array: pa.Array, destination: Tensor) -> None:
        """Write calendar, holiday, and UNIX features into destination rows."""
        unix = destination[:, self.index["unix"]]
        _copy_numpy(unix, _timestamp_nanoseconds(array, self.format))
        na_mask = unix == torch.iinfo(torch.int64).min
        unix[na_mask] = 0
        min_date = np.datetime64("1678-01-01", "ns").astype(np.int64).item()
        unix.clamp_(min=min_date)
        unix.div_(10**9, rounding_mode="floor")

        year_offset = self.helper.year_offset.to(unix.device)
        year_index = (
            torch.searchsorted(
                year_offset,
                unix,
                side="right",
            )
            - 1
        )
        year = year_index + 1678

        is_leap_year = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))

        day_of_year = unix - year_offset[year_index]
        day_of_year = day_of_year // (60 * 60 * 24)
        days_per_year = torch.where(is_leap_year, 366, 365)

        cumsum_common_days_per_month = self.helper.cumsum_common_days_per_month.to(
            unix.device
        )
        cumsum_leap_days_per_month = self.helper.cumsum_leap_days_per_month.to(
            unix.device
        )

        month = torch.searchsorted(
            cumsum_common_days_per_month[1:],
            day_of_year,
            side="right",
        )
        month[is_leap_year] = torch.searchsorted(
            cumsum_leap_days_per_month[1:],
            day_of_year[is_leap_year],
            side="right",
        )

        minute = (unix // 60) % 60
        hour = (unix // 3600) % 24

        # Janurary 1st, 1970 is a Thursday, so offset by 3 days such that
        # Monday gets label 0 and Sunday gets label 6:
        day_of_week = (unix // 86400 + 3) % 7

        day_of_month = day_of_year - cumsum_common_days_per_month[month]
        day_of_month[is_leap_year] = (
            day_of_year[is_leap_year] - cumsum_leap_days_per_month[month[is_leap_year]]
        )

        common_days_per_month = self.helper.common_days_per_month.to(unix.device)
        leap_days_per_month = self.helper.leap_days_per_month.to(unix.device)

        days_per_month = common_days_per_month[month]
        days_per_month[is_leap_year] = leap_days_per_month[month[is_leap_year]]

        week_of_year = day_of_year // 7  # Approximated.
        weeks_per_year = (days_per_year / 7).ceil().to(torch.long)

        closest_holiday = self.get_closest_holiday(day=unix // SECONDS_IN_DAY)

        for column, feature in enumerate(
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
            ]
        ):
            destination[:, column].copy_(feature)
        destination[na_mask] = -1

    def get_closest_holiday(self, day: Tensor) -> ClosestHoliday:
        r"""Returns the number of days and name of the closest previous and
        upcoming holidays.
        """
        holidays = self.helper.holidays.to(day.device)
        holiday_names = self.helper.holiday_names.to(day.device)

        to_index = torch.searchsorted(holidays, day, side="left")
        to_index = to_index.clamp(max=len(holidays) - 1)
        from_index = torch.searchsorted(holidays, day, side="right") - 1
        from_index = from_index.clamp(min=0)

        return ClosestHoliday(
            days_to_holiday=holidays[to_index] - day,
            to_holiday_name=holiday_names[to_index],
            days_from_holiday=day - holidays[from_index],
            from_holiday_name=holiday_names[from_index],
        )

    def backward(self, tensor: Tensor) -> pd.Series:
        data = tensor[:, torch.tensor([2, 9, 4, 1, 0], device=tensor.device)]
        return pd.Series(
            [
                pd.Timestamp(
                    year=year,
                    month=month + 1,
                    day=day + 1,
                    hour=hour,
                    minute=minute,
                )
                if year >= 0
                else None
                for year, month, day, hour, minute in data.tolist()
            ],
            dtype="datetime64[ns]",
        )

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> Tensor:
        if len(array) == 0:
            return torch.empty((0, len(self.index)), dtype=torch.long, device=device)
        values, offsets = _list_values_offsets(array)
        if not np.all(np.diff(offsets) == len(self.index)):
            raise ValueError(
                "Serialized timestamps must contain all timestamp features"
            )
        return (
            torch.from_numpy(values.to_numpy().copy())
            .to(device)
            .view(len(array), len(self.index))
        )


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
        array: pa.Array | pa.ChunkedArray,
        *,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        """Tokenize Arrow text through the configured embedding model.

        Raises:
            ValueError: If tokenization returns an invalid row-pointer count.
        """
        chunks: list[tuple[Tensor, Tensor]] = []
        for chunk in _arrow_chunks(array):
            rowptr, index = (
                torch.as_tensor(x, dtype=torch.long) for x in self.model.tokenize(chunk)
            )
            if rowptr.numel() != len(chunk) + 1:
                raise ValueError(
                    f"{self.model!r}.tokenize() returned {rowptr.numel()} row "
                    f"pointers for {len(chunk)} texts; it must return "
                    f"len(texts) + 1."
                )
            chunks.append((rowptr, index))
        encoded_values = torch.empty(
            sum(index.numel() for _, index in chunks), dtype=torch.long, device=device
        )
        encoded_offsets = torch.empty(len(array) + 1, dtype=torch.long, device=device)
        row = value_start = 0
        encoded_offsets[0] = 0
        for rowptr, index in chunks:
            encoded_values[value_start : value_start + index.numel()].copy_(index)
            destination = encoded_offsets[row + 1 : row + rowptr.numel()]
            destination.copy_(rowptr[1:])
            destination.add_(value_start)
            row += rowptr.numel() - 1
            value_start += index.numel()
        return MultiNestedTensor(
            num_rows=len(array),
            num_cols=1,
            values=encoded_values,
            offset=encoded_offsets,
        )

    def backward(self, tensor: MultiNestedTensor) -> pd.Series:
        assert tensor.num_cols == 1

        if tensor.num_rows == 0:
            return pd.Series([], dtype=str)

        values = self.model.vocabulary()[tensor.values.cpu().numpy()].to_numpy()
        ser = pd.Series(np.split(values, tensor.offset[1:-1].cpu().numpy()))
        ser = ser.map(lambda x: " ".join(x) if len(x) > 0 else None)
        return ser

    def post_forward(
        self,
        array: pa.Array,
        device: torch.device | None = None,
    ) -> MultiNestedTensor:
        # If the default `Null` encoder was manually overriden to
        # TextEmbedding, PMM didn't take place. In that case, we run a
        # regular `forward`.
        if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
            return self.forward(array, device=device)
        return _nested_from_arrow(array, device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _sort_category_counts(
    category_counts: Sequence[Sequence[Any]],
) -> list[Sequence[Any]]:
    r"""Sort the category counts in descending order."""
    return sorted(category_counts, key=lambda x: x[1], reverse=True)


def _create_mapping_from_category_counts(
    category_counts: Sequence[Sequence[Any]],
) -> pd.Index:
    # When mapping contains boolean values, pd.Index directly assigns object
    # data-type. Explicitly check for this case and correct the dtype.
    mapping = [cc[0] for cc in category_counts]
    dtype = pd.BooleanDtype() if mapping and isinstance(mapping[0], bool) else None
    index = pd.Index(mapping, dtype=dtype)
    if index.hasnans:
        raise ValueError("Categorical categories cannot be null")
    if not index.is_unique:
        raise ValueError("Categorical categories must be unique")
    return index


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
