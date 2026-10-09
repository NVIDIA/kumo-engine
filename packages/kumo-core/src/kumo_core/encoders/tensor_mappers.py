# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from kumoapi.typing import Stype
from torch import Tensor
from torch_frame import TensorFrame
from torch_frame.data import MultiEmbeddingTensor, MultiNestedTensor
from torch_frame.typing import TensorData
from torch_geometric.utils.mixin import CastMixin

import logging
from kumo_core.encoders.config import EncodersConfig
from kumo_core.encoders.tensor_mapper import TensorMapper

logger = logging.getLogger(__name__)


def _to_tf(
    df: pd.DataFrame | dict[str, pa.Array] | dict[str, np.ndarray],
    mappers: dict[str, TensorMapper],
    *,
    device: torch.device | None = None,
    post_forward_only: bool = False,
    col_name_hook: Callable[[str], str] | None = None,
) -> TensorFrame:
    """Converts a data frame into a TensorFrame.

    Args:
        df: The input data frame, dict of PyArrow arrays, or dict of numpy
            arrays to convert. Encoders that have a numpy-aware ``forward``
            consume the array directly, while the rest wrap it in
            ``pd.Series`` internally.
        mappers: A dictionary mapping column names to their respective
            TensorMapper objects for tensor conversion.
        device: The torch device to place tensors on. If None, uses the
            default device.
        post_forward_only: If True, expects the data to already be
            pre-materialized and only applies post-forward transformations.
            If False, applies full forward transformations on raw data.
        col_name_hook: An optional callable that transforms column names
            before looking them up in the dataframe. Used when column names
            in the dataframe differ from the mapper keys (e.g., with
            pre-materialized data).

    Returns:
        A TensorFrame containing the converted tensor representations grouped
        by semantic type.

    .. note::
        Stype.unsupported and Stype.image columns are not included in the
        TensorFrame because they are not consumed by our ML model.

    Note on device transfer: per-column tensors are produced on CPU
    (``device=None``) and a single H2D copy is issued per stype after the
    columns are concatenated. This collapses N small per-column transfers
    into one larger transfer per stype.
    """
    xs_dict: defaultdict[str, list[TensorData]] = defaultdict(list)
    col_names_dict: dict[str, list[str]] = defaultdict(list)
    for col_name, mapper in mappers.items():
        if mapper.stype in {Stype.unsupported, Stype.image}:
            continue

        mapped_col = col_name_hook(col_name) if col_name_hook else col_name
        if post_forward_only:
            assert mapped_col in df
            ser = df[mapped_col]
            x = mapper.post_forward(
                ser,
                device=None,
            )
        else:
            ser = df[col_name]
            if isinstance(ser, pa.Array):
                ser = ser.to_pandas()
            x = mapper.forward(
                ser,
                device=None,
            )
        # TODO (akihironitta) Use stype as key once PyTorch addresses:
        # https://github.com/pytorch/pytorch/issues/111603
        xs_dict[mapper.stype.value].append(x)
        col_names_dict[mapper.stype.value].append(col_name)

    x_dict: dict[str, TensorData] = {}
    for stype, xs in xs_dict.items():
        if Stype(stype) in {Stype.multicategorical, Stype.text}:
            stacked = MultiNestedTensor.cat(xs, dim=1)

        elif Stype(stype) == Stype.sequence:
            stacked = MultiEmbeddingTensor.cat(xs, dim=1)

        elif Stype(stype) in {Stype.categorical, Stype.numerical}:
            # [num_rows, num_cols]
            stacked = torch.cat(cast(list[Tensor], xs), dim=1)

        else:
            # [num_rows, num_cols, num_features]
            stacked = torch.stack(cast(list[Tensor], xs), dim=1)

        if device is not None:
            stacked = stacked.to(device, non_blocking=True)
        x_dict[stype] = stacked

    if isinstance(df, pd.DataFrame):
        num_rows = len(df)
    else:
        num_rows = len(list(df.values())[0]) if len(df) > 0 else 0
    return TensorFrame(x_dict, col_names_dict, num_rows=num_rows)


def _to_tf_vectorized(
    df: dict[str, np.ndarray],
    mappers: dict[str, TensorMapper],
    *,
    device: torch.device | None = None,
) -> TensorFrame:
    r"""Fast path of :func:`_to_tf` for ``dict[str, ndarray]`` inputs.

    Groups mappers by stype and processes each stype in one batched op:

    - Numerical: one ``np.stack`` + ``torch.from_numpy`` over all columns.
    - Categorical: per-column ``pd.Index.get_indexer`` into one
      preallocated ``(num_rows, num_cols)`` int64 buffer + one
      ``torch.from_numpy``.
    - Other stypes (multicategorical / sequence / timestamp / text): fall
      back to per-column ``mapper.forward`` + per-stype concat, same as
      :func:`_to_tf`.

    Only callable with ``dict[str, np.ndarray]``; ``post_forward_only`` and
    ``col_name_hook`` callers stay on :func:`_to_tf`.
    """
    # Group mappers by stype so we can pick a batched implementation per
    # stype instead of a per-column loop over the whole mapper dict.
    by_stype: dict[str, list[tuple[str, TensorMapper]]] = defaultdict(list)
    for col_name, mapper in mappers.items():
        if mapper.stype in {Stype.unsupported, Stype.image}:
            continue
        by_stype[mapper.stype.value].append((col_name, mapper))

    num_rows = len(next(iter(df.values()))) if len(df) > 0 else 0
    default_dtype_str = str(torch.get_default_dtype()).split('.')[-1]

    x_dict: dict[str, TensorData] = {}
    col_names_dict: dict[str, list[str]] = {}

    for stype_value, items in by_stype.items():
        stype = Stype(stype_value)
        col_names = [name for name, _ in items]
        col_names_dict[stype_value] = col_names

        if stype == Stype.numerical:
            # One stack + one torch.from_numpy for all numerical columns.
            # NumericalTensorMapper.forward's numpy path is:
            #   torch.from_numpy(arr.astype(dtype)).view(-1, 1)
            # so column-stacking the cast numpy arrays produces an
            # equivalent (num_rows, num_cols) tensor.
            if num_rows == 0 or len(items) == 0:
                stacked = torch.empty(
                    (num_rows, len(items)),
                    dtype=torch.get_default_dtype(),
                )
            else:
                casted = [
                    df[name].astype(default_dtype_str, copy=False)
                    for name in col_names
                ]
                stacked = torch.from_numpy(np.stack(casted, axis=1))
            if device is not None:
                stacked = stacked.to(device, non_blocking=True)
            x_dict[stype_value] = stacked

        elif stype == Stype.categorical:
            # Each col has its own mapping → get_indexer is per-col. But
            # we can write the int64 results into one preallocated buffer
            # and do a single torch.from_numpy + H2D for the whole stype.
            out = np.empty((num_rows, len(items)), dtype=np.int64)
            if num_rows > 0 and len(items) > 0:
                for i, (name, mapper) in enumerate(items):
                    # Mirrors CategoricalTensorMapper.forward's numpy
                    # fast path: pd.Index.get_indexer returns -1 for
                    # missing / unknown values, matching the
                    # categorical-encoder contract.
                    out[:, i] = mapper.mapper._index.get_indexer(df[name])
            stacked = torch.from_numpy(out)
            if device is not None:
                stacked = stacked.to(device, non_blocking=True)
            x_dict[stype_value] = stacked

        else:
            # Variable-length / less-frequent stypes: fall back to the
            # baseline per-col forward + per-stype concat. Behavior is
            # identical to _to_tf for these stypes.
            xs: list[TensorData] = []
            for name, mapper in items:
                xs.append(mapper.forward(df[name], device=None))

            if stype in {Stype.multicategorical, Stype.text}:
                stacked = MultiNestedTensor.cat(xs, dim=1)
            elif stype == Stype.sequence:
                stacked = MultiEmbeddingTensor.cat(xs, dim=1)
            else:
                # timestamp / text / anything else: each forward returns
                # (num_rows, num_features) per col; stack along dim=1.
                stacked = torch.stack(cast(list[Tensor], xs), dim=1)

            if device is not None:
                stacked = stacked.to(device, non_blocking=True)
            x_dict[stype_value] = stacked

    return TensorFrame(x_dict, col_names_dict, num_rows=num_rows)


@dataclass
class TargetTensorMapper(CastMixin):
    col_name: str
    mapper: TensorMapper


class TensorMappers:
    r"""A dataclass that contains the tensor mappers passed to a feature store.
    Can be serialized and deserialized to/from JSON.

    For each table, the feature store will then take raw features and maps each
    of its column to a tensor representation, grouped by semantic type.
    """
    def __init__(
        self,
        # Mapping from table and column to its tensor mapper:
        mappers: dict[str, dict[str, TensorMapper]] | None = None,
        # The target tensor mapper information used for the training table:
        target_mapper: TargetTensorMapper | None = None,
    ) -> None:
        self.mappers: dict[str, dict[str, TensorMapper]] = mappers or {}
        self.target = TargetTensorMapper.cast(target_mapper)

    @classmethod
    def from_encoders(
        cls,
        encoders: EncodersConfig,
    ) -> 'TensorMappers':
        mappers = cls()

        for table_name, table_encoders in encoders.items():
            mappers[table_name] = {}
            for col_name, encoder in table_encoders.items():
                mappers[table_name][col_name] = TensorMapper.from_encoder(
                    encoder)
        # TODO Store target column mapper as a dictionary for consistency.
        if encoders.target is not None:
            mappers.target = TargetTensorMapper(
                col_name=encoders.target.col_name,
                mapper=TensorMapper.from_encoder(encoders.target.encoder),
            )

        return mappers

    def __getitem__(self, table_name: str) -> dict[str, TensorMapper]:
        if table_name not in self.mappers:
            self.mappers[table_name] = {}
        return self.mappers[table_name]

    def __setitem__(
        self,
        table_name: str,
        table_mappers: dict[str, TensorMapper],
    ) -> None:
        self.mappers[table_name] = table_mappers

    def keys(self) -> Iterable[str]:
        return self.mappers.keys()

    def values(self) -> Iterable[dict[str, TensorMapper]]:
        return self.mappers.values()

    def items(self) -> Iterable[tuple[str, dict[str, TensorMapper]]]:
        return self.mappers.items()

    def convert_pmm_to_tensor_frame(
        self,
        table_name: str,
        df: dict[str, pa.Array],
        col_name_hook: Callable[[str], str],
        *,
        device: torch.device | None = None,
    ) -> TensorFrame:
        r"""Converts a dict of pyarrow arrays with pre-materialised data into
        a TensorFrame.
        """
        return self.convert_to_tensor_frame(
            table_name=table_name,
            df=df,
            device=device,
            post_forward_only=True,
            col_name_hook=col_name_hook,
        )

    def convert_raw_to_tensor_frame(
        self,
        table_name: str,
        df: dict[str, pa.Array],
        *,
        device: torch.device | None = None,
    ) -> TensorFrame:
        r"""Converts a dict of pyarrow arrays with raw data into a TensorFrame."""  # noqa: E501
        # Convert from response (proto) to arrow to pandas (note that
        # arrow -> pandas is zero-copy):
        df_dict = {}
        for col_name, arrow_array in df.items():
            self[table_name].get(col_name)
            if isinstance(arrow_array, pa.ListArray):
                df_dict[col_name] = self.list2ser(arrow_array)
            else:
                df_dict[col_name] = arrow_array.to_pandas()
        out_df = pd.DataFrame.from_dict(df_dict)
        return self.convert_to_tensor_frame(
            table_name=table_name,
            df=out_df,
            device=device,
            post_forward_only=False,
            col_name_hook=None,
        )

    @staticmethod
    def list2ser(array: pa.ListArray) -> pd.Series:
        """Convert a pyarrow list of array to a numpy tensor."""
        data = array.values.to_numpy(zero_copy_only=False)
        if len(data) == 0:
            return pd.Series([[] for _ in range(len(array))], dtype=object)

        offsets = array.offsets.to_numpy()
        # TODO(dong) We use for-loop because directly access array like
        # array[i] will segment fault. It is a bug in pyarrow.
        return pd.Series(
            [data[offsets[i]:offsets[i + 1]] for i in range(len(array))])

    def convert_to_tensor_frame(
        self,
        table_name: str,
        df: pd.DataFrame | dict[str, pa.Array] | dict[str, np.ndarray],
        *,
        device: torch.device | None = None,
        post_forward_only: bool = False,
        col_name_hook: Callable[[str], str] | None = None,
        vectorized: bool = False,
    ) -> TensorFrame:
        r"""Converts a DataFrame into a TensorFrame.

        When ``vectorized=True``, dispatches to :func:`_to_tf_vectorized`,
        which expects ``df`` to be a ``dict[str, np.ndarray]`` and does not
        support ``post_forward_only`` or ``col_name_hook``. Callers that
        already hold NumPy arrays may opt into this path; pandas and raw-arrow
        callers leave it False.

        Note:
            The order of columns within each stype must not change between
            calls, i.e., tf.feat_dict[stype] of the returned TensorFrame must
            be in the same order between calls. This is because subsequent
            fused encoders in TensorFrameEncoder assumes that the order
            of columns within a stype is consistent across calls so that it can
            efficiently encode columns in a batched manner.
        """
        if vectorized:
            tf = _to_tf_vectorized(
                df=cast(dict[str, np.ndarray], df),
                mappers=self[table_name],
                device=device,
            )
        else:
            tf = _to_tf(
                df=df,
                mappers=self[table_name],
                device=device,
                post_forward_only=post_forward_only,
                col_name_hook=col_name_hook,
            )

        # We store all null-encoded values in a special hidden data structure,
        # so that we can restore specific columns in `SubgraphCollector`, e.g.,
        # image columns.
        tf._unsupported_feat_dict: dict[str, Any] = {}
        for col_name, mapper in self[table_name].items():
            if mapper.stype != Stype.image:
                # We're only interested in including image columns in
                # _unsupported_feat_dict as we use them in XAI.
                continue

            tf_col_name = col_name
            if post_forward_only:
                assert col_name_hook is not None
                tf_col_name = col_name_hook(col_name)

            if tf_col_name in df:
                if isinstance(df, pd.DataFrame):
                    tf._unsupported_feat_dict[col_name] = \
                        df[tf_col_name].to_numpy()
                else:
                    val = df[tf_col_name]
                    if isinstance(val, np.ndarray):
                        tf._unsupported_feat_dict[col_name] = val
                    elif isinstance(val, pa.ListArray):
                        tf._unsupported_feat_dict[col_name] = \
                            self.list2ser(val).to_numpy()
                    else:
                        tf._unsupported_feat_dict[col_name] = \
                            val.to_numpy(zero_copy_only=False)

        return tf

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'    mappers={self.mappers},\n'
                f'    target={self.target},\n'
                f')')
