# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch_frame
import torch_geometric
from torch import Tensor
from torch_frame import TensorFrame
from torch_geometric.data.feature_store import _FieldStatus
from torch_geometric.utils.mixin import CastMixin

import logging
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.types.features import FeatureStoreData

logger = logging.getLogger(__name__)

ATTR_NAMES = {
    'tf',  # `TensorFrame` input features
    'time',  # Create date
    'end_time',  # End date
    'pkey',  # Primary key
    'df',  # Raw column data
}

TRAIN_ATTR_NAMES = {
    'tf',  # `TensorFrame` input features
    'fkey',  # Foreign keys
    'train_mask',  # Training masks
    'val_mask',  # Validation masks
    'test_mask',  # Test masks
    'y',  # Targets
    'time',  # Anchor time
    'weight',  # Custom training weights.
}


@dataclass
class FeatureStoreTableSchema(CastMixin):
    mapper_dict: dict[str, TensorMapper]
    pkey_name: str | None = None
    time_col_name: str | None = None
    end_time_col_name: str | None = None


@dataclass
class FeatureStoreTrainingTableSchema(CastMixin):
    table_name: str
    fkeys: list[tuple[str, str]]
    train_mask_name: str
    val_mask_name: str
    test_mask_name: str | None = None
    mapper_dict: dict[str, TensorMapper] = field(default_factory=dict)
    target_mapper: tuple[str, TensorMapper] | None = None
    time_col_name: str | None = None
    weight_col_name: str | None = None


@dataclass
class FeatureStoreAttr(CastMixin):
    table_name: str | None | _FieldStatus = _FieldStatus.UNSET
    attr_name: str | None | _FieldStatus = _FieldStatus.UNSET
    index: Tensor | slice | None | _FieldStatus = _FieldStatus.UNSET

    def is_set(self, key: str) -> bool:
        return getattr(self, key) != _FieldStatus.UNSET

    def is_fully_specified(self) -> bool:
        return all(self.is_set(key) for key in self.__dataclass_fields__)

    @property
    def group_name(self) -> str | None:
        assert not isinstance(self.table_name, _FieldStatus)
        return self.table_name


class FeatureStore(torch_geometric.data.FeatureStore, ABC):  # type: ignore
    @abstractmethod
    def _get_tensor(self, attr: FeatureStoreAttr) -> FeatureStoreData | None:
        r"""Obtains tensor data from the feature store."""

    @abstractmethod
    def _get_tensor_size(
        self,
        attr: FeatureStoreAttr,
    ) -> tuple[int, ...] | None:
        r"""Obtains tensor data sizes from the feature store."""

    @property
    @abstractmethod
    def table_names(self) -> list[str]:
        r"""Returns all graph table names registered to the feature store."""

    @property
    @abstractmethod
    def train_table_name(self) -> str | None:
        r"""Returns the training table name registered to the feature store."""

    @property
    @abstractmethod
    def entity_table_names(self) -> list[str]:
        r"""Returns the list of entity table names the training table is
        pointing to.
        """

    @property
    @abstractmethod
    def train_fkey_names(self) -> list[str]:
        r"""Returns the foreign key names of the training table."""

    @abstractmethod
    def get_pkey_name(self, table_name: str) -> str | None:
        r"""Returns the primary key name of :obj:`table_name` (in case it
        exists).
        """

    @abstractmethod
    def get_time_col_name(self, table_name: str) -> str | None:
        r"""Returns the time column name of :obj:`table_name` (in case it
        exists).
        """

    @abstractmethod
    def get_end_time_col_name(self, table_name: str) -> str | None:
        r"""Returns the end time column name of :obj:`table_name` (in case
        it exists).
        """

    @abstractmethod
    def get_tensor_mapper(
        self,
        table_name: str,
        col_name: str,
    ) -> TensorMapper:
        r"""Returns the tensor mapper according to a given :obj:`table_name`
        and :obj:`col_name`.
        """

    @abstractmethod
    def get_target_tensor_mapper(self) -> TensorMapper | None:
        r"""Returns the tensor mapper of the target column (if present)."""

    # Implementation ##########################################################

    def __init__(self) -> None:
        super().__init__(tensor_attr_cls=FeatureStoreAttr)
        self._batch_size: int | None = None
        self._min_batch_size: int | None = None

    def get_all_tensor_attrs(self) -> list[FeatureStoreAttr]:
        r"""Returns all tensor attributes queried during feature fetching."""
        attrs: list[FeatureStoreAttr] = []

        # NOTE All other attributes shall not be used during sampling.
        for table_name in self.table_names:
            attrs.append(FeatureStoreAttr(table_name, 'tf'))
        for table_name in self.table_names:
            if self.get_pkey_name(table_name) is not None:
                attrs.append(FeatureStoreAttr(table_name, 'pkey'))
        for table_name in self.table_names:
            if self.get_time_col_name(table_name) is not None:
                attrs.append(FeatureStoreAttr(table_name, 'time'))

        attrs.append(FeatureStoreAttr(self.train_table_name, 'tf'))
        attrs.append(FeatureStoreAttr(self.train_table_name, 'fkey'))
        if self.has_train_table_time_col:
            attrs.append(FeatureStoreAttr(self.train_table_name, 'time'))
        if self.has_train_table_target_col:
            attrs.append(FeatureStoreAttr(self.train_table_name, 'y'))
        if self.has_train_table_weight_col:
            attrs.append(FeatureStoreAttr(self.train_table_name, 'weight'))

        return attrs

    def _validate_query(self, attr: FeatureStoreAttr) -> None:
        if not attr.is_fully_specified():
            raise KeyError(f"The input '{attr}' is not fully specified")

        # Ensure that only mini-batch data is accessed within worker threads:
        if torch.utils.data.get_worker_info() is not None:
            assert attr.attr_name in {
                'tf', 'pkey', 'time', 'fkey', 'y', 'weight'
            }, attr.attr_name

        # Ensure that `table_name` and `attr_name` are valid:
        if attr.table_name in self.table_names:
            is_valid = attr.attr_name in ATTR_NAMES
        elif attr.table_name == self.train_table_name:
            is_valid = attr.attr_name in TRAIN_ATTR_NAMES
        else:
            is_valid = False

        if not is_valid:
            raise KeyError(f"Attribute '{attr.attr_name}' of table "
                           f"'{attr.table_name}' does not exist")

        # We only allow full querying of data for the training table or
        # time columns in tables:
        if (attr.table_name != self.train_table_name
                and attr.attr_name not in {'time', 'end_time', 'pkey'}):
            fetch_all = attr.index is None
            fetch_all |= (isinstance(attr.index, slice)
                          and attr.index.start is None
                          and attr.index.stop is None
                          and attr.index.step is None)
            if self._batch_size is None and fetch_all:
                raise KeyError(f"Can only fetch a mini-batch of data for a "
                               f"non-training table (got '{attr.table_name}')")

    def get_tensor(self, *args: Any, **kwargs: Any) -> FeatureStoreData | None:
        r"""Obtains tensor data from the feature store."""
        attr = self._tensor_attr_cls.cast(*args, **kwargs)
        self._validate_query(attr)

        if self._batch_size is None:
            return self._get_tensor(attr)

        batches: list[Tensor] | list[TensorFrame] | list[np.ndarray] = []
        index = self._get_index(attr)

        while True:
            try:
                for start in range(0, len(index), self._batch_size):
                    end = min(start + self._batch_size, len(index))
                    attr = FeatureStoreAttr(
                        table_name=attr.table_name,
                        attr_name=attr.attr_name,
                        index=index[start:end],
                    )
                    batch = self._get_tensor(attr)
                    assert batch is not None
                    batches.append(batch)
                break
            except Exception as e:
                if (self._min_batch_size is not None
                        and self._batch_size > self._min_batch_size
                        and self._batch_size > 1):
                    self._batch_size = max(self._batch_size // 2, 1)
                    logger.info(f"Retrying feature fetching with a batch size "
                                f"of {self._batch_size}")
                else:
                    raise e

        if len(batches) == 0:
            return self._get_tensor(attr)

        elem = batches[0]
        if isinstance(elem, Tensor):
            return torch.cat(batches)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batches, dim=0)
        elif isinstance(elem, np.ndarray):
            return np.concatenate(batches)
        elif isinstance(elem, pd.DataFrame):
            return pd.concat(batches, axis=0).reset_index(drop=True)
        else:  # pragma: no cover
            raise ValueError(f"batch_mode doesn't support type '{type(elem)}'")

    def multi_get_tensor(
        self,
        attrs: list[FeatureStoreAttr | dict[str, Any]],
    ) -> list[FeatureStoreData | None]:
        r"""Obtains a list of tensor data from the feature store."""
        _attrs = [self._tensor_attr_cls.cast(attr) for attr in attrs]
        for attr in _attrs:
            self._validate_query(attr)
        # NOTE We implicitly assume here that all attributes which point to the
        # same table reference the same indices.
        return self._multi_get_tensor(_attrs)

    def _multi_get_tensor(
        self,
        attrs: list[FeatureStoreAttr],
    ) -> list[FeatureStoreData | None]:
        r"""Obtains a list of tensor data from the feature store."""
        return [self._get_tensor(attr) for attr in attrs]

    @property
    def time_col_dict(self) -> dict[str, str]:
        time_col_dict: dict[str, str] = {}
        for table_name in self.table_names:
            time_col_name = self.get_time_col_name(table_name)
            if time_col_name is not None:
                time_col_dict[table_name] = time_col_name
        return time_col_dict

    @property
    @abstractmethod
    def has_train_table_test_mask(self) -> bool:
        r"""Returns whether the training table has a test mask."""

    @property
    @abstractmethod
    def has_train_table_target_col(self) -> bool:
        r"""Returns whether the training table has a target column."""

    @property
    @abstractmethod
    def has_train_table_weight_col(self) -> bool:
        r"""Returns whether the training table has a weight column."""

    @abstractmethod
    def get_node_id(self, table_name: str, pkey: np.ndarray) -> Tensor:
        r"""Returns the node ID given primary keys.

        Args:
            table_name: The table name.
            pkey: The primary keys to receive node IDs for.
        """

    @property
    def has_train_table_time_col(self) -> bool:
        r"""Returns whether the training table has a time column (*i.e.*
        whether the task is temporal).
        """
        if self.train_table_name is None:
            return False
        return self.get_time_col_name(self.train_table_name) is not None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tables={len(self.table_names)})'

    def _get_index(self, attr: FeatureStoreAttr) -> Tensor:
        # TODO: Ensure types are consistent with FeatureStoreAttr.index.
        if isinstance(attr.index, Tensor):
            return attr.index
        if isinstance(attr.index, np.ndarray):
            return torch.from_numpy(attr.index)
        elif isinstance(attr.index, int):
            return torch.tensor([attr.index])
        elif isinstance(attr.index, tuple | list):
            return torch.tensor(attr.index)
        elif attr.index is None:
            size = self._get_tensor_size(attr)
            assert size is not None
            return torch.arange(size[0])
        elif isinstance(attr.index, slice):
            size = self._get_tensor_size(attr)
            assert size is not None
            start = attr.index.start or 0
            stop = attr.index.stop or size[0]
            step = attr.index.step or 1
            return torch.arange(start, stop, step)

        raise ValueError(f"Unsupported index type: {type(attr.index)}")

    @contextmanager
    def batch_mode(self, batch_size: int) -> Generator[None, None, None]:
        """Context manager to fetch data in chunks.

        Args:
            batch_size: The size of each batch to fetch.
        """
        self._batch_size = batch_size
        yield
        self._batch_size = None

    @contextmanager
    def safe_batch_mode(
        self,
        batch_size: int,
        min_batch_size: int = 1024,
    ) -> Generator[None, None, None]:
        """Context manager to fetch data in chunks.

        If the specified batch size is too large and exceeds grpc message size
        (~1GB), will retry by dividing batch size by half until it succeeds.

        Args:
            batch_size: The size of each batch to fetch.
            min_batch_size: The minimum batch size to try before aborting.
        """
        self._batch_size = batch_size
        self._min_batch_size = min_batch_size
        yield
        self._batch_size = None
        self._min_batch_size = None

    # Not supported ###########################################################

    def _put_tensor(
        self,
        tensor: FeatureStoreData,
        attr: FeatureStoreAttr,
    ) -> bool:
        raise NotImplementedError

    def _remove_tensor(self, attr: FeatureStoreAttr) -> bool:
        raise NotImplementedError
