# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import functools
import sys
from typing import Generic, Optional, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self  # isort: skip
else:
    from typing_extensions import Self  # isort: skip # type: ignore

from io_utils import logging
from kumo.connector import Connector

logger = logging.getLogger(__name__)
T = TypeVar("T")


def raise_post_materialization(func):
    r"""Decorator that blocks mutating methods after a `_HasData`-bearing
    object has been materialized. The exception message is part of the
    public contract and is asserted by `test/graph/test_graph.py`; do not
    change the wording.
    """
    @functools.wraps(func)
    def _disallow_post_materialization(self, *args, **kwargs):
        if getattr(self, '_data', None) is not None:
            raise Exception(
                f"'{type(self).__name__}' cannot be modified post "
                "materialization. Tried calling "
                f"'{type(self).__name__}.{func.__name__}'")
        return func(self, *args, **kwargs)

    return _disallow_post_materialization


class _HasData(Generic[T]):
    r"""Mixin providing the materialized-data slot (`_data`) and its
    accessor / introspection helpers (`has_data`, `data`, `clear_data`).
    Each concrete subclass owns its own `materialize(...)` implementation.
    """
    _data: Optional[T] = None

    @property
    def has_data(self) -> bool:
        return getattr(self, '_data', None) is not None

    @property
    def data(self) -> T:
        if not self.has_data:
            raise AttributeError(
                f"Cannot access data because {self.__class__.__name__} has "
                f"not been materialized yet. Please ensure that you run "
                f"'materialize()' before accessing data.")
        assert self._data is not None
        return self._data

    def clear_data(self) -> Self:
        self._data = None
        return self


class ConnectorMixin:
    _connector: Optional[Connector] = None

    @property
    def has_connector(self) -> bool:
        return getattr(self, '_connector', None) is not None

    @property
    def connector(self) -> Connector:
        if not self.has_connector:
            if hasattr(self, 'name'):
                name = f"'{self.__class__.__name__}' with name '{self.name}'"
            else:
                name = f"'{self.__class__.__name__}'"
            raise ValueError(f"{name} is not part of any 'kumo.Graph'")
        return self._connector

    @connector.setter
    def connector(self, connector: Optional[Connector]):
        self._connector = connector

    def __deepcopy__(self, memo):
        # Connectors hold live connection handles that are not deep-copyable.
        # Share the connector by reference; deep-copy everything else.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_connector':
                setattr(result, k, v)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result


class TableMixin:
    _table_name: Optional[str] = None

    @property
    def has_table(self) -> bool:
        return getattr(self, '_table_name', None) is not None

    @property
    def table_name(self) -> str:
        if not self.has_table:
            if hasattr(self, 'name'):
                name = f"'{self.__class__.__name__}' with name '{self.name}'"
            else:
                name = f"'{self.__class__.__name__}'"
            raise ValueError(f"{name} is not part of any 'kumo.Table'")
        return self._table_name

    @table_name.setter
    def table_name(self, table_name: Optional[str]):
        self._table_name = table_name
