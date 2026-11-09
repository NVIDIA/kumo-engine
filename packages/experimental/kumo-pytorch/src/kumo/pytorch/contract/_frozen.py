# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A read-only mapping that survives pickling.

Contract objects are frozen, and their mappings must be too. `types.MappingProxyType`
is read-only but cannot be pickled, so a contract or fitted state could not reach a
DataLoader worker or another process. `FrozenMapping` holds its own copy and pickles
as that copy.
"""

from __future__ import annotations

from collections.abc import ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from typing import Any, TypeVar

K = TypeVar("K")
V = TypeVar("V")

__all__ = ["FrozenMapping"]


class FrozenMapping(Mapping[K, V]):
    """An immutable copy of a mapping; equal to any mapping with the same items."""

    __slots__ = ("_items",)

    _items: dict[K, V]

    def __init__(self, items: Mapping[K, V] | Iterable[tuple[K, V]] = ()) -> None:
        object.__setattr__(self, "_items", dict(items))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __reduce__(self) -> tuple[type[FrozenMapping[K, V]], tuple[dict[K, V]]]:
        return type(self), (self._items,)

    def __getitem__(self, key: K) -> V:
        return self._items[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def get(self, key: K, default: Any = None) -> Any:
        return self._items.get(key, default)

    def keys(self) -> KeysView[K]:
        return self._items.keys()

    def items(self) -> ItemsView[K, V]:
        return self._items.items()

    def values(self) -> ValuesView[V]:
        return self._items.values()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FrozenMapping):
            return self._items == other._items
        if isinstance(other, Mapping):
            return self._items == dict(other.items())
        return NotImplemented

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._items!r})"
