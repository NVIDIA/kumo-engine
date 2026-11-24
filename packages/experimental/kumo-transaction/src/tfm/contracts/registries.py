# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One registry object per configurable slot, mapping names to what YAML selects."""

from __future__ import annotations

import importlib

from tfm.contracts.errors import UnsupportedCombinationError


class Registry[T]:
    """Names to entries for one slot; `kind` names the slot in errors, e.g. "adapter"."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str, entry: T) -> T:
        """Add `entry` under `name`; registering the same entry again does nothing."""
        held = self._entries.get(name, entry)
        if held is not entry:
            raise UnsupportedCombinationError(
                f"{self.kind} {name!r} is already registered as {held!r}; choose another name."
            )
        self._entries[name] = entry
        return entry

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def entries(self) -> dict[str, T]:
        return dict(self._entries)

    def get(self, name: str) -> object:
        """The entry registered as `name`, or the object a `"package.module:attribute"`
        path names, imported now. Callers check an imported object's type."""
        if ":" in name:
            return _import(name, self.kind)
        if name not in self._entries:
            known = ", ".join(self.names()) or "<none>"
            raise UnsupportedCombinationError(
                f"Unknown {self.kind} {name!r}. Registered {self.kind}s: {known}; or give "
                "an import path, 'package.module:attribute'."
            )
        return self._entries[name]


def _import(path: str, kind: str) -> object:
    module, _, attribute = path.partition(":")
    try:
        return getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError) as error:
        raise UnsupportedCombinationError(
            f"Cannot import {kind} {path!r}: {error}. Give 'package.module:attribute' of a "
            "module importable here and in worker processes (on sys.path, not a notebook)."
        ) from error


__all__ = ["Registry"]
