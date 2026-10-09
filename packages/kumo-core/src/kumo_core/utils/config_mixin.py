# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static configuration conversion for portable training objects."""

from __future__ import annotations

import inspect
import sys
from dataclasses import asdict, fields, is_dataclass
from typing import IO, Any

from torch_geometric.config_store import (
    class_from_dataclass,
    dataclass_from_class,
)


def _to_config(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return [_to_config(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_config(item) for key, item in value.items()}
    config = getattr(value, "config", None)
    return config() if callable(config) else value


def _from_config(value: Any) -> Any:
    if isinstance(value, list | tuple):
        return [_from_config(item) for item in value]
    if isinstance(value, dict):
        return {key: _from_config(item) for key, item in value.items()}
    if not is_dataclass(value):
        return value
    owner = class_from_dataclass(value.__class__)
    if owner is None:
        return value
    from_config = getattr(owner, "from_config", None)
    if callable(from_config):
        return from_config(value)
    values = {
        field.name: _from_config(getattr(value, field.name))
        for field in fields(value) if field.init
    }
    return owner(**values)


class PortableConfigMixin:
    def config(self, data_cls: Any | None = None) -> Any:
        data_cls = data_cls or dataclass_from_class(self.__class__)
        if data_cls is None:
            raise ValueError(
                f"No configuration registered for {self.__class__.__name__}")
        values = {
            field.name: _to_config(getattr(self, field.name))
            for field in fields(data_cls) if hasattr(self, field.name)
        }
        values = self.customize_config(values)
        return data_cls(**values)

    @classmethod
    def from_config(cls, config: Any, **overrides: Any) -> Any:
        values = {
            field.name: _from_config(getattr(config, field.name))
            for field in fields(config)
        }
        values = cls.customize_from_config(values)
        parameters = set(inspect.signature(cls.__init__).parameters)
        parameters.discard("self")
        constructor_values = {
            name: value
            for name, value in values.items() if name in parameters
        }
        constructor_values.update(overrides)
        instance = cls(**constructor_values)
        for name, value in values.items():
            if name not in parameters:
                setattr(instance, name, overrides.get(name, value))
        return instance

    def customize_config(self, values: dict[str, Any]) -> dict[str, Any]:
        return values

    def diff_config(self, other: Any, file: IO | None = None) -> None:
        """Print a stable diff between two portable configurations."""
        import difflib
        import pprint

        file = file if file is not None else sys.stdout
        config = asdict(self.config())
        other_config = asdict(other.config())
        if config == other_config:
            print("No differences found.", file=file)
            return
        diff = difflib.unified_diff(
            pprint.pformat(config).splitlines(),
            pprint.pformat(other_config).splitlines(),
            n=6,
        )
        print("\n" + "\n".join(diff), file=file)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.config() == other.config()

    def __hash__(self) -> int:
        return super().__hash__()

    @classmethod
    def customize_from_config(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        return values
