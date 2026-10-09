# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyG configuration registration owned by the portable runtime."""

import contextlib
import dataclasses
import inspect
from collections.abc import Callable, Iterator
from typing import Any

import pydantic
from torch_geometric.config_store import register as pyg_register
from torch_geometric.config_store import to_dataclass as pyg_to_dataclass


@contextlib.contextmanager
def _field_signature(cls: Any) -> Iterator[None]:
    if not dataclasses.is_dataclass(cls):
        yield
        return
    parameters = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    for value in dataclasses.fields(cls):
        default = inspect.Parameter.empty
        if value.default is not dataclasses.MISSING:
            default = value.default
        parameters.append(
            inspect.Parameter(
                value.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=value.type,
            ))
    initializer = cls.__init__
    previous = getattr(initializer, "__signature__", None)
    initializer.__signature__ = inspect.Signature(parameters)
    try:
        yield
    finally:
        if previous is None:
            del initializer.__signature__
        else:
            initializer.__signature__ = previous


def register(
    cls: Any | None = None,
    data_cls: Any | None = None,
    group: str | None = None,
    **kwargs: Any,
) -> Any | Callable:
    if cls is None:

        def bounded_register(value: Any) -> Any:
            register(cls=value, data_cls=data_cls, group=group, **kwargs)
            return value

        return bounded_register

    if data_cls is None:
        with _field_signature(cls):
            data_cls = pyg_to_dataclass(cls, **kwargs)
        data_cls = pydantic.dataclasses.dataclass(data_cls)
        return pyg_register(cls, data_cls, group)
    return pyg_register(cls, data_cls, group, **kwargs)
