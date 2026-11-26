# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recipe input engines selected by registered name or an importable provider."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from tfm.contracts.errors import UnsupportedCombinationError
from tfm.contracts.registries import Registry


class EngineFactory(Protocol):
    """Construct a pinned Kumo PyTorch Engine; register extra cleanup on resources.

    Engine.open/Plan owns engine-source lifetime. Additional backend resources
    must register cleanup here so they outlive plans and batch producers.
    """

    def __call__(self, resources: ExitStack, *, work_dir: Path, num_threads: int) -> Any: ...


@dataclass(frozen=True)
class SourceLoader:
    """Source-table engine provider; model encoding and batching remain shared.

    ``engine_factory`` returns an implementation of ``kumo.pytorch.api.Engine``.
    Its immutable provenance must identify the implementation/version/settings
    that can affect delivered tensors. Changing it invalidates checkpoint reuse.
    The Python reference is the sole built-in without an engine factory.
    """

    engine_factory: EngineFactory | None
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.provenance or any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in self.provenance.items()
        ):
            raise ValueError("Loader provenance requires nonempty string keys and values")
        if self.engine_factory is not None and not callable(self.engine_factory):
            raise TypeError("Loader engine_factory must be callable")
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


_LOADERS: Registry[SourceLoader] = Registry("loader")


def register_loader(name: str, provider: SourceLoader) -> SourceLoader:
    """Register a source-table provider before parsing a recipe configuration."""
    if not name or name.strip() != name or name.lower() != name or ":" in name:
        raise ValueError("Loader registration requires a nonempty lowercase short name")
    if not isinstance(provider, SourceLoader) or provider.engine_factory is None:
        raise TypeError("Register a SourceLoader with an engine_factory")
    return _LOADERS.register(name, provider)


def resolve_loader(name: str) -> SourceLoader:
    """Resolve built-ins, registrations, or ``package.module:provider`` objects."""
    provider = _LOADERS.get(name)
    if not isinstance(provider, SourceLoader) or (
        name != "python" and provider.engine_factory is None
    ):
        raise UnsupportedCombinationError(
            f"data.loader {name!r} must name a SourceLoader with an engine_factory"
        )
    return provider


def _diskgraph(resources: ExitStack, *, work_dir: Path, num_threads: int) -> Any:
    del work_dir
    from tfm.extras.kumo_pytorch import temporary_engine

    return temporary_engine(resources, num_threads=num_threads)


_LOADERS.register("python", SourceLoader(None, {"name": "python"}))
_LOADERS.register(
    "kumo_pytorch", SourceLoader(_diskgraph, {"name": "kumo_pytorch", "engine": "diskgraph"})
)
