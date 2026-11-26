# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name registry for training backends (backend contracts).

A backend is a class with a `capabilities: BackendCapabilities` attribute, a
constructor taking `checkpoint_dir`, and the `Trainer.fit` method. Register one
with `@register_backend("name")`, or, for an optional extra that must not be
imported until selected, with `register_lazy_backend("name", "module:Class")`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tfm.contracts.capabilities import BackendCapabilities
from tfm.contracts.errors import UnsupportedCombinationError

if TYPE_CHECKING:
    from tfm.training.trainer import Trainer


@dataclass(frozen=True)
class _Entry:
    target: type | str
    """The class, or `module:attribute` to import on first use."""


_BACKENDS: dict[str, _Entry] = {}
_ALIASES: dict[str, str] = {"cpu": "eager"}


def register_backend(name: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        _check_declares_capabilities(cls, name)
        _BACKENDS[name] = _Entry(cls)
        return cls

    return decorator


def register_lazy_backend(name: str, target: str) -> None:
    """Register an optional backend by import path, so selecting it is the only import."""
    if ":" not in target:
        raise ValueError(f"Backend target must be 'module:attribute', got {target!r}.")
    _BACKENDS[name] = _Entry(target)


def advertised_backends() -> list[str]:
    return sorted(_BACKENDS)


def canonical_backend_name(name: str) -> str:
    requested = name.strip().lower()
    requested = _ALIASES.get(requested, requested)
    if requested not in _BACKENDS:
        known = ", ".join(advertised_backends()) or "<none>"
        raise UnsupportedCombinationError(
            f"Unknown training.backend {name!r}. Registered backends: {known}."
        )
    return requested


def backend_class(name: str) -> type:
    canonical = canonical_backend_name(name)
    entry = _BACKENDS[canonical]
    if isinstance(entry.target, type):
        return entry.target
    module_name, attribute = entry.target.split(":", 1)
    cls = getattr(importlib.import_module(module_name), attribute)
    _check_declares_capabilities(cls, canonical)
    return cls


def backend_capabilities(name: str) -> BackendCapabilities:
    capabilities = backend_class(name).capabilities  # type: ignore[attr-defined]
    assert isinstance(capabilities, BackendCapabilities)
    return capabilities


def build_trainer(name: str, *, checkpoint_dir: Path | None = None) -> Trainer:
    trainer: Any = backend_class(name)(checkpoint_dir=checkpoint_dir)
    return trainer


def check_backend_supports(
    capabilities: BackendCapabilities,
    *,
    backbone: str,
    packing: bool,
    device_type: str,
    world_size: int,
    layout: str | None = None,
    precision: str = "fp32",
) -> None:
    """Refuse a run the backend cannot execute, before any batch is built.

    `layout` may be omitted when the tokenizer is not built yet, so the rest
    can be checked before data preparation.
    """
    name = capabilities.name
    problems: list[str] = []
    if backbone not in capabilities.backbones:
        problems.append(
            f"model.backbone: {backbone} is not supported; it trains "
            f"{sorted(capabilities.backbones)}"
        )
    if layout is not None and layout not in capabilities.batch_layouts:
        problems.append(
            f"tokenizer layout {layout!r} is not supported; it consumes "
            f"{sorted(capabilities.batch_layouts)}"
        )
    if packing and not capabilities.segment_masks:
        problems.append(
            "training.packing needs segment masks, which it does not honor, so "
            "packed rows would attend across entities"
        )
    if capabilities.requires_cuda and device_type != "cuda":
        problems.append(f"it requires CUDA, and runtime.device resolved to {device_type}")
    if world_size > 1 and not capabilities.multi_process:
        problems.append(f"it is single-process, and the launcher set WORLD_SIZE={world_size}")
    if precision not in capabilities.precisions:
        problems.append(
            f"training.precision: {precision} is not supported; it computes in "
            f"{sorted(capabilities.precisions)}"
        )
    if problems:
        raise UnsupportedCombinationError(
            f"training.backend: {name} cannot run this configuration: " + "; ".join(problems) + "."
        )


def _check_declares_capabilities(cls: type, name: str) -> None:
    capabilities = getattr(cls, "capabilities", None)
    if not isinstance(capabilities, BackendCapabilities):
        raise UnsupportedCombinationError(
            f"Backend {name!r} ({cls.__name__}) must declare `capabilities: BackendCapabilities`."
        )
    if capabilities.name != name:
        # A subclass that forgets to redeclare would report its parent's name
        # in every run artifact.
        raise UnsupportedCombinationError(
            f"Backend {name!r} ({cls.__name__}) declares capabilities for {capabilities.name!r}."
        )


register_lazy_backend("eager", "tfm.training.trainer:EagerTrainer")
register_lazy_backend("ddp", "tfm.training.ddp:DdpTrainer")
register_lazy_backend("fsdp2", "tfm.extras.fsdp2:Fsdp2NextEventTrainer")
