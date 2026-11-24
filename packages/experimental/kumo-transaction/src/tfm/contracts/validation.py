# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled CPU validation records and ephemeral device execution views.

Only CPU constructors seal batches. Transfer plans preserve their proof without
rerunning tensor predicates. Version counters cannot certify untracked writes.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import torch
from torch import Tensor

from tfm.contracts.training import TrainableBatch


@runtime_checkable
class ExecutionLayout(Protocol):
    """Explicit field selection for an execution-only device transfer."""

    execution_fields: ClassVar[frozenset[str] | None]


@dataclass(frozen=True)
class ValidationMetadata:
    """Declared non-input state; excluded from mutation-layout recursion."""

    _metadata_layout: _Layout | None = field(
        default=None, init=False, repr=False, compare=False, metadata={"validation_cache": True}
    )
    cache_validation_layout: ClassVar[bool] = False
    execution_fields: ClassVar[frozenset[str] | None] = None


_ERROR = "Validated batch was mutated; rebuild and revalidate on CPU"


@dataclass(frozen=True, slots=True)
class _TensorState:
    value: Tensor
    version: int
    gradient: bool
    cpu: bool

    @classmethod
    def capture(cls, value: Tensor) -> _TensorState:
        if value.is_inference():
            raise ValueError(
                "Validated batches require versioned tensors; clone inference inputs inside "
                "torch.inference_mode(False) and torch.no_grad() before construction"
            )
        return cls(value, value._version, value.requires_grad, value.device.type == "cpu")


@dataclass(frozen=True, slots=True)
class _Binding:
    owner: Any
    key: Any
    expected: Any
    attribute: bool

    def unchanged(self) -> bool:
        owner = self.owner() if self.attribute else self.owner
        if owner is None:
            return False
        if self.attribute:
            return all(
                getattr(owner, key) is expected
                for key, expected in zip(self.key, self.expected, strict=True)
            )
        return owner[self.key] is self.expected


@dataclass(frozen=True, slots=True)
class _Container:
    value: Any
    keys: tuple[Any, ...] | int

    def unchanged(self) -> bool:
        return (
            tuple(self.value) == self.keys
            if isinstance(self.keys, tuple)
            else len(self.value) == self.keys
        )


@dataclass(frozen=True, slots=True)
class _Record:
    tensors: tuple[_TensorState, ...] = ()
    bindings: tuple[_Binding, ...] = ()
    containers: tuple[_Container, ...] = ()

    @classmethod
    def merge(cls, records: list[_Record]) -> _Record:
        tensors, bindings, containers = {}, {}, {}
        for record in records:
            tensors.update((id(x.value), x) for x in record.tensors)
            bindings.update(
                (
                    (
                        id(x.owner()) if x.attribute else id(x.owner),
                        None if x.attribute else x.key,
                        x.attribute,
                    ),
                    x,
                )
                for x in record.bindings
            )
            containers.update((id(x.value), x) for x in record.containers)
        return cls(tuple(tensors.values()), tuple(bindings.values()), tuple(containers.values()))

    def check(self) -> None:
        # No field discovery, tree reconstruction or tensor value reads here.
        for state in self.tensors:
            if state.value._version != state.version or state.value.requires_grad != state.gradient:
                raise ValueError(_ERROR)
        for container in self.containers:
            if not container.unchanged():
                raise ValueError(_ERROR)
        for binding in self.bindings:
            if not binding.unchanged():
                raise ValueError(_ERROR)


class _RecordAccumulator:
    """Collect construction guards in shared maps; freeze only the sealed root."""

    def __init__(self) -> None:
        self.tensors: dict[int, _TensorState] = {}
        self.bindings: dict[tuple[int, Any, bool], _Binding] = {}
        self.containers: dict[int, _Container] = {}

    def capture(self, value: Tensor) -> _TensorState:
        key = id(value)
        if key not in self.tensors:
            self.tensors[key] = _TensorState.capture(value)
        return self.tensors[key]

    def include(self, record: _Record) -> None:
        self.tensors.update((id(x.value), x) for x in record.tensors)
        self.bindings.update(
            (
                (
                    id(x.owner()) if x.attribute else id(x.owner),
                    None if x.attribute else x.key,
                    x.attribute,
                ),
                x,
            )
            for x in record.bindings
        )
        self.containers.update((id(x.value), x) for x in record.containers)

    def finish(self) -> _Record:
        return _Record(
            tuple(self.tensors.values()),
            tuple(self.bindings.values()),
            tuple(self.containers.values()),
        )


_EMPTY = _Record()


def require_cpu(value: Any) -> None:
    if isinstance(value, ValidatedBatch) and value._record is not None:
        value.assert_validated()
        if any(not state.cpu for state in value._record.tensors):
            raise ValueError(
                "Construct and validate graph batches on CPU, then use batch.to(device)"
            )
        return
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise ValueError(
                "Construct and validate graph batches on CPU, then use batch.to(device)"
            )
    elif is_dataclass(value) and not isinstance(value, type):
        for f in fields(value):
            if f.metadata.get("validation_cache", False):
                continue
            require_cpu(getattr(value, f.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_cpu(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            require_cpu(item)


@dataclass(frozen=True, slots=True)
class _Layout:
    kind: str
    value: Any
    record: _Record = _EMPTY
    children: tuple[tuple[Any, _Layout], ...] = ()
    movable: bool = False
    execution_fields: frozenset[str] | None = None

    def original(self) -> Any:
        return self.value() if self.kind in {"validated", "dataclass"} else self.value

    @classmethod
    def build(cls, value: Any, *, root: bool = False) -> _Layout:
        accumulator = _RecordAccumulator()
        node = cls._build(value, accumulator, root=root)
        return replace(node, record=accumulator.finish())

    @classmethod
    def _build(cls, value: Any, accumulator: _RecordAccumulator, *, root: bool = False) -> _Layout:
        if isinstance(value, Tensor):
            return cls("tensor", value, _Record((accumulator.capture(value),)), movable=True)
        if isinstance(value, ValidatedBatch) and not root:
            accumulator.include(value.validation_record)
            return value.validation_layout
        if is_dataclass(value) and not isinstance(value, type):
            cached = value._metadata_layout if isinstance(value, ValidationMetadata) else None
            if cached is not None:
                cached.record.check()
                accumulator.include(cached.record)
                return cached
            children = tuple(
                (f.name, cls._build(getattr(value, f.name), accumulator))
                for f in fields(value)
                if not f.metadata.get("validation_cache", False)
            )
            validated = isinstance(value, ValidatedBatch)
            bindings = (
                _Binding(
                    weakref.ref(value),
                    tuple(key for key, _ in children),
                    tuple(child.original() for _, child in children),
                    True,
                ),
            )
            local = _Record(bindings=bindings)
            accumulator.include(local)
            movable = any(child.movable for _, child in children)
            record = cls._construction_record(children, local, movable)
            node = cls(
                "validated" if validated else "dataclass",
                weakref.ref(value),
                record,
                children,
                movable,
                value.execution_fields if isinstance(value, ExecutionLayout) else None,
            )
            if isinstance(value, ValidationMetadata) and value.cache_validation_layout:
                object.__setattr__(value, "_metadata_layout", node)
            return node
        if isinstance(value, Mapping):
            children = tuple((key, cls._build(item, accumulator)) for key, item in value.items())
            mutable = not isinstance(value, MappingProxyType)
            bindings = (
                tuple(_Binding(value, key, child.original(), False) for key, child in children)
                if mutable
                else ()
            )
            containers = (_Container(value, tuple(value)),) if mutable else ()
            local = _Record(bindings=bindings, containers=containers)
            accumulator.include(local)
            movable = any(child.movable for _, child in children)
            return cls(
                "mapping",
                value,
                cls._construction_record(children, local, movable),
                children,
                movable,
            )
        if isinstance(value, (tuple, list)):
            children = tuple((key, cls._build(item, accumulator)) for key, item in enumerate(value))
            mutable = isinstance(value, list)
            bindings = (
                tuple(_Binding(value, key, child.original(), False) for key, child in children)
                if mutable
                else ()
            )
            containers = (_Container(value, len(value)),) if mutable else ()
            local = _Record(bindings=bindings, containers=containers)
            accumulator.include(local)
            movable = any(child.movable for _, child in children)
            return cls(
                "list" if mutable else "tuple",
                value,
                cls._construction_record(children, local, movable),
                children,
                movable,
            )
        return cls("constant", value)

    @staticmethod
    def _construction_record(
        children: tuple[tuple[Any, _Layout], ...], local: _Record, movable: bool
    ) -> _Record:
        # Transfer walks movable children; skipped execution fields are covered
        # by the sealed source's complete root record. Constant subtrees can be
        # reused without traversal, so retain their self-contained guards. No
        # identity dictionaries are rebuilt for these small metadata snapshots.
        if movable:
            return local
        records = [node.record for _, node in children] + [local]
        return _Record(
            tuple(state for record in records for state in record.tensors),
            tuple(binding for record in records for binding in record.bindings),
            tuple(container for record in records for container in record.containers),
        )

    def transfer(
        self,
        device: torch.device,
        non_blocking: bool,
        execution: bool,
        memo: dict[int, tuple[Any, _Layout]],
    ) -> tuple[Any, _Layout]:
        original = self.original()
        if not self.movable:
            return original, self
        key = id(original)
        if key in memo:
            return memo[key]
        if self.kind == "tensor":
            value = original.to(device, non_blocking=non_blocking)
            node = _Layout("tensor", value, _Record((_TensorState.capture(value),)), movable=True)
        else:
            moved = []
            for name, child in self.children:
                active = (
                    not execution or self.execution_fields is None or name in self.execution_fields
                )
                moved.append(
                    (
                        name,
                        child.transfer(device, non_blocking, execution, memo)
                        if active
                        else (child.original(), child),
                    )
                )
            children = tuple((name, node) for name, (_, node) in moved)
            bindings: tuple[_Binding, ...] = ()
            containers: tuple[_Container, ...] = ()
            if self.kind in {"validated", "dataclass"}:
                value = object.__new__(type(original))
                for name, (item, _) in moved:
                    object.__setattr__(value, name, item)
                bindings = (
                    _Binding(
                        weakref.ref(value),
                        tuple(name for name, _ in moved),
                        tuple(item for _, (item, _) in moved),
                        True,
                    ),
                )
            elif self.kind == "mapping":
                value = MappingProxyType({name: item for name, (item, _) in moved})
            else:
                items = [item for _, (item, _) in moved]
                value = items if self.kind == "list" else tuple(items)
                if self.kind == "list":
                    bindings = tuple(
                        _Binding(value, name, item, False) for name, (item, _) in moved
                    )
                    containers = (_Container(value, len(value)),)
            records = [child.record for _, child in children] + [
                _Record(bindings=bindings, containers=containers)
            ]
            if execution:
                records.append(self.record)  # Retain CPU source mutations, including copied inputs.
            record = _Record.merge(records)
            node = _Layout(
                self.kind,
                weakref.ref(value) if self.kind in {"validated", "dataclass"} else value,
                record,
                children,
                self.movable,
                self.execution_fields,
            )
            if self.kind == "validated":
                object.__setattr__(value, "_layout", node)
                object.__setattr__(value, "_record", record)
                object.__setattr__(value, "_execution_source", original if execution else None)
                object.__setattr__(value, "_execution_device", device if execution else None)
        memo[key] = value, node
        return value, node


def _constructor_args(value: Any) -> tuple[Any, ...]:
    return tuple(
        dict(item) if isinstance(item := getattr(value, f.name), Mapping) else item
        for f in fields(value)
        if f.init
    )


@dataclass(frozen=True)
class ValidatedBatch(ValidationMetadata, TrainableBatch):
    """Frozen CPU batches with compiled records and declared execution fields."""

    _layout: _Layout | None = field(
        default=None, init=False, repr=False, compare=False, metadata={"validation_cache": True}
    )
    _record: _Record | None = field(
        default=None, init=False, repr=False, compare=False, metadata={"validation_cache": True}
    )
    _execution_source: Self | None = field(
        default=None, init=False, repr=False, compare=False, metadata={"validation_cache": True}
    )
    _execution_device: torch.device | None = field(
        default=None, init=False, repr=False, compare=False, metadata={"validation_cache": True}
    )

    @property
    def validation_record(self) -> _Record:
        if self._record is None:
            raise ValueError("Batch validation has not completed")
        return self._record

    @property
    def validation_layout(self) -> _Layout:
        if self._layout is None:
            raise ValueError("Batch validation has not completed")
        return self._layout

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        self.assert_validated()
        require_cpu(self)
        return type(self), _constructor_args(self)

    def _seal(self) -> None:
        layout = _Layout.build(self, root=True)
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_record", layout.record)
        object.__setattr__(self, "_execution_source", None)
        object.__setattr__(self, "_execution_device", None)

    def assert_validated(self) -> None:
        """Check flat mutation records without recursive traversal or device reads."""
        self.validation_record.check()

    @torch.inference_mode(False)
    @torch.no_grad()
    def to(self, device: torch.device, *, non_blocking: bool = False) -> Self:
        """Complete device copy; same-device tensors may alias read-only inputs."""
        self.assert_validated()
        return self.validation_layout.transfer(torch.device(device), non_blocking, False, {})[0]

    @torch.inference_mode(False)
    @torch.no_grad()
    def for_execution(self, device: torch.device, *, non_blocking: bool = False) -> Self:
        """Move model inputs/targets; keep validation metadata and its guards on CPU.

        Views are ephemeral and never cached on the source. CPU source and copied
        device tensors remain read-only. A fully transferred non-CPU source is
        first copied completely to CPU, so views retain only CPU source ownership.
        Nonblocking copies require the normal
        PyTorch stream/lifetime rules. Inference contexts are restored on return.
        """
        self.assert_validated()
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if self._execution_source is not None:
            if self._execution_device == device:
                return self
            return self._execution_source.for_execution(device, non_blocking=non_blocking)
        if any(not state.cpu for state in self.validation_record.tensors):
            # Full .to() copies may have device-resident validation metadata.
            # Normalize synchronously before CPU-backed views/host guards reuse it.
            return self.to(torch.device("cpu")).for_execution(device, non_blocking=non_blocking)
        return self.validation_layout.transfer(device, non_blocking, True, {})[0]
