# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small protocols shared by the reference models in this repository.

Nothing here is part of the data contract — see `kumo.pytorch.contract` for
that. These describe what the example training code expects of a model and a
batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, NamedTuple, Protocol, Self, TypeVar, runtime_checkable

import torch


@runtime_checkable
class ModelBatch(Protocol):
    """The only batch behavior required by the shared training code."""

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def token_count(self) -> int: ...


class ModelOutput(NamedTuple):
    """Architecture-neutral result returned by pretraining and task models.

    The ``NamedTuple`` and its dictionaries are visible to PyTorch's distributed
    tree traversal. Architecture-specific tensors remain explicit, named values
    rather than being forced into one artificial shape.
    """

    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    predictions: dict[str, torch.Tensor]
    representations: dict[str, torch.Tensor]


class ModelState(NamedTuple):
    """Context produced by an encoder and consumed by one or more objectives."""

    contextual: torch.Tensor
    mask: torch.Tensor
    representations: dict[str, torch.Tensor]
    auxiliary: dict[str, torch.Tensor]


class ObjectiveOutput(NamedTuple):
    loss: torch.Tensor
    predictions: dict[str, torch.Tensor]
    components: dict[str, torch.Tensor]


BatchT = TypeVar("BatchT", bound=ModelBatch)


@dataclass
class LabeledBatch(Generic[BatchT]):
    """Add task labels and optional tabular features to any architecture batch."""

    inputs: BatchT
    labels: torch.Tensor
    features: dict[str, torch.Tensor] = field(default_factory=dict)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> LabeledBatch[BatchT]:
        return LabeledBatch(
            inputs=self.inputs.to(device, non_blocking=non_blocking),
            labels=self.labels.to(device, non_blocking=non_blocking),
            features={
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.features.items()
            },
        )

    @property
    def batch_size(self) -> int:
        return self.inputs.batch_size

    @property
    def token_count(self) -> int:
        return self.inputs.token_count


def model_inputs(batch: Any) -> Any:
    """Return the architecture batch inside an optional supervised wrapper."""

    return batch.inputs if isinstance(batch, LabeledBatch) else batch
