"""Dimensions for a graph, token-history and aggregate fusion model."""

from dataclasses import dataclass
from typing import Mapping

from ..graphsage import GraphSAGEConfig
from ..nuformer import NuFormerConfig


@dataclass(frozen=True)
class FusionConfig:
    graph: GraphSAGEConfig
    sequence: NuFormerConfig
    aggregate_fields: Mapping[str, int]
    aggregate_dim: int = 32
    fusion_dim: int = 64
    out_channels: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.graph, GraphSAGEConfig) or not isinstance(self.sequence, NuFormerConfig):
            raise TypeError("graph and sequence must be GraphSAGEConfig and NuFormerConfig")
        object.__setattr__(self, "aggregate_fields", dict(self.aggregate_fields))
        if not self.aggregate_fields:
            raise ValueError("aggregate_fields must select at least one field")
        for name, width in self.aggregate_fields.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Aggregate field names must be nonempty strings")
            if type(width) is not int or width < 1:
                raise ValueError("Aggregate field widths must be positive integers")
        for name in ("aggregate_dim", "fusion_dim", "out_channels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def aggregate_width(self) -> int:
        return sum(self.aggregate_fields.values())
