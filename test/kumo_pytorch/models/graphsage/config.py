"""Model-side feature and relation configuration; contains no sampling policy."""

from dataclasses import dataclass, field
from typing import Mapping

EdgeType = tuple[str, str, str]


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class NodeFeatureSpec:
    """Kumo graph field names, numeric widths, categorical vocabulary sizes.

    ``timestamp`` names an optional Raw(dtype='int64') field in epoch seconds.
    Numerical values must already be encoded by Kumo; widths include vectors.
    """

    numerical: Mapping[str, int] = field(default_factory=dict)
    categorical: Mapping[str, int] = field(default_factory=dict)
    timestamp: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "numerical", dict(self.numerical))
        object.__setattr__(self, "categorical", dict(self.categorical))
        for name, width in (*self.numerical.items(), *self.categorical.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("Feature names must be nonempty strings")
            _positive(width, f"Size of {name}")
        if set(self.numerical) & set(self.categorical):
            raise ValueError("A field cannot be both numerical and categorical")
        if self.timestamp is not None:
            if not isinstance(self.timestamp, str) or not self.timestamp:
                raise ValueError("timestamp must name a raw epoch-seconds field")
            if self.timestamp in self.numerical or self.timestamp in self.categorical:
                raise ValueError("The timestamp field must be separate from ordinary features")


@dataclass(frozen=True)
class GraphSAGEConfig:
    node_features: Mapping[str, NodeFeatureSpec]
    edge_types: tuple[EdgeType, ...]
    root_type: str
    channels: int = 64
    num_layers: int = 2
    out_channels: int = 1
    aggr: str = "mean"
    add_reverse_edges: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_features", dict(self.node_features))
        object.__setattr__(self, "edge_types", tuple(tuple(edge) for edge in self.edge_types))
        if not self.node_features or any(
            not isinstance(name, str) or not name or not isinstance(spec, NodeFeatureSpec)
            for name, spec in self.node_features.items()
        ):
            raise ValueError("node_features must map nonempty node names to NodeFeatureSpec")
        if self.root_type not in self.node_features:
            raise ValueError("root_type must be present in node_features")
        for name in ("channels", "num_layers", "out_channels"):
            _positive(getattr(self, name), name)
        if self.channels % 2 and any(spec.timestamp for spec in self.node_features.values()):
            raise ValueError("Temporal encoding requires an even channels value")
        if self.aggr not in ("mean", "sum", "max"):
            raise ValueError("aggr must be mean, sum or max")
        if type(self.add_reverse_edges) is not bool:
            raise TypeError("add_reverse_edges must be boolean")
        if not self.edge_types or any(
            len(edge) != 3 or any(not isinstance(part, str) or not part for part in edge)
            for edge in self.edge_types
        ):
            raise ValueError("edge_types must contain (source, relation, destination) triples")
        if len({edge[1] for edge in self.edge_types}) != len(self.edge_types):
            raise ValueError("Kumo relation names must be unique")
        for source, _, destination in self.edge_types:
            if source not in self.node_features or destination not in self.node_features:
                raise ValueError("Each relation endpoint must have a node feature specification")
        if self.add_reverse_edges:
            names = {edge[1] for edge in self.edge_types}
            if names & {"rev__" + name for name in names}:
                raise ValueError("Generated reverse relation names collide with configured relations")
        # RelBench only produces embeddings for destination types. Requiring an
        # incoming relation per type keeps all types available in deeper layers.
        destinations = {edge[2] for edge in self.message_edge_types}
        if destinations != set(self.node_features):
            raise ValueError("Every node type needs an incoming message relation; enable reverse edges or configure one")

    @property
    def message_edge_types(self) -> tuple[EdgeType, ...]:
        reverse = tuple((dst, "rev__" + rel, src) for src, rel, dst in self.edge_types)
        return self.edge_types + reverse if self.add_reverse_edges else self.edge_types
