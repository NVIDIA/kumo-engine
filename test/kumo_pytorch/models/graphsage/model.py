"""Kumo feature encoders and seed head around RelBench's unchanged GraphSAGE."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ._relbench import HeteroGraphSAGE, HeteroTemporalEncoder
from . import _kumo
from .adapter import GraphBatch
from .config import GraphSAGEConfig, NodeFeatureSpec


class _FeatureEncoder(nn.Module):
    def __init__(self, spec: NodeFeatureSpec, channels: int):
        super().__init__()
        self.spec = spec
        width = sum(spec.numerical.values())
        # Validity bits distinguish missing values from valid encoded zeros.
        self.numerical = nn.Linear(2 * width, channels) if width else None
        # Extra last embedding represents missingness independently of vocab ID 0.
        self.categorical = nn.ModuleList(
            [nn.Embedding(cardinality + 1, channels) for cardinality in spec.categorical.values()]
        )
        self.base = nn.Parameter(torch.zeros(channels))

    def forward(self, features: dict, count: int) -> Tensor:
        encoded = self.base.unsqueeze(0).expand(count, -1)
        if self.numerical is not None:
            values = torch.cat([features[name]["values"] for name in self.spec.numerical], dim=-1)
            valid = torch.cat([features[name]["valid"] for name in self.spec.numerical], dim=-1)
            values = torch.where(valid, values, 0).to(self.base.dtype)
            encoded = encoded + self.numerical(torch.cat((values, valid.to(values.dtype)), dim=-1))
        for (name, cardinality), embedding in zip(self.spec.categorical.items(), self.categorical):
            field = features[name]
            ids = torch.where(field["valid"], field["values"], cardinality)
            encoded = encoded + embedding(ids)
        return encoded

    def forward_kumo(self, features, count: int) -> Tensor:
        """Encode native field shapes and missingness within the model."""
        encoded = self.base.unsqueeze(0).expand(count, -1)
        if self.numerical is not None:
            values, validity = [], []
            for name, width in self.spec.numerical.items():
                if name in features:
                    value, valid = _kumo.numerical(features[name], width)
                    # Preserve the original consumer's float32 input precision,
                    # including when model parameters use a different dtype.
                    value = torch.where(valid, value.to(torch.float32), 0.0)
                else:
                    value = self.base.new_empty((0, width), dtype=torch.float32)
                    valid = self.base.new_empty((0, width), dtype=torch.bool)
                values.append(value)
                validity.append(valid)
            values = torch.cat(values, dim=-1).to(self.base.dtype)
            valid = torch.cat(validity, dim=-1).to(values.dtype)
            encoded = encoded + self.numerical(torch.cat((values, valid), dim=-1))
        for (name, cardinality), embedding in zip(self.spec.categorical.items(), self.categorical):
            if name in features:
                field = features[name]
                indices = torch.where(field["valid"], field["values"], cardinality)
            else:
                indices = self.base.new_empty(0, dtype=torch.long)
            encoded = encoded + embedding(indices)
        return encoded


@dataclass
class GraphSAGEEncoding:
    """Graph representations without running the standalone prediction head."""

    seed_embeddings: Tensor
    node_embeddings: dict[str, Tensor]


@dataclass
class GraphSAGEOutput:
    logits: Tensor
    seed_embeddings: Tensor
    node_embeddings: dict[str, Tensor]


class GraphSAGEModel(nn.Module):
    """Predict only the explicitly indexed request roots, in Kumo request order.

    ``forward_kumo`` consumes Kumo batches directly. Validate on CPU with
    ``validate_kumo_batch`` before transfer, then use ``validate_inputs=False``.
    The legacy ``forward(GraphBatch)`` interface remains available.
    Numerical/categorical feature encoding here is a local implementation, not
    RelBench's TensorFrame/ResNet encoder. The GraphSAGE class is unchanged.
    """

    def __init__(self, config: GraphSAGEConfig):
        super().__init__()
        self.config = config
        # Internal names avoid PyTorch/PyG restrictions on user table names.
        self._node_keys = {name: f"node_{i}" for i, name in enumerate(config.node_features)}
        self._edge_keys = {
            edge: (self._node_keys[edge[0]], f"relation_{i}", self._node_keys[edge[2]])
            for i, edge in enumerate(config.message_edge_types)
        }
        self.encoders = nn.ModuleDict({
            self._node_keys[name]: _FeatureEncoder(spec, config.channels)
            for name, spec in config.node_features.items()
        })
        temporal_types = [self._node_keys[name] for name, spec in config.node_features.items() if spec.timestamp]
        self.temporal_encoder = HeteroTemporalEncoder(temporal_types, config.channels)
        self.gnn = HeteroGraphSAGE(
            list(self._node_keys.values()), list(self._edge_keys.values()),
            config.channels, aggr=config.aggr, num_layers=config.num_layers,
        )
        self.head = nn.Linear(config.channels, config.out_channels)

    def validate_kumo_batch(self, batch, *, input_name="graph") -> None:
        """Check CPU input identities, ownership, fields and temporal bounds.

        This does not modify or replace the batch. Forward handles root lookup,
        missingness, feature shapes and the configured message directions.
        """
        _kumo.inspect(batch, self.config, input_name, check_values=True, cpu=True)

    def encode_kumo(self, batch, *, input_name="graph", validate_inputs=True) -> GraphSAGEEncoding:
        """Encode a native Kumo batch, preserving its request and node ordering.

        After CPU validation and batch transfer, ``validate_inputs=False``
        avoids value checks and accelerator readbacks. Structural and device
        checks still apply. No fitting, filtering or sampling occurs here.
        """
        device = next(iter(self.encoders.values())).base.device
        nodes, edges, metadata = _kumo.inspect(
            batch, self.config, input_name, check_values=validate_inputs, device=device,
        )
        x_dict, time_dict, batch_dict = {}, {}, {}
        for name, spec in self.config.node_features.items():
            key = self._node_keys[name]
            node = nodes.get(name, {})
            fields = node.get("features", {})
            owners = node.get("batch")
            if owners is None:
                owners = torch.empty(0, dtype=torch.long, device=device)
            batch_dict[key] = owners
            x_dict[key] = self.encoders[key].forward_kumo(fields, owners.numel())
            if spec.timestamp:
                raw_time = fields.get(spec.timestamp)
                time_dict[key] = (raw_time["values"] if raw_time is not None
                                  else torch.empty(0, dtype=torch.long, device=device))
        if time_dict:
            for key, value in self.temporal_encoder(
                    _kumo.prediction_time(metadata, check_values=False), time_dict, batch_dict).items():
                x_dict[key] = x_dict[key] + value.to(x_dict[key].dtype)
        message_edges = {}
        for edge in self.config.edge_types:
            source, relation, destination = edge
            raw_edge = edges.get(relation)
            index = (raw_edge["index"].contiguous() if raw_edge is not None
                     else torch.empty((2, 0), dtype=torch.long, device=device))
            message_edges[self._edge_keys[edge]] = index
            if self.config.add_reverse_edges:
                reverse = (destination, f"rev__{relation}", source)
                message_edges[self._edge_keys[reverse]] = index.flip(0).contiguous()
        embeddings = self.gnn(x_dict, message_edges)
        node_embeddings = {name: embeddings[key] for name, key in self._node_keys.items()}
        root = nodes[self.config.root_type]
        seed_index = _kumo.root_indices(root["ids"], root["batch"], metadata["entity_ids"])
        return GraphSAGEEncoding(node_embeddings[self.config.root_type][seed_index], node_embeddings)

    def forward_kumo(self, batch, *, input_name="graph", validate_inputs=True) -> GraphSAGEOutput:
        """Return predictions from native Kumo graph fields in request order."""
        encoding = self.encode_kumo(batch, input_name=input_name, validate_inputs=validate_inputs)
        return GraphSAGEOutput(
            self.head(encoding.seed_embeddings), encoding.seed_embeddings, encoding.node_embeddings,
        )

    def forward(self, batch: GraphBatch) -> GraphSAGEOutput:
        encoding = self.encode(batch)
        return GraphSAGEOutput(
            self.head(encoding.seed_embeddings),
            encoding.seed_embeddings,
            encoding.node_embeddings,
        )

    def encode(self, batch: GraphBatch) -> GraphSAGEEncoding:
        """Return request-ordered seed embeddings without computing predictions."""
        if not isinstance(batch, GraphBatch):
            raise TypeError("Pass KumoGraphAdapter(config)(kumo_batch), followed by .to(device)")
        if batch.root_type != self.config.root_type:
            raise ValueError("Batch root_type does not match the model configuration")
        if set(batch.features) != set(self._node_keys) or set(batch.edge_index_dict) != set(self._edge_keys):
            raise ValueError("Batch node/relation types do not match the model configuration")
        if set(batch.batch_dict) != set(self._node_keys):
            raise ValueError("Batch ownership node types do not match the model configuration")
        temporal_types = {name for name, spec in self.config.node_features.items() if spec.timestamp}
        if set(batch.time_dict) != temporal_types:
            raise ValueError("Batch temporal node types do not match the model configuration")
        first_encoder = self.encoders[next(iter(self._node_keys.values()))]
        if batch.seed_index.device != first_encoder.base.device:
            raise ValueError("Move the GraphBatch and model to the same device")
        x_dict = {
            key: self.encoders[key](batch.features[name], batch.batch_dict[name].numel())
            for name, key in self._node_keys.items()
        }
        if batch.time_dict:
            time_dict = {self._node_keys[name]: value for name, value in batch.time_dict.items()}
            batch_dict = {self._node_keys[name]: value for name, value in batch.batch_dict.items()}
            for key, value in self.temporal_encoder(batch.seed_time, time_dict, batch_dict).items():
                x_dict[key] = x_dict[key] + value.to(x_dict[key].dtype)
        edges = {self._edge_keys[edge]: index for edge, index in batch.edge_index_dict.items()}
        embeddings = self.gnn(x_dict, edges)
        node_embeddings = {name: embeddings[key] for name, key in self._node_keys.items()}
        seeds = node_embeddings[batch.root_type][batch.seed_index]
        return GraphSAGEEncoding(seeds, node_embeddings)
