# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, explicit LRP on captured heterogeneous message-passing activations.

The reference profile attributes encoded head outputs to encoded node inputs.
Bias and stabilizer relevance are reported separately; walks are not causal.
The gamma rule enhances positive source contributions z, including when input
features are signed; it is not described as positive-weight-only propagation.
No backward hooks, model mutations, resampling, or gradient reinterpretation.
"""
from dataclasses import dataclass
import math

import torch
from torch_geometric.nn.aggr import MeanAggregation, MultiAggregation, SumAggregation

from kumo_core.modeling.gnn.hetero import (
    HeteroGNN, HeteroLinear, HeteroIdentity, ForeignKeyToPrimaryKeyConv,
    PrimaryKeyToForeignKeyConv,
)
from kumo_core.modeling.heads.node_prediction import NodePredHead


class UnsupportedWalkAttribution(ValueError):
    """The complete computation is outside the qualified rule profile."""


@dataclass(frozen=True)
class WalkAttributionConfig:
    gamma: float = 0.0
    epsilon: float = 1e-9
    max_walks: int = 10000
    top_k: int = 20
    output_index: int = 0

    def __post_init__(self):
        for name in ('gamma', 'epsilon'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.epsilon == 0:
            raise ValueError('epsilon must be positive')
        for name in ('max_walks', 'top_k', 'output_index'):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == 'output_index' else 1):
                raise ValueError(f'Invalid {name}')


def _copy(x):
    return x.detach().clone()


def _linear(lin):
    return _copy(lin.weight), None if lin.bias is None else _copy(lin.bias)


class WalkCapture:
    """One scoring forward per context; call result before releasing the trace.

    Capability validation precedes hook registration. The caller owns failure
    isolation and source identities. Relevance is conditional on the captured
    encoded features and actual sampled graph, including layer trimming.
    """
    def __init__(self, module, config=None):
        self.module = module
        self.config = config or WalkAttributionConfig()
        self.handles = []
        self.layers = []
        self.relations = {}
        self.roots = {}
        self.output = None
        self.failure = None
        self.entity = module.entity_table_name

    def validate(self):
        model, head = self.module.model, self.module.head
        if type(model) is not HeteroGNN or type(head) is not NodePredHead:
            raise UnsupportedWalkAttribution('Requires HeteroGNN and NodePredHead')
        if any(m.training for m in self.module.modules()):
            raise UnsupportedWalkAttribution('Attribution requires evaluation mode')
        for m in (model, head):
            if m.norm is not None or m.act not in (None, 'relu'):
                raise UnsupportedWalkAttribution('Reference profile requires ReLU/identity and no normalization')
        if head.prediction_time_encodings or head.param_keys:
            raise UnsupportedWalkAttribution('Time/distribution readout requires separate rules')
        aggr = model.aggr
        if isinstance(aggr, list) and len(aggr) == 1:
            aggr = aggr[0]
        if aggr not in ('sum', 'mean'):
            raise UnsupportedWalkAttribution('Reference profile requires sum or mean aggregation')
        for post in list(model.post_lins) + list(model.post_convs):
            if any(type(op) not in (torch.nn.ReLU, torch.nn.Dropout)
                   for ops in post.layer_dict.values() for op in ops):
                raise UnsupportedWalkAttribution('Unregistered post-layer operator')
        if head.mlp.act is not None and type(head.mlp.act) is not torch.nn.ReLU:
            raise UnsupportedWalkAttribution('Unregistered readout activation')
        if any(type(op) is not torch.nn.Identity for op in head.mlp.norms):
            raise UnsupportedWalkAttribution('Unregistered readout normalization')
        if any(type(root) not in (HeteroLinear, HeteroIdentity)
               for root in model.root_skips):
            raise UnsupportedWalkAttribution('Unregistered root operator')
        if any(type(conv) not in (ForeignKeyToPrimaryKeyConv, PrimaryKeyToForeignKeyConv)
               for layer in model.hetero_convs for conv in layer.values()):
            raise UnsupportedWalkAttribution('Unregistered relation operator')
        expected_aggregation = SumAggregation if aggr == 'sum' else MeanAggregation
        for layer in model.hetero_convs:
            for conv in layer.values():
                if type(conv) is not ForeignKeyToPrimaryKeyConv:
                    continue
                actual = conv.aggr
                if type(actual) is MultiAggregation:
                    if actual.mode != 'cat' or len(actual.aggrs) != 1:
                        raise UnsupportedWalkAttribution('Unregistered aggregation combination')
                    actual = actual.aggrs[0]
                if type(actual) is not expected_aggregation:
                    raise UnsupportedWalkAttribution('Aggregation differs from qualified profile')
        self.aggr = aggr

    def __enter__(self):
        self.validate()
        model, head = self.module.model, self.module.head
        self.layers.clear()
        self.output = None
        self.relations.clear()
        self.roots.clear()
        self.failure = None
        def hook(m, fn):
            def guarded(module, args, output):
                if self.failure is not None:
                    return
                try:
                    fn(module, args, output)
                except Exception as error:
                    # Explanation failure cannot replace a completed model output.
                    self.failure = type(error).__name__
            self.handles.append(m.register_forward_hook(guarded))
        try:
            for lin in model.lins:
                hook(lin, self._pre)
            for index, convs in enumerate(model.hetero_convs):
                self.relations[index] = []
                for relation, conv in convs.items():
                    hook(conv, lambda m, a, o, i=index, r=relation: self._relation(i, r, m, a))
                hook(model.root_skips[index], lambda m, a, o, i=index: self._root(i, m, a))
                hook(model.post_convs[index], lambda m, a, o, i=index: self._finish_layer(i))
            for lin in head.mlp.lins:
                hook(lin, self._head_linear)
            hook(head, self._head_output)
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _pre(self, module, args, output):
        self.layers.append({'root': {k: (_copy(v), *_linear(module.lin_dict[k])) for k, v in args[0].items()}, 'relations': []})

    def _relation(self, index, relation, module, args):
        x, dst, edge = args[:3]
        self.relations[index].append((tuple(relation), _copy(x), dst.size(0), _copy(edge), *_linear(module.lin), isinstance(module, ForeignKeyToPrimaryKeyConv)))

    def _root(self, index, module, args):
        self.roots[index] = {}
        for k, x in args[0].items():
            w, b = _linear(module.lin_dict[k]) if isinstance(module, HeteroLinear) else (torch.eye(x.size(1), device=x.device, dtype=x.dtype), None)
            self.roots[index][k] = (_copy(x), w, b)

    def _finish_layer(self, index):
        self.layers.append({'root': self.roots.pop(index), 'relations': self.relations.pop(index)})

    def _head_linear(self, module, args, output):
        self.layers.append({'root': {self.entity: (_copy(args[0]), *_linear(module))}, 'relations': []})

    def _head_output(self, module, args, output):
        if self.output is not None:
            raise ValueError('WalkCapture supports one forward per context')
        self.output = _copy(output)

    def _incoming(self, layer, table, row, remaining):
        routes, biases = [], []
        if table in layer['root']:
            if remaining < 1:
                return None
            x, weight, bias = layer['root'][table]
            routes.append((table, row, {'kind': 'self', 'table': table, 'occurrence': row}, x[row, :, None] * weight.T))
            if bias is not None:
                biases.append(bias)
        for relation, x, n_dst, edges, weight, bias, aggregate in layer['relations']:
            if relation[2] != table:
                continue
            positions = (edges[1] == row).nonzero().flatten()
            if len(routes) + positions.numel() > remaining:
                return None
            if not aggregate and positions.numel() > 1:
                raise UnsupportedWalkAttribution('PK gather has multiple incoming occurrences')
            divisor = max(positions.numel(), 1) if aggregate and self.aggr == 'mean' else 1
            for pos in positions.tolist():
                src = int(edges[0, pos])
                routes.append((relation[0], src, {'kind': 'relation', 'type': list(relation), 'edge_occurrence': pos, 'source_occurrence': src, 'destination_occurrence': row}, x[src, :, None] * weight.T / divisor))
            if bias is not None:
                biases.append(bias)
        return routes, biases

    def result(self, seed_index=0):
        if self.failure is not None:
            raise ValueError(f'Activation capture failed: {self.failure}')
        if self.output is None:
            raise ValueError('No completed scoring forward captured')
        if type(seed_index) is not int or not 0 <= seed_index < self.output.size(0):
            raise ValueError('Invalid seed index')
        output = self.output.reshape(self.output.size(0), -1)[seed_index]
        if self.config.output_index >= output.numel():
            raise ValueError('Output selector outside head shape')
        relevance = torch.zeros_like(output)
        relevance[self.config.output_index] = output[self.config.output_index]
        states = [(self.entity, seed_index, relevance, [])]
        bias_total = residual = omitted = 0.0
        expanded = 0
        truncated = False
        for index in range(len(self.layers) - 1, -1, -1):
            next_states = []
            for table, row, rel, path in states:
                incoming = self._incoming(self.layers[index], table, row,
                                          self.config.max_walks - expanded)
                if incoming is None:
                    omitted += float(rel.sum())
                    truncated = True
                    continue
                routes, biases = incoming
                expanded += len(routes)
                zs = [z + self.config.gamma * z.clamp(min=0) for _, _, _, z in routes]
                bs = [b + self.config.gamma * b.clamp(min=0) for b in biases]
                total = torch.zeros_like(rel)
                for z in zs:
                    total += z.sum(0)
                for b in bs:
                    total += b
                denom = total + torch.where(total >= 0, self.config.epsilon, -self.config.epsilon)
                factor = rel / denom
                allocated = torch.zeros_like(rel)
                for (src, occurrence, route, _), z in zip(routes, zs):
                    share = z * factor
                    allocated += share.sum(0)
                    next_states.append((src, occurrence, share.sum(1), [{'layer': index, **route}] + path))
                for b in bs:
                    share = b * factor
                    bias_total += float(share.sum())
                    allocated += share
                residual += float((rel - allocated).sum())
            states = next_states
        walks = [{'source_table': table, 'source_occurrence': row, 'steps': path, 'relevance': float(rel.sum())} for table, row, rel, path in states]
        walks.sort(key=lambda w: abs(w['relevance']), reverse=True)
        omitted += sum(w['relevance'] for w in walks[self.config.top_k:])
        truncated |= len(walks) > self.config.top_k
        walks = walks[:self.config.top_k]
        score = float(output[self.config.output_index])
        values = [score, bias_total, residual, omitted] + [w['relevance'] for w in walks]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite relevance computation')
        return {'schema': 'walk_attribution/v1', 'method': 'hetero_lrp_gamma_reference_v1', 'coordinate': 'encoded_model_output', 'input_coordinate': 'encoded_node_features', 'output_index': self.config.output_index, 'score': score, 'gamma': self.config.gamma, 'epsilon': self.config.epsilon, 'walks': walks, 'bias_relevance': bias_total, 'stabilizer_residual': residual, 'omitted_signed_relevance': omitted, 'expanded_routes': expanded, 'truncated': truncated, 'complete': not truncated, 'search': 'bounded_exhaustive', 'accounting_error': score - sum(w['relevance'] for w in walks) - bias_total - residual - omitted}


def inspect_walk_capability(module):
    """Return an operator qualification record without running the model."""
    try:
        WalkCapture(module).validate()
    except (UnsupportedWalkAttribution, AttributeError) as error:
        return {'supported': False, 'reason': str(error),
                'method': 'hetero_lrp_gamma_reference_v1'}
    return {'supported': True, 'method': 'hetero_lrp_gamma_reference_v1',
            'coordinate': 'encoded_model_output',
            'input_coordinate': 'encoded_node_features',
            'qualification': 'reference_operator_profile',
            'qualified_devices': ['cpu'],
            'device_qualified': all(p.device.type == 'cpu' for p in module.parameters())}
