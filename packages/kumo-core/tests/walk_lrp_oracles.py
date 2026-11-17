# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent oracles for weight-gamma GNN-LRP (test-only).

Nothing here imports `kumo_core.explain` (an AST test enforces it), so the
oracle shares no code with the method, the shared enumerator or the typed
capture. It re-implements the float64 forward of `HeteroGNN` plus
`NodePredHead` from module parameters with Engine's layer trimming,
enumerates walks from the trimmed edges, and evaluates the paper's
forward-rewrite (Schnake et al., Section III-B1) with autograd. Every linear
step has the gamma-modified total

    P = sum over routes of x @ W_hat.T / d + b_hat,   W_hat = W + gamma * max(0, W),

and the rewritten step output is `Q = P * [H / (P + eps * sgn(P))]_cst`, with
`sgn(0) = +1` and `H` the true step output. A walk from source `J` scores

    R_W = < d out / d H_0[J], H_0[J] >

with every message off the walk detached at every layer (route masks).
Without masks, `H_0 * grad` is each source's complete vector, and
`grad_P . b_hat` and `grad_P . (eps * sgn(P))` are the bias and stabilizer
shares by kind.

Choices beyond the paper's GCN rewrite, which the oracle shares with the
contract as definitions rather than paper results: relation-specific
weights, biases as weights of a constant input (detached constants inside
`P`), the sign-preserving absolute epsilon inside the detached factor, the
identity root as the weight `I`, preprocessing and head linears under the
same rule, and the selected sign folded into the last head row. Route masks
replace the paper's node masks `M_K`: the two agree whenever consecutive walk
nodes are joined by a single route, and route masks stay exact for parallel
routes (`masks='node'` evaluates the paper's form).
"""
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch_geometric.nn.aggr import (
    MeanAggregation,
    MultiAggregation,
    SumAggregation,
)

from kumo_core.modeling.gnn.hetero import (
    ForeignKeyToPrimaryKeyConv,
    HeteroIdentity,
    PrimaryKeyToForeignKeyConv,
)

F64 = torch.float64
KINDS = ('preprocessing', 'message_passing', 'head')


# True forward ###############################################################

@dataclass
class Layer:
    """One message-passing layer of the true float64 forward."""
    inputs: dict
    rows: dict
    edges: dict
    pre: dict
    out: dict


@dataclass
class Trace:
    """The true forward of one seed: inputs, every step and the output."""
    module: Any
    seed: tuple
    h0: dict
    preprocessing: list
    layers: list
    head: list
    z: torch.Tensor


def _params(lin):
    weight = lin.weight.detach().to(F64)
    return weight, None if lin.bias is None else lin.bias.detach().to(F64)


def _activate(values, relu):
    return values.clamp(min=0) if relu else values


def _operator(conv):
    if isinstance(conv, ForeignKeyToPrimaryKeyConv):
        aggr = conv.aggr
        if isinstance(aggr, MultiAggregation):
            (aggr, ) = aggr.aggrs
        if isinstance(aggr, MeanAggregation):
            return 'mean'
        if isinstance(aggr, SumAggregation):
            return 'sum'
        raise ValueError('The oracle covers sum and mean aggregation')
    if isinstance(conv, PrimaryKeyToForeignKeyConv):
        return 'gather'
    raise ValueError('Unknown relation operator')


def _check(module):
    model, head = module.model, module.head
    if model.act not in (None, 'relu') or model.norm is not None:
        raise ValueError('The oracle covers ReLU or identity without normalization')
    if head.mlp.act is not None and type(head.mlp.act) is not torch.nn.ReLU:
        raise ValueError('The oracle covers ReLU or identity readouts')
    if any(type(norm) is not torch.nn.Identity for norm in head.mlp.norms):
        raise ValueError('The oracle covers readouts without normalization')
    if head.prediction_time_encodings or head.param_keys:
        raise ValueError('The oracle covers point readouts')


def forward_trace(module, x_dict, edge_index_dict, num_sampled_nodes_dict=None,
                  num_sampled_edges_dict=None, *, seed_index=0):
    """Float64 forward of `module.model` and `module.head` with Engine trimming.

    Layer `t` keeps the destination rows that are not among the last `t`
    hops and drops the edges sampled at the last `t - 1` hops; a relation
    convolution with no edges contributes its bias only if its `bias` flag
    is set; the head reads the seed row of the entity table.
    """
    _check(module)
    model, head = module.model, module.head
    relu = model.act is not None
    h0 = {table: x.detach().to(F64) for table, x in x_dict.items()}
    hidden = dict(h0)
    preprocessing = []
    for layer in model.lins:
        pre = {}
        for table, x in hidden.items():
            weight, bias = _params(layer.lin_dict[table])
            pre[table] = x @ weight.T + (0.0 if bias is None else bias)
        out = {table: _activate(a, relu) for table, a in pre.items()}
        preprocessing.append({'input': hidden, 'pre': pre, 'out': out})
        hidden = out
    T = len(model.hetero_convs)
    nodes = None if num_sampled_nodes_dict is None else {
        table: list(v) + [0] * max(T + 1 - len(v), 0)
        for table, v in num_sampled_nodes_dict.items()}
    counts = None if num_sampled_edges_dict is None else {
        tuple(relation): list(v) + [0] * max(T - len(v), 0)
        for relation, v in num_sampled_edges_dict.items()}
    edges = {tuple(relation): index for relation, index in edge_index_dict.items()}
    layers = []
    for i in range(T):
        rows = {table: x.size(0) - (nodes[table][-(i + 1)] if nodes else 0)
                for table, x in hidden.items()}
        if counts and i > 0:
            edges = {relation: index[:, :index.size(1) - counts[relation][-i]]
                     for relation, index in edges.items()}
        root = model.root_skips[i]
        pre = {}
        for table, x in hidden.items():
            x = x[:rows[table]]
            if isinstance(root, HeteroIdentity):
                pre[table] = x.clone()
            else:
                weight, bias = _params(root.lin_dict[table])
                pre[table] = x @ weight.T + (0.0 if bias is None else bias)
        for relation, index in edges.items():
            source, _, destination = relation
            conv = model.hetero_convs[i][relation]
            operator = _operator(conv)
            weight, bias = _params(conv.lin)
            n = rows[destination]
            channels = hidden[source].size(1)
            if index.size(1) == 0:
                message = torch.zeros(n, weight.size(0), dtype=F64)
                if conv.bias:
                    message = message + bias
            else:
                degree = torch.bincount(index[1], minlength=n)
                if operator == 'gather' and int(degree.max()) > 1:
                    raise ValueError('A gather destination has several edges')
                aggregated = torch.zeros(n, channels, dtype=F64).index_add(
                    0, index[1], hidden[source][index[0]])
                if operator == 'mean':
                    aggregated = aggregated / degree.clamp(min=1).to(F64)[:, None]
                message = aggregated @ weight.T + (0.0 if bias is None else bias)
            pre[destination] = pre[destination] + message
        out = {table: _activate(a, relu) for table, a in pre.items()}
        layers.append(Layer(inputs=hidden, rows=rows, edges=dict(edges), pre=pre, out=out))
        hidden = out
    entity = module.entity_table_name
    x = hidden[entity][seed_index]
    readout = []
    lins = list(head.mlp.lins)
    for j, lin in enumerate(lins):
        weight, bias = _params(lin)
        a = weight @ x + (0.0 if bias is None else bias)
        readout.append({'input': x, 'pre': a})
        x = _activate(a, head.mlp.act is not None) if j < len(lins) - 1 else a
    return Trace(module=module, seed=(entity, seed_index), h0=h0,
                 preprocessing=preprocessing, layers=layers, head=readout, z=x)


# Walks ######################################################################

def incoming(trace, t, node):
    """`[(step key, source node)]` of every route into `node` at layer t.

    Step keys follow the contract's walk keys: `(t, 'self_update', table,
    '', table, occurrence)` or `(t, 'relation', source table, relation,
    destination table, edge occurrence)`.
    """
    table, occurrence = node
    layer = trace.layers[t - 1]
    if not 0 <= occurrence < layer.rows[table]:
        raise ValueError(f'{node} is not a destination of layer {t}')
    routes = [((t, 'self_update', table, '', table, occurrence), node)]
    for relation in sorted(layer.edges):
        source, name, destination = relation
        if destination != table:
            continue
        index = layer.edges[relation]
        for edge in (index[1] == occurrence).nonzero().flatten().tolist():
            routes.append(((t, 'relation', source, name, destination, edge),
                           (source, int(index[0, edge]))))
    return routes


def walks(trace):
    """Every walk into the seed: `{walk key: source node}`, layer 1 first."""
    frontier = [((), trace.seed)]
    for t in range(len(trace.layers), 0, -1):
        frontier = [((key, ) + suffix, source) for suffix, node in frontier
                    for key, source in incoming(trace, t, node)]
    return dict(frontier)


def _destination(trace, key):
    t, kind, source, name, destination, occurrence = key
    if kind == 'self_update':
        return (destination, occurrence)
    index = trace.layers[t - 1].edges[(source, name, destination)]
    return (destination, int(index[1, occurrence]))


# Forward rewrite ############################################################

def _hat(tensor, gamma):
    return tensor + gamma * tensor.clamp(min=0)


def _sign_epsilon(total, epsilon):
    return torch.where(total >= 0, torch.full_like(total, epsilon),
                       torch.full_like(total, -epsilon))


def _live(values, rows):
    """Rows in `rows` (a bool vector) keep their gradient; others detach."""
    return torch.where(rows[:, None], values, values.detach())


class Rewrite:
    """The forward rewrite of one trace for one gamma, epsilon and selection.

    `sign` folds the selected output row of the last head linear (weight and
    bias); `output_index` is the explained output.
    """

    def __init__(self, trace, *, gamma, epsilon=1e-9, sign=1, output_index=0):
        self.trace, self.gamma, self.epsilon = trace, float(gamma), float(epsilon)
        self.sign, self.index = sign, output_index
        model, head = trace.module.model, trace.module.head
        self.relu_head = head.mlp.act is not None
        self.readout = []
        lins = list(head.mlp.lins)
        for j, lin in enumerate(lins):
            weight, bias = _params(lin)
            if j == len(lins) - 1:
                weight = weight.clone()
                weight[output_index] *= sign
                if bias is not None:
                    bias = bias.clone()
                    bias[output_index] *= sign
            self.readout.append((_hat(weight, gamma),
                                 None if bias is None else _hat(bias, gamma)))
        self.pre_params = [{table: tuple(None if p is None else _hat(p, gamma)
                                         for p in _params(layer.lin_dict[table]))
                            for table in trace.h0} for layer in model.lins]
        # Detached factors H / (P + eps * sgn(P)) and stabilizer terms from
        # the true forward.
        self.pre_factors, self.mp_factors, self.head_factors = [], [], []
        for p, step in enumerate(trace.preprocessing):
            factors = {}
            for table, x in step['input'].items():
                total = self._linear(x, *self.pre_params[p][table])
                factors[table] = self._factor(step['out'][table], total)
            self.pre_factors.append(factors)
        for t, layer in enumerate(trace.layers, start=1):
            totals, _ = self.layer_totals(t, layer.inputs)
            self.mp_factors.append({table: self._factor(layer.out[table], total)
                                    for table, total in totals.items()})
        for j, step in enumerate(trace.head):
            total = self._linear(step['input'], *self.readout[j])
            if j == len(trace.head) - 1:
                value = torch.zeros_like(total)
                value[output_index] = sign * trace.z[output_index]
            else:
                value = _activate(step['pre'], self.relu_head)
            self.head_factors.append(self._factor(value, total))

    @staticmethod
    def _linear(x, weight, bias):
        total = x @ weight.T
        return total if bias is None else total + bias

    def _factor(self, value, total):
        epsilon = _sign_epsilon(total, self.epsilon)
        return value / (total + epsilon), epsilon

    def layer_totals(self, t, hidden, live=None):
        """Gamma-modified totals `P` of layer t per destination table, and the
        constant modified bias sum added there. `live` is a step key whose
        message alone keeps its gradient (every other message is detached),
        or None."""
        layer = self.trace.layers[t - 1]
        model = self.trace.module.model
        root = model.root_skips[t - 1]
        totals, biases = {}, {}
        for table, x in hidden.items():
            n = layer.rows[table]
            x = x[:n]
            if isinstance(root, HeteroIdentity):
                part, bias = (1.0 + self.gamma) * x, None
            else:
                weight, bias = _params(root.lin_dict[table])
                part = x @ _hat(weight, self.gamma).T
            if live is not None:
                rows = torch.zeros(n, dtype=torch.bool)
                if live[1] == 'self_update' and live[2] == table:
                    rows[live[5]] = True
                part = _live(part, rows)
            totals[table] = part
            biases[table] = torch.zeros(part.size(1), dtype=F64)
            if bias is not None:
                biases[table] = biases[table] + _hat(bias, self.gamma)
        for relation in sorted(layer.edges):
            source, name, destination = relation
            index = layer.edges[relation]
            conv = model.hetero_convs[t - 1][relation]
            weight, bias = _params(conv.lin)
            if index.size(1):
                message = hidden[source][index[0]] @ _hat(weight, self.gamma).T
                if _operator(conv) == 'mean':
                    degree = torch.bincount(index[1], minlength=layer.rows[destination])
                    message = message / degree.to(F64)[index[1]][:, None]
                if live is not None:
                    rows = torch.zeros(index.size(1), dtype=torch.bool)
                    if live[1] == 'relation' and live[2:5] == relation:
                        rows[live[5]] = True
                    message = _live(message, rows)
                totals[destination] = totals[destination].index_add(0, index[1], message)
            if bias is not None and (index.size(1) or conv.bias):
                biases[destination] = biases[destination] + _hat(bias, self.gamma)
        return {table: totals[table] + biases[table] for table in totals}, biases

    def output(self, h0, *, route_keys=None, node_path=None, record=None):
        """The rewritten selected output from `h0`.

        `route_keys[t]` is the live step key at layer t (route masks);
        `node_path[t]` the live node of `H[t]` (the paper's node masks).
        `record` collects `(kind, P, bias term, stabilizer term)`.
        """
        hidden = h0
        for p, factors in enumerate(self.pre_factors):
            new = {}
            for table, x in hidden.items():
                weight, bias = self.pre_params[p][table]
                total = self._linear(x, weight, bias)
                factor, epsilon = factors[table]
                if record is not None:
                    record.append(('preprocessing', total,
                                   0.0 if bias is None else bias, epsilon))
                new[table] = total * factor
            hidden = new
        for t in range(1, len(self.trace.layers) + 1):
            live = None if route_keys is None else route_keys[t]
            totals, biases = self.layer_totals(t, hidden, live)
            new = {}
            for table, total in totals.items():
                factor, epsilon = self.mp_factors[t - 1][table]
                if record is not None:
                    record.append(('message_passing', total, biases[table], epsilon))
                values = total * factor
                if node_path is not None:
                    rows = torch.zeros(values.size(0), dtype=torch.bool)
                    if node_path[t][0] == table:
                        rows[node_path[t][1]] = True
                    values = _live(values, rows)
                new[table] = values
            hidden = new
        table, seed = self.trace.seed
        x = hidden[table][seed]
        for j, (weight, bias) in enumerate(self.readout):
            total = weight @ x if bias is None else weight @ x + bias
            factor, epsilon = self.head_factors[j]
            if record is not None:
                record.append(('head', total, 0.0 if bias is None else bias, epsilon))
            x = total * factor
        return x[self.index]

    def walk_score(self, keys, source, *, masks='route'):
        """`R_W` of one walk (route or node masks)."""
        h0 = {table: x.clone() for table, x in self.trace.h0.items()}
        table, occurrence = source
        leaf = h0[table][occurrence].clone().requires_grad_(True)
        h0[table] = torch.cat([h0[table][:occurrence], leaf[None], h0[table][occurrence + 1:]])
        if masks == 'route':
            out = self.output(h0, route_keys=dict(enumerate(keys, start=1)))
        elif masks == 'node':
            path = {t: _destination(self.trace, key) for t, key in enumerate(keys, start=1)}
            out = self.output(h0, node_path=path)
        else:
            raise ValueError(f'Unknown masks {masks!r}')
        if not out.requires_grad:
            return 0.0
        (grad, ) = torch.autograd.grad(out, leaf, allow_unused=True)
        if grad is None:
            return 0.0
        return math.fsum((grad * leaf.detach()).tolist())

    def totals(self):
        """Unmasked rewrite: per-source vectors and bias and stabilizer by kind."""
        h0 = {table: x.clone().requires_grad_(True) for table, x in self.trace.h0.items()}
        record = []
        out = self.output(h0, record=record)
        tensors = list(h0.values()) + [total for _, total, _, _ in record]
        grads = torch.autograd.grad(out, tensors, allow_unused=True)
        per_source = {}
        for (table, x), grad in zip(h0.items(), grads):
            grad = torch.zeros_like(x) if grad is None else grad
            for occurrence in range(x.size(0)):
                per_source[(table, occurrence)] = (grad[occurrence]
                                                   * x[occurrence].detach()).tolist()
        bias = {kind: [] for kind in KINDS}
        stabilizer = {kind: [] for kind in KINDS}
        for (kind, _, bias_term, epsilon), grad in zip(record, grads[len(h0):]):
            if grad is None:
                continue
            bias[kind].extend((grad * bias_term).reshape(-1).tolist())
            stabilizer[kind].extend((grad * epsilon).reshape(-1).tolist())
        return {'per_source': per_source,
                'bias': {kind: math.fsum(values) for kind, values in bias.items()},
                'stabilizer': {kind: math.fsum(values) for kind, values in stabilizer.items()},
                'output': float(out.detach())}


def explain(module, batch, *, gamma, epsilon=1e-9, sign=1, output_index=0, seed_index=0,
            masks='route'):
    """Oracle explanation of `module` on `batch = (x, edges[, nodes, edge counts])`.

    Returns walk scores keyed by walk key, the source of every walk, the
    unmasked per-source vectors, bias and stabilizer by kind and the trace.
    """
    trace = forward_trace(module, *batch, seed_index=seed_index)
    rewrite = Rewrite(trace, gamma=gamma, epsilon=epsilon, sign=sign,
                      output_index=output_index)
    sources = walks(trace)
    scores = {keys: rewrite.walk_score(keys, source, masks=masks)
              for keys, source in sources.items()}
    totals = rewrite.totals()
    return {'walks': scores, 'sources': sources, 'per_source': totals['per_source'],
            'bias': totals['bias'], 'stabilizer': totals['stabilizer'],
            'score': sign * float(trace.z[output_index]), 'trace': trace}


__all__ = ['Layer', 'Rewrite', 'Trace', 'explain', 'forward_trace', 'incoming', 'walks']
