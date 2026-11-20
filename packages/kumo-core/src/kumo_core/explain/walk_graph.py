# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed capture of one scoring forward for walk methods (frozen interface).

`GraphCapture` copies tensors from forward hooks around one `HeteroGNN` plus
`NodePredHead` forward. It never mutates the module, its parameters, the
calibrator or autograd state, and registers hooks only after qualification.
`WalkComputation` exposes the captured graph to walk rules: canonical
incoming routes per destination, the captured layer inputs and
pre-activations, and `forward_piece`, the float64 forward on the captured
linear piece. Everything a method computes is float64 (contract 4.2); `H_0` is the
`x_dict` argument of `HeteroGNN.forward` (the encoder output).
"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor

from kumo_core.explain.walk_attribution import UnsupportedWalkAttribution
from kumo_core.modeling.gnn.hetero import (
    ForeignKeyToPrimaryKeyConv, HeteroIdentity, HeteroLinear,
)

F64 = torch.float64
KINDS = ('preprocessing', 'message_passing', 'head')


# Numerics ###################################################################

def mkldnn_matmul_fp32_precision():
    """The derived oneDNN fp32 matmul setting, or None if torch lacks it."""
    matmul = getattr(getattr(torch.backends, 'mkldnn', None), 'matmul', None)
    value = getattr(matmul, 'fp32_precision', None)
    return None if value is None else str(value)


def normalized_mkldnn(value):
    """`none` (never set in this process) behaves as `ieee` (contract 4.8)."""
    return 'ieee' if value == 'none' else value


def capture_numerics():
    """Numerical settings in effect now (contract 4.8 and C9)."""
    capability = getattr(torch.backends.cpu, 'get_cpu_capability', None)
    return {
        'precision_mode': torch.get_float32_matmul_precision(),
        'mkldnn_matmul_fp32_precision': mkldnn_matmul_fp32_precision(),
        'num_threads': torch.get_num_threads(),
        'cpu_capability': None if capability is None else str(capability()),
        'torch_version': str(torch.__version__),
    }


@contextmanager
def float32_matmul_precision(mode):
    """Apply float32 matmul `mode` and restore both process settings after.

    `None` leaves the process unchanged. The setting is process-global, so
    replay must run on one thread; restoration happens even on exceptions.
    """
    if mode is None:
        yield
        return
    previous = torch.get_float32_matmul_precision()
    previous_mkldnn = mkldnn_matmul_fp32_precision()
    torch.set_float32_matmul_precision(mode)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)
        if (previous_mkldnn is not None
                and mkldnn_matmul_fp32_precision() != previous_mkldnn):
            torch.backends.mkldnn.matmul.fp32_precision = previous_mkldnn


# Qualification and records ##################################################

#: Structural qualification codes every walk method checks (contract 4.9).
STRUCTURAL_CODES = ('hetero_gnn', 'node_pred_head', 'eval_mode',
                    'relu_or_identity', 'no_normalization',
                    'sum_or_mean_aggregation', 'registered_operators',
                    'point_head', 'no_prediction_time_encoding',
                    'calibrator_outside_f')


def structural_qualification(module):
    """Shared operator profile of every walk method; raises before hooks.

    Reuses the v1 checks unchanged (contract section 7) and adds the head
    shape checks the typed capture relies on. Returns STRUCTURAL_CODES.
    """
    from kumo_core.explain.walk_attribution import WalkCapture
    try:
        WalkCapture(module).validate()
    except AttributeError as error:
        raise UnsupportedWalkAttribution(f'Unrecognized module: {error}') from error
    head = module.head
    if not getattr(head.mlp, 'plain_last', True):
        raise UnsupportedWalkAttribution('Readout must end with a plain linear')
    if any(not isinstance(lin, torch.nn.Module) or not hasattr(lin, 'weight')
           for lin in head.mlp.lins):
        raise UnsupportedWalkAttribution('Unregistered readout linear')
    return list(STRUCTURAL_CODES)


def _relation_operator(conv):
    if isinstance(conv, ForeignKeyToPrimaryKeyConv):
        aggr = conv.aggr
        if hasattr(aggr, 'aggrs'):
            aggr = aggr.aggrs[0]
        name = type(aggr).__name__.replace('Aggregation', '').lower()
        return 'fk_to_pk', name
    return 'pk_to_fk', 'gather'


def architecture_record(module):
    """The explained architecture, read from modules and actual parameters.

    Bias flags come from `lin.bias is None`, never from options or plans.
    """
    model, head = module.model, module.head
    aggr = model.aggr[0] if isinstance(model.aggr, list) and len(model.aggr) == 1 else model.aggr
    lins = [lin for layer in model.lins for lin in layer.lin_dict.values()]
    lins += [conv.lin for layer in model.hetero_convs for conv in layer.values()]
    lins += [lin for root in model.root_skips if isinstance(root, HeteroLinear)
             for lin in root.lin_dict.values()]
    relations = []
    if len(model.hetero_convs):
        for relation, conv in model.hetero_convs[0].items():
            direction, operator = _relation_operator(conv)
            relations.append({'relation': list(_relation_key(relation)),
                              'direction': direction, 'operator': operator})
    relations.sort(key=lambda r: r['relation'])
    return {
        'model': type(model).__name__,
        'head': type(head).__name__,
        'message_passing_layers': len(model.hetero_convs),
        'preprocessing_layers': len(model.lins),
        'head_layers': len(head.mlp.lins),
        'channels': int(model.channels),
        'aggregation': aggr,
        'activation': model.act,
        'normalization': model.norm,
        'gnn_bias': any(lin.bias is not None for lin in lins),
        'head_bias': any(lin.bias is not None for lin in head.mlp.lins),
        'root': ('linear' if any(isinstance(r, HeteroLinear) for r in model.root_skips)
                 else 'identity'),
        'prediction_time_encodings': bool(head.prediction_time_encodings),
        'relations': relations,
    }


def operations_record(module):
    """Preprocessing and head operations: never gated, never walk steps."""
    model, head = module.model, module.head
    act = None if model.act is None else 'relu'
    head_act = None if head.mlp.act is None else 'relu'
    last = len(head.mlp.lins) - 1
    return {
        'preprocessing': [{'index': i, 'kind': 'linear', 'activation': act,
                           'bias': any(lin.bias is not None for lin in layer.lin_dict.values())}
                          for i, layer in enumerate(model.lins)],
        'head': [{'index': j, 'kind': 'linear',
                  'activation': head_act if j < last else None,
                  'bias': lin.bias is not None}
                 for j, lin in enumerate(head.mlp.lins)],
    }


def _relation_key(relation):
    # PyG's ModuleDict yields edge types as tuples (the v1 capture relies on
    # the same behavior).
    if not isinstance(relation, tuple) or len(relation) != 3:
        raise ValueError(f'Unexpected relation key {relation!r}')
    return relation


# Captured structure ##########################################################

@dataclass(frozen=True)
class LinearOp:
    """A preprocessing or head linear with its captured rows (float64).

    Preprocessing ops are keyed by node table and cover every captured row;
    head ops are keyed by the entity table and their rows are the batch seeds
    (the explained seed is `WalkComputation.seed[1]`). `activation` follows
    the linear: 'relu' or None for identity.
    """
    index: int
    kind: str
    activation: str | None
    weight: dict
    bias: dict
    input: dict
    pre_activation: dict


@dataclass(frozen=True)
class Route:
    """One gate into `destination` at message-passing `layer` (contract 10).

    `source` is the occurrence of `H[layer - 1]` the route reads; a self
    update keeps the occurrence. `weight` is the float64 matrix `W`, the root
    `W` or the identity, not divided by `denominator`.
    """
    layer: int
    kind: str
    source: tuple
    destination: tuple
    relation: tuple | None
    direction: str | None
    operator: str
    edge_occurrence: int | None
    denominator: int
    weight: Tensor


@dataclass(frozen=True)
class Incoming:
    """Canonical routes into one destination and every bias added there.

    Routes: the self update first, then relations sorted by relation triple
    and edge occurrence (C6). `biases` holds the root bias and the bias of
    every relation convolution called into the destination table at this
    layer, including relations with no incoming edge at this destination.
    """
    routes: tuple
    biases: tuple


@dataclass(frozen=True)
class PieceForward:
    """Float64 forward from `h0` with every captured activation pattern fixed.

    `activation` is the message-passing and preprocessing activation ('relu',
    or None for identity, where nothing is gated). `masks[t][table]` and
    `preprocess_masks[i][table]` are the boolean ReLU masks applied
    (captured pre-activation > 0), or None under identity; `head_masks[j]` is
    the mask applied after head linear j for the seed row, None where no
    activation follows. `hidden[0]` is the message-passing input;
    `pre_activation[t]` and `hidden[t]` are `a~_t` and `H~_t` for every
    destination row of layer t. `z_piece` is the unsigned head output row of
    the seed; methods use the signed `f_piece = s * z_piece[output_index]`.
    """
    activation: str | None
    preprocess_masks: tuple
    preprocess_pre_activation: tuple
    preprocess_hidden: tuple
    masks: dict
    pre_activation: dict
    hidden: dict
    head_masks: tuple
    head_pre_activation: tuple
    z_piece: Tensor


@dataclass
class RelationOp:
    """One relation convolution called at a layer (float64 parameters).

    `edge_index` holds the retained edges at this layer; its columns are a
    prefix of the full edge index, so `edge_occurrence` names the same edge at
    every layer.
    """
    relation: tuple
    direction: str
    operator: str
    edge_index: Tensor
    n_src: int
    n_dst: int
    weight: Tensor
    bias: Tensor | None


@dataclass
class RootOp:
    """The root skip of one destination table at a layer."""
    kind: str
    weight: Tensor | None
    bias: Tensor | None
    n_dst: int


@dataclass
class _Layer:
    relations: list
    roots: dict
    pre: dict | None = None
    post: dict | None = None


def _copy(x):
    return x.detach().clone()


class GraphCapture:
    """Hook one scoring forward and expose it as a `WalkComputation`.

    Qualification (`structural_qualification` plus the method's `qualify`)
    precedes hook registration, so unsupported models never get hooks. Hook
    failures are recorded, never raised, so predictions continue.
    """

    def __init__(self, module, config=None, *, qualify=True):
        self.module = module
        self.config = config
        self.qualify = qualify
        self.qualification = None
        self.handles = []
        self.failure = None
        self._reset()

    def _reset(self):
        self.h0 = None
        self.full_edges = None
        self.num_sampled_nodes = None
        self.num_sampled_edges = None
        self.preprocess = []
        self.layers = []
        self.head_rows = []
        self.output = None

    def __enter__(self):
        if self.qualify:
            if self.config is None:
                self.qualification = structural_qualification(self.module)
            else:
                from kumo_core.explain.walk_attribution import method_module
                self.qualification = list(method_module(self.config.method).qualify(
                    self.module, self.config))
        self._reset()
        self.failure = None
        model, head = self.module.model, self.module.head
        try:
            self.handles.append(model.register_forward_pre_hook(
                self._guard(self._model_input), with_kwargs=True))
            for index, lin in enumerate(model.lins):
                self.handles.append(lin.register_forward_hook(
                    self._guard(partial(self._preprocess_linear, index))))
            for index, post in enumerate(model.post_lins):
                self.handles.append(post.register_forward_hook(
                    self._guard(partial(self._preprocess_post, index))))
            for index, convs in enumerate(model.hetero_convs):
                for relation, conv in convs.items():
                    self.handles.append(conv.register_forward_hook(
                        self._guard(partial(self._relation, index, _relation_key(relation)))))
                self.handles.append(model.root_skips[index].register_forward_hook(
                    self._guard(partial(self._root, index))))
                self.handles.append(model.post_convs[index].register_forward_hook(
                    self._guard(partial(self._post, index))))
            for index, lin in enumerate(head.mlp.lins):
                self.handles.append(lin.register_forward_hook(
                    self._guard(partial(self._head_linear, index))))
            self.handles.append(head.register_forward_hook(self._guard(self._head_output)))
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _guard(self, fn):
        def hook(*args):
            if self.failure is not None:
                return None
            try:
                fn(*args)
            except Exception as error:
                # An explanation failure never replaces a completed output.
                self.failure = type(error).__name__
            return None
        return hook

    # Hooks: copy tensors, return None, never touch the module. -------------
    def _model_input(self, module, args, kwargs):
        if self.h0 is not None:
            raise ValueError('GraphCapture supports one forward per context')
        names = ('x_dict', 'edge_index_dict', 'num_sampled_nodes_dict',
                 'num_sampled_edges_dict')
        values = dict(zip(names, args))
        values.update({k: v for k, v in kwargs.items() if k in names})
        self.h0 = {table: _copy(x) for table, x in values['x_dict'].items()}
        self.full_edges = {tuple(k): _copy(e) for k, e in values['edge_index_dict'].items()}
        nodes, edges = values.get('num_sampled_nodes_dict'), values.get('num_sampled_edges_dict')
        self.num_sampled_nodes = None if nodes is None else {k: list(v) for k, v in nodes.items()}
        self.num_sampled_edges = None if edges is None else {tuple(k): list(v) for k, v in edges.items()}

    def _preprocess_linear(self, index, module, args, output):
        if len(self.preprocess) != index:
            raise ValueError('Preprocessing layers out of order')
        self.preprocess.append({
            'weight': {t: _copy(lin.weight) for t, lin in module.lin_dict.items()},
            'bias': {t: None if lin.bias is None else _copy(lin.bias)
                     for t, lin in module.lin_dict.items()},
            'input': {t: _copy(x) for t, x in args[0].items()},
            'pre': {t: _copy(x) for t, x in output.items()},
            'post': None})

    def _preprocess_post(self, index, module, args, output):
        self.preprocess[index]['post'] = {t: _copy(x) for t, x in output.items()}

    def _layer(self, index):
        while len(self.layers) <= index:
            self.layers.append(_Layer(relations=[], roots={}))
        return self.layers[index]

    def _relation(self, index, relation, module, args, output):
        x_src, x_dst, edge_index = args[:3]
        direction, operator = _relation_operator(module)
        self._layer(index).relations.append(RelationOp(
            relation=relation, direction=direction, operator=operator,
            edge_index=_copy(edge_index), n_src=int(x_src.size(0)),
            n_dst=int(x_dst.size(0)), weight=_copy(module.lin.weight),
            bias=None if module.lin.bias is None else _copy(module.lin.bias)))

    def _root(self, index, module, args, output):
        layer = self._layer(index)
        for table, x in args[0].items():
            if isinstance(module, HeteroLinear):
                lin = module.lin_dict[table]
                layer.roots[table] = RootOp(
                    'linear', _copy(lin.weight),
                    None if lin.bias is None else _copy(lin.bias), int(x.size(0)))
            elif isinstance(module, HeteroIdentity):
                layer.roots[table] = RootOp('identity', None, None, int(x.size(0)))
            else:
                raise ValueError('Unregistered root operator')

    def _post(self, index, module, args, output):
        layer = self._layer(index)
        layer.pre = {t: _copy(x) for t, x in args[0].items()}
        layer.post = {t: _copy(x) for t, x in output.items()}

    def _head_linear(self, index, module, args, output):
        if len(self.head_rows) != index:
            raise ValueError('Readout layers out of order')
        self.head_rows.append({'weight': _copy(module.weight),
                               'bias': None if module.bias is None else _copy(module.bias),
                               'input': _copy(args[0]), 'pre': _copy(output)})

    def _head_output(self, module, args, output):
        if self.output is not None:
            raise ValueError('GraphCapture supports one forward per context')
        if not isinstance(output, Tensor):
            raise ValueError('Distribution readouts are unsupported')
        self.output = _copy(output)

    def computation(self, seed_index=0, numerics=None, *, preserve_device=False):
        """Build one seed's computation; opt in to the captured tensor device.

        The default retains the CPU contract of existing walk methods.
        """
        if self.failure is not None:
            raise ValueError(f'Activation capture failed: {self.failure}')
        if self.output is None or self.h0 is None:
            raise ValueError('No completed scoring forward captured')
        return WalkComputation(self, seed_index=seed_index, numerics=numerics,
                               preserve_device=preserve_device)


class WalkComputation:
    """One seed's captured forward, as the frozen method protocol sees it.

    All tensors are float64 copies of captured values, so the object is
    deep-copyable; treat it as read-only. Tests may deep-copy a computation
    and edit `pre_activation`, `hidden` or op tensors to inject sign flips or
    exact zeros, since masks and `forward_piece` read these attributes.

    Attributes: `seed` (entity table, seed occurrence), `T`, `channels`,
    `train_table_name`, `entity_table_name`, `activation` ('relu' or None),
    `h0` (`H_0` rows per table), `preprocessing` and `head` (`LinearOp`
    lists), `relations[t]` (`RelationOp` list), `roots[t]` (`RootOp` per
    table), `pre_activation[t]` (captured `a_t`), `hidden[t]` (captured
    `H_t`, 0 = message-passing input), `head_output` (captured `z` row),
    `numerics`, `model_dtype`, `gnn_bias_present`, `head_bias_present`.
    """

    def __init__(self, capture, *, seed_index=0, numerics=None,
                 preserve_device=False):
        self.capture_device = capture.output.device
        self.qualification = capture.qualification
        self.device = capture.output.device if preserve_device else torch.device(
            'cpu')

        def convert(x):
            if preserve_device and x.device != self.device:
                raise ValueError('Captured tensors span multiple devices')
            return x.detach().to(device=self.device, dtype=F64).clone()

        module = capture.module
        model, head = module.model, module.head
        self.T = len(model.hetero_convs)
        self.channels = int(model.channels)
        self.entity_table_name = module.entity_table_name
        self.train_table_name = getattr(module, 'train_table_name', None)
        self.model_dtype = str(capture.output.dtype).removeprefix('torch.')
        self.numerics = dict(numerics or {})
        if type(seed_index) is not int or not 0 <= seed_index < capture.output.size(0):
            raise ValueError('Invalid seed index')
        self.seed = (self.entity_table_name, seed_index)
        self.activation = None if model.act is None else 'relu'
        self.h0 = {t: convert(x) for t, x in capture.h0.items()}
        if len(capture.preprocess) != len(model.lins) or len(capture.layers) != self.T:
            raise ValueError('Incomplete layer capture')
        self.preprocessing = []
        hidden = self.h0
        for i, layer in enumerate(capture.preprocess):
            if layer['post'] is None:
                raise ValueError('Incomplete preprocessing capture')
            self.preprocessing.append(LinearOp(
                index=i, kind='preprocess', activation=self.activation,
                weight={t: convert(w) for t, w in layer['weight'].items()},
                bias={t: None if b is None else convert(b) for t, b in layer['bias'].items()},
                input={t: convert(x) for t, x in layer['input'].items()},
                pre_activation={t: convert(x) for t, x in layer['pre'].items()}))
            hidden = {t: convert(x) for t, x in layer['post'].items()}
        self.hidden = {0: hidden}
        self.pre_activation, self.relations, self.roots = {}, {}, {}
        for t, layer in enumerate(capture.layers, start=1):
            if preserve_device and any(r.edge_index.device != self.device
                                       for r in layer.relations):
                raise ValueError('Captured edges differ from the tensor device')
            if layer.pre is None or layer.post is None:
                raise ValueError(f'Incomplete capture at layer {t}')
            self._check_layer(t, layer, capture)
            self.relations[t] = [RelationOp(
                relation=r.relation, direction=r.direction, operator=r.operator,
                edge_index=r.edge_index.to(self.device).clone(), n_src=r.n_src, n_dst=r.n_dst,
                weight=convert(r.weight), bias=None if r.bias is None else convert(r.bias))
                for r in sorted(layer.relations, key=lambda r: r.relation)]
            self.roots[t] = {table: RootOp(
                root.kind, None if root.weight is None else convert(root.weight),
                None if root.bias is None else convert(root.bias), root.n_dst)
                for table, root in layer.roots.items()}
            self.pre_activation[t] = {k: convert(x) for k, x in layer.pre.items()}
            self.hidden[t] = {k: convert(x) for k, x in layer.post.items()}
        head_act = None if head.mlp.act is None else 'relu'
        last = len(capture.head_rows) - 1
        if last + 1 != len(head.mlp.lins):
            raise ValueError('Incomplete readout capture')
        table = self.entity_table_name
        self.head = [LinearOp(
            index=j, kind='head', activation=head_act if j < last else None,
            weight={table: convert(row['weight'])},
            bias={table: None if row['bias'] is None else convert(row['bias'])},
            input={table: convert(row['input'])}, pre_activation={table: convert(row['pre'])})
            for j, row in enumerate(capture.head_rows)]
        output = capture.output.reshape(capture.output.size(0), -1)
        self.head_output = convert(output[seed_index])
        # Actual captured parameters, never options or plans.
        self.gnn_bias_present = (
            any(b is not None for op in self.preprocessing for b in op.bias.values())
            or any(root.bias is not None for roots in self.roots.values()
                   for root in roots.values())
            or any(r.bias is not None for ops in self.relations.values() for r in ops))
        self.head_bias_present = any(b is not None for op in self.head
                                     for b in op.bias.values())
        self._incoming = {}
        self._eye = torch.eye(self.channels, dtype=F64, device=self.device)
        self._relevant = self._find_relevant()

    # Validation ------------------------------------------------------------
    def _check_layer(self, t, layer, capture):
        if t > 1:
            sources = capture.layers[t - 2].post
        elif capture.preprocess:
            sources = capture.preprocess[-1]['post']
        else:
            sources = capture.h0
        if sources is None:  # callers check every earlier capture first
            raise ValueError(f'Incomplete capture before layer {t}')
        for relation in layer.relations:
            src, _, dst = relation.relation
            full = capture.full_edges.get(relation.relation)
            count = relation.edge_index.size(1)
            if full is None or not torch.equal(full[:, :count], relation.edge_index):
                # Trimming keeps prefixes; anything else would rename edges.
                raise ValueError(f'Edge prefix broken for {relation.relation} at layer {t}')
            if relation.n_src != sources[src].size(0):
                raise ValueError(f'Source rows differ from H[{t - 1}] for {src}')
            if count and (int(relation.edge_index[0].max()) >= relation.n_src
                          or int(relation.edge_index[1].max()) >= relation.n_dst):
                raise ValueError('Edge endpoint outside its node block')
            if dst not in layer.roots or layer.roots[dst].n_dst != relation.n_dst:
                raise ValueError(f'Destination rows differ for {dst} at layer {t}')
        for table, root in layer.roots.items():
            if layer.pre[table].size(0) != root.n_dst or layer.post[table].size(0) != root.n_dst:
                raise ValueError(f'Destination rows differ for {table} at layer {t}')
            if sources[table].size(0) < root.n_dst:
                raise ValueError(f'Destinations are not a prefix of H[{t - 1}]')
        if set(layer.pre) != set(layer.roots):
            raise ValueError(f'Unexpected destination tables at layer {t}')

    def _find_relevant(self):
        """Nodes whose output reaches the seed; checks single-edge gathers."""
        relevant = {self.T: {self.seed}}
        for t in range(self.T, 0, -1):
            below = set()
            for node in sorted(relevant[t]):
                for route in self.incoming(t, node).routes:
                    below.add(route.source)
            relevant[t - 1] = below
        return relevant

    # Protocol ----------------------------------------------------------------
    def destinations(self, t):
        """Destination rows per table at layer t (after trimming)."""
        return {table: root.n_dst for table, root in self.roots[t].items()}

    def relevant_nodes(self, t):
        """Sorted occurrences of `H[t]` (0 = message-passing input) from which
        the seed is reachable through the remaining layers."""
        return sorted(self._relevant[t])

    def captured_input(self, t):
        """`H[t-1]` rows as captured, float64; t = 1 gives the MP input."""
        return self.hidden[t - 1]

    def captured_pre_activation(self, t):
        """`a[t]` rows as captured, float64; ReLU masks are `a > 0`."""
        return self.pre_activation[t]

    def gate_mask(self, t, node):
        """Captured activation pattern at `H[t, node]` as a float64 vector."""
        table, occurrence = node
        if self.activation is None:
            return torch.ones(self.channels, dtype=F64, device=self.device)
        return (self.pre_activation[t][table][occurrence] > 0).to(F64)

    def incoming(self, t, node):
        """Canonical `Incoming` of `node` at layer t (cached)."""
        key = (t, node)
        if key not in self._incoming:
            self._incoming[key] = self._build_incoming(t, node)
        return self._incoming[key]

    def _build_incoming(self, t, node):
        table, occurrence = node
        roots = self.roots[t]
        if table not in roots or not 0 <= occurrence < roots[table].n_dst:
            raise ValueError(f'{node} is not a destination of layer {t}')
        root = roots[table]
        routes = [Route(layer=t, kind='self_update', source=node, destination=node,
                        relation=None, direction=None, operator=root.kind,
                        edge_occurrence=None, denominator=1,
                        weight=self._eye if root.weight is None else root.weight)]
        biases = [] if root.bias is None else [root.bias]
        for relation in self.relations[t]:
            if relation.relation[2] != table:
                continue
            positions = (relation.edge_index[1] == occurrence).nonzero().flatten().tolist()
            if relation.operator == 'gather' and len(positions) > 1:
                raise UnsupportedWalkAttribution('PK gather has multiple incoming occurrences')
            denominator = len(positions) if relation.operator == 'mean' else 1
            for position in positions:
                routes.append(Route(
                    layer=t, kind='relation',
                    source=(relation.relation[0], int(relation.edge_index[0, position])),
                    destination=node, relation=relation.relation,
                    direction=relation.direction, operator=relation.operator,
                    edge_occurrence=int(position), denominator=denominator,
                    weight=relation.weight))
            if relation.bias is not None:
                biases.append(relation.bias)
        return Incoming(routes=tuple(routes), biases=tuple(biases))

    def forward_piece(self):
        """Float64 forward from `h0` with the captured masks held fixed."""
        relu = self.activation is not None

        def apply(values, captured):
            if not relu:
                return values, None
            mask = captured > 0
            return values * mask.to(F64), mask

        hidden = self.h0
        pre_list, hidden_list, mask_list = [], [], []
        for op in self.preprocessing:
            pre, masks, out = {}, {}, {}
            for table, x in hidden.items():
                a = x @ op.weight[table].T
                if op.bias[table] is not None:
                    a = a + op.bias[table]
                pre[table] = a
                out[table], masks[table] = apply(a, op.pre_activation[table])
            pre_list.append(pre)
            hidden_list.append(out)
            mask_list.append(masks if relu else None)
            hidden = out
        hiddens, pres, masks_by_layer = {0: hidden}, {}, {}
        for t in range(1, self.T + 1):
            a = {}
            for table, root in self.roots[t].items():
                x = hidden[table][:root.n_dst]
                a[table] = x @ root.weight.T if root.kind == 'linear' else x.clone()
                if root.bias is not None:
                    a[table] = a[table] + root.bias
            for relation in self.relations[t]:
                src, _, dst = relation.relation
                index = relation.edge_index
                message = hidden[src][index[0]] @ relation.weight.T
                if relation.operator == 'mean' and index.size(1):
                    counts = torch.bincount(index[1], minlength=relation.n_dst).to(F64)
                    message = message / counts[index[1]].unsqueeze(1)
                a[dst] = a[dst].index_add(0, index[1], message)
                if relation.bias is not None:
                    a[dst] = a[dst] + relation.bias
            pres[t] = a
            out, masks = {}, {}
            for table, values in a.items():
                out[table], masks[table] = apply(values, self.pre_activation[t][table])
            hiddens[t] = hidden = out
            masks_by_layer[t] = masks if relu else None
        table, seed = self.seed
        x = hidden[table][seed]
        head_pre, head_masks = [], []
        for op in self.head:
            a = op.weight[table] @ x
            if op.bias[table] is not None:
                a = a + op.bias[table]
            head_pre.append(a)
            if op.activation is not None:
                mask = op.pre_activation[table][seed] > 0
                x = a * mask.to(F64)
                head_masks.append(mask)
            else:
                x = a
                head_masks.append(None)
        return PieceForward(
            activation=self.activation, preprocess_masks=tuple(mask_list),
            preprocess_pre_activation=tuple(pre_list), preprocess_hidden=tuple(hidden_list),
            masks=masks_by_layer, pre_activation=pres, hidden=hiddens,
            head_masks=tuple(head_masks), head_pre_activation=tuple(head_pre), z_piece=x)


__all__ = [
    'F64', 'KINDS', 'STRUCTURAL_CODES', 'GraphCapture', 'Incoming', 'LinearOp',
    'PieceForward', 'RelationOp', 'RootOp', 'Route', 'WalkComputation',
    'architecture_record', 'capture_numerics', 'float32_matmul_precision',
    'mkldnn_matmul_fp32_precision', 'normalized_mkldnn', 'operations_record',
    'structural_qualification',
]
