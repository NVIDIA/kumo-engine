# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`hetero_gnn_gi_degree_t_v1`: exact degree-T decomposition by nested GNN-GI.

Explained value and variables (contract section 6). The method explains
`f = s * z`, the selected head output before calibration (`s = +1` for
regression). The encoder output `H_0` and every pre-message-passing output
are held fixed. The variables are gates: one per message-passing layer,
relation and retained edge occurrence, and one per layer and destination
occurrence for the self-update (root skip) branch. Mean denominators stay
frozen at the retained in-degrees, and the model is the point where every
gate equals 1. Head layers carry no gates, so the degree is the number of
message-passing layers `T`.

Computation. With the captured ReLU masks held fixed, `f` is the
multilinear polynomial `sum_W R_W prod_t lambda[t, v_t]`, and a walk score is
the mixed partial derivative `R_W = s * g^T D_T M_T ... D_1 M_1 h_J` (paper
Eq. 7, Prop. 2). It is evaluated in float64 as backward coefficient vectors:
`start` folds `s` into the selected head row, `transfer` applies the
destination mask and the step matrix (`W / d` for frozen means, `W` for sums
and single-edge gathers, the root weight or the identity for self updates),
and `finish` propagates through preprocessing and multiplies by the
source's `H_0` row. Captured intermediate activations supply masks only and
never enter scores. The prefix mass of an unexpanded state is `c . H~`, so
walks and prefix masses add up to the signed piece value `f_piece`.

Exactness. The claim (`exact_degree_t`) requires a module that passes
`qualify`, complete enumeration, an interior piece (no relevant ReLU at an
exactly zero captured pre-activation with a nonzero input, and captured
masks that agree with the float64 recomputation) and a precision residual
within the contract bound. Otherwise rows are `piecewise_gnn_gi`: the walks
still sum to `f_piece`, but are not claimed to be the unique degree-T
coefficients. The claim is an identity about the qualified model and these
graph variables only; it is not a raw-column expansion, does not cover
biased models and is not evidence of faithfulness.
"""
import math

import torch
from torch_geometric.nn import MLP
from torch_geometric.nn.dense.linear import Linear as PygLinear

from kumo_core.explain.walk_attribution import (
    DEGREE_T_METHOD,
    UnsupportedWalkAttribution,
    WalkAttributionConfig,
)
from kumo_core.explain.walk_enumerate import NONE
from kumo_core.explain.walk_graph import (
    F64,
    KINDS,
    STRUCTURAL_CODES,
    structural_qualification,
)
from kumo_core.explain.walk_schema import (
    WalkSchemaError,
    precision_limit,
    tolerance,
)
from kumo_core.modeling.gnn.hetero import (
    ForeignKeyToPrimaryKeyConv,
    HeteroIdentity,
    HeteroLinear,
    HeteroPostLayer,
    PrimaryKeyToForeignKeyConv,
)

METHOD_ID = DEGREE_T_METHOD
#: Contract 4.2: GNN-GI has no gamma rule, bias rule or stabilizer.
SETTINGS = {'gamma': None, 'gamma_rule': None, 'bias_rule': None, 'epsilon': None,
            'stabilizer': None, 'identity_root_rule': None,
            'sign_folding': 'selected_output_row'}
#: Codes `qualify` adds to the shared structural codes.
QUALIFY_CODES = ('bias_free_gnn', 'bias_free_head')
#: Row codes `summarize` reports when they hold. The row builder adds
#: `single_edge_gathers`, which every built computation satisfies.
ROW_CODES = ('frozen_mean_denominators', 'complete_enumeration', 'interior_piece',
             'tolerances_met')
#: Every qualification code a degree-T claim requires (contract 4.9).
CLAIM_CODES = STRUCTURAL_CODES + QUALIFY_CODES + ('single_edge_gathers', ) + ROW_CODES
#: Reasons a GNN-GI row withholds the claim, in contract order.
REASONS = ('enumeration_incomplete', 'boundary_piece', 'mask_disagreement',
           'precision_residual_exceeded')
DIAGNOSTIC_KEYS = ('f_piece', 'z_piece', 'claim_tolerance', 'tolerance_mode',
                   'gated_neurons')


def _unsupported(reason, detail):
    return UnsupportedWalkAttribution(f'{reason}: {detail}')


# Qualification (before hooks) ###############################################

def _check_config(config):
    if not isinstance(config, WalkAttributionConfig) or config.method != METHOD_ID:
        raise _unsupported('config_not_gnn_gi', f'{METHOD_ID} needs its own configuration')
    if config.gamma != 0.0 or config.epsilon != 1e-9:
        raise _unsupported('config_not_gnn_gi', 'GNN-GI has no gamma rule or stabilizer')


def _check_layout(model):
    layers = len(model.hetero_convs)
    if layers < 1:
        raise _unsupported('unsupported_model', 'GNN-GI needs a message-passing layer')
    if (model.num_mp_layers != layers or len(model.root_skips) != layers
            or len(model.post_convs) != layers or len(model.lins) != len(model.post_lins)
            or len(model.lins) != model.num_pre_mp_layers):
        raise _unsupported('unsupported_model', 'inconsistent layer lists')


def _check_post_layers(model):
    """Post layers must apply exactly the declared activation, then dropout."""
    relu = model.act is not None
    expected = [torch.nn.ReLU, torch.nn.Dropout] if relu else [torch.nn.Dropout]
    for kind, posts in (('preprocessing', model.post_lins),
                        ('message-passing', model.post_convs)):
        for index, post in enumerate(posts):
            if type(post) is not HeteroPostLayer:
                raise _unsupported('unregistered_operator',
                                   f'{kind} post layer {index} is {type(post).__name__}')
            for table, ops in post.layer_dict.items():
                kinds = [type(op) for op in ops]
                if kinds != expected:
                    reason = ('activation' if set(kinds) <= {torch.nn.ReLU, torch.nn.Dropout}
                              else 'unregistered_operator')
                    raise _unsupported(reason, f'{kind} post layer {index} of {table!r} applies '
                                       f'{[k.__name__ for k in kinds]}, model act {model.act!r}')
                if relu and ops[0].inplace:
                    raise _unsupported('unregistered_operator',
                                       'an in-place ReLU overwrites captured pre-activations')


def _check_operators(model, head):
    _check_post_layers(model)
    for index, layer in enumerate(model.lins):
        if type(layer) is not HeteroLinear or any(
                type(lin) is not torch.nn.Linear for lin in layer.lin_dict.values()):
            raise _unsupported('unregistered_operator', f'preprocessing layer {index}')
    for index, root in enumerate(model.root_skips):
        if type(root) is HeteroLinear:
            if any(type(lin) is not torch.nn.Linear for lin in root.lin_dict.values()):
                raise _unsupported('unregistered_operator', f'root skip {index}')
        elif type(root) is not HeteroIdentity:
            raise _unsupported('unregistered_operator', f'root skip {index}')
    for index, convs in enumerate(model.hetero_convs):
        for relation, conv in convs.items():
            if (type(conv) not in (ForeignKeyToPrimaryKeyConv, PrimaryKeyToForeignKeyConv)
                    or type(conv.lin) is not torch.nn.Linear):
                raise _unsupported('unregistered_operator',
                                   f'relation {tuple(relation)} at layer {index + 1}')
    mlp = head.mlp
    if (type(mlp) is not MLP or mlp.plain_last is not True
            or any(type(lin) is not PygLinear for lin in mlp.lins)):
        raise _unsupported('unregistered_operator', 'the readout must be a plain PyG MLP')
    if (mlp.act is None) != (head.act is None):
        raise _unsupported('activation', f'readout act {head.act!r} disagrees with its MLP')
    if mlp.act is not None and mlp.act.inplace:
        raise _unsupported('unregistered_operator', 'in-place readout activation')
    if getattr(head, 'pred_time_encoder', None) is not None:
        raise _unsupported('prediction_time_encoding', 'the readout holds a time encoder')


def _check_biases(model, head):
    """No additive bias anywhere between `H_0` and `z` (actual parameters).

    Convolution `bias` flags are read by `empty_output`, so they must agree
    with the parameters. The model and head flags are construction records
    the forward never reads; the parameters decide.
    """
    for index, layer in enumerate(model.lins):
        for table, lin in layer.lin_dict.items():
            if lin.bias is not None:
                raise _unsupported('gnn_bias_present',
                                   f'preprocessing layer {index} of {table!r}')
    for index, convs in enumerate(model.hetero_convs):
        for relation, conv in convs.items():
            present = conv.lin.bias is not None
            if bool(conv.bias) != present:
                raise _unsupported(
                    'inconsistent_bias_configuration',
                    f'relation {tuple(relation)} at layer {index + 1} has bias flag '
                    f'{conv.bias!r} but {"a" if present else "no"} bias parameter')
            if present:
                raise _unsupported('gnn_bias_present',
                                   f'relation {tuple(relation)} at layer {index + 1}')
        root = model.root_skips[index]
        if type(root) is HeteroLinear:
            for table, lin in root.lin_dict.items():
                if lin.bias is not None:
                    raise _unsupported('gnn_bias_present',
                                       f'root skip of {table!r} at layer {index + 1}')
    for index, lin in enumerate(head.mlp.lins):
        if lin.bias is not None:
            raise _unsupported('head_bias_present', f'readout linear {index}')


def _check_inventory(model, head):
    """Fail closed: the modules may hold exactly the weights the typed capture
    represents, and no buffers, so nothing else can silently enter `f`."""
    allowed = {id(lin.weight) for layer in model.lins for lin in layer.lin_dict.values()}
    allowed |= {id(conv.lin.weight) for convs in model.hetero_convs for conv in convs.values()}
    allowed |= {id(lin.weight) for root in model.root_skips if type(root) is HeteroLinear
                for lin in root.lin_dict.values()}
    for prefix, part, expected in (('model', model, allowed),
                                   ('head', head, {id(lin.weight) for lin in head.mlp.lins})):
        extra = [name for name, param in part.named_parameters() if id(param) not in expected]
        if extra:
            raise _unsupported('unregistered_parameter', f'{prefix}.{extra[0]}')
        buffers = [name for name, _ in part.named_buffers()]
        if buffers:
            raise _unsupported('unexpected_buffer', f'{prefix}.{buffers[0]}')


def qualify(module, config):
    """Fail-closed GNN-GI qualification of `module`, before any hook.

    Starts from the shared structural profile and adds the GNN-GI checks on
    actual modules and parameters (never plans, options or saved configs):
    at least one layer, consistent layer lists, post layers that apply
    exactly the declared ReLU or identity, registered linear types, a plain
    PyG readout without a time encoder, no bias parameter in any
    pre-message-passing, message-passing or readout linear, convolution bias
    flags that agree with their parameters, and no parameter or buffer
    beyond the captured weights. Encoder internals (including biases) are
    fixed features and are not inspected. Never mutates the module.
    Raises `UnsupportedWalkAttribution('<reason>: <detail>')`.
    """
    _check_config(config)
    codes = list(structural_qualification(module))
    model, head = module.model, module.head
    _check_layout(model)
    _check_operators(model, head)
    _check_biases(model, head)
    _check_inventory(model, head)
    return codes + list(QUALIFY_CODES)


# Row-level checks ###########################################################

def _finite(tensor):
    return bool(torch.isfinite(tensor).all())


def _check_selection(computation, selection):
    sign, index = selection.logit_sign, selection.output_index
    if type(sign) is not int or sign not in (1, -1):
        raise ValueError('logit_sign must be +1 or -1')
    if type(index) is not int or not 0 <= index < computation.head_output.numel():
        raise ValueError('Output index outside the captured head output')


def _check_finite(computation):
    """A nonfinite captured value makes the row fail (contract 4.2)."""
    tensors = [computation.head_output]
    for op in computation.preprocessing + computation.head:
        tensors += list(op.weight.values())
    for t in range(1, computation.T + 1):
        tensors += [op.weight for op in computation.relations[t]]
        tensors += [root.weight for root in computation.roots[t].values()
                    if root.weight is not None]
        tensors += [computation.pre_activation[t][table][occurrence]
                    for table, occurrence in computation.relevant_nodes(t)]
    for table, occurrence in computation.relevant_nodes(0):
        tensors.append(computation.h0[table][occurrence])
        tensors += [op.pre_activation[table][occurrence] for op in computation.preprocessing]
    table, seed = computation.seed
    tensors += [op.pre_activation[table][seed] for op in computation.head]
    if not all(_finite(tensor) for tensor in tensors):
        raise ValueError('Nonfinite captured value in the explained computation')


def _check_incoming(computation, t, node):
    """Routes into `node` at layer t against the retained edges (6.1).

    Every relation route count must equal the retained edges into the node,
    every mean denominator the frozen retained in-degree, and every other
    denominator 1; a gather may have at most one edge.
    """
    table, occurrence = node
    incoming = computation.incoming(t, node)
    if incoming.biases:
        raise _unsupported('biases_present', f'bias added at {node} in layer {t}')
    routes = incoming.routes
    selves = [route for route in routes if route.kind == 'self_update']
    if len(selves) != 1 or routes[0] is not selves[0]:
        raise ValueError(f'{node} at layer {t} needs exactly one leading self update')
    counts = {}
    for route in routes:
        if route.layer != t or route.destination != node:
            raise ValueError(f'Route outside {node} at layer {t}')
        if route.kind == 'self_update':
            if (route.source != node or route.denominator != 1
                    or route.operator not in ('linear', 'identity')):
                raise ValueError(f'Invalid self update at {node}, layer {t}')
        elif route.kind == 'relation':
            counts[route.relation] = counts.get(route.relation, 0) + 1
        else:
            raise ValueError(f'Unknown route kind {route.kind!r}')
    seen = set()
    for op in computation.relations[t]:
        if op.relation[2] != table:
            continue
        seen.add(op.relation)
        retained = int((op.edge_index[1] == occurrence).sum())
        if counts.get(op.relation, 0) != retained:
            raise ValueError(f'Routes of {op.relation} differ from its retained edges')
        if op.operator == 'gather' and retained > 1:
            raise _unsupported('multi_edge_gather', f'{op.relation} into {node} at layer {t}')
        expected = retained if op.operator == 'mean' else 1
        for route in routes:
            if route.relation == op.relation and (route.operator != op.operator
                                                  or route.denominator != expected):
                raise ValueError(f'{op.relation} at layer {t} is not frozen at the '
                                 'retained in-degree')
    if set(counts) - seen:
        raise ValueError(f'Routes into {node} name relations that were not called')


def _check_computation(computation, selection):
    if computation.gnn_bias_present or computation.head_bias_present:
        raise _unsupported('biases_present', 'the captured computation holds additive biases')
    _check_selection(computation, selection)
    _check_finite(computation)
    for t in range(1, computation.T + 1):
        for node in computation.relevant_nodes(t):
            _check_incoming(computation, t, node)


def _check_piece(computation, piece):
    """The piece applies the captured masks, and `H~ = D * a~` exactly."""
    if piece.activation != computation.activation:
        raise ValueError('The piece and the capture disagree on the activation')
    for t in range(1, computation.T + 1):
        for table, occurrence in computation.relevant_nodes(t):
            pre = piece.pre_activation[t][table][occurrence]
            expected = pre
            if computation.activation is not None:
                mask = piece.masks[t][table][occurrence]
                if not torch.equal(mask,
                                   computation.pre_activation[t][table][occurrence] > 0):
                    raise ValueError('The piece does not apply the captured mask')
                expected = pre * mask.to(F64)
            if not _finite(pre) or not torch.equal(piece.hidden[t][table][occurrence],
                                                   expected):
                raise ValueError('The piece output is not the masked pre-activation')
    if not _finite(piece.z_piece):
        raise ValueError('Nonfinite piece output')


def _checked_piece(computation, selection, config):
    _check_config(config)
    _check_computation(computation, selection)
    piece = computation.forward_piece()
    _check_piece(computation, piece)
    return piece


# The rule ###################################################################

def _step(route, gated):
    """`M^T g` for one route: frozen mean `W / d`, sum and gather `W`, root
    `W` or the identity."""
    if route.operator == 'identity':
        return gated
    child = route.weight.T @ gated
    return child / route.denominator if route.denominator != 1 else child


class DegreeTRule:
    """GNN-GI as a `WalkRule` with float64 backward coefficient vectors.

    A state at `(t, K)` is the row vector `c` of `H[t, K]` on the captured
    piece: every completion of the prefix receives, in total, `c . H~[t, K]`.
    States are never modified in place, so children may share tensors.
    """

    def __init__(self, computation, selection, piece):
        self.computation = computation
        self.piece = piece
        self.sign = selection.logit_sign
        self.output_index = selection.output_index
        self._masks = {}

    def _mask(self, layer, node):
        masks = self.piece.masks[layer]
        if masks is None:
            return None
        table, occurrence = node
        if (layer, table) not in self._masks:
            self._masks[(layer, table)] = masks[table].to(F64)
        return self._masks[(layer, table)][occurrence]

    def start(self):
        """Fold `s` into the selected row of the last head linear, then
        propagate through the hidden head layers (no gates, no change of T)."""
        computation, piece = self.computation, self.piece
        table, _ = computation.seed
        head = computation.head
        coefficient = self.sign * head[-1].weight[table][self.output_index]
        for j in range(len(head) - 2, -1, -1):
            mask = piece.head_masks[j]
            gated = coefficient if mask is None else coefficient * mask.to(F64)
            coefficient = head[j].weight[table].T @ gated
        return coefficient, NONE

    def transfer(self, layer, destination, state, incoming):
        mask = self._mask(layer, destination)
        gated = state if mask is None else state * mask
        children, shared = [], {}
        for route in incoming.routes:
            # Routes of one relation into one destination share `M^T g`.
            key = (route.operator, id(route.weight), route.denominator)
            if key not in shared:
                shared[key] = _step(route, gated)
            children.append(shared[key])
        return children, NONE

    def prefix_mass(self, layer, destination, state):
        table, occurrence = destination
        return float(state @ self.piece.hidden[layer][table][occurrence])

    def finish(self, source, state):
        """Propagate through bias-free preprocessing; the walk relevance is
        the sum of `b * H_0[J]` (exactly 0 on inactive input channels)."""
        computation, piece = self.computation, self.piece
        table, occurrence = source
        coefficient = state
        for p in range(len(computation.preprocessing) - 1, -1, -1):
            masks = piece.preprocess_masks[p]
            gated = (coefficient if masks is None
                     else coefficient * masks[table][occurrence].to(F64))
            coefficient = computation.preprocessing[p].weight[table].T @ gated
        vector = coefficient * computation.h0[table][occurrence]
        return math.fsum(vector.tolist()), vector, NONE


def make_rule(computation, selection, config):
    """The GNN-GI rule of one captured computation.

    Biases or a multi-edge gather make the row unsupported; a capture
    inconsistency (routes that differ from the retained edges, a mean not
    frozen at the retained in-degree, a piece that does not apply the
    captured masks, a nonfinite value) fails the row.
    """
    piece = _checked_piece(computation, selection, config)
    return DegreeTRule(computation, selection, piece)


# Exactness ##################################################################

class _Neurons:
    """Counts and per-layer margins of relevant gated neurons of one kind."""

    def __init__(self, layers):
        self.relevant = self.boundary = self.disagreements = 0
        self.margins = [None] * layers

    def add(self, layer, captured, piece, input_nonzero):
        self.relevant += captured.numel()
        zero = captured == 0  # also -0.0
        if input_nonzero:
            self.boundary += int(zero.sum())
        flipped = ~zero & (((captured > 0) != (piece > 0)) | (piece == 0))
        self.disagreements += int(flipped.sum())
        low = float(piece.abs().min())
        current = self.margins[layer]
        self.margins[layer] = low if current is None else min(current, low)

    def record(self):
        return {'relevant': self.relevant, 'boundary': self.boundary,
                'mask_disagreements': self.disagreements,
                'min_abs_preactivation_by_layer': list(self.margins)}


def gated_neurons(computation, piece):
    """Relevant gated neurons by kind (contract 6.3, review finding m3).

    Relevant: every channel of each (layer, occurrence) from which the seed
    is reachable, the head hidden neurons and the preprocessing neurons of
    walk sources; only ReLU layers gate. A boundary neuron has an exactly
    zero captured pre-activation while some incoming source vector (for head
    and preprocessing neurons, the input vector) is nonzero in the piece;
    zero source vectors are judged one layer lower. A disagreement is a
    nonzero captured pre-activation whose float64 piece pre-activation has
    the opposite sign or is exactly 0. Margins are `min |a~|`.
    """
    T = computation.T
    table, seed = computation.seed
    head = computation.head
    neurons = {'preprocessing': _Neurons(len(computation.preprocessing)),
               'message_passing': _Neurons(T),
               'head': _Neurons(len(head) - 1)}
    if computation.activation is not None:
        for t in range(1, T + 1):
            below = piece.hidden[t - 1]
            for node in computation.relevant_nodes(t):
                nonzero = any(bool((below[src][occ] != 0).any())
                              for src, occ in (route.source for route in
                                               computation.incoming(t, node).routes))
                neurons['message_passing'].add(
                    t - 1, computation.pre_activation[t][node[0]][node[1]],
                    piece.pre_activation[t][node[0]][node[1]], nonzero)
        for p, op in enumerate(computation.preprocessing):
            inputs = computation.h0 if p == 0 else piece.preprocess_hidden[p - 1]
            for source, occurrence in computation.relevant_nodes(0):
                neurons['preprocessing'].add(
                    p, op.pre_activation[source][occurrence],
                    piece.preprocess_pre_activation[p][source][occurrence],
                    bool((inputs[source][occurrence] != 0).any()))
    for j, op in enumerate(head[:-1]):
        if op.activation is None:
            continue
        inputs = (piece.hidden[T][table][seed] if j == 0 else
                  piece.head_pre_activation[j - 1] * piece.head_masks[j - 1].to(F64))
        neurons['head'].add(j, op.pre_activation[table][seed], piece.head_pre_activation[j],
                            bool((inputs != 0).any()))
    return neurons


def tolerance_mode(numerics):
    """Precision mode of the forward that produced the score: the replay
    mode for offline rows, else the capture mode (None if unrecorded)."""
    numerics = numerics or {}
    replay = numerics.get('replay_precision_mode')
    return replay if replay is not None else numerics.get('precision_mode')


def claim_tolerance(mode, score):
    """Contract 4.10 bound on `|precision_residual|` for a degree-T claim:
    `1e-5` under `highest`, else `1e-2` (a sanity bound), times
    `max(1, |score|)`."""
    return (1e-5 if mode == 'highest' else 1e-2) * max(1.0, abs(score))


def summarize(computation, selection, config, enumeration):
    """Method settings, precision residual, exactness and diagnostics.

    `precision_residual = score - f_piece` with `f_piece = s * z_piece`; the
    walks and prefix masses sum to `f_piece` whatever the masks or budget.
    `exactness.qualification` holds the row codes only (the builder adds
    the `qualify` codes and `single_edge_gathers`).
    """
    piece = _checked_piece(computation, selection, config)
    k, sign = selection.output_index, selection.logit_sign
    score = sign * float(computation.head_output[k])
    z_piece = float(piece.z_piece[k])
    f_piece = sign * z_piece
    residual = score - f_piece
    mode = tolerance_mode(computation.numerics)
    limit = claim_tolerance(mode, score)
    exceeded = abs(residual) > limit
    neurons = gated_neurons(computation, piece)
    relevant = sum(n.relevant for n in neurons.values())
    boundary = sum(n.boundary for n in neurons.values())
    disagreements = sum(n.disagreements for n in neurons.values())
    margins = [m for n in neurons.values() for m in n.margins if m is not None]
    interior = boundary == 0 and disagreements == 0
    conditions = {'enumeration_incomplete': not enumeration.complete,
                  'boundary_piece': boundary > 0, 'mask_disagreement': disagreements > 0,
                  'precision_residual_exceeded': exceeded}
    reasons = [reason for reason in REASONS if conditions[reason]]
    codes = ['frozen_mean_denominators']
    codes += ['complete_enumeration'] if enumeration.complete else []
    codes += ['interior_piece'] if interior else []
    codes += ['tolerances_met'] if not exceeded else []
    return {
        'method_settings': dict(SETTINGS),
        'precision_residual': residual,
        'exactness': {
            'decomposition': 'piecewise_gnn_gi' if reasons else 'exact_degree_t',
            'degree_t_claim': not reasons, 'reasons': reasons,
            'piece': 'interior' if interior else 'boundary',
            'boundary_neurons': boundary, 'mask_disagreements': disagreements,
            'relevant_gated_neurons': relevant,
            'min_abs_gated_preactivation': min(margins) if margins else None,
            'qualification': codes},
        'diagnostics': {
            'f_piece': f_piece, 'z_piece': z_piece, 'claim_tolerance': limit,
            'tolerance_mode': mode,
            'gated_neurons': {kind: neurons[kind].record() for kind in KINDS}},
    }


# Row validation #############################################################

def _require(condition, message):
    if not condition:
        raise WalkSchemaError(message)


def _absolute_mass(row):
    ledger = row['ledger']
    return math.fsum(
        [abs(walk['relevance']) for walk in row['walks']]
        + [ledger['omitted_display_abs_relevance'], ledger['omitted_search_abs_prefix_mass'],
           abs(ledger['precision_residual'])])


def _validate_diagnostics(row, limit):
    diagnostics = (row.get('diagnostics') or {}).get(METHOD_ID)
    if not isinstance(diagnostics, dict) or set(diagnostics) != set(DIAGNOSTIC_KEYS):
        raise WalkSchemaError('GNN-GI rows carry their diagnostics')
    exactness, ledger = row['exactness'], row['ledger']
    binary = row['output']['binary']
    sign = 1 if binary is None else binary['logit_sign']
    _require(type(diagnostics['z_piece']) is float
             and diagnostics['f_piece'] == sign * diagnostics['z_piece'],
             'f_piece must be the signed z_piece (m1)')
    _require(abs(diagnostics['f_piece'] - (row['score'] - ledger['precision_residual']))
             <= tolerance(_absolute_mass(row)), 'precision_residual must be score - f_piece')
    numerics = row['numerics']
    _require(diagnostics['tolerance_mode'] == tolerance_mode(numerics)
             and diagnostics['claim_tolerance'] == limit,
             'The claim tolerance must follow the precision mode of the row')
    neurons = diagnostics['gated_neurons']
    _require(isinstance(neurons, dict) and set(neurons) == set(KINDS), 'Gated neuron kinds')
    margins = []
    for kind in KINDS:
        entry = neurons[kind]
        _require(all(type(entry[name]) is int and entry[name] >= 0
                     for name in ('relevant', 'boundary', 'mask_disagreements')),
                 'Gated neuron counts')
        layers = entry['min_abs_preactivation_by_layer']
        _require(isinstance(layers, list) and all(
            m is None or (type(m) is float and m >= 0) for m in layers), 'Layer margins')
        margins += [m for m in layers if m is not None]
    _require(len(neurons['message_passing']['min_abs_preactivation_by_layer'])
             == row['architecture']['message_passing_layers'], 'One margin per layer')
    for name, total in (('relevant', 'relevant_gated_neurons'),
                        ('boundary', 'boundary_neurons'),
                        ('mask_disagreements', 'mask_disagreements')):
        _require(sum(neurons[kind][name] for kind in KINDS) == exactness[total],
                 f'Gated neuron {name} counts must add up to the exactness record')
    _require((min(margins) if margins else None) == exactness['min_abs_gated_preactivation'],
             'The margin must be the minimum layer margin')


def validate_row(row):
    """GNN-GI checks on top of the generic v2 validator (raises WalkSchemaError).

    Settings without gamma or stabilizer, zero bias and stabilizer items, a
    bias-free architecture, reasons in contract order that follow the row
    facts, row qualification codes that follow them too, every claim code
    behind a claim, consistent diagnostics, and bags that sum to `f_piece`.
    """
    _require(row['method'] == METHOD_ID, f'Not a {METHOD_ID} row')
    _require(row['method_settings'] == SETTINGS,
             'GNN-GI has no gamma rule, bias rule or stabilizer')
    config = WalkAttributionConfig(**row['config'])
    _require(config.gamma == 0.0 and config.epsilon == 1e-9, 'GNN-GI configuration')
    ledger = row['ledger']
    for name in ('bias_relevance', 'stabilizer_residual'):
        _require(ledger[name] == 0.0 and all(
            value == 0.0 for value in ledger[f'{name}_by_kind'].values()),
            f'GNN-GI rows have no {name}')
    architecture = row['architecture']
    _require(architecture['gnn_bias'] is False and architecture['head_bias'] is False,
             'GNN-GI rows explain bias-free models only')
    exactness = row['exactness']
    reasons = exactness['reasons']
    _require(reasons == [reason for reason in REASONS if reason in reasons],
             'GNN-GI reasons must be unique, known and in contract order')
    complete = row['coverage']['enumeration_complete']
    _require(('enumeration_incomplete' in reasons) == (not complete),
             'enumeration_incomplete must follow the coverage')
    _require(exactness['boundary_neurons'] + exactness['mask_disagreements']
             <= exactness['relevant_gated_neurons'], 'More flagged than relevant neurons')
    limit = precision_limit(row)
    exceeded = abs(ledger['precision_residual']) > limit
    _require(('precision_residual_exceeded' in reasons) == exceeded,
             'precision_residual_exceeded must follow the claim tolerance')
    codes = exactness['qualification']
    _require(len(set(codes)) == len(codes), 'Duplicate qualification codes')
    _require('frozen_mean_denominators' in codes and 'single_edge_gathers' in codes,
             'GNN-GI rows verify frozen means and single-edge gathers')
    _require(('complete_enumeration' in codes) == complete
             and ('interior_piece' in codes) == (exactness['piece'] == 'interior')
             and ('tolerances_met' in codes) == (not exceeded),
             'Row qualification codes must follow the row facts')
    if exactness['degree_t_claim']:
        missing = [code for code in CLAIM_CODES if code not in codes]
        _require(not missing, f'A degree-T claim needs every qualification code: {missing}')
    _validate_diagnostics(row, limit)
    if row['bags'] is not None:
        total = math.fsum(bag['relevance'] for bag in row['bags'])
        _require(abs(total - (row['score'] - ledger['precision_residual']))
                 <= tolerance(_absolute_mass(row)), 'Bags must sum to f_piece')


__all__ = [
    'CLAIM_CODES', 'DegreeTRule', 'METHOD_ID', 'QUALIFY_CODES', 'REASONS', 'ROW_CODES',
    'SETTINGS', 'claim_tolerance', 'gated_neurons', 'make_rule', 'qualify', 'summarize',
    'tolerance_mode', 'validate_row',
]
