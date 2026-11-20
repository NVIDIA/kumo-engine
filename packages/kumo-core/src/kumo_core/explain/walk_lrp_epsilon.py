# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned LRP-epsilon splits, not registered walk methods.

Canonical v1 uses float64, epsilon=1e-9 and pre-stabilization cancellation
K=sum(abs(terms))/abs(sum(terms)), including biases and explicit constants.
K>100 rejects the entire destination relevance. A nonempty cancellation to
zero has K=inf; an all-zero or empty group has K=0 and allocates relevance
to stabilization. Epsilon takes the total's sign, positive at either zero.
The smooth_floor_v1 policy uses c_i=z_i*S/(S*S+q*q), q=A/100+epsilon,
and stabilization R*q*q/(S*S+q*q), with one split across fitted encoder operands.
Its local operand coefficient L1 is at most 50, excluding stabilization.
It is continuous at zero; absolute-value kinks and global sensitivity remain.
"""
from dataclasses import dataclass

import torch

POLICY_ID = 'canonical_lrp_epsilon_v1'
SMOOTH_FLOOR_POLICY_ID = 'smooth_floor_v1'
EPSILON = 1e-9
CANCELLATION_LIMIT = 100.0


def validate_policy_id(policy_id):
    if policy_id not in (POLICY_ID, SMOOTH_FLOOR_POLICY_ID):
        raise ValueError(f'Unknown LRP policy: {policy_id!r}')


@dataclass(frozen=True)
class EpsilonSplit:
    """On-device allocations per output; contribution matrices retain inputs.

    Sum each contribution matrix over outputs to obtain child relevance.
    Rejected destinations must invalidate their dependent routes at integration.
    """
    contributions: tuple[torch.Tensor, ...]
    biases: tuple[torch.Tensor, ...]
    constants: tuple[torch.Tensor, ...]
    stabilization: torch.Tensor
    rejected: torch.Tensor
    residual: torch.Tensor
    rejection_mask: torch.Tensor
    cancellation: torch.Tensor
    policy_id: str = POLICY_ID


@torch.no_grad()
def canonical_epsilon_split(contributions, biases, relevance, *, constants=(),
                            policy_id=POLICY_ID):
    """Split finite same-device float64 tensors, refusing malformed arithmetic.

    Contributions have shape [inputs, outputs]; all other inputs are [outputs].
    CPU and CUDA are admitted by this primitive; no device transfer occurs.
    Nonfinite inputs, totals or allocations raise ValueError, even for zero R.
    Numerical residual is R minus child, bias, constant, stabilization and
    rejected allocations, per output. It is not folded into stabilization.
    """
    validate_policy_id(policy_id)
    if not isinstance(relevance, torch.Tensor) or relevance.ndim != 1:
        raise ValueError('relevance must be a one-dimensional tensor')
    if relevance.dtype != torch.float64 or relevance.device.type not in (
            'cpu', 'cuda'):
        raise ValueError('relevance requires float64 on CPU or CUDA')
    contributions, biases, constants = tuple(contributions), tuple(
        biases), tuple(constants)
    groups = ((contributions, 2), (biases, 1), (constants, 1), ((relevance, ),
                                                                1))
    for tensors, ndim in groups:
        for tensor in tensors:
            if (not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim
                    or tensor.shape[-1] != relevance.numel()
                    or tensor.dtype != relevance.dtype
                    or tensor.device != relevance.device
                    or tensor.layout != torch.strided):
                raise ValueError(
                    'terms require matching shape, float64 dtype and device')
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('nonfinite LRP input')

    total, mass = torch.zeros_like(relevance), torch.zeros_like(relevance)
    for tensor in contributions:
        total += tensor.sum(0)
        mass += tensor.abs().sum(0)
    for tensor in (*biases, *constants):
        total += tensor
        mass += tensor.abs()
    if not bool((torch.isfinite(total) & torch.isfinite(mass)).all()):
        raise ValueError('nonfinite LRP total or absolute mass')
    nonzero = total != 0
    cancellation = mass / torch.where(nonzero, total.abs(),
                                      torch.ones_like(total))
    cancellation = torch.where(~nonzero & (mass > 0),
                               torch.full_like(total, float('inf')),
                               cancellation)
    reject = cancellation > CANCELLATION_LIMIT
    signed_epsilon = torch.where(total >= 0, torch.full_like(total, EPSILON),
                                 torch.full_like(total, -EPSILON))
    denominator = total + signed_epsilon
    if policy_id == SMOOTH_FLOOR_POLICY_ID:
        floor = mass / CANCELLATION_LIMIT + EPSILON
        denominator = torch.maximum(total.abs(), floor)
        reject = torch.zeros_like(reject)
    if not bool(torch.isfinite(denominator).all()):
        raise ValueError('nonfinite stabilized LRP denominator')
    # A benign denominator and zero relevance avoid invalid rejected arithmetic.
    denominator = torch.where(reject, torch.ones_like(total), denominator)
    admitted = torch.where(reject, torch.zeros_like(relevance), relevance)
    if policy_id == SMOOTH_FLOOR_POLICY_ID:
        # Squaring normalized ratios avoids overflow and underflow in S*S.
        scaled_total, scaled_floor = total / denominator, floor / denominator
        norm = scaled_total.square() + scaled_floor.square()
        reciprocal = scaled_total / norm
        child_shares = tuple(
            (z / denominator * reciprocal) * admitted for z in contributions)
        bias_shares = tuple(
            (b / denominator * reciprocal) * admitted for b in biases)
        constant_shares = tuple(
            (c / denominator * reciprocal) * admitted for c in constants)
    else:
        child_shares = tuple(
            (torch.where(reject, torch.zeros_like(z), z) / denominator) *
            admitted for z in contributions)
        bias_shares = tuple(
            (torch.where(reject, torch.zeros_like(b), b) / denominator) *
            admitted for b in biases)
        constant_shares = tuple(
            (torch.where(reject, torch.zeros_like(c), c) / denominator) *
            admitted for c in constants)
    stabilization = torch.where(reject, torch.zeros_like(total),
                                signed_epsilon / denominator) * admitted
    if policy_id == SMOOTH_FLOOR_POLICY_ID:
        stabilization = (scaled_floor.square() / norm) * admitted
    rejected = torch.where(reject, relevance, torch.zeros_like(relevance))
    allocated = stabilization + rejected
    for share in child_shares:
        allocated = allocated + share.sum(0)
    for share in (*bias_shares, *constant_shares):
        allocated = allocated + share
    residual = relevance - allocated
    outputs = (*child_shares, *bias_shares, *constant_shares, stabilization,
               rejected, residual)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in outputs):
        raise ValueError('nonfinite LRP allocation')
    return EpsilonSplit(child_shares, bias_shares, constant_shares,
                        stabilization, rejected, residual, reject, cancellation,
                        policy_id)


@dataclass(frozen=True)
class RelevanceOperation:
    """Auditable split at one captured output occurrence.

    Contribution matrices and routes have matching order. Rejected output
    coordinates have zero child allocations on every dependent route.
    ``invalid_output`` also includes rejected dependencies from later layers.
    """
    kind: str
    index: int
    node: tuple
    routes: tuple
    relevance: torch.Tensor
    contributions: tuple
    biases: tuple
    constants: tuple
    split: EpsilonSplit
    invalid_output: torch.Tensor


@dataclass(frozen=True)
class CanonicalRelevance:
    """Encoded-input relevance and separate on-device operation accounting.

    ``h0`` is keyed by sampled occurrence, not by physical record identity.
    Route records retain layer, relation and edge-occurrence identities.
    ``invalid_h0`` conservatively marks channels linked to rejected outputs
    by nonzero captured weights, including zero-valued source activations.
    This result is not an enumeration of walks or a field allocation.
    """
    score: torch.Tensor
    h0: dict
    operations: tuple
    invalid_h0: dict
    policy_id: str = POLICY_ID

    def ledger(self):
        """Signed terminal amounts, without folding residual into epsilon."""
        result = {
            key: torch.zeros_like(self.score)
            for key in ('encoded_inputs', 'bias', 'constant', 'stabilization',
                        'rejected', 'residual')
        }
        for vector in self.h0.values():
            result['encoded_inputs'] += vector.sum()
        for op in self.operations:
            for value in op.split.biases:
                result['bias'] += value.sum()
            for value in op.split.constants:
                result['constant'] += value.sum()
            for key in ('stabilization', 'rejected', 'residual'):
                result[key] += getattr(op.split, key).sum()
        return result


@torch.no_grad()
def propagate(computation, *, output_index=0, sign=1, policy_id=POLICY_ID):
    """Selected-policy signed LRP through captured head, GNN and preprocessing.

    Reuses ``WalkComputation``'s actual layer inputs and canonical incoming
    routes. Relevance merges only at the same layer and occurrence; all
    route contribution matrices remain available in ``operations``. ReLU
    passes relevance unchanged through active outputs and requires zero
    relevance at inactive outputs. CUDA captures require preserve_device.
    """
    from kumo_core.explain.walk_graph import WalkComputation

    validate_policy_id(policy_id)
    if not isinstance(computation, WalkComputation):
        raise TypeError('propagate requires a WalkComputation')
    if (type(output_index) is not int
            or not 0 <= output_index < computation.head_output.numel()):
        raise ValueError('Invalid output index')
    if type(sign) is not int or sign not in (-1, 1):
        raise ValueError('sign must be -1 or 1')
    if computation.device != computation.capture_device:
        raise ValueError('Canonical relevance must preserve the capture device')
    if not computation.qualification:
        raise ValueError(
            'Canonical relevance requires structural qualification')

    operations = []

    def split(kind, index, node, routes, contributions, biases, relevance,
              invalid, weights, activation=None, pre_activation=None):
        if activation is not None and bool(
            ((pre_activation <= 0) & (relevance != 0)).any()):
            raise ValueError('Nonzero relevance at an inactive ReLU output')
        result = canonical_epsilon_split(contributions, biases, relevance,
                                         policy_id=policy_id)
        invalid_output = invalid | result.rejection_mask
        child_invalid = [((weight.T != 0) & invalid_output).any(1)
                         for weight in weights]
        operations.append(
            RelevanceOperation(kind, index, node, tuple(routes),
                               relevance.clone(), tuple(contributions),
                               tuple(biases), (), result, invalid_output))
        return [value.sum(1) for value in result.contributions], child_invalid

    table, seed = computation.seed
    score = sign * computation.head_output[output_index]
    relevance = torch.zeros_like(computation.head_output)
    relevance[output_index] = score
    invalid = torch.zeros_like(relevance, dtype=torch.bool)
    last = len(computation.head) - 1
    for op in reversed(computation.head):
        weight = op.weight[table].clone()
        bias = op.bias[table]
        bias = None if bias is None else bias.clone()
        if op.index == last:
            weight[output_index] *= sign
            if bias is not None:
                bias[output_index] *= sign
        contributions = [op.input[table][seed][:, None] * weight.T]
        children, invalid_children = split('head', op.index, computation.seed,
                                           (), contributions,
                                           [] if bias is None else [bias],
                                           relevance, invalid, [weight],
                                           op.activation,
                                           op.pre_activation[table][seed])
        relevance, = children
        invalid, = invalid_children

    frontier = {computation.seed: relevance}
    invalid_frontier = {computation.seed: invalid}
    for layer in range(computation.T, 0, -1):
        following, invalid_following = {}, {}
        inputs = computation.captured_input(layer)
        for node in sorted(frontier):
            incoming = computation.incoming(layer, node)
            contributions = [
                inputs[route.source[0]][route.source[1]][:, None] *
                route.weight.T / route.denominator for route in incoming.routes
            ]
            children, invalid_children = split(
                'message_passing', layer, node, incoming.routes, contributions,
                incoming.biases, frontier[node], invalid_frontier[node],
                [route.weight
                 for route in incoming.routes], computation.activation,
                computation.pre_activation[layer][node[0]][node[1]])
            for route, child, invalid_child in zip(incoming.routes, children,
                                                   invalid_children):
                if route.source in following:
                    following[route.source] = following[route.source] + child
                    invalid_following[route.source] |= invalid_child
                else:
                    following[route.source] = child
                    invalid_following[route.source] = invalid_child
        frontier = following
        invalid_frontier = invalid_following

    h0, invalid_h0 = {}, {}
    for node in sorted(frontier):
        table, occurrence = node
        relevance = frontier[node]
        invalid = invalid_frontier[node]
        for op in reversed(computation.preprocessing):
            contribution = (op.input[table][occurrence][:, None] *
                            op.weight[table].T)
            bias = op.bias[table]
            children, invalid_children = split(
                'preprocessing', op.index, node, (), [contribution],
                [] if bias is None else [bias], relevance, invalid,
                [op.weight[table]], op.activation,
                op.pre_activation[table][occurrence])
            relevance, = children
            invalid, = invalid_children
        h0[node] = relevance
        invalid_h0[node] = invalid
    return CanonicalRelevance(score, h0, tuple(operations), invalid_h0,
                              policy_id)


@dataclass(frozen=True)
class CanonicalWalks:
    """Bounded encoded-input routes, with independent merged accounting.

    Enumeration completeness certifies traversal, not rejected ancestry or
    faithfulness. Prefix mass is incoming signed relevance including future
    terminal allocations; it does not bound omitted absolute leaf relevance.
    Coverage measures signed scalar walk totals, not channel absolute mass.
    Invalid masks align with walks and prefixes in the enumeration.
    Captured inputs bind field composition numerically, not by execution or
    source-record provenance. Their tensors are detached snapshots.
    Prefix masks cover expanded suffix ancestry only; unexplored dependencies
    may introduce further rejection.
    """
    merged: CanonicalRelevance
    enumeration: object
    invalid_walks: tuple
    invalid_prefixes: tuple
    terminals: dict
    displayed_indices: tuple
    coverage: dict
    captured_h0: dict
    train_table_name: str | None

    @property
    def policy_id(self):
        return self.merged.policy_id


class _CanonicalWalkRule:
    """Canonical splits on the shared bounded enumerator's route states."""
    def __init__(self, computation, merged):
        self.computation = computation
        self.merged = merged
        self.operations = {
            (op.kind, op.index, op.node): op
            for op in merged.operations
        }
        self.invalid_walks = []
        self.invalid_prefixes = []
        self.pending_mass = []
        self.terminals = {
            key: torch.zeros_like(merged.score)
            for key in ('bias', 'constant', 'stabilization', 'rejected',
                        'residual')
        }

    def _split(self, op, state, weights):
        relevance, invalid = state
        result = canonical_epsilon_split(op.contributions, op.biases, relevance,
                                         constants=op.constants,
                                         policy_id=self.merged.policy_id)
        invalid = invalid | result.rejection_mask
        self.pending_mass.extend(
            float(value.abs().sum())
            for value in (*result.contributions, *result.biases,
                          *result.constants, result.stabilization,
                          result.rejected, result.residual))
        for key, values in (('bias', result.biases), ('constant',
                                                      result.constants)):
            for value in values:
                self.terminals[key] += value.sum()
        for key in ('stabilization', 'rejected', 'residual'):
            self.terminals[key] += getattr(result, key).sum()
        return [(value.sum(1), ((weight.T != 0) & invalid).any(1))
                for value, weight in zip(result.contributions, weights)]

    def absolute_mass(self, layer, node, state):
        import math

        mass = math.fsum([float(state[0].abs().sum()), *self.pending_mass])
        self.pending_mass.clear()
        return mass

    def start(self):
        from kumo_core.explain.walk_enumerate import NONE

        computation = self.computation
        head = [
            self.operations['head', op.index, computation.seed]
            for op in reversed(computation.head)
        ]
        state = (head[0].relevance.clone(),
                 torch.zeros_like(head[0].relevance, dtype=torch.bool))
        for op, captured in zip(head, reversed(computation.head)):
            state, = self._split(op, state,
                                 [captured.weight[computation.seed[0]]])
        return state, NONE

    def transfer(self, layer, destination, state, incoming):
        from kumo_core.explain.walk_enumerate import NONE

        op = self.operations['message_passing', layer, destination]
        return self._split(op, state, [r.weight for r in incoming.routes]), NONE

    def prefix_mass(self, layer, destination, state):
        self.invalid_prefixes.append(state[1].clone())
        return float(state[0].sum())

    def finish(self, source, state):
        from kumo_core.explain.walk_enumerate import NONE

        for captured in reversed(self.computation.preprocessing):
            op = self.operations['preprocessing', captured.index, source]
            state, = self._split(op, state, [captured.weight[source[0]]])
        relevance, invalid = state
        self.invalid_walks.append(invalid.clone())
        return float(relevance.sum()), relevance, NONE


@torch.no_grad()
def enumerate_canonical_walks(computation, *, output_index=0, sign=1,
                              max_walks=10000, top_k=20, policy_id=POLICY_ID):
    """Enumerate occurrence-aware canonical routes on the capture device.

    The budget counts message-passing route expansions, including zero
    relevance routes. Display ranks completed walks by absolute scalar
    relevance, with canonical traversal order breaking ties. Ranking and
    signed coverage denominators are available only for exhaustive search.
    Ranking certifies scalar allocation order, not faithfulness or stability.
    Rejected ancestry prevents ranking certification even when
    traversal is complete; coverage then describes partial allocations only.
    The merged propagation ledger is independent of the display budget;
    explored terminals plus completed inputs plus frontier mass account for
    the score separately. Scientific validity still requires ancestry masks.
    """
    import math

    from kumo_core.explain.walk_enumerate import enumerate_walks

    if type(top_k) is not int or top_k < 0:
        raise ValueError('top_k must be a nonnegative integer')
    if type(max_walks) is not int or max_walks < 1:
        raise ValueError('max_walks must be a positive integer')
    merged = propagate(computation, output_index=output_index, sign=sign,
                       policy_id=policy_id)
    rule = _CanonicalWalkRule(computation, merged)
    enumeration = enumerate_walks(computation, rule, max_walks=max_walks)
    indices = tuple(
        sorted(range(len(enumeration.walks)),
               key=lambda i: -abs(enumeration.walks[i].relevance))[:top_k])
    rejected_ancestry = any(
        bool(op.invalid_output.any()) for op in merged.operations)
    coverage = {
        'policy_id': policy_id,
        'ranking_scope': 'scalar_allocations_under_selected_policy',
        'ranking_certified': (enumeration.complete and not rejected_ancestry),
        'rejected_ancestry': rejected_ancestry,
        'denominator_scope': 'admitted_scalar_encoded_input_walks'
    }
    for name, orientation in (('positive', 1), ('negative', -1)):
        found = math.fsum(
            max(orientation * walk.relevance, 0.) for walk in enumeration.walks)
        displayed = math.fsum(
            max(orientation * enumeration.walks[i].relevance, 0.)
            for i in indices)
        total = found if enumeration.complete else None
        coverage[name] = {
            'enumerated': found,
            'displayed': displayed,
            'total': total,
            'fraction': displayed / total if total else None,
            'omitted': total - displayed if total is not None else None
        }
    return CanonicalWalks(merged, enumeration, tuple(rule.invalid_walks),
                          tuple(rule.invalid_prefixes), rule.terminals, indices,
                          coverage, {
                              key: value.detach().clone()
                              for key, value in computation.h0.items()
                          }, computation.train_table_name)
