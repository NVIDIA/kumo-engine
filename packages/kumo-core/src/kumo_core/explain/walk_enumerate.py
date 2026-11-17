# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared bounded walk enumerator and rule protocol (frozen interface).

Every walk method is a `WalkRule` driven by `enumerate_walks`. The enumerator
owns traversal order and the budget, so every method sees the same routes in
the same order; the rule owns the arithmetic. Display (`top_k`), walk keys,
sources, bags and the ledger are applied later by `walk_schema`.
"""
from dataclasses import dataclass, field
import math
from typing import Any, Protocol

from torch import Tensor

from kumo_core.explain.walk_graph import KINDS


@dataclass(frozen=True)
class Absorbed:
    """Relevance a rule absorbed in one call, by kind.

    Kinds are 'preprocessing', 'message_passing' and 'head'. `bias` holds
    shares absorbed by additive biases and `stabilizer` the epsilon effect.
    """
    bias: dict = field(default_factory=dict)
    stabilizer: dict = field(default_factory=dict)

    def __post_init__(self):
        for part in (self.bias, self.stabilizer):
            for kind, value in part.items():
                if kind not in KINDS:
                    raise ValueError(f'Unknown absorption kind {kind!r}')
                if type(value) is not float:
                    raise TypeError('Absorbed amounts must be Python floats')

    @classmethod
    def of(cls, kind, *, bias=0.0, stabilizer=0.0):
        return cls({kind: float(bias)}, {kind: float(stabilizer)})


NONE = Absorbed()


class WalkRule(Protocol):
    """A walk method's arithmetic over the shared enumerator.

    A state belongs to `(layer, destination)` and represents the output side
    of that node, `H[layer, destination]`; states are opaque to the
    enumerator. `start` returns the state at `H[T, seed]` after propagating
    through the head (folding the selected sign into the last head row);
    `transfer` returns one child state per route of `incoming`, at the
    route's source and in route order; `prefix_mass` is the signed relevance
    all completions of an unexpanded state would receive; `finish` propagates
    a state at a message-passing input row through preprocessing and returns
    the walk relevance and its float64 vector at `H_0` (or None).
    """

    def start(self) -> tuple[Any, Absorbed]:
        ...

    def transfer(self, layer: int, destination: tuple, state: Any,
                 incoming: Any) -> tuple[list, Absorbed]:
        ...

    def prefix_mass(self, layer: int, destination: tuple, state: Any) -> float:
        ...

    def finish(self, source: tuple, state: Any) -> tuple[float, Tensor | None, Absorbed]:
        ...

    # Optional (ledger D14), not part of the Protocol so that rules without it
    # still conform: `absolute_mass(layer, node, state) -> float`, called once
    # per state after the enumerator is done with it (transferred, left as an
    # unexpanded prefix, or finished at layer 0), returning the nonnegative
    # absolute mass the rule handled there. `enumerate_walks` sums it into
    # `Enumeration.intermediate_state_abs_mass` (None without the hook).


@dataclass(frozen=True)
class CompleteWalk:
    """One ordered walk: `steps[t - 1]` is its layer-t route (forward order)."""
    steps: tuple
    source: tuple
    relevance: float
    input_relevance: Tensor | None


@dataclass(frozen=True)
class UnexpandedPrefix:
    """A state whose incoming routes at `layer` did not fit in the budget.

    `steps` are the expanded suffix routes (layers `layer + 1 .. T`).
    """
    layer: int
    destination: tuple
    steps: tuple
    mass: float


@dataclass(frozen=True)
class Enumeration:
    """The traversal result.

    `walk_vector_abs_mass` is the sum over all complete walks of the L1 norm
    of their vector at `H_0` (walks without a vector contribute nothing);
    `intermediate_state_abs_mass` is the sum of the rule's `absolute_mass`
    hook over every state, or None when the rule has no hook (ledger D14).
    """
    walks: tuple
    prefixes: tuple
    bias: dict
    stabilizer: dict
    bias_terms: dict
    stabilizer_terms: dict
    expanded_routes: int
    max_walks: int
    walk_vector_abs_mass: float = 0.0
    intermediate_state_abs_mass: float | None = None

    @property
    def complete(self):
        return not self.prefixes


def enumerate_walks(computation, rule, *, max_walks):
    """Expand walks layer-synchronously from the seed at layer T down to 1.

    States are processed in creation order and routes in the canonical
    `Incoming` order, so results do not depend on edge-type insertion order.
    A destination's routes are expanded all or none: when they do not fit in
    the remaining budget of message-passing route expansions, the state
    becomes an unexpanded prefix and later states may still expand. Zero
    relevance is never pruned.

    A rule with the optional `absolute_mass` hook reports, once per state, the
    absolute mass it handled there; the sums feed the accounting tolerance
    together with the absolute mass of every walk's vector (ledger D14).
    """
    if type(max_walks) is not int or max_walks < 1:
        raise ValueError('max_walks must be a positive integer')
    bias = {kind: [] for kind in KINDS}
    stabilizer = {kind: [] for kind in KINDS}
    hook = getattr(rule, 'absolute_mass', None)
    masses = [] if callable(hook) else None

    def handled(layer, node, state):
        if masses is not None:
            value = float(hook(layer, node, state))
            if value < 0:
                raise ValueError('absolute_mass must be nonnegative')
            masses.append(value)

    def absorb(absorbed):
        if not isinstance(absorbed, Absorbed):
            raise TypeError('Rules must return Absorbed amounts')
        for kind, value in absorbed.bias.items():
            bias[kind].append(value)
        for kind, value in absorbed.stabilizer.items():
            stabilizer[kind].append(value)

    state, absorbed = rule.start()
    absorb(absorbed)
    frontier = [(computation.seed, (), state)]
    expanded = 0
    prefixes = []
    for layer in range(computation.T, 0, -1):
        following = []
        for destination, steps, state in frontier:
            incoming = computation.incoming(layer, destination)
            if expanded + len(incoming.routes) > max_walks:
                mass = float(rule.prefix_mass(layer, destination, state))
                prefixes.append(UnexpandedPrefix(layer, destination, steps, mass))
                handled(layer, destination, state)
                continue
            expanded += len(incoming.routes)
            children, absorbed = rule.transfer(layer, destination, state, incoming)
            absorb(absorbed)
            handled(layer, destination, state)
            if len(children) != len(incoming.routes):
                raise ValueError('A rule must return one state per route')
            for route, child in zip(incoming.routes, children):
                following.append((route.source, (route,) + steps, child))
        frontier = following
    walks = []
    vectors = []
    for source, steps, state in frontier:
        relevance, vector, absorbed = rule.finish(source, state)
        absorb(absorbed)
        handled(0, source, state)
        if vector is not None:
            vectors.append(float(vector.abs().sum()))
        walks.append(CompleteWalk(steps=steps, source=source,
                                  relevance=float(relevance), input_relevance=vector))
    return Enumeration(
        walks=tuple(walks), prefixes=tuple(prefixes),
        bias={kind: math.fsum(values) for kind, values in bias.items()},
        stabilizer={kind: math.fsum(values) for kind, values in stabilizer.items()},
        bias_terms={kind: tuple(values) for kind, values in bias.items()},
        stabilizer_terms={kind: tuple(values) for kind, values in stabilizer.items()},
        expanded_routes=expanded, max_walks=max_walks,
        walk_vector_abs_mass=math.fsum(vectors),
        intermediate_state_abs_mass=None if masses is None else math.fsum(masses))
