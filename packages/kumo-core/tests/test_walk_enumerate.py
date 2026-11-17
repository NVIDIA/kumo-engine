# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared bounded enumerator on fixture B1 with hand-computable rules.

The toy rules are independent of any model arithmetic: a walk scores
`h0(source) * product of route weights`, and prefix masses come from an
independent forward dynamic program over the captured routes.
"""
import math

import pytest
import torch

from kumo_core.explain.walk_enumerate import Absorbed, enumerate_walks
from kumo_core.explain.walk_schema import rank_walks, walk_key
from tests.test_walk_graph import (
    ART_REV,
    FK,
    REV,
    b1_batch,
    b1_forward,
    b1_module,
    capture,
)

F64 = torch.float64
NONE = Absorbed()


def b1_computation(order=None):
    module, batch = b1_module(), b1_batch(order=order)
    cap, _ = capture(module, lambda: b1_forward(module, batch))
    return cap.computation()


class PathProductRule:
    """Walk score = h0(source) * product of route weights."""

    def __init__(self, computation, rev=5.0):
        self.c = computation
        self.weights = {'self_update': 2.0, FK: 3.0, REV: rev, ART_REV: 7.0}

    def weight(self, route):
        return self.weights['self_update' if route.kind == 'self_update' else route.relation]

    def mass(self, t, node):  # independent forward dynamic program
        if t == 0:
            return float(self.c.h0[node[0]][node[1]].sum())
        return sum(self.weight(r) * self.mass(t - 1, r.source)
                   for r in self.c.incoming(t, node).routes)

    def start(self):
        return 1.0, NONE

    def transfer(self, layer, destination, state, incoming):
        return [state * self.weight(r) for r in incoming.routes], NONE

    def prefix_mass(self, layer, destination, state):
        return state * self.mass(layer, destination)

    def finish(self, source, state):
        return state * float(self.c.h0[source[0]][source[1]].sum()), None, NONE


class CountRule(PathProductRule):
    def weight(self, route):
        return 1.0

    def finish(self, source, state):
        return state, None, NONE


def summary(computation, rule, max_walks=10000, top_k=100):
    enumeration = enumerate_walks(computation, rule, max_walks=max_walks)
    ranked = rank_walks(enumeration.walks)
    shown, hidden = ranked[:top_k], ranked[top_k:]
    return {
        'walks': len(enumeration.walks), 'expanded': enumeration.expanded_routes,
        'prefixes': len(enumeration.prefixes),
        'displayed': math.fsum(w.relevance for w in shown),
        'omitted_display': math.fsum(w.relevance for w in hidden),
        'omitted_display_abs': math.fsum(abs(w.relevance) for w in hidden),
        'omitted_search': math.fsum(p.mass for p in enumeration.prefixes),
        'scores': sorted(w.relevance for w in enumeration.walks),
        'shown': [w.relevance for w in shown], 'enumeration': enumeration}


def test_toy_rule_complete_enumeration_matches_hand_values():
    computation = b1_computation()
    result = summary(computation, PathProductRule(computation))
    # c-s-c-s-c 4; t1-fk-c-s-c 12 and t2 18; t1-s-t1-fk-c 12, t2 18;
    # c-rev-t1-fk-c 15 and t2 15; a-rev-t1-fk-c 84.
    assert result['scores'] == [4, 12, 12, 15, 15, 18, 18, 84]
    assert result['walks'] == 8 and result['prefixes'] == 0
    assert result['expanded'] == 3 + 3 + 3 + 2
    assert result['displayed'] == 178 == PathProductRule(computation).mass(2, ('customer', 0))


@pytest.mark.parametrize('budget,expanded,prefixes,displayed,omitted', [
    (3, 3, 3, 0, 178), (6, 6, 2, 34, 144), (8, 8, 1, 67, 111), (11, 11, 0, 178, 0),
    (10, 9, 1, 145, 33)])
def test_budget_is_all_or_nothing_per_destination(budget, expanded, prefixes,
                                                  displayed, omitted):
    # Layer 2 expands 3 routes; layer 1 states are customer (3 routes), t0 (3)
    # and t1 (2). Budget 8 leaves t0 unexpanded while the later t1 expands.
    computation = b1_computation()
    result = summary(computation, PathProductRule(computation), max_walks=budget)
    assert (result['expanded'], result['prefixes']) == (expanded, prefixes)
    assert (result['displayed'], result['omitted_search']) == (displayed, omitted)
    assert result['displayed'] + result['omitted_search'] == 178  # exact closure


def test_signed_toy_shows_that_small_net_omission_is_not_absence():
    computation = b1_computation()
    result = summary(computation, PathProductRule(computation, rev=-5.0), top_k=3)
    assert result['shown'] == [84, 18, 18]
    assert result['omitted_display'] == -2.0
    assert result['omitted_display_abs'] == 58.0
    budget = summary(computation, PathProductRule(computation, rev=-5.0), max_walks=8)
    assert (budget['displayed'], budget['omitted_search']) == (37, 81)


def test_count_rule_counts_walks_and_route_expansions():
    computation = b1_computation()
    result = summary(computation, CountRule(computation))
    assert result['walks'] == 8 and result['expanded'] == 11
    assert result['displayed'] == 8


def test_zero_relevance_routes_are_never_pruned():
    computation = b1_computation()
    rule = PathProductRule(computation)
    rule.weights[ART_REV] = 0.0
    result = summary(computation, rule)
    assert result['walks'] == 8 and result['scores'][0] == 0.0


def test_enumeration_ignores_edge_type_insertion_order():
    def keys(order):
        computation = b1_computation(order)
        enumeration = enumerate_walks(computation, PathProductRule(computation),
                                      max_walks=10000)
        return [(walk_key(w.steps), w.relevance) for w in enumeration.walks]

    assert keys(None) == keys([FK, ART_REV, REV, ('transactions', 'article_id', 'article')])


def test_walk_steps_are_forward_ordered_and_contiguous():
    computation = b1_computation()
    for walk in enumerate_walks(computation, CountRule(computation), max_walks=100).walks:
        assert [route.layer for route in walk.steps] == [1, 2]
        assert walk.steps[0].source == walk.source
        assert walk.steps[0].destination == walk.steps[1].source
        assert walk.steps[-1].destination == computation.seed


class GradientInputSketch:
    """Test-only GNN-GI on the captured piece (bias-free B1, `s = +1`).

    It only demonstrates that the frozen protocol expresses degree-T
    coefficients; the method itself belongs to the degree-T module.
    """

    def __init__(self, computation):
        self.c = computation
        self.piece = computation.forward_piece()
        table, seed = computation.seed
        self.g = computation.head[0].weight[table][0].clone()

    def start(self):
        return self.g.clone(), NONE

    def transfer(self, layer, destination, state, incoming):
        gated = state * self.c.gate_mask(layer, destination)
        return [route.weight.T @ gated / route.denominator for route in incoming.routes], NONE

    def prefix_mass(self, layer, destination, state):
        return float(state @ self.piece.hidden[layer][destination[0]][destination[1]])

    def finish(self, source, state):
        vector = state * self.c.h0[source[0]][source[1]]
        return math.fsum(vector.tolist()), vector, NONE


def test_protocol_expresses_degree_t_scores_on_b1():
    computation = b1_computation()
    rule = GradientInputSketch(computation)
    result = summary(computation, rule)
    assert result['scores'] == [0.5, 0.5, 1, 1, 1, 1.5, 1.5, 2]
    assert result['displayed'] == float(rule.piece.z_piece[0]) == 9.0
    for budget, displayed, omitted in ((6, 3.5, 5.5), (8, 5.5, 3.5)):
        part = summary(computation, GradientInputSketch(computation), max_walks=budget)
        assert (part['displayed'], part['omitted_search']) == (displayed, omitted)


def test_protocol_errors_are_explicit():
    computation = b1_computation()

    class Short(CountRule):
        def transfer(self, layer, destination, state, incoming):
            return [state], NONE

    with pytest.raises(ValueError, match='one state per route'):
        enumerate_walks(computation, Short(computation), max_walks=100)
    with pytest.raises(ValueError, match='positive'):
        enumerate_walks(computation, CountRule(computation), max_walks=0)
    with pytest.raises(ValueError, match='Unknown absorption kind'):
        Absorbed(bias={'encoder': 1.0})
    with pytest.raises(TypeError, match='Python floats'):
        Absorbed(bias={'head': torch.tensor(1.0)})
    absorbed = Absorbed.of('head', bias=1, stabilizer=0.5)
    assert absorbed.bias == {'head': 1.0} and absorbed.stabilizer == {'head': 0.5}


# Optional absolute-mass hook (ledger D14) ####################################

class MassReportingRule(PathProductRule):
    """`PathProductRule` with the optional hook and a vector at `H_0`."""

    def __init__(self, computation):
        super().__init__(computation)
        self.calls = []

    def absolute_mass(self, layer, node, state):
        self.calls.append((layer, node))
        return abs(state) + 1.0

    def finish(self, source, state):
        relevance = state * float(self.c.h0[source[0]][source[1]].sum())
        return relevance, torch.tensor([relevance, -2.0 * relevance], dtype=F64), NONE


@pytest.mark.parametrize('max_walks', [10000, 3])
def test_absolute_mass_hook_reports_every_state_once(max_walks):
    computation = b1_computation()
    rule = MassReportingRule(computation)
    enumeration = enumerate_walks(computation, rule, max_walks=max_walks)
    # The seed's state plus one child per expanded route, each reported once.
    assert len(rule.calls) == 1 + enumeration.expanded_routes
    assert rule.calls[0] == (computation.T, computation.seed)
    finished = [call for call in rule.calls if call[0] == 0]
    assert len(finished) == len(enumeration.walks)
    assert len(rule.calls) - len(finished) >= len(enumeration.prefixes)
    assert enumeration.intermediate_state_abs_mass >= len(rule.calls)
    assert enumeration.walk_vector_abs_mass == pytest.approx(
        math.fsum(3.0 * abs(w.relevance) for w in enumeration.walks))


def test_rules_without_the_hook_report_no_state_mass():
    computation = b1_computation()
    enumeration = enumerate_walks(computation, PathProductRule(computation),
                                  max_walks=10000)
    assert enumeration.intermediate_state_abs_mass is None
    assert enumeration.walk_vector_abs_mass == 0.0  # no vectors either


def test_a_negative_absolute_mass_is_a_rule_error():
    computation = b1_computation()
    rule = MassReportingRule(computation)
    rule.absolute_mass = lambda layer, node, state: -1.0
    with pytest.raises(ValueError, match='nonnegative'):
        enumerate_walks(computation, rule, max_walks=10000)
