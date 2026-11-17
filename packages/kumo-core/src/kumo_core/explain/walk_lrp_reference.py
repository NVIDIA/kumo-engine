# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`hetero_lrp_gamma_reference_v1` on the shared walk enumerator (v2 rows).

The arithmetic is the v1 contribution-gamma rule of `WalkCapture.result`
(`z_hat = z + gamma * max(0, z)` on contributions `z = h * w / d`, biases
`b + gamma * max(0, b)`, sign-preserving epsilon), computed in float64 from
captured tensors. Differences from v1: the selected sign is folded into the
last head row, head and preprocessing operations are not walk steps, the
budget counts message-passing routes only, and the epsilon is the exact
float64 value (v1 adds `torch.where` of Python scalars, effectively the
float32 value 9.9999997e-10). The result is a conservative LRP
decomposition, never a degree-T decomposition.

`GammaLRPRule`, `lrp_qualify`, `lrp_summary` and `validate_lrp_row` are the
shared LRP skeleton: a rule differs only in `modified_contribution`, the
gamma-modified contribution of one linear input.
"""
import math

import torch

from kumo_core.explain.walk_attribution import LEGACY_METHOD
from kumo_core.explain.walk_enumerate import Absorbed
from kumo_core.explain.walk_graph import F64, structural_qualification

METHOD_ID = LEGACY_METHOD


def lrp_qualify(module, config):
    """Shared LRP profile: biases, identity activations and eval-mode
    dropout are allowed; everything structural is checked before hooks."""
    return structural_qualification(module)


qualify = lrp_qualify


def epsilon_split(contributions, biases, relevance, epsilon):
    """Redistribute `relevance` over modified contributions and biases.

    `contributions` are gamma-modified `[C_in, C_out]` tensors and `biases`
    gamma-modified `[C_out]` vectors. The denominator adds `+epsilon` where
    the total is `>= 0` and `-epsilon` elsewhere, in exact float64. Returns
    the child relevance vectors, the bias shares and the stabilizer residual
    `sum(relevance - allocated)`.
    """
    total = torch.zeros_like(relevance)
    for z in contributions:
        total += z.sum(0)
    for b in biases:
        total += b
    eps = torch.full_like(total, epsilon)
    factor = relevance / (total + torch.where(total >= 0, eps, -eps))
    allocated = torch.zeros_like(relevance)
    children = []
    for z in contributions:
        share = z * factor
        allocated += share.sum(0)
        children.append(share.sum(1))
    shares = []
    for b in biases:
        share = b * factor
        shares.append(float(share.sum()))
        allocated += share
    return children, shares, float((relevance - allocated).sum())


class GammaLRPRule:
    """LRP-gamma as a `WalkRule` with float64 states at node outputs.

    Subclasses define `gamma_rule` and `modified_contribution`; biases are
    the weight of a constant input 1, so `b + gamma * max(0, b)` for both
    rules. The selected sign `s` is folded into the selected row of the last
    head linear (weight and bias) before any rule is applied (contract 2.2).
    """
    gamma_rule = None

    def __init__(self, computation, selection, config):
        self.computation = computation
        self.gamma = float(config.gamma)
        self.epsilon = float(config.epsilon)
        self.output_index = selection.output_index
        self.sign = selection.logit_sign
        self.score = self.sign * float(computation.head_output[self.output_index])

    def modified_contribution(self, inputs, weight, denominator):
        """`[C_in, C_out]` gamma-modified contribution of `inputs` through
        `weight` (`[C_out, C_in]`) divided by the frozen `denominator`."""
        raise NotImplementedError

    def modified_bias(self, bias):
        return bias + self.gamma * bias.clamp(min=0)

    def _split(self, contributions, biases, relevance, kind):
        children, shares, residual = epsilon_split(
            contributions, [self.modified_bias(b) for b in biases], relevance,
            self.epsilon)
        return children, Absorbed.of(kind, bias=math.fsum(shares), stabilizer=residual)

    def start(self):
        computation = self.computation
        table, seed = computation.seed
        last = len(computation.head) - 1
        relevance = torch.zeros(computation.head[last].weight[table].size(0), dtype=F64)
        relevance[self.output_index] = self.score
        bias = stabilizer = 0.0
        for op in reversed(computation.head):
            weight, b = op.weight[table].clone(), op.bias[table]
            b = None if b is None else b.clone()
            if op.index == last:
                weight[self.output_index] *= self.sign
                if b is not None:
                    b[self.output_index] *= self.sign
            contribution = self.modified_contribution(op.input[table][seed], weight, 1)
            children, absorbed = self._split([contribution], [] if b is None else [b],
                                             relevance, 'head')
            relevance = children[0]
            bias += absorbed.bias['head']
            stabilizer += absorbed.stabilizer['head']
        return relevance, Absorbed.of('head', bias=bias, stabilizer=stabilizer)

    def transfer(self, layer, destination, state, incoming):
        inputs = self.computation.captured_input(layer)
        contributions = [
            self.modified_contribution(inputs[route.source[0]][route.source[1]],
                                       route.weight, route.denominator)
            for route in incoming.routes]
        return self._split(contributions, list(incoming.biases), state, 'message_passing')

    def prefix_mass(self, layer, destination, state):
        # Includes the bias and stabilizer terms its completions would absorb.
        return float(state.sum())

    def finish(self, source, state):
        table, occurrence = source
        relevance = state
        bias = stabilizer = 0.0
        for op in reversed(self.computation.preprocessing):
            contribution = self.modified_contribution(op.input[table][occurrence],
                                                      op.weight[table], 1)
            b = op.bias[table]
            children, absorbed = self._split([contribution], [] if b is None else [b],
                                             relevance, 'preprocessing')
            relevance = children[0]
            bias += absorbed.bias['preprocessing']
            stabilizer += absorbed.stabilizer['preprocessing']
        return (math.fsum(relevance.tolist()), relevance,
                Absorbed.of('preprocessing', bias=bias, stabilizer=stabilizer))


class ContributionGammaRule(GammaLRPRule):
    """The legacy rule: gamma applied to contributions `z = h * w / d`."""
    gamma_rule = 'contribution'

    def modified_contribution(self, inputs, weight, denominator):
        z = inputs[:, None] * weight.T / denominator
        return z + self.gamma * z.clamp(min=0)


def make_rule(computation, selection, config):
    return ContributionGammaRule(computation, selection, config)


def lrp_method_settings(config, gamma_rule):
    """Contract 4.2 `method_settings` of an LRP-gamma rule."""
    return {'gamma': float(config.gamma), 'gamma_rule': gamma_rule,
            'bias_rule': 'gamma_on_constant_input', 'epsilon': float(config.epsilon),
            'stabilizer': 'sign_preserving_epsilon',
            'identity_root_rule': 'identity_as_weight',
            'sign_folding': 'selected_output_row'}


def lrp_exactness(computation, enumeration):
    """Exactness record of an LRP row: never a degree-T claim."""
    reasons = ['method_is_lrp']
    if computation.gnn_bias_present or computation.head_bias_present:
        reasons.append('biases_present')
    if not enumeration.complete:
        reasons.append('enumeration_incomplete')
    return {'decomposition': 'conservative_lrp', 'degree_t_claim': False,
            'reasons': reasons, 'piece': None, 'boundary_neurons': None,
            'mask_disagreements': None, 'relevant_gated_neurons': None,
            'min_abs_gated_preactivation': None, 'qualification': []}


def lrp_summary(computation, config, enumeration, gamma_rule, diagnostics=None):
    """`summarize` output of an LRP rule (no precision residual: LRP
    redistributes the emitted score itself)."""
    summary = {'method_settings': lrp_method_settings(config, gamma_rule),
               'precision_residual': 0.0,
               'exactness': lrp_exactness(computation, enumeration)}
    if diagnostics is not None:
        summary['diagnostics'] = diagnostics
    return summary


def summarize(computation, selection, config, enumeration):
    return lrp_summary(computation, config, enumeration, ContributionGammaRule.gamma_rule)


def validate_lrp_row(row, gamma_rule):
    """Method checks shared by LRP rules."""
    from kumo_core.explain.walk_attribution import WalkAttributionConfig
    from kumo_core.explain.walk_schema import WalkSchemaError
    expected = lrp_method_settings(WalkAttributionConfig(**row['config']), gamma_rule)
    if row['method_settings'] != expected:
        raise WalkSchemaError('LRP settings disagree with the configuration')
    exactness = row['exactness']
    if (exactness['decomposition'] != 'conservative_lrp' or exactness['degree_t_claim']
            or 'method_is_lrp' not in exactness['reasons']
            or row['ledger']['precision_residual'] != 0.0):
        raise WalkSchemaError('LRP rows are conservative decompositions without a claim')


def validate_row(row):
    validate_lrp_row(row, ContributionGammaRule.gamma_rule)
