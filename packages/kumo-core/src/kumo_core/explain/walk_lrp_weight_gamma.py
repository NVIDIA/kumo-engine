# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`hetero_gnn_lrp_weight_gamma_v1`: paper-style weight-gamma GNN-LRP.

The rule of Schnake et al. (Table I, Eq. 9) on the shared walk enumerator:
at one destination neuron, an input `x_j` of a linear route with weight `w`
and frozen denominator `d` (the retained in-degree for `mean`, else 1)
receives the share `x_j * (w + gamma * max(0, w)) / d` of the neuron's
relevance, divided by the stabilized total of all such terms and of the
gamma-modified biases. One rule serves every captured linear: relation
messages (FK->PK mean or sum, PK->FK gather), learned self updates, the
identity self update (the weight `I`, so its share is `(1 + gamma) x`),
preprocessing linears and head linears. Biases are weights of a constant
input 1 (`b + gamma * max(0, b)`) and absorb relevance by kind; the exact
float64 epsilon takes the sign of the total (`+epsilon` at 0) and its effect
is the stabilizer residual by kind. The selected class is explained by
folding its sign into the last head row, which differs from negating the
explanation of `z` when `gamma > 0`.

Only the per-route modified contribution is specific to this method; the
traversal, folding, bias and epsilon arithmetic, accounting and row checks
are the shared LRP skeleton of `walk_lrp_reference`, so rows of both LRP
rules are built and validated identically. The rules coincide whenever every
rule input is nonnegative, which holds for Engine models with ReLU
activations; they differ only on signed inputs, where the paper validates
neither form. Relevance is conservative LRP of the selected logit, never a
degree-T decomposition, and a closed ledger is accounting, not faithfulness.

Rows also carry static denominator diagnostics (see `denominator_diagnostics`)
because `gamma > 0` can give the output layer a denominator of the opposite
sign to a negative selected logit, which inverts every share of that step.
"""
import math

import torch

from kumo_core.explain.walk_attribution import WEIGHT_GAMMA_METHOD
from kumo_core.explain.walk_graph import F64, KINDS, STRUCTURAL_CODES
from kumo_core.explain.walk_lrp_reference import (
    GammaLRPRule,
    lrp_qualify,
    lrp_summary,
    validate_lrp_row,
)
from kumo_core.explain.walk_schema import WalkSchemaError

METHOD_ID = WEIGHT_GAMMA_METHOD
GAMMA_RULE = 'weight'
#: Keys of `row['diagnostics'][METHOD_ID]`.
DIAGNOSTIC_KEYS = ('head_output_denominator', 'head_output_flipped',
                   'relevant_neurons', 'denominator_sign_flips',
                   'denominator_sign_flips_by_kind', 'min_denominator_margin',
                   'max_preactivation_mismatch')


def modified_weight(weight, gamma):
    """`W + gamma * max(0, W)`, elementwise (paper Table I).

    The frozen `1/d` is positive, so modifying `W` or `W/d` is the same; the
    identity weight `I` becomes `(1 + gamma) I`.
    """
    return weight + gamma * weight.clamp(min=0)


class WeightGammaRule(GammaLRPRule):
    """GNN-LRP with gamma on weights over the shared LRP skeleton."""
    gamma_rule = GAMMA_RULE

    def modified_contribution(self, inputs, weight, denominator):
        return inputs[:, None] * modified_weight(weight, self.gamma).T / denominator


def qualify(module, config):
    """The shared LRP operator profile; raises before any hook is registered.

    Contract section 7 rejects and allows the same cases for both LRP rules
    (identity activations, eval-mode dropout and biases are allowed).
    """
    return lrp_qualify(module, config)


def make_rule(computation, selection, config):
    return WeightGammaRule(computation, selection, config)


def summarize(computation, selection, config, enumeration):
    """LRP settings and exactness, plus the denominator diagnostics.

    Diagnostics are attached only when every value is finite, so a capture
    with nonfinite values fails as `nonfinite_relevance` (or, if its
    relevance happens to be finite, at validation, which requires them).
    """
    diagnostics = denominator_diagnostics(computation, selection, config)
    return lrp_summary(computation, config, enumeration, GAMMA_RULE,
                       diagnostics=diagnostics if _finite(diagnostics) else None)


# Diagnostics ##################################################################

def _fold(weight, bias, index, sign):
    """The selected output row times `sign`, on copies (as the skeleton)."""
    weight = weight.clone()
    weight[index] *= sign
    if bias is not None:
        bias = bias.clone()
        bias[index] *= sign
    return weight, bias


def _totals(rule, inputs, biases):
    """One linear step at its output neurons: the gamma-modified total `S`
    exactly as the rule forms it (before epsilon), the absolute mass of its
    terms, and the gamma-0 total (the float64 pre-activation recomputed from
    the captured inputs). `inputs` are `(x, W, d)` triples."""
    size = inputs[0][1].size(0)
    total = torch.zeros(size, dtype=F64)
    mass = torch.zeros(size, dtype=F64)
    plain = torch.zeros(size, dtype=F64)
    for x, weight, denominator in inputs:
        z = rule.modified_contribution(x, weight, denominator)
        total += z.sum(0)
        mass += z.abs().sum(0)
        plain += (x[:, None] * weight.T / denominator).sum(0)
    for bias in biases:
        modified = rule.modified_bias(bias)
        total += modified
        mass += modified.abs()
        plain += bias
    return total, mass, plain


class _Checks:
    """Accumulates sign, margin and precision checks over relevant neurons."""

    def __init__(self):
        self.count = 0
        self.flips = dict.fromkeys(KINDS, 0)
        self.margins = []
        self.mismatches = []

    def add(self, kind, totals, values, captured):
        """`values` are the step's output values whose relevance it splits
        (post-activation, or the selected logit at the output); a neuron is
        relevant iff its value is nonzero. `captured` is the captured
        pre-activation of the same neurons."""
        total, mass, plain = totals
        relevant = values != 0
        count = int(relevant.sum())
        if not count:
            return
        self.count += count
        flipped = ((total >= 0) != (values > 0)) & relevant
        self.flips[kind] += int(flipped.sum())
        margin = torch.where(mass > 0, total.abs() / mass, torch.zeros_like(mass))
        self.margins.append(float(margin[relevant].min()))
        self.mismatches.append(float((captured - plain).abs()[relevant].max()))


def denominator_diagnostics(computation, selection, config):
    """Static sign and cancellation checks of the rule's denominators.

    Computed from the captured computation alone, independent of the budget,
    so original and replayed rows agree. Relevant neurons are those whose
    value carries relevance: the selected head output (when the selected
    logit is nonzero), nonzero hidden head outputs, nonzero outputs of every
    message-passing destination from which the seed is reachable, and nonzero
    preprocessing outputs of reachable sources; relevance at a zero value
    (an inactive ReLU) is exactly 0.

    - `head_output_denominator`: the gamma-modified, sign-folded total `S` of
      the selected output before epsilon; `head_output_flipped`: the
      selected logit is nonzero and `S` has the other sign (`S = 0` counts as
      positive), so every share of the output step is inverted. This happens
      for negative selected logits when `gamma > 0`; at gamma 0 it can fire
      only when the captured logit lies within the capture's rounding of zero
      (the float64 recomputation then has the other sign). The flag is
      truthful whenever it fires. A near-zero `S` instead inflates the
      shares up to the epsilon bound.
    - `denominator_sign_flips` (and by kind): relevant neurons whose
      denominator sign differs from their value's sign. Below the head of a
      ReLU model with nonnegative inputs this needs a precision mismatch.
    - `min_denominator_margin`: minimum of `|S| / sum |terms|`; 1 means no
      cancellation, values near 0 mean large opposing shares, whose float64
      rounding the ledger's accounting tolerance (scaled by the absolute
      ledger mass) may not see, so ledger closure is weak evidence there.
    - `max_preactivation_mismatch`: maximum of `|captured a - recomputed a|`,
      the gap between the captured forward and the float64 recomputation
      from captured inputs (for example `medium` matmul precision).
    Margin and mismatch are None when no neuron is relevant.
    """
    rule = WeightGammaRule(computation, selection, config)
    table, seed = computation.seed
    index, sign = selection.output_index, selection.logit_sign
    checks = _Checks()
    last = len(computation.head) - 1
    head_denominator = 0.0
    for op in computation.head:
        weight, bias = op.weight[table], op.bias[table]
        captured = op.pre_activation[table][seed]
        if op.index == last:
            weight, bias = _fold(weight, bias, index, sign)
            captured = sign * captured
            values = torch.zeros_like(captured)
            values[index] = rule.score
        else:
            values = computation.head[op.index + 1].input[table][seed]
        totals = _totals(rule, [(op.input[table][seed], weight, 1)],
                         [] if bias is None else [bias])
        if op.index == last:
            head_denominator = float(totals[0][index])
        checks.add('head', totals, values, captured)
    for t in range(computation.T, 0, -1):
        inputs = computation.captured_input(t)
        for node in computation.relevant_nodes(t):
            name, occurrence = node
            incoming = computation.incoming(t, node)
            totals = _totals(rule, [(inputs[route.source[0]][route.source[1]], route.weight,
                                     route.denominator) for route in incoming.routes],
                             incoming.biases)
            checks.add('message_passing', totals, computation.hidden[t][name][occurrence],
                       computation.pre_activation[t][name][occurrence])
    count = len(computation.preprocessing)
    for op in computation.preprocessing:
        for name, occurrence in computation.relevant_nodes(0):
            values = (computation.preprocessing[op.index + 1].input[name][occurrence]
                      if op.index + 1 < count else computation.hidden[0][name][occurrence])
            bias = op.bias[name]
            totals = _totals(rule, [(op.input[name][occurrence], op.weight[name], 1)],
                             [] if bias is None else [bias])
            checks.add('preprocessing', totals, values, op.pre_activation[name][occurrence])
    score = rule.score
    return {
        'head_output_denominator': head_denominator,
        'head_output_flipped': bool(score != 0 and (head_denominator >= 0) != (score > 0)),
        'relevant_neurons': checks.count,
        'denominator_sign_flips': sum(checks.flips.values()),
        'denominator_sign_flips_by_kind': dict(checks.flips),
        'min_denominator_margin': min(checks.margins) if checks.margins else None,
        'max_preactivation_mismatch': max(checks.mismatches) if checks.mismatches else None,
    }


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


# Validation ###################################################################

def _require(condition, message):
    if not condition:
        raise WalkSchemaError(message)


def validate_row(row):
    """Method checks after the generic v2 validator; raises `WalkSchemaError`.

    Diagnostic values are checked for type and internal consistency only: a
    tiny logit can change sign between the captured float32 output and the
    float64 recomputation even at `gamma = 0`.
    """
    try:
        _validate_row(row)
    except WalkSchemaError:
        raise
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as error:
        raise WalkSchemaError(
            f'Malformed weight-gamma row: {type(error).__name__}: {error}') from error


def _validate_row(row):
    validate_lrp_row(row, GAMMA_RULE)
    _require(row['method'] == METHOD_ID and row['config']['method'] == METHOD_ID,
             'Not a weight-gamma row')
    architecture, coverage = row['architecture'], row['coverage']
    ledger, exactness = row['ledger'], row['exactness']
    biased = bool(architecture['gnn_bias'] or architecture['head_bias'])
    reasons = (['method_is_lrp'] + (['biases_present'] if biased else [])
               + ([] if coverage['enumeration_complete'] else ['enumeration_incomplete']))
    _require(exactness['reasons'] == reasons,
             'Weight-gamma reasons are method_is_lrp, then biases_present and '
             'enumeration_incomplete where they apply')
    _require(exactness['qualification'] == [*STRUCTURAL_CODES, 'single_edge_gathers'],
             'Weight-gamma rows pass the shared LRP qualification')
    preprocessing = bool(row['operations']['preprocessing'])
    for kind, present in (('head', architecture['head_bias']),
                          ('message_passing', architecture['gnn_bias']),
                          ('preprocessing', architecture['gnn_bias'] and preprocessing)):
        _require(present or ledger['bias_relevance_by_kind'][kind] == 0.0,
                 f'No {kind} bias exists to absorb relevance')
    _require(preprocessing or ledger['stabilizer_residual_by_kind']['preprocessing'] == 0.0,
             'No preprocessing step exists to stabilize')
    _validate_diagnostics(row)


def _validate_diagnostics(row):
    diagnostics = (row['diagnostics'] or {}).get(METHOD_ID)
    if not isinstance(diagnostics, dict) or sorted(diagnostics) != sorted(DIAGNOSTIC_KEYS):
        raise WalkSchemaError('Weight-gamma rows carry their denominator diagnostics')
    count, flips = diagnostics['relevant_neurons'], diagnostics['denominator_sign_flips']
    by_kind = diagnostics['denominator_sign_flips_by_kind']
    _require(type(count) is int and count >= 0 and type(flips) is int
             and isinstance(by_kind, dict) and sorted(by_kind) == sorted(KINDS)
             and all(type(v) is int and v >= 0 for v in by_kind.values())
             and flips == sum(by_kind.values()) and flips <= count,
             'Denominator sign flips are counts over relevant neurons')
    denominator, flipped = (diagnostics['head_output_denominator'],
                            diagnostics['head_output_flipped'])
    score = row['score']
    _require(type(denominator) is float and type(flipped) is bool
             and flipped == (score != 0 and (denominator >= 0) != (score > 0))
             and (not flipped or by_kind['head'] >= 1) and (score == 0 or count >= 1),
             'head_output_flipped follows the head denominator and the selected logit')
    margin, mismatch = (diagnostics['min_denominator_margin'],
                        diagnostics['max_preactivation_mismatch'])
    if count == 0:
        _require(margin is None and mismatch is None, 'Margins need relevant neurons')
    else:
        _require(type(margin) is float and 0.0 <= margin <= 1.0
                 and type(mismatch) is float and mismatch >= 0.0,
                 'Denominator margin lies in [0, 1] and the mismatch is nonnegative')


__all__ = [
    'DIAGNOSTIC_KEYS', 'GAMMA_RULE', 'METHOD_ID', 'WeightGammaRule',
    'denominator_diagnostics', 'make_rule', 'modified_weight', 'qualify',
    'summarize', 'validate_row',
]
