# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Signed fitted-field allocation below captured graph input channels."""
import math
from dataclasses import dataclass, field
from functools import partial

import torch
from kumo_core.encoders.fused.categorical import FusedCategoricalEncoder
from kumo_core.encoders.fused.numerical import FusedNumericalEncoder
from kumo_core.encoders.tensor_frame_encoder import TensorFrameEncoder
from kumo_core.explain.walk_lrp_epsilon import (
    POLICY_ID,
    SMOOTH_FLOOR_POLICY_ID,
    EpsilonSplit,
    canonical_epsilon_split,
    validate_policy_id,
)


@dataclass
class FieldAllocation:
    fields: dict = field(default_factory=dict)
    field_channels: dict = field(default_factory=dict)
    missingness: dict = field(default_factory=dict)
    constants: dict = field(default_factory=dict)
    ledgers: dict[str, EpsilonSplit] = field(default_factory=dict)
    complete: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    policy_id: str = POLICY_ID


@dataclass(frozen=True)
class CanonicalFieldWalks:
    """Fitted-field endpoints aligned with canonical encoded-input walks.

    Each allocation retains its route's ledgers and conservative row-level
    ancestry flags. Featureless constants are terminals, never field endpoints.
    The encoded enumeration owns route identities, budget and signed frontier.
    """
    encoded_walks: object
    merged: FieldAllocation
    allocations: tuple
    terminals: dict
    displayed_indices: tuple
    coverage: dict

    @property
    def policy_id(self):
        return self.merged.policy_id


def _copy(value):
    return value.detach().clone()


def _check_close(actual, expected, label):
    if (actual.shape != expected.shape or actual.device != expected.device
            or not bool(torch.isfinite(actual).all())
            or not bool(torch.isfinite(expected).all()) or not torch.allclose(
                actual.double(), expected.double(), atol=1e-5, rtol=1e-5)):
        raise ValueError(f'{label}: fitted-field reconstruction mismatch')


class FieldCapture:
    """Capture one fitted fuse-mode numerical/categorical encoder execution.

    Allocation ends at encoded fields, including fitted numerical preprocessing
    and categorical lookup. It does not attribute raw input transformations.
    No gradients, device transfers or predictor modifications are performed.
    """
    def __init__(self, module):
        self.module = module
        self.handles = []
        self.tables = {}
        self.failure = None

    def __enter__(self):
        self.tables = {}
        self.failure = None
        try:
            for name, encoder in self.module.encoder.enc_dict.items():
                if (type(encoder) is not TensorFrameEncoder
                        or encoder.enc_mode != 'fuse'
                        or encoder.rel_time_encoder is not None
                        or encoder.use_seq_id or type(
                            encoder.act) not in (type(None), torch.nn.ReLU)):
                    raise ValueError(f'Unsupported field adapter for {name}')
                for kind, fused in encoder.stype_encoder_dict.items():
                    if type(fused) not in (FusedNumericalEncoder,
                                           FusedCategoricalEncoder):
                        raise ValueError(
                            f'Unsupported field encoder {name}/{kind}')
                    if bool(fused.exclude_cols_mask.any()):
                        raise ValueError(
                            f'Unsupported excluded field {name}/{kind}')
                    self.handles.append(
                        fused.register_forward_hook(
                            self._guard(partial(self._fused, name, kind))))
                self.handles.append(
                    encoder.register_forward_pre_hook(
                        self._guard(partial(self._start, name)),
                        with_kwargs=True))
                self.handles.append(
                    encoder.register_forward_hook(
                        self._guard(partial(self._end, name)),
                        with_kwargs=True))
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _guard(self, callback):
        def hook(*args):
            if self.failure is None:
                try:
                    callback(*args)
                except Exception as error:
                    self.failure = f'{type(error).__name__}: {error}'
            return None

        return hook

    def _start(self, name, encoder, args, kwargs):
        if name in self.tables:
            raise ValueError('FieldCapture supports one forward per context')
        tf = kwargs.get('tf', args[0] if args else None)
        expected = set(encoder.stype_encoder_dict)
        if set(tf.feat_dict) != expected:
            raise ValueError(f'{name}: unaccounted input feature groups')
        for kind, fused in encoder.stype_encoder_dict.items():
            fused.validate_col_order(tf.col_names_dict[kind])
        self.tables[name] = {'groups': {}, 'relu': encoder.act is not None}

    @torch.no_grad()
    def _fused(self, name, kind, encoder, args, output):
        raw = args[0]
        fixed = encoder.forward_fixed(raw, encoder.col_names)
        bias = []
        if type(encoder) is FusedNumericalEncoder:
            missing = ~torch.isfinite(raw)
            na = fixed.isnan(
            ) if encoder.has_separate_na_encoder else torch.zeros_like(missing)
            value = fixed.nan_to_num(
            ) if encoder.has_separate_na_encoder else fixed
            terms = value.double().unsqueeze(
                -1) * encoder.lin.weight.detach().double().T
            if encoder.lin.bias is not None:
                bias.append(_copy(encoder.lin.bias).double())
            na_terms = torch.zeros_like(terms)
            if encoder.has_separate_na_encoder:
                na_terms = na.double().unsqueeze(
                    -1) * encoder.na_lin.weight.detach().double().T
                if encoder.na_lin.bias is not None:
                    bias.append(_copy(encoder.na_lin.bias).double())
            contributions = (terms, na_terms)
        else:
            missing = (raw == -1) | (raw >= encoder.na_thresholds)
            terms = encoder.emb.weight.detach()[fixed].double()
            # EmbeddingBag omits padding even if its stored row is nonzero.
            terms = torch.where(
                (fixed == encoder.emb.padding_idx).unsqueeze(-1),
                torch.zeros_like(terms), terms)
            contributions = (terms, )
        reconstructed = sum(z.sum(1) for z in contributions)
        for b in bias:
            reconstructed = reconstructed + b
        _check_close(output, reconstructed, f'{name}/{kind}')
        self.tables[name]['groups'][kind] = {
            'columns': tuple(encoder.col_names),
            'terms': contributions,
            'biases': tuple(bias),
            'output': _copy(output),
            'missing': _copy(missing),
            'fixed': _copy(fixed),
            'numerical': type(encoder) is FusedNumericalEncoder,
            'raw': _copy(raw),
            'parameters': {
                key: _copy(value)
                for key, value in encoder.named_parameters()
            },
            'buffers': {
                key: _copy(value)
                for key, value in encoder.named_buffers()
            },
        }

    def _end(self, name, encoder, args, kwargs, output):
        table = self.tables[name]
        groups = table['groups']
        pre = (sum(group['output'] for group in groups.values())
               if groups else torch.ones_like(output))
        expected = pre.relu() if table['relu'] else pre
        _check_close(output, expected, name)
        table['output'] = _copy(output)
        table['pre'] = _copy(pre)

    @torch.no_grad()
    def allocate(self, h0_relevance, *, h0, invalid=None, policy_id=POLICY_ID):
        """Return on-device signed fields and separate per-operation ledgers.

        ``h0_relevance`` maps (table, occurrence) to float64 channel vectors.
        ``h0`` is the graph capture's table tensor dictionary and must match
        the captured encoder outputs exactly. Optional ``invalid`` maps those
        occurrence keys to Boolean channel masks for rejected ancestor routes.
        Field completeness is conservative across all channels of its row.
        Missingness is an annotated subset of each field, not an extra total.
        Smooth floor uses one split across fitted field terms and biases.
        """
        validate_policy_id(policy_id)
        if self.failure is not None:
            raise ValueError(f'Field capture failed: {self.failure}')
        result = FieldAllocation(policy_id=policy_id)
        for name, value in h0.items():
            if (name not in self.tables or 'output' not in self.tables[name]
                    or value.device != self.tables[name]['output'].device
                    or not torch.equal(
                        value, self.tables[name]['output'].to(value.dtype))):
                raise ValueError(
                    f'{name}: graph h0 does not match fitted encoder capture')
        for key, relevance in h0_relevance.items():
            name, row = key
            if name not in h0 or not 0 <= row < h0[name].shape[0]:
                raise ValueError(f'Unknown field occurrence {key}')
            table = self.tables[name]
            output = table['output'][row]
            if relevance.shape != output.shape or relevance.device != output.device:
                raise ValueError(f'{key}: relevance shape/device mismatch')
            ancestor_invalid = (invalid or {}).get(
                key, torch.zeros_like(relevance, dtype=torch.bool))
            if (ancestor_invalid.shape != relevance.shape
                    or ancestor_invalid.device != relevance.device
                    or ancestor_invalid.dtype != torch.bool):
                raise ValueError(f'{key}: invalid ancestor mask')
            if table['relu'] and bool(
                ((table['pre'][row] <= 0) & (relevance != 0)).any()):
                raise ValueError(
                    f'{key}: nonzero relevance at inactive encoder ReLU')
            groups = table['groups']
            prefix = f'{name}/{row}'
            if not groups:
                split = canonical_epsilon_split([], [], relevance,
                                                constants=[output.double()],
                                                policy_id=policy_id)
                result.ledgers[prefix + '/constant'] = split
                result.constants[key] = split.constants[0].sum()
                result.complete[key] = not bool(ancestor_invalid.any()
                                                | split.rejection_mask.any())
                continue
            fused = policy_id == SMOOTH_FLOOR_POLICY_ID
            if fused:
                outer = canonical_epsilon_split(
                    [z[row] for g in groups.values() for z in g['terms']],
                    [b for g in groups.values()
                     for b in g['biases']], relevance, policy_id=policy_id)
                result.ledgers[prefix + '/fused_encoder'] = outer
            else:
                group_terms = torch.stack(
                    [g['output'][row].double() for g in groups.values()])
                outer = canonical_epsilon_split([group_terms], [], relevance,
                                                policy_id=policy_id)
                result.ledgers[prefix + '/fusion'] = outer
            offset = 0
            for index, (kind, group) in enumerate(groups.items()):
                if fused:
                    inner = outer
                    end = offset + len(group['terms'])
                    channels = sum(outer.contributions[offset:end])
                    offset = end
                else:
                    inner = canonical_epsilon_split(
                        [z[row] for z in group['terms']], group['biases'],
                        outer.contributions[0][index], policy_id=policy_id)
                    result.ledgers[prefix + '/' + str(kind)] = inner
                    channels = sum(inner.contributions)
                complete = not bool(ancestor_invalid.any()
                                    | outer.rejection_mask.any()
                                    | inner.rejection_mask.any())
                for column, field_name in enumerate(group['columns']):
                    field_key = (name, row, field_name)
                    result.field_channels[field_key] = channels[column]
                    result.fields[field_key] = channels[column].sum()
                    # Includes imputed numerical values or the fitted N/A category.
                    result.missingness[field_key] = (
                        channels[column].sum() if bool(group['missing'][row,
                                                                        column])
                        else relevance.new_zeros(()))
                    result.complete[field_key] = complete
                    result.metadata[field_key] = {
                        'kind': str(kind),
                        'missing_or_unknown': bool(group['missing'][row,
                                                                    column]),
                        'endpoint': 'fitted_encoded_field',
                    }
        return result

    @torch.no_grad()
    def allocate_walks(self, encoded_walks, *, h0, top_k=20):
        """Compose completed canonical routes through this fitted capture.

        Reuse field allocation independently for each route, preserving signed
        contributions even when merged occurrence relevance cancels to zero.
        Field completeness remains conservative across every channel of a row.
        Caller inputs must exactly match the retained graph input snapshot.
        This checks numerical compatibility, not common execution provenance.
        Ranking certifies scalar allocation order, not faithfulness or stability.
        Partial traversal leaves total coverage unknown; rejected graph or field
        ancestry prevents ranking certification even for exhaustive traversal.
        """
        from kumo_core.explain.walk_lrp_epsilon import CanonicalWalks

        if not isinstance(encoded_walks, CanonicalWalks):
            raise TypeError('allocate_walks requires CanonicalWalks')
        if type(top_k) is not int or top_k < 0:
            raise ValueError('top_k must be a nonnegative integer')
        captured = encoded_walks.captured_h0
        if set(h0) != set(captured):
            raise ValueError(
                'Graph h0 table set does not match canonical walks')
        # NodeGNNModule encodes the task row separately, then removes it from
        # the graph input. It is not a graph field-walk endpoint.
        extra_tables = set(self.tables) - set(captured)
        allowed_extra = {encoded_walks.train_table_name} - {None}
        if set(captured) - set(self.tables) or extra_tables - allowed_extra:
            raise ValueError('Fitted capture table set does not match graph h0')
        for name, expected in captured.items():
            value = h0[name]
            if (not isinstance(value, torch.Tensor)
                    or value.shape != expected.shape
                    or value.dtype != expected.dtype
                    or value.device != expected.device
                    or not bool(torch.isfinite(value).all())
                    or not bool(torch.isfinite(expected).all())
                    or not torch.equal(value, expected)):
                raise ValueError(
                    f'{name}: graph h0 does not match canonical walks')
        walks = encoded_walks.enumeration.walks
        if len(walks) != len(encoded_walks.invalid_walks):
            raise ValueError('Canonical walk ancestry masks are misaligned')
        merged = self.allocate(encoded_walks.merged.h0, h0=h0,
                               invalid=encoded_walks.merged.invalid_h0,
                               policy_id=encoded_walks.policy_id)
        allocations = tuple(
            self.allocate({walk.source: walk.input_relevance}, h0=h0, invalid={
                walk.source: invalid
            }, policy_id=encoded_walks.policy_id)
            for walk, invalid in zip(walks, encoded_walks.invalid_walks))
        terminals = {
            key: value.clone()
            for key, value in encoded_walks.terminals.items()
        }
        for allocation in allocations:
            for split in allocation.ledgers.values():
                for key, values in (('bias', split.biases), ('constant',
                                                             split.constants)):
                    for value in values:
                        terminals[key] += value.sum()
                for key in ('stabilization', 'rejected', 'residual'):
                    terminals[key] += getattr(split, key).sum()
        endpoints = [(index, key)
                     for index, allocation in enumerate(allocations)
                     for key in allocation.fields]
        displayed = tuple(
            sorted(
                endpoints, key=lambda item: -abs(
                    float(allocations[item[0]].fields[item[1]])))[:top_k])
        rejected = (encoded_walks.coverage['rejected_ancestry']
                    or not all(merged.complete.values())
                    or any(not all(value.complete.values())
                           for value in allocations))
        complete = encoded_walks.enumeration.complete
        coverage = {
            'policy_id': encoded_walks.policy_id,
            'ranking_scope': 'scalar_allocations_under_selected_policy',
            'ranking_certified': complete and not rejected,
            'rejected_ancestry': rejected,
            'denominator_scope': 'admitted_scalar_fitted_field_walks',
        }
        for name, orientation in (('positive', 1), ('negative', -1)):
            found = math.fsum(
                max(orientation * float(allocations[i].fields[key]), 0.)
                for i, key in endpoints)
            shown = math.fsum(
                max(orientation * float(allocations[i].fields[key]), 0.)
                for i, key in displayed)
            total = found if complete else None
            coverage[name] = {
                'enumerated': found,
                'displayed': shown,
                'total': total,
                'fraction': shown / total if total else None,
                'omitted': total - shown if total is not None else None,
            }
        return CanonicalFieldWalks(encoded_walks, merged, allocations,
                                   terminals, displayed, coverage)
