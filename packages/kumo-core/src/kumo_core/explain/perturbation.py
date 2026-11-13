# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controlled leaf-record removals on an original sampled scoring graph.

These are model-response experiments, not additive or causal attributions.
No replacement neighbors are sampled and no source database is modified.
"""

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterator

import torch
from torch_frame import TensorFrame, stype
from torch_geometric.data import HeteroData


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':'),
                   allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class RecordRemovalPlan:
    """Explicit source records and full schema, never a learned importance mask.

    ``groups`` contains dictionaries with ``table`` and ``pkey``. All sampled
    occurrences of that record are removed together. ``foreign_keys`` contains
    exact (source table, FK name, destination table) triples from graph schema.
    Only tables with no inbound foreign keys are eligible in this version.
    """

    groups: tuple[dict, ...]
    removable_tables: tuple[str, ...]
    foreign_keys: tuple[tuple[str, str, str], ...]
    max_groups: int = 32

    def __post_init__(self):
        if type(self.max_groups) is not int or not 1 <= self.max_groups <= 32:
            raise ValueError('max_groups must be between 1 and 32')
        if len(self.groups) > self.max_groups:
            raise ValueError('Removal selection exceeds configured budget')
        if len({_digest(group) for group in self.groups}) != len(self.groups):
            raise ValueError('Duplicate removal group')
        for relation in self.foreign_keys:
            if len(relation) != 3 or not all(
                    isinstance(x, str) and x for x in relation):
                raise ValueError('Invalid foreign-key relation schema')


def _counts(counts, keep):
    if counts is None or sum(counts) != len(keep):
        raise ValueError('Sample layer counts cannot be reconstructed')
    result, start = [], 0
    for count in counts:
        result.append(sum(keep[start:start + count]))
        start += count
    return result


def remove_record(sample: dict, group: dict, plan: RecordRemovalPlan) -> dict:
    """Preserve surviving node/edge order, reindex endpoints and layer counts."""
    table = group['table']
    if (table not in plan.removable_tables or table in {
            sample['seed_table'], sample['train_table_name'],
            sample['entity_table_name']
    }):
        raise ValueError(
            'Seed, task, or undeclared record removal is unsupported')
    if any(dst == table for _, _, dst in plan.foreign_keys):
        raise ValueError(
            'Record has inbound schema dependencies; removal unsupported')
    allowed = set(plan.foreign_keys)
    allowed.update(
        (dst, 'rev_' + fk, src) for src, fk, dst in plan.foreign_keys)
    if any(tuple(edge['type']) not in allowed for edge in sample['edges']):
        raise ValueError('Sample relation is absent from declared schema')
    if table not in sample['nodes']:
        raise ValueError('Selected table was not sampled')
    node = sample['nodes'][table]
    keep = [_digest(key) != _digest(group['pkey']) for key in node['pkeys']]
    if all(keep):
        raise ValueError('Selected source record was not sampled')
    if len(keep) != node['num_nodes']:
        raise ValueError('Incomplete source primary-key mapping')
    result = copy.deepcopy(sample)
    changed = result['nodes'][table]
    mapping = {}
    for old, retain in enumerate(keep):
        if retain:
            mapping[old] = len(mapping)
    for key in ('pkeys', 'times_ns', 'node_ids', 'batch'):
        if changed.get(key) is not None:
            if len(changed[key]) != len(keep):
                raise ValueError('Misaligned sampled node metadata')
            changed[key] = [
                v for v, retained in zip(changed[key], keep) if retained
            ]
    changed['num_nodes'] = sum(keep)
    changed['num_sampled_nodes'] = _counts(changed['num_sampled_nodes'], keep)
    for feature in changed['feature_tensors'].values():
        feature['values'] = [
            v for v, retained in zip(feature['values'], keep) if retained
        ]
        if feature.get('nonfinite') is not None:
            feature['nonfinite'] = [
                v for v, retained in zip(feature['nonfinite'], keep) if retained
            ]
        if feature.get('nonfinite_kind') is not None:
            feature['nonfinite_kind'] = [
                v for v, retained in zip(feature['nonfinite_kind'], keep)
                if retained
            ]
        feature['shape'][0] = sum(keep)
    for key, values in changed.get('numerical_features', {}).items():
        changed['numerical_features'][key] = [
            v for v, retained in zip(values, keep) if retained
        ]
    for edge in result['edges']:
        src, _, dst = edge['type']
        source, target = edge['edge_index']
        retained = [(src != table or s in mapping)
                    and (dst != table or t in mapping)
                    for s, t in zip(source, target)]
        edge['edge_index'] = [
            [
                mapping[s] if src == table else s
                for s, retain in zip(source, retained) if retain
            ],
            [
                mapping[t] if dst == table else t
                for t, retain in zip(target, retained) if retain
            ],
        ]
        edge['num_sampled_edges'] = _counts(edge['num_sampled_edges'], retained)
        # This is a derived computation graph, not another original sample.
        edge['original_edge_index'] = None
    result['origin'] = 'perturbation'
    return result


def reconstruct_batch(sample: dict) -> HeteroData:
    """Restore only supported, non-executable tensor payloads."""
    batch = HeteroData()
    batch.input_type = sample['seed_table']
    for table, node in sample['nodes'].items():
        frame, columns = {}, {}
        for name, feature in node['feature_tensors'].items():
            # Native TensorFrames use string Stype values. Preserve that key
            # coordinate; torch_frame.stype enums are not equal to strings.
            kind = stype(name.removeprefix('stype.')) if name.startswith(
                'stype.') else name
            dtype = getattr(torch, feature['dtype'].removeprefix('torch.'),
                            None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError('Unsupported captured tensor dtype')
            values = feature['values']

            # JSON null represents a captured nonfinite numerical coordinate.

            def restore(value):
                if isinstance(value, list):
                    return [restore(v) for v in value]
                return float('nan') if value is None else value

            tensor = torch.tensor(restore(values),
                                  dtype=dtype).reshape(feature['shape'])
            if feature.get('nonfinite_kind') is not None:
                kinds = torch.tensor(feature['nonfinite_kind']).reshape(
                    feature['shape'])
                tensor[kinds == 1] = float('nan')
                tensor[kinds == 2] = float('inf')
                tensor[kinds == 3] = float('-inf')
            frame[kind], columns[kind] = tensor, feature['columns']
        batch[table].tf = TensorFrame(frame, columns,
                                      num_rows=node['num_nodes'])
        batch[table].num_nodes = node['num_nodes']
        for source, target in (('batch', 'batch'), ('times_ns', 'time'),
                               ('seed_time', 'seed_time'), ('node_ids', 'n_id'),
                               ('input_id', 'input_id')):
            if node.get(source) is not None:
                batch[table][target] = torch.tensor(node[source],
                                                    dtype=torch.long)
        batch[table].pkey = node['pkeys']
        if node.get('batch_size') is not None:
            batch[table].batch_size = node['batch_size']
        batch[table].num_sampled_nodes = node['num_sampled_nodes']
    for edge in sample['edges']:
        relation = tuple(edge['type'])
        batch[relation].edge_index = torch.tensor(edge['edge_index'],
                                                  dtype=torch.long).reshape(
                                                      2, -1)
        batch[relation].num_sampled_edges = edge['num_sampled_edges']
    batch.validate(raise_on_error=True)
    return batch


def record_removal_responses(module, sample: dict, *, prediction: float,
                             prediction_id: str, plan: RecordRemovalPlan,
                             atol: float = 1e-6,
                             rtol: float = 1e-6) -> Iterator[dict]:
    """Execute the actual initialized native scalar model on retained tensors.

    Caller must bind the module to the exact retained checkpoint. The public
    adapter restores that checkpoint and checks its hash before calling here.
    """
    from kumo_core.train.loss import MSELoss
    from kumo_core.train.node_gnn_module import NodeGNNLightningModule

    if not isinstance(module, NodeGNNLightningModule) or not isinstance(
            module.loss, MSELoss):
        raise ValueError(
            'Record removals require the native scalar MSE node module')
    if module.device.type != 'cpu' or module.training:
        raise ValueError(
            'Record removals require a CPU model in evaluation mode')
    if sample.get('origin') != 'original_forward':
        raise ValueError('Record removals require an original scoring sample')

    def score(graph):
        batch = reconstruct_batch(graph)
        with torch.no_grad():
            logits = module.common_step(batch)
            decoded = module.target_encoder.decode_tensor(
                module.loss.to_prediction(logits))
        if decoded.numel() != 1 or not torch.isfinite(decoded).all():
            raise ValueError('Unsupported or nonfinite scalar model response')
        return float(decoded.reshape(()))

    baseline = score(sample)
    if not math.isclose(baseline, prediction, abs_tol=atol, rel_tol=rtol):
        raise ValueError('Retained graph does not reproduce emitted prediction')
    for group in plan.groups:
        modified = remove_record(sample, group, plan)
        perturbed = score(modified)
        yield {
            'origin': 'perturbation',
            'method': 'fixed_graph_record_removal',
            'prediction_id': prediction_id,
            'group': group,
            'original_sample_sha256': _digest(sample),
            'perturbed_sample_sha256': _digest(modified),
            'baseline': baseline,
            'perturbed_prediction': perturbed,
            'signed_change': baseline - perturbed,
            'absolute_change': abs(baseline - perturbed),
            'output_coordinate': 'decoded_model_output',
            'selected_groups': len(plan.groups),
            'resampled': False,
            'interpretation': 'model_response_change_not_causal_or_additive'
        }
