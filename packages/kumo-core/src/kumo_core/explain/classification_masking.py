# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original-context binary message ranking and bounded verified replacement."""
import math

import torch
from kumo_core.explain.message_intervention import (
    AmortizedMessageCompanion,
    OriginalMessageContext,
)
from kumo_core.explain.walk_graph import _relation_key
from kumo_core.modeling.gnn.hetero import ForeignKeyToPrimaryKeyConv

POLICY = 'runtime_verified_learned_greedy_k12_v1'


def fidelity(original, result, threshold):
    """Compare calibrated scores to one original with fixed fidelity bounds."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Invalid decision threshold')
    for value in (original, result):
        if (not all(math.isfinite(value[k]) for k in ('raw', 'served'))
                or not 0 <= value['served'] <= 1):
            raise ValueError('Invalid or nonfinite masking output')
    raw_error = abs(result['raw'] - original['raw'])
    served_error = abs(result['served'] - original['served'])
    same = (result['served'] >= threshold) == (original['served'] >= threshold)
    reasons = [name for name, failed in (
        ('raw_error', raw_error > .05), ('served_error', served_error > .01),
        ('decision_changed', not same)) if failed]
    return dict(raw_error=raw_error, served_error=served_error,
                decision_matches=same, fidelity_pass=not reasons,
                rejection_reasons=reasons)


def verified_greedy(ranked, original, keep_all, evaluate, threshold):
    """Verify the first twelve original-context ranks cumulatively, with rollback."""
    ranked = list(ranked)
    identities = []
    for probability, key, edge in ranked:
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Nonfinite or invalid keep probability')
        if (not isinstance(key, tuple) or len(key) != 2
                or type(key[0]) is not int or key[0] < 1
                or not isinstance(key[1], tuple) or len(key[1]) != 3
                or not all(isinstance(v, str) and v for v in key[1])
                or type(edge) is not int or edge < 0):
            raise ValueError('Invalid occurrence')
        identities.append((key, edge))
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate occurrence')
    if not fidelity(original, keep_all, threshold)['fidelity_pass']:
        raise ValueError('Keep-all fidelity failed')
    accepted, accepted_ranks, attempts = [], [], []
    selected = keep_all
    for rank, (probability, key, edge) in enumerate(sorted(ranked)[:12], 1):
        result = evaluate(tuple(accepted + [(key, edge)]))
        check = fidelity(original, result, threshold)
        trial = dict(rank=rank, probability=probability, layer=key[0],
                     relation=list(key[1]), edge_column=edge, **result, **check,
                     accepted_ranks_before=list(accepted_ranks),
                     trial_replaced_ranks=accepted_ranks + [rank],
                     accepted=check['fidelity_pass'])
        if trial['accepted']:
            accepted.append((key, edge))
            accepted_ranks.append(rank)
            selected = trial
        trial['accepted_ranks_after'] = list(accepted_ranks)
        attempts.append(trial)
    return dict(selected_rank=accepted_ranks[-1] if accepted else None,
                selected=selected, attempts=attempts,
                rejected_attempts=len(attempts) - len(accepted),
                fallback=not accepted, accepted_ranks=accepted_ranks,
                selected_mask=[dict(rank=rank, layer=key[0], relation=list(key[1]),
                                    edge_column=edge)
                               for rank, (key, edge) in zip(accepted_ranks, accepted)])


def capture_context(module, batch):
    """Capture detached messages, receivers and first-layer entity context together."""
    entity = module.entity_table_name
    if batch[module.train_table_name].num_nodes != 1 or batch[entity].batch_size != 1:
        raise ValueError('Masking requires exactly one entity seed')
    receivers, targets, occurrences, handles = {}, [], {}, []
    for layer, convs in enumerate(module.model.hetero_convs, 1):
        for relation, conv in convs.items():
            key = layer, _relation_key(relation)

            def hook(current, inputs, key=key):
                source, destination, edges = inputs[:3]
                if key[0] == 1 and key[1][0] == entity:
                    targets.append(source[0].detach().clone())
                if edges.shape[1]:
                    if key in receivers:
                        raise ValueError('Context convolution executed twice')
                    receivers[key] = destination[edges[1]].detach().clone()
                    occurrences[key] = dict(
                        edges=edges.detach().clone(),
                        direction=('fk_to_pk' if type(current) is ForeignKeyToPrimaryKeyConv
                                   else 'pk_to_fk'))
            handles.append(conv.register_forward_pre_hook(hook))
    try:
        with torch.no_grad(), OriginalMessageContext(module) as context:
            raw = module.common_step(batch)
    finally:
        for handle in handles:
            handle.remove()
    if not targets or any(not torch.equal(t, targets[0]) for t in targets):
        raise ValueError('First-layer entity context is unavailable or inconsistent')
    if context.messages.keys() != receivers.keys():
        raise ValueError('Receiver/message occurrence mismatch')
    contexts = {key: torch.cat((value, receivers[key],
                                targets[0].expand(value.size(0), -1)), dim=1)
                for key, value in context.messages.items()}
    if any(value.requires_grad or not torch.isfinite(value).all()
           for value in contexts.values()):
        raise ValueError('Invalid original context')
    return raw, contexts, occurrences


class FeasibilityCompanion(AmortizedMessageCompanion):
    """Receiver/target MLP gates with supplied frozen, per-layer learned stand-ins."""
    def __init__(self, keys, channels, standins):
        if type(channels) is not int or channels < 1:
            raise ValueError('channels must be positive')
        keys = tuple(keys)
        for layer, relation in keys:
            if (type(layer) is not int or layer < 1 or len(relation) != 3
                    or not all(isinstance(v, str) and v for v in relation)):
                raise ValueError('Invalid companion key')
        super().__init__(keys, channels)
        self.channels = channels
        self.networks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(3 * channels, 16),
                                torch.nn.ReLU(), torch.nn.Linear(16, 1))
            for _ in self.keys])
        for network in self.networks:
            torch.nn.init.constant_(network[-1].bias, 2.0)
        if set(standins) != {key[0] for key in self.keys}:
            raise ValueError('Stand-ins must cover every companion layer')
        for layer, value in standins.items():
            if (not isinstance(value, torch.Tensor) or value.shape != (channels,)
                    or value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError('Stand-ins require finite float32 channel vectors')
            self.standins[str(layer)] = torch.nn.Parameter(value.detach().clone(),
                                                           requires_grad=False)

    def forward(self, contexts, *, hard=False):
        for value in contexts.values():
            if (value.ndim != 2 or value.shape[1] != 3 * self.channels
                    or value.requires_grad or not torch.isfinite(value).all()):
                raise ValueError('Invalid detached receiver/target context')
        return super().forward(contexts, hard=hard)


def fit_feasibility(companion, cases, *, epochs=20):
    """Case-balanced feasibility BCE; only gate networks enter Adam at lr .01."""
    if type(epochs) is not int or not 1 <= epochs <= 120 or not cases:
        raise ValueError('Require nonempty fitting cases and 1..120 epochs')
    for contexts, targets in cases:
        if not contexts or set(contexts) != set(targets):
            raise ValueError('Fitting targets must cover nonempty contexts')
        for key, value in targets.items():
            if (value.shape != (contexts[key].shape[0],)
                    or value.device != contexts[key].device
                    or value.dtype != contexts[key].dtype or value.requires_grad
                    or not ((value == 0) | (value == 1)).all()):
                raise ValueError('Feasibility labels must be detached binary vectors')
    if any(p.requires_grad or p.grad is not None for p in companion.standins.parameters()):
        raise ValueError('Stand-ins must remain frozen')
    optimizer = torch.optim.Adam(companion.networks.parameters(), lr=.01)
    curve = []
    companion.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for contexts, targets in cases:
            gates, _ = companion(contexts)
            losses.append(torch.nn.functional.binary_cross_entropy(
                torch.cat([gates[k] for k in sorted(gates)]),
                torch.cat([targets[k] for k in sorted(gates)])))
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite feasibility loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in companion.networks.parameters()):
            raise ValueError('Nonfinite companion gradient')
        optimizer.step()
        curve.append({'epoch': epoch, 'loss': float(loss.detach())})
    companion.eval()
    return optimizer.state_dict(), curve
