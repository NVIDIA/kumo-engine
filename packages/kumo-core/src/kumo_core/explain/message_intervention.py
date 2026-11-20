# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic occurrence replacement in native relation convolutions.

GraphCapture observes outputs but cannot replace individual pre-aggregation
messages. This context uses the native convolution seam without cached replay.
"""
from functools import partial

import torch
from kumo_core.explain.walk_graph import (_relation_key,
                                          structural_qualification)
from kumo_core.modeling.gnn.hetero import (ForeignKeyToPrimaryKeyConv,
                                           PrimaryKeyToForeignKeyConv)


class MessageIntervention:
    """Apply scalar zero or one gates to native source-message occurrences.

    ``selection`` is (one-based layer, typed relation, original edge column).
    With no selection, ``gate`` applies to every executed relation message.
    Relation bias, self updates, topology and mean denominators are preserved.
    The context is single-forward, exclusive, and restores modules on errors.
    """
    def __init__(self, module, *, gate, selection=None):
        if type(gate) not in (int, float) or gate not in (0, 1):
            raise ValueError('Only deterministic scalar gates 0 and 1 admitted')
        if selection is not None:
            layer, relation, edge = selection
            if type(layer) is not int or layer < 1 or type(
                    edge) is not int or edge < 0:
                raise ValueError('Invalid message occurrence identity')
            selection = (layer, tuple(relation), edge)
        self.module, self.gate, self.selection = module, gate, selection
        self.used = False
        self.bound = []
        self.observations = []
        self.selected_hits = 0

    def __enter__(self):
        structural_qualification(self.module)
        if self.used:
            raise ValueError('Message intervention contexts are single-use')
        self.used = True
        candidates = []
        for layer, convs in enumerate(self.module.model.hetero_convs, 1):
            for relation, conv in convs.items():
                if type(conv) not in (ForeignKeyToPrimaryKeyConv,
                                      PrimaryKeyToForeignKeyConv):
                    raise ValueError('Unsupported message convolution')
                if hasattr(conv, '_message_intervention'):
                    raise ValueError(
                        'An intervention already owns this convolution')
                candidates.append((layer, _relation_key(relation), conv))
        for layer, relation, conv in candidates:
            conv._message_intervention = partial(self._apply, layer, relation,
                                                 conv)
            self.bound.append(conv)
        return self

    def __exit__(self, exc_type, *args):
        for conv in self.bound:
            del conv._message_intervention
        self.bound.clear()
        if exc_type is None and self.selection is not None and self.selected_hits != 1:
            raise ValueError(
                'Selected message occurrence was not executed exactly once')

    def _validate_messages(self, layer, relation, conv, messages, edges):
        if not torch.isfinite(messages).all():
            raise ValueError('Nonfinite native messages')
        if type(conv) is PrimaryKeyToForeignKeyConv and edges[1].unique().numel(
        ) != edges.shape[1]:
            raise ValueError('Gather has repeated destination occurrences')
        if any(o['layer'] == layer and tuple(o['relation']) == relation
               for o in self.observations):
            raise ValueError('Convolution executed more than once')

    def _apply(self, layer, relation, conv, messages, edges):
        self._validate_messages(layer, relation, conv, messages, edges)
        selected = self.selection is None or self.selection[:2] == (layer,
                                                                    relation)
        indices = []
        if selected:
            indices = list(range(
                messages.shape[0])) if self.selection is None else [
                    self.selection[2]
                ]
            if any(i >= messages.shape[0] for i in indices):
                raise ValueError(
                    'Selected edge occurrence is outside retained layer')
            if self.selection is not None:
                self.selected_hits += 1
        gates = messages.new_ones((messages.shape[0], 1))
        gates[indices] = self.gate
        result = gates * messages
        self.observations.append(
            dict(
                layer=layer, relation=list(relation), count=messages.shape[0],
                gate=self.gate, selected_indices=indices,
                device=str(messages.device),
                selected_edges=edges[:, indices].detach().cpu().tolist(),
                zeroed_nonzero_elements=int(
                    ((messages != 0) & (result == 0)).sum().item())))
        return result


class DifferentiableMessageMasking(MessageIntervention):
    """Mix each native occurrence with a shared learnable layer stand-in.

    Gates map (one-based layer, typed relation) to a vector in [0, 1].
    Baselines map executed layers to channel vectors. Both retain autograd.
    Predictor parameters must be frozen and every module in evaluation mode.
    Only the structurally qualified sum/mean/gather profile is admitted.
    """
    def __init__(self, module, *, gates, baselines):
        super().__init__(module, gate=1)
        self.gates = dict(gates)
        self.baselines = dict(baselines)

    def __enter__(self):
        if any(m.training for m in self.module.modules()):
            raise ValueError('Message masking requires evaluation mode')
        if any(p.requires_grad or p.grad is not None
               for p in self.module.parameters()):
            raise ValueError('Message masking requires a frozen predictor')
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise ValueError('Message masking requires autograd enabled')
        return super().__enter__()

    def __exit__(self, exc_type, *args):
        super().__exit__(exc_type, *args)
        if exc_type is None:
            used = {(o['layer'], tuple(o['relation']))
                    for o in self.observations}
            if used != set(self.gates):
                raise ValueError('Gates must cover exactly executed relations')
            if {key[0] for key in used} != set(self.baselines):
                raise ValueError('Baselines must cover exactly executed layers')

    def _apply(self, layer, relation, conv, messages, edges):
        self._validate_messages(layer, relation, conv, messages, edges)
        key = (layer, relation)
        if key not in self.gates or layer not in self.baselines:
            raise ValueError('Missing occurrence gates or layer baseline')
        gates, baseline = self.gates[key], self.baselines[layer]
        for name, value, shape in (('gates', gates, (messages.shape[0], )),
                                   ('baseline', baseline,
                                    (messages.shape[1], ))):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.device != messages.device
                    or value.dtype != messages.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(
                    f'Invalid {name}: shape/device/dtype/finiteness')
        if not ((gates >= 0) & (gates <= 1)).all():
            raise ValueError('Occurrence gates must lie in [0, 1]')
        result = gates[:, None] * messages + (1 - gates[:, None]) * baseline
        self.observations.append(
            dict(layer=layer, relation=list(relation), count=messages.shape[0],
                 device=str(messages.device)))
        return result


class OriginalMessageContext(MessageIntervention):
    """Capture detached original source messages for shared gate inference.

    Reuses the native intervention seam; GraphCapture retains aggregated paths
    rather than the exact per-occurrence input required by the companion.
    """
    def __init__(self, module):
        super().__init__(module, gate=1)
        self.messages = {}

    def _apply(self, layer, relation, conv, messages, edges):
        self._validate_messages(layer, relation, conv, messages, edges)
        self.messages[(layer, relation)] = messages.detach().clone()
        self.observations.append(
            dict(layer=layer, relation=list(relation), count=messages.shape[0]))
        return messages


class AmortizedMessageCompanion(torch.nn.Module):
    """Shared layer/relation message MLPs and layer replacement vectors.

    DifferentiableMessageMasking owns native replacement; this module only
    predicts gates from detached original messages, shared across all cases.
    """
    def __init__(self, keys, channels, hidden_channels=16):
        super().__init__()
        self.keys = tuple(
            sorted((int(layer), tuple(relation)) for layer, relation in keys))
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError('Companion requires distinct layer/relation keys')
        self.networks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(channels, hidden_channels),
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden_channels, 1))
            for _ in self.keys
        ])
        for network in self.networks:
            torch.nn.init.constant_(network[-1].bias, 2.0)
        self.standins = torch.nn.ParameterDict({
            str(layer):
            torch.nn.Parameter(torch.zeros(channels))
            for layer in sorted({key[0]
                                 for key in self.keys})
        })

    def forward(self, contexts, *, hard=False):
        if not set(contexts).issubset(self.keys):
            raise ValueError('Unknown companion layer/relation context')
        gates = {}
        for key, network in zip(self.keys, self.networks):
            if key not in contexts:
                continue
            context = contexts[key]
            if context.requires_grad or context.ndim != 2:
                raise ValueError(
                    'Original message contexts must be detached matrices')
            gate = network(context).squeeze(-1).sigmoid()
            gates[key] = (gate >= 0.5).to(gate.dtype) if hard else gate
        baselines = {
            layer: self.standins[str(layer)]
            for layer in {key[0]
                          for key in gates}
        }
        return gates, baselines
