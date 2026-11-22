# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small package consumers for the registered graph, token and combined workloads."""

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from e2e.datasets.aml import l01_loss
from kumo.pytorch.models.causal import CausalEncoderConfig, CausalTokenEncoder
from tests.reference_models.batch_adapters import nuformer_batch, pragma_batch
from tests.test_model import _smoke_model


def make_model(task, binding):
    if binding is None:
        from tests.e2e.streaming_training import CountSAGE

        return CountSAGE(task)
    return BindingModel(task, binding)


class BindingModel(nn.Module):
    def __init__(self, task, binding):
        super().__init__()
        self.binding = binding
        self.heads = {
            "L01": ("amount", "currency"),
            "M03": ("event_type", "event_subtype"),
            "M04": ("count",),
            "E03": ("masked_values",),
            "S01": ("masked_values",),
            "E04": ("next_token",),
            "S02": ("next_token",),
        }.get(binding, ("label",))
        if self.heads == ("masked_values",):
            self.masked = _smoke_model(task.spaces["history"])
            return
        if binding == "L01":
            from tests.e2e.l01_model import _Combined

            self.combined = _Combined(task)
            return
        inputs = task.contract.batch.inputs
        if "graph" in inputs:
            self.graph = SAGEConv((2, 2), 16, root_weight=False)
        if "history" in inputs:
            space = task.spaces["history"]
            self.sequence = CausalTokenEncoder(
                CausalEncoderConfig(
                    d_model=16,
                    d_ffn=32,
                    attention_heads=2,
                    layers=1,
                    dropout=0,
                ),
                vocabulary_size=space.size,
                pad_id=space.special(role="pad"),
            )
        if self.heads == ("next_token",):
            return
        aggregates = inputs.get("aggregates")
        width = 16 * (int("graph" in inputs) + int("history" in inputs))
        if aggregates is not None:
            width += 2 * len(aggregates.representation.fields)
        self.predict = nn.ModuleDict(
            {
                name: nn.Linear(
                    width, task.output_schema["targets"][name]["value"].get("vocabulary_size", 1)
                )
                for name in self.heads
            }
        )

    def _masked(self, batch):
        # These contracts have no entity-profile tokens. Supply the adapter's empty profile.
        empty = torch.empty(batch.batch_size, 0, dtype=torch.long)
        profile = dict(field_ids=empty, value_ids=empty, position_ids=empty, valid=empty.bool())
        inputs = dict(
            batch.inputs, entity=profile, profile_timing={"age_seconds": {"values": empty}}
        )
        return pragma_batch(replace(batch, inputs=inputs))

    def forward(self, batch):
        if hasattr(self, "masked"):
            adapted = self._masked(batch)
            state = self.masked.encoder(adapted)
            return state.representations["entity"]
        if hasattr(self, "combined"):
            return self.combined(batch)
        vectors = []
        if hasattr(self, "graph"):
            graph = batch.inputs["graph"]
            values, owners = [], []
            for edge in graph["edges"].values():
                src, dst = (graph["nodes"][edge[k]] for k in ("source_type", "destination_type"))
                src_ids, dst_ids = edge["index"]
                batch_ids = src["batch"][src_ids]
                keep = src["ids"][src_ids] == batch.metadata["entity_ids"][batch_ids]
                fields = dst["features"]
                # Each real edge contributes its observed feature values and missingness.
                x = torch.stack(
                    [
                        torch.asinh(f["values"].float()).masked_fill(~f["valid"], 0)
                        for f in fields.values()
                    ],
                    1,
                )
                valid = torch.stack([f["valid"] for f in fields.values()], 1).float()
                values.append(torch.stack((x.mean(1), valid.mean(1)), 1)[dst_ids[keep]])
                owners.append(batch_ids[keep])
            x, dest = torch.cat(values), torch.cat(owners)
            edge_index = torch.stack((torch.arange(len(x)), dest))
            vectors.append(self.graph((x, None), edge_index, size=(len(x), batch.batch_size)))
        if hasattr(self, "sequence"):
            tokens = nuformer_batch(batch)
            if tokens.input_ids.shape[1]:
                state = self.sequence(tokens)
                if self.heads == ("next_token",):
                    return state.contextual
                hidden = state.representations["entity"]
            else:
                hidden = self.sequence.embedding.weight.new_zeros(batch.batch_size, 16)
            vectors.append(hidden)
        if "aggregates" in batch.inputs:
            for name in sorted(batch.inputs["aggregates"]):
                value = batch.inputs["aggregates"][name]
                if isinstance(value, dict):
                    vectors.extend(
                        (
                            value["values"].masked_fill(~value["valid"], 0)[:, None],
                            value["valid"][:, None],
                        )
                    )
        joined = torch.cat(vectors, -1)
        return torch.cat([layer(joined) for layer in self.predict.values()], 1)

    def loss(self, batch):
        if hasattr(self, "masked"):
            return self.masked(self._masked(batch)).loss
        predicted = self(batch)
        if self.binding == "L01":
            return l01_loss(batch, predicted[:, 0], predicted[:, 1:])[0]
        if self.heads == ("next_token",):
            valid = batch.targets["next_token"]["valid"]
            scores = predicted[valid] @ self.sequence.embedding.weight.T
            return F.cross_entropy(scores, batch.targets["next_token"]["values"][valid])
        losses = []
        for name, scores in zip(
            self.heads,
            predicted.split([m.out_features for m in self.predict.values()], 1),
            strict=True,
        ):
            target = batch.targets[name]
            values, scores = target["values"][target["valid"]], scores[target["valid"]]
            if not len(values):
                losses.append(scores.sum() * 0)
            elif name == "count":
                losses.append(F.smooth_l1_loss(scores[:, 0], values))
            elif name == "label":
                losses.append(F.binary_cross_entropy_with_logits(scores[:, 0], values))
            else:
                losses.append(F.cross_entropy(scores, values))
        return sum(losses)
