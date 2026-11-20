# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-owned heads for independently valid stored scalar targets."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..targets import fitted_target_encoder


@dataclass
class StoredStep:
    logits: dict
    loss: torch.Tensor
    weight: int
    weights: dict
    loss_sums: dict


def output_widths(task, rules):
    state = task.fitted_state.to_dict()
    widths = {}
    for name, rule in rules.items():
        encoder = fitted_target_encoder(state, name)
        if rule["kind"] == "category":
            assert encoder["kind"] == "vocabulary"
            assert encoder["roles"] == {"missing": 0, "unknown": 1}
            widths[name] = encoder["vocabulary_size"]
        else:
            assert encoder["kind"] == "raw" and encoder["dtype"]["name"] == "float32"
            widths[name] = 1
    return widths


class StoredTargetModel(nn.Module):
    """Reuse input encoders; sum independently normalized per-output losses.

    Binary targets use BCE, categories use cross-entropy (including UNKNOWN),
    and raw numerical targets use Smooth L1 with beta=1 in their original units.
    This avoids squaring large raw amounts/delays. No target scaling is fitted.
    """

    def __init__(self, backbone, family, rules, widths, dim, *, check_targets=()):
        super().__init__()
        self.backbone, self.family = backbone, family
        self.rules, self.widths = rules, widths
        self.check_targets = set(check_targets)
        self.head = (
            nn.Linear(dim, sum(widths.values()))
            if family in {"pragma", "nuformer"}
            else None
        )

    def validate_batch(self, batch):
        self.backbone.validate_kumo_batch(batch)

    def forward(self, batch, *, validate_inputs=True):
        if self.family in {"graphsage", "combined"}:
            records = batch.metadata.get("request_metadata", ())
            if len(records) != len(batch.metadata["sample_ids"]) or any(
                "prediction_time_ns" not in row for row in records
            ):
                raise ValueError("Graph heads require original prediction clocks")
            logits = self.backbone.forward_kumo(
                batch, validate_inputs=validate_inputs
            ).logits
        else:
            encoded = (
                self.backbone.encode_kumo(batch, validate_inputs=validate_inputs)
                if self.family == "nuformer"
                else self.backbone.forward_kumo(batch, validate_inputs=validate_inputs)
            )
            logits = self.head(encoded.user_repr)
        if logits.shape != (
            len(batch.metadata["sample_ids"]),
            sum(self.widths.values()),
        ):
            raise ValueError("Stored-target head shape differs from declared outputs")
        return dict(
            zip(
                self.widths,
                logits.split(tuple(self.widths.values()), dim=-1),
                strict=True,
            )
        )

    def training_step(self, batch, *, validate_inputs=True):
        if validate_inputs:
            self.validate_batch(batch)
        logits = self(batch, validate_inputs=validate_inputs)
        size = len(batch.metadata["sample_ids"])
        padding = batch.metadata["is_padding"]
        if padding.shape != (size,) or padding.dtype != torch.bool:
            raise ValueError("Expected one boolean padding flag per anchor")
        if set(batch.targets) != set(self.rules) | self.check_targets:
            raise ValueError("Stored targets differ from the model's outputs")
        loss = sum(value.sum() * 0 for value in logits.values())
        weights, sums = {}, {}
        for name, rule in self.rules.items():
            values, valid = batch.targets[name]["values"], batch.targets[name]["valid"]
            if (
                values.shape != (size,)
                or valid.shape != (size,)
                or valid.dtype != torch.bool
            ):
                raise ValueError(
                    f"Expected scalar values and boolean validity for {name}"
                )
            selected = valid & ~padding
            count = int(selected.sum())
            target, prediction = values[selected], logits[name][selected]
            if not torch.isfinite(prediction).all():
                raise ValueError(f"Nonfinite predictions for {name}")
            if rule["kind"] == "category":
                if (
                    target.dtype != torch.int64
                    or (target < 1).any()
                    or (target >= self.widths[name]).any()
                ):
                    raise ValueError(f"Invalid categorical target IDs for {name}")
                total = (
                    F.cross_entropy(prediction, target, reduction="sum")
                    if count
                    else prediction.sum() * 0
                )
            else:
                if not target.is_floating_point() or not torch.isfinite(target).all():
                    raise ValueError(f"Nonfinite or nonnumerical targets for {name}")
                if rule["kind"] == "binary":
                    if not ((target == 0) | (target == 1)).all():
                        raise ValueError(f"Nonbinary target for {name}")
                    total = F.binary_cross_entropy_with_logits(
                        prediction[:, 0], target, reduction="sum"
                    )
                else:
                    total = F.smooth_l1_loss(
                        prediction[:, 0].double(),
                        target.double(),
                        reduction="sum",
                        beta=1.0,
                    )
            weights[name], sums[name] = count, total
            if count:
                loss = loss + total / count
        return StoredStep(logits, loss, sum(weights.values()), weights, sums)
