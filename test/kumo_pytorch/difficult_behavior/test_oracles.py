# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove that semantic tie handling does not accept corrupted history answers."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

import kumo.pytorch.api as r

from .test_history_graph import fixture, verify_history


@pytest.mark.parametrize("packed", [False, True], ids=["padded", "packed"])
@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "clock",
        "label",
        "age",
        "validity",
        "count",
        "age-mask",
        "age-layout",
        "identity-mask",
        "component-mask",
        "clock-mask",
    ],
)
def test_history_oracle_rejects_corruption(packed, fault):
    _, rows = fixture(r.Packed() if packed else r.Padded(), "history")

    def tensor(values):
        return torch.tensor(
            values if packed else [values],
            dtype=torch.float64 if isinstance(values[0], float) else None,
        )

    valid = tensor([True] * 6)
    present = tensor([True] * 4 + [False] * 2)
    history = {
        "valid": valid.clone(),
        **{
            name: {
                "values": tensor(values),
                "valid": (present if name == "value" else valid).clone(),
            }
            for name, values in {
                "identity": [300, 301, 300, 301, 302, 302],
                "component": [0, 0, 1, 1, 0, 1],
                "clock": [99, 99, 99, 99, 100, 100],
                "value": [5.0, 5.0, 5.0, 5.0, 0.0, 0.0],
            }.items()
        },
    }
    ages = {
        "valid": valid.clone(),
        "age": {
            "values": tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
            "valid": valid.clone(),
        },
    }
    aligned = {
        "hidden": {"values": tensor([10, 11, 10, 11, 0, 0]), "valid": present.clone()}
    }
    if packed:
        for output in (history, ages, aligned):
            output["offsets"] = torch.tensor([0, 6])
    batch = SimpleNamespace(
        inputs={"history": history, "ages": ages},
        targets={"aligned": aligned},
        metadata={"sample_ids": ["duplicates"]},
    )
    expected = {"duplicates": (3, 100)}
    verify_history(batch, rows, expected)
    # Tied permutations and payload storage beneath false masks are legitimate.
    permutation = [2, 0, 3, 1, 5, 4]

    def permute(tree):
        if isinstance(tree, dict):
            return {
                key: value if key == "offsets" else permute(value)
                for key, value in tree.items()
            }
        return tree[..., permutation] if isinstance(tree, torch.Tensor) else tree

    allowed = deepcopy(batch)
    allowed.inputs = permute(allowed.inputs)
    allowed.targets = permute(allowed.targets)
    allowed.inputs["history"]["value"]["values"].reshape(-1)[-2:] = 5.0
    verify_history(allowed, rows, expected)
    broken = deepcopy(batch)
    changes = {
        "identity": (broken.inputs["history"]["identity"]["values"], 999),
        "clock": (broken.inputs["history"]["clock"]["values"], 101),
        "label": (broken.targets["aligned"]["hidden"]["values"], 777),
        "age": (broken.inputs["ages"]["age"]["values"], -1.0),
        "validity": (broken.inputs["history"]["value"]["valid"], False),
        "count": (broken.inputs["history"]["valid"], False),
        "age-mask": (broken.inputs["ages"]["age"]["valid"], False),
        "age-layout": (broken.inputs["ages"]["valid"], False),
        "identity-mask": (broken.inputs["history"]["identity"]["valid"], False),
        "component-mask": (broken.inputs["history"]["component"]["valid"], False),
        "clock-mask": (broken.inputs["history"]["clock"]["valid"], False),
    }
    values, replacement = changes[fault]
    values.reshape(-1)[0] = replacement
    with pytest.raises(AssertionError):
        verify_history(broken, rows, expected)
