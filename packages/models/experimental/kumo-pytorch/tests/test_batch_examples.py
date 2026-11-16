# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paper-shaped reference batches and explicit model adapters agree on fitted ID spaces."""

import pytest
import torch

from tests.reference_models.fixtures import adapted_batch, architecture_for, model_task


@pytest.mark.parametrize("name", ("pragma", "nuformer", "panther", "treasure"))
def test_each_paper_batch_runs_through_its_reference_model(name):
    with model_task(name) as task:
        public = next(task.batches(batch_size=2, seed=1729))
        architecture = architecture_for(task, name)
        model_batch = architecture.adapt(public)
        output = architecture.model(model_batch)
        assert public.inputs
        assert public.targets
        if name == "panther":
            assert tuple(public.inputs) == ("entity", "history")
            assert "profile_ids" not in public.inputs["history"]
            assert (
                public.targets["next_behavior"]["values"].shape[1]
                == public.inputs["history"]["value_ids"].shape[1] + 1
            )
        if name == "treasure":
            assert tuple(public.inputs) == ("entity", "events")
        assert torch.isfinite(output.loss)
        output.loss.backward()


def test_masking_exposes_validity_corruption_and_target_masks_separately():
    with model_task("pragma") as task:
        first = next(task.batches(batch_size=2, epoch=3, seed=99))
        repeated = next(task.batches(batch_size=2, epoch=3, seed=99))
        target = first.targets["masked_values"]
        assert torch.equal(target["corruption"], repeated.targets["masked_values"]["corruption"])
        assert torch.equal(target["valid"], repeated.targets["masked_values"]["valid"])
        assert not bool(target["valid"].logical_and(~target["corruption"]).any())
        assert set(first.inputs) == {
            "entity",
            "history",
            "profile_timing",
            "history_calendar",
            "history_timing",
        }
        assert "valid" in first.inputs["history"]
        assert "sequence" in first.inputs["history"]
        assert adapted_batch(task, "pragma").history_elapsed_seconds.dtype == torch.float64
