# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Each sample/batch recipe is executed before its model is constructed."""

import torch

from tests.reference_models.architecture_definitions import ARCHITECTURES
from tests.reference_models.fixtures import (
    adapted_batch,
    architecture_for,
    model_task,
    requests_for,
)


def test_four_data_recipes_feed_the_same_model_api():
    for name in ARCHITECTURES:
        with model_task(name) as task:
            samples = task.preview(requests_for(name)).to_dict()["samples"]
            architecture = architecture_for(task, name)
            output = architecture.model(adapted_batch(task, name))
            assert len(samples) == 2
            assert all(sample["events"] for sample in samples)
            assert output.representations["entity"].shape == (2, 24)
            assert torch.isfinite(output.loss)


def test_four_data_recipes_feed_the_model_independent_batch_api():
    for name in ARCHITECTURES:
        with model_task(name) as task:
            batch = next(task.batches(batch_size=2, seed=17))
            assert batch.batch_size == 2
            assert batch.inputs
            assert batch.targets
            assert batch.metadata["sample_ids"] == (f"{name}-1", f"{name}-2")
