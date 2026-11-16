# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PANTHER combines explicit composite IDs with separately numbered profiles."""

from dataclasses import replace

import torch

from kumo.pytorch.api import SampleRequest
from kumo.pytorch.architectures import ClassificationConfig, for_classification
from kumo.pytorch.models.protocols import LabeledBatch
from tests.helpers import written_state
from tests.reference_models.batch_definitions import PANTHER_BATCH
from tests.reference_models.fixtures import _tables, adapted_batch, architecture_for, model_task


def test_panther_composite_tokens_round_trip(tmp_path):
    history = PANTHER_BATCH.inputs["history"]
    history = replace(
        history,
        representation=replace(
            history.representation, row=replace(history.representation.row, top_k=1)
        ),
    )
    policy = replace(PANTHER_BATCH, inputs={**PANTHER_BATCH.inputs, "history": history}, targets={})
    with model_task("panther", batch=policy) as task:
        original = next(task.batches(batch_size=2))
        state = task.fitted_state
        task.to_artifact().save(tmp_path / "composite")
        assert written_state(tmp_path / "composite") == state.to_dict()
        assert task.spaces["panther"].size == 4  # Three roles plus one retained tuple.
        assert original.inputs["history"]["value_ids"].shape == (2, 2)
        assert original.inputs["history"]["value_ids"].tolist() == [[3, 1], [3, 1]]
    with model_task("panther", batch=policy, fitted=state) as task:
        restored = next(task.batches(batch_size=2))
        assert torch.equal(
            restored.inputs["history"]["value_ids"], original.inputs["history"]["value_ids"]
        )


def test_panther_pretraining_combines_behavior_and_contrastive_losses():
    tables = _tables()
    tables["users"][1]["country"] = "C002"
    requests = (
        SampleRequest(sample_id="one", entity_id=1, as_of_time=100),
        SampleRequest(sample_id="two", entity_id=2, as_of_time=100),
        SampleRequest(sample_id="three", entity_id=1, as_of_time=100),
    )
    with model_task("panther", tables=tables, requests=requests) as task:
        architecture = architecture_for(task, "panther")
        batch = adapted_batch(task, "panther")
        output = architecture.model(batch)
        assert set(output.losses) == {"next_behavior", "profile_contrastive"}
        assert output.predictions["next_token_logits"].shape == (
            batch.token_count,
            task.spaces["panther"].size,
        )
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert architecture.model.encoder.prototypes.grad is not None
        assert architecture.model.encoder.convolutions[0].weight.grad is not None


def test_panther_classification_uses_the_complete_history():
    with model_task("panther", targets=False) as task:
        architecture = architecture_for(task, "panther")
        inputs = adapted_batch(task, "panther")
        model = for_classification(
            architecture, ClassificationConfig(classes=2, lora=None, tabular_fusion=None)
        )
        output = model(LabeledBatch(inputs, torch.tensor([0, 1])))
        assert inputs.token_count == 2 * 3  # Two card events plus a model profile slot.
        assert output.predictions["class_logits"].shape == (2, 2)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert model.encoder.prototypes.grad is not None


def test_panther_never_substitutes_status_for_missing_merchant_risk():
    tables = _tables()
    for merchant in tables["merchants"]:
        merchant["risk_score"] = None
    with model_task("panther", tables=tables) as task:
        batch = next(task.batches(batch_size=2))
        # Required risk is absent, so neither card event is an eligible tuple.
        # Empty histories remain empty rather than using status as a fallback.
        assert not batch.inputs["history"]["valid"].any()
        assert not batch.targets["next_behavior"]["valid"].any()
