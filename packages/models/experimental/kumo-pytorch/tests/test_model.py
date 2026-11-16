# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical model behavior over the explicitly fitted public loader outputs."""

from dataclasses import replace

import pytest
import torch
from torch.utils._pytree import tree_flatten

from kumo.pytorch.models import HierarchicalFieldEncoder, MultiObjectiveModel
from kumo.pytorch.models.protocols import LabeledBatch, ModelBatch, ModelOutput
from kumo.pytorch.objectives import ClassificationObjective, MaskedValueObjective
from tests.reference_models.batch_definitions import PRAGMA_BATCH
from tests.reference_models.fixtures import (
    adapted_batch,
    model_task,
    requests_for,
    small_config,
)


def _masked_batch():
    target = replace(
        PRAGMA_BATCH.targets["masked_values"],
        token_probability=1.0,
        event_probability=0.0,
        key_probability=0.0,
        unknown_probability=0.0,
    )
    return replace(PRAGMA_BATCH, targets={"masked_values": target})


def _smoke_model(space, event_encoder_layout="padded"):
    config = small_config("pragma").encoder
    from kumo.pytorch.config import ModelConfig

    encoder = HierarchicalFieldEncoder(
        ModelConfig(
            variant="test",
            d_model=config.width,
            d_ffn=config.ffn_width,
            attention_heads=config.attention_heads,
            profile_encoder_layers=1,
            event_encoder_layers=1,
            history_encoder_layers=1,
            dropout=0.0,
            event_encoder_layout=event_encoder_layout,
        ),
        vocabulary_size=space.size,
        pad_id=space.special(role="pad"),
        entity_token_id=space.special(role="entity"),
        event_token_id=space.special(role="event"),
    )
    return MultiObjectiveModel(encoder, {"masked_value": MaskedValueObjective(encoder.embedding)})


def test_model_forward_and_backward():
    with model_task("pragma", batch=_masked_batch()) as task:
        batch = adapted_batch(task, "pragma")
        model = _smoke_model(task.spaces["pragma"])
        output = model(batch)
        output_tensors, _ = tree_flatten(output)
        assert any(tensor is output.loss for tensor in output_tensors)
        assert any(tensor is output.predictions["masked_value_logits"] for tensor in output_tensors)
        assert isinstance(batch, ModelBatch)
        assert isinstance(output, ModelOutput)
        assert output.predictions["masked_value_logits"].shape == (
            batch.prediction_count,
            task.spaces["pragma"].size,
        )
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert model.encoder.embedding.weight.grad is not None
        assert model.encoder.module_times_seconds() == {}


def test_tiny_batch_can_be_overfit():
    torch.manual_seed(3)
    with model_task("pragma", batch=_masked_batch(), requests=requests_for("pragma")[:1]) as task:
        batch = adapted_batch(task, "pragma")
        model = _smoke_model(task.spaces["pragma"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        initial = float(model(batch).loss.detach())
        for _step in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch).loss
            loss.backward()
            optimizer.step()
        assert float(model(batch).loss.detach()) < initial * 0.25


def test_packed_event_encoder_matches_padded_reference():
    torch.manual_seed(7)
    with model_task("pragma", batch=_masked_batch()) as task:
        batch = adapted_batch(task, "pragma")
        padded = _smoke_model(task.spaces["pragma"], "padded")
        packed = _smoke_model(task.spaces["pragma"], "packed")
        packed.load_state_dict(padded.state_dict())
        padded_output, packed_output = padded(batch), packed(batch)
        torch.testing.assert_close(
            packed_output.predictions["masked_value_logits"],
            padded_output.predictions["masked_value_logits"],
        )
        torch.testing.assert_close(packed_output.loss, padded_output.loss)
        torch.testing.assert_close(
            packed_output.representations["entity"], padded_output.representations["entity"]
        )
        padded_output.loss.backward()
        packed_output.loss.backward()
        for (padded_name, padded_parameter), (packed_name, packed_parameter) in zip(
            padded.named_parameters(), packed.named_parameters(), strict=True
        ):
            assert padded_name == packed_name
            torch.testing.assert_close(packed_parameter.grad, padded_parameter.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_packed_event_encoder_runs_with_cuda_fp16():
    with model_task("pragma", batch=_masked_batch()) as task:
        batch = adapted_batch(task, "pragma").to("cuda")
        model = _smoke_model(task.spaces["pragma"], "packed").to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(batch)
        output.loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(output.loss)
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_pragma_classification_uses_unmasked_entity_representation():
    with model_task("pragma", targets=False) as task:
        inputs = adapted_batch(task, "pragma")
        assert inputs.prediction_count == 0
        batch = LabeledBatch(inputs, torch.tensor([0, 1]))
        encoder = _smoke_model(task.spaces["pragma"]).encoder
        model = MultiObjectiveModel(
            encoder, {"classification": ClassificationObjective(encoder.config.d_model)}
        )
        output = model(batch)
        assert output.predictions["class_logits"].shape == (2, 2)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert model.encoder.embedding.weight.grad is not None


def test_pragma_timing_is_explicit_and_latest_relative():
    with model_task("pragma") as task:
        public = next(task.batches(batch_size=2))
        adapted = adapted_batch(task, "pragma")
        # Source times are60,65,80,85; anchor100 produces40,35,20,15.
        assert (
            public.inputs["history_timing"]["age_seconds"]["values"].tolist()
            == [[40.0, 35.0, 20.0, 15.0]] * 2
        )
        # The leading zero is model-owned summary; real newest-event age isalsozero.
        assert adapted.history_elapsed_seconds.tolist() == [[0.0, 25.0, 20.0, 5.0, 0.0]] * 2
        assert "elapsed_seconds" not in public.inputs["history"]["sequence"]
