# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nuFormer uses explicit sampled direction and the task's fitted token space."""

from dataclasses import replace

import torch

from kumo.pytorch.api import DataLoaderSpec
from kumo.pytorch.architectures import (
    ClassificationConfig,
    LoraConfig,
    TabularFusionConfig,
    for_classification,
)
from kumo.pytorch.data.sequences import TokenSequenceBatch
from kumo.pytorch.models.common import LowRankLinear
from kumo.pytorch.models.protocols import LabeledBatch
from tests.helpers import written_state
from tests.reference_models.batch_definitions import NUFORMER_BATCH
from tests.reference_models.fixtures import SAMPLES, adapted_batch, architecture_for, model_task


def test_nuformer_fitted_state_and_contract_round_trip(tmp_path):
    with model_task("nuformer") as task:
        original = adapted_batch(task, "nuformer")
        saved = task.fitted_state
        task.to_artifact().save(tmp_path / "nuformer")
        assert written_state(tmp_path / "nuformer") == saved.to_dict()
        contract = DataLoaderSpec.from_json(task.contract.to_json())
        assert contract == task.contract
    with model_task("nuformer", fitted=saved) as task:
        restored = adapted_batch(task, "nuformer")
        assert torch.equal(restored.input_ids, original.input_ids)
        assert restored.batch_size == 2
        assert restored.token_count > 0
        assert task.fitted_state.to_dict() == saved.to_dict()


def test_nuformer_pretraining_forward_backward():
    with model_task("nuformer") as task:
        architecture = architecture_for(task, "nuformer")
        batch = adapted_batch(task, "nuformer")
        output = architecture.model(batch)
        assert output.predictions["next_token_logits"].shape == (
            batch.token_count,
            task.spaces["nuformer"].size,
        )
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert architecture.model.encoder.embedding.weight.grad is not None


def test_nuformer_uses_declared_direction_instead_of_positive_amount_sign():
    with model_task("nuformer") as task:
        public = next(task.batches(batch_size=2))
        direction = task.output("history").field(name="direction")
        emitted = public.inputs["history"]["value_ids"][0].tolist()
        assert direction.encode(value="inflow") in emitted
        assert direction.encode(value="outflow") in emitted
        # Both raw amounts are positive; the explicit transaction type mapping
        # still distinguishes deposit from withdrawal.
        assert direction.encode(value="inflow") != direction.encode(value="outflow")


def test_omitting_direction_does_not_read_an_implicit_direction_field():
    history = NUFORMER_BATCH.inputs["history"]
    fields = tuple(field for field in history.representation.fields if field.field != "direction")
    policy = replace(
        NUFORMER_BATCH,
        inputs={
            "history": replace(
                history, representation=replace(history.representation, fields=fields)
            )
        },
    )
    sample = SAMPLES["nuformer"]
    events = {
        name: replace(
            event,
            fields={
                key: value
                for key, value in event.fields.items()
                if key not in {"direction", "transaction_type"}
            },
        )
        for name, event in sample.events.items()
    }
    with model_task("nuformer", batch=policy, sample=replace(sample, events=events)) as task:
        assert "direction" not in task.output("history").bindings
        assert adapted_batch(task, "nuformer").token_count > 0


def test_nuformer_tiny_batch_can_be_overfit():
    torch.manual_seed(9)
    with model_task("nuformer") as task:
        architecture = architecture_for(task, "nuformer")
        full = adapted_batch(task, "nuformer")
        batch = TokenSequenceBatch(
            **{
                name: getattr(full, name)[:1]
                for name in (
                    "entity_ids",
                    "input_ids",
                    "target_ids",
                    "token_mask",
                    "profile_ids",
                    "profile_mask",
                    "contrastive_groups",
                )
            }
        )
        optimizer = torch.optim.AdamW(architecture.model.parameters(), lr=0.03)
        initial = float(architecture.model(batch).loss.detach())
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = architecture.model(batch).loss
            loss.backward()
            optimizer.step()
        assert float(architecture.model(batch).loss.detach()) < initial * 0.25


def test_nuformer_lora_joint_fusion():
    with model_task("nuformer", targets=False) as task:
        architecture = architecture_for(task, "nuformer")
        inputs = adapted_batch(task, "nuformer")
        config = ClassificationConfig(
            classes=2,
            lora=LoraConfig(rank=4, dropout=0.1),
            tabular_fusion=TabularFusionConfig(
                numerical_means=(0.0, 0.0),
                numerical_scales=(1.0, 1.0),
                periodic_frequencies=3,
                categorical_cardinalities=(3,),
                categorical_embedding_width=4,
                layers=2,
                rank=3,
            ),
        )
        source_qkv = architecture.model.encoder.transformer.blocks[0].attention.qkv
        model = for_classification(architecture, config)
        second_model = for_classification(architecture, config)
        batch = LabeledBatch(
            inputs,
            torch.tensor([0, 1]),
            {
                "tabular_numerical": torch.tensor([[0.2, -0.5], [0.7, 0.3]]),
                "tabular_categorical": torch.tensor([[0], [2]]),
            },
        )
        output = model(batch)
        assert architecture.model.encoder.transformer.blocks[0].attention.qkv is source_qkv
        assert not isinstance(source_qkv, LowRankLinear)
        assert second_model is not model
        assert output.predictions["class_logits"].shape == (2, 2)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert model.encoder.sequence.embedding.weight.grad is None
        assert model.encoder.sequence.transformer.blocks[0].attention.qkv.up.weight.grad is not None
        assert model.encoder.tabular_features.frequencies.grad is not None
        assert model.encoder.tabular_features.categorical[0].weight.grad is not None
