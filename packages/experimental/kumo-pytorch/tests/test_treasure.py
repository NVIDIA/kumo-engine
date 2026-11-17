# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TREASURE reads typed task columns and withholds its current-event signals."""

from dataclasses import replace

import pytest
import torch

from kumo.pytorch.api import AlignedEventTarget, Encode, Field, Fields, SampleRequest, Vocabulary
from kumo.pytorch.data.typed_vectors import AttributeSchema, TypedVectorBatch
from kumo.pytorch.models.causal import CausalEncoderConfig
from kumo.pytorch.models.model import MultiObjectiveModel
from kumo.pytorch.models.typed import TypedVectorEncoder
from kumo.pytorch.objectives import TypedAttributeObjective
from tests.helpers import written_state
from tests.reference_models.batch_definitions import TREASURE_BATCH
from tests.reference_models.fixtures import (
    SAMPLES,
    _tables,
    adapted_batch,
    architecture_for,
    model_task,
)


def test_treasure_fitted_state_round_trip_and_output_only_signal(tmp_path):
    sample = SAMPLES["treasure"]
    event = sample.events["card_transaction"]
    sample = replace(
        sample,
        events={
            "card_transaction": replace(
                event, fields={**event.fields, "response_code": Field(column="status")}
            )
        },
    )
    policy = replace(
        TREASURE_BATCH,
        targets={
            "current_event": AlignedEventTarget(
                input="events", fields=(Encode(field="response_code", encoding=Vocabulary()),)
            )
        },
    )
    with model_task("treasure", sample=sample, batch=policy) as task:
        original = next(task.batches(batch_size=2))
        state = task.fitted_state
        task.to_artifact().save(tmp_path / "typed-vectors")
        assert written_state(tmp_path / "typed-vectors") == state.to_dict()
        assert "response_code" in task.target("current_event").bindings
        assert "response_code" not in task.output("events").bindings
        assert "response_code" not in original.inputs["events"]
    with model_task("treasure", sample=sample, batch=policy, fitted=state) as task:
        restored = next(task.batches(batch_size=2))
        assert torch.equal(
            restored.targets["current_event"]["response_code"]["values"],
            original.targets["current_event"]["response_code"]["values"],
        )


def test_treasure_predicts_current_signals_and_next_event_attributes():
    with model_task("treasure") as task:
        architecture = architecture_for(task, "treasure")
        output = architecture.model(adapted_batch(task, "treasure"))
        encoder = architecture.model.encoder
        objective = architecture.model.objectives["typed_attributes"]
        assert encoder.static.schema.numerical == ("active_account_count",)
        assert encoder.static.schema.categorical == ("country",)
        assert encoder.dynamic.schema.numerical == ("amount",)
        assert set(encoder.dynamic.schema.categorical) == {
            "channel",
            "country",
            "currency",
            "entry_mode",
            "mcc",
            "recurring_candidate_flag",
        }
        assert objective.signal_schema.categorical == ("status", "synthetic_abnormal")
        assert "current.synthetic_abnormal" in output.losses
        assert "next.amount" in output.losses
        assert "next.mcc" in output.losses
        assert output.predictions["next.amount.distribution"].shape == (2, 2)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert encoder.static.projection[0].weight.grad is not None
        assert objective.signal_embeddings[0].weight.grad is not None


def test_treasure_high_cardinality_head_uses_one_shared_negative_set():
    tables = _tables()
    for index, row in enumerate(tables["card_transactions"]):
        row["description"] = f"merchant-{index}"
    sample = SAMPLES["treasure"]
    event = sample.events["card_transaction"]
    sample = replace(
        sample,
        events={
            "card_transaction": replace(
                event, fields={**event.fields, "description": Field(column="description")}
            )
        },
    )
    events = TREASURE_BATCH.inputs["events"]
    events = replace(
        events, representation=Fields(fields=(Encode(field="description", encoding=Vocabulary()),))
    )
    policy = replace(
        TREASURE_BATCH,
        inputs={**TREASURE_BATCH.inputs, "events": events},
        targets={
            "current_event": AlignedEventTarget(
                input="events", fields=(Encode(field="status", encoding=Vocabulary()),)
            )
        },
    )
    with model_task("treasure", tables=tables, sample=sample, batch=policy) as task:
        public = next(task.batches(batch_size=2))
        description_size = task.output("events").field(name="description").encoder.size
        country_size = task.output("entity").field(name="country").encoder.size
        status_size = task.target("current_event").field(name="status").encoder.size
        static = AttributeSchema(
            numerical=(),
            categorical=("country",),
            categories={"country": tuple(range(country_size))},
        )
        dynamic = AttributeSchema(
            numerical=(),
            categorical=("description",),
            categories={"description": tuple(range(description_size))},
            high_cardinality=frozenset({"description"}),
        )
        signals = AttributeSchema(
            numerical=(), categorical=("status",), categories={"status": tuple(range(status_size))}
        )
        config = CausalEncoderConfig(d_model=24, d_ffn=48, attention_heads=3, layers=1, dropout=0.0)
        encoder = TypedVectorEncoder(config, static, dynamic)
        objective = TypedAttributeObjective(
            config.d_model, dynamic, signals, encoder.dynamic.categorical, negative_samples=2
        )
        model = MultiObjectiveModel(encoder, {"typed_attributes": objective})
        shape = public.inputs["events"]["valid"].shape
        batch = TypedVectorBatch(
            entity_ids=public.metadata["entity_ids"],
            static_numerical=torch.empty((2, 0)),
            static_numerical_present=torch.empty((2, 0), dtype=torch.bool),
            static_categorical=public.inputs["entity"]["country"]["values"][:, None],
            dynamic_numerical=torch.empty((*shape, 0)),
            dynamic_numerical_present=torch.empty((*shape, 0), dtype=torch.bool),
            dynamic_categorical=public.inputs["events"]["description"]["values"][..., None],
            dynamic_categorical_present=public.inputs["events"]["description"]["valid"][..., None],
            signal_numerical=torch.empty((*shape, 0)),
            signal_numerical_present=torch.empty((*shape, 0), dtype=torch.bool),
            signal_categorical=public.targets["current_event"]["status"]["values"][..., None],
            signal_categorical_present=public.targets["current_event"]["status"]["valid"][
                ..., None
            ],
            event_mask=public.inputs["events"]["valid"],
        )
        output = model(batch)
        assert description_size == 6  # Missing, unknown, and four observations.
        assert output.predictions["next.description.logits"].shape == (2, 3)
        assert torch.isfinite(output.loss)


def test_treasure_distinguishes_missing_from_unseen_categories():
    with model_task("treasure") as task:
        state = task.fitted_state
        assert task.output("events").field(name="mcc").encoder.roles == {"missing": 0, "unknown": 1}
    tables = _tables()
    tables["card_transactions"][0].update(amount=None, mcc=None)
    tables["card_transactions"][1].update(amount=0.0, mcc="unseen-mcc")
    with model_task("treasure", tables=tables, fitted=state) as task:
        events = next(task.batches(batch_size=2)).inputs["events"]
        assert events["mcc"]["values"][0].tolist() == [0, 1]
        assert events["mcc"]["valid"][0].tolist() == [False, True]
        assert events["amount"]["values"][0].tolist() == [0.0, 0.0]
        assert events["amount"]["valid"][0].tolist() == [False, True]


def test_treasure_missing_next_field_contributes_no_target_loss():
    tables = _tables()
    tables["card_transactions"][1]["mcc"] = None
    tables["card_transactions"][3]["mcc"] = None
    with model_task("treasure", tables=tables) as task:
        architecture = architecture_for(task, "treasure")
        output = architecture.model(adapted_batch(task, "treasure"))
        assert "next.amount" in output.losses
        assert "next.mcc" not in output.losses
        assert torch.isfinite(output.loss)


def test_treasure_requires_entities_from_its_declared_account_table():
    requests = tuple(
        SampleRequest(sample_id=f"wrong-{index}", entity_id=index, as_of_time=100)
        for index in (1, 2)
    )
    with pytest.raises(ValueError, match="accounts"):
        with model_task("treasure", requests=requests):
            pass


def test_treasure_current_signal_must_exist_in_the_sample_shape():
    target = TREASURE_BATCH.targets["current_event"]
    policy = replace(
        TREASURE_BATCH,
        targets={
            "current_event": replace(
                target, fields=(Encode(field="missing_signal", encoding=Vocabulary()),)
            )
        },
    )
    with pytest.raises(ValueError, match="missing_signal"):
        with model_task("treasure", batch=policy):
            pass
