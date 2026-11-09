# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model construction reads fitted task metadata and performs no fitting."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from kumo.pytorch.api import DType, Encode, Raw
from kumo.pytorch.architectures import (
    ArchitectureConfig,
    NextTokenConfig,
    ObjectiveDefinition,
    build_architecture,
    for_classification,
)
from kumo.pytorch.models import CausalTokenEncoder
from kumo.pytorch.models.common import LowRankLinear
from kumo.pytorch.models.layers import TransformerStack
from kumo.pytorch.models.protocols import LabeledBatch, ModelBatch, ModelOutput
from tests.reference_models.architecture_definitions import ARCHITECTURES, CLASSIFICATION_TASKS
from tests.reference_models.batch_adapters import _token_batch
from tests.reference_models.batch_definitions import BATCH_SPECS
from tests.reference_models.data_schema import SYNTHETIC_SCHEMA
from tests.reference_models.fixtures import (
    adapted_batch,
    architecture_for,
    model_task,
    small_config,
)


@pytest.mark.parametrize("name", tuple(ARCHITECTURES))
def test_every_configuration_uses_the_same_pretraining_interface(name):
    with model_task(name) as task:
        before = task.fitted_state.to_dict()
        with patch(
            "kumo.pytorch.assembly.fit_encoder",
            side_effect=AssertionError("model construction fitted data"),
        ):
            architecture = architecture_for(task, name)
            batch = architecture.adapt(next(task.batches(batch_size=2)))
        output = architecture.model(batch)
        assert isinstance(batch, ModelBatch)
        assert isinstance(output, ModelOutput)
        assert output.representations["entity"].shape == (2, 24)
        assert torch.isfinite(output.loss)
        assert task.fitted_state.to_dict() == before
        output.loss.backward()


@pytest.mark.parametrize("name", tuple(CLASSIFICATION_TASKS))
def test_supported_finetuning_paths_share_one_labeled_batch_contract(name):
    with model_task(name, targets=False) as task:
        architecture = architecture_for(task, name)
        inputs = adapted_batch(task, name)
        task_config = CLASSIFICATION_TASKS[name]
        features = {}
        if task_config.tabular_fusion is not None:
            tabular = task_config.tabular_fusion
            features = {
                "tabular_numerical": torch.zeros((2, len(tabular.numerical_means))),
                "tabular_categorical": torch.zeros(
                    (2, len(tabular.categorical_cardinalities)), dtype=torch.long
                ),
            }
        output = for_classification(architecture, task_config)(
            LabeledBatch(inputs, torch.tensor([0, 1]), features)
        )
        assert output.predictions["class_logits"].shape == (2, 2)
        assert output.representations["entity"].shape[0] == 2
        assert torch.isfinite(output.loss)
        output.loss.backward()


def test_pragma_lora_copies_the_encoder_and_adapts_all_three_stacks():
    with model_task("pragma", targets=False) as task:
        architecture = architecture_for(task, "pragma")
        source_stacks = [
            module
            for module in architecture.model.encoder.modules()
            if isinstance(module, TransformerStack)
        ]
        adapted = for_classification(architecture, CLASSIFICATION_TASKS["pragma"])
        adapted_stacks = [
            module for module in adapted.encoder.modules() if isinstance(module, TransformerStack)
        ]
        assert len(source_stacks) == len(adapted_stacks) == 3
        assert all(
            not isinstance(block.attention.qkv, LowRankLinear)
            for stack in source_stacks
            for block in stack.blocks
        )
        assert all(
            isinstance(block.attention.qkv, LowRankLinear)
            for stack in adapted_stacks
            for block in stack.blocks
        )
        adapted(LabeledBatch(adapted_batch(task, "pragma"), torch.tensor([0, 1]))).loss.backward()
        assert architecture.model.encoder.embedding.weight.grad is None
        assert all(
            block.attention.qkv.up.weight.grad is not None
            for stack in adapted_stacks
            for block in stack.blocks
        )


def test_model_options_can_be_recombined_without_adding_a_named_preset():
    policy = BATCH_SPECS["panther"]
    policy = replace(
        policy,
        targets={"next_behavior": replace(policy.targets["next_behavior"], prefix_input=None)},
    )
    encoder = replace(
        small_config("panther").encoder.transformer, profile_conditioned=False, profile_mix=0.0
    )
    config = ArchitectureConfig(
        encoder=encoder,
        objectives=(
            ObjectiveDefinition(name="next_composite_event", config=NextTokenConfig(), weight=1.0),
        ),
    )
    with model_task("panther", batch=policy) as task:
        architecture = build_architecture(
            task=task,
            config=config,
            bindings={"space": "panther"},
            adapt=lambda batch: _token_batch(batch, target_name="next_behavior"),
        )
        output = architecture.model(architecture.adapt(next(task.batches(batch_size=2))))
        assert isinstance(architecture.model.encoder, CausalTokenEncoder)
        assert torch.isfinite(output.loss)


def test_treasure_field_location_and_type_come_from_fitted_output_bindings():
    with model_task("treasure") as task:
        assert task.output("entity").field(name="active_account_count").encoder.dtype == DType(
            name="int64"
        )
        assert task.output("entity").field(name="country").encoder.dtype == DType(name="string")
        assert task.output("events").field(name="amount").encoder.dtype == DType(name="float64")
        assert tuple(task.target("current_event").bindings) == ("status", "synthetic_abnormal")


def test_model_field_roles_must_exist_in_the_bound_sample_shape():
    policy = BATCH_SPECS["nuformer"]
    history = policy.inputs["history"]
    fields = (
        replace(history.representation.fields[0], field="missing_amount"),
        *history.representation.fields[1:],
    )
    policy = replace(
        policy,
        inputs={
            "history": replace(
                history, representation=replace(history.representation, fields=fields)
            )
        },
    )
    with pytest.raises(ValueError, match="missing_amount"):
        with model_task("nuformer", batch=policy):
            pass


def test_model_field_roles_use_declared_raw_types():
    tables = tuple(
        replace(table, columns={**table.columns, "amount": DType(name="string")})
        if table.name == "banking_transactions"
        else table
        for table in SYNTHETIC_SCHEMA.tables
    )
    schema = replace(SYNTHETIC_SCHEMA, tables=tables)
    with pytest.raises(ValueError, match="abs.*cannot produce"):
        with model_task("nuformer", schema=schema):
            pass


def test_profile_conditioning_roles_must_exist_in_the_bound_sample_shape():
    policy = BATCH_SPECS["panther"]
    profile = policy.inputs["entity"]
    profile = replace(
        profile,
        representation=replace(
            profile.representation,
            fields=(
                *profile.representation.fields,
                Encode(field="missing_profile_value", encoding=Raw()),
            ),
        ),
    )
    policy = replace(policy, inputs={**policy.inputs, "entity": profile})
    with pytest.raises(ValueError, match="missing_profile_value"):
        with model_task("panther", batch=policy):
            pass


def test_model_objective_names_are_unique():
    config = small_config("nuformer")
    config = replace(config, objectives=config.objectives * 2)
    with model_task("nuformer") as task, pytest.raises(ValueError, match="unique"):
        architecture_for(task, "nuformer", config=config)
