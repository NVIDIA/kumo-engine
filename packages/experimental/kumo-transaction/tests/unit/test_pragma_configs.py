# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The PRAGMA-S/M/L configs must stay on the published size ladder.

Dimensions are dynamiq-ai/pragmatiq's `small` / `medium` / `large` presets.
The roadmap's PRAGMA comparison records which rung produced a result
(docs/reference/roadmap.md), so the rungs cannot drift silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tfm.api.builders import profile_history_from_config
from tfm.config import TrainingConfig, load_recipe_config
from tfm.training.backends import backend_capabilities, check_backend_supports

RECIPE = Path("recipes/pragma_like")

# size: (hidden, heads, profile_depth, event_depth, history_depth)
LADDER = {
    "s": (192, 3, 1, 5, 2),
    "m": (512, 8, 3, 16, 6),
    "l": (1024, 16, 9, 45, 18),
}


@pytest.mark.parametrize("rung", sorted(LADDER))
def test_ladder_config_matches_its_pragmatiq_preset(rung: str) -> None:
    config = load_recipe_config(RECIPE / f"config.pragma_{rung}.yaml")
    gran = config.model.composition.granularity
    hidden, heads, profile_depth, event_depth, history_depth = LADDER[rung]

    assert gran.decoder.hidden_size == hidden
    assert gran.decoder.num_heads == heads
    assert gran.profile_depth == profile_depth
    assert gran.event_depth == event_depth
    assert gran.history_depth == history_depth
    assert gran.type == "profile_history"


@pytest.mark.parametrize("rung", sorted(LADDER))
def test_every_rung_uses_the_pragma_objective_and_readout(rung: str) -> None:
    config = load_recipe_config(RECIPE / f"config.pragma_{rung}.yaml")
    composition = config.model.composition

    assert composition.objective == "token_event_key_mlm"
    assert composition.readout == "usr_token"
    assert composition.position == "time_rope"
    # Paper masking rates plus label smoothing.
    assert config.objective.mlm.p_token == 0.15
    assert config.objective.mlm.p_event == 0.10
    assert config.objective.mlm.p_key == 0.10
    assert config.objective.mlm.p_unk == 0.10
    assert config.objective.mlm.label_smoothing == 0.1


@pytest.mark.parametrize("rung", sorted(LADDER))
def test_every_rung_runs_padding_free_with_a_lora_stage(rung: str) -> None:
    config = load_recipe_config(RECIPE / f"config.pragma_{rung}.yaml")

    assert config.model.composition.granularity.varlen is True
    assert config.model.composition.attention == "sdpa"
    assert config.fraud_task.finetune.enabled is True
    assert config.fraud_task.finetune.method == "lora"
    # The frozen probe stays the reference baseline alongside it.
    assert config.fraud_task.use_tfm_embeddings is True
    assert config.fraud_task.use_raw_features is True


@pytest.mark.parametrize("rung", sorted(LADDER))
def test_every_rung_is_runnable_on_its_declared_backend(rung: str) -> None:
    config = load_recipe_config(RECIPE / f"config.pragma_{rung}.yaml")
    capabilities = backend_capabilities(config.training.backend)
    check_backend_supports(
        capabilities,
        backbone=config.model.backbone,
        packing=config.training.packing,
        device_type="cuda",
        world_size=2 if capabilities.multi_process else 1,
        layout="key_value_time",
        precision=config.training.precision,
    )


def test_the_large_rung_keeps_its_global_batch_on_two_ranks() -> None:
    config = load_recipe_config(RECIPE / "config.pragma_l.yaml")
    assert config.training.precision == "bf16"
    assert config.training.backend == "ddp"
    assert config.training.batch_size * 2 == 8


def test_training_precision_rejects_unknown_values() -> None:
    config = load_recipe_config(RECIPE / "config.pragma_s.yaml")
    with pytest.raises(ValidationError, match="precision"):
        TrainingConfig.model_validate({**config.training.model_dump(), "precision": "fp16"})


def test_default_is_matched_to_blueprint_v2_on_everything_but_the_architecture() -> None:
    pragma = load_recipe_config(RECIPE / "config.yaml")
    v2 = load_recipe_config(Path("recipes/blueprint_v2/config.yaml"))

    for section in ("seed", "data", "features", "training", "fraud_task"):
        assert getattr(pragma, section) == getattr(v2, section), section
    gran, v2_gran = pragma.model.composition.granularity, v2.model.composition.granularity
    assert gran.type == "profile_history"
    assert (gran.decoder.hidden_size, gran.decoder.num_heads) == (
        v2_gran.decoder.hidden_size,
        v2_gran.decoder.num_heads,
    )
    assert (gran.profile_depth, gran.event_depth) == (v2_gran.profile_depth, v2_gran.event_depth)
    # PRAGMA has no next-event head, so the surprise arm is the only probe it skips.
    assert pragma.evaluation == v2.evaluation.model_copy(update={"surprise": False})


def test_the_small_rung_builds_and_honors_its_composition() -> None:
    config = load_recipe_config(RECIPE / "config.pragma_s.yaml")

    model = profile_history_from_config(config, vocab_size=1024)

    assert model.spec.hidden_size == 192
    assert model.spec.varlen is True
    assert model.attention_spec.name == "sdpa"
    assert model.objective_spec.label_smoothing == 0.1
    assert len(model.event_encoder.blocks) == 5
    assert len(model.profile_encoder.blocks) == 1
    assert len(model.history_blocks) == 2
