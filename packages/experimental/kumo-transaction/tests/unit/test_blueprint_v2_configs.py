# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Blueprint v2 configs build the model ADR 0012 describes."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tfm.api.builders import profile_event_decoder_from_config
from tfm.config import RecipeConfig, load_recipe_config

RECIPE = Path("recipes/blueprint_v2")
ABLATIONS = Path("benchmarks/ablations")


def _build(name: str | Path, vocab_size: int = 1024):
    config = load_recipe_config(name if isinstance(name, Path) else RECIPE / name)
    return config, profile_event_decoder_from_config(config, vocab_size=vocab_size)


@pytest.mark.parametrize(
    "path",
    [
        RECIPE / "config.smoke.yaml",
        RECIPE / "config.scale.yaml",
        RECIPE / "config.yaml",
        ABLATIONS / "blueprint_v2_lab.yaml",
        ABLATIONS / "blueprint_v2_final_d.yaml",
    ],
    ids=lambda p: p.name,
)
def test_config_trains_next_event_fields_with_mlm(path: Path) -> None:
    config, model = _build(path)
    assert config.model.composition.readout == "last_token"
    assert config.model.composition.position == "time_rope"
    assert model.objective_spec.name == "next_event_fields"
    assert model.mlm_spec is not None
    assert model.mlm_spec.label_smoothing == 0.1


def test_final_sweep_candidate_d_is_the_recipe_with_its_four_overrides() -> None:
    recipe = load_recipe_config(RECIPE / "config.yaml").model_dump(mode="json")
    candidate = load_recipe_config(ABLATIONS / "blueprint_v2_final_d.yaml").model_dump(mode="json")

    overrides = {
        ("training", "max_steps"): (1500, 3000),
        ("training", "optimizer", "warmup_steps"): (75, 150),
        ("fraud_task", "finetune", "rank"): (8, 16),
        ("fraud_task", "finetune", "alpha"): (8.0, 16.0),
    }
    for keys, (in_recipe, in_candidate) in overrides.items():
        recipe_block, candidate_block = recipe, candidate
        for key in keys[:-1]:
            recipe_block, candidate_block = recipe_block[key], candidate_block[key]
        assert (recipe_block[keys[-1]], candidate_block[keys[-1]]) == (in_recipe, in_candidate)
        candidate_block[keys[-1]] = in_recipe
    assert candidate == recipe


def test_pragmatiq_config_is_about_100m_parameters() -> None:
    _config, model = _build("config.scale.yaml")
    spec = model.spec
    assert (spec.hidden_size, spec.num_heads) == (512, 8)
    assert (spec.profile_depth, spec.event_depth, spec.decoder_depth) == (3, 16, 12)
    assert spec.varlen is True
    assert model.attention_spec.name == "sdpa"
    non_embedding = sum(p.numel() for p in model.parameters()) - model.embed.weight.numel()
    assert 95e6 < non_embedding < 105e6


def test_unknown_auxiliary_objective_is_rejected() -> None:
    raw = load_recipe_config(RECIPE / "config.smoke.yaml").model_dump(mode="json")
    raw["objective"]["auxiliary"] = ["not_an_objective"]
    with pytest.raises(ValidationError, match="auxiliary"):
        RecipeConfig.model_validate(raw)


def test_auxiliary_objectives_are_built_in_order() -> None:
    raw = load_recipe_config(RECIPE / "config.smoke.yaml").model_dump(mode="json")
    raw["objective"]["auxiliary"] = ["past_reconstruction", "subsequence_contrastive"]
    model = profile_event_decoder_from_config(RecipeConfig.model_validate(raw), vocab_size=64)
    assert model.mlm_spec is None
    assert model.auxiliary_names == ("past_reconstruction", "subsequence_contrastive")


def test_usr_readout_on_the_causal_decoder_names_rule_3() -> None:
    from tfm.contracts.errors import IncompatibleCompositionError

    raw = load_recipe_config(RECIPE / "config.smoke.yaml").model_dump(mode="json")
    raw["model"]["composition"]["readout"] = "usr_token"
    with pytest.raises(IncompatibleCompositionError, match="rule 3"):
        profile_event_decoder_from_config(RecipeConfig.model_validate(raw), vocab_size=64)


def _leaves(tree: object, prefix: str = "") -> dict[str, object]:
    if isinstance(tree, dict):
        out: dict[str, object] = {}
        for key, value in tree.items():
            out.update(_leaves(value, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): tree}


def test_v1_flat_baseline_changes_only_the_model_and_its_tokens() -> None:
    """Criterion 6 compares models on the same rows; data, training, and probes must match."""
    hybrid = _leaves(load_recipe_config(ABLATIONS / "blueprint_v2_lab.yaml").model_dump())
    flat = _leaves(load_recipe_config(ABLATIONS / "blueprint_v2_lab_v1_flat.yaml").model_dump())

    changed = {key for key in hybrid.keys() | flat.keys() if hybrid.get(key) != flat.get(key)}

    allowed = ("model.", "features.tokenizer", "data.max_tokens", "objective.")
    assert changed and all(key.startswith(allowed) for key in changed), sorted(changed)
    assert flat["model.composition.granularity.type"] == "flat"
    assert flat["model.composition.position"] == "nope"
    assert flat["model.composition.objective"] == "next_event"


def test_tuned_lab_config_keeps_the_model_and_the_probe_rows() -> None:
    """The tuned config changes tokens, corpus, optimizer, fine-tuning, and how arms are
    scored, not which rows are probed."""
    lab = _leaves(load_recipe_config(ABLATIONS / "blueprint_v2_lab.yaml").model_dump())
    tuned = _leaves(load_recipe_config(RECIPE / "config.yaml").model_dump())

    changed = {key for key in lab.keys() | tuned.keys() if lab.get(key) != tuned.get(key)}

    allowed = (
        "features.",
        "training.",
        "fraud_task.finetune.",
        "data.pretrain_windows",
        "data.token_cache",
        "evaluation.surprise",
        "evaluation.tabular_baselines.xgboost_trials",
        "evaluation.tabular_baselines.xgboost_params",
    )
    assert changed and all(key.startswith(allowed) for key in changed), sorted(changed)
    assert tuned["training.optimizer.name"] == "muon"
    assert tuned["features.hashed_keys"] == ["merchant", "device_id", "counterparty"]


def test_the_tuned_config_scores_with_the_lab_notebooks_fixed_setting() -> None:
    from tfm.runtime.lab import LAB_XGBOOST_PARAMS

    evaluation = load_recipe_config(RECIPE / "config.yaml").evaluation
    assert evaluation.surprise
    assert evaluation.tabular_baselines.xgboost_trials == 0
    assert evaluation.tabular_baselines.xgboost_params == LAB_XGBOOST_PARAMS
    assert isinstance(evaluation.tabular_baselines.xgboost_params["max_depth"], int)


def _with_evaluation(**changes: object) -> dict:
    raw = load_recipe_config(RECIPE / "config.yaml").model_dump(mode="json")
    for dotted, value in changes.items():
        *path, leaf = dotted.split("__")
        node = raw["evaluation"]
        for key in path:
            node = node[key]
        node[leaf] = value
    return raw


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"tabular_baselines__xgboost_trials": 30}, "not both"),
        ({"tabular_baselines__xgboost_params": {"max_depth": 3, "eta": 0.1}}, "Unknown"),
        ({"tabular_baselines__enabled": False}, "evaluation.surprise"),
        (
            {"tabular_baselines__xgboost_params": None, "tabular_baselines__xgboost_trials": 0},
            "evaluation.surprise",
        ),
    ],
)
def test_an_ambiguous_or_unscorable_evaluation_is_refused(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        RecipeConfig.model_validate(_with_evaluation(**changes))
