# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Blueprint v2 configs build the model Blueprint v2 guide describes."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from tfm.api.builders import profile_event_decoder_from_config
from tfm.config import RecipeConfig, load_recipe_config
from tfm.contracts.errors import IncompatibleCompositionError
from tfm.contracts.kvt import CyclicTime
from tfm.layers.kvt_encoders import CalendarEmbedding

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


def _with_time_features(time_features: dict, granularity: str = "profile_event_decoder"):
    raw = load_recipe_config(RECIPE / "config.smoke.yaml").model_dump(mode="json")
    raw["model"]["composition"]["granularity"].update(type=granularity)
    raw["model"]["composition"]["granularity"]["time_features"] = time_features
    return RecipeConfig.model_validate(raw)


def test_time_features_set_the_arrays_the_calendar_reads() -> None:
    features = {"minute": 60, "dom": {"period": 31, "origin": 1}, "dow": 7, "hour": 24}
    model = profile_event_decoder_from_config(_with_time_features(features), vocab_size=64)
    assert model.event_arrays == ("hour", "dow", "dom", "minute")
    arrays = {name: torch.ones(2, 3, dtype=torch.long) for name in model.event_arrays}
    before = model.calendar(SimpleNamespace(event_array=arrays.__getitem__))
    arrays["minute"] = torch.full((2, 3), 30)
    after = model.calendar(SimpleNamespace(event_array=arrays.__getitem__))
    assert not torch.equal(before, after)


def test_the_default_calendar_reads_hour_weekday_and_day_of_month_as_before() -> None:
    torch.manual_seed(0)
    calendar = CalendarEmbedding(16)
    arrays = {
        "hour": torch.randint(0, 24, (3, 5)),
        "dow": torch.randint(0, 7, (3, 5)),
        "dom": torch.randint(1, 32, (3, 5)),
    }
    h = 2.0 * math.pi * arrays["hour"].float() / 24.0
    w = 2.0 * math.pi * arrays["dow"].float() / 7.0
    d = 2.0 * math.pi * (arrays["dom"].float() - 1.0) / 31.0
    feats = torch.stack([h.sin(), h.cos(), w.sin(), w.cos(), d.sin(), d.cos()], dim=-1)
    got = calendar(SimpleNamespace(event_array=arrays.__getitem__))
    assert calendar.arrays == ("hour", "dow", "dom")
    assert torch.equal(got, calendar.net(feats))


def test_a_time_feature_keeps_its_phase_far_from_zero() -> None:
    # float32 spacing at 2**50 is 2**27: a cast before the subtraction loses the phase.
    calendar = CalendarEmbedding(8, (CyclicTime("t", 2_000_000.0, 2.0**50),))
    t = torch.tensor([[2**50, 2**50 + 1_000_000]])
    got = calendar(SimpleNamespace(event_array={"t": t}.__getitem__))
    want = calendar.net(torch.tensor([[0.0, 1.0], [0.0, -1.0]]))
    assert torch.allclose(got[0], want, atol=1e-5)


def test_a_dumped_config_with_the_default_time_features_reloads_for_any_granularity() -> None:
    config = load_recipe_config(Path("recipes/blueprint_v1/config.small.yaml"))
    RecipeConfig.model_validate(config.model_dump(mode="json"))


@pytest.mark.parametrize(
    ("time_features", "granularity", "match"),
    [
        ({"minute": 60}, "flat", "does not use \\['time_features'\\]"),
        ({"minute": 0}, "profile_event_decoder", "positive period"),
        ({"minute": {"period": -1}}, "profile_history", "greater than 0"),
        ({}, "profile_event_decoder", "at least one array"),
    ],
)
def test_time_features_are_refused_where_they_cannot_apply(
    time_features: dict, granularity: str, match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        _with_time_features(time_features, granularity)


def test_a_time_feature_the_tokenizer_does_not_emit_is_refused_at_setup(tmp_path: Path) -> None:
    from tfm.runtime.pipeline import setup_run

    config = _with_time_features({"hour": 24, "minute": 60})
    with pytest.raises(IncompatibleCompositionError, match=r"\['minute'\], which this tokenizer"):
        setup_run(config=config, run_dir=tmp_path / "run", device="cpu")
