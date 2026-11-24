# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from tfm.config import ModelConfig, RecipeConfig
from tfm.contracts.errors import UnsupportedCombinationError
from tfm.features.tokenizers.registry import tokenizer_for


def test_shared_config_accepts_typed_nested_values():
    raw = yaml.safe_load(Path("recipes/blueprint_v2/config.smoke.yaml").read_text())
    expected = RecipeConfig.model_validate(raw)
    raw["model"] = expected.model
    raw["data"] = expected.data
    assert RecipeConfig.model_validate(raw) == expected
    raw["model"] = {"composition": expected.model.composition}
    assert RecipeConfig.model_validate(raw) == expected
    assert isinstance(expected.model, ModelConfig)


@pytest.mark.parametrize(
    "source,n_entities,max_entities,accepted",
    [
        ("synthetic", 8, 48, False),
        ("pragmatiq", 8, 48, False),
        ("local", 8, 48, True),
        ("local", 48, 8, False),
    ],
)
def test_graph_owner_minimum_uses_the_selected_source(source, n_entities, max_entities, accepted):
    raw = yaml.safe_load(Path("recipes/gnn_encoder/config.smoke.yaml").read_text())
    raw["data"].update(
        adapter="pragmatiq" if source == "local" else source,
        n_entities=n_entities,
        max_entities=max_entities,
    )
    if source == "local":
        raw["data"]["path"] = "/a/local/book"
    if accepted:
        RecipeConfig.model_validate(raw)
    else:
        with pytest.raises(ValueError, match="12 selected owners"):
            RecipeConfig.model_validate(raw)


@pytest.mark.parametrize("options", [False, True])
def test_hash_bucket_settings_are_rejected_by_tokenizers_that_ignore_them(options):
    raw = yaml.safe_load(Path("recipes/blueprint_v2/config.smoke.yaml").read_text())
    raw["features"] = {"tokenizer": "event_fields", "hash_buckets": 2}
    if options:
        raw["features"]["tokenizer_options"] = {}
    with pytest.raises(UnsupportedCombinationError, match="hash_buckets"):
        tokenizer_for(RecipeConfig.model_validate(raw))


def test_kvt_hash_bucket_setting_is_consumed():
    raw = yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())
    raw["features"] = {
        "tokenizer": "key_value_time",
        "hashed_keys": ["merchant_name"],
        "hash_buckets": 2,
    }
    tokenizer = tokenizer_for(RecipeConfig.model_validate(raw))
    assert tokenizer.config.hash_buckets == 2


def test_local_graph_data_samples_max_entities_instead_of_generated_count(tmp_path):
    from tfm.data.adapters.pragmatiq import generate_pragmatiq_fixture
    from tfm.data.graph_recipe import prepare_graph_data

    book = generate_pragmatiq_fixture(tmp_path / "book", n_users=16, events_per_user=6)
    raw = yaml.safe_load(Path("recipes/gnn_encoder/config.smoke.yaml").read_text())
    raw["data"].update(adapter="pragmatiq", path=str(book), n_entities=8, max_entities=12)
    data = prepare_graph_data(RecipeConfig.model_validate(raw))
    assert sum(len(owners) for owners in data.split_manifest["probe_owners"].values()) == 12


def test_sequence_config_preserves_plugin_objects_in_any_typed_options():
    from pydantic import BaseModel

    class PluginOptions(BaseModel):
        width: int

    raw = yaml.safe_load(Path("recipes/blueprint_v2/config.smoke.yaml").read_text())
    options = PluginOptions(width=3)
    raw["features"] = {"tokenizer": "key_value_time", "tokenizer_options": {"spec": options}}
    config = RecipeConfig.model_validate(raw)
    assert config.features.tokenizer_options["spec"] is options


@pytest.mark.parametrize("position", [None, "nope", "time_rope"])
@pytest.mark.parametrize("typed_section", ["model", "composition"])
def test_graph_position_defaults_preserve_explicit_typed_fields(position, typed_section):
    from tfm.config import CompositionConfig

    raw = yaml.safe_load(Path("recipes/gnn_sequence/config.smoke.yaml").read_text())
    raw["model"]["composition"].pop("position")
    if position is not None:
        raw["model"]["composition"]["position"] = position
    expected = RecipeConfig.model_validate(raw)
    if typed_section == "model":
        raw["model"] = ModelConfig.model_validate(raw["model"])
    else:
        raw["model"]["composition"] = CompositionConfig.model_validate(raw["model"]["composition"])
    actual = RecipeConfig.model_validate(raw)
    assert actual == expected
    assert actual.model.composition.position == ("time_rope" if position is None else position)


def test_graph_omitted_typed_defaults_match_minimal_yaml():
    expected = RecipeConfig.model_validate({"recipe_family": "graph", "data": {}})
    assert RecipeConfig(recipe_family="graph", data=expected.data, model=ModelConfig()) == expected
