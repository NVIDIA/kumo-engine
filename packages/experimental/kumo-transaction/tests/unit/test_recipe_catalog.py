# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from tfm.contracts.errors import SchemaError
from tfm.data.sampling import stratified_sample_ids


def test_recipe_catalog_matches_directories() -> None:
    catalog = yaml.safe_load(Path("recipes/catalog.yaml").read_text(encoding="utf-8"))
    ids = {row["id"] for row in catalog["recipes"]}
    dirs = {p.name for p in Path("recipes").iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert ids == dirs
    series = {row["notebooks"] for row in catalog["recipes"] if row.get("notebooks")}
    notebook_dirs = {
        p.name for p in Path("notebooks").iterdir() if p.is_dir() and not p.name.startswith(".")
    }
    assert series == notebook_dirs - {"outputs"}
    for row in catalog["recipes"]:
        recipe = Path("recipes") / row["id"]
        assert (recipe / "README.md").is_file()
        assert (recipe / "config.smoke.yaml").is_file(), row["id"]
        for variant in row["configs"]:
            name = "config.yaml" if variant == "default" else f"config.{variant}.yaml"
            assert (recipe / name).is_file(), f"{row['id']}: {name}"
        listed = {"config.yaml" if v == "default" else f"config.{v}.yaml" for v in row["configs"]}
        assert {p.name for p in recipe.glob("config*.yaml")} == listed, row["id"]


def test_cli_resolves_recipe_variants() -> None:
    from tfm.cli.recipe import resolve_config

    assert (
        resolve_config("blueprint_v1", None, command="smoke")
        == Path("recipes/blueprint_v1/config.smoke.yaml").resolve()
    )
    assert resolve_config("blueprint_v2", None, command="report").name == "config.yaml"
    assert resolve_config("blueprint_v2", "scale", command="train").name == "config.scale.yaml"
    assert resolve_config(None, "my.yaml", command="train") == Path("my.yaml")
    try:
        resolve_config("decoder_fraud_baseline", None, command="train")
    except SystemExit as exc:
        assert "blueprint_v1" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_stratified_sample_keeps_positives() -> None:
    chosen = stratified_sample_ids(
        list(range(20)),
        [0, 1, 2],
        cap=6,
        seed=0,
    )
    assert len(chosen) == 6
    assert {0, 1, 2} & set(chosen)


def test_stratified_sample_rejects_nonpositive_cap() -> None:
    try:
        stratified_sample_ids([1, 2], [1], cap=0, seed=0)
    except SchemaError:
        return
    raise AssertionError("expected SchemaError")


@pytest.mark.parametrize(
    ("recipe", "variant"), [("blueprint_v1", "multi_gpu"), ("blueprint_v2", "scale")]
)
def test_distributed_blueprint_presets_select_python(recipe: str, variant: str) -> None:
    from tfm.cli.recipe import resolve_config
    from tfm.config import load_recipe_config

    config = load_recipe_config(resolve_config(recipe, variant, command="train"))
    assert config.data.loader == "python"
