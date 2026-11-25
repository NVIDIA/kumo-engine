# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File declarations share typed keys/allowlists without loading event objects."""

from pathlib import Path

import pytest

from tfm.contracts.relational import RelationalForeignKey, RelationalTable
from tfm.contracts.table_source import TableSource


def test_table_source_snapshots_only_metadata(tmp_path):
    tables = [RelationalTable("users", ("id",), immutable=True)]
    paths = {"users": tmp_path / "not-read-yet.parquet"}
    source = TableSource(tables, (), paths)
    tables.clear()
    paths.clear()
    assert source.tables[0].name == "users"
    assert source.files["users"].is_absolute()
    assert not source.files["users"].exists()
    with pytest.raises(TypeError):
        source.files["extra"] = Path("wrong")


@pytest.mark.parametrize("files", [{}, {"other": Path("elsewhere.parquet")}])
def test_table_source_requires_complete_binding(files):
    with pytest.raises(ValueError, match="exactly match"):
        TableSource((RelationalTable("users", ("id",), immutable=True),), (), files)


def test_table_source_checks_relation_key_arity(tmp_path):
    with pytest.raises(ValueError, match="parent key"):
        TableSource(
            (
                RelationalTable("users", ("tenant", "id"), immutable=True),
                RelationalTable("events", ("id",), immutable=True),
            ),
            (RelationalForeignKey("activity", "users", "events", ("owner",)),),
            {"users": tmp_path / "users.parquet", "events": tmp_path / "events.parquet"},
        )


def test_adapter_table_source_does_not_call_object_loader(monkeypatch, tmp_path):
    from tfm.config import load_recipe_config
    from tfm.data.adapters import registry

    def forbidden(*args, **kwargs):
        raise AssertionError("Native source preparation must not load Python event objects")

    declaration = TableSource(
        (RelationalTable("users", ("id",), immutable=True),),
        (),
        {"users": tmp_path / "users.parquet"},
    )
    adapter = registry.Adapter(forbidden, table_source=lambda *args, **kwargs: declaration)
    monkeypatch.setattr(registry, "_adapter", lambda config: adapter)
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    assert registry.table_source(config.data, seed=42, fixture_dir=tmp_path) is declaration


def test_native_source_missing_capability_does_not_fall_back(monkeypatch, tmp_path):
    from tfm.config import load_recipe_config
    from tfm.contracts.errors import UnsupportedCombinationError
    from tfm.data.adapters import registry

    def forbidden(*args, **kwargs):
        raise AssertionError("No object fallback")

    monkeypatch.setattr(registry, "_adapter", lambda config: registry.Adapter(forbidden))
    config = load_recipe_config("recipes/blueprint_v1/config.smoke.yaml")
    with pytest.raises(UnsupportedCombinationError, match="no table_source"):
        registry.table_source(config.data, seed=42, fixture_dir=tmp_path)
