# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from kumo_core.encoder_selection import _EncoderStat, select_encoder
from kumo_core.model_plan import TrainingTableGenerationPlan
from kumo_core.pquery import (
    _column_encoder_stats,
    build_native_pquery_task_plan,
    normalize_graph,
    parse_and_validate_query,
)
from kumo_core.types import ColStatType
from kumoapi.encoder import Hash, Index
from kumoapi.typing import ColStatType as ApiColStatType
from kumoapi.typing import Dtype, Stype
from pydantic import TypeAdapter

from kumo.graph.column import Column, EncoderGenerator
from kumo.graph.table import Table
from kumo.stats import Stats
from kumo.task.task import collect_encoders


def _hosted_stats(
    values: dict[ApiColStatType, _EncoderStat], ) -> dict[ColStatType, Stats]:
    return {
        ColStatType(stat_type.value):
        Stats(
            value=stat.value,
            approx=stat.approx,
            num_entries=stat.num_entries,
        )
        for stat_type, stat in values.items()
    }


def _select_both(
    *,
    dtype: Dtype,
    stype: Stype,
    stats: dict[ApiColStatType, _EncoderStat],
) -> tuple[dict, dict]:
    shared = select_encoder("feature", dtype, stype, stats)
    hosted = EncoderGenerator.get_encoder(
        Column("feature", dtype=dtype, stype=stype),
        _hosted_stats(stats),
    )
    return (
        TypeAdapter(type(shared)).dump_python(shared, mode="json"),
        TypeAdapter(type(hosted)).dump_python(hosted, mode="json"),
    )


def test_hosted_warnings_use_graph_table_names_in_override_keys(caplog):
    stats = _hosted_stats({
        ApiColStatType.NA_FRACTION: _EncoderStat(0.0, approx=True),
        ApiColStatType.NUM_UNIQUE: _EncoderStat(8_000, approx=False),
        ApiColStatType.STRING_GLOVE_OVERLAP: _EncoderStat(0.0, approx=True),
        ApiColStatType.STRING_AVG_TOKENS: _EncoderStat(1.0, approx=True),
        ApiColStatType.STRING_MAX_TOKENS: _EncoderStat(1.0, approx=True),
        ApiColStatType.STRING_AVG_NON_CHAR: _EncoderStat(0.0, approx=True),
    })
    for table_name in ("users", "orders"):
        table = Table(
            table_name=table_name,
            source_name="warehouse_source",
            data_source="snapshot",
            cols=[Column("category_01", dtype=Dtype.string, stype=Stype.text)],
        )
        table.infer_encoders(encoder_stats=SimpleNamespace(col_stats={
            "category_01": {key: stat.config() for key, stat in stats.items()},
        }))
        assert table["category_01"].encoder.name == "Null"

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    for table_name, message in zip(("users", "orders"), messages):
        assert f"Text column '{table_name}.category_01'" in message
        assert (
            f"encoder_overrides['{table_name}.category_01']" in message)
        assert "warehouse_source" not in message


@pytest.mark.parametrize("method,dtype,stype,missing_stat", [
    ("get_numerical_encoder", Dtype.float, Stype.numerical, "MEAN"),
    ("get_sequence_encoder", Dtype.floatlist, Stype.sequence,
     "SEQUENCE_MIN_LENGTH"),
])
def test_direct_hosted_selectors_qualify_column_warnings(
        method, dtype, stype, missing_stat, caplog):
    column = Column("feature", dtype=dtype, stype=stype)
    column.table_name = "users"

    assert getattr(EncoderGenerator, method)(column, {}).name == "Null"
    assert "users.feature" in caplog.text
    assert missing_stat in caplog.text


def test_real_entry_points_share_prepared_snapshot_selection(
        tmp_path, monkeypatch):
    stats = {
        ApiColStatType.NA_FRACTION:
        _EncoderStat(0.0, approx=True, num_entries=10_000),
        ApiColStatType.NUM_UNIQUE:
        _EncoderStat(8_000, approx=True, num_entries=10_000),
    }
    hosted_column = Column(
        "repeated_id",
        dtype=Dtype.int64,
        stype=Stype.ID,
    )
    hosted_column.encoder = EncoderGenerator.get_encoder(
        hosted_column, _hosted_stats(stats))
    hosted_table = Table(
        table_name="users",
        data_source="snapshot",
        data_version="snapshot-v1",
        cols=[hosted_column],
    )
    with patch("kumo.graph.utils.fetch_and_set_encoder_stats"):
        hosted = collect_encoders(
            SimpleNamespace(name="prepared-snapshot", tables=[hosted_table]))

    producer_stats = {
        "repeated_id": {
            stat_type.value: {
                "value": stat.value,
                "approx": stat.approx,
                "num_entries": stat.num_entries,
            }
            for stat_type, stat in stats.items()
        },
    }
    monkeypatch.setattr(
        "kumo_core.pquery.read_col_stats",
        lambda _index, _table, _kind: json.dumps(producer_stats),
    )
    graph = normalize_graph({
        "tables": [{
            "table_name":
            "users",
            "source_table_name":
            "main.schema.users",
            "pkey": {
                "name": "id",
                "dtype": "int64",
                "stype": "ID",
            },
            "cols": [{
                "name": "label",
                "dtype": "bool",
                "stype": "categorical",
            }, {
                "name": "repeated_id",
                "dtype": "int64",
                "stype": "ID",
            }],
            "fkeys": [],
        }],
    })
    query = parse_and_validate_query(
        "PREDICT users.label FOR EACH users.id",
        graph,
    )
    portable = build_native_pquery_task_plan(
        query=query,
        graph=graph,
        training_table_generation_plan=TrainingTableGenerationPlan(),
        run_id="prepared-snapshot",
        graph_index_path=str(tmp_path / "index"),
        scratch_dir=str(tmp_path / "scratch"),
        artifact_dir=str(tmp_path / "artifacts"),
    ).task_dataset.feature_schema["node_types"]["users"]

    hosted_configs = {
        name: TypeAdapter(type(encoder)).dump_python(encoder, mode="json")
        for name, encoder in hosted["users"].items()
    }
    portable_configs = {
        spec["column_name"]: spec["encoder"]
        for spec in portable["column_specs"]
    }
    assert portable["feature_columns"] == ["repeated_id"]
    assert list(hosted_configs) == portable["feature_columns"]
    assert hosted_configs == portable_configs
    assert EncoderGenerator._shared_stats(
        _hosted_stats(stats)) == _column_encoder_stats(producer_stats,
                                                       "repeated_id")


@pytest.mark.parametrize(
    "num_unique,expected",
    [
        (8_000, {
            "name": "Hash",
            "num_components": 3_989
        }),
        (8_001, {
            "name": "Null"
        }),
    ],
)
def test_hosted_and_shared_id_selection_use_the_row_count_boundary(
        num_unique, expected):
    stats = {
        ApiColStatType.NA_FRACTION:
        _EncoderStat(0.0, approx=True),
        ApiColStatType.NUM_UNIQUE:
        _EncoderStat(num_unique, approx=True, num_entries=10_000),
    }

    shared, hosted = _select_both(
        dtype=Dtype.int64,
        stype=Stype.ID,
        stats=stats,
    )

    for selected in (shared, hosted):
        assert selected["name"] == expected["name"]
        if "num_components" in expected:
            assert selected["num_components"] == expected["num_components"]


@pytest.mark.parametrize("num_unique,num_components", [(4_000, 189),
                                                       (4_001, 3_989)])
def test_hosted_and_shared_id_selection_use_the_category_limit(
        num_unique, num_components):
    stats = {
        ApiColStatType.NA_FRACTION:
        _EncoderStat(0.0, approx=True),
        ApiColStatType.COUNT:
        _EncoderStat(400_000, approx=False),
        ApiColStatType.NUM_NA:
        _EncoderStat(0, approx=False),
        ApiColStatType.NUM_UNIQUE:
        _EncoderStat(num_unique, approx=False, num_entries=10_000),
        ApiColStatType.CATEGORY_COUNTS:
        _EncoderStat([[str(index), 100] for index in range(1_000)],
                     approx=False),
    }

    shared, hosted = _select_both(
        dtype=Dtype.string,
        stype=Stype.ID,
        stats=stats,
    )

    assert shared == hosted
    assert shared["name"] == "Hash"
    assert shared["num_components"] == num_components


@pytest.mark.parametrize("num_unique,expected", [(8_000, True), (8_001, False)])
def test_hosted_id_validation_uses_the_shared_uniqueness_boundary(
        num_unique, expected):
    column = Column(
        "feature",
        dtype=Dtype.int64,
        stype=Stype.ID,
        encoder=Hash(num_components=3_989),
    )
    stats = {
        column.base_config(): {
            ColStatType.NUM_UNIQUE:
            Stats(
                value=num_unique,
                approx=True,
                num_entries=10_000,
            ),
        },
    }

    if expected:
        assert column.validate_encoders("version", True, stats) is True
    else:
        with pytest.warns(UserWarning, match="mostly unique embeddings"):
            assert column.validate_encoders("version", True, stats) is False


def test_hosted_id_validation_skips_uniqueness_without_population(caplog):
    column = Column(
        "feature",
        dtype=Dtype.int64,
        stype=Stype.ID,
        encoder=Hash(num_components=3_989),
    )
    stats = {
        column.base_config(): {
            ColStatType.NUM_UNIQUE: Stats(value=8_001, approx=True),
        },
    }

    assert column.validate_encoders("version", True, stats) is True
    assert "has no num_entries" in caplog.text


def test_hosted_and_shared_numerical_selection_handle_nonfinite_stats():
    stats = {
        ApiColStatType.NA_FRACTION: _EncoderStat(0.0, approx=True),
        ApiColStatType.MEAN: _EncoderStat(float("nan"), approx=True),
        ApiColStatType.MEDIAN: _EncoderStat(1.0, approx=True),
        ApiColStatType.STD: _EncoderStat(float("inf"), approx=True),
        ApiColStatType.MIN: _EncoderStat(0.0, approx=True),
        ApiColStatType.MAX: _EncoderStat(2.0, approx=True),
        ApiColStatType.KURTOSIS: _EncoderStat(0.0, approx=True),
        ApiColStatType.QUANTILES: _EncoderStat([0.0, 1.0, 2.0], approx=True),
    }

    shared, hosted = _select_both(
        dtype=Dtype.float,
        stype=Stype.numerical,
        stats=stats,
    )

    assert shared["name"] == "Numerical"
    assert hosted["name"] == "Numerical"


def test_hosted_and_shared_selection_warn_for_approximate_category_stats(caplog):
    stats = {
        ApiColStatType.NA_FRACTION:
        _EncoderStat(0.0, approx=True),
        ApiColStatType.COUNT:
        _EncoderStat(100, approx=True),
        ApiColStatType.NUM_NA:
        _EncoderStat(0, approx=False),
        ApiColStatType.NUM_UNIQUE:
        _EncoderStat(2, approx=False),
        ApiColStatType.CATEGORY_COUNTS:
        _EncoderStat([["a", 60], ["b", 40]], approx=False),
    }

    shared, hosted = _select_both(dtype=Dtype.string, stype=Stype.categorical,
                                  stats=stats)

    assert shared["name"] == hosted["name"] == "Null"
    assert len(caplog.records) == 2
    assert "COUNT must be exact" in caplog.text


def test_hosted_adapter_ignores_stats_outside_the_selector_contract():
    stats = _hosted_stats({
        ApiColStatType.NA_FRACTION:
        _EncoderStat(0.0, approx=True),
    })
    stats[ColStatType.DROPPED_ROWS] = Stats(value=3, approx=False)

    encoder = EncoderGenerator.get_encoder(
        Column("feature", dtype=Dtype.bool, stype=Stype.categorical),
        stats,
    )

    assert isinstance(encoder, Index)
