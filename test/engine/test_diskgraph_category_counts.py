# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the DiskGraph categorical-statistics contract on real Parquet."""

import json
from collections import Counter

import pandas as pd
import torch
from diskgraph import compute_col_stats_for_parquet
from kumo_core import pquery
from kumo_core.encoders.fused.categorical import FusedCategoricalEncoder
from kumo_core.encoders.tensor_mapper import TensorMapper
from kumo_core.pquery import NativeGraphColumn, _infer_encoder
from kumo_core.types import ColStatType
from kumo_core.types.schema import VersionedSourceTableID
from kumoapi.encoder import Hash, Index
from kumoapi.typing import Dtype, Stype

from kumo.data.backend_cache import ExactStats, PredStats
from kumo.graph.column import Column, EncoderGenerator
from kumo.stats import Stats


def _raw_exact_stat(stats, column_name, stat_name):
    matches = [(key, value) for key, value in stats.items()
               if key.startswith(f"{column_name}__")
               and key.rsplit("__", 2)[-2] == stat_name]
    assert len(matches) == 1
    key, entry = matches[0]
    assert key.endswith("__False")
    assert entry["approx"] is False
    return entry


def _compute_stats(tmp_path, source_values, stypes, category_limit):
    parquet = tmp_path / f"categories-{len(source_values)}.parquet"
    pd.DataFrame(source_values).to_parquet(parquet, index=False)
    specs = [{
        "name": name,
        "stype": stypes[name],
        "dtype": "int64" if stypes[name] == "ID" else "float",
    } for name in source_values]
    full_json, min_json = compute_col_stats_for_parquet(
        parquet_paths=[str(parquet)],
        kumo_column_specs_json=json.dumps(specs),
        data_version="v1",
        _category_count_limit_for_testing=category_limit,
    )
    raw_stats = json.loads(full_json)
    for stats in (raw_stats, json.loads(min_json)):
        assert not any("CATEGORY_COUNTS_CAPPED" in key for key in stats)
        assert not any("MULTI_CATEGORY_COUNTS_CAPPED" in key for key in stats)
    return full_json, raw_stats


def _parse_stats(full_json, num_rows):
    table_json = json.dumps({
        "NUM_ROWS__False": {
            "value": num_rows,
            "approx": False,
            "data_version": "v1",
            "num_entries": num_rows,
        },
    }).encode()
    return (
        ExactStats.parse("generated", table_json, full_json.encode()),
        PredStats.parse(
            VersionedSourceTableID("connector", "generated", "v1"),
            full_json.encode(),
        ),
    )


def _runtime_stats(stats):
    return {
        stat_type: Stats.from_config(stat)
        for stat_type, stat in stats.items()
    }


def test_capped_categorical_stats_preserve_legacy_consumer_behavior(tmp_path):
    category_limit = 4
    num_rows = 100
    exactly_limit = ([0.0, -0.0] * 10 + [1.5] * 30 + [2.5] * 25 + [3.5] * 25)
    common_over_limit = ([0.0, -0.0] * 12 + [0.0] + [1.5] * 25 + [2.5] * 20 +
                         [3.5] * 20 + [4.5] * 10)
    rare_over_limit = [0.0] * 96 + [1.5, 2.5, 3.5, 4.5]
    source_values = {
        "EXACTLY_LIMIT": exactly_limit,
        "COMMON_OVER_LIMIT": common_over_limit,
        "RARE_OVER_LIMIT": rare_over_limit,
    }
    assert all(len(values) == num_rows for values in source_values.values())

    full_json, raw_stats = _compute_stats(
        tmp_path,
        source_values,
        {name: "categorical"
         for name in source_values},
        category_limit,
    )

    retained = {}
    for column_name, values in source_values.items():
        retained[column_name] = _raw_exact_stat(raw_stats, column_name,
                                                "CATEGORY_COUNTS")["value"]
        expected = Counter(values)
        assert len(retained[column_name]) == category_limit
        assert all(expected[value] == count
                   for value, count in retained[column_name])
        _raw_exact_stat(raw_stats, column_name, "NUM_UNIQUE")
        _raw_exact_stat(raw_stats, column_name, "UNIQUE_FRACTION")

    assert dict(retained["EXACTLY_LIMIT"]) == Counter(exactly_limit)
    assert dict(retained["EXACTLY_LIMIT"])[0.0] == 20
    for column_name in ("COMMON_OVER_LIMIT", "RARE_OVER_LIMIT"):
        source_categories = set(source_values[column_name])
        retained_categories = {value for value, _ in retained[column_name]}
        assert len(source_categories) == category_limit + 1
        assert len(source_categories - retained_categories) == 1

    parsed_stats = _parse_stats(full_json, num_rows)
    for parsed in parsed_stats:
        for column_name in source_values:
            stats = parsed.col_stats[column_name]
            assert stats[
                ColStatType.CATEGORY_COUNTS].value == retained[column_name]
            assert stats[ColStatType.NUM_UNIQUE].approx is False
            assert stats[ColStatType.UNIQUE_FRACTION].approx is False

    native_encoders = {}
    service_encoders = {}
    for column_name in ("COMMON_OVER_LIMIT", "RARE_OVER_LIMIT"):
        native_encoders[column_name] = _infer_encoder(
            NativeGraphColumn(
                name=column_name,
                dtype="float",
                stype="categorical",
            ),
            pquery._column_encoder_stats(raw_stats, column_name),
        )
        service_encoders[column_name] = EncoderGenerator.get_encoder(
            Column(
                name=column_name,
                dtype=Dtype.float,
                stype=Stype.categorical,
            ),
            _runtime_stats(parsed_stats[0].col_stats[column_name]),
        )
    assert native_encoders["COMMON_OVER_LIMIT"]["name"] == "Hash"
    assert isinstance(service_encoders["COMMON_OVER_LIMIT"], Hash)
    assert native_encoders["RARE_OVER_LIMIT"]["name"] == "Index"
    assert isinstance(service_encoders["RARE_OVER_LIMIT"], Index)

    common_stats = pquery._column_stats(raw_stats, "COMMON_OVER_LIMIT")
    omitted_value = (set(common_over_limit) -
                     {value
                      for value, _ in retained["COMMON_OVER_LIMIT"]}).pop()
    second_retained_value = retained["COMMON_OVER_LIMIT"][1][0]
    last_retained_value = retained["COMMON_OVER_LIMIT"][-1][0]
    encoder = Hash(
        num_components=native_encoders["COMMON_OVER_LIMIT"]["num_components"])
    encoder._stats = {
        ColStatType.CATEGORY_COUNTS: common_stats["CATEGORY_COUNTS"],
    }
    mapper = TensorMapper.from_encoder(encoder)
    mapped = mapper(
        pd.Series(
            [
                0.0,
                -0.0,
                second_retained_value,
                last_retained_value,
                omitted_value,
            ],
            dtype="float64",
        )).view(-1)
    assert mapped[0].item() == mapped[1].item()
    assert mapped.tolist() == [0, 0, 1, category_limit - 1, -1]

    fused = FusedCategoricalEncoder(
        {"COMMON_OVER_LIMIT": encoder},
        [],
        out_channels=4,
        enc_mode="column",
    )
    fixed = fused.forward_fixed(mapped.view(-1, 1), ["COMMON_OVER_LIMIT"])
    assert fixed[:, 0].tolist() == [0, 0, 1, 0, 3]
    embedded = fused(mapped.view(-1, 1), ["COMMON_OVER_LIMIT"])
    assert embedded.shape == (5, 1, 4)
    assert torch.isfinite(embedded).all()


def test_capped_id_stats_preserve_legacy_service_inference(tmp_path):
    category_limit = 4
    num_rows = 10_000
    values = (list(range(4_501)) + list(range(4_499)) + list(range(1_000)))
    assert len(values) == num_rows

    full_json, raw_stats = _compute_stats(
        tmp_path,
        {"ID_OVER_LIMIT": values},
        {"ID_OVER_LIMIT": "ID"},
        category_limit,
    )
    retained = _raw_exact_stat(raw_stats, "ID_OVER_LIMIT",
                               "CATEGORY_COUNTS")["value"]
    expected = Counter(values)
    assert len(retained) == category_limit
    assert all(expected[value] == count for value, count in retained)

    unique_entry = _raw_exact_stat(raw_stats, "ID_OVER_LIMIT", "NUM_UNIQUE")
    _raw_exact_stat(raw_stats, "ID_OVER_LIMIT", "UNIQUE_FRACTION")
    estimated_unique = unique_entry["value"]
    exact_unique = len(expected)
    assert abs(estimated_unique - exact_unique) < 0.05 * exact_unique
    assert 4_000 < estimated_unique <= 0.8 * num_rows

    parsed_stats = _parse_stats(full_json, num_rows)
    for parsed in parsed_stats:
        stats = parsed.col_stats["ID_OVER_LIMIT"]
        assert stats[ColStatType.CATEGORY_COUNTS].value == retained
        assert stats[ColStatType.NUM_UNIQUE].value == estimated_unique
        assert stats[ColStatType.NUM_UNIQUE].approx is False
        assert stats[ColStatType.UNIQUE_FRACTION].approx is False

    encoder = EncoderGenerator.get_encoder(
        Column(name="ID_OVER_LIMIT", dtype=Dtype.int64, stype=Stype.ID),
        _runtime_stats(parsed_stats[0].col_stats["ID_OVER_LIMIT"]),
    )
    assert isinstance(encoder, Hash)
    assert encoder.num_components == 3_989
