# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source normalization is row-bounded, keyed, and independent of anchors."""

from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.relational import RelationalForeignKey, RelationalTable
from tfm.contracts.table_source import TableSource
from tfm.data.native_source import AVAILABLE_US, EVENT_US, ROW_ID, NativeSourceFiles

START = datetime(2024, 1, 1, tzinfo=UTC)


def declaration(tmp_path, *, ids=None, owners=None, times=None, known=None):
    pq.write_table(
        pa.table({"tenant": ["one", "two"], "id": ["same", "same"]}), tmp_path / "users.parquet"
    )
    pq.write_table(
        pa.table(
            {
                "id": ["x", "y"] if ids is None else ids,
                "tenant": ["one", "two"],
                "owner": ["same", "same"] if owners is None else owners,
                "at": pa.array(
                    times or [START, START + timedelta(seconds=1)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "known": pa.array(
                    known or [START, START + timedelta(seconds=1)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "value": [1.0, 2.0],
            }
        ),
        tmp_path / "events.parquet",
    )
    return TableSource(
        (
            RelationalTable("users", ("tenant", "id"), immutable=True),
            RelationalTable(
                "events",
                ("id",),
                numeric={"amount": "value"},
                event_time="at",
                available_at="known",
            ),
        ),
        (RelationalForeignKey("activity", "users", "events", ("tenant", "owner")),),
        {name: tmp_path / f"{name}.parquet" for name in ("users", "events")},
    )


def test_bounded_projection_preserves_composite_key_and_source_row_cardinality(tmp_path):
    source = declaration(tmp_path)
    with NativeSourceFiles(source, tmp_path / "normalized", block_size=1) as normalized:
        assert normalized.counts == {"users": 2, "events": 2}
        assert normalized.code("users", ("two", "same")) == 1
        table = pq.read_table(normalized.source.files["events"])
        assert table.num_rows == 2
        assert table[ROW_ID].to_pylist() == [0, 1]
        assert table["__tfm_fk_0"].to_pylist() == [0, 1]
        assert table[AVAILABLE_US].to_pylist() == table[EVENT_US].to_pylist()
        assert normalized.source.tables[0].key == (ROW_ID,)
        assert normalized.source.relationships[0].columns == ("__tfm_fk_0",)
        assert len(normalized.source_identity["events"]) == 64
    assert source.tables[0].key == ("tenant", "id")
    assert pq.read_table(source.files["events"]).column_names == [
        "id",
        "tenant",
        "owner",
        "at",
        "known",
        "value",
    ]


@pytest.mark.parametrize("ids", [["same", "same"], ["x", None]])
def test_primary_key_corruption_is_rejected(tmp_path, ids):
    with pytest.raises(SchemaError, match="primary key|key contains a null"):
        NativeSourceFiles(declaration(tmp_path, ids=ids), tmp_path / "normalized", block_size=1)


def test_missing_fk_is_rejected_before_indexing(tmp_path):
    with pytest.raises(SchemaError, match="missing source row"):
        NativeSourceFiles(
            declaration(tmp_path, owners=["missing", "same"]), tmp_path / "normalized"
        )


def test_subsecond_clock_is_never_rounded(tmp_path):
    source = declaration(tmp_path, times=[START, START + timedelta(microseconds=1)])
    with pytest.raises(UnsupportedCombinationError, match="whole-second"):
        NativeSourceFiles(source, tmp_path / "normalized")


def test_event_and_availability_clocks_remain_separate(tmp_path):
    source = declaration(tmp_path, known=[START + timedelta(days=2)] * 2)
    with NativeSourceFiles(source, tmp_path / "normalized") as normalized:
        table = pq.read_table(normalized.source.files["events"])
        assert table[AVAILABLE_US][0].as_py() > table[EVENT_US][0].as_py()


def test_normalization_never_reuses_existing_registry_implicitly(tmp_path):
    source = declaration(tmp_path)
    with NativeSourceFiles(source, tmp_path / "normalized"):
        pass
    with pytest.raises(ValueError, match="already has a key registry"):
        NativeSourceFiles(source, tmp_path / "normalized")


def test_observation_cannot_precede_event(tmp_path):
    source = declaration(tmp_path, known=[START - timedelta(seconds=1), START])
    with pytest.raises(SchemaError, match="observation predates"):
        NativeSourceFiles(source, tmp_path / "normalized")


@pytest.mark.parametrize("kind", ["numeric", "categorical", "text", "graph"])
@pytest.mark.parametrize("column", ["id", "tenant", "owner", "at", "known"])
def test_original_key_fk_and_clock_columns_cannot_hide_behind_normalization(tmp_path, kind, column):
    from dataclasses import replace

    source = declaration(tmp_path)
    if kind == "graph":
        source = replace(source, graph_numeric={"events": {"observed": column}})
    else:
        source = replace(
            source,
            tables=tuple(
                replace(table, **{kind: {"observed": column}}) if table.name == "events" else table
                for table in source.tables
            ),
        )
    with pytest.raises(SchemaError, match="keys, clocks.*cannot be model inputs"):
        NativeSourceFiles(source, tmp_path / "normalized")
    assert not (tmp_path / "normalized/keys.sqlite").exists()


def test_date_composite_keys_roundtrip_without_aliasing_strings(tmp_path):
    from datetime import date

    from tfm.data.native_source import _key, source_key_values

    day = date(2024, 1, 1)
    assert _key((1, day), "test") != _key((1, day.isoformat()), "test")
    assert source_key_values(_key((1, day, START), "test")) == (1, day, START)
    pq.write_table(pa.table({"id": [1], "day": [day]}), tmp_path / "owners.parquet")
    pq.write_table(
        pa.table({"id": [1], "owner": [1], "day": [day], "amount": [2.0]}),
        tmp_path / "activity.parquet",
    )
    source = TableSource(
        (
            RelationalTable("owners", ("id", "day"), immutable=True),
            RelationalTable(
                "activity",
                ("id",),
                numeric={"amount": "amount"},
                event_time="day",
                available_at="day",
            ),
        ),
        (RelationalForeignKey("activity", "owners", "activity", ("owner", "day")),),
        {name: tmp_path / f"{name}.parquet" for name in ("owners", "activity")},
    )
    with NativeSourceFiles(source, tmp_path / "normalized", block_size=1) as normalized:
        assert normalized.code("owners", (1, day)) == 0
        actual = pq.read_table(normalized.source.files["activity"])
        assert actual["__tfm_fk_0"].to_pylist() == [0]
        assert actual[EVENT_US].to_pylist() == [1704067200000000]
        assert actual.schema.field("day").type == pa.timestamp("us", tz="UTC")


def test_normalization_rejects_payload_replacement_between_key_and_feature_passes(
    tmp_path, monkeypatch
):
    source = declaration(tmp_path)
    original = NativeSourceFiles._write_sources

    def swap(self):
        path = source.files["events"]
        rows = pq.read_table(path)
        pq.write_table(rows.take(pa.array([1, 0])), path)
        return original(self)

    monkeypatch.setattr(NativeSourceFiles, "_write_sources", swap)
    with pytest.raises(SchemaError, match="source snapshot changed"):
        NativeSourceFiles(source, tmp_path / "normalized")


@pytest.mark.parametrize("has_label_source", [False, True])
def test_task_replay_cannot_inject_labels_without_a_validated_source_match(
    tmp_path, has_label_source
):
    from tfm.config import load_recipe_config
    from tfm.data.adapters.synthetic_relational import source_files
    from tfm.data.native_anchors import task_anchors

    config = load_recipe_config("recipes/relational_inputs/config.yaml")
    task = config.data.relational
    assert task is not None
    source = source_files(config.data, seed=config.seed, fixture_dir=tmp_path / "source")
    requested = tmp_path / "requested.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["entity0"],
                "sample_id": ["unmatched"],
                "split": ["predict"],
                "cutoff": [START],
                "target": [1.0],
                "label_available_us": [1],
                "label_cutoff_us": [2],
            }
        ),
        requested,
    )
    task = task.model_copy(
        update={"anchors": str(requested), "labels": task.labels if has_label_source else None}
    )
    with NativeSourceFiles(source, tmp_path / "normalized") as normalized:
        output = task_anchors(normalized, task, tmp_path / "anchors.parquet")
        row = pq.read_table(output).to_pylist()[0]
        assert row["target"] is row["label_available_us"] is row["label_cutoff_us"] is None


def test_task_labels_use_the_same_owned_source_snapshot_as_features(tmp_path):
    from tfm.config import load_recipe_config
    from tfm.data.adapters.synthetic_relational import source_files
    from tfm.data.native_anchors import task_anchors

    config = load_recipe_config("recipes/relational_inputs/config.yaml")
    task = config.data.relational
    assert task is not None and task.labels is not None
    source = source_files(config.data, seed=config.seed, fixture_dir=tmp_path / "source")
    with NativeSourceFiles(source, tmp_path / "normalized") as normalized:
        labels = pq.read_table(source.files[task.labels.table])
        # The source snapshot is already bound; changes to the caller's file
        # must not change targets or the data identity of this preparation.
        target = labels[task.labels.target].to_pylist()
        changed = labels.set_column(
            labels.schema.get_field_index(task.labels.target),
            task.labels.target,
            pa.array([1 - value for value in target]),
        )
        pq.write_table(changed, source.files[task.labels.table])
        output = task_anchors(normalized, task, tmp_path / "anchors.parquet")
        assert pq.read_table(output)["target"].to_pylist() == target
