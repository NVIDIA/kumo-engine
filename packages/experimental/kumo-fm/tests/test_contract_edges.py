# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Population, plan and budget contracts, including outstanding review findings."""
from datetime import UTC, date, datetime, timedelta, timezone

import pyarrow as pa
import pytest
from _correctness import instant, make_case, prepare, sample_for
from kumo.fm._snapshot import _rows, _schema_view, _selected_graph
from kumo.pytorch import api

from kumo import fm


@pytest.fixture
def case(tmp_path):
    return make_case(tmp_path)


@pytest.mark.parametrize("population,split,exception,match", [
    (fm.Query(), "predict", ValueError, "at least one"),
    (fm.Context(), "context", ValueError, "at least one"),
    (fm.Query(entities=[1], at=instant(1), pairs=[(1, instant(1))]), "predict", ValueError, "Choose"),
    (fm.Context(entities=[1], pairs=[(1, instant(1))]), "context", ValueError, "Choose"),
    (fm.Query(pairs=[(1,)]), "predict", ValueError, "Occurrences require"),
    (fm.Query(pairs=[(1, instant(1), 2, 3)]), "predict", ValueError, "Occurrences require"),
    (fm.Query(pairs=[(None, instant(1))]), "predict", ValueError, "null"),
    (fm.Query(pairs=[(True, instant(1))]), "predict", ValueError, "int64"),
    (fm.Query(pairs=[(2**63, instant(1))]), "predict", ValueError, "int64"),
    (fm.Query(pairs=[(1.0, instant(1))]), "predict", ValueError, "int64"),
    (fm.Query(pairs=[("1", instant(1))]), "predict", ValueError, "int64"),
    (fm.Query(pairs=[(1, datetime(2025, 1, 1))]), "predict", ValueError, "UTC offset"),
    (fm.Query(pairs=[(1, None)]), "predict", ValueError, "UTC offset"),
    (fm.Query(pairs=[(1, instant(1), True)]), "predict", ValueError, "int64"),
    (fm.Query(pairs=[(1, instant(1), 2**63)]), "predict", ValueError, "int64"),
    (fm.Context(entities=[1], times=[instant(1)]), "predict", TypeError, "fm.Query"),
    (fm.Query(entities=[1], at=instant(1)), "context", TypeError, "fm.Context"),
])
def test_population_rejects_invalid_inputs(population, split, exception, match):
    with pytest.raises(exception, match=match):
        _rows(population, split, "int64")


@pytest.mark.parametrize("value", ["1970-01-01T02:00:10+02:00", instant(10),
                                   instant(10).astimezone(timezone(timedelta(hours=-8)))])
def test_offsets_normalize_to_utc(value):
    assert _rows(fm.Query(pairs=[(1, value)]), "predict", "int64")[0]["anchor_time"] == instant(10)


def test_date_and_generator_population():
    rows = _rows(fm.Context(entities=iter(["001", "用户"]), times=iter([date(2025, 1, 1)])),
                 "context", "string", max_instances=2)
    assert [r["entity_key"] for r in rows] == ["001", "用户"]
    assert [r["instance_id"] for r in rows] == [0, 1]
    assert rows[0]["anchor_time"] == datetime(2025, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "10", None])
def test_invalid_budget_values(case, budget):
    with pytest.raises(ValueError, match="positive integers"):
        prepare(case[0], budgets={"max_instances": budget})


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, "10"])
def test_invalid_batch_size(case, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        prepare(case[0], batch_size=batch_size)


@pytest.mark.parametrize("classes", [None, {}, {0: "no"}, {0: "same", 1: "same"},
                                      {0: "a", 1: "b", 2: "c"}])
def test_invalid_class_maps(case, classes):
    with pytest.raises(ValueError, match="two-entry"):
        prepare(case[0], task="classification", classes=classes)


@pytest.mark.parametrize("features", [{"users": []}, {"users": [], "events": ["missing"]},
                                       {"users": [], "events": ["text", "text"]}])
def test_invalid_projections(case, features):
    with pytest.raises(ValueError, match="features|projection"):
        prepare(case[0], features=features)


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                   reason="#391 review 4687200524: unknown budget keys pass silently")
def test_unknown_budget_key_is_rejected(case):
    with (pytest.raises((TypeError, ValueError), match="max_payload_byte|[Uu]nknown"),
          prepare(case[0], budgets={"max_payload_byte": 1})):
        pass


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="#391 review 4687200537: default strategy should resolve to recent")
def test_default_strategy_matches_explicit_recent(case):
    engine, keys, _ = case
    population = fm.Query(entities=keys, at=instant(30))
    with prepare(engine) as explicit:
        expected = explicit.load(query=population)
    with prepare(engine, sample=sample_for(strategy=None)) as implicit:
        actual = implicit.load(query=population)
    assert actual.encode() == expected.encode()


@pytest.mark.parametrize("dtype", [pa.int32(), pa.float32(), pa.binary(), pa.decimal128(10, 2)])
@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="#391 review 4687200488: unprojected columns block preparation")
def test_unprojected_unsupported_columns_do_not_block_fm(tmp_path, dtype):
    engine, keys, _ = make_case(tmp_path, ignored_dtype=dtype)
    with prepare(engine) as loader:
        batch = loader.load(query=fm.Query(entities=[keys[0]], at=instant(30)))
    assert "unused" not in batch.tables("predict")["related_tables"]["events"].column_names


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception,
                   reason="#391 review 4687200483: generated edge feature name collision")
def test_generated_edge_feature_name_is_reserved(case):
    engine, _, _ = case
    tables = tuple(api.TableSpec(t.name, t.primary_key,
                                {**t.columns, "__fm_edge_activity": api.DType("int64")},
                                time_column=t.time_column) for t in engine.schema.tables)
    schema = api.DataSchema(tables, engine.schema.relations)
    with pytest.raises(ValueError, match="reserv|colli"):
        _selected_graph(schema, sample_for(),
                        {"users": [], "events": ["__fm_edge_activity"]}, _schema_view(schema))


@pytest.mark.parametrize("batch_size", [pytest.param(1, marks=pytest.mark.xfail(
    strict=True, raises=pytest.fail.Exception,
    reason="#391 review 4687200497: candidate budget checked per batch instead of request")), 100])
def test_candidate_budget_is_independent_of_batch_size(case, batch_size):
    engine, keys, _ = case
    with (prepare(engine, batch_size=batch_size, budgets={"max_candidate_rows": 10}) as loader,
          pytest.raises(ValueError, match="budget")):
        loader.load(query=fm.Query(pairs=[(keys[0], instant(30), i) for i in range(4)]))


@pytest.mark.parametrize("ids,values,valid,classes,match", [
    ([1], [1], [], None, "misaligned"),
    ([1], [], [True], None, "misaligned"),
    ([1], [1], [False], None, "Missing, duplicated, or unexpected"),
    ([99], [1], [True], None, "unexpected"),
    ([1, 1], [1, 2], [True, True], None, "duplicated"),
    ([1], [2], [True], {0: "no", 1: "yes"}, "outside declared classes"),
    ([], [], [], None, "Missing FM context labels"),
])
def test_context_label_corruption_fails_closed(ids, values, valid, classes, match):
    from types import SimpleNamespace

    import numpy as np
    from kumo.fm._snapshot import SnapshotFMLoader

    class Task:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def batches(self, **_):
            yield SimpleNamespace(metadata={"sample_ids": ids}, targets={"label": {
                "values": np.array(values), "valid": np.array(valid)}})

    loader = object.__new__(SnapshotFMLoader)
    loader._labels = SimpleNamespace(materialize=lambda _: Task())
    loader._batch_size = 2
    loader._classes = classes
    with pytest.raises(ValueError, match=match):
        loader._labels_for(None, [{"instance_id": 1}])


@pytest.mark.parametrize("dtype", [pa.int32(), pa.float32()])
@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="#391 review 4687200488: projected narrow scalars need lossless widening")
def test_projected_narrow_scalars_widen_losslessly(tmp_path, dtype):
    engine, keys, events = make_case(tmp_path, amount_dtype=dtype)
    with prepare(engine, features={"users": [], "events": ["amount"]}) as loader:
        batch = loader.load(query=fm.Query(entities=[keys[0]], at=instant(30)))
    table = batch.tables("predict")["related_tables"]["events"]
    assert table["amount"].type == pa.float64()
    source = {e["id"]: e["amount"] for e in events}
    assert table["amount"].to_pylist() == [source[key] for key in table["id"].to_pylist()]
