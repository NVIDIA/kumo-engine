# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warehouse integer ids in a Parquet anchor table read as the entity's int64 key.

Snowflake unloads every NUMBER(p,0) column, including INT and BIGINT, as a
scale-zero Parquet decimal, and other writers may narrow integer ids.
"""

from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.engines.native_anchors import NativeAnchors
from kumo.pytorch.engines.parquet_anchors import _entity_id_type, _entity_ids
from tests.test_preview import _equal

pytest.importorskip("diskgraph", reason="requires the optional native DiskGraph engine")

BIG = 2**40
INT64 = api.DType("int64")
DECIMALS = pytest.mark.parametrize(
    "decimal", [pa.decimal128(38, 0), pa.decimal256(76, 0)], ids=["decimal128", "decimal256"]
)
# entity, sample ID, split, label
ANCHORS = [
    (1, "a", "train", 1.0),
    (2, "b", "train", 2.0),
    (BIG, "c", "valid", 3.0),
    (3, "d", "train", 4.0),
    (BIG, "e", "train", 5.0),
    (1, "f", "valid", 6.0),
]
SCHEMA = api.DataSchema(
    tables=(
        api.TableSpec(
            "users",
            "id",
            {"id": INT64, "score": api.DType("float64")},
            time_column=api.NON_TEMPORAL,
        ),
        api.TableSpec(
            "events",
            "id",
            {
                "id": INT64,
                "user": INT64,
                "time": api.DType("timestamp", unit="s", timezone="UTC"),
                "amount": api.DType("float64"),
            },
            time_column="time",
        ),
    ),
    relations=(api.RelationSpec("activity", "users", "id", "events", "user"),),
)


def _contract(kind):
    """``graph`` hands the anchors straight to `NativeAnchors`; ``scalar`` first
    materializes per-anchor values through ``materialize_scalar_batches``."""
    label = {"label": api.ExternalTarget(dtype=api.DType("float64"))}
    entity = api.EntitySpec("users", entity_type="user")
    if kind == "graph":
        sample = api.SampleSpec(
            entity=entity,
            neighborhoods={
                "graph": api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("activity", 2, strategy="recent"),)),)
                )
            },
            targets=label,
        )
        inputs = {
            "graph": api.Input(
                api.NeighborhoodSource("graph"),
                api.Graph(node_fields={"events": api.Fields((api.Encode("amount", api.Raw()),))}),
                None,
            )
        }
    else:
        past = api.Rows(("activity",), time=api.TimeWindow("time", None, 0))
        sample = api.SampleSpec(
            entity=entity,
            entity_values={
                "score": api.Field("score"),
                "seen": api.Aggregate(past, "count", dtype=INT64),
            },
            targets=label,
        )
        inputs = {
            "entity": api.Input(
                api.EntitySource(),
                api.Fields((api.Encode("score", api.Raw()), api.Encode("seen", api.Raw()))),
                api.Separate(),
            )
        }
    return api.DataLoaderSpec(
        SCHEMA,
        sample,
        api.BatchSpec(
            inputs=inputs,
            targets={"label": api.LabelTarget("label", api.Raw())},
            batching=api.Batching(max_samples=2),
        ),
    )


@pytest.fixture
def dataset(tmp_path):
    users = tmp_path / "users.parquet"
    events = tmp_path / "events.parquet"
    ids = [1, 2, 3, BIG]
    pq.write_table(pa.table({"id": ids, "score": [float(i % 97) for i in ids]}), users)
    pq.write_table(
        pa.table(
            {
                "id": list(range(8)),
                "user": [ids[i % 4] for i in range(8)],
                "time": pa.array([10, 20] * 4, type=pa.timestamp("s", tz="UTC")),
                "amount": [float(i) for i in range(8)],
            }
        ),
        events,
    )
    return api.Dataset(
        tables={"users": api.Parquet(str(users)), "events": api.Parquet(str(events))}
    )


def _anchors(directory, parts, rows=ANCHORS):
    """Write ``rows`` in order over one file per (row count, entity id type) part."""
    directory.mkdir()
    start = 0
    for index, (count, dtype) in enumerate(parts):
        chunk = rows[start : start + count]
        start += count
        ids = [Decimal(row[0]) if pa.types.is_decimal(dtype) else row[0] for row in chunk]
        pq.write_table(
            pa.table(
                {
                    "user": pa.array(ids, type=dtype),
                    "sid": [row[1] for row in chunk],
                    "split": [row[2] for row in chunk],
                    "label": [row[3] for row in chunk],
                    "cutoff": [25] * count,
                }
            ),
            directory / f"part-{index}.parquet",
        )
    assert start == len(rows)
    # The entity id column is also metadata: that role sees the same int64 ids.
    return api.AnchorTable(
        api.Parquet(str(directory)),
        "user",
        "cutoff",
        sample_id="sid",
        split="split",
        targets={"label": "label"},
        metadata=("user",),
    )


def _engine(tmp_path):
    return api.DiskGraphEngine(index=tmp_path / "index", num_threads=1)


def _assert_same_batches(actual, expected, **kwargs):
    left = list(actual.batches(shuffle=False, **kwargs))
    right = list(expected.batches(shuffle=False, **kwargs))
    assert len(left) == len(right) > 0
    for got, want in zip(left, right, strict=True):
        _equal(got.inputs, want.inputs)
        _equal(got.targets, want.targets)
        _equal(got.metadata, want.metadata)


def _assert_int_requests(task):
    requests = task._anchors.fetch(api.ALL_ANCHORS, range(len(ANCHORS)))
    assert {r.sample_id: r.entity_id for r in requests} == {row[1]: row[0] for row in ANCHORS}
    assert all(type(r.entity_id) is int and r.metadata["user"] == r.entity_id for r in requests)
    assert all(type(r.metadata["user"]) is int for r in requests)
    return requests


MIXED = pytest.mark.parametrize(
    "order", ["int32-then-decimal", "decimal-then-int64"], ids=lambda order: order
)


def _parts(order, decimal):
    # BIG never fits int32, so the first order also needs the int64 output type.
    if order == "int32-then-decimal":
        return [(2, pa.int32()), (4, decimal)]
    return [(3, decimal), (3, pa.int64())]


@DECIMALS
@MIXED
def test_native_anchors_read_narrow_and_decimal_ids_as_int64(tmp_path, dataset, decimal, order):
    mixed = _anchors(tmp_path / "mixed", _parts(order, decimal))
    baseline = _anchors(tmp_path / "baseline", [(len(ANCHORS), pa.int64())])
    with api.prepare(_contract("graph"), dataset, _engine(tmp_path)) as plan:
        with (
            plan.materialize(baseline, fit_split="train") as expected,
            plan.materialize(mixed, fit_split="train") as actual,
        ):
            assert isinstance(actual._anchors, NativeAnchors)
            requests = _assert_int_requests(actual)
            assert requests == expected._anchors.fetch(api.ALL_ANCHORS, range(len(ANCHORS)))
            assert actual._anchors.fitting_requests(api.ALL_ANCHORS) == (
                expected._anchors.fitting_requests(api.ALL_ANCHORS)
            )
            _assert_same_batches(actual, expected, split="train", output_format="model")
            _assert_same_batches(actual, expected, split="valid", output_format="inspectable")


@DECIMALS
@MIXED
def test_scalar_values_read_narrow_and_decimal_ids_as_int64(tmp_path, dataset, decimal, order):
    mixed = _anchors(tmp_path / "mixed", _parts(order, decimal))
    baseline = _anchors(tmp_path / "baseline", [(len(ANCHORS), pa.int64())])
    with api.prepare(_contract("scalar"), dataset, _engine(tmp_path)) as plan:
        values = plan._materialize_values(mixed, tmp_path / "values", batch_size=2)
        stored = pq.read_table(values.anchors.source.path)
        assert stored.schema.field("_anchor_entity").type == pa.int64()
        assert stored.schema.field("user").type == pa.int64()
        assert dict(
            zip(stored["_anchor_id"].to_pylist(), stored["_anchor_entity"].to_pylist())
        ) == {row[1]: row[0] for row in ANCHORS}
        with (
            plan.materialize(baseline, fit_split="train") as expected,
            plan.materialize(mixed, fit_split="train") as actual,
        ):
            assert isinstance(actual._anchors, NativeAnchors)
            assert _assert_int_requests(actual) == expected._anchors.fetch(
                api.ALL_ANCHORS, range(len(ANCHORS))
            )
            _assert_same_batches(actual, expected, split="train", output_format="model")
            _assert_same_batches(actual, expected, split="valid", output_format="inspectable")


@DECIMALS
@pytest.mark.parametrize("kind", ["graph", "scalar"])
@pytest.mark.parametrize("value", [2**63, -(2**63) - 1], ids=["above", "below"])
def test_ids_outside_int64_fail_without_leaving_a_stage(tmp_path, dataset, decimal, kind, value):
    rows = [*ANCHORS[:4], (value, "x", "train", 0.0)]
    anchors = _anchors(tmp_path / "anchors", [(3, pa.int64()), (2, decimal)], rows)
    index = tmp_path / "index"
    with api.prepare(_contract(kind), dataset, _engine(tmp_path)) as plan:
        with pytest.raises(ValueError, match=r"anchors\.entity_id in .*part-1\.parquet.*int64"):
            plan.materialize(anchors, fit_split="train")
        assert not [*index.glob("tasks/*"), *index.glob("train_tables/*")]


@pytest.mark.parametrize(
    "physical",
    [
        pa.int64(),
        pa.uint8(),
        pa.uint32(),
        pa.float64(),
        pa.decimal128(38, 2),
        pa.decimal128(5, -2),
        pa.string(),
        pa.timestamp("s"),
        pa.list_(pa.int64()),
        # DType has no decimal32; older supported pyarrow releases lack the type.
        *([pa.decimal32(9, 0)] if hasattr(pa, "decimal32") else []),
    ],
    ids=str,
)
def test_other_entity_id_types_are_left_to_the_per_value_check(physical):
    assert _entity_id_type(physical, INT64) == physical
    values = pa.nulls(1, type=physical)
    assert _entity_ids(values, INT64, "anchors.parquet") is values


@pytest.mark.parametrize("key", ["int32", "uint64", "string"])
def test_only_an_int64_key_reads_other_physical_types(key):
    for physical in (pa.int8(), pa.int32(), pa.decimal128(38, 0)):
        assert _entity_id_type(physical, api.DType(key)) == physical
