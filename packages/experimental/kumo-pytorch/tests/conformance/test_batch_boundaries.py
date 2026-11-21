# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent batching witnesses ported from Arena's G-BATCH goldens.

No Arena imports or data are needed. Required engine answers fail normally.
The existing core/native tests retain ownership of general planner behavior.
"""

from contextlib import contextmanager

import numpy as np
import pytest

from conformance.assertions import bound_tables
from kumo.pytorch import api as r
from tests.native_fixtures import request_anchor_table

pytestmark = pytest.mark.engine("diskgraph")


def pytest_generate_tests(metafunc):
    metafunc.parametrize("engine_name", ("diskgraph",))


@contextmanager
def history_task(
    engine_name,
    lengths,
    *,
    packed=False,
    max_events=None,
    fit_all=False,
    population_count=None,
    **policy,
):
    """Values, labels and histories encode ownership independently of batch order."""
    integer = r.DType("int64")
    schema = r.DataSchema(
        (
            r.TableSpec(
                "entities",
                "id",
                {"id": integer, "value": integer},
                time_column=r.NON_TEMPORAL,
            ),
            r.TableSpec(
                "events",
                "id",
                {
                    "id": integer,
                    "entity_id": integer,
                    "time": r.DType("timestamp", unit="s", timezone="UTC"),
                    "value": integer,
                },
                time_column="time",
            ),
        ),
        (r.RelationSpec("history", "entities", "id", "events", "entity_id"),),
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(
            r.EntitySpec("entities"),
            entity_values={"value": r.Field(column="value")},
            events={
                "events": r.EventSpec(
                    r.Rows(("history",), time=r.TimeWindow("time", stop=-1)),
                    {"value": r.Field(column="value")},
                )
            },
            targets={"label": r.ExternalTarget(integer)},
        ),
        r.BatchSpec(
            {
                "entity": r.Input(
                    r.EntitySource(), r.Fields((r.Encode("value", r.Raw()),)), r.Separate()
                ),
                "history": r.Input(
                    r.EventSource(("events",), max_events=max_events, keep="last"),
                    r.Fields((r.Encode("value", r.Raw()),)),
                    r.Packed() if packed else r.Padded(),
                ),
            },
            {"label": r.LabelTarget("label", r.Raw())},
            batching=r.Batching(**({"max_samples": 16} | policy)),
        ),
    )
    tables = {
        "entities": [{"id": i, "value": 10 * i} for i in range(1, len(lengths) + 1)],
        "events": [
            {"id": 1000 * i + j, "entity_id": i, "time": j, "value": 100 * i + j}
            for i, length in enumerate(lengths, 1)
            for j in range(1, length + 1)
        ],
    }
    requests = [
        r.SampleRequest(f"s{i:02d}", i, 100, targets=(r.TargetValue("label", 100 + i),))
        for i in range(1, len(lengths) + 1)
    ]
    with bound_tables(contract, tables, engine_name) as (dataset, engine):
        with r.prepare(contract, dataset, engine) as plan:
            # Fit on a legal one-event history; batching witnesses need no learned state.
            with plan.materialize(
                request_anchor_table(requests if fit_all else [requests[-1]])
            ) as fitted:
                with plan.materialize(
                    request_anchor_table(requests[:population_count]), fitted=fitted.fitted_state
                ) as task:
                    yield task, requests


def check_batches(batches, groups, lengths, *, packed=False, cap=None):
    """Check authored owners, values, validity, labels and packed offsets."""
    assert [list(b.metadata["sample_ids"]) for b in batches] == [
        [f"s{i:02d}" for i in group] for group in groups
    ]
    for batch, owners in zip(batches, groups, strict=True):
        assert batch.batch_size == len(owners)
        assert batch.metadata["entity_ids"].tolist() == owners
        for field, values in (
            (batch.inputs["entity"]["value"], [10 * i for i in owners]),
            (batch.targets["label"], [100 + i for i in owners]),
        ):
            np.testing.assert_array_equal(field["values"], values)
            np.testing.assert_array_equal(field["valid"], [True] * len(owners))
        sequences = [list(range(1, lengths[i - 1] + 1)) for i in owners]
        if cap is not None:
            sequences = [seq[-cap:] for seq in sequences]
        values = [[100 * i + j for j in seq] for i, seq in zip(owners, sequences)]
        history = batch.inputs["history"]
        if packed:
            offsets = [0, *np.cumsum([len(seq) for seq in sequences]).tolist()]
            expected = np.array([v for seq in values for v in seq], dtype=np.int64)
            valid = np.ones(len(expected), dtype=bool)
            np.testing.assert_array_equal(history["offsets"], offsets)
            np.testing.assert_array_equal(history["cu_seqlens"], offsets)
            assert history["max_seqlen"] == max(map(len, sequences))
        else:
            shape = (len(owners), max(map(len, sequences)))
            expected, valid = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=bool)
            for row, seq in enumerate(values):
                expected[row, : len(seq)], valid[row, : len(seq)] = seq, True
        np.testing.assert_array_equal(history["value"]["values"], expected)
        np.testing.assert_array_equal(history["value"]["valid"], valid)
        np.testing.assert_array_equal(history["valid"], valid)


@pytest.mark.parametrize("count", (0, 1, 15, 16, 17, 32, 33), ids=lambda n: f"n{n}")
@pytest.mark.parametrize("drop", (False, True), ids=("keep", "drop"))
def test_sample_count_boundaries(engine_name, count, drop):
    """G-BATCH-001/002: empty, just below/at/above one and two full batches."""
    lengths = [1] * max(1, count)
    with history_task(engine_name, lengths, drop_last=drop, population_count=count) as (
        task,
        requests,
    ):
        batches = list(task.batches(shuffle=False, output_format="model"))
        stop = count - count % 16 if drop else count
        groups = [list(range(i + 1, min(i + 16, stop) + 1)) for i in range(0, stop, 16)]
        check_batches(batches, groups, lengths)


@pytest.mark.parametrize(
    "lengths", ((7,), (1, 7), (7, 1), (1, 2, 7), (6,)), ids=("7", "1,7", "7,1", "1,2,7", "6")
)
@pytest.mark.parametrize("packed", (False, True), ids=("padded", "packed"))
def test_oversized_position_and_exact_budget(engine_name, lengths, packed):
    """G-BATCH-009: a too-long sample is never emitted, wherever it occurs."""
    with history_task(
        engine_name,
        [*lengths, 1],
        packed=packed,
        max_tokens=6,
        length_input="history",
        population_count=len(lengths),
    ) as (task, requests):
        if lengths == (6,):
            check_batches(
                list(task.batches(shuffle=False, output_format="model")),
                [[1]],
                lengths,
                packed=packed,
            )
            return
        import diskgraph

        with pytest.raises(
            (ValueError, diskgraph.DiskgraphConfigError), match=r"exceeds Batching\.max_tokens"
        ):
            for batch in task.batches(shuffle=False, output_format="model"):
                owners = batch.metadata["entity_ids"].tolist()
                assert all(lengths[i - 1] <= 6 for i in owners), "oversized sample emitted"
                check_batches([batch], [owners], lengths, packed=packed)


@pytest.mark.parametrize("bucket", (False, True), ids=("unbucketed", "bucketed"))
def test_bucketing_has_exact_padding_cost(engine_name, bucket):
    """G-BATCH-012: the authored four-sample example saves 54 padding slots."""
    lengths = [1, 30, 2, 29]
    with history_task(
        engine_name,
        [*lengths, 1],
        max_samples=2,
        bucket_by_length=bucket,
        length_input="history",
        population_count=len(lengths),
    ) as (task, requests):
        batches = list(task.batches(shuffle=False, output_format="model"))
        check_batches(batches, [[1, 3], [4, 2]] if bucket else [[1, 2], [3, 4]], lengths)
        allocated = sum(batch.inputs["history"]["valid"].numel() for batch in batches)
        assert allocated - sum(lengths) == (2 if bucket else 56)
        # The only three pair partitions establish the minimum for this fixture,
        # without imposing global optimality on the streaming scheduler.
        costs = [
            2 * max(lengths[a], lengths[b]) + 2 * max(lengths[c], lengths[d]) - sum(lengths)
            for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))
        ]
        assert costs == [56, 2, 56]


def test_bucketing_uses_truncated_lengths(engine_name):
    """G-BATCH-013: 40/3/31 raw lengths become 30/3/30 before stable sorting."""
    lengths = [40, 3, 31]
    with history_task(
        engine_name,
        [*lengths, 1],
        max_events=30,
        max_samples=2,
        bucket_by_length=True,
        length_input="history",
        population_count=len(lengths),
    ) as (task, requests):
        batches = list(task.batches(shuffle=False, output_format="model"))
        check_batches(batches, [[2, 1], [3]], lengths, cap=30)


@pytest.mark.parametrize("chunks", (1, 2), ids=("chunks1", "chunks2"))
def test_bucketing_respects_chunk_scope(engine_name, chunks):
    """G-BATCH-014: materialized bucketing sorts within each requested chunk."""
    lengths = [1, 30, 2, 29]
    ordered = [1, 2, 3, 4]  # Authored Parquet row order, with shuffle disabled.
    with history_task(
        engine_name,
        lengths,
        max_samples=2,
        fit_all=True,
        bucket_by_length=True,
        length_input="history",
    ) as (task, requests):
        # Sort only within each chunk of the authored anchor order.
        expected = []
        for start in range(0, len(ordered), 2 * chunks):
            chunk = sorted(ordered[start : start + 2 * chunks], key=lambda i: lengths[i - 1])
            expected.extend([chunk[i : i + 2] for i in range(0, len(chunk), 2)])
        batches = list(task.batches(shuffle=False, chunk_batches=chunks, output_format="model"))
        check_batches(batches, expected, lengths)
