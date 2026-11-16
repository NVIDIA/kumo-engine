# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent source-size × fit-anchor correctness sweeps, not RAM benchmarks."""

import math
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conformance.assertions import bound_tables
from kumo.pytorch import api as r
from tests.test_anchor_tables import _table
from tests.test_materialized_feature_stats import aggregate_contract


@pytest.fixture(scope="module", params=(64, 1024, 16384), ids=lambda n: f"source-{n}")
def source(request):
    count = request.param
    per_root = count // 8
    tables = {
        "users": [dict(user_id=i, score=float(i)) for i in range(1, 9)],
        "events": [
            dict(
                event_id=(root - 1) * per_root + stamp,
                user_id=root,
                occurred_at=stamp,
                amount=float(root * stamp),
            )
            for root in range(1, 9)
            for stamp in range(1, per_root + 1)
        ][::-1],
    }
    assert len(tables["events"]) == count
    return tables, per_root


@pytest.mark.parametrize("fit_count", (8, 64, 512), ids=lambda n: f"fit-{n}")
@pytest.mark.parametrize("engine", ("in_memory", "diskgraph"))
def test_source_rows_and_fitting_anchors(source, tmp_path, fit_count, engine):
    tables, per_root = source
    base = aggregate_contract()
    score = base.sample.entity_values["score"]
    contract = replace(
        base,
        sample=replace(
            base.sample,
            entity_values={
                "score": score,
                "seen": r.Aggregate(score.source, "count", dtype=r.DType("int64")),
            },
        ),
        batch=replace(
            base.batch,
            inputs={
                **base.batch.inputs,
                "raw": r.Input(
                    r.EntitySource(),
                    r.Fields(
                        (
                            r.Encode("score", r.Raw()),
                            r.Encode("seen", r.Raw()),
                        )
                    ),
                    r.Separate(),
                ),
            },
        ),
    )
    # The source is unchanged when fit_count changes. Unique IDs may repeat a
    # root/cutoff: each fitting anchor must contribute, not just unique contexts.
    rows = [
        dict(
            sid=f"train-{i}",
            user_id=(i * 5) % 8 + 1,
            at=1 + (i * 7 + i // 8) % (per_root - 1),
            split="train",
        )
        for i in range(fit_count)
    ]
    rows += [
        dict(sid="valid-full-1", user_id=1, at=per_root, split="valid"),
        dict(sid="valid-full-8", user_id=8, at=per_root, split="valid"),
        dict(sid="test-full-3", user_id=3, at=per_root, split="test"),
        dict(sid="test-empty", user_id=7, at=0, split="test"),
        dict(sid="test-first", user_id=8, at=1, split="test"),
    ]
    assert sum(row["split"] == "train" for row in rows) == fit_count
    # Inclusive source windows: n events at times 1..n, amounts root*t.
    expected = {
        row["sid"]: (row["at"], float(row["user_id"] * row["at"] * (row["at"] + 1) // 2))
        for row in rows
    }
    scores = [expected[row["sid"]][1] for row in rows[:fit_count]]
    mean = math.fsum(scores) / fit_count
    scale = math.sqrt(math.fsum((value - mean) ** 2 for value in scores) / fit_count)
    assert scale > 0
    assert 0.0 not in scores  # Empty held-out context must remain unknown to vocabulary.
    if engine == "diskgraph":
        path = tmp_path / "anchors.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=17)
        anchors = _table(source=r.Parquet(str(path)))
    else:
        anchors = _table(source=r.InMemoryRows(rows))

    with bound_tables(contract, tables, engine) as (dataset, backend):
        with r.prepare(contract, dataset, backend) as plan:
            with plan.materialize(anchors, fit_split="train") as task:
                encoder = task.output("entity").field(name="score").encoder
                assert encoder.mean == pytest.approx(mean, rel=1e-10)
                assert encoder.scale == pytest.approx(scale, rel=1e-10)
                vocabulary = dict(task.output("tokens").field(name="score").encoder.entries)
                assert set(vocabulary) == set(scores), (
                    "Fitting included held-out anchors or lost training values"
                )
                fingerprint = task.fitted_state.fingerprint
                for split in ("train", "valid", "test"):
                    seen, sizes = [], []
                    for batch in task.batches(
                        split=split,
                        batch_size=31,
                        chunk_batches=2,
                        output_format="model" if engine == "diskgraph" else "inspectable",
                    ):
                        ids = list(batch.metadata["sample_ids"])
                        seen.extend(ids)
                        sizes.append(len(ids))
                        assert not batch.metadata["is_padding"].any()
                        assert set(batch.inputs) == {"entity", "tokens", "raw"}
                        counts = [expected[sid][0] for sid in ids]
                        sums = [expected[sid][1] for sid in ids]
                        for name, answers in (("seen", counts), ("score", sums)):
                            field = batch.inputs["raw"][name]
                            assert field["valid"].all()
                            assert field["values"].tolist() == answers
                        standardized = batch.inputs["entity"]["score"]
                        assert standardized["valid"].all()
                        assert standardized["values"].tolist() == pytest.approx(
                            [(value - mean) / scale for value in sums],
                            rel=2e-6,
                            abs=2e-6,
                        )
                        tokens = batch.inputs["tokens"]["score"]
                        assert tokens["valid"].all()
                        assert tokens["values"].tolist() == [
                            vocabulary.get(value, 1) for value in sums
                        ]
                    selected = [row["sid"] for row in rows if row["split"] == split]
                    assert seen == selected, "Wrong split, dropped or duplicated anchors"
                    assert sizes == [31] * (len(selected) // 31) + (
                        [len(selected) % 31] if len(selected) % 31 else []
                    )
                assert task.fitted_state.fingerprint == fingerprint
