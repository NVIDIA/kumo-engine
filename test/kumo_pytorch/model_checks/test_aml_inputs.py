# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import timedelta
from test.kumo_pytorch.aml_inputs import SECOND
from test.kumo_pytorch.contracts import build_contract, counts

import torch

import kumo.pytorch.api as r

from .fixtures import build_fixture, probe_anchor_table


def test_combined_history_matches_unshifted_sequence_and_count_boundaries(tmp_path):
    fixture = build_fixture()
    dataset = fixture.parquet_dataset(tmp_path / "source")
    originals = fixture.training
    shifted = tuple(
        replace(
            a,
            as_of_time=a.as_of_time - timedelta(seconds=1),
            metadata={"prediction_time_ns": int(a.as_of_time.timestamp()) * SECOND},
        )
        for a in originals
    )
    cutoff = max(a.as_of_time for a in shifted)
    histories = []
    for family, anchors in (("nuformer", originals), ("combined", shifted)):
        contract = build_contract(
            fixture.schema, family, max_events=len(fixture.tables["transactions"])
        )
        engine = r.DiskGraphEngine(index=tmp_path / family, num_threads=1)
        with r.prepare(contract, dataset, engine) as plan:
            task = plan.materialize(
                probe_anchor_table(plan, anchors),
                fit_split="train",
                encoder_cutoff=cutoff,
            )
            batch = next(task.batches(seed=0, output_format="model"))
            histories.append(batch.inputs["history"])
            learned = [
                value
                for state in task.fitted_state["encoders"].values()
                for value, _ in state.get("entries", ())
            ]
            assert "wire" in learned and "new_format" not in learned
    for name in ("value_ids", "valid"):
        torch.testing.assert_close(
            histories[0][name], histories[1][name], rtol=0, atol=0
        )

    # Independently count [T-24h,T) in raw rows; events at T and after are excluded.
    contract = counts(
        fixture.schema,
        "accounts",
        [("count", ("account_touches",), "event_time", -86400)],
        normalize=False,
        inclusive=True,
    )
    anchors = tuple(replace(a, targets=()) for a in shifted)
    engine = r.DiskGraphEngine(index=tmp_path / "counts", num_threads=1)
    with r.prepare(contract, dataset, engine) as plan:
        task = plan.materialize(
            probe_anchor_table(plan, anchors), fit_split="train", encoder_cutoff=cutoff
        )
        batch = next(task.batches(batch_size=2, output_format="model"))
        expected = [
            sum(
                row["account_id"] == a.entity_id
                and a.as_of_time - timedelta(days=1) <= row["event_time"] < a.as_of_time
                for row in fixture.tables["account_events"]
            )
            for a in originals
        ]
        assert batch.inputs["aggregates"]["count"]["values"].tolist() == expected
