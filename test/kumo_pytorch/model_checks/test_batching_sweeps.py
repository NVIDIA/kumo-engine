# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AML sample limits and token budgets through materialized DiskGraph anchors.

Generic planner/layout rules stay in package tests. These checks exercise the
five AML consumers and independently verify delivery and source/task alignment.
"""

from collections import Counter
from dataclasses import replace
from test.kumo_pytorch.contracts import build_contract
from test.kumo_pytorch.involvement import history, verify_inputs
from test.kumo_pytorch.masked_values import original_tokens
from test.kumo_pytorch.masked_values import verify_targets as verify_masked
from test.kumo_pytorch.models.aml import build_model
from test.kumo_pytorch.next_token import verify_targets as verify_next
from test.kumo_pytorch.reference_tokens import history_tokens

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import kumo.pytorch.api as r

from .fixtures import CUTOFF, VALUE_FIELDS, build_fixture
from .test_aml import CASES

T = int(CUTOFF.timestamp())
PROFILES = [(size, None, False) for size in (32, 128, 512)] + [
    (32, 4, False),
    (32, 17, True),
]
SWEEPS = [
    pytest.param(
        selection,
        family,
        size,
        capacity,
        bucket,
        id=f"{selection['selector']['workload_id']}-{family}-"
        + (
            f"samples-{size}"
            if capacity is None
            else f"budget-{capacity}-bucket-{bucket}"
        ),
    )
    for selection, family in CASES
    for size, capacity, bucket in PROFILES
    if capacity is None or family != "graphsage"
]


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    fixture = build_fixture()
    directory = tmp_path_factory.mktemp("aml-batching-source")
    reference = {
        "events": [
            dict(
                id=row["transaction_id"],
                time=row["event_time_seconds"],
                accounts={str(row["sender_id"]), str(row["receiver_id"])},
                sender=str(row["sender_id"]),
                receiver=str(row["receiver_id"]),
                values={name: row[name] for name in VALUE_FIELDS},
            )
            for row in fixture.tables["transactions"]
        ]
    }
    return (
        fixture,
        fixture.parquet_dataset(directory / "sources"),
        directory / "index",
        reference,
    )


@pytest.mark.parametrize("selection,family,size,capacity,bucket", SWEEPS)
def test_aml_sample_limits_and_token_budgets(
    source, tmp_path, selection, family, size, capacity, bucket
):
    fixture, dataset, index, reference = source
    graph = family in {"graphsage", "combined"}
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    # Unique sample IDs over repeated roots test ownership independently of IDs.
    # Two complete sample-limited batches plus a five-sample final tail.
    count = 2 * size + 5
    rows = [
        dict(
            sample_id=f"{split}-{i}",
            entity_id=i % 2 + 1,
            cutoff=T - int(graph),
            prediction_time_ns=T * 10**9,
            label=float(i % 2 == 0),
            split=split,
        )
        for split, total in (("train", 2), ("test", count))
        for i in range(total)
    ]
    anchors = {
        row["sample_id"]: dict(
            sample_id=row["sample_id"],
            entity_id=str(row["entity_id"]),
            as_of_time=T * 10**9,
        )
        for row in rows
    }
    labels = {row["sample_id"]: row["label"] for row in rows}
    expected_ids = [row["sample_id"] for row in rows if row["split"] == "test"]
    training = [anchors["train-0"], anchors["train-1"]]
    identities = {str(i): i for i in range(1, 5)}
    counts = {sid: len(history(anchor, reference)) for sid, anchor in anchors.items()}
    assert set(counts.values()) == {6, 7}
    # Costs come from source fields/events, before next-token shifting.
    lengths = {
        sid: 6 * n if family == "pragma" else 11 * n - 1 for sid, n in counts.items()
    }
    budget = None if capacity is None else capacity * max(lengths.values())
    contract = build_contract(fixture.schema, family, native=native, max_events=11)
    contract = replace(
        contract,
        batch=replace(
            contract.batch,
            batching=r.Batching(
                max_samples=size,
                max_tokens=budget,
                length_input="history" if budget else None,
                bucket_by_length=bucket,
            ),
        ),
    )
    path = tmp_path / "anchors.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    table = r.AnchorTable(
        r.Parquet(str(path)),
        "entity_id",
        "cutoff",
        sample_id="sample_id",
        split="split",
        targets={} if native else {"label": "label"},
        metadata=("prediction_time_ns",) if graph else (),
    )

    def cost(ids):
        if family == "pragma":  # Packed field values; padded event metadata.
            return sum(lengths[sid] for sid in ids)
        return len(ids) * max(lengths[sid] for sid in ids)

    engine = r.DiskGraphEngine(index=index, num_threads=1)
    with r.prepare(contract, dataset, engine) as plan:
        with plan.materialize(table, fit_split="train", encoder_cutoff=T - 1) as task:
            fingerprint = task.fitted_state.fingerprint
            state = task.fitted_state.to_dict()
            model = build_model(task, selection, family, dim=8, seed=17)
            delivered = []
            for batch in task.batches(
                split="test",
                chunk_batches=1,
                output_format="model",
                shuffle=bucket,
                seed=7,
                epoch=0,
            ):
                ids = list(batch.metadata["sample_ids"])
                assert 0 < len(ids) == batch.batch_size <= size
                assert not batch.metadata["is_padding"].any()
                delivered.append(ids)
                if budget is not None:
                    assert cost(ids) <= budget
                    stream = batch.inputs["history"]
                    # NextTokenTarget removes one token after budget planning.
                    slots = cost(ids) - (
                        len(ids) if native and family == "nuformer" else 0
                    )
                    assert stream["value_ids"].numel() == slots
                    assert slots <= budget
                if family == "pragma":
                    original = original_tokens(
                        batch, anchors, reference, state, identities
                    )
                    assert len(original) == sum(lengths[sid] for sid in ids)
                    verify_masked(batch, original)
                    model.backbone.validate_kumo_batch(
                        batch, target_name="masked_values"
                    )
                elif native:
                    verify_next(batch, anchors, reference, state, identities)
                    model.backbone.validate_kumo_batch(batch, target_name="next_token")
                else:
                    verify_inputs(
                        batch, family, anchors, reference, identities, training
                    )
                    assert set(batch.targets) == {"label"}
                    assert batch.targets["label"]["valid"].all()
                    assert batch.targets["label"]["values"].flatten().tolist() == [
                        labels[sid] for sid in ids
                    ]
                    if family != "graphsage":
                        stream = batch.inputs["history"]
                        for owner, sid in enumerate(ids):
                            tokens = history_tokens(
                                anchors[sid], history(anchors[sid], reference), state
                            )
                            valid = stream["valid"][owner]
                            assert stream["value_ids"][owner][valid].tolist() == tokens
                            assert not stream["value_ids"][owner][~valid].any()
                    model.backbone.validate_kumo_batch(batch)
            flat = [sid for ids in delivered for sid in ids]
            assert Counter(flat) == Counter(expected_ids), (
                "Missing, duplicate or wrong-split sample"
            )
            if not bucket:
                assert flat == expected_ids
            if budget is None:
                assert [len(ids) for ids in delivered] == [size, size, 5]
            else:
                assert len(delivered) > 3, "Token budget did not constrain delivery"
                # Chunk boundaries/bucketing can close batches early. Require
                # the budget to bind at least once, without imposing a packing
                # algorithm or order on the package's different input paths.
                assert any(
                    len(ids) < size and cost(ids + ids[:1]) > budget
                    for ids in delivered
                )
            assert task.fitted_state.fingerprint == fingerprint
