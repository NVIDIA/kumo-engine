# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BPE clipping, supervision, native length budgets and interrupted epochs together."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import kumo.pytorch.api as r
from tests.conformance.fixtures import (
    INT,
    TEXT,
    TIME,
    assert_tree,
    batches,
    materialized,
    record,
)


def token_fixture(layout, objective):
    golden = (
        Path(__file__).parents[2]
        / "conformance/spec/guarantees/G-ENCODING-001/golden/C-ENCODING-003.json"
    )
    declaration = r.DataLoaderSpec.from_dict(json.loads(golden.read_text())["contract"])
    text = declaration.batch.inputs["entity"].representation.fields[0].encoding
    schema = r.DataSchema(
        (
            r.TableSpec("users", "id", {"id": INT}, time_column=r.NON_TEMPORAL),
            r.TableSpec(
                "events",
                "id",
                {"id": INT, "owner": INT, "at": TIME, "text": TEXT},
                time_column="at",
            ),
        ),
        (r.RelationSpec("events", "users", "id", "events", "owner"),),
    )
    roles = ("pad", "unknown", "missing", "mask", "bos", "eos")
    target = (
        r.NextTokenTarget("history")
        if objective == "next"
        else r.MaskedValuesTarget(
            "history", token_probability=1.0, event_probability=0.0, key_probability=0.0
        )
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(
            r.EntitySpec("users"),
            events={
                "events": r.EventSpec(
                    r.Rows(("events",), r.TimeWindow("at", stop=0)),
                    {"text": r.Field("text")},
                )
            },
        ),
        r.BatchSpec(
            {
                "history": r.Input(
                    r.EventSource(("events",)),
                    r.Fields(
                        (r.Encode("text", text, "tokens"),),
                        row=r.Tokens(missing="emit"),
                        history=r.Flatten(
                            6,
                            keep="last",
                            prefer_event_boundary=False,
                            before="bos",
                            after="eos",
                        ),
                    ),
                    layout,
                )
            },
            targets={"answer": target},
            id_spaces={"tokens": r.IdSpace({name: name for name in roles})},
            batching=r.Batching(
                max_samples=3,
                max_tokens=12,
                length_input="history",
                bucket_by_length=True,
            ),
        ),
    )
    events = [
        (1, 90, "red bike " * 4),
        (1, 99, "red"),
        (1, 100, None),
        (2, 90, ""),
        (3, 90, None),
        (3, 95, "é"),
        (4, 101, "red"),
        (5, 90, "bike bike"),
    ]
    rows = {
        "users": [{"id": i} for i in range(1, 6)],
        "events": [
            {"id": i, "owner": owner, "at": at, "text": text}
            for i, (owner, at, text) in enumerate(events)
        ],
    }
    # Six role IDs precede the golden's eleven ordinary BPE IDs: red=13, bike=16.
    # Clipping retains the final four content IDs, then adds BOS=4 and EOS=5.
    expected = {
        1: [4, 13, 16, 13, 2, 5],
        2: [4, 5],
        3: [4, 2, 1, 5],
        4: [4, 5],
        5: [4, 16, 16, 5],
    }
    records = [record(f"anchor-{i}", entity) for i, entity in enumerate([1, 2, 3, 4, 5, 1, 2])]
    return (
        contract,
        rows,
        records,
        {row["sample"]: expected[row["entity"]] for row in records},
    )


def verify_tokens(batch, expected, objective):
    source, target = batch.inputs["history"], batch.targets["answer"]
    ids = batch.metadata["sample_ids"]
    assert len(ids) <= 3
    assert source["value_ids"].numel() <= 12, "Padded slots also consume the batch token budget"
    assert source["valid"].dtype == target["valid"].dtype == torch.bool
    if "offsets" in source:
        assert_tree(source["offsets"], target["offsets"])
        offsets = source["offsets"].tolist()
        assert offsets[0] == 0 and offsets[-1] == source["value_ids"].numel()
    for i, identity in enumerate(ids):
        original = expected[identity]
        inputs = (
            original[:-1] if objective == "next" else [x if x in (4, 5) else 3 for x in original]
        )
        labels = (
            original[1:] if objective == "next" else [-100 if x in (4, 5) else x for x in original]
        )
        valid = [True] * len(labels) if objective == "next" else [x not in (4, 5) for x in original]
        if "offsets" in source:
            index = slice(offsets[i], offsets[i + 1])
            assert offsets[i + 1] - offsets[i] == len(inputs)
        else:
            index = (i, slice(0, len(inputs)))
            padding = (i, slice(len(inputs), None))
            assert not source["valid"][padding].any()
            assert not target["valid"][padding].any()
        assert source["valid"][index].all()
        assert source["value_ids"][index].tolist() == inputs, identity
        assert target["values"][index].tolist() == labels, identity
        assert target["valid"][index].tolist() == valid, identity
        if objective == "masked":
            assert target["corruption"][index].tolist() == valid


@pytest.mark.parametrize("layout", [r.Padded(), r.Packed()], ids=["padded", "packed"])
@pytest.mark.parametrize("objective", ["next", "masked"])
def test_bpe_supervision_budget_cache_and_resume(tmp_path, layout, objective):
    contract, rows, records, expected = token_fixture(layout, objective)
    # A serialized declaration must preserve the same production path and answers.
    contract = r.DataLoaderSpec.from_dict(json.loads(json.dumps(contract.to_dict())))
    with materialized(contract, rows, records, tmp_path) as (task, _, _):
        native = type(task._anchors)
        original_iter = native._iter
        traces = []

        def observe(owner, *args, **kwargs):
            assert kwargs.get("length_planning") is not None
            iterator = original_iter(owner, *args, **kwargs)
            traces.append(iterator)
            return iterator

        def forbidden(*args, **kwargs):
            raise AssertionError("Native delivery fell back to per-anchor requests")

        with (
            patch.object(native, "_iter", observe),
            patch.object(native, "_requests", forbidden),
            patch.object(task._assembler._source, "build_model_ordered", forbidden),
        ):
            first = list(batches(task, epoch=0))
            assert traces, "Native iterator was not exercised"
            assert sum(t.length_planning_stats()[0] for t in traces) == len(records)
            assert sorted(s for batch in first for s in batch.metadata["sample_ids"]) == sorted(
                expected
            )
            for batch in first:
                verify_tokens(batch, expected, objective)
            traces.clear()
            second = list(batches(task, epoch=1))
            assert sorted(s for batch in second for s in batch.metadata["sample_ids"]) == sorted(
                expected
            )
            assert traces
            assert sum(t.length_planning_stats()[0] for t in traces) == 0
            assert sum(t.length_planning_stats()[1] for t in traces) == len(records)
            for batch in second:
                verify_tokens(batch, expected, objective)
            interrupted = batches(task, epoch=1)
            prefix = next(interrupted)
            saved = interrupted.state_dict()
            torch.save(saved, tmp_path / "iterator.pt")
            restored = batches(task, epoch=1)
            restored.load_state_dict(torch.load(tmp_path / "iterator.pt", weights_only=True))
            remaining = list(restored)
            assert len(remaining) == len(second) - 1
            for actual, baseline in zip([prefix, *remaining], second, strict=True):
                assert_tree(actual.inputs, baseline.inputs)
                assert_tree(actual.targets, baseline.targets)
                assert_tree(actual.metadata, baseline.metadata)
