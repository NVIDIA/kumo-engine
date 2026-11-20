# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen encoders meet nulls, wide identities, shared dictionaries and evaluation poison."""

import pytest
import torch

import kumo.pytorch.api as r
from conformance.refusals import forbid_fitting
from tests.conformance.fixtures import (
    FLOAT,
    INT,
    TEXT,
    TIME,
    assert_tree,
    batches,
    materialized,
    record,
)


@pytest.mark.parametrize("seed", [0, 17])
def test_frozen_constant_and_empty_statistics_with_shared_sparse_ids(tmp_path, seed):
    columns = {
        "id": INT,
        "at": TIME,
        "constant": FLOAT,
        "empty": FLOAT,
        "left": TEXT,
        "right": TEXT,
        "wide": INT,
    }
    schema = r.DataSchema((r.TableSpec("items", "id", columns, time_column="at"),))
    encoders = {
        "constant": r.ZScore(dtype="float64"),
        "empty": r.ZScore(dtype="float64"),
        "left": r.UseEncoding("shared"),
        "right": r.UseEncoding("shared"),
        "wide": r.Vocabulary(values=((2**53 + 1, 2**53 + 3), (2**53 + 2, 2**53 + 4))),
    }
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(r.EntitySpec("items"), entity_values={k: r.Field(k) for k in encoders}),
        r.BatchSpec(
            {
                "values": r.Input(
                    r.EntitySource(),
                    r.Fields(tuple(r.Encode(k, v) for k, v in encoders.items())),
                    r.Separate(),
                )
            },
            encodings={
                "shared": r.Vocabulary(
                    values=(("", 7), ("NULL", 11), ("é", 13), ("e\u0301", 17)),
                    missing_id=99,
                    unknown_id=100,
                )
            },
        ),
    )
    rows = {
        "items": [
            {
                "id": 1,
                "at": 0,
                "constant": 7.0,
                "empty": None,
                "left": "",
                "right": "NULL",
                "wide": 2**53 + 1,
            },
            {
                "id": 2,
                "at": 0,
                "constant": 7.0,
                "empty": None,
                "left": "é",
                "right": "e\u0301",
                "wide": 2**53 + 2,
            },
            {
                "id": 3,
                "at": 200,
                "constant": 99.0,
                "empty": 3.0,
                "left": None,
                "right": "FUTURE_ONLY",
                "wide": 2**53 + 1,
            },
        ]
    }
    records = [
        record("one", 1),
        record("two", 2),
        record("held", 3, at=200, split="valid"),
    ]
    with materialized(contract, rows, records, tmp_path, seed=seed) as (
        task,
        plan,
        anchors,
    ):
        before = task.fitted_state.to_dict()
        train = next(batches(task, batch_size=2)).inputs["values"]
        assert train["wide"]["values"].dtype == torch.int64
        assert train["wide"]["values"].tolist() == [2**53 + 3, 2**53 + 4]
        assert train["left"]["values"].tolist() == [7, 13]
        assert train["right"]["values"].tolist() == [11, 17]
        assert train["constant"]["values"].tolist() == [0.0, 0.0]
        assert train["empty"]["valid"].tolist() == [False, False]
        held = next(batches(task, split="valid")).inputs["values"]
        assert held["constant"]["values"].tolist() == [92.0]
        assert held["empty"]["values"].tolist() == [3.0]
        assert held["left"]["values"].tolist() == [99] and held["left"]["valid"].tolist() == [False]
        assert held["right"]["values"].tolist() == [100] and held["right"]["valid"].tolist() == [
            True
        ]
        task.to_artifact().save(tmp_path / "artifact")
        artifact = r.Artifact.load(tmp_path / "artifact")
        # Forbid fitting in the consumer when frozen state is supplied.
        with forbid_fitting():
            with plan.materialize(
                anchors, fitted=artifact.fitted_state, inference=True
            ) as restored:
                actual = next(batches(restored, split="valid")).inputs["values"]
                assert_tree(held, actual)
                assert not next(batches(restored, split="valid")).targets
        assert task.fitted_state.to_dict() == before


@pytest.mark.parametrize("layout", [r.Padded(), r.Packed()], ids=["padded", "packed"])
def test_candidate_sets_keep_empty_unknown_and_partial_labels_separate(tmp_path, layout):
    schema = r.DataSchema(
        (
            r.TableSpec("users", "id", {"id": INT}, time_column=r.NON_TEMPORAL),
            r.TableSpec("items", "id", {"id": INT, "kind": TEXT}, time_column=r.NON_TEMPORAL),
        )
    )
    candidate = r.CandidateSetSpec(
        r.EntitySpec("items"),
        values={"id": r.Field("id"), "kind": r.Field("kind")},
        targets={"clicked": r.ExternalTarget(INT)},
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(r.EntitySpec("users"), candidate_sets={"choices": candidate}),
        r.BatchSpec(
            {
                "choices": r.Input(
                    r.CandidateSource("choices"),
                    r.Fields(
                        (
                            r.Encode("id", r.Raw()),
                            r.Encode("kind", r.Vocabulary(values=(("known", 2),))),
                        )
                    ),
                    layout,
                )
            },
            targets={"clicked": r.CandidateTarget("choices", "clicked", r.Raw())},
        ),
    )
    rows = {
        "users": [{"id": 1}, {"id": 2}, {"id": 3}],
        "items": [
            {"id": 10, "kind": "known"},
            {"id": 20, "kind": None},
            {"id": 30, "kind": "unknown"},
        ],
    }
    records = [
        record("empty", 1, choices=[]),
        record(
            "mixed",
            2,
            choices=[{"id": 30, "y": 0}, {"id": 20, "y": None}, {"id": 10, "y": 1}],
        ),
        record("single", 3, choices=[{"id": 10, "y": None}]),
    ]
    with materialized(
        contract,
        rows,
        records,
        tmp_path,
        candidates={"choices": r.CandidateColumn("choices", "id", {"clicked": "y"})},
    ) as (task, _, _):
        batch = next(batches(task, batch_size=3))
        values, target = batch.inputs["choices"], batch.targets["clicked"]
        if isinstance(layout, r.Packed):
            assert values["offsets"].tolist() == [0, 0, 3, 4]
            assert values["id"]["values"].tolist() == [30, 20, 10, 10]
            assert values["kind"]["valid"].tolist() == [True, False, True, True]
            assert values["kind"]["values"].tolist() == [1, 0, 2, 2]
            assert values["valid"].tolist() == [True] * 4
            assert target["offsets"].tolist() == [0, 0, 3, 4]
            assert target["valid"].tolist() == [True, False, True, False]
            assert target["values"][[0, 2]].tolist() == [0, 1]
        else:
            assert values["valid"].tolist() == [
                [False] * 3,
                [True] * 3,
                [True, False, False],
            ]
            assert values["id"]["values"][1].tolist() == [30, 20, 10]
            assert values["id"]["values"][2, 0].item() == 10
            assert values["kind"]["values"][1].tolist() == [1, 0, 2]
            assert values["kind"]["valid"][1].tolist() == [True, False, True]
            assert target["values"][1, [0, 2]].tolist() == [0, 1]
            assert target["valid"].tolist() == [
                [False] * 3,
                [True, False, True],
                [False] * 3,
            ]
