# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Documented package bugs remain ordinary failures.

Evidence: conformance/documented-regressions.md relative to the package root.
"""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import kumo.pytorch.api as r
from conformance.refusals import forbid_fitting
from tests import graph_semantics
from tests.conformance.fixtures import (
    FLOAT,
    INT,
    TEXT,
    TIME,
    assert_tree,
    batches,
    fields,
    materialized,
    record,
)
from tests.conformance.test_token_interactions import token_fixture, verify_tokens


@pytest.mark.parametrize("reload", [False, True], ids=["same-task-control", "artifact-reload"])
def test_artifact_reload_preserves_iterator_identity(tmp_path, reload):
    """Saved iterators resume after reloading an artifact.

    Expected: Reloading identical anchors and fitted state preserves task identity
    and emits exactly the two remaining samples, without refitting.

    Evidence: G-ARTIFACT-001;
    conformance/documented-regressions.md#db01--artifact-identity-and-iterator-continuation
    """
    schema = r.DataSchema(
        (r.TableSpec("users", "id", {"id": INT, "at": TIME, "x": FLOAT}, time_column="at"),)
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(r.EntitySpec("users"), entity_values={"x": r.Field("x")}),
        r.BatchSpec(
            {"values": r.Input(r.EntitySource(), fields("x"), r.Separate())},
            batching=r.Batching(max_samples=2),
        ),
    )
    rows = {"users": [{"id": i, "at": 0, "x": float(i)} for i in range(4)]}
    records = [record(str(i), i) for i in range(4)]
    with materialized(contract, rows, records, tmp_path) as (task, plan, table):
        iterator = batches(task)
        first = next(iterator)
        assert list(first.metadata["sample_ids"]) == ["0", "1"]
        assert first.inputs["values"]["x"]["values"].tolist() == [0.0, 1.0]
        expected = list(iterator)
        assert len(expected) == 1
        saved_iterator = batches(task)
        next(saved_iterator)
        torch.save(saved_iterator.state_dict(), tmp_path / "iterator.pt")
        task.to_artifact().save(tmp_path / "artifact")
        artifact = r.Artifact.load(tmp_path / "artifact")
        with (
            forbid_fitting(),
            (
                plan.materialize(table, fitted=artifact.fitted_state)
                if reload
                else nullcontext(task)
            ) as restored,
        ):
            resumed = batches(restored)
            resumed.load_state_dict(torch.load(tmp_path / "iterator.pt", weights_only=True))
            remaining = list(resumed)
            assert restored.fingerprints == task.fingerprints
            assert len(remaining) == 1
            assert list(remaining[0].metadata["sample_ids"]) == ["2", "3"]
            assert remaining[0].inputs["values"]["x"]["values"].tolist() == [2.0, 3.0]
            assert_tree(remaining[0].inputs, expected[0].inputs)
            assert_tree(remaining[0].metadata, expected[0].metadata)


@pytest.mark.parametrize("layout", [r.Packed(), r.Padded()], ids=["packed", "padded"])
@pytest.mark.parametrize("budget", [6, 5], ids=["spare-slot-control", "exact-five-slots"])
def test_next_token_budget_counts_emitted_input_slots(tmp_path, layout, budget):
    """Next-token batches fit a budget equal to their emitted input length.

    Expected: The shifted sequence contains five input tokens and fits a budget
    of five or six, in both packed and padded layouts.

    Evidence: Batching.max_tokens and G-TARGET-001;
    conformance/documented-regressions.md#db02--next-token-budget-after-shifting
    """
    contract, rows, records, expected = token_fixture(layout, "next")
    contract = replace(
        contract,
        batch=replace(contract.batch, batching=replace(contract.batch.batching, max_tokens=budget)),
    )
    with materialized(contract, rows, records[:1], tmp_path) as (task, _, _):
        actual = list(batches(task))
        assert len(actual) == 1
        assert list(actual[0].metadata["sample_ids"]) == [records[0]["sample"]]
        verify_tokens(actual[0], expected, "next")
        assert actual[0].inputs["history"]["value_ids"].numel() == 5 <= budget


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0.0, 2.0**-499], [-1.0, 1.0]),
        ([0.0, 2.0**-599], [-1.0, 1.0]),
        ([2.0**600, 2.0**600], [0.0, 0.0]),
    ],
    ids=["small-scale-control", "nonconstant-tiny-scale", "large-finite-constant"],
)
def test_numerical_fitting_preserves_representable_normalization(tmp_path, values, expected):
    """Numerical fitting handles finite values at very small and large scales.

    Expected: The two distinct inputs normalize to [-1, 1]; identical finite
    inputs normalize to [0, 0] using scale one, without a fitting error.

    Evidence: G-ENCODING-001;
    conformance/documented-regressions.md#db03--population-normalization-on-finite-binary64-inputs
    """
    schema = r.DataSchema(
        (r.TableSpec("items", "id", {"id": INT, "at": TIME, "value": FLOAT}, time_column="at"),)
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(r.EntitySpec("items"), entity_values={"value": r.Field("value")}),
        r.BatchSpec(
            {
                "values": r.Input(
                    r.EntitySource(),
                    r.Fields((r.Encode("value", r.ZScore(dtype="float64")),)),
                    r.Separate(),
                )
            }
        ),
    )
    rows = {"items": [{"id": i, "at": 0, "value": value} for i, value in enumerate(values)]}
    with materialized(contract, rows, [record(str(i), i) for i in range(2)], tmp_path) as (
        task,
        _,
        _,
    ):
        batch = next(batches(task))
        result = batch.inputs["values"]["value"]
        assert result["values"].dtype == torch.float64
        assert result["valid"].tolist() == [True, True]
        # Powers of two make every expected mean, scale and ratio exactly representable.
        assert result["values"].tolist() == expected


@pytest.mark.parametrize(
    "identity",
    [2**63 - 1, 2**63, 2**64 - 1],
    ids=["signed-boundary-control", "unsigned-high-bit", "unsigned-maximum"],
)
def test_masking_supports_declared_uint64_entity_keys(tmp_path, identity):
    """Masking supports the full declared range of unsigned entity keys.

    Expected: Each uint64 key emits the mask token [3], original label [4] and
    valid supervision, including keys above the signed integer maximum.

    Evidence: DType.uint64 and G-TARGET-001;
    conformance/documented-regressions.md#db04--uint64-identities-with-masking
    """
    dtype = r.DType("uint64")
    schema = r.DataSchema(
        (
            r.TableSpec(
                "items",
                "id",
                {"id": dtype, "at": TIME, "value": TEXT},
                time_column="at",
            ),
        )
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(r.EntitySpec("items"), entity_values={"value": r.Field("value")}),
        r.BatchSpec(
            {
                "values": r.Input(
                    r.EntitySource(),
                    r.Fields(
                        (r.Encode("value", r.Vocabulary(values=(("x", 2),)), "tokens"),),
                        row=r.Tokens(missing="emit"),
                    ),
                    r.Padded(),
                )
            },
            id_spaces={
                "tokens": r.IdSpace({role: role for role in ("pad", "unknown", "missing", "mask")})
            },
            targets={
                "masked": r.MaskedValuesTarget(
                    "values",
                    token_probability=1.0,
                    event_probability=0.0,
                    key_probability=0.0,
                )
            },
        ),
    )
    with materialized(
        contract,
        {"items": [{"id": identity, "at": 0, "value": "x"}]},
        [record("one", identity)],
        tmp_path,
    ) as (task, _, _):
        batch = next(batches(task))
        assert list(batch.metadata["sample_ids"]) == ["one"]
        assert batch.inputs["values"]["value_ids"].tolist() == [[3]]
        assert batch.inputs["values"]["valid"].tolist() == [[True]]
        assert batch.targets["masked"]["values"].tolist() == [[4]]
        assert batch.targets["masked"]["valid"].tolist() == [[True]]
        assert batch.targets["masked"]["corruption"].tolist() == [[True]]


@pytest.mark.parametrize("unit,ticks", [("us", 10**6), ("ns", 10**9)])
@pytest.mark.parametrize("days", [104, 105])
def test_event_age_survives_long_history_at_fine_clock_resolution(tmp_path, unit, ticks, days):
    """Event ages remain valid across long histories and fine clock resolutions.

    Expected: A 104-day or 105-day interval returns its exact age in seconds
    (8,985,600 or 9,072,000), for microsecond and nanosecond source clocks.

    Evidence: EventTimeDelta and DType;
    conformance/documented-regressions.md#db05--event-ages-at-nanosecond-resolution
    """
    time = r.DType("timestamp", unit=unit, timezone="UTC")
    schema = r.DataSchema(
        (
            r.TableSpec("users", "id", {"id": INT}, time_column=r.NON_TEMPORAL),
            r.TableSpec("events", "id", {"id": INT, "owner": INT, "at": time}, time_column="at"),
        ),
        (r.RelationSpec("events", "users", "id", "events", "owner"),),
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(
            r.EntitySpec("users"),
            events={
                "history": r.EventSpec(
                    r.Rows(("events",), r.TimeWindow("at")),
                    {"age": r.EventTimeDelta("seconds", dtype=FLOAT)},
                )
            },
        ),
        r.BatchSpec({"history": r.Input(r.EventSource(("history",)), fields("age"), r.Packed())}),
    )
    cutoff, elapsed = 20_000_000, days * 86_400
    rows = {
        "users": [{"id": 1}],
        "events": [{"id": 1, "owner": 1, "at": (cutoff - elapsed) * ticks}],
    }
    with materialized(contract, rows, [record("one", 1, at=cutoff)], tmp_path) as (
        task,
        _,
        _,
    ):
        batch = next(batches(task))
        history = batch.inputs["history"]
        assert history["offsets"].tolist() == [0, 1]
        assert history["age"]["valid"].tolist() == [True]
        assert history["age"]["values"].dtype == torch.float64
        assert history["age"]["values"].tolist() == [float(elapsed)]


@pytest.mark.parametrize("fanout", [1, 3], ids=["same-fanout-control", "expand-revisited-root"])
def test_later_graph_layer_reexpands_visited_nodes(tmp_path, fanout):
    """Later graph layers expand a previously visited node with their own fanout.

    Expected: Returning to the root with fanout three includes all three eligible
    events and their edges; fanout one keeps the latest. Output nodes are unique.

    Evidence: G-GRAPH-001;
    conformance/documented-regressions.md#db06--expanding-a-revisited-graph-node
    """
    schema = r.DataSchema(
        (
            r.TableSpec("users", "id", {"id": INT, "at": TIME}, time_column="at"),
            r.TableSpec("events", "id", {"id": INT, "owner": INT, "at": TIME}, time_column="at"),
        ),
        (r.RelationSpec("events", "users", "id", "events", "owner"),),
    )
    layers = (
        r.GraphLayer((r.GraphSample("events", 1),)),
        r.GraphLayer((r.GraphSample("events", 1, direction="reverse"),)),
        r.GraphLayer((r.GraphSample("events", fanout),)),
    )
    contract = r.DataLoaderSpec(
        schema,
        r.SampleSpec(r.EntitySpec("users"), neighborhoods={"graph": r.NeighborhoodSpec(layers)}),
        r.BatchSpec(
            {
                "graph": r.Input(
                    r.NeighborhoodSource("graph"),
                    r.Graph({"events": fields("id")}),
                    None,
                )
            }
        ),
    )
    rows = {
        "users": [{"id": 1, "at": 0}],
        "events": [{"id": i, "owner": 1, "at": 10 * i} for i in (1, 2, 3)],
    }
    with materialized(contract, rows, [record("one", 1)], tmp_path) as (task, _, _):
        batch = next(batches(task))
        graph = batch.inputs["graph"]
        nodes = graph["nodes"]["events"]
        wanted = {3} if fanout == 1 else {1, 2, 3}
        assert set(nodes["ids"].tolist()) == wanted, (
            "Later layer omitted eligible neighbors of a revisited node"
        )
        assert len(nodes["ids"]) == len(wanted)
        assert nodes["batch"].tolist() == [0] * len(wanted)
        oracle = graph_semantics
        oracle.assert_graph(
            graph,
            contract,
            "graph",
            [SimpleNamespace(sample_id="one", entity_id=1, as_of_time=100)],
            oracle.SourceRows(schema, rows),
            task.fitted_state.to_dict(),
        )


@pytest.mark.parametrize("constant", ["red", None], ids=["text-control", "typed-null"])
@pytest.mark.parametrize(
    "planned", [False, True], ids=["no-budget-control", "native-length-budget"]
)
def test_native_length_planning_handles_typed_null_text_constants(tmp_path, constant, planned):
    """Native length planning supports typed NULL text constants.

    Expected: Three constant values emit five tokens including BOS/EOS, fit the
    budget of twelve, and retain the correct masked labels and validity.

    Evidence: Constant, Tokens.missing and G-TARGET-001;
    conformance/documented-regressions.md#db07--typed-null-text-constants-under-length-planning
    """
    contract, rows, records, _ = token_fixture(r.Packed(), "masked")
    event = replace(contract.sample.events["events"], fields={"text": r.Constant(constant, TEXT)})
    policy = contract.batch.batching if planned else r.Batching(max_samples=3)
    contract = replace(
        contract,
        sample=replace(contract.sample, events={"events": event}),
        batch=replace(contract.batch, batching=policy),
    )
    with materialized(contract, rows, records[:1], tmp_path) as (task, _, _):
        actual = list(batches(task))
        assert len(actual) == 1
        batch = actual[0]
        assert batch.inputs["history"]["value_ids"].tolist() == [4, 3, 3, 3, 5]
        assert batch.inputs["history"]["valid"].tolist() == [True] * 5
        assert batch.inputs["history"]["offsets"].tolist() == [0, 5]
        target = batch.targets["answer"]
        assert target["values"].tolist() == [-100, *([13 if constant else 2] * 3), -100]
        assert target["valid"].tolist() == [False, True, True, True, False]
        assert target["corruption"].tolist() == target["valid"].tolist()


@pytest.mark.parametrize("copies", [1, 2], ids=["one-field-control", "two-named-fields-review"])
def test_ambiguous_two_named_fields_reading_one_text_column(tmp_path, copies):
    """CONTRACT AMBIGUITY A01: two named fields read the same text column.

    Expected: Proposed: emit both named fields in declaration order. Alternative:
    reject source aliases before execution. Reviewers must settle the contract.

    Evidence: G-FIELDS-001 and EventSpec.fields;
    conformance/documented-regressions.md#ambiguous-a01
    """
    contract, rows, records, _ = token_fixture(r.Packed(), "masked")
    encoding = contract.batch.inputs["history"].representation.fields[0].encoding
    names = ("first", "second")[:copies]
    event = replace(
        contract.sample.events["events"],
        fields={name: r.Field("text") for name in names},
    )
    representation = r.Fields(
        tuple(r.Encode(name, r.UseEncoding("words"), "tokens") for name in names),
        row=r.Tokens(missing="emit"),
        history=r.Flatten(8, before="bos", after="eos"),
    )
    contract = replace(
        contract,
        sample=replace(contract.sample, events={"events": event}),
        batch=replace(
            contract.batch,
            inputs={"history": r.Input(r.EventSource(("events",)), representation, r.Packed())},
            encodings={"words": encoding},
            targets={},
        ),
    )
    rows = {
        "users": [{"id": 1}],
        "events": [{"id": 1, "owner": 1, "at": 90, "text": "red bike"}],
    }
    with materialized(contract, rows, [record("one", 1)], tmp_path) as (task, _, _):
        message = (
            "CONTRACT AMBIGUITY A01: "
            "conformance/documented-regressions.md#ambiguous-a01. "
            "G-FIELDS-001 forbids duplicate resolved values; EventSpec allows named fields. "
            "Proposed: emit both aliases in declaration order; "
            "alternative: reject before execution."
        )
        try:
            actual = list(batches(task))
        except Exception as error:
            if copies == 2:
                raise AssertionError(
                    f"{message} Observed {type(error).__name__}: {error}"
                ) from error
            raise
        assert len(actual) == 1, message
        output = actual[0].inputs["history"]
        assert output["value_ids"].tolist() == [4, *([13, 16] * copies), 5], message
        assert output["valid"].tolist() == [True] * (2 + 2 * copies), message
