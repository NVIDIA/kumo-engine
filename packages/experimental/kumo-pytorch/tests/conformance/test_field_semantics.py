# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public lifecycle and alignment obligations that need more than one batch."""

import copy
import warnings
from dataclasses import replace
from unittest.mock import patch

from conformance.assertions import (
    assert_answer,
    assert_execution_error,
    assert_projection,
    bound_case,
)
from conformance.cases import discover_cases
from kumo.pytorch.api import (
    Columns,
    Encode,
    EntitySource,
    Fields,
    Input,
    Separate,
    Tokens,
    UseEncoding,
    Vocabulary,
    check,
    prepare,
)


def pytest_generate_tests(metafunc):
    if "engine_name" in metafunc.fixturenames:
        selected = metafunc.config.getoption("--conformance-engine")
        metafunc.parametrize(
            "engine_name", ("in_memory", "diskgraph") if selected == "all" else (selected,)
        )


def test_calendar_timing_outputs_have_only_explicit_values(engine_name):
    case = discover_cases()["C-HISTORY-002"]
    with (
        bound_case(case, engine_name) as (dataset, engine),
        prepare(case.contract, dataset, engine) as plan,
        plan.materialize(case.requests) as task,
    ):
        batch = next(task.batches(case.requests, batch_size=3))
        forbidden = {"timestamps", "elapsed_seconds", "auxiliary", "hour", "weekday", "day", "age"}
        assert not forbidden & set(batch.inputs["history"])
        assert not forbidden & set(batch.inputs["history"]["sequence"])
        assert "age" in batch.inputs["timing"]


def test_aligned_columns_follow_parent_row_removal(engine_name):
    case = discover_cases()["C-HISTORY-002"]
    parent = case.contract.batch.inputs["history"]
    parent = replace(
        parent, representation=replace(parent.representation, row=Tokens(missing="skip_event"))
    )
    contract = replace(
        case.contract,
        batch=replace(
            case.contract.batch, inputs={**case.contract.batch.inputs, "history": parent}
        ),
    )
    expected = {
        "inputs": {
            "timing": {
                "age": {
                    "values": {
                        "dtype": "float64",
                        "shape": [3, 1],
                        "values": [[60.0], [20.0], [0.0]],
                    },
                    "valid": {
                        "dtype": "bool",
                        "shape": [3, 1],
                        "values": [[True], [True], [False]],
                    },
                }
            }
        }
    }
    # Event at90 is removed because its second selected field isNULL.
    # The follower must not independently reselect it from the source rows.
    assert_answer(replace(case, contract=contract, batch=expected), engine_name)


def test_complete_reuse_never_fits_or_grows(engine_name):
    case = discover_cases()["C-IDSPACE-001"]
    with (
        bound_case(case, engine_name) as (dataset, engine),
        prepare(case.contract, dataset, engine) as plan,
        plan.materialize(case.requests) as task,
    ):
        saved = task.fitted_state
        before = copy.deepcopy(saved.to_dict())
    later_tables = copy.deepcopy(case.tables)
    for row in later_tables["users"]:
        row.update(currency="GBP", amount=1_000.0)
    later = replace(case, tables=later_tables)
    with (
        bound_case(later, engine_name) as (dataset, engine),
        prepare(case.contract, dataset, engine) as plan,
        patch(
            "kumo.pytorch.assembly.fit_encoder", side_effect=AssertionError("refitted reused state")
        ),
        plan.materialize(case.requests, fitted=saved) as task,
    ):
        batch = next(task.batches(case.requests, batch_size=3))
        assert task.fitted_state.to_dict() == before
        assert saved.to_dict() == before
        assert_projection(
            batch.inputs["spending"]["value_ids"],
            {"dtype": "int64", "shape": [3, 2], "values": [[11, 1], [11, 1], [11, 1]]},
        )
        assert_projection(
            batch.inputs["local_currency"]["currency"]["values"],
            {"dtype": "int64", "shape": [3], "values": [1, 1, 1]},
        )


def test_held_out_anchor_does_not_change_fitted_mean(engine_name):
    case = discover_cases()["C-ENCODING-002"]
    tables = copy.deepcopy(case.tables)
    tables["users"][2]["amount"] = 1_000_000.0
    requests = tuple(
        replace(request, split="train" if index < 2 else "test")
        for index, request in enumerate(case.requests)
    )
    with (
        bound_case(replace(case, tables=tables), engine_name) as (dataset, engine),
        prepare(case.contract, dataset, engine) as plan,
        plan.materialize(requests, fit_split="train") as task,
    ):
        binding = task.output("entity").field(name="amount")
        assert binding.encoder.mean == 2.0
        assert binding.encoder.scale == 1.0
        batch = next(task.batches(requests, batch_size=3))
        assert_projection(
            batch.inputs["entity"]["amount"]["values"],
            {"dtype": "float32", "shape": [3], "values": [-1.0, 1.0, 999_998.0]},
        )


def test_exposure_warning_names_overlap_and_masks_only_named_input(engine_name):
    case = discover_cases()["C-TARGET-001"]
    parent = case.contract.batch.inputs["entity"]
    shared = Vocabulary(values=(("USD", 2), ("EUR", 3)))
    parent = replace(
        parent,
        representation=replace(
            parent.representation,
            fields=(
                Encode(
                    field="currency", encoding=UseEncoding(name="currencies"), id_space="tokens"
                ),
            ),
        ),
    )
    visible = Input(
        source=EntitySource(),
        representation=Fields(
            fields=(Encode(field="currency", encoding=UseEncoding(name="currencies")),),
            row=Columns(),
        ),
        layout=Separate(),
    )
    contract = replace(
        case.contract,
        batch=replace(
            case.contract.batch,
            inputs={"entity": parent, "visible": visible},
            encodings={"currencies": shared},
        ),
    )
    with bound_case(replace(case, contract=contract), engine_name) as (dataset, engine):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert not check(contract, dataset, engine)
        assert len(caught) == 1
        assert caught[0].category.__name__ == "InputExposureWarning"
        for word in ("masked", "entity", "visible", "currency"):
            assert word in str(caught[0].message)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with (
                prepare(contract, dataset, engine) as plan,
                plan.materialize(case.requests) as task,
            ):
                with warnings.catch_warnings(record=True) as during_batches:
                    batch = next(task.batches(case.requests, batch_size=2))
                assert not during_batches
                assert_projection(
                    batch.inputs["entity"]["value_ids"],
                    {"dtype": "int64", "shape": [2, 1], "values": [[3], [3]]},
                )
                assert_projection(
                    batch.inputs["visible"]["currency"]["values"],
                    {"dtype": "int64", "shape": [2], "values": [2, 3]},
                )


def test_physical_dtype_mismatch_fails_during_binding(engine_name):
    case = discover_cases()["C-ENCODING-001"]
    tables = copy.deepcopy(case.tables)
    for row in tables["users"]:
        row["category"] = None if row["category"] is None else str(row["category"])
    bad = replace(case, tables=tables, physical_types={"users": {"category": "string"}})
    assert_execution_error(bad, engine_name, stage="bind", message_contains="category")


def test_invalid_output_mode_fails_at_batch_setup(engine_name):
    case = discover_cases()["C-TYPE-001"]
    assert_execution_error(
        case,
        engine_name,
        stage="build",
        message_contains="output_format",
        output_format="undefined",
    )
