# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Physical wide tables, distinct from derived output-width sweeps."""

import pytest

from conformance.assertions import bound_tables
from kumo.pytorch import api as r
from tests.native_fixtures import request_anchor_table


@pytest.mark.parametrize("engine", ("diskgraph",))
@pytest.mark.parametrize("width", (32, 128, 512))
@pytest.mark.parametrize("sparse", (False, True), ids=("dense-all", "null-heavy-projected"))
def test_physical_wide_rows_and_projection(engine, width, sparse):
    names = [f"field_{i:03}" for i in range(width)]
    columns = {
        "id": r.DType("int64"),
        **{name: r.DType(("int64", "float64", "string")[i % 3]) for i, name in enumerate(names)},
    }
    rows = []
    for identity in range(8):
        row = {"id": identity}
        for i, name in enumerate(names):
            row[name] = (
                None
                if sparse and (i == 0 or (identity + i) % 4 != 0)
                else identity * 1000 + i
                if i % 3 == 0
                else identity * 1000 + i + 0.25
                if i % 3 == 1
                else ("a", "b", "unseen")[identity % 3]
            )
        rows.append(row)
    selected = names[::7] if sparse else names
    assert len(rows[0]) == width + 1
    vocabulary = r.Vocabulary(values=(("a", 2), ("b", 3)))
    contract = r.DataLoaderSpec(
        r.DataSchema((r.TableSpec("items", "id", columns, time_column=r.NON_TEMPORAL),)),
        r.SampleSpec(
            r.EntitySpec("items"), entity_values={name: r.Field(name) for name in selected}
        ),
        r.BatchSpec(
            {
                "entity": r.Input(
                    r.EntitySource(),
                    r.Fields(
                        tuple(
                            r.Encode(
                                name, vocabulary if columns[name].name == "string" else r.Raw()
                            )
                            for name in selected
                        )
                    ),
                    r.Separate(),
                )
            }
        ),
    )
    requests = tuple(r.SampleRequest(f"row-{i}", i, 100) for i in range(8))
    with bound_tables(contract, {"items": rows}, engine) as (dataset, backend):
        with r.prepare(contract, dataset, backend) as plan:
            with plan.materialize(
                request_anchor_table(requests, schema=plan.contract.schema)
            ) as task:
                fingerprint = task.fitted_state.fingerprint
                seen = []
                for batch in task.batches(
                    batch_size=3,
                    output_format="model",
                ):
                    ids = list(batch.metadata["sample_ids"])
                    seen.extend(ids)
                    assert set(batch.inputs) == {"entity"}
                    assert set(batch.inputs["entity"]) == set(selected)
                    for name in selected:
                        raw = [rows[int(sid.removeprefix("row-"))][name] for sid in ids]
                        field = batch.inputs["entity"][name]
                        assert field["valid"].tolist() == [value is not None for value in raw]
                        if columns[name].name == "string":
                            expected = [
                                0 if value is None else {"a": 2, "b": 3}.get(value, 1)
                                for value in raw
                            ]
                            assert field["values"].tolist() == expected
                        else:
                            assert field["values"][field["valid"]].tolist() == [
                                value for value in raw if value is not None
                            ]
                assert seen == [row.sample_id for row in requests]
                assert task.fitted_state.fingerprint == fingerprint
