# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full populations retain every anchor, including tails and invalid targets."""

from dataclasses import replace
from test.kumo_pytorch import nightly
from test.kumo_pytorch.bindings import binding_for
from test.kumo_pytorch.population_anchors import project_binding

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.pytorch.api as r

from .test_run_aml import mapped  # noqa: F401 - pytest fixture
from .test_run_aml import CASES


def source_anchors(selections, selection, tmp_path, *, repeats=3):
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    groups = selections[selection["selector"]["anchor_version"]][
        "split" if native else "forecast_split"
    ]
    rows = [
        dict(row, sample_id=f"{row['sample_id']}-{i}" if repeats > 1 else row["sample_id"])
        for group in groups.values()
        for row in group
        for i in range(repeats)
    ]
    table = pa.Table.from_pylist(rows)
    table = table.set_column(
        table.schema.get_field_index("involvement"),
        "involvement",
        table["involvement"].cast(pa.int32()),
    )
    for clock in ("as_of_time", "graph_as_of_time"):
        table = table.set_column(
            table.schema.get_field_index(clock),
            clock,
            table[clock].cast(pa.timestamp("ns")),
        )
    path = tmp_path / "source-anchors.parquet"
    pq.write_table(table, path)
    return path


def projected_case(mapped, selection, family, directory, *, limit=None, repeats=3):
    """Project fixture anchors; execution belongs to the shared nightly runner."""
    directory.mkdir(parents=True)
    dataset, identities, selections = mapped
    binding = binding_for(selection, family)
    source = source_anchors(selections, selection, directory, repeats=repeats)
    anchors, manifest = project_binding(
        source, directory / "anchors.parquet", identities, binding, limit=limit
    )
    return dataset, anchors, manifest, binding


def test_preflight_refusal_does_not_claim_population(mapped, tmp_path, monkeypatch):
    original = nightly.contract_for

    def filtered(*args, **kwargs):
        contract = original(*args, **kwargs)
        entity = replace(
            contract.sample.entity,
            filters=(r.RowFilter("account_id", "greater_than", 0),),
        )
        return replace(contract, sample=replace(contract.sample, entity=entity))

    monkeypatch.setattr(nightly, "contract_for", filtered)
    selection, family = CASES[0]
    directory = tmp_path / "refused-population"
    result = nightly.run_case(
        *projected_case(mapped, selection, family, directory),
        directory,
        capacity=5,
        mode="full",
        mapping={"scope": {"kind": "full_source"}},
    )
    assert result["status"] == "passed" and result["outcome"] == "refused"
    assert result["full_population_verified"] is False
    assert set(result["splits"]) == {"train", "valid", "test"}
    assert all(
        s["delivered"] == 0 and s["status"] == "not_executed"
        for s in result["splits"].values()
    )


@pytest.mark.parametrize(
    "selection,family",
    CASES,
    ids=[f"{s['selector']['workload_id']}-{f}" for s, f in CASES],
)
@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_raw_aml_population_completion(mapped, selection, family, mode, tmp_path):
    directory = tmp_path / "case"
    result = nightly.run_case(
        *projected_case(
            mapped, selection, family, directory, limit=1 if mode == "smoke" else None
        ),
        directory,
        capacity=5,
        mode=mode,
        mapping={"scope": {"kind": "full_source"}},
        batch_size=2,
    )
    assert result["status"] == "passed", result.get("traceback", result)
    if result.get("outcome") == "refused":
        assert result["full_population_verified"] is False
        assert all(s["status"] == "not_executed" for s in result["splits"].values())
        return
    assert result["full_population_verified"] == (mode == "full")
    native = selection["selector"]["workload_id"] != "account-involvement-next24h"
    groups = mapped[2][selection["selector"]["anchor_version"]][
        "split" if native else "forecast_split"
    ]
    for name, split in result["splits"].items():
        expected = len(groups[name]) * 3 if mode == "full" else 1
        assert split["delivered"] == split["expected"] == expected
        assert split["batches"] == (expected + 1) // 2
        assert split["status"] == "passed"
