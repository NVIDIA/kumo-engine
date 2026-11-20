# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-derived masked account histories, including variable-length BPE fields."""

import hashlib
import json
from test.kumo_pytorch.bindings import contract_for, dataset_definition
from test.kumo_pytorch.data import schema_for
from test.kumo_pytorch.history_features import masked_features
from test.kumo_pytorch.nightly import run_case

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import kumo.pytorch.api as r

from .test_relational_history import STAMP, inputs, prepared


@pytest.mark.parametrize("mode", ["smoke", "training", "full", "lifecycle"])
def test_account_bpe_binding_campaigns(tmp_path, mode):
    binding, output, (dataset, anchors, population, mapping, capacity) = prepared(
        tmp_path, binding_ref="S01"
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode=mode,
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert report["status"] == "passed", report
    if report.get("outcome") != "refused" and mode in {"smoke", "training"}:
        assert report["feature_coverage"]["F09"] == {"train": 3, "valid": 1, "test": 1}
        assert report["feature_assertions"]["bpe_pieces"]["status"] == "passed"


@pytest.fixture
def masked_case(tmp_path):
    binding, output, (dataset, anchors, population, mapping, capacity) = prepared(
        tmp_path, binding_ref="S01"
    )
    contract = contract_for(schema_for(dataset, binding), binding, max_events=capacity)
    assert contract.batch.inputs["history"].representation.row.max_tokens is None
    with r.prepare(
        contract,
        dataset,
        r.DiskGraphEngine(index=output / "probe-index", num_threads=1),
    ) as plan:
        with plan.materialize(
            anchors,
            fit_split="train",
            encoder_cutoff=population["encoder_cutoff_seconds"],
        ) as task:
            batch = next(
                iter(
                    task.batches(
                        split="train",
                        epoch=0,
                        seed=7,
                        shuffle=False,
                        batch_size=2,
                        output_format="model",
                    )
                )
            )
            state = task.fitted_state.to_dict()
            features = masked_features(
                binding, output / "anchors.parquet", mapping, state, mode="smoke"
            )
            features(batch)
            yield batch, features, state


def test_account_bpe_uses_multiple_pieces_and_one_field_identity(masked_case):
    batch, _, state = masked_case
    history = batch.inputs["history"]
    member = dict(state["spaces"]["history"]["members"]['["fields","fields"]'])
    field = next(
        member[index]
        for path, index in state["field_names"]["fields"]
        if path == ["events", "disputes", "reason"]
    )
    selected = history["field_ids"] == field
    assert history["position_ids"][selected].tolist() == [0, 1, 0, 1]
    # With the declared ByteLevel preprocessing, these are distinct words even
    # when training merges every character within each word.
    encoder = state["encoders"]['["inputs","history","fields","reason"]']
    assert encoder["kind"] == "text"
    assert "unknown" in json.loads(encoder["tokenizer_json"])["model"]["vocab"]


@pytest.mark.parametrize(
    "row,stamp,changes_state",
    [
        (2, STAMP - 1, True),
        (2, STAMP, False),
        (3, STAMP + 1, False),
        (4, STAMP - 10, True),
    ],
)
def test_text_fitting_obeys_table_cutoff_including_held_out_roots(
    tmp_path, row, stamp, changes_state
):
    def fitted(root, files, binding, output):
        from test.kumo_pytorch.data import prepare_binding

        output.mkdir()
        dataset, anchors, population, _, capacity = prepare_binding(
            root, output, binding, files, limit=None
        )
        assert population["encoder_cutoff_seconds"] == STAMP - 1
        contract = contract_for(
            schema_for(dataset, binding), binding, max_events=capacity
        )
        with r.prepare(
            contract, dataset, r.DiskGraphEngine(index=output / "index", num_threads=1)
        ) as plan:
            with plan.materialize(
                anchors,
                fit_split="train",
                encoder_cutoff=population["encoder_cutoff_seconds"],
            ) as task:
                return task.fitted_state.to_dict()["encoders"]

    root, files, binding = inputs(tmp_path, binding_ref="S01")
    key = next(key for key in files if "/disputes/" in key)
    table = pq.read_table(root / key).to_pydict()

    def write():
        pq.write_table(pa.table(table), root / key)
        files[key].update(
            bytes=(root / key).stat().st_size,
            sha256=hashlib.sha256((root / key).read_bytes()).hexdigest(),
        )

    clock = dataset_definition(binding)["source_options"]["events"]["disputes"]["time"]
    table[clock][row] = stamp
    write()
    before = fitted(root, files, binding, tmp_path / "before")
    # Stored-column encoders fit all table rows at/before encoder_cutoff,
    # including held-out roots. Strict history eligibility is a separate rule.
    table["reason"][row] = "distinct Ω text"
    write()
    after = fitted(root, files, binding, tmp_path / "after")
    key = '["inputs","history","fields","reason"]'
    assert (before.pop(key) != after.pop(key)) == changes_state
    assert before == after


@pytest.mark.parametrize("text", [None, "", "[UNK]", "[MISSING] unknown purchase"])
def test_masked_text_null_empty_and_literal_special_tokens(tmp_path, text):
    from test.kumo_pytorch.data import prepare_binding

    root, files, binding = inputs(tmp_path, binding_ref="S01")
    key = next(key for key in files if "/disputes/" in key)
    table = pq.read_table(root / key)
    table = table.set_column(
        table.schema.get_field_index("reason"),
        "reason",
        pa.array([text] * len(table), pa.string()),
    )
    pq.write_table(table, root / key)
    files[key].update(
        bytes=(root / key).stat().st_size,
        sha256=hashlib.sha256((root / key).read_bytes()).hexdigest(),
    )
    output = tmp_path / "output"
    output.mkdir()
    dataset, anchors, population, mapping, capacity = prepare_binding(
        root, output, binding, files, limit=None
    )
    report = run_case(
        dataset,
        anchors,
        population,
        binding,
        output,
        mode="smoke",
        capacity=capacity,
        mapping=mapping,
        batch_size=2,
    )
    assert report["status"] == "passed", report
