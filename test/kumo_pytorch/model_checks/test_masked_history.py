# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-derived masked account histories, including variable-length BPE fields."""

import hashlib
import json
from test.kumo_pytorch.bindings import contract_for, dataset_definition
from test.kumo_pytorch.data import schema_for
from test.kumo_pytorch.history_features import _matching, masked_features
from test.kumo_pytorch.nightly import run_case
from test.kumo_pytorch.relational_sources import prepare_sources

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import kumo.pytorch.api as r

from .test_relational_history import STAMP, inputs, prepared


@pytest.mark.parametrize("kind", ["large", "ambiguous", "duplicate"])
def test_tied_record_matching_is_bijective_without_a_recursion_limit(kind):
    candidates = (
        [list(range(1200)) for _ in range(1200)]
        if kind == "large"
        else [[0, 1], [0, 2], [0, 2]]
        if kind == "ambiguous"
        else [[0], [0]]
    )
    if kind == "duplicate":
        with pytest.raises(AssertionError, match="cannot match"):
            _matching(candidates)
    else:
        matched = _matching(candidates)
        assert set(matched) == set(matched.values()) == set(range(len(candidates)))
        assert all(
            expected in candidates[actual] for actual, expected in matched.items()
        )


def test_account_sources_exclude_sibling_finances_and_include_parent_events(tmp_path):
    binding, _, (dataset, _, _, mapping, _) = prepared(tmp_path, binding_ref="S01")
    cards = pq.read_table(dataset.tables["card_txns"].path).to_pylist()
    assert {(row["root_id"], row["source_id"]) for row in cards} == {
        (100, 1),
        (101, 2),
        (100, 3),
        (101, 4),
        (200, 5),
    }
    transfers = pq.read_table(dataset.tables["transfers"].path).to_pylist()
    assert {
        (row["root_id"], row["source_id"], row["direction"]) for row in transfers
    } == {
        (100, 1, "outgoing"),
        (101, 1, "incoming"),
        (101, 2, "outgoing"),
        (200, 2, "incoming"),
        (100, 3, "outgoing"),
        (200, 3, "incoming"),
        (100, 4, "incoming"),
        (200, 5, "outgoing"),
    }
    logins = pq.read_table(dataset.tables["logins"].path).to_pylist()
    assert {(row["root_id"], row["source_id"]) for row in logins} == {
        *((account, event) for account in (100, 101) for event in (1, 2, 3, 4)),
        (200, 5),
    }
    assert len({row["event_id"] for row in logins}) == len(logins)
    assert {row["event_time_seconds"] for row in logins} == {
        STAMP - 30,
        STAMP - 20,
        STAMP - 10,
        STAMP,
        STAMP + 1,
    }
    assert mapping["scope"]["complete_selected_root_histories"]
    assert dataset_definition(binding)["identity_column"] == "account_id"


@pytest.mark.parametrize("account", [100, 101, 200])
def test_account_mapping_ids_stay_stable_when_selecting_roots(tmp_path, account):
    root, files, binding = inputs(tmp_path, binding_ref="S01")
    full, _ = prepare_sources(
        root, tmp_path / "full", dataset_definition(binding), files, None
    )
    bounded, _ = prepare_sources(
        root, tmp_path / "bounded", dataset_definition(binding), files, {str(account)}
    )
    for name in full.tables:
        expected = [
            row
            for row in pq.read_table(full.tables[name].path).to_pylist()
            if row["root_id"] == account
        ]
        actual = pq.read_table(bounded.tables[name].path).to_pylist()
        key = "root_id" if name == "roots" else "event_id"
        assert sorted(actual, key=lambda r: r[key]) == sorted(
            expected, key=lambda r: r[key]
        )


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
    "fault",
    [
        "value",
        "target",
        "validity",
        "corruption",
        "age",
        "calendar",
        "field",
        "piece-position",
        "offset",
        "owner",
        "sequence",
        "identity",
        "cutoff",
        "dtype",
        "derived-dtype",
        "derived-shape",
        "derived-padding",
    ],
)
def test_masked_history_assertions_detect_corruption(masked_case, fault):
    batch, features, _ = masked_case
    history, target = batch.inputs["history"], batch.targets["masked_values"]
    if fault == "value":
        history["value_ids"][0] = 999999
    elif fault == "target":
        target["values"][target["valid"]] += 1
    elif fault == "validity":
        target["valid"][0] = ~target["valid"][0]
    elif fault == "corruption":
        target["corruption"][0] = ~target["corruption"][0]
    elif fault == "age":
        batch.inputs["history_timing"]["age_seconds"]["values"][0, 0] += 1
    elif fault == "calendar":
        batch.inputs["history_calendar"]["hour"]["values"][0, 0] += 1
    elif fault == "field":
        history["field_ids"][0] = 999999
    elif fault == "piece-position":
        history["position_ids"][history["position_ids"] == 1] = 0
    elif fault == "offset":
        history["field_offsets"][1] += 1
    elif fault == "owner":
        history["owner"][0] = 1
    elif fault == "sequence":
        history["sequence"]["event_indices"][0, 0] += 1
    elif fault == "identity":
        batch.metadata["entity_ids"][0] = 200
    elif fault == "cutoff":
        batch.metadata["as_of_time"][0] += 1
    elif fault.startswith("derived-"):
        field = batch.inputs["history_calendar"]["hour"]
        if fault == "derived-dtype":
            field["values"] = field["values"].float()
        elif fault == "derived-shape":
            field["values"] = torch.cat(
                (field["values"], field["values"][:, :1]), dim=1
            )
        else:
            assert not field["valid"].all(), "Fixture must include event padding"
            field["valid"] = torch.ones_like(field["valid"])
    else:
        history["position_ids"] = history["position_ids"].float()
    with pytest.raises(AssertionError, match="Feature assertions failed"):
        features(batch)


def reorder_events(batch, order):
    history, target = batch.inputs["history"], batch.targets["masked_values"]
    offsets = history["field_offsets"].tolist()
    indices = [i for event in order for i in range(offsets[event], offsets[event + 1])]
    lengths = [offsets[event + 1] - offsets[event] for event in order]
    for name in ("field_ids", "value_ids", "position_ids", "valid", "field_valid"):
        history[name] = history[name][indices]
    for name in ("values", "valid", "corruption"):
        target[name] = target[name][indices]
    history["field_offsets"] = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()]
    )
    if "offsets" in target:
        target["offsets"] = history["field_offsets"].clone()


@pytest.mark.parametrize("change", ["tie", "cross-time", "duplicate"])
def test_masked_records_allow_ties_but_preserve_time_and_multiplicity(
    masked_case, change
):
    batch, features, _ = masked_case
    order = list(range(len(batch.inputs["history"]["owner"])))
    if change == "duplicate":
        order[0] = 1
    else:
        other = 1 if change == "tie" else 7
        order[0], order[other] = order[other], order[0]
    reorder_events(batch, order)
    if change == "tie":
        features(batch)
    else:
        with pytest.raises(AssertionError, match="Feature assertions failed"):
            features(batch)


def test_missing_bpe_piece_cannot_pass_with_consistent_offsets(masked_case):
    batch, features, _ = masked_case
    history, target = batch.inputs["history"], batch.targets["masked_values"]
    drop = int((history["position_ids"] == 1).nonzero()[0])
    keep = [i for i in range(history["value_ids"].numel()) if i != drop]
    for name in ("field_ids", "value_ids", "position_ids", "valid", "field_valid"):
        history[name] = history[name][keep]
    for name in ("values", "valid", "corruption"):
        target[name] = target[name][keep]
    history["field_offsets"][history["field_offsets"] > drop] -= 1
    if "offsets" in target:
        target["offsets"] = history["field_offsets"].clone()
    with pytest.raises(AssertionError, match="Feature assertions failed"):
        features(batch)


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


def test_bpe_merge_oracle_matches_hand_calculated_pieces():
    from test.kumo_pytorch.reference_tokens import encoded_pieces

    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(
        models.BPE(
            vocab={"[MISSING]": 0, "[UNK]": 1, "a": 2, "b": 3, "ab": 4, "abab": 5},
            merges=[("a", "b"), ("ab", "ab")],
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    tokenizer.add_special_tokens(["[MISSING]", "[UNK]"])
    key = '["inputs","history","fields","text"]'
    state = {
        "encoders": {
            key: {
                "kind": "text",
                "roles": {"missing": 0, "unknown": 1},
                "tokenizer_json": tokenizer.to_str(),
            }
        },
        "spaces": {"history": {"members": {key: [[i, i + 10] for i in range(6)]}}},
    }
    assert encoded_pieces(state, "text", "ababa") == [15, 12]
    assert encoded_pieces(state, "text", "abaX") == [14, 12, 11]
    assert encoded_pieces(state, "text", "") == []
    assert encoded_pieces(state, "text", None) == [10]
    assert encoded_pieces(state, "text", "[UNK]") == [11]
    assert encoded_pieces(state, "text", "[MISSING]") == [10]
    assert encoded_pieces(state, "text", "ab[UNK]a") == [14, 11, 12]


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
