# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace

import pyarrow as pa
import pytest

from kumo.pytorch import api, encoding
from kumo.pytorch.encoding import fit_encoder, train_text
from tests.test_column_stats import LATEST, _column, _contract, _requests, _setup

pytest.importorskip("diskgraph")

TEXT = api.Text(vocabulary_size=300)


def _text_contract() -> api.DataLoaderSpec:
    contract = _contract()
    transactions = contract.batch.inputs["transactions"]
    fields = (
        *transactions.representation.fields,
        api.Encode(field="description", encoding=TEXT),
    )
    events = contract.sample.events["transactions"]
    sample = replace(
        contract.sample,
        events={
            "transactions": replace(
                events, fields={**events.fields, "description": api.Field(column="description")}
            )
        },
    )
    inputs = {
        **contract.batch.inputs,
        "transactions": replace(
            transactions, representation=replace(transactions.representation, fields=fields)
        ),
    }
    return replace(contract, sample=sample, batch=replace(contract.batch, inputs=inputs))


def _tokenizer(task: api.Task) -> str:
    return task.output("transactions").field(name="description").encoder.tokenizer_json


@pytest.mark.parametrize("engine_name", ("in_memory", "diskgraph"))
def test_a_stored_text_column_trains_on_its_rows_up_to_the_cutoff(tmp_path, engine_name):
    dataset, engine = _setup(engine_name, tmp_path)
    expected = train_text(TEXT, [_column("card_transactions", "description", LATEST)])
    published = []
    for _ in range(2):
        with api.prepare(_text_contract(), dataset, engine) as plan:
            with plan.materialize(_requests(), fit_split="train") as task:
                assert _tokenizer(task) == expected
                statistics = task.fitted_state.fit["inputs"]["statistics"]
        texts = tmp_path.glob("index/col_stats/text-*/tokenizer.json")
        published.append({path: path.stat().st_mtime_ns for path in texts})
    if engine_name == "diskgraph":
        assert any(name.startswith("text-") for name in statistics["col_stats"])
        assert published[0] == published[1]
        (path,) = published[0]
        assert json.loads(path.read_text())["tokenizer"] == expected


def test_both_engines_and_every_thread_count_train_the_same_tokenizer(tmp_path):
    tokenizers = set()
    for name, threads in (("in_memory", None), ("diskgraph", 1), ("diskgraph", 4)):
        dataset, engine = _setup(name, tmp_path / f"{name}-{threads}")
        if threads:
            engine = api.DiskGraphEngine(index=engine.index, num_threads=threads)
        with api.prepare(_text_contract(), dataset, engine) as plan:
            with plan.materialize(_requests(), fit_split="train", encoder_cutoff=None) as task:
                tokenizers.add(_tokenizer(task))
    assert len(tokenizers) == 1


def test_chunked_encoding_matches_one_string_at_a_time(monkeypatch):
    monkeypatch.setattr(encoding, "TEXT_BATCH", 3)
    fitted = fit_encoder("e", TEXT, api.DType("string"), ["red bike", "blue car", "red car"])
    unknown = replace(fitted, config=replace(TEXT, empty="unknown"))
    values = ["red bike", None, "", "blue car red car red bike", "", None, "x"]
    for encoder in (fitted, unknown):
        encoded = encoder.encode_arrow(pa.array(values, pa.string()))
        pieces = [encoder.encode(value) for value in values]
        assert encoded["values"].tolist() == [piece for item in pieces for piece in item]
        assert encoded["piece_offsets"].diff().tolist() == [len(item) for item in pieces]
