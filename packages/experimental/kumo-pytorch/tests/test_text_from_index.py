# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
import shutil
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from kumo.pytorch import api, encoding
from kumo.pytorch.column_stats import TableColumnEncoder
from kumo.pytorch.encoding import fit_encoder, train_text
from kumo.pytorch.engines.diskgraph import DiskGraphSource
from kumo.pytorch.materialization import ensure_diskgraph_index
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


def test_the_index_trains_the_tokenizer_python_trains_on_every_thread_count(tmp_path):
    pieces = [
        "naïve café",
        "東京タワー",
        "🎉 party",
        "  two spaces",
        "tab\tand\nline",
        "it's",
        "",
        "1234 5678",
        "e\u0301",
        "Привет мир",
        "x" * 30,
        "a  b   c",
        "\u00a0nbsp",
    ]
    rng = random.Random(0)
    values = [
        None if i % 7 == 0 else " ".join(rng.choices(pieces, k=rng.randint(1, 4)))
        for i in range(2_000)
    ]
    pq.write_table(
        pa.table({"id": range(len(values)), "body": pa.array(values, pa.string())}),
        tmp_path / "notes.parquet",
    )
    schema = api.DataSchema(
        tables=(
            api.TableSpec(
                "notes",
                "id",
                {"id": api.DType("int64"), "body": api.DType("string")},
                time_column=api.NON_TEMPORAL,
            ),
        )
    )
    dataset = api.Dataset({"notes": api.Parquet(str(tmp_path / "notes.parquet"))})
    index = ensure_diskgraph_index(schema, dataset, tmp_path / "index", num_shards=3)
    encoder = TableColumnEncoder(TEXT, api.DType("string"), (("notes", "body"),))
    expected = train_text(TEXT, [[value for value in values if value is not None]])
    for threads in (1, 4):
        source = DiskGraphSource(index, schema, num_threads=threads)
        tokenizer, directory = source.column_text(encoder, None)
        assert tokenizer == expected
        shutil.rmtree(tmp_path / "index" / "col_stats" / directory)


def test_chunked_encoding_matches_one_string_at_a_time(monkeypatch):
    monkeypatch.setattr(encoding, "TEXT_BATCH", 3)
    fitted = fit_encoder("e", TEXT, api.DType("string"), ["red bike", "blue car", "red car"])
    unknown = replace(fitted, config=replace(TEXT, empty="unknown"))
    tokenizer = Tokenizer.from_str(fitted.tokenizer_json)
    tokenizer.enable_padding()
    padded = fit_encoder("e", replace(TEXT, state=tokenizer.to_str()), api.DType("string"), ())
    values = ["red bike", None, "", "blue car red car red bike", "", None, "x"]
    for encoder in (fitted, unknown, padded):
        encoded = encoder.encode_arrow(pa.array(values, pa.string()))
        pieces = [encoder.encode(value) for value in values]
        assert encoded["values"].tolist() == [piece for item in pieces for piece in item]
        assert encoded["piece_offsets"].diff().tolist() == [len(item) for item in pieces]
