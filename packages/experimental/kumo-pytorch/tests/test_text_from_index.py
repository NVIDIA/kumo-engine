# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import random
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from kumo.pytorch import api, encoding
from kumo.pytorch.column_stats import TableColumnEncoder, read_token_lengths
from kumo.pytorch.encoding import fit_encoder, train_text
from kumo.pytorch.engines.diskgraph import DiskGraphSource
from kumo.pytorch.materialization import ensure_diskgraph_index
from tests.test_column_stats import LATEST, _column, _contract, _requests, _setup
from tests.test_diskgraph_backend import _fixture_rows

pytest.importorskip("diskgraph")

TEXT = api.Text(vocabulary_size=300)


def _text_contract(encoding: api.Text = TEXT) -> api.DataLoaderSpec:
    contract = _contract()
    transactions = contract.batch.inputs["transactions"]
    fields = (
        *transactions.representation.fields,
        api.Encode(field="description", encoding=encoding),
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
        texts = tmp_path.glob("index/col_stats/*/*.text-*.tokenizer.json")
        published.append({path: path.stat().st_mtime_ns for path in texts})
    if engine_name == "diskgraph":
        assert any(".text-" in name for name in statistics["col_stats"])
        assert published[0] == published[1]
        (path,) = published[0]
        assert json.loads(path.read_text())["tokenizer"] == expected
        umask = os.umask(0)
        os.umask(umask)
        assert path.stat().st_mode & 0o777 == 0o666 & ~umask


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
        tokenizer, name = source.column_text(encoder, None)
        assert tokenizer == expected
        (tmp_path / "index" / "col_stats" / name).unlink()


@pytest.mark.parametrize("padded", (False, True))
def test_every_fit_keeps_each_event_rows_token_count(tmp_path, padded):
    import diskgraph

    encoding = TEXT
    if padded:
        supplied = Tokenizer.from_str(
            train_text(TEXT, [_column("card_transactions", "description", None)])
        )
        supplied.enable_padding(length=16)
        encoding = replace(TEXT, state=supplied.to_str())
    dataset, engine = _setup("diskgraph", tmp_path)
    kept = []
    for _ in range(2):
        with api.prepare(_text_contract(encoding), dataset, engine) as plan:
            with plan.materialize(_requests(), fit_split="train") as task:
                encoder = task.output("transactions").field(name="description").encoder
                tokenizer = encoder.tokenizer
                files = task.fitted_state.fit["inputs"]["statistics"]["token_lengths"]
        path = tmp_path / "index" / "col_stats" / files["card_transactions"]
        kept.append((files, path.stat().st_mtime_ns))
    assert kept[0] == kept[1]
    rows = _fixture_rows()["card_transactions"]
    nodes = diskgraph.SamplerEngine(str(tmp_path / "index")).lookup_node_ids(
        "card_transactions", pa.array([row["card_transaction_id"] for row in rows])
    )
    lengths = read_token_lengths(path)
    assert len(lengths) == len(rows)
    for node, row in zip(nodes, rows, strict=True):
        pieces = len(tokenizer.encode(row["description"] or "").ids)
        assert lengths[node] == 2 + pieces  # amount, channel and the description's pieces


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
