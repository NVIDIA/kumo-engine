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
from kumo.pytorch.column_stats import DROPPED, TableColumnEncoder, read_token_lengths
from kumo.pytorch.encoding import fit_encoder, train_text
from kumo.pytorch.engines.diskgraph import DiskGraphSource
from kumo.pytorch.materialization import ensure_diskgraph_index
from tests.test_column_stats import LATEST, _column, _contract, _requests, _setup
from tests.test_diskgraph_backend import _fixture_rows, _write_parquet

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


def _measured(contract: api.DataLoaderSpec) -> api.DataLoaderSpec:
    batching = replace(contract.batch.batching, length_input="transactions")
    return replace(contract, batch=replace(contract.batch, batching=batching))


def _reading(contract: api.DataLoaderSpec, name: str, column: str, encoding) -> api.DataLoaderSpec:
    events = contract.sample.events["transactions"]
    events = replace(events, fields={**events.fields, "again": api.Field(column=column)})
    transactions = contract.batch.inputs["transactions"]
    group = transactions.representation
    kept = group.fields if name == "transactions" else ()
    group = replace(group, fields=(*kept, api.Encode(field="again", encoding=encoding)))
    inputs = {**contract.batch.inputs, name: replace(transactions, representation=group)}
    return replace(
        contract,
        sample=replace(contract.sample, events={"transactions": events}),
        batch=replace(contract.batch, inputs=inputs),
    )


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


def test_a_fit_without_length_input_keeps_no_token_lengths(tmp_path):
    dataset, engine = _setup("diskgraph", tmp_path)
    with api.prepare(_text_contract(), dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            assert "token_lengths" not in task.fitted_state.fit["inputs"]["statistics"]
    assert not list(tmp_path.glob("index/col_stats/*/*.lengths-*.bin"))


@pytest.mark.parametrize("padded", (False, True))
def test_a_measured_fit_keeps_each_event_rows_token_count(tmp_path, padded):
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
        with api.prepare(_measured(_text_contract(encoding)), dataset, engine) as plan:
            with plan.materialize(_requests(), fit_split="train") as task:
                encoder = task.output("transactions").field(name="description").encoder
                tokenizer = encoder.tokenizer
                lengths = task.fitted_state.fit["inputs"]["statistics"]["token_lengths"]
        path = tmp_path / "index" / "col_stats" / lengths["card_transactions"]["file"]
        kept.append((lengths, path.stat().st_mtime_ns))
    assert kept[0] == kept[1]
    rows = _fixture_rows()["card_transactions"]
    nodes = diskgraph.SamplerEngine(str(tmp_path / "index")).lookup_node_ids(
        "card_transactions", pa.array([row["card_transaction_id"] for row in rows])
    )
    constant, pieces = read_token_lengths(tmp_path / "index", lengths["card_transactions"])
    assert constant == 2  # amount and channel
    assert len(pieces) == len(rows)
    for node, row in zip(nodes, rows, strict=True):
        assert pieces[node] == len(tokenizer.encode(row["description"] or "").ids)


def _edited(directory, column: str, value, step: int):
    rows = _fixture_rows()
    for row in rows["card_transactions"][::step]:
        row[column] = value
    dataset = _write_parquet(rows, directory / "parquet")
    engine = api.DiskGraphEngine(index=directory / "index", num_threads=2)
    return rows["card_transactions"], dataset, engine


def _entry(contract: api.DataLoaderSpec, dataset, engine) -> dict:
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            (entry,) = task.fitted_state.fit["inputs"]["statistics"]["token_lengths"].values()
    return entry


def _nodes(directory, rows):
    import diskgraph

    engine = diskgraph.SamplerEngine(str(directory / "index"))
    ids = pa.array([row["card_transaction_id"] for row in rows])
    return engine.lookup_node_ids("card_transactions", ids)


def _required(fields: tuple[str, ...]) -> api.DataLoaderSpec:
    contract = _measured(_contract())
    transactions = contract.batch.inputs["transactions"]
    group = replace(transactions.representation, row=api.Composite(required_fields=fields))
    inputs = {**contract.batch.inputs, "transactions": replace(transactions, representation=group)}
    return replace(contract, batch=replace(contract.batch, inputs=inputs))


def test_an_empty_string_counts_the_unknown_id_it_emits(tmp_path):
    rows, dataset, engine = _edited(tmp_path, "description", "", 4)
    with api.prepare(
        _measured(_text_contract(replace(TEXT, empty="unknown"))), dataset, engine
    ) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            encoder = task.output("transactions").field(name="description").encoder
            (entry,) = task.fitted_state.fit["inputs"]["statistics"]["token_lengths"].values()
    _, pieces = read_token_lengths(tmp_path / "index", entry)
    descriptions = pa.array([row["description"] for row in rows])
    emitted = encoder.encode_arrow(descriptions)["piece_offsets"].diff().tolist()
    nodes = _nodes(tmp_path, rows)
    assert pieces[nodes].tolist() == emitted
    empty = [node for node, row in zip(nodes, rows, strict=True) if row["description"] == ""]
    assert pieces[empty].tolist() == [1] * len(empty)


@pytest.mark.parametrize("length_input", ("transactions", "follower"))
def test_a_row_with_a_null_required_field_is_marked_dropped(tmp_path, length_input):
    rows, dataset, engine = _edited(tmp_path, "channel", None, 3)
    contract = _required(("channel",))
    follower = api.Input(
        api.EventsOf("transactions"),
        api.Fields((api.Encode(field="amount", encoding=api.ZScore()),)),
        api.Padded(),
    )
    batch = replace(
        contract.batch,
        inputs={**contract.batch.inputs, "follower": follower},
        batching=replace(contract.batch.batching, length_input=length_input),
    )
    entry = _entry(replace(contract, batch=batch), dataset, engine)
    _, counts = read_token_lengths(tmp_path / "index", entry)
    expected = [DROPPED if row["channel"] is None else 0 for row in rows]
    assert counts[_nodes(tmp_path, rows)].tolist() == expected


@pytest.mark.parametrize(
    ("again", "message"),
    (
        ({"channel": api.Field(column="description")}, "reads several columns"),
        ({"channel": api.Constant("ok", dtype=api.DType("string"))}, "reads several columns"),
        ({}, "has no stored column"),
    ),
)
def test_a_required_field_every_component_must_store_once(tmp_path, again, message):
    contract = _required(("channel",))
    events = contract.sample.events["transactions"]
    again = replace(events, fields={"amount": api.Field(column="amount"), **again})
    sample = replace(contract.sample, events={"transactions": events, "again": again})
    transactions = contract.batch.inputs["transactions"]
    source = replace(transactions.source, events=("transactions", "again"))
    inputs = {**contract.batch.inputs, "transactions": replace(transactions, source=source)}
    contract = replace(contract, sample=sample, batch=replace(contract.batch, inputs=inputs))
    dataset, engine = _setup("diskgraph", tmp_path)
    with api.prepare(contract, dataset, engine) as plan:
        with pytest.raises(ValueError, match=message):
            plan.materialize(_requests(), fit_split="train")


def test_a_changed_empty_policy_or_required_field_never_reuses_counts(tmp_path):
    dataset, engine = _setup("diskgraph", tmp_path)
    contracts = (
        _measured(_text_contract()),
        _measured(_text_contract(replace(TEXT, empty="unknown"))),
        _required(("channel",)),
        _required(("amount", "channel")),
    )
    files = [_entry(contract, dataset, engine)["file"] for contract in contracts]
    assert len(set(files)) == len(files)
    assert all((tmp_path / "index" / "col_stats" / file).is_file() for file in files)


def test_a_fixed_width_recipe_records_a_constant_without_a_scan(tmp_path, monkeypatch):
    import diskgraph

    def scan(*args, **kwargs):
        raise AssertionError("a fixed-width recipe scanned the index")

    monkeypatch.setattr(diskgraph, "token_lengths", scan)
    rows = _fixture_rows()
    for row in rows["card_transactions"][::3]:
        row["channel"] = None
    dataset = _write_parquet(rows, tmp_path / "parquet")
    engine = api.DiskGraphEngine(index=tmp_path / "index", num_threads=2)
    contract = _measured(_contract())
    transactions = contract.batch.inputs["transactions"]
    group = transactions.representation
    group = replace(
        group,
        fields=tuple(replace(field, id_space="tokens") for field in group.fields),
        row=api.Tokens(missing="emit"),
    )
    transactions = replace(
        transactions, representation=group, layout=api.Nested(api.Packed(), api.Padded())
    )
    spec = replace(
        contract.batch,
        inputs={**contract.batch.inputs, "transactions": transactions},
        id_spaces={"tokens": api.IdSpace(specials={"pad": "pad", "missing": "missing"})},
        batching=replace(contract.batch.batching, max_tokens=64),
    )
    with api.prepare(replace(contract, batch=spec), dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            (entry,) = task.fitted_state.fit["inputs"]["statistics"]["token_lengths"].values()
            emitted = [
                batch.inputs["transactions"]["field_offsets"].diff().tolist()
                for batch in task.batches(_requests())
            ]
    assert read_token_lengths(tmp_path / "index", entry) == (2, None)
    assert {count for counts in emitted for count in counts} == {2}
    assert not list(tmp_path.glob("index/col_stats/*/*.lengths-*"))


def _lengths(contract: api.DataLoaderSpec, directory) -> tuple[dict, bytes]:
    dataset, engine = _setup("diskgraph", directory)
    with api.prepare(contract, dataset, engine) as plan:
        with plan.materialize(_requests(), fit_split="train") as task:
            (entry,) = task.fitted_state.fit["inputs"]["statistics"]["token_lengths"].values()
    return entry, (directory / "index" / "col_stats" / entry["file"]).read_bytes()


def test_only_the_length_inputs_columns_count(tmp_path):
    alone = _measured(_text_contract())
    beside = _reading(alone, "merchants", "merchant_id", api.Vocabulary())
    assert _lengths(alone, tmp_path / "alone") == _lengths(beside, tmp_path / "beside")


def test_a_column_two_encoders_read_counts_once(tmp_path):
    alone = _measured(_text_contract())
    again = _reading(alone, "transactions", "description", api.Vocabulary())
    assert _lengths(alone, tmp_path) == _lengths(again, tmp_path)
    twice = _reading(alone, "transactions", "description", replace(TEXT, min_frequency=3))
    dataset, engine = _setup("diskgraph", tmp_path)
    with api.prepare(twice, dataset, engine) as plan:
        with pytest.raises(ValueError, match="different tokenizers"):
            plan.materialize(_requests(), fit_split="train")


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


def test_a_nonnull_constant_required_field_needs_no_validity_scan(tmp_path, monkeypatch):
    import diskgraph

    contract = _required(("channel",))
    events = contract.sample.events["transactions"]
    events = replace(
        events, fields={**events.fields, "channel": api.Constant("ok", dtype=api.DType("string"))}
    )
    contract = replace(contract, sample=replace(contract.sample, events={"transactions": events}))

    def scan(*args, **kwargs):
        raise AssertionError("an always-present required constant needs no validity scan")

    monkeypatch.setattr(diskgraph, "token_lengths", scan)
    dataset, engine = _setup("diskgraph", tmp_path)
    entry = _entry(contract, dataset, engine)
    assert entry["file"] is None
