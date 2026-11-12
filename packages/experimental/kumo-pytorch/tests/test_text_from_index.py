# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text encoders over stored columns train on table rows up to the encoder cutoff."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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
        texts = tmp_path.glob("index/col_stats/*/*/*.text-*")
        published.append({path: path.stat().st_mtime_ns for path in texts})
    (file,) = statistics['["inputs","transactions","fields","description"]']["files"]
    assert file.startswith("card_transactions/description.text-")
    if engine_name == "diskgraph":
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


def test_training_is_independent_of_batching_order_and_threads():
    words = [f"w{index}x" for index in range(500)]
    values = [
        " ".join(words[(row * 7 + step) % 500] for step in range(row % 9 + 1))
        for row in range(5000)
    ]
    script = (
        "import json, sys; from kumo.pytorch.api import Text; "
        "from kumo.pytorch.encoding import train_text; "
        "values, size = json.load(sys.stdin); "
        "print(train_text(Text(vocabulary_size=300), "
        "(values[i : i + size] for i in range(0, len(values), size))))"
    )
    outputs = set()
    for threads, size, ordered in (("1", 1, values), ("8", 1000, values), ("8", 7, values[::-1])):
        run = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps([ordered, size]),
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "RAYON_NUM_THREADS": threads},
        )
        outputs.add(run.stdout)
    assert len(outputs) == 1


def test_text_on_per_anchor_values_still_fits_on_examples():
    values = ["red bike", None, "blue car", "red car"] * 3
    fitted = fit_encoder("e", TEXT, api.DType("string"), values)
    assert fitted.tokenizer_json == train_text(TEXT, [[value for value in values if value]])


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


_RSS = """
import sys
from kumo.pytorch import api
from kumo.pytorch.encoding import train_text
from kumo.pytorch.engines.diskgraph import DiskGraphSource
from kumo.pytorch.materialization import ensure_diskgraph_index
from tests.test_text_from_index import TEXT


def memory(field):
    status = dict(line.split(":", 1) for line in open("/proc/self/status"))
    return int(status[field].split()[0]) * 1024


schema = api.DataSchema(
    tables=(
        api.TableSpec(
            "notes",
            "id",
            {"id": api.DType("int64"), "body": api.DType("string")},
            time_column=api.NON_TEMPORAL,
        ),
    ),
)
dataset = api.Dataset({"notes": api.Parquet(sys.argv[1])})
index = ensure_diskgraph_index(schema, dataset, sys.argv[2], num_threads=4)
if sys.argv[3] == "train":
    source = DiskGraphSource(index, schema, num_threads=4)
    with open("/proc/self/clear_refs", "w") as peak:
        peak.write("5")
    before = memory("VmRSS")
    source.column_text([("notes", "body")], None, "text-test", lambda b: train_text(TEXT, b))
    print(memory("VmHWM") - before)
"""


def _training_growth(directory: Path, rows: int) -> int:
    body = pa.array(
        [f"note {index % 997} about item {index % 5003} and more" for index in range(rows)]
    )
    directory.mkdir()
    pq.write_table(
        pa.table({"id": pa.array(range(rows), pa.int64()), "body": body}),
        directory / "notes.parquet",
    )
    for step in ("build", "train"):
        run = subprocess.run(
            [
                sys.executable,
                "-c",
                _RSS,
                str(directory / "notes.parquet"),
                str(directory / "index"),
                step,
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parents[1],
        )
    return int(run.stdout.split()[-1])


def test_training_from_the_index_does_not_hold_the_column(tmp_path):
    # Training holds a bounded number of values, so 4x the rows is not 4x the peak; what
    # does grow is mostly the index pages the scan maps.
    small = _training_growth(tmp_path / "small", 500_000)
    large = _training_growth(tmp_path / "large", 2_000_000)
    assert large < small * 2 + 32 * 2**20
