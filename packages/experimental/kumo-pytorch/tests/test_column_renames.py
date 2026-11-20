# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parquet bindings preserve logical names through native ingest and batch assembly."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from tests.test_preview import _equal
from tests.test_type_resolution import _source_contract


def _parquet(path, data, renamed=False):
    names = data.column_names
    if renamed:
        data = data.select(list(reversed(names)))
        data = data.rename_columns([name.upper() for name in data.column_names])
    pq.write_table(data, path)
    return api.Parquet(str(path), {name.upper(): name for name in names} if renamed else {})


def _binding_contract():
    contract = _source_contract(
        {"id": api.DType("int64"), "a": api.DType("float64"), "b": api.DType("float64")}
    )
    return replace(
        contract, sample=replace(contract.sample, entity_values={"value": api.Field("a")})
    )


_ROW = {"id": 1, "a": 2.0, "b": 9.0}
_BINDINGS = {
    "identity": ([_ROW], {"a": "a"}, 2.0),
    "swap": ([_ROW], {"a": "b", "b": "a"}, 9.0),
    "chain": ([{"id": 1, "A": 2.0, "a": 9.0}], {"A": "a", "a": "b"}, 2.0),
    "unused": ([_ROW], {"absent": "unused"}, 2.0),
    "ambiguous": ([{**_ROW, "A": 3.0}], {"A": "a"}, "ambiguous logical column 'a'"),
    "renamed-away": ([_ROW], {"a": "gone"}, "is missing columns ['a']"),
    "duplicate": (
        [[("id", 1), ("a", 2.0), ("a", 9.0), ("b", 3.0)]],
        {},
        "duplicate physical column names",
    ),
    "missing": ([{"id": 1}], {}, "Parquet table 'users' in {path!r} is missing columns ['a', 'b']"),
    "different-binding": (
        [{"id": 1, "A": 2.0, "b": 9.0}, {"id": 2, "a": 4.0, "b": 8.0}],
        {"A": "a"},
        "column binding differs across Parquet files",
    ),
    "subset": (
        [{"id": 1, "A": 2.0, "b": 9.0, "extra": 3}, {"id": 2, "A": 4.0, "b": 8.0, "other": 5}],
        {"A": "a"},
        4.0,
    ),
}


@pytest.mark.parametrize("files,renames,expected", _BINDINGS.values(), ids=_BINDINGS)
def test_column_bindings(tmp_path, files, renames, expected):
    pytest.importorskip("diskgraph")
    for i, columns in enumerate(files):
        names, values = zip(*(columns.items() if isinstance(columns, dict) else columns))
        _parquet(tmp_path / f"{i}.parquet", pa.table([[v] for v in values], names=names))
    dataset = api.Dataset({"users": api.Parquet(str(tmp_path), renames)})
    engine = api.DiskGraphEngine(index=tmp_path / "index")
    if isinstance(expected, str):
        with pytest.raises(ValueError) as error:
            api.prepare(_binding_contract(), dataset, engine)
        if expected.startswith("Parquet table"):
            assert str(error.value) == expected.format(path=str(tmp_path / "0.parquet"))
        else:
            assert expected in str(error.value)
    else:
        with api.prepare(_binding_contract(), dataset, engine) as plan:
            with plan.materialize((api.SampleRequest("one", len(files), 0),)) as task:
                batch = next(task.batches(batch_size=1, output_format="model"))
                assert batch.inputs["values"]["value"]["values"].tolist() == [expected]


def test_index_reuse_includes_rename_map(tmp_path, monkeypatch):
    diskgraph = pytest.importorskip("diskgraph")
    source = _parquet(tmp_path / "users.parquet", pa.table({"id": [1], "a": [2.0], "b": [9.0]}))
    source = replace(source, columns={"a": "b", "b": "a"})
    contract = _binding_contract()
    dataset = api.Dataset({"users": source})
    engine = api.DiskGraphEngine(index=tmp_path / "index")
    with api.prepare(contract, dataset, engine):
        pass
    monkeypatch.setattr(diskgraph, "ingest", lambda *a, **kw: pytest.fail("index rebuilt"))
    with api.prepare(contract, dataset, engine):
        pass
    changed = api.Dataset({"users": replace(dataset.tables["users"], columns={})})
    with pytest.raises(ValueError, match="different dataset"):
        api.prepare(contract, changed, engine)


def _parity_contract(topology, tables):
    schema = api.DataSchema(
        tables=tuple(
            api.TableSpec(
                name,
                "id",
                {field.name: api.DType.from_arrow(field.type) for field in data.schema},
                time_column=api.NON_TEMPORAL if name == "users" else "at",
            )
            for name, data in tables.items()
        ),
        relations=(api.RelationSpec("history", "users", "id", "events", "user_id"),),
    )
    fields = (api.Encode("amount", api.ZScore()), api.Encode("kind", api.Vocabulary()))
    sample = api.SampleSpec(
        entity=api.EntitySpec("users"),
        targets={"label": api.ExternalTarget(api.DType("float64"))},
    )
    if topology == "graph":
        sample = replace(
            sample,
            neighborhoods={
                "past": api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("history", 3, strategy="recent"),)),)
                )
            },
        )
        value = api.Input(
            api.NeighborhoodSource("past"),
            api.Graph(node_fields={"events": api.Fields(fields)}),
            None,
        )
        batching = api.Batching(max_samples=2)
    else:
        sample = replace(
            sample,
            events={
                "past": api.EventSpec(
                    api.Rows(("history",), time=api.TimeWindow("at", stop=0)),
                    fields={name: api.Field(name) for name in ("amount", "kind", "description")},
                )
            },
        )
        fields = (
            api.Encode("amount", api.Buckets(bins=3)),
            fields[1],
            api.Encode(
                "description", api.Text(vocabulary_size=32, min_frequency=1, empty="unknown")
            ),
        )
        fields = tuple(replace(field, id_space="tokens") for field in fields)
        value = api.Input(
            api.EventSource(("past",)),
            api.Fields(fields, row=api.Tokens()),
            api.Nested(api.Packed(), api.Padded()),
        )
        batching = api.Batching(
            max_samples=2, max_tokens=16, length_input="past", bucket_by_length=True
        )
    return api.DataLoaderSpec(
        schema,
        sample,
        api.BatchSpec(
            inputs={"past": value},
            id_spaces={"tokens": api.IdSpace(specials={"pad": "[PAD]", "unknown": "[UNK]"})},
            batching=batching,
        ),
    )


@pytest.mark.parametrize("topology", ["graph", "sequence"])
def test_uppercase_reordered_parquet_has_identical_batches_and_fitted_state(tmp_path, topology):
    pytest.importorskip("diskgraph")
    tables = {
        "users": pa.table({"id": [1, 2]}),
        "events": pa.table(
            {
                "id": [10, 11, 12, 13],
                "user_id": [1, 1, 2, 1],
                "at": pa.array([1000, 2000, 1500, 4000], type=pa.timestamp("ms", tz="UTC")),
                "amount": [2.0, None, 6.0, 99.0],
                "kind": ["a", "b", "a", "future"],
                "description": ["small purchase", "", "a larger purchase", "future text"],
            }
        ),
    }
    contract = _parity_contract(topology, tables)
    # Includes an empty history and a NULL feature; the last event is after every cutoff.
    anchor_data = pa.Table.from_pylist(
        [
            {"sid": "one", "entity": 1, "cutoff": 3, "label": 1.0},
            {"sid": "two", "entity": 2, "cutoff": 3, "label": 0.0},
            {"sid": "empty", "entity": 2, "cutoff": 0, "label": 2.0},
        ]
    )
    results = []
    for renamed in (False, True):
        anchors = api.AnchorTable(
            _parquet(tmp_path / f"anchors-{renamed}.parquet", anchor_data, renamed),
            entity_id="entity",
            as_of_time="cutoff",
            sample_id="sid",
            targets={"label": "label"},
        )
        dataset = api.Dataset(
            {
                name: _parquet(tmp_path / f"{name}-{renamed}.parquet", data, renamed)
                for name, data in tables.items()
            }
        )
        index = tmp_path / f"index-{renamed}"
        with api.prepare(
            contract,
            dataset,
            api.DiskGraphEngine(
                index=index,
                num_shards=1,
                num_threads=1,
            ),
        ) as plan:
            with plan.materialize(anchors) as task:
                batches = {
                    fmt: list(task.batches(output_format=fmt, seed=7))
                    for fmt in ("inspectable", "model")
                }
                artifacts = {
                    str(path.relative_to(index / "col_stats")): (
                        json.loads(path.read_text())
                        if path.suffix == ".json"
                        else path.read_bytes()
                    )
                    for path in (index / "col_stats").glob("*/*")
                    if path.suffix in {".json", ".bin"}
                }
                assert any(name.endswith(".col_stats.json") for name in artifacts)
                if topology == "sequence":
                    assert any(name.endswith(".bin") for name in artifacts)
                results.append((task.fitted_state, batches, artifacts))
    before, after = results
    assert before[0] == after[0]
    assert before[2] == after[2]
    for fmt in before[1]:
        assert before[1][fmt]
        for expected, actual in zip(before[1][fmt], after[1][fmt], strict=True):
            _equal(expected.inputs, actual.inputs)
            _equal(expected.targets, actual.targets)
            _equal(expected.metadata, actual.metadata)
