# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-free, read-only reuse through the public preparation lifecycle."""

import hashlib
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kumo.pytorch import api
from kumo.pytorch.column_stats import TableColumnEncoder, read_token_lengths

diskgraph = pytest.importorskip("diskgraph")


def inventory(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@contextmanager
def read_only(root):
    paths = [root, *root.rglob("*")]
    modes = {p: p.stat().st_mode for p in paths}
    for p in paths:
        p.chmod(0o555 if p.is_dir() else 0o444)
    try:
        yield
    finally:
        for p, mode in modes.items():
            p.chmod(mode)


def contract(schema):
    return api.DataLoaderSpec(
        schema,
        api.SampleSpec(
            entity=api.EntitySpec("users"),
            neighborhoods={
                "history": api.NeighborhoodSpec(
                    (api.GraphLayer((api.GraphSample("orders", fanout=2, strategy="recent"),)),)
                ),
            },
        ),
        api.BatchSpec(
            inputs={
                "graph": api.Input(
                    api.NeighborhoodSource("history"),
                    api.Graph(
                        node_fields={
                            table.name: api.Fields(
                                tuple(api.Encode(name, api.Raw()) for name in table.columns)
                            )
                            for table in schema.tables
                        }
                    ),
                    None,
                )
            },
            batching=api.Batching(max_samples=2),
        ),
    )


@pytest.fixture
def indexed(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    data = {
        "users": pa.table({"id": [-2, 7], "name": ["Alice", "Bob"]}),
        "events": pa.table(
            {
                "id": ["a", "b", "c", "d"],
                "user": [-2, -2, -2, 7],
                "reviewer": [7, 7, 7, -2],
                "parent": pa.array([None, "a", "b", None], pa.string()),
                "at": pa.array([10, 20, 30, 10], pa.timestamp("s")),
                "value": [2.0, 4.0, 100.0, None],
                "note": ["first item", "second item", "future item", None],
            }
        ),
    }
    for name, table in data.items():
        pq.write_table(table, source / f"{name}.parquet")
    schema = api.DataSchema(
        tuple(
            api.TableSpec(
                name,
                "id",
                {f.name: api.DType.from_arrow(f.type) for f in table.schema},
                time_column="at" if name == "events" else api.NON_TEMPORAL,
            )
            for name, table in data.items()
        ),
        (
            api.RelationSpec("orders", "users", "id", "events", "user"),
            api.RelationSpec("reviews", "events", "reviewer", "users", "id"),
            api.RelationSpec("parent", "events", "parent", "events", "id"),
        ),
    )
    dataset = api.Dataset({name: api.Parquet(str(source / f"{name}.parquet")) for name in data})
    index = tmp_path / "index"
    spec = contract(schema)
    with api.prepare(spec, dataset, api.DiskGraphEngine(index=index, num_threads=1)):
        pass
    anchor_path = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.table(
            {
                "user": [-2, 7],
                "at": pa.array([20, 20], pa.timestamp("s")),
                "sample": ["left", "right"],
            }
        ),
        anchor_path,
    )
    anchors = api.AnchorTable(api.Parquet(str(anchor_path)), "user", "at", "sample")
    return index, source, dataset, spec, anchors


def sample(engine, spec, anchors):
    with api.prepare(contract=spec, engine=engine) as plan:
        with plan.materialize(anchors) as task:
            graph = next(iter(task.batches(output_format="columnar"))).inputs["graph"]
            events = graph.nodes["events"]
            assert set(events.ids.to_pylist()) == {"a", "b", "d"}
            assert events.features["at"].type == pa.timestamp("s", tz="UTC")
            assert set(events.features["value"].to_pylist()) == {2.0, 4.0, None}
            assert graph.edges["orders"].source_type == "users"
            assert graph.edges["orders"].destination_type == "events"
            for left, right in zip(
                *[graph.edges["orders"].index[n].to_pylist() for n in ("source", "destination")],
                strict=True,
            ):
                assert graph.nodes["users"].owners[left] == events.owners[right]
            return graph


def test_relocated_index_without_sources_or_base_writes(indexed, tmp_path, monkeypatch):
    index, source, dataset, spec, anchors = indexed
    relocated = tmp_path / "relocated"
    shutil.move(str(index), relocated)
    shutil.rmtree(source)
    before = inventory(relocated)
    original_ingest = diskgraph.ingest
    extensions = []

    def ingest(*args, **kwargs):
        assert kwargs.get("parent_index_path") == str(relocated)
        assert not str(args[1]).startswith(str(relocated))
        extensions.append(args[1])
        return original_ingest(*args, **kwargs)

    monkeypatch.setattr(diskgraph, "ingest", ingest)
    with read_only(relocated):
        engine = api.DiskGraphEngine.from_index(
            relocated, work_dir=tmp_path / "work", num_threads=1
        )
        assert engine.schema == spec.schema
        sample(engine, spec, anchors)
        assert extensions
        spec_path = tmp_path / "contract.json"
        spec_path.write_text(spec.to_json())
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from pathlib import Path
from kumo.pytorch import api
index, work, spec, anchors = sys.argv[1:]
engine = api.DiskGraphEngine.from_index(index, work_dir=work, num_threads=1)
contract = api.DataLoaderSpec.from_json(Path(spec).read_text())
with api.prepare(contract=contract, engine=engine) as plan:
    with plan.materialize(api.AnchorTable(api.Parquet(anchors), 'user', 'at', 'sample')) as task:
        graph = next(iter(task.batches(output_format='columnar'))).inputs['graph']
        assert set(graph.nodes['events'].ids.to_pylist()) == {'a', 'b', 'd'}
""",
                str(relocated),
                str(tmp_path / "fresh-process"),
                str(spec_path),
                anchors.source.path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    assert inventory(relocated) == before
    assert all(not Path(path).exists() for path in extensions)
    assert not (relocated / "tasks").exists()


def test_snapshot_binding_needs_no_pytorch_manifest(indexed, tmp_path):
    index, source, _, spec, anchors = indexed
    config = (index / "diskgraph_config.json").read_text()
    index = tmp_path / "engine-index"
    diskgraph.ingest(config, str(index), col_stats=False, success_marker_content="{}")
    # The real engine integration test covers connector -> GraphSnapshot creation.
    snapshot = SimpleNamespace(index_path=index, id="graph@version", graph_config={})
    engine = api.DiskGraphEngine.from_snapshot(snapshot, work_dir=tmp_path / "work", num_threads=1)
    shutil.rmtree(source)
    before = inventory(index)
    sample(engine, spec, anchors)  # Custom logical relation aliases resolve by endpoint AND key.
    assert inventory(index) == before


def test_external_fitting_statistics_tokenizer_and_lengths(indexed, tmp_path):
    index, _, _, spec, _ = indexed
    before = inventory(index)
    engine = api.DiskGraphEngine.from_index(index, work_dir=tmp_path / "work", num_threads=1)
    with read_only(index), api.prepare(contract=spec, engine=engine) as plan:
        source = plan._source
        stats = source.column_stats([("events", "value", "numerical")], 20 * 10**9)
        assert stats.columns[("events", "value", "numerical")]["count"] == 3
        assert stats.columns[("events", "value", "numerical")]["mean"] == 3.0
        encoder = TableColumnEncoder(
            api.Text(vocabulary_size=300, min_frequency=1),
            api.DType("string"),
            (("events", "note"),),
        )
        tokenizer, name = source.column_text(encoder, 20 * 10**9)
        assert source.column_text(encoder, 20 * 10**9) == (tokenizer, name)
        file = source.token_lengths("events", [("note", tokenizer, 0)], [], 20 * 10**9)
        _, values = read_token_lengths(
            index, {"file": file, "constant": 0}, cache_root=source._cache_root
        )
        assert len(values) == 4
    assert inventory(index) == before


def test_existing_base_statistics_are_reused_and_reported(indexed, tmp_path, monkeypatch):
    index, _, _, spec, _ = indexed
    from kumo.pytorch.column_stats import read_column_stats

    requests = [("events", "value", "numerical")]
    stored = read_column_stats(index, requests, 20, "seconds")
    before = inventory(index)

    def unexpected_compute(*args, **kwargs):
        pytest.fail("the requested statistics already exist in the base")

    monkeypatch.setattr(diskgraph, "cutoff_col_stats", unexpected_compute)
    engine = api.DiskGraphEngine.from_index(index, work_dir=tmp_path / "work")
    with read_only(index), api.prepare(contract=spec, engine=engine) as plan:
        reused = plan._source.column_stats(requests, 20 * 10**9)
        assert reused.columns == stored.columns
        assert reused.files == tuple(str(index / "col_stats" / f) for f in stored.files)
    assert inventory(index) == before


def test_external_statistics_compute_only_missing_requests(indexed, tmp_path, monkeypatch):
    index, _, _, spec, _ = indexed
    from kumo.pytorch.column_stats import read_column_stats

    known = ("events", "value", "numerical")
    missing = ("events", "value", "categorical")
    stored = read_column_stats(index, [known], 20, "seconds")
    before = inventory(index)
    calls = []
    original = diskgraph.cutoff_col_stats

    def compute(index_path, request_json, **options):
        request = json.loads(request_json)
        assert request["columns"] == {"events": {"value": ["categorical"]}}
        assert not Path(options["output_dir"]).is_relative_to(index)
        calls.append(request)
        return original(index_path, request_json, **options)

    monkeypatch.setattr(diskgraph, "cutoff_col_stats", compute)
    engine = api.DiskGraphEngine.from_index(index, work_dir=tmp_path / "work", num_threads=1)
    with read_only(index), api.prepare(contract=spec, engine=engine) as plan:
        source = plan._source
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda _: source.column_stats([known, missing], 20 * 10**9), range(2))
            )
        for result in results:
            assert result.columns[known]["mean"] == stored.columns[known]["mean"]
            assert sorted(result.columns[missing]["category_counts"]) == [["2", 1], ["4", 1]]
        count = len(calls)
        assert count >= 1
        assert source.column_stats([known, missing], 20 * 10**9) == results[0]
        assert len(calls) == count
    assert inventory(index) == before


@pytest.mark.parametrize(
    "unit,clock", [("ms", "milliseconds"), ("us", "microseconds"), ("ns", "nanoseconds")]
)
def test_native_clock_precision_preserves_cutoff(indexed, tmp_path, unit, clock):
    index, source, _, spec, _ = indexed
    path = source / "events.parquet"
    table = pq.read_table(path)
    table = table.set_column(
        table.schema.get_field_index("at"),
        "at",
        pa.array([1001, 1002, 1003, 1001], pa.timestamp(unit)),
    )
    pq.write_table(table, path)
    config = json.loads((index / "diskgraph_config.json").read_text())
    config["timestamp_unit"] = clock
    native = tmp_path / "native"
    diskgraph.ingest(json.dumps(config), str(native), col_stats=False, success_marker_content="{}")
    shutil.rmtree(source)
    anchor_path = tmp_path / "precise-anchors.parquet"
    pq.write_table(
        pa.table({"user": [-2, 7], "at": pa.array([1002, 1002], pa.timestamp(unit))}), anchor_path
    )
    engine = api.DiskGraphEngine.from_index(native, work_dir=tmp_path / "work", num_threads=1)
    schema = replace(engine.schema, relations=spec.schema.relations)
    with api.prepare(contract=contract(schema), engine=engine) as plan:
        with plan.materialize(api.AnchorTable(api.Parquet(str(anchor_path)), "user", "at")) as task:
            graph = next(iter(task.batches(output_format="columnar"))).inputs["graph"]
            events = graph.nodes["events"]
            assert set(events.ids.to_pylist()) == {"a", "b", "d"}
            assert events.features["at"].type == pa.timestamp(unit, tz="UTC")
            assert set(events.features["at"].cast(pa.int64()).to_pylist()) == {1001, 1002}


def test_concurrent_consumers_keep_task_cleanup_separate(indexed, tmp_path):
    index, _, _, spec, anchors = indexed
    before = inventory(index)

    def run(_):
        return sample(
            api.DiskGraphEngine.from_index(index, work_dir=tmp_path / "shared", num_threads=1),
            spec,
            anchors,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert len(list(pool.map(run, range(2)))) == 2
    assert inventory(index) == before


def test_projection_alias_validation_and_bad_bindings(indexed, tmp_path):
    index, _, dataset, spec, anchors = indexed
    engine = api.DiskGraphEngine.from_index(index, work_dir=tmp_path / "work")
    assert api.check(spec, engine=engine) == ()
    with pytest.raises(api.ValidationError, match="does not accept Dataset"):
        api.prepare(spec, dataset, engine)
    unresolved = replace(
        spec,
        schema=replace(
            spec.schema,
            tables=tuple(
                replace(table, columns={name: None for name in table.columns})
                for table in spec.schema.tables
            ),
        ),
    )
    with api.prepare(contract=unresolved, engine=engine) as plan:
        assert plan.contract.schema == spec.schema
    wrong = replace(spec.schema.relations[0], right_key="reviewer")
    projected = replace(spec.schema, relations=(wrong,))
    with api.prepare(contract=contract(projected), engine=engine) as plan:
        with plan.materialize(anchors) as task:
            graph = next(iter(task.batches(output_format="columnar"))).inputs["graph"]
            edge = graph.edges["orders"]
            pairs = {
                (graph.nodes["users"].ids[left].as_py(), graph.nodes["events"].ids[right].as_py())
                for left, right in zip(
                    edge.index["source"].to_pylist(),
                    edge.index["destination"].to_pylist(),
                    strict=True,
                )
            }
            assert pairs == {(-2, "d"), (7, "a"), (7, "b")}
    missing = replace(wrong, right_key="value")
    assert api.check(
        replace(spec, schema=replace(spec.schema, relations=(missing,))), engine=engine
    )
    with pytest.raises(ValueError, match="outside"):
        api.DiskGraphEngine.from_index(index, work_dir=index / "work")


def test_physically_present_self_relation(indexed, tmp_path):
    index, source, _, spec, anchors = indexed
    shutil.rmtree(source)
    engine = api.DiskGraphEngine.from_index(index, work_dir=tmp_path / "work", num_threads=1)
    spec = replace(
        spec,
        sample=replace(
            spec.sample,
            neighborhoods={
                "history": api.NeighborhoodSpec(
                    (
                        api.GraphLayer((api.GraphSample("orders", fanout=1, strategy="recent"),)),
                        api.GraphLayer((api.GraphSample("parent", fanout=1, strategy="recent"),)),
                    )
                ),
            },
        ),
    )
    with api.prepare(contract=spec, engine=engine) as plan:
        with plan.materialize(anchors) as task:
            graph = next(iter(task.batches(output_format="columnar"))).inputs["graph"]
            events, edge = graph.nodes["events"], graph.edges["parent"]
            assert {
                (events.ids[left].as_py(), events.ids[right].as_py())
                for left, right in zip(
                    edge.index["source"].to_pylist(),
                    edge.index["destination"].to_pylist(),
                    strict=True,
                )
            } == {("b", "a")}


@pytest.mark.parametrize(
    "change,match",
    [
        ("legacy", "diskgraph_config"),
        ("incomplete", "completion manifest"),
        ("edge", "edge table"),
        ("list", "unsupported stored type"),
        ("pair", "edge pair"),
    ],
)
def test_incompatible_artifacts_fail_before_sampling(indexed, tmp_path, change, match):
    index, _, _, _, _ = indexed
    if change == "legacy":
        (index / "diskgraph_config.json").unlink()
    elif change == "incomplete":
        (index / "index_manifest.json").unlink()
    else:
        filename = "index_metadata.json" if change == "pair" else "diskgraph_config.json"
        path = index / filename
        value = json.loads(path.read_text())
        if change == "edge":
            value["tables"][0]["edge_table"] = {"src_table": "users", "dst_table": "users"}
        elif change == "list":
            value["tables"][0]["columns"][0]["ingested_dtype"] = "string_array"
        else:
            value["edge_relations"][0]["reverse_of"] = "absent"
        path.write_text(json.dumps(value))
        # Keep the native size inventory internally consistent for this semantic refusal.
        manifest_path = index / "_MANIFEST.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            for entry in manifest["files"]:
                if entry["rel_path"] == filename:
                    entry["size_bytes"] = path.stat().st_size
            manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, RuntimeError), match=match):
        api.DiskGraphEngine.from_index(index, work_dir=tmp_path / "work")


def test_connector_versions_allow_restage_but_reject_changes(indexed, tmp_path):
    legacy, source, dataset, spec, _ = indexed
    versions = {
        name: {"source": f"connector/catalog/{name}", "version": "v1"} for name in dataset.tables
    }
    path = tmp_path / "versioned"
    with api.prepare(spec, dataset, api.DiskGraphEngine(index=path, source_versions=versions)):
        pass
    # Version-aware reuse explicitly trusts the connector, even if its token is stale.
    # A caller that needs file-change detection must keep the default legacy binding.
    event_path = source / "events.parquet"
    events = pq.read_table(event_path)
    pq.write_table(
        events.set_column(
            events.schema.get_field_index("value"), "value", pa.array([200.0, 400.0, 10000.0, None])
        ),
        event_path,
    )
    with pytest.raises(ValueError, match="different dataset"):
        api.prepare(spec, dataset, api.DiskGraphEngine(index=legacy))
    moved = tmp_path / "new_sources"
    shutil.move(source, moved)
    restaged = api.Dataset(
        {name: api.Parquet(str(moved / f"{name}.parquet")) for name in dataset.tables}
    )
    before = inventory(path)
    with api.prepare(spec, restaged, api.DiskGraphEngine(index=path, source_versions=versions)):
        pass
    assert inventory(path) == before
    for field in ("source", "version"):
        changed = {name: {**entry, field: "different"} for name, entry in versions.items()}
        with pytest.raises(ValueError, match="different dataset"):
            api.prepare(spec, restaged, api.DiskGraphEngine(index=path, source_versions=changed))
    with pytest.raises(ValueError, match="every dataset table"):
        api.prepare(spec, restaged, api.DiskGraphEngine(index=path, source_versions={}))


@pytest.mark.parametrize("alias", ["absolute", "relative", "symlink"])
@pytest.mark.parametrize("explicit_first", [False, True])
def test_explicit_index_work_dir_reuses_legacy_state(
    indexed, tmp_path, monkeypatch, alias, explicit_first
):
    index, _, dataset, spec, anchors = indexed
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        work = Path("index")
    elif alias == "symlink":
        work = tmp_path / "index-alias"
        work.symlink_to(index, target_is_directory=True)
    else:
        work = index
    encoder = TableColumnEncoder(
        api.Text(vocabulary_size=300, min_frequency=1),
        api.DType("string"),
        (("events", "note"),),
    )

    def run(explicit):
        options = {"work_dir": work} if explicit else {}
        engine = api.DiskGraphEngine(index=index, num_threads=1, **options)
        with api.prepare(spec, dataset, engine) as plan:
            source = plan._source
            stats = source.column_stats([("events", "value", "numerical")], 20 * 10**9)
            tokenizer, name = source.column_text(encoder, 20 * 10**9)
            lengths = source.token_lengths("events", [("note", tokenizer, 0)], [], 20 * 10**9)
            _, values = read_token_lengths(index, {"file": lengths, "constant": 0})
            assert all((index / "col_stats" / path).is_file() for path in stats.files)
            assert (index / "col_stats" / name).is_file()
            with plan.materialize(anchors) as task:
                assert any((index / "train_tables").iterdir())
                assert any((index / "tasks").iterdir())
                graph = next(iter(task.batches(output_format="columnar"))).inputs["graph"]
                assert set(graph.nodes["events"].ids.to_pylist()) == {"a", "b", "d"}
            assert not list((index / "train_tables").iterdir())
            assert not list((index / "tasks").iterdir())
            return stats, tokenizer, name, lengths, values.tolist()

    expected = run(explicit_first)
    before = inventory(index)

    def unexpected_compute(*args, **kwargs):
        pytest.fail("existing in-index cache must be reused")

    for function in ("cutoff_col_stats", "train_tokenizer", "token_lengths"):
        monkeypatch.setattr(diskgraph, function, unexpected_compute)
    assert run(not explicit_first) == expected
    assert inventory(index) == before
    assert not (index / "cache").exists()


@pytest.mark.parametrize("alias", ["absolute", "relative", "symlink", "nested"])
def test_adopted_indexes_require_external_work_dir(indexed, tmp_path, monkeypatch, alias):
    index, _, _, _, _ = indexed
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        work = Path("index")
    elif alias == "symlink":
        work = tmp_path / "index-alias"
        work.symlink_to(index, target_is_directory=True)
    else:
        work = index if alias == "absolute" else index / "work"
    with pytest.raises(ValueError, match="outside"):
        api.DiskGraphEngine.from_index(index, work_dir=work)
    with pytest.raises(ValueError, match="outside"):
        api.DiskGraphEngine.from_snapshot(
            SimpleNamespace(index_path=index, id="test"), work_dir=work
        )
    if alias == "nested":
        with pytest.raises(ValueError, match="outside"):
            api.DiskGraphEngine(index=index, work_dir=work)


@pytest.mark.parametrize("root", ["tasks", "train_tables", "cache"])
@pytest.mark.parametrize("symlink", [False, True])
def test_external_writable_roots_cannot_overlap_base(indexed, tmp_path, root, symlink):
    index, _, _, _, _ = indexed
    work = tmp_path / "work"
    work.mkdir()
    if symlink:
        (work / root).symlink_to(index, target_is_directory=True)
    else:
        moved = work / root
        shutil.move(index, moved)
        index = moved
    before = inventory(index)
    with pytest.raises(ValueError, match="outside"):
        api.DiskGraphEngine.from_index(index, work_dir=work)
    assert inventory(index) == before
    assert set(p.name for p in work.iterdir()) == {root}


def test_external_statistics_reuse_overlapping_requests(indexed, tmp_path, monkeypatch):
    index, _, _, spec, _ = indexed
    known = ("events", "value", "numerical")
    missing = ("events", "value", "categorical")
    engine = api.DiskGraphEngine.from_index(index, work_dir=tmp_path / "work", num_threads=1)
    with api.prepare(contract=spec, engine=engine) as plan:
        source = plan._source
        initial = source.column_stats([known], 20 * 10**9)
        original = diskgraph.cutoff_col_stats
        calls = []

        def compute(index_path, request_json, **options):
            request = json.loads(request_json)
            assert request["columns"] == {"events": {"value": ["categorical"]}}
            calls.append(request)
            return original(index_path, request_json, **options)

        monkeypatch.setattr(diskgraph, "cutoff_col_stats", compute)
        result = source.column_stats([known, missing], 20 * 10**9)
        assert result.columns[known]["mean"] == initial.columns[known]["mean"]
        assert result.files == initial.files
        assert len(calls) == 1
        assert source.column_stats([missing], 20 * 10**9).files == initial.files
        assert len(calls) == 1


def test_trusted_reuse_refuses_column_renames(indexed, tmp_path):
    _, _, dataset, spec, _ = indexed
    versions = {name: {"source": name, "version": "v1"} for name in dataset.tables}
    index = tmp_path / "trusted"
    with api.prepare(spec, dataset, api.DiskGraphEngine(index=index, source_versions=versions)):
        pass
    renamed = api.Dataset(
        {
            name: api.Parquet(source.path, columns={"value": "renamed"})
            if name == "events"
            else source
            for name, source in dataset.tables.items()
        }
    )
    with pytest.raises(ValueError, match="DG-COLUMN-RENAME"):
        api.prepare(spec, renamed, api.DiskGraphEngine(index=index, source_versions=versions))


def test_legacy_ingest_checks_the_snapshot_recorded_in_provenance(indexed, tmp_path, monkeypatch):
    from kumo.pytorch import materialization

    _, source, dataset, spec, _ = indexed
    original = materialization._dataset_payload
    calls = 0

    def capture(*args, **kwargs):
        nonlocal calls
        payload = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            path = source / "events.parquet"
            table = pq.read_table(path)
            pq.write_table(table.slice(0, 2), path)
        return payload

    monkeypatch.setattr(materialization, "_dataset_payload", capture)
    index = tmp_path / "new-index"
    with pytest.raises(ValueError, match="changed while DiskGraph was ingesting"):
        api.prepare(spec, dataset, api.DiskGraphEngine(index=index, num_threads=1))
    assert not index.exists()


def test_external_storage_refuses_old_native_api_before_task_writes(indexed, tmp_path, monkeypatch):
    index, _, dataset, spec, _ = indexed

    def old_cutoff_col_stats(index_dir, request_json, num_threads=None):
        pytest.fail("compatibility check must not execute statistics")

    monkeypatch.setattr(diskgraph, "cutoff_col_stats", old_cutoff_col_stats)
    work = tmp_path / "work"
    engine = api.DiskGraphEngine.from_index(index, work_dir=work)
    with pytest.raises(RuntimeError, match="matching kumo-engine revision"):
        api.prepare(contract=spec, engine=engine)
    assert not work.exists()
    with api.prepare(spec, dataset, api.DiskGraphEngine(index=index)):
        pass


def test_index_local_statistics_read_each_file_once(indexed, tmp_path, monkeypatch):
    from kumo.pytorch.column_stats import read_column_stats

    index, _, _, _, _ = indexed
    request = [("events", "value", "numerical")]
    expected = read_column_stats(index, request, 20, "seconds")
    alias = tmp_path / "alias"
    alias.symlink_to(index, target_is_directory=True)
    original = Path.read_text
    reads = []

    def read(path, *args, **kwargs):
        if path.name.endswith(".col_stats.json"):
            reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    assert read_column_stats(index, request, 20, "seconds", cache_root=alias) == expected
    assert len(reads) == 1
