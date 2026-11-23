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
from kumo.pytorch._storage import ColumnCache
from kumo.pytorch.column_stats import TableColumnEncoder, read_token_lengths

diskgraph = pytest.importorskip("diskgraph")


def test_unbound_engine_has_no_schema():
    engine = api.DiskGraphEngine()
    assert not hasattr(engine, "schema")


@pytest.mark.parametrize(
    "option", ["index", "index_build_id", "root", "root_dir", "output_dir", "work_dir"]
)
def test_engine_rejects_public_storage_options(tmp_path, option):
    with pytest.raises(TypeError, match=option):
        api.DiskGraphEngine(**{option: str(tmp_path / "outside")})
    with pytest.raises(TypeError, match=option):
        api.engine_from_name("diskgraph", **{option: str(tmp_path / "outside")})


def test_engine_has_no_public_index_opener():
    assert not hasattr(api.DiskGraphEngine, "from_index")


@pytest.mark.parametrize("policy", ["if_missing", "never"])
def test_engine_rejects_public_ingest_policy(policy):
    with pytest.raises(TypeError, match="ingest"):
        api.DiskGraphEngine(ingest=policy)
    with pytest.raises(TypeError, match="ingest"):
        api.engine_from_name("diskgraph", ingest=policy)


def snapshot(index):
    return SimpleNamespace(index_path=index, id="graph@version", graph_config={})


def test_snapshot_factory_uses_stored_identity_and_one_root(indexed, tmp_path, monkeypatch):
    import kumo.settings

    index, _, _, spec, _ = indexed
    roots = []

    def configured_root():
        roots.append(tmp_path)
        assert len(roots) == 1, "snapshot construction reread configured storage"
        return tmp_path

    def forbidden():
        raise AssertionError("snapshot construction allocated a fresh build identity")

    class Engine(api.DiskGraphEngine):
        pass

    monkeypatch.setattr(kumo.settings, "scratch_dir", configured_root)
    monkeypatch.setattr(diskgraph, "new_index_build_id", forbidden)
    engine = Engine.from_snapshot(snapshot(index), num_threads=2)
    assert isinstance(engine, Engine)
    assert engine.index == index
    assert engine.index_build_id == index.name
    assert engine.schema == spec.schema
    assert engine.num_threads == 2
    assert engine.random_seed == 0
    assert engine.num_shards == 1
    assert engine.encoder_cutoffs == ()
    assert engine._paths.task_dir("task") == tmp_path / "tasks" / index.name / "task"
    assert engine._paths.cache_dir == tmp_path / "cache" / index.name
    assert roots == [tmp_path]


def test_external_cache_refuses_redirected_cutoff(tmp_path):
    index = tmp_path / "index"
    (index / "col_stats").mkdir(parents=True)
    cache = tmp_path / "cache"
    (cache / "col_stats").mkdir(parents=True)
    (cache / "col_stats" / "cutoff").symlink_to(index / "col_stats", target_is_directory=True)
    with pytest.raises(ValueError, match="col_stats path must not redirect"):
        ColumnCache(index, cache).destination("cutoff/table.col_stats.json")


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
def indexed(tmp_path, monkeypatch):
    import kumo.settings

    monkeypatch.setattr(kumo.settings, "_configured", tmp_path)
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
    spec = contract(schema)
    engine = api.DiskGraphEngine(num_threads=1)
    index = engine.index
    with api.prepare(spec, dataset, engine):
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


def test_managed_index_without_sources_or_base_writes(indexed, tmp_path, monkeypatch):
    index, source, dataset, spec, anchors = indexed
    shutil.rmtree(source)
    before = inventory(index)
    original_ingest = diskgraph.ingest
    extensions = []

    def ingest(*args, **kwargs):
        assert kwargs.get("parent_index_path") == str(index)
        assert Path(args[1]).parent == tmp_path / "tasks" / index.name
        extensions.append(args[1])
        return original_ingest(*args, **kwargs)

    monkeypatch.setattr(diskgraph, "ingest", ingest)
    with read_only(index):
        engine = api.DiskGraphEngine.from_snapshot(snapshot(index), num_threads=1)
        assert engine.schema == spec.schema
        sample(engine, spec, anchors)
        assert extensions
        spec_path = tmp_path / "contract.json"
        spec_path.write_text(spec.to_json())
        snapshot_file = tmp_path / "snapshot.json"
        snapshot_file.write_text(json.dumps({"id": "graph@version", "index_path": str(index)}))
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from pathlib import Path
import kumo.settings
from kumo.pytorch import api
root, snapshot_file, spec, anchors = sys.argv[1:]
kumo.settings.configure(scratch_dir=root)
from types import SimpleNamespace
import json
engine = api.DiskGraphEngine.from_snapshot(
    SimpleNamespace(**json.loads(Path(snapshot_file).read_text())), num_threads=1
)
contract = api.DataLoaderSpec.from_json(Path(spec).read_text())
with api.prepare(contract=contract, engine=engine) as plan:
    with plan.materialize(api.AnchorTable(api.Parquet(anchors), 'user', 'at', 'sample')) as task:
        graph = next(iter(task.batches(output_format='columnar'))).inputs['graph']
        assert set(graph.nodes['events'].ids.to_pylist()) == {'a', 'b', 'd'}
""",
                str(tmp_path),
                str(snapshot_file),
                str(spec_path),
                anchors.source.path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    assert inventory(index) == before
    assert all(not Path(path).exists() for path in extensions)
    assert not (index / "tasks").exists()


def test_snapshot_binding_needs_no_pytorch_manifest(indexed, tmp_path):
    index, source, _, spec, anchors = indexed
    config = (index / "diskgraph_config.json").read_text()
    index = tmp_path / "snapshots" / "graph" / "version"
    diskgraph.ingest(
        config,
        str(index),
        col_stats=False,
        success_marker_content="{}",
        index_build_id=diskgraph.new_index_build_id(),
    )
    # The real engine integration test covers connector -> GraphSnapshot creation.
    snapshot = SimpleNamespace(index_path=index, id="graph@version", graph_config={})
    engine = api.DiskGraphEngine.from_snapshot(snapshot, num_threads=1)
    shutil.rmtree(source)
    before = inventory(index)
    sample(engine, spec, anchors)  # Custom logical relation aliases resolve by endpoint AND key.
    assert inventory(index) == before


def test_external_fitting_statistics_tokenizer_and_lengths(indexed, tmp_path):
    index, _, _, spec, _ = indexed
    before = inventory(index)
    engine = api.DiskGraphEngine.from_snapshot(snapshot(index), num_threads=1)
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
    monkeypatch.setattr(diskgraph._c, "_cutoff_col_stats_at", unexpected_compute)
    engine = api.DiskGraphEngine.from_snapshot(snapshot(index))
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
    from diskgraph._c import _cutoff_col_stats_at

    original = _cutoff_col_stats_at

    def compute(index_path, output_dir, request_json, **options):
        request = json.loads(request_json)
        assert request["columns"] == {"events": {"value": ["categorical"]}}
        assert not Path(output_dir).is_relative_to(index)
        calls.append(request)
        return original(index_path, output_dir, request_json, **options)

    monkeypatch.setattr(diskgraph._c, "_cutoff_col_stats_at", compute)
    engine = api.DiskGraphEngine.from_snapshot(snapshot(index), num_threads=1)
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
    native = tmp_path / "indexes" / diskgraph.new_index_build_id()
    diskgraph.ingest(
        json.dumps(config),
        str(native),
        col_stats=False,
        success_marker_content="{}",
        index_build_id=native.name,
    )
    shutil.rmtree(source)
    anchor_path = tmp_path / "precise-anchors.parquet"
    pq.write_table(
        pa.table({"user": [-2, 7], "at": pa.array([1002, 1002], pa.timestamp(unit))}), anchor_path
    )
    engine = api.DiskGraphEngine.from_snapshot(snapshot(native), num_threads=1)
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
            api.DiskGraphEngine.from_snapshot(snapshot(index), num_threads=1),
            spec,
            anchors,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert len(list(pool.map(run, range(2)))) == 2
    assert inventory(index) == before


def test_projection_alias_validation_and_bad_bindings(indexed, tmp_path):
    index, _, dataset, spec, anchors = indexed
    engine = api.DiskGraphEngine.from_snapshot(snapshot(index))
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


def test_physically_present_self_relation(indexed, tmp_path):
    index, source, _, spec, anchors = indexed
    shutil.rmtree(source)
    engine = api.DiskGraphEngine.from_snapshot(snapshot(index), num_threads=1)
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
        api.DiskGraphEngine.from_snapshot(snapshot(index))


def test_rebuilt_managed_index_changes_plan_identity(indexed, tmp_path):
    _, _, dataset, spec, _ = indexed
    first_engine = api.DiskGraphEngine(num_threads=1)
    with api.prepare(spec, dataset, first_engine) as first:
        identity = first._dataset_fingerprint
    shutil.rmtree(first_engine.index)
    second_engine = api.DiskGraphEngine(num_threads=1)
    with api.prepare(spec, dataset, second_engine) as second:
        assert second._dataset_fingerprint != identity


def test_managed_layout_and_physical_identity(indexed, tmp_path):
    index, _, dataset, spec, anchors = indexed
    build = index.name
    assert index.parent == tmp_path / "indexes"
    assert json.loads((index / "index_metadata.json").read_text())["index_build_id"] == build
    before = inventory(index)
    engine = api.DiskGraphEngine.from_snapshot(snapshot(index), num_threads=1)
    with api.prepare(contract=spec, engine=engine) as plan:
        source = plan._source
        source.column_stats([("events", "value", "categorical")], 20 * 10**9)
        assert (tmp_path / "cache" / build / "col_stats").is_dir()
        with plan.materialize(anchors) as task:
            task_root = tmp_path / "tasks" / build
            task_dirs = [
                path for path in task_root.iterdir() if not path.name.startswith("scratch-")
            ]
            assert len(task_dirs) == 1
            metadata = json.loads((task_dirs[0] / "index_metadata.json").read_text())
            assert metadata["parent_index_build_id"] == build
            assert next(iter(task.batches(output_format="columnar"))).inputs["graph"]
        assert not list(task_root.iterdir())
    assert inventory(index) == before
    second = api.DiskGraphEngine(num_threads=1)
    with api.prepare(spec, dataset, second):
        pass
    assert second.index_build_id != build
    assert second.index.parent == index.parent


def test_independent_builds_keep_node_indexed_caches_separate(indexed, tmp_path):
    first_index, _, dataset, spec, _ = indexed
    second = api.DiskGraphEngine(num_threads=1)
    with api.prepare(spec, dataset, second):
        pass
    roots = []
    for index in (first_index, second.index):
        engine = api.DiskGraphEngine.from_snapshot(snapshot(index))
        with api.prepare(contract=spec, engine=engine) as plan:
            source = plan._source
            encoder = TableColumnEncoder(
                api.Text(vocabulary_size=300, min_frequency=1),
                api.DType("string"),
                (("events", "note"),),
            )
            tokenizer, _ = source.column_text(encoder, 20 * 10**9)
            file = source.token_lengths("events", [("note", tokenizer, 0)], [], 20 * 10**9)
            root = source._cache_root
            _, lengths = read_token_lengths(index, {"file": file, "constant": 0}, cache_root=root)
            assert len(lengths) == 4
            roots.append(root)
    assert roots == [tmp_path / "cache" / first_index.name, tmp_path / "cache" / second.index.name]
    assert roots[0] != roots[1]


@pytest.mark.parametrize("from_snapshot", [False, True])
def test_engine_keeps_managed_root_after_reconfigure(indexed, tmp_path, from_snapshot):
    import kumo.settings

    index, _, dataset, spec, anchors = indexed
    engine = (
        api.DiskGraphEngine.from_snapshot(snapshot(index), num_threads=1)
        if from_snapshot else api.DiskGraphEngine(num_threads=1)
    )
    base = engine.index
    new_root = tmp_path / "new-root"
    kumo.settings.configure(scratch_dir=new_root)

    prepare_options = {} if from_snapshot else {"dataset": dataset}
    with api.prepare(contract=spec, engine=engine, **prepare_options) as plan:
        plan._source.column_stats([("events", "value", "categorical")], 20 * 10**9)
        assert (tmp_path / "cache" / engine.index_build_id / "col_stats").is_dir()
        with plan.materialize(anchors) as task:
            task_root = tmp_path / "tasks" / engine.index_build_id
            assert any(task_root.iterdir())
            assert next(iter(task.batches(output_format="columnar"))).inputs["graph"]
    assert base.parent == tmp_path / "indexes"
    assert base.is_dir()
    assert not new_root.exists()
    assert api.DiskGraphEngine().index.parent == new_root / "indexes"


def test_snapshot_binding_detects_replaced_metadata(indexed):
    index, _, _, spec, _ = indexed
    engine = api.DiskGraphEngine.from_snapshot(snapshot(index))
    metadata = index / "index_metadata.json"
    value = json.loads(metadata.read_text())
    value["index_build_id"] = "replacement-build"
    metadata.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="metadata changed after binding"):
        api.prepare(contract=spec, engine=engine)


def test_core_snapshot_requires_same_root_and_physical_id(indexed, tmp_path):
    index, _, _, _, _ = indexed
    outside = tmp_path.parent / "outside-snapshot"
    with pytest.raises(ValueError, match="configured Kumo root"):
        api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=outside, id="old"))
    with pytest.raises(ValueError, match="cannot be adopted"):
        legacy = tmp_path / "legacy"
        shutil.copytree(index, legacy)
        metadata = legacy / "index_metadata.json"
        value = json.loads(metadata.read_text())
        value.pop("index_build_id")
        metadata.write_text(json.dumps(value))
        api.DiskGraphEngine.from_snapshot(SimpleNamespace(index_path=legacy, id="old"))


@pytest.mark.parametrize("root", ["tasks", "cache"])
def test_managed_writable_roots_cannot_overlap_base(indexed, tmp_path, root):
    index, _, _, _, _ = indexed
    path = tmp_path / root / index.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(index, target_is_directory=True)
    with pytest.raises(ValueError, match="disjoint"):
        api.DiskGraphEngine.from_snapshot(snapshot(index))


def test_managed_col_stats_redirect_is_refused(indexed, tmp_path):
    index, _, _, _, _ = indexed
    cache = tmp_path / "cache" / index.name
    cache.mkdir(parents=True)
    (cache / "col_stats").symlink_to(index / "col_stats", target_is_directory=True)
    with pytest.raises(ValueError, match="col_stats path must not redirect"):
        api.DiskGraphEngine.from_snapshot(snapshot(index))


def test_external_statistics_reuse_overlapping_requests(indexed, tmp_path, monkeypatch):
    index, _, _, spec, _ = indexed
    known = ("events", "value", "numerical")
    missing = ("events", "value", "categorical")
    engine = api.DiskGraphEngine.from_snapshot(snapshot(index), num_threads=1)
    with api.prepare(contract=spec, engine=engine) as plan:
        source = plan._source
        initial = source.column_stats([known], 20 * 10**9)
        from diskgraph._c import _cutoff_col_stats_at

        original = _cutoff_col_stats_at
        calls = []

        def compute(index_path, output_dir, request_json, **options):
            request = json.loads(request_json)
            assert request["columns"] == {"events": {"value": ["categorical"]}}
            calls.append(request)
            return original(index_path, output_dir, request_json, **options)

        monkeypatch.setattr(diskgraph._c, "_cutoff_col_stats_at", compute)
        result = source.column_stats([known, missing], 20 * 10**9)
        assert result.columns[known]["mean"] == initial.columns[known]["mean"]
        assert result.files == initial.files
        assert len(calls) == 1
        assert source.column_stats([missing], 20 * 10**9).files == initial.files
        assert len(calls) == 1


def test_source_ingest_refuses_column_renames(indexed, tmp_path):
    _, _, dataset, spec, _ = indexed
    engine = api.DiskGraphEngine()
    with api.prepare(spec, dataset, engine):
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
        api.prepare(spec, renamed, engine)


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
    engine = api.DiskGraphEngine(num_threads=1)
    index = engine.index
    with pytest.raises(ValueError, match="changed while DiskGraph was ingesting"):
        api.prepare(spec, dataset, engine)
    assert not index.exists()


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
