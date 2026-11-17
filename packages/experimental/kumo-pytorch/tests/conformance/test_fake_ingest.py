# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""kumo-pytorch's side of DiskGraph ingest, with a recorder in place of the native call (no wheel).

The config kumo-pytorch compiles for generated and hand-written schemas, the relation
mapping it publishes, and the index lifecycle: reuse, refusal, atomic publish.
Then the generated DiskGraph settings: each case runs at its own shards and threads.
"""

import inspect
import itertools
import json
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conformance.assertions import bound_tables, case_settings, require_engine, write_parquet
from conformance.cases import REPOSITORY
from conformance.fake_ingest import (
    FakeDiskGraph,
    IngestCall,
    config_violations,
    edge_relations,
    mapping_violations,
    option_violations,
    recording,
)
from conformance.generate import (
    PATHS,
    PHYSICAL_TYPES,
    GeneratedCase,
    Row,
    generate_batch,
    generate_case,
)
from conformance.properties import (
    PROPERTIES,
    PropertyFailure,
    irrelevant,
    obligation_cases,
    obligations,
    replay,
    run_obligation,
)
from conformance.registry import BY_NAME, CROSSED, KNOWN_GAPS, active_values
from conformance.repro import load_repro
from kumo.pytorch.api import (
    NON_TEMPORAL,
    DataSchema,
    Dataset,
    DiskGraphEngine,
    DType,
    InMemoryEngine,
    Parquet,
    RelationSpec,
    TableSpec,
)
from kumo.pytorch.contract.sample import BuiltSample
from kumo.pytorch.data.records import FieldValue
from kumo.pytorch.materialization import _DiskGraphIndex, ensure_diskgraph_index

Tables = Mapping[str, Sequence[Row]]
INT64 = DType("int64")
TIMESTAMP = DType("timestamp", unit="s", timezone="UTC")


@dataclass(frozen=True)
class Shape:
    id: str
    schema: DataSchema
    tables: Tables
    num_shards: int = 1
    num_threads: int | None = None


class _Source:
    """In place of DiskGraphSource: what the engine hands the sampler."""

    def __init__(self, artifact: _DiskGraphIndex, schema: DataSchema, *, num_threads: int | None):
        self.artifact, self.schema, self.num_threads = artifact, schema, num_threads


def _write(schema: DataSchema, tables: Tables, directory: Path, parts: int = 1) -> Dataset:
    bindings = {}
    for table in schema.tables:
        path = directory / (f"{table.name}.parquet" if parts == 1 else table.name)
        write_parquet(
            table, tables[table.name], path, parts=parts, types=PHYSICAL_TYPES.get(table.name)
        )
        bindings[table.name] = Parquet(str(path))
    return Dataset(bindings)


def _arrow_types(dataset: Dataset) -> dict[str, dict[str, str]]:
    types = {}
    for name, source in dataset.tables.items():
        path = Path(source.path)
        file = next(iter(sorted(path.glob("*.parquet")))) if path.is_dir() else path
        arrow = pq.read_schema(file)
        types[name] = {column: str(arrow.field(column).type) for column in arrow.names}
    return types


def _open(
    engine: DiskGraphEngine, dataset: Dataset, schema: DataSchema, fake: FakeDiskGraph
) -> _Source:
    with recording(fake), patch("kumo.pytorch.engines.diskgraph.DiskGraphSource", _Source):
        source = engine.open(dataset, schema)
    assert isinstance(source, _Source)
    return source


def _leftovers(index: Path) -> list[str]:
    return sorted(path.name for path in index.parent.glob(f".{index.name}-*"))


# -- The config, over generated and hand-written schemas ---------------------


def _generated_shapes() -> list[Shape]:
    shapes = []
    for case in generate_batch(0, "diskgraph") + generate_batch(0, "in_memory"):
        if case.dimensions["schema_shape.parallel_relation"] == "present":
            continue  # GAP-PARALLEL-FK-DG, checked below
        shards = int(case.dimensions.get("execution.num_shards", 1))
        threads = case.dimensions.get("execution.num_threads")
        shapes.append(
            Shape(
                f"{case.id}-{case.scale}",
                case.contract.schema,
                case.tables,
                shards,
                None if threads is None else int(threads),
            )
        )
    # Every path shape, including depth2_timed, which no obligation generates on either engine.
    first = generate_batch(0, "in_memory")[0]
    for shape in PATHS:
        case = generate_case(0, 0, {**first.dimensions, "schema_shape.path": shape}, "XS")
        shapes.append(Shape(shape, case.contract.schema, case.tables, 3, 4))
        for position in range(3):
            variant, contract, tables = irrelevant(case, position)
            shapes.append(Shape(f"{shape}-{variant}", contract.schema, tables, 2, 2))
    return shapes


def _hand_shapes() -> list[Shape]:
    users = TableSpec(
        "users", "user_id", {"user_id": INT64, "signup_at": TIMESTAMP}, time_column="signup_at"
    )
    accounts = TableSpec(
        "accounts",
        "account_id",
        {"opened_at": TIMESTAMP, "account_id": INT64, "user_id": INT64},
        time_column="opened_at",
    )
    events = TableSpec(
        "events",
        "event_id",
        {
            "amount": DType("float64"),
            "event_id": INT64,
            "account_id": INT64,
            "user_id": INT64,
            "at": TIMESTAMP,
        },
        time_column="at",
    )
    referrals = TableSpec(
        "people",
        "person_id",
        {"person_id": INT64, "referrer_id": INT64},
        time_column=NON_TEMPORAL,
    )
    rows: dict[str, list[Row]] = {
        "users": [{"user_id": 1, "signup_at": 10}, {"user_id": 2, "signup_at": 20}],
        "accounts": [{"opened_at": 30, "account_id": 5, "user_id": 1}],
        "events": [
            {"amount": 1.0, "event_id": 9, "account_id": 5, "user_id": 1, "at": 40},
            {"amount": 2.0, "event_id": 8, "account_id": 5, "user_id": 2, "at": 50},
        ],
        "people": [{"person_id": 1, "referrer_id": 2}, {"person_id": 2, "referrer_id": 1}],
    }
    forward = RelationSpec("user_events", "users", "user_id", "events", "user_id")
    backward = RelationSpec("event_user", "events", "user_id", "users", "user_id")
    return [
        # Both ends timed: the forward and reverse edges take different clocks.
        Shape("two_timed_forward", DataSchema((users, events), (forward,)), rows),
        Shape("two_timed_reverse", DataSchema((events, users), (backward,)), rows, 3, 4),
        Shape(
            "two_fkeys_one_table",
            DataSchema(
                (users, accounts, events),
                (
                    backward,
                    RelationSpec(
                        "account_events", "accounts", "account_id", "events", "account_id"
                    ),
                    RelationSpec("account_user", "accounts", "user_id", "users", "user_id"),
                ),
            ),
            rows,
            2,
            1,
        ),
        Shape(
            "self_reference",
            DataSchema(
                (referrals,),
                (RelationSpec("referred_by", "people", "referrer_id", "people", "person_id"),),
            ),
            rows,
        ),
    ]


@pytest.mark.parametrize("shape", _generated_shapes() + _hand_shapes(), ids=lambda shape: shape.id)
def test_ingest_config_says_what_the_schema_declares(shape: Shape, tmp_path: Path) -> None:
    dataset = _write(shape.schema, shape.tables, tmp_path)
    index = tmp_path / "index"
    engine = DiskGraphEngine(
        index=index, num_shards=shape.num_shards, num_threads=shape.num_threads
    )
    fake = FakeDiskGraph()
    source = _open(engine, dataset, shape.schema, fake)

    (call,) = fake.calls
    types = _arrow_types(dataset)
    assert (
        config_violations(shape.schema, dataset, call.config, types, num_shards=shape.num_shards)
        == []
    )
    assert option_violations(call.options, num_threads=shape.num_threads) == []
    assert source.num_threads == shape.num_threads
    manifest = source.artifact.manifest
    edges = json.loads((index / "index_metadata.json").read_text())["edge_relations"]
    assert mapping_violations(shape.schema, manifest["relation_mapping"], edges) == []
    assert manifest["num_shards"] == shape.num_shards
    # Staged beside the index, so publishing is one same-filesystem rename.
    assert call.output_dir.parent.parent == index.parent and not call.output_dir.exists()
    assert (index / "index_manifest.json").is_file() and _leftovers(index) == []


def test_a_wrong_config_or_an_unknown_type_is_a_violation(tmp_path: Path) -> None:
    """Clock dtypes are checked whatever unit Parquet reads back; no type is skipped."""

    dataset = _write(SCHEMA, ROWS, tmp_path)
    fake = FakeDiskGraph()
    _open(DiskGraphEngine(index=tmp_path / "index"), dataset, SCHEMA, fake)
    (call,) = fake.calls
    types = _arrow_types(dataset)
    assert types["events"]["at"] == "timestamp[ms, tz=UTC]"
    config = json.loads(json.dumps(call.config))
    assert config["tables"][0]["columns"][0]["name"] == "at"
    config["tables"][0]["columns"][0]["ingested_dtype"] = "int64"
    config |= {"edge_format": "csv", "dataset_name": "x"}
    listed = {**types, "users": {"user_id": "list<item: int64>"}}
    assert config_violations(SCHEMA, dataset, config, listed, num_shards=1) == [
        "dataset_name 'x' != 'default'",
        "edge_format 'csv' != 'binary'",
        "events.at: dtype int64",
        "users.user_id: Arrow type list<item: int64> is not in the table",
    ]
    assert option_violations({**call.options, "col_stats": True}, num_threads=None) == [
        "col_stats=True != False"
    ]


def _parallel_gap_case() -> GeneratedCase:
    gap = next(gap for gap in KNOWN_GAPS if gap.id == "GAP-PARALLEL-FK-DG")
    return load_repro(REPOSITORY / gap.repro).case


def test_parallel_relation_emits_a_duplicate_fkey(tmp_path: Path) -> None:
    """F2 on kumo-pytorch's side, the root of GAP-PARALLEL-FK-DG; when this fails, lift that gap."""

    case = _parallel_gap_case()
    schema = case.contract.schema
    assert [r.left_key for r in schema.relations] == ["user_id", "user_id"]
    dataset = _write(schema, case.tables, tmp_path)
    fake = FakeDiskGraph()
    _open(DiskGraphEngine(index=tmp_path / "index"), dataset, schema, fake)
    (call,) = fake.calls
    problems = config_violations(schema, dataset, call.config, _arrow_types(dataset), num_shards=1)
    assert problems == ["events.user_id: 2 fkeys"]


def test_only_the_gap_keeps_parallel_relations_off_diskgraph() -> None:
    """Lifting GAP-PARALLEL-FK-DG generates them on DiskGraph; the reference has them."""

    assert BY_NAME["schema_shape.parallel_relation"].values == ("absent", "present")
    assert active_values("schema_shape.parallel_relation", "diskgraph") == ("absent",)
    for engine in ("in_memory", "diskgraph"):
        parallel = set()
        for case in generate_batch(0, engine):
            columns = [
                (r.left_table, r.left_key, r.right_table, r.right_key)
                for r in case.contract.schema.relations
            ]
            present = case.dimensions["schema_shape.parallel_relation"] == "present"
            assert (len(set(columns)) < len(columns)) == present, case.id
            parallel.add(present)
        assert parallel == ({False} if engine == "diskgraph" else {False, True})
    with patch("conformance.registry.KNOWN_GAPS", ()):
        assert active_values("schema_shape.parallel_relation", "diskgraph") == ("absent", "present")


# -- The index lifecycle -----------------------------------------------------

SCHEMA = DataSchema(
    tables=(
        TableSpec("users", "user_id", {"user_id": INT64}, time_column=NON_TEMPORAL),
        TableSpec(
            "events",
            "event_id",
            {"event_id": INT64, "user_id": INT64, "at": TIMESTAMP},
            time_column="at",
        ),
    ),
    relations=(RelationSpec("user_events", "users", "user_id", "events", "user_id"),),
)
ROWS: dict[str, list[Row]] = {
    "users": [{"user_id": 1}, {"user_id": 2}],
    "events": [{"event_id": 1, "user_id": 1, "at": 5}, {"event_id": 2, "user_id": 2, "at": 6}],
}


@contextmanager
def _built(tmp_path: Path, parts: int = 1) -> Iterator[tuple[Dataset, Path]]:
    dataset = _write(SCHEMA, ROWS, tmp_path, parts=parts)
    index = tmp_path / "index"
    with recording(FakeDiskGraph()):
        ensure_diskgraph_index(SCHEMA, dataset, index)
    yield dataset, index


def _ensure_again(
    dataset: Dataset, index: Path, schema: DataSchema = SCHEMA, **arguments: int | bool
) -> FakeDiskGraph:
    fake = FakeDiskGraph()
    with recording(fake):
        ensure_diskgraph_index(schema, dataset, index, **arguments)
    return fake


def _snapshot(index: Path) -> dict[str, bytes]:
    return {str(p.relative_to(index)): p.read_bytes() for p in index.rglob("*") if p.is_file()}


def test_an_unchanged_index_is_reused_without_ingest(tmp_path: Path) -> None:
    with _built(tmp_path) as (dataset, index):
        before = _snapshot(index)
        assert _ensure_again(dataset, index).calls == []
        assert _snapshot(index) == before


def _touch(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))


def _resize(path: Path) -> None:
    """Different content, same mtime."""

    stat = path.stat()
    at = pa.array([5, 6, 7], pa.timestamp("s"))
    pq.write_table(pa.table({"event_id": [1, 2, 3], "user_id": [1, 2, 2], "at": at}), path)
    assert path.stat().st_size != stat.st_size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


@pytest.mark.parametrize("change", ("mtime", "size", "extra_file"))
def test_a_changed_dataset_is_refused_not_reused(change: str, tmp_path: Path) -> None:
    """M06: the dataset fingerprint covers each file's path, size and mtime."""

    with _built(tmp_path, parts=2) as (dataset, index):
        before = _snapshot(index)
        files = sorted((tmp_path / "events").glob("*.parquet"))
        if change == "mtime":
            _touch(files[0])
        elif change == "size":
            mtime = files[0].stat().st_mtime_ns
            _resize(files[0])
            assert files[0].stat().st_mtime_ns == mtime
        else:
            (tmp_path / "events" / "part-9.parquet").write_bytes(files[0].read_bytes())
        fake = FakeDiskGraph()
        with recording(fake), pytest.raises(ValueError, match="different dataset"):
            ensure_diskgraph_index(SCHEMA, dataset, index)
        assert fake.calls == [] and _snapshot(index) == before


@pytest.mark.parametrize("change", ("column_type", "clock", "relation"))
def test_a_changed_schema_is_refused_not_reused(change: str, tmp_path: Path) -> None:
    """M20: the schema fingerprint covers column types, the clock and relations."""

    events, users = SCHEMA.tables
    changed = {
        "column_type": replace(
            SCHEMA,
            tables=(
                replace(events, columns={**events.columns, "user_id": DType("int32")}),
                users,
            ),
        ),
        "clock": replace(SCHEMA, tables=(replace(events, time_column=NON_TEMPORAL), users)),
        "relation": replace(
            SCHEMA,
            relations=(RelationSpec("event_user", "events", "user_id", "users", "user_id"),),
        ),
    }[change]
    with _built(tmp_path) as (dataset, index):
        before = _snapshot(index)
        fake = FakeDiskGraph()
        with recording(fake), pytest.raises(ValueError, match="different schema"):
            ensure_diskgraph_index(changed, dataset, index)
        assert fake.calls == [] and _snapshot(index) == before


def _changes_source(dataset: Dataset) -> FakeDiskGraph:
    def during(call: IngestCall) -> None:
        _touch(Path(dataset.tables["events"].path))

    return FakeDiskGraph(during)


def _creates_destination(index: Path) -> FakeDiskGraph:
    def during(call: IngestCall) -> None:
        index.mkdir()
        (index / "someone_else").write_text("x")

    return FakeDiskGraph(during)


def _fails(index: Path) -> FakeDiskGraph:
    def during(call: IngestCall) -> None:
        assert not index.exists(), "the destination exists before ingest finished"
        raise RuntimeError("native ingest failed")

    return FakeDiskGraph(during)


@pytest.mark.parametrize(
    "failure,error,message",
    (
        ("source_changed", ValueError, "Parquet files changed while DiskGraph was ingesting"),
        ("destination_appeared", FileExistsError, "index appeared during ingest"),
        ("ingest_failed", RuntimeError, "native ingest failed"),
    ),
)
def test_a_failed_build_publishes_nothing(
    failure: str, error: type[Exception], message: str, tmp_path: Path
) -> None:
    """M07 and the publish race: the index is complete or absent; no staging remains."""

    dataset = _write(SCHEMA, ROWS, tmp_path)
    index = tmp_path / "index"
    fake = {
        "source_changed": lambda: _changes_source(dataset),
        "destination_appeared": lambda: _creates_destination(index),
        "ingest_failed": lambda: _fails(index),
    }[failure]()
    with recording(fake), pytest.raises(error, match=message):
        ensure_diskgraph_index(SCHEMA, dataset, index)
    assert len(fake.calls) == 1 and _leftovers(index) == []
    if failure == "destination_appeared":
        assert _snapshot(index) == {"someone_else": b"x"}  # not overwritten
    else:
        assert not index.exists()


@pytest.mark.parametrize(
    "damage,message",
    (
        ("manifest_version", "unsupported DiskGraph index manifest version"),
        ("no_manifest", "missing the DiskGraph index manifest"),
        ("no_metadata", "missing DiskGraph index_metadata.json"),
    ),
)
def test_an_incompatible_index_is_refused_and_never_overwritten(
    damage: str, message: str, tmp_path: Path
) -> None:
    with _built(tmp_path) as (dataset, index):
        manifest = index / "index_manifest.json"
        if damage == "manifest_version":
            payload = json.loads(manifest.read_text())
            manifest.write_text(json.dumps({**payload, "manifest_version": 1}))
        else:
            (manifest if damage == "no_manifest" else index / "index_metadata.json").unlink()
        before = _snapshot(index)
        fake = FakeDiskGraph()
        with recording(fake), pytest.raises(ValueError, match=message):
            ensure_diskgraph_index(SCHEMA, dataset, index)
        assert fake.calls == [] and _snapshot(index) == before


def test_a_missing_index_is_not_built_when_ingest_is_never(tmp_path: Path) -> None:
    dataset = _write(SCHEMA, ROWS, tmp_path)
    fake = FakeDiskGraph()
    with recording(fake), pytest.raises(FileNotFoundError, match="does not exist"):
        ensure_diskgraph_index(SCHEMA, dataset, tmp_path / "index", if_missing=False)
    assert fake.calls == [] and not (tmp_path / "index").exists()


def _always_rebuilt(schema: DataSchema, dataset: Dataset, index: Path, **keywords: int | bool):
    if Path(index).exists():
        shutil.rmtree(index)
    return ensure_diskgraph_index(schema, dataset, index, **keywords)


@pytest.mark.parametrize(
    "layout,defect,expected",
    (
        ("as_built", False, None),
        ("as_built", True, "reopening as built rebuilt the index"),
        ("other_shards", False, "reused a 1-shard index for num_shards=3"),
    ),
)
def test_index_layout_reuses_only_the_layout_built(
    layout: str, defect: bool, expected: str | None
) -> None:
    """The last row is GAP-INDEX-LAYOUT-DG: today another shard count reuses the index."""

    base = generate_batch(0, "diskgraph")[0]
    case = replace(base, dimensions={**base.dimensions, "execution.index_layout": layout})
    rebuild = _always_rebuilt if defect else ensure_diskgraph_index
    with (
        recording(FakeDiskGraph()),
        patch("conformance.assertions.require_engine"),
        patch("kumo.pytorch.engines.diskgraph.DiskGraphSource", _Source),
        patch("kumo.pytorch.engines.diskgraph.ensure_diskgraph_index", rebuild),
        case_settings(case.dimensions),
    ):
        if expected is None:
            PROPERTIES["index_layout"].check(case, "diskgraph", 0)
            return
        with pytest.raises(PropertyFailure, match=expected):
            PROPERTIES["index_layout"].check(case, "diskgraph", 0)


def test_dropped_rows_are_reported(tmp_path: Path) -> None:
    dataset = _write(SCHEMA, ROWS, tmp_path)
    with recording(FakeDiskGraph(dropped={"events": 2, "users": 0})):
        with pytest.warns(UserWarning, match=r"dropped source rows \{'events': 2\}"):
            ensure_diskgraph_index(SCHEMA, dataset, tmp_path / "index")


@pytest.mark.engine("diskgraph")
def test_the_recorder_matches_the_pinned_diskgraph(tmp_path: Path) -> None:
    """The recorder's signature and what it writes are the real ingest's."""

    require_engine("diskgraph")
    import diskgraph

    real = inspect.signature(diskgraph.ingest)
    recorded = list(inspect.signature(FakeDiskGraph().ingest).parameters.values())
    assert [(p.name, p.default, p.kind) for p in recorded] == [
        (p.name, p.default, p.kind) for p in real.parameters.values()
    ]
    dataset = _write(SCHEMA, ROWS, tmp_path)
    fake = FakeDiskGraph()
    with recording(fake):
        ensure_diskgraph_index(SCHEMA, dataset, tmp_path / "recorded")
    ensure_diskgraph_index(SCHEMA, dataset, tmp_path / "real")
    (call,) = fake.calls
    for name in ("index_metadata.json", "ingest_stats.json"):
        recorded_file = json.loads((tmp_path / "recorded" / name).read_text())
        real_file = json.loads((tmp_path / "real" / name).read_text())
        if name == "ingest_stats.json":
            recorded_file, real_file = recorded_file["stats"], real_file["stats"]
        assert recorded_file.keys() <= real_file.keys(), name
    metadata = json.loads((tmp_path / "real" / "index_metadata.json").read_text())
    assert metadata["dataset_name"] == call.config["dataset_name"]
    assert metadata["timestamp_unit"] == "seconds"
    assert sorted(
        (edge["name"], edge["src_type"], edge["dst_type"]) for edge in metadata["edge_relations"]
    ) == sorted(
        (edge["name"], edge["src_type"], edge["dst_type"]) for edge in edge_relations(call.config)
    )


# -- Engine settings come from the case --------------------------------------


@pytest.mark.parametrize("seed", (0, 1))
def test_every_diskgraph_batch_crosses_shards_and_threads(seed: int) -> None:
    names = CROSSED["diskgraph"]
    expected = set(itertools.product(*(active_values(name, "diskgraph") for name in names)))
    cases = generate_batch(seed, "diskgraph")
    assert {tuple(case.dimensions[name] for name in names) for case in cases} == expected
    assert expected == {(1, 1), (1, 4), (3, 1), (3, 4)}
    assert not any(
        name in case.dimensions for case in generate_batch(seed, "in_memory") for name in names
    )


def test_every_diskgraph_obligation_sees_every_crossed_setting() -> None:
    """Per-case and per-batch obligations alike, graph and sequence ones included."""

    names = CROSSED["diskgraph"]
    expected = set(itertools.product(*(active_values(name, "diskgraph") for name in names)))
    for name, engine, seed in obligations(("diskgraph",)):
        cases = obligation_cases(name, engine, seed)
        seen = {tuple(case.dimensions[n] for n in names) for case in cases}
        assert seen == expected, name


def test_bindings_take_shards_and_threads_from_the_case() -> None:
    case = generate_batch(0, "in_memory")[0]
    with patch("conformance.assertions.require_engine"):
        with case_settings({"execution.num_shards": 3, "execution.num_threads": 4}):
            with bound_tables(case.contract, case.tables, "diskgraph", refusal=True) as bound:
                assert (bound[1].num_shards, bound[1].num_threads) == (3, 4)
        with bound_tables(case.contract, case.tables, "diskgraph", refusal=True) as bound:
            engine = bound[1]
            assert (engine.num_shards, engine.num_threads) == (1, 1)


class ShardSensitive(InMemoryEngine):
    """The reference, off by one in every float value when the index has several shards."""

    def __init__(self, num_shards: int):
        super().__init__()
        self.num_shards = num_shards

    def open(self, dataset, schema):
        source = super().open(dataset, schema)
        build = source.build

        def sharded(spec, requests) -> list[BuiltSample]:
            built = build(spec, requests)
            return built if self.num_shards == 1 else [_off_by_one(sample) for sample in built]

        source.build = sharded
        return source


def _off_by_one(sample: BuiltSample) -> BuiltSample:
    values = tuple(
        FieldValue(f.key, f.value + 1.0, f.dtype) if isinstance(f.value, float) else f
        for f in sample.history.entity.values
    )
    entity = replace(sample.history.entity, values=values)
    return replace(sample, history=replace(sample.history, entity=entity))


@contextmanager
def _settings_binding(contract, tables, engine_name, *, refusal=False, parts=1, types=None):
    """The real binding's DiskGraph settings, served by a shard-sensitive stand-in."""

    with patch("conformance.assertions.require_engine"):
        with bound_tables(contract, tables, "diskgraph", refusal=True) as (_, native):
            shards = native.num_shards
    with bound_tables(contract, tables, "in_memory", refusal=refusal) as (dataset, reference):
        yield dataset, reference if engine_name == "in_memory" else ShardSensitive(shards)


@pytest.mark.parametrize("shards", (1, 3))
def test_differential_judges_each_case_at_its_own_shards(shards: int, tmp_path: Path) -> None:
    base = generate_batch(0, "in_memory")[0]
    settings = {"execution.num_shards": shards, "execution.num_threads": 4}
    case = replace(base, dimensions={**base.dimensions, **settings})
    with (
        patch("conformance.properties.bound_tables", _settings_binding),
        patch("conformance.properties.generate_batch", lambda seed, engine: [case]),
    ):
        if shards == 1:
            run_obligation("differential", "diskgraph", 0, tmp_path)
            return
        with pytest.raises(PropertyFailure, match="vs reference changed samples"):
            run_obligation("differential", "diskgraph", 0, tmp_path)
        (path,) = tmp_path.glob("repro-*.json")
        with pytest.raises(PropertyFailure, match="vs reference changed samples"):
            replay(load_repro(path))
