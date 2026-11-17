# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated-case machinery: wrong engines fail every property, repros round-trip."""

import json
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conformance.assertions import bound_tables, write_parquet
from conformance.cases import ENGINES, REPOSITORY
from conformance.generate import (
    PHYSICAL_TYPES,
    STRATEGIES,
    WINDOW_STARTS,
    GeneratedCase,
    _pairs_of,
    delivery_configs,
    entity_cutoffs,
    generate_batch,
    pairwise,
    required_pairs,
)
from conformance.graph_cases import GRAPH_CHECKS
from conformance.properties import (
    PROPERTIES,
    SEEDS,
    PropertyFailure,
    answer,
    obligations,
    replay,
    run_obligation,
    seed_batches,
)
from conformance.registry import (
    BUNDLED,
    BY_NAME,
    DELIVERY,
    KNOWN_GAPS,
    REPLAYED,
    ROTATED,
    SENSITIVE,
    active_values,
    blocked_regions,
)
from conformance.repro import CORPUS, load_corpus, load_repro, repro_payload, write_repro
from conformance.sequence_cases import SEQUENCE_CHECKS
from conformance.variants import VARIANTS
from kumo.pytorch.api import (
    Dataset,
    InMemoryEngine,
    InMemoryRows,
    Parquet,
    ValidationIssue,
)
from kumo.pytorch.contract.sample import BuiltSample
from kumo.pytorch.data.records import FieldValue
from kumo.pytorch.engines.in_memory.anchors import InMemoryAnchors

CASES = generate_batch(0, "in_memory")
FILTERS = "sample.entity_values.filtered.source.filters"


def _case(**dimensions):
    return next(
        (position, case)
        for position, case in enumerate(CASES)
        if all(case.dimensions[name] == value for name, value in dimensions.items())
    )


def _shifted(sample: BuiltSample, delta: float) -> BuiltSample:
    values = tuple(
        FieldValue(field.key, field.value + delta, field.dtype)
        if type(field.value) in (int, float)
        else field
        for field in sample.history.entity.values
    )
    entity = replace(sample.history.entity, values=values)
    return replace(sample, history=replace(sample.history, entity=entity))


class Mutant(InMemoryEngine):
    """The reference with exactly one deliberate defect."""

    def __init__(self, defect: str):
        super().__init__()
        self.defect = defect
        self.calls = self.orders = self.opened = 0

    def open(self, dataset, schema):
        tables = {name: list(source.rows) for name, source in dataset.tables.items()}
        if self.defect == "first_row_dropped":
            tables["events"] = tables["events"][1:]
        source = super().open(Dataset({n: InMemoryRows(r) for n, r in tables.items()}), schema)
        build, columns = source.build, sum(len(table.columns) for table in schema.tables)

        def defective(spec, requests):
            self.calls += 1
            if self.defect == "future_visible":
                requests = [replace(r, as_of_time=r.as_of_time + 10**6) for r in requests]
            if self.defect == "absolute_time":
                requests = [replace(r, as_of_time=r.as_of_time // 1000 * 1000) for r in requests]
            if self.defect == "late_refusal" and any(
                getattr(value.source, "filters", ())
                for value in spec.entity_values.values()
                if hasattr(value, "source")
            ):
                raise ValueError("row filters are not supported")
            if self.defect == "growing_work":
                source.last_temporal_profile = {"native_stats": {"profiled": self.calls}}
            delta = {
                "reads_declared_columns": float(columns),
                "noise": random.random(),
                "off_by_one": 1.0,
            }.get(self.defect, 0.0)
            return [_shifted(sample, delta) for sample in build(spec, requests)]

        source.build = defective
        return source

    def check_contract(self, contract):
        filtered = any(
            getattr(value.source, "filters", ())
            for value in contract.sample.entity_values.values()
            if hasattr(value, "source")
        )
        if self.defect == "refusal_reads_rows":
            list(self._dataset.tables["events"].rows)
        refusing = ("no_suggestion", "refusal_reads_rows", "vague_refusal", "unnamed_refusal")
        if filtered and self.defect in refusing:
            suggestion = None if self.defect == "no_suggestion" else "use another engine"
            path = "sample" if self.defect == "vague_refusal" else FILTERS
            issue = ValidationIssue(path, "row filters are unsupported", suggestion)
            if self.defect == "unnamed_refusal":
                return (issue, ValidationIssue("sample", "unsupported", suggestion))
            return (issue,)
        return ()

    def materialize_anchors(self, table):
        anchors = InMemoryAnchors(table)
        self.opened += 1
        if self.defect in ("anchor_dropped", "unstable_order", "no_shuffle", "reseeded_on_open"):
            order, opened = anchors.order, self.opened

            def defective(selection, **arguments):
                self.orders += 1
                if self.defect == "reseeded_on_open":
                    return order(selection, **{**arguments, "seed": arguments["seed"] + opened})
                if self.defect == "no_shuffle":
                    arguments.update(seed=0, epoch=0)
                positions = list(order(selection, **arguments))
                if self.defect == "anchor_dropped":
                    return iter(positions[:-1])
                if self.defect == "no_shuffle":
                    return iter(positions)
                return iter(positions[self.orders % 2 :] + positions[: self.orders % 2])

            anchors.order = defective
        return anchors


class ParquetReference(InMemoryEngine):
    """Reads Parquet into the reference, with a defect in how files are read."""

    def __init__(self, defect: str):
        super().__init__()
        self.defect = defect

    @staticmethod
    def check_sources(kinds, path):
        return ()

    def open(self, dataset, schema):
        tables = {}
        for name, source in dataset.tables.items():
            path = Path(source.path)
            files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
            if self.defect == "first_file_only" and name != "users":
                files = files[:1]
            rows = []
            for file in files:
                try:
                    table = pq.read_table(file)
                except Exception:
                    if self.defect == "corrupt_file_skipped":
                        continue
                    if self.defect == "bare_error":
                        raise RuntimeError("boom") from None
                    raise
                for column in table.column_names:
                    if pa.types.is_timestamp(table.schema.field(column).type):
                        index = table.column_names.index(column)
                        seconds = table[column].cast(pa.timestamp("s")).cast(pa.int64())
                        table = table.set_column(index, column, seconds)
                rows.extend(table.to_pylist())
            tables[name] = InMemoryRows(rows)
        return super().open(Dataset(tables), schema)


def _binding(native: InMemoryEngine | None = None, reference: Mutant | None = None):
    """Stand in for one engine of `bound_tables`, without the native package."""

    @contextmanager
    def binding(contract, tables, engine_name, *, refusal=False, parts=1, types=None):
        if engine_name == "in_memory" or isinstance(native, Mutant):
            with bound_tables(contract, tables, "in_memory", refusal=refusal) as (dataset, engine):
                chosen = (reference if engine_name == "in_memory" else native) or engine
                chosen._dataset = dataset
                yield dataset, chosen
            return
        with tempfile.TemporaryDirectory() as directory:
            bindings = {}
            for table in contract.schema.tables:
                path = Path(directory) / (table.name if parts > 1 else f"{table.name}.parquet")
                if not refusal:
                    types_of = (types or {}).get(table.name)
                    write_parquet(table, tables[table.name], path, parts=parts, types=types_of)
                bindings[table.name] = Parquet(str(path))
            yield Dataset(bindings), native

    return binding


#: Property -> (case dimensions, stand-in engine kind, its defect, expected failure).
WRONG_ENGINES = {
    "point_in_time": ({}, "reference", "future_visible", "future rows changed samples"),
    "refusal_timing": (
        {"contract.strategy": "row_filter"},
        "reference",
        "late_refusal",
        r"accepted at check\(\), then",
    ),
    "differential": ({}, "native", "off_by_one", "vs reference changed samples"),
    "permutation": ({}, "reference", "first_row_dropped", "row and request order changed"),
    "partition": ({}, "parquet", "first_file_only", "files per table changed samples"),
    "time_shift": ({}, "reference", "absolute_time", "shift changed samples"),
    "irrelevance": ({}, "reference", "reads_declared_columns", "an unrelated .* changed"),
    "delivery": ({}, "reference", "anchor_dropped", "one pass served|not exactly once"),
    "epoch_trend": ({}, "reference", "growing_work", "!= epoch 2"),
    "truncated_parquet": (
        {"source.files": "split"},
        "parquet",
        "corrupt_file_skipped",
        "was accepted",
    ),
}
MORE_DEFECTS = (
    ("refusal_timing", "no_suggestion", "needs a suggestion"),
    ("refusal_timing", "refusal_reads_rows", "refusal read rows"),
    ("refusal_timing", "vague_refusal", "no refusal is as specific as " + re.escape(FILTERS)),
    ("refusal_timing", "unnamed_refusal", "refusal names none of"),
    ("delivery", "unstable_order", "same seed, different output|resume"),
    ("delivery", "noise", "differs from one pass"),
    ("delivery", "no_shuffle", "shuffle ignores the epoch or the seed"),
    ("delivery", "reseeded_on_open", "reopening changed the shuffled order"),
    ("truncated_parquet", "bare_error", "doesn't name Parquet"),
)


def _check(name: str, defect: str) -> None:
    dimensions, kind, _, _ = WRONG_ENGINES[name]
    position, case = _case(**dimensions)
    if kind == "reference":
        binding, engine = _binding(reference=Mutant(defect)), "in_memory"
    else:
        native = Mutant(defect) if kind == "native" else ParquetReference(defect)
        binding, engine = _binding(native=native), "diskgraph"
    with patch("conformance.properties.bound_tables", binding):
        PROPERTIES[name].check(case, engine, position)


def test_every_property_has_a_wrong_engine():
    # That module imports this one, so it is imported here.
    from test_functional_harness import FUNCTIONAL_COVERED

    # Graph and sequence properties have theirs in test_sampling_harness.py.
    sampling = GRAPH_CHECKS.keys() | SEQUENCE_CHECKS.keys()
    assert WRONG_ENGINES.keys() | FUNCTIONAL_COVERED == PROPERTIES.keys() - sampling


def test_the_parquet_stand_in_answers_like_the_reference():
    position, case = _case()
    expected = answer(case.contract, case.tables, case.requests, "in_memory")
    with patch("conformance.properties.bound_tables", _binding(native=ParquetReference("none"))):
        for parts in (1, 2):
            actual = answer(case.contract, case.tables, case.requests, "diskgraph", parts=parts)
            assert actual == expected and any(sample["count"] for sample in actual.values())


@pytest.mark.parametrize("name", sorted(WRONG_ENGINES))
def test_the_stand_in_engine_passes_without_its_defect(name):
    _check(name, "none")


@pytest.mark.parametrize(
    "name,defect,expected",
    [(name, defect, expected) for name, (_, _, defect, expected) in sorted(WRONG_ENGINES.items())]
    + list(MORE_DEFECTS),
)
def test_a_wrong_engine_fails_the_property(name, defect, expected):
    with pytest.raises(PropertyFailure, match=expected):
        _check(name, defect)


def test_longer_runs_add_seed_batches_and_keep_the_required_ones():
    for engine in ENGINES:
        assert seed_batches(engine) == SEEDS[engine]
        assert set(SEEDS[engine]) < set(seed_batches(engine, 8)) == set(range(8))
        rotated = seed_batches(engine, 4, 1000)
        assert set(rotated) == set(SEEDS[engine]) | {1000, 1001, 1002, 1003}
        with pytest.raises(ValueError, match="positive count"):
            seed_batches(engine, 0)
    assert set(obligations(ENGINES)) < set(obligations(ENGINES, 8))
    assert set(obligations(ENGINES)) < set(obligations(ENGINES, 4, 1000))


def test_an_empty_batch_fails_the_obligation():
    with patch("conformance.properties.generate_batch", lambda seed, engine: []):
        with pytest.raises(PropertyFailure, match="the batch has no case"):
            run_obligation("point_in_time", "in_memory", 0)


def test_a_failing_obligation_writes_a_repro_that_replays(tmp_path):
    binding = _binding(None, Mutant("future_visible"))
    with patch("conformance.properties.bound_tables", binding):
        with pytest.raises(PropertyFailure, match="repro: ") as failure:
            run_obligation("point_in_time", "in_memory", 0, tmp_path)
        paths = sorted(tmp_path.glob("repro-*.json"))
        assert paths and all(str(path) in str(failure.value) for path in paths)
        repro = load_repro(paths[0])
        with pytest.raises(AssertionError):
            replay(repro)
    replay(repro)  # the same file passes on the correct reference
    payload = json.loads(paths[0].read_text())
    assert payload["engine"] == "in_memory" and payload["seed"] == 0
    assert set(payload["versions"]) == {"package", "diskgraph", "registry", "python"}
    assert GeneratedCase.from_dict(payload["case"]).to_dict() == payload["case"]


def test_repro_files_are_named_by_content(tmp_path):
    position, case = _case()
    first = write_repro("permutation", "in_memory", case, position, ValueError("a"), tmp_path)
    second = write_repro("permutation", "in_memory", case, position, ValueError("b"), tmp_path)
    assert first == second
    payload = repro_payload("permutation", "in_memory", case, position, ValueError("a"))
    for damage in ({"issue": ""}, {"extra": 1}, {"format_version": 2}, {"seed": 99}):
        broken = tmp_path / "broken" / first.name
        broken.parent.mkdir(exist_ok=True)
        broken.write_text(json.dumps({**payload, **damage}))
        with pytest.raises(ValueError):
            load_repro(broken, require_issue="issue" in damage)
    renamed = tmp_path / "repro-0000000000000000.json"
    renamed.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="file name"):
        load_repro(renamed)


def test_corpus_loader(tmp_path):
    assert CORPUS.is_dir() and isinstance(load_corpus(), dict)
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")
    assert load_corpus(tmp_path) == {}
    position, case = _case()
    path = write_repro("time_shift", "in_memory", case, position, ValueError("x"), tmp_path)
    with pytest.raises(ValueError, match="issue"):
        load_corpus(tmp_path)
    path.write_text(json.dumps({**json.loads(path.read_text()), "issue": "#1"}))
    entry = load_corpus(tmp_path)[path.stem]
    assert entry.issue == "#1" and entry.case.to_dict() == case.to_dict()
    replay(entry, "in_memory")


@pytest.mark.parametrize("seed", (0, 1))
@pytest.mark.parametrize("engine", ENGINES)
def test_batches_cover_every_sensitive_pair_outside_known_gaps(seed, engine):
    cases = generate_batch(seed, engine)
    covered = set().union(*(_pairs_of(case.dimensions) for case in cases))
    sensitive = {
        key: [n for n in names if n in cases[0].dimensions] for key, names in SENSITIVE.items()
    }
    domains = {name: BY_NAME[name].values for name in cases[0].dimensions}
    regions = blocked_regions(engine)
    required = {
        pair
        for pair in required_pairs(sensitive, domains)
        if not any(region.covers(dict(pair)) for region in regions)
    }
    assert required <= covered
    assert not any(region.covers(case.dimensions) for region in regions for case in cases)
    for name in ROTATED:
        assert {case.dimensions[name] for case in cases} == set(active_values(name, engine))
    assert [case.to_dict() for case in generate_batch(seed, engine)] == [c.to_dict() for c in cases]
    configs = delivery_configs(seed, engine)
    assert required_pairs({"d": DELIVERY}, {n: BY_NAME[n].values for n in DELIVERY}) <= set().union(
        *(_pairs_of(config) for config in configs)
    )


def test_pairwise_avoids_a_known_gap_region():
    gap = replace(KNOWN_GAPS[0], when={"time.ties": ("tied",), "values.null_rate": ("all",)})
    domains = {"time.ties": ("distinct", "tied"), "values.null_rate": ("none", "all")}
    rows = pairwise(domains, required_pairs({"x": list(domains)}, domains), random.Random(0), [gap])
    assert {"time.ties": "tied", "values.null_rate": "all"} not in rows and len(rows) == 3


def _eligible(case: GeneratedCase) -> Mapping[str, Sequence[int]]:
    owner = {row["account_id"]: row["user_id"] for row in case.tables.get("accounts", [])}
    times: dict[int, list[int]] = {}
    for row in case.tables["events"]:
        user = owner[row["account_id"]] if "account_id" in row else row["user_id"]
        times.setdefault(user, []).append(row["at"])
    return {
        str(request.sample_id): [
            t for t in times.get(request.entity_id, []) if t <= request.as_of_time
        ]
        for request in case.requests
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_generated_cases_realize_their_dimensions(case):
    dims, eligible = case.dimensions, _eligible(case)
    counts = sorted(len(times) for times in eligible.values())
    assert (0 in counts) == (dims["values.empty_window"] == "present")
    assert (1 in counts) == (dims["values.single_row"] == "present")
    owner = {row["account_id"]: row["user_id"] for row in case.tables.get("accounts", [])}
    times = [
        (owner[row["account_id"]] if "account_id" in row else row["user_id"], row["at"])
        for row in case.tables["events"]
    ]
    assert (len(set(times)) < len(times)) == (dims["time.ties"] == "tied")
    cutoffs = entity_cutoffs(case.requests)
    assert (len({c for cs in cutoffs.values() for c in cs}) > 1) == (
        dims["time.cutoffs"] == "mixed"
    )
    if dims["time.cutoffs"] == "mixed":
        assert any(len(set(cs)) > 1 for cs in cutoffs.values())
    offset = {"before": -1, "at": 0, "after": 1}[dims["time.cutoff_placement"]]
    event_times = {row["at"] for row in case.tables["events"]}
    assert {r.as_of_time + offset for r in case.requests} & event_times
    window = {r.as_of_time + WINDOW_STARTS["from_row"] for r in case.requests}
    assert window & event_times and {t - 1 for t in window} & event_times
    nulls = {row["amount"] is None for row in case.tables["events"]}
    assert (
        nulls == {"none": {False}, "some": {False, True}, "all": {True}}[dims["values.null_rate"]]
    )
    negative = any(row["at"] < 0 for row in case.tables["events"])
    assert negative == (dims["time.range"] == "pre_epoch")
    assert set(PHYSICAL_TYPES) <= set(case.tables)


def test_registry_names_what_generators_use():
    assert set(STRATEGIES) | set(VARIANTS) == set(BY_NAME["contract.strategy"].values)
    used = {name for names in SENSITIVE.values() for name in names} | set(DELIVERY)
    assert used | set(BUNDLED) | set(ROTATED) <= set(BY_NAME)
    assert BY_NAME["physical_type.columns"].deferred == (
        "deferred: generated type variation; scalar pairs use a fixed matrix"
    )
    assert all((REPOSITORY / gap.repro).is_file() for gap in REPLAYED)
