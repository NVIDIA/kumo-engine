# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The generated-case machinery: wrong engines fail every property, repros round-trip."""

import copy
import json
import random
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conformance.assertions import bound_tables
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
    DiskGraphEngine,
    ValidationIssue,
)
from kumo.pytorch.contract.sample import BuiltSample
from kumo.pytorch.data.records import FieldValue
from kumo.pytorch.engines.diskgraph import DiskGraphSource
from kumo.pytorch.engines.native_anchors import NativeAnchors
from tests.native_fixtures import directory, parquet_dataset

CASES = generate_batch(0, "diskgraph")
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


class FaultAnchors(NativeAnchors):
    """Faults at the actual native anchor iteration boundary."""

    def _selection(self, selection):
        stages = super()._selection(selection)
        if self._fault.defect == "null_split_dropped":
            return [stage for stage in stages if stage["split"] is not None]
        return stages

    def batches(self, assembler, *args, **kwargs):
        owner = self._fault
        assembler = copy.copy(assembler)
        assembler._model_sample = owner.sample(assembler._model_sample)
        owner.orders += 1
        if owner.defect == "no_shuffle":
            kwargs.update(shuffle=False)
        elif owner.defect == "reseeded_on_open":
            kwargs["seed"] += owner.opened
        elif owner.defect == "unstable_order":
            kwargs["seed"] += owner.orders
        batches = list(super().batches(assembler, *args, **kwargs))
        if owner.defect == "anchor_dropped":
            batches = batches[:-1]
        if owner.defect == "union_unsorted":
            batches.reverse()
        yield from batches


class FaultSource(DiskGraphSource):
    """Corrupt native requests/results, including materialized task views."""

    def _fault_requests(self, requests):
        defect = self._fault.defect
        if defect in {"future_visible", "absolute_time"}:
            requests = [
                replace(r, as_of_time=int(r.as_of_time.timestamp()))
                if isinstance(r.as_of_time, datetime)
                else r
                for r in requests
            ]
        if defect == "future_visible":
            return [replace(r, as_of_time=r.as_of_time + 10**6) for r in requests]
        if defect == "absolute_time":
            return [replace(r, as_of_time=r.as_of_time // 1000 * 1000) for r in requests]
        return requests

    def _delta(self):
        defect = self._fault.defect
        return (
            sum(len(t.columns) for t in self.schema.tables)
            if defect == "reads_declared_columns"
            else random.random()
            if defect == "noise"
            else 1.0
            if defect == "off_by_one"
            else 0.0
        )

    def build(self, spec, requests):
        result = super().build(self._fault.sample(spec), self._fault_requests(requests))
        if self._fault.defect == "growing_work":
            self._fault.calls += 1
            self._last_temporal_profile = {"native_stats": {"profiled": self._fault.calls}}
        return [_shifted(sample, self._delta()) for sample in result]

    def build_model_ordered(self, spec, requests, **kwargs):
        result = super().build_model_ordered(
            self._fault.sample(spec), self._fault_requests(requests), **kwargs
        )
        return self._fault_model(result)

    def _fault_model(self, result):
        import pyarrow.compute as pc

        delta = self._delta()
        if not delta:
            return result
        return replace(
            result,
            entity_values={
                key: pc.add(values, pa.scalar(delta, type=values.type))
                if pa.types.is_integer(values.type) or pa.types.is_floating(values.type)
                else values
                for key, values in result.entity_values.items()
            },
        )

    def build_model_batch(self, spec, runtime, anchors, compiled):
        result = super().build_model_batch(spec, runtime, anchors, compiled)
        if self._fault.defect == "growing_work":
            self._fault.calls += 1
            self._last_temporal_profile = {"native_stats": {"profiled": self._fault.calls}}
        return self._fault_model(result)

    def materialize_scalar_batches(self, sample, anchors, batch_size):
        if self._fault.defect == "late_refusal":
            raise ValueError("row filters are unsupported")
        try:
            yield from super().materialize_scalar_batches(
                self._fault.sample(sample), anchors, batch_size
            )
        except ValueError:
            if self._fault.defect == "missing_root_bare":
                raise RuntimeError("lookup failed") from None
            raise

    def _cutoff_array(self, values):
        import pyarrow.compute as pc

        cutoffs = super()._cutoff_array(values)
        second = 10**9 // self._tick_nanos
        if self._fault.defect == "future_visible":
            return pc.add(cutoffs, 10**6 * second)
        if self._fault.defect == "absolute_time":
            return pa.array([t // (1000 * second) * (1000 * second) for t in cutoffs.to_pylist()])
        return cutoffs

    def materialize_anchors(self, table, sample, **kwargs):
        anchors = super().materialize_anchors(table, self._fault.sample(sample), **kwargs)
        anchors.__class__ = FaultAnchors
        anchors._fault = self._fault
        return anchors

    def _root_ids(self, table, requests):
        try:
            return super()._root_ids(table, requests)
        except ValueError:
            if self._fault.defect == "missing_root_bare":
                raise RuntimeError("lookup failed") from None
            raise


class Mutant(DiskGraphEngine):
    """Real DiskGraph with exactly one deliberate fault."""

    def __init__(self, defect: str):
        super().__init__(index=directory() / "index", num_threads=1, num_shards=1)
        self.defect = defect
        self.calls = self.orders = self.opened = 0

    def sample(self, spec):
        return spec

    def open(self, dataset, schema):
        self.opened += 1
        if self.defect == "first_row_dropped":
            tables = {
                name: pq.read_table(source.path).to_pylist()
                for name, source in dataset.tables.items()
            }
            tables["events"] = tables["events"][1:]
            dataset = parquet_dataset(schema, tables)
        source = super().open(dataset, schema)
        source.__class__ = FaultSource
        source._fault = self
        return source

    def check_contract(self, contract):
        filtered = any(
            getattr(value.source, "filters", ())
            for value in contract.sample.entity_values.values()
            if hasattr(value, "source")
        )
        if filtered and self.defect == "refusal_reads_rows":
            raise PropertyFailure("refusal read rows")
        refusing = ("no_suggestion", "vague_refusal", "unnamed_refusal")
        if filtered and self.defect in refusing:
            suggestion = None if self.defect == "no_suggestion" else "remove the filter"
            path = "sample" if self.defect == "vague_refusal" else FILTERS
            issue = ValidationIssue(path, "row filters are unsupported", suggestion)
            if self.defect == "unnamed_refusal":
                return (issue, ValidationIssue("sample", "unsupported", suggestion))
            return (issue,)
        if self.defect in ("late_refusal", "accept_refusal"):
            return ()
        return super().check_contract(contract)


class ParquetFault(DiskGraphEngine):
    """Read real Parquet with a deliberate file-selection or error-reporting fault."""

    def __init__(self, defect: str):
        super().__init__(index=directory() / "index", num_threads=1, num_shards=1)
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
            tables[name] = rows
        return super().open(parquet_dataset(schema, tables), schema)


def _binding(native=None):
    """Inject a fault into the real native binding, never an alternate engine."""

    @contextmanager
    def binding(contract, tables, engine_name, *, refusal=False, parts=1, types=None):
        with bound_tables(
            contract, tables, "diskgraph", refusal=refusal, parts=parts, types=types
        ) as (dataset, engine):
            chosen = native or engine
            chosen.index = engine.index
            chosen.num_shards, chosen.num_threads = engine.num_shards, engine.num_threads
            chosen._dataset = dataset
            yield dataset, chosen

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
    "aggregate_answers": ({}, "native", "off_by_one", "fixture answers changed samples"),
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
        binding, engine = _binding(native=Mutant(defect)), "diskgraph"
    else:
        native = Mutant(defect) if kind == "native" else ParquetFault(defect)
        binding, engine = _binding(native=native), "diskgraph"
    with patch("conformance.properties.bound_tables", binding):
        PROPERTIES[name].check(case, engine, position)


def test_every_property_has_a_wrong_engine():
    # That module imports this one, so it is imported here.
    from test_functional_harness import FUNCTIONAL_COVERED

    # Graph and sequence properties have theirs in test_sampling_harness.py.
    sampling = GRAPH_CHECKS.keys() | SEQUENCE_CHECKS.keys()
    assert WRONG_ENGINES.keys() | FUNCTIONAL_COVERED == PROPERTIES.keys() - sampling


def test_parquet_fault_wrapper_matches_independent_fixture_answers():
    position, case = _case()
    with patch("conformance.properties.bound_tables", _binding(native=ParquetFault("none"))):
        PROPERTIES["aggregate_answers"].check(case, "diskgraph", position)


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
            run_obligation("point_in_time", "diskgraph", 0)


def test_a_failing_obligation_writes_a_repro_that_replays(tmp_path):
    binding = _binding(Mutant("future_visible"))
    with patch("conformance.properties.bound_tables", binding):
        with pytest.raises(PropertyFailure, match="repro: ") as failure:
            run_obligation("point_in_time", "diskgraph", 0, tmp_path)
        paths = sorted(tmp_path.glob("repro-*.json"))
        assert paths and all(str(path) in str(failure.value) for path in paths)
        repro = load_repro(paths[0])
        with pytest.raises(AssertionError):
            replay(repro)
    replay(repro)  # the same file passes on the correct reference
    payload = json.loads(paths[0].read_text())
    assert payload["engine"] == "diskgraph" and payload["seed"] == 0
    assert set(payload["versions"]) == {"package", "diskgraph", "registry", "python"}
    assert GeneratedCase.from_dict(payload["case"]).to_dict() == payload["case"]


def test_repro_files_are_named_by_content(tmp_path):
    position, case = _case()
    first = write_repro("permutation", "diskgraph", case, position, ValueError("a"), tmp_path)
    second = write_repro("permutation", "diskgraph", case, position, ValueError("b"), tmp_path)
    assert first == second
    payload = repro_payload("permutation", "diskgraph", case, position, ValueError("a"))
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
    path = write_repro("time_shift", "diskgraph", case, position, ValueError("x"), tmp_path)
    with pytest.raises(ValueError, match="issue"):
        load_corpus(tmp_path)
    path.write_text(json.dumps({**json.loads(path.read_text()), "issue": "#1"}))
    entry = load_corpus(tmp_path)[path.stem]
    assert entry.issue == "#1" and entry.case.to_dict() == case.to_dict()
    replay(entry, "diskgraph")


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
