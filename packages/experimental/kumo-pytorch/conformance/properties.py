# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metamorphic properties over generated cases (design §3.4).

Each property needs no expected answer: it runs related inputs through the real
public `check()` → `prepare()` → `materialize()` path and compares the results
as sets keyed by ``sample_id``. A failing case is written as a repro file.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from conformance.assertions import PropertyFailure, _same_value, bound_tables, case_settings
from conformance.assertions import require as _require
from conformance.cases import REPOSITORY
from conformance.fixtures import values as fixture_values
from conformance.generate import (
    ACCOUNT_EVENTS,
    OPENED,
    PHYSICAL_TYPES,
    GeneratedCase,
    Row,
    Scalar,
    Tables,
    case_digest,
    crossed_settings,
    delivery_configs,
    entity_cutoffs,
    event_row,
    generate_batch,
    permuted,
    shifted,
)
from conformance.graph_cases import _RUNS as graph_runs
from conformance.graph_cases import GRAPH_CHECKS, graph_batch
from conformance.outputs import (
    HISTORY,
    SELECTIONS,
    TARGET,
    availability_case,
    batch_cost,
    decode_batches,
    entity_only,
    expected_targets,
    fitted_structure,
    fitting_contract,
    flat_targets,
    model_contract,
    packing_contract,
    packing_requests,
    plant_label_rows,
    selected,
    sequence_tokens,
    split_anchor_rows,
    state_differences,
    typed_fitting_contract,
)
from conformance.registry import ENGINE_REFUSALS, late_refusal
from conformance.repro import Repro, write_repro
from conformance.sequence_cases import _RUNS as sequence_runs
from conformance.sequence_cases import SEQUENCE_CHECKS, sequence_batch
from conformance.transfer import TRANSFER_ARGUMENTS, batches_json, transfer_command
from conformance.variants import Variant, variant_for
from kumo.pytorch.api import (
    ALL_ANCHORS,
    AnchorTable,
    Batch,
    DataLoaderSpec,
    DataSchema,
    Dataset,
    DiskGraphEngine,
    DType,
    InMemoryEngine,
    InMemoryRows,
    Packed,
    Padded,
    Parquet,
    RelationSpec,
    SampleRequest,
    TableSpec,
    Task,
    ValidationError,
    ValidationIssue,
    check,
    prepare,
)
from kumo.pytorch.column_stats import table_column_encoders
from kumo.pytorch.contract.batch.encodings import Buckets
from kumo.pytorch.contract.sample import BuiltSample
from kumo.pytorch.contract.schema import table_clock
from kumo.pytorch.data.model import ModelSourceBatch
from kumo.pytorch.data.records import cutoff_to_epoch

Answer = dict[str, dict[str, Scalar]]
#: One delivered row: sample ID, padding flag, and each numeric field as (valid, value).
Delivered = tuple[str, bool, tuple[tuple[str, bool, float | None], ...]]
Engine = InMemoryEngine | DiskGraphEngine
Format = Literal["inspectable", "model"]
PREVIEW_LIMIT = 8
#: Reductions whose float result depends on summation order and algorithm.
SUMMATION_DEPENDENT = frozenset(
    {"mean", "std_population", "std_sample", "var_population", "var_sample"}
)
#: Allowed discrepancy (conformance/README.md, Differential testing): G-CALC-001
#: leaves the rounding of these approximate results open. Inputs are small integers,
#: so a rounding difference is ~1e-16 relative (measured: ≤4 ULP) while any semantic
#: error (ddof, nulls, a missed row) moves the value by far more. Also the absolute
#: floor for results near zero.
#: Everything else is exact.
MOMENT_TOLERANCE = 1e-12
#: Allowed discrepancy: Buckets cuts over stored columns are t-digest quantiles, and
#: the digest depends on how the index splits the rows into shards. Each engine's cut
#: lies within 1.5 ranks of the exact quantile (tests/test_column_stats.py), so two
#: engines' cuts lie within 3 ranks of each other.
BUCKET_RANKS = 3


# -- Running a case ----------------------------------------------------------


def answer(
    contract: DataLoaderSpec,
    tables: Tables,
    requests: Sequence[SampleRequest],
    engine_name: str,
    *,
    parts: int = 1,
) -> Answer:
    """Raw entity values per sample, through the real public path."""

    with bound_tables(contract, tables, engine_name, parts=parts, types=PHYSICAL_TYPES) as bound:
        dataset, engine = bound
        issues = check(contract, dataset, engine)
        _require(not issues, f"{engine_name} refused a generated contract: {issues}")
        with prepare(contract, dataset, engine) as plan:
            return _collect(plan.materialize(requests), requests)


def _collect(task: Task, requests: Sequence[SampleRequest]) -> Answer:
    result: Answer = {}
    cutoffs = {str(request.sample_id): int(request.as_of_time) for request in requests}
    for start in range(0, len(requests), PREVIEW_LIMIT):
        chunk = requests[start : start + PREVIEW_LIMIT]
        for sample in task.preview(chunk).to_dict()["samples"]:
            _require(sample["sample_id"] not in result, f"duplicate sample {sample['sample_id']}")
            targets = flat_targets(sample["targets"], cutoffs[str(sample["sample_id"])])
            result[sample["sample_id"]] = fixture_values({**sample["entity"]["values"], **targets})
    expected = {str(request.sample_id) for request in requests}
    _require(result.keys() == expected, f"samples {sorted(result)} != requests {sorted(expected)}")
    return result


def _close(actual: Scalar, expected: Scalar) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=MOMENT_TOLERANCE, abs_tol=MOMENT_TOLERANCE)
    return _same_value(actual, expected)


def _same_sample(
    actual: Mapping[str, Scalar] | None, expected: Mapping[str, Scalar] | None
) -> bool:
    if actual is None or expected is None or actual.keys() != expected.keys():
        return _same_value(actual, expected)
    return all(
        _close(actual[name], value)
        if name in SUMMATION_DEPENDENT
        else _same_value(actual[name], value)
        for name, value in expected.items()
    )


def _same(
    before: Answer,
    after: Answer,
    label: str,
    only: set[str] | None = None,
    *,
    rounding: bool = False,
) -> None:
    """Exact per sample; ``rounding`` allows MOMENT_TOLERANCE on summation-dependent values."""

    keys = sorted(only if only is not None else before.keys() | after.keys())
    agree = _same_sample if rounding else _same_value
    differences = [
        f"{key}: {before.get(key)} != {after.get(key)}"
        for key in keys
        if not agree(after.get(key), before.get(key))
    ]
    _require(not differences, f"{label} changed samples:\n  " + "\n  ".join(differences[:8]))


#: One base answer per (binding, engine, case, reader) serves every property.
_RUNS: dict[tuple[Callable[..., object], str, str, str], Answer] = {}


def _cached(case: GeneratedCase, engine: str, reader: str, run: Callable[[], Answer]) -> Answer:
    key = (bound_tables, engine, case_digest(case), reader)
    if key not in _RUNS:
        _RUNS[key] = run()
    return _RUNS[key]


def clear_runs() -> None:
    """Forget cached engine runs, for checks that patch an engine in place."""

    _RUNS.clear()
    graph_runs.clear()
    sequence_runs.clear()


def _base(case: GeneratedCase, engine: str) -> Answer:
    return _cached(
        case, engine, "preview", lambda: answer(case.contract, case.tables, case.requests, engine)
    )


# -- Properties --------------------------------------------------------------


def with_future_rows(case: GeneratedCase, rng: random.Random) -> tuple[Tables, set[str]]:
    """Rows dated after one cutoff per entity; the anchors they must not reach.

    Future rows go to the destination table, through an existing and a new
    intermediate row. When the intermediate table is timed, the new intermediate
    row is itself dated after the cutoff and also links an older destination row.
    """

    tables = {name: [dict(row) for row in rows] for name, rows in case.tables.items()}
    events = tables["events"]
    by_account = case.dimensions["schema_shape.path"] in ACCOUNT_EVENTS
    accounts = tables.get("accounts") if by_account else None
    guarded: set[str] = set()
    thresholds = {
        entity: rng.choice(cutoffs) for entity, cutoffs in entity_cutoffs(case.requests).items()
    }
    for entity, threshold in sorted(thresholds.items()):
        owned: list[list[int]] = [[], []]
        if accounts is not None:
            existing = [row["account_id"] for row in accounts if row["user_id"] == entity]
            accounts.append(
                {"account_id": len(accounts) + 1, "user_id": entity, OPENED: threshold + 1}
            )
            owned = [existing[:1] or [len(accounts)], [len(accounts)]]
        for time, via in zip((threshold + 1, threshold + rng.randint(2, 100)), owned, strict=True):
            events.append(event_row(rng, len(events) + 1, entity, time, via, "none"))
            if "links" in tables:
                link = {"user_id": entity, "event_id": events[-1]["event_id"]}
                tables["links"].append({"link_id": len(tables["links"]) + 1, **link})
        if case.dimensions["schema_shape.path"] == "depth2_timed":
            older = min(entity_cutoffs(case.requests)[entity]) - 1
            events.append(event_row(rng, len(events) + 1, entity, older, owned[1], "none"))
    for request in case.requests:
        if int(request.as_of_time) <= thresholds[int(request.entity_id)]:
            guarded.add(str(request.sample_id))
    return tables, guarded


def point_in_time(case: GeneratedCase, engine: str, position: int) -> None:
    """Rows dated after an anchor's cutoff leave its sample unchanged.

    Covers the destination table and a static intermediate. No passing obligation
    checks a cutoff at a timed intermediate yet: the in-memory engine has the known
    divergence DIV-TIMED-HOP-MEM there, and DiskGraph refuses that shape.
    """

    tables, guarded = with_future_rows(case, random.Random(f"{case.id}:future"))
    _require(bool(guarded), "no anchor is guarded by the future rows")
    after = _entities(_judged(case, engine, tables))
    _same(_entities(_judged(case, engine)), after, "future rows", only=guarded)


_DEVICE_GRAPH = "sample.neighborhoods.devices.layers[0].samples[0]"
#: Per refused strategy: the most specific path some refusal reaches today, and words
#: one of which every refusal's path or message holds, naming the refused construct.
REFUSAL_CONSTRUCTS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    "invalid_unknown_column": ("sample.entity_values.broken.column", ("broken",)),
    "invalid_unknown_relation": ("sample.entity_values.broken.source.path[0]", ("broken",)),
    "active_interval": ("sample", ("active interval",)),
    "clock_override": (f"{_DEVICE_GRAPH}.time_column", ("time_column",)),
    "comparison": ("sample.entity_values.busy", ("busy",)),
    "current_events": ("sample", ("current_events",)),
    "dated_values": ("sample", ("dated_entity_values",)),
    "derived": ("sample.entity_values.total_per_row", ("total_per_row",)),
    "non_direct_field": ("sample", ("'owner'",)),
    "object_store": ("dataset.users.path", ("local parquet",)),
    "row_filter": ("sample.entity_values.filtered.source.filters", ("filter",)),
    "static_traversal": (f"{_DEVICE_GRAPH}.temporal", ("temporal",)),
    "timed_intermediate": ("sample.entity_values.count.source.path[0]", ("intermediate",)),
    "uniform_graph": (f"{_DEVICE_GRAPH}.strategy", ("uniform",)),
    "window_on_value_column": ("sample.entity_values.broken.source.time.column", ("amount",)),
}


def refusal_timing(case: GeneratedCase, engine: str, position: int) -> None:
    """Anything refused is refused at check(), before data is opened or read.

    Anything accepted runs to completion. The generated base contract is
    accepted. Support refusals (of valid contracts) carry a suggestion;
    structurally invalid contracts are refused at the broken value; a region
    the registry declares refused by this engine is refused. Each refusal names
    its strategy's construct, at a path as specific as REFUSAL_CONSTRUCTS records.
    A batches() argument check() never sees is refused at batches() instead,
    before any batch, when the registry declares it inapplicable there.
    """

    strategy = str(case.dimensions["contract.strategy"])
    variant = variant_for(case, strategy)
    contract = variant.contract
    with bound_tables(case.contract, case.tables, engine, types=PHYSICAL_TYPES) as bound_base:
        _require(not check(case.contract, *bound_base), f"{engine} refused the base contract")
    with _bound_variant(variant, engine) as (dataset, bound):
        issues = check(contract, dataset, bound)
        declared = any(r.engine == engine and r.strategy == strategy for r in ENGINE_REFUSALS)
        _require(bool(issues) or not declared, f"{engine} accepted {strategy}, declared refused")
        if not issues:
            _require(not strategy.startswith("invalid_"), f"{engine} accepted {strategy}")
            _runs_or_refuses_late(case, strategy, engine, variant, dataset, bound)
            return
    for issue in issues:
        _require(bool(issue.path and issue.message), f"{strategy}: refusal needs a path: {issue}")
        if strategy.startswith("invalid_"):
            broken = issue.path.startswith("sample.entity_values.broken")
            _require(broken, f"{strategy}: refusal doesn't point at the broken value: {issue}")
        else:
            _require(bool(issue.suggestion), f"{strategy}: support refusal needs a suggestion")
    _names_construct(strategy, issues)
    with bound_tables(contract, variant.tables, engine, refusal=True) as refused:
        dataset, bound = variant.bind(*refused)
        with patch.object(bound, "open", side_effect=PropertyFailure("opened before refusal")):
            try:
                again = check(contract, dataset, bound)
            except AssertionError as error:
                raise PropertyFailure(f"{strategy}: {error}") from error
            _require(again == issues, f"{strategy}: refusal depends on the data")
            try:
                prepare(contract, dataset, bound)
            except ValidationError as error:
                _require(error.issues == issues, f"{strategy}: prepare() refused differently")
            else:
                raise PropertyFailure(f"{strategy}: prepare() accepted a refused request")


def _names_construct(strategy: str, issues: Sequence[ValidationIssue]) -> None:
    construct = REFUSAL_CONSTRUCTS.get(strategy)
    if construct is None:
        raise PropertyFailure(f"{strategy}: refused, but no construct is recorded for it")
    path, words = construct
    _require(
        any(issue.path.startswith(path) for issue in issues),
        f"{strategy}: no refusal is as specific as {path}: {[i.path for i in issues]}",
    )
    for issue in issues:
        text = f"{issue.path} {issue.message}".lower()
        _require(
            any(word in text for word in words),
            f"{strategy}: refusal names none of {list(words)}: {issue}",
        )


def _runs_or_refuses_late(
    case: GeneratedCase,
    strategy: str,
    engine: str,
    variant: Variant,
    dataset: Dataset,
    bound: Engine,
) -> None:
    """An accepted request runs to completion, or is refused at batches() as declared."""

    late = late_refusal(engine, strategy)
    try:
        with prepare(variant.contract, dataset, bound) as plan:
            task = plan.materialize(case.requests)
            _collect(task, case.requests)
            if late is None:
                list(variant.run(task, case.requests))
                return
            try:
                next(iter(variant.run(task, case.requests)))
            except ValueError as error:
                _require(
                    str(late.message) in str(error),
                    f"{strategy}: {late.refused_at} refused unclearly: {error!r}",
                )
            else:
                raise PropertyFailure(f"{strategy}: {late.refused_at} delivered before refusing")
    except PropertyFailure:
        raise
    except Exception as error:
        raise PropertyFailure(f"{strategy}: accepted at check(), then {error!r}") from error


def differential(case: GeneratedCase, engine: str, position: int) -> None:
    """DiskGraph equals the reference, once the reference holds point-in-time invariance."""

    try:
        point_in_time(case, "in_memory", position)
    except AssertionError as error:
        raise PropertyFailure(f"reference failed point_in_time; not judged: {error}") from error
    reference, actual = _judged(case, "in_memory"), _judged(case, engine)
    _same(reference, actual, f"{engine} vs reference", rounding=True)


def permutation(case: GeneratedCase, engine: str, position: int) -> None:
    """Row order and request order don't change outputs (no duplicate primary keys)."""

    shuffled = permuted(case, random.Random(f"{case.id}:permute"))
    after = answer(shuffled.contract, shuffled.tables, shuffled.requests, engine)
    _same(_base(case, engine), after, "row and request order")


def partition(case: GeneratedCase, engine: str, position: int) -> None:
    """Splitting every table into several Parquet files doesn't change outputs."""

    parts = 2 + position % 2
    after = answer(case.contract, case.tables, case.requests, engine, parts=parts)
    _same(_base(case, engine), after, f"{parts} files per table")


def time_shift(case: GeneratedCase, engine: str, position: int) -> None:
    """Moving every time and cutoff by the same positive constant changes nothing."""

    seconds = random.Random(f"{case.id}:shift").randint(1, 10**7)
    moved = shifted(case, seconds)
    after = answer(moved.contract, moved.tables, moved.requests, engine)
    _same(_base(case, engine), after, f"a {seconds}s shift")


def irrelevant(case: GeneratedCase, position: int) -> tuple[str, DataLoaderSpec, Tables]:
    """An unused column, an unrelated table, or an unused relation to a new table."""

    variant = ("column", "table", "relation")[position % 3]
    schema = case.contract.schema
    tables = {name: [dict(row) for row in rows] for name, rows in case.tables.items()}
    specs = list(schema.tables)
    relations = list(schema.relations)
    if variant == "column":
        index = next(i for i, table in enumerate(specs) if table.name == "events")
        specs[index] = replace(
            specs[index], columns={**specs[index].columns, "note": DType("string")}
        )
        for row in tables["events"]:
            row["note"] = f"n{row['event_id']}"
    else:
        columns = {
            "device_id": DType("int64"),
            "label": DType("string"),
            "seen_at": DType("timestamp", unit="s", timezone="UTC"),
        }
        if variant == "relation":
            columns = {**columns, "user_id": DType("int64")}
            relations.append(RelationSpec("user_devices", "users", "user_id", "devices", "user_id"))
        specs.append(TableSpec("devices", "device_id", columns, time_column="seen_at"))
        tables["devices"] = [
            {"device_id": device, "label": "x", "seen_at": int(request.as_of_time) - 1}
            | ({"user_id": int(request.entity_id)} if variant == "relation" else {})
            for device, request in enumerate(case.requests, 1)
        ]
    contract = DataLoaderSpec(
        DataSchema(tables=tuple(specs), relations=tuple(relations)),
        case.contract.sample,
        case.contract.batch,
    )
    return variant, contract, tables


def irrelevance(case: GeneratedCase, engine: str, position: int) -> None:
    """Adding an unrelated table, relation or column changes nothing."""

    variant, contract, tables = irrelevant(case, position)
    after = answer(contract, tables, case.requests, engine)
    _same(_base(case, engine), after, f"an unrelated {variant}")


@contextmanager
def _anchored(
    case: GeneratedCase,
    engine: str,
    contract: DataLoaderSpec | None = None,
    anchor_rows: Sequence[Row] | None = None,
) -> Iterator[tuple[Dataset, Engine, AnchorTable]]:
    """Bind the case's rows and its requests as an anchor table of the engine's kind.

    ``anchor_rows`` with a ``split`` column replace the requests.
    """

    rows: list[Row] = list(anchor_rows or ()) or [
        {
            "sample_id": str(request.sample_id),
            "entity_id": int(request.entity_id),
            "as_of_time": int(request.as_of_time),
        }
        for request in case.requests
    ]
    split = "split" if anchor_rows else None
    with (
        tempfile.TemporaryDirectory(prefix="conformance-anchors-") as directory,
        bound_tables(contract or case.contract, case.tables, engine, types=PHYSICAL_TYPES) as bound,
    ):
        dataset, engine_object = bound
        source: InMemoryRows | Parquet = InMemoryRows(rows)
        if isinstance(dataset.tables["users"], Parquet):
            import pyarrow as pa
            import pyarrow.parquet as pq

            path = Path(directory) / "anchors.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            source = Parquet(str(path))
        yield (
            dataset,
            engine_object,
            AnchorTable(source, "entity_id", "as_of_time", sample_id="sample_id", split=split),
        )


def _delivered(batches: Sequence[Batch]) -> list[Delivered]:
    rows: list[Delivered] = []
    for batch in batches:
        fields = batch.inputs["entity"]
        flags = batch.metadata["is_padding"].tolist()
        for index, sample_id in enumerate(batch.metadata["sample_ids"]):
            values = tuple(
                (
                    name,
                    bool(field["valid"][index]),
                    float(field["values"][index]) if bool(field["valid"][index]) else None,
                )
                for name, field in sorted(fields.items())
            )
            rows.append((str(sample_id), bool(flags[index]), values))
    return rows


def delivery(case: GeneratedCase, engine: str, position: int) -> None:
    """Sharding, resume and determinism on real engine batches.

    For every covering configuration: the ranks' union is every anchor exactly
    once per epoch; ranks take equal steps; padding is marked and repeats one of
    the rank's own anchors with an identical sample; a resumed iterator continues
    the same sequence; the same seed gives the same output, in the same order, also after reopening.
    """

    _delivery(case, engine, case.contract, "inspectable")


def _delivery(
    case: GeneratedCase, engine: str, contract: DataLoaderSpec, output_format: Format
) -> None:
    count = len(case.requests)
    batch_size = max(1, count // 16)
    with _anchored(case, engine, contract) as (dataset, engine_object, anchors):
        with prepare(contract, dataset, engine_object) as plan:
            task = plan.materialize(anchors)
            one_pass = task.batches(batch_size=1, output_format=output_format)
            whole = {row[0]: row[2] for row in _delivered(list(one_pass))}
            _require(len(whole) == count, f"one pass served {len(whole)} of {count} anchors")
            if count >= 3:
                orders = [
                    [
                        row[0]
                        for row in _delivered(
                            list(
                                task.batches(batch_size=1, output_format=output_format, **arguments)
                            )
                        )
                    ]
                    for arguments in (
                        dict(seed=case.seed, epoch=0, shuffle=True),
                        dict(seed=case.seed, epoch=1, shuffle=True),
                        dict(seed=case.seed + 1, epoch=0, shuffle=True),
                    )
                ]
                _require(
                    orders[0] != orders[1] and orders[0] != orders[2],
                    "shuffle ignores the epoch or the seed",
                )
            for config in delivery_configs(case.seed, engine):
                _deliver(task, whole, config, batch_size, case.seed, output_format)
            world = min(2, count)
            shuffled = dict(seed=case.seed, epoch=1, shuffle=True, rank=world - 1, world_size=world)
            ordered = _delivered(
                list(task.batches(batch_size=batch_size, output_format=output_format, **shuffled))
            )
        with prepare(contract, dataset, engine_object) as plan:
            task = plan.materialize(anchors)
            again = _delivered(list(task.batches(batch_size=1, output_format=output_format)))
            reopened = _delivered(
                list(task.batches(batch_size=batch_size, output_format=output_format, **shuffled))
            )
        _require({row[0]: row[2] for row in again} == whole, "reopening changed the samples")
        _require(reopened == ordered, "reopening changed the shuffled order for the same seed")


def _deliver(
    task: Task,
    whole: Mapping[str, object],
    config: Mapping[str, object],
    size: int,
    seed: int,
    output_format: Format = "inspectable",
) -> None:
    world, epochs = int(config["execution.world_size"]), int(config["execution.epochs"])
    label = f"world_size={world} epochs={epochs} resume={config['execution.resume']}"
    for epoch in range(epochs):

        def run(rank: int) -> list[Delivered]:
            arguments = dict(epoch=epoch, seed=seed, shuffle=True, rank=rank, world_size=world)
            batches = task.batches(batch_size=size, output_format=output_format, **arguments)
            return _delivered(list(batches))

        ranks = [run(rank) for rank in range(world)]
        _require(len({len(rows) for rows in ranks}) == 1, f"{label}: unequal rank lengths")
        real = [row[0] for rows in ranks for row in rows if not row[1]]
        _require(Counter(real) == Counter(whole.keys()), f"{label}: epoch {epoch} not exactly once")
        padding = sum(row[1] for rows in ranks for row in rows)
        _require(padding == world * math.ceil(len(whole) / world) - len(whole), f"{label}: padding")
        for rows in ranks:
            own = {row[0] for row in rows if not row[1]}
            for sample_id, is_padding, values in rows:
                original = sample_id.split("#padding")[0] if is_padding else sample_id
                _require(not is_padding or original in own, f"{label}: foreign padding {sample_id}")
                _require(values == whole[original], f"{label}: {sample_id} differs from one pass")
        _require(run(0) == ranks[0], f"{label}: same seed, different output")
        if config["execution.resume"] == "mid_epoch":
            arguments = dict(epoch=epoch, seed=seed, shuffle=True, rank=0, world_size=world)
            first = task.batches(batch_size=size, output_format=output_format, **arguments)
            taken = [next(first) for _ in range(math.ceil(len(ranks[0]) / size) // 2)]
            resumed = task.batches(batch_size=size, output_format=output_format, **arguments)
            resumed.load_state_dict(first.state_dict())
            _require(
                _delivered(taken) + _delivered(list(resumed)) == ranks[0],
                f"{label}: resume from state_dict changed the sequence",
            )


TREND_EPOCHS = 4


def _instrument(engine: Engine, epochs: list[Counter[str]]) -> Callable[..., object]:
    """Wrap ``engine.open`` so the source counts work into the current epoch's counter."""

    original = engine.open

    def opened(dataset: Dataset, schema: DataSchema) -> object:
        source = original(dataset, schema)
        build = source.build

        def counted(spec: object, requests: Sequence[SampleRequest]) -> object:
            built = build(spec, requests)
            work = epochs[-1]
            work["build_calls"] += 1
            work["requests"] += len(requests)
            profile = getattr(source, "last_temporal_profile", None) or {}
            for name, value in profile.get("native_stats", {}).items():
                work[f"native.{name}"] += int(value)
            return built

        source.build = counted
        # Batches build through build_model_ordered when a source has it; build_model calls it.
        model_method = (
            "build_model_ordered" if hasattr(source, "build_model_ordered") else "build_model"
        )
        build_model = getattr(source, model_method, None)
        if build_model is not None:

            def counted_model(
                spec: object, requests: Sequence[SampleRequest], **keywords: object
            ) -> object:
                built = build_model(spec, requests, **keywords)
                work = epochs[-1]
                work["build_calls"] += 1
                work["requests"] += len(requests)
                profile = getattr(source, "last_temporal_profile", None) or {}
                for name, value in profile.get("native_stats", {}).items():
                    work[f"native.{name}"] += int(value)
                return built

            setattr(source, model_method, counted_model)
        return source

    return opened


def epoch_trend(case: GeneratedCase, engine: str, position: int) -> None:
    """Epoch N does the same work as epoch 2, measured by work counters, not time.

    Guards against work being recomputed every epoch. Work also matches what
    is served: one build request per delivered row.
    """

    _epoch_trend(case, engine, case.contract, "inspectable")


def _epoch_trend(
    case: GeneratedCase, engine: str, contract: DataLoaderSpec, output_format: Format
) -> None:
    epochs: list[Counter[str]] = [Counter()]
    with _anchored(case, engine, contract) as (dataset, engine_object, anchors):
        with patch.object(engine_object, "open", _instrument(engine_object, epochs)):
            with prepare(contract, dataset, engine_object) as plan:
                task = plan.materialize(anchors)
                served = []
                for epoch in range(TREND_EPOCHS):
                    epochs.append(Counter())
                    rows = [
                        row
                        for rank in range(2)
                        for row in _delivered(
                            list(
                                task.batches(
                                    batch_size=2,
                                    epoch=epoch,
                                    shuffle=True,
                                    rank=rank,
                                    world_size=2,
                                    output_format=output_format,
                                )
                            )
                        )
                    ]
                    served.append(len(rows))
    per_epoch = epochs[1:]
    for epoch in range(2, TREND_EPOCHS):
        _require(
            per_epoch[epoch] == per_epoch[1],
            f"epoch {epoch + 1} work {dict(per_epoch[epoch])} != epoch 2 {dict(per_epoch[1])}",
        )
    _require(
        [work["requests"] for work in per_epoch] == served,
        f"built {[w['requests'] for w in per_epoch]} requests to serve {served} rows",
    )


def truncated_parquet(case: GeneratedCase, engine: str, position: int) -> None:
    """A truncated Parquet file gives a clean error naming Parquet and the table or file
    (rule C6: the stage, resource or path).

    Nothing is published: no index and no staging directory remain.
    """

    parts = 1 if case.dimensions["source.files"] == "single" else 2
    contract = case.contract
    with bound_tables(contract, case.tables, engine, parts=parts, types=PHYSICAL_TYPES) as bound:
        dataset, engine_object = bound
        location = Path(dataset.tables["events"].path)
        victim = location if parts == 1 else sorted(location.glob("*.parquet"))[-1]
        data = victim.read_bytes()
        victim.write_bytes(data[: len(data) // 2])
        try:
            with prepare(contract, dataset, engine_object) as plan:
                _collect(plan.materialize(case.requests), case.requests)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001 - a native panic is a finding
            _require(isinstance(error, Exception), f"not a clean error: {error!r}")
            message = str(error)
            _require("parquet" in message.lower(), f"error doesn't name Parquet: {message}")
            _require(
                "events" in message or str(victim) in message,
                f"error names neither the table nor the file: {message}",
            )
        else:
            raise PropertyFailure(f"a truncated file ({victim.name}) was accepted")
        if isinstance(engine_object, DiskGraphEngine):
            index = engine_object.index
            leftovers = [path.name for path in index.parent.glob(f".{index.name}-*")]
            _require(not index.exists() and not leftovers, f"left {leftovers or index}")


# -- Functional areas: outputs, model path, anchors, fitted state ------------

#: Rows per batch when a property reads batches rather than previews.
BATCH_ROWS = 16
MISSING_ROOT = 987_654_321


@contextmanager
def _bound_variant(variant: Variant, engine: str) -> Iterator[tuple[Dataset, Engine]]:
    with bound_tables(variant.contract, variant.tables, engine, types=PHYSICAL_TYPES) as bound:
        yield variant.bind(*bound)


def _entities(result: Answer) -> Answer:
    return {sample: entity_only(values) for sample, values in result.items()}


def batch_answer(
    contract: DataLoaderSpec,
    tables: Tables,
    requests: Sequence[SampleRequest],
    engine_name: str,
    output_format: Format,
) -> Answer:
    """Entity columns and targets per sample, decoded from real batches."""

    with bound_tables(contract, tables, engine_name, types=PHYSICAL_TYPES) as bound:
        dataset, engine = bound
        issues = check(contract, dataset, engine)
        _require(not issues, f"{engine_name} refused a generated contract: {issues}")
        with prepare(contract, dataset, engine) as plan:
            task = plan.materialize(requests)
            batches = task.batches(requests, batch_size=BATCH_ROWS, output_format=output_format)
            rows = decode_batches(batches)
    result = dict(rows)
    _require(len(result) == len(rows), "a sample was delivered twice")
    expected = {str(request.sample_id) for request in requests}
    _require(result.keys() == expected, f"samples {sorted(result)} != requests {sorted(expected)}")
    return result


def _judged(case: GeneratedCase, engine: str, tables: Tables | None = None) -> Answer:
    """Previews, or for an ``execution.output_format=model`` case, decoded batches:
    DiskGraph's model batches and the reference's inspectable ones.
    """

    if tables is None:
        return _cached(case, engine, "judged", lambda: _judged(case, engine, case.tables))
    rows = tables
    if case.dimensions.get("execution.output_format") != "model":
        return answer(case.contract, rows, case.requests, engine)
    output_format: Format = "model" if engine == "diskgraph" else "inspectable"
    return batch_answer(case.contract, rows, case.requests, engine, output_format)


def label_window(case: GeneratedCase, engine: str, position: int) -> None:
    """Targets read only rows in (cutoff+a, cutoff+b]: the start is exclusive, the stop inclusive.

    Rows planted just inside and outside every target edge (and before the
    cutoff) change each target exactly as the rule says; which of several rows
    tied for first or last is chosen is not judged, but their time is.
    """

    planted = plant_label_rows(case, random.Random(f"{case.id}:labels"))
    before = _base(case, engine)
    after = answer(case.contract, planted.tables, case.requests, engine)
    differences = []
    for request in case.requests:
        sample, entity = str(request.sample_id), int(request.entity_id)
        expected = expected_targets(
            before[sample],
            planted.rows[entity],
            int(request.as_of_time),
            planted.multiplicity[entity],
        )
        targets = {k: v for k, v in after[sample].items() if k.startswith(TARGET)}
        for name, value in expected.items():
            if not _same_value(targets.get(name), value):
                differences.append(f"{sample} {name}: {targets.get(name)} != expected {value}")
    _require(not differences, "label window:\n  " + "\n  ".join(differences[:8]))


def model_delivery(case: GeneratedCase, engine: str, position: int) -> None:
    """The delivery property on output_format='model' batches."""

    _delivery(case, engine, model_contract(case), "model")


def model_epoch_trend(case: GeneratedCase, engine: str, position: int) -> None:
    """The epoch work trend on output_format='model', counted at the source's model builder."""

    _epoch_trend(case, engine, model_contract(case), "model")


def _batch_ids(batches: Sequence[Batch]) -> list[tuple[str, ...]]:
    return [tuple(str(sample) for sample in batch.metadata["sample_ids"]) for batch in batches]


def packing(case: GeneratedCase, engine: str, position: int) -> None:
    """Token budgets, bucketing, layouts and drop_last on an event-history input.

    Histories are empty, one or two rows long, or at the cap (packing_requests).
    Under ``max_tokens`` no batch materializes more token slots; bucketing delivers
    a permutation of the requests; Padded and Packed, and model and inspectable
    batches, carry the same tokens per sample and the same batches. drop_last
    drops only a rank's final short batch, whatever ``chunk_batches`` is, and under
    a token budget at most the final batch (``execution.drop_last=token_budget``).
    """

    capped, case = case, replace(case, requests=packing_requests(case))
    formats: tuple[Format, ...] = (
        ("inspectable", "model") if engine == "diskgraph" else ("inspectable",)
    )
    budget, rows, width = PACKING
    expected_ids = sorted(str(request.sample_id) for request in case.requests)
    tokens: dict[str, dict[str, tuple[object, ...]]] = {}
    compositions: dict[str, list[tuple[str, ...]]] = {}  # per layout: budgets differ
    bounded = {
        type(layout).__name__: packing_contract(
            case, layout, max_samples=rows, max_tokens=budget, keep="last", max_events=width
        )
        for layout in (Padded(), Packed())
    }
    with bound_tables(bounded["Packed"], case.tables, engine, types=PHYSICAL_TYPES) as bound:
        dataset, engine_object = bound
        for layout, contract in bounded.items():
            with prepare(contract, dataset, engine_object) as plan:
                task = plan.materialize(case.requests)
                for output_format in formats:
                    label = f"{layout}/{output_format}"
                    batches = list(task.batches(case.requests, output_format=output_format))
                    for batch in batches:
                        cost, size = batch_cost(batch, HISTORY), len(batch.metadata["sample_ids"])
                        _require(cost <= budget, f"{label}: a batch takes {cost} > {budget} tokens")
                        _require(size <= rows, f"{label}: a batch holds {size} > {rows} samples")
                    ids = _batch_ids(batches)
                    delivered = sorted(sample for batch in ids for sample in batch)
                    _require(delivered == expected_ids, f"{label}: bucketing is not a permutation")
                    tokens[label] = dict(
                        pair for batch in batches for pair in sequence_tokens(batch, HISTORY)
                    )
                    compositions.setdefault(layout, ids)
                    _require(compositions[layout] == ids, f"{label}: batches differ by format")
        first = next(iter(tokens))
        for label in tokens:
            _require(tokens[label] == tokens[first], f"{label} tokens differ from {first}")
        unbounded = packing_contract(
            case, Packed(), max_samples=rows, max_tokens=None, keep="last", max_events=width
        )
        _check_drop_last(capped, engine, unbounded, formats)
    if case.dimensions.get("execution.drop_last") == "token_budget":
        for anchors in (capped, case):
            _check_budgeted_drop_last(anchors, engine, bounded["Packed"], formats)


#: Token budget, samples per batch and history cap of the packing contracts.
PACKING = (12, 8, 3)


def _check_budgeted_drop_last(
    case: GeneratedCase, engine: str, contract: DataLoaderSpec, formats: Sequence[Format]
) -> None:
    """Under a token budget, drop_last may drop the final batch and nothing else."""

    budget = PACKING[0]
    with _anchored(case, engine, contract) as (dataset, engine_object, anchors):
        with prepare(contract, dataset, engine_object) as plan:
            task = plan.materialize(anchors)
            for output_format in formats:
                kept, dropped = (
                    _batch_ids(list(task.batches(drop_last=drop, output_format=output_format)))
                    for drop in (False, True)
                )
                _require(
                    dropped in (kept, kept[:-1]),
                    f"{output_format}: drop_last under max_tokens={budget} kept {len(dropped)} "
                    f"of {len(kept)} batches ({[len(b) for b in dropped]} of "
                    f"{[len(b) for b in kept]} samples)",
                )


def _check_drop_last(
    case: GeneratedCase, engine: str, contract: DataLoaderSpec, formats: Sequence[Format]
) -> None:
    size, world = 5, 2
    with _anchored(case, engine, contract) as (dataset, engine_object, anchors):
        with prepare(contract, dataset, engine_object) as plan:
            task = plan.materialize(anchors)
            for output_format in formats:
                for rank in range(world):

                    def ids(chunks: int, drop: bool) -> list[tuple[str, ...]]:
                        batches = task.batches(
                            batch_size=size,
                            rank=rank,
                            world_size=world,
                            drop_last=drop,
                            chunk_batches=chunks,
                            output_format=output_format,
                        )
                        return _batch_ids(list(batches))

                    kept = ids(8, False)
                    short = kept and len(kept[-1]) < size
                    label = f"{output_format} rank {rank}"
                    for chunks in (1, 3, 8):
                        _require(ids(chunks, False) == kept, f"{label}: chunk_batches={chunks}")
                        dropped = ids(chunks, True)
                        _require(
                            dropped == (kept[:-1] if short else kept),
                            f"{label}: drop_last with chunk_batches={chunks} kept "
                            f"{[len(b) for b in dropped]} of {[len(b) for b in kept]}",
                        )


def anchor_splits(case: GeneratedCase, engine: str, position: int) -> None:
    """Split selection from an anchor table: names, unions and the null split.

    Each selection counts and serves exactly its anchors, in table order when
    not shuffled, with the same samples as one pass; DiskGraph's shuffled order
    equals the reference's.
    """

    rows = split_anchor_rows(case)
    orders: dict[str, list[list[str]]] = {}
    engines = (engine, "in_memory") if engine != "in_memory" else (engine,)
    for name in engines:
        with _anchored(case, name, anchor_rows=rows) as (dataset, engine_object, anchors):
            with prepare(case.contract, dataset, engine_object) as plan:
                task = plan.materialize(anchors, fit_split=ALL_ANCHORS)
                whole = {row[0]: row[2] for row in _delivered(list(task.batches(batch_size=1)))}
                orders[name] = []
                for selection in SELECTIONS:
                    expected = selected(rows, selection)
                    shown = (
                        sorted(selection, key=str)
                        if isinstance(selection, frozenset)
                        else selection
                    )
                    label = f"{name} split={shown}"
                    count = task.count(selection)
                    _require(count == len(expected), f"{label}: count {count} != {len(expected)}")
                    served = _delivered(list(task.batches(split=selection, batch_size=3)))
                    _require(
                        [row[0] for row in served] == expected,
                        f"{label}: served {[row[0] for row in served]} != {expected}",
                    )
                    _require(
                        all(row[2] == whole[row[0]] for row in served), f"{label}: samples differ"
                    )
                    shuffled = task.batches(split=selection, batch_size=3, shuffle=True, seed=7)
                    orders[name].append([row[0] for row in _delivered(list(shuffled))])
    _require(orders[engine] == orders[engines[-1]], f"{engine} shuffles splits differently")


def root_availability(case: GeneratedCase, engine: str, position: int) -> None:
    """``available_from`` refuses an anchor at exactly that time and accepts one a
    second later; a missing root primary key is a clean error naming the table and key.
    """

    contract, tables, joined = availability_case(case)
    with bound_tables(contract, tables, engine, types=PHYSICAL_TYPES) as bound:
        dataset, engine_object = bound
        _require(not check(contract, dataset, engine_object), f"{engine} refused available_from")
        with prepare(contract, dataset, engine_object) as plan:
            task = plan.materialize(case.requests)
            _same(
                _entities(_base(case, engine)),
                _entities(_collect(task, case.requests)),
                "available_from before every anchor",
            )
            entity = case.requests[0].entity_id
            task.preview([SampleRequest("after_join", entity, joined + 1)])
            try:
                task.preview([SampleRequest("at_join", entity, joined)])
            except ValueError as error:
                _require("did not exist" in str(error), f"available_from: {error!r}")
            else:
                raise PropertyFailure("an anchor at available_from was answered")
            missing = SampleRequest("missing_root", MISSING_ROOT, joined + 1)
            try:
                task.preview([missing])
            except Exception as error:
                message = str(error)
                _require(
                    "users" in message and str(MISSING_ROOT) in message,
                    f"a missing root key's error names neither table nor key: {message}",
                )
            else:
                raise PropertyFailure("a missing root primary key was answered")


def fitting_contracts(case: GeneratedCase) -> dict[str, DataLoaderSpec]:
    """Fitted encoders over entity columns, and field tokens over entity values and an
    event history. DiskGraph fits both from typed columns (``build_model``), the reference
    from records.
    """

    return {"columns": fitting_contract(case), "tokens": typed_fitting_contract(case)}


def fitted_state(case: GeneratedCase, engine: str, position: int) -> None:
    """The state DiskGraph fits equals the reference's, for both fitting contracts: the
    same fit inputs (entity values and history events per sample) and the same fitted
    structure: vocabulary entries, not their IDs; floats within MOMENT_TOLERANCE.
    Every contract is judged, and the failure names each that differs.
    """

    failures = []
    for label, contract in fitting_contracts(case).items():
        states, inputs = {}, {}
        for name in ("in_memory", engine):
            with bound_tables(contract, case.tables, name, types=PHYSICAL_TYPES) as bound:
                dataset, engine_object = bound
                fitted: Answer = {}
                with patch.object(engine_object, "open", _recording(engine_object, fitted)):
                    with prepare(contract, dataset, engine_object) as plan:
                        states[name] = plan.materialize(case.requests).fitted_state.to_dict()
                inputs[name] = fitted
        try:
            _same(inputs["in_memory"], inputs[engine], f"{label}: fit inputs", rounding=True)
        except PropertyFailure as error:
            failures.append(str(error))
            continue
        reference, actual = fitted_structure(states["in_memory"]), fitted_structure(states[engine])
        for identity, values in _bucket_values(case, contract).items():
            cuts = [state["encoders"][identity]["boundaries"] for state in (reference, actual)]
            ranks = [[bisect.bisect_right(values, cut) for cut in side] for side in cuts]
            if len(ranks[0]) == len(ranks[1]) and all(
                abs(left - right) <= BUCKET_RANKS for left, right in zip(*ranks, strict=True)
            ):
                actual["encoders"][identity]["boundaries"] = cuts[0]
        differences = state_differences(reference, actual, MOMENT_TOLERANCE)
        if differences:
            failures.append(f"{label}: fitted state differs:\n  " + "\n  ".join(differences[:8]))
    _require(not failures, "\n".join(failures))


def _bucket_values(case: GeneratedCase, contract: DataLoaderSpec) -> dict[str, list]:
    """The sorted values each Buckets encoder over stored columns fits on."""

    cutoff = (
        max(cutoff_to_epoch(request.as_of_time) for request in case.requests)
        if contract.requires_anchor_time
        else None
    )
    clocks = {table.name: table_clock(table) for table in contract.schema.tables}
    return {
        identity: sorted(
            row[column]
            for table, column in encoder.columns
            for row in case.tables[table]
            if row[column] is not None
            and (cutoff is None or clocks[table] is None or row[clocks[table]] <= cutoff)
        )
        for identity, encoder in table_column_encoders(contract).items()
        if isinstance(encoder.config, Buckets)
    }


def _recording(engine: Engine, inputs: Answer) -> Callable[..., object]:
    """Wrap ``engine.open`` so what the source builds, as records or typed columns
    (``build_model``), is recorded per sample as flat fit inputs.
    """

    original = engine.open

    def opened(dataset: Dataset, schema: DataSchema) -> object:
        source = original(dataset, schema)
        build = source.build

        def recorded(spec: object, requests: Sequence[SampleRequest]) -> object:
            samples = build(spec, requests)
            for sample in samples:
                inputs[str(sample.sample_id)] = _record_inputs(sample)
            return samples

        source.build = recorded
        build_model = getattr(source, "build_model", None)
        if build_model is not None:

            def recorded_model(spec: object, requests: Sequence[SampleRequest]) -> object:
                values = build_model(spec, requests)
                inputs.update(_typed_inputs(values))
                return values

            source.build_model = recorded_model
        return source

    return opened


FitEvent = tuple[int, tuple[tuple[str, Scalar], ...]]


def _flat_inputs(entity: Mapping[str, Scalar], events: Sequence[FitEvent]) -> dict[str, Scalar]:
    """One sample's fit inputs as flat values; a null is the same as an absent value."""

    flat: dict[str, Scalar] = dict(entity)
    for index, (timestamp, fields) in enumerate(sorted(events, key=repr)):
        flat[f"{HISTORY}[{index}].at"] = timestamp
        flat.update({f"{HISTORY}[{index}].{name}": value for name, value in fields})
    return {name: value for name, value in flat.items() if value is not None}


def _record_inputs(sample: BuiltSample) -> dict[str, Scalar]:
    events = [
        (event.timestamp, tuple(sorted((f.key, f.value) for f in event.fields)))
        for event in sample.history.events
        if event.source == HISTORY
    ]
    return _flat_inputs({v.key: v.value for v in sample.history.entity.values}, events)


def _py(column: Sequence[object], index: int) -> Scalar:
    value = column[index]
    return value.as_py() if hasattr(value, "as_py") else value  # type: ignore[return-value]


def _typed_inputs(values: ModelSourceBatch) -> dict[str, dict[str, Scalar]]:
    history = values.events.get(HISTORY)
    result = {}
    for position, sample_id in enumerate(values.sample_ids):
        events: list[FitEvent] = []
        if history is not None:
            start, stop = (int(str(_py(history.offsets, position + k))) for k in (0, 1))
            events = [
                (
                    int(str(_py(history.timestamps, row))),
                    tuple(
                        sorted((name, _py(column, row)) for name, column in history.fields.items())
                    ),
                )
                for row in range(start, stop)
            ]
        entity = {name: _py(column, position) for name, column in values.entity_values.items()}
        result[str(sample_id)] = _flat_inputs(entity, events)
    return result


def contract_transfer(case: GeneratedCase, engine: str, position: int) -> None:
    """A bound contract written as JSON gives identical batches in a fresh process, for
    state fitted for entity columns and for field tokens.
    """

    contracts = fitting_contracts(case)
    requests = [
        {
            "sample_id": str(r.sample_id),
            "entity_id": int(r.entity_id),
            "as_of_time": int(r.as_of_time),
        }
        for r in case.requests
    ]
    expected = {}
    with (
        bound_tables(contracts["columns"], case.tables, engine, types=PHYSICAL_TYPES) as bound,
        tempfile.TemporaryDirectory(prefix="conformance-transfer-") as directory,
    ):
        dataset, engine_object = bound
        paths = {name: source.path for name, source in dataset.tables.items()}
        for label, contract in contracts.items():
            with prepare(contract, dataset, engine_object) as plan:
                task = plan.materialize(case.requests)
                folder = Path(directory) / label
                folder.mkdir()
                (folder / "contract.json").write_text(task.bound_contract().to_json())
                expected[label] = batches_json(task.batches(case.requests, **TRANSFER_ARGUMENTS))
            inputs = json.dumps({"tables": paths, "requests": requests})
            (folder / "inputs.json").write_text(inputs)
        folders = [Path(directory) / label for label in contracts]
        done = subprocess.run(
            transfer_command(*folders), cwd=REPOSITORY, capture_output=True, text=True
        )
        _require(done.returncode == 0, f"the fresh process failed:\n{done.stderr[-2000:]}")
        for label, folder in zip(contracts, folders, strict=True):
            _require(
                json.loads(json.dumps(expected[label]))
                == json.loads((folder / "batches.json").read_text()),
                f"{label}: a fresh process gave different batches from the bound contract",
            )


def index_layout(case: GeneratedCase, engine: str, position: int) -> None:
    """An existing index is reused only with the layout it was built with.

    Reopened as built, it is reused without ingest. Opened with another shard count
    (``execution.index_layout=other_shards``), it is refused or rebuilt, never
    reused silently: ties at a graph fanout limit may differ by shards.
    """

    built = int(case.dimensions.get("execution.num_shards", 1))
    other = case.dimensions.get("execution.index_layout") == "other_shards"
    shards = (1 if built == 3 else 3) if other else built
    schema = case.contract.schema
    with bound_tables(case.contract, case.tables, engine, types=PHYSICAL_TYPES) as bound:
        dataset, engine_object = bound
        if not isinstance(engine_object, DiskGraphEngine):
            raise PropertyFailure(f"{engine} builds no index")
        engine_object.open(dataset, schema)
        manifest = engine_object.index / "index_manifest.json"
        before = manifest.read_text()
        again = DiskGraphEngine(
            index=engine_object.index, num_threads=engine_object.num_threads, num_shards=shards
        )
        try:
            again.open(dataset, schema)
        except ValueError as error:
            _require(other and "shard" in str(error), f"num_shards={shards} refused: {error}")
            return
        if not other:
            _require(manifest.read_text() == before, "reopening as built rebuilt the index")
            return
        reused = json.loads(manifest.read_text())["num_shards"]
        _require(reused == shards, f"reused a {reused}-shard index for num_shards={shards}")


# -- Obligations -------------------------------------------------------------


@dataclass(frozen=True)
class Property:
    name: str
    engines: tuple[str, ...]
    check: Callable[[GeneratedCase, str, int], None]
    #: Runs on the batch's first cases only, one per crossed engine setting.
    per_batch: bool = False
    #: The property's own generator; the aggregate batch when None.
    cases: Callable[[int, str], Sequence[GeneratedCase]] | None = None


BOTH = ("in_memory", "diskgraph")
NATIVE = ("diskgraph",)
PROPERTIES: Mapping[str, Property] = {
    prop.name: prop
    for prop in (
        Property("point_in_time", BOTH, point_in_time),
        Property("refusal_timing", BOTH, refusal_timing),
        Property("differential", NATIVE, differential),
        Property("permutation", BOTH, permutation),
        Property("partition", NATIVE, partition),
        Property("time_shift", BOTH, time_shift),
        Property("irrelevance", BOTH, irrelevance),
        Property("delivery", BOTH, delivery, per_batch=True),
        Property("epoch_trend", BOTH, epoch_trend, per_batch=True),
        Property("truncated_parquet", NATIVE, truncated_parquet),
        # -- Functional areas --
        Property("label_window", BOTH, label_window),
        Property("anchor_splits", BOTH, anchor_splits, per_batch=True),
        Property("root_availability", BOTH, root_availability, per_batch=True),
        Property("packing", BOTH, packing, per_batch=True),
        Property("model_delivery", NATIVE, model_delivery, per_batch=True),
        Property("model_epoch_trend", NATIVE, model_epoch_trend, per_batch=True),
        Property("fitted_state", NATIVE, fitted_state, per_batch=True),
        Property("contract_transfer", NATIVE, contract_transfer, per_batch=True),
        Property("index_layout", NATIVE, index_layout, per_batch=True),
    )
}
# -- Raw sampling: graph neighborhoods and event sequences --------------------
PROPERTIES = {
    **PROPERTIES,
    **{
        name: Property(
            name,
            NATIVE if name in ("graph_differential", "graph_inference_exact") else BOTH,
            check,
            cases=partial(graph_batch, property_name=name),
        )
        for name, check in GRAPH_CHECKS.items()
    },
    **{
        name: Property(
            name,
            NATIVE if name == "sequence_differential" else BOTH,
            check,
            cases=partial(sequence_batch, property_name=name),
        )
        for name, check in SEQUENCE_CHECKS.items()
    },
}
#: Seed batches per engine: XS in memory, S on DiskGraph (see ENGINE_SCALE).
SEEDS: Mapping[str, tuple[int, ...]] = {"in_memory": (0, 1, 2), "diskgraph": (0,)}


def seed_batches(engine: str, count: int | None = None, start: int = 0) -> tuple[int, ...]:
    """The engine's seed batches: :data:`SEEDS`, plus ``start .. start + count - 1`` for
    longer runs. The required batches are always included."""

    if count is None:
        return SEEDS[engine]
    if count < 1 or start < 0:
        raise ValueError(f"need a positive count and a non-negative start, not {count}, {start}")
    return tuple(sorted(set(SEEDS[engine]) | set(range(start, start + count))))


def obligations(
    engines: Sequence[str], seeds: int | None = None, start: int = 0
) -> list[tuple[str, str, int]]:
    """Every required (property, engine, seed batch) for the selected engines."""

    return [
        (name, engine, seed)
        for name, prop in PROPERTIES.items()
        for engine in engines
        if engine in prop.engines
        for seed in seed_batches(engine, seeds, start)
    ]


def obligation_id(name: str, engine: str, seed: int) -> str:
    return f"{name}-{engine}-seed{seed}"


def obligation_cases(name: str, engine: str, seed: int) -> list[GeneratedCase]:
    """The cases one obligation checks: every DiskGraph obligation sees each crossed
    (shards, threads) setting, since generators assign them by case position.
    """

    prop = PROPERTIES[name]
    cases = list((prop.cases or generate_batch)(seed, engine))
    return cases[: len(crossed_settings(engine))] if prop.per_batch else cases


def run_obligation(name: str, engine: str, seed: int, repro_dir: Path | None = None) -> None:
    """Check one property on every case of a seed batch; write a repro per failure."""

    prop = PROPERTIES[name]
    cases = obligation_cases(name, engine, seed)
    _require(bool(cases), f"{name} on {engine}, seed {seed}: the batch has no case")
    failures = []
    for position, case in enumerate(cases):
        try:
            with case_settings(case.dimensions):
                prop.check(case, engine, position)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:  # noqa: BLE001 - every failure gets a repro
            path = write_repro(name, engine, case, position, error, repro_dir)
            failures.append(f"{case.id}: {error}\n  repro: {path}")
    if failures:
        raise PropertyFailure(f"{name} on {engine}, seed {seed}:\n" + "\n".join(failures))


def replay(repro: Repro, engine: str | None = None) -> None:
    """Rerun a repro's property (on its own engine by default); raises while it fails."""

    with case_settings(repro.case.dimensions):
        PROPERTIES[repro.property].check(repro.case, engine or repro.engine, repro.position)
