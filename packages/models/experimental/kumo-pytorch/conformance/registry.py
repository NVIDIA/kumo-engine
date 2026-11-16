# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The declared input space that generated cases are drawn from (design §3.2).

Generators read this registry, so adding a value here widens coverage without
touching a generator. Values that are declared but not generated carry the
reason, and every known implementation gap names its repro file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from kumo.pytorch.contract.sample import ALL_REDUCTIONS

REGISTRY_VERSION = 3

DimensionValue = str | int | bool


class Group(Enum):
    SOURCE = "source"
    PHYSICAL_TYPE = "physical_type"
    VALUES = "values"
    TIME = "time"
    SCHEMA_SHAPE = "schema_shape"
    CONTRACT = "contract"
    EXECUTION = "execution"


#: A whole group that is declared but not varied, with the reason.
DEFERRED_GROUPS: Mapping[Group, str] = MappingProxyType(
    {Group.PHYSICAL_TYPE: "deferred: generated type variation; scalar pairs use a fixed matrix"}
)


@dataclass(frozen=True)
class Dimension:
    """One axis of variation; ``values`` are generated, ``excluded`` are not."""

    name: str
    group: Group
    values: tuple[DimensionValue, ...]
    description: str
    excluded: Mapping[DimensionValue, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.startswith(f"{self.group.value}."):
            raise ValueError(f"{self.name}: dimension names start with their group")
        if self.group not in DEFERRED_GROUPS and not self.values:
            raise ValueError(f"{self.name}: an active dimension needs a value")
        if len(set(self.values)) != len(self.values) or set(self.values) & set(self.excluded):
            raise ValueError(f"{self.name}: values must be unique and not excluded")
        object.__setattr__(self, "excluded", MappingProxyType(dict(self.excluded)))

    @property
    def deferred(self) -> str | None:
        return DEFERRED_GROUPS.get(self.group)


DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "source.files",
        Group.SOURCE,
        ("single", "split"),
        "Parquet files per table when the truncated-file property corrupts one",
    ),
    Dimension(
        "physical_type.columns",
        Group.PHYSICAL_TYPE,
        ("int64", "float64", "string", "timestamp[s]"),
        "the Arrow types the conformance Parquet writer already uses; not varied",
    ),
    Dimension(
        "time.cutoff_placement",
        Group.TIME,
        ("before", "at", "after"),
        "a planted row one second before, exactly at, or one second after a cutoff",
    ),
    Dimension(
        "time.ties",
        Group.TIME,
        ("distinct", "tied"),
        "whether one entity has several rows with the same timestamp",
    ),
    Dimension(
        "time.cutoffs",
        Group.TIME,
        ("shared", "mixed"),
        "one cutoff for every anchor, or per-anchor cutoffs with an entity at several",
    ),
    Dimension(
        "time.range",
        Group.TIME,
        ("positive", "pre_epoch"),
        "clocks and cutoffs well after the Unix epoch, or straddling it in whole seconds",
    ),
    Dimension(
        "values.null_rate",
        Group.VALUES,
        ("none", "some", "all"),
        "the share of null values in the reduced columns (never keys or clocks)",
    ),
    Dimension(
        "values.empty_window",
        Group.VALUES,
        ("absent", "present"),
        "whether some anchor has no eligible row",
    ),
    Dimension(
        "values.single_row",
        Group.VALUES,
        ("absent", "present"),
        "whether some anchor has exactly one eligible row",
    ),
    Dimension(
        "schema_shape.path",
        Group.SCHEMA_SHAPE,
        (
            "depth1_forward",
            "depth1_reverse",
            "depth2_static",
            "depth2_timed",
            "depth1_parent",
            "depth2_sibling",
            "depth3_static",
        ),
        "FK→PK relations from the entity: one hop declared either way round, or two "
        "hops through a static or a timed intermediate table; one hop plus an FK→PK "
        "hop into a static parent; two hops through a static link table that can "
        "reach an event twice; three hops out, back and out again through accounts",
    ),
    Dimension(
        "contract.reduction",
        Group.CONTRACT,
        tuple(sorted(ALL_REDUCTIONS)),
        "the reduction of a Rows aggregate; every case declares one value per reduction",
    ),
    Dimension(
        "contract.window",
        Group.CONTRACT,
        ("open", "last_second", "from_row", "lag"),
        "TimeWindow start: open, one second before the cutoff, or exactly at a planted row; "
        "or a lag window ending before the cutoff, with planted rows at both edges",
    ),
    Dimension(
        "contract.strategy",
        Group.CONTRACT,
        (
            "comparison",
            "derived",
            "invalid_unknown_column",
            "invalid_unknown_relation",
            "row_filter",
            "timed_intermediate",
            "valid",
            "window_on_value_column",
            # Refused by DiskGraph at check() by design (ENGINE_REFUSALS).
            "active_interval",
            "clock_override",
            "current_events",
            "dated_values",
            "non_direct_field",
            "static_traversal",
            "uniform_graph",
            # Accepted at check(), then refused later on DiskGraph (KNOWN_GAPS).
            "column_rename",
            "duplicate_layer_relation",
            "int64_clock",
            "max_tokens_world_size",
            "missing_column",
            "model_format_limit",
            "object_store",  # refused at check() (ENGINE_REFUSALS)
            "unsupported_type",
        ),
        "how the refusal-timing property alters the contract; invalid_* must be refused",
    ),
    Dimension(
        "contract.distinct",
        Group.CONTRACT,
        (False, True),
        "Rows.distinct on every aggregate of the case",
    ),
    Dimension(
        "execution.engine",
        Group.EXECUTION,
        ("in_memory", "diskgraph"),
        "the engine executing the case",
    ),
    # -- Functional areas: aggregation shapes, outputs, model path ----------
    Dimension(
        "contract.group_by",
        Group.CONTRACT,
        ("none", "nested"),
        "whether the case also declares aggregates over a GroupBy with two keys",
    ),
    Dimension(
        "contract.target",
        Group.CONTRACT,
        ("window_from_cutoff", "window_ahead", "row_first", "row_last"),
        "targets every case declares: future Aggregate windows [0,W) and [a,b) with a>0, "
        "and the first and last future row",
    ),
    Dimension(
        "execution.output_format",
        Group.EXECUTION,
        ("inspectable", "model"),
        "batches(output_format=...) the differential and point-in-time checks read",
    ),
    Dimension("execution.world_size", Group.EXECUTION, (1, 2, 7), "ranks sharing one epoch"),
    Dimension("execution.epochs", Group.EXECUTION, (1, 2, 3), "passes over the anchors"),
    Dimension(
        "execution.resume",
        Group.EXECUTION,
        ("none", "mid_epoch"),
        "whether a rank stops mid-epoch and resumes from state_dict",
    ),
)

# -- DiskGraph index layout and ingest config ---------------------------------
DIMENSIONS += (
    Dimension(
        "execution.num_shards",
        Group.EXECUTION,
        (1, 3),
        "DiskGraph index shards; answers must not depend on them",
    ),
    Dimension(
        "execution.num_threads",
        Group.EXECUTION,
        (1, 4),
        "DiskGraph ingest and sampler threads; answers must not depend on them",
    ),
    Dimension(
        "schema_shape.parallel_relation",
        Group.SCHEMA_SHAPE,
        ("absent", "present"),
        "a second relation declared over the same FK column as the first one",
    ),
    Dimension(
        "execution.index_layout",
        Group.EXECUTION,
        ("other_shards", "as_built"),
        "an existing DiskGraph index opened again with another shard count, or the same",
    ),
    Dimension(
        "execution.drop_last",
        Group.EXECUTION,
        ("token_budget", "sample_count"),
        "drop_last checked under a token budget as well as a sample count, or a sample count only",
    ),
)
#: Engine settings crossed whole: every combination occurs in each seed batch.
CROSSED: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {"diskgraph": ("execution.num_shards", "execution.num_threads")}
)
# -- Raw sampling: graph neighborhoods and event sequences --------------------
#: Read by conformance.graph_cases and conformance.sequence_cases only, so a gap
#: scoped by one of these never narrows the aggregate generator.
SAMPLING_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        "schema_shape.graph_path",
        Group.SCHEMA_SHAPE,
        (
            "one_hop",
            "many_to_one",
            "wide",
            "three_hop",
            "wide_back",
            "fan_in",
            "back_and_forth",
        ),
        "graph layers from a users root: 1-3 layers, 1-2 samples per layer, many-to-one "
        "hops; the last three traverse a relation again",
    ),
    Dimension(
        "schema_shape.graph_orientation",
        Group.SCHEMA_SHAPE,
        ("aligned", "opposite"),
        "the events-shops relation declared FK to PK as DiskGraph stores it, or PK to FK",
    ),
    Dimension(
        "contract.graph_direction",
        Group.CONTRACT,
        ("forward", "reverse"),
        "the users-events relation declared so the users-to-events hop runs forward "
        "(PK to FK, opposite to storage) or reverse",
    ),
    Dimension("contract.graph_fanout", Group.CONTRACT, (1, 2), "fanout of every traversal"),
    Dimension(
        "contract.graph_temporal",
        Group.CONTRACT,
        ("inherit", "static"),
        "users-to-events traversals filter by the cutoff, or are static ('recent' still orders)",
    ),
    Dimension(
        "time.graph_ties",
        Group.TIME,
        ("distinct", "tied"),
        "all neighbor candidates dated apart, or more candidates than the fanout share the "
        "timestamp at the fanout limit",
    ),
    Dimension(
        "contract.event_order",
        Group.CONTRACT,
        ("ascending", "descending"),
        "EventSpec.order of the limited event component",
    ),
    Dimension(
        "contract.event_limit",
        Group.CONTRACT,
        ("none", "limited"),
        "EventSpec.limit of that component: none or 2",
    ),
    Dimension(
        "contract.event_keep",
        Group.CONTRACT,
        ("first", "last"),
        "EventSource.keep of the merged and the single-component inputs",
    ),
    Dimension(
        "contract.event_max",
        Group.CONTRACT,
        ("none", "limited"),
        "EventSource.max_events of the merged input; the single-component input is always "
        "capped, so the model path pushes it down",
    ),
    Dimension(
        "contract.event_delta_unit",
        Group.CONTRACT,
        ("seconds", "minutes"),
        "EventTimeDelta.unit",
    ),
)
DIMENSIONS += SAMPLING_DIMENSIONS
#: Each generator covers every value pair of its own dimensions.
GRAPH_DIMENSIONS: tuple[str, ...] = (
    "schema_shape.graph_path",
    "schema_shape.graph_orientation",
    "contract.graph_direction",
    "contract.graph_fanout",
    "contract.graph_temporal",
    "time.graph_ties",
    "time.cutoffs",
)
SEQUENCE_DIMENSIONS: tuple[str, ...] = (
    "contract.event_order",
    "contract.event_limit",
    "contract.event_keep",
    "contract.event_max",
    "contract.event_delta_unit",
    "time.ties",
    "time.cutoffs",
)

BY_NAME: Mapping[str, Dimension] = MappingProxyType({d.name: d for d in DIMENSIONS})

_ROW_DIMENSIONS = (
    "time.range",
    "time.cutoff_placement",
    "time.ties",
    "time.cutoffs",
    "values.null_rate",
    "values.empty_window",
    "values.single_row",
    "schema_shape.path",
    "contract.distinct",
)

#: What each contract feature or reduction is sensitive to; coverage goals apply only there.
SENSITIVE: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        **{f"reduction:{operation}": _ROW_DIMENSIONS for operation in sorted(ALL_REDUCTIONS)},
        "time.per_anchor_cutoff": ("time.cutoffs", "time.cutoff_placement", "time.ties"),
        "rows.distinct": ("contract.distinct", "schema_shape.path"),
        "rows.multi_hop_path": ("schema_shape.path", "values.empty_window"),
        "time.relative_windows": ("contract.window", "time.cutoff_placement", "time.ties"),
        "aggregate.nested_group_by": (
            "contract.group_by",
            "contract.distinct",
            "schema_shape.path",
            "values.null_rate",
        ),
        "targets": ("contract.target", "time.cutoffs", "time.cutoff_placement", "time.ties"),
        "output_format:model": (
            "execution.output_format",
            "schema_shape.path",
            "time.cutoffs",
            "values.empty_window",
            "values.null_rate",
        ),
    }
)

#: Execution dimensions the delivery property (sharding, resume, determinism) varies.
DELIVERY: tuple[str, ...] = ("execution.world_size", "execution.epochs", "execution.resume")
#: Covered whole rather than pairwise: every case declares every reduction and
#: window; strategies and file layouts rotate so each occurs in every batch.
BUNDLED: tuple[str, ...] = ("contract.reduction", "contract.window", "contract.target")
ROTATED: tuple[str, ...] = ("contract.strategy", "source.files")
#: Also rotated; a known gap or an inapplicable rule keeps a value out (active_values).
ROTATED += ("schema_shape.parallel_relation", "execution.index_layout", "execution.drop_last")


@dataclass(frozen=True)
class KnownGap:
    """An engine defect a property found; the generator stays out of it.

    ``when`` is a conjunction: the gap covers cases whose dimension values all lie
    in the listed sets. ``property`` names the property that exposed it, and
    replaying ``repro`` must still fail with a message containing ``symptom``.
    """

    id: str
    engine: str
    property: str
    when: Mapping[str, tuple[DimensionValue, ...]]
    reason: str
    repro: str
    symptom: str

    def __post_init__(self) -> None:
        if not self.when or any(name not in BY_NAME for name in self.when):
            raise ValueError(f"{self.id}: scope the gap by registered dimensions")
        object.__setattr__(self, "when", MappingProxyType(dict(self.when)))

    def applies(self, engine: str) -> bool:
        return self.engine == engine

    def covers(self, values: Mapping[str, DimensionValue]) -> bool:
        return all(name in values and values[name] in self.when[name] for name in self.when)


_ENGINE_TAG = {"in_memory": "REF", "diskgraph": "DG"}

#: DiskGraph's shortfalls against the spec, each replayed from conformance/known_gaps/.
KNOWN_GAPS: tuple[KnownGap, ...] = (
    # -- Functional areas: accepted at check(), refused at prepare() or batches() --
    *(
        KnownGap(
            f"GAP-LATE-{strategy.upper().replace('_', '-')}-{_ENGINE_TAG[engine]}",
            engine,
            "refusal_timing",
            {"contract.strategy": (strategy,)},
            f"accepted at check(), fails at {stage}: {reason}",
            f"conformance/known_gaps/{repro}.json",
            symptom,
        )
        for strategy, engine, stage, reason, repro, symptom in (
            (
                "column_rename",
                "diskgraph",
                "prepare()",
                "Parquet(columns=...) renames are refused by the materializer",
                "repro-347e53900d7b9027",
                "requires matching physical column names",
            ),
            (
                "unsupported_type",
                "diskgraph",
                "prepare()",
                "a list-typed Parquet column is refused when the binding is checked",
                "repro-6d4a103649f844e5",
                "unsupported physical type list<element: double>",
            ),
            (
                "missing_column",
                "diskgraph",
                "prepare()",
                "a declared column absent from the Parquet file",
                "repro-ea4df9342ba46532",
                "is missing columns ['amount']",
            ),
            (
                "int64_clock",
                "diskgraph",
                "prepare()",
                "an int64 clock column under a declared timestamp is refused at binding",
                "repro-17377cd26ed0ecf8",
                "physical int64 resolves to DType(name='int64'",
            ),
            (
                "duplicate_layer_relation",
                "diskgraph",
                "batches()",
                "two samples of one physical relation in one graph layer",
                "repro-0e6731dc166334bb",
                "is sampled more than once in one layer",
            ),
        )
    ),
)

# -- DiskGraph index layout and ingest config (continued) ---------------------
KNOWN_GAPS += (
    KnownGap(
        "GAP-PARALLEL-FK-DG",
        "diskgraph",
        "refusal_timing",
        {"schema_shape.parallel_relation": ("present",)},
        "F2: two relations over one FK column pass check(); the loader emits a duplicate fkey "
        "(materialization.py _compile_diskgraph_config) and native ingest fails at prepare()",
        "conformance/known_gaps/repro-99c19aba95fd9583.json",
        "fkey bitmap length mismatch",
    ),
    KnownGap(
        "GAP-INDEX-LAYOUT-DG",
        "diskgraph",
        "index_layout",
        {"execution.index_layout": ("other_shards",)},
        "index layout settings are not part of the index binding; a mismatched index is "
        "reused silently (materialization.py ensure_diskgraph_index), though ties at a "
        "graph fanout limit differ by shard count (GAP-GRAPH-TIES-DG)",
        "conformance/known_gaps/repro-f18d19cbed41f126.json",
        "reused a 1-shard index for num_shards=3",
    ),
)

KNOWN_GAPS += (
    KnownGap(
        "GAP-GRAPH-TIES-DG",
        "diskgraph",
        "graph_tie_invariance",
        {"time.graph_ties": ("tied",)},
        "which tied neighbors 'recent' keeps at the fanout limit follows the index's physical "
        "row layout, not the data: it depends on Parquet row order and num_shards, and with "
        "3 shards and 4 threads differs across fresh builds of the same data, while sampling "
        "one built index repeats (G-GRAPH-001 rule 2); graph_inference_exact leaves tied "
        "samples out where builds aren't deterministic (G-INFERENCE-001 boundaries)",
        "conformance/known_gaps/repro-7657161068432210.json",
        "num_shards=3 changed graphs",
    ),
)


#: In-memory engine bugs (conformance/README.md, Differential testing): replayed like
#: known gaps, but not spec gaps.
IN_MEMORY_DIVERGENCES: tuple[KnownGap, ...] = (
    KnownGap(
        "DIV-TIMED-HOP-MEM",
        "in_memory",
        "point_in_time",
        {"schema_shape.path": ("depth2_timed",)},
        "the cutoff applies only at the destination, so rows reached through an "
        "intermediate row dated after the cutoff leak (engines/in_memory/samples.py _select_rows)",
        "conformance/in_memory_divergences/repro-aec68015a13dd97d.json",
        "a0: {'count': 16, 'count_distinct': 3",
    ),
)

#: Every recorded defect whose repro must still fail as recorded.
REPLAYED: tuple[KnownGap, ...] = KNOWN_GAPS + IN_MEMORY_DIVERGENCES


@dataclass(frozen=True)
class EngineRefusal:
    """A region an engine refuses at check() by design; ``strategy`` keeps checking it."""

    engine: str
    when: Mapping[str, tuple[DimensionValue, ...]]
    reason: str
    strategy: str

    def covers(self, values: Mapping[str, DimensionValue]) -> bool:
        return all(name in values and values[name] in self.when[name] for name in self.when)


ENGINE_REFUSALS: tuple[EngineRefusal, ...] = (
    EngineRefusal(
        "diskgraph",
        {"schema_shape.path": ("depth2_timed",)},
        "DiskGraph refuses time-filtering an intermediate table",
        "timed_intermediate",
    ),
    # -- Functional areas: refusals keyed by their own strategy stay generated --
    *(
        EngineRefusal("diskgraph", {"contract.strategy": (strategy,)}, reason, strategy)
        for strategy, reason in (
            ("active_interval", "DiskGraph does not filter rows by an active interval"),
            ("clock_override", "DiskGraph cannot override a destination table's clock"),
            ("current_events", "DiskGraph does not build current events"),
            ("dated_values", "DiskGraph does not build dated entity values"),
            ("non_direct_field", "DiskGraph event fields read the event row only"),
            ("object_store", "DiskGraph reads local Parquet paths only"),
            ("static_traversal", "DiskGraph cannot disable filtering of a timed table"),
            ("uniform_graph", "DiskGraph samples graphs by recency only"),
        )
    ),
)


@dataclass(frozen=True)
class Inapplicable:
    """Values an engine has no implementation of by design, so none are generated.

    Not a check() refusal: the option lives outside the contract (a binding or a
    batches() argument), so nothing can refuse it there. One with ``refused_at``
    is generated instead: refusal_timing requires that call to refuse it before
    delivering anything, with ``message`` in the error.
    """

    engine: str
    when: Mapping[str, tuple[DimensionValue, ...]]
    reason: str
    refused_at: str | None = None
    message: str | None = None

    def covers(self, values: Mapping[str, DimensionValue]) -> bool:
        return all(name in values and values[name] in self.when[name] for name in self.when)


INAPPLICABLE: tuple[Inapplicable, ...] = (
    Inapplicable(
        "in_memory",
        {"execution.output_format": ("model",)},
        "the reference builds inspectable batches only; output_format='model' is DiskGraph's",
    ),
    Inapplicable(
        "in_memory",
        {
            "contract.strategy": (
                "column_rename",
                "int64_clock",
                "missing_column",
                "model_format_limit",
                "object_store",
                "unsupported_type",
            )
        },
        "Parquet bindings and the model path don't exist for in-memory rows",
    ),
    Inapplicable(
        "in_memory",
        {"execution.index_layout": ("other_shards",)},
        "the reference builds no index",
    ),
    # -- batches() arguments check() never sees; refused there, eagerly --
    *(
        Inapplicable(
            engine,
            {"contract.strategy": (strategy,)},
            reason,
            refused_at="batches()",
            message=message,
        )
        for engine, strategy, reason, message in (
            (
                "diskgraph",
                "max_tokens_world_size",
                "Batching.max_tokens with world_size > 1 (shared task code)",
                "Batching.max_tokens with world_size > 1 is not supported yet",
            ),
            (
                "in_memory",
                "max_tokens_world_size",
                "Batching.max_tokens with world_size > 1 (shared task code)",
                "Batching.max_tokens with world_size > 1 is not supported yet",
            ),
            (
                "diskgraph",
                "model_format_limit",
                "output_format='model' refuses a Raw string column",
                "requires inspectable output or an explicit encoder",
            ),
        )
    ),
)

ENGINE_REFUSALS += (
    EngineRefusal(
        "diskgraph",
        {"contract.graph_temporal": ("static",)},
        "DiskGraph refuses a static traversal into a timed table",
        "graph_static",
    ),
)


def known_gaps(engine: str, *, property: str | None = None) -> tuple[KnownGap, ...]:
    """Known gaps and in-memory divergences on ``engine``."""

    return tuple(
        gap
        for gap in KNOWN_GAPS + IN_MEMORY_DIVERGENCES
        if gap.applies(engine) and (property is None or gap.property == property)
    )


def blocked_regions(engine: str) -> tuple[KnownGap | EngineRefusal | Inapplicable, ...]:
    """Regions generated data stays out of: known gaps, declared engine refusals and
    inapplicable values. A refusal keyed only by its own strategy is what
    refusal_timing checks, so it stays generated.
    """

    refused = tuple(
        r
        for r in ENGINE_REFUSALS
        if r.engine == engine and dict(r.when) != {"contract.strategy": (r.strategy,)}
    )
    inapplicable = tuple(r for r in INAPPLICABLE if r.engine == engine and r.refused_at is None)
    return known_gaps(engine) + refused + inapplicable


def late_refusal(engine: str, strategy: str) -> Inapplicable | None:
    """The batches() refusal ``engine`` declares for ``strategy``, if any."""

    return next(
        (
            r
            for r in INAPPLICABLE
            if r.engine == engine
            and r.refused_at is not None
            and dict(r.when) == {"contract.strategy": (strategy,)}
        ),
        None,
    )


def active_values(name: str, engine: str) -> tuple[DimensionValue, ...]:
    """Values the generator may use for ``engine``, minus single-dimension blocked regions."""

    blocked = {
        value
        for region in blocked_regions(engine)
        if set(region.when) == {name}
        for value in region.when[name]
    }
    return tuple(value for value in BY_NAME[name].values if value not in blocked)
