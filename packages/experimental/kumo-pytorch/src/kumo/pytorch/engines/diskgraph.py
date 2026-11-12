# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DiskGraph-backed graph and temporal-row execution."""

from __future__ import annotations

import datetime as dt
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from kumo.pytorch.anchors import AnchorTable
from kumo.pytorch.column_stats import (
    ColumnStats,
    StatRequest,
    cutoff_seconds,
    read_column_stats,
)
from kumo.pytorch.contract.batch import EntitySource, EventSource, Input
from kumo.pytorch.contract.dtypes import DType, arrow_float64
from kumo.pytorch.contract.sample import (
    Aggregate,
    BuiltSample,
    CalendarPart,
    Compare,
    Constant,
    Derived,
    EventSpec,
    EventTimeDelta,
    ExternalTarget,
    Field,
    GroupBy,
    MapValues,
    Rows,
    RowTarget,
    SampleRequest,
    SampleSpec,
    TargetValue,
    _normalize_supplied_targets,
    _typed_map_groups,
    _value_evaluation_order,
    destination_table,
    evaluate_derived,
    sample_value_types,
    source_column_types,
    value_dtype,
)
from kumo.pytorch.contract.schema import DataSchema, table_clock
from kumo.pytorch.data.model import ModelRowTarget, ModelSequence, ModelSourceBatch
from kumo.pytorch.data.records import (
    EntityHistory,
    EntityState,
    Event,
    FieldValue,
    cutoff_to_epoch,
    timestamp_to_epoch,
)
from kumo.pytorch.dataset import Dataset, Parquet, TableSource
from kumo.pytorch.engine import AnchorTask, GraphRoot, SampleSource, register_engine
from kumo.pytorch.engines.parquet_anchors import ParquetAnchors, check_parquet_anchors
from kumo.pytorch.engines.unsupported import DISKGRAPH_UNSUPPORTED, Unsupported
from kumo.pytorch.materialization import _DiskGraphIndex, ensure_diskgraph_index
from kumo.pytorch.spec import DataLoaderSpec, ValidationIssue

__all__ = ["DiskGraphEngine"]


def _refusal(code: str, path: str, message: str) -> ValidationIssue:
    entry = DISKGRAPH_UNSUPPORTED[code]
    return ValidationIssue(path, message, entry.next_step, code)


def _sample_refusals(spec: SampleSpec, path: str = "sample") -> list[ValidationIssue]:
    """Refusals that need only the sample declaration, in contract order."""

    issues: list[ValidationIssue] = []
    for name, code in (
        ("dated_entity_values", "DG-DATED-ENTITY-VALUES"),
        ("current_events", "DG-CURRENT-EVENTS"),
        ("candidate_sets", "DG-CANDIDATE-SETS"),
    ):
        if getattr(spec, name):
            issues.append(_refusal(code, path, f"DiskGraph does not support SampleSpec.{name}"))
    if spec.entity.filters:
        issues.append(
            _refusal(
                "DG-ROW-FILTER",
                f"{path}.entity.filters",
                "DiskGraph does not support entity row filters",
            )
        )

    def rows(value: Rows, location: str) -> None:
        if value.filters:
            issues.append(
                _refusal(
                    "DG-ROW-FILTER", f"{location}.filters", "DiskGraph does not support row filters"
                )
            )
        if value.active is not None:
            issues.append(
                _refusal(
                    "DG-ACTIVE-INTERVAL",
                    f"{location}.active",
                    "DiskGraph does not support active intervals",
                )
            )

    def source(value: Rows | GroupBy, location: str) -> None:
        if isinstance(value, Rows):
            rows(value, location)
        else:
            source(value.source, f"{location}.source")

    def fields(values: Mapping[str, Any], location: str, context: str) -> None:
        for name, field in values.items():
            if isinstance(field, Field) and (field.path or field.time_column is not None):
                issues.append(
                    _refusal(
                        "DG-JOINED-FIELD",
                        f"{location}.{name}",
                        f"DiskGraph {context} field {name!r} must be a direct event-row "
                        "column or constant",
                    )
                )

    for name, value in spec.entity_values.items():
        location = f"{path}.entity_values.{name}"
        if isinstance(value, Aggregate):
            source(value.source, f"{location}.source")
        elif isinstance(value, Compare):
            source(value.left.source, f"{location}.source")
        elif (
            isinstance(value, Field) and (value.path or value.time_column is not None)
        ) or not isinstance(value, (Field, Constant, Derived)):
            issues.append(
                _refusal(
                    "DG-ENTITY-VALUE-KIND",
                    location,
                    f"DiskGraph entity value {name!r} must be an aggregate or a direct root field",
                )
            )
    for name, event in spec.events.items():
        location = f"{path}.events.{name}"
        rows(event.rows, f"{location}.rows")
        fields(event.fields, f"{location}.fields", f"event {name!r}")
    for name, target in spec.targets.items():
        location = f"{path}.targets.{name}"
        if isinstance(target, ExternalTarget):
            continue
        if isinstance(target, (Aggregate, Compare)):
            aggregate = target.left if isinstance(target, Compare) else target
            source(aggregate.source, f"{location}.source")
        else:
            rows(target.rows, f"{location}.rows")
            fields(target.fields, f"{location}.fields", f"row target {name!r}")
    for name, neighborhood in spec.neighborhoods.items():
        for layer_index, layer in enumerate(neighborhood.layers):
            for sample_index, traversal in enumerate(layer.samples):
                # Native uniform sampling ignores each seed's own timestamp (one coarse
                # NodeCutoff per request, diskgraph-engine engine.rs); only recent is exact.
                if traversal.strategy == "uniform":
                    issues.append(
                        _refusal(
                            "DG-UNIFORM-GRAPH",
                            f"{path}.neighborhoods.{name}.layers[{layer_index}]"
                            f".samples[{sample_index}].strategy",
                            "DiskGraph does not sample graph neighbors uniformly",
                        )
                    )
    return issues


class DiskGraphEngine:
    """Ensure a local base index during ``prepare()``, then sample it.

    The DiskGraph engine is the one to use for real data. It indexes data on
    local SSDs and retrieves time-correct context for training and batch
    prediction, such as event histories and graph neighborhoods, without keeping
    the full index in RAM.

    The base index is built from local Parquet files and is independent of
    anchors, fitted encoders and output layout, so several contracts can reuse
    it. Its manifest binds the schema, physical file metadata (not a content
    snapshot) and resolved types; every file's declared columns and physical
    types are checked before ingestion. Native ingestion rules apply: rows with
    NULL clocks and other rejected rows are dropped and counted.

    Supported: direct entity fields, historical event fields, typed scalar
    calculations, aggregates and comparisons, nested grouping, generated scalar
    and row labels, recent graph neighborhoods with node fields, and columnar
    ``output_format="model"`` batches. What it doesn't support yet is listed in
    `DiskGraphEngine.unsupported`; each entry is refused with its code and a next
    step, before any data is read.

    :param index: directory of the base index.
    :param num_threads: native sampling threads.
    :param num_shards: index shards.
    :param ingest: ``"if_missing"`` builds the index when absent; ``"never"``
        requires an existing one. A mismatching manifest is an error, never
        permission to reuse.
    :param encoder_cutoffs: ``encoder_cutoff`` values whose column statistics an ingest computes.
    """

    name = "diskgraph"
    #: What this engine doesn't support yet, keyed by issue code (read-only).
    unsupported: Mapping[str, Unsupported] = DISKGRAPH_UNSUPPORTED

    def __init__(
        self,
        *,
        index: str | Path,
        num_threads: int | None = None,
        num_shards: int = 1,
        ingest: Literal["if_missing", "never"] = "if_missing",
        encoder_cutoffs: Sequence[str | dt.datetime | None] = (),
    ):
        if ingest not in {"if_missing", "never"}:
            raise ValueError("ingest must be 'if_missing' or 'never'")
        if isinstance(encoder_cutoffs, str):
            raise TypeError("encoder_cutoffs is a list of cutoffs; wrap a single one in a list")
        self.index = Path(index)
        self.num_threads = num_threads
        self.num_shards = num_shards
        self.ingest = ingest
        self.encoder_cutoffs = tuple(dict.fromkeys(cutoff_seconds(v) for v in encoder_cutoffs))

    def check_contract(self, contract: DataLoaderSpec) -> tuple[ValidationIssue, ...]:
        """Refuse the entries of `unsupported` the contract uses, before any data is read."""
        issues = _sample_refusals(contract.sample)
        tables = {table.name: table for table in contract.schema.tables}
        relations = {relation.name: relation for relation in contract.schema.relations}
        for index, relation in enumerate(contract.schema.relations):
            if (relation.left_key == tables[relation.left_table].primary_key) == (
                relation.right_key == tables[relation.right_table].primary_key
            ):
                issues.append(
                    _refusal(
                        "DG-RELATION-KEYS",
                        f"schema.relations[{index}]",
                        f"DiskGraph relation {relation.name!r} needs exactly one primary-key side",
                    )
                )

        def check_rows(rows: Rows, path: str) -> None:
            current = contract.sample.entity.table
            for index, relation_name in enumerate(rows.path):
                relation = relations.get(relation_name)
                if relation is None:
                    break  # General contract validation reports unknown relations.
                if current == relation.left_table:
                    destination = relation.right_table
                elif current == relation.right_table:
                    destination = relation.left_table
                else:
                    break  # General contract validation reports disconnected paths.
                if index < len(rows.path) - 1 and table_clock(tables[destination]) is not None:
                    issues.append(
                        _refusal(
                            "DG-TIMED-INTERMEDIATE",
                            f"{path}.path[{index}]",
                            f"DiskGraph cannot time-filter intermediate table {destination!r}",
                        )
                    )
                current = destination

        def check_source(source: Rows | GroupBy, path: str) -> None:
            if isinstance(source, Rows):
                check_rows(source, path)
            else:
                check_source(source.source, f"{path}.source")

        def check_reduction(reduction: Any, path: str, columns: Mapping[str, DType]) -> None:
            source = columns.get(reduction.column) if reduction.column is not None else None
            if source is not None and source.name in {"decimal128", "decimal256", "bytes"}:
                message = (
                    f"DiskGraph does not support {source.name} operands "
                    f"in {reduction.operation} reductions"
                )
                issues.append(_refusal("DG-REDUCTION-OPERAND", path, message))

        def check_aggregate(value: Aggregate, path: str) -> None:
            check_source(value.source, f"{path}.source")
            try:
                columns = source_column_types(
                    contract.schema, contract.sample.entity.table, value.source
                )
            except (KeyError, ValueError, StopIteration):
                return
            check_reduction(value, path, columns)
            source = value.source
            location = f"{path}.source"
            while isinstance(source, GroupBy):
                columns = source_column_types(
                    contract.schema, contract.sample.entity.table, source.source
                )
                for name, reduction in source.values.items():
                    check_reduction(reduction, f"{location}.values.{name}", columns)
                source = source.source
                location += ".source"

        for name, value in contract.sample.entity_values.items():
            if isinstance(value, (Aggregate, Compare)):
                check_aggregate(
                    value.left if isinstance(value, Compare) else value,
                    f"sample.entity_values.{name}",
                )
        for name, event in contract.sample.events.items():
            check_rows(event.rows, f"sample.events.{name}.rows")
        for name, target in contract.sample.targets.items():
            if isinstance(target, (Aggregate, Compare)):
                check_aggregate(
                    target.left if isinstance(target, Compare) else target, f"sample.targets.{name}"
                )
            elif isinstance(target, RowTarget):
                check_rows(target.rows, f"sample.targets.{name}.rows")

        for name, neighborhood in contract.sample.neighborhoods.items():
            for layer_index, layer in enumerate(neighborhood.layers):
                for sample_index, traversal in enumerate(layer.samples):
                    relation = relations.get(traversal.relation)
                    if relation is None:
                        continue
                    destination = (
                        relation.right_table
                        if traversal.direction == "forward"
                        else relation.left_table
                    )
                    clock = table_clock(tables[destination])
                    path = (
                        f"sample.neighborhoods.{name}.layers[{layer_index}].samples[{sample_index}]"
                    )
                    if traversal.time_column is not None and traversal.time_column != clock:
                        issues.append(
                            _refusal(
                                "DG-GRAPH-CLOCK-OVERRIDE",
                                path + ".time_column",
                                "DiskGraph cannot override the destination table clock",
                            )
                        )
                    if traversal.temporal == "static" and clock is not None:
                        issues.append(
                            _refusal(
                                "DG-STATIC-TIMED-TRAVERSAL",
                                path + ".temporal",
                                "DiskGraph cannot disable filtering of a timed table",
                            )
                        )
        return tuple(issues)

    @staticmethod
    def check_sources(sources: Mapping[str, TableSource], path: str) -> tuple[ValidationIssue, ...]:
        """Refuse sources other than local Parquet, before any data is read.

        ``sources`` maps table names to their sources; an empty name means the
        source at ``path`` itself, such as an anchor table's.
        """
        unreadable = sorted({source.kind for source in sources.values()} - {"parquet"})
        issues = []
        if unreadable:
            message = f"engine 'diskgraph' does not read {unreadable} sources"
            issues.append(_refusal("DG-SOURCE-KIND", path, message))
        for name, source in sources.items():
            if isinstance(source, Parquet) and "://" in source.path:
                issues.append(
                    _refusal(
                        "DG-REMOTE-PARQUET",
                        f"{path}.{name}.path" if name else f"{path}.path",
                        f"local Parquet paths are required; got {source.path!r}",
                    )
                )
        return tuple(issues)

    @staticmethod
    def check_anchors(table: AnchorTable) -> tuple[ValidationIssue, ...]:
        """Issues with a Parquet anchor table's binding, without reading its rows."""
        return check_parquet_anchors(table)

    def open(
        self,
        dataset: Dataset,
        schema: DataSchema,
        *,
        column_stats: Sequence[StatRequest] = (),
    ) -> SampleSource:
        ingests = bool(column_stats and self.encoder_cutoffs) and not self.index.exists()
        artifact = ensure_diskgraph_index(
            schema,
            dataset,
            self.index,
            num_shards=self.num_shards,
            num_threads=self.num_threads,
            if_missing=self.ingest == "if_missing",
            column_stats=(column_stats, self.encoder_cutoffs) if ingests else None,
        )
        return DiskGraphSource(
            artifact,
            schema,
            num_threads=self.num_threads,
        )

    def materialize_anchors(self, table: AnchorTable) -> AnchorTask:
        # anchors are read from Parquet independently of the graph index: an anchor
        # table is per task, the index is per dataset
        return ParquetAnchors(table)


class DiskGraphSource:
    """Adapt direct DiskGraph graph/temporal results to kumo-pytorch's source contract."""

    def __init__(
        self,
        artifact: _DiskGraphIndex,
        schema: DataSchema,
        *,
        num_threads: int | None,
    ) -> None:
        try:
            import diskgraph
        except ImportError as error:  # pragma: no cover - optional installation
            raise ImportError(
                "DiskGraphEngine requires the optional 'diskgraph' package"
            ) from error
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
        except ImportError as error:  # pragma: no cover - installed with DiskGraph
            raise ImportError("DiskGraphEngine requires pyarrow") from error

        self.schema = schema
        self._index = artifact.path
        metadata = json.loads(Path(artifact.path, "index_metadata.json").read_text())
        self._clock = metadata.get("timestamp_unit", "seconds")
        self._num_threads = num_threads
        self._tables = {table.name: table for table in schema.tables}
        self._relations = {relation.name: relation for relation in schema.relations}
        self._relation_mapping = artifact.manifest["relation_mapping"]
        self._diskgraph = diskgraph
        self.pack_token_stream = diskgraph.pack_token_stream
        self._pa = pa
        self._pc = pc
        self._sampler = diskgraph.SamplerEngine(
            artifact.path,
            num_threads=num_threads,
        )
        mapping = json.loads(self._sampler.get_type_mapping())
        self._node_types = tuple(mapping["node_types"])
        self._relations_by_id = tuple(mapping["edge_relations"])
        self._last_temporal_profile: Mapping[str, Any] | None = None
        self._last_length_profile: Mapping[str, Any] | None = None
        self._last_graph_profile: Mapping[str, Any] | None = None

    def string_lookup(self, keys):
        return self._diskgraph.StringLookup(keys).lookup

    def build(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
    ) -> list[BuiltSample]:
        self._validate_sample(spec)
        requests = tuple(
            dataclass_replace(request, targets=_normalize_supplied_targets(spec, request.targets))
            for request in requests
        )
        self._validate_root_availability(spec, requests)
        if not spec.entity_values and not spec.events and not spec.targets:
            self._last_temporal_profile = None
            return [self._empty_sample(spec, request) for request in requests]

        total_started = time.perf_counter_ns()
        lookup_started = time.perf_counter_ns()
        root_ids = self._root_ids(spec.entity.table, requests)
        lookup_ns = time.perf_counter_ns() - lookup_started
        cutoffs = [cutoff_to_epoch(request.as_of_time) for request in requests]

        feature_started = time.perf_counter_ns()
        direct_values = self._direct_entity_arrays(spec, root_ids)
        feature_ns = time.perf_counter_ns() - feature_started

        if not self._needs_temporal_query(spec):
            built = self._direct_samples(spec, requests, cutoffs, direct_values)
            self._last_temporal_profile = {
                "samples": len(requests),
                "output_format": "inspectable",
                "root_lookup_ns": lookup_ns,
                "root_feature_fetch_ns": feature_ns,
                "plan_compile_ns": 0,
                "native_call_ns": 0,
                "python_adaptation_ns": 0,
                "total_ns": time.perf_counter_ns() - total_started,
                "native_timings_ns": {},
                "native_stats": {},
            }
            return built

        compile_started = time.perf_counter_ns()
        compiled = _TemporalPlanCompiler(self).compile(spec)
        compile_ns = time.perf_counter_ns() - compile_started
        native_started = time.perf_counter_ns()
        result = self._sampler.query_temporal(
            {spec.entity.table: root_ids},
            {spec.entity.table: cutoffs},
            json.dumps(compiled.plan),
        )
        native_call_ns = time.perf_counter_ns() - native_started

        adapt_started = time.perf_counter_ns()
        built = self._adapt_temporal_result(
            spec,
            requests,
            cutoffs,
            compiled,
            result,
            direct_values,
        )
        adaptation_ns = time.perf_counter_ns() - adapt_started
        self._last_temporal_profile = {
            "samples": len(requests),
            "output_format": "inspectable",
            "root_lookup_ns": lookup_ns,
            "root_feature_fetch_ns": feature_ns,
            "plan_compile_ns": compile_ns,
            "native_call_ns": native_call_ns,
            "python_adaptation_ns": adaptation_ns,
            "total_ns": time.perf_counter_ns() - total_started,
            "native_timings_ns": dict(result["timings_ns"]),
            "native_stats": dict(result["stats"]),
        }
        return built

    def build_model(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
    ) -> ModelSourceBatch:
        return self.build_model_ordered(spec, requests)

    def build_model_ordered(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
        *,
        event_orders: Mapping[str, Sequence[str]] | None = None,
    ) -> ModelSourceBatch:
        """Keep temporal values in Arrow buffers for direct model assembly."""

        self._validate_sample(spec)
        requests = tuple(
            dataclass_replace(request, targets=_normalize_supplied_targets(spec, request.targets))
            for request in requests
        )
        self._validate_root_availability(spec, requests)
        if not spec.entity_values and not spec.events and not spec.targets:
            self._last_temporal_profile = None
            request_targets: dict[str, list[Any]] = {}
            for anchor, request in enumerate(requests):
                for target in request.targets:
                    request_targets.setdefault(target.name, [None] * len(requests))[anchor] = (
                        target.value
                    )
            return ModelSourceBatch(
                sample_ids=tuple(request.sample_id for request in requests),
                entity_ids=tuple(request.entity_id for request in requests),
                evaluation_times=tuple(cutoff_to_epoch(request.as_of_time) for request in requests),
                entity_type=spec.entity.entity_type,
                entity_values={},
                events={},
                targets=request_targets,
                row_targets={},
                splits=tuple(request.split for request in requests),
                request_metadata=tuple(
                    {"split": request.split, **request.metadata} for request in requests
                ),
            )
        total_started = time.perf_counter_ns()
        lookup_started = time.perf_counter_ns()
        root_ids = self._root_ids(spec.entity.table, requests)
        lookup_ns = time.perf_counter_ns() - lookup_started
        cutoffs = [cutoff_to_epoch(request.as_of_time) for request in requests]

        feature_started = time.perf_counter_ns()
        direct_values = self._direct_entity_arrays(spec, root_ids)
        feature_ns = time.perf_counter_ns() - feature_started

        if not self._needs_temporal_query(spec):
            built = self._direct_model_batch(spec, requests, cutoffs, direct_values)
            self._last_temporal_profile = {
                "samples": len(requests),
                "output_format": "model",
                "root_lookup_ns": lookup_ns,
                "root_feature_fetch_ns": feature_ns,
                "plan_compile_ns": 0,
                "native_call_ns": 0,
                "model_adaptation_ns": 0,
                "total_ns": time.perf_counter_ns() - total_started,
                "native_timings_ns": {},
                "native_stats": {},
            }
            return built

        compile_started = time.perf_counter_ns()
        compiled = _TemporalPlanCompiler(self).compile(
            spec, event_orders=event_orders or {name: ("ascending",) for name in spec.events}
        )
        compile_ns = time.perf_counter_ns() - compile_started
        native_started = time.perf_counter_ns()
        result = self._sampler.query_temporal(
            {spec.entity.table: root_ids},
            {spec.entity.table: cutoffs},
            json.dumps(compiled.plan),
            output_format="model",
        )
        native_call_ns = time.perf_counter_ns() - native_started

        adapt_started = time.perf_counter_ns()
        built = self._adapt_model_result(
            spec,
            requests,
            cutoffs,
            compiled,
            result,
            direct_values,
        )
        adaptation_ns = time.perf_counter_ns() - adapt_started
        self._last_temporal_profile = {
            "samples": len(requests),
            "output_format": "model",
            "root_lookup_ns": lookup_ns,
            "root_feature_fetch_ns": feature_ns,
            "plan_compile_ns": compile_ns,
            "native_call_ns": native_call_ns,
            "model_adaptation_ns": adaptation_ns,
            "total_ns": time.perf_counter_ns() - total_started,
            "native_timings_ns": dict(result["timings_ns"]),
            "native_stats": dict(result["stats"]),
        }
        return built

    def model_lengths(
        self,
        spec: SampleSpec,
        input_spec: Any,
        requests: Sequence[SampleRequest],
    ) -> list[int]:
        """Measure exact model lengths without decoding event row values."""

        if not isinstance(input_spec, Input):
            raise ValueError("typed length planning requires a modern Input")
        if isinstance(input_spec.source, EntitySource):
            width = len(input_spec.representation.fields)
            return [width] * len(requests)
        if not isinstance(input_spec.source, EventSource):
            raise ValueError("typed length planning currently supports entity/event inputs")
        if input_spec.source.collection != "history":
            raise ValueError("typed length planning requires historical events")
        self._validate_sample(spec)
        selected_events = {
            name: dataclass_replace(
                spec.events[name],
                fields={"__length__": Constant(value=0.0, dtype=DType("float64"))},
            )
            for name in input_spec.source.events
        }
        length_spec = dataclass_replace(
            spec,
            entity_values={},
            events=selected_events,
            targets={},
        )
        started = time.perf_counter_ns()
        lookup_ns = 0
        native_ns = 0
        native_timings = defaultdict(int)
        native_stats = defaultdict(int)
        measured: list[int] = []
        chunk_size = 4096
        for start in range(0, len(requests), chunk_size):
            chunk = requests[start : start + chunk_size]
            lookup_started = time.perf_counter_ns()
            root_ids = self._root_ids(spec.entity.table, chunk)
            lookup_ns += time.perf_counter_ns() - lookup_started
            cutoffs = [cutoff_to_epoch(request.as_of_time) for request in chunk]
            compiled = _TemporalPlanCompiler(self).compile(length_spec)
            native_started = time.perf_counter_ns()
            result = self._sampler.query_temporal_lengths(
                {spec.entity.table: root_ids},
                {spec.entity.table: cutoffs},
                json.dumps(compiled.plan),
            )
            native_ns += time.perf_counter_ns() - native_started
            for name, value in result["timings_ns"].items():
                native_timings[name] += int(value)
            for name, value in result["stats"].items():
                native_stats[name] += int(value)
            counts = [0] * len(chunk)
            for sequence in result["sequences"]:
                lengths = sequence["lengths"].to_numpy(zero_copy_only=False)
                counts = [
                    total + int(length)
                    for total, length in zip(
                        counts,
                        lengths,
                        strict=True,
                    )
                ]
            if input_spec.source.max_events is not None:
                counts = [min(value, input_spec.source.max_events) for value in counts]
            measured.extend(counts)
        self._last_length_profile = {
            "samples": len(requests),
            "root_lookup_ns": lookup_ns,
            "native_call_ns": native_ns,
            "total_ns": time.perf_counter_ns() - started,
            "native_timings_ns": dict(native_timings),
            "native_stats": dict(native_stats),
        }
        return measured

    @property
    def last_temporal_profile(self) -> Mapping[str, Any] | None:
        """Timing and row counters for the most recent temporal build call."""

        return self._last_temporal_profile

    @property
    def last_length_profile(self) -> Mapping[str, Any] | None:
        """Timing for the most recent typed batch-length planning pass."""

        return self._last_length_profile

    @property
    def last_graph_profile(self) -> Mapping[str, Any] | None:
        """Timing for the most recent graph sampling and feature assembly call."""

        return self._last_graph_profile

    @staticmethod
    def _needs_temporal_query(spec: SampleSpec) -> bool:
        return bool(
            spec.events
            or any(not isinstance(value, ExternalTarget) for value in spec.targets.values())
            or any(isinstance(value, (Aggregate, Compare)) for value in spec.entity_values.values())
        )

    def _direct_entity_arrays(
        self,
        spec: SampleSpec,
        root_ids: Sequence[int],
    ) -> dict[str, Any]:
        direct = {
            name: value
            for name, value in spec.entity_values.items()
            if isinstance(value, (Field, Constant))
        }
        columns = tuple(
            dict.fromkeys(value.column for value in direct.values() if isinstance(value, Field))
        )
        fetched = self._feature_arrays(spec.entity.table, root_ids, columns)
        result = {}
        for name, field in direct.items():
            if isinstance(field, Constant):
                values = self._pa.repeat(
                    self._pa.scalar(field.value, type=field.dtype.to_arrow()),
                    len(root_ids),
                )
            else:
                values = fetched[field.column]
            result[name] = self._typed_array(
                values, value_dtype(self.schema, spec.entity.table, field)
            )
            if isinstance(field, Field) and field.selection == "all":
                result[name] = self._occurrence_array(result[name])
        for name in _value_evaluation_order(spec.entity_values):
            field = spec.entity_values[name]
            if isinstance(field, Derived) and all(source in result for source in field.inputs):
                result[name] = self._derived_array(field, result, path=f"entity_values.{name}")
        return result

    def _direct_samples(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
        cutoffs: Sequence[int],
        direct_values: Mapping[str, Any],
    ) -> list[BuiltSample]:
        built = []
        for anchor, request in enumerate(requests):
            entity_values = self._entity_records(spec, direct_values, anchor)
            built.append(
                BuiltSample(
                    request.sample_id,
                    EntityHistory(
                        request.entity_id,
                        cutoffs[anchor],
                        EntityState(entity_values, ()),
                        (),
                        spec.entity.entity_type,
                    ),
                    request.targets,
                    metadata={"split": request.split, **request.metadata},
                )
            )
        return built

    def _direct_model_batch(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
        cutoffs: Sequence[int],
        direct_values: Mapping[str, Any],
    ) -> ModelSourceBatch:
        request_targets: dict[str, list[Any]] = {}
        for anchor, request in enumerate(requests):
            for target in request.targets:
                request_targets.setdefault(target.name, [None] * len(requests))[anchor] = (
                    target.value
                )
        return ModelSourceBatch(
            sample_ids=tuple(request.sample_id for request in requests),
            entity_ids=tuple(request.entity_id for request in requests),
            evaluation_times=tuple(cutoffs),
            entity_type=spec.entity.entity_type,
            entity_values=dict(direct_values),
            events={},
            targets=request_targets,
            row_targets={},
            splits=tuple(request.split for request in requests),
            request_metadata=tuple(
                {"split": request.split, **request.metadata} for request in requests
            ),
        )

    def column_stats(self, requests: Sequence[StatRequest], cutoff: int | None) -> ColumnStats:
        ticks = None if cutoff is None else self._diskgraph.cutoff_ticks(cutoff, self._clock)
        return read_column_stats(self._index, requests, ticks, self._clock, self._num_threads)

    def fit_graph_fields(
        self,
        sample: SampleSpec,
        spec: Any,
        roots: Sequence[Any],
    ) -> Mapping[tuple[str, str], tuple[Any, ...]]:
        if not spec.node_fields:
            return {}
        _neighborhood, events, _physical_relations = self._sample_graph_events(
            sample,
            spec.source,
            roots,
        )
        global_ids: dict[str, list[int]] = defaultdict(list)
        for event in events:
            if event["type"] == "node":
                table = self._node_types[event["type_id"]]
                global_ids[table].extend(self._column(event["batch"], "id_b"))
        values = {}
        for table, fields in spec.node_fields.items():
            fetched = self._feature_arrays(
                table, global_ids.get(table, ()), tuple(use.field for use in fields.fields)
            )
            for use in fields.fields:
                field = use.field
                values[(table, field)] = tuple(fetched[field].to_pylist())
        return values

    def sample_graph(
        self,
        sample: SampleSpec,
        spec: Any,
        state: Any,
        roots: Sequence[GraphRoot],
        seed: int,
        epoch: int,
    ) -> dict[str, Any]:
        del seed, epoch
        if self._sampler is None:
            raise RuntimeError("DiskGraph source is closed")
        total_started = time.perf_counter_ns()
        sampling_started = time.perf_counter_ns()
        neighborhood, events, physical_relations = self._sample_graph_events(
            sample,
            spec.source,
            roots,
        )
        sampling_ns = time.perf_counter_ns() - sampling_started
        stages: dict[str, int] = {}
        output = self._graph_output(
            sample.entity.table,
            neighborhood,
            events,
            physical_relations,
            spec.node_fields,
            state,
            stages,
        )
        self._last_graph_profile = {
            "samples": len(roots),
            "sampling_ns": sampling_ns,
            **stages,
            "total_ns": time.perf_counter_ns() - total_started,
        }
        return output

    def _sample_graph_events(
        self,
        sample: SampleSpec,
        source: str,
        roots: Sequence[GraphRoot],
    ) -> tuple[Any, list[Mapping[str, Any]], Mapping[str, str]]:
        neighborhood = sample.neighborhoods.get(source)
        if neighborhood is None:
            raise ValueError(f"unknown neighborhood {source!r}")
        root_ids = self._root_ids(sample.entity.table, roots)

        plan, physical_relations = self._sampling_plan(neighborhood)
        events = list(
            self._sampler.sample(
                {sample.entity.table: root_ids},
                json.dumps(plan),
                fused=False,
                seed_timestamps={sample.entity.table: [root.cutoff for root in roots]},
                swap_edge_direction=False,
                seed_timestamp_unit=self._diskgraph.TimeUnit.Seconds,
            )
        )
        return neighborhood, events, physical_relations

    def close(self) -> None:
        self._sampler = None

    @staticmethod
    def _validate_sample(spec: SampleSpec) -> None:
        if issues := _sample_refusals(spec):
            raise DISKGRAPH_UNSUPPORTED[issues[0].code].error(issues[0].message)

    def _empty_sample(self, spec: SampleSpec, request: SampleRequest) -> BuiltSample:
        return BuiltSample(
            request.sample_id,
            EntityHistory(
                entity_id=request.entity_id,
                evaluation_time=cutoff_to_epoch(request.as_of_time),
                entity=EntityState((), ()),
                events=(),
                entity_type=spec.entity.entity_type,
            ),
            request.targets,
            metadata={"split": request.split, **request.metadata},
        )

    def _validate_root_availability(
        self, spec: SampleSpec, requests: Sequence[SampleRequest]
    ) -> None:
        if not requests or spec.entity.available_from is None:
            return
        table = spec.entity.table
        column = spec.entity.available_from
        node_ids = self._root_ids(table, requests)
        times = self._feature_values(table, node_ids, column)
        for request, value in zip(requests, times, strict=True):
            if value is None:
                raise ValueError(f"{table!r} row has no timestamp in {column!r}")
            if timestamp_to_epoch(value) >= cutoff_to_epoch(request.as_of_time):
                raise ValueError("entity did not exist before the requested sample time")

    def _root_ids(
        self,
        table: str,
        requests: Sequence[SampleRequest | GraphRoot],
    ) -> list[int]:
        definition = next(item for item in self.schema.tables if item.name == table)
        dtype = definition.columns[definition.primary_key]
        root_ids = self._sampler.lookup_node_ids(
            table,
            self._pa.array(
                [
                    dtype.normalize(request.entity_id, path="requests.entity_id")
                    for request in requests
                ],
                type=dtype.to_arrow(),
            ),
        )
        missing = [
            request.entity_id
            for request, node_id in zip(requests, root_ids, strict=True)
            if node_id is None
        ]
        if missing:
            raise ValueError(f"DiskGraph index has no {table!r} primary key(s) {missing!r}")
        return [int(node_id) for node_id in root_ids if node_id is not None]

    def _adapt_temporal_result(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
        cutoffs: Sequence[int],
        compiled: _CompiledTemporalPlan,
        result: Mapping[str, Any],
        direct_values: Mapping[str, Any],
    ) -> list[BuiltSample]:
        sequences = {item["name"]: item for item in result["sequences"]}
        aggregates = {item["name"]: item["values"] for item in result["aggregates"]}
        events: list[list[tuple[tuple[Any, ...], Event]]] = [[] for _ in requests]
        row_targets: list[dict[str, Event]] = [{} for _ in requests]

        for native_name, output in compiled.sequences.items():
            sequence = sequences[native_name]
            batch = sequence["batch"]
            offsets = sequence["offsets"]
            timestamps = sequence["timestamps"]
            columns = {
                name: batch.column(index).to_pylist()
                for index, name in enumerate(batch.schema.names)
            }
            primary_keys = self._result_column(columns, output.primary_key)
            for anchor, cutoff in enumerate(cutoffs):
                start, stop = offsets[anchor : anchor + 2]
                if output.kind == "row_target":
                    if stop - start > 1:
                        raise RuntimeError(
                            f"DiskGraph returned multiple rows for target {output.logical_name!r}"
                        )
                    if start != stop:
                        timestamp = timestamps[start]
                        if timestamp is None:
                            raise RuntimeError("row target has no timestamp")
                        row_targets[anchor][output.logical_name] = Event(
                            output.logical_name,
                            int(timestamp),
                            self._project_fields(
                                output.spec.fields,
                                columns,
                                start,
                                cutoff,
                                int(timestamp),
                                destination_table(
                                    self.schema, spec.entity.table, output.spec.rows.path
                                ),
                            ),
                        )
                    continue

                for row in range(start, stop):
                    timestamp = timestamps[row]
                    if timestamp is None:
                        raise RuntimeError("event sequence row has no timestamp")
                    event = Event(
                        output.logical_name,
                        int(timestamp),
                        self._project_fields(
                            output.spec.fields,
                            columns,
                            row,
                            cutoff,
                            int(timestamp),
                            destination_table(
                                self.schema, spec.entity.table, output.spec.rows.path
                            ),
                        ),
                    )
                    events[anchor].append(
                        (
                            (
                                int(timestamp),
                                output.logical_name,
                                _sort_value(primary_keys[row]),
                            ),
                            event,
                        )
                    )

        typed_entity_values = self._entity_arrays(spec, direct_values, aggregates)
        built = []
        for anchor, request in enumerate(requests):
            entity_values = self._entity_records(spec, typed_entity_values, anchor)

            targets = {target.name: target for target in request.targets}
            for name, target in spec.targets.items():
                if isinstance(target, ExternalTarget):
                    continue
                if isinstance(target, Aggregate):
                    value = self._typed_array(aggregates[f"target:{name}"], target.dtype)[
                        anchor
                    ].as_py()
                    if value is not None:
                        targets[name] = TargetValue(
                            name, target.dtype.normalize(value, path=f"targets.{name}")
                        )
                elif isinstance(target, Compare):
                    values = self._comparison_array(target, aggregates[f"target:{name}"])
                    if values[anchor].is_valid:
                        targets[name] = TargetValue(name, values[anchor].as_py())
                elif name in row_targets[anchor]:
                    targets[name] = TargetValue(name, row_targets[anchor][name])

            ordered_events = tuple(event for _key, event in sorted(events[anchor]))
            built.append(
                BuiltSample(
                    request.sample_id,
                    EntityHistory(
                        entity_id=request.entity_id,
                        evaluation_time=cutoffs[anchor],
                        entity=EntityState(tuple(entity_values), ()),
                        events=ordered_events,
                        entity_type=spec.entity.entity_type,
                    ),
                    tuple(targets.values()),
                    metadata={"split": request.split, **request.metadata},
                )
            )
        return built

    def _adapt_model_result(
        self,
        spec: SampleSpec,
        requests: Sequence[SampleRequest],
        cutoffs: Sequence[int],
        compiled: _CompiledTemporalPlan,
        result: Mapping[str, Any],
        direct_values: Mapping[str, Any],
    ) -> ModelSourceBatch:
        sequences = {item["name"]: item for item in result["sequences"]}
        aggregates = {item["name"]: item["values"] for item in result["aggregates"]}
        events: dict[str, ModelSequence] = {}
        ordered_events: dict[tuple[str, str], ModelSequence] = {}
        row_targets: dict[str, ModelRowTarget] = {}

        for native_name, output in compiled.sequences.items():
            sequence = sequences[native_name]
            batch = sequence["batch"]
            offsets = sequence["offsets"]
            timestamps = sequence["timestamps"]
            table = destination_table(self.schema, spec.entity.table, output.spec.rows.path)
            fields = {}
            for name in _value_evaluation_order(output.spec.fields):
                field = output.spec.fields[name]
                fields[name] = (
                    self._derived_array(field, fields, path=f"events.{output.logical_name}.{name}")
                    if isinstance(field, Derived)
                    else self._model_field(field, batch, offsets, timestamps, cutoffs, table)
                )
            if output.kind == "event":
                sequence = ModelSequence(offsets, timestamps, fields, output.order)
                events.setdefault(output.logical_name, sequence)
                ordered_events[output.logical_name, output.order] = sequence
            else:
                row_targets[output.logical_name] = ModelRowTarget(
                    self._align_optional_rows(timestamps, offsets, len(requests)),
                    {
                        name: self._align_optional_rows(values, offsets, len(requests))
                        for name, values in fields.items()
                    },
                )

        entity_values = self._entity_arrays(spec, direct_values, aggregates)
        targets: dict[str, Any] = {}
        for anchor, request in enumerate(requests):
            for target in request.targets:
                if target.name in spec.targets and not isinstance(
                    spec.targets[target.name], ExternalTarget
                ):
                    raise ValueError(f"target {target.name!r} was both supplied and generated")
                targets.setdefault(target.name, [None] * len(requests))[anchor] = target.value
        targets.update(
            {
                name: self._typed_array(aggregates[f"target:{name}"], target.dtype)
                for name, target in spec.targets.items()
                if isinstance(target, Aggregate)
            }
        )
        targets.update(
            {
                name: self._comparison_array(target, aggregates[f"target:{name}"])
                for name, target in spec.targets.items()
                if isinstance(target, Compare)
            }
        )
        return ModelSourceBatch(
            sample_ids=tuple(request.sample_id for request in requests),
            entity_ids=tuple(request.entity_id for request in requests),
            evaluation_times=tuple(cutoffs),
            entity_type=spec.entity.entity_type,
            entity_values=entity_values,
            events=events,
            ordered_events=ordered_events,
            targets=targets,
            row_targets=row_targets,
            splits=tuple(request.split for request in requests),
            request_metadata=tuple(
                {"split": request.split, **request.metadata} for request in requests
            ),
        )

    def _model_field(
        self,
        field: Field | Constant | EventTimeDelta,
        batch: Any,
        offsets: Any,
        timestamps: Any,
        cutoffs: Sequence[int],
        table: str,
    ) -> Any:
        if isinstance(field, EventTimeDelta):
            boundaries = offsets.to_numpy(zero_copy_only=False)
            owners = self._pa.array(
                [
                    anchor
                    for anchor in range(len(boundaries) - 1)
                    for _ in range(int(boundaries[anchor + 1] - boundaries[anchor]))
                ],
                type=self._pa.int64(),
            )
            event_cutoffs = self._pc.take(self._pa.array(cutoffs), owners)
            elapsed = self._pc.subtract(event_cutoffs, timestamps)
            return self._pc.divide(
                self._pc.cast(elapsed, self._pa.float64()),
                self._pa.scalar(field.seconds_per_unit, type=self._pa.float64()),
            )
        if isinstance(field, Constant):
            return self._pa.repeat(
                self._pa.scalar(field.value, type=field.dtype.to_arrow()), len(timestamps)
            )
        index = batch.schema.get_field_index(field.column)
        if index < 0:
            raise RuntimeError(f"DiskGraph result has no projected column {field.column!r}")
        values = self._typed_array(batch.column(index), value_dtype(self.schema, table, field))
        return self._occurrence_array(values) if field.selection == "all" else values

    def _occurrence_array(self, values: Any) -> Any:
        """A direct repeated selection has one occurrence per present scalar."""
        if isinstance(values, self._pa.ChunkedArray):
            values = values.combine_chunks()
        lengths = self._pc.cast(self._pc.is_valid(values), self._pa.int64())
        boundaries = self._pa.concat_arrays(
            [
                self._pa.array([0], type=self._pa.int64()),
                self._pc.cumulative_sum(lengths),
            ]
        )
        return self._pa.LargeListArray.from_arrays(boundaries, self._pc.drop_null(values))

    def _entity_records(
        self, spec: SampleSpec, arrays: Mapping[str, Any], row: int
    ) -> tuple[FieldValue, ...]:
        types = sample_value_types(self.schema, spec)
        result = []
        for name, column in arrays.items():
            value = column[row].as_py()
            repeated = self._pa.types.is_list(column.type) or self._pa.types.is_large_list(
                column.type
            )
            for item in value if repeated else (value,):
                if item is not None:
                    result.append(FieldValue(name, item, types[name]))
        return tuple(result)

    def _typed_array(self, values: Any, dtype: DType) -> Any:
        if isinstance(values, (list, tuple)):
            integer_values = all(value is None or type(value) is int for value in values)
            values = self._pa.array(
                values, type=dtype.to_arrow() if dtype.integer and integer_values else None
            )
        if dtype.name == "date32" and self._pa.types.is_integer(values.type):
            # Native scalar reductions expose dates as epoch seconds.
            days = self._pc.divide_checked(values, 86_400)
            return self._pc.cast(self._pc.cast(days, self._pa.int32(), safe=True), dtype.to_arrow())
        return self._pc.cast(values, dtype.to_arrow(), safe=True)

    def _entity_arrays(
        self, spec: SampleSpec, direct: Mapping[str, Any], aggregates: Mapping[str, Any]
    ) -> dict[str, Any]:
        result = dict(direct)
        for name in _value_evaluation_order(spec.entity_values):
            expression = spec.entity_values[name]
            if isinstance(expression, Aggregate):
                result[name] = self._typed_array(aggregates[f"entity:{name}"], expression.dtype)
            elif isinstance(expression, Compare):
                result[name] = self._comparison_array(expression, aggregates[f"entity:{name}"])
            elif isinstance(expression, Derived):
                result[name] = self._derived_array(expression, result, path=f"entity_values.{name}")
        return result

    def _comparison_array(self, spec: Compare, values: Any) -> Any:
        values = self._typed_array(values, spec.left.dtype)
        operation = {
            "greater_than": "greater",
            "less_than": "less",
            "greater_or_equal": "greater_equal",
            "less_or_equal": "less_equal",
        }.get(spec.operation, spec.operation)
        return getattr(self._pc, operation)(
            values, self._pa.scalar(spec.right, type=spec.left.dtype.to_arrow())
        )

    def _derived_array(self, spec: Derived, values: Mapping[str, Any], *, path: str) -> Any:
        try:
            return self._calculate_array(spec, values, path=path)
        except (self._pa.ArrowException, ValueError, OverflowError) as error:
            raise ValueError(f"{path}: {error}") from error

    def _calculate_array(self, spec: Derived, values: Mapping[str, Any], *, path: str) -> Any:
        operands = [values[name] for name in spec.inputs]
        operation = spec.operation
        if (
            spec.dtype.name.startswith("decimal")
            and isinstance(operation, str)
            and operation in {"add", "subtract", "multiply", "abs"}
        ):
            # Arrow's static precision expansion can exceed decimal256 even when
            # actual results fit. Use the shared exact scalar kernel for decimals.
            types = {name: DType.from_arrow(values[name].type) for name in spec.inputs}
            rows = zip(*(value.to_pylist() for value in operands), strict=True)
            return self._pa.array(
                [
                    evaluate_derived(
                        spec, dict(zip(spec.inputs, row, strict=True)), path=path, input_types=types
                    )
                    for row in rows
                ],
                type=spec.dtype.to_arrow(),
            )
        if isinstance(operation, CalendarPart):
            function = {"weekday": "day_of_week"}.get(operation.part, operation.part)
            result = getattr(self._pc, function)(operands[0])
        elif isinstance(operation, MapValues):
            fallback = operation.fallback if operation.unknown == "fallback" else None
            result = self._pa.repeat(
                self._pa.scalar(fallback, type=spec.dtype.to_arrow()), len(operands[0])
            )
            known = self._pc.is_null(operands[0])
            operand_dtype = DType.from_arrow(operands[0].type)
            for output, members in _typed_map_groups(operation, operand_dtype):
                selected = self._pa.repeat(self._pa.scalar(False), len(operands[0]))
                for member in members:
                    selected = self._pc.or_kleene(
                        selected,
                        self._pc.equal(operands[0], self._pa.scalar(member, type=operands[0].type)),
                    )
                selected = self._pc.fill_null(selected, False)
                known = self._pc.or_(known, selected)
                result = self._pc.if_else(
                    selected, self._pa.scalar(output, type=spec.dtype.to_arrow()), result
                )
            if operation.unknown == "error" and not self._pc.all(known).as_py():
                raise ValueError(f"{path}: unmapped value in MapValues")
            result = self._pc.if_else(self._pc.is_valid(operands[0]), result, None)
        elif operation in {"add", "subtract", "multiply"}:
            # Float arithmetic follows Arrow's numeric promotion, including
            # integer-to-float rounding; integer kernels retain checked widths.
            operands = [
                self._pc.cast(
                    arrow_float64(value)
                    if spec.dtype.name.startswith("float") and self._pa.types.is_decimal(value.type)
                    else value,
                    spec.dtype.to_arrow(),
                    safe=spec.dtype.integer,
                )
                for value in operands
            ]
            result = getattr(self._pc, operation + "_checked")(*operands)
        elif operation == "abs":
            result = self._pc.abs_checked(self._typed_array(operands[0], spec.dtype))
        elif operation == "sign":
            result = self._pc.sign(operands[0])
        elif operation == "divide":
            left, right = [arrow_float64(value) for value in operands]
            right = self._pc.if_else(self._pc.equal(right, 0), None, right)
            result = self._pc.divide(left, right)
        else:
            operand = arrow_float64(operands[0])
            operand = self._pc.if_else(self._pc.less_equal(operand, -1), None, operand)
            result = self._pc.log1p(operand)
        try:
            result = self._typed_array(result, spec.dtype)
            return result
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{path}: result cannot be represented as {spec.dtype}") from error

    def _align_optional_rows(self, values: Any, offsets: Any, size: int) -> Any:
        boundaries = offsets.to_numpy(zero_copy_only=False)
        indices = []
        for anchor in range(size):
            start = int(boundaries[anchor])
            stop = int(boundaries[anchor + 1])
            if stop - start > 1:
                raise RuntimeError("DiskGraph returned multiple rows for a scalar row target")
            indices.append(start if start != stop else None)
        return self._pc.take(values, self._pa.array(indices, type=self._pa.int64()))

    def _project_fields(
        self,
        fields: Mapping[str, Any],
        columns: Mapping[str, Sequence[Any]],
        row: int,
        cutoff: int,
        event_time: int,
        table: str,
    ) -> tuple[FieldValue, ...]:
        output = []
        available = {}
        for name in _value_evaluation_order(fields):
            field = fields[name]
            if isinstance(field, EventTimeDelta):
                output.append(
                    FieldValue(
                        name,
                        float(cutoff - event_time) / field.seconds_per_unit,
                        field.dtype,
                    )
                )
            elif isinstance(field, Constant):
                output.append(FieldValue(name, field.value, field.dtype))
            elif isinstance(field, Derived):
                result = evaluate_derived(
                    field,
                    available,
                    path=f"event.{name}",
                    input_types={value.key: value.dtype for value in output},
                )
                if result is not None:
                    output.append(FieldValue(name, result, field.dtype))
            else:
                values = self._result_column(columns, field.column)
                if values[row] is not None:
                    dtype = value_dtype(self.schema, table, field)
                    output.append(
                        FieldValue(name, dtype.normalize(values[row], path=f"event.{name}"), dtype)
                    )
            if output and output[-1].key == name:
                available[name] = output[-1].value
        return tuple(output)

    @staticmethod
    def _result_column(columns: Mapping[str, Sequence[Any]], name: str | None) -> Sequence[Any]:
        if name is None:
            raise RuntimeError("a projected field has no source column")
        if name not in columns:
            raise RuntimeError(f"DiskGraph result has no projected column {name!r}")
        return columns[name]

    def _sampling_plan(self, neighborhood: Any) -> tuple[dict[str, Any], dict[str, str]]:
        layers = []
        physical_relations = {}
        for layer in neighborhood.layers:
            fanouts = {}
            for traversal in layer.samples:
                if traversal.strategy != "recent":
                    raise ValueError("DiskGraph does not support uniform graph sampling")
                relation = self._relations[traversal.relation]
                target_table = (
                    relation.right_table
                    if traversal.direction == "forward"
                    else relation.left_table
                )
                clock = table_clock(self._tables[target_table])
                if traversal.time_column is not None and traversal.time_column != clock:
                    raise ValueError(
                        "the DiskGraph backend cannot override a destination "
                        f"table clock; {target_table!r} is indexed by {clock!r}, but the "
                        f"traversal requests {traversal.time_column!r}"
                    )
                if traversal.temporal == "static" and clock is not None:
                    raise ValueError(
                        "the DiskGraph backend cannot disable temporal filtering "
                        f"for timestamped destination table {target_table!r}"
                    )
                mapping = self._relation_mapping[traversal.relation]
                physical = mapping[traversal.direction]
                if physical in fanouts:
                    raise DISKGRAPH_UNSUPPORTED["DG-DUPLICATE-LAYER-RELATION"].error(
                        f"physical relation {physical!r} is sampled more than once in one layer"
                    )
                if (
                    physical in physical_relations
                    and physical_relations[physical] != traversal.relation
                ):
                    raise ValueError(
                        f"logical relations {physical_relations[physical]!r} and "
                        f"{traversal.relation!r} resolve to the same physical relation {physical!r}"
                    )
                physical_relations[physical] = traversal.relation
                fanouts[physical] = {
                    "fanout": traversal.fanout,
                    "mode": "temporal_most_recent",
                    "absent_key_policy": "reject_plan",
                }
            layers.append({"fanouts": fanouts, "default_fanout": None})
        return {"layers": layers}, physical_relations

    def _graph_output(
        self,
        root_table: str,
        neighborhood: Any,
        events: Sequence[Mapping[str, Any]],
        physical_relations: Mapping[str, str],
        node_fields: Mapping[str, Mapping[str, Any]],
        encoders: Mapping[str, Mapping[str, Any]],
        profile: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        decode_started = time.perf_counter_ns()
        global_ids: dict[str, list[np.ndarray]] = defaultdict(list)
        owners: dict[str, list[np.ndarray]] = defaultdict(list)
        physical_edges: dict[str, list[np.ndarray]] = defaultdict(list)

        def indices(batch: Any, name: str) -> np.ndarray:
            values = batch.column(batch.schema.get_field_index(name))
            if values.null_count:
                raise RuntimeError(f"DiskGraph graph column {name!r} contains NULL indices")
            return values.cast(self._pa.int64()).to_numpy(zero_copy_only=False)

        for event in events:
            if event["type"] == "node":
                node_type = self._node_types[event["type_id"]]
                batch = event["batch"]
                owners[node_type].append(indices(batch, "id_a"))
                global_ids[node_type].append(batch.column("id_b").to_numpy(zero_copy_only=False))
            elif event["type"] == "edge":
                relation = self._relations_by_id[event["rel_id"]]["name"]
                if relation in physical_relations:
                    batch = event["batch"]
                    physical_edges[relation].append(
                        np.stack((indices(batch, "id_a"), indices(batch, "id_b")))
                    )
        if profile is not None:
            profile["event_decode_ns"] = time.perf_counter_ns() - decode_started

        relation_names = {
            traversal.relation for layer in neighborhood.layers for traversal in layer.samples
        }
        tables = {root_table, *node_fields}
        for name in relation_names:
            relation = self._relations[name]
            tables.update((relation.left_table, relation.right_table))

        nodes = {}
        identity_fetch_ns = 0
        feature_fetch_ns = 0
        feature_encode_ns = 0
        for table in sorted(tables):
            # Native feature lookup still accepts a list of global node IDs.
            ids = np.concatenate(global_ids[table]).tolist() if global_ids[table] else []
            primary_key = self._tables[table].primary_key
            started = time.perf_counter_ns()
            feature_arrays = self._feature_arrays(
                table,
                ids,
                tuple(
                    dict.fromkeys(
                        (
                            primary_key,
                            *(
                                (use.field for use in node_fields[table].fields)
                                if table in node_fields
                                else ()
                            ),
                        )
                    )
                ),
            )
            feature_fetch_ns += time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            node_ids = _graph_node_ids(feature_arrays[primary_key])
            identity_fetch_ns += time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            nodes[table] = {
                "ids": node_ids,
                "batch": torch.from_numpy(
                    np.concatenate(owners[table]) if owners[table] else np.empty(0, dtype=np.int64)
                ),
                "features": {
                    field: encoder.encode_arrow(feature_arrays[field])
                    for field, encoder in encoders.get(table, {}).items()
                },
            }
            feature_encode_ns += time.perf_counter_ns() - started
        if profile is not None:
            profile.update(
                {
                    "identity_fetch_ns": identity_fetch_ns,
                    "feature_fetch_ns": feature_fetch_ns,
                    "feature_encode_ns": feature_encode_ns,
                }
            )

        topology_started = time.perf_counter_ns()
        logical_edges: dict[str, list[np.ndarray]] = defaultdict(list)
        for physical, logical in physical_relations.items():
            # Table names cannot distinguish the two ends of a self-relation.
            # The ingest mapping records direction relative to the declaration.
            aligned = physical == self._relation_mapping[logical]["forward"]
            pairs = physical_edges.get(physical, [])
            logical_edges[logical].extend(pairs if aligned else (part[::-1] for part in pairs))

        edges = {}
        for name in sorted(relation_names):
            relation = self._relations[name]
            pairs = logical_edges.get(name, [])
            edges[name] = {
                "index": (
                    torch.from_numpy(_first_occurrences(np.concatenate(pairs, axis=1)))
                    if pairs
                    else torch.empty((2, 0), dtype=torch.long)
                ),
                "source_type": relation.left_table,
                "destination_type": relation.right_table,
            }
        if profile is not None:
            profile["topology_assembly_ns"] = time.perf_counter_ns() - topology_started
        return {"nodes": nodes, "edges": edges}

    def _feature_values(self, table: str, node_ids: Sequence[int], column: str) -> list[Any]:
        if not node_ids:
            return []
        return self._feature_arrays(table, node_ids, (column,))[column].to_pylist()

    def _feature_arrays(
        self,
        table: str,
        node_ids: Sequence[int],
        names: Sequence[str],
    ) -> dict[str, Any]:
        if not names:
            return {}
        if not node_ids:
            columns = self._tables[table].columns or {}
            return {
                name: self._pa.nulls(0)
                if columns.get(name) is None
                else self._pa.array([], type=columns[name].to_arrow())
                for name in names
            }
        chunks: dict[str, list[Any]] = {name: [] for name in names}
        for event in self._sampler.fetch_features(
            [{"node_type": table, "node_ids": list(node_ids), "feature_names": list(names)}]
        ):
            if event["type"] == "feature":
                batch = event["batch"]
                for name in names:
                    index = batch.schema.get_field_index(name)
                    if index < 0:
                        raise RuntimeError(f"DiskGraph returned no {name!r} feature for {table!r}")
                    chunks[name].append(batch.column(index))
        result = {
            name: (
                parts[0]
                if len(parts) == 1
                else self._pa.concat_arrays(parts)
                if parts
                else self._pa.nulls(0)
            )
            for name, parts in chunks.items()
        }
        for name, values in result.items():
            if len(values) != len(node_ids):
                raise RuntimeError(
                    f"DiskGraph returned {len(values)} {name!r} values for "
                    f"{len(node_ids)} {table!r} nodes"
                )
        return result

    @staticmethod
    def _column(batch: Any, name: str) -> list[int]:
        index = batch.schema.get_field_index(name)
        return [int(value) for value in batch.column(index).to_pylist()]


class _SequenceOutput:
    def __init__(
        self,
        kind: str,
        logical_name: str,
        spec: EventSpec | RowTarget,
        primary_key: str,
        order: str = "ascending",
    ) -> None:
        self.kind = kind
        self.logical_name = logical_name
        self.spec = spec
        self.primary_key = primary_key
        self.order = order


class _CompiledTemporalPlan:
    def __init__(
        self,
        plan: Mapping[str, Any],
        sequences: Mapping[str, _SequenceOutput],
    ) -> None:
        self.plan = plan
        self.sequences = sequences


class _TemporalPlanCompiler:
    """Translate kumo-pytorch's row contract into DiskGraph's small physical query plan."""

    def __init__(self, source: DiskGraphSource) -> None:
        self.source = source
        self.root_table = ""
        self.frames: list[dict[str, Any]] = []
        self.frames_by_rows: dict[Rows, dict[str, Any]] = {}
        self.sequences: list[dict[str, Any]] = []
        self.sequence_outputs: dict[str, _SequenceOutput] = {}
        self.aggregates: list[dict[str, Any]] = []

    def compile(
        self, spec: SampleSpec, *, event_orders: Mapping[str, Sequence[str]] | None = None
    ) -> _CompiledTemporalPlan:
        self.root_table = spec.entity.table
        for name, expression in spec.entity_values.items():
            if isinstance(expression, (Aggregate, Compare)):
                self._aggregate(
                    f"entity:{name}",
                    expression.left if isinstance(expression, Compare) else expression,
                )
        for name, event in spec.events.items():
            orders = (None,) if event_orders is None else event_orders.get(name, ("ascending",))
            for index, order in enumerate(orders):
                native_name = (
                    f"event:{name}" if index == 0 else f"event-order:{len(self.sequences)}"
                )
                self._sequence(native_name, "event", name, event, output_order=order)
        for name, target in spec.targets.items():
            if isinstance(target, (Aggregate, Compare)):
                self._aggregate(
                    f"target:{name}", target.left if isinstance(target, Compare) else target
                )
            elif isinstance(target, RowTarget):
                self._sequence(f"target:{name}", "row_target", name, target)
        frames = [
            {
                key: (sorted(value) if key == "columns" else value)
                for key, value in frame.items()
                if key != "destination_type"
            }
            for frame in self.frames
        ]
        return _CompiledTemporalPlan(
            {
                "frames": frames,
                "sequences": self.sequences,
                "aggregates": self.aggregates,
            },
            self.sequence_outputs,
        )

    def _sequence(
        self,
        native_name: str,
        kind: str,
        logical_name: str,
        spec: EventSpec | RowTarget,
        *,
        output_order: str | None = None,
    ) -> None:
        frame, destination = self._frame(spec.rows)
        primary_key = self.source._tables[destination].primary_key
        columns = [primary_key]
        for field in spec.fields.values():
            if isinstance(field, Field) and field.column is not None:
                self._physical_column(frame, destination, field.column)
                if field.column not in columns:
                    columns.append(field.column)
        if isinstance(spec, EventSpec):
            order = spec.order
            limit = spec.limit
        else:
            order = "ascending" if spec.selection == "first" else "descending"
            limit = 1
        self.sequences.append(
            {
                "name": native_name,
                "frame": frame["name"],
                "columns": columns,
                "order": order,
                "limit": limit,
            }
        )
        if output_order is not None:
            self.sequences[-1]["output_order"] = output_order
        self.sequence_outputs[native_name] = _SequenceOutput(
            kind,
            logical_name,
            spec,
            primary_key,
            output_order or "ascending",
        )

    def _aggregate(self, native_name: str, spec: Aggregate) -> None:
        frame, groups, available, destination = self._source(spec.source)
        if spec.column is not None:
            if spec.column not in available:
                raise ValueError(
                    f"aggregate {native_name!r} references unavailable column {spec.column!r}"
                )
            if not groups:
                self._physical_column(frame, destination, spec.column)
        self.aggregates.append(
            {
                "name": native_name,
                "frame": frame["name"],
                "groups": groups,
                "operation": spec.operation,
                "column": spec.column,
                "sum_type": spec.dtype.name
                if spec.operation == "sum" and spec.dtype.integer
                else None,
            }
        )

    def _source(
        self,
        source: Rows | GroupBy,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], set[str], str]:
        if isinstance(source, Rows):
            frame, destination = self._frame(source)
            return frame, [], set(self.source._tables[destination].columns), destination

        frame, groups, available, destination = self._source(source.source)
        missing_keys = set(source.keys) - available
        if missing_keys:
            raise ValueError(f"group-by references unavailable keys {sorted(missing_keys)}")
        values = []
        for name, reduction in source.values.items():
            if reduction.column is not None and reduction.column not in available:
                raise ValueError(
                    f"group-by reduction {name!r} references unavailable column "
                    f"{reduction.column!r}"
                )
            values.append(
                {
                    "name": name,
                    "operation": reduction.operation,
                    "column": reduction.column,
                    "sum_type": reduction.dtype.name
                    if reduction.operation == "sum" and reduction.dtype.integer
                    else None,
                }
            )
        if not groups:
            for column in source.keys:
                self._physical_column(frame, destination, column)
            for reduction in source.values.values():
                if reduction.column is not None:
                    self._physical_column(frame, destination, reduction.column)
        groups.append({"keys": list(source.keys), "values": values})
        return frame, groups, set(source.keys) | set(source.values), destination

    def _frame(self, rows: Rows) -> tuple[dict[str, Any], str]:
        existing = self.frames_by_rows.get(rows)
        if existing is not None:
            return existing, str(existing["destination_type"])

        current = self.root_table
        path = []
        for relation_name in rows.path:
            relation = self.source._relations[relation_name]
            if current == relation.left_table:
                direction = "forward"
                destination = relation.right_table
            elif current == relation.right_table:
                direction = "reverse"
                destination = relation.left_table
            else:
                raise ValueError(f"relation {relation_name!r} cannot be followed from {current!r}")
            path.append(
                {
                    "relation": self.source._relation_mapping[relation_name][direction],
                    "destination_type": destination,
                }
            )
            current = destination

        clock = (
            rows.time.column if rows.time is not None else table_clock(self.source._tables[current])
        )
        if clock is not None and clock not in self.source._tables[current].columns:
            raise ValueError(f"table {current!r} has no temporal column {clock!r}")
        frame = {
            "name": f"frame:{len(self.frames)}",
            "root_type": self.root_table,
            "path": path,
            "time_column": clock,
            "start_seconds": rows.time.start
            if clock is not None and rows.time is not None
            else None,
            "stop_seconds": rows.time.stop
            if clock is not None and rows.time is not None
            else 0
            if clock is not None
            else None,
            "distinct": rows.distinct,
            "columns": set(),
            # Kept only while compiling; stripped below.
            "destination_type": current,
        }
        self.frames.append(frame)
        self.frames_by_rows[rows] = frame
        return frame, current

    def _physical_column(
        self,
        frame: dict[str, Any],
        destination: str,
        column: str,
    ) -> None:
        if column not in self.source._tables[destination].columns:
            raise ValueError(f"table {destination!r} has no column {column!r}")
        frame["columns"].add(column)


def _sort_value(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _graph_node_ids(values):
    import pyarrow as pa
    import pyarrow.compute as pc

    if not len(values):
        return torch.empty(0, dtype=torch.long)
    if not values.null_count and (
        pa.types.is_integer(values.type) or pa.types.is_boolean(values.type)
    ):
        if pa.types.is_uint64(values.type) and (largest := pc.max(values).as_py()) >= 2**63:
            raise ValueError(f"Overflow: uint64 node ID {largest} does not fit in int64")
        return torch.from_numpy(
            values.cast(pa.int64()).to_numpy(zero_copy_only=False, writable=True)
        )
    raw = values.to_pylist()
    return (
        torch.tensor(raw, dtype=torch.long)
        if all(isinstance(value, int) for value in raw)
        else tuple(raw)
    )


def _first_occurrences(pairs: np.ndarray) -> np.ndarray:
    """Drop repeated ``(source, destination)`` columns, keeping each first one in order.

    Node indices are disjoint per sample, so this deduplicates per sample.
    """

    if pairs.shape[1] < 2:
        return pairs
    _, first = np.unique(pairs, axis=1, return_index=True)
    if first.size == pairs.shape[1]:
        return pairs
    return np.ascontiguousarray(pairs[:, np.sort(first)])


register_engine(DiskGraphEngine.name, DiskGraphEngine)
