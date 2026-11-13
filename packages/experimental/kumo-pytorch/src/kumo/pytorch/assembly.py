# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit typed encoders once and assemble engine-independent model batches."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from dataclasses import field as dataclass_field
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch

from kumo.pytorch.anchors import shard_positions
from kumo.pytorch.config import MaskingConfig
from kumo.pytorch.contract.batch import (
    AlignedEventTarget,
    Batch,
    BatchSpec,
    CandidateSource,
    CandidateTarget,
    Columns,
    Composite,
    Encode,
    EntitySource,
    EventsOf,
    EventSource,
    Fields,
    Flatten,
    Graph,
    Input,
    LabelTarget,
    MaskedValuesTarget,
    MatchingGroupTarget,
    NeighborhoodSource,
    Nested,
    NextTokenTarget,
    OccurrencesOf,
    Packed,
    Padded,
    Raw,
    StructuredTarget,
    Tokens,
    UseEncoding,
)
from kumo.pytorch.contract.batch.encodings import Buckets, Text, Vocabulary, ZScore
from kumo.pytorch.contract.batch.tokens import ROLE_ORDER
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    BuiltSample,
    ExternalTarget,
    SampleRequest,
    SampleSpec,
    _normalize_supplied_targets,
)
from kumo.pytorch.data.masking import mask_event_values
from kumo.pytorch.data.model import ModelSourceBatch
from kumo.pytorch.data.records import EntityHistory, Event, cutoff_to_epoch
from kumo.pytorch.encoding import (
    FittedEncoder,
    FittedSpace,
    OutputBinding,
    fit_encoder,
    json_value,
    learns_from_values,
    typed_key,
    unjson_value,
)
from kumo.pytorch.engine import GraphRoot
from kumo.pytorch.output import (
    TokenRows,
    flatten_rows,
    kernel_metadata,
    layout_array,
    offsets,
    sequence_layout,
)
from kumo.pytorch.spec import sample_without_targets

HistoryInput = EntityHistory | BuiltSample
_FIT_CHUNK = 4096


def _chunks(values: Any) -> Any:
    return (values[start : start + _FIT_CHUNK] for start in range(0, len(values), _FIT_CHUNK))


class _Mapped:
    """``self[i:j]`` is ``function(items[i:j])``, computed when sliced."""

    def __init__(self, function: Any, items: Any) -> None:
        self._function, self._items = function, items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: slice) -> Any:
        return self._function(self._items[index])


def _history(value: HistoryInput) -> EntityHistory:
    return value.history if isinstance(value, BuiltSample) else value


def _sample_id(value: HistoryInput) -> str | int:
    return value.sample_id if isinstance(value, BuiltSample) else value.entity_id


def _targets(value: HistoryInput | SampleRequest) -> dict[str, Any]:
    return (
        {item.name: item.value for item in value.targets}
        if isinstance(value, (BuiltSample, SampleRequest))
        else {}
    )


def _jsonable(value: Any) -> Any:
    # Most request leaves are built-in scalars; avoid structural checks for each one.
    # Floats still go through json_value, which keeps them lossless.
    if type(value) in (str, int, bool) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "values": value.tolist(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_pylist"):
        return {"dtype": str(value.type), "values": json_value(value.to_pylist())}
    return json_value(value)


#: The manifest entry holding every artifact file's digest.
ARTIFACT_DIGESTS = "file_digests"


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def artifact_digests(texts: Mapping[str, str]) -> dict[str, str]:
    """Each artifact file's SHA-256; the manifest's covers it without this entry."""

    return {
        name: hashlib.sha256(
            (_json_text(_without_digests(text)) if name == "manifest.json" else text).encode()
        ).hexdigest()
        for name, text in texts.items()
    }


def _without_digests(manifest_text: str) -> Any:
    manifest = json.loads(manifest_text)
    if isinstance(manifest, dict):
        manifest.pop(ARTIFACT_DIGESTS, None)
    return manifest


def _fingerprint(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _identity(*parts: str) -> str:
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class _Rows:
    boundaries: np.ndarray
    columns: Mapping[str, Any]
    sources: np.ndarray
    occurrence_times: Mapping[str, Any] = dataclass_field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.boundaries[-1])

    def select(self, indices: np.ndarray) -> _Rows:
        import pyarrow as pa
        import pyarrow.compute as pc

        owners = np.repeat(np.arange(len(self.boundaries) - 1), np.diff(self.boundaries))
        lengths = np.bincount(owners[indices], minlength=len(self.boundaries) - 1)
        return _Rows(
            boundaries=offsets(lengths),
            columns={
                name: pc.take(value, pa.array(indices)) for name, value in self.columns.items()
            },
            sources=self.sources[indices],
            occurrence_times={
                name: pc.take(value, pa.array(indices))
                for name, value in self.occurrence_times.items()
            },
        )


def _record_rows(
    source: Any, values: Sequence[HistoryInput], declarations: Mapping[str, Any]
) -> _Rows:
    import pyarrow as pa

    selected = []
    selected_times = []
    lengths = []
    sources = []
    for value in values:
        history = _history(value)
        if isinstance(source, EntitySource):
            rows = [
                (
                    "entity",
                    (
                        *history.entity.values,
                        *sorted(history.entity.dated_values, key=lambda item: item.timestamp),
                    ),
                )
            ]
        elif isinstance(source, CandidateSource):
            rows = [
                (source.name, (*item.history.entity.values, *item.history.entity.dated_values))
                for item in value.candidates.get(source.name, ())
            ]
        else:
            events = (
                history.events
                if source.collection == "history"
                else tuple(
                    event for name in source.events for event in value.current_events.get(name, ())
                )
            )
            events = sorted(
                (event for event in events if event.source in source.events),
                key=lambda event: (event.timestamp, source.events.index(event.source)),
            )
            if source.max_events is not None:
                events = (
                    events[-source.max_events :]
                    if source.keep == "last"
                    else events[: source.max_events]
                )
            if source.order == "descending":
                events.reverse()
            rows = [(event.source, event.fields) for event in events]
        lengths.append(len(rows))
        for name, row in rows:
            grouped = {}
            row_times = {}
            for field in row:
                grouped.setdefault(field.key, []).append(field.value)
                row_times.setdefault(field.key, []).append(getattr(field, "timestamp", None))
            selected.append(grouped)
            selected_times.append(row_times)
            sources.append(name)
    columns = {}
    occurrence_times = {}
    for name, resolved in declarations.items():
        dtype = resolved[0][1]
        repeated = any(item[2] == "repeated" for item in resolved)
        collected = []
        collected_times = []
        for row, row_times, component in zip(selected, selected_times, sources, strict=True):
            local = name.split("/")[-1]
            allowed = (
                {item[0][-2] for item in resolved} if isinstance(source, EventSource) else None
            )
            items = row.get(local, ()) if allowed is None or component in allowed else ()
            collected.append(list(items) if repeated else (items[0] if items else None))
            times = row_times.get(local, ()) if allowed is None or component in allowed else ()
            collected_times.append(list(times) if repeated else (times[0] if times else None))
        columns[name] = pa.array(
            collected, type=pa.list_(dtype.to_arrow()) if repeated else dtype.to_arrow()
        )
        occurrence_times[name] = pa.array(
            collected_times, type=pa.list_(pa.int64()) if repeated else pa.int64()
        )
    return _Rows(
        boundaries=offsets(lengths),
        columns=columns,
        sources=np.asarray(sources, dtype=object),
        occurrence_times=occurrence_times,
    )


def _model_rows(source: Any, values: ModelSourceBatch, declarations: Mapping[str, Any]) -> _Rows:
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(source, EntitySource):
        return _Rows(
            boundaries=np.arange(values.batch_size + 1, dtype=np.int64),
            columns={
                name: values.entity_values[resolved[0][0][-1]]
                for name, resolved in declarations.items()
            },
            sources=np.full(values.batch_size, "entity", dtype=object),
        )
    sequences = [
        (name, getattr(values, "ordered_events", {}).get((name, source.order), values.events[name]))
        for name in source.events
    ]
    if len(sequences) == 1:
        name, sequence = sequences[0]
        boundaries = sequence.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        times = sequence.timestamps.to_numpy(zero_copy_only=False)
        falls = (
            np.flatnonzero(
                times[1:] < times[:-1] if source.order == "ascending" else times[1:] > times[:-1]
            )
            + 1
        )
        ordered = (
            sequence.order == source.order
            and not sequence.timestamps.null_count
            and np.isin(falls, boundaries).all()
        )
        if ordered:
            columns = {}
            for field, resolved in declarations.items():
                local = field.split("/")[-1]
                columns[field] = (
                    sequence.fields[local]
                    if local in sequence.fields and name in {item[0][-2] for item in resolved}
                    else pa.nulls(
                        len(times),
                        type=pa.list_(resolved[0][1].to_arrow())
                        if any(item[2] == "repeated" for item in resolved)
                        else resolved[0][1].to_arrow(),
                    )
                )
            counts = np.diff(boundaries)
            if source.max_events is not None and bool((counts > source.max_events).any()):
                kept = np.minimum(counts, source.max_events)
                starts = boundaries[:-1] + (
                    counts - kept if (source.keep == "last") == (source.order == "ascending") else 0
                )
                boundaries = offsets(kept)
                take = (
                    np.arange(int(boundaries[-1]))
                    - np.repeat(boundaries[:-1], kept)
                    + np.repeat(starts, kept)
                )
                columns = {
                    field: pc.take(column, pa.array(take)) for field, column in columns.items()
                }
            return _Rows(
                boundaries=boundaries,
                columns=columns,
                sources=np.full(int(boundaries[-1]), name, dtype=object),
            )
    merge_key = (tuple(source.events), source.max_events, source.keep, source.order)
    premerged = getattr(values, "merged_events", {}).get(merge_key)
    if premerged is not None or getattr(values, "merge_event_streams", None) is not None:
        native_bounds, native_order = (
            premerged
            if premerged is not None
            else values.merge_event_streams(
                [
                    (item.offsets, item.timestamps, item.order == "ascending")
                    for _, item in sequences
                ],
                source.max_events,
                source.keep == "last",
                source.order == "descending",
            )
        )
        bounds = native_bounds.to_numpy(zero_copy_only=False)
        order = native_order.to_numpy(zero_copy_only=False)
        ends = np.cumsum([len(item.timestamps) for _, item in sequences])
        components = np.searchsorted(ends, order, side="right")
        selected_sources = np.asarray([name for name, _ in sequences], dtype=object)[components]
    else:
        lengths = [
            np.diff(item.offsets.to_numpy(zero_copy_only=False).astype(np.int64))
            for _, item in sequences
        ]
        owners = np.concatenate([np.repeat(np.arange(values.batch_size), size) for size in lengths])
        times = np.concatenate(
            [item.timestamps.to_numpy(zero_copy_only=False) for _, item in sequences]
        )
        ranks = np.concatenate(
            [
                np.full(len(item.timestamps), rank, dtype=np.int64)
                for rank, (_, item) in enumerate(sequences)
            ]
        )
        within = np.concatenate(
            [
                np.arange(len(item.timestamps)) * (1 if item.order == "ascending" else -1)
                for _, item in sequences
            ]
        )
        names = np.concatenate(
            [np.full(len(item.timestamps), name, dtype=object) for name, item in sequences]
        )
        order = np.lexsort((within, ranks, times, owners))
        counts = np.bincount(owners[order], minlength=values.batch_size)
        starts = offsets(counts)[:-1]
        kept = counts if source.max_events is None else np.minimum(counts, source.max_events)
        if source.keep == "last":
            starts += counts - kept
        bounds = offsets(kept)
        positions = np.arange(int(bounds[-1])) - np.repeat(bounds[:-1], kept)
        if source.order == "descending":
            positions = np.repeat(kept, kept) - 1 - positions
        order = order[np.repeat(starts, kept) + positions]
        selected_sources = names[order]
    columns = {}
    for name, resolved in declarations.items():
        dtype = resolved[0][1]
        column_type = (
            pa.list_(dtype.to_arrow())
            if any(item[2] == "repeated" for item in resolved)
            else dtype.to_arrow()
        )
        local = name.split("/")[-1]
        allowed = {item[0][-2] for item in resolved}
        parts = [
            item.fields[local]
            if component in allowed and local in item.fields
            else pa.nulls(len(item.timestamps), type=column_type)
            for component, item in sequences
        ]
        columns[name] = pc.take(pa.concat_arrays(parts), pa.array(order))
    return _Rows(boundaries=bounds, columns=columns, sources=selected_sources)


def _eligible(rows: _Rows, representation: Fields, declarations: Mapping[str, Any]) -> _Rows:
    import pyarrow.compute as pc

    required = (
        representation.row.required_fields
        if isinstance(representation.row, Composite)
        else tuple(value.field for value in representation.fields)
        if getattr(representation.row, "missing", None) == "skip_event"
        else ()
    )
    keep = np.ones(rows.size, dtype=bool)
    for field in required:
        declared = (
            np.isin(rows.sources, [item[0][-2] for item in declarations[field]])
            if declarations[field][0][0][0] in {"events", "current_events"}
            else np.ones(rows.size, dtype=bool)
        )
        valid = pc.is_valid(rows.columns[field]).to_numpy(zero_copy_only=False)
        keep &= valid if isinstance(representation.row, Composite) else ~declared | valid
    return rows.select(np.flatnonzero(keep)) if not keep.all() else rows


@dataclass
class _PreparedInput:
    spec: Input
    declarations: Mapping[str, Any]
    state: Mapping[str, OutputBinding]
    composite: Mapping[Any, int] | None = None


@dataclass(frozen=True)
class _InputState:
    prepared_inputs: Mapping[str, _PreparedInput]
    encoders: Mapping[str, FittedEncoder]
    spaces: Mapping[str, FittedSpace]
    field_names: Mapping[str, Mapping[tuple[str, ...], int]]


@dataclass(frozen=True)
class OutputView:
    bindings: Mapping[str, OutputBinding]

    def __post_init__(self):
        object.__setattr__(self, "bindings", MappingProxyType(dict(self.bindings)))

    def field(self, *, name: str) -> OutputBinding:
        if name not in self.bindings:
            raise ValueError(f"output has no field {name!r}")
        return self.bindings[name]


def _declarations(
    sample: SampleSpec, spec: BatchSpec, contract: Any, values: Sequence[HistoryInput]
) -> dict[str, dict[str, Any]]:
    if contract is not None:
        from kumo.pytorch.contract.batch.validation import resolved_fields

        result = {
            name: {
                use.field: resolved_fields(contract, name, use.field)
                for use in value.representation.fields
            }
            for name, value in spec.inputs.items()
            if isinstance(value.representation, Fields)
        }
        for target in spec.targets.values():
            if isinstance(target, AlignedEventTarget):
                for field in (
                    *(use.field for use in target.fields),
                    *(item.field for item in target.derived),
                ):
                    result[target.input][field] = resolved_fields(contract, target.input, field)
        return result
    result = {}
    for name, value in spec.inputs.items():
        if not isinstance(value.representation, Fields):
            continue
        if isinstance(value.source, OccurrencesOf):
            parent = spec.inputs[value.source.input].representation
            selected = {use.field.split("/")[-1] for use in parent.fields}
            found = {
                (dated.key, field.key): field.dtype
                for item in values
                for dated in _history(item).entity.dated_values
                if dated.key in selected
                for field in dated.fields
            }
            result[name] = {
                use.field: tuple(
                    (("dated_entity", dated, field), dtype, "scalar")
                    for (dated, field), dtype in found.items()
                    if field == use.field
                )
                for use in value.representation.fields
            }
            if any(not declarations for declarations in result[name].values()):
                raise ValueError(
                    "occurrence projection requires typed dated fields or a complete contract"
                )
            continue
        source = (
            spec.inputs[value.source.input].source
            if isinstance(value.source, EventsOf)
            else value.source
        )
        found = {}
        for sample_value in values:
            history = _history(sample_value)
            if isinstance(source, EntitySource):
                groups = [("entity", (*history.entity.values, *history.entity.dated_values))]
            elif isinstance(source, CandidateSource):
                groups = [
                    (source.name, (*item.history.entity.values, *item.history.entity.dated_values))
                    for item in sample_value.candidates.get(source.name, ())
                ]
            else:
                events = (
                    history.events
                    if source.collection == "history"
                    else tuple(
                        event
                        for name in source.events
                        for event in sample_value.current_events.get(name, ())
                    )
                )
                groups = [
                    (event.source, event.fields)
                    for event in events
                    if event.source in source.events
                ]
            for component, fields_ in groups:
                for field in fields_:
                    found[(component, field.key)] = field.dtype
        result[name] = {}
        selectors = [use.field for use in value.representation.fields]
        for target in spec.targets.values():
            if isinstance(target, AlignedEventTarget) and target.input == name:
                selectors.extend(use.field for use in target.fields)
                selectors.extend(item.field for item in target.derived)
        for selector in dict.fromkeys(selectors):
            variants = []
            for (component, field), dtype in found.items():
                if isinstance(source, EntitySource):
                    identity = (
                        "dated_entity" if field in sample.dated_entity_values else "entity",
                        field,
                    )
                    declaration = sample.entity_values.get(field)
                elif isinstance(source, CandidateSource):
                    identity = ("candidates", source.name, field)
                    declaration = sample.candidate_sets[source.name].values.get(field)
                else:
                    current = source.collection == "current"
                    identity = ("current_events" if current else "events", component, field)
                    declaration = (sample.current_events if current else sample.events)[
                        component
                    ].fields.get(field)
                if selector == field or selector == "/".join(identity):
                    cardinality = (
                        "repeated"
                        if identity[0] == "dated_entity"
                        or getattr(declaration, "selection", None) == "all"
                        else "scalar"
                    )
                    variants.append((identity, dtype, cardinality))
            if not variants:
                raise ValueError(
                    f"input {name!r}: cannot resolve type of {selector!r}; supply a typed contract"
                )
            result[name][selector] = tuple(variants)
    return result


def _rows_for_inputs(
    spec: BatchSpec, declarations: Any, values: Any, *, model: bool = False
) -> dict[str, _Rows]:
    result = {}
    # A follower projects the parent's retained rows. Include follower columns in
    # the same source read so it cannot independently select or reorder events.
    for name, value in spec.inputs.items():
        if not isinstance(value.representation, Fields) or isinstance(
            value.source, (EventsOf, OccurrencesOf)
        ):
            continue
        fields_ = dict(declarations[name])
        for follower_name, follower in spec.inputs.items():
            if isinstance(follower.source, EventsOf) and follower.source.input == name:
                fields_.update(declarations[follower_name])
        rows = (
            _model_rows(value.source, values, fields_)
            if model
            else _record_rows(value.source, values, fields_)
        )
        result[name] = _eligible(rows, value.representation, fields_)
    for name, value in spec.inputs.items():
        if isinstance(value.source, EventsOf):
            result[name] = result[value.source.input]
        elif isinstance(value.source, OccurrencesOf):
            import pyarrow as pa

            size = values.batch_size if model else len(values)
            result[name] = _Rows(
                boundaries=np.zeros(size + 1, dtype=np.int64),
                columns={
                    field: pa.array([], type=resolved[0][1].to_arrow())
                    for field, resolved in declarations[name].items()
                },
                sources=np.empty(0, dtype=object),
            )
    return result


def _encoder_identity(input_name: str, use: Encode) -> str:
    return (
        _identity("shared", use.encoding.name)
        if isinstance(use.encoding, UseEncoding)
        else _identity("inputs", input_name, "fields", use.field)
    )


def _graph_encoder_identity(input_name: str, table: str, use: Encode) -> str:
    return (
        _identity("shared", use.encoding.name)
        if isinstance(use.encoding, UseEncoding)
        else _identity("inputs", input_name, "graph", table, use.field)
    )


def _fit_inputs(
    sample: SampleSpec,
    spec: BatchSpec,
    declarations: Mapping[str, Any],
    values_for: Any,
    contract: Any,
    *,
    model: bool = False,
    source: Any = None,
    pinned: Mapping[str, FittedEncoder] = MappingProxyType({}),
) -> _InputState:
    import pyarrow as pa

    slots = {}
    definitions = {}
    field_names = {name: {} for name in spec.field_names}
    graph_fields = {}
    for name in sorted(spec.inputs):
        value = spec.inputs[name]
        if not isinstance(value.representation, Fields):
            graph_fields[name] = {}
            tables = {table.name: table for table in contract.schema.tables}
            for table, group in value.representation.node_fields.items():
                graph_fields[name][table] = {}
                for use in group.fields:
                    identity = _graph_encoder_identity(name, table, use)
                    dtype = tables[table].columns[use.field]
                    config = (
                        spec.encodings[use.encoding.name]
                        if isinstance(use.encoding, UseEncoding)
                        else use.encoding
                    )
                    if identity in definitions and definitions[identity][1] != dtype:
                        raise ValueError(
                            f"encoder {identity}: sharing requires identical declared dtypes"
                        )
                    definitions[identity] = (config, dtype)
                    if identity not in pinned and learns_from_values(config, dtype):
                        slots[(name, table, use.field)] = identity
                    graph_fields[name][table][use.field] = identity
            continue
        for use in value.representation.fields:
            identity = _encoder_identity(name, use)
            dtype = declarations[name][use.field][0][1]
            config = (
                spec.encodings[use.encoding.name]
                if isinstance(use.encoding, UseEncoding)
                else use.encoding
            )
            if identity in definitions and definitions[identity][1] != dtype:
                raise ValueError(f"encoder {identity}: sharing requires identical declared dtypes")
            definitions[identity] = (config, dtype)
            if identity not in pinned and learns_from_values(config, dtype):
                slots[(name, use.field)] = identity
            if value.representation.field_names is not None:
                field_names[value.representation.field_names].update(
                    {tuple(item[0]): 0 for item in declarations[name][use.field]}
                )
    observed = {slot: [] for slot in slots}
    composites = {
        name: Counter()
        for name, value in spec.inputs.items()
        if isinstance(value.representation, Fields)
        and isinstance(value.representation.row, Composite)
        and value.representation.row.vocabulary is None
    }
    for name, value in spec.inputs.items():
        graph_slots = [slot for slot in observed if slot[0] == name]
        if isinstance(value.representation, Graph) and graph_slots:
            graph = SimpleNamespace(
                source=value.source.name, node_fields=value.representation.node_fields
            )
            roots = [
                GraphRoot(request.sample_id, request.entity_id, cutoff_to_epoch(request.as_of_time))
                for requests in _chunks(values_for(None))
                for request in requests
            ]
            found = source.fit_graph_fields(sample, graph, roots)
            for slot in graph_slots:
                observed[slot].extend(found[slot[1:]])
    builds = composites or any(
        isinstance(spec.inputs[slot[0]].representation, Fields) for slot in observed
    )
    for values in _chunks(values_for(sample_without_targets(sample))) if builds else ():
        rows = _rows_for_inputs(spec, declarations, values, model=model)
        for name, value in spec.inputs.items():
            if not isinstance(value.representation, Fields):
                continue
            fitting_rows = rows[name]
            if isinstance(value.representation.row, Composite):
                fitting_rows = (
                    _model_rows(value.source, values, declarations[name])
                    if model
                    else _record_rows(value.source, values, declarations[name])
                )
            for use in value.representation.fields:
                if (name, use.field) in observed:
                    column = fitting_rows.columns[use.field]
                    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
                        column = column.flatten()
                    observed[(name, use.field)].extend(column.to_pylist())
            if name in composites:
                columns = (rows[name].columns[use.field] for use in value.representation.fields)
                composites[name].update(
                    zip(*(column.to_pylist() for column in columns), strict=True)
                )
    contributions = {}
    for slot, identity in slots.items():
        contributions.setdefault(identity, []).extend(observed[slot])
    encoders = {
        name: pinned[name]
        if name in pinned
        else fit_encoder(
            identity=name, config=config, dtype=dtype, values=contributions.get(name, ())
        )
        for name, (config, dtype) in sorted(definitions.items())
    }
    field_names = {
        name: {identity: index + 1 for index, identity in enumerate(sorted(identities))}
        for name, identities in field_names.items()
    }
    prepared = {}
    for name, value in spec.inputs.items():
        if isinstance(value.representation, Graph):
            prepared[name] = _PreparedInput(
                spec=value,
                declarations={},
                state={
                    table: {
                        field: OutputBinding(encoder=encoders[identity])
                        for field, identity in fields_.items()
                    }
                    for table, fields_ in graph_fields[name].items()
                },
            )
            continue
        bindings = {
            use.field: OutputBinding(encoder=encoders[_encoder_identity(name, use)])
            for use in value.representation.fields
        }
        composite = None
        if isinstance(value.representation.row, Composite):
            config = value.representation.row
            if config.vocabulary is None:
                counts = Counter()
                for key, count in composites[name].items():
                    parts = zip(value.representation.fields, key, strict=True)
                    key = tuple(_component_key(bindings[use.field], part) for use, part in parts)
                    counts[key] += count
                ordered = sorted(counts, key=lambda key: (-counts[key], key))[: config.top_k]
                composite = {key: index + 2 for index, key in enumerate(ordered)}
            else:
                composite = {
                    tuple(tuple(part) if isinstance(part, list) else part for part in key): index
                    for key, index in config.vocabulary
                }
        prepared[name] = _PreparedInput(
            spec=value, declarations=declarations[name], state=bindings, composite=composite
        )
    spaces = _allocate_spaces(spec, prepared, field_names)
    for name, value in prepared.items():
        if not isinstance(value.spec.representation, Fields):
            for table, group in value.spec.representation.node_fields.items():
                for use in group.fields:
                    if use.id_space:
                        value.state[table][use.field] = replace(
                            value.state[table][use.field],
                            destination=spaces[use.id_space],
                            id_space=use.id_space,
                        )
            continue
        for use in value.spec.representation.fields:
            binding = value.state[use.field]
            value.state[use.field] = replace(
                binding,
                destination=spaces[use.id_space] if use.id_space else None,
                id_space=use.id_space,
            )
    return _InputState(
        prepared_inputs=prepared, encoders=encoders, spaces=spaces, field_names=field_names
    )


def _component_key(binding: OutputBinding, value: Any) -> Any:
    if value is None:
        return ("missing", str(binding.encoder.dtype))
    result = binding.encoder.encode(value=value)
    return (
        typed_key(binding.encoder.dtype, result)
        if isinstance(binding.encoder.config, Raw)
        else (type(binding.encoder.config).__name__, result)
    )


def _allocate_spaces(
    spec: BatchSpec, prepared: Any, field_names: Any, *, extra_members=()
) -> dict[str, FittedSpace]:
    members = {name: {} for name in spec.id_spaces}
    for input_name, value in prepared.items():
        group = value.spec.representation
        if not isinstance(group, Fields):
            for table, attributes in group.node_fields.items():
                for use in attributes.fields:
                    if use.id_space:
                        encoder = value.state[table][use.field].encoder
                        members[use.id_space][encoder.identity] = (encoder.local_ids, encoder.roles)
            continue
        for use in group.fields:
            if use.id_space:
                encoder = value.state[use.field].encoder
                members[use.id_space][encoder.identity] = (encoder.local_ids, encoder.roles)
        if group.field_space:
            identity = _identity("fields", group.field_names)
            members[group.field_space][identity] = (
                (0, *field_names[group.field_names].values()),
                {"pad": 0},
            )
        if isinstance(group.row, Composite) and group.row.id_space:
            members[group.row.id_space][_identity("composite", input_name)] = (
                (0, 1, *value.composite.values()),
                {"missing": 0, "unknown": 1},
            )
    for space_name, encoder in extra_members:
        members[space_name][encoder.identity] = (encoder.local_ids, encoder.roles)
    spaces = {}
    for name, definition in spec.id_spaces.items():
        role_names = [role for role in ROLE_ORDER if role in definition.specials] + sorted(
            set(definition.specials) - set(ROLE_ORDER)
        )
        roles = {role: index for index, role in enumerate(role_names)}
        cursor = len(roles)
        mappings = {}
        for identity, (ids, local_roles) in sorted(members[name].items()):
            inverse_roles = {value: role for role, value in local_roles.items()}
            mapping = {}
            for index in sorted(set(ids)):
                role = inverse_roles.get(index)
                if role in roles:
                    mapping[index] = roles[role]
                else:
                    mapping[index] = cursor
                    cursor += 1
            mappings[identity] = mapping
        spaces[name] = FittedSpace(roles=roles, members=mappings, size=cursor)
    return spaces


def _occurrence_rows(
    name: str, state: _InputState, rows: Mapping[str, _Rows], values: Any
) -> _Rows:
    """Repeat explicitly sampled occurrence fields over retained parent positions."""
    import pyarrow as pa

    prepared = state.prepared_inputs[name]
    parent_name = prepared.spec.source.input
    parent = state.prepared_inputs[parent_name]
    tokens = _token_rows(parent_name, parent, rows[parent_name], state)
    occurrences = []
    for value in values:
        grouped = {}
        for dated in sorted(_history(value).entity.dated_values, key=lambda item: item.timestamp):
            grouped.setdefault(dated.key, []).append(
                {field.key: field.value for field in dated.fields}
            )
        occurrences.append(grouped)
    columns = {field: [] for field in prepared.declarations}
    owners = np.repeat(np.arange(len(values)), np.diff(tokens.boundaries))
    sources = []
    for owner, slot, occurrence in zip(owners, tokens.slots, tokens.occurrences, strict=True):
        field = parent.spec.representation.fields[int(slot)].field.split("/")[-1]
        candidates = occurrences[int(owner)].get(field, ())
        projected = candidates[int(occurrence)] if candidates else {}
        for name in columns:
            columns[name].append(projected.get(name))
        sources.append(field)
    return _Rows(
        boundaries=tokens.boundaries,
        columns={
            field: pa.array(items, type=prepared.declarations[field][0][1].to_arrow())
            for field, items in columns.items()
        },
        sources=np.asarray(sources, dtype=object),
    )


def _encoded_column(binding: OutputBinding, column: Any, *, model=False) -> dict[str, Any]:
    import pyarrow as pa

    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        result = binding.encode_arrow(column.flatten(), model=model)
        result["occurrence_offsets"] = torch.tensor(
            column.offsets.to_numpy(zero_copy_only=False), dtype=torch.long
        )
        return result
    return binding.encode_arrow(column, model=model)


def _encoded_rows(encoded: dict[str, Any], boundaries: np.ndarray, layout: Any) -> dict[str, Any]:
    """Lay out scalar rows; expanding fields keep explicit piece/occurrence boundaries."""
    if "occurrence_offsets" in encoded or "piece_offsets" in encoded:
        return {**encoded, "offsets": torch.from_numpy(boundaries.copy())}
    return {
        name: layout_array(item, boundaries, layout, fill=False if name == "valid" else 0)
        for name, item in encoded.items()
    }


def _column_output(
    prepared: _PreparedInput, rows: _Rows, *, model=False, state=None
) -> dict[str, Any]:
    result = {}
    entity = isinstance(prepared.spec.source, EntitySource)
    layout = prepared.spec.layout
    for field, binding in prepared.state.items():
        encoded = _encoded_column(binding, rows.columns[field], model=model)
        group = prepared.spec.representation
        if group.field_names is not None:
            if state is None:
                raise ValueError("field-name column output requires its fitted mapping state")
            ids = _field_ids(
                prepared,
                field,
                rows,
                state.field_names[group.field_names],
                state.spaces.get(group.field_space),
            )
            owned = np.ones(rows.size, dtype=bool)
            if prepared.declarations[field][0][0][0] in {"events", "current_events"}:
                owned = np.isin(
                    rows.sources, [item[0][-2] for item in prepared.declarations[field]]
                )
            for key in ("occurrence_offsets", "piece_offsets"):
                if key in encoded:
                    lengths = np.diff(encoded[key].numpy())
                    ids, owned = np.repeat(ids, lengths), np.repeat(owned, lengths)
            encoded["field_ids"] = torch.from_numpy(ids.copy())
            encoded["field_valid"] = torch.from_numpy(owned.copy())
        if not entity:
            encoded = _encoded_rows(encoded, rows.boundaries, layout)
        result[field] = encoded
    if not entity:
        result["valid"] = layout_array(
            torch.ones(rows.size, dtype=torch.bool), rows.boundaries, layout, fill=False
        )
        if isinstance(layout, Packed):
            result["offsets"] = torch.from_numpy(rows.boundaries.copy())
    return result


def _field_ids(
    prepared: _PreparedInput,
    field: str,
    rows: _Rows,
    names: Mapping[Any, int],
    space: FittedSpace | None,
) -> np.ndarray:
    result = np.zeros(rows.size, dtype=np.int64)
    for identity, _dtype, _cardinality in prepared.declarations[field]:
        index = names[tuple(identity)]
        if space is not None:
            index = space.members[_identity("fields", prepared.spec.representation.field_names)][
                index
            ]
        mask = (
            rows.sources == identity[-2]
            if identity[0] in {"events", "current_events"}
            else np.ones(rows.size, dtype=bool)
        )
        result[mask] = index
    return result


def _token_rows(
    name: str, prepared: _PreparedInput, rows: _Rows, state: _InputState, *, model=False
) -> TokenRows:
    import pyarrow as pa
    import pyarrow.compute as pc

    group = prepared.spec.representation
    if isinstance(group.row, Composite):
        raw = {use.field: rows.columns[use.field].to_pylist() for use in group.fields}
        keys = [
            tuple(
                _component_key(prepared.state[use.field], raw[use.field][index])
                for use in group.fields
            )
            for index in range(rows.size)
        ]
        ids = np.asarray([prepared.composite.get(key, 1) for key in keys], dtype=np.int64)
        if group.row.id_space:
            mapping = state.spaces[group.row.id_space].members[_identity("composite", name)]
            ids = np.asarray([mapping[int(index)] for index in ids], dtype=np.int64)
        return TokenRows(
            values=ids,
            boundaries=np.arange(rows.size + 1),
            positions=np.zeros(rows.size, dtype=np.int64),
            fields=np.zeros(rows.size, dtype=np.int64),
            slots=np.full(rows.size, -1, dtype=np.int64),
            field_valid=np.zeros(rows.size, dtype=bool),
            events=np.arange(rows.size),
            occurrences=np.zeros(rows.size, dtype=np.int64),
        )
    owners, ranks, positions, keys, slots, values, occurrences, times, dated = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for rank, use in enumerate(group.fields):
        column = rows.columns[use.field]
        binding = prepared.state[use.field]
        repeated = pa.types.is_list(column.type) or pa.types.is_large_list(column.type)
        row_ids = (
            np.repeat(np.arange(rows.size), np.diff(column.offsets.to_numpy(zero_copy_only=False)))
            if repeated
            else np.arange(rows.size)
        )
        occurrence_ids = (
            np.arange(len(row_ids))
            - np.repeat(
                column.offsets.to_numpy(zero_copy_only=False)[:-1],
                np.diff(column.offsets.to_numpy(zero_copy_only=False)),
            )
            if repeated
            else np.zeros(rows.size, dtype=np.int64)
        )
        occurrence_times = rows.occurrence_times.get(use.field)
        if occurrence_times is not None:
            occurrence_times = occurrence_times.flatten() if repeated else occurrence_times
            is_dated = pc.is_valid(occurrence_times).to_numpy(zero_copy_only=False)
            time_values = pc.fill_null(occurrence_times, 0).to_numpy(zero_copy_only=False)
        else:
            is_dated, time_values = (
                np.zeros(len(row_ids), dtype=bool),
                np.zeros(len(row_ids), dtype=np.int64),
            )
        encoded = _encoded_column(binding, column, model=model)
        scalar_valid = encoded["valid"].numpy()
        present = np.ones(len(row_ids), dtype=bool)
        declared = prepared.declarations[use.field]
        if declared[0][0][0] in {"events", "current_events"}:
            present &= np.isin(rows.sources[row_ids], [item[0][-2] for item in declared])
        if group.row.missing == "error" and bool((present & ~scalar_valid).any()):
            raise ValueError(f"input {name!r}, field {use.field!r}: NULL under missing='error'")
        include = present & (scalar_valid if group.row.missing in {"omit", "skip_event"} else True)
        if "piece_offsets" in encoded:
            piece_lengths = np.diff(encoded["piece_offsets"].numpy())
            piece_ids = np.repeat(np.arange(len(row_ids)), piece_lengths)
            field_positions = np.arange(len(encoded["values"])) - np.repeat(
                encoded["piece_offsets"].numpy()[:-1], piece_lengths
            )
            include = include[piece_ids]
            row_ids = row_ids[piece_ids]
            occurrence_ids = occurrence_ids[piece_ids]
            is_dated, time_values = is_dated[piece_ids], time_values[piece_ids]
        else:
            field_positions = np.zeros(len(row_ids), dtype=np.int64)
        row_ids = row_ids[include]
        owners.append(row_ids)
        ranks.append(np.full(len(row_ids), rank, dtype=np.int64))
        positions.append(field_positions[include])
        occurrences.append(occurrence_ids[include])
        times.append(time_values[include])
        dated.append(is_dated[include])
        values.append(encoded["values"].numpy()[include])
        slots.append(np.full(len(row_ids), rank, dtype=np.int64))
        if group.field_names:
            field_space = state.spaces.get(group.field_space)
            ids = _field_ids(
                prepared, use.field, rows, state.field_names[group.field_names], field_space
            )
            keys.append(ids[row_ids])
        else:
            keys.append(np.zeros(len(row_ids), dtype=np.int64))
    owners = np.concatenate(owners)
    ranks = np.concatenate(ranks)
    order = np.lexsort((np.arange(len(owners)), ranks, owners))
    if group.row.occurrence_order == "chronological":
        order = np.lexsort(
            (np.arange(len(owners)), ranks, np.concatenate(times), np.concatenate(dated), owners)
        )
    counts = np.bincount(owners, minlength=rows.size)
    boundaries = offsets(counts)
    if group.row.max_tokens is not None:
        kept = np.minimum(counts, group.row.max_tokens)
        starts = boundaries[:-1] + (counts - kept if group.row.keep == "last" else 0)
        bounds = offsets(kept)
        within = np.arange(int(bounds[-1])) - np.repeat(bounds[:-1], kept)
        order = order[np.repeat(starts, kept) + within]
        boundaries = bounds
    return TokenRows(
        values=np.concatenate(values)[order],
        boundaries=boundaries,
        positions=np.concatenate(positions)[order],
        fields=np.concatenate(keys)[order],
        slots=np.concatenate(slots)[order],
        field_valid=np.ones(len(order), dtype=bool),
        events=owners[order],
        occurrences=np.concatenate(occurrences)[order],
    )


def _mask_tokens(
    tokens: TokenRows,
    rows: _Rows,
    target: MaskedValuesTarget,
    group: Fields,
    space: FittedSpace,
    entity_ids: Sequence[Any],
    *,
    epoch: int,
    seed: int,
) -> tuple[TokenRows, dict[str, Any]]:
    ids = layout_array(
        torch.from_numpy(tokens.values.copy()), tokens.boundaries, Padded(), fill=space.roles["pad"]
    )
    key_ids = layout_array(torch.from_numpy(tokens.slots.copy()), tokens.boundaries, Padded())
    valid = layout_array(
        torch.ones(len(tokens.values), dtype=torch.bool), tokens.boundaries, Padded(), fill=False
    )
    actual = valid.clone()
    if target.fields:
        allowed = torch.tensor(
            [index for index, use in enumerate(group.fields) if use.field in target.fields],
            dtype=torch.long,
        )
        valid &= torch.isin(key_ids, allowed)
    owners = np.repeat(np.arange(len(rows.boundaries) - 1), np.diff(rows.boundaries))
    history_position = np.arange(rows.size) - np.repeat(
        rows.boundaries[:-1], np.diff(rows.boundaries)
    )
    entity_keys = [
        int(value)
        if isinstance(value, (int, bool))
        else int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:4], "big")
        for value in entity_ids
    ]
    masking = MaskingConfig(
        token_probability=target.token_probability,
        event_probability=target.event_probability,
        semantic_type_probability=target.key_probability,
        unknown_probability=target.unknown_probability,
        combine=target.combine,
        ensure_at_least_one=target.ensure_at_least_one,
        supervise_unknown=target.supervise_unknown,
    )
    corrupted, answers, supervised = mask_event_values(
        key_ids=key_ids,
        value_ids=ids,
        valid=valid,
        event_owner=torch.tensor(owners),
        event_history_position=torch.tensor(history_position),
        entity_ids=torch.tensor(entity_keys),
        masking=masking,
        seed=seed,
        epoch=epoch,
        mask_id=space.roles["mask"],
        unknown_id=space.roles.get("unknown", space.roles["mask"]),
    )
    changed = replace(tokens, values=corrupted[actual].numpy())
    result = {
        "values": answers[actual],
        "valid": supervised[actual],
        "corruption": (corrupted != ids)[actual],
    }
    return changed, result


def _masked_layout(targets, boundaries, layout, *, source_indices=None):
    for name, target in targets.items():
        if source_indices is not None:
            real = source_indices >= 0
            selected = {}
            for key, values in target.items():
                gathered = torch.full(
                    (len(source_indices),), -100 if key == "values" else False, dtype=values.dtype
                )
                gathered[real] = values[source_indices[real]]
                selected[key] = gathered
            target = selected
        targets[name] = {
            key: layout_array(values, boundaries, layout, fill=-100 if key == "values" else False)
            for key, values in target.items()
        }
        if isinstance(layout, Packed):
            targets[name]["offsets"] = torch.from_numpy(boundaries.copy())


def _token_output(
    name: str,
    prepared: _PreparedInput,
    rows: _Rows,
    state: _InputState,
    batch_spec: BatchSpec,
    entity_ids,
    *,
    model=False,
    packer=None,
    seed=0,
    epoch=0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    group = prepared.spec.representation
    tokens = _token_rows(name, prepared, rows, state, model=model)
    space_name = (
        group.row.id_space if isinstance(group.row, Composite) else group.fields[0].id_space
    )
    space = state.spaces.get(space_name)
    pad = space.roles.get("pad", 0) if space is not None else 0
    targets = {}
    masked_targets = [
        (target_name, target)
        for target_name, target in batch_spec.targets.items()
        if isinstance(target, MaskedValuesTarget) and target.input == name
    ]
    if not isinstance(group.history, Flatten):
        for target_name, target in masked_targets:
            tokens, targets[target_name] = _mask_tokens(
                tokens, rows, target, group, space, entity_ids, epoch=epoch, seed=seed
            )
    if isinstance(group.history, Flatten):
        source_tokens = tokens
        if masked_targets:
            tokens = replace(tokens, source_indices=np.arange(len(tokens.values)))
        tokens = flatten_rows(tokens, rows.boundaries, group.history, space.roles, packer=packer)
        if masked_targets:
            real = tokens.source_indices >= 0
            kept = source_tokens.select(
                tokens.source_indices[real],
                offsets(np.bincount(tokens.events[real], minlength=rows.size)),
            )
            for target_name, target in masked_targets:
                changed, targets[target_name] = _mask_tokens(
                    kept, rows, target, group, space, entity_ids, epoch=epoch, seed=seed
                )
                corrupted = tokens.values.copy()
                corrupted[real] = changed.values
                tokens = replace(tokens, values=corrupted)
            indices = np.full(len(tokens.values), -1, dtype=np.int64)
            indices[real] = np.arange(int(real.sum()))
            _masked_layout(targets, tokens.boundaries, prepared.spec.layout, source_indices=indices)
        next_targets = [
            (target_name, target)
            for target_name, target in batch_spec.targets.items()
            if isinstance(target, NextTokenTarget) and target.input == name
        ]
        if next_targets:
            target_name, target = next_targets[0]
            lengths = np.diff(tokens.boundaries)
            input_lengths = np.maximum(lengths - 1, 0)
            new_bounds = offsets(input_lengths)
            indices = (
                np.arange(int(new_bounds[-1]))
                - np.repeat(new_bounds[:-1], input_lengths)
                + np.repeat(tokens.boundaries[:-1], input_lengths)
            )
            answer_indices = indices + 1
            answer_bounds = new_bounds
            if target.prefix_input is not None:
                answer_indices = np.arange(len(tokens.values))
                answer_bounds = tokens.boundaries
            target_values = torch.from_numpy(tokens.values[answer_indices].copy())
            answers = sequence_layout(answer_bounds, prepared.spec.layout, values=target_values)
            answers["values"][~answers["valid"]] = -100
            targets[target_name] = answers
            tokens = tokens.select(indices, new_bounds)
        return tokens.output(
            prepared.spec.layout, pad=pad, field_ids=group.field_names is not None
        ), targets
    if isinstance(prepared.spec.source, EntitySource):
        tokens = tokens.select(np.arange(len(tokens.values)), tokens.boundaries[rows.boundaries])
        _masked_layout(targets, tokens.boundaries, prepared.spec.layout)
        return tokens.output(
            prepared.spec.layout, pad=pad, field_ids=group.field_names is not None
        ), targets
    if isinstance(group.row, Composite):
        return tokens.select(np.arange(len(tokens.values)), rows.boundaries).output(
            prepared.spec.layout, pad=pad, field_ids=False
        ), targets
    layout = prepared.spec.layout
    result = tokens.output(layout.fields, pad=pad, field_ids=group.field_names is not None)
    if "offsets" in result:
        result["field_offsets"] = result.pop("offsets")
    lengths = np.diff(rows.boundaries)
    result["owner"] = torch.from_numpy(np.repeat(np.arange(len(lengths)), lengths))
    result["history_position"] = torch.from_numpy(
        np.arange(rows.size) - np.repeat(rows.boundaries[:-1], lengths)
    )
    result["sequence"] = sequence_layout(
        rows.boundaries, layout.events, event_indices=torch.arange(rows.size)
    )
    _masked_layout(targets, tokens.boundaries, layout.fields)
    return result, targets


@dataclass(frozen=True)
class _TargetState:
    bindings: Mapping[str, OutputBinding]
    thresholds: Mapping[str, float] | None = None
    groups: Mapping[Any, int] | None = None


def _raw_targets(
    values: Any, source: str, *, field: str | None = None, candidate: str | None = None
):
    if isinstance(values, ModelSourceBatch):
        if field is not None:
            return values.row_targets[source].fields[field].to_pylist()
        column = values.targets[source]
        return column.to_pylist() if hasattr(column, "to_pylist") else list(column)
    result = []
    for value in values:
        records = value.candidates.get(candidate, ()) if candidate else (value,)
        for record in records:
            target = _targets(record).get(source)
            result.append(
                next((item.value for item in target.fields if item.key == field), None)
                if field is not None and isinstance(target, Event)
                else target
            )
    return result


def _record_target_dtype(sample, source, values, *, field=None, candidate=None):
    definition = (sample.candidate_sets[candidate].targets if candidate else sample.targets).get(
        source
    )
    if definition is None:
        raise ValueError(f"target {source!r} requires an explicit SampleSpec declaration")
    if field is None:
        return definition.dtype
    if hasattr(definition.fields[field], "dtype"):
        return definition.fields[field].dtype
    for value in values:
        target = _targets(value).get(source)
        if isinstance(target, Event):
            matched = next((item for item in target.fields if item.key == field), None)
            if matched is not None:
                return matched.dtype
    raise ValueError(
        f"target {source!r}, field {field!r}: typed records or a complete contract are required"
    )


def fitted_target_names(spec: BatchSpec) -> list[str]:
    """The targets fitting gives state to; the rest (masking, next token) carry none."""

    return sorted(
        name
        for name, target in spec.targets.items()
        if not isinstance(target, (MaskedValuesTarget, NextTokenTarget))
    )


def _fit_targets(
    sample: SampleSpec,
    spec: BatchSpec,
    declarations: Mapping[str, Any],
    values_for: Any,
    state: _InputState,
    contract: Any,
    *,
    model=False,
    reused=False,
) -> tuple[_InputState, Mapping[str, _TargetState]]:
    from kumo.pytorch.contract.batch.validation import target_dtype

    if reused:
        _check_reused_target_spaces(spec, state)
    slots, derived, definitions, uses, groups = {}, {}, {}, {}, {}
    for name in sorted(spec.targets):
        target = spec.targets[name]
        if isinstance(target, (MaskedValuesTarget, NextTokenTarget)):
            continue
        if isinstance(target, MatchingGroupTarget):
            groups[name] = set()
            uses[name] = []
            continue
        fields_ = (
            target.fields
            if isinstance(target, (StructuredTarget, AlignedEventTarget))
            else (Encode(field="value", encoding=target.encoding, id_space=target.id_space),)
        )
        uses[name] = list(fields_)
        for use in fields_:
            if isinstance(target, AlignedEventTarget):
                dtype = declarations[target.input][use.field][0][1]
            else:
                candidate = (
                    spec.inputs[target.input].source.name
                    if isinstance(target, CandidateTarget)
                    else None
                )
                dtype = (
                    target_dtype(
                        contract,
                        target.source,
                        candidate=candidate,
                        field=use.field if isinstance(target, StructuredTarget) else None,
                    )
                    if contract is not None
                    else _record_target_dtype(
                        sample,
                        target.source,
                        values_for(sample),
                        candidate=candidate,
                        field=use.field if isinstance(target, StructuredTarget) else None,
                    )
                )
            identity = (
                _identity("shared", use.encoding.name)
                if isinstance(use.encoding, UseEncoding)
                else _identity("targets", name, use.field)
            )
            config = (
                spec.encodings[use.encoding.name]
                if isinstance(use.encoding, UseEncoding)
                else use.encoding
            )
            definitions[identity] = (config, dtype)
            if identity not in state.encoders and learns_from_values(config, dtype):
                slots[(name, use.field)] = identity
        if isinstance(target, AlignedEventTarget):
            for item in target.derived:
                use = Encode(field=item.name, encoding=item.encoding)
                uses[name].append(use)
                identity = (
                    _identity("shared", use.encoding.name)
                    if isinstance(use.encoding, UseEncoding)
                    else _identity("targets", name, use.field)
                )
                config = (
                    spec.encodings[use.encoding.name]
                    if isinstance(use.encoding, UseEncoding)
                    else use.encoding
                )
                definitions[identity] = (config, item.dtype)
                derived[(name, item.name)] = item
                if identity not in state.encoders and learns_from_values(config, item.dtype):
                    slots[(name, item.name)] = identity
    observed = {
        key: []
        for key in (*slots, *(key for key, item in derived.items() if item.threshold is None))
    }
    learned = {name for name, _ in observed} | set(groups)
    reads_rows = any(
        isinstance(spec.targets[name], (AlignedEventTarget, MatchingGroupTarget, CandidateTarget))
        for name in learned
    )
    computed = any(
        not isinstance(sample.targets.get(spec.targets[name].source), ExternalTarget)
        for name in learned
        if isinstance(spec.targets[name], (LabelTarget, StructuredTarget))
    )
    if reads_rows:
        built = sample
    elif computed:
        built = replace(
            sample,
            entity_values={},
            dated_entity_values={},
            events={},
            current_events={},
            neighborhoods={},
        )
    else:
        built = None
    for values in _chunks(values_for(built)) if learned else ():
        rows = _rows_for_inputs(spec, declarations, values, model=model) if reads_rows else {}
        for name in groups:
            target = spec.targets[name]
            bindings = state.prepared_inputs[target.input].state
            columns = [rows[target.input].columns[field].to_pylist() for field in target.fields]
            groups[name].update(
                tuple(
                    typed_key(bindings[field].encoder.dtype, value)
                    for field, value in zip(target.fields, row, strict=True)
                )
                for row in zip(*columns, strict=True)
            )
        for name, field in observed:
            target = spec.targets[name]
            if isinstance(target, AlignedEventTarget):
                column = derived[(name, field)].field if (name, field) in derived else field
                observed[(name, field)].extend(rows[target.input].columns[column].to_pylist())
            else:
                observed[(name, field)].extend(
                    _raw_targets(
                        values,
                        target.source,
                        field=field if isinstance(target, StructuredTarget) else None,
                        candidate=spec.inputs[target.input].source.name
                        if isinstance(target, CandidateTarget)
                        else None,
                    )
                )
    groups = {
        name: {key: index for index, key in enumerate(sorted(keys))}
        for name, keys in groups.items()
    }
    thresholds = {}
    for (name, field), item in derived.items():
        raw = observed.get((name, field), [])
        numeric = [float(value) for value in raw if value is not None]
        threshold = (
            item.threshold
            if item.threshold is not None
            else float(np.median(numeric))
            if numeric
            else 0.0
        )
        if not np.isfinite(numeric).all() or not np.isfinite(threshold):
            raise ValueError(f"target {name!r}: threshold values must be finite")
        thresholds.setdefault(name, {})[field] = threshold
        if (name, field) in slots:
            above = (
                (lambda value: value > threshold)
                if item.comparison == "greater_than"
                else (lambda value: value >= threshold)
            )
            observed[(name, field)] = [
                None if value is None else item.above if above(value) else item.below
                for value in raw
            ]
    contributors = {}
    for key, identity in slots.items():
        contributors.setdefault(identity, []).extend(observed[key])
    encoders = dict(state.encoders)
    for identity, (config, dtype) in sorted(definitions.items()):
        if identity in encoders:
            existing = encoders[identity]
            if existing.dtype != dtype or existing.config != config:
                raise ValueError(f"target encoder {identity}: incompatible shared encoding")
            continue  # Input-shared encoders never collect target observations.
        encoders[identity] = fit_encoder(
            identity=identity, config=config, dtype=dtype, values=contributors.get(identity, ())
        )
    target_states = {}
    target_members = []
    for name, declarations_ in uses.items():
        bindings = {}
        for use in declarations_:
            identity = (
                _identity("shared", use.encoding.name)
                if isinstance(use.encoding, UseEncoding)
                else _identity("targets", name, use.field)
            )
            bindings[use.field] = OutputBinding(encoder=encoders[identity])
        target_states[name] = _TargetState(
            bindings=bindings, thresholds=thresholds.get(name), groups=groups.get(name)
        )
        target_members.extend(
            (use.id_space, bindings[use.field].encoder)
            for use in declarations_
            if use.id_space is not None
        )
    allocated = _allocate_spaces(
        spec, state.prepared_inputs, state.field_names, extra_members=target_members
    )
    if reused:
        for name, space in state.spaces.items():
            if any(identity not in space.members for identity in allocated[name].members):
                raise ValueError(f"reused ID space {name!r} cannot acquire target members")
        spaces = {**allocated, **state.spaces}
    else:
        spaces = allocated
    prepared = {}
    for name, value in state.prepared_inputs.items():
        if isinstance(value.spec.representation, Fields):
            destinations = {use.field: use.id_space for use in value.spec.representation.fields}
            bindings = {
                field: replace(
                    binding,
                    destination=spaces[destinations[field]] if destinations[field] else None,
                    id_space=destinations[field],
                )
                for field, binding in value.state.items()
            }
            prepared[name] = replace(value, state=bindings, declarations=declarations[name])
        else:
            prepared[name] = replace(
                value,
                state={
                    table: {
                        field: replace(
                            binding,
                            destination=spaces[use.id_space] if use.id_space else None,
                            id_space=use.id_space,
                        )
                        for use in value.spec.representation.node_fields[table].fields
                        for field, binding in ((use.field, bindings[use.field]),)
                    }
                    for table, bindings in value.state.items()
                },
            )
    for name, target in target_states.items():
        destinations = {use.field: use.id_space for use in uses[name]}
        target_states[name] = replace(
            target,
            bindings={
                field: replace(
                    binding,
                    destination=spaces[destinations[field]] if destinations[field] else None,
                    id_space=destinations[field],
                )
                for field, binding in target.bindings.items()
            },
        )
    return _InputState(
        prepared_inputs=prepared, encoders=encoders, spaces=spaces, field_names=state.field_names
    ), target_states


def _check_reused_target_spaces(spec: BatchSpec, state: _InputState) -> None:
    for name, target in spec.targets.items():
        if isinstance(target, (StructuredTarget, AlignedEventTarget)):
            uses = target.fields
        elif isinstance(target, (LabelTarget, CandidateTarget)):
            uses = (Encode(field="value", encoding=target.encoding, id_space=target.id_space),)
        else:
            continue
        for use in uses:
            if use.id_space not in state.spaces:
                continue
            identity = (
                _identity("shared", use.encoding.name)
                if isinstance(use.encoding, UseEncoding)
                else _identity("targets", name, use.field)
            )
            config = (
                spec.encodings[use.encoding.name]
                if isinstance(use.encoding, UseEncoding)
                else use.encoding
            )
            if identity not in state.spaces[use.id_space].members:
                raise ValueError(
                    f"reused ID space {use.id_space!r} cannot acquire target member {identity}"
                )
            if identity in state.encoders and state.encoders[identity].config != config:
                raise ValueError(f"reused encoder {identity} cannot change its configuration")


def _target_outputs(
    spec: BatchSpec,
    states: Mapping[str, _TargetState],
    inputs: Any,
    rows: Any,
    values: Any,
    *,
    model=False,
) -> dict[str, Any]:
    result = {}
    for name, state in states.items():
        target = spec.targets[name]
        if isinstance(target, MatchingGroupTarget):
            columns = {
                field: rows[target.input].columns[field].to_pylist() for field in target.fields
            }
            bindings = inputs[target.input].state
            keys = [
                tuple(
                    typed_key(bindings[field].encoder.dtype, columns[field][index])
                    for field in target.fields
                )
                for index in range(rows[target.input].size)
            ]
            ids = torch.tensor([state.groups.get(key, -1) for key in keys], dtype=torch.long)
            result[name] = {"values": ids, "valid": ids >= 0}
        elif isinstance(target, (LabelTarget, CandidateTarget, StructuredTarget)):
            candidate = (
                spec.inputs[target.input].source.name
                if isinstance(target, CandidateTarget)
                else None
            )
            output = {}
            for field, binding in state.bindings.items():
                raw = _raw_targets(
                    values,
                    target.source,
                    field=field if isinstance(target, StructuredTarget) else None,
                    candidate=candidate,
                )
                encoded = binding.encode_values(raw, model=model)
                if isinstance(target, CandidateTarget):
                    parent = rows[target.input]
                    encoded = _encoded_rows(
                        encoded, parent.boundaries, spec.inputs[target.input].layout
                    )
                    if isinstance(spec.inputs[target.input].layout, Packed):
                        encoded["offsets"] = torch.from_numpy(parent.boundaries.copy())
                    encoded["candidate_valid"] = layout_array(
                        torch.ones(parent.size, dtype=torch.bool),
                        parent.boundaries,
                        spec.inputs[target.input].layout,
                        fill=False,
                    )
                output[field] = encoded
            result[name] = output if isinstance(target, StructuredTarget) else output["value"]
        elif isinstance(target, AlignedEventTarget):
            parent = rows[target.input]
            output = {}
            derived_by_name = {item.name: item for item in target.derived}
            for field, binding in state.bindings.items():
                if field in derived_by_name:
                    derived = derived_by_name[field]
                    threshold = state.thresholds[field]
                    raw = parent.columns[derived.field].to_pylist()
                    labels = [
                        None
                        if value is None
                        else derived.above
                        if (
                            value > threshold
                            if derived.comparison == "greater_than"
                            else value >= threshold
                        )
                        else derived.below
                        for value in raw
                    ]
                    encoded = binding.encode_values(labels, model=model)
                else:
                    encoded = binding.encode_arrow(parent.columns[field], model=model)
                output[field] = _encoded_rows(
                    encoded, parent.boundaries, spec.inputs[target.input].layout
                )
            if isinstance(spec.inputs[target.input].layout, Packed):
                output["offsets"] = torch.from_numpy(parent.boundaries.copy())
            result[name] = output
    return result


def _binding_records(bindings):
    return [
        [field, binding.encoder.identity, binding.id_space] for field, binding in bindings.items()
    ]


def _input_record(prepared):
    if isinstance(prepared.spec.representation, Graph):
        return {
            "node_fields": {
                table: _binding_records(bindings) for table, bindings in prepared.state.items()
            }
        }
    result = {"bindings": _binding_records(prepared.state)}
    if prepared.composite is not None:
        result["composite"] = json_value(tuple(prepared.composite.items()))
    return result


class PreparedBatchPlan:
    """Frozen encoder state with repeatable selection, assembly, and delivery.

    The lower-level object behind a `Task`; most code uses the task instead.
    """

    def __init__(
        self,
        sample: SampleSpec,
        spec: BatchSpec,
        input_state: _InputState,
        target_encoders: Mapping[str, _TargetState],
        *,
        source: Any = None,
        contract: Any = None,
        schema_fingerprint=None,
        contract_fingerprint=None,
        inference=False,
    ):
        self.inference = inference
        # inference reads no row a target would read, and emits no targets, so neither
        # masking nor shifted next-token inputs apply
        if inference:
            sample = sample_without_targets(sample)
        self._emitted = replace(spec, targets={}) if inference else spec
        self.sample, self.spec = sample, spec
        self._input_state = input_state
        self._prepared_inputs = dict(input_state.prepared_inputs)
        self._target_encoders = dict(target_encoders)
        self._source, self._contract = source, contract
        lookup_factory = getattr(source, "string_lookup", None)
        if lookup_factory is not None:
            for encoder in input_state.encoders.values():
                encoder._cache_lookup(lookup_factory)
        self._model_sample = _model_retrieval_sample(sample, spec)
        self.contract_fingerprint = contract_fingerprint
        self._request_builder = (
            (lambda requests: source.build(sample, requests)) if source is not None else None
        )
        self._last_batch_profile = None
        self._length_shapes = {}
        self.encoders = MappingProxyType(
            {
                name: encoder
                for identity, encoder in input_state.encoders.items()
                for name in (
                    json.loads(identity)[1] if json.loads(identity)[0] == "shared" else identity,
                )
            }
        )
        self.spaces = MappingProxyType(dict(input_state.spaces))
        inputs = {}
        for name, value in self._prepared_inputs.items():
            if isinstance(value.spec.representation, Fields):
                summary = {
                    field: {
                        **binding.encoder.description,
                        "id_space": binding.id_space,
                    }
                    for field, binding in value.state.items()
                }
            else:
                summary = {
                    table: {
                        field: binding.encoder.description for field, binding in bindings.items()
                    }
                    for table, bindings in value.state.items()
                }
            inputs[name] = {"summary": summary}
        target_states = {
            name: {
                "bindings": _binding_records(value.bindings),
                "thresholds": dict(value.thresholds or {}),
                "groups": json_value(
                    tuple((key, index) for key, index in (value.groups or {}).items())
                ),
            }
            for name, value in self._target_encoders.items()
        }
        self.fitted_state = {
            "format_version": 2,
            "encoders": {
                identity: encoder.state_dict()
                for identity, encoder in sorted(input_state.encoders.items())
            },
            "spaces": {name: space.state_dict() for name, space in sorted(self.spaces.items())},
            "field_names": {
                name: [[list(identity), index] for identity, index in sorted(entries.items())]
                for name, entries in input_state.field_names.items()
            },
            "inputs": {name: _input_record(value) for name, value in self._prepared_inputs.items()},
            "targets": target_states,
        }
        self.manifest = {
            "format_version": 2,
            "schema_fingerprint": schema_fingerprint,
            "contract_fingerprint": contract_fingerprint,
            "inputs": inputs,
            "targets": {
                name: {
                    field: binding.encoder.description for field, binding in state.bindings.items()
                }
                for name, state in self._target_encoders.items()
            },
            "spaces": {
                name: {"size": value.size, "roles": dict(value.roles)}
                for name, value in self.spaces.items()
            },
            "fitted_state_fingerprint": _fingerprint(self.fitted_state),
        }
        self.manifest["artifact_fingerprint"] = _fingerprint(self.manifest)

    def output(self, *, name: str) -> OutputView:
        """Read-only view of an input's field encoders and output conversions."""
        if name not in self._prepared_inputs:
            raise ValueError(f"unknown output {name!r}")
        return OutputView(bindings=self._prepared_inputs[name].state)

    def target(self, *, name: str) -> OutputView:
        """Read-only view of an independently encoded target's fitted encodings."""
        if name not in self._target_encoders:
            raise ValueError(f"target {name!r} has no independent fitted encoder")
        return OutputView(bindings=self._target_encoders[name].bindings)

    def _fitted_input_state(self) -> _InputState:
        encoders, spaces, names = set(), set(), set()
        for prepared in self._prepared_inputs.values():
            representation = prepared.spec.representation
            groups = (
                ((representation, prepared.state),)
                if isinstance(representation, Fields)
                else tuple(
                    (group, prepared.state[table])
                    for table, group in representation.node_fields.items()
                )
            )
            for group, bindings in groups:
                encoders.update(binding.encoder.identity for binding in bindings.values())
                spaces.update(use.id_space for use in group.fields if use.id_space is not None)
                if group.field_names is not None:
                    names.add(group.field_names)
                if group.field_space is not None:
                    spaces.add(group.field_space)
                if isinstance(group.row, Composite) and group.row.id_space is not None:
                    spaces.add(group.row.id_space)
        for name in spaces:
            encoders.update(self.spaces[name].members)
        return _InputState(
            prepared_inputs=dict(self._prepared_inputs),
            encoders={
                name: value
                for name, value in self._input_state.encoders.items()
                if name in encoders
            },
            spaces={name: self.spaces[name] for name in spaces},
            field_names={name: self._input_state.field_names[name] for name in names},
        )

    def attach_source(self, source: Any) -> None:
        """Take ownership of ``source``, which `close` then closes."""
        self._owned_source = source

    def with_source(self, source: Any) -> PreparedBatchPlan:
        """This plan reading through ``source``, sharing every fitted value; this one is unchanged.

        Rebinding an already-built plan to new rows skips rebuilding and
        re-fingerprinting its fitted state.
        """

        plan = copy.copy(self)
        plan.__dict__.pop("_owned_source", None)
        plan._source = source
        plan._request_builder = lambda requests: source.build(plan.sample, requests)
        plan._last_batch_profile = None
        plan._length_shapes = {}
        return plan

    def close(self) -> None:
        """Close the attached source, if any."""
        if getattr(self, "_owned_source", None) is not None:
            self._owned_source.close()
            self._owned_source = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    @property
    def last_batch_profile(self):
        """Timing profile of the most recent batch, or ``None``."""
        return self._last_batch_profile

    def bound_contract(self):
        """Pin input encoders; unrelated learned targets may still require fitting.

        Targets are left as declared, so a fine-tune of the bound contract fits
        its own labels. A written artifact restores target state from
        ``fitted-state.json`` instead.
        """

        if self._contract is None:
            raise ValueError("bound_contract requires a complete DataLoaderSpec")
        inputs = {}
        for name, value in self.spec.inputs.items():
            if isinstance(value.representation, Graph):
                fields_ = {
                    table: replace(
                        group,
                        fields=tuple(
                            replace(
                                use,
                                encoding=self._prepared_inputs[name]
                                .state[table][use.field]
                                .encoder.bound_config(),
                            )
                            if not isinstance(use.encoding, UseEncoding)
                            else use
                            for use in group.fields
                        ),
                    )
                    for table, group in value.representation.node_fields.items()
                }
                representation = replace(value.representation, node_fields=fields_)
            else:
                group = value.representation
                fields_ = tuple(
                    replace(
                        use,
                        encoding=self._prepared_inputs[name]
                        .state[use.field]
                        .encoder.bound_config(),
                    )
                    if not isinstance(use.encoding, UseEncoding)
                    else use
                    for use in group.fields
                )
                row = group.row
                if isinstance(row, Composite):
                    row = replace(
                        row, vocabulary=tuple(self._prepared_inputs[name].composite.items())
                    )
                representation = replace(group, fields=fields_, row=row)
            inputs[name] = replace(value, representation=representation)
        dependency_encoders = self._fitted_input_state().encoders
        encodings = {
            name: dependency_encoders[_identity("shared", name)].bound_config()
            if _identity("shared", name) in dependency_encoders
            else value
            for name, value in self.spec.encodings.items()
        }
        targets = dict(self.spec.targets)
        for name, target in self.spec.targets.items():
            if name not in self._target_encoders:
                continue
            bindings = self._target_encoders[name].bindings

            def bound(use):
                encoder = bindings[use.field].encoder
                return (
                    replace(use, encoding=encoder.bound_config())
                    if encoder.identity in dependency_encoders
                    and not isinstance(use.encoding, UseEncoding)
                    else use
                )

            if isinstance(target, (StructuredTarget, AlignedEventTarget)):
                targets[name] = replace(target, fields=tuple(bound(use) for use in target.fields))
            elif isinstance(target, (LabelTarget, CandidateTarget)):
                use = bound(Encode(field="value", encoding=target.encoding))
                targets[name] = replace(target, encoding=use.encoding)
        return replace(
            self._contract,
            batch=replace(self.spec, inputs=inputs, encodings=encodings, targets=targets),
        )

    def _validate_model_contract(self) -> None:
        if self._source is None or not callable(getattr(self._source, "build_model", None)):
            raise ValueError("this source does not implement output_format='model'")
        if self.sample.dated_entity_values:
            raise ValueError(
                "dated entity values require inspectable output; "
                "the typed model source does not expose their occurrences"
            )
        for name, value in self._prepared_inputs.items():
            if isinstance(value.spec.source, CandidateSource):
                raise ValueError(
                    f"input {name!r}: candidate model output is not supported; use inspectable"
                )
            bindings = (
                value.state.values()
                if isinstance(value.spec.representation, Fields)
                else (
                    binding
                    for attributes in value.state.values()
                    for binding in attributes.values()
                )
            )
            for binding in bindings:
                if (
                    isinstance(binding, OutputBinding)
                    and isinstance(binding.encoder.config, Raw)
                    and binding.encoder.dtype.name
                    in {"string", "bytes", "date32", "timestamp", "decimal128", "decimal256"}
                ):
                    raise ValueError(
                        f"input {name!r}: Raw {binding.encoder.dtype.name} requires "
                        "inspectable output or an explicit encoder"
                    )
        for name, state in () if self.inference else self._target_encoders.items():
            for binding in state.bindings.values():
                if isinstance(binding.encoder.config, Raw) and binding.encoder.dtype.name in {
                    "string",
                    "bytes",
                    "date32",
                    "timestamp",
                    "decimal128",
                    "decimal256",
                }:
                    raise ValueError(
                        f"target {name!r}: selected Raw dtype requires "
                        "inspectable output or an explicit encoder"
                    )

    def _collate(self, values: Any, *, epoch: int, seed: int, model=False, requests=()) -> Batch:
        declarations = {
            name: value.declarations
            for name, value in self._prepared_inputs.items()
            if isinstance(value.spec.representation, Fields)
        }
        rows = _rows_for_inputs(self.spec, declarations, values, model=model)
        for name, prepared in self._prepared_inputs.items():
            if isinstance(prepared.spec.source, OccurrencesOf):
                rows[name] = _occurrence_rows(name, self._input_state, rows, values)
        entity_ids = (
            values.entity_ids if model else tuple(_history(value).entity_id for value in values)
        )
        result, targets = {}, {}
        for name, prepared in self._prepared_inputs.items():
            if isinstance(prepared.spec.representation, Graph):
                graph = SimpleNamespace(
                    source=prepared.spec.source.name,
                    node_fields=prepared.spec.representation.node_fields,
                )
                if model and getattr(values, "runtime_batch", None) is not None:
                    result[name] = self._source.sample_graph_batch(
                        self.sample, graph, prepared.state, values.runtime_batch
                    )
                    continue
                roots = (
                    [
                        GraphRoot(
                            request.sample_id,
                            request.entity_id,
                            cutoff_to_epoch(request.as_of_time),
                        )
                        for request in requests
                    ]
                    if model
                    else [
                        GraphRoot(
                            _sample_id(value),
                            _history(value).entity_id,
                            _history(value).evaluation_time,
                        )
                        for value in values
                    ]
                )
                result[name] = self._source.sample_graph(
                    self.sample, graph, prepared.state, roots, seed, epoch
                )
            elif isinstance(prepared.spec.representation.row, Columns):
                result[name] = _column_output(
                    prepared, rows[name], model=model, state=self._input_state
                )
            else:
                result[name], produced = _token_output(
                    name,
                    prepared,
                    rows[name],
                    self._input_state,
                    self._emitted,
                    entity_ids,
                    model=model,
                    packer=getattr(self._source, "pack_token_stream", None) if model else None,
                    seed=seed,
                    epoch=epoch,
                )
                targets.update(produced)
        if not self.inference:
            targets.update(
                _target_outputs(
                    self.spec,
                    self._target_encoders,
                    self._prepared_inputs,
                    rows,
                    values,
                    model=model,
                )
            )
        if model:
            metadata = {
                "sample_ids": values.sample_ids,
                "entity_ids": entity_ids,
                "as_of_time": torch.tensor(values.evaluation_times),
                "splits": values.splits,
                "request_metadata": values.request_metadata,
            }
        else:
            metadata = {
                "sample_ids": tuple(_sample_id(value) for value in values),
                "entity_ids": entity_ids,
                "as_of_time": torch.tensor([_history(value).evaluation_time for value in values]),
                "splits": tuple(
                    value.metadata.get("split") if isinstance(value, BuiltSample) else None
                    for value in values
                ),
                "request_metadata": tuple(
                    dict(value.metadata) if isinstance(value, BuiltSample) else {}
                    for value in values
                ),
            }
        metadata.update(
            artifact_fingerprint=self.manifest["artifact_fingerprint"],
            contract_fingerprint=self.contract_fingerprint,
        )
        if all(type(value) is int and -(2**63) <= value < 2**63 for value in entity_ids):
            metadata["entity_ids"] = torch.tensor(entity_ids, dtype=torch.long)
        if not model and self.sample.candidate_sets:
            metadata["candidate_entity_ids"] = {
                name: tuple(
                    tuple(
                        candidate.history.entity_id for candidate in value.candidates.get(name, ())
                    )
                    for value in values
                )
                for name in self.sample.candidate_sets
            }
        return Batch(
            inputs=kernel_metadata(result), targets=kernel_metadata(targets), metadata=metadata
        )

    def _input_length(self, value: HistoryInput, input_name: str | None) -> int:
        if input_name is None:
            return 1
        return self._measure_values([value], input_name)[0]

    def _measure_values(self, values, input_name, *, model=False):
        prepared = self._prepared_inputs[input_name]
        declarations = {
            name: item.declarations
            for name, item in self._prepared_inputs.items()
            if isinstance(item.spec.representation, Fields)
        }
        rows = _rows_for_inputs(self.spec, declarations, values, model=model)
        if isinstance(prepared.spec.source, OccurrencesOf):
            rows[input_name] = _occurrence_rows(input_name, self._input_state, rows, values)
        if isinstance(prepared.spec.representation.row, Columns):
            return dict(enumerate(int(length) for length in np.diff(rows[input_name].boundaries)))
        tokens = _token_rows(input_name, prepared, rows[input_name], self._input_state, model=model)
        if isinstance(prepared.spec.representation.history, Flatten):
            group = prepared.spec.representation
            space_name = (
                group.row.id_space if isinstance(group.row, Composite) else group.fields[0].id_space
            )
            tokens = flatten_rows(
                tokens,
                rows[input_name].boundaries,
                group.history,
                self.spaces[space_name].roles,
                packer=getattr(self._source, "pack_token_stream", None) if model else None,
            )
            return dict(enumerate(int(length) for length in np.diff(tokens.boundaries)))
        lengths = np.diff(tokens.boundaries)
        if isinstance(prepared.spec.source, EntitySource):
            return dict(enumerate(int(length) for length in lengths))
        bounds = rows[input_name].boundaries
        prefix = offsets(lengths)
        self._length_shapes = {
            index: (int(stop - start), int(lengths[start:stop].max(initial=0)))
            for index, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
        }
        return dict(enumerate(int(length) for length in prefix[bounds[1:]] - prefix[bounds[:-1]]))

    def _batch_cost(self, indices, lengths, name):
        if not indices:
            return 0
        layout = self.spec.inputs[name].layout if name is not None else Padded()
        if isinstance(layout, Nested):
            if isinstance(layout.fields, Packed):
                return sum(lengths[index] for index in indices)
            return sum(self._length_shapes[index][0] for index in indices) * max(
                self._length_shapes[index][1] for index in indices
            )
        if isinstance(layout, Packed):
            return sum(lengths[index] for index in indices)
        return len(indices) * max(lengths[index] for index in indices)

    def _compose_batches(self, order, lengths, policy, max_samples):
        batches, current = [], []

        def cost(indices):
            return self._batch_cost(indices, lengths, policy.length_input)

        for index in order:
            if policy.max_tokens is not None and cost([index]) > policy.max_tokens:
                raise ValueError("one sample exceeds Batching.max_tokens")
            if current and (
                max_samples is not None
                and len(current) >= max_samples
                or policy.max_tokens is not None
                and cost([*current, index]) > policy.max_tokens
            ):
                batches.append(current)
                current = []
            current.append(index)
        if current:
            batches.append(current)
        return batches

    def batches(
        self,
        values,
        *,
        batch_size=None,
        epoch=0,
        seed=0,
        shuffle=False,
        rank=0,
        world_size=1,
        drop_last=None,
        output_format: Literal["inspectable", "model"] = "inspectable",
    ):
        """Encode ``values`` into batches under the frozen state; see `Task.batches`."""
        if output_format not in {"inspectable", "model"}:
            raise ValueError("output_format must be inspectable or model")
        model = output_format == "model"
        if model:
            self._validate_model_contract()
        policy = self.spec.batching
        maximum = batch_size if batch_size is not None else policy.max_samples
        if maximum is None and policy.max_tokens is None:
            raise ValueError("set max_samples/max_tokens or pass batch_size")
        if maximum is not None and maximum <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must identify one member of a positive world size")
        requests = bool(values and isinstance(values[0], SampleRequest))
        if model and not requests:
            raise ValueError("model output requires SampleRequest values")
        order = list(
            shard_positions(
                len(values),
                seed=seed,
                epoch=epoch,
                shuffle=shuffle,
                rank=rank,
                world_size=world_size,
            )
        )
        built = self._request_builder(values) if requests and not model else values
        if model and (policy.max_tokens is not None or policy.bucket_by_length):
            length_spec = self.spec.inputs[policy.length_input]
            native_counter = getattr(self._source, "model_lengths", None)
            if isinstance(length_spec.source, EventsOf):
                parent = self.spec.inputs[length_spec.source.input]
                row = parent.representation.row
                if (
                    isinstance(row, Columns)
                    or isinstance(row, Tokens)
                    and row.missing in {"omit", "emit"}
                    or isinstance(row, Composite)
                    and not row.required_fields
                ):
                    length_spec = replace(length_spec, source=parent.source)
            if (
                callable(native_counter)
                and isinstance(length_spec.source, EventSource)
                and isinstance(length_spec.representation.row, Columns)
            ):
                measured = native_counter(self.sample, length_spec, values)
                lengths = {index: int(measured[index]) for index in order}
            else:
                lengths = self._model_lengths(values, policy.length_input)
        else:
            lengths = (
                self._measure_values(built, policy.length_input)
                if not model and policy.length_input is not None
                else {index: 1 for index in order}
            )
        if policy.bucket_by_length:
            order.sort(key=lambda index: lengths[index])
        chunks = self._compose_batches(order, lengths, policy, maximum)
        if (
            (policy.drop_last if drop_last is None else drop_last)
            and maximum is not None
            and chunks
            and len(chunks[-1]) < maximum
        ):
            chunks = chunks[:-1]
        for chunk in chunks:
            started = time.perf_counter_ns()
            if model:
                selected_requests = [values[index] for index in chunk]
                builder = getattr(self._source, "build_model_ordered", None)
                orders = {}
                for value in self.spec.inputs.values():
                    if isinstance(value.source, EventSource):
                        for event in value.source.events:
                            orders.setdefault(event, set()).add(value.source.order)
                source_values = (
                    builder(
                        self._model_sample,
                        selected_requests,
                        event_orders={
                            name: tuple(sorted(directions)) for name, directions in orders.items()
                        },
                    )
                    if callable(builder)
                    else self._source.build_model(self._model_sample, selected_requests)
                )
                batch = self._collate(
                    source_values, epoch=epoch, seed=seed, model=True, requests=selected_requests
                )
            else:
                batch = self._collate([built[index] for index in chunk], epoch=epoch, seed=seed)
            self._last_batch_profile = {
                "output_format": output_format,
                "samples": len(chunk),
                "model_assembly_ns": time.perf_counter_ns() - started,
            }
            yield batch

    def _model_lengths(self, values, input_name):
        source_values = self._source.build_model(self.sample, values)
        return self._measure_values(source_values, input_name, model=True)


def prepare_batch_spec(
    sample: SampleSpec, spec: BatchSpec, fitting_values: Sequence[HistoryInput], *, contract=None
) -> PreparedBatchPlan:
    """Fit a batch spec's encoders on already built samples and return the frozen plan.

    The lower-level path behind `Plan.materialize`.

    :raises ValueError: when ``fitting_values`` is empty.
    """
    if not fitting_values:
        raise ValueError("at least one fitting sample is required")
    fitting_spec = _fitting_batch_spec(spec)
    declarations = _declarations(sample, fitting_spec, contract, fitting_values)
    state = _fit_inputs(sample, fitting_spec, declarations, lambda _: fitting_values, contract)
    state, targets = _fit_targets(
        sample, fitting_spec, declarations, lambda _: fitting_values, state, contract
    )
    state = replace(
        state,
        prepared_inputs={
            name: replace(value, spec=spec.inputs[name])
            for name, value in state.prepared_inputs.items()
        },
    )
    return PreparedBatchPlan(sample, spec, state, targets, contract=contract)


def _model_retrieval_sample(sample: SampleSpec, spec: BatchSpec) -> SampleSpec:
    """Push unambiguous retrieval bounds without narrowing raw samples or fitting."""

    sources = [
        value.source
        for value in spec.inputs.values()
        if isinstance(value.source, EventSource) and value.source.collection == "history"
    ]
    consumers = defaultdict(list)
    for source in sources:
        for name in source.events:
            consumers[name].append(source)
    bounded = {}
    for name, uses in consumers.items():
        event = sample.events[name]
        if event.limit is not None:
            continue  # An explicit event limit defines membership before model capping.
        if any(type(use.max_events) is not int for use in uses):
            continue
        if len({use.keep for use in uses}) != 1:
            continue
        limit = max(use.max_events for use in uses)
        if limit > np.iinfo(np.intp).max:
            continue
        # The global first/last K cannot contain an event outside the first/last
        # K of its component. Bound each component before native feature decoding.
        bounded[name] = replace(
            event, limit=limit, order="ascending" if uses[0].keep == "first" else "descending"
        )
    return replace(sample, events={**sample.events, **bounded}) if bounded else sample


def _fitting_batch_spec(spec: BatchSpec) -> BatchSpec:
    return replace(
        spec,
        inputs={
            name: replace(value, source=replace(value.source, order="ascending"))
            if isinstance(value.source, EventSource)
            else value
            for name, value in spec.inputs.items()
        },
    )


def fit_plan(
    sample: SampleSpec,
    spec: BatchSpec,
    fitting_requests: Sequence[SampleRequest],
    source: Any,
    *,
    schema_fingerprint=None,
    contract_fingerprint=None,
    contract=None,
    input_state=None,
    target_encoders=None,
    inference=False,
    pinned=MappingProxyType({}),
) -> PreparedBatchPlan:
    """Fit one plan by reading raw samples through a source.

    ``input_state`` and ``target_encoders`` reuse previously fitted state
    instead of fitting it; whatever is not supplied is fitted on
    ``fitting_requests``, except the input encoders in ``pinned``. ``inference``
    requires both, and builds samples without their row-computed targets so no
    target row is read.

    Samples are built in chunks of ``_FIT_CHUNK`` requests, only for encodings
    that learn from values, and only their observations are kept: inputs build
    without their targets, targets build alone unless they read input rows, and
    supplied labels are read from the requests.
    """

    needs_fitting = input_state is None or target_encoders is None
    if inference and needs_fitting:
        raise ValueError("inference never fits: supply complete input state and target state")
    if needs_fitting and not fitting_requests:
        raise ValueError("at least one fitting request is required")
    if input_state is not None and target_encoders is None:
        _check_reused_target_spaces(spec, input_state)
    model = (
        callable(getattr(source, "build_model", None))
        and not sample.dated_entity_values
        and all(
            not isinstance(value.source, (CandidateSource, NeighborhoodSource))
            for value in spec.inputs.values()
        )
    )

    def values_for(sample_spec):
        def read(requests):
            requests = [
                replace(request, targets=_normalize_supplied_targets(sample, request.targets))
                for request in requests
            ]
            if sample_spec is None:
                return requests
            return (source.build_model if model else source.build)(sample_spec, requests)

        return _Mapped(read, fitting_requests)

    reused = input_state is not None
    fitting_spec = _fitting_batch_spec(spec)
    declarations = _declarations(sample, fitting_spec, contract, ()) if needs_fitting else {}
    if input_state is None:
        input_state = _fit_inputs(
            sample,
            fitting_spec,
            declarations,
            values_for,
            contract,
            model=model,
            source=source,
            pinned=pinned,
        )
    else:
        input_state = replace(
            input_state,
            prepared_inputs={
                name: replace(value, spec=spec.inputs[name])
                for name, value in input_state.prepared_inputs.items()
            },
        )
    if target_encoders is None:
        input_state, target_encoders = _fit_targets(
            sample,
            fitting_spec,
            declarations,
            values_for,
            input_state,
            contract,
            model=model,
            reused=reused,
        )
    else:
        encoders, spaces = dict(input_state.encoders), dict(input_state.spaces)
        for name, target in target_encoders.items():
            declaration = spec.targets[name]
            uses = (
                declaration.fields
                if isinstance(declaration, (StructuredTarget, AlignedEventTarget))
                else (
                    Encode(
                        field="value", encoding=declaration.encoding, id_space=declaration.id_space
                    ),
                )
                if isinstance(declaration, (LabelTarget, CandidateTarget))
                else ()
            )
            for field, binding in target.bindings.items():
                encoders[binding.encoder.identity] = binding.encoder
                destination = next((use.id_space for use in uses if use.field == field), None)
                if destination is not None:
                    spaces[destination] = binding.destination
        empty_spaces = _allocate_spaces(spec, {}, {})
        spaces = {**empty_spaces, **spaces}
        input_state = replace(input_state, encoders=encoders, spaces=spaces)
    input_state = replace(
        input_state,
        prepared_inputs={
            name: replace(value, spec=spec.inputs[name])
            for name, value in input_state.prepared_inputs.items()
        },
    )
    return PreparedBatchPlan(
        sample,
        spec,
        input_state,
        target_encoders,
        source=source,
        contract=contract,
        schema_fingerprint=schema_fingerprint,
        contract_fingerprint=contract_fingerprint,
        inference=inference,
    )


def restore_fitted_state(
    contract: Any, serialized: Mapping[str, Any]
) -> tuple[_InputState, dict[str, _TargetState], dict[str, Any], Any]:
    """Rebuild a contract's fitted state from its written form, reading no data.

    Every encoder is rebuilt from its written state under the encoding the
    contract declares for it, and every ID space, field-name map and binding
    as written. The result is refused unless it serializes back to
    ``serialized`` exactly, so state the contract cannot carry is never
    silently refitted or dropped. A state without ``targets`` is the input half.

    Returns the input state (what the inputs depend on), the target state, the
    manifest the state gives, and the bound contract: the contract with that
    state's encodings pinned.
    """

    spec = contract.batch
    if serialized.get("format_version") != 2:
        raise ValueError(
            f"fitted-state format_version {serialized.get('format_version')!r}; "
            "this version reads 2"
        )
    configs = _declared_encoders(spec)
    encoders = {
        identity: _restored_encoder(
            identity, configs.get(identity) or _described_encoding(stored), stored
        )
        for identity, stored in serialized["encoders"].items()
    }
    spaces = {
        name: FittedSpace(
            roles={role: int(index) for role, index in stored["roles"].items()},
            members={
                identity: {int(local): int(final) for local, final in pairs}
                for identity, pairs in stored["members"].items()
            },
            size=int(stored["size"]),
        )
        for name, stored in serialized["spaces"].items()
    }
    field_names = {
        name: {tuple(identity): int(index) for identity, index in entries}
        for name, entries in serialized["field_names"].items()
    }

    def binding(record: Sequence[Any]) -> tuple[str, OutputBinding]:
        field, identity, id_space = record
        return field, OutputBinding(
            encoder=encoders[identity],
            destination=spaces[id_space] if id_space is not None else None,
            id_space=id_space,
        )

    declarations = _declarations(contract.sample, spec, contract, [])
    stored_inputs = serialized["inputs"]
    prepared = {}
    for name, value in spec.inputs.items():
        if name not in stored_inputs:
            raise ValueError(_not_reproduced([f"inputs.{name}"]))
        record = stored_inputs[name]
        if isinstance(value.representation, Graph):
            prepared[name] = _PreparedInput(
                spec=value,
                declarations={},
                state={
                    table: dict(binding(item) for item in items)
                    for table, items in record["node_fields"].items()
                },
            )
            continue
        composite = record.get("composite")
        prepared[name] = _PreparedInput(
            spec=value,
            declarations=declarations[name],
            state=dict(binding(item) for item in record["bindings"]),
            composite=None if composite is None else dict(unjson_value(composite)),
        )
    stored_targets = serialized.get("targets", {})
    unknown = sorted(set(stored_targets) - set(spec.targets))
    if unknown:
        raise ValueError(f"target state for {unknown}, which this contract does not declare")
    targets = {
        name: _TargetState(
            bindings=dict(binding(item) for item in stored["bindings"]),
            thresholds=(
                {key: float(value) for key, value in stored["thresholds"].items()}
                if isinstance(spec.targets[name], AlignedEventTarget)
                else None
            ),
            groups=(
                dict(unjson_value(stored["groups"]))
                if isinstance(spec.targets[name], MatchingGroupTarget)
                else None
            ),
        )
        for name, stored in stored_targets.items()
    }
    _check_spaces(spaces, encoders, prepared, field_names, targets)
    restored = PreparedBatchPlan(
        contract.sample,
        spec,
        _InputState(
            prepared_inputs=prepared, encoders=encoders, spaces=spaces, field_names=field_names
        ),
        targets,
        contract=contract,
        schema_fingerprint=_fingerprint(contract.schema),
        contract_fingerprint=contract.fingerprint,
    )
    expected = json.loads(json.dumps({"targets": {}, **serialized}, sort_keys=True))
    actual = json.loads(json.dumps(restored.fitted_state, sort_keys=True))
    if actual != expected:
        raise ValueError(_not_reproduced(_state_differences(expected, actual)))
    return restored._fitted_input_state(), targets, restored.manifest, restored.bound_contract()


#: What a written fitted-state file keeps beside the fitted values, outside every fingerprint.
WRITTEN_BESIDE_STATE = ("fit", "fitted_for")


def pin_fitted_state(contract: Any, serialized: Mapping[str, Any]) -> Any:
    """The contract with the encodings ``serialized`` records pinned.

    What `PreparedBatchPlan.bound_contract` pins, read from the written state
    rather than the fitted objects; refused as `restore_fitted_state` refuses.
    ``serialized`` may be a file this version's `Artifact.save` wrote, with what it keeps
    beside the state.
    """

    state = {key: value for key, value in serialized.items() if key not in WRITTEN_BESIDE_STATE}
    return restore_fitted_state(contract, state)[3]


def _not_reproduced(differences: Sequence[str]) -> str:
    return (
        "the contract's encodings do not reproduce the written fitted state "
        f"(differs at {list(differences)}); it is inconsistent, or was written by a version "
        "that cannot reload it"
    )


def _check_spaces(
    spaces: Mapping[str, FittedSpace],
    encoders: Mapping[str, FittedEncoder],
    prepared: Mapping[str, _PreparedInput],
    field_names: Mapping[str, Mapping[tuple[str, ...], int]],
    targets: Mapping[str, _TargetState],
) -> None:
    """Refuse ID spaces that do not number exactly the IDs their members produce."""

    local = {identity: set(encoder.local_ids) for identity, encoder in encoders.items()}
    for name, value in prepared.items():
        if value.composite is not None:
            local[_identity("composite", name)] = {0, 1, *value.composite.values()}
    for name, entries in field_names.items():
        local[_identity("fields", name)] = {0, *entries.values()}
    for space_name, space in spaces.items():
        for identity, mapping in space.members.items():
            if set(mapping) != local.get(identity):
                raise ValueError(
                    f"ID space {space_name!r} does not number the IDs of {identity}; "
                    + _not_reproduced([f"spaces.{space_name}"])
                )
    bindings = [
        binding
        for value in prepared.values()
        for group in (
            value.state.values() if isinstance(value.spec.representation, Graph) else (value.state,)
        )
        for binding in group.values()
    ]
    bindings += [binding for target in targets.values() for binding in target.bindings.values()]
    for binding in bindings:
        if binding.id_space is not None and binding.encoder.identity not in (
            spaces[binding.id_space].members
        ):
            raise ValueError(
                f"ID space {binding.id_space!r} has no member {binding.encoder.identity}; "
                + _not_reproduced([f"spaces.{binding.id_space}"])
            )


def _declared_encoders(spec: BatchSpec) -> dict[str, Any]:
    """Each encoder identity fitting gives this contract, with its declared encoding."""

    configs: dict[str, Any] = {
        _identity("shared", name): config for name, config in spec.encodings.items()
    }
    for name, value in spec.inputs.items():
        if isinstance(value.representation, Graph):
            for table, group in value.representation.node_fields.items():
                for use in group.fields:
                    if not isinstance(use.encoding, UseEncoding):
                        configs[_identity("inputs", name, "graph", table, use.field)] = use.encoding
            continue
        for use in value.representation.fields:
            if not isinstance(use.encoding, UseEncoding):
                configs[_encoder_identity(name, use)] = use.encoding
    for name, target in spec.targets.items():
        if isinstance(target, (StructuredTarget, AlignedEventTarget)):
            uses = list(target.fields)
            if isinstance(target, AlignedEventTarget):
                uses += [Encode(field=item.name, encoding=item.encoding) for item in target.derived]
        elif isinstance(target, (LabelTarget, CandidateTarget)):
            uses = [Encode(field="value", encoding=target.encoding)]
        else:
            continue
        for use in uses:
            if not isinstance(use.encoding, UseEncoding):
                configs[_identity("targets", name, use.field)] = use.encoding
    return configs


def _described_encoding(stored: Mapping[str, Any]) -> Any:
    """The encoding a written encoder describes, for a member this contract does not declare.

    Input state carries every member of the ID spaces its inputs use, which can include
    another contract's target; only what encoding it reads is rebuilt.
    """

    kind, roles = stored.get("kind"), stored.get("roles", {})
    if kind == "vocabulary":
        return Vocabulary(missing_id=roles["missing"], unknown_id=roles["unknown"])
    if kind == "buckets":
        return Buckets(
            boundaries=tuple(stored["boundaries"]), separate_zero=stored["separate_zero"]
        )
    if kind == "zscore":
        return ZScore(dtype=stored["output_dtype"])
    if kind == "text":
        return Text(
            state=stored["tokenizer_json"],
            missing_id=roles["missing"],
            unknown_id=roles["unknown"],
        )
    if kind == "raw":
        return Raw()
    raise ValueError(f"no encoding is written as {kind!r}")


def _restored_encoder(identity: str, config: Any, stored: Mapping[str, Any]) -> FittedEncoder:
    """One encoder rebuilt from its written state under its declared encoding."""

    kind = type(config).__name__.lower()
    if stored.get("kind") != kind:
        raise ValueError(
            f"encoder {identity}: written {stored.get('kind')!r} state does not match the "
            f"declared {type(config).__name__}"
        )
    dtype = DType(**stored["dtype"])
    if isinstance(config, Vocabulary):
        return FittedEncoder(
            identity=identity,
            config=config,
            dtype=dtype,
            entries=tuple(
                (dtype.normalize(unjson_value(value)), int(index))
                for value, index in stored["entries"]
            ),
        )
    if isinstance(config, Buckets):
        return FittedEncoder(
            identity=identity,
            config=config,
            dtype=dtype,
            boundaries=tuple(float(value) for value in stored["boundaries"]),
        )
    if isinstance(config, ZScore):
        return FittedEncoder(
            identity=identity,
            config=config,
            dtype=dtype,
            mean=float(stored["mean"]),
            scale=float(stored["scale"]),
        )
    if isinstance(config, Text):
        return FittedEncoder(
            identity=identity, config=config, dtype=dtype, tokenizer_json=stored["tokenizer_json"]
        )
    return FittedEncoder(identity=identity, config=config, dtype=dtype)


def _state_differences(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    paths = []
    for section in sorted(expected.keys() | actual.keys()):
        left, right = expected.get(section), actual.get(section)
        if left == right:
            continue
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            paths.extend(
                f"{section}.{name}"
                for name in sorted(left.keys() | right.keys())
                if left.get(name) != right.get(name)
            )
        else:
            paths.append(section)
    return paths


@dataclass(frozen=True)
class PreviewResult:
    """What `preview` returns.

    :param samples: the raw built samples, as JSON-compatible mappings.
    :param batch: the encoded batch.
    :param encoding_summary: the fitted encodings the batch used.
    """

    samples: tuple[Mapping[str, Any], ...]
    batch: Batch
    encoding_summary: Mapping[str, Any]

    def to_dict(self):
        """JSON-compatible form of the whole result."""
        return _jsonable(
            {
                "samples": self.samples,
                "batch": self.batch,
                "encoding_summary": self.encoding_summary,
            }
        )

    def __str__(self):
        return f"Preview: {len(self.samples)} sample(s); inputs: {', '.join(self.batch.inputs)}"


def _group_values(values):
    grouped = {}
    for value in values:
        grouped.setdefault(value.key, []).append(value.value)
    return {name: items[0] if len(items) == 1 else tuple(items) for name, items in grouped.items()}


def _sample_preview(value: BuiltSample):
    history = value.history
    events = {}
    for event in history.events:
        events.setdefault(event.source, []).append(
            {"timestamp": event.timestamp, "values": _group_values(event.fields)}
        )
    dated = {}
    for field in history.entity.dated_values:
        dated.setdefault(field.key, []).append({"timestamp": field.timestamp, "value": field.value})
    return {
        "sample_id": value.sample_id,
        "as_of_time": history.evaluation_time,
        "entity": {
            "id": history.entity_id,
            "type": history.entity_type,
            "values": _group_values(history.entity.values),
            "dated_values": dated,
        },
        "events": events,
        "current_events": {
            name: tuple(
                {"timestamp": event.timestamp, "values": _group_values(event.fields)}
                for event in events
            )
            for name, events in value.current_events.items()
        },
        "targets": _targets(value),
        "metadata": dict(value.metadata),
        "candidates": {
            name: tuple(_sample_preview(item) for item in items)
            for name, items in value.candidates.items()
        },
    }


def preview_plan(
    plan: PreparedBatchPlan, requests: Sequence[SampleRequest], *, seed=0
) -> PreviewResult:
    if plan._request_builder is None:
        raise ValueError("preview requires a plan bound to a relational source")
    samples = plan._request_builder(requests)
    return PreviewResult(
        samples=tuple(_sample_preview(value) for value in samples),
        batch=plan._collate(samples, epoch=0, seed=seed),
        encoding_summary=plan.manifest["inputs"],
    )
