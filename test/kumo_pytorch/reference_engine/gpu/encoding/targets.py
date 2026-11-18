# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Labels: sample targets computed or supplied for each anchor, encoded beside the inputs, on the GPU.

Kumo PyTorch's G-TARGET-001 rules:

- An ``ExternalTarget`` is supplied by the caller: an anchor table's target
  column, or a request's ``targets`` (checked by upstream's own
  ``_normalize_supplied_targets``). A label not supplied is missing.
- An ``Aggregate`` or ``Compare`` target is computed as an entity value is,
  over its window after the cutoff.
- A ``RowTarget`` is the first (or last) row its rows select, ordered by time
  and then by printed primary key, as events are. Its fields are worked out with
  that row as an event (``EventTimeDelta``: the cutoff minus the row's time). An
  anchor with no such row has every field missing.
- A ``LabelTarget`` encodes one sample target, and a ``StructuredTarget`` each
  field it names of a ``RowTarget``. Their encoders are fitted on the fit
  split's labels, NULLs excluded. A shared encoder (``UseEncoding``) that an
  input also uses keeps the input's fit and never sees a label. With
  ``id_space``, a target's encoder joins that space.
- An ``AlignedEventTarget`` encodes fields of an event input's own events,
  withheld from the input (read as the input's other fields are), and lays them
  out as the input is. A ``ThresholdEventLabel`` labels each event ``above``
  where its field passes the threshold (given, or the median of the fit split's
  values; non-finite values are refused) and ``below`` elsewhere, NULL staying
  missing. Integers are compared with the threshold exactly.
- A ``MatchingGroupTarget`` numbers the distinct tuples of an entity input's
  typed field values from 0, in canonical typed order (`test.kumo_pytorch.reference_engine.gpu.encoding.composite`); an
  unseen tuple is -1, not valid.
- A batch carries them in ``targets`` by target name, in name order:
  ``{"values", "valid"}`` per sample, one such pair per field of a structured
  or aligned target (an aligned one laid out per event, with its input's
  offsets when packed).

Only the sample targets a batch target reads are computed.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import Values
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        CompositeVocabulary,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        GpuEncoder,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler

    from kumo.pytorch.spec import DataLoaderSpec

__all__ = ["TargetState", "aligned_reads", "encode_targets", "fit_targets", "sample_targets", "target_identity",
           "target_uses"]


def target_identity(name: str, use: Any) -> str:
    """A target field's encoder identity, spelled as upstream spells it (a shared encoder by its name)."""

    from kumo.pytorch.contract.batch.encodings import UseEncoding

    parts = ("shared", use.encoding.name) if isinstance(use.encoding, UseEncoding) else ("targets", name, use.field)
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def target_uses(contract: DataLoaderSpec) -> list[tuple[str, Any, Any]]:
    """Every encoded target field as (target name, target, ``Encode``), targets by name, fields in order: a
    ``LabelTarget`` is one field named ``"value"``; an ``AlignedEventTarget`` its fields, then its derived labels
    by name."""

    from kumo.pytorch.contract.batch import (
        AlignedEventTarget,
        Encode,
        LabelTarget,
        StructuredTarget,
    )

    uses = []
    for name in sorted(contract.batch.targets):
        target = contract.batch.targets[name]
        if isinstance(target, LabelTarget):
            uses.append((name, target, Encode(field="value", encoding=target.encoding, id_space=target.id_space)))
        elif isinstance(target, StructuredTarget):
            uses.extend((name, target, use) for use in target.fields)
        elif isinstance(target, AlignedEventTarget):
            uses.extend((name, target, use) for use in target.fields)
            uses.extend((name, target, Encode(field=label.name, encoding=label.encoding)) for label in target.derived)
    return uses


def aligned_reads(contract: DataLoaderSpec) -> dict[str, tuple[str, ...]]:
    """The fields aligned targets read from each event input's rows, besides the input's own: their fields
    and their derived labels' source fields."""

    from kumo.pytorch.contract.batch import AlignedEventTarget

    reads: dict[str, tuple[str, ...]] = {}
    for target in contract.batch.targets.values():
        if isinstance(target, AlignedEventTarget):
            fields = (*(use.field for use in target.fields), *(label.field for label in target.derived))
            reads[target.input] = tuple(dict.fromkeys((*reads.get(target.input, ()), *fields)))
    return reads


def _reads(contract: DataLoaderSpec) -> dict[str, tuple[str, ...] | None]:
    """The sample targets the batch targets read: a ``RowTarget``'s fields read, None for any other."""

    from kumo.pytorch.contract.batch import LabelTarget, StructuredTarget
    from kumo.pytorch.contract.sample import RowTarget

    reads: dict[str, Any] = {}
    for _, target, use in target_uses(contract):
        if not isinstance(target, (LabelTarget, StructuredTarget)):
            continue
        if isinstance(contract.sample.targets[target.source], RowTarget):
            reads[target.source] = tuple(dict.fromkeys((*(reads.get(target.source) or ()), use.field)))
        else:
            reads[target.source] = None
    return reads


def sample_targets(contract: DataLoaderSpec, sampler: Sampler, anchors: GpuAnchors, rows: Any, entity: Any,
                   cutoffs: Any, flags: list[tuple[Any, Any]]) -> dict[str, Any]:
    """Each sample target the batch targets read, for the anchors at `rows` (their entity rows and cutoffs
    given): `Values` per anchor, or for a ``RowTarget`` a field-to-`Values` mapping."""

    from kumo.pytorch.contract.sample import (
        Aggregate,
        Compare,
        ExternalTarget,
        RowTarget,
    )

    targets = contract.sample.targets
    reads = _reads(contract)
    supplied = _request_labels(contract, anchors, rows) if anchors._requests is not None else \
        _table_labels(contract, anchors)
    computed = [name for name in reads if isinstance(targets[name], (Aggregate, Compare))]
    result: dict[str, Any] = {}
    if computed:
        result.update(sampler.scope_values(computed, targets, sampler.entity, entity, cutoffs, None, flags))
    for name, fields in reads.items():
        spec = targets[name]
        if isinstance(spec, ExternalTarget):
            result[name] = _supplied(anchors, rows, name, spec.dtype, supplied, flags)
        elif isinstance(spec, RowTarget):
            result[name] = _row_target(sampler, spec, fields, entity, cutoffs, flags)
    return result


def _row_target(sampler: Sampler, spec: Any, fields: tuple[str, ...], entity: Any, cutoffs: Any,
                flags: list[tuple[Any, Any]]) -> dict[str, Values]:
    """A ``RowTarget``'s fields for each anchor: its first (or last) row, as an event; missing without one."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.events import (
        _component,
    )
    from types import SimpleNamespace

    import cupy

    count = entity.size
    order = "ascending" if spec.selection == "first" else "descending"  # one row kept from that end
    part = _component(sampler, SimpleNamespace(rows=spec.rows, limit=1, order=order), entity, cutoffs, count, flags)
    values = sampler.scope_values(list(fields), spec.fields, part.table, part.rows, cutoffs[part.owner], part.times,
                                  flags)
    at = cupy.full(count, -1, dtype=cupy.int64)
    at[part.owner] = cupy.arange(part.owner.size, dtype=cupy.int64)
    return {field: _gathered(values[field], at) for field in fields}


def _gathered(values: Values, at: Any) -> Values:
    """`values` at positions `at` (CuPy int64), NULL where `at` is -1."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _nulls,
    )

    import cupy

    size = values.size
    if size == 0:
        return _nulls(values.dtype, at.size, values.texts)
    has = at >= 0
    index = cupy.where(has, at, 0)
    present = has & values.valid()[index]
    if values.values is None:  # decimals, on the host
        import pyarrow as pa
        import pyarrow.compute as pc

        host = pc.if_else(pa.array(present.get()), values.arrow().take(pa.array(index.get())), None)
        numbers = None if values.numbers is None else values.numbers[index]
        return Values(values.dtype, None, present, host=host, numbers=numbers)
    return Values(values.dtype, values.values[index], present, values.texts)


def _request_labels(contract: DataLoaderSpec, anchors: GpuAnchors, rows: Any) -> dict[str, list[Any]]:
    """The labels requests built in code supply, by name, one per anchor at `rows` (None where not supplied),
    each request's checked and normalized by upstream's own rule: every name a declared ``ExternalTarget``,
    every value of its type."""

    from kumo.pytorch.contract.sample import _normalize_supplied_targets

    requests = anchors.fetch_rows(rows)
    labels: dict[str, list[Any]] = {}
    for index, request in enumerate(requests):
        if request.targets:
            for value in _normalize_supplied_targets(contract.sample, request.targets):
                labels.setdefault(value.name, [None] * len(requests))[index] = value.value
    return labels


def _table_labels(contract: DataLoaderSpec, anchors: GpuAnchors) -> None:
    """An anchor table's label columns are checked as upstream checks each anchor's labels: every name a
    declared ``ExternalTarget`` (every anchor supplies every one; values are typed as they are read)."""

    from kumo.pytorch.contract.sample import ExternalTarget

    for name in sorted(anchors._table.targets):
        declaration = contract.sample.targets.get(name)
        if declaration is None:
            raise ValueError(f"unknown supplied target: {name!r}")
        if not isinstance(declaration, ExternalTarget):
            raise ValueError(f"target {name!r} was both supplied and generated")


def _supplied(anchors: GpuAnchors, rows: Any, name: str, dtype: Any, supplied: dict[str, list[Any]] | None,
              flags: list[tuple[Any, Any]]) -> Values:
    """Supplied label `name` of the anchors at `rows`, of type `dtype`; missing where none is supplied
    (`supplied`: the labels requests built in code give; None for an anchor table)."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _column_values,
        _nulls,
    )

    import cupy

    if supplied is not None:
        return _host_values(dtype, supplied.get(name, [None] * int(rows.size)))
    column = anchors._table.targets.get(name)
    if column is None:
        return _nulls(dtype, rows.size, None)
    table = anchors._columns
    stored = table.dtypes[column]
    if dtype.name == "timestamp" and stored.name == "timestamp":
        return _column_values(_whole_seconds(anchors, column), column, rows)
    if stored == dtype or (dtype.name.startswith("decimal") and stored.name == dtype.name
                           and (stored.precision, stored.scale) == (dtype.precision, dtype.scale)):
        values = _column_values(table, column, rows)
        return dataclasses.replace(values, dtype=dtype)
    if stored.integer and dtype.integer:  # an integer of another width: refused where it does not fit
        import numpy as np

        values = _column_values(table, column, rows)
        info = np.iinfo(np.dtype(dtype.name))
        if stored.name == "uint64":  # compared as uint64, exactly (as float64, 2**63 + 1024 rounds to 2**63)
            bad = values.values > cupy.uint64(info.max) if info.max < 2**64 - 1 else cupy.zeros(values.values.shape, bool)
        else:
            bad = (values.values < info.min) | (values.values > info.max)
        bad &= values.valid()
        flags.append((lambda: _invalid(name, dtype, values.values[cupy.flatnonzero(bad)[0]].item()), bad.any()))
        return Values(dtype, values.values.astype(np.dtype(dtype.name)), values.present)
    if stored.integer and dtype.name == "float64" or stored.name == "float32" and dtype.name == "float64":
        values = _column_values(table, column, rows)
        return Values(dtype, values.values.astype(cupy.float64), values.present)
    if stored.name == "float64" and dtype.name == "float32":  # upstream packs each as a C float: rounded to the
        values = _column_values(table, column, rows)          # nearest, a finite value rounding to infinity refused
        narrow = values.values.astype(cupy.float32)
        bad = cupy.isinf(narrow) & ~cupy.isinf(values.values) & values.valid()
        flags.append((lambda: _invalid(name, dtype, values.values[cupy.flatnonzero(bad)[0]].item()), bad.any()))
        return Values(dtype, narrow, values.present)
    raise ValueError(f"test.kumo_pytorch.reference_engine.gpu does not read anchor column {column!r} ({stored.name}) as target {name!r} "
                     f"({dtype.name}) yet")


def _invalid(name: str, dtype: Any, value: Any) -> str:
    """Upstream's message for a supplied label its type refuses (``DType.normalize``)."""

    return f"targets.{name}: {value!r} is not a valid {dtype} value"


def _whole_seconds(anchors: GpuAnchors, column: str) -> Any:
    """The anchor table with timestamp label `column` in whole seconds, rounded up (as upstream normalizes a
    supplied time); cached."""

    from test.kumo_pytorch.reference_engine.gpu.data.decimal_storage import (
        _stored,
    )
    from test.kumo_pytorch.reference_engine.gpu.data.tables import GpuTable

    cache = anchors.__dict__.setdefault("_label_seconds", {})
    if column not in cache:
        table = anchors._columns
        stored = _stored(table.columns[column], table.dtypes[column], f"anchors.{column}")
        cache[column] = GpuTable("labels", table.num_rows, {column: stored}, {column: table.dtypes[column]})
    return cache[column]


def _host_values(dtype: Any, labels: list[Any]) -> Values:
    """Normalized Python labels as `Values` of `dtype` (text as codes into sorted texts)."""

    from test.kumo_pytorch.reference_engine.gpu.data.tables import (
        GpuTable,
        _with_codes,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _column_values,
    )

    import cupy
    import pyarrow as pa
    import pylibcudf as plc

    if dtype.name == "timestamp":
        arrow = pa.array([None if value is None else int(value.timestamp()) for value in labels], pa.int64())
        arrow = arrow.cast(pa.timestamp("s", tz="UTC"))
    else:
        arrow = pa.array(labels, dtype.to_arrow())
    if dtype.name.startswith("decimal"):
        present = cupy.asarray(arrow.is_valid().to_numpy(zero_copy_only=False))
        return Values(dtype, None, present, host=arrow)
    table = _with_codes(GpuTable("labels", len(labels), {"v": plc.Column.from_arrow(arrow)}, {"v": dtype}))
    return _column_values(table, "v", cupy.arange(len(labels), dtype=cupy.int64))


@dataclass
class TargetState:
    """What fitting the targets learns besides encoders.

    :param thresholds: each aligned target's fitted (or given) threshold per derived label.
    :param groups: each matching group target's groups (`test.kumo_pytorch.reference_engine.gpu.encoding.composite.CompositeVocabulary`, unknown -1).
    """

    thresholds: dict[str, dict[str, float]] = dataclass_field(default_factory=dict)
    groups: dict[str, CompositeVocabulary] = dataclass_field(default_factory=dict)


def fit_targets(contract: DataLoaderSpec, labels: dict[str, Any], sampled: dict[str, Any],
                encoders: dict[str, GpuEncoder]) -> TargetState:
    """Fit each target on the fit split: thresholds and matching groups, then each target field's encoder
    (NULLs excluded) into `encoders`. An encoder an input already fitted is kept, and refused if the target
    declares it differently. `sampled`: the inputs' values (`test.kumo_pytorch.reference_engine.gpu.batching.inputs._sampled`)."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        component,
        fit_groups,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        GpuEncoder,
        _raise_flags,
        concatenated,
        fit_encoder,
    )

    from kumo.pytorch.contract.batch import (
        AlignedEventTarget,
        MatchingGroupTarget,
    )
    from kumo.pytorch.contract.batch.encodings import Raw, UseEncoding

    state = TargetState()
    flags: list[tuple[Any, Any]] = []
    for name in sorted(contract.batch.targets):
        target = contract.batch.targets[name]
        if isinstance(target, MatchingGroupTarget):
            specs = _group_specs(contract, target, encoders)
            state.groups[name] = fit_groups(specs, [component(spec, sampled[target.input].values[spec.field], None,
                                                              flags) for spec in specs])
    _raise_flags(flags)
    # Current Kumo collects matching-group keys before fitting derived thresholds.
    for name in sorted(contract.batch.targets):
        target = contract.batch.targets[name]
        if isinstance(target, AlignedEventTarget):
            for label in target.derived:
                state.thresholds.setdefault(name, {})[label.name] = _threshold(
                    name, label, sampled[target.input].values[label.field])
    parts: dict[str, list[Values]] = {}
    definitions: dict[str, tuple[Any, Any]] = {}
    for name, target, use in target_uses(contract):
        values = _target_values(labels, sampled, state, name, target, use)
        identity = target_identity(name, use)
        config = contract.batch.encodings[use.encoding.name] if isinstance(use.encoding, UseEncoding) else use.encoding
        definitions[identity] = (config, values.dtype)
        parts.setdefault(identity, []).append(values)
    for identity in sorted(definitions):
        config, dtype = definitions[identity]
        if identity in encoders:
            existing = encoders[identity]
            # upstream's check already refuses a shared encoder of two dtypes; kept, as upstream keeps it
            if existing.dtype != dtype or existing.config != config:
                raise ValueError(f"target encoder {identity}: incompatible shared encoding")
            continue  # an encoder an input shares never collects target observations
        if any(part.values is None for part in parts[identity]):  # decimals: only Raw is built
            if not isinstance(config, Raw):
                raise ValueError(f"encoder {identity}: test.kumo_pytorch.reference_engine.gpu does not encode {dtype.name} values "
                                 f"with {type(config).__name__} yet")
            encoders[identity] = GpuEncoder(identity, config, dtype)
            continue
        encoders[identity] = fit_encoder(identity, config, dtype, concatenated(parts[identity]))
    return state


def _group_specs(contract: DataLoaderSpec, target: Any, encoders: dict[str, GpuEncoder]) -> list[Any]:
    """A matching group target's components: the typed values of its fields, a NULL spelled
    ``("null", "")``."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        ComponentSpec,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        encoder_identity,
    )

    uses = {use.field: use for use in contract.batch.inputs[target.input].representation.fields}
    specs = []
    for name in target.fields:
        encoder = encoders[encoder_identity(target.input, uses[name])]
        specs.append(ComponentSpec(name, encoder, True, encoder.dtype.name, ("null", "")))
    return specs


def _threshold(name: str, label: Any, values: Values) -> float:
    """A derived label's threshold: given, or numpy's median of the non-NULL values (0.0 without any). A
    non-finite value is refused, as upstream refuses it."""

    import cupy

    numbers = threshold_numbers(name, values)
    ordered = cupy.sort(numbers) if label.threshold is None else numbers
    return fitted_threshold(name, label, int(numbers.size), lambda at: ordered[cupy.asarray(at)].get())


def threshold_numbers(name: str, values: Values) -> Any:
    """The numbers a derived label's threshold is the median of (CuPy float64): the non-NULL values, refused
    unless finite. (A median of zero is 0.0 whichever zeros the sort leaves in the middle: numpy's mean of them
    adds from +0.0.)"""

    import cupy

    numbers = values.values.astype(cupy.float64)[values.valid()]
    if not bool(cupy.isfinite(numbers).all()):
        raise ValueError(f"target {name!r}: threshold values must be finite")
    return numbers


def fitted_threshold(name: str, label: Any, count: int, smallest: Any) -> float:
    """A derived label's threshold over `count` numbers (`threshold_numbers`): given, or numpy's median, the
    middle value or the mean of the middle two, from the values at given positions of their sorted order
    (`smallest`: host int positions to float64 values); 0.0 without any."""

    import numpy as np

    if label.threshold is not None:
        threshold = float(label.threshold)
    elif count == 0:
        threshold = 0.0
    else:
        middle = count // 2
        picked = smallest([middle]) if count % 2 else smallest([middle - 1, middle])
        threshold = float(np.mean(np.asarray(picked, dtype=np.float64)))
    if not np.isfinite(threshold):
        raise ValueError(f"target {name!r}: threshold values must be finite")
    return threshold


def _threshold_labels(label: Any, threshold: float, values: Values) -> Values:
    """Each row's derived label: ``above`` where its value passes `threshold`, ``below`` elsewhere, NULL
    staying missing. Integers are compared exactly, as Python compares an int with a float."""

    import math

    import cupy
    import numpy as np

    strict = label.comparison == "greater_than"
    if values.dtype.integer:
        info = np.iinfo(values.values.dtype)
        bound = math.floor(threshold) if strict else math.ceil(threshold)  # v > t: v > floor(t); v >= t: v >= ceil(t)
        if bound > info.max or (bound == info.max and strict):
            above = cupy.zeros(values.values.shape, dtype=bool)
        elif bound < info.min:
            above = cupy.ones(values.values.shape, dtype=bool)
        else:
            bound = values.values.dtype.type(bound)
            above = values.values > bound if strict else values.values >= bound
    else:
        numbers = values.values.astype(cupy.float64)
        above = numbers > threshold if strict else numbers >= threshold
    choices = _host_values(label.dtype, [label.dtype.normalize(label.below), label.dtype.normalize(label.above)])
    return _gathered(choices, cupy.where(values.valid(), above.astype(cupy.int64), -1))


def _target_values(labels: dict[str, Any], sampled: dict[str, Any], state: TargetState, name: str, target: Any,
                   use: Any) -> Values:
    """What a target field encodes, for a call's anchors (per anchor) or rows (an aligned target, per row)."""

    from kumo.pytorch.contract.batch import (
        AlignedEventTarget,
        StructuredTarget,
    )

    if isinstance(target, AlignedEventTarget):
        derived = {label.name: label for label in target.derived}
        if use.field in derived:
            label = derived[use.field]
            return _threshold_labels(label, state.thresholds[name][label.name],
                                     sampled[target.input].values[label.field])
        return sampled[target.input].values[use.field]
    return labels[target.source][use.field] if isinstance(target, StructuredTarget) else labels[target.source]


def encode_targets(contract: DataLoaderSpec, labels: dict[str, Any], sampled: dict[str, Any],
                   encoders: dict[str, GpuEncoder], spaces: dict[str, Any], state: TargetState,
                   flags: list[tuple[Any, Any]], model: bool) -> dict[str, Any]:
    """Every batch target, encoded for all of a call's anchors: ``{"values", "valid"}`` per target field (a
    structured or aligned target: per field it names; an aligned target's flat over its input's rows, to be
    laid out per batch), as upstream encodes them."""

    from test.kumo_pytorch.reference_engine.gpu.batching.layout import _raw
    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        component,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.tokens import (
        translated,
    )

    import cupy
    import torch

    from kumo.pytorch.contract.batch import (
        AlignedEventTarget,
        MatchingGroupTarget,
        StructuredTarget,
    )
    from kumo.pytorch.contract.batch.encodings import Raw

    result: dict[str, Any] = {}
    for name in sorted(contract.batch.targets):
        target = contract.batch.targets[name]
        if isinstance(target, MatchingGroupTarget):
            groups = state.groups[name]
            ids = groups.lookup([component(spec, sampled[target.input].values[spec.field], None, flags)
                                 for spec in groups.specs])
            result[name] = {"values": torch.from_dlpack(ids), "valid": torch.from_dlpack(ids >= 0)}
    for name, target, use in target_uses(contract):
        identity = target_identity(name, use)
        encoder = encoders[identity]
        values = _target_values(labels, sampled, state, name, target, use)
        if isinstance(encoder.config, Raw):
            encoded = _raw(identity, values, model)
        else:
            ids, valid = encoder.encode(values, flags)
            if use.id_space:
                ids = translated(spaces[use.id_space], identity, ids)
            encoded = {"values": torch.from_dlpack(cupy.ascontiguousarray(ids)), "valid": torch.from_dlpack(valid)}
        if isinstance(target, (StructuredTarget, AlignedEventTarget)):
            result.setdefault(name, {})[use.field] = encoded
        else:
            result[name] = encoded
    return {name: result[name] for name in sorted(result)}


def mask_tokens(target: Any, values: Any, slots: Any, owner: Any, history: Any, place: Any, entities: Any,
                fields: list[str], space: Any, *, seed: int, epoch: int) -> tuple[Any, dict[str, tuple[Any, Any]]]:
    """A ``MaskedValuesTarget`` applied to tokens (CuPy, one per token, grouped by sample): `values` their final
    IDs, `slots` their fields' places in `fields`, `owner` their samples, `history` their events' places in
    their samples, `place` their places among their events' tokens; `entities` (CuPy int64) each sample's entity
    as upstream's draws read it (`GpuAnchors.mask_keys`).

    Upstream's ``mask_event_values`` with its own seeded draws (``_uniform``, ``_below``), so a decision
    depends on the sample's entity and the token's coordinates, never on the batch. Returns the corrupted
    IDs and the target's per-token arrays (with their padding fills): ``values`` (the original ID where
    supervised, else -100), ``valid`` (supervised) and ``corruption`` (changed in the input).
    """

    import cupy
    import torch

    from kumo.pytorch.data.masking import _below, _uniform

    ids = torch.from_dlpack(cupy.ascontiguousarray(values))
    keys = torch.from_dlpack(cupy.ascontiguousarray(slots))
    owners = torch.from_dlpack(cupy.ascontiguousarray(owner))
    events = torch.from_dlpack(cupy.ascontiguousarray(history))
    positions = torch.from_dlpack(cupy.ascontiguousarray(place))
    who = torch.from_dlpack(cupy.ascontiguousarray(entities))[owners] if ids.numel() else torch.zeros_like(ids)
    zeros = torch.zeros_like(keys)
    eligible = torch.ones_like(keys, dtype=torch.bool)
    if target.fields:
        allowed = [rank for rank, name in enumerate(fields) if name in target.fields]
        eligible = torch.isin(keys, torch.tensor(allowed, dtype=keys.dtype, device=keys.device))
    draws = {"seed": seed, "epoch": epoch}
    token = _below(target.token_probability, _uniform(keys, who, events, positions, **draws, domain=1))
    event = _below(target.event_probability, _uniform(keys, who, events, zeros, **draws, domain=2))
    key = _below(target.key_probability, _uniform(keys, who, keys, zeros, **draws, domain=3))
    selected = eligible & ((token | event | key) if target.combine == "union" else (token & event & key))
    unknown = selected & _below(target.unknown_probability,
                                _uniform(keys, who, events, positions, keys, **draws, domain=4))
    if target.ensure_at_least_one and ids.numel():
        count = int(entities.size)
        index = torch.arange(ids.numel(), device=ids.device)
        none = ids.numel()

        def first(where: Any) -> Any:  # each sample's first token where `where` holds (`none` without one)
            start = torch.full((count,), none, dtype=torch.long, device=ids.device)
            return start.scatter_reduce(0, owners, torch.where(where, index, none), reduce="amin")

        candidates = first(eligible)
        # a sample with an eligible token and none selected: its first eligible token is selected, not unknown
        pick = candidates[(candidates < none) & (first(selected) == none)]
        selected[pick] = True
        unknown[pick] = False
        if not target.supervise_unknown:  # and with none supervised: its first selected token is not unknown
            pick = first(selected)[(candidates < none) & (first(selected & ~unknown) == none)]
            unknown[pick] = False
    supervised = selected if target.supervise_unknown else selected & ~unknown
    corrupted = ids.clone()
    corrupted[unknown] = space.roles.get("unknown", space.roles["mask"])
    corrupted[selected & ~unknown] = space.roles["mask"]
    answers = torch.where(supervised, ids, torch.full_like(ids, -100))
    as_cupy = cupy.from_dlpack
    return as_cupy(corrupted), {"values": (as_cupy(answers), -100), "valid": (as_cupy(supervised), False),
                                "corruption": (as_cupy(corrupted != ids), False)}


def next_tokens(values: Any, bounds: Any, prefix: bool) -> tuple[Any, Any, Any]:
    """A ``NextTokenTarget`` on flattened sequences (`values` CuPy, `bounds` host): (which tokens stay in the
    input, each sequence's last one dropped; the input's new bounds; the answers). The answers are each
    kept token's successor, or with a prefix input every token of the sequence, with their bounds (host)."""

    import cupy
    import numpy as np

    lengths = np.diff(bounds)
    kept = np.maximum(lengths - 1, 0)
    new_bounds = np.concatenate(([0], np.cumsum(kept, dtype=np.int64)))
    total = int(new_bounds[-1])
    starts = cupy.asarray(np.repeat(bounds[:-1] - new_bounds[:-1], kept))
    indices = cupy.arange(total, dtype=cupy.int64) + starts
    if prefix:
        return indices, new_bounds, (values, bounds)
    return indices, new_bounds, (values[indices + 1], new_bounds)
