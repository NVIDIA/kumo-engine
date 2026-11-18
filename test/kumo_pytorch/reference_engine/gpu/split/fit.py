# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit a contract on a split dataset (`test.kumo_pytorch.reference_engine.gpu.split.plan`), a partition at a time.

The fit split's anchors are read partition by partition, a chunk at a time, as
`test.kumo_pytorch.reference_engine.gpu.batching.fit.fit_inputs` reads them all at once. Everything fitted keeps
only what its fit needs, merged across chunks, then takes the same final step
as the whole dataset's fit:

- Vocabulary and learned Text: how often each value or text occurs (a learned
  BPE depends only on that, not on the order of the texts), held on the host.
- ZScore: each chunk's count, mean and sum of squared deviations, merged
  (equal to the whole dataset's mean and deviation up to rounding).
- Buckets cuts and derived labels' thresholds (medians): each chunk's numbers
  (a Buckets field's -0.0 made 0.0, as the whole dataset's fit makes it) sorted
  on the GPU as keys ordered as the floats are, and kept on the host; the values
  at the positions the formula needs are found exactly among the sorted runs
  (`_smallest`).
- Composite IDs and matching groups: each chunk's distinct tuples of component
  results and how many rows give each (`TupleCounts`), a Raw text component's
  texts recoded into one dictionary when merged.

The first pass fits the inputs' and graph node fields' encoders (unless the
inputs' half of a fitted state is reused), the thresholds, the matching groups
and the target encoders of labels and event fields. A second pass, only when
the contract needs one, fits what depends on those: Composite IDs (tuples of
encoded results) and derived labels' encoders (labels by threshold). The
fitted samples' lengths (``Batching.length_input``) are measured with the
fitted state in a pass of their own (`measure_split_lengths`), back in the fit
anchors' order. Raw encoders, and those given their state, read nothing.
"""

from __future__ import annotations

from test.kumo_pytorch.reference_engine.gpu.data.text import (
    arrow_column,
    is_text,
)
from test.kumo_pytorch.reference_engine.gpu.split.fit_storage import (
    NumericRuns,
    TupleParts,
    ValueCounts,
)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        Component,
        ComponentSpec,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        GpuEncoder,
        Values,
    )
    from test.kumo_pytorch.reference_engine.gpu.split.plan import SplitPlan

    from kumo.pytorch.contract.dtypes import DType

__all__ = ["SPLIT_FIT_ANCHORS", "EncoderFit", "TupleCounts", "fit_split_state", "measure_split_lengths"]

#: Fitting anchors sampled at once within a partition (memory grows with their values).
SPLIT_FIT_ANCHORS = 1 << 18

_TOP = 1 << 63


def _held(plan: SplitPlan, anchors: GpuAnchors, rows: Any) -> dict[int, tuple[Any, Any]]:
    """The fit anchors each partition holds: (their rows, their places among `rows`), in `rows`' order."""

    import cupy
    import numpy as np

    where = plan.partition_of(anchors, rows).get()  # a fit split has anchors (upstream's check)
    places = {int(k): np.flatnonzero(where == k) for k in np.unique(where)}
    return {k: (rows[cupy.asarray(place)], place) for k, place in places.items()}


def _passes(plan: SplitPlan, held: dict[int, tuple[Any, Any]]) -> Any:
    """Each partition holding fit anchors in turn, a chunk at a time: (sampler, index, chunk rows, their places)."""

    for partition in plan.stream(sorted(held)):
        sampler, index = plan.reader(partition)
        mine, places = held[partition.id]
        del partition
        for start in range(0, int(mine.size), SPLIT_FIT_ANCHORS):
            yield sampler, index, mine[start:start + SPLIT_FIT_ANCHORS], places[start:start + SPLIT_FIT_ANCHORS]
        del sampler, index  # this partition freed before the one after the next is read


def fit_split_state(plan: SplitPlan, anchors: GpuAnchors, rows: Any,
                    field_maps: dict[str, dict[tuple[str, ...], int]] | None,
                    reused: tuple[Any, Any, Any] | None = None, *, budget: Any = None, pinned=None) -> tuple[Any, ...]:
    """`test.kumo_pytorch.reference_engine.gpu.batching.fit.fit_inputs` for the anchors at `rows` of a split dataset: (encoders, Composite IDs, ID
    spaces, target state), fitted partition by partition. With `reused` (a prior task's input state), only the
    targets are fitted, and an ID space the inputs reuse cannot gain a target's member."""

    from test.kumo_pytorch.reference_engine.gpu.batching.fit import (
        _node_values,
    )
    from test.kumo_pytorch.reference_engine.gpu.batching.inputs import (
        _config,
        _inputs,
        _sampled,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        ComponentSpec,
        component,
        component_specs,
        fit_composite,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        GpuEncoder,
        _raise_flags,
        encoder_identity,
        graph_identity,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import (
        TargetState,
        _group_specs,
        _target_values,
        _threshold_labels,
        fitted_threshold,
        target_identity,
        target_uses,
        threshold_numbers,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.tokens import (
        allocate_spaces,
    )

    import cupy

    from kumo.pytorch.contract.batch import (
        AlignedEventTarget,
        Composite,
        MatchingGroupTarget,
    )
    from kumo.pytorch.contract.batch.encodings import Raw, UseEncoding

    contract = plan.contract
    targets = contract.batch.targets
    held = _held(plan, anchors, rows)
    composite_inputs = {name: value for name, value in contract.batch.inputs.items()
                        if isinstance(getattr(value.representation, "row", None), Composite)}
    derived = [(name, target, label) for name, target in sorted(targets.items())
               if isinstance(target, AlignedEventTarget) for label in target.derived]
    groups = [(name, target) for name, target in sorted(targets.items()) if isinstance(target, MatchingGroupTarget)]
    input_keys = set(reused[0]) if reused is not None else \
        {encoder_identity(name, use) for name, use in _inputs(contract)} | _node_keys(contract)
    labelled = {(name, label.name) for name, _, label in derived}
    plain, later, shared = [], [], {}  # target fields fitted in the first pass, the second, or by an input
    for name, target, use in target_uses(contract):
        identity = target_identity(name, use)
        config = contract.batch.encodings[use.encoding.name] if isinstance(use.encoding, UseEncoding) else use.encoding
        if identity in input_keys:
            shared[identity] = config
        elif (name, use.field) in labelled:
            later.append((name, target, use, identity, config))
        else:
            plain.append((name, target, use, identity, config))

    fits: dict[str, EncoderFit] = {}
    target_fits: dict[str, EncoderFit] = {}
    target_dtypes: dict[str, Any] = {}
    medians = {(name, label.name): (NumericRuns(budget) if budget is not None else []) for name, _, label in derived if label.threshold is None}
    group_tuples = {name: TupleCounts(budget) for name, _ in groups}
    stand_ins: dict[tuple[str, str], Any] = {}

    def add(store: dict[str, EncoderFit], key: str, config: Any, values: Values, *, target: bool = False) -> None:
        if not target and pinned is not None and key in pinned:
            return
        if key not in store:
            store[key] = EncoderFit(key, config, values.dtype, numeric_decimals=not target, budget=budget)
        store[key].add(values)

    for sampler, index, chunk, _ in _passes(plan, held):
        sampled, labels = _sampled(contract, sampler, index, anchors, chunk, labels=bool(targets))
        if reused is None:
            for name, use in _inputs(contract):
                source = sampled[name].fitting if sampled[name].fitting is not None else sampled[name].values
                add(fits, encoder_identity(name, use), _config(contract, use), source[use.field])
            for name, table, use, values in _node_values(contract, sampler, index, anchors, chunk):
                add(fits, graph_identity(name, table, use), _config(contract, use), values)
        flags: list[tuple[Any, Any]] = []
        for name, target in groups:  # the fields' typed values: raw, whatever their inputs' encoders
            parts = []
            for field in target.fields:
                values = sampled[target.input].values[field]
                if (name, field) not in stand_ins:
                    stand_ins[(name, field)] = ComponentSpec(field, GpuEncoder(field, Raw(), values.dtype), True,
                                                             values.dtype.name, ("null", ""))
                parts.append(component(stand_ins[(name, field)], values, None, flags))
            group_tuples[name].add(parts)
        _raise_flags(flags)
        for name, target, label in derived:
            numbers = threshold_numbers(name, sampled[target.input].values[label.field])
            if label.threshold is None and numbers.size:
                medians[(name, label.name)].append(cupy.sort(_keys(numbers)).get())
        for name, target, use, identity, config in plain:
            values = _target_values(labels, sampled, TargetState(), name, target, use)
            target_dtypes[identity] = values.dtype
            add(target_fits, identity, config, values, target=True)
        for name, target, use in target_uses(contract):  # what a shared encoder's target values are
            identity = target_identity(name, use)
            if identity in shared:
                target_dtypes[identity] = next(label.dtype for label in target.derived if label.name == use.field) \
                    if (name, use.field) in labelled else \
                    _target_values(labels, sampled, TargetState(), name, target, use).dtype
        del sampled, labels

    if reused is not None:
        encoders, composites, kept = dict(reused[0]), reused[1], reused[2]
    else:
        encoders, composites, kept = {**{key: fits[key].encoder() for key in sorted(fits)}, **(pinned or {})}, {}, None
    state = TargetState()
    for name, target, label in derived:
        runs = medians.get((name, label.name), [])
        state.thresholds.setdefault(name, {})[label.name] = fitted_threshold(
            name, label, _run_size(runs), lambda at, runs=runs: _smallest(runs, at))
    for name, target in groups:
        specs = _group_specs(contract, target, encoders)
        components, counts = group_tuples[name].components(specs)
        state.groups[name] = _weighted_known(specs, components, counts, groups=True)

    composite_tuples = {} if reused is not None else \
        {name: TupleCounts(budget) for name, value in composite_inputs.items() if value.representation.row.vocabulary is None}
    if composite_tuples or later:  # the second pass: what needs the encoders or the thresholds
        for sampler, index, chunk, _ in _passes(plan, held):
            sampled, _ = _sampled(contract, sampler, index, anchors, chunk, labels=False)
            flags = []
            for name in composite_tuples:
                specs = component_specs(composite_inputs[name].representation, name, encoders)
                composite_tuples[name].add([component(spec, sampled[name].values[spec.field],
                                                      None if spec.raw else
                                                      spec.encoder.encode(sampled[name].values[spec.field], flags),
                                                      flags) for spec in specs])
            _raise_flags(flags)
            for name, target, use, identity, config in later:
                label = next(label for label in target.derived if label.name == use.field)
                values = _threshold_labels(label, state.thresholds[name][label.name],
                                           sampled[target.input].values[label.field])
                add(target_fits, identity, config, values, target=True)
            del sampled
    if reused is None:
        for name, value in composite_inputs.items():
            specs = component_specs(value.representation, name, encoders)
            if name in composite_tuples:
                components, counts = composite_tuples[name].components(specs)
                composites[name] = _weighted_known(specs, components, counts, top_k=value.representation.row.top_k)
            else:
                composites[name] = fit_composite(specs, value.representation.row, [])  # supplied
    for identity in sorted({*target_fits, *shared}):  # as `test.kumo_pytorch.reference_engine.gpu.encoding.targets.fit_targets` fits them
        if identity in shared:
            existing = encoders[identity]
            if existing.dtype != target_dtypes[identity] or existing.config != shared[identity]:
                raise ValueError(f"target encoder {identity}: incompatible shared encoding")
            continue
        encoders[identity] = target_fits[identity].encoder()
    spaces = allocate_spaces(contract, encoders, composites, field_maps)
    if kept is not None:
        for name, space in kept.items():
            if any(identity not in space.members for identity in spaces[name].members):
                raise ValueError(f"reused ID space {name!r} cannot acquire target members")
        spaces = {**spaces, **kept}
    return encoders, composites, spaces, state


def measure_split_lengths(plan: SplitPlan, anchors: GpuAnchors, rows: Any, encoders: Any, composites: Any,
                          spaces: Any, targets: Any, field_maps: Any) -> dict[str, Any]:
    """Each fitted sample's length in ``Batching.length_input`` (`test.kumo_pytorch.reference_engine.gpu.plan.groups._measured`), partition by
    partition with the fitted state, back in `rows`' order (the length summary's ties keep it)."""

    from functools import partial
    from test.kumo_pytorch.reference_engine.gpu.batching.build import (
        build_batches,
    )
    from test.kumo_pytorch.reference_engine.gpu.plan.groups import _measured

    import cupy
    import numpy as np

    held = _held(plan, anchors, rows)
    measured: dict[str, Any] = {}
    for sampler, index, chunk, places in _passes(plan, held):
        build = partial(build_batches, sampler=sampler, index=index, encoders=encoders, composites=composites,
                        spaces=spaces, target_state=targets, anchors=anchors, maximum=None, drop_last=False,
                        model=False, fingerprint="", field_maps=field_maps)
        part = _measured(build, plan.contract, chunk, cupy.zeros(int(chunk.size), dtype=bool), plan.build_bytes)
        for key, values in part.items():
            measured.setdefault(key, np.zeros(int(rows.size), dtype=np.int64))[places] = values
    return measured


def _node_keys(contract: Any) -> set[str]:
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        graph_identity,
    )

    from kumo.pytorch.contract.batch import Graph

    return {graph_identity(name, table, use) for name, value in contract.batch.inputs.items()
            if isinstance(value.representation, Graph)
            for table, group in value.representation.node_fields.items() for use in group.fields}


def _weighted_known(specs: list[ComponentSpec], components: list[Component], counts: Any, *,
                    top_k: int | None = None, groups: bool = False) -> Any:
    """`test.kumo_pytorch.reference_engine.gpu.encoding.composite.fit_composite` (the ``top_k`` most frequent tuples from ID 2, ties in canonical typed
    order) or `fit_groups` (every tuple from 0, in that order) over distinct tuples each given `counts` times."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        _distinct,
        _known,
    )

    import cupy

    order, starts, _ = _distinct(components)
    if groups:
        return _known(specs, components, order[starts], 0, -1)
    if not starts.size:
        return _known(specs, components, order, 2, 1)
    weights = cupy.cumsum(counts[order])
    ends = cupy.append(starts[1:], order.size) - 1
    totals = weights[ends] - cupy.where(starts > 0, weights[cupy.maximum(starts - 1, 0)], 0)
    by = cupy.lexsort(cupy.stack([cupy.arange(starts.size), -totals]))[:top_k]
    return _known(specs, components, order[starts[by]], 2, 1)


class TupleCounts:
    """Distinct tuples of component results (`test.kumo_pytorch.reference_engine.gpu.encoding.composite.Component`) and how many rows give each, merged
    part by part on the host; a Raw text component's texts kept as texts, coded into one dictionary when merged."""

    def __init__(self, budget: Any = None) -> None:
        self.parts: Any = TupleParts(budget) if budget is not None else []  # per part: missing, identity or texts, counts

    def add(self, components: list[Component]) -> None:
        from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
            _distinct,
        )
        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
            _texts_at,
        )

        import pyarrow as pa

        order, starts, counts = _distinct(components)
        if not starts.size:
            return
        chosen = order[starts]
        missing, identity = [], []
        for part in components:
            gap = part.missing[chosen]
            missing.append(gap.get())
            if part.spec.raw and is_text(part.spec.dtype):
                words = _texts_at(part.texts, part.identity[chosen], part.spec.dtype) if part.texts is not None \
                    and part.texts.size() else [None] * int(chosen.size)
                identity.append(pa.array([None if absent else word for word, absent in
                                          zip(words, gap.get().tolist(), strict=True)], part.spec.dtype.to_arrow()))
            else:
                identity.append(part.identity[chosen].get())
        self.parts.append((missing, identity, counts.get()))

    def components(self, specs: list[ComponentSpec]) -> tuple[list[Component], Any]:
        """The tuples of every part as components (texts coded into one sorted dictionary), and each one's count
        (CuPy); a tuple several parts gave appears once per part."""

        from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
            Component,
        )
        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
            _view,
        )

        import cupy
        import numpy as np
        import pyarrow as pa
        import pylibcudf as plc

        parts = self.parts.materialized() if isinstance(self.parts, TupleParts) else self.parts
        result = []
        for position, spec in enumerate(specs):
            missing = cupy.asarray(np.concatenate([part[0][position] for part in parts]) if parts else
                                   np.zeros(0, dtype=bool))
            if spec.raw and is_text(spec.dtype):
                words = pa.concat_arrays([part[1][position] for part in parts]) if parts else \
                    pa.array([], spec.dtype.to_arrow())
                column = arrow_column(words)
                distinct = plc.stream_compaction.distinct(  # the texts given (a NULL's place is a missing one)
                    plc.Table([arrow_column(words.drop_null())]), [0],
                    plc.stream_compaction.DuplicateKeepOption.KEEP_ANY, plc.types.NullEquality.EQUAL,
                    plc.types.NanEquality.ALL_EQUAL)
                dictionary = plc.sorting.sort(distinct, [plc.types.Order.ASCENDING], [plc.types.NullOrder.AFTER])
                codes = _view(plc.search.lower_bound(dictionary, plc.Table([column]), [plc.types.Order.ASCENDING],
                                                     [plc.types.NullOrder.AFTER])).astype(cupy.int64)
                result.append(Component(spec, missing, cupy.where(missing, 0, codes), dictionary.columns()[0]))
            else:
                identity = cupy.asarray(np.concatenate([part[1][position] for part in parts]) if parts else
                                        np.zeros(0, dtype=np.int64))
                result.append(Component(spec, missing, identity))
        counts = cupy.asarray(np.concatenate([part[2] for part in parts]) if parts else
                              np.zeros(0, dtype=np.int64))
        return result, counts


class EncoderFit:
    """One encoder, fitted over values given part by part as `test.kumo_pytorch.reference_engine.gpu.encoding.encoders.fit_encoder` fits it over all of
    them at once (NULLs excluded; a decimal's numbers for a numeric transform)."""

    def __init__(self, identity: str, config: Any, dtype: DType, *, numeric_decimals: bool = True, budget: Any = None) -> None:
        from kumo.pytorch.contract.batch.encodings import (
            Buckets,
            Text,
            Vocabulary,
            ZScore,
        )

        self.identity, self.config, self.dtype = identity, config, dtype
        self.numeric_decimals = numeric_decimals  # an input's decimals may take ZScore or Buckets; a target's only Raw
        given = {Vocabulary: "values", ZScore: "mean", Buckets: "boundaries", Text: "state"}
        kinds = {Vocabulary: "vocabulary", ZScore: "zscore", Buckets: "buckets", Text: "text"}
        kind = kinds.get(type(config))
        self.kind = None if kind is None or getattr(config, given[type(config)]) is not None else kind
        self.decimal = False  # a part with no CuPy values: Raw, or a numeric transform of its numbers
        self.counts: Any = ValueCounts(budget) if budget is not None else []  # per part: distinct keys (numpy, or texts as Arrow) and counts
        self.moments = (0, 0.0, 0.0)  # count, mean, sum of squared deviations
        self.runs: Any = NumericRuns(budget) if budget is not None else []  # per part: sorted keys of its numbers (numpy uint64, `_keys`)

    def add(self, values: Values) -> None:
        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
            _normalized,
            _numbers,
            _raise_flags,
            _texts_at,
            bucket_numbers,
        )

        import cupy

        if values.values is None:
            self.decimal = True
        if self.kind is None or (self.decimal and self.kind in {"vocabulary", "text"}):
            return  # nothing read (Raw, given state), or refused when fitted
        if self.kind in {"vocabulary", "text"}:
            if values.texts is not None:
                codes = values.values[values.valid()]
                distinct, counts = cupy.unique(codes, return_counts=True)
                if distinct.size:
                    import pyarrow as pa

                    self.counts.append((pa.array(_texts_at(values.texts, distinct, self.dtype), self.dtype.to_arrow()),
                                        counts.get()))
            elif self.kind == "vocabulary":
                flags: list[tuple[str, Any]] = []
                keys = _normalized(values, flags, self.identity)[values.valid()]
                _raise_flags(flags)
                distinct, counts = cupy.unique(keys, return_counts=True)
                self.counts.append((distinct.get(), counts.get()))
            return
        if self.kind == "zscore":
            numeric = _numbers(self.identity, values)
            if numeric.size:
                mean = float(numeric.mean())
                self.moments = _merged(self.moments, (int(numeric.size), mean, float(((numeric - mean) ** 2).sum())))
            return
        numeric = bucket_numbers(self.identity, self.config, values)
        if numeric.size:
            self.runs.append(cupy.sort(_keys(numeric)).get())

    def encoder(self) -> GpuEncoder:
        """The fitted encoder, from every part given."""

        import math
        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
            GpuEncoder,
            bucket_cuts,
            fit_encoder,
            learned_tokenizer,
            zscore_encoder,
        )

        from kumo.pytorch.contract.batch.encodings import Buckets, Raw, ZScore

        identity, config, dtype = self.identity, self.config, self.dtype
        if self.decimal:  # as `test.kumo_pytorch.reference_engine.gpu.batching.fit.fit_inputs` takes decimals
            if isinstance(config, Raw):
                return GpuEncoder(identity, config, dtype)
            if not isinstance(config, (ZScore, Buckets)) or not self.numeric_decimals:
                raise ValueError(f"encoder {identity}: test.kumo_pytorch.reference_engine.gpu does not encode {dtype.name} values "
                                 f"with {type(config).__name__} yet")
        if self.kind is None:  # Raw, a given state, or an encoding not built: none reads values
            return fit_encoder(identity, config, dtype, None)
        if isinstance(config, (ZScore, Buckets)) and not dtype.numeric:
            raise ValueError(f"encoder {identity}: {type(config).__name__} requires a numeric dtype")
        if self.kind == "zscore":
            count, mean, squares = self.moments
            return zscore_encoder(identity, config, dtype, *((mean, math.sqrt(squares / count)) if count else
                                                             (0.0, 1.0)))
        if self.kind == "buckets":
            count = _run_size(self.runs)
            return GpuEncoder(identity, config, dtype,
                              boundaries=bucket_cuts(identity, config, count, lambda at: _smallest(self.runs, at)))
        if self.kind == "text":
            if dtype.name != "string":
                raise ValueError(f"encoder {identity}: Text requires string dtype")
            texts, counts = self._texts()
            pairs = zip(texts.to_pylist(), counts.tolist(), strict=True)
            tokenizer = learned_tokenizer(config, (text for text, times in pairs for _ in range(times)))
            return GpuEncoder(identity, config, dtype, tokenizer_json=tokenizer.to_str())
        return self._vocabulary()

    def _texts(self) -> tuple[Any, Any]:
        """Each distinct text given and how often, merged across parts (host: Arrow, numpy)."""

        import numpy as np
        import pyarrow as pa

        if isinstance(self.counts, ValueCounts) and self.counts:
            return self.counts.arrays()
        if not self.counts:
            return pa.array([], self.dtype.to_arrow()), np.zeros(0, dtype=np.int64)
        table = pa.table({"text": pa.concat_arrays([texts for texts, _ in self.counts]),
                          "count": pa.array(np.concatenate([counts for _, counts in self.counts]), pa.int64())})
        merged = table.group_by("text").aggregate([("count", "sum")])
        return merged.column("text").combine_chunks(), merged.column("count_sum").to_numpy()

    def _vocabulary(self) -> GpuEncoder:
        """The Vocabulary of every key given, from how often each occurs (texts: as codes into their sorted
        dictionary, as the whole dataset's fit has them)."""

        from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
            _view,
            counted_vocabulary,
        )

        import cupy
        import numpy as np

        identity, config, dtype = self.identity, self.config, self.dtype
        is_text_counts = (self.counts.arrow if isinstance(self.counts, ValueCounts) else
                          bool(self.counts) and not isinstance(self.counts[0][0], np.ndarray))
        if self.counts and is_text_counts:
            import pylibcudf as plc

            texts, counts = self._texts()
            column = arrow_column(texts)
            order = plc.sorting.sorted_order(plc.Table([column]), [plc.types.Order.ASCENDING],
                                             [plc.types.NullOrder.AFTER])
            dictionary = plc.copying.gather(plc.Table([column]), order,
                                            plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
            return counted_vocabulary(identity, config, dtype, cupy.arange(len(texts), dtype=cupy.int32),
                                      cupy.asarray(counts)[_view(order).astype(cupy.int64)], dictionary)
        if not self.counts:
            empty = cupy.zeros(0, dtype=cupy.int64)
            return counted_vocabulary(identity, config, dtype, empty, empty, None)
        if isinstance(self.counts, ValueCounts):
            distinct, counts = self.counts.arrays()
            return counted_vocabulary(identity, config, dtype, cupy.asarray(distinct), cupy.asarray(counts), None)
        keys = np.concatenate([keys for keys, _ in self.counts])
        distinct, inverse = np.unique(keys, return_inverse=True)
        counts = np.zeros(distinct.size, dtype=np.int64)
        np.add.at(counts, inverse, np.concatenate([counts for _, counts in self.counts]))
        return counted_vocabulary(identity, config, dtype, cupy.asarray(distinct), cupy.asarray(counts), None)


def _merged(first: tuple[int, float, float], second: tuple[int, float, float]) -> tuple[int, float, float]:
    """Two parts' (count, mean, sum of squared deviations) as one (Chan et al.'s parallel update)."""

    (n_a, mean_a, m2_a), (n_b, mean_b, m2_b) = first, second
    if not n_a:
        return second
    count = n_a + n_b
    delta = mean_b - mean_a
    return count, mean_a + delta * n_b / count, m2_a + m2_b + delta * delta * n_a * n_b / count


def _keys(numbers: Any) -> Any:
    """float64 values (CuPy) as uint64 keys in the same order (-0.0 just before 0.0; no NaN reaches here)."""

    import cupy

    bits = numbers.view(cupy.uint64)
    return cupy.where(bits >> cupy.uint64(63), ~bits, bits | cupy.uint64(_TOP))


def _values(keys: Any) -> Any:
    """`_keys` back to float64 values (numpy)."""

    import numpy as np

    keys = np.asarray(keys, dtype=np.uint64)
    bits = np.where(keys >> np.uint64(63), keys & np.uint64(_TOP - 1), ~keys)
    return bits.view(np.float64)


def _run_size(runs: Any) -> int:
    return runs.size if isinstance(runs, NumericRuns) else sum(int(run.size) for run in runs)


def _smallest(runs: Any, positions: Any) -> Any:
    """The values at `positions` (0-based) of all `runs`' keys sorted together: for each, the smallest key with
    more than `position` keys at or below it, by binary search over the keys (every run sorted)."""

    import numpy as np

    positions = np.asarray(positions, dtype=np.int64)
    if isinstance(runs, NumericRuns):
        return _values(runs.select(positions))
    if not positions.size:
        return np.zeros(0, dtype=np.float64)
    low = np.full(positions.size, min(run[0] for run in runs), dtype=np.uint64)
    high = np.full(positions.size, max(run[-1] for run in runs), dtype=np.uint64)
    while (low < high).any():
        middle = low + (high - low) // np.uint64(2)
        below = sum(np.searchsorted(run, middle, side="right") for run in runs)
        above = below > positions
        high = np.where(above, middle, high)
        low = np.where(above, low, middle + np.uint64(1))
    return _values(low)
