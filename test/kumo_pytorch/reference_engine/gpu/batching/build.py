# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Building a task's batches on the GPU: inputs, graphs, masked tokens and answers.

Fitting a task's inputs and building its batches on the GPU.
What is built so far: inputs read from the sample's entity values (``EntitySource``,
`test.kumo_pytorch.reference_engine.gpu.sampling`) as separate typed columns (``Columns``, ``Separate`` layout);
event sequences (``EventSource``, ``EventsOf``, `test.kumo_pytorch.reference_engine.gpu.sampling.events`) as columns per
event, padded or packed; entity rows or events as tokens in a final ID space
(``Tokens``, `test.kumo_pytorch.reference_engine.gpu.encoding.tokens`), grouped per event (``Nested``) or flattened per
sample (``Flatten``); entity rows or events as one outer ID each
(``Composite``, `test.kumo_pytorch.reference_engine.gpu.encoding.composite`), padded, packed or flattened; and graph
neighborhoods (``Graph``, `test.kumo_pytorch.reference_engine.gpu.sampling.graphs`), sampled a few batches at a time. Fields use
Raw, Vocabulary, ZScore or Buckets, their own or shared (``UseEncoding``).
Targets (labels, fields aligned with an event input, matching groups;
`test.kumo_pytorch.reference_engine.gpu.encoding.targets`) come beside the inputs in ``targets``. A batch has upstream Kumo PyTorch's shape:
``inputs[input][field] = {"values": ..., "valid": ...}`` with upstream's dtypes,
and upstream's per-sample metadata. Tensors are PyTorch tensors on the GPU,
handed over through DLPack without a copy; Raw text, timestamp, date and decimal
fields are Arrow arrays, as upstream's are (inspectable output only).

A `batches()` call works out every requested anchor at once, in one pass, and
then slices the results into batches: the entity rows are found through a key
index built once per plan, every field is encoded for all rows, and every check
is read back once per call.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from test.kumo_pytorch.reference_engine.gpu.data.anchors import (
    batch_entity_ids,
)
from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
    GpuEncoder,
    encoder_identity,
    graph_identity,
)
from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.encoding import FittedSpace
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import CompositeVocabulary
    from test.kumo_pytorch.reference_engine.gpu.split.parts import SampledPart
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import TargetState

from test.kumo_pytorch.reference_engine.gpu.batching import \
    inputs as call_inputs
from test.kumo_pytorch.reference_engine.gpu.batching.compose import (
    BatchPlan,
    _lengths,
)
from test.kumo_pytorch.reference_engine.gpu.batching.inputs import (
    _config,
    _inputs,
)
from test.kumo_pytorch.reference_engine.gpu.batching.layout import (
    _flat_batch,
    _kernel,
    _laid_out,
    _Lists,
    _named,
    _Nested,
    _offsets,
    _Pieces,
    _projected,
    _raw,
    _sample_ids,
    _sequence,
    _slice,
)
from test.kumo_pytorch.reference_engine.gpu.sampling.indexes import EntityIndex

__all__ = ['build_batches']


def build_batches(contract: DataLoaderSpec, sampler: Sampler, index: EntityIndex, encoders: Mapping[str, GpuEncoder],
                  composites: Mapping[str, CompositeVocabulary], spaces: Mapping[str, FittedSpace],
                  target_state: TargetState, anchors: GpuAnchors, rows: Any, padding: Any, *, maximum: int | None,
                  drop_last: bool, model: bool, fingerprint: str, artifact: str = "", targets: bool = True,
                  seed: int = 0, epoch: int = 0, chunk: int = 8, plan: BatchPlan | None = None,
                  measure: str | None = None, field_maps: dict[str, dict[tuple[str, ...], int]] | None = None,
                  parts: Sequence[SampledPart] | None = None, first: int = 0) -> Iterator[Any]:
    """Batches over the anchors at `rows`, in that order (padding rows marked): as `plan` cuts them, else
    `maximum` samples each (the last dropped if short and `drop_last`); with their targets unless `targets` is
    false (inference: no target, so no masking or shift either). `seed` and `epoch` seed masked-value targets.
    Graph inputs are sampled `chunk` batches at a time (a neighborhood grows with its fanouts); everything
    else for every batch at once. With `measure` (an input's name), yields only that input's `lengths`.
    `fingerprint` and `artifact` are the contract's and the fitted state's (its manifest's) fingerprints, which
    every batch's metadata carries. An input naming a field-name dictionary (`field_maps`) gives every value
    and token its field's ID (``field_ids``, ``field_valid``; `test.kumo_pytorch.reference_engine.gpu.encoding.field_names`). With `parts`, the anchors
    at `rows` were sampled already, part by part against different tables (`test.kumo_pytorch.reference_engine.gpu.split.parts.sample_part`, one
    part after another in `rows`' order): their values, labels and graphs are joined instead of sampled, and
    `sampler` and `index` are not read; the call is then one graph chunk. `first` is the place of `rows`' first
    anchor in a call built a group at a time (`test.kumo_pytorch.reference_engine.gpu.plan.groups._in_groups`), so that a padding row's sample ID is
    the one the whole call would give it."""

    from dataclasses import replace
    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        Component,
        component,
        component_specs,
        composite_identity,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _raise_flags,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.field_names import (
        field_lookup,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import (
        encode_targets,
        next_tokens,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.tokens import (
        Flat,
        TokenField,
        TokenRows,
        flatten,
        per_sample,
        token_rows,
        translated,
    )

    import cupy
    import torch

    from kumo.pytorch.contract.batch import (
        AlignedEventTarget,
        Composite,
        EntitySource,
        Flatten,
        Graph,
        MaskedValuesTarget,
        NextTokenTarget,
        OccurrencesOf,
        StructuredTarget,
        Tokens,
    )
    from kumo.pytorch.contract.batch.encodings import Raw, Text
    from kumo.pytorch.contract.batch.spec import Batch

    specs = contract.batch.inputs
    structured = {name for name, target in contract.batch.targets.items() if isinstance(target, StructuredTarget)}
    masks = {target.input: (name, target) for name, target in contract.batch.targets.items()
             if isinstance(target, MaskedValuesTarget)}
    shifts = {target.input: (name, target) for name, target in contract.batch.targets.items()
              if isinstance(target, NextTokenTarget)}
    aligned = {name: target.input for name, target in contract.batch.targets.items()
               if isinstance(target, AlignedEventTarget)}
    graph_inputs = [name for name, value in specs.items() if isinstance(value.representation, Graph)]
    if plan is None:
        plan = BatchPlan.fixed(int(rows.size), maximum, drop_last)
    with cupy.cuda.Stream.from_external(torch.cuda.current_stream()):  # the caller's stream
        joined_graphs: dict[str, Any] = {}
        if parts is None:
            sampled, labels = call_inputs._sampled(contract, sampler, index, anchors, rows, labels=measure is None)
        else:
            from test.kumo_pytorch.reference_engine.gpu.split.parts import (
                join_parts,
            )

            if measure is not None or sum(part.size for part in parts) != int(rows.size):
                raise ValueError("sampled parts serve a whole batch call, as many anchors as its rows")
            sampled, labels, joined_graphs = join_parts(parts)
        bounds = {name: _offsets(value.lengths.get()) for name, value in sampled.items()
                  if value.lengths is not None}
        flags: list[tuple[Any, Any]] = []
        lookups = {name: field_lookup(contract, name, field_maps or {}, spaces) for name in specs
                   if getattr(specs[name].representation, "field_names", None) is not None}
        columns: dict[str, dict[str, dict[str, Any]]] = {}
        token_fields: dict[str, list[TokenField]] = {}
        components: dict[str, list[Component]] = {}
        for name, value in specs.items():
            if isinstance(getattr(value.representation, "row", None), Composite):
                components[name] = [component(spec, sampled[name].values[spec.field], None if spec.raw else
                                              spec.encoder.encode(sampled[name].values[spec.field], flags), flags)
                                    for spec in component_specs(value.representation, name, encoders)]
        for name, use in _inputs(contract):
            if name in components or isinstance(specs[name].source, OccurrencesOf):
                continue
            key = encoder_identity(name, use)
            values = sampled[name].values[use.field]
            dated = sampled[name].dated.get(use.field)  # a list per row: every occurrence
            if isinstance(_config(contract, use), Raw):
                raw = _raw(key, values, model)
                columns.setdefault(name, {})[use.field] = _Lists(raw["values"], raw["valid"], dated.bounds) \
                    if dated is not None else raw
                _named(columns[name], use.field, lookups.get(name), specs[name], sampled[name])
                continue
            pieces = None
            if isinstance(_config(contract, use), Text):
                result, valid, pieces = encoders[key].pieces(values)
            else:
                result, valid = encoders[key].encode(values, flags)
            if use.id_space:
                result = translated(spaces[use.id_space], key, result)
            if isinstance(specs[name].representation.row, Tokens):
                token_fields.setdefault(name, []).append(
                    TokenField(use.field, result, valid, sampled[name].declared.get(use.field), pieces,
                               None if dated is None else cupy.asarray(dated.bounds),
                               None if dated is None else dated.times))
            elif dated is not None:
                columns.setdefault(name, {})[use.field] = _Lists(
                    torch.from_dlpack(result), torch.from_dlpack(valid), dated.bounds,
                    None if pieces is None else pieces.get())
            elif pieces is not None:
                columns.setdefault(name, {})[use.field] = _Pieces.of(result, valid, pieces)
            else:
                columns.setdefault(name, {})[use.field] = {"values": torch.from_dlpack(result),
                                                           "valid": torch.from_dlpack(valid)}
            if name in columns and use.field in columns[name]:
                _named(columns[name], use.field, lookups.get(name), specs[name], sampled[name])
        sequences: dict[str, Any] = {}
        parents: dict[str, Any] = {}  # entity token inputs' token rows and row bounds
        produced: dict[str, _Answers] = {}  # masked-value and next-token targets, by target name
        # each sample's entity, as masked-value draws read it: only for them, as upstream works it out only for
        # them (a uint64 ID of 2**63 or more has none: upstream's int64 tensor of them overflows)
        entities = anchors.mask_keys(rows) if targets and masks else None
        for name in specs:
            representation = specs[name].representation
            if name not in token_fields and name not in components:
                continue
            row_bounds = bounds[name]
            total = int(row_bounds[-1])
            if name in components:  # one outer ID per row
                row = representation.row
                ids = composites[name].lookup(components[name])
                if row.id_space:
                    ids = translated(spaces[row.id_space], composite_identity(name), ids)
                tokens = TokenRows(ids, cupy.arange(total + 1, dtype=cupy.int64), cupy.arange(total, dtype=cupy.int64),
                                   positions=cupy.zeros(total, dtype=cupy.int64))
                space = spaces.get(row.id_space)
            else:
                owner = cupy.searchsorted(cupy.asarray(row_bounds[1:]), cupy.arange(total), side="right")
                tokens = token_rows(name, representation.row, token_fields[name], total, owner, plan.cuts, flags)
                if name in lookups:  # each token's field: its slot's, on its row's component
                    found = sampled[name].components
                    tokens.fields = lookups[name][tokens.slots, 0 if found is None else found[tokens.rows]]
                parents[name] = (tokens, row_bounds)  # unmasked: what `OccurrencesOf` projects onto
                space = spaces[representation.fields[0].id_space]
            pad = space.roles.get("pad", 0) if space is not None else 0
            masked = masks.get(name) if targets else None
            flat = isinstance(representation.history, Flatten)
            if masked is not None and not flat:  # masked as rows, before they are laid out
                place = cupy.arange(tokens.values.size, dtype=cupy.int64) - tokens.bounds[tokens.rows]
                values, arrays = _masked(masked[1], tokens.values, tokens.slots, tokens.rows, place, owner, row_bounds,
                                         entities, representation, space, seed, epoch)
                tokens = replace(tokens, values=values)
            if flat:
                built = flatten(tokens, row_bounds, representation.history, space.roles)
                if masked is not None:  # the tokens each sample keeps, each counted among its event's kept ones
                    real = cupy.flatnonzero(built.sources >= 0)
                    source = built.sources[real]
                    kept_rows = tokens.rows[source]
                    starts = cupy.ones(real.size, dtype=bool)
                    starts[1:] = kept_rows[1:] != kept_rows[:-1]
                    runs = cupy.flatnonzero(starts)
                    index = cupy.arange(real.size, dtype=cupy.int64)
                    place = index - runs[cupy.cumsum(starts) - 1] if real.size else index
                    values, kept = _masked(masked[1], built.values[real], tokens.slots[source], kept_rows, place, owner,
                                           row_bounds, entities, representation, space, seed, epoch)
                    built.values[real] = values
                    arrays = {}
                    for key, (array, fill) in kept.items():
                        whole = cupy.full(built.values.size, fill, dtype=array.dtype)
                        whole[real] = array
                        arrays[key] = (whole, fill)
                    produced[masked[0]] = _Answers(arrays, built.bounds, None, exists=False)
                if targets and name in shifts:
                    target_name, target = shifts[name]
                    keep, kept_bounds, (answers, answer_bounds) = next_tokens(built.values, built.bounds,
                                                                              target.prefix_input is not None)
                    produced[target_name] = _Answers({"values": (answers, -100)}, answer_bounds, None, exists=True)
                    built = Flat(built.values[keep], kept_bounds, built.rows[keep], built.sources[keep],
                                 built.positions[keep])
                sequences[name] = (built, pad, tokens.fields)
            elif isinstance(specs[name].source, EntitySource) or name in components:
                built = per_sample(tokens, row_bounds)
                if masked is not None:
                    produced[masked[0]] = _Answers(arrays, built.bounds, None, exists=False)
                sequences[name] = (built, pad, tokens.fields)
            else:
                sequences[name] = (_Nested.of(tokens, row_bounds, owner, plan.cuts), pad, None)
                if masked is not None:
                    produced[masked[0]] = _Answers(arrays, None, sequences[name][0], exists=False)
        for name, value in specs.items():  # occurrence fields over the parent's tokens
            if isinstance(value.source, OccurrencesOf):
                tokens, row_bounds = parents[value.source.input]
                bounds[name] = tokens.bounds[cupy.asarray(row_bounds)].get()
                columns[name] = {use.field: _raw(encoder_identity(name, use), _projected(
                    contract, name, use.field, tokens, sampled[value.source.input]), model)
                    for use in value.representation.fields}
        if measure is not None:
            _raise_flags(flags)
            yield _lengths(measure, sequences, bounds, int(rows.size))
            return
        encoded_targets = encode_targets(contract, labels, sampled, encoders, spaces, target_state, flags, model) \
            if targets else {}
        _raise_flags(flags)
        entity_ids = anchors.entity_metadata(rows)
        times = torch.from_dlpack(anchors.cutoffs[rows])
        is_padding = torch.from_dlpack(padding)
        # graphs grow from each anchor's entity row
        roots = index.find(anchors, rows)[0] if graph_inputs and parts is None else None
        cutoffs = anchors.cutoffs[rows]
    sample_ids, splits, metadata = anchors.request_fields(rows)
    sample_ids, request_metadata = _sample_ids(sample_ids, splits, metadata, padding.get().tolist(), first)
    count = len(sample_ids)
    graphs: dict[str, Any] = {}
    first = 0  # the first sample of the sampled graphs
    for number, (start, stop) in enumerate(plan.bounds[:plan.emit]):
        if number == plan.failure_at:
            raise ValueError(plan.failure)
        part = slice(start, stop)
        if graph_inputs and number % chunk == 0:
            first, last = start, plan.bounds[min(number + chunk, len(plan.bounds)) - 1][1]
            if parts is not None:  # sampled with the parts, for the whole call
                if (first, last) != (0, count):
                    raise ValueError("a call of sampled parts is one graph chunk")
                graphs = joined_graphs
            else:
                with cupy.cuda.Stream.from_external(torch.cuda.current_stream()):
                    graphs = _neighborhoods(contract, sampler, graph_inputs, roots[first:last], cutoffs[first:last],
                                            encoders, spaces, model, seed, epoch)
        inputs = {}
        for name in specs:
            layout = specs[name].layout
            if name in graphs:
                with cupy.cuda.Stream.from_external(torch.cuda.current_stream()):
                    inputs[name] = graphs[name].batch(start - first, stop - first)
            elif name in sequences:
                built, pad, fields = sequences[name]
                inputs[name] = built.batch(bounds[name], start, stop, layout, pad) if isinstance(built, _Nested) \
                    else _flat_batch(built, bounds[name], start, stop, layout, pad, fields)
            elif name in bounds:
                inputs[name] = _laid_out(
                    columns[name], bounds[name][start:stop + 1], layout,
                    pads={use.field: spaces[use.id_space].roles.get("pad", 0) for use in specs[name].representation.fields
                          if use.id_space},
                    field_pad=spaces[specs[name].representation.field_space].roles.get("pad", 0)
                    if specs[name].representation.field_space else 0)
            else:
                inputs[name] = {field: encoded.rows(start, stop) if isinstance(encoded, (_Pieces, _Lists)) else
                                {key: _slice(value, part) for key, value in encoded.items()}
                                for field, encoded in columns[name].items()}
        for name, output in inputs.items():
            field_space = getattr(specs[name].representation, "field_space", None)
            if field_space is not None:
                _field_padding(output, spaces[field_space].roles.get("pad", 0))
        batch_targets = {}
        for name, answers in produced.items():  # first, by input (upstream's order), then the others by name
            batch_targets[name] = answers.batch(bounds[contract.batch.targets[name].input], start, stop,
                                                specs[contract.batch.targets[name].input].layout)
        for name, output in encoded_targets.items():
            if name in aligned:  # laid out as its input is, without the input's own validity
                from test.kumo_pytorch.reference_engine.gpu.encoding.targets import (
                    target_uses,
                )
                pads = {use.field: spaces[use.id_space].roles.get("pad", 0)
                        for target_name, _, use in target_uses(contract) if target_name == name and use.id_space}
                laid = _laid_out(output, bounds[aligned[name]][start:stop + 1], specs[aligned[name]].layout, pads=pads)
                del laid["valid"]
                batch_targets[name] = laid
            elif name in structured:
                batch_targets[name] = {field: {key: _slice(value, part) for key, value in encoded.items()}
                                       for field, encoded in output.items()}
            else:
                batch_targets[name] = {key: _slice(value, part) for key, value in output.items()}
        yield Batch(inputs=inputs, targets=batch_targets, metadata={
            "sample_ids": tuple(sample_ids[part]),
            "entity_ids": batch_entity_ids(entity_ids, part),
            "as_of_time": times[part],
            "splits": tuple(splits[part]),
            "request_metadata": tuple(request_metadata[part]),
            "artifact_fingerprint": artifact,
            "contract_fingerprint": fingerprint,
            "is_padding": is_padding[part],
        })


@dataclass
class _GraphInput:
    """One graph input's sampled neighborhoods and its encoded node fields, for consecutive samples: per table,
    per field, ``values`` and ``valid`` of every node, sliced per batch."""

    graph: Any  # test.kumo_pytorch.reference_engine.gpu.sampling.graphs.Neighborhoods
    features: dict[str, dict[str, dict[str, Any]]]

    def batch(self, start: int, stop: int) -> dict[str, Any]:
        result = self.graph.batch(start, stop)
        for table, fields in self.features.items():
            nodes = slice(int(self.graph.node_bounds[table][start]), int(self.graph.node_bounds[table][stop]))
            result["nodes"][table]["features"] = {
                field: encoded.rows(nodes.start, nodes.stop) if isinstance(encoded, _Pieces) else
                {key: _slice(value, nodes) for key, value in encoded.items()} for field, encoded in fields.items()}
        return result


def _neighborhoods(contract: DataLoaderSpec, sampler: Sampler, names: list[str], roots: Any, cutoffs: Any,
                   encoders: Mapping[str, GpuEncoder], spaces: Mapping[str, FittedSpace], model: bool, seed: int,
                   epoch: int) -> dict[str, _GraphInput]:
    """Each graph input's sampled neighborhoods (`test.kumo_pytorch.reference_engine.gpu.sampling.graphs`) for samples grown from entity rows `roots`
    with their cutoffs (uniform draws seeded by `seed` and `epoch`), and its node fields encoded (as a column
    input's fields are); inputs reading one neighborhood with the same node tables share its sampling."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _raise_flags,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.tokens import (
        translated,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.graphs import (
        sample_neighborhoods,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _column_values,
    )

    import torch

    from kumo.pytorch.contract.batch.encodings import Raw, Text

    flags: list[tuple[Any, Any]] = []
    built: dict[tuple[str, tuple[str, ...]], Any] = {}
    result = {}
    for name in names:
        value = contract.batch.inputs[name]
        key = (value.source.name, tuple(sorted(value.representation.node_fields)))
        if key not in built:
            built[key] = sample_neighborhoods(sampler, contract.sample.neighborhoods[key[0]], key[1], roots, cutoffs,
                                              flags, seed=seed, epoch=epoch)
        graph, features = built[key], {}
        for table, group in value.representation.node_fields.items():
            features[table] = {}
            for use in group.fields:
                identity = graph_identity(name, table, use)
                values = _column_values(sampler.tables[table], use.field, graph.rows[table])
                config = _config(contract, use)
                if isinstance(config, Raw):
                    features[table][use.field] = _raw(identity, values, model)
                    continue
                pieces = None
                if isinstance(config, Text):
                    ids, valid, pieces = encoders[identity].pieces(values)
                else:
                    ids, valid = encoders[identity].encode(values, flags)
                if use.id_space:
                    ids = translated(spaces[use.id_space], identity, ids)
                features[table][use.field] = _Pieces.of(ids, valid, pieces) if pieces is not None else \
                    {"values": torch.from_dlpack(ids), "valid": torch.from_dlpack(valid)}
        result[name] = _GraphInput(graph, features)
    _raise_flags(flags)
    return result


def _masked(target: Any, values: Any, slots: Any, rows: Any, place: Any, owner: Any, row_bounds: Any, entities: Any,
            representation: Any, space: Any, seed: int, epoch: int) -> tuple[Any, dict[str, tuple[Any, Any]]]:
    """`test.kumo_pytorch.reference_engine.gpu.encoding.targets.mask_tokens` for tokens of event (or entity) `rows`, each event's sample and place in
    it found from `owner` and `row_bounds`."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import (
        mask_tokens,
    )

    import cupy

    sample = owner[rows]
    history = rows - cupy.asarray(row_bounds)[sample]
    return mask_tokens(target, values, slots, sample, history, place, entities,
                       [use.field for use in representation.fields], space, seed=seed, epoch=epoch)


@dataclass
class _Answers:
    """A masked-value or next-token target's arrays for a call (CuPy, with their padding fills), laid out per
    batch as its input's tokens are: per sample (`bounds`, host), or per event of a grouped input (`nested`).
    `exists` adds upstream's ``valid`` of the positions that exist (next token)."""

    arrays: dict[str, tuple[Any, Any]]
    bounds: Any
    nested: _Nested | None
    exists: bool

    def batch(self, row_bounds: Any, start: int, stop: int, layout: Any) -> dict[str, Any]:
        import cupy
        import numpy as np

        from kumo.pytorch.contract.batch.layouts import Packed

        if self.nested is not None:  # tokens per event of the batch, as `_Nested.batch` lays them out
            number = int(np.searchsorted(self.nested.cuts, start, side="left"))
            first, last = int(row_bounds[start]), int(row_bounds[stop])
            begin, end = int(self.nested.starts[number]), int(self.nested.starts[number + 1])
            offsets = self.nested.tokens.bounds[first:last + 1] - begin
            count, width, layout = last - first, int(self.nested.longest[number]), layout.fields
        else:
            begin, end = int(self.bounds[start]), int(self.bounds[stop])
            local = self.bounds[start:stop + 1] - begin
            offsets, count, width = cupy.asarray(local), stop - start, int(np.diff(local).max(initial=0))
        arrays = {key: (array[begin:end], fill) for key, (array, fill) in self.arrays.items()}
        result = _sequence(arrays, offsets, end - begin, count, width, layout, exists=self.exists)
        if isinstance(layout, Packed):
            _kernel(result, "", end - begin, width if count else 0)
        return result


def _field_padding(output, pad):
    """Apply the declared field-space pad to markers and padded rows."""
    if not isinstance(output, dict):
        return
    if "field_ids" in output and "field_valid" in output:
        output["field_ids"] = output["field_ids"].masked_fill(~output["field_valid"], pad)
    for value in output.values():
        if isinstance(value, dict):
            _field_padding(value, pad)
