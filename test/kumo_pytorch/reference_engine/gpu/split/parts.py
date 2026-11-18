# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A batch whose samples are read from different tables: sampled part by part, then joined.

On a split dataset (`test.kumo_pytorch.reference_engine.gpu.split.plan`) a batch at a partition boundary holds
the last anchors of one partition and the first of the next (or of several, when
a partition holds fewer anchors than a batch). Each part is sampled while its
partition is on the GPU (`sample_part`): every value, label and graph its anchors
read, before anything is encoded. When the batch is complete, the parts are
joined (`join_parts`) and the batch is built from them as from any call's
sampled values (`test.kumo_pytorch.reference_engine.gpu.batching.build.build_batches` with ``parts=``), so its layout,
padding and encoding are the ones every other batch gets.

A sampled value depends only on its anchor and the rows its routes reach, never
on the other anchors of its call (uniform draws and masks are seeded per anchor),
so joining parts gives exactly what sampling all of them at once gives. A part
keeps no reference to its partition's tables: a text column's dictionary is cut
down to the texts the part uses (`_detached`), so the partition is freed when
the next one is read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        GpuEncoder,
        Values,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.indexes import (
        EntityIndex,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler

    from kumo.pytorch.spec import DataLoaderSpec

__all__ = ["SampledPart", "join_parts", "sample_part"]


@dataclass
class SampledPart:
    """Consecutive anchors of a call, sampled against one set of tables: each input's values
    (`test.kumo_pytorch.reference_engine.gpu.batching.inputs._InputValues`), each sample target's labels, and each graph input's neighborhoods with
    its node fields encoded (`test.kumo_pytorch.reference_engine.gpu.batching.build._GraphInput`)."""

    size: int
    sampled: dict[str, Any]
    labels: dict[str, Any]
    graphs: dict[str, Any]


def sample_part(contract: DataLoaderSpec, sampler: Sampler, index: EntityIndex, anchors: GpuAnchors, rows: Any,
                encoders: Mapping[str, GpuEncoder], spaces: Mapping[str, Any], *, model: bool, seed: int,
                epoch: int, labels: bool = True) -> SampledPart:
    """The anchors at `rows`, sampled against `sampler`'s tables as `test.kumo_pytorch.reference_engine.gpu.batching.build.build_batches` samples a
    call (its checks raised here), and detached from those tables."""

    from test.kumo_pytorch.reference_engine.gpu.batching.build import (
        _neighborhoods,
    )
    from test.kumo_pytorch.reference_engine.gpu.batching.inputs import _sampled

    import cupy
    import torch

    from kumo.pytorch.contract.batch import Graph

    graph_inputs = [name for name, value in contract.batch.inputs.items() if isinstance(value.representation, Graph)]
    with cupy.cuda.Stream.from_external(torch.cuda.current_stream()):  # the caller's stream
        sampled, found = _sampled(contract, sampler, index, anchors, rows, labels=labels)
        graphs = {}
        if graph_inputs:
            roots = index.find(anchors, rows)[0]
            graphs = _neighborhoods(contract, sampler, graph_inputs, roots, anchors.cutoffs[rows], encoders, spaces,
                                    model, seed, epoch)
        sampled = {name: _detached_input(value) for name, value in sampled.items()}
        found = {name: _detached_label(value) for name, value in found.items()}
    return SampledPart(int(rows.size), sampled, found, graphs)


def join_parts(parts: Sequence[SampledPart]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(sampled values, labels, graphs) of consecutive `parts`, as sampling their anchors at once gives them."""

    import cupy
    import torch

    sizes = [part.size for part in parts]
    with cupy.cuda.Stream.from_external(torch.cuda.current_stream()):
        sampled = {name: _join_input([part.sampled[name] for part in parts], sizes) for name in parts[0].sampled}
        labels = {name: _join_label([part.labels[name] for part in parts]) for name in parts[0].labels}
        graphs = {name: _join_graph_input([part.graphs[name] for part in parts], sizes) for name in parts[0].graphs}
    return sampled, labels, graphs


# -- values -----------------------------------------------------------------------------------

def _join_values(parts: Sequence[Values]) -> Values:
    """Several parts' values of one field, in order: texts in one dictionary, a decimal's numbers joined and
    its Arrow values made when asked."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        Values,
        recoded,
        shared_texts,
    )

    import cupy
    import pyarrow as pa

    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    present = None if all(part.present is None for part in parts) else \
        cupy.concatenate([part.valid() for part in parts])
    if first.values is None:  # a decimal: its numbers, and its Arrow values when something asks for them
        numbers = None if any(part.numbers is None for part in parts) else \
            cupy.concatenate([part.numbers for part in parts])
        kept = list(parts)
        return Values(first.dtype, None, present, numbers=numbers,
                      build=lambda: pa.concat_arrays([_whole(part.arrow()) for part in kept]))
    if all(part.texts is None for part in parts):
        return Values(first.dtype, cupy.concatenate([part.values for part in parts]), present)
    # a part without a dictionary holds only NULL texts: its codes say nothing, so 0 stands for each
    holders = [part for part in parts if part.texts is not None]
    texts, maps = shared_texts(holders)
    remaps = iter(maps)
    codes = [recoded(part.values, next(remaps)) if part.texts is not None else
             cupy.zeros(part.values.shape, dtype=part.values.dtype) for part in parts]
    return Values(first.dtype, cupy.concatenate(codes), present, texts)


def _detached(values: Values | None) -> Values | None:
    """`values` holding no reference to its table: a text dictionary cut down to the texts it uses (code 0's
    among them, so every code stays in range); anything else is already its own."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _column,
    )

    import cupy
    import pylibcudf as plc

    if values is None or values.texts is None or values.values is None or values.texts.size() == 0:
        return values
    codes = cupy.clip(values.values, 0, values.texts.size() - 1).astype(cupy.int32)
    used = cupy.unique(cupy.concatenate([codes[values.valid()], cupy.zeros(1, dtype=cupy.int32)]))
    texts = plc.copying.gather(plc.Table([values.texts]), _column(used),
                               plc.copying.OutOfBoundsPolicy.DONT_CHECK).columns()[0]
    moved = cupy.minimum(cupy.searchsorted(used, codes), used.size - 1)  # a NULL's code: any, still in range
    return replace(values, values=moved.astype(values.values.dtype), texts=texts)


def _detached_input(value: Any) -> Any:
    fitting = None if value.fitting is None else {key: _detached(item) for key, item in value.fitting.items()}
    dated = {key: replace(item, values=_detached(item.values),
                          fields={name: _detached(field) for name, field in item.fields.items()})
             for key, item in value.dated.items()}
    values = {key: dated[key].values if key in dated and item is value.dated[key].values else _detached(item)
              for key, item in value.values.items()}
    return replace(value, values=values, fitting=fitting, dated=dated)


def _detached_label(value: Any) -> Any:
    return {key: _detached(item) for key, item in value.items()} if isinstance(value, dict) else _detached(value)


def _join_label(parts: Sequence[Any]) -> Any:
    if isinstance(parts[0], dict):  # a row target's fields
        return {key: _join_values([part[key] for part in parts]) for key in parts[0]}
    return _join_values(parts)


def _bounds(parts: Sequence[Any]) -> Any:
    """Host offsets (each from 0) of consecutive parts, as one."""

    import numpy as np

    joined, total = [np.asarray(parts[0], dtype=np.int64)], int(parts[0][-1])
    for bounds in parts[1:]:
        joined.append(np.asarray(bounds[1:], dtype=np.int64) + total)
        total += int(bounds[-1])
    return np.concatenate(joined)


def _join_input(parts: Sequence[Any], sizes: Sequence[int]) -> Any:
    """One input's `_InputValues` of consecutive parts (`sizes` anchors each)."""

    import cupy

    first = parts[0]
    dated = {key: _join_occurrences([part.dated[key] for part in parts], sizes) for key in first.dated}
    values = {key: dated[key].values if key in dated and all(part.values[key] is part.dated[key].values
                                                             for part in parts)
              else _join_values([part.values[key] for part in parts]) for key in first.values}
    # which events' components declare a field: None in every part (all do: the contract says so) or in none
    declared = {key: None if mask is None else cupy.concatenate([part.declared[key] for part in parts])
                for key, mask in first.declared.items()}
    fitting = None if all(part.fitting is None for part in parts) else \
        {key: _join_values([part.fitting[key] for part in parts]) for key in first.fitting}
    lengths = None if first.lengths is None else cupy.concatenate([part.lengths for part in parts])
    components = None if first.components is None else cupy.concatenate([part.components for part in parts])
    return replace(first, lengths=lengths, values=values, declared=declared, fitting=fitting, dated=dated,
                   components=components)


def _join_occurrences(parts: Sequence[Any], sizes: Sequence[int]) -> Any:
    """A dated value's `test.kumo_pytorch.reference_engine.gpu.sampling.dated.Occurrences` of consecutive parts (`sizes` anchors each)."""

    import cupy

    offsets = [sum(sizes[:number]) for number in range(len(parts))]
    first = parts[0]
    return replace(first, owner=cupy.concatenate([part.owner + offset for part, offset in zip(parts, offsets,
                                                                                              strict=True)]),
                   values=_join_values([part.values for part in parts]),
                   times=cupy.concatenate([part.times for part in parts]),
                   fields={name: _join_values([part.fields[name] for part in parts]) for name in first.fields},
                   bounds=_bounds([part.bounds for part in parts]))


# -- graphs -----------------------------------------------------------------------------------

def _join_graph_input(parts: Sequence[Any], sizes: Sequence[int]) -> Any:
    """A graph input's `_GraphInput` of consecutive parts: their neighborhoods, then each table's node fields."""

    from test.kumo_pytorch.reference_engine.gpu.batching.build import (
        _GraphInput,
    )

    graph = _join_neighborhoods([part.graph for part in parts], sizes)
    features = {table: {field: _join_feature([part.features[table][field] for part in parts])
                        for field in fields} for table, fields in parts[0].features.items()}
    return _GraphInput(graph, features)


def _join_neighborhoods(parts: Sequence[Any], sizes: Sequence[int]) -> Any:
    """`test.kumo_pytorch.reference_engine.gpu.sampling.graphs.Neighborhoods` of consecutive parts: each part's samples, nodes and edges after the
    previous parts' (an edge's ends count among all samples' nodes of their table)."""

    import cupy

    first = parts[0]
    samples = [sum(sizes[:number]) for number in range(len(parts))]
    nodes = {table: [sum(int(part.node_bounds[table][-1]) for part in parts[:number]) for number in range(len(parts))]
             for table in first.tables}
    edges = {}
    for name, (left, right) in first.relations.items():
        edges[name] = (cupy.concatenate([part.edges[name][0] + nodes[left][number] for number, part in enumerate(parts)]),
                       cupy.concatenate([part.edges[name][1] + nodes[right][number]
                                         for number, part in enumerate(parts)]))
    return replace(
        first,
        owners={table: cupy.concatenate([part.owners[table] + offset for part, offset in zip(parts, samples,
                                                                                             strict=True)])
                for table in first.tables},
        ids={table: _join_ids([part.ids[table] for part in parts]) for table in first.tables},
        rows={table: cupy.concatenate([part.rows[table] for part in parts]) for table in first.tables},
        node_bounds={table: _bounds([part.node_bounds[table] for part in parts]) for table in first.tables},
        edges=edges,
        edge_bounds={name: _bounds([part.edge_bounds[name] for part in parts]) for name in first.relations},
    )


def _join_ids(parts: Sequence[Any]) -> Any:
    """Nodes' primary keys of consecutive parts (`test.kumo_pytorch.reference_engine.gpu.sampling.graphs.Neighborhoods.ids`): integers end to end, a
    text-keyed table's texts end to end, or keys of other types (`TypedIds`) so (a part with no node holds an
    empty integer array)."""

    from test.kumo_pytorch.reference_engine.gpu.sampling.graphs import TypedIds

    import cupy
    import pylibcudf as plc

    held = [part for part in parts if not isinstance(part, cupy.ndarray)]
    if not held:
        return cupy.concatenate(parts)
    if len(held) == 1:
        return held[0]
    if isinstance(held[0], TypedIds):
        return TypedIds.join(held)
    return plc.concatenate.concatenate(held)


def _join_feature(parts: Sequence[Any]) -> Any:
    """A node field's encoded values of consecutive parts: tensors (or Arrow arrays) end to end, or a Text
    field's pieces (`test.kumo_pytorch.reference_engine.gpu.batching.layout._Pieces`)."""

    from test.kumo_pytorch.reference_engine.gpu.batching.layout import _Pieces

    import pyarrow as pa
    import torch

    first = parts[0]
    if isinstance(first, _Pieces):
        tensor = (lambda key: None if getattr(first, key) is None else
                  torch.cat([getattr(part, key) for part in parts]))
        return _Pieces(tensor("values"), tensor("valid"), _bounds([part.offsets for part in parts]),
                       tensor("field_ids"), tensor("field_valid"))
    return {key: torch.cat([part[key] for part in parts]) if isinstance(value, torch.Tensor) else
            pa.concat_arrays([_whole(part[key]) for part in parts]) for key, value in first.items()}


def _whole(array: Any) -> Any:
    return array.combine_chunks() if hasattr(array, "combine_chunks") else array
