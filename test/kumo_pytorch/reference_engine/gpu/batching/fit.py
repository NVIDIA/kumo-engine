# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fitting a task's input encoders on the GPU from its fit split's anchors."""

from __future__ import annotations

from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
    GpuEncoder,
    Values,
    concatenated,
    encoder_identity,
    fit_encoder,
    graph_identity,
)
from test.kumo_pytorch.reference_engine.gpu.sampling.sampler import Sampler
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.encoding import FittedSpace
    from kumo.pytorch.spec import DataLoaderSpec

    from test.kumo_pytorch.reference_engine.gpu.data.anchors import GpuAnchors
    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import CompositeVocabulary
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import TargetState

from test.kumo_pytorch.reference_engine.gpu.batching import \
    inputs as call_inputs
from test.kumo_pytorch.reference_engine.gpu.batching.inputs import (
    _config,
    _inputs,
)
from test.kumo_pytorch.reference_engine.gpu.sampling.indexes import EntityIndex

__all__ = ['fit_inputs']


#: Anchors whose neighborhoods are sampled at once when fitting node fields (memory grows with the fanouts).
GRAPH_FIT_ANCHORS = 1 << 15


def fit_inputs(contract: DataLoaderSpec, sampler: Sampler, index: EntityIndex, anchors: GpuAnchors, rows: Any,
               reused: tuple[dict[str, GpuEncoder], dict[str, CompositeVocabulary], dict[str, FittedSpace]] | None = None,
               field_maps: dict[str, dict[tuple[str, ...], int]] | None = None,
               pinned: dict[str, GpuEncoder] | None = None) -> tuple[dict[str, GpuEncoder], dict[str, CompositeVocabulary], dict[str, FittedSpace], TargetState]:
    """Fit every encoder and Composite input on the anchors at `rows` (the fit split's), and number every ID
    space.

    An encoder is fitted over its field's values for those anchors, or over every row an event or token
    input keeps for them (a Composite input: every row, before its required fields drop any); a shared
    encoder (``UseEncoding``) over every field that uses it; NULLs excluded. A graph input's node field: over
    every node of those anchors' neighborhoods (a row in several samples' counts in each), as upstream fits
    it. A Composite input's outer IDs
    are then fitted on the rows it keeps (`test.kumo_pytorch.reference_engine.gpu.encoding.composite`), and the targets (`test.kumo_pytorch.reference_engine.gpu.encoding.targets`): thresholds,
    matching groups, and each target field's encoder.

    With `reused` (a prior task's input state: encoders, Composite IDs and ID spaces, applied unchanged), only
    the targets are fitted; an ID space the inputs reuse cannot gain a target's member (upstream's
    ``_fit_targets``). `field_maps` (`test.kumo_pytorch.reference_engine.gpu.encoding.field_names`) join their field spaces.
    """

    from test.kumo_pytorch.reference_engine.gpu.encoding.composite import (
        component,
        component_specs,
        fit_composite,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _raise_flags,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.targets import (
        fit_targets,
    )
    from test.kumo_pytorch.reference_engine.gpu.encoding.tokens import (
        allocate_spaces,
    )

    from kumo.pytorch.contract.batch import Composite
    from kumo.pytorch.contract.batch.encodings import Buckets, Raw, ZScore

    sampled, labels = call_inputs._sampled(contract, sampler, index, anchors, rows)
    if reused is not None:
        encoders, composites, kept = dict(reused[0]), reused[1], reused[2]
        state = fit_targets(contract, labels, sampled, encoders)
        spaces = allocate_spaces(contract, encoders, composites, field_maps)
        for name, space in kept.items():
            if any(identity not in space.members for identity in spaces[name].members):
                raise ValueError(f"reused ID space {name!r} cannot acquire target members")
        return encoders, composites, {**spaces, **kept}, state
    contributions: list[tuple[str, str, Values]] = []  # (input, encoder, values)
    configs = {}
    for name, use in _inputs(contract):
        key = encoder_identity(name, use)
        source = sampled[name].fitting if sampled[name].fitting is not None else sampled[name].values
        contributions.append((name, key, source[use.field]))
        configs[key] = _config(contract, use)
    for name, table, use, values in _node_values(contract, sampler, index, anchors, rows):
        key = graph_identity(name, table, use)
        contributions.append((name, key, values))
        configs[key] = _config(contract, use)
    parts: dict[str, list[Values]] = {}
    # upstream's order: inputs by name, each its fields in order (a learned Text tokenizer reads them in it)
    for name, key, values in sorted(contributions, key=lambda item: item[0]):
        parts.setdefault(key, []).append(values)
    encoders = dict(pinned or {})
    for key in sorted(parts):
        if key in encoders:
            continue
        config, dtype = configs[key], parts[key][0].dtype
        if any(part.values is None for part in parts[key]):  # decimals: Raw, or a numeric transform of numbers
            if isinstance(config, Raw):
                encoders[key] = GpuEncoder(key, config, dtype)
                continue
            if not isinstance(config, (ZScore, Buckets)) or any(part.numbers is None for part in parts[key]):
                raise ValueError(f"encoder {key}: test.kumo_pytorch.reference_engine.gpu does not encode {dtype.name} values "
                                 f"with {type(config).__name__} yet")
        encoders[key] = fit_encoder(key, config, dtype, concatenated(parts[key]))
    composites = {}
    for name, value in contract.batch.inputs.items():
        row = getattr(value.representation, "row", None)
        if isinstance(row, Composite):
            specs = component_specs(value.representation, name, encoders)
            if row.vocabulary is not None:
                composites[name] = fit_composite(specs, row, [])
                continue
            flags: list[tuple[Any, Any]] = []
            results = [component(spec, sampled[name].values[spec.field],
                                 None if spec.raw else spec.encoder.encode(sampled[name].values[spec.field], flags),
                                 flags) for spec in specs]
            _raise_flags(flags)
            composites[name] = fit_composite(specs, row, results)
    state = fit_targets(contract, labels, sampled, encoders)
    return encoders, composites, allocate_spaces(contract, encoders, composites, field_maps), state


def _node_values(contract: DataLoaderSpec, sampler: Sampler, index: EntityIndex, anchors: GpuAnchors,
                 rows: Any) -> list[tuple[str, str, Any, Values]]:
    """Every graph input's node-field values over the neighborhoods of the anchors at `rows`: (input, table,
    field use, values of every node of every sample), the neighborhoods sampled `GRAPH_FIT_ANCHORS` anchors at a
    time, uniform draws with seed 0 and epoch 0 (as upstream fits)."""

    from test.kumo_pytorch.reference_engine.gpu.encoding.encoders import (
        _raise_flags,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.graphs import (
        sample_neighborhoods,
    )
    from test.kumo_pytorch.reference_engine.gpu.sampling.reductions import (
        _column_values,
    )

    import cupy

    from kumo.pytorch.contract.batch import Graph

    graphs = [(name, value) for name, value in contract.batch.inputs.items()
              if isinstance(value.representation, Graph) and value.representation.node_fields]
    if not graphs:
        return []
    roots, cutoffs = index.find(anchors, rows)[0], anchors.cutoffs[rows]
    nodes: dict[tuple[str, str], list[Any]] = {}
    for first in range(0, int(rows.size), GRAPH_FIT_ANCHORS):
        part = slice(first, first + GRAPH_FIT_ANCHORS)
        flags: list[tuple[Any, Any]] = []
        built = {}
        for name, value in graphs:
            key = (value.source.name, tuple(sorted(value.representation.node_fields)))
            if key not in built:
                built[key] = sample_neighborhoods(sampler, contract.sample.neighborhoods[key[0]], key[1], roots[part],
                                                  cutoffs[part], flags)
            for table in value.representation.node_fields:
                nodes.setdefault((name, table), []).append(built[key].rows[table])
        _raise_flags(flags)
    result = []
    for name, value in graphs:
        for table, group in value.representation.node_fields.items():
            found = nodes.get((name, table), [])
            table_rows = cupy.concatenate(found) if found else cupy.zeros(0, dtype=cupy.int64)
            for use in group.fields:
                result.append((name, table, use, _column_values(sampler.tables[table], use.field, table_rows)))
    return result
