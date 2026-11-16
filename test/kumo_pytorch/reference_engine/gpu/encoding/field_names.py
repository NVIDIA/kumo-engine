"""Field-name IDs (``FieldNames``): which declared field each value or token came from.

The rules are upstream Kumo PyTorch's (G-IDSPACE-001; its ``_fit_inputs``, ``_field_ids``
and ``_allocate_spaces``). A ``FieldNames`` dictionary numbers every field
identity the inputs naming it declare, from 1 in sorted order: an entity
field's is ``("entity", name)``, an event field's ``("events", component,
name)``, one per component that declares it. Its members come from the
declarations, not from rows, so nothing is fitted. With a ``field_space`` the
dictionary joins that final ID space as one member (``["fields", name]``, 0 its
pad). Each value of a column input and each token of a token input then carries
its field's ID (``field_ids``; an event field's by its event's component, 0 on
an event of a component that does not declare it) and whether it has one
(``field_valid``; a marker, a token no field owns, does not).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kumo.pytorch.encoding import FittedSpace
    from kumo.pytorch.spec import DataLoaderSpec

__all__ = ["field_lookup", "field_maps", "fields_identity"]


def fields_identity(dictionary: str) -> str:
    """A field-name dictionary's identity as an ID-space member, as upstream spells it."""

    import json

    return json.dumps(("fields", dictionary), ensure_ascii=False, separators=(",", ":"))


def field_maps(contract: DataLoaderSpec) -> dict[str, dict[tuple[str, ...], int]]:
    """Every ``FieldNames`` dictionary the batch declares: each field identity the inputs naming it declare,
    numbered from 1 in sorted order (a dictionary no input names: empty)."""

    from kumo.pytorch.contract.batch import Fields
    from kumo.pytorch.contract.batch.validation import resolved_fields

    found: dict[str, set[tuple[str, ...]]] = {name: set() for name in contract.batch.field_names}
    for name, value in contract.batch.inputs.items():
        group = value.representation
        if isinstance(group, Fields) and group.field_names is not None:
            for use in group.fields:
                found[group.field_names].update(tuple(identity) for identity, _, _ in
                                                resolved_fields(contract, name, use.field))
    return {name: {identity: index + 1 for index, identity in enumerate(sorted(identities))}
            for name, identities in found.items()}


def field_lookup(contract: DataLoaderSpec, input_name: str, maps: dict[str, dict[tuple[str, ...], int]],
                 spaces: dict[str, FittedSpace]) -> Any | None:
    """An input's field-name IDs, None if it names no dictionary: for each of its fields (rows, in declared
    order), the ID on an event of each component of its source (columns, in the source's order; an entity
    input: one), 0 where the component does not declare the field, translated into its ``field_space``."""

    import cupy
    from kumo.pytorch.contract.batch import EventsOf
    from kumo.pytorch.contract.batch.validation import resolved_fields

    value = contract.batch.inputs[input_name]
    group = value.representation
    if getattr(group, "field_names", None) is None:
        return None
    source = value.source
    if isinstance(source, EventsOf):
        source = contract.batch.inputs[source.input].source
    components = list(getattr(source, "events", ()) or ())
    numbering = maps[group.field_names]
    space = spaces.get(group.field_space) if group.field_space is not None else None
    member = None if space is None else space.members[fields_identity(group.field_names)]
    table = [[0] * max(len(components), 1) for _ in group.fields]
    for rank, use in enumerate(group.fields):
        for identity, _, _ in resolved_fields(contract, input_name, use.field):
            index = numbering[tuple(identity)]
            if member is not None:
                index = member[index]
            if identity[0] in {"events", "current_events"}:
                table[rank][components.index(identity[-2])] = index
            else:
                table[rank] = [index] * len(table[rank])
    return cupy.asarray(table, dtype=cupy.int64)
