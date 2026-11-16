# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve selected typed fields and reject incompatible output declarations."""

from __future__ import annotations

import warnings
from typing import Any

from kumo.pytorch.contract.batch import (
    AlignedEventTarget,
    Buckets,
    CandidateSource,
    CandidateTarget,
    Composite,
    EntitySource,
    EventsOf,
    EventSource,
    Fields,
    Graph,
    Grouped,
    LabelTarget,
    MaskedValuesTarget,
    MatchingGroupTarget,
    Nested,
    NextTokenTarget,
    OccurrencesOf,
    Raw,
    StructuredTarget,
    Text,
    Tokens,
    UseEncoding,
    Vocabulary,
    ZScore,
)
from kumo.pytorch.contract.dtypes import DType
from kumo.pytorch.contract.sample import (
    Field,
    RowTarget,
    dated_value_types,
    destination_table,
    event_value_types,
    sample_value_types,
    value_dtype,
)


class InputExposureWarning(UserWarning):
    """A masked value is also selected by an unmasked named input.

    Emitted while checking the contract. The other input is not masked; the
    warning does not prove the absence of every indirect leak.
    """


def resolved_fields(
    contract: Any, input_name: str, field: str
) -> tuple[tuple[tuple[str, ...], DType, str], ...]:
    """Resolve a short/qualified selector into source identities, types, and shape."""

    value = contract.batch.inputs[input_name]
    source, sample, schema = value.source, contract.sample, contract.schema
    if isinstance(source, EventsOf):
        return resolved_fields(contract, source.input, field)
    if isinstance(source, OccurrencesOf):
        parent = contract.batch.inputs[source.input]
        types = dated_value_types(schema, sample)
        matches = []
        for use in parent.representation.fields:
            dated_name = use.field.removeprefix("entity/")
            if dated_name not in types:
                continue
            name = field.removeprefix(f"dated_entity/{dated_name}/")
            if name in types[dated_name]:
                matches.append(
                    (("dated_entity", dated_name, name), types[dated_name][name], "scalar")
                )
        if not matches:
            raise ValueError(f"unknown dated occurrence field {field!r}")
        return tuple(matches)
    if isinstance(source, EntitySource):
        name = field.removeprefix("entity/")
        types = sample_value_types(schema, sample)
        if name not in types:
            raise ValueError(f"unknown entity field {field!r}")
        dated = name in sample.dated_entity_values
        declaration = (
            sample.dated_entity_values[name].field if dated else sample.entity_values[name]
        )
        repeated = dated or isinstance(declaration, Field) and declaration.selection == "all"
        return (
            (
                ("dated_entity" if dated else "entity", name),
                types[name],
                "repeated" if repeated else "scalar",
            ),
        )
    if isinstance(source, CandidateSource):
        candidate = sample.candidate_sets.get(source.name)
        if candidate is None:
            raise ValueError(f"unknown candidate set {source.name!r}")
        candidate_sample = candidate.sample_spec()
        name = field.removeprefix(f"candidates/{source.name}/")
        types = sample_value_types(schema, candidate_sample)
        if name not in types:
            raise ValueError(f"unknown candidate field {field!r}")
        dtype = types[name]
        declaration = candidate.values[name]
        return (
            (
                ("candidates", source.name, name),
                dtype,
                "repeated"
                if isinstance(declaration, Field) and declaration.selection == "all"
                else "scalar",
            ),
        )
    if isinstance(source, EventSource):
        collection = "current_events" if source.collection == "current" else "events"
        events = getattr(sample, collection)
        types = event_value_types(schema, sample, current=source.collection == "current")
        matches = []
        for name in source.events:
            if name not in events:
                raise ValueError(f"unknown event component {name!r}")
            short = field.removeprefix(f"{collection}/{name}/")
            if short not in events[name].fields:
                continue
            declaration = events[name].fields[short]
            repeated = isinstance(declaration, Field) and declaration.selection == "all"
            matches.append(
                (
                    (collection, name, short),
                    types[name][short],
                    "repeated" if repeated else "scalar",
                )
            )
        if not matches:
            raise ValueError(f"unknown event field {field!r}")
        return tuple(matches)
    raise ValueError("field selection requires an entity, event, or candidate source")


def target_dtype(
    contract: Any, name: str, *, candidate: str | None = None, field: str | None = None
) -> DType:
    sample = contract.sample
    if candidate is not None:
        sample = sample.candidate_sets[candidate].sample_spec()
    target = sample.targets[name]
    if field is not None and not isinstance(target, RowTarget):
        raise ValueError("a structured target requires a sampled RowTarget source")
    if isinstance(target, RowTarget):
        if field is None:
            raise ValueError(f"structured target {name!r} requires a field")
        declaration = target.fields[field]
        if isinstance(declaration, Field) and declaration.selection == "all":
            raise ValueError("structured target fields must be scalar, not repeated")
        table = destination_table(contract.schema, sample.entity.table, target.rows.path)
        return value_dtype(contract.schema, table, target.fields[field], target.fields)
    return target.dtype


def validate_batch(contract: Any) -> tuple[Any, ...]:
    from kumo.pytorch.spec import ValidationIssue

    issues = []
    shared: dict[str, DType] = {}
    selections: dict[str, set[tuple[str, ...]]] = {}
    batch = contract.batch

    def add(
        path: str,
        message: str,
        suggestion: str = "select compatible typed fields or choose a supported output arrangement",
    ) -> None:
        issues.append(ValidationIssue(path, message, suggestion))

    def encoding(config: Any, dtype: DType, path: str, *, space: str | None = None) -> Any:
        if isinstance(config, UseEncoding):
            if config.name not in batch.encodings:
                add(path, f"unknown encoder {config.name!r}")
                return None
            previous = shared.setdefault(config.name, dtype)
            if previous != dtype:
                add(
                    path,
                    f"shared encoder {config.name!r} requires identical dtypes, "
                    f"got {previous} and {dtype}",
                )
            config = batch.encodings[config.name]
        if isinstance(config, (ZScore, Buckets)) and not dtype.numeric:
            add(path, f"{type(config).__name__} requires a numerical dtype, not {dtype.name}")
        elif isinstance(config, Text) and dtype.name != "string":
            add(path, f"Text requires string, not {dtype.name}")
        if space is not None:
            if space not in batch.id_spaces:
                add(path, f"unknown ID space {space!r}")
            elif isinstance(config, (Raw, ZScore)):
                add(path, "Raw and ZScore outputs cannot join an ID space")
        if isinstance(config, Vocabulary) and config.values is not None:
            from kumo.pytorch.encoding import typed_key

            keys = set()
            for scalar, _local_id in config.values:
                try:
                    key = typed_key(dtype, scalar)
                    if scalar is None or key in keys:
                        raise ValueError(
                            "supplied vocabulary values must be unique non-NULL typed entries"
                        )
                    keys.add(key)
                except ValueError as error:
                    add(path, str(error))
        return config

    def layout(value: Any, path: str) -> None:
        if isinstance(value, Nested):
            layout(value.fields, f"{path}.fields")
            layout(value.events, f"{path}.events")

    def group(input_name: str, fields: Fields, path: str, table: str | None = None) -> None:
        seen = set()
        identities = selections.setdefault(input_name, set())
        for index, use in enumerate(fields.fields):
            location = f"{path}.fields[{index}]"
            try:
                matches = (
                    resolved_fields(contract, input_name, use.field)
                    if table is None
                    else (
                        (
                            ("graph", table, use.field),
                            next(
                                item for item in contract.schema.tables if item.name == table
                            ).columns[use.field],
                            "scalar",
                        ),
                    )
                )
            except (KeyError, ValueError, StopIteration) as error:
                add(location, str(error))
                continue
            types = {dtype for _identity, dtype, _cardinality in matches}
            if len(types) != 1:
                add(location, f"field {use.field!r} selects incompatible event dtypes {types}")
            for identity, dtype, cardinality in matches:
                if identity in seen:
                    add(location, f"the same sampled value is encoded twice: {identity}")
                seen.add(identity)
                identities.add(identity)
                if (
                    use.field in fields.forbidden_fields
                    or identity[-1] in fields.forbidden_fields
                    or "/".join(identity) in fields.forbidden_fields
                ):
                    add(location, f"forbidden field {use.field!r} is selected")
                config = encoding(use.encoding, dtype, location, space=use.id_space)
                if isinstance(fields.row, Composite) and (
                    isinstance(config, Text) or cardinality != "scalar"
                ):
                    add(location, "Composite requires scalar components and does not accept Text")
                if isinstance(fields.row, Tokens) and isinstance(config, (Raw, ZScore)):
                    add(location, "token rows require ID-producing encoders")

    for name, value in batch.inputs.items():
        path = f"batch.inputs.{name}"
        layout(value.layout, f"{path}.layout")
        if isinstance(value.representation, Graph):
            if value.source.name not in contract.sample.neighborhoods:
                add(f"{path}.source", f"unknown neighborhood {value.source.name!r}")
            reachable = {contract.sample.entity.table}
            neighborhood = contract.sample.neighborhoods.get(value.source.name)
            if neighborhood is not None:
                relations = {item.name: item for item in contract.schema.relations}
                for layer_value in neighborhood.layers:
                    for traversal in layer_value.samples:
                        relation = relations.get(traversal.relation)
                        if relation:
                            reachable.update((relation.left_table, relation.right_table))
            for table, fields in value.representation.node_fields.items():
                if fields.field_names is not None:
                    add(
                        f"{path}.node_fields.{table}.field_names",
                        "graph attributes do not emit field-name IDs",
                        "leave field_names unset for graph attributes",
                    )
                if table not in reachable:
                    add(
                        f"{path}.node_fields.{table}",
                        f"table {table!r} is not reachable in this neighborhood",
                    )
                group(name, fields, f"{path}.node_fields.{table}", table)
        else:
            if isinstance(value.source, OccurrencesOf):
                if any(not isinstance(use.encoding, Raw) for use in value.representation.fields):
                    add(
                        f"{path}.representation",
                        "occurrence projections require explicit Raw encodings",
                    )
            if isinstance(value.source, EventsOf):
                parent = batch.inputs.get(value.source.input)
                if (
                    parent is None
                    or not isinstance(parent.source, EventSource)
                    or not isinstance(parent.representation.history, Grouped)
                ):
                    add(f"{path}.source", "EventsOf requires a direct grouped EventSource parent")
            group(name, value.representation, f"{path}.representation")

    for name, target in batch.targets.items():
        path = f"batch.targets.{name}"
        try:
            if isinstance(target, LabelTarget):
                encoding(
                    target.encoding,
                    target_dtype(contract, target.source),
                    f"{path}.encoding",
                    space=target.id_space,
                )
            elif isinstance(target, StructuredTarget):
                for use in target.fields:
                    encoding(
                        use.encoding,
                        target_dtype(contract, target.source, field=use.field),
                        f"{path}.fields.{use.field}",
                        space=use.id_space,
                    )
            elif isinstance(target, CandidateTarget):
                source = batch.inputs[target.input].source
                encoding(
                    target.encoding,
                    target_dtype(contract, target.source, candidate=source.name),
                    f"{path}.encoding",
                    space=target.id_space,
                )
            elif isinstance(target, AlignedEventTarget):
                for use in target.fields:
                    for identity, dtype, cardinality in resolved_fields(
                        contract, target.input, use.field
                    ):
                        if cardinality != "scalar":
                            add(
                                f"{path}.fields.{use.field}",
                                "aligned target fields must be scalar, not repeated",
                            )
                        for input_name, selected_identities in selections.items():
                            if identity in selected_identities:
                                add(
                                    f"{path}.fields",
                                    f"target field {use.field!r} also appears "
                                    f"in input {input_name!r}",
                                )
                        encoding(
                            use.encoding, dtype, f"{path}.fields.{use.field}", space=use.id_space
                        )
                for label in target.derived:
                    matches = resolved_fields(contract, target.input, label.field)
                    if any(not dtype.numeric for _identity, dtype, _shape in matches):
                        add(f"{path}.derived", "threshold labels require numerical source values")
                    if any(shape != "scalar" for _identity, _dtype, shape in matches):
                        add(f"{path}.derived", "threshold labels require scalar source values")
                    label.dtype.normalize(label.below, path=f"{path}.below")
                    label.dtype.normalize(label.above, path=f"{path}.above")
                    encoding(label.encoding, label.dtype, f"{path}.derived.{label.name}")
            elif isinstance(target, MaskedValuesTarget):
                selected = batch.inputs[target.input].representation
                if target.fields and not set(target.fields) <= {
                    use.field for use in selected.fields
                }:
                    add(f"{path}.fields", "masking selects an undeclared input field")
                masked = selections.get(target.input, set())
                if target.fields:
                    masked = {
                        identity
                        for field in target.fields
                        for identity, _dtype, _shape in resolved_fields(
                            contract, target.input, field
                        )
                    }
                for input_name, identities in selections.items():
                    if input_name != target.input and masked & identities:
                        overlap = ", ".join(
                            "/".join(identity) for identity in sorted(masked & identities)
                        )
                        warnings.warn(
                            f"{path}: input {input_name!r} also selects {overlap}, "
                            f"masked in {target.input!r}; masking changes only the named input",
                            InputExposureWarning,
                            stacklevel=3,
                        )
            elif isinstance(target, MatchingGroupTarget):
                for field in target.fields:
                    matches = resolved_fields(contract, target.input, field)
                    if any(shape != "scalar" for _identity, _dtype, shape in matches):
                        add(f"{path}.fields", "matching groups require scalar fields")
            elif isinstance(target, NextTokenTarget) and target.prefix_input is not None:
                if target.prefix_input not in batch.inputs:
                    add(f"{path}.prefix_input", f"unknown input {target.prefix_input!r}")
        except (KeyError, ValueError, AttributeError) as error:
            add(path, str(error))
    return tuple(issues)
