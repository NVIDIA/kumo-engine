# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent answers for the finite generated aggregate fixture family.

Only relational joins, window inequalities and elementary reductions are used.
This is a test oracle for these fixtures, not a data-loading backend.
"""

import math
from collections import Counter, defaultdict

from kumo.pytorch.api import Aggregate, GroupBy, RowTarget


def reduction(rows, operation, column=None, cutoff=None):
    values = [row[column] for row in rows if row[column] is not None] if column else []
    if operation == "count":
        return len(rows)
    if operation == "count_distinct":
        return len(set(values))
    if operation == "sum":
        return sum(values) if values or not rows else None
    if not values:
        return None
    if operation == "recency":
        return float(cutoff - max(values))
    if operation in ("min", "max"):
        return (min if operation == "min" else max)(values)
    if operation == "mode":
        counts = Counter(values)
        return min(value for value, count in counts.items() if count == max(counts.values()))
    mean = sum(values) / len(values)
    if operation == "mean":
        return mean
    if operation in ("std_sample", "std_population", "var_sample", "var_population"):
        denominator = len(values) - int(operation.endswith("sample"))
        if not denominator:
            return None
        variance = sum((value - mean) ** 2 for value in values) / denominator
        return math.sqrt(variance) if operation.startswith("std") else variance
    raise AssertionError(f"oracle has no rule for reduction {operation}")


def selected_rows(case, request, source):
    if isinstance(source, GroupBy):
        groups = defaultdict(list)
        for row in selected_rows(case, request, source.source):
            groups[tuple(row[key] for key in source.keys)].append(row)
        return [
            dict(zip(source.keys, keys))
            | {
                name: reduction(rows, item.operation, item.column, request.as_of_time)
                for name, item in source.values.items()
            }
            for keys, rows in groups.items()
        ]
    schemas = {table.name: table for table in case.contract.schema.tables}
    relations = {relation.name: relation for relation in case.contract.schema.relations}
    table = case.contract.sample.entity.table
    rows = [
        row for row in case.tables[table] if row[schemas[table].primary_key] == request.entity_id
    ]
    for index, name in enumerate(source.path):
        relation = relations[name]
        if relation.left_table == table:
            target, key, foreign = relation.right_table, relation.left_key, relation.right_key
        else:
            target, key, foreign = relation.left_table, relation.right_key, relation.left_key
        rows = [
            other
            for row in rows
            for other in case.tables[target]
            if row[key] is not None and row[key] == other[foreign]
        ]
        table = target
        clock = schemas[table].time_column
        if index < len(source.path) - 1 and isinstance(clock, str):
            rows = [
                row for row in rows if row[clock] is not None and row[clock] <= request.as_of_time
            ]
    if source.time:
        window = source.time
        rows = [
            row
            for row in rows
            if row[window.column] is not None
            and (window.start is None or row[window.column] > request.as_of_time + window.start)
            and (window.stop is None or row[window.column] <= request.as_of_time + window.stop)
        ]
    if source.distinct:
        rows = list({row[schemas[table].primary_key]: row for row in rows}.values())
    assert not source.filters, "generated oracle does not define filtered fixtures"
    return rows


def aggregate_answer(case):
    expected = {}
    for request in case.requests:
        values = {}
        for name, item in case.contract.sample.entity_values.items():
            assert isinstance(item, Aggregate), f"oracle has no scalar rule for {name}"
            value = reduction(
                selected_rows(case, request, item.source),
                item.operation,
                item.column,
                request.as_of_time,
            )
            values[name] = (
                float(value) if value is not None and item.dtype.name.startswith("float") else value
            )
        for name, item in case.contract.sample.targets.items():
            if isinstance(item, Aggregate):
                value = reduction(
                    selected_rows(case, request, item.source),
                    item.operation,
                    item.column,
                    request.as_of_time,
                )
                if value is not None:
                    values[f"target:{name}"] = value
            else:
                assert isinstance(item, RowTarget)
                rows = selected_rows(case, request, item.rows)
                if rows:
                    rows.sort(key=lambda row: (row[item.rows.time.column], row["event_id"]))
                    row = rows[0 if item.selection == "first" else -1]
                    values[f"target:{name}.offset"] = (
                        row[item.rows.time.column] - request.as_of_time
                    )
                    for field, source in item.fields.items():
                        if row[source.column] is not None:
                            values[f"target:{name}.{field}"] = row[source.column]
        expected[str(request.sample_id)] = values
    return expected


def target_choices(case):
    """A row target may select any complete eligible row tied at its boundary time."""
    choices = {}
    for request in case.requests:
        by_target = {}
        for name, item in case.contract.sample.targets.items():
            if not isinstance(item, RowTarget):
                continue
            rows = selected_rows(case, request, item.rows)
            if not rows:
                continue
            clock = item.rows.time.column
            boundary = (min if item.selection == "first" else max)(row[clock] for row in rows)
            by_target[name] = [
                {f"target:{name}.offset": boundary - request.as_of_time}
                | {
                    f"target:{name}.{field}": row[source.column]
                    for field, source in item.fields.items()
                }
                for row in rows
                if row[clock] == boundary
            ]
        choices[str(request.sample_id)] = by_target
    return choices


def encoder_expectations(case, contract):
    """Encoder inventory and state derived from declarations and fixture populations."""
    import json
    from dataclasses import asdict

    import numpy as np

    from kumo.pytorch.api import Buckets, DType, Raw, UseEncoding, Vocabulary, ZScore

    aggregates = aggregate_answer(case)
    cutoff = max(request.as_of_time for request in case.requests)
    stored = [row for row in case.tables["events"] if row["at"] <= cutoff]
    events_schema = next(table for table in contract.schema.tables if table.name == "events")
    uses = {}
    sampled_history = (
        [
            row
            for request in case.requests
            for row in selected_rows(case, request, contract.sample.events["history"].rows)
        ]
        if "history" in contract.sample.events
        else []
    )
    computed_shared = {
        field.encoding.name
        for name, input_spec in contract.batch.inputs.items()
        if name == "entity"
        for field in input_spec.representation.fields
        if isinstance(field.encoding, UseEncoding)
    }

    def add(path, config, dtype, values, stored_column=False):
        key = json.dumps(path, separators=(",", ":"))
        if key in uses:
            uses[key][2].extend(values)
        else:
            uses[key] = [config, dtype, list(values), stored_column]

    for name, input_spec in contract.batch.inputs.items():
        for field in input_spec.representation.fields:
            is_entity = name == "entity"
            values = (
                [row[field.field] for row in aggregates.values()]
                if is_entity
                else [
                    row[field.field]
                    for row in (
                        sampled_history
                        if isinstance(field.encoding, UseEncoding)
                        and field.encoding.name in computed_shared
                        else stored
                    )
                ]
            )
            dtype = (
                contract.sample.entity_values[field.field].dtype
                if is_entity
                else events_schema.columns[field.field]
            )
            config = field.encoding
            path = ["inputs", name, "fields", field.field]
            if isinstance(config, UseEncoding):
                path = ["shared", config.name]
                config = contract.batch.encodings[config.name]
            # Native scalar materialization uses the stored-column t-digest collector too.
            add(path, config, dtype, values, not isinstance(field.encoding, UseEncoding))
    for name, target in contract.batch.targets.items():
        if hasattr(target, "fields"):
            for field in target.fields:
                column = contract.sample.targets[target.source].fields[field.field].column
                add(
                    ["targets", name, field.field],
                    field.encoding,
                    events_schema.columns[column],
                    [],
                )
        else:
            sample_target = contract.sample.targets[target.source]
            dtype = sample_target.dtype or DType(
                "int64" if sample_target.operation == "count" else "float64"
            )
            add(["targets", name, "value"], target.encoding, dtype, [])
    result, approximate = {}, {}
    for identity, (config, dtype, values, stored_column) in uses.items():
        values = [value for value in values if value is not None]
        state = {"kind": type(config).__name__.lower(), "dtype": asdict(dtype)}
        if isinstance(config, Vocabulary):
            counts = Counter(values)
            ordered = sorted(counts, key=lambda value: (-counts[value], value))
            roles = {"missing": config.missing_id, "unknown": config.unknown_id}
            available = (
                i
                for i in range(len(ordered) + len(roles) + max(roles.values()) + 1)
                if i not in roles.values()
            )
            entries = [[value, next(available)] for value in ordered]
            state.update(
                entries=entries,
                roles=roles,
                vocabulary_size=max([*roles.values(), *(i for _, i in entries)]) + 1,
            )
        elif isinstance(config, ZScore):
            mean = sum(values) / len(values) if values else 0.0
            scale = (
                (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
                if values
                else 1.0
            )
            state.update(mean=mean, scale=scale or 1.0, output_dtype=config.dtype)
        elif isinstance(config, Buckets):
            values = sorted(value for value in values if not config.separate_zero or value != 0)
            ranges = config.bins - int(config.separate_zero)
            cuts = (
                [
                    float(value)
                    for value in np.unique(np.quantile(values, np.linspace(0, 1, ranges + 1)[1:-1]))
                ]
                if values
                else []
            )
            state.update(
                boundaries=cuts,
                separate_zero=config.separate_zero,
                roles={"missing": 0},
                vocabulary_size=len(cuts) + 2 + int(config.separate_zero),
            )
            if stored_column:
                approximate[identity] = (values, ranges - 1)
        else:
            assert isinstance(config, Raw), f"fixture oracle has no encoder rule for {config}"
        result[identity] = state
    return result, approximate


def fitted_structure(contract, encoders):
    """Bindings and final ID maps from G-IDSPACE-001, for these generated fields."""
    import json

    from kumo.pytorch.api import UseEncoding

    def identity(parts):
        return json.dumps(parts, separators=(",", ":"))

    inputs, targets, field_sets, space_members = {}, {}, {}, {}
    for name, input_spec in contract.batch.inputs.items():
        fields = input_spec.representation
        bindings = []
        for use in fields.fields:
            key = (
                identity(["shared", use.encoding.name])
                if isinstance(use.encoding, UseEncoding)
                else identity(["inputs", name, "fields", use.field])
            )
            bindings.append([use.field, key, use.id_space])
            if use.id_space:
                space_members.setdefault(use.id_space, {})[key] = encoders[key]
            if fields.field_names:
                qualified = (
                    ("entity", use.field) if name == "entity" else ("events", "history", use.field)
                )
                field_sets.setdefault(fields.field_names, set()).add(qualified)
        inputs[name] = {"bindings": bindings}
    field_names = {
        name: [[list(key), i + 1] for i, key in enumerate(sorted(members))]
        for name, members in field_sets.items()
    }
    for input_spec in contract.batch.inputs.values():
        fields = input_spec.representation
        if fields.field_space:
            key = identity(["fields", fields.field_names])
            space_members.setdefault(fields.field_space, {})[key] = {
                "vocabulary_size": len(field_names[fields.field_names]) + 1,
                "roles": {"pad": 0},
            }
    for name, target in contract.batch.targets.items():
        fields = (
            [field.field for field in target.fields] if hasattr(target, "fields") else ["value"]
        )
        targets[name] = {
            "bindings": [[field, identity(["targets", name, field]), None] for field in fields],
            "thresholds": {},
            "groups": [],
        }
    spaces = {}
    order = ["pad", "unknown", "missing", "mask", "entity", "event", "separator", "bos", "eos"]
    for name, config in contract.batch.id_spaces.items():
        role_names = [role for role in order if role in config.specials]
        role_names += sorted(set(config.specials) - set(order))
        roles = (
            dict(config.special_ids)
            if config.special_ids
            else {role: i for i, role in enumerate(role_names)}
        )
        next_id = max(roles.values(), default=-1) + 1
        members = {}
        for key, encoder in sorted(
            space_members[name].items(), key=lambda pair: json.loads(pair[0])
        ):
            local_roles = {number: role for role, number in encoder.get("roles", {}).items()}
            mapping = []
            for local in range(encoder["vocabulary_size"]):
                role = local_roles.get(local)
                if role in roles:
                    final = roles[role]
                else:
                    final = next_id
                    next_id += 1
                mapping.append([local, final])
            members[key] = mapping
        spaces[name] = {"roles": roles, "size": next_id, "members": members}
    return {
        "format_version": 2,
        "encoders": encoders,
        "inputs": inputs,
        "targets": targets,
        "field_names": field_names,
        "spaces": spaces,
    }
