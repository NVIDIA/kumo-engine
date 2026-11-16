"""Static arena-engine scope checks, independent of DiskGraph's implementation.

Kumo validates the contract language and calls these engine-specific checks.
These checks add the selected DiskGraph restrictions. They inspect declarations only and never open source files.
"""

from __future__ import annotations

from kumo.pytorch.contract.sample import (
    Aggregate,
    Compare,
    Constant,
    Derived,
    EventTimeDelta,
    ExternalTarget,
    Field,
    GroupBy,
    Rows,
    RowTarget,
    SampleSpec,
)
from kumo.pytorch.contract.schema import DataSchema, table_clock
from kumo.pytorch.spec import DataLoaderSpec, ValidationIssue

__all__ = ["contract_issues", "sample_issues"]


def _scope_issues(sample: SampleSpec, schema: DataSchema) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def refuse(path: str, message: str, suggestion: str) -> None:
        issues.append(ValidationIssue(path, message, suggestion))

    for name in ("dated_entity_values", "current_events", "candidate_sets"):
        if getattr(sample, name):
            refuse(
                f"sample.{name}",
                f"{name} is outside the selected DiskGraph scope",
                "Use supported historical events, root fields and supplied labels.",
            )
    if sample.entity.filters:
        refuse(
            "sample.entity.filters",
            "Entity filters are outside the selected DiskGraph scope",
            "Select a contract without entity predicates.",
        )

    tables = {table.name: table for table in schema.tables}
    relations = {relation.name: relation for relation in schema.relations}
    physical: dict[str, tuple[tuple[str, str, str, str], bool]] = {}
    seen_relations: dict[tuple[str, str, str, str], str] = {}
    for index, relation in enumerate(schema.relations):
        left, right = tables.get(relation.left_table), tables.get(relation.right_table)
        if left is None or right is None:
            continue  # Kumo reports unknown tables; avoid masking it with a KeyError.
        left_pk, right_pk = (
            relation.left_key == left.primary_key,
            relation.right_key == right.primary_key,
        )
        path = f"schema.relations[{index}]"
        if left_pk == right_pk:
            refuse(
                path,
                f"Relation {relation.name!r} requires exactly one primary-key endpoint",
                "Express this relation as one primary key joined to one foreign key.",
            )
            continue
        identity = (
            (left.name, relation.left_key, right.name, right.primary_key)
            if right_pk
            else (right.name, relation.right_key, left.name, left.primary_key)
        )
        physical[relation.name] = (identity, right_pk)
        previous = seen_relations.setdefault(identity, relation.name)
        if previous != relation.name:
            refuse(
                path,
                f"Relations {previous!r} and {relation.name!r} describe the same physical relation",
                "Declare the PK/FK relation once and select its traversal direction explicitly.",
            )

    def rows_issues(rows: Rows, path: str) -> None:
        if rows.filters:
            refuse(
                path + ".filters",
                "Row filters are outside the selected DiskGraph scope",
                "Select a contract without row predicates.",
            )
        if rows.active is not None:
            refuse(
                path + ".active",
                "Active intervals are outside the selected DiskGraph scope",
                "Use a supported relative time window.",
            )
        current = sample.entity.table
        for index, name in enumerate(rows.path):
            relation = relations.get(name)
            if relation is None:
                break
            if current == relation.left_table:
                destination = relation.right_table
            elif current == relation.right_table:
                destination = relation.left_table
            else:
                break  # Kumo reports disconnected paths.
            table = tables.get(destination)
            if table is None:
                break
            if index < len(rows.path) - 1 and table_clock(table) is not None:
                refuse(
                    f"{path}.path[{index}]",
                    f"Clocked intermediate table {destination!r} is unsupported",
                    "Keep the declared clock and use an equivalent supported direct path.",
                )
            current = destination
        # Rows.time may select a declared alternative column; this is different
        # from graph traversal clocks, which must match the destination clock.

    def source_issues(source, path: str) -> None:
        if isinstance(source, Rows):
            rows_issues(source, path)
        elif isinstance(source, GroupBy):
            source_issues(source.source, path + ".source")
        else:
            refuse(
                path,
                "Unsupported aggregate row source",
                "Use Rows or nested GroupBy sources.",
            )

    def fields_issues(fields, path: str) -> None:
        for name, field in fields.items():
            if isinstance(field, (EventTimeDelta, Constant, Derived)):
                continue
            if (
                not isinstance(field, Field)
                or field.path
                or field.time_column is not None
            ):
                refuse(
                    f"{path}.{name}",
                    "Only direct event/row-target fields, constants and event time deltas are supported",
                    "Read the current row directly without a field path or clock override.",
                )

    for name, value in sample.entity_values.items():
        path = f"sample.entity_values.{name}"
        if isinstance(value, (Constant, Derived)):
            continue
        if isinstance(value, Compare):
            value = value.left
        if isinstance(value, Aggregate):
            source_issues(value.source, path + ".source")
        elif (
            not isinstance(value, Field) or value.path or value.time_column is not None
        ):
            refuse(
                path,
                "Entity values require an aggregate or a direct root field/constant",
                "Remove joined, derived or comparison values and per-field clock overrides.",
            )
    for name, event in sample.events.items():
        rows_issues(event.rows, f"sample.events.{name}.rows")
        fields_issues(event.fields, f"sample.events.{name}.fields")
    for name, target in sample.targets.items():
        if isinstance(target, ExternalTarget):
            continue
        path = f"sample.targets.{name}"
        if isinstance(target, Compare):
            target = target.left
        if isinstance(target, Aggregate):
            source_issues(target.source, path + ".source")
        elif isinstance(target, RowTarget):
            rows_issues(target.rows, path + ".rows")
            fields_issues(target.fields, path + ".fields")
        else:
            refuse(
                path,
                "Generated targets require Aggregate or RowTarget in this scope",
                "Select an aggregate/row target or supply the label on the request.",
            )

    for name, neighborhood in sample.neighborhoods.items():
        for layer_index, layer in enumerate(neighborhood.layers):
            selected = set()
            for index, traversal in enumerate(layer.samples):
                path = f"sample.neighborhoods.{name}.layers[{layer_index}].samples[{index}]"
                if traversal.strategy != "recent":
                    refuse(
                        path + ".strategy",
                        "Uniform graph sampling is outside the selected DiskGraph scope",
                        "Use the recent sampling strategy.",
                    )
                relation = relations.get(traversal.relation)
                if relation is None:
                    continue
                destination = (
                    relation.right_table
                    if traversal.direction == "forward"
                    else relation.left_table
                )
                table = tables.get(destination)
                if table is not None:
                    clock = table_clock(table)
                    if (
                        traversal.time_column is not None
                        and traversal.time_column != clock
                    ):
                        refuse(
                            path + ".time_column",
                            "Graph traversal cannot override its destination table clock",
                            "Use the declared destination clock.",
                        )
                    if traversal.temporal == "static" and clock is not None:
                        refuse(
                            path + ".temporal",
                            "Graph traversal cannot disable a clocked destination's time filter",
                            "Use temporal='inherit' for a clocked destination.",
                        )
                relation_identity = physical.get(traversal.relation)
                if relation_identity is not None:
                    identity, forward = relation_identity
                    key = (
                        identity,
                        forward if traversal.direction == "forward" else not forward,
                    )
                    if key in selected:
                        refuse(
                            path,
                            "A physical graph relation is selected more than once in one layer",
                            "Select each physical relation and direction only once per layer.",
                        )
                    selected.add(key)
    return issues


def sample_issues(
    sample: SampleSpec, schema: DataSchema
) -> tuple[ValidationIssue, ...]:
    """The executable DiskGraph-scoped sample contract, independent of batch layout."""
    issues = _scope_issues(sample, schema)
    tables = {table.name: table for table in schema.tables}
    root = tables.get(sample.entity.table)
    if root is None:
        issues.append(
            ValidationIssue(
                "sample.entity.table",
                "Root table is not declared",
                "Select a declared table.",
            )
        )
    else:
        for name, field in sample.entity_values.items():
            if (
                isinstance(field, Field)
                and field.column is not None
                and root.columns is not None
                and field.column not in root.columns
            ):
                issues.append(
                    ValidationIssue(
                        f"sample.entity_values.{name}",
                        "Root field column is not declared",
                        "Use a declared column.",
                    )
                )
        if (
            sample.entity.available_from is not None
            and sample.entity.available_from not in root.columns
        ):
            issues.append(
                ValidationIssue(
                    "sample.entity.available_from",
                    "Availability column is not declared",
                    "Use a declared column.",
                )
            )
    return tuple(issues)


def contract_issues(contract: DataLoaderSpec) -> tuple[ValidationIssue, ...]:
    """Additional static restrictions for the selected engine scope."""
    issues = list(sample_issues(contract.sample, contract.schema))
    batching = contract.batch.batching
    if batching.length_input in contract.batch.inputs and (
        batching.max_tokens is not None or batching.bucket_by_length
    ):
        from kumo.pytorch.contract.batch import Graph

        selected = contract.batch.inputs[batching.length_input]
        if isinstance(selected.representation, Graph):
            issues.append(
                ValidationIssue(
                    "batch.batching.length_input",
                    "Graph length budgets and buckets are unsupported",
                    "Use max_samples with no graph length planning.",
                )
            )
    return tuple(issues)
