"""The contract checks `prepare` runs: upstream's, then a refusal for each part not built yet."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:

    from kumo.pytorch.anchors import AnchorTable
    from kumo.pytorch.spec import DataLoaderSpec, ValidationIssue



NAME = "cuda"

class _Engine:
    """The face this backend shows upstream's validation, as upstream's engines do: its name, and the checks
    upstream's `check` and `prepare` call before any data is read (``check_sources``, ``check_contract``,
    ``check_anchors``)."""

    name = NAME

    @property
    def unsupported(self) -> Mapping[str, Any]:
        """What this backend refuses, by code (`test.kumo_pytorch.reference_engine.gpu.plan.unsupported`), as `DiskGraphEngine.unsupported`."""

        from test.kumo_pytorch.reference_engine.gpu.plan.unsupported import declared

        return declared()

    @staticmethod
    def check_sources(sources: Mapping[str, Any], path: str) -> tuple[ValidationIssue, ...]:
        """Refuse sources other than local Parquet files: the tables', or (an empty name) an anchor table's."""

        from kumo.pytorch.dataset import Parquet

        issues = []
        unreadable = sorted({source.kind for source in sources.values()} - {"parquet"})
        if unreadable:
            issues.append(_refusal("GPU-SOURCE-KIND", path, f"engine {NAME!r} does not read {unreadable} sources"))
        for name, source in sources.items():
            if isinstance(source, Parquet) and "://" in source.path:
                issues.append(_refusal("GPU-REMOTE-PARQUET", f"{path}.{name}.path" if name else f"{path}.path",
                                       f"test.kumo_pytorch.reference_engine.gpu does not read remote Parquet paths yet; got {source.path!r}"))
        return tuple(issues)

    def check_contract(self, contract: DataLoaderSpec) -> tuple[ValidationIssue, ...]:
        return _unbuilt(contract)

    @staticmethod
    def check_anchors(anchors: AnchorTable) -> tuple[ValidationIssue, ...]:
        from kumo.pytorch.engines.parquet_anchors import check_parquet_anchors

        if getattr(anchors.source, "kind", None) != "parquet":
            return ()  # check_sources refuses the source kind, by name
        return check_parquet_anchors(anchors)

def _refusal(code: str, path: str, message: str) -> ValidationIssue:
    """A refusal of a declared entry (`test.kumo_pytorch.reference_engine.gpu.plan.unsupported`): its code, and its next step as the suggestion."""

    from kumo.pytorch.spec import ValidationIssue

    from test.kumo_pytorch.reference_engine.gpu.plan.unsupported import declared

    return ValidationIssue(path, message, declared()[code].next_step, code)

def _unbuilt(contract: DataLoaderSpec) -> tuple[ValidationIssue, ...]:
    """The parts of a contract this backend does not build yet, each refused with its code (`test.kumo_pytorch.reference_engine.gpu.plan.unsupported`)."""

    from kumo.pytorch.contract.batch import (
        AlignedEventTarget,
        CandidateSource,
        CandidateTarget,
        Columns,
        Composite,
        EntitySource,
        EventSource,
        EventsOf,
        Fields,
        Graph,
        LabelTarget,
        MaskedValuesTarget,
        MatchingGroupTarget,
        NeighborhoodSource,
        NextTokenTarget,
        OccurrencesOf,
        StructuredTarget,
        Tokens,
        UseEncoding,
    )
    from kumo.pytorch.contract.batch.encodings import Buckets, Raw, Text, Vocabulary, ZScore
    from kumo.pytorch.contract.batch.validation import resolved_fields
    from kumo.pytorch.contract.sample import (
        Aggregate,
        Compare,
        Constant,
        Derived,
        EventTimeDelta,
        Field,
        RowTarget,
        destination_table,
        dated_value_types,
    )

    def refused(code: str, path: str, what: str) -> ValidationIssue:
        return _refusal(code, path, f"test.kumo_pytorch.reference_engine.gpu does not build {what} yet")

    sample, issues = contract.sample, []
    entity = sample.entity

    def scope(values: Any, table_name: str, prefix: str) -> None:
        """Refuse by name what one scope's values (the entity's, or an event component's fields) need."""

        for name, value in values.items():
            where = f"{prefix}.{name}"
            aggregate = value.left if isinstance(value, Compare) else value
            if isinstance(value, Field):
                if value.selection == "all":
                    issues.append(refused("GPU-REPEATED-FIELD", f"{where}.selection",
                                          "repeated field values (selection='all')"))
            elif isinstance(value, Derived):
                continue  # every calculation upstream checks is built
            elif isinstance(aggregate, Aggregate):
                continue  # every aggregate and grouping upstream checks is built
            elif not isinstance(value, (Constant, EventTimeDelta)):
                issues.append(refused("GPU-OTHER", where, f"{type(value).__name__} values"))

    scope(sample.entity_values, entity.table, "sample.entity_values")
    for collection in ("events", "current_events"):
        for name, spec in getattr(sample, collection).items():
            try:
                table_name = destination_table(contract.schema, entity.table, spec.rows.path)
            except ValueError:
                continue
            scope(spec.fields, table_name, f"sample.{collection}.{name}.fields")
    for name in sample.candidate_sets:
        issues.append(refused("GPU-CANDIDATE-SETS", f"sample.candidate_sets.{name}", "candidate sets"))
    try:
        dated_types = dated_value_types(contract.schema, sample)
    except (ValueError, KeyError, StopIteration):  # upstream's own check reports these
        dated_types = {}
    for name, dated in sample.dated_entity_values.items():
        where = f"sample.dated_entity_values.{name}"
        if len(dated.rows.path) > 1:
            issues.append(refused("GPU-DATED-PATH", f"{where}.rows.path",
                                  "dated values through more than one relation"))
        if isinstance(dated.field, Field) and (dated.field.path or dated.field.time_column is not None
                                               or dated.field.selection != "one"):
            issues.append(refused("GPU-DATED-FIELD", f"{where}.field",
                                  "dated values read through relations or other clocks"))
        dtype = dated_types.get(name, {}).get(name)
        if dtype is not None and not (dtype.integer or dtype.name in {"string", "bool"}
                                      or dtype.name.startswith("float")):
            issues.append(refused("GPU-DATED-TYPE", f"{where}.field", f"dated values of {dtype.name}"))
        try:
            table_name = destination_table(contract.schema, entity.table, dated.rows.path)
        except ValueError:
            continue
        scope(dated.fields, table_name, f"{where}.fields")
    for name, target in sample.targets.items():
        if isinstance(target, (Aggregate, Compare)):
            scope({name: target}, entity.table, "sample.targets")
        elif isinstance(target, RowTarget):
            try:
                table_name = destination_table(contract.schema, entity.table, target.rows.path)
            except ValueError:
                continue
            scope(target.fields, table_name, f"sample.targets.{name}.fields")
    batch = contract.batch
    for name, value in batch.inputs.items():
        where = f"batch.inputs.{name}"
        representation = value.representation
        if isinstance(value.source, NeighborhoodSource) and isinstance(representation, Graph):
            for table, group in representation.node_fields.items():
                for index, use in enumerate(group.fields):
                    config = batch.encodings.get(use.encoding.name) if isinstance(use.encoding, UseEncoding) \
                        else use.encoding  # an unknown shared encoder: upstream's own check names it
                    if config is not None and not isinstance(config, (Raw, Vocabulary, ZScore, Buckets, Text)):
                        issues.append(refused("GPU-OTHER",
                                              f"{where}.representation.node_fields.{table}.fields[{index}].encoding",
                                              f"{type(config).__name__} encodings"))
            continue
        if not isinstance(value.source, (EntitySource, EventSource, EventsOf, OccurrencesOf)):
            issues.append(refused("GPU-CANDIDATE-SETS" if isinstance(value.source, CandidateSource) else "GPU-OTHER",
                                  f"{where}.source", f"{type(value.source).__name__} inputs"))
        elif not isinstance(representation, Fields):
            issues.append(refused("GPU-OTHER", f"{where}.representation", f"{type(representation).__name__} inputs"))
        elif not isinstance(representation.row, (Columns, Tokens, Composite)):
            issues.append(refused("GPU-OTHER", f"{where}.representation.row",
                                  f"{type(representation.row).__name__} rows"))
        else:
            for index, use in enumerate(representation.fields):
                config = batch.encodings.get(use.encoding.name) if isinstance(use.encoding, UseEncoding) \
                    else use.encoding  # an unknown shared encoder: upstream's own check names it
                if config is not None and not isinstance(config, (Raw, Vocabulary, ZScore, Buckets, Text)):
                    issues.append(refused("GPU-OTHER", f"{where}.representation.fields[{index}].encoding",
                                          f"{type(config).__name__} encodings"))
                elif isinstance(config, Raw) and isinstance(representation.row, Composite):
                    try:
                        dtype = resolved_fields(contract, name, use.field)[0][1]
                    except (ValueError, KeyError, IndexError, StopIteration):
                        continue  # upstream's own check names it
                    if dtype.name.startswith("decimal"):
                        issues.append(refused("GPU-DECIMAL-COMPOSITE", f"{where}.representation.fields[{index}]",
                                              "Composite components of decimal values"))
    def decimal(input_name: str, selector: str) -> bool:
        try:
            return resolved_fields(contract, input_name, selector)[0][1].name.startswith("decimal")
        except (ValueError, KeyError, IndexError, StopIteration):
            return False  # upstream's own check names it

    for name, target in batch.targets.items():
        where = f"batch.targets.{name}"
        if not isinstance(target, (LabelTarget, StructuredTarget, AlignedEventTarget, MatchingGroupTarget,
                                   MaskedValuesTarget, NextTokenTarget)):
            issues.append(refused("GPU-CANDIDATE-SETS" if isinstance(target, CandidateTarget) else "GPU-OTHER", where,
                                  f"{type(target).__name__} targets"))
            continue
        if isinstance(target, (MaskedValuesTarget, NextTokenTarget)):
            continue  # upstream's own check ties them to a token input (and a flattened one)
        if isinstance(target, MatchingGroupTarget):
            for selector in target.fields:
                if target.input in batch.inputs and decimal(target.input, selector):
                    issues.append(refused("GPU-DECIMAL-TARGET", f"{where}.fields", "matching groups of decimal values"))
            continue
        places = [(f"{where}.encoding", target.encoding)] if isinstance(target, LabelTarget) else \
            [(f"{where}.fields[{index}].encoding", use.encoding) for index, use in enumerate(target.fields)]
        if isinstance(target, AlignedEventTarget):
            places += [(f"{where}.derived[{index}].encoding", label.encoding)
                       for index, label in enumerate(target.derived)]
            for index, label in enumerate(target.derived):
                if target.input in batch.inputs and decimal(target.input, label.field):
                    issues.append(refused("GPU-DECIMAL-TARGET", f"{where}.derived[{index}].field",
                                          "threshold labels of decimal values"))
        for place, encoding in places:
            config = batch.encodings.get(encoding.name) if isinstance(encoding, UseEncoding) else encoding
            if config is not None and not isinstance(config, (Raw, Vocabulary, ZScore, Buckets)):
                issues.append(refused("GPU-TARGET-ENCODING", place, f"{type(config).__name__} encodings"))
    return tuple(issues)
