# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Point-in-time resolution of the shared bounded relational declaration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType
from typing import Any

from tfm.config import RelationalTaskConfig
from tfm.contracts.errors import SchemaError, UnsupportedCombinationError
from tfm.contracts.graph import GraphAnchor, utc_us
from tfm.contracts.relational import (
    OwnershipVersion,
    RelationalDataset,
    RelationalForeignKey,
    RelationalTable,
)
from tfm.contracts.schema import EntityProfile, TransactionEvent, TransactionSequence, ensure_utc
from tfm.data.graph import GraphLink, GraphRow, GraphTable, RelationalGraph
from tfm.data.validation.schema import PROHIBITED_INPUT_KEYS, validate_events


@dataclass(frozen=True)
class RelationalAnchor:
    """Prediction identity and input cutoff, with separate optional label clocks."""

    anchor: GraphAnchor
    split: str
    target: float | None = None
    label_available_at: datetime | None = None
    label_cutoff: datetime | None = None


@dataclass(frozen=True)
class ResolvedHistoryEvent:
    """One deduplicated event and the earliest complete observable path to it."""

    table: str
    key: str
    event: TransactionEvent
    path_available_at: datetime


@dataclass(frozen=True)
class _Binding:
    parent: str
    child: str
    available_at: datetime | None
    version: OwnershipVersion | None = None


def timestamp(value: Any, path: str) -> datetime:
    """Normalize declared date/timestamp values; never infer a clock column."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SchemaError(f"{path}: invalid declared timestamp {value!r}") from exc
    if isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    if not isinstance(value, datetime):
        raise SchemaError(f"{path}: expected a declared date/timestamp, got {type(value).__name__}")
    return ensure_utc(value)


def key_token(values: Sequence[Any]) -> str:
    """Canonical composite identity; raw key values never become model features."""
    if not values or any(value is None for value in values):
        raise SchemaError("Relational primary keys must be nonempty and non-null")
    encoded = []
    for value in values:
        # Tag types so integer 1, float 1.0, text "1" and ISO-looking strings
        # cannot alias. Reject objects rather than relying on repr/default=str.
        if isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, int):
            kind = "int"
        elif isinstance(value, float) and math.isfinite(value):
            kind = "float"
        elif isinstance(value, datetime):
            kind, value = "datetime", ensure_utc(value).isoformat()
        elif isinstance(value, date):
            kind, value = "date", value.isoformat()
        elif isinstance(value, str):
            kind = "str"
        else:
            raise SchemaError(f"Unsupported relational key type: {type(value).__name__}")
        encoded.append([kind, value])
    return json.dumps(
        encoded,
        ensure_ascii=False,
        separators=(",", ":"),
    )


class RelationalSource:
    """Validated tables, FK bindings and point-in-time history traversal.

    Fixed projections and versioned traversal use the same observation checks.
    A history event is deduplicated by table and primary key across paths.
    """

    def __init__(self, dataset: RelationalDataset, task: RelationalTaskConfig) -> None:
        task = RelationalTaskConfig.model_validate(task.model_dump(mode="python"))
        self.dataset, self.task = dataset, task
        self.tables = {table.name: table for table in dataset.tables}
        self.relations = {relation.name: relation for relation in dataset.relationships}
        if len(self.tables) != len(dataset.tables) or len(self.relations) != len(
            dataset.relationships
        ):
            raise SchemaError("Relational table and relationship names must be unique")
        if task.root not in self.tables or self.tables[task.root].role != "input":
            raise SchemaError(f"Root path {task.root}: missing table declaration")
        if set(dataset.rows) != set(self.tables):
            raise SchemaError("Relational rows must match the declared tables exactly")
        self.rows: dict[str, dict[str, Mapping[str, Any]]] = {}
        self.times: dict[tuple[str, str], tuple[datetime | None, datetime | None]] = {}
        forbidden = {
            key.strip().casefold() for key in PROHIBITED_INPUT_KEYS | dataset.forbidden_fields
        }
        if task.labels is not None:
            forbidden.add(task.labels.target.strip().casefold())
        for table in dataset.tables:
            metadata = set(table.key) | {table.event_time, table.available_at}
            metadata.update(
                column
                for fk in dataset.relationships
                if fk.child == table.name
                for column in fk.columns
            )
            metadata.update(
                fk.available_at
                for fk in dataset.relationships
                if fk.child == table.name and fk.available_at is not None
            )
            if (
                task.labels is not None
                and task.labels.table == table.name
                and (table.numeric or table.categorical or table.text)
            ):
                raise SchemaError(
                    f"Label path {table.name}: label tables cannot declare input features"
                )
            for fields in (table.numeric, table.categorical, table.text):
                for alias, column in fields.items():
                    if (
                        alias.strip().casefold() in forbidden
                        or column.strip().casefold() in forbidden
                        or column.strip().casefold()
                        in {str(key).strip().casefold() for key in metadata if key is not None}
                    ):
                        raise SchemaError(
                            f"{table.name}.{column}: keys, clocks and label-derived fields cannot be model inputs"
                        )
                    if dataset.rows[table.name] and not any(
                        column in row for row in dataset.rows[table.name]
                    ):
                        raise SchemaError(
                            f"{table.name}.{column}: declared feature column is absent"
                        )
            aliases = [
                alias
                for fields in (table.numeric, table.categorical, table.text)
                for alias in fields
            ]
            if len(aliases) != len(set(aliases)):
                raise SchemaError(f"{table.name}: feature aliases must be unique across kinds")
            keyed = {}
            for row in dataset.rows[table.name]:
                token = key_token([row.get(column) for column in table.key])
                if token in keyed:
                    raise SchemaError(f"{table.name}: duplicate primary key")
                event, available = self._row_clocks(table, row)
                keyed[token] = row
                self.times[table.name, token] = event, available
            self.rows[table.name] = keyed
        self.edges: dict[str, tuple[_Binding, ...]] = {}
        self.adjacency: dict[tuple[str, bool, str], list[_Binding]] = {}
        versioned = {version.relation for version in dataset.ownership_versions}
        for relation in dataset.relationships:
            if relation.parent not in self.tables or relation.child not in self.tables:
                raise SchemaError(f"Path {relation.name}: undeclared relationship endpoint")
            parent = self.tables[relation.parent]
            if len(relation.columns) != len(parent.key):
                raise SchemaError(f"Path {relation.name}: FK and primary-key arity differ")
            for row in dataset.rows[relation.child]:
                missing = [column for column in relation.columns if column not in row]
                if missing:
                    raise SchemaError(
                        f"Path {relation.name}: declared FK columns are absent from {relation.child}: {missing}; use explicit None for an optional link"
                    )
            bindings = []
            if relation.name in versioned:
                if not relation.ownership:
                    raise SchemaError(
                        f"Path {relation.name}: version records require an ownership relationship"
                    )
                for version in dataset.ownership_versions:
                    if version.relation != relation.name:
                        continue
                    left, right = key_token(version.parent_key), key_token(version.child_key)
                    self._endpoints(relation, left, right)
                    if version.valid_until is not None and utc_us(version.valid_until) <= utc_us(
                        version.valid_from
                    ):
                        raise SchemaError(
                            f"Path {relation.name}: empty ownership validity interval"
                        )
                    if version.known_until is not None and utc_us(version.known_until) <= utc_us(
                        version.known_from
                    ):
                        raise SchemaError(
                            f"Path {relation.name}: empty relationship knowledge interval"
                        )
                    bindings.append((left, right, ensure_utc(version.known_from), version))
                covered = {right for _, right, _, _ in bindings}
                if any(
                    right not in covered
                    and any(row.get(column) is not None for column in relation.columns)
                    for right, row in self.rows[relation.child].items()
                ):
                    raise SchemaError(
                        f"Path {relation.name}: version records must cover every child with a fixed FK; partial ownership logs cannot replace a role"
                    )
            else:
                for right, row in self.rows[relation.child].items():
                    values = [row.get(column) for column in relation.columns]
                    if all(value is None for value in values):
                        continue
                    if any(value is None for value in values):
                        raise SchemaError(f"Path {relation.name}: partially null composite FK")
                    left = key_token(values)
                    self._endpoints(relation, left, right)
                    available = self.times[relation.child, right][1]
                    if relation.available_at is not None:
                        available = timestamp(
                            row.get(relation.available_at),
                            f"{relation.name}.{relation.available_at}",
                        )
                    bindings.append((left, right, available, None))
            records = tuple(_Binding(*binding) for binding in bindings)
            self._validate_versions(relation.name, records)
            self.edges[relation.name] = records
            for binding in records:
                self.adjacency.setdefault((relation.name, True, binding.parent), []).append(binding)
                self.adjacency.setdefault((relation.name, False, binding.child), []).append(binding)
        unknown = versioned - set(self.relations)
        if unknown:
            raise SchemaError(f"Ownership versions reference undeclared paths: {sorted(unknown)}")
        self.paths = {name: self._path(name, path) for name, path in task.history_paths.items()}
        if task.labels is not None:
            if task.labels.table == task.root or any(
                table == task.labels.table for path in self.paths.values() for _, _, table in path
            ):
                raise SchemaError(f"History path cannot include label table {task.labels.table}")

    @staticmethod
    def _validate_versions(role: str, bindings: tuple[_Binding, ...]) -> None:
        by_child: dict[str, list[OwnershipVersion]] = {}
        for binding in bindings:
            version = binding.version
            if version is None:
                continue
            for other in by_child.setdefault(binding.child, []):
                effective = (
                    other.valid_until is None
                    or utc_us(version.valid_from) < utc_us(other.valid_until)
                ) and (
                    version.valid_until is None
                    or utc_us(other.valid_from) < utc_us(version.valid_until)
                )
                known = (
                    other.known_until is None
                    or utc_us(version.known_from) < utc_us(other.known_until)
                ) and (
                    version.known_until is None
                    or utc_us(other.known_from) < utc_us(version.known_until)
                )
                if effective and known:
                    raise SchemaError(
                        f"Path {role}: ambiguous overlapping ownership versions for child {binding.child}"
                    )
            by_child[binding.child].append(version)

    def _row_clocks(
        self, table: RelationalTable, row: Mapping[str, Any]
    ) -> tuple[datetime | None, datetime | None]:
        if table.immutable:
            return None, None
        assert table.event_time is not None and table.available_at is not None
        event = timestamp(row.get(table.event_time), f"{table.name}.{table.event_time}")
        available = timestamp(row.get(table.available_at), f"{table.name}.{table.available_at}")
        if available < event:
            raise SchemaError(f"{table.name}: observation predates event time")
        return event, available

    def _endpoints(self, relation: RelationalForeignKey, left: str, right: str) -> None:
        if left not in self.rows[relation.parent] or right not in self.rows[relation.child]:
            raise SchemaError(f"Path {relation.name}: dangling FK endpoint")

    def _path(self, name: str, names: Sequence[str]) -> tuple[tuple[str, bool, str], ...]:
        current, steps = self.task.root, []
        for role in names:
            relation = self.relations.get(role)
            if relation is None:
                raise SchemaError(f"History path {name}/{role}: undeclared relationship")
            if relation.parent == relation.child:
                raise SchemaError(
                    f"History path {name}/{role}: self-relationship direction is ambiguous"
                )
            if current == relation.parent:
                forward, current = True, relation.child
            elif current == relation.child:
                forward, current = False, relation.parent
            else:
                raise SchemaError(f"History path {name}/{role}: disconnected from table {current}")
            if self.tables[current].role != "input":
                raise SchemaError(
                    f"History path {name}/{role}: label table {current} cannot be an input"
                )
            steps.append((role, forward, current))
        if self.tables[current].immutable:
            raise SchemaError(f"History path {name}: terminal {current} must have an event clock")
        return tuple(steps)

    def visible(self, table: str, token: str, cutoff: datetime) -> bool:
        available = self.times[table, token][1]
        return available is None or available <= cutoff

    def _candidates(
        self, root: str, path: Sequence[tuple[str, bool, str]], cutoff: datetime
    ) -> list[tuple[str, datetime | None, tuple[Any, ...]]]:
        current = [(root, self.times[self.task.root, root][1], ())]
        for role, forward, table in path:
            following = []
            for token, observed, versions in current:
                for binding in self.adjacency.get((role, forward, token), ()):
                    left, right, known, version = (
                        binding.parent,
                        binding.child,
                        binding.available_at,
                        binding.version,
                    )
                    if (
                        (left if forward else right) != token
                        or known is not None
                        and known > cutoff
                    ):
                        continue
                    child = right if forward else left
                    if not self.visible(table, child, cutoff):
                        continue
                    if (
                        version is not None
                        and version.known_until is not None
                        and cutoff >= ensure_utc(version.known_until)
                    ):
                        continue
                    clocks = [
                        clock
                        for clock in (observed, known, self.times[table, child][1])
                        if clock is not None
                    ]
                    following.append(
                        (
                            child,
                            max(clocks) if clocks else None,
                            versions + (() if version is None else (version,)),
                        )
                    )
            current = following
        return current

    def resolve_history(self, anchor: GraphAnchor) -> tuple[ResolvedHistoryEvent, ...]:
        """Retain path observation clocks independently of terminal event time."""
        cutoff = ensure_utc(anchor.cutoff)
        if self.dataset.observed_through is not None and utc_us(cutoff) > utc_us(
            self.dataset.observed_through
        ):
            raise SchemaError(
                f"Root path {self.task.root}: cutoff exceeds declared source coverage"
            )
        if anchor.entity_id not in self.rows[self.task.root] or not self.visible(
            self.task.root, anchor.entity_id, cutoff
        ):
            raise SchemaError(f"Root path {self.task.root}: root absent or not yet observable")
        selected = {}
        for path in self.paths.values():
            terminal = path[-1][2]
            for token, known, versions in self._candidates(anchor.entity_id, path, cutoff):
                event_time = self.times[terminal, token][0]
                assert event_time is not None
                if event_time > cutoff or known is not None and known > cutoff:
                    continue
                if anchor.observation_start is not None and event_time < anchor.observation_start:
                    continue
                reference = event_time if self.task.ownership == "event_time" else cutoff
                if any(
                    reference < ensure_utc(v.valid_from)
                    or v.valid_until is not None
                    and reference >= ensure_utc(v.valid_until)
                    for v in versions
                ):
                    continue
                row, spec = self.rows[terminal][token], self.tables[terminal]
                numeric = {}
                for alias, column in spec.numeric.items():
                    value = row.get(column)
                    try:
                        numeric[alias] = math.nan if value is None else float(value)
                    except (TypeError, ValueError) as exc:
                        raise SchemaError(
                            f"{terminal}.{column}: expected numeric value or missing"
                        ) from exc
                categorical = {
                    alias: str(row[column])
                    for alias, column in spec.categorical.items()
                    if row.get(column) is not None
                }
                text = {
                    alias: str(row[column])
                    for alias, column in spec.text.items()
                    if row.get(column) is not None
                }
                event = TransactionEvent(
                    anchor.entity_id,
                    key_token((terminal, token)),
                    event_time,
                    numeric=numeric,
                    categorical=categorical,
                    text=text,
                    group=terminal,
                )
                assert known is not None
                previous = selected.get((terminal, token))
                if previous is None or known < previous.path_available_at:
                    selected[terminal, token] = ResolvedHistoryEvent(terminal, token, event, known)
        return tuple(
            sorted(
                selected.values(),
                key=lambda item: (item.event.event_timestamp, item.event.event_id),
            )
        )

    def history(self, anchor: GraphAnchor, *, max_events: int) -> TransactionSequence:
        """Select event-time owners using only relationship facts known at cutoff."""
        if max_events < 1:
            raise ValueError("max_events must be positive")
        ordered = self.resolve_history(anchor)
        kept = tuple(item.event for item in ordered[-max_events:])
        if kept:
            validate_events(kept)
        return TransactionSequence(
            anchor.entity_id, kept, ensure_utc(anchor.cutoff), None, sequence_id=anchor.sample_id
        )

    def profiles(self, anchors: Sequence[GraphAnchor]) -> dict[str, EntityProfile]:
        """Project visible root allowlists into the existing profile-token contract.

        A bounded table has one row per key; changing attribute snapshots require
        a versioned table declaration rather than a latest-state profile join.
        Immutable profiles use the earliest requested cutoff as snapshot metadata.
        """
        root = self.tables[self.task.root]
        result = {}
        for anchor in sorted(anchors, key=lambda item: item.cutoff):
            if anchor.entity_id in result:
                continue
            if anchor.entity_id not in self.rows[root.name] or not self.visible(
                root.name, anchor.entity_id, anchor.cutoff
            ):
                raise SchemaError(f"Root path {root.name}: profile is not observable at cutoff")
            row = self.rows[root.name][anchor.entity_id]
            attributes = {}
            for kind, mapped in (
                ("numeric", root.numeric),
                ("categorical", root.categorical),
                ("text", root.text),
            ):
                for alias, column in mapped.items():
                    value = row.get(column)
                    if kind == "numeric" and value is not None:
                        try:
                            finite = math.isfinite(float(value))
                        except (TypeError, ValueError) as exc:
                            raise SchemaError(
                                f"{root.name}.{column}: expected a numeric profile value"
                            ) from exc
                        if not finite:
                            continue
                    if value is not None:
                        attributes[alias] = str(value)
            result[anchor.entity_id] = EntityProfile(
                anchor.entity_id,
                self.times[root.name, anchor.entity_id][1] or anchor.cutoff,
                attributes,
            )
        return result

    def graph(self) -> RelationalGraph:
        """Project fixed FKs without dropping any row or link observation clocks.

        Numeric fields have explicit finite-value masks. Categorical/text graph
        encoding and version-expanded topology require separate fitted encoders;
        refuse them rather than silently omitting declared inputs or flattening
        ownership. Histories already support versioned ownership directly.
        """
        excluded = {name for name, table in self.tables.items() if table.role == "labels"}
        for role, bindings in self.edges.items():
            relation = self.relations[role]
            if relation.parent in excluded or relation.child in excluded:
                continue
            if any(binding.version is not None for binding in bindings):
                raise UnsupportedCombinationError(
                    f"Graph path {role}: event-time versioned ownership requires version-expanded topology; fixed projection is unsupported"
                )
        terminals = {path[-1][2] for path in self.paths.values()}
        intermediate = {table for path in self.paths.values() for _, _, table in path[:-1]}
        structural = {self.task.root} | (intermediate - terminals)
        tables, rows, links, relations = [], [], [], []
        for name, table in self.tables.items():
            if name in excluded:
                continue
            if table.categorical or table.text:
                raise UnsupportedCombinationError(
                    f"Graph path {name}: categorical/text allowlists require a fitted graph feature encoder"
                )
            features = tuple(
                field for alias in table.numeric for field in (f"value:{alias}", f"present:{alias}")
            ) or ("bias",)
            tables.append(
                GraphTable(
                    name,
                    features,
                    () if table.immutable else tuple(f"raw:{alias}" for alias in table.numeric),
                    immutable=table.immutable,
                    windowed=name not in structural,
                )
            )
            for token, row in self.rows[name].items():
                values = {} if table.numeric else {"bias": 1.0}
                for alias, column in table.numeric.items():
                    raw = row.get(column)
                    try:
                        number = math.nan if raw is None else float(raw)
                    except (TypeError, ValueError) as exc:
                        raise SchemaError(
                            f"{name}.{column}: expected numeric value or missing"
                        ) from exc
                    finite = math.isfinite(number)
                    if not table.immutable:
                        values[f"raw:{alias}"] = number
                    values[f"value:{alias}"] = number if finite else 0.0
                    values[f"present:{alias}"] = float(finite)
                event, available = self.times[name, token]
                rows.append(GraphRow(name, token, MappingProxyType(values), event, available))
        for name, relation in self.relations.items():
            if relation.parent in excluded or relation.child in excluded:
                continue
            role = (relation.parent, name, relation.child)
            relations.append(role)
            links.extend(
                GraphLink(role, binding.parent, binding.child, binding.available_at)
                for binding in self.edges[name]
            )
        return RelationalGraph(
            tuple(tables),
            tuple(relations),
            tuple(rows),
            tuple(links),
            self.dataset.forbidden_fields | PROHIBITED_INPUT_KEYS,
            observed_through=self.dataset.observed_through,
        )

    def _label_anchors(
        self, requested: set[tuple[str, datetime, str]] | None = None
    ) -> list[RelationalAnchor]:
        task = self.task
        spec = task.labels
        assert spec is not None
        if len(spec.root_columns) != len(self.tables[task.root].key):
            raise SchemaError(f"Label path {spec.table}: root key arity differs")
        if spec.table not in self.rows or self.tables[spec.table].role != "labels":
            raise SchemaError(f"Label path {spec.table}: declare a labels-role table")
        anchors: list[RelationalAnchor] = []
        seen = set()
        for row in self.dataset.rows[spec.table]:
            owner = key_token([row[column] for column in spec.root_columns])
            cutoff = timestamp(row[spec.timestamp], f"{spec.table}.{spec.timestamp}")
            split = str(row[spec.split])
            if requested is not None and (owner, cutoff, split) not in requested:
                continue
            if owner not in self.rows[task.root]:
                raise SchemaError(f"Label path {spec.table}: unknown root key")
            if split not in task.label_cutoffs:
                raise SchemaError(
                    f"Label path {spec.table}: declare fitting label_cutoffs[{split!r}]"
                )
            available = timestamp(row[spec.available_at], f"{spec.table}.{spec.available_at}")
            fitting_cutoff = timestamp(task.label_cutoffs[split], f"label_cutoffs.{split}")
            raw_target = row.get(spec.target)
            if raw_target is None:
                raise SchemaError(
                    f"Label path {spec.table}.{spec.target}: expected a binary target"
                )
            try:
                target = float(raw_target)
            except (TypeError, ValueError) as exc:
                raise SchemaError(
                    f"Label path {spec.table}.{spec.target}: expected a binary target"
                ) from exc
            if available < cutoff + timedelta(days=spec.horizon_days + spec.label_delay_days):
                raise SchemaError(
                    f"Label path {spec.table}: label observed before its declared horizon/delay"
                )
            if target not in {0.0, 1.0} or available > fitting_cutoff:
                raise SchemaError(
                    f"Label path {spec.table}: binary label unavailable at declared fitting cutoff"
                )
            sid = key_token((owner, cutoff.isoformat(), split))
            if sid in seen:
                raise SchemaError(f"Label path {spec.table}: duplicate root/cutoff/split")
            seen.add(sid)
            anchors.append(
                RelationalAnchor(
                    GraphAnchor(sid, owner, cutoff), split, target, available, fitting_cutoff
                )
            )
        return anchors

    def anchors(self) -> tuple[RelationalAnchor, ...]:
        """Generate anchor identities from task labels/cutoffs, or read replay anchors."""
        task, root = self.task, self.tables[self.task.root]
        if task.anchors is not None:
            import pyarrow.parquet as pq

            rows = pq.read_table(task.anchors).to_pylist()
            anchors = []
            for row in rows:
                owner = key_token([row[column] for column in root.key])
                cutoff = timestamp(row["cutoff"], "replay.cutoff")
                if row.get("sample_id") is None or not str(row["sample_id"]).strip():
                    raise SchemaError("Replay anchors require a nonempty sample_id")
                anchors.append(
                    RelationalAnchor(
                        GraphAnchor(
                            str(row["sample_id"]),
                            owner,
                            cutoff,
                            observation_start=None
                            if row.get("observation_start") is None
                            else timestamp(row["observation_start"], "replay.observation_start"),
                        ),
                        str(row.get("split", "predict")),
                    )
                )
        elif task.labels is not None and not task.cutoffs:
            anchors = self._label_anchors()
        else:
            anchors = []
            for split, raw_cutoff in task.cutoffs.items():
                cutoff = timestamp(raw_cutoff, f"cutoffs.{split}")
                for owner in sorted(self.rows[task.root]):
                    if self.visible(task.root, owner, cutoff):
                        anchors.append(
                            RelationalAnchor(
                                GraphAnchor(
                                    key_token((owner, cutoff.isoformat(), split)), owner, cutoff
                                ),
                                split,
                            )
                        )
        if (task.anchors is not None or task.cutoffs) and task.labels is not None:
            labels = {
                (item.anchor.entity_id, item.anchor.cutoff, item.split): item
                for item in self._label_anchors(
                    {(item.anchor.entity_id, item.anchor.cutoff, item.split) for item in anchors}
                )
            }
            replay = []
            for item in anchors:
                match = labels.get((item.anchor.entity_id, item.anchor.cutoff, item.split))
                if match is None:
                    if item.split != "predict":
                        raise SchemaError(
                            f"Replay anchor {item.anchor.sample_id}: missing available entity label"
                        )
                    replay.append(item)
                else:
                    replay.append(
                        RelationalAnchor(
                            item.anchor,
                            item.split,
                            match.target,
                            match.label_available_at,
                            match.label_cutoff,
                        )
                    )
            anchors = replay
        ids = [item.anchor.sample_id for item in anchors]
        if not ids or len(ids) != len(set(ids)):
            raise SchemaError("Task anchors must be nonempty and have unique sample IDs")
        split_by_root_cutoff: dict[tuple[str, int], str] = {}
        for item in anchors:
            pair = item.anchor.entity_id, utc_us(item.anchor.cutoff)
            previous = split_by_root_cutoff.setdefault(pair, item.split)
            if previous != item.split:
                raise SchemaError(
                    f"Task root/cutoff appears in more than one split ({previous!r}, {item.split!r})"
                )
            if self.dataset.observed_through is not None and utc_us(item.anchor.cutoff) > utc_us(
                self.dataset.observed_through
            ):
                raise SchemaError(
                    f"Root path {task.root}: anchor cutoff exceeds declared source coverage"
                )
            if item.anchor.entity_id not in self.rows[task.root] or not self.visible(
                task.root, item.anchor.entity_id, item.anchor.cutoff
            ):
                raise SchemaError(
                    f"Root path {task.root}: anchor is missing or not observable at cutoff"
                )
        return tuple(anchors)
