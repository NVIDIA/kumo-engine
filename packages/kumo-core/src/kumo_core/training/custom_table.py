# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""One check for a supplied training table, wherever one arrives.

DiskGraph reads a custom training table, so what a column has to be called and
what Arrow types it may be are DiskGraph's to say. It says so in
:data:`diskgraph.CUSTOM_TT_CONTRACT`, exported beside the reader arms that
enforce it and asserted against them there. This module reads that and refuses
an arriving table by name and by type. It declares no column names and no
accepted types of its own -- a second spelling of the spec, free to drift from
the first, is the thing this exists to remove.

Both backends call it. The hosted validation activity
(``rest/backends/data_backend/activities/custom_train_table.py``) runs it on
the parquet a connector produced; the in-process one
(:mod:`kumoai_studio.backend.local.train_table`) runs it on the parquet a caller
named. A table the cloud refuses is refused locally, in the same words.

Nothing here renames or casts a column. A table naming ``ENTITY`` where the
reader wants ``ENTITY1`` is refused with a message naming ``ENTITY1``, on
every backend and every connector.
"""

from __future__ import annotations

import base64
import functools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
from kumo_core.types.conversions import pyarrow_to_dtype
from kumoapi.common import KumoValidationError, ValidationWarning
from kumoapi.typing import Dtype

if TYPE_CHECKING:
    from kumo_core.types import TaskType


@functools.lru_cache(maxsize=1)
def contract() -> Mapping[str, Any]:
    """The column contract, read from the reader that enforces it."""
    from diskgraph import CUSTOM_TT_CONTRACT

    return CUSTOM_TT_CONTRACT


#: How a contract token is decoded into a question about an Arrow type. The
#: only thing this module says of its own accord, and mechanical: a token
#: DiskGraph grows without an entry here raises rather than reading as "no".
_PREDICATES = {
    'int8': pa.types.is_int8,
    'int16': pa.types.is_int16,
    'int32': pa.types.is_int32,
    'int64': pa.types.is_int64,
    'float32': pa.types.is_float32,
    'float64': pa.types.is_float64,
    'string': pa.types.is_string,
    'boolean': pa.types.is_boolean,
    'decimal128': pa.types.is_decimal128,
    'timestamp': pa.types.is_timestamp,
    'list': pa.types.is_list,
    'large_list': pa.types.is_large_list,
    'fixed_size_list': pa.types.is_fixed_size_list,
}


def _reads(dtype: pa.DataType, tokens: Sequence[str]) -> bool:
    return any(_PREDICATES[token](dtype) for token in tokens)


#: Which of the contract's type lists answers for each role. A list-valued
#: target and a per-element weight are asked about by container first, so they
#: are not here.
_ROLE_TYPES = {
    'entity': 'entity_types',
    'target': 'scalar_types',
    'split': 'scalar_types',
    'timestamp': 'timestamp_types',
    'weight': 'weight_types',
}


def reads(dtype: pa.DataType, role: str) -> bool:
    """Whether DiskGraph reads :obj:`dtype` as a column of :obj:`role`."""
    return _reads(dtype, contract()[_ROLE_TYPES[role]])


def accepted(role: str) -> str:
    """What a column of :obj:`role` may be, for a message that says so."""
    return _named(contract()[_ROLE_TYPES[role]])


def _named(tokens: Sequence[str]) -> str:
    return ', '.join(tokens)


def entity_column(index: int = 1) -> str:
    return f"{contract()['entity_column_prefix']}{index}"


def target_column(index: int = 1) -> str:
    return f"{contract()['target_column_prefix']}{index}"


def timestamp_column() -> str:
    return str(contract()['timestamp_column'])


def anchor_is_sentinel(
    task_type: 'TaskType',
    *,
    temporal: bool,
    entity_has_time_column: bool,
    list_distinct_target_is_array: bool,
) -> bool:
    """Whether generation uses a zero anchor for these resolved facts."""
    if temporal:
        return False
    if task_type.is_node_pred:
        return not entity_has_time_column
    return task_type.is_link_pred and list_distinct_target_is_array


def timestamp_span_ns(
    metadata: Iterable[Optional[Any]],
    column: str,
) -> Optional[tuple[float, float]]:
    """Fold one timestamp column's footer bounds without reading its rows."""
    low: Any = None
    high: Any = None
    for footer in metadata:
        if footer is None:
            return None
        if column not in footer.schema.names:
            return None
        index = footer.schema.names.index(column)
        for group in range(footer.num_row_groups):
            stats = footer.row_group(group).column(index).statistics
            if stats is None or not stats.has_min_max:
                return None
            low = stats.min if low is None else min(low, stats.min)
            high = stats.max if high is None else max(high, stats.max)
    if low is None:
        return None
    return float(pd.Timestamp(low).value), float(pd.Timestamp(high).value)


@functools.lru_cache(maxsize=1)
def _positional() -> re.Pattern:
    """Matches the numbered columns and nothing else: ``ENTITY1``, not
    ``ENTITY``, and not ``ENTITY_ID``."""
    prefixes = '|'.join(
        re.escape(str(contract()[key]))
        for key in ('entity_column_prefix', 'target_column_prefix'))
    return re.compile(f'({prefixes})[0-9]+')


def is_positional(name: str) -> bool:
    return _positional().fullmatch(name) is not None


def column_role(
    name: str,
    *,
    weight_col: Optional[str] = None,
    split_col: Optional[str] = None,
) -> str:
    r"""Which of the reader's columns :obj:`name` is, if any.

    ``'other'`` for a feature column the reader copies through without reading
    it as anything. A column is its role by its own name: ``ENTITY1`` is an
    entity key and ``ENTITY`` is a feature column that happens to be spelled
    like one.
    """
    if weight_col is not None and name == weight_col:
        return 'weight'
    if split_col is not None and name == split_col:
        return 'split'
    if name == timestamp_column():
        return 'timestamp'
    if is_positional(name):
        prefix = str(contract()['entity_column_prefix'])
        return 'entity' if name.startswith(prefix) else 'target'
    return 'other'


@dataclass(frozen=True)
class SuppliedTableRules:
    r"""What a run requires of a supplied table, beside the contract.

    Args:
        task_type: Decides whether a target is one value or a list of them,
            and whether a negative weight is meaningful.
        weight_col: The column carrying per-row weights, if the specification
            names one. A specification naming none describes the table as it
            stands, which DiskGraph reads with every row weighing one.
        split_col: The column carrying a manual split, if one is expected.
        anchored: Whether generation writes a real anchor for this query. A
            static query's anchor is filled by DiskGraph, so the table may
            omit it; a temporal query's may not.
        allow_row_count_mismatch: The deployment's escape hatch, honoured by
            both backends under the same name.
    """
    task_type: 'TaskType'
    weight_col: Optional[str] = None
    split_col: Optional[str] = None
    anchored: bool = True
    allow_row_count_mismatch: bool = False

    def __post_init__(self) -> None:
        # A Temporal activity serialises its arguments, and `TaskType` is a
        # `str` subclass -- so a hosted caller's enum arrives here as a bare
        # string, without the properties below. Coerced once, at the boundary
        # the two backends share, rather than at each call site.
        from kumo_core.types import TaskType

        if not isinstance(self.task_type, TaskType):
            object.__setattr__(self, 'task_type', TaskType(self.task_type))

    @property
    def list_target(self) -> bool:
        return bool(self.task_type.is_link_pred)

    @property
    def allow_negative_weight(self) -> bool:
        from kumo_core.types import TaskType

        return self.task_type is TaskType.TEMPORAL_LINK_PREDICTION


def _error(title: str, message: str) -> KumoValidationError:
    return KumoValidationError(title=title, message=message)


# Names #####################################################################


def contract_required_columns(rules: SuppliedTableRules) -> list[str]:
    """What the reader looks up, for a table with no generated one behind it."""
    # A supplied training table is labelled rows: the target is what makes it
    # one, so it is required for every task, not only where it happens to be a
    # list. `list_target` answers what shape the target has, not whether one
    # is needed.
    required = [entity_column(), target_column()]
    if rules.anchored:
        required.append(timestamp_column())
    return required


def generated_required_columns(
    generated_names: Sequence[str],
    *,
    anchor_optional: bool = False,
) -> list[str]:
    """What a table standing in for a generated one has to carry.

    Every generated column, because only *additions* are supported. The anchor
    is the one column a caller may leave to be filled: a static query's is a
    sentinel that ingest writes, so at that point the table does not have it
    yet.
    """
    anchor = timestamp_column()
    return [
        name for name in generated_names
        if not (anchor_optional and name == anchor)
    ]


def missing_column_errors(
    names: Sequence[str],
    *,
    required: Sequence[str],
    weight_col: Optional[str] = None,
    split_col: Optional[str] = None,
) -> list[KumoValidationError]:
    r"""Refuses a table that does not carry a column it has to.

    Checked on names alone, so a warehouse's column metadata answers it as
    well as a parquet footer does -- which is what lets the cloud refuse a
    misnamed table before it pays to export one.

    Nothing is matched by any name but its own. A table calling its entity
    column ``ENTITY`` is refused with a message naming ``ENTITY1``, rather
    than accepted here and read as missing later.
    """
    carried = set(names)
    errors: list[KumoValidationError] = []
    for name in required:
        if name not in carried:
            errors.append(
                _error(
                    'Required column not found',
                    f'Source table is missing {name}, which is the name '
                    f'DiskGraph reads that column under and the only name it '
                    f'reads it under. It carries {sorted(carried)}.'))
    for named, what in ((weight_col, 'weight'), (split_col, 'split')):
        if named and named not in carried:
            errors.append(
                _error(
                    'Column not found',
                    f'The training-table specification names {named!r} as its '
                    f'{what} column, which the supplied table does not have. '
                    f'It carries {sorted(carried)}.'))
    return errors


# What the file stores ######################################################


def stored_schema(
    schema: pa.Schema,
    footer_metadata: Optional[Mapping[bytes, bytes]],
) -> pa.Schema:
    r"""The schema the reader opens the file with, which pyarrow may not
    report.

    arrow-rs restores a column's dictionary encoding from the parquet
    footer's ``ARROW:schema``; pyarrow reports a dictionary over ints as its
    value type, so a gate reading the reported schema cannot see the encoding
    that reaches ``cell_to_value`` as a ``DictionaryArray`` it drops every row
    of. Both backends read a supplied table in place, so both gate on this
    rather than on what pyarrow says.

    :obj:`footer_metadata` is the parquet key/value metadata of the file the
    schema came from, however the caller opened it. ``None`` -- or a footer
    this cannot be read out of -- leaves the schema as reported.
    """
    payload = (footer_metadata or {}).get(b'ARROW:schema')
    if payload is None:
        return schema
    try:
        embedded = pa.ipc.read_schema(
            pa.BufferReader(base64.b64decode(payload)))
    except ValueError:
        return schema
    return pa.schema([
        embedded.field(field_.name) if
        (field_.name in embedded.names and pa.types.is_dictionary(
            embedded.field(field_.name).type)) else field_ for field_ in schema
    ])


# Types #####################################################################


def column_type_errors(
    schema: pa.Schema,
    rules: SuppliedTableRules,
) -> list[KumoValidationError]:
    r"""Refuses every column the reader would not read as what it is.

    DiskGraph does not refuse these itself: an entity key it cannot read
    resolves to no node and the row is skipped, a target it cannot read is
    substituted, and the run reports success over whatever survived. So they
    are refused here, before a run is launched on them.
    """
    errors: list[KumoValidationError] = []
    for field_ in schema:
        role = column_role(field_.name, weight_col=rules.weight_col,
                           split_col=rules.split_col)
        for error in _field_errors(field_, role, rules):
            errors.append(error)
    return errors


def _field_errors(
    field_: pa.Field,
    role: str,
    rules: SuppliedTableRules,
) -> Iterable[KumoValidationError]:
    name, dtype = field_.name, field_.type

    if role == 'entity':
        if not reads(dtype, 'entity'):
            yield _error(
                'Column type cannot be read',
                f'Column {name!r} is {dtype}. An entity key is read as one '
                f"of {accepted('entity')}; DiskGraph resolves anything else "
                f'to no node and drops the row without counting it. Cast the '
                f'column to one of those.')
    elif role == 'timestamp':
        if not reads(dtype, 'timestamp'):
            yield _error(
                'Column type cannot be read',
                f'Column {name!r} is {dtype}. An anchor is read as one of '
                f"{accepted('timestamp')}; DiskGraph reads anything else as a "
                f'null anchor and drops the row. Cast the column to one of '
                f'those.')
    elif role == 'target':
        yield from _target_errors(name, dtype, rules)
    elif role == 'weight':
        yield from _weight_type_errors(name, dtype, rules)
    elif role == 'split':
        if not reads(dtype, 'split'):
            yield _error(
                'Column type cannot be read',
                f'Column {name!r} is {dtype}. A split column is read as one '
                f"of {accepted('split')}. Cast the column to one of those.")


def is_list_target(dtype: pa.DataType) -> bool:
    r"""Whether :obj:`dtype` is a target column shaped as a list of targets.

    Wider than the contract's ``list_target_containers`` on purpose, and the
    only place that is widened. The contract names the containers DiskGraph
    *reads*; a ``fixed_size_list`` target is list-shaped and read as an empty
    target list, so it has to reach the refusal that says so rather than fall
    through to the scalar rule. Everything that asks whether a target is a
    list -- the type gate, the weight's arity, the paired-length check -- asks
    here, so none of them can disagree about one column.
    """
    return (_reads(dtype, contract()['list_target_containers'])
            or pa.types.is_fixed_size_list(dtype))


def _target_errors(
    name: str,
    dtype: pa.DataType,
    rules: SuppliedTableRules,
) -> Iterable[KumoValidationError]:
    spec = contract()
    is_list = is_list_target(dtype)
    if rules.list_target and not is_list:
        yield _error(
            f'{name} must be a list',
            f'A link-prediction training table carries one list of targets '
            f'per row, so {name!r} has to be one of '
            f'{_named(spec["list_target_containers"])}, and it is {dtype}. '
            f'Cast the column to one of those.')
        return
    if is_list and not rules.list_target:
        yield _error(
            f'{name} must not be a list',
            f'A {rules.task_type.value} training table carries one target per '
            f'row, so {name!r} has to be one of '
            f"{accepted('target')}, and it is {dtype}. DiskGraph reads no "
            f'value out of a list here and labels every row NaN. Cast the '
            f'column to one of those.')
        return
    if is_list:
        if not _reads(dtype, spec['list_target_containers']):
            yield _error(
                'Column type cannot be read',
                f'Column {name!r} is {dtype}, which DiskGraph does not read: '
                f'a list target has to be one of '
                f'{_named(spec["list_target_containers"])}, and anything else '
                f'reads as an empty target list. Cast the column to one of '
                f'those.')
            return
        item = dtype.value_type
        if not _reads(item, spec['list_target_item_types']):
            yield _error(
                'Column type cannot be read',
                f'Column {name!r} holds {item} inside its list, which '
                f'DiskGraph does not read: a list target is read as one of '
                f'{_named(spec["list_target_item_types"])}, and anything else '
                f'as an empty target list. Cast the column to a list of one '
                f'of those.')
        return
    if not reads(dtype, 'target'):
        yield _error(
            'Column type cannot be read',
            f'Column {name!r} is {dtype}. A target is read as one of '
            f"{accepted('target')}. Cast the column to one of those.")


def _weight_type_errors(
    name: str,
    dtype: pa.DataType,
    rules: SuppliedTableRules,
) -> Iterable[KumoValidationError]:
    spec = contract()
    if _reads(dtype, spec['weight_list_containers']):
        item = dtype.value_type
        if not _reads(item, spec['weight_list_item_types']):
            yield _error(
                'Column type cannot be read',
                f'Column {name!r} holds {item} inside its list. A '
                f'per-element weight is read as one of '
                f'{_named(spec["weight_list_item_types"])}. Cast the column '
                f'to a list of one of those.')
        return
    if not reads(dtype, 'weight'):
        yield _error(
            'Column type cannot be read',
            f'Column {name!r} is {dtype}. A weight is read as one of '
            f'{_named(spec["weight_types"])}, or as a list of '
            f'{_named(spec["weight_list_item_types"])} beside a list target.')


# The graph's primary key ###################################################
#
# What an entity key has to be is two questions, and only the first is the
# contract's: DiskGraph reads `ENTITY1` as one of `entity_types`, and then
# resolves the value against the graph's own primary key. A string key against
# an int64-keyed graph passes the first and resolves to nothing under the
# second, so every row is skipped and the run reports success on no seeds.
#
# The pairing lives here because both the training and the prediction path ask
# it, of the same two things, and the answer is the reader's rather than
# either caller's. It was `custom_projection_type_compatible` in the
# direct-Databricks projection; that projection went with the vendor path
# (#30914) and this is what survived it.


def entity_key_dtype(arrow_type: pa.DataType) -> Dtype:
    """The Kumo dtype an ``ENTITY1`` column of :obj:`arrow_type` carries."""
    if pa.types.is_decimal(arrow_type):
        return Dtype.int64 if arrow_type.scale == 0 else Dtype.float64
    try:
        return pyarrow_to_dtype(arrow_type)
    except KeyError:
        return Dtype.unsupported


def entity_key_resolves(
    arrow_type: pa.DataType,
    entity_pk_dtype: Dtype,
) -> bool:
    r"""Whether a key of :obj:`arrow_type` resolves against the graph's own.

    Integer against integer and string against string, and nothing else. A
    float key does not appear: the graph refuses one before it can be a
    primary key at all, because a float column yields zero nodes at ingest.
    """
    supplied = entity_key_dtype(arrow_type)
    if supplied == Dtype.unsupported or entity_pk_dtype == Dtype.unsupported:
        return False
    if entity_pk_dtype.is_string():
        return supplied.is_string()
    if entity_pk_dtype.is_int():
        return supplied.is_int()
    return False


def entity_key_errors(
    schema: pa.Schema,
    *,
    entity_pk_dtype: Optional[Dtype],
) -> list[KumoValidationError]:
    r"""Refuses an entity column the graph's primary key cannot resolve.

    The check the prediction path already makes
    (``materialize_prediction_task.py``), made by the training path too. On
    every path where a supplied table stands in for a *generated* one this is
    implied by :func:`correspondence_errors`, which demands exact Arrow parity
    against the generated entity column and so against the graph's key type.
    A registered table has no generated one behind it, and had nothing.

    :obj:`entity_pk_dtype` of ``None`` -- a graph whose entity table declares
    no typed primary key -- checks nothing, leaving that to the graph's own
    validation rather than reporting it as a fault of the supplied table.
    """
    entity = entity_column()
    if entity_pk_dtype is None or entity not in schema.names:
        return []
    dtype = schema.field(entity).type
    if not reads(dtype, 'entity'):
        return []  # Refused by type already; one message, not two.
    supplied = entity_key_dtype(dtype)
    if entity_key_resolves(dtype, entity_pk_dtype):
        return []
    return [
        _error(
            'Entity column does not match the graph primary key',
            f'Column {entity!r} is {supplied.value} while the graph primary '
            f'key it is resolved against is {entity_pk_dtype.value}. '
            f'DiskGraph resolves a key of the wrong type to no node and drops '
            f'the row without counting it, so the run would train on the rows '
            f'that happened to match and report success. Cast the column to '
            f'the graph primary-key type.')
    ]


# Contents ##################################################################

#: The roles :class:`Tally` reads, in the order it names them. A feature
#: column is copied through and never decoded.
_SCANNED_ROLES = ('entity', 'target', 'timestamp', 'weight', 'split')


def scanned_columns(
    schema: pa.Schema,
    rules: SuppliedTableRules,
) -> list[str]:
    r"""The columns :class:`Tally` reads, and so the only ones a caller has to
    project off disk. A supplied table may carry any number of feature columns
    beside these, and none of them is read here.
    """
    by_role: dict[str, list[str]] = {role: [] for role in _SCANNED_ROLES}
    for name in schema.names:
        role = column_role(name, weight_col=rules.weight_col,
                           split_col=rules.split_col)
        if role in by_role:
            by_role[role].append(name)
    names = [name for role in _SCANNED_ROLES for name in by_role[role]]
    return names or schema.names[:1]


@dataclass
class Tally:
    r"""Every check that needs rows, folded batch by batch.

    One pass, bounded by the batch: a supplied table is the size of the
    generated one and neither backend may hold it whole.
    """
    schema: pa.Schema
    rules: SuppliedTableRules
    known_keys: Optional[Any] = None

    rows: int = 0
    null_entities: list[str] = field(default_factory=list)
    null_targets: list[str] = field(default_factory=list)
    null_target_items: list[str] = field(default_factory=list)
    empty_target_counts: dict = field(default_factory=dict)
    list_target_null_counts: dict = field(default_factory=dict)
    null_timestamp: bool = False
    weight_nulls: bool = False
    weight_nan: bool = False
    weight_inf: bool = False
    weight_negative: bool = False
    weight_values: int = 0
    weight_max: float = float('-inf')
    lengths_differ: bool = False
    anchor_max: Optional[float] = None
    stale: int = 0
    stale_sample: list[Any] = field(default_factory=list)
    split_nulls: bool = False
    split_values: set = field(default_factory=set)

    SAMPLE = 5

    #: What a manual split column means. Required rather than compared
    #: against the plan: every producer of one on this path hardcodes these
    #: three, and `ColumnDeciderManual` reads nothing else as a split.
    SPLIT_VALUES = {0: 'train', 1: 'validation', 2: 'test'}

    def __post_init__(self) -> None:
        names = self.schema.names
        self._entities = [
            name for name in names
            if column_role(name, weight_col=self.rules.weight_col,
                           split_col=self.rules.split_col) == 'entity'
        ]
        self._targets = [
            name for name in names
            if column_role(name, weight_col=self.rules.weight_col,
                           split_col=self.rules.split_col) == 'target'
        ]
        self._anchor = timestamp_column() if timestamp_column() in names \
            else None
        self._int_anchor = (self._anchor is not None
                            and pa.types.is_int64(self.schema.field(
                                self._anchor).type))
        split = self.rules.split_col
        self._split = split if split in names else None
        weight = self.rules.weight_col
        self._weight = weight if weight in names else None
        self._weight_is_list = (self._weight is not None and _reads(
            self.schema.field(self._weight).type,
            contract()['weight_list_containers']))
        self._list_targets = {
            name
            for name in self._targets
            if is_list_target(self.schema.field(name).type)
        }
        target = target_column()
        self._paired = (self._weight_is_list and target in names
                        and is_list_target(self.schema.field(target).type))
        self._primary = entity_column()

    def feed(self, batch: pa.RecordBatch) -> None:
        self.rows += batch.num_rows
        names = batch.schema.names
        if self.known_keys is not None and self._primary in names:
            keys = batch.column(self._primary).to_numpy(zero_copy_only=False)
            missing = keys[~np.isin(keys, self.known_keys)]
            self.stale += len(missing)
            room = self.SAMPLE - len(self.stale_sample)
            self.stale_sample.extend(missing[:room].tolist())
        for name in self._entities:
            if name in names and batch.column(
                    name).null_count and name not in self.null_entities:
                self.null_entities.append(name)
        for name in self._targets:
            if name not in names:
                continue
            targets = batch.column(name)
            if targets.null_count and name not in self.null_targets:
                self.null_targets.append(name)
            if name not in self._list_targets:
                continue
            # A null list *slot* is what `null_count` answers, and neither
            # thing DiskGraph does quietly is one: a null inside the list is
            # substituted, and an empty list routes its row to UNUSED_STAGE.
            if (targets.flatten().null_count
                    and name not in self.null_target_items):
                self.null_target_items.append(name)
            if targets.null_count:
                self.list_target_null_counts[name] = (
                    self.list_target_null_counts.get(name, 0)
                    + targets.null_count)
            empty = int(np.sum(_lengths(targets) == 0))
            if empty:
                self.empty_target_counts[name] = (
                    self.empty_target_counts.get(name, 0) + empty)
        if self._anchor is not None and self._anchor in names:
            anchors = batch.column(self._anchor)
            self.null_timestamp |= bool(anchors.null_count)
            if self._int_anchor and len(anchors):
                span = anchors.to_numpy(zero_copy_only=False)
                high = float(np.nanmax(span))
                self.anchor_max = (high if self.anchor_max is None else max(
                    self.anchor_max, high))
        if self._split is not None and self._split in names:
            splits = batch.column(self._split)
            self.split_nulls |= bool(splits.null_count)
            if len(splits) and len(self.split_values) <= len(
                    self.SPLIT_VALUES) + self.SAMPLE:
                # Bounded: enough distinct values to name what is wrong,
                # without holding a distinct set over the whole table.
                self.split_values.update(
                    splits.drop_null().unique().to_pylist())
        if self._weight is None or self._weight not in names:
            return
        values = batch.column(self._weight)
        if self._paired and not values.null_count:
            self.lengths_differ |= not np.array_equal(
                _lengths(values), _lengths(batch.column(target_column())))
        flat = (values.flatten() if self._weight_is_list else values).cast(
            pa.float64())
        weights = flat.to_numpy(zero_copy_only=False)
        self.weight_nulls |= bool(values.null_count)
        self.weight_nan |= bool(np.isnan(weights).any())
        self.weight_inf |= bool(np.isinf(weights).any())
        self.weight_negative |= bool((weights < 0).any())
        self.weight_values += len(weights)
        if len(weights):
            self.weight_max = max(self.weight_max, float(weights.max()))


def _lengths(values: Any) -> Any:
    return values.value_lengths().to_numpy(zero_copy_only=False)


def content_errors(
    tally: Tally,
    rules: SuppliedTableRules,
) -> list[KumoValidationError]:
    r"""Refuses what only the rows can say.

    Every rule here names something DiskGraph does *quietly*: an empty table
    trains on nothing, a null entity key skips its row, a null target is
    relabelled, a null anchor drops its row. Each reports success.
    """
    errors: list[KumoValidationError] = []
    if not tally.rows:
        errors.append(
            _error(
                'Supplied table is empty',
                'The supplied training table has no rows, so the run would '
                'train on nothing and still report success.'))
    for name in tally.null_entities:
        errors.append(
            _error(
                f'{name} contains N/A',
                f'Column {name} has null entries, and DiskGraph skips a row '
                f'whose entity key is null rather than refusing it.'))
    for name in tally.null_targets:
        errors.append(
            _error(
                f'{name} contains N/A',
                f'Column {name} has null entries, and DiskGraph keeps such a '
                f'row with the label substituted rather than refusing it -- a '
                f'phantom class for a categorical target, a NaN label for a '
                f'numerical one, and success reported either way.'))
    for name in tally.null_target_items:
        errors.append(
            _error(
                f'{name} contains N/A inside its lists',
                f'Column {name} has null elements inside its target lists, '
                f'and DiskGraph substitutes each of them -- NaN, zero or the '
                f'empty string -- rather than refusing the row.'))
    for name, empty in tally.empty_target_counts.items():
        nulls = tally.list_target_null_counts.get(name, 0)
        if tally.rows - empty - nulls <= 0:
            errors.append(
                _error(
                    f'{name} contains empty lists',
                    f'Column {name} has an empty target list on every row, '
                    f'and DiskGraph puts such a row in no split at all '
                    f'rather than refusing it, so the run would train on '
                    f'nothing and report success.'))
    if tally.null_timestamp:
        errors.append(
            _error(
                f'{timestamp_column()} contains N/A',
                f'Column {timestamp_column()} has null entries, and DiskGraph '
                f'drops a row whose anchor is null rather than refusing it.'))
    if rules.weight_col:
        errors.extend(_weight_content_errors(tally, rules))
    if rules.split_col:
        errors.extend(_split_content_errors(tally, rules))
    return errors


def split_key(value: Any) -> str:
    r"""How DiskGraph renders a split cell before it matches one.

    ``value_to_string_key`` and then ``ColumnDeciderManual.decide_str``
    (``split_decider.rs``): the plan's split values cross as strings and the
    cell is rendered to one, so ``0`` and ``'0'`` are the same value to it,
    and so is a whole float -- Rust prints ``0f64`` as ``0``, not ``0.0``. A
    bool renders ``true``/``false`` and a list renders empty, so neither ever
    matches.
    """
    if isinstance(value, bool):  # before int: a bool is an int in Python
        return 'true' if value else 'false'
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    return str(value)


def _split_content_errors(
    tally: Tally,
    rules: SuppliedTableRules,
) -> list[KumoValidationError]:
    r"""Refuses a split column that is not the 0/1/2 encoding that is read.

    Nothing downstream checks these. A column valued ``'train'``/``'val'``/
    ``'test'`` is dtype-legal and reads fine, and :func:`split_key` renders it
    to something the decider matches against none of the three sets -- so
    every row lands in ``UNUSED_STAGE``, outside every split, and the run
    trains on what is left and reports success.
    """
    split = rules.split_col
    encoding = ', '.join(f'{value}={name}'
                         for value, name in sorted(Tally.SPLIT_VALUES.items()))
    if tally.split_nulls:
        return [
            _error(
                'Split column contains N/A',
                f'Column {split!r} has null entries, which the decider reads '
                f'as no split at all. A split column assigns every row, and '
                f'is read as {encoding}.')
        ]
    expected = {str(value) for value in Tally.SPLIT_VALUES}
    unexpected = sorted(
        {repr(value)
         for value in tally.split_values if split_key(value) not in expected})
    if not unexpected:
        return []
    return [
        _error(
            'Split column is not the encoding that is read',
            f'Column {split!r} carries {", ".join(unexpected)}, which is not '
            f'how a split is read: {encoding}. A value outside those three '
            f'puts its row in no split at all, so the run would train on what '
            f'is left and report success. The value may be a number or the '
            f'digit as a string; both are read the same way.')
    ]


def _weight_content_errors(
    tally: Tally,
    rules: SuppliedTableRules,
) -> list[KumoValidationError]:
    weight = rules.weight_col
    if tally.weight_nulls or tally.weight_nan:
        return [
            _error(
                'Weight column contains N/A',
                f'Weight column {weight!r} contains null or NaN entries. '
                f'Replace them with valid weights.')
        ]
    if tally.lengths_differ:
        return [
            _error(
                f'Weight column length does not match {target_column()}',
                f'Weight column {weight!r} has rows whose list length differs '
                f'from {target_column()}\'s. DiskGraph weights each target '
                f'element, so the two lists must be the same length.')
        ]
    if tally.weight_inf:
        return [
            _error(
                'Weight column contains infinite values',
                f'Weight column {weight!r} contains infinite entries, which '
                f'DiskGraph refuses row by row.')
        ]
    errors = []
    if tally.weight_values and tally.weight_max <= 0:
        errors.append(
            _error(
                'Weight column contains no positive values',
                f'Weight column {weight!r} is non-positive throughout, which '
                f'leaves the loss unbounded or undefined.'))
    if tally.weight_negative and not rules.allow_negative_weight:
        errors.append(
            _error(
                'Weight column contains negative values',
                f'Weight column {weight!r} contains negative entries; a '
                f'weight scales a loss and must be non-negative for this '
                f'task.'))
    return errors


def weight_arity_errors(
    schema: pa.Schema,
    rules: SuppliedTableRules,
) -> list[KumoValidationError]:
    r"""Refuses a weight whose shape does not answer the target's.

    DiskGraph mirrors the weight's arity on the target's: a list target reads
    a list of weights and a scalar target reads one, so a mismatch is read as
    the wrong column rather than as an error.
    """
    weight = rules.weight_col
    target = target_column()
    if not weight or weight not in schema.names or target not in schema.names:
        return []
    weight_is_list = _reads(
        schema.field(weight).type, contract()['weight_list_containers'])
    target_is_list = is_list_target(schema.field(target).type)
    if weight_is_list == target_is_list:
        return []
    return [
        _error(
            'Weight column type mismatch for array target',
            f'Weight column {weight!r} is '
            f'{"a list" if weight_is_list else "a scalar"} while {target} is '
            f'{"a scalar" if weight_is_list else "a list"}. DiskGraph mirrors '
            f'the weight\'s arity on the target\'s, so the two must agree.')
    ]


def stale_key_warnings(tally: Tally) -> list[ValidationWarning]:
    if not tally.stale:
        return []
    sample = ', '.join(repr(key) for key in tally.stale_sample)
    return [
        ValidationWarning(
            title='Entity keys that are not in the graph',
            message=(f'{tally.stale} of {tally.rows} rows name an '
                     f'{entity_column()} the graph snapshot does not have, '
                     f'and DiskGraph skips such a row without counting it -- '
                     f'the run would train on {tally.rows - tally.stale} rows '
                     f'and report success. Missing keys include: {sample}.'))
    ]


def content_warnings(tally: Tally) -> list[ValidationWarning]:
    warnings = []
    for name, empty in tally.empty_target_counts.items():
        nulls = tally.list_target_null_counts.get(name, 0)
        if tally.rows - empty - nulls <= 0:
            continue
        warnings.append(
            ValidationWarning(
                title=f'{name} contains empty lists',
                message=(f'{empty} of {tally.rows} rows in column {name} '
                         f'have an empty target list, and DiskGraph puts '
                         f'each in no split at all rather than refusing it '
                         f'-- the run would train on '
                         f'{tally.rows - empty - nulls} rows and report '
                         f'success.')))
    return warnings


# Correspondence with the generated table ###################################


@dataclass(frozen=True)
class GeneratedTable:
    r"""What generation produced, which a supplied table stands in for."""
    schema: pa.Schema
    rows: int
    #: The anchor's range in nanoseconds, off the footers, where generation
    #: wrote a typed one. Used only to catch a supplied integer anchor in the
    #: wrong unit.
    anchor_span_ns: Optional[tuple[float, float]] = None


_WRONG_UNIT = {
    1_000_000_000: 'seconds',
    1_000_000: 'milliseconds',
    1_000: 'microseconds',
}


def correspondence_errors(
    schema: pa.Schema,
    tally: Tally,
    generated: GeneratedTable,
    rules: SuppliedTableRules,
) -> tuple[list[KumoValidationError], list[ValidationWarning]]:
    r"""Refuses a supplied table that is not the generated one plus columns.

    Only additions are supported: every generated column has to be present and
    to have kept its type, and the two row counts have to agree, because each
    supplied row stands for exactly one generated row.
    """
    errors: list[KumoValidationError] = []
    warnings: list[ValidationWarning] = []
    corresponds = tally.rows == generated.rows
    if not corresponds:
        message = (f'The supplied table has {tally.rows} rows against the '
                   f"generated table's {generated.rows}. Each supplied row "
                   f'corresponds to exactly one generated row.')
        if not rules.allow_row_count_mismatch:
            return [_error('Row count mismatch', message)], []
        warnings.append(
            ValidationWarning(
                title='Row count mismatch allowed',
                message=(f'{message} Allowed by the deployment; the columns '
                         f'are still compared, the anchors are not.')))

    for field_ in generated.schema:
        if field_.name not in schema.names:
            if field_.name == timestamp_column() and not rules.anchored:
                continue
            errors.append(
                _error(
                    'Column missing in the supplied table',
                    f'Column {field_.name!r} is in the generated training '
                    f'table and not in this one.'))
            continue
        actual = schema.field(field_.name).type
        if not _corresponds(field_.name, field_.type, actual):
            errors.append(
                _error(
                    'Column type mismatch',
                    f'Column {field_.name!r} is {actual} here and '
                    f'{field_.type} in the generated training table.'))
    if corresponds:
        errors += _anchor_unit_errors(tally, generated)
    return errors, warnings


def _corresponds(name: str, generated: pa.DataType,
                 supplied: pa.DataType) -> bool:
    r"""Whether a supplied column stands for a generated one.

    Exact Arrow parity, with the anchor the one exception: generation writes a
    typed timestamp and the reader takes every unit the contract names, so a
    supplied anchor is compared against the contract instead.
    """
    if generated == supplied:
        return True
    if name != timestamp_column() or not pa.types.is_timestamp(generated):
        return False
    return _reads(supplied, contract()['timestamp_types'])


def _anchor_unit_errors(
    tally: Tally,
    generated: GeneratedTable,
) -> list[KumoValidationError]:
    r"""An integer anchor carries no unit and is read as nanoseconds, so one
    written in any other unit dates every row wrongly without failing. Only a
    whole-unit discrepancy against the generated anchor is reported.
    """
    span = generated.anchor_span_ns
    if not tally.anchor_max or span is None or not span[1]:
        return []
    ratio = span[1] / tally.anchor_max
    unit = next((name for factor, name in _WRONG_UNIT.items()
                 if abs(ratio - factor) <= factor * 0.01), None)
    if unit is None:
        return []
    return [
        _error(
            f'{timestamp_column()} looks like {unit}, not nanoseconds',
            f'Column {timestamp_column()} is {tally.anchor_max:.0f} at its '
            f"largest against the generated table's {span[1]:.0f} "
            f'nanoseconds -- a factor of {ratio:.0f}, which is {unit}. An '
            f'integer anchor is read as nanoseconds and carries no unit of '
            f'its own, so one in any other unit dates every row wrongly '
            f'without failing.')
    ]
