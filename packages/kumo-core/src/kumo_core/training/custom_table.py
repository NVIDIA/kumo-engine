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

Row contents are validated by DiskGraph during ingestion. Both backends call
this module for schema checks. The hosted validation activity
(``rest/backends/data_backend/activities/custom_train_table.py``) runs it on
the parquet a connector produced; the in-process one
(:mod:`kumo.engine.pquery._custom_source`) runs it on the parquet a caller
named. A table the cloud refuses is refused locally, in the same words.

Nothing here renames or casts a column. A table naming ``ENTITY`` where the
reader wants ``ENTITY1`` is refused with a message naming ``ENTITY1``, on
every backend and every connector.
"""

from __future__ import annotations

import base64
import functools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional, Sequence

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
            and which target schema is required.
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


# Correspondence with the generated table ###################################


@dataclass(frozen=True)
class GeneratedTable:
    r"""What generation produced, which a supplied table stands in for."""
    schema: pa.Schema
    rows: int


def correspondence_errors(
    schema: pa.Schema,
    supplied_rows: int,
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
    corresponds = supplied_rows == generated.rows
    if not corresponds:
        message = (f'The supplied table has {supplied_rows} rows against the '
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
