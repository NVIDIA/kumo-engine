# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one check a supplied training table goes through, on either backend."""

from __future__ import annotations

import pyarrow as pa
import pytest
from kumo_core.training import custom_table
from kumo_core.types import TaskType
from kumoapi.typing import Dtype


def _rules(**overrides):
    settings = dict(task_type=TaskType.REGRESSION, weight_col='w')
    settings.update(overrides)
    return custom_table.SuppliedTableRules(**settings)


def _schema(**columns) -> pa.Schema:
    return pa.schema(list(columns.items()))


def _titles(errors) -> list[str]:
    return [error.title for error in errors]


_BASE = dict(ENTITY1=pa.int64(), TARGET1=pa.float64(), TIMESTAMP=pa.int64(),
             w=pa.float64())


# The contract itself ########################################################


def test_the_contract_comes_from_diskgraph() -> None:
    """Not declared here. If this ever stops being true, everything below is
    checking Kumo's opinion of the reader rather than the reader."""
    import diskgraph

    assert custom_table.contract() is diskgraph.CUSTOM_TT_CONTRACT


def test_the_column_names_are_the_contracts() -> None:
    spec = custom_table.contract()

    assert custom_table.entity_column() == f"{spec['entity_column_prefix']}1"
    assert custom_table.target_column() == f"{spec['target_column_prefix']}1"
    assert custom_table.timestamp_column() == spec['timestamp_column']


@pytest.mark.parametrize('task_type', list(TaskType))
@pytest.mark.parametrize('temporal', [False, True])
@pytest.mark.parametrize('entity_has_time_column', [False, True])
@pytest.mark.parametrize('list_distinct_target_is_array', [False, True])
def test_the_anchor_decision_uses_only_resolved_query_facts(
        task_type, temporal, entity_has_time_column,
        list_distinct_target_is_array) -> None:
    expected = (not temporal and
                ((task_type.is_node_pred and not entity_has_time_column) or
                 (task_type.is_link_pred and
                  list_distinct_target_is_array)))

    assert custom_table.anchor_is_sentinel(
        task_type,
        temporal=temporal,
        entity_has_time_column=entity_has_time_column,
        list_distinct_target_is_array=list_distinct_target_is_array,
    ) is expected


def test_every_token_the_contract_names_can_be_decoded() -> None:
    """A token DiskGraph grows without a decoder here has to raise, not read
    as "no". This is the test that fails when that happens."""
    spec = custom_table.contract()
    tokens = {
        token
        for key, value in spec.items() if key.endswith('_types')
        or key.endswith('_containers') for token in value
    }

    for token in tokens:
        assert custom_table._PREDICATES[token](pa.int64()) in (True, False)


def test_an_undecodable_token_raises_rather_than_reading_as_no(
        monkeypatch) -> None:
    monkeypatch.setattr(custom_table, '_PREDICATES', {})

    with pytest.raises(KeyError):
        custom_table.reads(pa.int64(), 'entity')


def test_the_rules_follow_the_contract_rather_than_a_copy_of_it(
        monkeypatch) -> None:
    """Nothing below is hard-coded: change what DiskGraph says and the
    validator changes with it."""
    spec = dict(custom_table.contract())
    spec['entity_types'] = ['string']
    spec['entity_column_prefix'] = 'KEY'
    monkeypatch.setattr(custom_table, 'contract', lambda: spec)
    custom_table._positional.cache_clear()
    try:
        assert not custom_table.reads(pa.int64(), 'entity')
        assert custom_table.entity_column() == 'KEY1'
        assert custom_table.contract_required_columns(
            _rules(anchored=False)) == ['KEY1', 'TARGET1']
    finally:
        custom_table._positional.cache_clear()


# Names ######################################################################


def test_the_entity_column_is_required_by_the_only_name_it_has() -> None:
    rules = _rules()

    errors = custom_table.missing_column_errors(
        ['ENTITY', 'TARGET1', 'TIMESTAMP', 'w'],
        required=custom_table.contract_required_columns(rules),
        weight_col='w')

    assert _titles(errors) == ['Required column not found']
    assert 'missing ENTITY1' in errors[0].message


def test_every_task_requires_the_target_by_name() -> None:
    rules = _rules(task_type=TaskType.TEMPORAL_LINK_PREDICTION)

    assert custom_table.contract_required_columns(rules) == [
        'ENTITY1', 'TARGET1', 'TIMESTAMP'
    ]


def test_a_static_query_does_not_require_the_anchor() -> None:
    """The anchor goes; the labels do not. A supplied training table is
    labelled rows whatever the query's shape."""
    assert custom_table.contract_required_columns(
        _rules(anchored=False)) == ['ENTITY1', 'TARGET1']


def test_a_weight_column_the_specification_names_has_to_be_there() -> None:
    errors = custom_table.missing_column_errors(['ENTITY1'],
                                                required=['ENTITY1'],
                                                weight_col='w')

    assert _titles(errors) == ['Column not found']
    assert "'w'" in errors[0].message


def test_the_anchor_may_be_left_for_ingest_to_synthesize() -> None:
    assert custom_table.generated_required_columns(
        ['ENTITY1', 'TIMESTAMP'], anchor_optional=True) == ['ENTITY1']
    assert custom_table.generated_required_columns(
        ['ENTITY1', 'TIMESTAMP']) == ['ENTITY1', 'TIMESTAMP']


# Types ######################################################################


@pytest.mark.parametrize('dtype', [pa.int64(), pa.float64(), pa.string()])
def test_an_entity_key_may_be_what_the_index_resolves(dtype) -> None:
    assert custom_table.reads(dtype, 'entity')


@pytest.mark.parametrize('dtype',
                         [pa.bool_(), pa.int32(),
                          pa.uint64(),
                          pa.large_string()])
def test_an_entity_key_may_not_be_anything_else(dtype) -> None:
    assert not custom_table.reads(dtype, 'entity')


def test_an_entity_key_is_narrower_than_a_target() -> None:
    """`cell_to_value` reads a boolean and `value_to_pk_key` does not, so a
    boolean entity loses every row while a boolean target is fine."""
    assert custom_table.reads(pa.bool_(), 'target')
    assert not custom_table.reads(pa.bool_(), 'entity')


def test_a_weight_may_be_a_decimal_and_a_list_of_them_may_not() -> None:
    spec = custom_table.contract()

    assert 'decimal128' in spec['weight_types']
    assert 'decimal128' not in spec['weight_list_item_types']


def test_a_list_target_may_use_either_supported_list_container() -> None:
    assert custom_table.contract()['list_target_containers'] == [
        'list', 'large_list']


@pytest.mark.parametrize(('column', 'dtype'), [
    ('ENTITY1', pa.bool_()),
    ('ENTITY1', pa.int32()),
    ('TARGET1', pa.binary()),
    ('TIMESTAMP', pa.date32()),
    ('TIMESTAMP', pa.string()),
    ('w', pa.string()),
])
def test_a_column_the_reader_would_not_read_is_refused(column, dtype) -> None:
    columns = dict(_BASE)
    columns[column] = dtype

    errors = custom_table.column_type_errors(_schema(**columns), _rules())

    assert _titles(errors) == ['Column type cannot be read']
    assert column in errors[0].message


def test_a_table_the_reader_reads_raises_nothing() -> None:
    assert custom_table.column_type_errors(_schema(**_BASE), _rules()) == []


def test_a_feature_column_is_carried_through_whatever_it_is() -> None:
    """Only the columns the reader decodes are gated. A feature column is
    copied and never read as anything."""
    columns = dict(_BASE, embedding=pa.list_(pa.float32()))

    assert custom_table.column_type_errors(_schema(**columns), _rules()) == []


def test_a_link_task_needs_its_target_to_be_a_list() -> None:
    rules = _rules(task_type=TaskType.TEMPORAL_LINK_PREDICTION)

    errors = custom_table.column_type_errors(_schema(**_BASE), rules)

    assert f'{custom_table.target_column()} must be a list' in _titles(errors)


def test_a_scalar_task_refuses_a_list_target() -> None:
    """The inverse of the rule above, and the same reason: DiskGraph's scalar
    ``cell_to_value`` reads nothing out of a ListArray, so every row would be
    labelled NaN and the run would report success over them."""
    columns = dict(_BASE, TARGET1=pa.list_(pa.float64()))

    errors = custom_table.column_type_errors(_schema(**columns), _rules())

    assert _titles(errors) == [
        f'{custom_table.target_column()} must not be a list'
    ]


# Arity ######################################################################


def test_a_weight_mirrors_its_targets_arity() -> None:
    columns = dict(_BASE, TARGET1=pa.list_(pa.float64()))

    errors = custom_table.weight_arity_errors(_schema(**columns), _rules())

    assert _titles(errors) == ['Weight column type mismatch for array target']


def test_a_list_weight_beside_a_list_target_agrees() -> None:
    columns = dict(_BASE, TARGET1=pa.list_(pa.float64()),
                   w=pa.list_(pa.float64()))

    assert custom_table.weight_arity_errors(_schema(**columns),
                                            _rules()) == []


# Correspondence #############################################################


def _generated(rows: int = 2, **columns) -> custom_table.GeneratedTable:
    return custom_table.GeneratedTable(schema=_schema(**(columns or _BASE)),
                                       rows=rows)


def test_a_row_count_that_does_not_correspond_is_refused() -> None:
    table = pa.table({
        'ENTITY1': pa.array([1], pa.int64()),
        'TARGET1': pa.array([0.1], pa.float64()),
        'TIMESTAMP': pa.array([0], pa.int64()),
        'w': pa.array([1.0], pa.float64()),
    })
    errors, warnings = custom_table.correspondence_errors(
        table.schema, table.num_rows, _generated(rows=2), _rules())

    assert _titles(errors) == ['Row count mismatch']
    assert warnings == []


def test_the_deployments_escape_hatch_downgrades_it_to_a_warning() -> None:
    rules = _rules(allow_row_count_mismatch=True)
    table = pa.table({
        'ENTITY1': pa.array([1], pa.int64()),
        'TARGET1': pa.array([0.1], pa.float64()),
        'TIMESTAMP': pa.array([0], pa.int64()),
        'w': pa.array([1.0], pa.float64()),
    })

    errors, warnings = custom_table.correspondence_errors(
        table.schema, table.num_rows, _generated(rows=2), rules)

    assert errors == []
    assert [warning.title for warning in warnings
            ] == ['Row count mismatch allowed']


def test_only_additions_are_supported() -> None:
    table = pa.table({
        'ENTITY1': pa.array([1, 2], pa.int64()),
        'TARGET1': pa.array([0.1, 0.2], pa.float64()),
        'TIMESTAMP': pa.array([0, 0], pa.int64()),
        'w': pa.array([1.0, 1.0], pa.float64()),
    })
    generated = _generated(**dict(_BASE, extra=pa.string()))

    errors, _ = custom_table.correspondence_errors(table.schema, table.num_rows,
                                                   generated, _rules())

    assert _titles(errors) == ['Column missing in the supplied table']


def test_a_generated_column_keeps_its_type() -> None:
    table = pa.table({
        'ENTITY1': pa.array([1, 2], pa.int64()),
        'TARGET1': pa.array([1, 2], pa.int64()),
        'TIMESTAMP': pa.array([0, 0], pa.int64()),
        'w': pa.array([1.0, 1.0], pa.float64()),
    })

    errors, _ = custom_table.correspondence_errors(table.schema, table.num_rows,
                                                   _generated(), _rules())

    assert _titles(errors) == ['Column type mismatch']


def test_the_anchor_is_the_one_column_compared_against_the_contract() -> None:
    """Generation writes a typed anchor; the reader takes an integer one too,
    so exact Arrow parity is relaxed for it and nothing else."""
    table = pa.table({
        'ENTITY1': pa.array([1, 2], pa.int64()),
        'TARGET1': pa.array([0.1, 0.2], pa.float64()),
        'TIMESTAMP': pa.array([0, 0], pa.int64()),
        'w': pa.array([1.0, 1.0], pa.float64()),
    })
    generated = _generated(**dict(_BASE, TIMESTAMP=pa.timestamp('ns')))

    errors, _ = custom_table.correspondence_errors(table.schema, table.num_rows,
                                                   generated, _rules())

    assert errors == []


# The graph's primary key ####################################################


def test_an_entity_the_graph_key_cannot_resolve_is_refused() -> None:
    """The pairing the correspondence check makes implicitly, for a table
    registered without a generated one behind it. A string key against an
    int64-keyed graph passes the contract's union and resolves to no node."""
    schema = _schema(**dict(_BASE, ENTITY1=pa.string()))

    errors = custom_table.entity_key_errors(schema,
                                            entity_pk_dtype=Dtype.int64)

    assert _titles(errors) == [
        'Entity column does not match the graph primary key'
    ]


@pytest.mark.parametrize(('dtype', 'pkey'), [
    (pa.int64(), Dtype.int64),
    (pa.int64(), Dtype.int),
    (pa.string(), Dtype.string),
])
def test_an_entity_the_graph_key_resolves_raises_nothing(dtype, pkey) -> None:
    schema = _schema(**dict(_BASE, ENTITY1=dtype))

    assert custom_table.entity_key_errors(schema, entity_pk_dtype=pkey) == []


def test_a_graph_with_no_typed_key_is_not_reported_as_a_table_fault() -> None:
    schema = _schema(**dict(_BASE, ENTITY1=pa.string()))

    assert custom_table.entity_key_errors(schema, entity_pk_dtype=None) == []


def test_an_unreadable_entity_is_named_once_and_not_twice() -> None:
    """An int32 key is refused by the contract's union already, so the
    pairing stays quiet rather than adding a second message about it."""
    schema = _schema(**dict(_BASE, ENTITY1=pa.int32()))

    assert custom_table.entity_key_errors(schema,
                                          entity_pk_dtype=Dtype.int64) == []


# What the file stores #######################################################


def test_a_dictionary_column_is_read_as_the_dictionary_it_is(tmp_path) -> None:
    """pyarrow reports a dictionary over ints as its value type, so a gate on
    the reported schema sees ``int64`` where arrow-rs restores a
    ``DictionaryArray`` and drops every row."""
    import pyarrow.parquet as pq

    path = str(tmp_path / 'dict.parquet')
    pq.write_table(
        pa.table({'ENTITY1': pa.array([1, 2], pa.int64()).dictionary_encode()
                  }), path)
    parquet = pq.ParquetFile(path)
    assert pa.types.is_int64(parquet.schema_arrow.field('ENTITY1').type)

    stored = custom_table.stored_schema(parquet.schema_arrow,
                                        parquet.metadata.metadata)

    assert pa.types.is_dictionary(stored.field('ENTITY1').type)
    assert custom_table.column_type_errors(
        stored, _rules(weight_col=None)) != []


def test_a_footer_naming_no_arrow_schema_is_read_as_reported() -> None:
    schema = _schema(**_BASE)

    assert custom_table.stored_schema(schema, None) is schema
    assert custom_table.stored_schema(schema, {}) is schema


# One definition of a list-shaped target #####################################


def test_a_list_shaped_target_the_reader_refuses_is_named_once() -> None:
    """``fixed_size_list`` is list-shaped and unreadable, so it reaches the type
    refusal that says so -- and the arity check, asking the same question,
    does not also call a list weight beside it a mismatch."""
    rules = _rules(task_type=TaskType.TEMPORAL_LINK_PREDICTION)
    columns = dict(_BASE, TARGET1=pa.list_(pa.float64(), list_size=2),
                   w=pa.list_(pa.float64()))
    schema = _schema(**columns)

    errors = (custom_table.column_type_errors(schema, rules) +
              custom_table.weight_arity_errors(schema, rules))

    assert _titles(errors) == ['Column type cannot be read']


def test_a_task_type_that_crossed_a_wire_is_still_a_task_type():
    r"""A Temporal activity serialises its arguments, and ``TaskType`` is a
    ``str`` subclass -- so a hosted caller's enum arrives as a bare string.
    Every rule that asks the task a question would then raise
    ``AttributeError`` inside the activity, surfaced to the caller as the
    generic internal-error message.
    """
    from kumo_core.types import TaskType

    rules = custom_table.SuppliedTableRules(
        task_type='temporal_link_prediction', weight_col='weight')

    assert rules.task_type is TaskType.TEMPORAL_LINK_PREDICTION
    assert rules.list_target is True
