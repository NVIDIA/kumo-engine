# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from kumoapi.graph import GraphDefinition
from kumoapi.pquery import ValidatedPredictiveQuery
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    DateOffsetRange,
    Filter,
    Join,
    LogicalOperation,
)
from kumoapi.pquery.AST.location_interval import ASTQueryLocationInterval
from kumoapi.typing import (
    AggregationType,
    BoolOp,
    Dtype,
    RelOp,
    StrOp,
    Stype,
    TimeUnit,
)
from kumopql.parser import PQLParser
from kumopql.rewriting import SplitFilter, StaticFilterSplitter
from kumopql.validator import PredictiveQueryValidator


def get_validated_query(graph: GraphDefinition,
                        query: str) -> ValidatedPredictiveQuery:
    parser = PQLParser()
    validator = PredictiveQueryValidator(graph)
    parsed_query = parser.to_parsed_predictive_query(query)
    validated_query, response = validator.validate_predictive_query(
        parsed_query)
    assert response.ok, response
    assert validated_query is not None
    return validated_query


def test_full_static() -> None:
    # Test single fully static condition
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=Condition(
            target=Column(fqn="entity_table.string_attribute"),
            op=RelOp.EQ,
            value="target_string_value",
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=entity.condition,
        non_applicable_filter=None,
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=0,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


def test_multiple_static() -> None:
    # Test multiple fully static conditions
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=LogicalOperation(
            left=Condition(
                target=Column(fqn="entity_table.numeric_attribute"),
                op=RelOp.GT,
                value=100,
            ),
            bool_op=BoolOp.AND,
            right=Condition(
                target=Column(fqn="entity_table.string_attribute"),
                op=RelOp.EQ,
                value="test_value",
            ),
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    expected = SplitFilter(
        applicable_filter=entity.condition,
        non_applicable_filter=None,
        rewritten=False,
        n_applicable_conditions=2,
        n_non_applicable_conditions=0,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )
    assert result == expected


def test_negation() -> None:
    # Test negation
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=LogicalOperation(
            left=Condition(
                target=Column(fqn="entity_table.numeric_attribute"),
                op=RelOp.GT,
                value=100,
            ),
            bool_op=BoolOp.NOT,
            right=None,
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=entity.condition,
        non_applicable_filter=None,
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=0,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


def test_mixed_and() -> None:
    # Test mixed static and non-static conditions separated by AND
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=LogicalOperation(
            left=Condition(
                target=Column(fqn="entity_table.numeric_attribute"),
                op=RelOp.GT,
                value=100,
            ),
            bool_op=BoolOp.AND,
            right=Condition(
                target=Aggregation(
                    aggr=AggregationType.COUNT,
                    target=Column(fqn="linked_fact_table.*"),
                    aggr_time_range=DateOffsetRange(
                        start=0,
                        end=7,
                        unit=TimeUnit.DAYS,
                    ),
                ),
                op=RelOp.GT,
                value=0,
            ),
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=Condition(
            target=Column(fqn="entity_table.numeric_attribute"),
            op=RelOp.GT,
            value=100,
        ),
        non_applicable_filter=Condition(
            target=Aggregation(
                aggr=AggregationType.COUNT,
                target=Column(fqn="linked_fact_table.*"),
                aggr_time_range=DateOffsetRange(
                    start=0,
                    end=7,
                    unit=TimeUnit.DAYS,
                ),
            ),
            op=RelOp.GT,
            value=0,
        ),
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


def test_mixed_or() -> None:
    # Test mixed static and non-static conditions separated by OR
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=LogicalOperation(
            left=Condition(
                target=Column(fqn="entity_table.string_attribute"),
                op=RelOp.EQ,
                value="test_value",
            ),
            bool_op=BoolOp.OR,
            right=Condition(
                target=Aggregation(
                    aggr=AggregationType.COUNT,
                    target=Column(fqn="linked_fact_table.*"),
                    aggr_time_range=DateOffsetRange(
                        start=0,
                        end=7,
                        unit=TimeUnit.DAYS,
                    ),
                ),
                op=RelOp.GT,
                value=0,
            ),
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=None,
        non_applicable_filter=entity.condition,
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=2,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


def test_single_temporal() -> None:
    # Test single temporal condition
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=Condition(
            target=Aggregation(
                aggr=AggregationType.COUNT,
                target=Column(fqn="linked_fact_table.*"),
                aggr_time_range=DateOffsetRange(
                    start=0,
                    end=7,
                    unit=TimeUnit.DAYS,
                ),
            ),
            op=RelOp.GT,
            value=0,
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=None,
        non_applicable_filter=entity.condition,
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


def test_multiple_temporal() -> None:
    # Test multiple temporal conditions
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=LogicalOperation(
            left=Condition(
                target=Aggregation(
                    aggr=AggregationType.COUNT,
                    target=Column(fqn="linked_fact_table.*"),
                    aggr_time_range=DateOffsetRange(
                        start=0,
                        end=7,
                        unit=TimeUnit.DAYS,
                    ),
                ),
                op=RelOp.GT,
                value=0,
            ),
            bool_op=BoolOp.AND,
            right=Condition(
                target=Aggregation(
                    aggr=AggregationType.SUM,
                    target=Column(fqn="linked_fact_table.value"),
                    aggr_time_range=DateOffsetRange(
                        start=7,
                        end=14,
                        unit=TimeUnit.DAYS,
                    ),
                ),
                op=RelOp.GT,
                value=1000,
            ),
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=None,
        non_applicable_filter=entity.condition,
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=2,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


def test_static_aggregation() -> None:
    # Test that includes a static aggregation
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=LogicalOperation(
            left=Condition(
                target=Aggregation(
                    aggr=AggregationType.LIST_DISTINCT,
                    target=Column(
                        fqn="entity_table.multicategorical_attribute"),
                    aggr_time_range=None,  # Static aggregation
                ),
                op=StrOp.CONTAINS,
                value="specific_category",
            ),
            bool_op=BoolOp.AND,
            right=Condition(
                target=Column(fqn="entity_table.status"),
                op=RelOp.EQ,
                value="active",
            ),
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=entity.condition,
        non_applicable_filter=None,
        rewritten=False,
        n_applicable_conditions=2,
        n_non_applicable_conditions=0,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


def test_static_aggregation_static_subfilter() -> None:
    # Test that includes a static aggregation with a static sub-filter
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=Condition(
            target=Aggregation(
                aggr=AggregationType.LIST_DISTINCT,
                target=Filter(
                    target=Column(fqn="linked_fact_table.category"),
                    condition=Condition(
                        target=Column(fqn="linked_fact_table.value"),
                        op=RelOp.GT,
                        value=1000,
                    ),
                ),
                aggr_time_range=None,  # Static aggregation
            ),
            op=StrOp.CONTAINS,
            value="high_value_category",
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=entity.condition,
        non_applicable_filter=None,
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=0,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


def test_static_aggregation_temporal_subfilter() -> None:
    # Test that includes a static aggregation with a temporal sub-filter
    splitter = StaticFilterSplitter()

    entity = Filter(
        target=Column(fqn="entity_table.entity_id"),
        condition=Condition(
            target=Aggregation(
                aggr=AggregationType.LIST_DISTINCT,
                target=Filter(
                    target=Column(fqn="linked_fact_table.category"),
                    condition=Condition(
                        target=Aggregation(
                            aggr=AggregationType.SUM,
                            target=Column(fqn="linked_fact_table.value"),
                            aggr_time_range=DateOffsetRange(
                                start=0,
                                end=7,
                                unit=TimeUnit.DAYS,
                            )),
                        op=RelOp.GT,
                        value=1000,
                    ),
                ),
                aggr_time_range=None,  # Static aggregation
            ),
            op=StrOp.CONTAINS,
            value="high_value_category",
        ),
    )

    result = splitter.split(entity)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=None,
        non_applicable_filter=entity.condition,
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=entity.condition,
    )


STATIC_JOIN_FILTER = Condition(
    target=Join(
        rhs_target=Column(
            fqn="LOCATIONS.NAME",
            dtype_maybe=Dtype.string,
            stype_maybe=Stype.categorical,
        ),
        rhs_key="LOCATIONS.LOCATION_ID",
        lhs_key="USERS.LOCATION_ID",
    ),
    op=RelOp.EQ,
    value="Pallet Town",
    dtype_maybe=Dtype.bool,
    stype_maybe=Stype.categorical,
)

TEMPORAL_COUNT_FILTER = Condition(
    target=Join(
        rhs_target=Aggregation(
            aggr=AggregationType.COUNT,
            target=Column(
                fqn="TRANS.*",
                dtype_maybe=Dtype.int,
                stype_maybe=Stype.categorical,
            ),
            aggr_time_range=DateOffsetRange(
                start=None,  # None -> -inf
                end=0,
            ),
            dtype_maybe=Dtype.int,
            stype_maybe=Stype.numerical,
            group_by="TRANS.USER",
            time_col="TRANS.TIME",
        ),
        rhs_key="TRANS.USER",
        lhs_key="USERS.USERS_ID",
    ),
    op=RelOp.GT,
    value=0,
    dtype_maybe=Dtype.bool,
    stype_maybe=Stype.categorical,
)

SIMPLE_STATIC_FILTER = Condition(
    target=Column(
        fqn="USERS.AGE",
        dtype_maybe=Dtype.int,
        stype_maybe=Stype.numerical,
    ),
    op=RelOp.GT,
    value=10,
    dtype_maybe=Dtype.bool,
    stype_maybe=Stype.categorical,
)


def test_query_no_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = ("PREDICT LIST_DISTINCT(TRANS.STORE "
                      "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
                      "RANK TOP 10 "
                      "FOR EACH USERS.USERS_ID")
    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert result == SplitFilter(
        applicable_filter=None,
        non_applicable_filter=None,
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=0,
        non_applicable_is_valid=True,
        original_filter=None,
    )


def test_query_join_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
        "RANK TOP 10 "
        "FOR EACH USERS.USERS_ID WHERE LOCATIONS.NAME = \"Pallet Town\"")
    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    expected = SplitFilter(
        applicable_filter=STATIC_JOIN_FILTER,
        non_applicable_filter=None,
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=0,
        non_applicable_is_valid=True,
        original_filter=STATIC_JOIN_FILTER,
    )

    # Locations are different in each AST node but the content is the same
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def test_query_temporal_filter(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
        "RANK TOP 10 "
        "FOR EACH USERS.USERS_ID WHERE COUNT(TRANS.*, -inf, 0) > 0 ")
    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    expected = SplitFilter(
        applicable_filter=None,
        non_applicable_filter=TEMPORAL_COUNT_FILTER,
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=TEMPORAL_COUNT_FILTER,
    )

    # Locations are different in each AST node but the content is the same
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def test_query_mixed_and(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
        "RANK TOP 10 "
        "FOR EACH USERS.USERS_ID WHERE COUNT(TRANS.*, -inf, 0) > 0 AND "
        "LOCATIONS.NAME = \"Pallet Town\"")
    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert isinstance(validated_training_query.entity_ast, Filter)
    expected = SplitFilter(
        applicable_filter=STATIC_JOIN_FILTER,
        non_applicable_filter=TEMPORAL_COUNT_FILTER,
        rewritten=False,
        n_applicable_conditions=1,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=validated_training_query.entity_ast.condition,
    )
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def test_query_mixed_or(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
        "RANK TOP 10 "
        "FOR EACH USERS.USERS_ID WHERE COUNT(TRANS.*, -inf, 0) > 0 OR "
        "LOCATIONS.NAME = \"Pallet Town\"")
    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert isinstance(validated_training_query.entity_ast, Filter)
    expected = SplitFilter(
        applicable_filter=None,
        non_applicable_filter=LogicalOperation(
            left=TEMPORAL_COUNT_FILTER,
            bool_op=BoolOp.OR,
            right=STATIC_JOIN_FILTER,
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
        rewritten=False,
        n_applicable_conditions=0,
        n_non_applicable_conditions=2,
        non_applicable_is_valid=True,
        original_filter=validated_training_query.entity_ast.condition,
    )
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def test_duplicated_aggregation(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()

    # An aggregation may be duplicated in the following case if it is
    # R_non_appl or L_non_appl

    # Split the expression into Conjunctive Normal Form (CNF)
    # (L_appl AND L_non_appl) OR (R_appl AND R_non_appl) =
    # (L_appl OR R_appl) AND (
    #      (L_appl OR R_non_appl) AND
    #      (L_non_appl OR R_appl) AND
    #      (L_non_appl OR R_non_appl)
    # )

    training_query = (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
        "RANK TOP 10 "
        "FOR EACH USERS.USERS_ID WHERE (COUNT(TRANS.*, -inf, 0) > 0 AND "
        "USERS.AGE > 10) OR USERS.AGE > 30")

    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert splitter.can_be_split_for_entity_optimization(result)

    training_query = (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
        "RANK TOP 10 "
        "FOR EACH USERS.USERS_ID WHERE (COUNT(TRANS.*, -inf, 0) > 0 AND "
        "USERS.AGE > 10) OR (USERS.AGE > 30 AND COUNT(TRANS.*, -inf, 0) > 1)")

    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert splitter.duplicates_inapplicable_nodes(result)
    assert not splitter.can_be_split_for_entity_optimization(result)


def test_query_mixed_or_negation_rewritten(
        create_mock_graph: Callable[..., GraphDefinition]) -> None:
    graph = create_mock_graph()
    training_query = (
        "PREDICT LIST_DISTINCT(TRANS.STORE "
        "WHERE TRANS.TYPE = 'A' AND STORES.TYPE = 'A', 0, 7) "
        "RANK TOP 10 "
        "FOR EACH USERS.USERS_ID WHERE NOT(COUNT(TRANS.*, -inf, 0) > 0 OR "
        "LOCATIONS.NAME = \"Pallet Town\") AND USERS.AGE > 10")

    validated_training_query = get_validated_query(graph, training_query)

    splitter = StaticFilterSplitter()
    result = splitter.split(validated_training_query.entity_ast)
    assert not splitter.duplicates_inapplicable_nodes(result)
    assert isinstance(validated_training_query.entity_ast, Filter)
    expected = SplitFilter(
        applicable_filter=LogicalOperation(
            left=LogicalOperation(
                left=STATIC_JOIN_FILTER,
                bool_op=BoolOp.NOT,
                right=None,
                dtype_maybe=Dtype.bool,
                stype_maybe=Stype.categorical,
            ),
            bool_op=BoolOp.AND,
            right=SIMPLE_STATIC_FILTER,
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
        non_applicable_filter=LogicalOperation(
            left=TEMPORAL_COUNT_FILTER,
            bool_op=BoolOp.NOT,
            right=None,
            dtype_maybe=Dtype.bool,
            stype_maybe=Stype.categorical,
        ),
        rewritten=True,
        n_applicable_conditions=2,
        n_non_applicable_conditions=1,
        non_applicable_is_valid=True,
        original_filter=validated_training_query.entity_ast.condition,
    )
    _recursively_strip_location(result.applicable_filter)
    _recursively_strip_location(result.non_applicable_filter)
    _recursively_strip_location(expected.applicable_filter)
    _recursively_strip_location(expected.non_applicable_filter)
    assert result == expected


def test_exponential_blowup() -> None:
    # condition of type "(x1 and y1) or ... or (xn and yn)" creates 2^n terms
    # when rewritten into DNF. But only one of them "x1 or .. or xn" is
    # applicable. The system must handle this correctly and without hiccups
    n_terms = 40
    splitter = StaticFilterSplitter()
    expected_applicable_condition = 'entity_table.col = 0'
    condition = LogicalOperation(
        left=Condition(
            target=Aggregation(
                aggr=AggregationType.COUNT,
                target=Column(fqn='linked_fact_table.col'),
                aggr_time_range=DateOffsetRange(
                    start=None,
                    end=0,
                ),
            ), op='=', value=0), bool_op=BoolOp.AND,
        right=Condition(target=Column(fqn='entity_table.col'), op='=',
                        value=0))
    for i in range(1, n_terms):
        expected_applicable_condition = (
            f'({expected_applicable_condition}) OR (entity_table.col = {i})')
        condition = LogicalOperation(
            left=condition, bool_op=BoolOp.OR, right=LogicalOperation(
                left=Condition(
                    target=Aggregation(
                        aggr=AggregationType.COUNT,
                        target=Column(fqn='linked_fact_table.col'),
                        aggr_time_range=DateOffsetRange(
                            start=None,
                            end=0,
                        ),
                    ), op='=', value=i), bool_op=BoolOp.AND,
                right=Condition(target=Column(fqn='entity_table.col'), op='=',
                                value=i)))
    split_filter = splitter._recursively_split_filter(condition, condition)
    assert str(split_filter.applicable_filter) == expected_applicable_condition
    assert split_filter.non_applicable_filter is None
    assert split_filter.rewritten
    assert split_filter.n_applicable_conditions == n_terms
    # order of magnitude 10**12 - would be OOM if in memory
    assert split_filter.n_non_applicable_conditions > 6
    assert not split_filter.non_applicable_is_valid
    assert not splitter.can_be_split_for_entity_optimization(split_filter)


def _recursively_strip_location(ast: ASTNode | None) -> ASTNode | None:
    if ast is None:
        return None
    ast.location = ASTQueryLocationInterval(
        start_row=0,
        start_col=0,
        end_row=0,
        end_col=0,
        data_available=True,
    )
    for child in ast.children:
        _recursively_strip_location(child)
    return ast
