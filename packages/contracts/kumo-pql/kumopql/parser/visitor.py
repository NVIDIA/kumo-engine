# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import antlr4
from kumoapi.pquery.AST import (
    Aggregation,
    ASTNode,
    Column,
    Condition,
    Constant,
    DateOffsetRange,
    Filter,
    LogicalOperation,
)
from kumoapi.pquery.AST.ast_node import ArrayDtype
from kumoapi.pquery.AST.condition import SQL_IS, SQL_IS_NOT
from kumoapi.pquery.AST.location_interval import ASTQueryLocationInterval
from kumoapi.typing import (
    AggregationType,
    BoolOp,
    Dtype,
    MemberOp,
    RelOp,
    StrOp,
    TimeUnit,
)
from kumopql.grammar.PQLGrammarParser import PQLGrammarParser
from kumopql.grammar.PQLGrammarVisitor import PQLGrammarVisitor

logger = logging.getLogger(__name__)

PREDICT_INDEX = 0
TARGET_CONDITION_INDEX = 1
PROBLEM_SPEC_INDEX = 2
FOR_EACH_INDEX = 3
ENTITY_INDEX = 4
ASSUMING_INDEX = 5
WHERE = 'WHERE'
ASSUMING = 'ASSUMING'
TARGET = 'target'
ENTITY = 'entity'
WHAT_IF = 'whatif'
FOR_EACH = 'for_each'
PROBLEM_TYPE = 'problem_type'
TOP_K = 'top_k'
FORECAST = 'forecast'


class PQLVisitor(PQLGrammarVisitor):
    r"""A Visitor class that traverses the syntax tree and produces a
    dict of :class:`ASTNode` classes, each entry corresponding to one
    of the three parts of the query: entity, target, and whatif clause
    and the following arguments: top_k, forecast, and problem_type.

    The child index from this tree is used to parse and retrieve information
    about specific nodes.
    """
    @staticmethod
    def get_interval(
            ctx: antlr4.ParserRuleContext) -> ASTQueryLocationInterval:
        r"""Interval for error reporting. Extracts what interval of the input
        text corresponds to this syntax tree node.
        """
        if ctx.start is None:
            start_row = 0
            start_col = 0
        else:
            start_row = ctx.start.line
            start_col = ctx.start.column
        if ctx.stop is None:
            end_row = start_row
            end_col = start_col
        else:
            end_row = ctx.stop.line
            end_col = ctx.stop.column
        return ASTQueryLocationInterval(start_row, start_col, end_row, end_col)

    # Visit a parse tree produced by PQLGrammarParser#prog.
    def visitProg(
            self, ctx: PQLGrammarParser.ProgContext
    ) -> dict[str, ASTNode | str | int]:
        r"""Called by the :class:`PQLGrammarParser` to parse the root node.

        Returns:
            A `dict` that can be used to initialise :class:`PredictiveQuery`.
        """
        # PREDICT target problem_spec
        # FOR EACH entity whatif EOF
        predict = ctx.getChild(PREDICT_INDEX)
        target = ctx.getChild(TARGET_CONDITION_INDEX)
        problem_spec = ctx.getChild(PROBLEM_SPEC_INDEX)
        for_each = ctx.getChild(FOR_EACH_INDEX)
        entity = ctx.getChild(ENTITY_INDEX)
        whatif = ctx.getChild(ASSUMING_INDEX)
        assert predict is not None
        assert str(predict).upper() == 'PREDICT'
        assert str(for_each).upper() == 'FOR EACH'
        complete_dict = {}

        # Process entity
        assert entity is not None
        complete_dict[ENTITY] = entity.accept(self)

        # Process target
        assert target is not None
        complete_dict[TARGET] = target.accept(self)

        # Process problem_spec
        complete_dict[PROBLEM_TYPE] = None
        complete_dict[TOP_K] = None
        complete_dict[FORECAST] = None
        assert problem_spec is not None
        spec = problem_spec.accept(self)
        if spec is not None:
            complete_dict.update(spec)

        # Process whatif
        assert whatif is not None
        complete_dict[WHAT_IF] = whatif.accept(self)

        # Process for_each
        assert for_each is not None
        complete_dict[FOR_EACH] = str(for_each).upper()

        return complete_dict

    # Visit a parse tree produced by PQLGrammarParser#problem_type.
    def visitProblem_type(
            self, ctx: PQLGrammarParser.Problem_typeContext) -> str | None:
        r"""Called by the :class:`PQLGrammarParser` to parse the problem
        type node.

        Returns:
            A `str` or :obj:`None`. Possible valid return values are:
                - RANK
                - CLASSIFY (TODO, subject to change)
                - None (the generator later chooses the default value).
        """
        # (RANK|CLASSIFY)?
        if ctx.getChildCount() == 0:
            return None
        return str(ctx.getChild(0)).upper()

    # Visit a parse tree produced by PQLGrammarParser#problem_spec.
    def visitProblem_spec(
        self, ctx: PQLGrammarParser.Problem_specContext
    ) -> dict[str, int | str | None] | None:
        r"""Called by the :class:`PQLGrammarParser` to parse the problem
        specification node (forecast or problem type/top_k).

        Returns:
            A `dict` containing any of: problem_type, top_k, forecast.
        """
        # forecast | problem_type top_k | top_k | problem_type | (empty)
        if ctx.getChildCount() == 0:
            return {}
        forecast = ctx.forecast()
        if forecast is not None:
            return {FORECAST: forecast.accept(self)}
        result: dict[str, int | str | None] = {}
        problem_type = ctx.problem_type()
        if problem_type is not None:
            result[PROBLEM_TYPE] = problem_type.accept(self)
        top_k = ctx.top_k()
        if top_k is not None:
            result[TOP_K] = top_k.accept(self)
        return result

    # Visit a parse tree produced by PQLGrammarParser#top_k.
    def visitTop_k(self, ctx: PQLGrammarParser.Top_kContext) -> int | None:
        r"""Called by the :class:`PQLGrammarParser` to parse the top_k
        node.

        Returns:
            An `int` (value of K) or :class:`None` if not given.
        """
        # (TOP INT)?
        if ctx.getChildCount() == 0:
            return None
        assert str(ctx.getChild(0)).upper() == 'TOP'
        return int(str(ctx.getChild(1)))

    # Visit a parse tree produced by PQLGrammarParser#forecast.
    def visitForecast(self,
                      ctx: PQLGrammarParser.ForecastContext) -> int | None:
        r"""Called by the :class:`PQLGrammarParser` to parse the forecast
        node.

        Returns:
            An `int` (number of forecast timeframes) or :class:`None` if
            not given.
        """
        # (FORECAST INT TIMEFRAMES)?
        if ctx.getChildCount() == 0:
            return None
        assert str(ctx.getChild(0)).upper() == 'FORECAST'
        assert str(ctx.getChild(2)).upper() == 'TIMEFRAMES'
        return int(str(ctx.getChild(1)))

    # Visit a parse tree produced by PQLGrammarParser#target.
    def visitTarget(self, ctx: PQLGrammarParser.TargetContext) -> ASTNode:
        r"""Called by the :class:`PQLGrammarParser` to parse the target
        node.

        Returns:
            An :class:`ASTNode` corresponding to the target.
        """
        # condition | aggregation | column
        assert ctx.getChildCount() == 1
        target = ctx.getChild(0)
        assert target is not None
        return target.accept(self)

    # Visit a parse tree produced by PQLGrammarParser#entity.
    def visitEntity(
        self, ctx: PQLGrammarParser.EntityContext
    ) -> Column | Filter:
        r"""Called by the :class:`PQLGrammarParser` to parse the entity
        node.

        Returns:
            A :class:`Column` or :class:`Filter` corresponding to the entity.
        """
        # column | filtered_column
        assert ctx.getChildCount() == 1
        target = ctx.getChild(0)
        assert target is not None
        return target.accept(self)

    # Visit a parse tree produced by PQLGrammarParser#whatif.
    def visitWhatif(self,
                    ctx: PQLGrammarParser.WhatifContext) -> ASTNode | None:
        r"""Called by the :class:`PQLGrammarParser` to parse the whatif
        node.

        Returns:
            An :class:`ASTNode` (whatif condition) or :class:`None` if
                not given.
        """
        # (ASSUMING condition)?
        if ctx.getChildCount() == 0:
            return None
        assert str(ctx.getChild(0)).upper() == 'ASSUMING'
        whatif = ctx.getChild(1)
        assert whatif is not None
        return whatif.accept(self)

    # Visit a parse tree produced by PQLGrammarParser#column.
    def visitColumn(self, ctx: PQLGrammarParser.ColumnContext) -> Column:
        r"""Called by the :class:`PQLGrammarParser` to parse the column
        node.

        Returns:
            A :class:`Column`.
        """
        # "ID.ID" or "ID.*"
        location = self.get_interval(ctx)
        return Column(fqn=str(ctx.getChild(0)), location=location)

    # Visit a parse tree produced by PQLGrammarParser#filtered_column.
    def visitFiltered_column(
            self, ctx: PQLGrammarParser.Filtered_columnContext) -> Filter:
        r"""Called by the :class:`PQLGrammarParser` to parse the filtered
        column node.

        Returns:
            A :class:`Filter`.
        """
        # column WHERE condition
        assert ctx.getChildCount() == 3
        child_column = ctx.getChild(0)
        child_where = ctx.getChild(1)
        child_condition = ctx.getChild(2)
        assert child_column is not None
        assert str(child_where).upper() == 'WHERE'
        assert child_condition is not None
        column = child_column.accept(self)
        assert isinstance(column, Column)
        condition = child_condition.accept(self)
        assert isinstance(condition, Condition | LogicalOperation)
        location = self.get_interval(ctx)
        return Filter(target=column, condition=condition, location=location)

    # Visit a parse tree produced by PQLGrammarParser#condition.
    def visitCondition(
        self,
        ctx: PQLGrammarParser.ConditionContext,
    ) -> Condition | LogicalOperation:
        r"""Called by the :class:`PQLGrammarParser` to parse the condition
        node.

        Returns:
            :class:`Condition` or :class:`LogicalOperation`, corresponding to
            the condition.
        """
        # "aggregation REL_OP const" or "column REL_OP const" or
        # "NOT condition" or "condition AND condition"
        # or "condition OR condition" or "(condition)"

        child_l = ctx.getChild(0)
        child_mid = ctx.getChild(1)
        assert child_l is not None
        assert child_mid is not None
        location = self.get_interval(ctx)
        # NOT
        if ctx.getChildCount() == 2:
            assert str(child_l).upper() == BoolOp.NOT.value
            nested_filter = child_mid.accept(self)
            return LogicalOperation(left=nested_filter, bool_op=BoolOp.NOT,
                                    location=location)

        child_r = ctx.getChild(2)
        assert child_r is not None

        # AND/OR
        if isinstance(child_mid, antlr4.tree.Tree.TerminalNodeImpl) and str(
                child_mid).upper() in {BoolOp.OR.value, BoolOp.AND.value}:
            nested_l_filter = child_l.accept(self)
            nested_r_filter = child_r.accept(self)
            bool_op = BoolOp(str(child_mid).upper())
            return LogicalOperation(left=nested_l_filter, bool_op=bool_op,
                                    right=nested_r_filter, location=location)

        # (condition)
        if (isinstance(child_l, antlr4.tree.Tree.TerminalNodeImpl)
                and str(child_l) == '('
                and isinstance(child_r, antlr4.tree.Tree.TerminalNodeImpl)
                and str(child_r) == ')'):
            return child_mid.accept(self)

        # (column|aggregation) REL_OP const
        assert isinstance(child_mid, antlr4.tree.Tree.TerminalNodeImpl)
        rel_op = str(child_mid).upper()
        value = child_r.accept(self)
        target = child_l.accept(self)
        arg_dict = {'target': target, 'value': value, 'location': location}
        if rel_op in set(op.value for op in RelOp):
            arg_dict['op'] = RelOp(rel_op)
            new_filt = Condition(**arg_dict)
        elif rel_op in set(op.value for op in MemberOp):
            arg_dict['op'] = MemberOp(rel_op)
            new_filt = Condition(**arg_dict)
        elif rel_op.upper() in {SQL_IS, SQL_IS_NOT}:
            arg_dict['op'] = RelOp.EQ if rel_op.upper(
            ) == SQL_IS else RelOp.NEQ
            new_filt = Condition(**arg_dict)
        elif rel_op.upper() in set(op.value
                                   for op in StrOp) | {'LIKE', 'NOT LIKE'}:
            arg_dict['op'] = rel_op
            new_filt = Condition(**arg_dict)
        else:
            raise NotImplementedError(f'Unknown operation {rel_op}.')
        return new_filt

    # Visit a parse tree produced by PQLGrammarParser#aggregation.
    def visitAggregation(
            self, ctx: PQLGrammarParser.AggregationContext) -> Aggregation:
        r"""Called by the :class:`PQLGrammarParser` to parse the
        aggregation node.

        Returns:
            :class:`Aggregation` - the :class:`ASTNode` corresponding to the
                aggregation.
        """
        # "AGGR ( column )" or "AGGR ( column , int , int ) " or
        # "AGGR (column , int , int , time_unit )"
        # The first of the two ints that determine the time range can be "-INF"

        n_children = ctx.getChildCount()
        aggr = AggregationType(str(ctx.getChild(0)).upper())
        assert str(ctx.getChild(1)) == '('
        assert str(ctx.getChild(n_children - 1)) == ')'
        column_child = ctx.getChild(2)
        assert column_child is not None
        arg_dict = {}
        target = column_child.accept(self)
        arg_dict[TARGET] = target
        arg_dict['aggr'] = aggr
        arg_dict['location'] = self.get_interval(ctx)
        if n_children == 4:
            return Aggregation(**arg_dict)
        assert str(ctx.getChild(3)) == ','
        assert str(ctx.getChild(5)) == ','
        start_offset_child = ctx.getChild(4)  # -NEG or INT
        assert start_offset_child is not None
        offset_start = None
        if start_offset_child.symbol.type == PQLGrammarParser.INT:
            offset_start = int(str(start_offset_child))
        end_offset_child = ctx.getChild(6)  # INT
        offset_end = int(str(end_offset_child))
        time_unit = TimeUnit.DAYS
        if n_children == 10:  # time_unit is given
            assert str(ctx.getChild(7)) == ','
            time_unit = TimeUnit(str(ctx.getChild(8)).lower())
        arg_dict['aggr_time_range'] = DateOffsetRange(offset_start, offset_end,
                                                      unit=time_unit)
        return Aggregation(**arg_dict)

    # Visit a parse tree produced by PQLGrammarParser#const.
    def visitConstant(self, ctx: PQLGrammarParser.ConstantContext) -> Constant:
        r"""Called by the :class:`PQLGrammarParser` to parse the constant
        node.

        Returns:
            A constant cast into the correct type.
        """
        # BOOL or INT or STR or DECIMAL or NULL or array or datetime
        assert ctx.getChildCount() == 1
        location = self.get_interval(ctx)
        child = ctx.getChild(0)
        assert child is not None

        # array or datetime
        if isinstance(child, antlr4.RuleContext):
            return child.accept(self)

        # BOOL
        if child.symbol.type == PQLGrammarParser.BOOL:
            return Constant(value=str(child), dtype_maybe=Dtype.bool,
                            location=location)

        # INT
        if child.symbol.type == PQLGrammarParser.INT:
            return Constant(value=str(child), dtype_maybe=Dtype.int,
                            location=location)

        # STR
        if child.symbol.type == PQLGrammarParser.STR:
            # Handling invalid escape characters in grammar results in
            # confusing errors, so we handle them here.
            valid_escape_chr = '\\\"\'/bfnrtu'
            escaped = False
            for c in str(child)[1:-1]:
                if escaped and c not in valid_escape_chr:
                    raise ValueError(f'\\{c} is not a valid escape pattern.')
                if not escaped and c == '\\':
                    escaped = True
                else:
                    escaped = False
            # transform escaped characters and drop quotes
            return Constant(value=str(child), dtype_maybe=Dtype.string,
                            location=location)

        # DECIMAL
        if child.symbol.type == PQLGrammarParser.DECIMAL:
            return Constant(value=str(child), dtype_maybe=Dtype.float,
                            location=location)

        # NULL
        if child.symbol.type == PQLGrammarParser.NULL:
            return Constant(value=str(child), dtype_maybe=None,
                            location=location)
        raise ValueError(f'Cannot parse the symbol: "{str(child)}", please '
                         'check Predictive Query documentation for correct '
                         'formatting.')

    # Visit a parse tree produced by PQLGrammarParser#datetime.
    def visitDatetime(self, ctx: PQLGrammarParser.DatetimeContext) -> Constant:
        r"""Called by the :class:`PQLGrammarParser` to parse the datetime
        node.

        Returns:
            :class:`pd.Timestamp` - the corresponding datetime object.
        """
        # DATE TIME?
        location = self.get_interval(ctx)
        if ctx.getChildCount() == 1:
            return Constant(value=str(ctx.getChild(0)), dtype_maybe=Dtype.time,
                            location=location)
        assert ctx.getChildCount() == 2
        return Constant(
            value=str(ctx.getChild(0)) + ' ' + str(ctx.getChild(1)),
            dtype_maybe=Dtype.time, location=location)

    # Visit a parse tree produced by PQLGrammarParser#array.
    def visitArray(self, ctx: PQLGrammarParser.ArrayContext) -> Constant:
        r"""Called by the :class:`PQLGrammarParser` to parse the array node.

        Returns:
            List[Any] - a list of constants.
        """
        # ( const, const, ... const)
        child_trees = []
        location = self.get_interval(ctx)
        array_len = ctx.getChildCount()
        for i in range(array_len):
            child = ctx.getChild(i)
            if i % 2 == 0:
                assert isinstance(child, antlr4.tree.Tree.TerminalNodeImpl)
                if i == 0:
                    assert str(child) == '('
                elif i + 1 == array_len:
                    assert str(child) == ')'
                else:
                    assert str(child) == ','
            else:
                assert child is not None
                child_trees.append(child.accept(self))
        # We validate this later
        dtype = ArrayDtype(child_trees[0].dtype_maybe)
        return Constant(value=child_trees, dtype_maybe=dtype,
                        location=location)
