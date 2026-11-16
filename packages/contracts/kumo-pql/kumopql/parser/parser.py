# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import antlr4
from antlr4 import CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener
from antlr4.InputStream import InputStream
from kumoapi.common import ValidationResponse
from kumoapi.pquery import ParsedPredictiveQuery
from kumoapi.pquery.AST import (
    ASTNode,
    Column,
    Condition,
    Filter,
    LogicalOperation,
)
from kumoapi.typing import ProblemType
from kumopql.grammar.PQLGrammarLexer import PQLGrammarLexer
from kumopql.grammar.PQLGrammarParser import PQLGrammarParser
from kumopql.parser.error_translator import Antlr4SyntaxError, ErrorTranslator
from kumopql.parser.visitor import PQLVisitor


def state_from_pql_ctx(ctxType: Any) -> int:
    if ctxType == PQLGrammarParser.ProgContext:
        return PQLGrammarParser.RULE_prog
    if ctxType == PQLGrammarParser.TargetContext:
        return PQLGrammarParser.RULE_target
    if ctxType == PQLGrammarParser.Problem_specContext:
        return PQLGrammarParser.RULE_problem_spec
    if ctxType == PQLGrammarParser.Problem_typeContext:
        return PQLGrammarParser.RULE_problem_type
    if ctxType == PQLGrammarParser.Top_kContext:
        return PQLGrammarParser.RULE_top_k
    if ctxType == PQLGrammarParser.ForecastContext:
        return PQLGrammarParser.RULE_forecast
    if ctxType == PQLGrammarParser.EntityContext:
        return PQLGrammarParser.RULE_entity
    if ctxType == PQLGrammarParser.WhatifContext:
        return PQLGrammarParser.RULE_whatif
    if ctxType == PQLGrammarParser.ConditionContext:
        return PQLGrammarParser.RULE_condition
    if ctxType == PQLGrammarParser.AggregationContext:
        return PQLGrammarParser.RULE_aggregation
    if ctxType == PQLGrammarParser.ColumnContext:
        return PQLGrammarParser.RULE_column
    if ctxType == PQLGrammarParser.Filtered_columnContext:
        return PQLGrammarParser.RULE_filtered_column
    if ctxType == PQLGrammarParser.ConstantContext:
        return PQLGrammarParser.RULE_constant
    if ctxType == PQLGrammarParser.ArrayContext:
        return PQLGrammarParser.RULE_array
    if ctxType == PQLGrammarParser.DatetimeContext:
        return PQLGrammarParser.RULE_datetime
    return -1


class Delegate(ErrorListener):
    r"""Subclass of Antlr4 Error listener to implement custom error handling.
    For the official Antlr4 documentation, see below Java docs.
    https://www.antlr.org/api/Java/org/antlr/v4/runtime/ANTLRErrorListener.html
    Python docs are incomplete but the implementation is analogous.

    This is the most basic implementation that collects all errors instead of
    printing them out (the default).
    """
    def __init__(self) -> None:
        super().__init__()
        self.errors: list[Antlr4SyntaxError] = []

    def syntaxError(
        self,
        recognizer: PQLGrammarParser | PQLGrammarLexer,
        offendingSymbol: antlr4.Token,
        line: int,
        column: int,
        msg: str,
        e: antlr4.RecognitionException,
    ) -> None:
        r"""This method is called by the parser when a syntax error occurs.

        Args:
            recognizer: Antlr recognizer object.
            offendingSymbol: Unmatched token.
            line: Line in which the error occurred.
            column: Column in which the error occurred.
            msg: Parser message to the error.
            e: Exception corresponding to the error.
        """
        # We need to capture the recognizer state and lineage here because it
        # gets wiped after the error is reported
        lineage = []
        if isinstance(recognizer, PQLGrammarParser):
            currCtx = recognizer._ctx
            while (currCtx is not None
                   and not isinstance(currCtx, PQLGrammarParser.ProgContext)):
                lineage.append(state_from_pql_ctx(type(currCtx)))
                currCtx = currCtx.parentCtx
            lineage.append(state_from_pql_ctx(type(currCtx)))

        self.errors.append(
            Antlr4SyntaxError(
                recognizer=recognizer,
                offendingSymbol=offendingSymbol,
                line=line,
                column=column,
                msg=msg,
                e=e,
                state=recognizer.state,
                lineage=lineage,
            ))


class PQLParser:
    r"""Parses the input string according to the PQLGrammar.g4 grammar file."""
    def parse_tree(
            self,
            query: str) -> tuple[antlr4.ParserRuleContext, PQLGrammarParser]:
        r"""Parses the input query and returns the ANTLR4 tree.

        Args:
            query (str): The input query.

        Raises:
            ValueError: If the query is not valid
        """
        tree, parser, response = self._parse_tree(query)
        if response.ok:
            return tree, parser
        else:
            raise ValueError("Encountered the following issues during "
                             f"parsing: {response.message()}")

    def validate(self, query: str) -> ValidationResponse:
        r"""Validates syntactic correctness of the query. Does not check for
        semantic correctness, types, or consistency with the graph.

        Args:
            query: The input query.

        Returns:
            Errors encountered during parsing, if any.
        """
        _, _, response = self._parse_tree(query)
        return response

    def _parse_tree(
        self, query: str
    ) -> tuple[antlr4.ParserRuleContext, PQLGrammarParser, ValidationResponse]:
        input_stream = InputStream(query)
        lexer = PQLGrammarLexer(input_stream)
        error_listener = Delegate()
        lexer.removeErrorListeners()
        lexer.addErrorListener(error_listener)
        token_stream = CommonTokenStream(lexer)
        parser = PQLGrammarParser(token_stream)
        parser.removeErrorListeners()
        parser.addErrorListener(error_listener)
        tree = parser.prog()
        translator = ErrorTranslator()
        response = translator.translate_errors(error_listener.errors, query)
        return tree, parser, response

    def to_lisp_tree(self, query: str) -> str:
        r"""Parses the input query and returns the LISP-formatted syntax tree.
        Primarily used for internal testing and debugging.

        ..code-block::python
            p = PQLParser()
            assert p.to_lisp_tree(
                'predict table.value for each table.user'
            ) == (
                'prog predict (target (column table.value)) problem_spec '
                'for each (entity (column table.user)) '
                'whatif <EOF>)'
            )

        Args:
            query: Input query.

        Raises:
            ValueError: If the query is not valid
        """
        tree, parser = self.parse_tree(query)
        return tree.toStringTree(recog=parser)

    def to_ast(self, query: str) -> dict[str, ASTNode | str | int]:
        r"""Parses the input query.

        Args:
            query: Input query.

        Returns: A dictionary with two or three
            abstract syntax trees and other arguments.
                - entity: AST that defines valid entities
                - target: AST that defines labels corresponding
                    to each entity
                - whatif: AST that defines the whatif
                    condition
                - top_k: The top_k argument for ranking
                - forecast: Number of forecast timeframes, if given.
                - problem_type: Problem type, if given.

        Raises:
            ValueError: If the query is not valid
        """
        tree, _ = self.parse_tree(query)
        visitor = PQLVisitor()
        return tree.accept(visitor)

    def to_parsed_predictive_query(self, query: str) -> ParsedPredictiveQuery:
        r"""Parses the input query and returns an initialised
        :class:`ParsedPredictiveQuery` object.

        Args:
            query: Input query.

        Returns:
            ParsedPredictiveQuery

        Raises:
            ValueError: If the query is not valid
        """
        ast_dict = self.to_ast(query)
        entity_ast = ast_dict['entity']
        assert isinstance(entity_ast, Filter | Column)
        target_ast = ast_dict['target']
        assert isinstance(target_ast, ASTNode)
        whatif_ast: ASTNode | None = None
        if 'whatif' in ast_dict:
            assert isinstance(ast_dict['whatif'], Condition) or isinstance(
                ast_dict['whatif'],
                LogicalOperation) or ast_dict['whatif'] is None
            whatif_ast = ast_dict['whatif']
        top_k: int | None = None
        if 'top_k' in ast_dict:
            assert isinstance(ast_dict['top_k'],
                              int) or ast_dict['top_k'] is None
            top_k = ast_dict['top_k']
        num_forecasts: int = 1
        has_forecast_clause = False
        if 'forecast' in ast_dict:
            assert isinstance(ast_dict['forecast'],
                              int) or ast_dict['forecast'] is None
            if ast_dict['forecast'] is not None:
                num_forecasts = ast_dict['forecast']
                has_forecast_clause = True
        problem_type: ProblemType | str | None = None
        if 'problem_type' in ast_dict:
            assert isinstance(ast_dict['problem_type'],
                              str) or ast_dict['problem_type'] is None
            problem_type = ast_dict['problem_type']
        if has_forecast_clause and problem_type is None:
            problem_type = ProblemType.FORECAST
        for_each: str = str(ast_dict['for_each'])
        return ParsedPredictiveQuery(
            entity_ast,
            target_ast,
            whatif_ast=whatif_ast,
            top_k=top_k,
            num_forecasts=num_forecasts,
            problem_type=problem_type,
            for_each=for_each,
        )
