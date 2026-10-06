# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import antlr4
from antlr4.error.Errors import InputMismatchException, NoViableAltException
from kumoapi.common import ValidationError, ValidationResponse
from kumopql.grammar.PQLGrammarLexer import PQLGrammarLexer
from kumopql.grammar.PQLGrammarParser import PQLGrammarParser


class Antlr4SyntaxError:
    r"""Dataclass with all the data returned by an AntlrSyntaxError.

    Args:
        recognizer: Antlr recognizer object.
        offendingSymbol: Unmatched token.
        line: Line in which the error occurred.
        column: Column in which the error occurred.
        msg: Parser message to the error.
        e: Exception corresponding to the error.
        state: Corresponds to the state the parser was in before it
            threw an error. This helps us figure out what part of the query
            has the error
        lineage: Corresponds to the list of states that the parser
            had passed through before throwing this error. The int values
            correspond to the RULE enums in the parser code
    """
    def __init__(
        self,
        recognizer: PQLGrammarParser | PQLGrammarLexer,
        offendingSymbol: antlr4.Token,
        line: int,
        column: int,
        msg: str,
        e: antlr4.RecognitionException,
        state: int,
        lineage: list[int],
    ):
        self.recognizer = recognizer
        self.offendingSymbol = offendingSymbol
        self.line = line
        self.column = column
        self.msg = msg
        self.e = e
        self.state = state
        self.lineage = lineage


logger = logging.getLogger(__name__)


class ErrorTranslator:
    r"""Error translator class. Given a list of errors produced by Antlr4
    parser, this class returns user-friendly error descriptions.
    """
    def translate_errors(
        self,
        errors: list[Antlr4SyntaxError],
        query: str,
    ) -> ValidationResponse:
        r"""Given a list of ANTLR4 errors, returns all possible user-readable
        responses for each of the errors.
        """
        response = ValidationResponse()
        for error in errors:
            error_message = self._translate_error(error)
            log_error = self._log_error_description(error)

            response.errors.append(
                ValidationError("Invalid Syntax", message=error_message))
            logger.warning(f'Parser error {log_error} was translated to: '
                           f'{error_message}. Input query: `{query}`.')
        return response

    def _summarize_error(self, err: Antlr4SyntaxError) -> str | None:
        # Get the high-level state of the recognizer so we can know where
        # the error is coming from
        if not isinstance(err.recognizer, PQLGrammarParser):
            return None
        input_stream = err.recognizer.getInputStream()
        tokens: list[antlr4.Token] = []
        if input_stream is None:
            return None

        tokens = input_stream.getTokens(0, 2**30)
        token_strings = [token.text for token in tokens]
        if not token_strings:
            return 'Query does not start with the PREDICT keyword. '
        lineage = err.lineage
        if (lineage in ([PQLGrammarParser.RULE_prog], [
                PQLGrammarParser.RULE_target, PQLGrammarParser.RULE_prog
        ]) and 'WHERE' in token_strings and 'FOR EACH' in token_strings):
            where_idx = token_strings.index('WHERE')
            for_each_idx = token_strings.index('FOR EACH')
            if (where_idx < for_each_idx and err.offendingSymbol.tokenIndex
                    < tokens[for_each_idx].tokenIndex):
                if '(' in token_strings[:where_idx]:
                    return ('The target filter (PREDICT ... WHERE) clause in '
                            'this query is empty or invalid.')
                return ('The target (PREDICT) clause in this query is '
                        'invalid.')
        if lineage == [PQLGrammarParser.RULE_prog]:
            expected = err.recognizer.atn.nextTokens(
                err.recognizer.atn.states[err.state])
            if PQLGrammarParser.PREDICT in expected:
                return ('Query does not start with the PREDICT keyword. ')
            elif PQLGrammarParser.FOR_EACH in expected:
                if 'FOR EACH' in token_strings:
                    for_each_idx = token_strings.index('FOR EACH')
                    return ('The target (PREDICT) clause in this query is '
                            'invalid. We found the following extra characters '
                            f'before the FOR EACH keyword: '
                            f'{token_strings[for_each_idx-1]}')
                else:
                    return ('Query is missing the FOR EACH keyword.')
        if lineage == [
                PQLGrammarParser.RULE_target, PQLGrammarParser.RULE_prog
        ]:
            return ('The target (PREDICT) clause in this query is '
                    'empty or invalid.')
        if lineage == [
                PQLGrammarParser.RULE_entity, PQLGrammarParser.RULE_prog
        ]:
            return ('The entity (FOR EACH) clause in this query is '
                    'empty or invalid.')
        if ('FOR EACH' not in token_strings
                and err.offendingSymbol.text == '<EOF>'):
            return 'Query is missing the FOR EACH keyword.'
        if lineage == []:
            return ('We found some syntactical errors in your query. We\'re '
                    'unable to parse some characters. See the detailed error '
                    'below.')
        if lineage == [
                PQLGrammarParser.RULE_condition,
                PQLGrammarParser.RULE_filtered_column,
                PQLGrammarParser.RULE_entity,
                PQLGrammarParser.RULE_prog,
        ]:
            return (
                'The entity filter (FOR EACH ... WHERE) clause in this query '
                'is empty or invalid.')
        return None

    def _translate_error(self, err: Antlr4SyntaxError) -> str:
        r"""Given an ANTLR error, returns a user-readable response as a
        single string.
        """
        error_parts = []

        if (err.lineage and err.lineage[0] == PQLGrammarParser.RULE_array
                and err.offendingSymbol.text == ')'):
            return (f"Line {err.line}, col {err.column}; "
                    "Array with single element is not supported. "
                    "Suggestion: If you are using 'IS IN (const)' please use "
                    "'= const' instead.")

        # Add error summary if available
        error_summary = self._summarize_error(err)
        if error_summary is not None:
            error_parts.append(error_summary)

        # Add specific error messages
        if isinstance(err.e, NoViableAltException):
            error_parts.append('Specific error: ')
            error_parts.extend(self._handleNoViableAltException(err))
        elif isinstance(err.e, InputMismatchException):
            error_parts.append('Specific error: ')
            error_parts.extend(self._handleInputMismatchException(err))
        elif 'extraneous input' in err.msg and err.e is None:
            error_parts.append('Specific error: ')
            error_parts.extend(self._handleExtraneousInput(err))
        elif 'Array with single element' in err.msg:
            error_parts.append(f"Line {err.line}, col {err.column}; {err.msg} "
                               f"Suggestion: If you are using 'IS IN (const)' "
                               f"please use '= const'  instead.")

        # If no specific errors were found, use default error
        if not error_parts:
            error_parts.append(self._default_error(err))

        # Join all error parts with newlines
        return "\n".join(error_parts)

    def _handleNoViableAltException(self, err: Antlr4SyntaxError) -> list[str]:
        # The parser was not able to apply any of the rules.
        # Seems the trickiest exception to handle since it happens at the root
        # so there's no context or expected tokens available.
        assert isinstance(err.e, NoViableAltException)
        # All the rules that the parser tried to apply
        hints = []
        for dead_end_conf in err.e.deadEndConfigs:
            rule_name = err.recognizer.ruleNames[dead_end_conf.alt]
            if rule_name == 'column':
                hints.append(
                    'If you were trying to add a column, syntax for a column '
                    'is "ID.ID", where ID stands for a valid table/column '
                    'name.')
            if rule_name == 'condition':
                hints.append(
                    'If you were trying to add a condition on aggregation, '
                    'syntax for a condition is "COLUMN REL_OP CONST", where '
                    'COLUMN should be a valid column name or an aggregation, '
                    'REL_OP should be a valid operation, and CONST is a '
                    'constant. Multiple conditions can be combined with '
                    '"AND", "OR", and "NOT" logical operators. You can refer '
                    'to our documentation at docs.kumo.ai for more details '
                    'on allowed operations.')
            if rule_name == 'aggregation':
                hints.append(
                    'If you were trying to add an aggregation, syntax for '
                    'aggregation is "AGGR(COLUMN, <start>, <end>, <time_unit'
                    '>)", where AGGR stands for a valid aggregation name, '
                    'COLUMN stands for a column name, <start> and <end> are '
                    'integers that stand for time offset, and <time_unit> is '
                    'a valid time unit. You can refer '
                    'to our documentation at docs.kumo.ai for more details '
                    'on allowed operations.')

        maybe_hints_text = ""
        # avoid repeating hints
        hints = list(set(hints))
        if len(hints) > 0:
            maybe_hints_text = '\nHINTS:\n' + '\n'.join(hints)
        return [
            f'Could not process the text: "{err.offendingSymbol.text}"; '
            f'in line {err.line} and col {err.column}; Encountered the '
            f'following error: "{err.msg}"; '
            f'{maybe_hints_text}'
        ]

    def _handleInputMismatchException(self,
                                      err: Antlr4SyntaxError) -> list[str]:
        # The next token does not match any of the next expected token
        assert isinstance(err.e, InputMismatchException)
        offending_token_text = err.recognizer.getTokenErrorDisplay(
            err.offendingSymbol)
        # The parser is already in recovery mode and is trying to parse input
        # without the token - hence we can't rely on the parser to find the
        # expected tokens and we need to parse the default error
        if err.msg[-1] == '}':
            # multiple tokens
            expected_tokens = err.msg[err.msg.rfind('expecting {') + 11:-1]
        else:
            # only one
            expected_tokens = err.msg[err.msg.rfind('expecting ') + 10:]
        if expected_tokens == '<EOF>':
            return [
                f'Line {err.line}, col {err.column}; '
                f'Input Mismatch, got {offending_token_text}, but expected '
                f'the end of input.'
            ]

        # The default error displays rule name rather than the actual text
        # it stands for. We replace the internal names with the more helpful
        # text that they stand for.
        expected_tokens = expected_tokens.replace("TARGET_FILTER",
                                                  "TARGET FILTER IS")
        expected_tokens = expected_tokens.replace("TEMPORAL_ENTITY_FILTER",
                                                  "TEMPORAL ENTITY FILTER IS")
        expected_tokens = expected_tokens.replace("ENTITY_FILTER",
                                                  "ENTITY FILTER IS")
        return [
            f'Line {err.line}, col {err.column}; '
            f'Input Mismatch, got {offending_token_text}, but expected one '
            f'of {expected_tokens}.'
        ]

    def _handleExtraneousInput(self, err: Antlr4SyntaxError) -> list[str]:
        # The next token does not match any of the next expected token
        offending_token_text = err.recognizer.getTokenErrorDisplay(
            err.offendingSymbol)
        # The parser is already in recovery mode and is trying to parse input
        # without the token - hence we can't rely on the parser to find the
        # expected tokens and we need to parse the default error
        if err.msg[-1] == '}':
            # multiple tokens
            expected_tokens = err.msg[err.msg.rfind('expecting {') + 11:-1]
        else:
            # only one
            expected_tokens = err.msg[err.msg.rfind('expecting ') + 10:]
        if expected_tokens == '<EOF>':
            return [
                f'Line {err.line}, col {err.column}; '
                f'Extraneous input, got {offending_token_text}, but expected '
                f'the end of input.'
            ]
        # We do not report $TODAY as an option for now since we do not
        # encourage its use
        expected_tokens = expected_tokens.replace("'$TODAY', ", "")
        return [
            f'Line {err.line}, col {err.column}; '
            f'Extraneous input, got {offending_token_text}, but expected one '
            f'of {expected_tokens}.'
        ]

    def _default_error(self, err: Antlr4SyntaxError) -> str:
        r"""Given an ANTLR error, returns a default message."""
        return (f" Line {err.line}, col {err.column};  "
                f"Syntax Error: {err.msg}")

    def _log_error_description(self, err: Antlr4SyntaxError) -> str:
        r"""Given an ANTLR error, returns a message for logging with all
        the details, relevant to debugging.
        """
        if isinstance(err.e, NoViableAltException):
            e_str = (f'{str(err.e)}, startToken {err.e.startToken}, '
                     f'offendingSymbol {err.offendingSymbol}, '
                     f'deadEndConfigs {err.e.deadEndConfigs}')
        else:
            e_str = str(err.e)

        maybe_context = ""
        return (
            f'{type(err.e)} {str(err.offendingSymbol)}, line {err.line}, col '
            f'{err.column}{maybe_context} {err.msg}, {e_str}')
