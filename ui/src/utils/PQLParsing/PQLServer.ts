// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CharStreams, CommonTokenStream } from "antlr4ts";

import { PQLGrammarLexer } from "./grammar/PQLGrammarLexer";

// Token types that the language server supports
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const supportedTokens = [
    "T__0",
    "T__1",
    "T__2",
    "BOOL",
    "FQN_COLUMN",
    "WILDCARD_COLUMN",
    "PREDICT",
    "CLASSIFY",
    "RANK",
    "TOP",
    "FORECAST",
    "TIMEFRAMES",
    "FOR_EACH",
    "WHERE",
    "ASSUMING",
    "AND",
    "OR",
    "NOT",
    "AGGR",
    "REL_OP",
    "NULL",
    "NEG_INFTY",
    "TIME_UNIT",
    "ID",
    "INT",
    "DECIMAL",
    "STR",
    "DATE",
    "TIME",
    "WS",
    "unknown"
] as const;

export type Token = (typeof supportedTokens)[number];

// Tokenize the lexer's tokens
const lexerTokenToTokenType: { [key: number]: Token } = {
    [PQLGrammarLexer.T__0]: "T__0",
    [PQLGrammarLexer.T__1]: "T__1",
    [PQLGrammarLexer.T__2]: "T__2",
    [PQLGrammarLexer.BOOL]: "BOOL",
    [PQLGrammarLexer.FQN_COLUMN]: "FQN_COLUMN",
    [PQLGrammarLexer.WILDCARD_COLUMN]: "WILDCARD_COLUMN",
    [PQLGrammarLexer.PREDICT]: "PREDICT",
    [PQLGrammarLexer.CLASSIFY]: "CLASSIFY",
    [PQLGrammarLexer.RANK]: "RANK",
    [PQLGrammarLexer.TOP]: "TOP",
    [PQLGrammarLexer.FORECAST]: "FORECAST",
    [PQLGrammarLexer.TIMEFRAMES]: "TIMEFRAMES",
    [PQLGrammarLexer.FOR_EACH]: "FOR_EACH",
    [PQLGrammarLexer.WHERE]: "WHERE",
    [PQLGrammarLexer.ASSUMING]: "ASSUMING",
    [PQLGrammarLexer.AND]: "AND",
    [PQLGrammarLexer.OR]: "OR",
    [PQLGrammarLexer.NOT]: "NOT",
    [PQLGrammarLexer.AGGR]: "AGGR",
    [PQLGrammarLexer.REL_OP]: "REL_OP",
    [PQLGrammarLexer.NULL]: "NULL",
    [PQLGrammarLexer.NEG_INFTY]: "NEG_INFTY",
    [PQLGrammarLexer.TIME_UNIT]: "TIME_UNIT",
    [PQLGrammarLexer.ID]: "ID",
    [PQLGrammarLexer.INT]: "INT",
    [PQLGrammarLexer.DECIMAL]: "DECIMAL",
    [PQLGrammarLexer.STR]: "STR",
    [PQLGrammarLexer.DATE]: "DATE",
    [PQLGrammarLexer.TIME]: "TIME",
    [PQLGrammarLexer.WS]: "WS"
};

export class PQLServer {
    public getTokenStream(value: string) {
        const chars = CharStreams.fromString(value);
        const lexer = new PQLGrammarLexer(chars);
        const tokenStream: CommonTokenStream = new CommonTokenStream(lexer);
        tokenStream.fill();
        return tokenStream.getTokens();
    }

    public getTokenTypeForIndex(tokenIndex: number): Token {
        if (tokenIndex in lexerTokenToTokenType) {
            return lexerTokenToTokenType[tokenIndex];
        }

        return "unknown";
    }
}
