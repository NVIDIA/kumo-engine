// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { defineLanguageFacet, languageDataProp } from "@codemirror/language";
import { NodeSet, NodeType } from "@lezer/common";
import { styleTags, tags } from "@lezer/highlight";

import { Token } from "./PQLServer";

/**
 * Lezer - CodeMirror's parsing language.
 * Lezer parse tree is a collection of nodes represented with the NodeType class.
 */
export const tokenToNodeType: { [key in Token | "document"]: NodeType } = {
    // document needed as lezer as entry point; top node
    document: NodeType.define({
        id: 0,
        name: "document",
        top: true,
        props: [
            [
                languageDataProp,
                defineLanguageFacet({
                    commentTokens: {
                        line: "//"
                    }
                })
            ]
        ]
    }),
    T__0: NodeType.define({ id: 1, name: "T__0" }),
    T__1: NodeType.define({ id: 2, name: "T__1" }),
    T__2: NodeType.define({ id: 3, name: "T__2" }),
    BOOL: NodeType.define({ id: 4, name: "BOOL" }),
    FQN_COLUMN: NodeType.define({ id: 5, name: "FQN_COLUMN" }),
    WILDCARD_COLUMN: NodeType.define({ id: 6, name: "WILDCARD_COLUMN" }),
    PREDICT: NodeType.define({ id: 7, name: "PREDICT" }),
    CLASSIFY: NodeType.define({ id: 8, name: "CLASSIFY" }),
    RANK: NodeType.define({ id: 9, name: "RANK" }),
    TOP: NodeType.define({ id: 10, name: "TOP" }),
    FORECAST: NodeType.define({ id: 11, name: "FORECAST" }),
    TIMEFRAMES: NodeType.define({ id: 12, name: "TIMEFRAMES" }),
    FOR_EACH: NodeType.define({ id: 13, name: "FOR_EACH" }),
    WHERE: NodeType.define({ id: 14, name: "WHERE" }),
    ASSUMING: NodeType.define({ id: 15, name: "ASSUMING" }),
    AND: NodeType.define({ id: 16, name: "AND" }),
    OR: NodeType.define({ id: 17, name: "OR" }),
    NOT: NodeType.define({ id: 18, name: "NOT" }),
    AGGR: NodeType.define({ id: 19, name: "AGGR" }),
    REL_OP: NodeType.define({ id: 20, name: "REL_OP" }),
    NULL: NodeType.define({ id: 21, name: "NULL" }),
    NEG_INFTY: NodeType.define({ id: 22, name: "NEG_INFTY" }),
    TIME_UNIT: NodeType.define({ id: 23, name: "TIME_UNIT" }),
    ID: NodeType.define({ id: 24, name: "ID" }),
    INT: NodeType.define({ id: 25, name: "INT" }),
    DECIMAL: NodeType.define({ id: 26, name: "DECIMAL" }),
    STR: NodeType.define({ id: 27, name: "STR" }),
    DATE: NodeType.define({ id: 28, name: "DATE" }),
    TIME: NodeType.define({ id: 29, name: "TIME" }),
    WS: NodeType.define({ id: 30, name: "WS" }),
    unknown: NodeType.define({ id: 31, name: "unknown" }) //unsure about this line
};

// set of all NodeTypes.
// styleTags method creates a key/value mapping between a node type's name and the particular style tag we want to associate with this node type.
export const parserAdapterNodeSet = new NodeSet(
    Object.values(tokenToNodeType)
).extend(
    styleTags({
        ASSUMING: tags.keyword,
        CLASSIFY: tags.keyword,
        RANK: tags.keyword,
        TOP: tags.keyword,
        FORECAST: tags.keyword,
        TIMEFRAMES: tags.keyword,
        PREDICT: tags.keyword,
        FOR_EACH: tags.keyword,
        WHERE: tags.keyword,

        REL_OP: tags.operator,
        NULL: tags.operator,
        BOOL: tags.operator,
        AND: tags.operatorKeyword,
        OR: tags.operatorKeyword,
        NOT: tags.operatorKeyword,

        T__0: tags.string,
        T__1: tags.string,
        T__2: tags.string,
        FQN_COLUMN: tags.string,
        WILDCARD_COLUMN: tags.string,
        STR: tags.string,
        ID: tags.string,

        AGGR: tags.modifier,

        INT: tags.number,
        DECIMAL: tags.number,
        NEG_INFTY: tags.number,
        TIME_UNIT: tags.number,
        DATE: tags.number,
        TIME: tags.number
    })
);
