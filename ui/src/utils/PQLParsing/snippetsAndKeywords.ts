// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Completion } from "@codemirror/autocomplete";

const keyWordCompletion = (name: string) => ({
    label: name.toUpperCase(),
    type: "variable"
});

/**
 * The array of keywords is used to generate the autocompletion options.
 * If by any change the grammar changes and a keyword is added/removed,
 * that change needs to be reflected in the list below.
 */
export const keywords = [
    "!=",
    "<",
    "<=",
    "=",
    ">",
    ">=",
    "and",
    "contains",
    "ends with",
    "in",
    "is not null",
    "is null",
    "like",
    "not like",
    "not contains",
    "not",
    "or",
    "starts with",
    "avg",
    "count",
    "count_distinct",
    "first",
    "last",
    "list_distinct",
    "max",
    "min",
    "sum",
    "classify",
    "rank top K",
    "where",
    "for each",
    "assuming",
    "predict",
    "forecast",
    "timeframes"
].map(keyWordCompletion);

export const multiWordKeywordStart = [
    "ends",
    "is",
    "not",
    "starts",
    "top",
    "rank",
    "for"
];

export const multiKeywordMap = new Map<string, Array<Completion>>()
    .set("ends", ["with"].map(keyWordCompletion))
    .set("is", ["not", "null"].map(keyWordCompletion))
    .set("not", ["null", "like", "contains"].map(keyWordCompletion))
    .set("starts", ["with"].map(keyWordCompletion))
    .set("rank", ["top"].map(keyWordCompletion))
    .set("for", ["each"].map(keyWordCompletion));

// export const snippetsAndKeywords = snippets.concat(keywords);
