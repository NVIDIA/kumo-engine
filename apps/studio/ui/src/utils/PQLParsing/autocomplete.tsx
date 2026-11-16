// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Completion, CompletionContext } from "@codemirror/autocomplete";

import { JSTable } from "types/tables";

import { ParserAdapter } from "./PQLParserAdapter";
import {
    keywords,
    multiKeywordMap,
    multiWordKeywordStart
} from "./snippetsAndKeywords";

export function autocompleteWithGraphs(graphTables: JSTable[]) {
    let tablesAndColumns: {
        [key: string]: {
            label: string;
        }[];
    };
    const graphTableNames: {
        label: string;
        type: string;
    }[] = [];
    if (graphTables && graphTables.length !== 0) {
        graphTables.forEach((table) => {
            const columnNames: { label: string }[] = [];
            const tableName = table.name_alias as string;
            table.columns?.forEach((column) => {
                columnNames.push({ label: column.name as string });
            });
            // Dictionary object table names and their column names
            tablesAndColumns = {
                ...tablesAndColumns,
                [tableName]: columnNames
            };
            // Object of only table names
            graphTableNames.push({ label: tableName, type: "constant" });
        });
    }

    function myAutocompleteSource(context: CompletionContext) {
        const { state, pos } = context;
        const word = context.matchBefore(/\w*/);
        const customParser = new ParserAdapter();

        if (!word || word.from === word.to) return null;

        // Access the syntax tree at the cursor position
        const tree = customParser.parse(state.doc.toString());
        const cursor = tree.cursorAt(pos);

        // Analyze the tree to determine the current context
        let completions: Completion[] = [];

        // Limit what completions can be based on context
        // Match only table name
        const table = context.matchBefore(/\w*\.\w*/);

        if (cursor.name === "document") {
            completions = keywords.concat(graphTableNames);
            if (table !== null) {
                const tableName = table.text.split(".")[0];
                if (
                    tablesAndColumns &&
                    Object.keys(tablesAndColumns).indexOf(tableName) > -1
                ) {
                    // Completions should only be the column names for the table
                    completions = tablesAndColumns[tableName];
                } else {
                    // If entry doesn't make sense (not a table name or a keyword),
                    // don't suggest anything after a "."
                    completions = [];
                }
            }
            // Since the grammar file doesn't support parent child relationship
            // between multi-word keys (e.g. FOR EACH), handle that relationship
            // by looking at the preceeding word.
            const splitState = state.doc.toString().split(/\s+/);
            let latestWord = "";
            if (splitState.length > 1) {
                latestWord = splitState[splitState.length - 2].toLowerCase();
            }
            if (multiWordKeywordStart.includes(latestWord)) {
                if (latestWord === "for" || latestWord === "not") {
                    completions = completions.concat(
                        multiKeywordMap.get(latestWord) ?? []
                    );
                } else {
                    completions = multiKeywordMap.get(latestWord) ?? [];
                }
            }
        }

        return {
            from: word.from,
            options: completions
        };
    }

    return myAutocompleteSource;
}
