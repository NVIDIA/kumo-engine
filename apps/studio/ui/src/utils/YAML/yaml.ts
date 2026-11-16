// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { completeFromList } from "@codemirror/autocomplete";
import {
    HighlightStyle,
    LanguageSupport,
    StreamLanguage,
    StringStream
} from "@codemirror/language";
import { tags as t } from "@lezer/highlight";

import { HighlightTypes } from "types/codeEditor";

import { getKeys } from "utils/Helpers/objectHelpers";

import { yamlDictionary } from "./yamlDictionary";

const keys: Array<string> = getKeys(yamlDictionary);

export const highlights = HighlightStyle.define([
    { tag: t.keyword, color: "#C12C89 !important" },
    { tag: t.comment, color: "gray !important" },
    { tag: t.typeName, color: "#443E82 !important" },
    { tag: t.variableName, color: "#4F888E !important" },
    { tag: t.name, color: "#4F888E !important" },
    { tag: t.operator, color: "#522BB2 !important" },
    { tag: t.number, color: "#4F888E !important" },
    { tag: t.string, color: "#C12C89 !important" }
]);

const YAML = StreamLanguage.define({
    name: "yaml",
    startState: () => {
        return {};
    },
    token: (
        stream: StringStream,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        state: any = {}
    ): string | null => {
        for (const keyword of keys) {
            if (stream.match(keyword, true, true)) {
                state.db = true;
                // let current = stream.current();
                const peek = stream.peek();

                // Syntax highlight only keywords
                if (!(peek === undefined)) {
                    // DOWNSTREAM:
                    // if peek === char in regex  or is a valid key --> valid
                    // regex query to ignore characters: "(", ")", ",", ":", "'", "[", "]"
                    if (
                        !(
                            peek.trim() === "" ||
                            // (/[\(\),'_:\[\].e-]+/.test(peek)) ||
                            /[(),'_:[\].e-]+/.test(peek) ||
                            keys.includes(peek)
                        )
                    ) {
                        return null;
                    }
                    // UPSTREAM:
                    else if (stream.column() !== 0) {
                        // stream.column() is the position where the keyword starts.
                        const currentStream = stream.string;
                        const shouldBeSpace = currentStream.charAt(
                            stream.column() - 1
                        );
                        if (
                            shouldBeSpace !== " " &&
                            !keys.includes(shouldBeSpace) &&
                            !/[(),'_:[\].e-]+/.test(shouldBeSpace)
                        ) {
                            return null;
                        }
                    }
                    return yamlDictionary[keyword].type as HighlightTypes;
                }
                return yamlDictionary[keyword].type as HighlightTypes;
            }
        }
        stream.next();
        return null;
    },

    blankLine: (): void => {},
    copyState: () => {},
    indent: (): number | null => {
        return 0;
    },
    languageData: {
        commentTokens: { line: "--" }
    },
    tokenTable: {
        predict: t.keyword,
        assuming: t.keyword,
        where: t.keyword
    }
});

const autocompleteList: Array<{ label: string; type: string }> = [];

for (let i = 0; i < keys.length; i++) {
    // Example transformation: square each element
    autocompleteList.push({ label: keys[i], type: "keyword" });
}

const yamlCompletion = YAML.data.of({
    autocomplete: completeFromList(autocompleteList)
});

export function yaml() {
    return new LanguageSupport(YAML, [yamlCompletion]);
}
