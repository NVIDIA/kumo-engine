// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Diagnostic, linter } from "@codemirror/lint";
import { EditorView, Extension } from "@uiw/react-codemirror";

export function pqlLinter(errorMsg: string): Extension {
    // Parse error message:
    const splitErrorList: string[] | null = errorMsg.split(/\d+\.\s+/).slice(1); // slice(1) to remove the empty string before the first item

    const errorSet = new Set<string>();
    const errorObj: { [key: string]: string } = {};

    if (splitErrorList.length !== 0) {
        // Parse error substring from error message
        const regex = /'(\w*)'/;
        for (let i = 0, n = splitErrorList.length; i < n; ++i) {
            const item = splitErrorList[i];
            const removeLocation = item.split(
                /(row \d+, column \d+: )|(line \d+, col \d+; )|(line \d+ and col \d+; )|(in row \d+, column \d+: )|(in line \d+, col \d+; )|(in line \d+ and col \d+; )/i
            );
            // clean up error messages + create mapping between errors and error messages
            const errorStr = item.match(regex);
            if (errorStr !== null) {
                const cleanError = errorStr[0].replaceAll("'", "");
                errorSet.add(cleanError);
                errorObj[cleanError] = removeLocation[0].concat(
                    "",
                    removeLocation[removeLocation.length - 1]
                );
            }
        }
    }

    const errorLinter = linter((view: EditorView) => {
        const diagnostics: Diagnostic[] = [];

        if (errorSet.size !== 0) {
            errorSet.forEach((errorStr: string) => {
                const docString = view.state.doc.toString();

                // Get indeces of all occurances of an error
                const indices: number[] = [];
                const regex = new RegExp(`\\b${errorStr}\\b`, "g");
                let match;
                while ((match = regex.exec(docString)) !== null) {
                    indices.push(match.index);
                }

                for (let i = 0, n = indices.length; i < n; ++i) {
                    const startIndex = indices[i];
                    diagnostics.push({
                        from: startIndex,
                        to: startIndex + errorStr.length,
                        severity: "error",
                        message: errorObj[errorStr]
                    });
                }
            });
        }
        return diagnostics;
    });
    return errorLinter;
}
