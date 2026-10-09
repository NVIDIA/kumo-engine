// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import CodeEditor from "components/CodeEditor/CodeEditor";

import { JSTable } from "types/tables";

import { pqObj } from "utils/PQLParsing/pqDictionary";

import { parseHintsFromMintlify } from "./parseHints";

interface PQCodeEditorProps {
    language: "PQLANG_V2" | "PYTHON";
    value: string;
    onChange: (newval: string) => void;
    editable?: boolean;
    isInvalid?: boolean;
    errorMsg?: string;
    graphTables?: JSTable[];
    hideBorder?: boolean;
    height?: string;
    lineNumbers?: boolean;
    highlightActiveLineGutter?: boolean;
    highlightActiveLine?: boolean;
    showCopyButton?: boolean;
    enableHighlight?: boolean;
    highlightPositions?: { start: number; end: number }[];
    fontSize?: number;
    backgroundColor?: string;
}

const PQCodeEditor: React.FC<PQCodeEditorProps> = ({
    value,
    language,
    onChange,
    editable,
    isInvalid,
    errorMsg,
    graphTables,
    hideBorder = false,
    height = "300px",
    lineNumbers = true,
    highlightActiveLineGutter = true,
    highlightActiveLine = true,
    showCopyButton = false,
    enableHighlight = false,
    highlightPositions = [],
    fontSize = 16,
    backgroundColor
}) => {
    const parsedDict = parseHintsFromMintlify(language, pqObj);

    return (
        <div>
            <CodeEditor
                value={value}
                placeholder={"Start typing your predictive query here..."}
                language={language}
                hoverHints={parsedDict}
                basicSetup={{
                    editable: true,
                    lineNumbers: lineNumbers,
                    highlightActiveLineGutter: highlightActiveLineGutter,
                    highlightSpecialChars: true,
                    history: true,
                    foldGutter: false,
                    drawSelection: true,
                    indentOnInput: true,
                    syntaxHighlighting: true,
                    bracketMatching: true,
                    closeBrackets: true,
                    autocompletion: true,
                    highlightActiveLine: highlightActiveLine,
                    highlightSelectionMatches: true,
                    closeBracketsKeymap: true,
                    dropCursor: true,
                    allowMultipleSelections: true,
                    rectangularSelection: true,
                    crosshairCursor: false,
                    defaultKeymap: true,
                    searchKeymap: true,
                    historyKeymap: true,
                    foldKeymap: true,
                    completionKeymap: true
                }}
                onChange={(value: string) => {
                    onChange(value);
                }}
                editable={editable}
                isInvalid={isInvalid}
                errorMsg={errorMsg}
                graphTables={graphTables}
                hideBorder={hideBorder}
                height={height}
                showCopyButton={showCopyButton}
                enableHighlight={enableHighlight}
                highlightPositions={highlightPositions}
                fontSize={fontSize}
                backgroundColor={backgroundColor}
            />
        </div>
    );
};

export default WithErrorBoundary(PQCodeEditor);
