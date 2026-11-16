// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { SQLNamespace } from "@codemirror/lang-sql";

import WithErrorBoundary from "components/@hoc/WithErrorBoundary";
import CodeEditor from "components/CodeEditor/CodeEditor";

interface SQLCodeEditorProps {
    value: string;
    onChange: (newval: string) => void;
    // Connector catalog (tables/columns) for autocompletion. Warehouse-neutral:
    // completion is StandardSQL keywords + these names, nothing dialect-specific.
    catalog?: SQLNamespace;
    editable?: boolean;
    isInvalid?: boolean;
    errorMsg?: string;
    hideBorder?: boolean;
    height?: string;
    lineNumbers?: boolean;
    showCopyButton?: boolean;
    placeholder?: string;
    fontSize?: number;
    backgroundColor?: string;
}

/**
 * SQL authoring editor for Custom SQL Tables, mirroring `PQCodeEditor` but for
 * `language="SQL"`. Autocomplete is intentionally minimal — StandardSQL keywords
 * plus a connector catalog wired through lang-sql's `schema` option. Warehouse
 * validation stays authoritative on the backend.
 */
const SQLCodeEditor: React.FC<SQLCodeEditorProps> = ({
    value,
    onChange,
    catalog,
    editable = true,
    isInvalid = false,
    errorMsg = "",
    hideBorder = false,
    height = "300px",
    lineNumbers = true,
    showCopyButton = false,
    placeholder = "Write a read-only SELECT that runs in your warehouse…",
    fontSize = 16,
    backgroundColor
}) => (
    <div>
        <CodeEditor
            value={value}
            language="SQL"
            sqlSchema={catalog}
            placeholder={placeholder}
            basicSetup={{
                editable: true,
                lineNumbers: lineNumbers,
                highlightActiveLineGutter: true,
                highlightSpecialChars: true,
                history: true,
                foldGutter: false,
                drawSelection: true,
                indentOnInput: true,
                syntaxHighlighting: true,
                bracketMatching: true,
                closeBrackets: true,
                autocompletion: true,
                highlightActiveLine: true,
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
            onChange={onChange}
            editable={editable}
            isInvalid={isInvalid}
            errorMsg={errorMsg}
            hideBorder={hideBorder}
            height={height}
            showCopyButton={showCopyButton}
            fontSize={fontSize}
            backgroundColor={backgroundColor}
        />
    </div>
);

export default WithErrorBoundary(SQLCodeEditor);
