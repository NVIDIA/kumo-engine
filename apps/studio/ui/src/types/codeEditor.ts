// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export type CodeEditorBasicSetup = {
    editable: boolean;
    lineNumbers: boolean;
    highlightActiveLineGutter: boolean;
    highlightSpecialChars: boolean;
    history: boolean;
    foldGutter: boolean;
    drawSelection: boolean;
    indentOnInput: boolean;
    syntaxHighlighting: boolean;
    bracketMatching: boolean;
    closeBrackets: boolean;
    autocompletion: boolean;
    highlightActiveLine: boolean;
    highlightSelectionMatches: boolean;
    closeBracketsKeymap: boolean;
    dropCursor: boolean;
    allowMultipleSelections: boolean;
    rectangularSelection: boolean;
    crosshairCursor: boolean;
    defaultKeymap: boolean;
    searchKeymap: boolean;
    historyKeymap: boolean;
    foldKeymap: boolean;
    completionKeymap: boolean;
};

export enum HighlightTypes {
    KEYWORD = "keyword",
    STRING = "string",
    TYPE_NAME = "typeName",
    NAME = "name",
    VARIABLE_NAME = "variableName",
    OPERATOR = "operator",
    NUMBER = "number"
}

export type HoverHintsType = {
    [key: string]: {
        title: string;
        subtitle: string;
        hint: string;
        link?: string;
        type?: HighlightTypes;
        body_html?: string;
        slug?: string;
    };
};
