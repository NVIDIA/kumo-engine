// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import React from "react";

import { acceptCompletion, autocompletion } from "@codemirror/autocomplete";
import { sql, SQLNamespace, StandardSQL } from "@codemirror/lang-sql";
import { syntaxHighlighting } from "@codemirror/language";
import {
    EditorSelection,
    Extension,
    RangeSetBuilder,
    StateCommand
} from "@codemirror/state";
import {
    Decoration,
    DecorationSet,
    EditorView,
    hoverTooltip,
    keymap,
    ViewPlugin,
    ViewUpdate
} from "@codemirror/view";
import { CopyToClipBoard } from "@kumo-ai/kumo-ui-catalog";
import { langs } from "@uiw/codemirror-extensions-langs";
import { xcodeLight } from "@uiw/codemirror-themes-all";
import CodeMirror from "@uiw/react-codemirror";

import {
    CodeEditorBasicSetup,
    HighlightTypes,
    HoverHintsType
} from "types/codeEditor";
import { JSTable } from "types/tables";

import { syntaxHighlight as pythonSyntaxHighlight } from "utils/codeEditor/pythonHighlight";
import { autocompleteWithGraphs } from "utils/PQLParsing/autocomplete";
import { pql1 } from "utils/PQLParsing/PQL1";
import { pqlLinter } from "utils/PQLParsing/PQLinter";
import { highlights, yaml } from "utils/YAML/yaml";

/**
 * Provides HTML compatible styling for the tooltip text.
 *
 * @val Value associated with keyword in the hoverHints object.
 * @returns .
 */
export function styleHints(val: {
    title: string;
    subtitle: string;
    hint: string;
    link?: string;
    type?: HighlightTypes;
}): string {
    const styleTitle = `<h2 style="margin-bottom: 3%; font-family: 'NVIDIA Sans', sans-serif">${val?.title.toUpperCase()}</h2>`;
    const styleSubtitle = `<p style="color: rgba(0, 0, 0, 0.8)">${val?.subtitle}</p>`;
    const description = `<h3>Description</h3>`;
    const styleHint = `<div style="color: #fc1373; margin: 0;">${val?.hint}</div>`;
    if (val?.link === "") {
        return styleTitle + styleSubtitle + description + styleHint;
    } else {
        const divider = '<hr style="margin-bottom: 1.5%;" class="solid">';

        const styleLink = `<a href="${val.link}" id="doc-link"
        rel="noreferrer" target="_blank" >See full documentation</a>`;
        return (
            styleTitle +
            styleSubtitle +
            description +
            styleHint +
            divider +
            styleLink
        );
    }
}

export function handleAtK(key: string): string {
    let lowerKey = key?.toLowerCase();
    if (/@\d+/.test(lowerKey)) {
        lowerKey = lowerKey.replace(/@\d+/, "@K");
        return lowerKey;
    }
    return lowerKey;
}

/**
 * Handles PQL multi-word keys, such as is not null, for each, etc., to return appropriate hoverhint.
 * @key Currently hovered key.
 * @fullText The whole line of text where the currently hovered key is part of.
 * @end The position where @key ends.
 * @start The position where @key starts.
 * @from The position of the start of the line.
 * @returns String that matches the key in the hoverhints JSON object.
 */
export function handleMultiwordKeys(
    key: string,
    fullText: string,
    end: number,
    start: number,
    from: number
): string {
    const lowerKey = key.toLowerCase();
    const lowerFullText = fullText.toLowerCase();

    function handleIs(
        key: string,
        fullText: string,
        end: number,
        from: number
    ): string {
        // always at the beginning
        // ["is in", "is not null", "is null"]
        const slice = key + fullText?.slice(end - from, end - from + 3);
        switch (slice) {
            case "is in":
                return "is in";
            case "is no":
                return "is not null";
            case "is nu":
                return "is null";
            default:
                return "is";
        }
    }
    function handleNull(
        key: string,
        fullText: string,
        start: number,
        from: number
    ) {
        // always at the end
        // ["is not null", "is null"]
        const slice = fullText?.slice(start - from - 3, start - from) + key;
        switch (slice) {
            case "is null":
            case " = null":
                return "is null";
            case "ot null":
            case "!= null":
                return "is not null";
            default:
                return "null";
        }
    }
    function handleNot(
        key: string,
        fullText: string,
        end: number,
        from: number
    ) {
        // middle or beginning; check after
        // ["is not null", "not like", "not contains"]
        const slice = key + fullText?.slice(end - from, end - from + 3);
        switch (slice) {
            case "not nu":
                return "is not null";
            case "not li":
                return "not like";
            case "not co":
                return "not contains";
            default:
                return "not";
        }
    }
    function handleLike(
        key: string,
        fullText: string,
        start: number,
        from: number
    ) {
        // always at the end
        // ["like", "not like"];
        const slice = fullText?.slice(start - from - 3, start - from) + key;
        switch (slice) {
            case "ot like":
                return "not like";
            default:
                return "like";
        }
    }
    function handleContains(
        key: string,
        fullText: string,
        start: number,
        from: number
    ) {
        // always at the end
        // ["contains", "not contains"];
        const slice = fullText?.slice(start - from - 3, start - from) + key;
        switch (slice) {
            case "ot contains":
                return "not contains";
            default:
                return "contains";
        }
    }
    function handleWith(
        key: string,
        fullText: string,
        start: number,
        from: number
    ) {
        // always at the end
        // ["starts with", "ends with"];
        const slice = fullText?.slice(start - from - 3, start - from) + key;
        switch (slice) {
            case "ts with":
                return "starts with";
            case "ds with":
                return "ends with";
            default:
                return "with";
        }
    }

    switch (lowerKey) {
        case "for":
        case "each":
            key = "for each";
            return key;
        case "in":
            key = "is in";
            return key;
        case "is":
            key = handleIs(lowerKey, lowerFullText, end, from);
            return key;
        case "null":
            key = handleNull(lowerKey, lowerFullText, start, from);
            return key;
        case "not":
            key = handleNot(lowerKey, lowerFullText, end, from);
            return key;
        case "like":
            key = handleLike(lowerKey, lowerFullText, start, from);
            return key;
        case "contains":
            key = handleContains(lowerKey, lowerFullText, start, from);
            return key;
        case "with":
            key = handleWith(lowerKey, lowerFullText, start, from);
            return key;
        case "ends":
            return "ends with";
        case "starts":
            return "starts with";
        case "rank":
        case "top":
        case "classify":
            return "rank";
        default:
            return key;
    }
}

/**
 * Enables displaying tooltips over the editor.
 *
 * @hoverHints Typescript Object of type HoverHintsType.
 * @returns Extension type because the returned value is passed as an extension in the CodeMirror element.
 *
 * The @hoverTooltip function determines the word boundaries around a given position.
 * @view An editor view represents the editor's user interface.
 * @pos Relative position to determine word start and end.
 * @side Start or end of word.
 * @regex query (/[\w!=<>[\]^_%*@=]+/) Captures all alphanumeric characters (\w), characters that are part of the boolean operations, and characters that are part of keywords (such as _ in list_distinct and @ in map@K).
 * @returns The DOM element to position over the editor.
 */
function tooltip(
    hoverHints: HoverHintsType,
    language?: "PQLANG_V2" | "YAML"
): Extension {
    return hoverTooltip((view: EditorView, pos: number, side: number) => {
        // eslint-disable-next-line no-unsafe-optional-chaining
        const { from, to, text } = view?.state?.doc.lineAt(pos);
        let start = pos,
            end = pos;

        while (
            start > from &&
            /[\w!=<>[\]^_%*@=]+/.test(text[start - from - 1])
        ) {
            start--;
        }

        while (end < to && /[\w!=<>[\]^_%*@=]+/.test(text[end - from])) {
            end++;
        }

        if ((start === pos && side < 0) || (end === pos && side > 0)) {
            return null;
        }

        let hoveredText: string = text?.slice(start - from, end - from);
        hoveredText = hoveredText?.toLowerCase();

        /// Exceptions:
        // 1. Accept keywords with more than one word and space
        const hoveredException =
            language === "PQLANG_V2"
                ? handleMultiwordKeys(hoveredText, text, end, start, from)
                : handleAtK(hoveredText);

        // 2. don't show hover hint if the hovered text is not a PQL/YAML keyword.
        if (!(hoveredException in hoverHints)) {
            return null;
        }

        // 3. Don't show hover hint if the hint entry for the keyword is an empty string.
        const hoveredVal = hoverHints[hoveredException];
        if (hoveredVal?.hint?.trim() === "") {
            return null;
        }

        return {
            pos: start,
            end,
            above: true,
            create() {
                const dom = document?.createElement("div");
                if (dom) {
                    dom.innerHTML = styleHints(hoveredVal);

                    // Add click handler to bypass DOMPurify restrictions
                    const link = dom.querySelector(
                        "#doc-link"
                    ) as HTMLAnchorElement;
                    if (link) {
                        link.onclick = (e) => {
                            e.preventDefault();
                            window.open(link.href, "_blank");
                        };
                    }
                }
                return { dom };
            }
        };
    });
}

/**
 * CSS Styling of the code editor.
 *
 * @validValue Used to conditionally change the stylin of the border depending on whether the value in the code editor is valid or not.
 * @returns CodeEditorStyleType which is passed in the style attribute of the CodeMirror element.
 */
function styleCodeEditor(
    validValue: boolean,
    hideBorder: boolean
): React.CSSProperties {
    const returnStyle = {
        height: "fit-content",
        borderRadius: "4px",
        border: hideBorder
            ? "none"
            : validValue === false
              ? "1px solid #cccccc"
              : "1px solid #c12c89",
        overflow: "hidden",
        whiteSpace: "pre-line",
        paddingRight: "0px",
        paddingLeft: "0px"
    } as React.CSSProperties;
    return returnStyle;
}

/**
 * Creates a ViewPlugin to handle text highlighting based on positions.
 *
 * @param positions - Array of {start, end} positions to highlight
 * @returns ViewPlugin extension for CodeMirror
 */
function createHighlightPlugin(
    positions: { start: number; end: number }[]
): Extension {
    const highlightMark = Decoration.mark({
        class: "cm-text-highlight"
    });

    return ViewPlugin.fromClass(
        class {
            decorations: DecorationSet;

            constructor() {
                this.decorations = this.buildDecorations();
            }

            buildDecorations(): DecorationSet {
                if (!positions || positions.length === 0) {
                    return Decoration.none;
                }

                const builder = new RangeSetBuilder<Decoration>();
                // Sort positions by start to ensure proper order
                const sortedPositions = [...positions].sort(
                    (a, b) => a.start - b.start
                );

                for (const { start, end } of sortedPositions) {
                    if (start >= 0 && end > start) {
                        builder.add(start, end, highlightMark);
                    }
                }
                return builder.finish();
            }

            update(update: ViewUpdate) {
                // Rebuild decorations if document changed
                if (update.docChanged) {
                    this.decorations = this.buildDecorations();
                }
            }
        },
        {
            decorations: (v) => v.decorations
        }
    );
}

/**
 * CSS Styling of the tooltip popup.
 *
 * @returns Extension type that adds extension to the base theme.
 */
function styleTooltip(
    fontSize: number,
    backgroundColor?: string,
    hideGutterBorder?: boolean
): Extension {
    return EditorView.theme({
        ...(backgroundColor && {
            "&": { backgroundColor },
            ".cm-content": { backgroundColor },
            ".cm-gutters": { backgroundColor },
            ".cm-gutter": { backgroundColor }
        }),
        ...(hideGutterBorder && {
            ".cm-gutters": { borderRight: "none" }
        }),
        ".cm-tooltip-autocomplete.cm-tooltip.cm-tooltip-below": {
            backgroundColor: "rgba(255, 255, 255, 0.89)",
            backdropFilter: "blur(1.5px)",
            borderColor: "#e3ebf6",
            borderRadius: "4px"
        },
        ".cm-tooltip-hover.cm-tooltip": {
            overflow: "auto",
            color: "black",
            padding: "15px",
            maxWidth: "500px",
            boxSizing: "border-box",
            backgroundColor: "rgba(255, 255, 255, 0.89)",
            backdropFilter: "blur(1.5px)",
            borderRadius: "4px",
            boxShadow: "1px 1px 1px 1px rgba(206, 195, 200, 0.2)"
        },
        ".cm-tooltip": {
            borderRadius: "4px",
            boxShadow: "1px 1px 1px 1px rgba(206, 195, 200, 0.2)",
            fontFamily: "var(--font-mono)"
        },
        ".cm-completionIcon-constant": {
            "&:after": {
                content: "'tbl' !important;",
                fontSize: "65%",
                verticalAlign: "baseline"
            }
        },
        ".cm-completionIcon-variable": {
            "&:after": {
                content: "'key' !important;",
                fontSize: "65%",
                verticalAlign: "baseline"
            }
        },
        ".cm-lintRange-error": {
            backgroundImage: "none !important;",
            // background: "rgba(255, 0, 0, 0.25);",
            // borderRadius: "1.5px",
            // padding: "1px"
            textDecoration: "underline red"
        },
        ".cm-gutterElement": {
            fontFamily: "var(--font-mono)",
            fontSize: fontSize + "px;",
            fontStyle: "normal;",
            fontWeight: "500;",
            lineHeight: "20px;",
            color: "var(--title-secondary)"
        },
        ".cm-line": {
            fontFamily: "var(--font-mono)",
            fontSize: fontSize + "px;",
            fontStyle: "normal;",
            fontWeight: "500;",
            lineHeight: "20px;"
        },
        ".cm-line span": {
            fontWeight: "500;"
        },
        ".cm-line span[class*='ͼpw']": {
            color: "#292D36"
        },
        ".cm-text-highlight": {
            fontWeight: "900"
        }
    });
}
const createCursorTabHandler = (spaces: number = 2): Extension => {
    // Create a command that inserts spaces at cursor position
    const insertSpacesAtCursor: StateCommand = ({ state, dispatch }) => {
        if (!state) return false;

        const spacesToInsert = " ".repeat(spaces || 2);

        try {
            const changes = state.changeByRange((range) => {
                if (range == null || typeof range.from !== "number") {
                    return {
                        changes: [],
                        range: range
                    };
                }
                return {
                    changes: [
                        {
                            from: range.from,
                            to: range.from,
                            insert: spacesToInsert
                        }
                    ],
                    range: EditorSelection.range(
                        range.from + (spaces || 2),
                        range.from + (spaces || 2)
                    )
                };
            });
            if (dispatch && changes) {
                dispatch(state.update(changes));
            }

            return true;
        } catch (error) {
            console.error("Error in tab handler:", error);
            return false;
        }
    };
    return keymap.of([
        {
            key: "Tab",
            run: insertSpacesAtCursor
        }
    ]);
};

/**
 * `CodeEditor` Component to show code snippets
 *
 * @param language -- "SQL", "YAML", or "PQLANG_V2".
 * @param hoverHints -- Object defining the language's keywords and their definitions. Used in the hovering tooltip.
 * @param theme -- Any custom theme from `@uiw/codemirror-themes-all'.
 * @param value -- Value.
 * @param isInvalid -- Boolean based on whether input is valid or not.
 * @returns ```<CodeMirror/>```
 */
export interface CodeEditorProps {
    language: "SQL" | "YAML" | "PQLANG_V2" | "PYTHON";
    hoverHints?: HoverHintsType;
    basicSetup: CodeEditorBasicSetup;
    theme?: "light" | "dark" | "none" | Extension;
    value: string;
    onChange: (newval: string) => void;
    editable?: boolean;
    isInvalid?: boolean;
    errorMsg?: string;
    graphTables?: JSTable[];
    hideBorder?: boolean;
    height?: string;
    isTabKeyEnabled?: boolean;
    showCopyButton?: boolean;
    enableHighlight?: boolean;
    highlightPositions?: { start: number; end: number }[];
    placeholder?: string;
    fontSize?: number;
    backgroundColor?: string;
    // Connector catalog (schemas → tables → columns) fed to lang-sql for
    // autocompletion. Only used by the "SQL" language.
    sqlSchema?: SQLNamespace;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
    language = "SQL",
    hoverHints,
    basicSetup,
    theme = xcodeLight,
    value,
    onChange,
    editable = true,
    isInvalid = false,
    errorMsg = "Required field",
    graphTables,
    hideBorder = false,
    isTabKeyEnabled = false,
    height = "300px",
    showCopyButton = false,
    enableHighlight = false,
    highlightPositions = [],
    placeholder,
    fontSize = 16,
    backgroundColor,
    sqlSchema
}) => {
    const width = "100%";
    let extensions: Array<Extension> = [
        styleTooltip(fontSize, backgroundColor, !basicSetup.lineNumbers),
        EditorView.lineWrapping
    ];

    switch (language) {
        case "SQL":
            // StandardSQL keyword completion plus optional catalog-driven
            // (tables/columns) completion. Deliberately minimal: no per-connector
            // dialects or ANTLR grammar.
            extensions = [
                ...extensions,
                sql({
                    dialect: StandardSQL,
                    schema: sqlSchema,
                    upperCaseKeywords: true
                })
            ];
            break;
        case "YAML":
            extensions = [
                ...extensions,
                yaml(),
                tooltip(hoverHints as HoverHintsType, language),
                syntaxHighlighting(highlights)
            ];
            break;
        case "PQLANG_V2":
            extensions = [
                ...extensions,
                pql1(),
                tooltip(hoverHints as HoverHintsType, language),
                autocompletion({
                    override: [autocompleteWithGraphs(graphTables as JSTable[])]
                }),
                pqlLinter(errorMsg as string)
            ];
            break;
        case "PYTHON":
            extensions = [...extensions, langs.python(), pythonSyntaxHighlight];
            break;
        default:
            console.log("Unsupported extension");
            return null;
    }

    let nHeight = height;
    if (editable === false) {
        value = value?.replace(/^\s+|\s+$/g, "");
        nHeight = "fit-content";
    }

    const tabExtension = isTabKeyEnabled
        ? createCursorTabHandler(2)
        : keymap.of([
              {
                  key: "Tab",
                  run: acceptCompletion
              }
          ]);

    const forceMonospaceTheme = EditorView.theme({
        ".cm-line, .cm-line *": {
            fontFamily: "var(--font-mono) !important"
        }
    });

    // Add highlight extension if enabled
    if (enableHighlight && highlightPositions.length > 0) {
        extensions = [...extensions, createHighlightPlugin(highlightPositions)];
    }

    return (
        <>
            <CodeMirror
                id={language}
                data-testid="pql-editor"
                style={styleCodeEditor(isInvalid as boolean, hideBorder)}
                value={value}
                height={nHeight}
                width={width}
                extensions={[...extensions, tabExtension, forceMonospaceTheme]}
                onChange={onChange}
                editable={editable}
                theme={theme}
                basicSetup={basicSetup}
                indentWithTab={false}
                placeholder={placeholder}
            />
            {showCopyButton && (
                <div className="absolute top-3 right-2">
                    <CopyToClipBoard
                        copiedText={value as string}
                        tooltipText={"Query copied!"}
                        size="sm"
                        variant="outline"
                        iconSize={16}
                    />
                </div>
            )}
            <div>
                {isInvalid && (
                    <div className="w-full mt-1 text-kumo-pink-600 break-words whitespace-pre-line">
                        {errorMsg}
                    </div>
                )}
            </div>
        </>
    );
};

export default CodeEditor;
