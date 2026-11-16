// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";

import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { CodeEditorBasicSetup } from "types/codeEditor";

import CodeEditor, {
    CodeEditorProps,
    handleAtK,
    handleMultiwordKeys,
    styleHints
} from "../CodeEditor";

const defaultBasicSetup: CodeEditorBasicSetup = {
    editable: true,
    lineNumbers: true,
    highlightActiveLineGutter: true,
    highlightSpecialChars: true,
    history: true,
    foldGutter: true,
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
    crosshairCursor: true,
    defaultKeymap: true,
    searchKeymap: true,
    historyKeymap: true,
    foldKeymap: true,
    completionKeymap: true
};

const mockOnChange = vi.fn();

const defaultProps: CodeEditorProps = {
    language: "SQL",
    value: "SELECT * FROM users;",
    onChange: mockOnChange,
    basicSetup: defaultBasicSetup,
    hoverHints: {},
    theme: "light",
    editable: true,
    isInvalid: false,
    errorMsg: "Required field",
    graphTables: [],
    hideBorder: false,
    height: "300px",
    isTabKeyEnabled: false
};

const getEditor = () => screen.getByTestId("pql-editor");

const getEditorTextbox = () => screen.getByRole("textbox");

const queryEditorAttribute = (selector: string) => {
    return getEditor().querySelector(selector);
};

const getCopyButtonContainer = () => {
    return document.querySelector("div.absolute.top-3.right-2");
};

describe("CodeEditor", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const renderCodeEditor = async (props: CodeEditorProps) => {
        await act(async () => {
            render(<CodeEditor {...props} />);
        });
    };

    it("Should render the CodeEditor component", async () => {
        await renderCodeEditor(defaultProps);
        expect(getEditor()).toBeInTheDocument();
    });

    it("Should render non-editable editor when editable is false", async () => {
        await renderCodeEditor({ ...defaultProps, editable: false });
        const nonEditableContent = queryEditorAttribute(
            '[contenteditable="false"]'
        );
        expect(nonEditableContent).toBeInTheDocument();
    });

    it("Should trim value when editable is false", async () => {
        await renderCodeEditor({
            ...defaultProps,
            value: "   SELECT * FROM users;   ",
            editable: false
        });
        const line = queryEditorAttribute(".cm-line");
        expect(line?.textContent?.trim()).toBe("SELECT * FROM users;");
    });

    it("Should hide outer container border when hideBorder is true", async () => {
        await renderCodeEditor({ ...defaultProps, hideBorder: true });
        await waitFor(() => {
            const editor = getEditor();
            expect(getComputedStyle(editor).borderStyle).toBe("none");
            const gutter = document.querySelector(".cm-gutters");
            expect(gutter).toBeInTheDocument();
        });
    });

    describe("Language Support", () => {
        it.each([
            {
                language: "SQL" as const,
                expectedAttribute: '[data-language="sql"]'
            },
            {
                language: "YAML" as const,
                expectedAttribute: '[data-language="yaml"]'
            },
            {
                language: "PQLANG_V2" as const,
                expectedAttribute: '[data-language="PQL1"]'
            }
        ])(
            "Should render $language language editor",
            async ({ language, expectedAttribute }) => {
                await renderCodeEditor({ ...defaultProps, language });
                expect(
                    queryEditorAttribute(expectedAttribute)
                ).toBeInTheDocument();
            }
        );
    });

    describe("Theme Support", () => {
        it.each([
            { theme: "light" as const, expectedClass: "cm-theme-light" },
            { theme: "dark" as const, expectedClass: "cm-theme-dark" },
            { theme: "none" as const, expectedClass: "cm-theme-none" }
        ])(
            "Should apply $theme theme class",
            async ({ theme, expectedClass }) => {
                await renderCodeEditor({ ...defaultProps, theme });
                expect(getEditor().className).toContain(expectedClass);
            }
        );
    });

    describe("Error Handling", () => {
        it("Should display error message when isInvalid is true", async () => {
            await renderCodeEditor({
                ...defaultProps,
                isInvalid: true,
                errorMsg: "Field is required"
            });
            expect(screen.getByText("Field is required")).toBeInTheDocument();
        });

        it("Should display default error message when isInvalid is true and no errorMsg is passed", async () => {
            await renderCodeEditor({
                ...defaultProps,
                isInvalid: true,
                errorMsg: undefined
            });
            expect(screen.getByText("Required field")).toBeInTheDocument();
        });

        it("Should not display error message when isInvalid is false", async () => {
            await renderCodeEditor({
                ...defaultProps,
                isInvalid: false,
                errorMsg: "Some error"
            });
            expect(screen.queryByText("Some error")).not.toBeInTheDocument();
        });
    });

    describe("Tab Key Behavior", () => {
        it("Should handle Tab key press when isTabKeyEnabled is true", async () => {
            await renderCodeEditor({ ...defaultProps, isTabKeyEnabled: true });
            const textbox = getEditorTextbox();
            textbox.focus();
            fireEvent.keyDown(textbox, {
                key: "Tab",
                code: "Tab",
                charCode: 9
            });
            expect(mockOnChange).toHaveBeenCalled();
        });

        it("Should not handle Tab key press when isTabKeyEnabled is false", async () => {
            await renderCodeEditor({ ...defaultProps, isTabKeyEnabled: false });
            const textbox = getEditorTextbox();
            textbox.focus();
            fireEvent.keyDown(textbox, {
                key: "Tab",
                code: "Tab",
                charCode: 9
            });
            expect(mockOnChange).not.toHaveBeenCalled();
        });
    });

    describe("Copy Button", () => {
        it("Should render copy button container when showCopyButton is true", async () => {
            await renderCodeEditor({ ...defaultProps, showCopyButton: true });
            expect(getCopyButtonContainer()).toBeInTheDocument();
        });

        it("Should not render copy button when showCopyButton is false", async () => {
            await renderCodeEditor({ ...defaultProps, showCopyButton: false });
            expect(getCopyButtonContainer()).not.toBeInTheDocument();
        });
    });
});

describe("styleHints", () => {
    const createHintParams = (overrides = {}) => ({
        title: "Sample Title",
        subtitle: "This is a subtitle",
        hint: "Use this hint",
        link: "https://example.com",
        ...overrides
    });

    const expectCommonElements = (html: string) => {
        expect(html).toContain("SAMPLE TITLE");
        expect(html).toContain("This is a subtitle");
        expect(html).toContain("Use this hint");
    };

    it("Should return formatted HTML with all elements when link is provided", () => {
        const html = styleHints(createHintParams());

        expectCommonElements(html);
        expect(html).toContain("Description");
        expect(html).toContain("See full documentation");
        expect(html).toContain("https://example.com");
    });

    it("Should return formatted HTML without link section when link is empty", () => {
        const html = styleHints(createHintParams({ link: "" }));

        expectCommonElements(html);
        expect(html).not.toContain("See full documentation");
        expect(html).not.toContain("<hr");
    });

    it("Should uppercase the title", () => {
        const html = styleHints(createHintParams({ title: "lowercase title" }));
        expect(html).toContain("LOWERCASE TITLE");
    });
});

describe("handleAtK", () => {
    it("Should replace @<digits> with @K", () => {
        expect(handleAtK("@123")).toBe("@K");
        expect(handleAtK("@456")).toBe("@K");
        expect(handleAtK("@0")).toBe("@K");
    });

    it("Should replace only leading digits after @", () => {
        expect(handleAtK("@123ABC")).toBe("@Kabc");
    });

    it("Should leave non-digit @ keys unchanged but lowercase", () => {
        expect(handleAtK("@key")).toBe("@key");
        expect(handleAtK("@VALUE")).toBe("@value");
    });

    it("Should lowercase text without @", () => {
        expect(handleAtK("someText")).toBe("sometext");
        expect(handleAtK("UPPERCASE")).toBe("uppercase");
    });
});

describe("handleMultiwordKeys", () => {
    describe("'is' keyword variations", () => {
        it("Should return 'is in' when key is 'in'", () => {
            expect(
                handleMultiwordKeys("in", "select * is in (1, 2, 3)", 10, 5, 0)
            ).toBe("is in");
        });

        it("Should return 'is in' when key is 'is' followed by 'in'", () => {
            expect(
                handleMultiwordKeys("is", "select * is in (1, 2, 3)", 11, 5, 0)
            ).toBe("is in");
        });

        it("Should return 'is not null' when key is 'is' followed by 'not'", () => {
            expect(
                handleMultiwordKeys("is", "column is not null", 9, 5, 0)
            ).toBe("is not null");
        });

        it("Should return 'is null' when key is 'is' followed by 'null'", () => {
            expect(handleMultiwordKeys("is", "column is null", 9, 2, 0)).toBe(
                "is null"
            );
        });

        it("Should return 'is' when not followed by 'in', 'not', or 'null'", () => {
            expect(handleMultiwordKeys("is", "this is a test", 7, 2, 0)).toBe(
                "is"
            );
        });
    });

    describe("'null' keyword variations", () => {
        it("Should return 'is null' when preceded by 'is'", () => {
            expect(
                handleMultiwordKeys("null", "column is null", 14, 10, 0)
            ).toBe("is null");
        });

        it("Should return 'is null' when preceded by ' ='", () => {
            expect(handleMultiwordKeys("null", "column = null", 13, 9, 0)).toBe(
                "is null"
            );
        });

        it("Should return 'is not null' when preceded by 'ot'", () => {
            expect(
                handleMultiwordKeys("null", "column not null", 15, 11, 0)
            ).toBe("is not null");
        });

        it("Should return 'is not null' when preceded by '!='", () => {
            expect(
                handleMultiwordKeys("null", "column != null", 14, 10, 0)
            ).toBe("is not null");
        });

        it("Should return 'null' for standalone null", () => {
            expect(
                handleMultiwordKeys("null", "value null check", 10, 6, 0)
            ).toBe("null");
        });
    });

    describe("'not' keyword variations", () => {
        it("Should return 'is not null' when 'not' is followed by 'null'", () => {
            expect(
                handleMultiwordKeys("not", "column is not null", 13, 5, 0)
            ).toBe("is not null");
        });

        it("Should return 'not like' when 'not' is followed by 'like'", () => {
            expect(
                handleMultiwordKeys("not", "column not like", 10, 5, 0)
            ).toBe("not like");
        });

        it("Should return 'not contains' when 'not' is followed by 'contains'", () => {
            expect(
                handleMultiwordKeys(
                    "not",
                    "column not contains 'value'",
                    10,
                    5,
                    0
                )
            ).toBe("not contains");
        });

        it("Should return 'not' when no matching pattern found", () => {
            expect(
                handleMultiwordKeys(
                    "not",
                    "this is just some random text",
                    10,
                    5,
                    0
                )
            ).toBe("not");
        });
    });

    describe("'like' and 'contains' keywords", () => {
        it("Should return 'not like' when 'like' is preceded by 'ot'", () => {
            expect(
                handleMultiwordKeys(
                    "like",
                    "column not like '%value%'",
                    15,
                    11,
                    0
                )
            ).toBe("not like");
        });

        it("Should return 'like' for standalone like", () => {
            expect(
                handleMultiwordKeys("like", "column like '%value%'", 11, 7, 0)
            ).toBe("like");
        });

        it("Should return 'not contains' when 'contains' is preceded by 'ot'", () => {
            expect(
                handleMultiwordKeys(
                    "contains",
                    "column not contains 'value'",
                    19,
                    11,
                    0
                )
            ).toBe("not contains");
        });

        it("Should return 'contains' for standalone contains", () => {
            expect(
                handleMultiwordKeys(
                    "contains",
                    "column contains 'value'",
                    15,
                    7,
                    0
                )
            ).toBe("contains");
        });
    });

    describe("'starts with' and 'ends with' keywords", () => {
        it("Should return 'starts with' when key is 'starts'", () => {
            expect(
                handleMultiwordKeys(
                    "starts",
                    "column starts with 'text'",
                    13,
                    6,
                    0
                )
            ).toBe("starts with");
        });

        it("Should return 'ends with' when key is 'ends'", () => {
            expect(
                handleMultiwordKeys("ends", "column ends with 'text'", 11, 4, 0)
            ).toBe("ends with");
        });

        it("Should return 'starts with' when 'with' is preceded by 'ts'", () => {
            expect(
                handleMultiwordKeys(
                    "with",
                    "column starts with 'text'",
                    18,
                    14,
                    0
                )
            ).toBe("starts with");
        });

        it("Should return 'ends with' when 'with' is preceded by 'ds'", () => {
            expect(
                handleMultiwordKeys(
                    "with",
                    "column ends with 'text'",
                    16,
                    12,
                    0
                )
            ).toBe("ends with");
        });

        it("Should return 'with' when not preceded by 'starts' or 'ends'", () => {
            expect(
                handleMultiwordKeys(
                    "with",
                    "column value with another value",
                    17,
                    13,
                    0
                )
            ).toBe("with");
        });
    });

    describe("'for each' keyword", () => {
        it("Should return 'for each' when key is 'for'", () => {
            expect(
                handleMultiwordKeys("for", "for each column", 10, 3, 0)
            ).toBe("for each");
        });

        it("Should return 'for each' when key is 'each'", () => {
            expect(
                handleMultiwordKeys("each", "for each column", 10, 4, 0)
            ).toBe("for each");
        });
    });

    describe("rank-related keywords", () => {
        it("Should return 'rank' when key is 'rank'", () => {
            expect(
                handleMultiwordKeys("rank", "rank by column", 10, 4, 0)
            ).toBe("rank");
        });

        it("Should return 'rank' when key is 'top'", () => {
            expect(handleMultiwordKeys("top", "top 10 results", 10, 3, 0)).toBe(
                "rank"
            );
        });

        it("Should return 'rank' when key is 'classify'", () => {
            expect(
                handleMultiwordKeys("classify", "classify by type", 10, 8, 0)
            ).toBe("rank");
        });
    });

    describe("default behavior", () => {
        it("Should return key itself when no match is found", () => {
            expect(
                handleMultiwordKeys("random", "this is a test", 10, 3, 0)
            ).toBe("random");
        });
    });
});
