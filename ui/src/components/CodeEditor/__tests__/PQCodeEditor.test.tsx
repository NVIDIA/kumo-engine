// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";

import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { JSTable } from "types/tables";

import PQCodeEditor from "../PQCodeEditor";

const mockOnChange = vi.fn();

const defaultProps = {
    language: "PQLANG_V2" as const,
    value: "",
    onChange: mockOnChange,
    editable: true,
    isInvalid: false,
    errorMsg: "",
    graphTables: [] as JSTable[],
    hideBorder: false,
    height: "300px"
};

const getEditor = () => screen.getByTestId("pql-editor");

const renderPQCodeEditor = async (props = defaultProps) => {
    await act(async () => {
        render(<PQCodeEditor {...props} />);
    });
};

describe("PQCodeEditor", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render the PQCodeEditor component", async () => {
        await renderPQCodeEditor();
        expect(getEditor()).toBeInTheDocument();
    });

    it("Should display placeholder text when value is empty", async () => {
        await renderPQCodeEditor();
        expect(getEditor()).toHaveTextContent(
            "Start typing your predictive query here..."
        );
    });

    it("Should display provided value when value is not empty", async () => {
        await renderPQCodeEditor({
            ...defaultProps,
            value: "SELECT * FROM table"
        });
        expect(getEditor()).toHaveTextContent("SELECT * FROM table");
    });

    it("Should show error message when isInvalid is true", async () => {
        await renderPQCodeEditor({
            ...defaultProps,
            isInvalid: true,
            errorMsg: "Invalid query syntax"
        });
        expect(screen.getByText("Invalid query syntax")).toBeInTheDocument();
    });

    it("Should not show error message when isInvalid is false", async () => {
        await renderPQCodeEditor({
            ...defaultProps,
            isInvalid: false,
            errorMsg: "Invalid query syntax"
        });
        expect(
            screen.queryByText("Invalid query syntax")
        ).not.toBeInTheDocument();
    });

    it("Should apply custom height style", async () => {
        await renderPQCodeEditor({
            ...defaultProps,
            height: "fit-content"
        });
        expect(getEditor()).toHaveStyle({ height: "fit-content" });
    });

    it("Should handle border visibility when hideBorder is true", async () => {
        await renderPQCodeEditor({
            ...defaultProps,
            hideBorder: true
        });
        expect(getEditor()).toHaveStyle({ border: "none" });
    });

    it("Should keep placeholder visible on click when value is empty", async () => {
        await renderPQCodeEditor();
        const editor = getEditor();

        expect(editor).toHaveTextContent(
            "Start typing your predictive query here..."
        );
        await act(async () => {
            fireEvent.click(editor);
        });

        // Placeholder remains visible until user starts typing (new behavior)
        expect(editor).toHaveTextContent(
            "Start typing your predictive query here..."
        );
    });

    it("Should call onChange when editor content changes", async () => {
        await renderPQCodeEditor({ ...defaultProps, value: "test" });

        const textbox = screen.getByRole("textbox");
        await act(async () => {
            fireEvent.input(textbox, { target: { textContent: "new value" } });
        });

        expect(mockOnChange).toHaveBeenCalled();
    });
});
