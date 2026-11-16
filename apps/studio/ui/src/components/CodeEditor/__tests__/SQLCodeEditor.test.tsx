// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";

import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import SQLCodeEditor from "../SQLCodeEditor";

const mockOnChange = vi.fn();

const defaultProps = {
    value: "",
    onChange: mockOnChange
};

const getEditor = () => screen.getByTestId("pql-editor");

const renderSQLCodeEditor = async (props = defaultProps) => {
    await act(async () => {
        render(<SQLCodeEditor {...props} />);
    });
};

describe("SQLCodeEditor", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("Should render the SQLCodeEditor component", async () => {
        await renderSQLCodeEditor();
        expect(getEditor()).toBeInTheDocument();
    });

    it("Should render the SQL language extension", async () => {
        await renderSQLCodeEditor();
        expect(
            getEditor().querySelector('[data-language="sql"]')
        ).toBeInTheDocument();
    });

    it("Should display the placeholder when value is empty", async () => {
        await renderSQLCodeEditor();
        expect(getEditor()).toHaveTextContent(
            "Write a read-only SELECT that runs in your warehouse"
        );
    });

    it("Should display the provided SQL value", async () => {
        await renderSQLCodeEditor({
            ...defaultProps,
            value: "SELECT customer_id FROM customers"
        });
        expect(getEditor()).toHaveTextContent(
            "SELECT customer_id FROM customers"
        );
    });

    it("Should accept a connector catalog without crashing", async () => {
        await act(async () => {
            render(
                <SQLCodeEditor
                    value="SELECT * FROM orders"
                    onChange={mockOnChange}
                    catalog={{ orders: ["order_id", "amount"] }}
                />
            );
        });
        expect(getEditor()).toBeInTheDocument();
    });
});
