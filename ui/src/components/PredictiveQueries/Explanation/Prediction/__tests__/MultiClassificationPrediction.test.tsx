// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { render, screen, within } from "@testing-library/react";

import { multiclassEntityOutput } from "../../__tests__/MockResponses/EntityOutputResponses";
import MultiClassificationPrediction from "../MultiClassificationPrediction";

const defaultProps = {
    entityOutput: multiclassEntityOutput,
    isDataLoading: false,
    showGroundTruth: true
};

describe("MultiClassificationPrediction", () => {
    const renderMultiClassificationPrediction = async (
        props = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <MultiClassificationPrediction {...props} />
                </MemoryRouter>
            );
        });
    };

    const verifyTableRows = (tableId: string, expectedRows: string[][]) => {
        const table = document.getElementById(tableId);
        expect(table).toBeInTheDocument();

        const rows = within(table!).getAllByRole("row").slice(1);

        expectedRows.forEach(([label, value], idx) => {
            const rowCells = within(rows[idx]).getAllByRole("cell");
            expect(rowCells[0]).toHaveTextContent(label);
            expect(rowCells[1]).toHaveTextContent(value);
        });
    };

    it("Should render both tables with correct headers and values", async () => {
        await renderMultiClassificationPrediction();

        expect(screen.getByText("Predictions")).toBeInTheDocument();
        expect(screen.getByText("Ground Truth")).toBeInTheDocument();

        const tables = screen.getAllByRole("table");
        expect(tables.length).toBe(2);

        const headers = screen.getAllByTestId("table-header");
        expect(headers.length).toBe(2);
        headers.forEach((header) => {
            expect(header).toHaveTextContent("Label");
            expect(header).toHaveTextContent("Value");
        });

        verifyTableRows("prediction-table", [
            ["SIPO", "0.99"],
            ["UVER", "0.01"],
            ["POJISTNE", "0.00"],
            ["LEASING", "0.00"]
        ]);

        verifyTableRows("ground-truth-table", [
            ["SIPO", "1"],
            ["UVER", "0"],
            ["POJISTNE", "0"],
            ["LEASING", "0"]
        ]);
    });

    it("Should display skeletons when isDataLoading is true", async () => {
        await renderMultiClassificationPrediction({
            ...defaultProps,
            isDataLoading: true
        });

        ["prediction-table", "ground-truth-table"].forEach((tableId) => {
            const table = document.getElementById(tableId);
            expect(table).toBeInTheDocument();

            const rows = within(table!).getAllByRole("row").slice(1);
            rows.forEach((row) => {
                const cells = within(row).getAllByRole("cell");
                cells.forEach((cell) => {
                    expect(
                        cell.querySelector('[data-slot="skeleton"]')
                    ).toBeInTheDocument();
                });
            });
        });
    });

    it("Should not render ground truth table when showGroundTruth is false", async () => {
        await renderMultiClassificationPrediction({
            ...defaultProps,
            showGroundTruth: false
        });

        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
        expect(
            document.getElementById("ground-truth-table")
        ).not.toBeInTheDocument();
    });

    it.each([
        {
            testName: "prediction data is null",
            dataKey: "prediction" as const,
            expectedText: "Predictions"
        },
        {
            testName: "target data is null",
            dataKey: "target" as const,
            expectedText: "Ground Truth"
        }
    ])(
        "Should render skeleton when $testName",
        async ({ dataKey, expectedText }) => {
            const entityOutputWithNullData = {
                ...multiclassEntityOutput,
                [dataKey]: null
            };

            await renderMultiClassificationPrediction({
                ...defaultProps,
                entityOutput:
                    entityOutputWithNullData as unknown as typeof multiclassEntityOutput
            });

            expect(screen.getByText(expectedText)).toBeInTheDocument();

            const section = screen.getByText(expectedText).closest("div");
            const skeleton = section?.querySelector('[data-slot="skeleton"]');
            expect(skeleton).toBeInTheDocument();
            expect(skeleton).toHaveClass("w-12", "h-6");
        }
    );

    it("Should return null when entityOutput is null", async () => {
        await renderMultiClassificationPrediction({
            ...defaultProps,
            entityOutput: null as unknown as typeof multiclassEntityOutput
        });

        expect(screen.queryByText("Predictions")).not.toBeInTheDocument();
        expect(screen.queryByText("Ground Truth")).not.toBeInTheDocument();
    });
});
