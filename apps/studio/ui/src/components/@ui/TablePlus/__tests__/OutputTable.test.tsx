// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, within } from "@testing-library/react";
import {
    clickWithoutPointerCheck,
    getDataRows,
    getTable,
    verifyTextExists
} from "tests/testHelpers";

import OutputTable, { OutputTableProps } from "../OutputTable";

import {
    outputTableResponse,
    repeatedColumnNameResponse,
    repeatedColumnSamplesResponse
} from "./MockResponses/OutputTableResponse";

const defaultProps: OutputTableProps = {
    title: "",
    cols: [
        {
            name: "",
            colType: "",
            samples: []
        }
    ],
    loading: false,
    areSamplesAvailable: false
};

describe("OutputTable", () => {
    const renderOutputTable = async (props: OutputTableProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <OutputTable {...props} />
                </MemoryRouter>
            );
        });
    };

    const getCellTexts = (
        rows: NodeListOf<Element>,
        cellIndex: number
    ): string[] => {
        return Array.from(rows).map((row) => {
            const cells = row.querySelectorAll('[data-slot="table-cell"]');
            return cells[cellIndex]?.textContent?.trim() ?? "";
        });
    };

    const clickHeaderAndGetCellTexts = async (
        table: HTMLElement,
        headerText: string,
        cellIndex: number
    ) => {
        const header = within(table).getByText(headerText);
        await clickWithoutPointerCheck(header);
        const rows = getDataRows(table);
        return getCellTexts(rows, cellIndex);
    };

    it("Should render OutputTable without crashing", async () => {
        await renderOutputTable({
            ...defaultProps,
            title: "Predictions table"
        });

        const schemaTitle = document.querySelector(
            '[data-testid="output-table-title"]'
        );
        expect(schemaTitle).toHaveTextContent("Predictions table");
    });

    it("Should display OutputTable column headers", async () => {
        await renderOutputTable({
            ...defaultProps,
            areSamplesAvailable: true
        });

        const table = getTable();
        const expectedHeaders = ["Column name", "Data type", "Column samples"];

        expectedHeaders.forEach((header) => {
            expect(within(table).getByText(header)).toBeInTheDocument();
        });
    });

    it("Should render table rows with correct column values", async () => {
        await renderOutputTable({
            title: "Predictions table",
            cols: outputTableResponse,
            loading: false,
            areSamplesAvailable: true
        });

        const table = getTable();
        const dataRows = getDataRows(table);

        expect(dataRows.length).toBe(outputTableResponse.length);

        dataRows.forEach((row, index) => {
            const cells = row.querySelectorAll('[data-slot="table-cell"]');
            const colData = outputTableResponse[index];
            const expectedSampleText = colData.samples?.join(", ") ?? "";

            expect(cells[0]).toHaveTextContent(colData.name);
            expect(cells[1]).toHaveTextContent(colData.colType);
            expect(cells[2]).toHaveTextContent(expectedSampleText);
        });
    });

    it("Should show loading skeletons inside each table cell when loading is true", async () => {
        await renderOutputTable({
            ...defaultProps,
            loading: true
        });

        const table = getTable();
        const dataCells = table.querySelectorAll(
            'tbody [data-slot="table-cell"]'
        );

        dataCells.forEach((cell) => {
            const skeleton = cell.querySelector('[data-slot="skeleton"]');
            expect(skeleton).toBeInTheDocument();
        });
    });

    it("Should render empty state when no columns are provided", async () => {
        await renderOutputTable({
            ...defaultProps,
            cols: [],
            areSamplesAvailable: true
        });

        await verifyTextExists(/No data found./i);
    });

    it("Should hide samples column when samples are not available", async () => {
        await renderOutputTable({
            ...defaultProps,
            areSamplesAvailable: false
        });

        const table = getTable();

        expect(
            within(table).queryByText("Column samples")
        ).not.toBeInTheDocument();

        const dataRows = getDataRows(table);
        dataRows.forEach((row) => {
            const cells = row.querySelectorAll('[data-slot="table-cell"]');
            expect(cells[2]).toBeUndefined();
        });
    });

    it("Should sort rows correctly by column name", async () => {
        await renderOutputTable({
            ...defaultProps,
            areSamplesAvailable: true,
            cols: outputTableResponse
        });

        const table = getTable();

        const cellTextsAsc = await clickHeaderAndGetCellTexts(
            table,
            "Column name",
            0
        );
        expect(cellTextsAsc).toEqual([
            outputTableResponse[0].name,
            outputTableResponse[1].name
        ]);

        const cellTextsDesc = await clickHeaderAndGetCellTexts(
            table,
            "Column name",
            0
        );
        expect(cellTextsDesc).toEqual([
            outputTableResponse[1].name,
            outputTableResponse[0].name
        ]);
    });

    it("Should not sort rows when column names are the same", async () => {
        await renderOutputTable({
            ...defaultProps,
            areSamplesAvailable: true,
            cols: repeatedColumnNameResponse
        });

        const table = getTable();
        const nameCells = await clickHeaderAndGetCellTexts(
            table,
            "Column name",
            0
        );

        expect(nameCells).toEqual(["ENTITY", "ENTITY"]);

        const dataRows = getDataRows(table);
        const samplesCells = getCellTexts(dataRows, 2);

        expect(samplesCells).toEqual([
            repeatedColumnNameResponse[0].samples.join(", "),
            repeatedColumnNameResponse[1].samples.join(", ")
        ]);
    });

    it("Should sort rows correctly by column samples", async () => {
        await renderOutputTable({
            ...defaultProps,
            areSamplesAvailable: true,
            cols: outputTableResponse
        });

        const table = getTable();

        const samplesAsc = await clickHeaderAndGetCellTexts(
            table,
            "Column samples",
            2
        );
        expect(samplesAsc).toEqual([
            outputTableResponse[0].samples?.join(", "),
            outputTableResponse[1].samples?.join(", ")
        ]);

        const samplesDesc = await clickHeaderAndGetCellTexts(
            table,
            "Column samples",
            2
        );
        expect(samplesDesc).toEqual([
            outputTableResponse[1].samples?.join(", "),
            outputTableResponse[0].samples?.join(", ")
        ]);
    });

    it("Should not sort rows when column samples are the same", async () => {
        await renderOutputTable({
            ...defaultProps,
            areSamplesAvailable: true,
            cols: repeatedColumnSamplesResponse
        });

        const table = getTable();
        const samplesCells = await clickHeaderAndGetCellTexts(
            table,
            "Column samples",
            2
        );

        expect(samplesCells).toEqual([
            repeatedColumnSamplesResponse[0].samples?.join(", "),
            repeatedColumnSamplesResponse[1].samples?.join(", ")
        ]);

        const dataRows = getDataRows(table);
        const nameCells = getCellTexts(dataRows, 0);

        expect(nameCells).toEqual([
            repeatedColumnSamplesResponse[0].name,
            repeatedColumnSamplesResponse[1].name
        ]);
    });

    it("Should handle for undefined samples", async () => {
        await renderOutputTable({
            ...defaultProps,
            cols: [
                {
                    name: "Test Col",
                    colType: "string",
                    samples: undefined
                }
            ],
            areSamplesAvailable: true
        });

        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("Should handle UI if samples not provided for a row", async () => {
        await renderOutputTable({
            ...defaultProps,
            areSamplesAvailable: true,
            cols: [
                {
                    name: "Col without samples",
                    colType: "string"
                }
            ]
        });

        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });
});
