// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { RenderResult } from "@testing-library/react";
import { waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { vi } from "vitest";

import {
    defaultGraphSelectTablesProps,
    resetQueryClient,
    setupGraphTests
} from "components/Graphs/__tests__/graphTestUtils";

import { TableListResponse } from "types/tables";

import { GraphSelectTablesProps } from "../SelectTables/GraphSelectTables";

import {
    mockAllTables,
    mockColumnStats
} from "./MockResponses/MockColumnStats";

describe("GraphSelectTables", () => {
    const { graphSelectTablesMock, renderGraphSelectTables } =
        setupGraphTests();

    beforeEach(() => {
        resetQueryClient();
    });

    afterEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
    });

    const renderComponent = async (
        overrides: Partial<GraphSelectTablesProps> = {}
    ) => {
        return await renderGraphSelectTables({
            ...defaultGraphSelectTablesProps,
            ...overrides
        });
    };

    const getCheckboxes = (getAllByRole: RenderResult["getAllByRole"]) =>
        getAllByRole("checkbox");

    const getSelectAllCheckbox = (getAllByRole: RenderResult["getAllByRole"]) =>
        getCheckboxes(getAllByRole)[0];

    const getFirstRowCheckbox = (getAllByRole: RenderResult["getAllByRole"]) =>
        getCheckboxes(getAllByRole)[1];

    const expectTableSelection = (
        setGfc: ReturnType<typeof vi.fn>,
        expectedTables: string[]
    ) => {
        expect(setGfc).toHaveBeenCalledWith(
            expect.objectContaining({
                selected_tables: expectedTables
            })
        );
    };

    const expectUpdateGroupsCalled = (
        updateGroups: ReturnType<typeof vi.fn>
    ) => {
        expect(updateGroups).toHaveBeenCalled();
    };

    it("Should render DataTable with table headers when list is provided", async () => {
        const { getByTestId } = await renderComponent();

        const header = getByTestId("table-header");
        const headerCells = within(header).getAllByRole("columnheader");

        const headerText = headerCells.map((th) => th.textContent?.trim());
        const expectedHeaders = [
            "Table Name",
            "Created date",
            "Connector",
            "Source Table",
            "Preview"
        ];

        expect(headerText).toEqual(expect.arrayContaining(expectedHeaders));
    });

    it("Should not render DataTable when list is empty", async () => {
        const { queryByRole } = await renderComponent({
            tableList: []
        });

        expect(queryByRole("table")).toBeNull();
    });

    it("Should select and deselect a table when its checkbox is clicked, and update selected_tables state", async () => {
        const { setGfc, updateGroups } = graphSelectTablesMock;

        const { getAllByRole } = await renderComponent({
            setGraphFlowInfo: setGfc,
            updateGroups
        });

        const checkbox = getFirstRowCheckbox(getAllByRole);
        expect(checkbox).toBeInTheDocument();

        await userEvent.click(checkbox);
        expect(checkbox).toBeChecked();

        await waitFor(() => {
            expectTableSelection(setGfc, ["customer"]);
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                expect.stringContaining("v1/tables/"),
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });
        expectUpdateGroupsCalled(updateGroups);

        await userEvent.click(checkbox);
        expect(checkbox).not.toBeChecked();

        await waitFor(() => expectTableSelection(setGfc, []));
        expectUpdateGroupsCalled(updateGroups);
    });

    it("Should select all rows when 'select all' checkbox is clicked", async () => {
        const { setGfc, updateGroups } = graphSelectTablesMock;

        const { getAllByRole } = await renderComponent({
            setGraphFlowInfo: setGfc,
            updateGroups
        });

        const selectAllCheckbox = getSelectAllCheckbox(getAllByRole);
        await userEvent.click(selectAllCheckbox);
        expect(selectAllCheckbox).toBeChecked();

        await waitFor(() =>
            expectTableSelection(setGfc, ["customer", "transaction_ltv1"])
        );
        expectUpdateGroupsCalled(updateGroups);
    });

    it("Should remove table from selected_tables when checkbox is deselected", async () => {
        const { setGfc, updateGroups } = graphSelectTablesMock;

        const { getAllByRole } = await renderComponent({
            graphFlowInfo: {
                ...defaultGraphSelectTablesProps.graphFlowInfo,
                selected_tables: ["customer", "transaction_ltv1"]
            },
            setGraphFlowInfo: setGfc,
            updateGroups
        });

        const checkbox = getFirstRowCheckbox(getAllByRole);
        expect(checkbox).toBeChecked();
        await userEvent.click(checkbox);

        expectTableSelection(setGfc, ["transaction_ltv1"]);
        expectUpdateGroupsCalled(updateGroups);
    });

    it("Should deselect all tables when 'select all' checkbox is toggled off", async () => {
        const { setGfc, updateGroups } = graphSelectTablesMock;

        const { getAllByRole } = await renderComponent({
            graphFlowInfo: {
                ...defaultGraphSelectTablesProps.graphFlowInfo,
                selected_tables: ["customer", "transaction_ltv1"]
            },
            setGraphFlowInfo: setGfc,
            updateGroups
        });

        const selectAllCheckbox = getSelectAllCheckbox(getAllByRole);
        expect(selectAllCheckbox).toBeChecked();
        await userEvent.click(selectAllCheckbox);

        expectTableSelection(setGfc, []);
        expectUpdateGroupsCalled(updateGroups);
    });

    it("Should open graph side panel and show sample rows when Preview icon is clicked", async () => {
        const { getByTestId, getAllByText, getByText, findByText } =
            await renderComponent({
                allTables: mockAllTables
            });

        const previewButton = getByTestId("preview-0");
        await userEvent.click(previewButton);

        const headings = [
            "Source",
            "Source Table",
            "Column Name",
            "Data type",
            "Semantic Type",
            "Properties"
        ];

        headings.forEach((heading) => {
            const headingElement = getAllByText(heading);
            expect(headingElement.length).toBeGreaterThanOrEqual(1);
        });

        const [columnStats] = mockColumnStats;

        if (columnStats && typeof columnStats !== "boolean") {
            columnStats.columns.forEach((col) => {
                expect(getByText(col.name)).toBeInTheDocument();
                expect(getByText(col.stype)).toBeInTheDocument();
                expect(
                    getByText(new RegExp(col.dtype, "i"))
                ).toBeInTheDocument();

                col.sample_rows.forEach(async (sample) => {
                    const sampleElement = await findByText(sample);
                    expect(sampleElement).toBeInTheDocument();
                });
            });
        }
    });

    it("Should display `Something went wrong` for invalid table list", async () => {
        const { queryByRole } = await renderComponent({
            tableList: { name: false } as unknown as TableListResponse[]
        });

        const alert = queryByRole("alert");
        expect(alert).toBeInTheDocument();
        expect(alert).toHaveTextContent(/Something went wrong in/i);
        expect(alert).toHaveTextContent(/GraphSelectTables/i);
        expect(alert).toHaveTextContent(/Please contact Kumo support/i);
    });

    it("Should not render DataTable when tableList is null", async () => {
        const { queryByRole } = await renderComponent({
            tableList: null
        });

        expect(queryByRole("table")).toBeNull();
    });
});
