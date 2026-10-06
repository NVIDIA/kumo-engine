// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockNavigateCall } from "tests/mockRouter";
import { createQueryClient } from "tests/testUtil";

import { GraphFlowInfo, GraphTables } from "types/graphs";
import { HandleErrorFunc } from "types/validation";

import {
    SelectTableDropdown,
    SelectTableDropdownProps
} from "../SelectTables/SelectTableDropDown";

const defaultProps: SelectTableDropdownProps = {
    tableNameList: ["customers", "orders", "products"],
    graphFlowInfo: {
        name: "",
        id: "",
        description: "",
        tables: {},
        selected_tables: [],
        selected_table_ids: [],
        flow_type: ""
    } as GraphFlowInfo,
    setGraphFlowInfo: () => {},
    allTables: {} as GraphTables,
    setAllTables: () => {},
    tableIdMap: {},
    updateGroups: () => {},
    suggestLinks: () => {},
    resetErrLabel: () => {},
    toggleTableModel: () => {},
    newTable: null,
    setNewTable: () => {},
    handleError: {} as HandleErrorFunc,
    setLoading: () => {}
};

describe("SelectTableDropdown", () => {
    const queryClient = createQueryClient();
    const renderSelectTableDropdown = async (
        props: SelectTableDropdownProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <SelectTableDropdown {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    beforeEach(() => {
        mockNavigateCall();
    });

    afterEach(() => {
        queryClient.clear();
    });

    it("Should display table options when dropdown is opened", async () => {
        await renderSelectTableDropdown({
            ...defaultProps,
            tableNameList: ["customers", "orders"]
        });

        const selectButton = screen.getByRole("button");
        await userEvent.click(selectButton);

        await waitFor(() => {
            expect(screen.getByText("customers")).toBeInTheDocument();
            expect(screen.getByText("orders")).toBeInTheDocument();
            expect(screen.getByText("Add new table")).toBeInTheDocument();
        });
    });

    it.each([
        { tableCount: 1, tables: ["customers"], expected: "1 Tables" },
        {
            tableCount: 2,
            tables: ["customers", "orders"],
            expected: "2 Tables"
        },
        {
            tableCount: 3,
            tables: ["customers", "orders", "products"],
            expected: "3 Tables"
        }
    ])(
        "Should display $expected when $tableCount tables are selected",
        async ({ tables, expected }) => {
            const graphFlowInfo = {
                name: "test-graph",
                id: "graph-1",
                description: "",
                tables: Object.fromEntries(
                    tables.map((table) => [
                        table,
                        { columns: {}, cols: [], source_table: {} }
                    ])
                ),
                selected_tables: tables,
                selected_table_ids: [],
                flow_type: ""
            } as unknown as GraphFlowInfo;

            await renderSelectTableDropdown({
                ...defaultProps,
                tableNameList: tables,
                graphFlowInfo
            });

            const selectedText = await screen.findByText(expected);
            expect(selectedText).toBeInTheDocument();
        }
    );

    it("Should render with empty table list", async () => {
        await renderSelectTableDropdown({
            ...defaultProps,
            tableNameList: []
        });

        const selectButton = screen.getByRole("button");
        expect(selectButton).toBeInTheDocument();

        expect(
            screen.queryByTestId("combobox-option-0")
        ).not.toBeInTheDocument();
    });

    it("Should call toggleTableModel when Add new table is clicked", async () => {
        let toggleCalled = false;
        const toggleTableModel = () => {
            toggleCalled = true;
        };

        await renderSelectTableDropdown({
            ...defaultProps,
            tableNameList: ["customers", "orders"],
            toggleTableModel
        });

        const selectButton = screen.getByRole("button");
        await userEvent.click(selectButton);

        const addNewTableOption = await screen.findByText("Add new table");
        await userEvent.click(addNewTableOption);

        expect(toggleCalled).toBe(true);
    });

    it("Should select a table when clicking on a table option", async () => {
        await renderSelectTableDropdown({
            ...defaultProps,
            tableNameList: ["customers", "orders", "products"]
        });

        const selectButton = screen.getByRole("button");
        await userEvent.click(selectButton);

        const customersOption = await screen.findByText("customers");
        await userEvent.click(customersOption);

        const selectedText = await screen.findByText("1 Tables");
        expect(selectedText).toBeInTheDocument();
    });

    it("Should not call setNewTable when newTable is not in tableNameList", async () => {
        const setNewTable = vi.fn();

        await renderSelectTableDropdown({
            ...defaultProps,
            tableNameList: ["customers", "orders"],
            newTable: "nonexistent",
            setNewTable
        });

        const selectButton = screen.getByRole("button");
        expect(selectButton).toBeInTheDocument();
        expect(setNewTable).not.toHaveBeenCalled();
    });
});
