// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { tables } from "tests/fixtures/MockGraphInformationPanel";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { getSkeletonElement, verifyTextExists } from "tests/testHelpers";

import { JSTable } from "types/tables";

import ConnectedTables, { ConnectedTablesProps } from "../ConnectedTables";

import multipleTables from "./MockResponses/ConnectedTablesResponse.json";

const onConnectTableClick = vi.fn();

const defaultProps: ConnectedTablesProps<object> = {
    projectId: "project-test",
    tables: tables,
    onConnectTableClick: onConnectTableClick,
    dialogRef: {
        current: {
            toggleDialog: vi.fn()
        }
    },
    fetchingTables: false
};

describe("ConnectedTables", () => {
    const renderConnectedTables = async (
        props: ConnectedTablesProps<object>
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <ConnectedTables {...props} />
                </MemoryRouter>
            );
        });
    };

    const verifyConnectedTablesHeader = async () => {
        await verifyTextExists("Connected Tables");
        await verifyTextExists(
            /You have connected the following tables from this connector/i
        );
        expect(
            screen.getByRole("button", { name: "Add table" })
        ).toBeInTheDocument();
    };

    const verifyTableHeaders = () => {
        const tableHeader = screen.getByTestId("table-header");
        expect(tableHeader).toBeInTheDocument();

        const connectedTableHeaders = [
            "Table Name",
            "Dimensions",
            "Fact",
            "Updated",
            "Action"
        ];

        const headers = within(tableHeader).getAllByRole("columnheader");
        const headerTexts = headers.map((th) =>
            th.textContent?.trim().toLowerCase()
        );

        connectedTableHeaders.forEach((expectedHeader) => {
            expect(headerTexts).toContain(expectedHeader.toLowerCase());
        });
    };

    const verifyTableNamesPresent = (tableList: JSTable[]) => {
        tableList.forEach((table) => {
            expect(
                screen.getByRole("link", { name: table.name_alias })
            ).toBeInTheDocument();
        });
    };

    const clickAddTableButton = async () => {
        await userEvent.click(
            screen.getByRole("button", { name: "Add table" })
        );
    };

    const clickTableLink = async (tableName: string) => {
        const tableLink = screen.getByRole("link", { name: tableName });
        await userEvent.click(tableLink);
    };

    const clickDeleteIcon = async (index: number) => {
        const deleteIcons = screen.getAllByTestId("delete-icon");
        await userEvent.click(deleteIcons[index]);
    };

    const searchForTable = async (searchText: string) => {
        const searchInput = screen.getByPlaceholderText("Search");
        await userEvent.type(searchInput, searchText);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
    });

    it("Should display the table content for `ConnectedTables`", async () => {
        await renderConnectedTables(defaultProps);

        await verifyConnectedTablesHeader();
        verifyTableHeaders();
        verifyTableNamesPresent(tables);
    });

    it("Should display `No connected tables` when the props are set empty", async () => {
        await renderConnectedTables({
            projectId: "project-test",
            tables: [],
            onConnectTableClick,
            dialogRef: {
                current: null
            },
            fetchingTables: false
        });

        await verifyTextExists("No connected tables");
        expect(
            screen.getByRole("button", { name: "Add table" })
        ).toBeInTheDocument();
    });

    it("Should display table header with loading skeleton for table content when `fetchingTables` prop is set as true", async () => {
        await renderConnectedTables({
            ...defaultProps,
            fetchingTables: true
        });

        await verifyConnectedTablesHeader();

        getSkeletonElement();
    });

    it("Should call `onConnectTableClick` when the `Add table` button is clicked", async () => {
        await renderConnectedTables({
            ...defaultProps,
            onConnectTableClick
        });

        await verifyConnectedTablesHeader();
        await clickAddTableButton();

        expect(onConnectTableClick).toHaveBeenCalled();
    });

    it("Should navigate to the Table Overview page when the `Connected table` is clicked", async () => {
        await renderConnectedTables(defaultProps);

        await verifyConnectedTablesHeader();

        const tableName = tables[1].name_alias as string;
        await clickTableLink(tableName);

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                "/project/project-test/tables/" + `${tableName}`
            );
        });
    });

    it("Should open delete dialog with correct content when delete icon is clicked", async () => {
        const mockToggleDialog = vi.fn();
        await renderConnectedTables({
            ...defaultProps,
            dialogRef: {
                current: {
                    toggleDialog: mockToggleDialog
                }
            }
        });

        await verifyConnectedTablesHeader();

        await clickDeleteIcon(0);

        expect(mockToggleDialog).toHaveBeenCalledWith(
            "Delete Table",
            expect.stringContaining(`Are you sure you want to delete table`),
            expect.objectContaining({
                data: tables[0],
                isConnector: false
            }),
            false,
            "Delete",
            "destructive"
        );
    });

    it("Should `enableGlobalFilter` when there are more than 3 tables", async () => {
        await renderConnectedTables({
            ...defaultProps,
            tables: multipleTables as unknown as JSTable[]
        });

        await verifyConnectedTablesHeader();

        const searchInput = screen.getByPlaceholderText("Search");
        expect(searchInput).toBeInTheDocument();
    });

    it("Should filter the connected tables based on Search input", async () => {
        await renderConnectedTables({
            ...defaultProps,
            tables: multipleTables as unknown as JSTable[]
        });

        await verifyConnectedTablesHeader();

        // Before search - verify all tables are present
        verifyTableNamesPresent(multipleTables as unknown as JSTable[]);

        // Search for specific table
        const searchText = `${multipleTables[2].name_alias}`;
        await searchForTable(searchText);

        // After search - verify filtered results
        expect(
            screen.getByRole("link", { name: multipleTables[2].name_alias })
        ).toBeInTheDocument();
        expect(
            screen.queryByRole("link", { name: multipleTables[0].name_alias })
        ).not.toBeInTheDocument();
    });

    it("Should display empty cell if the number of column, rows are empty under response", async () => {
        await renderConnectedTables({
            ...defaultProps,
            tables: multipleTables as unknown as JSTable[]
        });

        await verifyConnectedTablesHeader();

        const allCells = document.querySelectorAll('[data-slot="table-cell"]');
        expect(allCells[6]?.textContent?.trim()).toBe(
            "4 Col, 1.2K Rows, 29.4 KB"
        ); // Cell with Col and Row
        expect(allCells[11]?.textContent?.trim()).toBe(""); // Cell without Col or Row
    });

    it("Should have correct styling for the table name link", async () => {
        await renderConnectedTables(defaultProps);

        await verifyConnectedTablesHeader();

        const tableLink = screen.getByRole("link", {
            name: tables[0].name_alias
        });

        const linkWrapper = tableLink.closest("span");
        expect(linkWrapper).toBeInTheDocument();

        expect(linkWrapper).toHaveClass(
            "transition-colors",
            "text-text-primary",
            "underline"
        );

        expect(tableLink).toHaveAttribute(
            "href",
            `/project/project-test/tables/${tables[0].name_alias}`
        );
        expect(tableLink).toHaveAttribute(
            "id",
            `connected-table-cols-${tables[0].name_alias}`
        );
    });
});
