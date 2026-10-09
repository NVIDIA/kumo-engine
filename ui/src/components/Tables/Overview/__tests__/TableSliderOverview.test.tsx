// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Column } from "gen/v1/types";
import { mockAxiosInstance } from "tests/mockAxios";
import { createQueryClient } from "tests/testUtil";

import { mockAllTables } from "components/Graphs/__tests__/MockResponses/MockColumnStats";

import { GraphTables } from "types/graphs";

import { storageKeys } from "utils/Constants";

import TableSliderOverview, {
    TableSliderOverviewProps
} from "../TableSliderOverview";

const defProps = {
    name: "",
    tableId: "",
    allTablesInGraph: mockAllTables,
    onClose: vi.fn(),
    suggestLinks: vi.fn(),
    editable: false,
    graphNameOrId: undefined,
    updateTable: vi.fn(),
    callCreateGraph: vi.fn()
};

const tableName = "transaction_ltv1";

describe("TableSliderOverview", async () => {
    const queryClient = createQueryClient();

    const renderTableSliderOverview = async (
        props: TableSliderOverviewProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <TableSliderOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async (table = tableName) => {
        await waitFor(() => {
            const APIS_TO_CALL = [
                `v1/tables/${table}`,
                `v1/tables/${table}/col_stats`
            ];
            APIS_TO_CALL.forEach((API) => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `${API}`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            });
        });
    };

    const callSampleRowsAndGetColumns = async () => {
        const connectorId = "ltv_test";
        const sourceTable = "transaction";

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/columns`,
                undefined,
                {
                    "headers": undefined,
                    "params": {
                        "table_name": sourceTable
                    }
                }
            );
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/sample`,
                undefined,
                {
                    "headers": undefined,
                    "params": {
                        "sample_rows": 10,
                        "table_name": sourceTable
                    }
                }
            );
        });
    };

    beforeEach(() => {
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");
        queryClient.clear();
        vi.clearAllMocks();
    });

    it("Should show skeleton loader when loading", async () => {
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true
        });

        const skeletonElement = document.querySelector(
            '[data-slot="skeleton"]'
        );
        expect(skeletonElement).toBeInTheDocument();
    });

    it("Should display graph side window view when details provided", async () => {
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName
        });

        const sideView = document.getElementById("graph-side-window-view");
        expect(sideView).toBeInTheDocument();
    });

    it("Should call `onClose` when close button is clicked", async () => {
        const onCloseMock = vi.fn();
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            onClose: onCloseMock
        });

        const closeButton = screen.getByTestId("close-icon");
        await userEvent.click(closeButton);

        await waitFor(
            () => {
                expect(onCloseMock).toHaveBeenCalled();
            },
            { timeout: 1500 }
        );
    });

    it("Should render table metadata labels with correct values", async () => {
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName
        });

        await checkApiCall();

        const expectedValues = { Source: "File" };

        for (const [labelText, expectedValue] of Object.entries(
            expectedValues
        )) {
            const label = await screen.findByText(labelText, {
                selector: "label"
            });
            const form = label.closest("form");
            expect(form).toBeInTheDocument();

            const valueElement = within(form!).getByText(expectedValue);
            expect(valueElement).toBeInTheDocument();
        }
    });

    it("Should display column headers and row data from mock data", async () => {
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName
        });
        await checkApiCall();

        const expectedHeaders = [
            "Column Name",
            "Data type",
            "Semantic Type",
            "Properties",
            "Preview"
        ];
        const tableHeader = await screen.findByTestId("table-header");
        const thElements = within(tableHeader).getAllByRole("columnheader");

        expectedHeaders.forEach((header, index) => {
            expect(thElements[index]).toHaveTextContent(header);
        });

        const tableData = mockAllTables.transaction_ltv1;
        const firstCol = tableData.cols[0];

        const expectedRowValues = [
            firstCol.name,
            firstCol.dtype,
            firstCol.stype,
            tableData.pkey === firstCol.name ? "Primary key" : "",
            ""
        ];

        const tableBody = document.querySelector(
            '[data-slot="table-body"]'
        ) as HTMLElement;
        expect(tableBody).toBeInTheDocument();

        const firstRow = within(tableBody).getAllByRole("row")[0];
        const cells = within(firstRow).getAllByRole("cell");

        expectedRowValues.forEach((expectedValue, index) => {
            expect(cells[index].textContent).toContain(expectedValue);
        });
    });

    it("Should show `Primary key` tag for primary key column", async () => {
        const tableName = "customer";
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName
        });

        const primaryKeyCell = screen.getByText("Primary key");
        expect(primaryKeyCell).toBeInTheDocument();
    });

    it("Should show `Time Column` tag for create date column", async () => {
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName
        });

        await checkApiCall();
        await waitFor(async () => {
            const timeColumnSelect = await screen.findByText("Time Column");
            expect(timeColumnSelect).toBeInTheDocument();
        });
    });

    it("Should not show connector label when `connector_id` is set empty", async () => {
        const partialTables = {
            [tableName]: {
                ...mockAllTables[tableName],
                source_table: {
                    ...mockAllTables[tableName].source_table,
                    data_source_type: "S3",
                    connector_id: ""
                }
            }
        };

        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            allTablesInGraph: partialTables as unknown as GraphTables
        });

        expect(screen.queryByLabelText("Connector")).not.toBeInTheDocument();
    });

    it("Should not show source table label when `source_table_name` is set empty", async () => {
        const partialTables = {
            [tableName]: {
                ...mockAllTables[tableName],
                source_table: {
                    ...mockAllTables[tableName].source_table,
                    data_source_type: "S3",
                    source: { mode: "relative", path: "" }
                }
            }
        };

        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            allTablesInGraph: partialTables as unknown as GraphTables
        });

        expect(screen.queryByLabelText("Source Table")).not.toBeInTheDocument();
    });

    it("Should render Preprocessing component when editable is true", async () => {
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true
        });

        expect(
            screen.getByText(
                /The edits made here will be saved and used as a new temp table without altering the original table./i
            )
        ).toBeInTheDocument();
        const button = screen.getByRole("button", { name: "Save" });
        expect(button).toBeInTheDocument();
    });

    it("Should call `callCreateGraph` when Save button is clicked", async () => {
        const callCreateGraph = vi.fn();
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true,
            callCreateGraph
        });

        await callSampleRowsAndGetColumns();

        const saveButton = await screen.findByText("Save");
        await userEvent.click(saveButton);

        expect(callCreateGraph).toHaveBeenCalled();
    });

    it("Should disable Save button when required fields are empty", async () => {
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true
        });

        const saveButton = await screen.findByText("Save");
        expect(saveButton).toHaveClass("disabled:pointer-events-none");
    });

    it("Should call `suggestLinks` when Save button is clicked", async () => {
        const suggestLinks = vi.fn();
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true,
            suggestLinks
        });

        await callSampleRowsAndGetColumns();

        const saveButton = await screen.findByText("Save");
        await userEvent.click(saveButton);

        await waitFor(() => {
            expect(suggestLinks).toHaveBeenCalled();
        });
    });

    it("Should call `updateTable` with updated table args on save", async () => {
        const updateTableMock = vi.fn();
        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true,
            updateTable: updateTableMock
        });

        await callSampleRowsAndGetColumns();

        const saveButton = await screen.findByText("Save");
        expect(saveButton).not.toBeDisabled();

        await userEvent.click(saveButton);

        expect(updateTableMock).toHaveBeenCalledWith(
            tableName,
            expect.objectContaining({
                cols: expect.arrayContaining([
                    expect.objectContaining({
                        name: expect.any(String),
                        dtype: expect.any(String),
                        stype: expect.any(String)
                    })
                ])
            })
        );
    });

    it("Should clear invalid pkey, time_col, and end_time_col when saving", async () => {
        const updateTableMock = vi.fn();

        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true,
            updateTable: updateTableMock,
            allTablesInGraph: {
                [tableName]: {
                    cols: [{ name: "valid_col1" }] as Column[],
                    pkey: "invalid_pkey",
                    time_col: "invalid_time_col",
                    end_time_col: "invalid_end_time_col",
                    source_table: mockAllTables[tableName].source_table
                }
            }
        });

        const saveButton = await screen.findByText("Save");
        expect(saveButton).not.toBeDisabled();

        await userEvent.click(saveButton);

        expect(updateTableMock).toHaveBeenCalledWith(
            tableName,
            expect.objectContaining({
                pkey: "",
                time_col: "",
                end_time_col: "",
                cols: [],
                source_table: expect.objectContaining({
                    source: { mode: "relative", path: "transaction" }
                })
            })
        );
    });

    it("Should display 'No data found' when column list is empty", async () => {
        const tableName = "customer";
        const emptyTables = {
            customer: { ...mockAllTables.customer, cols: [] }
        };

        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            allTablesInGraph: emptyTables
        });

        expect(screen.getByText(/No data found./i)).toBeInTheDocument();
    });

    it("Should display `Something went wrong` for `Invalid` tables list", async () => {
        await renderTableSliderOverview({
            ...defProps,
            allTablesInGraph: null as unknown as GraphTables
        });

        const alert = screen.queryByRole("alert");
        expect(alert).toBeInTheDocument();
        expect(alert).toHaveTextContent(/Something went wrong in/i);
        expect(alert).toHaveTextContent(/Please contact Kumo support/i);
    });

    it("Should call `suggestLinks` and `callCreateGraph` when updateTable is not provided", async () => {
        const suggestLinksMock = vi.fn();
        const callCreateGraphMock = vi.fn();

        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true,
            updateTable: undefined,
            suggestLinks: suggestLinksMock,
            callCreateGraph: callCreateGraphMock
        });

        await waitFor(async () => {
            expect(await screen.findByText("Save")).not.toBeDisabled();
        });

        await userEvent.click(screen.getByText("Save"));

        await waitFor(() => {
            expect(suggestLinksMock).toHaveBeenCalled();
            expect(callCreateGraphMock).toHaveBeenCalled();
        });
    });

    it("Should handle valid pkey, time_col, and end_time_col that exist in columns", async () => {
        const updateTableMock = vi.fn();
        const connectorId = "ltv_test";
        const sourceTable = "transaction";

        await renderTableSliderOverview({
            ...defProps,
            name: tableName,
            tableId: tableName,
            editable: true,
            updateTable: updateTableMock,
            allTablesInGraph: {
                [tableName]: {
                    cols: [
                        { name: "id", dtype: "string", stype: "categorical" },
                        {
                            name: "created_at",
                            dtype: "datetime",
                            stype: "datetime"
                        },
                        {
                            name: "updated_at",
                            dtype: "datetime",
                            stype: "datetime"
                        }
                    ] as Column[],
                    pkey: "id",
                    time_col: "created_at",
                    end_time_col: "updated_at",
                    source_table: mockAllTables[tableName].source_table
                }
            }
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/columns`,
                undefined,
                {
                    headers: undefined,
                    params: {
                        table_name: sourceTable
                    }
                }
            );
        });

        await waitFor(async () => {
            expect(await screen.findByText("Save")).not.toBeDisabled();
        });

        await userEvent.click(screen.getByText("Save"));

        await waitFor(() => {
            expect(updateTableMock).toHaveBeenCalledWith(
                tableName,
                expect.objectContaining({
                    pkey: expect.any(String),
                    cols: expect.any(Array)
                })
            );
        });
    });
});
