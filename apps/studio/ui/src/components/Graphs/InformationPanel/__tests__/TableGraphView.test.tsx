// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import {
    cleanup,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import {
    allTables,
    graphInfo,
    groups,
    invalidGraphInfo,
    tables
} from "tests/fixtures/MockGraphInformationPanel";
import { mockAxiosInstance } from "tests/mockAxios";
import {
    clickEditIcon,
    findAndClickButton,
    getEditIcon,
    getSkeletonElement,
    verifyControlButtons,
    verifyElementDoesNotExist,
    verifyElementExists
} from "tests/testHelpers";
import { createQueryClient } from "tests/testUtil";

import { GraphTables, JSGraph } from "types/graphs";
import { DataSourceType, FlowType, JSTable, TableDetails } from "types/tables";

import { storageKeys } from "utils/Constants";

import TableGraphView, { TableGraphViewProps } from "../TableGraphView";

const defaultProps: TableGraphViewProps = {
    graphData: {
        id: "",
        name: "",
        totalLinks: 0,
        totalTables: 0,
        tables: {},
        groups: [],
        created: "",
        updated: "",
        isActive: true,
        table_name_to_fpid_mapping: {}
    },
    setSelectedTable: vi.fn(),
    tableData: [],
    weakEdges: [],
    height: 600,
    width: 800,
    open: true,
    enableEditTable: false,
    gfc: {
        name: "",
        id: "",
        description: "",
        tables: {},
        selected_tables: [],
        selected_table_ids: [],
        flow_type: FlowType.CREATE
    },
    setGfc: vi.fn(),
    allTables: {},
    setAllTables: vi.fn(),
    groups: [],
    setGroups: vi.fn(),
    callCreateGraph: vi.fn(),
    suggestLinks: vi.fn()
};

describe("TableGraphView", () => {
    const queryClient = createQueryClient();
    const renderTableGraphView = async (props: TableGraphViewProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <TableGraphView {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const waitForEditDialogText = async () => {
        await waitFor(async () => {
            verifyElementExists(
                await screen.findByText(
                    "The edits made here will be saved and used as a new temp table without altering the original table."
                )
            );
        });
    };

    const clickSaveChangesButton = async () => {
        await findAndClickButton("Save");
    };

    const verifyReactFlowElements = () => {
        verifyElementExists(screen.getByTestId("rf__wrapper"));
        verifyElementExists(screen.getByTestId("rf__background"));
        verifyElementExists(screen.getByTestId("rf__controls"));
        verifyElementExists(screen.getByTestId("rf__minimap"));
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
        cleanup();
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");
    });

    it("Should render the initial graph UI correctly", async () => {
        await renderTableGraphView(defaultProps);

        verifyReactFlowElements();
        verifyControlButtons();
        verifyElementExists(getSkeletonElement());
    });

    it("Should not render graph info when open is false", async () => {
        await renderTableGraphView({
            ...defaultProps,
            open: false
        });

        verifyElementDoesNotExist(screen.queryByTestId("rf__wrapper"));
        verifyElementDoesNotExist(getSkeletonElement());
    });

    it("Should render all table nodes", async () => {
        await renderTableGraphView({
            ...defaultProps,
            tableData: tables
        });

        tables.forEach((table) => {
            verifyElementExists(screen.getByTestId(`rf__node-${table.id}`));
        });
    });

    it("Should render all table names with pkey and fkey", async () => {
        await renderTableGraphView({
            ...defaultProps,
            graphData: graphInfo,
            tableData: tables
        });

        const tableNames = tables.map((table) => table.name_alias as string);
        const foreignKeys = ["CustomerID", "StockCode"];

        tableNames.forEach((name) => {
            const node = screen.getByTestId(`rf__node-${name}`);
            verifyElementExists(node);
            verifyElementExists(within(node).getByText(name));

            const pkey = graphInfo.tables?.[name]?.pkey;
            if (pkey) {
                verifyElementExists(within(node).getByText(`PK: ${pkey}`));
            }
        });

        foreignKeys.forEach((fk) => {
            verifyElementExists(screen.getByText(`FK: ${fk}`));
        });
    });

    it("Should render show icon for each table", async () => {
        await renderTableGraphView({
            ...defaultProps,
            graphData: graphInfo,
            tableData: tables
        });

        const tableNames = tables.map((table) => table.name_alias as string);

        for (const name of tableNames) {
            const icon = await screen.findByTestId(`${name}_show_icon`);
            verifyElementExists(icon);
        }
    });

    it("Should show edit controls when enableEditTable is true", async () => {
        await renderTableGraphView({
            ...defaultProps,
            tableData: tables,
            enableEditTable: true
        });

        verifyElementExists(getEditIcon("transaction"));
    });

    it("Should not show edit controls when enableEditTable is false", async () => {
        await renderTableGraphView({
            ...defaultProps,
            tableData: tables,
            enableEditTable: false
        });

        verifyElementDoesNotExist(getEditIcon("transaction"));
    });

    it("Should update graph state after editing a table and clicking Save button", async () => {
        const setGfc = vi.fn();
        const setAllTables = vi.fn();
        const setGroups = vi.fn();

        await renderTableGraphView({
            ...defaultProps,
            graphData: graphInfo,
            tableData: tables,
            enableEditTable: true,
            allTables,
            groups,
            gfc: {
                name: graphInfo.name ?? "",
                id: graphInfo.id ?? "",
                description: "",
                tables: graphInfo.tables ?? {},
                selected_tables: ["stock", "transaction", "customer"],
                selected_table_ids: [],
                flow_type: FlowType.CREATE
            },
            setGfc,
            setAllTables,
            setGroups
        });

        await clickEditIcon("transaction");
        await waitForEditDialogText();
        await clickSaveChangesButton();

        await waitFor(() => {
            expect(setGfc).toHaveBeenCalled();
            expect(setAllTables).toHaveBeenCalled();
            expect(setGroups).toHaveBeenCalled();
        });
    });

    it("Should clear primary, time, and end time columns when no columns are selected in a table", async () => {
        const setGfc = vi.fn();
        const setAllTables = vi.fn();
        const setGroups = vi.fn();
        const tableName = "transaction";

        const allTablesWithEmptyCols: GraphTables = {
            ...allTables,
            [tableName]: {
                ...allTables[tableName],
                cols: [],
                pkey: "SomeOldPkey",
                time_col: "SomeOldTimeCol",
                end_time_col: "SomeOldEndTimeCol"
            }
        };

        const graphData: JSGraph = {
            ...graphInfo,
            tables: {
                ...graphInfo.tables,
                [tableName]: {
                    ...(graphInfo.tables?.[tableName] as TableDetails),
                    cols: [],
                    pkey: "SomePkey",
                    time_col: "SomeTimeCol",
                    end_time_col: "SomedEndTimeCol"
                }
            }
        };

        await renderTableGraphView({
            ...defaultProps,
            graphData,
            tableData: tables,
            allTables: allTablesWithEmptyCols,
            enableEditTable: true,
            gfc: {
                name: "",
                id: "",
                description: "",
                tables: {
                    transaction: {
                        cols: [],
                        pkey: "SomeOldPkey",
                        time_col: "SomeOldTimeCol",
                        end_time_col: "SomeOldEndTimeCol",
                        source_table: {
                            connector_id: "connector_missing",
                            file_type: "PARQUET",
                            data_source_type: DataSourceType.S3,
                            source: { mode: "relative", path: "" }
                        }
                    }
                },
                selected_tables: [],
                selected_table_ids: [],
                flow_type: FlowType.CREATE
            },
            setGfc,
            setAllTables,
            setGroups
        });

        await clickEditIcon(tableName);

        await findAndClickButton("Save");

        await waitFor(() => {
            expect(setAllTables).toHaveBeenCalledWith(
                expect.objectContaining({
                    [tableName]: expect.objectContaining({
                        cols: [],
                        pkey: "",
                        time_col: "",
                        end_time_col: ""
                    })
                })
            );
            expect(setGfc).toHaveBeenCalled();
            expect(setGroups).toHaveBeenCalled();
        });
    });

    it("Should reset invalid pkey, time_col, and end_time_col to empty strings when not found in column list", async () => {
        const setGfc = vi.fn();
        const setAllTables = vi.fn();
        const setGroups = vi.fn();
        const tableName = "transaction";

        await renderTableGraphView({
            ...defaultProps,
            graphData: invalidGraphInfo,
            tableData: tables,
            enableEditTable: true,
            allTables: invalidGraphInfo.tables as GraphTables,
            gfc: {
                name: invalidGraphInfo.name ?? "",
                id: invalidGraphInfo.id ?? "",
                description: "",
                tables: invalidGraphInfo.tables ?? {},
                selected_tables: ["stock", "transaction", "customer"],
                selected_table_ids: [],
                flow_type: FlowType.CREATE
            },
            setGfc,
            setAllTables,
            setGroups
        });

        await clickEditIcon(tableName);
        await waitForEditDialogText();
        await clickSaveChangesButton();

        await waitFor(() => {
            expect(setAllTables).toHaveBeenCalledWith(
                expect.objectContaining({
                    [tableName]: expect.objectContaining({
                        pkey: "",
                        time_col: "",
                        end_time_col: ""
                    })
                })
            );
            expect(setGfc).toHaveBeenCalled();
            expect(setGroups).toHaveBeenCalled();
        });
    });

    it("Should show table heading when clicking the edit icon", async () => {
        const tableName = "transaction";

        await renderTableGraphView({
            ...defaultProps,
            tableData: tables,
            enableEditTable: true
        });

        await clickEditIcon(tableName);

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${tableName}`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        const headings = [
            "Source",
            "Source table",
            "Column Name",
            "Data Type",
            "Semantic Type",
            "Properties",
            "Preview"
        ];

        await waitFor(async () => {
            for (const heading of headings) {
                verifyElementExists(await screen.findByText(heading));
            }
        });

        const tableText = "transaction";
        const connectorText = "ltv_test";

        const findButtonByTestIdAndText = (testId: string, text: string) => {
            return screen
                .getAllByTestId(testId)
                .find((el) => el.textContent?.includes(text));
        };

        const sourcetableSelect = findButtonByTestIdAndText(
            "testid-select-source",
            tableText
        );
        expect(sourcetableSelect).toBeDefined();
        expect(sourcetableSelect).toHaveTextContent(tableText);

        const connectorSelect = screen.getByTestId("testid-select-connector");
        expect(connectorSelect).toBeDefined();
        expect(connectorSelect).toHaveTextContent(connectorText);
    });

    it("Should display `Something went wrong` for `Invalid` table list", async () => {
        await renderTableGraphView({
            ...defaultProps,
            tableData: { name: false } as unknown as JSTable[]
        });

        const alert = screen.queryByRole("alert");
        verifyElementExists(alert);
        expect(alert).toHaveTextContent(/Something went wrong in/i);
        expect(alert).toHaveTextContent(/getNodesFromTableData/i);
        expect(alert).toHaveTextContent(/Please contact Kumo support/i);
    });
});
