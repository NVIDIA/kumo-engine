// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import {
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import {
    allTables,
    graphInfo,
    groups,
    mockTableIds,
    tables
} from "tests/fixtures/MockGraphInformationPanel";
import {
    clickEditIcon,
    getEditIcon,
    verifyElementDoesNotExist,
    verifyElementExists,
    verifyTextDoesNotExist,
    verifyTextExists
} from "tests/testHelpers";
import { createQueryClient } from "tests/testUtil";

import { FlowType } from "types/tables";

import GraphInformationPanel, {
    GraphInformationPanelProps
} from "../GraphInformationPanel";

const defaultProps: GraphInformationPanelProps = {
    graphInfo: {
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
    tables: [],
    weakGraphEdges: [],
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
    tableIds: {},
    callCreateGraph: vi.fn(),
    suggestLinks: vi.fn()
};

describe("GraphInformationPanel", () => {
    const renderGraphInformationPanel = async (
        props: GraphInformationPanelProps
    ) => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <GraphInformationPanel {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const verifyControlButtons = () => {
        verifyElementExists(screen.getByLabelText(/Zoom In/i));
        verifyElementExists(screen.getByLabelText(/Zoom Out/i));
        verifyElementExists(screen.getByLabelText(/Fit View/i));
        verifyElementExists(screen.getByLabelText(/Toggle Interactivity/i));
    };

    it("Should render React Flow container and control buttons correctly", async () => {
        await renderGraphInformationPanel(defaultProps);

        verifyControlButtons();

        verifyElementExists(screen.getByTestId("rf__wrapper"));
        verifyElementExists(screen.getByTestId("rf__background"));
    });

    it("Should render all table nodes", async () => {
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo,
            tables
        });

        tables.forEach((table) => {
            verifyElementExists(screen.getByTestId(`rf__node-${table.id}`));
        });
    });

    it("Should render all table names with pkey and fkey", async () => {
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo,
            tables
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
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo,
            tables
        });

        const tableNames = tables.map((table) => table.name_alias as string);

        for (const name of tableNames) {
            const icon = await screen.findByTestId(`${name}_show_icon`);
            verifyElementExists(icon);
        }
    });

    it("Should show edit controls when enableEditTable is true", async () => {
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo,
            tables,
            enableEditTable: true
        });

        verifyElementExists(getEditIcon("transaction"));
    });

    it("Should not show edit controls when enableEditTable is false", async () => {
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo,
            tables,
            enableEditTable: false
        });

        verifyElementDoesNotExist(getEditIcon("transaction"));
    });

    it("Should handle click on edit icon", async () => {
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo,
            tables,
            enableEditTable: true
        });

        clickEditIcon("transaction");

        verifyTextExists(
            "The edits made here will be saved and used as a new temp table without altering the original table."
        );
    });

    it("Should navigate back to Graph View from Table View when 'Back` is clicked", async () => {
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo,
            tables,
            allTables,
            groups,
            tableIds: mockTableIds
        });

        const tableName = tables[0].name_alias as string;
        const showIcon = screen.getByTestId(`${tableName}_show_icon`);
        fireEvent.click(showIcon);

        await waitFor(() => {
            verifyTextExists(tableName);
        });
        await verifyTextExists(/Back/i);
        await verifyTextExists(/Column Statistics/i);

        fireEvent.click(screen.getByText(/Back/i));
        await waitFor(() => {
            verifyTextDoesNotExist(/Column Statistics/i);
        });
    });

    it("Should navigate back to Graph View from Table edit", async () => {
        vi.useFakeTimers({ shouldAdvanceTime: true });
        const setAllTables = vi.fn();
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo,
            tables,
            enableEditTable: true,
            allTables,
            groups,
            tableIds: mockTableIds,
            setAllTables
        });

        clickEditIcon("transaction");

        verifyTextExists(
            "The edits made here will be saved and used as a new temp table without altering the original table."
        );

        const saveChanges = await screen.findByRole("button", {
            name: "Save"
        });
        fireEvent.click(saveChanges);

        await act(async () => {
            vi.advanceTimersByTime(1200);
        });

        await waitFor(() => {
            verifyTextDoesNotExist(
                /The edits made here will be saved and used as a new temp table without altering the original table./i
            );
        });
        vi.useRealTimers();
    });

    it("Should render with empty tables array", async () => {
        await renderGraphInformationPanel({
            ...defaultProps,
            graphInfo: { ...graphInfo, tables: {} },
            tables: []
        });

        verifyElementDoesNotExist(screen.queryByTestId(/rf__node-/));
    });
});
