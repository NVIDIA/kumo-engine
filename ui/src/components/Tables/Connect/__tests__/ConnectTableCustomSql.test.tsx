// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import {
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KumoapiTableTableDefinition } from "gen/v1/types";
import { setDeploymentTypeInStore } from "stores/__tests__/debugInfoStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { storageKeys } from "utils/Constants";

// Enable the Custom SQL feature flag for this suite so the "Custom SQL" source
// option is available.
vi.mock("utils/Constants", async (importOriginal) => {
    const actual = await importOriginal<typeof import("utils/Constants")>();
    return { ...actual, ENABLE_CUSTOM_SQL_TABLES: true };
});

// This suite covers the *parent* wiring — Add gating + create — not the editor.
// Stand in for the authoring surface with a control that flips the lifted
// `validated`/`sqlText`/columns exactly as a successful Validate would, so the
// flow is deterministic without driving CodeMirror. The surface's own
// discovery/validate/preview behavior is covered by CustomSqlTableSource.test.
const { MOCK_SQL, MOCK_COLS } = vi.hoisted(() => ({
    MOCK_SQL: "SELECT id, amount FROM orders",
    MOCK_COLS: [{ name: "id", dtype: "int64", stype: "ID" }]
}));

vi.mock("../CustomSqlTableSource", () => ({
    default: ({
        connectorId,
        validated,
        setSqlText,
        setValidated,
        setTableArgs
    }: {
        connectorId: string;
        validated: boolean;
        setSqlText: (v: string) => void;
        setValidated: (v: boolean) => void;
        setTableArgs: (
            updater: (
                args: KumoapiTableTableDefinition
            ) => KumoapiTableTableDefinition
        ) => void;
    }) => (
        <div data-testid="mock-custom-sql">
            <span data-testid="mock-connector-id">{connectorId}</span>
            <span data-testid="mock-validated">{String(validated)}</span>
            <button
                type="button"
                onClick={() => {
                    setSqlText(MOCK_SQL);
                    setTableArgs((args) => ({
                        ...args,
                        cols: MOCK_COLS as KumoapiTableTableDefinition["cols"],
                        pkey: "id"
                    }));
                    setValidated(true);
                }}>
                mock-validate
            </button>
        </div>
    )
}));

import { setConnectorListResponseType } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";

import ConnectTables from "../ConnectTables";

import { setColumnResponseType } from "./MockFunction/ColumnResponse.test.mock";
import { setCreateTableResponseType } from "./MockFunction/CreateTable.mock";
import { setSampleRowsResponseType } from "./MockFunction/SampleRowsResponse.test.mock";
import { setSourceTablesResponsesType } from "./MockFunction/SourceTables.mock";
import { setTableNameResponseType } from "./MockFunction/TableName.mock";

const PROJECT_ID = "project-tests";
const TABLE_NAME = "my_sql_table";

describe("ConnectTables - Custom SQL create wiring (flag on)", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });
    const queryClient = createQueryClient();

    const renderPage = async () => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        `/project/${PROJECT_ID}/tables/new-table`
                    ]}>
                    <QueryClientProvider client={queryClient}>
                        <ConnectTables />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    // Select the Custom SQL source and wait for the warehouse connector to be
    // auto-selected (Custom SQL restricts the list to SQL warehouses).
    const selectCustomSqlSource = async () => {
        const sourceDropdown = await screen.findByTestId(
            "testid-select-source"
        );
        await userEvent.click(sourceDropdown);
        const menu = await screen.findByTestId("listbox");
        await userEvent.click(await within(menu).findByText("Custom SQL"));
        expect(screen.getByTestId("testid-select-source")).toHaveTextContent(
            "Custom SQL"
        );
        await waitFor(() =>
            expect(
                screen.getByTestId("mock-connector-id").textContent
            ).toBeTruthy()
        );
    };

    const enterTableName = async (name: string) => {
        const nameInput = await screen.findByPlaceholderText("Use default");
        fireEvent.change(nameInput, { target: { value: name } });
        // The name-uniqueness check gates Add while it runs; wait for it to fire
        // (and settle) so the create path isn't blocked as "validating".
        await waitFor(() =>
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${name}/validate_name`,
                undefined,
                { headers: undefined, params: { project_id: PROJECT_ID } }
            )
        );
    };

    const clickValidate = async () => {
        await act(async () => {
            await userEvent.click(screen.getByText("mock-validate"));
        });
    };

    const findCreateCall = () =>
        mockAxiosInstance.post.mock.calls.find(
            (call) =>
                call[0] === "v1/tables/" &&
                call[2]?.params?.name_alias !== undefined
        );

    beforeEach(() => {
        queryClient.clear();
        localStorage.clear();
        // `useProjectId` falls back to the stored project (no `:projectId` route
        // param is mounted here); re-seed it after the clear.
        storeToLocalStorage(storageKeys.selectedProject, PROJECT_ID);
        vi.clearAllMocks();
        mockNavigateCall();
        setDeploymentTypeInStore(deploymentType.SAAS);
        setConnectorListResponseType(testResponseType.ALL);
        setSourceTablesResponsesType(testResponseType.ALL);
        setColumnResponseType(testResponseType.ALL);
        setSampleRowsResponseType(testResponseType.ALL);
        setTableNameResponseType(testResponseType.ALL);
        setCreateTableResponseType(testResponseType.ALL);
    });

    it("Should render the Custom SQL authoring surface", async () => {
        await renderPage();
        await selectCustomSqlSource();
        expect(
            await screen.findByTestId("mock-custom-sql")
        ).toBeInTheDocument();
    });

    it("Should keep Add disabled until the SQL definition is validated", async () => {
        await renderPage();
        await selectCustomSqlSource();
        await enterTableName(TABLE_NAME);

        // Named + connector selected, but not validated → Add stays disabled.
        const addButton = screen.getByRole("button", { name: "Add" });
        await waitFor(() => expect(addButton).toBeDisabled());

        await clickValidate();

        // Validation confirms the schema, so Add is now enabled.
        await waitFor(() => expect(addButton).toBeEnabled());
    });

    it("Should create the SQL table on Add once validated", async () => {
        await renderPage();
        await selectCustomSqlSource();
        await enterTableName(TABLE_NAME);
        await clickValidate();

        const connectorId = screen.getByTestId("mock-connector-id").textContent;
        expect(connectorId).toBeTruthy();

        const addButton = screen.getByRole("button", { name: "Add" });
        await waitFor(() => expect(addButton).toBeEnabled());
        await act(async () => {
            await userEvent.click(addButton);
        });

        // Add fires the project-scoped create with a CustomSQLSourceTable
        // definition (connector + defining SELECT + declared columns).
        await waitFor(() => expect(findCreateCall()).toBeDefined());
        const [, body, options] = findCreateCall()!;
        expect(body.source_table).toEqual(
            expect.objectContaining({
                data_source_type: "CUSTOM_SQL",
                connector_id: connectorId,
                sql: MOCK_SQL,
                table: TABLE_NAME
            })
        );
        // `connector_type` and `referenced_tables` are server-owned: the server
        // derives them from the connector and the SQL, so the client must not
        // send them.
        expect(body.source_table).not.toHaveProperty("connector_type");
        expect(body.source_table).not.toHaveProperty("referenced_tables");
        expect(body.cols).toEqual([
            expect.objectContaining({
                name: "id",
                dtype: "int64",
                stype: "ID"
            })
        ]);
        expect(options.params).toEqual(
            expect.objectContaining({
                project_id: PROJECT_ID,
                name_alias: TABLE_NAME,
                force_rename: false
            })
        );

        // The SQL create is validated server-side, so no front-end
        // physical-table validate fires.
        const postUrls = mockAxiosInstance.post.mock.calls.map((c) => c[0]);
        expect(postUrls).not.toContain("v1/tables/validate");

        await waitFor(() =>
            expect(mockUseNavigate).toHaveBeenCalledWith(
                `/project/${PROJECT_ID}/tables/${TABLE_NAME}`
            )
        );
    });

    it("Should surface a create-time validation error rather than navigating", async () => {
        // A 422 stands in for the server-side SQL re-validation the create path
        // runs; the backend message surfaces and the flow does not navigate.
        setCreateTableResponseType(testResponseType.INVALID);
        await renderPage();
        await selectCustomSqlSource();
        await enterTableName(TABLE_NAME);
        await clickValidate();

        const addButton = screen.getByRole("button", { name: "Add" });
        await waitFor(() => expect(addButton).toBeEnabled());
        await act(async () => {
            await userEvent.click(addButton);
        });

        expect(
            await screen.findByText(
                "Failed to create table!",
                {},
                { timeout: 5000 }
            )
        ).toBeInTheDocument();
        expect(screen.getByText("Error in Creating Table")).toBeInTheDocument();
        expect(mockUseNavigate).not.toHaveBeenCalledWith(
            `/project/${PROJECT_ID}/tables/${TABLE_NAME}`
        );
    });
});
