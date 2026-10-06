// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { setDeploymentTypeInStore } from "stores/__tests__/debugInfoStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockLocationStateValue, mockNavigateCall } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient } from "tests/testUtil";

import ConnectTables from "components/Tables/Connect/ConnectTables";

import { expectedTableDetailsResponse } from "pages/Tables/TablesList/__tests__/MockFunction/TableDetail.test.mock";

import { CreateTableArgs, FlowType } from "types/tables";

import { storageKeys } from "utils/Constants";
import {
    getConnectorIDFromTableSource,
    getSourceNameFromTableSource
} from "utils/Converters/tablesConverter";

import { setCreateTableResponseType } from "./MockFunction/CreateTable.mock";
import { setSampleRowsResponseType } from "./MockFunction/SampleRowsResponse.test.mock";
import { setSourceTablesResponsesType } from "./MockFunction/SourceTables.mock";
import { setValidateTableResponseType } from "./MockFunction/ValidateTable.mock";
import { mockNewDuplicateTable } from "./MockResponse/NewTableResponse";
import { executeTableCreationFlow } from "./tableCreationHelpers";

describe("DuplicateTablePage", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });

    const queryClient = createQueryClient();
    const renderDuplicateTablePage = async () => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={["/project/project-test/tables/new-table"]}>
                    <QueryClientProvider client={queryClient}>
                        <ConnectTables />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const pageLoadTest = async () => {
        await waitFor(() => {
            const APIS_TO_CALL = ["v0/projects/project-tests/connectors/"];
            APIS_TO_CALL.forEach((API) => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(`${API}`);
            });
        });

        const headerText = "Add New Table";
        await waitFor(() => {
            const pageTitle = screen.getByText(headerText);
            expect(pageTitle).toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: "Cancel" })
            ).toBeInTheDocument();
        });
    };

    const checkNameValidationApi = async (name: string) => {
        await waitFor(
            () => {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    `v1/tables/${name}/validate_name`,
                    undefined,
                    {
                        headers: undefined,
                        params: { project_id: "project-tests" }
                    }
                );
            },
            { timeout: 2000 }
        );
    };

    const callSampleRowsAndGetColumns = async (
        connectorId: string,
        sourceTableToSelect: { name?: string } | null
    ) => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/columns`,
                undefined,
                {
                    headers: undefined,
                    params: {
                        table_name: sourceTableToSelect?.name
                    }
                }
            );
        });

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/sample`,
                undefined,
                {
                    headers: undefined,
                    params: {
                        sample_rows: 10,
                        table_name: sourceTableToSelect?.name
                    }
                }
            );
        });
    };

    const clickAddTableButton = async () => {
        await waitFor(() => {
            const addTable = screen.getByRole("button", {
                name: "Add"
            });
            expect(addTable).toBeEnabled();
        });
        const addTable = screen.getByRole("button", { name: "Add" });
        await userEvent.click(addTable);
    };

    const createDuplicateApiCallTest = async (
        args: CreateTableArgs,
        options: {
            validateWithFailure?: boolean;
            validateWithInvalid?: boolean;
            createWithFailure?: boolean;
            createWithInvalid?: boolean;
            newTableName?: string;
        }
    ) => {
        await executeTableCreationFlow(args, clickAddTableButton, {
            ...options,
            newTableName: options.newTableName || "transaction_ltv1_copy",
            forceRename: false // Duplicate table uses force_rename: false
        });
    };

    const setupDuplicateTableTest = () => {
        const resolvedTableArgs = structuredClone(
            expectedTableDetailsResponse()
        );

        const TableInfo: CreateTableArgs = {
            cols: resolvedTableArgs?.table?.cols,
            source_table: resolvedTableArgs?.table?.source_table,
            pkey: resolvedTableArgs?.table?.pkey ?? "",
            time_col: resolvedTableArgs?.table?.time_col ?? "",
            end_time_col: resolvedTableArgs?.table?.end_time_col ?? ""
        };

        mockLocationStateValue({
            createtableargs: TableInfo,
            flowtype: FlowType.DUPLICATE,
            tableName: resolvedTableArgs?.name_alias + "_copy"
        });

        const connectorId = getConnectorIDFromTableSource(
            resolvedTableArgs?.table?.source_table
        );
        const resolvedSourceTableName = getSourceNameFromTableSource(
            resolvedTableArgs?.table?.source_table
        );

        return {
            resolvedTableArgs,
            TableInfo,
            connectorId,
            resolvedSourceTableName
        };
    };

    beforeEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
        setDeploymentTypeInStore(deploymentType.SAAS);
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");
        mockNavigateCall();
        queryClient.clear();

        // Reset mock response types to default state
        setSampleRowsResponseType(testResponseType.ALL);
        setSourceTablesResponsesType(testResponseType.ALL);
        setValidateTableResponseType(testResponseType.ALL);
        setCreateTableResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
    });

    describe("Duplicate table with property selection", () => {
        beforeEach(async () => {
            const { resolvedTableArgs, connectorId, resolvedSourceTableName } =
                setupDuplicateTableTest();

            await renderDuplicateTablePage();
            await pageLoadTest();

            const tableName = resolvedTableArgs?.name_alias + "_copy";
            await checkNameValidationApi(tableName);

            await callSampleRowsAndGetColumns(connectorId, {
                name: resolvedSourceTableName
            });

            //Property selection
            const propertyDropdown = screen.getByTestId(
                "testid-properties-select-4"
            );
            expect(propertyDropdown).toHaveTextContent("Select");
            await userEvent.click(propertyDropdown);
            await waitFor(() => {
                expect(screen.getByRole("listbox")).toBeInTheDocument();
            });
            const optionToSelect = await screen.findByText("Create date");
            await userEvent.click(optionToSelect);

            await waitFor(() => {
                const updatedDropdown = screen.getByTestId(
                    "testid-properties-select-4"
                );
                expect(updatedDropdown).toHaveTextContent("Create date");
            });
        });

        it("Should successfully duplicate table with all columns selected", async () => {
            //All Checkbox selection
            const checkboxes = screen.getAllByRole("checkbox");
            expect(checkboxes.length).toBeGreaterThan(0);
            checkboxes.forEach((checkbox) => {
                expect(checkbox).toBeChecked();
            });

            await createDuplicateApiCallTest(
                mockNewDuplicateTable as CreateTableArgs,
                {}
            );
        });
    });

    it("Should pre-populate table name in `S3` config for Duplicate entity and handle table name change", async () => {
        const { connectorId, resolvedSourceTableName } =
            setupDuplicateTableTest();

        await renderDuplicateTablePage();
        await pageLoadTest();

        await callSampleRowsAndGetColumns(connectorId, {
            name: resolvedSourceTableName
        });

        const tableNameInput = screen.getByPlaceholderText("Use default");
        expect(tableNameInput).toHaveValue("transaction_ltv1_copy");

        //Change name
        await userEvent.clear(tableNameInput);
        const newTableName = "test_table";
        fireEvent.change(tableNameInput, { target: { value: newTableName } });
        await checkNameValidationApi(newTableName);
        expect(tableNameInput).toHaveValue(newTableName);

        await waitFor(() => {
            const addButton = screen.getByRole("button", {
                name: "Add"
            });
            expect(addButton).toBeEnabled();
        });
    });
});
