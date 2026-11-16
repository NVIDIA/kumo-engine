// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
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
import { setTableNameResponseType } from "./MockFunction/TableName.mock";
import { setValidateTableResponseType } from "./MockFunction/ValidateTable.mock";
import { mockNewEditTable } from "./MockResponse/NewTableResponse";
import { executeTableCreationFlow } from "./tableCreationHelpers";

describe("EditTablePage", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });

    const queryClient = createQueryClient();
    const renderEditTablePage = async () => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={[
                        "/project/project-test/tables/edit-table"
                    ]}>
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

        const headerText = "Edit Table";
        await waitFor(() => {
            const pageTitle = screen.getByText(headerText);
            expect(pageTitle).toBeInTheDocument();
            expect(
                screen.getByRole("button", { name: "Cancel" })
            ).toBeInTheDocument();
        });
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

    const setupEditTableTest = () => {
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
            flowtype: FlowType.EDIT,
            tableName: resolvedTableArgs?.name_alias,
            islocal: false
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

    const clickSaveTableButton = async () => {
        const saveTable = screen.getByRole("button", { name: "Save" });
        await userEvent.click(saveTable);
    };

    const createTableApiCallTest = async (
        args: CreateTableArgs,
        options: {
            validateWithFailure?: boolean;
            validateWithInvalid?: boolean;
            createWithFailure?: boolean;
            createWithInvalid?: boolean;
            newTableName?: string;
        }
    ) => {
        await executeTableCreationFlow(args, clickSaveTableButton, {
            ...options,
            forceRename: true
        });
    };

    beforeEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
        mockNavigateCall();
        queryClient.clear();
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");

        setDeploymentTypeInStore(deploymentType.SAAS);
        setTableNameResponseType(testResponseType.ALL);
        setValidateTableResponseType(testResponseType.ALL);
        setCreateTableResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
    });

    describe("Edit Table with property selection", () => {
        beforeEach(async () => {
            const { connectorId, resolvedSourceTableName } =
                setupEditTableTest();

            await renderEditTablePage();
            await pageLoadTest();

            await callSampleRowsAndGetColumns(connectorId, {
                name: resolvedSourceTableName
            });

            const tableNameInput = screen.getByPlaceholderText("Use default");
            await waitFor(() => {
                expect(tableNameInput).toHaveValue("transaction_ltv1");
                expect(tableNameInput).toHaveAttribute("readonly");
            });

            //Property selection
            const propertyDropdown = screen.getByTestId(
                "testid-properties-select-4"
            );
            expect(propertyDropdown).toHaveTextContent("Select");
            await userEvent.click(propertyDropdown);
            const optionToSelect = await screen.findByText("Create date");
            expect(optionToSelect).toBeInTheDocument();
            await userEvent.click(optionToSelect);
            await waitFor(() => {
                const updatedDropdown = screen.getByTestId(
                    "testid-properties-select-4"
                );
                expect(updatedDropdown).toHaveTextContent("Create date");
            });
        });

        it("Should successfully edit table with column and data type changes", async () => {
            //Checkbox deselect
            const checkboxes = await screen.findAllByRole("checkbox");
            expect(checkboxes.length).toBeGreaterThanOrEqual(1);
            const targetCheckbox = checkboxes[7]; //CustomerID row
            expect(targetCheckbox).toBeInTheDocument();
            await userEvent.click(targetCheckbox!);
            await waitFor(() => {
                const updatedCheckbox = screen.getAllByRole("checkbox")[7];
                expect(updatedCheckbox).toHaveAttribute(
                    "data-state",
                    "unchecked"
                );
            });

            //Data Type change
            const dataTypeDropdowns = screen.getAllByTestId(
                "testid-select-d-type"
            );
            expect(dataTypeDropdowns.length).toBeGreaterThan(0);
            const dataTypeDropdown = dataTypeDropdowns[5];
            expect(dataTypeDropdown).toHaveTextContent("Float64");
            await userEvent.click(dataTypeDropdown);
            const newDataTypeOption = await screen.findByText("Float32");
            expect(newDataTypeOption).toBeInTheDocument();
            await userEvent.click(newDataTypeOption);
            await waitFor(() => {
                const updatedDataTypeDropdown = screen.getAllByTestId(
                    "testid-select-d-type"
                )[5];
                expect(updatedDataTypeDropdown).toHaveTextContent("Float32");
            });

            await createTableApiCallTest(
                mockNewEditTable as CreateTableArgs,
                {}
            );
        });
    });
});
