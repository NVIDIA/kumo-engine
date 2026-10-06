// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { act } from "react";
import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import {
    fireEvent,
    render,
    screen,
    waitFor,
    within
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { setDeploymentTypeInStore } from "stores/__tests__/debugInfoStore";
import { useUserStore } from "stores/userStore";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import { testResponseType } from "tests/testResponseType";
import { deploymentType } from "tests/testType";
import { createQueryClient } from "tests/testUtil";

import {
    expectedConnectorListResponse,
    setConnectorListResponseType
} from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";
import { expectedTableDetailsResponse } from "pages/Tables/TablesList/__tests__/MockFunction/TableDetail.test.mock";
import { setTableListResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/TableList.test.mock";

import { CreateTableArgs } from "types/tables";

import { storageKeys } from "utils/Constants";
import { getConnectorIDFromTableSource } from "utils/Converters/tablesConverter";

import ConnectTables from "../ConnectTables";

import { setColumnResponseType } from "./MockFunction/ColumnResponse.test.mock";
import { setCreateTableResponseType } from "./MockFunction/CreateTable.mock";
import { expectedNewTableResponse } from "./MockFunction/NewTableResponse.mock";
import { setSampleRowsResponseType } from "./MockFunction/SampleRowsResponse.test.mock";
import {
    expectedSourceTablesResponse,
    setSourceTablesResponsesType
} from "./MockFunction/SourceTables.mock";
import { setTableNameResponseType } from "./MockFunction/TableName.mock";
import { setValidateTableResponseType } from "./MockFunction/ValidateTable.mock";
import { mockNewTable } from "./MockResponse/NewTableResponse";
import { executeTableCreationFlow } from "./tableCreationHelpers";

describe("ConnectTablesPage", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });

    const queryClient = createQueryClient();
    const renderConnectTablesPage = async (
        initialEntry:
            | string
            | {
                  pathname: string;
                  state?: Record<string, unknown>;
              } = "/project/project-test/tables/new-table"
    ) => {
        await act(async () => {
            render(
                <MemoryRouter initialEntries={[initialEntry]}>
                    <QueryClientProvider client={queryClient}>
                        <ConnectTables />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const pageLoadTest = async ({
        expectSourceTablesRequest = true
    }: { expectSourceTablesRequest?: boolean } = {}) => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v0/projects/project-tests/connectors/"
            );
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                "v1/tables/",
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );

            if (expectSourceTablesRequest) {
                expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                    "v0/projects/project-tests/connectors/ltv_test/source-tables"
                );
            }
        });

        const headerText = "Add New Table";

        await waitFor(() => {
            screen.getByText(headerText);
            screen.getByRole("button", { name: "Cancel" });
            screen.getByRole("button", { name: "Add" });
        });

        const sourceDropdown = screen.queryByText("Source");
        {
            await waitFor(() => {
                expect(sourceDropdown).not.toBeNull();
            });
        }
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
        ); //takes 1-2 seconds
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
    };

    const sourceTableSelection = async (tableIndex: number = 2) => {
        const tableDropdown = await screen.findByTestId(
            "testid-select-source-table"
        );
        await userEvent.click(tableDropdown);
        const source_table_menu = await screen.findByTestId("listbox");
        const expectedSourceTable = structuredClone(
            expectedSourceTablesResponse()
        );
        const sourceTables = expectedSourceTable.table_names.map((name) => ({
            name
        }));
        for (const option of sourceTables) {
            expect(
                await within(source_table_menu).findByText(option?.name)
            ).toBeInTheDocument();
        }
        const sourceTableToSelect = sourceTables[tableIndex];
        const sourceTableElement = await within(source_table_menu).findByText(
            sourceTableToSelect?.name
        );
        await userEvent.click(sourceTableElement);

        const inferenceText = screen.queryByText(
            "Inferring column types based on sample rows."
        );
        if (inferenceText) {
            expect(inferenceText).toBeInTheDocument();
            const skipButton = screen.getByText("Skip");
            await userEvent.click(skipButton);
            await waitFor(() => {
                expect(
                    document.querySelector(".login-anim-container")
                ).toBeNull();
            });
        }

        const resolvedConnector = structuredClone(
            expectedTableDetailsResponse()
        );
        const connectorId = getConnectorIDFromTableSource(
            resolvedConnector?.table?.source_table
        );
        await callSampleRowsAndGetColumns(connectorId, sourceTableToSelect);

        return sourceTableToSelect;
    };

    const closeDropdown = () => {
        document.body.style.pointerEvents = "";
        fireEvent.click(document);
    };

    const selectConnector = async (connectorIndex: number) => {
        const dropdown = await screen.findByTestId("testid-select-connector");
        await userEvent.click(dropdown);
        const connector_menu = await screen.findByTestId("listbox");
        const expectedConnectorlist = expectedConnectorListResponse();
        const optionToSelect = expectedConnectorlist[connectorIndex];
        const optionElement = await within(connector_menu).findByText(
            optionToSelect?.name as string
        );
        await userEvent.click(optionElement);
        await waitFor(() => {
            expect(
                screen.getByText(optionToSelect.name as string)
            ).toBeInTheDocument();
        });
        return optionToSelect;
    };

    const selectSourceType = async (
        sourceType: "Connector" | "Local Upload"
    ) => {
        const sourceDropdown = await screen.findByTestId(
            "testid-select-source"
        );
        await userEvent.click(sourceDropdown);
        const source_type_menu = await screen.findByTestId("listbox");
        const option = await within(source_type_menu).findByText(sourceType);
        await userEvent.click(option);
        expect(screen.getByTestId("testid-select-source")).toHaveTextContent(
            sourceType
        );
    };

    const selectPropertyForColumn = async (
        columnIndex: number,
        propertyName: string
    ) => {
        const propertiesDropdowns = screen.getAllByTestId(
            /^testid-properties-select-\d+$/
        );
        await userEvent.click(propertiesDropdowns[columnIndex]);
        await waitFor(() => {
            screen.getByRole("listbox");
        });
        const propertyToSelect = await screen.findByText(propertyName);
        await userEvent.click(propertyToSelect);
        await waitFor(() => {
            const updatedDropdown = screen.getByTestId(
                `testid-properties-select-${columnIndex}`
            );
            expect(updatedDropdown).toHaveTextContent(propertyName);
        });
    };

    const changeSTypeForColumn = async (
        columnIndex: number,
        newSType: string
    ) => {
        const sTypeDropdowns = screen.getAllByTestId("testid-select-s-type");
        const sTypeDropdown = sTypeDropdowns[columnIndex];
        await userEvent.click(sTypeDropdown);
        const newSTypeOption = await screen.findByText(newSType);
        expect(newSTypeOption).toBeInTheDocument();
        await userEvent.click(newSTypeOption);
    };

    const uncheckColumnCheckbox = async (checkboxIndex: number) => {
        const checkBoxes = screen.getAllByRole("checkbox");
        checkBoxes.forEach((checkBox) => {
            expect(checkBox).toBeChecked();
        });
        await userEvent.click(checkBoxes[checkboxIndex]);
        expect(
            screen.getAllByRole("checkbox")[checkboxIndex]
        ).not.toBeChecked();
    };

    const verifyPropertyErrorAlert = async (
        _errorType: "Create Date" | "End Date" | "Primary Key",
        expectedText: string | RegExp
    ) => {
        await waitFor(
            async () => {
                const alert = await screen.findAllByRole("alert");
                expect(alert[0]).toHaveTextContent(expectedText);
            },
            { timeout: 3000 }
        );
    };

    const clearPropertyError = async (columnIndex: number) => {
        await userEvent.click(
            screen.getAllByTestId(/^testid-properties-select-\d+$/)[columnIndex]
        );
        const property = await screen.findByText("Primary Key");
        await userEvent.click(property);
        await waitFor(
            () => {
                expect(screen.queryByRole("alert")).not.toBeInTheDocument();
            },
            { timeout: 2000 }
        );
    };

    const setupConnectTablePage = async () => {
        await renderConnectTablesPage();
        await pageLoadTest();
    };

    const renameTable = async (tableName: string) => {
        const nameInput = screen.getByPlaceholderText("Use default");
        await userEvent.clear(nameInput);
        fireEvent.change(nameInput, { target: { value: tableName } });
        await checkNameValidationApi(tableName);
    };

    const clickAddTableButton = async () => {
        const addTable = screen.getByText("Add");
        await userEvent.click(addTable);
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
        await executeTableCreationFlow(args, clickAddTableButton, {
            ...options,
            newTableName: options.newTableName || "test_table",
            forceRename: false // Connect table uses force_rename: false
        });
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
        vi.resetAllMocks();
        mockNavigateCall();
        sessionStorage.removeItem(storageKeys.isLocalUpload);
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");

        setDeploymentTypeInStore(deploymentType.SAAS);
        setConnectorListResponseType(testResponseType.ALL);
        setSourceTablesResponsesType(testResponseType.ALL);
        setColumnResponseType(testResponseType.ALL);
        setSampleRowsResponseType(testResponseType.ALL);
        setTableNameResponseType(testResponseType.ALL);
        setValidateTableResponseType(testResponseType.ALL);
        focusManager.setFocused(false);
        setCreateTableResponseType(testResponseType.ALL);
    });

    afterEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
        focusManager.setFocused(undefined);
    });

    it("Should load the Add New Table page and verify API calls", async () => {
        await setupConnectTablePage();
    });

    it("groups the form into sections and holds Columns open before a source table is picked", async () => {
        await setupConnectTablePage();

        for (const section of ["Source", "Table details", "Columns"]) {
            expect(
                screen.getByRole("heading", { name: section })
            ).toBeInTheDocument();
        }
        // Columns stays visible from the start; the grid needs a source table
        // before it has anything to show, so it holds this in the meantime.
        expect(
            screen.getByText("Select a source table to see its columns")
        ).toBeInTheDocument();
    });

    it("disables the Add button until a name and source table are set", async () => {
        await renderConnectTablesPage();

        const addTable = await screen.findByRole("button", { name: "Add" });
        expect(addTable).toBeDisabled();

        const nameInput = await screen.findByPlaceholderText("Use default");
        fireEvent.change(nameInput, { target: { value: "test_table" } });

        // A name alone is not enough while no source table is selected.
        await waitFor(() => expect(addTable).toBeDisabled());
    });

    it("Add button remains disabled when required inputs are incomplete", async () => {
        await renderConnectTablesPage();

        const addTable = await screen.findByRole("button", { name: "Add" });
        expect(addTable).toBeDisabled();
    });

    it("Should render default Add New Table page - (databricks-deployment)", async () => {
        setDeploymentTypeInStore(deploymentType.Databricks);
        await setupConnectTablePage();
    });

    it("Should navigate back table list page on `Cancel` click", async () => {
        await setupConnectTablePage();

        setTableListResponseType(testResponseType.ALL);

        const cancelButton = await screen.findByRole("button", {
            name: "Cancel"
        });
        await userEvent.click(cancelButton);
        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                "/project/project-tests/tables/",
                { "replace": true }
            );
        });
    });

    it("Should handle Table Name validation for valid and invalid message", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        // With no table name entered the Add button is disabled, so the
        // legacy click-time "Please enter table name." path is unreachable.
        const addButton = await screen.findByRole("button", { name: "Add" });
        expect(addButton).toBeDisabled();

        const nameInput = screen.getByPlaceholderText("Use default");
        const validName = "test_table";
        fireEvent.change(nameInput, { target: { value: validName } });

        await checkNameValidationApi(validName);

        setTableNameResponseType(testResponseType.INVALID);
        const invalidName = "test_table table";
        await waitFor(() => {
            userEvent.clear(nameInput);
            fireEvent.change(nameInput, { target: { value: invalidName } });
        });

        await checkNameValidationApi(invalidName);

        await waitFor(() => {
            screen.getByText(
                `Table identifier may only contain alphanumeric characters and underscores, and must begin with a letter. Got: "${invalidName}"`
            );
        });

        setTableNameResponseType(testResponseType.EXISTS);
        const alreadyExists = "financial";
        await userEvent.clear(nameInput);
        fireEvent.change(nameInput, { target: { value: alreadyExists } });

        await checkNameValidationApi(alreadyExists);

        await waitFor(() => {
            screen.getByText(
                `Table with name ${alreadyExists} already exists. Please choose a different name.`
            );
        });
        setTableNameResponseType(testResponseType.ALL);
    });

    it("Should retain source data type when the source table is switched", async () => {
        await renderConnectTablesPage();
        await pageLoadTest();

        await sourceTableSelection();
        closeDropdown();

        const expectedSourceTable = structuredClone(
            expectedSourceTablesResponse()
        );
        await callSampleRowsAndGetColumns("ltv_test", {
            name: expectedSourceTable.table_names[2]
        });

        //Data Type change
        const dTypeDropdowns = await screen.findAllByTestId(
            "testid-select-d-type"
        );
        expect(dTypeDropdowns.length).toBeGreaterThan(0);
        const dataTypeDropdown = dTypeDropdowns[4];
        expect(dataTypeDropdown).toHaveTextContent("String");
        await userEvent.click(dataTypeDropdown);
        const newDataTypeOption = await screen.findByText("Date");
        expect(newDataTypeOption).toBeInTheDocument();
        await userEvent.click(newDataTypeOption);

        const initialColumnsCallCount = mockAxiosInstance.get.mock.calls.filter(
            (call) => call[0].includes("/columns")
        ).length;

        //Source Table change
        const tableDropdown = await screen.findByRole("button", {
            name: "transaction"
        });

        // Selecting the first source table derives and validates the table
        // name. Source-table selection is intentionally disabled while that
        // validation is in flight, so wait for the control to become usable
        // before opening it again.
        await waitFor(() => expect(tableDropdown).toBeEnabled(), {
            timeout: 3000
        });
        await userEvent.click(tableDropdown, { pointerEventsCheck: 0 });

        const sourceTableElement = await screen.findByTestId(
            "select-source-table-option-1"
        ); //customer table
        await userEvent.click(sourceTableElement);

        await waitFor(() => {
            const newColumnsCallCount = mockAxiosInstance.get.mock.calls.filter(
                (call) => call[0].includes("/columns")
            ).length;
            expect(newColumnsCallCount).toBeGreaterThan(
                initialColumnsCallCount
            );
        });

        const afterFirstSwitchColumnsCount =
            mockAxiosInstance.get.mock.calls.filter((call) =>
                call[0].includes("/columns")
            ).length;

        //Source Table change back to original table
        const tableDropdownAgain = await screen.findByRole("button", {
            name: "customer"
        });
        // Each switch re-derives the table name, and selection stays disabled
        // while that name validates — same wait as the first switch above.
        await waitFor(() => expect(tableDropdownAgain).toBeEnabled(), {
            timeout: 3000
        });
        await userEvent.click(tableDropdownAgain, { pointerEventsCheck: 0 });
        const originalSourceTableElement = await screen.findByTestId(
            "select-source-table-option-2"
        ); //transaction table
        await userEvent.click(originalSourceTableElement);

        //Data type check
        await waitFor(() => {
            const dTypes = screen.getAllByTestId("testid-select-d-type");
            expect(dTypes.length).toBeGreaterThan(4);
            expect(dTypes[4]).toHaveTextContent("String");
        });

        const afterSecondSwitchColumnsCount =
            mockAxiosInstance.get.mock.calls.filter((call) =>
                call[0].includes("/columns")
            ).length;

        expect(afterSecondSwitchColumnsCount).toBeGreaterThan(
            afterFirstSwitchColumnsCount
        );
    });

    describe("Table creation with S3 connector and column modifications", () => {
        beforeEach(async () => {
            setTableNameResponseType(testResponseType.VALID_NAME);
            await setupConnectTablePage();

            //The Source(Connector) and Connector(ltv_test) are initially set in this case
            await sourceTableSelection();

            //Renaming the table
            await renameTable("test_table");

            ///Table elements check
            const tableHeader = await screen.findByTestId("table-header");
            expect(tableHeader).toHaveTextContent("Column Name");
            const columnName = screen.getByTestId("column-name-4");
            expect(columnName).toHaveTextContent("InvoiceDate");

            //Checkbox selected check
            const checkBoxes = await screen.findAllByRole("checkbox");
            checkBoxes.forEach((checkBox) => {
                expect(checkBox).toBeChecked();
            });

            //Data Type change
            const dTypeDropdowns = await screen.findAllByTestId(
                "testid-select-d-type"
            );
            expect(dTypeDropdowns.length).toBeGreaterThan(0);
            const dataTypeDropdown = dTypeDropdowns[4];
            expect(dataTypeDropdown).toHaveTextContent("String");
            await userEvent.click(dataTypeDropdown);
            const newDataTypeOption = await screen.findByText("Date");
            expect(newDataTypeOption).toBeInTheDocument();
            await userEvent.click(newDataTypeOption);

            //SType change
            const sTypeDropdowns = await screen.findAllByTestId(
                "testid-select-s-type"
            );
            expect(sTypeDropdowns.length).toBeGreaterThan(0);
            const sTypeDropdown = sTypeDropdowns[4];
            expect(sTypeDropdown).toHaveTextContent("Multicategorical");
            await userEvent.click(sTypeDropdown);
            const newSTypeOption = await screen.findByText("Timestamp");
            expect(newSTypeOption).toBeInTheDocument();
            await userEvent.click(newSTypeOption);

            //Property change
            const propertiesDropdowns = await screen.findAllByTestId(
                /^testid-properties-select-\d+$/
            );
            expect(propertiesDropdowns.length).toBeGreaterThan(0);
            await userEvent.click(propertiesDropdowns[4]);
            const propertyToSelect = await screen.findByText("Create date");
            expect(propertyToSelect).toBeInTheDocument();
            await userEvent.click(propertyToSelect);
            await waitFor(() => {
                const updatedDropdown = screen.getByTestId(
                    "testid-properties-select-4"
                );
                expect(updatedDropdown).toHaveTextContent("Create date");
            });
        });

        it("Should successfully create table with all column modifications", async () => {
            await createTableApiCallTest(mockNewTable as CreateTableArgs, {});
        });

        it("Should handle validation API failures when creating table", async () => {
            await createTableApiCallTest(mockNewTable as CreateTableArgs, {
                validateWithFailure: true,
                validateWithInvalid: false
            });
        });

        it("Should handle validation API invalid responses when creating table", async () => {
            await createTableApiCallTest(mockNewTable as CreateTableArgs, {
                validateWithFailure: false,
                validateWithInvalid: true
            });
        });

        it("Should handle table creation API failures", async () => {
            await createTableApiCallTest(mockNewTable as CreateTableArgs, {
                createWithFailure: true,
                createWithInvalid: true
            });
        });

        it("Should handle table creation API invalid responses", async () => {
            await createTableApiCallTest(mockNewTable as CreateTableArgs, {
                createWithFailure: false,
                createWithInvalid: true
            });
        });
    });

    it("Should display `Add a Primary Key?` pop up when primary key not selected", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        const sourceTableToSelect = await sourceTableSelection(0);

        //Renaming the table
        await renameTable("test_table");

        const addTableButton = await screen.findByRole("button", {
            name: "Add"
        });
        await userEvent.click(addTableButton);

        //Primary Key pop up
        const role = await screen.findByRole("dialog");
        expect(role).toHaveTextContent("Add a Primary Key?");

        expect(
            screen.getByText(
                /Would you like to set column.*CustomerID.*as the primary key/i
            )
        ).toBeInTheDocument();

        const learnMoreLink = screen.getByTestId("testid-pkey_selection");
        expect(learnMoreLink).toBeInTheDocument();
        expect(learnMoreLink).toHaveAttribute(
            "href",
            expect.stringContaining(
                "https://kumo.ai/docs/column-preprocessing#primary-key-column"
            )
        );
        expect(learnMoreLink).toHaveAttribute("target", "_blank");

        const skipBtn = screen.getByRole("button", { name: "Skip" });
        const setPkeyBtn = screen.getByRole("button", {
            name: "Set as primary key"
        });
        [setPkeyBtn, skipBtn].forEach((btn) =>
            expect(btn).toHaveAttribute("data-slot", "button")
        );

        //Should create table with Primary Key selected
        await userEvent.click(setPkeyBtn);

        const newTable = structuredClone(
            expectedNewTableResponse(sourceTableToSelect?.name)
        );
        const args = {
            "cols": newTable.cols ?? [],
            "source_table": newTable?.source_table,
            "pkey": "CustomerID",
            "time_col": "",
            "end_time_col": ""
        };

        await waitFor(() => {
            const url = `v1/tables/validate`;
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
                headers: { "Content-Type": "application/json" },
                params: { project_id: "project-tests" }
            });
        });
    });

    it("Should display `Add a Create Date?` pop up when create date not selected", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        await sourceTableSelection();

        //Table name
        await checkNameValidationApi("transaction");

        //SType change
        await waitFor(() => {
            const sTypeDropdowns = screen.getAllByTestId(
                "testid-select-s-type"
            );
            expect(sTypeDropdowns.length).toBeGreaterThanOrEqual(0);
            expect(sTypeDropdowns[4]).toHaveTextContent("Multicategorical");
        });

        const sTypeDropdowns = screen.getAllByTestId("testid-select-s-type");
        const sTypeDropdown = sTypeDropdowns[4];
        await userEvent.click(sTypeDropdown);
        const newSTypeOption = await screen.findByText("Timestamp");
        expect(newSTypeOption).toBeInTheDocument();
        await userEvent.click(newSTypeOption);

        const addTableButton = screen.getByRole("button", {
            name: "Add"
        });
        await userEvent.click(addTableButton);

        //Create date pop up
        const role = await screen.findByRole("dialog");
        expect(role).toHaveTextContent("Add a Create Date?");

        expect(
            screen.getByText(
                /Would you like to set column.*InvoiceDate.*as the create date/i
            )
        ).toBeInTheDocument();

        const learnMoreLink = screen.getByTestId(
            "testid-create_date_selection"
        );
        expect(learnMoreLink).toBeInTheDocument();
        expect(learnMoreLink).toHaveAttribute(
            "href",
            expect.stringContaining(
                "https://kumo.ai/docs/column-preprocessing#create-date-column"
            )
        );
        expect(learnMoreLink).toHaveAttribute("target", "_blank");

        const skipBtn = screen.getByRole("button", { name: "Skip" });
        const setDateBtn = screen.getByRole("button", {
            name: "Set as create date"
        });
        [setDateBtn, skipBtn].forEach((btn) =>
            expect(btn).toHaveAttribute("data-slot", "button")
        );

        //Should create table with Create Date selected
        await userEvent.click(setDateBtn);

        const newTable = structuredClone(expectedNewTableResponse());
        const args = {
            "cols": newTable?.cols ?? [],
            "source_table": newTable?.source_table,
            "pkey": "",
            "time_col": "InvoiceDate",
            "end_time_col": ""
        };
        await waitFor(() => {
            const url = `v1/tables/validate`;
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
                headers: { "Content-Type": "application/json" },
                params: { project_id: "project-tests" }
            });
        });
    });

    it("Should clear `Create Date Type is not valid!` error after selecting a valid column", async () => {
        await setupConnectTablePage();

        await sourceTableSelection();

        await checkNameValidationApi("transaction");

        await selectPropertyForColumn(1, "Create date");

        await verifyPropertyErrorAlert(
            "Create Date",
            /Create Date Type is not valid!.*Please select Timestamp columns only/i
        );

        await clearPropertyError(1);
    });

    it("Should clear `End Date Type is not valid!` error after selecting a valid column", async () => {
        await setupConnectTablePage();

        await sourceTableSelection();

        await checkNameValidationApi("transaction");

        await selectPropertyForColumn(1, "End date");

        await verifyPropertyErrorAlert(
            "End Date",
            /End Date Type is not valid!.*Please select Timestamp columns only/i
        );

        await clearPropertyError(1);
    });

    it("Should clear `Primary Key (pkey) must be an ID.` error after selecting a valid column", async () => {
        await setupConnectTablePage();

        await sourceTableSelection();

        await checkNameValidationApi("transaction");

        await selectPropertyForColumn(4, "Primary Key");

        const sTypeDropdowns = screen.getAllByTestId("testid-select-s-type");
        expect(sTypeDropdowns[4]).toHaveTextContent("ID");
        await changeSTypeForColumn(4, "Timestamp");

        await verifyPropertyErrorAlert(
            "Primary Key",
            "Primary Key (pkey) must be an ID.Please change the semantic type (stype) of your column to kumo.ID and try again!"
        );

        await userEvent.click(
            screen.getAllByTestId(/^testid-properties-select-\d+$/)[4]
        );
        const property = await screen.findByTestId(
            "properties-select-4-option-0"
        );
        await userEvent.click(property);

        expect(
            screen.getByTestId("testid-properties-select-4")
        ).toHaveTextContent("Select");

        await waitFor(
            () => {
                expect(screen.queryByRole("alert")).not.toBeInTheDocument();
            },
            { timeout: 2000 }
        );
    });

    it("Should create table without the Primary Key selection when `No` button is clicked under the Primary Key pop-up", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        const sourceTableToSelect = await sourceTableSelection(0);

        //Renaming the table
        await renameTable("test_table");

        const addTableButton = await screen.findByRole("button", {
            name: "Add"
        });
        await userEvent.click(addTableButton);

        //Primary Key pop up
        const role = await screen.findByRole("dialog");
        expect(role).toHaveTextContent("Add a Primary Key?");

        const skipBtn = screen.getByRole("button", { name: "Skip" });
        await userEvent.click(skipBtn);

        const newTable = structuredClone(
            expectedNewTableResponse(sourceTableToSelect?.name)
        );
        const args = {
            "cols": newTable?.cols?.map((col) => ({
                ...col,
                stype: "numerical"
            })),
            "source_table": newTable?.source_table,
            "pkey": "",
            "time_col": "",
            "end_time_col": ""
        };

        await waitFor(() => {
            const url = `v1/tables/validate`;
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
                headers: { "Content-Type": "application/json" },
                params: { project_id: "project-tests" }
            });
        });
    });

    it("Should create table without the create date selection when `No` button is clicked under the Create date pop-up", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        await sourceTableSelection();

        //Renaming the table
        await renameTable("test_table");

        //SType change
        const sTypeDropdowns = await screen.findAllByTestId(
            "testid-select-s-type"
        );
        expect(sTypeDropdowns.length).toBeGreaterThan(0);
        const sTypeDropdown = sTypeDropdowns[4];
        await waitFor(() => {
            expect(sTypeDropdown).toHaveTextContent("Multicategorical");
        });
        await userEvent.click(sTypeDropdown);
        await waitFor(async () => {
            const newSTypeOption = await screen.findByText("Timestamp");
            expect(newSTypeOption).toBeInTheDocument();
        });

        await userEvent.click(await screen.findByText("Timestamp"));

        const addTableButton = screen.getByRole("button", {
            name: "Add"
        });
        await userEvent.click(addTableButton);

        //Create date pop up
        const role = await screen.findByRole("dialog");
        expect(role).toHaveTextContent("Add a Create Date?");

        const skipBtn = screen.getByRole("button", { name: "Skip" });
        await userEvent.click(skipBtn);

        const newTable = structuredClone(expectedNewTableResponse());
        const args = {
            "cols": newTable?.cols ?? [],
            "source_table": newTable?.source_table,
            "pkey": "",
            "time_col": "",
            "end_time_col": ""
        };
        await waitFor(() => {
            const url = `v1/tables/validate`;
            expect(mockAxiosInstance.post).toHaveBeenCalledWith(url, args, {
                headers: { "Content-Type": "application/json" },
                params: { project_id: "project-tests" }
            });
        });
    });

    it("Should display error message when the row with Primary Key is deselected", async () => {
        await setupConnectTablePage();

        await sourceTableSelection();

        await checkNameValidationApi("transaction");

        await selectPropertyForColumn(1, "Primary Key");

        await uncheckColumnCheckbox(2);

        const addTable = screen.getByText("Add");
        expect(addTable).toBeEnabled();
        await userEvent.click(addTable);

        const rowName = "StockCode";
        await waitFor(() => {
            const alert = screen.getAllByRole("alert")[0];
            expect(alert).toHaveTextContent(
                `Please select the column "${rowName}" that is selected as Primary Key`
            );
        });
    });

    it("Should display error message when the row with Create date is deselected", async () => {
        await setupConnectTablePage();

        await sourceTableSelection();

        await checkNameValidationApi("transaction");

        await selectPropertyForColumn(4, "Create date");

        await uncheckColumnCheckbox(5);

        const addTable = screen.getByText("Add");
        await userEvent.click(addTable);

        const rowName = "InvoiceDate";
        await waitFor(() => {
            const alert = screen.getAllByRole("alert")[0];
            expect(alert).toHaveTextContent(
                `Please select the column "${rowName}" that is selected as Create Date`
            );
        });
    });

    it("Should display error message when the row with End date is deselected", async () => {
        await setupConnectTablePage();

        await sourceTableSelection();

        await checkNameValidationApi("transaction");

        await selectPropertyForColumn(4, "End date");

        await uncheckColumnCheckbox(5);

        const addTable = screen.getByText("Add");
        await userEvent.click(addTable);

        const rowName = "InvoiceDate";
        await waitFor(() => {
            const alert = screen.getAllByRole("alert")[0];
            expect(alert).toHaveTextContent(
                `Please select the column "${rowName}" that is selected as End Date`
            );
        });
    });

    it("Should display error message when all the rows are unchecked and the add table button is clicked", async () => {
        await setupConnectTablePage();

        await sourceTableSelection();

        //Table name
        await checkNameValidationApi("transaction");

        //Uncheck all the rows
        const checkBoxes = await screen.findAllByRole("checkbox");
        checkBoxes.forEach((checkBox) => {
            expect(checkBox).toBeChecked();
        });
        await userEvent.click(checkBoxes[0]);

        const addTable = screen.getByText("Add");
        await userEvent.click(addTable);

        //Error alert
        await waitFor(() => {
            const alert = screen.getAllByRole("alert");
            expect(alert[0]).toHaveTextContent(`Please select the column !`);
            expect(alert[0]).toHaveTextContent(
                `Please select one or more column(s).`
            );
        });
    });

    it("Should show `Please select Connector` for empty Connector List", async () => {
        setConnectorListResponseType(testResponseType.EMPTY);
        await renderConnectTablesPage();

        const dropdown = await screen.findByTestId("testid-select-connector");
        await userEvent.click(dropdown);
        screen.getByText("No connector found.");
        closeDropdown();

        // No table name and no source table selected, so the Add button is
        // disabled. The "Please select Connector" message below is reactive
        // name validation, not the disabled click-time path.
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();

        const nameInput = await screen.findByPlaceholderText("Use default");
        await userEvent.type(nameInput, "test_table");

        await waitFor(
            async () => {
                expect(
                    await screen.findByText(/Please select Connector/i)
                ).toBeVisible();
            },
            { timeout: 5000 }
        );

        const tableDropdown = screen.getByTestId("testid-select-source-table");
        await userEvent.click(tableDropdown);
        screen.getByText("No tables found.");
        closeDropdown();
    });

    it("Add button stays disabled for empty Source Table", async () => {
        setSourceTablesResponsesType(testResponseType.EMPTY);
        await renderConnectTablesPage();

        await selectConnector(0);

        const tableDropdown = screen.getByTestId("testid-select-source-table");
        await userEvent.click(tableDropdown);
        screen.getByText("No tables found.");
        closeDropdown();

        const addButton = screen.getByRole("button", { name: "Add" });
        expect(addButton).toBeDisabled();

        // Entering a valid name is not enough; without a selected source
        // table the Add button stays disabled (inputs incomplete).
        const nameInput = screen.getByPlaceholderText("Use default");
        fireEvent.change(nameInput, { target: { value: "test_table" } });

        expect(addButton).toBeDisabled();
    });

    it("Should render the Local Upload source UI correctly on Connect Tables page", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        await selectSourceType("Local Upload");

        expect(
            screen.getByText(/or drag and drop it here/i)
        ).toBeInTheDocument();
        expect(
            screen.getByText(
                /Only '.CSV' or '.Parquet' files are allowed with max size 100 GiB/i
            )
        ).toBeInTheDocument();
    });

    it("Should stack local-upload fields behind the file-source radio", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        await selectSourceType("Local Upload");

        // The two ways to point at a file are mutually exclusive, so only the
        // selected one renders. Upload is the default.
        const uploadChoice = screen.getByRole("radio", {
            name: "Upload a new file"
        });
        expect(uploadChoice).toBeChecked();
        const dropZone = screen.getByTestId("local-upload-dropzone");
        expect(dropZone).toBeInTheDocument();
        expect(
            screen.queryByTestId("testid-select-source-table")
        ).not.toBeInTheDocument();

        // Fields are stacked top-to-bottom in decision order: the Source
        // section comes first, then Table details holds the file choice, the
        // control it selects, and finally the table name.
        const tableNameInput = screen.getByPlaceholderText("Use default");
        const sourceSelector = screen.getByTestId("table-source-selector");
        const isBefore = (a: Element, b: Element) =>
            Boolean(
                a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING
            );
        expect(isBefore(sourceSelector, uploadChoice)).toBe(true);
        expect(isBefore(uploadChoice, dropZone)).toBe(true);
        expect(isBefore(dropZone, tableNameInput)).toBe(true);

        // #24679: picking the other option swaps in the dropdown of
        // previously-uploaded / SDK-pushed files and retires the drop zone.
        await userEvent.click(
            screen.getByRole("radio", { name: "Choose an existing file" })
        );
        expect(
            await screen.findByTestId("testid-select-source-table")
        ).toBeInTheDocument();
        expect(
            screen.queryByTestId("local-upload-dropzone")
        ).not.toBeInTheDocument();
    });

    it("Should default to Local Upload when opened from connector flow", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        sessionStorage.setItem(
            storageKeys.isLocalUpload,
            JSON.stringify("true")
        );
        await renderConnectTablesPage("/project/project-test/tables/new-table");
        await pageLoadTest({ expectSourceTablesRequest: false });

        expect(screen.getByTestId("testid-select-source")).toHaveTextContent(
            "Local Upload"
        );
        expect(
            screen.getByText(/or drag and drop it here/i)
        ).toBeInTheDocument();

        const nameInput = screen.getByPlaceholderText("Use default");
        fireEvent.change(nameInput, { target: { value: "test_table" } });
        await checkNameValidationApi("test_table");

        // No file uploaded yet, so the Add button stays disabled.
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("Should suggest Table name from the chosen local file", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        await selectSourceType("Local Upload");

        const simulateFileDrop = async (file: File) => {
            const input = screen
                .getByTestId("local-upload-dropzone")
                .querySelector("input[type=file]") as HTMLInputElement;
            Object.defineProperty(input, "files", {
                value: [file],
                configurable: true
            });
            await act(() => fireEvent.change(input));
        };

        const file = new File(["sample file"], "sample file.csv", {
            type: "text/csv"
        });
        await simulateFileDrop(file);

        // Completion replaces the upload row with a new success card.
        await waitFor(() => {
            expect(
                screen.queryByTestId("local-upload-dropzone")
            ).not.toBeInTheDocument();
            expect(screen.getByText("sample file.csv")).toBeInTheDocument();
            expect(screen.getByPlaceholderText("Use default")).toHaveValue(
                "sample_file"
            );
        });
        // The upload starts on selection, so there is no Upload button.
        expect(
            screen.queryByRole("button", { name: "Upload" })
        ).not.toBeInTheDocument();

        await checkNameValidationApi("sample_file");
    });

    it("Should keep a table name the user typed when the source changes", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        // First pick fills the name in; the user then makes it their own.
        await sourceTableSelection();
        closeDropdown();
        const tableNameInput = screen.getByPlaceholderText("Use default");
        fireEvent.change(tableNameInput, { target: { value: "my_own_name" } });
        await checkNameValidationApi("my_own_name");

        const tableDropdown = await screen.findByRole("button", {
            name: "transaction"
        });
        await waitFor(() => expect(tableDropdown).toBeEnabled(), {
            timeout: 3000
        });
        await userEvent.click(tableDropdown, { pointerEventsCheck: 0 });
        await userEvent.click(
            await screen.findByTestId("select-source-table-option-1")
        );

        // Their name survives; only a name this page suggested is replaced.
        await waitFor(() =>
            expect(screen.getByPlaceholderText("Use default")).toHaveValue(
                "my_own_name"
            )
        );
    });

    it("Add button stays disabled without uploading a table under local upload setup", async () => {
        setTableNameResponseType(testResponseType.VALID_NAME);
        await setupConnectTablePage();

        await selectSourceType("Local Upload");

        const nameInput = screen.getByPlaceholderText("Use default");
        const validName = "sample_table";
        fireEvent.change(nameInput, { target: { value: validName } });

        await checkNameValidationApi(validName);

        // Without a file uploaded the Add button remains disabled.
        expect(screen.getByRole("button", { name: "Add" })).toBeDisabled();
    });

    it("Should handle connector change and reset form state", async () => {
        await setupConnectTablePage();

        await selectConnector(0);

        //Change connector
        await selectConnector(1);

        const nameInput = screen.getByPlaceholderText("Use default");
        expect(nameInput).toHaveValue("");
    });

    it("Should validate multiple property type combinations", async () => {
        await setupConnectTablePage();

        await sourceTableSelection();

        await checkNameValidationApi("transaction");

        await selectPropertyForColumn(1, "Primary Key");
        await selectPropertyForColumn(4, "Create date");

        expect(
            screen.getByTestId("testid-properties-select-1")
        ).toHaveTextContent("Primary Key");
        expect(
            screen.getByTestId("testid-properties-select-4")
        ).toHaveTextContent("Create date");
    });

    it("Should handle cancel button click during different deployment types", async () => {
        setDeploymentTypeInStore(deploymentType.Databricks);
        await setupConnectTablePage();

        setTableListResponseType(testResponseType.ALL);

        const cancelButton = await screen.findByRole("button", {
            name: "Cancel"
        });
        await userEvent.click(cancelButton);

        await waitFor(() => {
            expect(mockUseNavigate).toHaveBeenCalledWith(
                "/project/project-tests/tables/",
                { "replace": true }
            );
        });
    });
});
