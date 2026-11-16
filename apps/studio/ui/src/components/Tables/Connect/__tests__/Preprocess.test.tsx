// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { KumoapiTableTableDefinition } from "gen/v1/types/kumoapiTableTableDefinition";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";

import { DropdownOption } from "types/inputs";
import { DataSourceType, FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { storageKeys } from "utils/Constants";

import Preprocessing, { PreprocessingProps } from "../Preprocess";

import { setColumnResponseType } from "./MockFunction/ColumnResponse.test.mock";
import { setSampleRowsResponseType } from "./MockFunction/SampleRowsResponse.test.mock";
import { tableWithConnector } from "./MockResponse/PreprocessResponses";

const connectorId = "ltv_test";
const selectedSourceTableName = "transaction";
const setTableArgs = vi.fn();

const defaultProps: Partial<PreprocessingProps> = {
    tableArgs: {
        cols: [],
        source_table: {
            connector_id: "connector_missing",
            file_type: "PARQUET",
            data_source_type: DataSourceType.S3,
            source: { mode: "relative", path: "" }
        }
    } as KumoapiTableTableDefinition,
    flowType: FlowType.CREATE,
    handleError: {} as HandleErrorFunc,
    setLoading: vi.fn(),
    isLoading: false,
    setTableArgs,
    selectedConnector: {} as DropdownOption,
    cancelFetchSampleCal: false,
    connectorId: "",
    selectedSourceTableName: "",
    selectFromTableArgs: false,
    connectorNotExists: false,
    setUnnamedConSourceTable: vi.fn(),
    sourcetableNotExists: false,
    setColumnNamesAdjusted: vi.fn(),
    columnNamesAdjusted: false
};

describe("Preprocessing", async () => {
    const queryClient = createQueryClient();
    const renderPreprocessing = async (
        props: Partial<PreprocessingProps> = defaultProps
    ) => {
        await act(async () => {
            render(
                <MemoryRouter
                    initialEntries={["/project/project-test/tables/new-table"]}>
                    <QueryClientProvider client={queryClient}>
                        <Preprocessing {...(props as PreprocessingProps)} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const callSampleRowsAndGetColumns = async (
        connectorId: string,
        selectedSourceTableName: { name?: string } | null
    ) => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/columns`,
                undefined,
                {
                    headers: undefined,
                    params: {
                        table_name: selectedSourceTableName?.name
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
                        table_name: selectedSourceTableName?.name
                    }
                }
            );
        });
    };

    const clickSkipButton = async () => {
        const skipButton = screen.queryByRole("button", { name: "Skip" });
        if (skipButton) {
            await userEvent.click(skipButton);
        }
    };

    const setupTestWithConnector = (
        overrides: Partial<PreprocessingProps> = {}
    ) => ({
        ...defaultProps,
        tableArgs: tableWithConnector,
        connectorId,
        selectedSourceTableName,
        ...overrides
    });

    const setupAndRenderWithData = async (
        overrides: Partial<PreprocessingProps> = {}
    ) => {
        await renderPreprocessing(setupTestWithConnector(overrides));
        await clickSkipButton();
        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });
    };

    const selectPropertyOption = async (
        dropdownIndex: number,
        optionText: string
    ) => {
        const propertiesDropdowns = await screen.findAllByTestId(
            /^testid-properties-select-\d+$/
        );
        await userEvent.click(propertiesDropdowns[dropdownIndex]);

        const options = await screen.findAllByText(optionText);
        await userEvent.click(options[dropdownIndex] || options[0]);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        vi.resetAllMocks();
        queryClient.clear();
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");
        setColumnResponseType(testResponseType.DEFAULT);
        setSampleRowsResponseType(testResponseType.DEFAULT);
    });

    it.each([
        [FlowType.CREATE, true, "CREATE"],
        [FlowType.DUPLICATE, true, "DUPLICATE"],
        [FlowType.EDIT, false, "EDIT"]
    ])(
        "Should %s display `Skip` button in %s flow",
        async (flowType, shouldDisplay) => {
            setSampleRowsResponseType(testResponseType.INVALID);
            await renderPreprocessing(setupTestWithConnector({ flowType }));

            const skipButton = screen.queryByRole("button", { name: "Skip" });
            if (shouldDisplay) {
                expect(
                    screen.getByText(
                        "Inferring column types based on sample rows."
                    )
                ).toBeInTheDocument();
                expect(skipButton).toBeInTheDocument();
            } else {
                expect(
                    screen.queryByText(
                        "Inferring column types based on sample rows."
                    )
                ).not.toBeInTheDocument();
                expect(skipButton).not.toBeInTheDocument();
            }
        }
    );

    it("Should handle hover over the table headers", async () => {
        await renderPreprocessing(setupTestWithConnector());

        await clickSkipButton();

        expect(
            screen.getByText(
                /Select the columns & any preprocessing you want to apply to new table/i
            )
        ).toBeInTheDocument();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const tableHeader = screen.getByTestId("table-header");
        expect(tableHeader).toBeInTheDocument();
        expect(tableHeader).toHaveTextContent(
            "Column NameData TypeSemantic TypePropertiesPreview"
        );

        const headers = [
            { name: "Column Name", expectedText: "name of each column" },
            {
                name: "Data Type",
                expectedText: "represents how the data is physically stored"
            },
            {
                name: "Semantic Type",
                expectedText: "determines how data is processed and encoded"
            },
            { name: "Properties", expectedText: "property for this column" },
            {
                name: "Preview",
                expectedText:
                    "This shows you an example of raw data from this column to better inform type decisions"
            }
        ];

        for (const header of headers) {
            const headerElement = screen.getByText(header.name);
            expect(headerElement).toBeInTheDocument();
            await userEvent.hover(headerElement);
            await waitFor(() => {
                const tooltips = screen.getAllByText((text) =>
                    text.includes(header.expectedText)
                );
                const visibleTooltip = tooltips.find(
                    (el) => el.offsetParent !== null
                );
                expect(visibleTooltip).toBeTruthy();
                expect(visibleTooltip).toBeInTheDocument();
            });
        }
    });

    it("Should handle `Data Type` change", async () => {
        await renderPreprocessing(setupTestWithConnector({ setTableArgs }));

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const dTypeDropdowns = screen.getAllByTestId("testid-select-d-type");
        expect(dTypeDropdowns.length).toBeGreaterThan(0);
        const dataTypeDropdown = dTypeDropdowns[5];
        expect(dataTypeDropdown).toHaveTextContent("Float64");
        await userEvent.click(dataTypeDropdown);

        const menu = await screen.findByRole("listbox");
        expect(menu).toBeInTheDocument();
        const newDataTypeOption = await screen.findByText("Int");
        expect(newDataTypeOption).toBeInTheDocument();
        await userEvent.click(newDataTypeOption);

        await waitFor(() => {
            expect(
                screen.getAllByTestId("testid-select-d-type")[5]
            ).toHaveTextContent("Int");
            expect(setTableArgs).toHaveBeenCalled();
        });
    });

    it("Should handle `Type` change", async () => {
        await renderPreprocessing(setupTestWithConnector({ setTableArgs }));

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const sTypeDropdowns = screen.getAllByTestId("testid-select-s-type");
        expect(sTypeDropdowns.length).toBeGreaterThan(0);
        const sTypeDropdown = sTypeDropdowns[6];
        expect(sTypeDropdown).toHaveTextContent("Numerical");
        await userEvent.click(sTypeDropdown);

        const menu = await screen.findByRole("listbox");
        expect(menu).toBeInTheDocument();
        const newTypeOption = await screen.findByText(
            "Numbers or text that uniquely identify an entity in your graph (Eg PK or FKs)."
        );
        expect(newTypeOption).toBeInTheDocument();
        await userEvent.click(newTypeOption);

        await waitFor(() => {
            expect(
                screen.getAllByTestId("testid-select-s-type")[6]
            ).toHaveTextContent("ID");
            expect(setTableArgs).toHaveBeenCalled();
        });
    });

    it.each([
        ["Primary Key", 0],
        ["Create date", 0],
        ["End date", 0]
    ])("Should handle `%s` selection", async (propertyOption) => {
        await setupAndRenderWithData({ setTableArgs });

        await selectPropertyOption(0, propertyOption);

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        await waitFor(() => {
            const propertiesDropdowns = screen.getAllByTestId(
                /^testid-properties-select-\d+$/
            )[0];
            expect(propertiesDropdowns).toHaveTextContent(propertyOption);
            expect(setTableArgs).toHaveBeenCalled();
        });
    });

    it("Should display `No data found.` when the API return Invalid response", async () => {
        setColumnResponseType(testResponseType.INVALID);
        const handleError = vi.fn();

        await renderPreprocessing(
            setupTestWithConnector({ selectFromTableArgs: true, handleError })
        );

        await clickSkipButton();

        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/columns`,
                undefined,
                {
                    headers: undefined,
                    params: {
                        table_name: selectedSourceTableName
                    }
                }
            );
        });

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });

        const [errorArg, messageArg] = handleError.mock.calls[0];

        expect(errorArg).toMatchObject({
            message: expect.any(String),
            name: expect.any(String),
            response: {
                data: expect.anything(),
                status: 422
            }
        });

        expect(messageArg).toBe("Failed to load columns");
        expect(screen.getByText("No data found."));
    });

    it("Should auto-change data type to date when semantic type changes to timestamp", async () => {
        await renderPreprocessing(setupTestWithConnector({ setTableArgs }));

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const sTypeDropdowns = screen.getAllByTestId("testid-select-s-type");
        const dTypeDropdowns = screen.getAllByTestId("testid-select-d-type");

        expect(sTypeDropdowns[4]).toHaveTextContent("Multicategorical");
        expect(dTypeDropdowns[4]).toHaveTextContent("String");

        await userEvent.click(sTypeDropdowns[4]);
        const timestampOption = await screen.findByText(
            "Numbers or strings that represent a date or time (eg date_created)."
        ); //Timestamp option
        await userEvent.click(timestampOption);

        await waitFor(() => {
            expect(setTableArgs).toHaveBeenCalled();
            expect(
                screen.getAllByTestId("testid-select-s-type")[4]
            ).toHaveTextContent("Timestamp");
            expect(
                screen.getAllByTestId("testid-select-d-type")[4]
            ).toHaveTextContent("Date");
        });
    });

    it("Should display Snowflake-specific message when connector is Snowflake", async () => {
        await renderPreprocessing(
            setupTestWithConnector({
                selectedConnector: {
                    label: "Snowflake",
                    value: "snowflake",
                    type: "snowflake"
                }
            })
        );

        expect(
            screen.getByText(/Inferring column types based on sample rows./)
        ).toBeInTheDocument();
        expect(
            screen.getByText(/The duration depends on your warehouse size/)
        ).toBeInTheDocument();
    });

    it("Should display column names adjusted warning", async () => {
        const setColumnNamesAdjusted = vi.fn();

        await renderPreprocessing(
            setupTestWithConnector({
                columnNamesAdjusted: true,
                setColumnNamesAdjusted
            })
        );

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const alerts = await screen.findAllByRole("alert");
        const warningAlert = alerts.find((alert) =>
            /Column names adjusted/i.test(alert.textContent)
        );
        expect(warningAlert).toBeTruthy();

        const closeButton = screen.getByTestId("alert-close");
        await userEvent.click(closeButton);

        await waitFor(() => {
            expect(
                screen.queryByText(/Column names adjusted/i)
            ).not.toBeInTheDocument();
        });
    });

    it.each([
        ["time_col", "InvoiceDate", 4, "Create date"],
        ["pkey", "InvoiceNo", 0, "Primary Key"]
    ])(
        "Should preselect column property in UI when %s defined in tableArgs",
        async (property, columnName, columnIndex, expectedPropertyLabel) => {
            await setupAndRenderWithData({
                setTableArgs,
                tableArgs: {
                    ...tableWithConnector,
                    [property]: columnName
                }
            });

            const propertiesDropdowns = await screen.findAllByTestId(
                /^testid-properties-select-\d+$/
            );
            expect(propertiesDropdowns[columnIndex]).toHaveTextContent(
                expectedPropertyLabel
            );

            if (property === "pkey") {
                const sTypeDropdowns = screen.getAllByTestId(
                    "testid-select-s-type"
                );
                expect(sTypeDropdowns[columnIndex]).toHaveTextContent("ID");
            }
        }
    );

    it("Should display documentation link", async () => {
        await renderPreprocessing(setupTestWithConnector());

        await clickSkipButton();

        const docLink = await screen.findByTestId(
            "testid-column_preprocessing_doc"
        );
        expect(docLink).toBeInTheDocument();
        expect(docLink).toHaveTextContent("docs");

        expect(docLink).toHaveAttribute(
            "href",
            "https://docs.nvidia.com/sdgm/kumo-engine/1.0.0/kumo-studio-guide-coming-soon/preliminary-documentation/connect-data/column-preprocessing"
        );

        expect(screen.getByText(/See/)).toBeInTheDocument();
        expect(screen.getByText(/for more details/)).toBeInTheDocument();
    });

    it("Should display `Request Failed!` alert when sample rows API returns error", async () => {
        setSampleRowsResponseType(testResponseType.ERROR);

        await renderPreprocessing(setupTestWithConnector());

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const alert = screen.getByRole("alert");
        expect(alert).toHaveTextContent("Request Failed!");

        const closeButton = screen.getByTestId("alert-close");
        await userEvent.click(closeButton);

        await waitFor(() => {
            expect(
                screen.queryByText("Request Failed!")
            ).not.toBeInTheDocument();
        });
    });

    it("Should display skeleton loader when sample rows API returns empty response", async () => {
        setSampleRowsResponseType(testResponseType.EMPTY);

        await renderPreprocessing(setupTestWithConnector());

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        await waitFor(() => {
            expect(screen.getByTestId("table-header")).toBeInTheDocument();
        });

        await waitFor(() => {
            expect(
                document.querySelector('[data-slot="skeleton"]')
            ).toBeInTheDocument();
        });
    });

    it("Should display DataTypeSuggestions warning for categorical columns with float dtype", async () => {
        const tableArgsWithFloatCategorical: KumoapiTableTableDefinition = {
            ...tableWithConnector,
            cols: [
                {
                    name: "category_col",
                    dtype: "float64",
                    stype: "categorical",
                    timestamp_format: undefined
                }
            ]
        };

        await renderPreprocessing(
            setupTestWithConnector({ tableArgs: tableArgsWithFloatCategorical })
        );

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const alert = screen.getAllByRole("alert");
        expect(alert.length).toBeGreaterThan(0);
        expect(alert[0]).toHaveTextContent("Data type suggestion");

        await waitFor(() => {
            expect(
                screen.getByText(
                    /Use of data type .* for column "category_col"/i
                )
            ).toBeInTheDocument();
            expect(
                screen.getByText(/may result in data mismatch/i)
            ).toBeInTheDocument();
        });

        const closeButton = screen.getByTestId("alert-close");
        await userEvent.click(closeButton);
        await waitFor(() => {
            expect(
                screen.queryByText("Data type suggestion")
            ).not.toBeInTheDocument();
        });
    });

    it("Should display and handle invalidKeyError", async () => {
        const handleError = vi.fn();
        setColumnResponseType(testResponseType.INVALID);

        await renderPreprocessing(
            setupTestWithConnector({ handleError, selectFromTableArgs: true })
        );

        await clickSkipButton();

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should handle select all columns checkbox", async () => {
        await renderPreprocessing(setupTestWithConnector({ setTableArgs }));

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const checkboxes = screen.getAllByRole("checkbox");
        const selectAllCheckbox = checkboxes[0];

        await userEvent.click(selectAllCheckbox);

        await waitFor(() => {
            expect(setTableArgs).toHaveBeenCalled();
        });
    });

    it("Should not display column warnings when columnWarnings is null", async () => {
        await renderPreprocessing(setupTestWithConnector());

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const alerts = screen.queryAllByRole("alert");
        const warningAlert = alerts.find((alert) =>
            /Column Type Change/i.test(alert.textContent || "")
        );

        expect(warningAlert).toBeUndefined();
    });

    it("Should not display column warnings when warnings array is empty", async () => {
        await renderPreprocessing(setupTestWithConnector());

        await clickSkipButton();

        await callSampleRowsAndGetColumns(connectorId, {
            name: selectedSourceTableName
        });

        const alerts = screen.queryAllByRole("alert");
        const warningAlert = alerts.find((alert) =>
            /Column Type Change/i.test(alert.textContent || "")
        );

        expect(warningAlert).toBeUndefined();
    });
});
