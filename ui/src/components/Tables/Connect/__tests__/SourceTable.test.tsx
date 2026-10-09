// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createMockResponse, createQueryClient } from "tests/testUtil";

import { expectedConnectorListResponse } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";

import { ConnectorDropdownOption, InputState } from "types/inputs";
import { FlowType } from "types/tables";
import { HandleErrorFunc } from "types/validation";

import { storageKeys } from "utils/Constants";

import SourceTable, { SourceTableProps } from "../SourceTable";

import { setSourceTablesResponsesType } from "./MockFunction/SourceTables.mock";

const connectors = expectedConnectorListResponse().map((c) => ({
    value: c.id,
    label: c.name,
    type: c.config.type,
    config: c.config
})) as ConnectorDropdownOption[];

const connectorId = String(connectors[0].value ?? "");
const onTableChange = vi.fn();
const setLoading = vi.fn();

const defaultProps: SourceTableProps = {
    onTableChange,
    nameValidationState: InputState.None,
    selectedConnector: connectors[0],
    flowType: FlowType.CREATE,
    handleError: vi.fn() as HandleErrorFunc,
    setLoading,
    connectorId,
    sourceTableName: "",
    resetTableSelection: vi.fn()
};

describe("SourceTable", () => {
    const queryClient = createQueryClient();
    const renderSourceTable = async (props: SourceTableProps) => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <SourceTable {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const checkApiCall = async () => {
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/source-tables`
            );
        });
    };

    const sourceTableDropdownClick = async () => {
        const dropdown = screen.getByTestId("testid-select-source-table");
        expect(dropdown).toBeInTheDocument();
        expect(dropdown).toBeEnabled();
        await userEvent.click(dropdown);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
        setSourceTablesResponsesType(testResponseType.ALL);
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");
    });

    it("Should enable source table selection when nameValidationState is Valid, Invalid, Warning, ValidationError, or None", async () => {
        await renderSourceTable({
            ...defaultProps,
            nameValidationState: InputState.None
        });

        const dropdown = screen.getByTestId("testid-select-source-table");
        expect(dropdown).toBeEnabled();
    });

    it("Should disable source table selection when nameValidationState is `Validating`", async () => {
        await renderSourceTable({
            ...defaultProps,
            nameValidationState: InputState.Validating
        });

        const dropdown = screen.getByTestId("testid-select-source-table");
        expect(dropdown).toBeDisabled();
    });

    it("Should disable source table selection in EDIT flow type", async () => {
        await renderSourceTable({
            ...defaultProps,
            nameValidationState: InputState.Validating,
            flowType: FlowType.EDIT
        });

        const dropdown = screen.getByTestId("testid-select-source-table");
        expect(dropdown).toBeDisabled();
    });

    it("Should show source table name as selected in `EDIT` flow type", async () => {
        const sourceTableName = "transaction";
        await renderSourceTable({
            ...defaultProps,
            flowType: FlowType.EDIT,
            sourceTableName: sourceTableName
        });

        expect(screen.getByRole("button")).toHaveTextContent(sourceTableName);
    });

    it("shows a selected exact path that is absent from discovery", async () => {
        const sourceTableName = "nested/dataset/events";
        await renderSourceTable({ ...defaultProps, sourceTableName });

        expect(
            screen.getByTestId("testid-select-source-table")
        ).toHaveTextContent(sourceTableName);
    });

    it("Should show selected source table and keep dropdown enabled in `DUPLICATE` flow type", async () => {
        const sourceTableName = "transaction";
        await renderSourceTable({
            ...defaultProps,
            flowType: FlowType.DUPLICATE,
            setLoading: (isLoading) => expect(isLoading).toBe(false),
            sourceTableName
        });

        expect(screen.getByText("Source table")).toBeInTheDocument();

        const dropdown = screen.getByTestId("testid-select-source-table");
        expect(dropdown).toBeInTheDocument();
        expect(dropdown).toBeDisabled();
    });
    it("Should update and display selected source table name after selection in `CREATE` flow type", async () => {
        await renderSourceTable({
            ...defaultProps,
            setLoading: (isLoading) => expect(isLoading).toBe(false)
        });

        expect(screen.getByText("Source table")).toBeInTheDocument();

        await sourceTableDropdownClick();

        const sourceTableToCheck = ["customer", "stock", "transaction"];
        const sourceTableList = await screen.findByRole("listbox");
        sourceTableToCheck.forEach((sourceTable) => {
            expect(sourceTableList).toHaveTextContent(sourceTable);
        });

        const sourceTableOption = screen.getByTestId(
            "select-source-table-option-0"
        );
        await userEvent.click(sourceTableOption);
        expect(onTableChange).toHaveBeenCalledWith(
            expect.objectContaining({ label: "customer" })
        );
        expect(sourceTableList).not.toBeInTheDocument();
    });

    it("accepts an arbitrary-depth relative path for an object-store connector", async () => {
        await renderSourceTable(defaultProps);

        const pathInput = screen.getByLabelText("Or enter a table path");
        await userEvent.type(
            pathInput,
            "2026/07/daily/events/transactions.parquet"
        );
        await userEvent.click(screen.getByRole("button", { name: "Use path" }));

        await waitFor(() => {
            expect(onTableChange).toHaveBeenCalledWith({
                value: "2026/07/daily/events/transactions.parquet",
                label: "2026/07/daily/events/transactions.parquet",
                valid: true,
                connectorId,
                fileType: "PARQUET",
                objectStoreSource: {
                    mode: "relative",
                    path: "2026/07/daily/events/transactions.parquet"
                }
            });
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v0/projects/project-tests/connectors/${connectorId}/source-table-config`,
                undefined,
                expect.objectContaining({
                    params: {
                        table_name: "2026/07/daily/events/transactions.parquet"
                    }
                })
            );
        });
    });

    it("skips discovery for an any-URI GCS connector", async () => {
        const anyUriGcs: ConnectorDropdownOption = {
            value: "any-uri-gcs",
            label: "Any URI GCS",
            config: {
                type: "gcs",
                target: { mode: "any_uri" }
            }
        };
        await renderSourceTable({
            ...defaultProps,
            selectedConnector: anyUriGcs,
            connectorId: "any-uri-gcs"
        });

        expect(
            screen.getByPlaceholderText("gs://bucket/deep/path/table.parquet")
        ).toBeInTheDocument();
        expect(screen.getByTestId("testid-select-source-table")).toBeDisabled();
        expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
            "v0/projects/project-tests/connectors/any-uri-gcs/source-tables"
        );
    });

    it.each([
        {
            provider: "GCS",
            connector: connectors[5],
            uri: "gs://training-data/raw/2026/07/deep/examples.csv",
            expected: "2026/07/deep/examples.csv"
        },
        {
            provider: "ADLS",
            connector: connectors[6],
            uri: "abfss://curated@companydata.dfs.core.windows.net/orders/2026/07/deep/facts.parquet",
            expected: "2026/07/deep/facts.parquet"
        }
    ])(
        "normalizes an arbitrary-depth full $provider URI",
        async ({ connector, uri, expected }) => {
            const id = String(connector.value);
            await renderSourceTable({
                ...defaultProps,
                selectedConnector: connector,
                connectorId: id
            });

            await userEvent.type(
                screen.getByLabelText("Or enter a table path"),
                uri
            );
            await userEvent.click(
                screen.getByRole("button", { name: "Use path" })
            );

            await waitFor(() => {
                expect(onTableChange).toHaveBeenCalledWith(
                    expect.objectContaining({
                        value: expected,
                        connectorId: id
                    })
                );
            });
        }
    );

    it("normalizes a full same-root URI before selecting the table", async () => {
        await renderSourceTable(defaultProps);

        await userEvent.type(
            screen.getByLabelText("Or enter a table path"),
            "s3://kumo-public-datasets/customerltv_integ_test/2026/07/events.csv"
        );
        await userEvent.click(screen.getByRole("button", { name: "Use path" }));

        await waitFor(() => {
            expect(onTableChange).toHaveBeenCalledWith(
                expect.objectContaining({
                    value: "2026/07/events.csv",
                    label: "2026/07/events.csv",
                    fileType: "CSV"
                })
            );
        });
    });

    it("uses the server-detected file type for a directory dataset", async () => {
        await renderSourceTable(defaultProps);

        await userEvent.type(
            screen.getByLabelText("Or enter a table path"),
            "2026/deep/directory-dataset"
        );
        await userEvent.click(screen.getByRole("button", { name: "Use path" }));

        await waitFor(() => {
            expect(onTableChange).toHaveBeenCalledWith(
                expect.objectContaining({
                    value: "2026/deep/directory-dataset",
                    fileType: "CSV"
                })
            );
        });
    });

    it("disables table selection while an exact path lookup is in flight", async () => {
        await renderSourceTable(defaultProps);
        await checkApiCall();

        let resolveRequest!: (value: unknown) => void;
        const pendingRequest = new Promise<unknown>((resolve) => {
            resolveRequest = resolve;
        });
        mockAxiosInstance.get.mockReturnValueOnce(pendingRequest as never);
        await userEvent.type(
            screen.getByLabelText("Or enter a table path"),
            "deep/directory-dataset"
        );
        await userEvent.click(screen.getByRole("button", { name: "Use path" }));

        expect(screen.getByTestId("testid-select-source-table")).toBeDisabled();
        expect(screen.getByRole("button", { name: "Use path" })).toBeDisabled();

        resolveRequest(
            createMockResponse({
                source_table: {
                    connector_id: connectorId,
                    source: {
                        mode: "relative",
                        path: "deep/directory-dataset"
                    },
                    file_type: "PARQUET",
                    data_source_type: "S3"
                }
            })
        );
        await waitFor(() => {
            expect(onTableChange).toHaveBeenCalledWith(
                expect.objectContaining({ fileType: "PARQUET" })
            );
        });
    });

    it("does not select a manual path when exact validation fails", async () => {
        const handleError = vi.fn();
        await renderSourceTable({ ...defaultProps, handleError });
        await checkApiCall();
        mockAxiosInstance.get.mockRejectedValueOnce(
            new Error("table is not readable")
        );

        await userEvent.type(
            screen.getByLabelText("Or enter a table path"),
            "deep/missing-table"
        );
        await userEvent.click(screen.getByRole("button", { name: "Use path" }));

        await waitFor(() => {
            expect(handleError).toHaveBeenCalledOnce();
            expect(
                screen.getByText("The table path could not be validated.")
            ).toBeInTheDocument();
        });
        expect(onTableChange).not.toHaveBeenCalled();
    });

    it("shows invalid manual paths and clears that state when a discovered table is selected", async () => {
        await renderSourceTable(defaultProps);

        const pathInput = screen.getByLabelText("Or enter a table path");
        await userEvent.type(pathInput, "../outside.csv");
        await userEvent.click(screen.getByRole("button", { name: "Use path" }));
        expect(
            screen.getByText(
                "Path cannot contain empty, current, or parent components."
            )
        ).toBeInTheDocument();

        await sourceTableDropdownClick();
        await userEvent.click(
            await screen.findByTestId("select-source-table-option-0")
        );

        expect(pathInput).toHaveValue("");
        expect(
            screen.queryByText(
                "Table path cannot contain empty, current, or parent components."
            )
        ).not.toBeInTheDocument();
        expect(onTableChange).toHaveBeenCalledWith(
            expect.objectContaining({ label: "customer" })
        );
    });

    it("does not show manual path entry for a warehouse connector", async () => {
        await renderSourceTable({
            ...defaultProps,
            selectedConnector: connectors[1],
            connectorId: String(connectors[1].value)
        });

        expect(
            screen.queryByLabelText("Or enter a table path")
        ).not.toBeInTheDocument();
    });

    it("offers full-path entry without listing for a rootless S3 config", async () => {
        await renderSourceTable({
            ...defaultProps,
            selectedConnector: {
                value: "rootless-s3",
                label: "rootless-s3",
                type: "s3",
                config: { type: "s3", target: { mode: "any_uri" } }
            },
            connectorId: "rootless-s3"
        });

        const input = screen.getByPlaceholderText(
            "s3://bucket/deep/path/table.parquet"
        );
        await userEvent.type(
            input,
            "s3://remote-bucket/deep/path/table.parquet"
        );
        await userEvent.click(screen.getByRole("button", { name: "Use path" }));

        await waitFor(() =>
            expect(onTableChange).toHaveBeenCalledWith(
                expect.objectContaining({
                    value: "s3://remote-bucket/deep/path/table.parquet",
                    objectStoreSource: {
                        mode: "uri",
                        uri: "s3://remote-bucket/deep/path/table.parquet"
                    }
                })
            )
        );
        expect(mockAxiosInstance.get).not.toHaveBeenCalledWith(
            "v0/projects/project-tests/connectors/rootless-s3/source-tables"
        );
    });

    it("Should filter and display matching source table names on valid search", async () => {
        await renderSourceTable({
            ...defaultProps,
            setLoading: (isLoading) => expect(isLoading).toBe(false),
            connectorId
        });

        await checkApiCall();

        await sourceTableDropdownClick();

        const search = await screen.findByPlaceholderText("Search...");
        expect(search).toBeInTheDocument();
        const searchInput = "stock";
        await userEvent.type(search, searchInput);

        const connectorsToCheck = "stock";
        const connectorList = await screen.findByRole("listbox");
        expect(connectorList).toHaveTextContent(connectorsToCheck);
    });

    it("Should display 'No tables found.' on invalid search input", async () => {
        await renderSourceTable({
            ...defaultProps,
            nameValidationState: InputState.None
        });

        await sourceTableDropdownClick();

        const search = await screen.findByPlaceholderText("Search...");
        expect(search).toBeInTheDocument();
        const searchInput = "03052000%";
        await userEvent.type(search, searchInput);
        await waitFor(() => {
            expect(screen.getByText("No tables found.")).toBeInTheDocument();
        });
    });

    it("Should display 'No tables found.' when source table are not available", async () => {
        await renderSourceTable({
            ...defaultProps,
            connectorId: "",
            selectedConnector: {} as ConnectorDropdownOption,
            nameValidationState: InputState.None
        });

        await sourceTableDropdownClick();
        await waitFor(() => {
            expect(screen.getByText("No tables found.")).toBeInTheDocument();
        });
    });

    it("Should not call `handleError` when the source tables are fetched correctly", async () => {
        const handleError = vi.fn();
        await renderSourceTable({
            ...defaultProps,
            selectedConnector: {
                value: "ltv_test"
            } as ConnectorDropdownOption,
            handleError
        });

        await checkApiCall();
        await waitFor(() => {
            expect(handleError).not.toHaveBeenCalled();
        });
    });

    it("Should call `handleError` when fetching source table fails", async () => {
        setSourceTablesResponsesType(testResponseType.ERROR);
        const handleError = vi.fn();
        await renderSourceTable({
            ...defaultProps,
            selectedConnector: {
                value: "ltv_test"
            } as ConnectorDropdownOption,
            handleError
        });

        await checkApiCall();
        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should not fetch source tables when connectorId is missing", async () => {
        await renderSourceTable({
            ...defaultProps,
            connectorId: undefined
        });

        expect(mockAxiosInstance.get).not.toHaveBeenCalled();
    });

    it("Should close dropdown when clicking outside", async () => {
        await renderSourceTable({
            ...defaultProps,
            handleError: vi.fn()
        });

        await sourceTableDropdownClick();

        expect(screen.getByRole("listbox")).toBeInTheDocument();

        await userEvent.click(document.body); // click outside
        await waitFor(() => {
            expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
        });
    });

    it("Should not render schema input for Databricks connector in EDIT mode", async () => {
        const connectorId = "test_databricks";
        await renderSourceTable({
            ...defaultProps,
            selectedConnector: connectors[6],
            connectorId,
            flowType: FlowType.EDIT
        });

        expect(screen.queryByText("Schema")).not.toBeInTheDocument();
        expect(screen.getByText("Source table")).toBeInTheDocument();
    });

    it("Should trigger API and call onTableChange for each selected table", async () => {
        await renderSourceTable(defaultProps);

        await checkApiCall();

        const dropdown = screen.getByTestId("testid-select-source-table");
        await userEvent.click(dropdown);

        const firstOption = await screen.findByTestId(
            "select-source-table-option-0"
        );
        await userEvent.click(firstOption);

        await checkApiCall();

        expect(onTableChange).toHaveBeenCalledWith(
            expect.objectContaining({ label: "customer" })
        );

        await userEvent.click(dropdown);

        const secondOption = screen.getByTestId("select-source-table-option-1");
        await userEvent.click(secondOption);

        await checkApiCall();

        expect(onTableChange).toHaveBeenCalledWith(
            expect.objectContaining({ label: "stock" })
        );
    });
});
