// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { focusManager, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import { mockAxiosInstance } from "tests/mockAxios";
import { mockNavigateCall, mockUseNavigate } from "tests/mockRouter";
import {
    clickWithoutPointerCheck,
    hoverWithoutPointerCheck
} from "tests/testHelpers";
import { createQueryClient } from "tests/testUtil";

import { getConnectorResponseById } from "pages/Connectors/ConnectorsList/__tests__/MockFunctions/ConnectorList.mock";
import { mockGetColStats } from "../../../../pages/Tables/TablesList/__tests__/MockResponse/Colstats";
import { mockGetTablesDetail } from "../../../../pages/Tables/TablesList/__tests__/MockResponse/TableDetail.test.response";

import { JsConnector } from "types/connectors";
import { ExtendedColStatsResponse } from "types/snapshots";
import { DataSourceType, JSTable, TableKind } from "types/tables";

import { storageKeys } from "utils/Constants";
import { connectorListFromConnectorResp } from "utils/Converters/connectorsConverter";
import { parseDateTime } from "utils/timeConverter";

// Force the Custom SQL feature flag OFF for this suite so the detail panel keeps
// its prior layout (no SQL definition panel). The flag-on behavior is covered in
// TableOverviewCustomSql.test.tsx.
vi.mock("utils/Constants", async (importOriginal) => {
    const actual = await importOriginal<typeof import("utils/Constants")>();
    return { ...actual, ENABLE_CUSTOM_SQL_TABLES: false };
});

import TableOverview, { TableOverviewProps } from "../TableOverview";

const mockTableInfo = {
    id: mockGetTablesDetail?.id,
    name_alias: mockGetTablesDetail?.name_alias,
    created: mockGetTablesDetail?.timestamps?.created_at,
    dateCreated: parseDateTime(mockGetTablesDetail?.timestamps?.created_at),
    updated: mockGetTablesDetail?.timestamps?.updated_at,
    connectorId: mockGetTablesDetail?.connector?.id,
    num_cols: mockGetTablesDetail?.table?.cols?.length,
    num_rows: mockGetTablesDetail?.stats?.num_rows,
    size: `${(mockGetTablesDetail?.stats?.size_bytes / 1_000_000).toFixed(1)}MB`,
    source_table: {
        connector_id: "ltv_test",
        data_source_type: DataSourceType.S3,
        file_type: "PARQUET",
        source: { mode: "relative", path: "transaction" }
    },
    s3Connector: {
        id: mockGetTablesDetail?.table?.source_table?.connector_id,
        name: "ltv_test"
    },
    pkey: "",
    timeCol: "",
    endTimeCol: "",
    timeRange: `${mockGetTablesDetail?.time_range?.min?.split("T")[0]} to ${mockGetTablesDetail?.time_range?.max?.split("T")[0]}`
} as unknown as JSTable;

const defaultProps: TableOverviewProps = {
    actionItems: [],
    columnStats: null,
    fetchingSamples: false,
    tableInfo: mockTableInfo,
    loadingStatus: false
};

describe("TableOverview", () => {
    const renderTableOverview = async (props: TableOverviewProps) => {
        const queryClient = createQueryClient();
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={queryClient}>
                        <TableOverview {...props} />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const hoverAndExpectTooltip = async (
        testId: string,
        expectations: Array<string | RegExp>
    ) => {
        const element = screen.getByTestId(testId);
        await hoverWithoutPointerCheck(element);

        for (const expectation of expectations) {
            await waitFor(() => {
                const matches = screen.queryAllByText(expectation);
                expect(matches.length).toBeGreaterThan(0);
            });
        }
    };

    const testConnectorTooltip = async (
        tableInfo: JSTable,
        connectorConfig: JsConnector | undefined,
        expectedTexts: Array<string | RegExp>
    ) => {
        await renderTableOverview({
            ...defaultProps,
            tableInfo,
            connectorConfig
        });

        await hoverAndExpectTooltip("table-connector", expectedTexts);
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
        focusManager.setFocused(false);
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");
    });

    afterEach(() => {
        focusManager.setFocused(undefined);
    });

    it("Should not render the SQL definition panel when the feature flag is off", async () => {
        await renderTableOverview({
            ...defaultProps,
            tableInfo: {
                ...mockTableInfo,
                table_kind: TableKind.Sql,
                viewSql: "SELECT 1"
            } as JSTable
        });
        expect(screen.queryByText("SQL Definition")).not.toBeInTheDocument();
    });

    it.each([
        { tableInfo: null, scenario: "null" },
        { tableInfo: {} as JSTable, scenario: "empty" }
    ])(
        "Should display 'No data found' when tableInfo is $scenario",
        async ({ tableInfo }) => {
            await renderTableOverview({
                ...defaultProps,
                tableInfo
            });

            expect(
                await screen.findByText("No data found.")
            ).toBeInTheDocument();
        }
    );

    it("Should display `No data found.` for `Invalid` tableInfo response", async () => {
        const invalidMockGetTablesDetail = {
            id: "table-001",
            name_alias: "Broken Table",
            timestamps: {
                created_at: "2023-01-01T00:00:00Z",
                updated_at: "2023-01-01T00:00:00Z"
            },
            connector: { id: "conn-001" },
            table: {
                cols: null,
                source_table: {
                    connector_id: "ltv_test"
                }
            },
            stats: null,
            time_range: {
                min: "2020-01-01T00:00:00Z",
                max: "2020-12-31T00:00:00Z"
            }
        } as unknown as JSTable;
        await renderTableOverview({
            ...defaultProps,
            columnStats: {} as ExtendedColStatsResponse,
            tableInfo: invalidMockGetTablesDetail,
            connectorConfig: {} as JsConnector
        });

        const tableData = screen.getByText("No data found.");
        expect(tableData).toBeInTheDocument();
    });

    it("Should display `table skeleton` when loadingStatus is set true", async () => {
        await renderTableOverview({
            ...defaultProps,
            loadingStatus: true
        });

        const skeleton = document.querySelector('[data-slot="skeleton"]');
        expect(skeleton).toBeInTheDocument();
    });

    it("Should render metadata label/value pairs for valid tableInfo response", async () => {
        await renderTableOverview(defaultProps);

        expect(
            screen.getByText(`${mockTableInfo.name_alias}`)
        ).toBeInTheDocument();

        // Metadata is rendered as label/value pairs in the new card style.
        expect(screen.getByText("Created")).toBeInTheDocument();
        expect(screen.getByText(mockTableInfo.created!)).toBeInTheDocument();

        expect(screen.getByText("Dimension")).toBeInTheDocument();
        expect(
            screen.getByText(
                `${mockTableInfo.num_cols} columns × ${mockTableInfo.num_rows} rows · ${mockTableInfo.size}`
            )
        ).toBeInTheDocument();

        const connector = screen.getByTestId("table-connector");
        expect(connector.textContent).toBe(`${mockTableInfo.connectorId}`);

        expect(screen.getByText("Time Range")).toBeInTheDocument();
        expect(
            screen.getByText(`${mockTableInfo.timeRange}`)
        ).toBeInTheDocument();
    });

    it("Should render Column stats table and skeleton loader in cells", async () => {
        const colStatsResponse = { ...mockGetColStats };
        const resp = getConnectorResponseById(`ltv_test`);
        const conInfo = connectorListFromConnectorResp([resp], false)[0];

        await renderTableOverview({
            ...defaultProps,
            columnStats: colStatsResponse as ExtendedColStatsResponse,
            fetchingSamples: true,
            tableInfo: mockTableInfo,
            connectorConfig: conInfo
        });

        expect(screen.getByText("Column Statistics")).toBeInTheDocument();
        expect(
            screen.getByText(
                "The following stats are computed from a sample of table data."
            )
        ).toBeInTheDocument();

        const skeletonLoaders = screen
            .getAllByRole("generic", {
                name: ""
            })
            .filter((el) => el.getAttribute("data-slot") === "skeleton");

        expect(skeletonLoaders.length).toBeGreaterThan(0);

        await waitFor(() => {
            const table = screen.getByTestId("table-header");
            expect(table).toBeInTheDocument();
        });

        await waitFor(() => {
            const tableHeaders = screen.getAllByRole("columnheader");
            expect(tableHeaders).toHaveLength(6); //including the column with chevron
            expect(tableHeaders[0]).toHaveTextContent("Column");
            expect(tableHeaders[1]).toHaveTextContent("Data Type");
            expect(tableHeaders[2]).toHaveTextContent("Semantic Type");
            expect(tableHeaders[3]).toHaveTextContent("Missing");
            expect(tableHeaders[4]).toHaveTextContent("Invalid");

            const rows = screen.getAllByRole("row");
            expect(rows.length).toBeGreaterThan(0);
        });
    });

    it("Should display `No data found.` under Column Stats when the `fetchingSamples` is set as false", async () => {
        await renderTableOverview(defaultProps);

        const table = screen.getByTestId("table-header").closest("table");
        expect(within(table!).getByText("No data found.")).toBeInTheDocument();
    });

    it("Should handle missing `num_cols` or `num_rows` or `size` without crashing", async () => {
        const partialInfo = {
            ...mockTableInfo,
            num_cols: undefined,
            num_rows: undefined,
            size: undefined
        };

        await renderTableOverview({
            ...defaultProps,
            tableInfo: partialInfo
        });

        expect(screen.queryByTestId("table-size")).not.toBeInTheDocument();
    });

    it("Should handle missing `created` without crashing", async () => {
        const partialInfo = {
            ...mockTableInfo,
            created: undefined
        };

        await renderTableOverview({
            ...defaultProps,
            tableInfo: partialInfo
        });

        expect(screen.queryByTestId("Created")).not.toBeInTheDocument();
    });

    it("Should handle missing `updated` without crashing and not show 'Last edited' in tooltip", async () => {
        const partialInfo = {
            ...mockTableInfo,
            updated: undefined
        };

        await renderTableOverview({
            ...defaultProps,
            tableInfo: partialInfo
        });

        const createdValue = screen.getByText(mockTableInfo.created!);
        await hoverWithoutPointerCheck(createdValue);

        await waitFor(() => {
            expect(screen.getByText("Created")).toBeInTheDocument();
            expect(
                screen.getAllByText(
                    parseDateTime(mockGetTablesDetail?.timestamps?.created_at)
                ).length
            ).toBeGreaterThan(0);
            expect(screen.queryByText(/Last edited/i)).not.toBeInTheDocument();
        });
    });

    it("Should show tooltip content on hover over 'Created' value & 'Connector'", async () => {
        await renderTableOverview(defaultProps);

        // Created value reveals the formatted creation date on hover.
        const createdValue = screen.getByText(mockTableInfo.created!);
        await hoverWithoutPointerCheck(createdValue);
        await waitFor(() => {
            expect(
                screen.getAllByText(mockTableInfo.dateCreated!).length
            ).toBeGreaterThan(0);
        });

        expect(screen.getByTestId("table-connector")).toHaveTextContent(
            "ltv_test"
        );
    });

    it("Should handle an S3 source when the connector ID display field is empty", async () => {
        const s3TableInfo: JSTable = {
            ...mockTableInfo,
            connectorId: "",
            source_table: {
                connector_id: "s3_conn_test",
                data_source_type: DataSourceType.S3,
                file_type: "PARQUET",
                source: { mode: "relative", path: "test/table" }
            },
            s3Connector: {
                id: "s3_conn_test",
                name: "s3_connector"
            }
        };

        await renderTableOverview({
            ...defaultProps,
            tableInfo: s3TableInfo,
            connectorConfig: {
                id: "s3_conn_test",
                name: "s3_connector",
                source: "S3",
                config: {
                    type: "s3",
                    target: {
                        mode: "rooted",
                        bucket: "kumo-public-datasets",
                        prefix: "test/table"
                    }
                }
            } as JsConnector
        });

        const connector = screen.getByTestId("table-connector");
        expect(connector.textContent).toBe("s3_conn_test");

        await hoverAndExpectTooltip("table-connector", [
            /Path/i,
            "../test/table"
        ]);
    });

    it("Should display Snowflake connector details in tooltip", async () => {
        const snowflakeTable: JSTable = {
            ...mockTableInfo,
            source_table: {
                data_source_type: DataSourceType.SnowFlake,
                snowflake_connector_id: "mock-connector-id",
                database: "mock_db",
                schema_name: "mock_schema",
                table: "mock_table"
            }
        };

        const snowflakeConnector: JsConnector = {
            id: "snowflake_conn",
            name: "Snowflake",
            config: {
                warehouse: "WH_XS",
                schema_name: "PUBLIC",
                database: "DB_TEST"
            }
        } as unknown as JsConnector;

        await testConnectorTooltip(snowflakeTable, snowflakeConnector, [
            /Warehouse/i,
            "WH_XS",
            /Schema Name/i,
            "PUBLIC",
            /Database/i,
            "DB_TEST"
        ]);
    });

    it("Should display BigQuery connector details in tooltip", async () => {
        const bqTable: JSTable = {
            ...mockTableInfo,
            source_table: {
                ...mockTableInfo.source_table,
                data_source_type: DataSourceType.BigQuery,
                bigquery_connector_id: "bq-connector-id",
                project_id: "my-gcp-project",
                dataset_id: "my_dataset",
                table_name: "my_table"
            }
        };

        const bqConnector: JsConnector = {
            id: "bq_conn",
            name: "BigQuery",
            config: {
                project_id: "my_project",
                dataset_id: "my_dataset"
            }
        } as unknown as JsConnector;

        await testConnectorTooltip(bqTable, bqConnector, [
            /Project/i,
            "my_project",
            /Dataset/i,
            "my_dataset"
        ]);
    });

    it("Should display Databricks connector details in tooltip", async () => {
        const dbxTable: JSTable = {
            ...mockTableInfo,
            source_table: {
                ...mockTableInfo.source_table,
                data_source_type: DataSourceType.Databricks,
                databricks_connector_id: "dbx-connector-id",
                table: "my_table"
            }
        };

        const dbxConnector: JsConnector = {
            id: "dbx_conn",
            name: "Databricks",
            config: {
                cluster_id: "cluster-123",
                warehouse_id: "wh-456",
                schema_name: "analytics",
                catalog: "main_catalog"
            }
        } as unknown as JsConnector;

        await testConnectorTooltip(dbxTable, dbxConnector, [
            /Cluster/i,
            "cluster-123",
            /Warehouse/i,
            "wh-456",
            /Schema/i,
            "analytics",
            /Catalog/i,
            "main_catalog"
        ]);
    });

    it("Should handle long connector IDs with truncation", async () => {
        const longConnectorId =
            "this-is-a-very-long-connector-id-that-exceeds-twenty-characters";
        const longIdTable: JSTable = {
            ...mockTableInfo,
            connectorId: longConnectorId
        };

        await renderTableOverview({
            ...defaultProps,
            tableInfo: longIdTable
        });

        const connector = screen.getByTestId("table-connector");
        expect(connector.textContent).toBe(
            longConnectorId.substring(0, 20) + "..."
        );
    });

    it("Should handle connectors with null/undefined config values", async () => {
        const snowflakeTable: JSTable = {
            ...mockTableInfo,
            source_table: {
                data_source_type: DataSourceType.SnowFlake,
                snowflake_connector_id: "mock-connector-id",
                database: "mock_db",
                schema_name: "mock_schema",
                table: "mock_table"
            }
        };

        const snowflakeConnectorWithNulls: JsConnector = {
            id: "snowflake_conn",
            name: "Snowflake",
            config: {
                warehouse: undefined,
                schema_name: undefined,
                database: undefined
            }
        } as unknown as JsConnector;

        await renderTableOverview({
            ...defaultProps,
            tableInfo: snowflakeTable,
            connectorConfig: snowflakeConnectorWithNulls
        });

        expect(screen.getByTestId("table-connector")).toHaveTextContent(
            "Snowflake"
        );
    });

    it("Should render 'SDK Code' button and open SDK modal", async () => {
        const colStatsResponse = { ...mockGetColStats };
        const resp = getConnectorResponseById(`ltv_test`);
        const conInfo = connectorListFromConnectorResp([resp], false)[0];

        await renderTableOverview({
            ...defaultProps,
            columnStats: colStatsResponse as ExtendedColStatsResponse,
            fetchingSamples: true,
            tableInfo: {
                ...mockTableInfo,
                id: "18e2b1a6e499fff45a6f249f3024662f"
            },
            connectorConfig: conInfo
        });

        const dropdownButton = screen.getByTestId("more-options-dropdown");
        await clickWithoutPointerCheck(dropdownButton);

        const sdkCodeItem = screen.getByText(/SDK Code/i);
        expect(sdkCodeItem).toBeInTheDocument();

        await clickWithoutPointerCheck(sdkCodeItem);

        const tableId = "NA.NA-18e2b1a6e499fff45a6f249f3024662f";
        await waitFor(() => {
            expect(mockAxiosInstance.get).toHaveBeenCalledWith(
                `v1/tables/${tableId}/sdk`,
                undefined,
                {
                    headers: undefined,
                    params: { project_id: "project-tests" }
                }
            );
        });

        const modal = await screen.findByTestId("sdk-content");
        expect(modal).toBeInTheDocument();
        expect(modal).toHaveTextContent(/View SDK Code/i);

        expect(
            screen.getByRole("button", { name: "Copy" })
        ).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /Download/i })
        ).toBeInTheDocument();
        expect(screen.getByTestId("close-icon")).toBeInTheDocument();

        await clickWithoutPointerCheck(screen.getByTestId("close-icon"));

        await waitFor(() => {
            expect(screen.queryByTestId("sdk-content")).not.toBeInTheDocument();
        });
    });

    it("Should display a clickable connector link with proper styling inside the tooltip", async () => {
        await renderTableOverview(defaultProps);

        const connectorElement = screen.getByTestId("table-connector");
        expect(connectorElement.tagName).toBe("SPAN");
        expect(connectorElement.querySelectorAll("a")).toHaveLength(1);
        await hoverWithoutPointerCheck(connectorElement);

        expect(await screen.findByText(/CONNECTOR/i)).toBeInTheDocument();
        const connectorName = screen.getByRole("link", {
            name: `${mockTableInfo.connectorId}`
        });

        expect(connectorName).toHaveAttribute(
            "href",
            `/project/project-tests/connectors/${mockTableInfo.connectorId}`
        );
        expect(connectorName).toHaveAttribute(
            "id",
            `connecter-${mockTableInfo.connectorId}`
        );

        await clickWithoutPointerCheck(connectorName);

        expect(mockUseNavigate).toHaveBeenCalledWith(
            "/project/project-tests/connectors/" +
                `${mockTableInfo.connectorId}`
        );
    });

    it("Should display a local upload without linking the hidden connector", async () => {
        await renderTableOverview({
            ...defaultProps,
            isProjectUploadConnector: true
        });

        expect(screen.getByTestId("table-connector")).toHaveTextContent(
            "Local Upload"
        );
        expect(
            screen.queryByRole("link", {
                name: `${mockTableInfo.connectorId}`
            })
        ).not.toBeInTheDocument();
    });

    it("Should show S3 connector location details from the connector config", async () => {
        const s3TableInfo: JSTable = {
            ...mockTableInfo,
            connectorId: "",
            source_table: {
                connector_id: "s3_conn_test",
                data_source_type: DataSourceType.S3,
                file_type: "PARQUET",
                source: { mode: "relative", path: "test/table" }
            },
            s3Connector: {
                id: "s3_conn_test",
                name: "s3_connector"
            }
        };

        await renderTableOverview({
            ...defaultProps,
            tableInfo: s3TableInfo,
            connectorConfig: {
                id: "s3_conn_test",
                name: "s3_connector",
                source: "S3",
                config: {
                    type: "s3",
                    target: {
                        mode: "rooted",
                        bucket: "kumo-public-datasets",
                        prefix: "test/table"
                    }
                }
            } as JsConnector
        });

        const connector = screen.getByTestId("table-connector");
        expect(connector.textContent).toBe("s3_conn_test");

        await hoverWithoutPointerCheck(connector);

        await waitFor(() => {
            expect(screen.getByText(/Path/i)).toBeInTheDocument();
            expect(screen.getByText("../test/table")).toBeInTheDocument();
        });
    });

    it("shows the sovereign ADLS connector location", async () => {
        await renderTableOverview({
            ...defaultProps,
            tableInfo: {
                ...mockTableInfo,
                source_table: {
                    connector_id: "adls-connector",
                    data_source_type: DataSourceType.ADLS,
                    file_type: "PARQUET",
                    source: { mode: "relative", path: "tables/events" }
                }
            } as JSTable,
            connectorConfig: {
                id: "adls-connector",
                name: "ADLS Government",
                source: "ADLS",
                config: {
                    type: "adls",
                    target: {
                        mode: "rooted",
                        storage_account: "agencydata",
                        filesystem: "curated",
                        prefix: "prod",
                        endpoint_suffix: "dfs.core.usgovcloudapi.net"
                    }
                }
            } as JsConnector
        });

        await hoverWithoutPointerCheck(screen.getByTestId("table-connector"));
        const location = await screen.findByText("../prod");
        await hoverWithoutPointerCheck(location);
        expect(
            await screen.findAllByText(
                "abfss://curated@agencydata.dfs.core.usgovcloudapi.net/prod"
            )
        ).not.toHaveLength(0);
    });
});
