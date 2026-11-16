// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen } from "@testing-library/react";
import { mockNavigateCall } from "tests/mockRouter";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { JsConnector } from "types/connectors";
import { DataSourceType, JSTable, TableKind } from "types/tables";

// Enable the Custom SQL feature flag so the detail panel exposes the SQL badge
// and definition card.
vi.mock("utils/Constants", async (importOriginal) => {
    const actual = await importOriginal<typeof import("utils/Constants")>();
    return { ...actual, ENABLE_CUSTOM_SQL_TABLES: true };
});

import TableOverview, {
    ConnectorItemsSection,
    TableOverviewProps
} from "../TableOverview";

const sqlTableInfo = {
    id: "t1",
    name_alias: "my_sql_table",
    connectorId: "sf_conn",
    table_kind: TableKind.Sql,
    viewSql: "SELECT customer_id FROM customers",
    // Q3: the SQL definition panel reuses the entry's standard `updated`.
    updated: "2024-02-01",
    source_table: {
        data_source_type: DataSourceType.SnowFlake
    },
    pkey: "",
    timeCol: "",
    endTimeCol: ""
} as unknown as JSTable;

const defaultProps: TableOverviewProps = {
    actionItems: [],
    columnStats: null,
    fetchingSamples: false,
    tableInfo: sqlTableInfo,
    loadingStatus: false
};

describe("TableOverview - Custom SQL (flag on)", () => {
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

    beforeEach(() => {
        vi.clearAllMocks();
        mockNavigateCall();
    });

    it("Should render the SQL badge and definition card for a SQL table", async () => {
        await renderTableOverview(defaultProps);
        expect(screen.getByText("SQL")).toBeInTheDocument();
        expect(screen.getByText("SQL Definition")).toBeInTheDocument();
        expect(screen.getByText(/Last updated/)).toBeInTheDocument();
    });

    it("Should recognize a SQL table by its CUSTOM_SQL source when table_kind is absent", async () => {
        // The table GET response doesn't stamp `table_kind` yet, but the source
        // carries the defining SELECT — the badge + card must still render.
        await renderTableOverview({
            ...defaultProps,
            tableInfo: {
                ...sqlTableInfo,
                table_kind: undefined,
                source_table: { data_source_type: "CUSTOM_SQL" }
            } as unknown as JSTable
        });
        expect(screen.getByText("SQL")).toBeInTheDocument();
        expect(screen.getByText("SQL Definition")).toBeInTheDocument();
    });

    it("Should not render the SQL definition card for a raw table", async () => {
        await renderTableOverview({
            ...defaultProps,
            tableInfo: {
                ...sqlTableInfo,
                table_kind: TableKind.Raw,
                viewSql: undefined
            } as JSTable
        });
        expect(screen.queryByText("SQL Definition")).not.toBeInTheDocument();
    });
});

describe("ConnectorItemsSection - Custom SQL connector hover", () => {
    const dbxCustomSqlTable = {
        source_table: {
            data_source_type: "CUSTOM_SQL",
            connector_type: "databricks"
        }
    } as unknown as JSTable;

    const dbxConnector = {
        created: "2026-07-10",
        source: "DATABRICKS",
        config: {
            warehouse_id: "wh-abc",
            schema_name: "bakehouse",
            catalog: "samples",
            cluster_id: null
        }
    } as unknown as JsConnector;

    it("Should surface Created + Warehouse and omit Schema/Catalog", () => {
        const items = ConnectorItemsSection(dbxCustomSqlTable, dbxConnector);
        const byLabel = (label: string) =>
            items.find((item) => item.label === label);

        expect(byLabel("Created")?.value).toBe("2026-07-10");
        // The Warehouse is the actual compute the SELECT runs on, sourced from
        // the connector (Databricks -> warehouse_id).
        expect(byLabel("Warehouse")?.value).toBe("wh-abc");
        // Schema/Catalog are only unqualified-reference defaults for a Custom SQL
        // table, so they are intentionally omitted from the hover.
        expect(items.some((item) => item.label === "Schema")).toBe(false);
        expect(items.some((item) => item.label === "Schema Name")).toBe(false);
        expect(items.some((item) => item.label === "Catalog")).toBe(false);
    });

    it("Should read the Snowflake warehouse for a Snowflake-backed Custom SQL table", () => {
        const items = ConnectorItemsSection(
            {
                source_table: {
                    data_source_type: "CUSTOM_SQL",
                    connector_type: "snowflake"
                }
            } as unknown as JSTable,
            {
                created: "2026-07-10",
                source: "SNOWFLAKE",
                config: {
                    warehouse: "COMPUTE_WH",
                    schema_name: "s",
                    database: "d"
                }
            } as unknown as JsConnector
        );
        expect(items.find((item) => item.label === "Warehouse")?.value).toBe(
            "COMPUTE_WH"
        );
        expect(items.some((item) => item.label === "Schema Name")).toBe(false);
    });
});
