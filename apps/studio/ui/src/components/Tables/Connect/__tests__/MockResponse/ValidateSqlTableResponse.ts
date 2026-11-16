// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SqlTableResolveResponse } from "gen/v1/types";

// Columns mirror the discovery mock (`ColumnResponse`) names so validation's
// authoritative stypes land on the discovered rows. `StockCode` flips from the
// discovered "ID" to "categorical", which lets tests assert the override.
export const mockValidateSqlTableResponse: SqlTableResolveResponse = {
    table_definition: {
        cols: [
            { name: "InvoiceNo", dtype: "string", stype: "categorical" },
            { name: "StockCode", dtype: "string", stype: "categorical" },
            {
                name: "Description",
                dtype: "string",
                stype: "multicategorical",
                multicategorical_delimiter: "|"
            },
            { name: "Quantity", dtype: "int64", stype: "numerical" },
            {
                name: "InvoiceDate",
                dtype: "string",
                stype: "timestamp",
                timestamp_format: "yyyy-MM-dd"
            },
            { name: "UnitPrice", dtype: "float64", stype: "numerical" },
            { name: "CustomerID", dtype: "float64", stype: "numerical" },
            { name: "Country", dtype: "string", stype: "categorical" }
        ],
        source_table: {
            connector_id: "connector-1",
            connector_type: "snowflake",
            sql: "SELECT * FROM customers",
            table: "custom_sql_mock",
            // "orders" is absent from the query text, so it uniquely
            // identifies the rendered referenced-table badge in tests.
            referenced_tables: ["customers", "orders"],
            data_source_type: "CUSTOM_SQL"
        }
    },
    errors: [],
    // Bounded preview (row-major, aligned to `columns`) returned alongside the
    // schema (Q2), so no separate sample fetch is needed.
    preview_rows: [
        [
            "536365",
            "85123A",
            "WHITE HANGING HEART",
            "6",
            "2010-12-01",
            "2.55",
            "17850",
            "United Kingdom"
        ],
        [
            "536366",
            "71053",
            "WHITE METAL LANTERN",
            "6",
            "2010-12-01",
            "3.39",
            "17850",
            "United Kingdom"
        ]
    ],
    preview_note: null
};

// Cost-gated result: the schema is confirmed but the query was too large for a
// preview, so the backend returns no rows and a note explaining the skip.
export const mockCostGatedValidateResponse: SqlTableResolveResponse = {
    ...mockValidateSqlTableResponse,
    preview_rows: [],
    preview_note: "Preview skipped — query too large."
};

// Empty column set: the backend marks the definition invalid and asks the user
// to run column discovery first.
export const mockEmptyColumnsValidateResponse: SqlTableResolveResponse = {
    errors: [
        {
            title: "Invalid output schema",
            message:
                "The query returned no columns. Run column discovery first, " +
                "then validate."
        }
    ]
};

// Message the backend raises when the query path is used on a scoped (RBAC
// group) connector.
export const scopedConnectorErrorMessage =
    "Custom SQL Table authoring is not supported on scoped (RBAC group) " +
    "connectors.";
