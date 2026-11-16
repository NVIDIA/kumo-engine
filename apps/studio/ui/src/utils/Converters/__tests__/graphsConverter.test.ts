// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { JSGraph } from "types/graphs";

import {
    formatToLongLocalDateString,
    formatToShortLocalDateString,
    parseDate
} from "utils/timeConverter";

import {
    jsGraphFromGraphResponse,
    tablesFromGraphResponse
} from "../graphsConverter";

import { mockGraphInfo, mockGraphResponse } from "./MockResponse/mockResponses";

describe("jsGraphFromGraphResponse", () => {
    it("Should correctly parse basic graph info with fullTimeStamp = false", () => {
        const result = jsGraphFromGraphResponse(mockGraphResponse, false);

        expect(result.id).toBe(mockGraphResponse.id);
        expect(result.name).toBe(mockGraphResponse.name_alias);
        expect(result.totalLinks).toBe(2);
        expect(result.totalTables).toBe(3);
        expect(result.tables).toBe(mockGraphResponse.graph?.tables);
        expect(result.groups).toBe(mockGraphResponse.graph?.col_groups);

        // Mapping gets prefixed
        expect(result.table_name_to_fpid_mapping!["transactions_2020"]).toBe(
            "NA.NA-d5828d994e8679119b05ec8f07de920a"
        );

        // Time parsing checks
        expect(result.created).toBe(
            parseDate(mockGraphResponse.timestamps.created_at)
        );
        expect(result.updated).toBe(
            parseDate(mockGraphResponse.timestamps.updated_at)
        );

        expect(result).not.toHaveProperty("dateCreated");
        expect(result).not.toHaveProperty("dateUpdated");
    });

    it("Should correctly parse timestamps and include full date fields with fullTimeStamp = true", () => {
        const result = jsGraphFromGraphResponse(mockGraphResponse, true);

        expect(result.created).toBe(
            formatToShortLocalDateString(
                mockGraphResponse.timestamps.created_at
            )
        );
        expect(result.updated).toBe(
            formatToLongLocalDateString(mockGraphResponse.timestamps.updated_at)
        );

        expect(result.dateCreated).toBe(
            formatToLongLocalDateString(mockGraphResponse.timestamps.created_at)
        );
        expect(result.dateUpdated).toBe(
            formatToLongLocalDateString(mockGraphResponse.timestamps.updated_at)
        );
    });

    it("Should handle empty table_name_to_fpid_mapping", () => {
        const response = {
            ...mockGraphResponse,
            table_name_to_fpid_mapping: {}
        };
        const result = jsGraphFromGraphResponse(response);

        expect(result.table_name_to_fpid_mapping).toEqual({});
    });

    it("Should not override FPID if it already contains '-'", () => {
        const response = {
            ...mockGraphResponse,
            table_name_to_fpid_mapping: {
                transactions_2020: "projId.fpId-uuid"
            }
        };
        const result = jsGraphFromGraphResponse(response);

        expect(result.table_name_to_fpid_mapping!["transactions_2020"]).toBe(
            "projId.fpId-uuid"
        );
    });
});

describe("tablesFromGraphResponse", () => {
    it("Should return a table array with correctly mapped values", () => {
        const result = tablesFromGraphResponse(mockGraphInfo);
        expect(result).toHaveLength(1);

        const table = result[0];
        expect(table.name_alias).toBe("artlcles");
        expect(table.id).toBe("artlcles");
        expect(table.columns).toEqual([
            {
                id: "article_id",
                name: "article_id",
                stype: "ID",
                dtype: "int64",
                value: 1
            },
            {
                id: "prod_name",
                name: "prod_name",
                stype: "text",
                dtype: "string",
                value: 1
            }
        ]);
        expect(table.connector).toBe("hnm_integ");
        expect(table.table_location).toBe("articles/hnm_integ");
        expect(table.pkey).toBe("article_id");
        expect(table.timeCol).toBe("");
        expect(table.endTimeCol).toBe("");
        expect(table.earliestFact).toBe("-");
        expect(table.latestFact).toBe("-");
    });

    it.each([
        {
            description: "Should return an empty array for null input",
            input: null
        },
        {
            description:
                "Should return an empty array when tables is undefined",
            input: {
                id: "graph_2",
                name: "Empty Graph",
                totalLinks: 0,
                totalTables: 0
            } as JSGraph
        }
    ])("$description", ({ input }) => {
        const result = tablesFromGraphResponse(input);
        expect(result).toEqual([]);
    });
});
