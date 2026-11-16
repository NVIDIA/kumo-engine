// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { vi } from "vitest";

import { setGraphCreateResponseType } from "pages/Graphs/__tests__/MockFunction/GraphCreate.test.mock";
import { setGraphValidateResponseType } from "pages/Graphs/__tests__/MockFunction/GraphValidate.test.mock";
import { graphDetail } from "pages/Graphs/__tests__/MockResponse/GraphDetail.response";

import { ColumnGroup, GraphFlowInfo, GraphTables } from "types/graphs";
import { FlowType } from "types/tables";

import { storageKeys } from "utils/Constants";
import { errorStatusCode } from "utils/Constants";

import {
    useBackwardLinks,
    useCreateGraph,
    useGraphValidation
} from "../newGraph";

import "tests/mockAxios";

vi.mock("utils/navigator", () => ({
    useCustomNavigation: vi.fn(() => vi.fn())
}));

let queryClient: QueryClient;

const handleError = vi.fn();

const mockGraphFlowInfo: GraphFlowInfo = {
    name: "test_graph",
    id: "",
    description: "",
    tables: graphDetail.graph.tables as GraphTables,
    selected_tables: Object.keys(graphDetail.graph.tables),
    selected_table_ids: [],
    flow_type: FlowType.CREATE
};

const mockGroups: ColumnGroup[] = [
    {
        columns: [
            { table_name: "transactions", col_name: "customer_id" },
            { table_name: "customers", col_name: "customer_id" }
        ]
    }
];

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    storeToLocalStorage(storageKeys.selectedProject, "project-tests");

    setGraphValidateResponseType();
    setGraphCreateResponseType();
});

afterEach(() => {
    queryClient.clear();
});

describe("useGraphValidation", () => {
    const renderUseGraphValidation = () => {
        return renderHook(() => useGraphValidation(), {
            wrapper: createWrapper(queryClient)
        });
    };

    const callValidateGraph = async (
        result: ReturnType<typeof renderUseGraphValidation>["result"]
    ) => {
        let response:
            | { errors: string[]; statusCode: errorStatusCode }
            | undefined;
        await act(async () => {
            response = await result.current.validateGraph(
                mockGraphFlowInfo,
                mockGroups
            );
        });
        return response;
    };

    it("Should provide validateGraph function", () => {
        const { result } = renderUseGraphValidation();

        expect(typeof result.current.validateGraph).toBe("function");
    });

    it("Should return success with no errors for valid graph", async () => {
        const { result } = renderUseGraphValidation();

        const response = await callValidateGraph(result);

        expect(response).toEqual({
            errors: [],
            statusCode: errorStatusCode.Success
        });
    });

    it.each([
        {
            description: "validation fails",
            responseType: testResponseType.INVALID,
            expectedStatus: errorStatusCode.UnProcessableEntityError
        },
        {
            description: "API errors occur",
            responseType: testResponseType.ERROR,
            expectedStatus: undefined
        }
    ])(
        "Should return errors when $description",
        async ({ responseType, expectedStatus }) => {
            setGraphValidateResponseType(responseType);

            const { result } = renderUseGraphValidation();

            const response = await callValidateGraph(result);

            if (expectedStatus) {
                expect(response?.statusCode).toBe(expectedStatus);
            }
            expect(response?.errors?.length).toBeGreaterThan(0);
        }
    );
});

describe("useCreateGraph", () => {
    const renderUseCreateGraph = (invalidateCache: boolean = true) => {
        return renderHook(() => useCreateGraph(invalidateCache), {
            wrapper: createWrapper(queryClient)
        });
    };

    const callCreateGraph = async (
        result: ReturnType<typeof renderUseCreateGraph>["result"],
        isEdit: boolean = false
    ) => {
        let response:
            | {
                  graphId: string | null;
                  error: string;
                  statusCode: errorStatusCode;
              }
            | undefined;
        await act(async () => {
            response = await result.current.createGraph(
                mockGraphFlowInfo,
                mockGroups,
                handleError,
                isEdit
            );
        });
        return response;
    };

    it("Should provide createGraph function", () => {
        const { result } = renderUseCreateGraph();

        expect(typeof result.current.createGraph).toBe("function");
    });

    it.each([
        { description: "create mode", isEdit: false },
        { description: "edit mode", isEdit: true }
    ])("Should handle graph creation in $description", async ({ isEdit }) => {
        const { result } = renderUseCreateGraph();

        const response = await callCreateGraph(result, isEdit);

        expect(response?.graphId).toBeTruthy();
        expect(response?.error).toBe("");
        expect(response?.statusCode).toBe(errorStatusCode.Success);
    });

    it("Should handle creation errors and call handleError", async () => {
        setGraphCreateResponseType(testResponseType.ERROR);

        const { result } = renderUseCreateGraph();

        const response = await callCreateGraph(result);

        expect(response?.graphId).toBeNull();
        expect(response?.error).toBeTruthy();
        expect(handleError).toHaveBeenCalledWith(
            expect.any(Object),
            "Graph Creation failed!"
        );
    });
});

describe("useBackwardLinks", () => {
    const renderUseBackwardLinks = (
        tables?: GraphTables,
        groups?: ColumnGroup[]
    ) => {
        return renderHook(() => useBackwardLinks(tables, groups), {
            wrapper: createWrapper(queryClient)
        });
    };

    const createTable = (
        name: string,
        cols: Array<{ name: string; stype: string; dtype: string }>,
        pkey: string | null
    ) =>
        ({
            cols,
            source_table: {
                connector_id: "test",
                source: { mode: "relative", path: name },
                data_source_type: "S3" as const,
                file_type: "PARQUET" as const
            },
            pkey
        }) as GraphTables[string];

    it("Should return empty backward links and isPkeyExists true when no tables", () => {
        const { result } = renderUseBackwardLinks();

        expect(result.current.backwardLinks).toEqual({});
        expect(result.current.isPkeyExists).toBe(true);
    });

    it("Should identify tables without pkeys", () => {
        const tablesWithoutPkey: GraphTables = {
            table1: createTable(
                "table1",
                [
                    { name: "id", stype: "ID", dtype: "string" },
                    { name: "value", stype: "numerical", dtype: "float64" }
                ],
                null
            )
        };

        const { result } = renderUseBackwardLinks(tablesWithoutPkey, []);

        expect(result.current.isPkeyExists).toBe(false);
        expect(result.current.backwardLinks).toHaveProperty("table1");
        expect(result.current.backwardLinks.table1).toEqual([]);
    });

    it("Should set isPkeyExists to true when at least one table has pkey", () => {
        const mixedTables: GraphTables = {
            table1: createTable(
                "table1",
                [{ name: "id", stype: "ID", dtype: "string" }],
                "id"
            ),
            table2: createTable(
                "table2",
                [{ name: "value", stype: "numerical", dtype: "float64" }],
                null
            )
        };

        const { result } = renderUseBackwardLinks(mixedTables, []);

        expect(result.current.isPkeyExists).toBe(true);
    });

    it("Should compute backward links from column groups", () => {
        const tables: GraphTables = {
            customers: createTable(
                "customers",
                [{ name: "customer_id", stype: "ID", dtype: "string" }],
                "customer_id"
            ),
            transactions: createTable(
                "transactions",
                [
                    { name: "id", stype: "ID", dtype: "string" },
                    { name: "customer_id", stype: "ID", dtype: "string" }
                ],
                "id"
            )
        };

        const groups: ColumnGroup[] = [
            {
                columns: [
                    { table_name: "customers", col_name: "customer_id" },
                    { table_name: "transactions", col_name: "customer_id" }
                ]
            }
        ];

        const { result } = renderUseBackwardLinks(tables, groups);

        expect(result.current.backwardLinks.transactions).toEqual([
            {
                tableName: "customers",
                columnName: "customer_id",
                pkeyName: "customer_id"
            }
        ]);
    });
});
