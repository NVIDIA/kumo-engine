// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClient } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { vi } from "vitest";

import { setColStatsResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/Colstats.test.mock";
import { setTestResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/TableDetail.test.mock";
import { setTableListResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/TableList.test.mock";

import { storageKeys } from "utils/Constants";

import {
    useColumnStatsInfo,
    useCreateTable,
    useGetTableInfo,
    useTables
} from "../tables";

import "tests/mockAxios";

vi.mock("utils/navigator", () => ({
    useCustomNavigation: vi.fn(() => vi.fn())
}));

let queryClient: QueryClient;

const handleError = vi.fn();
const setPageLoading = vi.fn();

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    storeToLocalStorage(storageKeys.selectedProject, "project-tests");
    setTableListResponseType();
    setTestResponseType();
    setColStatsResponseType();
});

afterEach(() => {
    queryClient.clear();
});

describe("useTables", () => {
    const renderUseTables = (
        handleErrorFn?: typeof handleError,
        setPageLoadingFn?: typeof setPageLoading,
        enabled = true
    ) => {
        return renderHook(
            () => useTables(handleErrorFn, setPageLoadingFn, enabled),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it("Should return loading state initially", () => {
        const { result } = renderUseTables();

        expect(result.current.isDataLoading).toBe(true);
    });

    it("Should return table list sorted by updated_at descending after loading", async () => {
        const { result } = renderUseTables(handleError, setPageLoading);

        await waitFor(() => {
            expect(result.current.isDataLoading).toBe(false);
        });

        expect(result.current.tableList.length).toBeGreaterThan(0);
        expect(result.current.tablesPresent).toBe(true);

        const timestamps = result.current.tableList.map((t) =>
            new Date(t?.timestamps?.updated_at || "").getTime()
        );
        const sorted = [...timestamps].sort((a, b) => b - a);
        expect(timestamps).toEqual(sorted);
    });

    it("Should set connectorPresent to false and tablesPresent to false when list is empty", async () => {
        setTableListResponseType(testResponseType.EMPTY);

        const { result } = renderUseTables(handleError, setPageLoading);

        await waitFor(() => {
            expect(result.current.isDataLoading).toBe(false);
        });

        expect(result.current.connectorPresent).toBe(false);
        expect(result.current.tablesPresent).toBe(false);
        expect(result.current.tableList).toEqual([]);
    });

    it("Should call handleError when API fails", async () => {
        setTableListResponseType(testResponseType.ERROR);

        renderUseTables(handleError, setPageLoading);

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });

    it("Should call setPageLoading(false) when loading completes", async () => {
        renderUseTables(handleError, setPageLoading);

        await waitFor(() => {
            expect(setPageLoading).toHaveBeenCalledWith(false);
        });
    });
});

describe("useGetTableInfo", () => {
    const renderGetTableInfo = (tableId: string | null, enabled = true) => {
        return renderHook(
            () => useGetTableInfo(tableId, handleError, enabled),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it("Should return table info after loading", async () => {
        const { result } = renderGetTableInfo(
            "NA.NA-18e2b1a6e499fff45a6f249f3024662f"
        );

        await waitFor(() => {
            expect(result.current.loading).toBe(false);
        });

        expect(result.current.selectedTableInfo).not.toBeNull();
        expect(result.current.selectedTableInfo?.id).toBe(
            "18e2b1a6e499fff45a6f249f3024662f"
        );
    });

    it("Should return null selectedTableInfo when disabled", async () => {
        const { result } = renderGetTableInfo(null, false);

        expect(result.current.selectedTableInfo).toBeNull();
    });

    it("Should call handleError when API fails", async () => {
        setTestResponseType(testResponseType.ERROR);

        renderGetTableInfo("some-table-id");

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });
});

describe("useColumnStatsInfo", () => {
    const renderColumnStats = (id: string | null, enabled = true) => {
        const isMounted = { current: true };
        return renderHook(
            () => useColumnStatsInfo(id, handleError, isMounted, enabled),
            { wrapper: createWrapper(queryClient) }
        );
    };

    it("Should return column stats after loading", async () => {
        const { result } = renderColumnStats(
            "NA.NA-18e2b1a6e499fff45a6f249f3024662f"
        );

        await waitFor(() => {
            expect(result.current.loading).toBe(false);
        });

        expect(result.current.columnStats).not.toBeNull();
        expect(result.current.columnStats?.statistics_available).toBe(true);
    });

    it("Should not load when id is null", async () => {
        const { result } = renderColumnStats(null);

        expect(result.current.loading).toBe(false);
        expect(result.current.columnStats).toBeNull();
    });

    it("Should call handleError when API fails", async () => {
        setColStatsResponseType(testResponseType.ERROR);

        renderColumnStats("some-table-id");

        await waitFor(() => {
            expect(handleError).toHaveBeenCalled();
        });
    });
});

describe("useCreateTable", () => {
    const renderCreateTable = () => {
        return renderHook(() => useCreateTable(handleError), {
            wrapper: createWrapper(queryClient)
        });
    };

    const tableArgs = {
        cols: [{ name: "id", stype: "ID", dtype: "string" }],
        pkey: "id",
        source_table: {
            connector_id: "test",
            source: { mode: "relative", path: "test_table" },
            data_source_type: "S3",
            file_type: "PARQUET"
        }
    };

    it("Should return createTable and validateTables functions", () => {
        const { result } = renderCreateTable();

        expect(result.current.createTable).toBeDefined();
        expect(result.current.validateTables).toBeDefined();
    });

    it("Should validate table args via validateTables", async () => {
        const { result } = renderCreateTable();

        let response: unknown;
        await act(async () => {
            response = await result.current.validateTables(
                tableArgs as unknown as Parameters<
                    typeof result.current.validateTables
                >[0]
            );
        });

        expect(response).toBeDefined();
    });

    it("Should create table via createTable", async () => {
        const { result } = renderCreateTable();

        let response: unknown;
        await act(async () => {
            response = await result.current.createTable(
                tableArgs as unknown as Parameters<
                    typeof result.current.createTable
                >[0],
                {
                    project_id: "project_00000000000000000000000000000001",
                    name_alias: "test_table",
                    force_rename: false
                }
            );
        });

        expect(response).toBeDefined();
    });
});
