// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
    expectedGraphSnapshotResponse,
    setTestGraphSnapshotResponseType
} from "components/Jobs/Graph/__tests__/MockFunctions/GraphSnapshot.mock";

import { jobStatus } from "utils/Constants";

import {
    useGraphEdges,
    useGraphSnapshot,
    useGraphSnapshotStats,
    useSnapshotColumnStats
} from "../graphSnapshot";

import "tests/mockAxios";

const SNAPSHOT_ID =
    "graph-d445eeb8b2aa1b79a142406239fa74e8@6f0ae33ac31edb0c321fbe182e39e460";

const FINANCE_TRANS_ID = "table-bc150313117da0324bfe31ca4ff499e5@1664307208";

let queryClient: QueryClient;

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    setTestGraphSnapshotResponseType();
});

afterEach(() => {
    queryClient.clear();
});

describe("useGraphSnapshot", () => {
    const setPageLoading = vi.fn();
    const handleError = vi.fn();
    const setStatus = vi.fn();
    const setWarnings = vi.fn();

    const renderGraphSnapshot = ({
        id = SNAPSHOT_ID,
        errorFn,
        statusFn,
        warningsFn
    }: {
        id?: string;
        errorFn?: typeof handleError;
        statusFn?: typeof setStatus;
        warningsFn?: typeof setWarnings;
    } = {}) =>
        renderHook(
            () =>
                useGraphSnapshot(
                    id,
                    undefined,
                    undefined,
                    setPageLoading,
                    errorFn,
                    statusFn,
                    warningsFn
                ),
            { wrapper: createWrapper(queryClient) }
        );

    it("Should call setPageLoading(false) when data loads", async () => {
        renderGraphSnapshot();
        await waitFor(() => expect(setPageLoading).toHaveBeenCalledWith(false));
    });

    it("Should return graphSnapshot and stages from response", async () => {
        const { result } = renderGraphSnapshot();
        const expected = expectedGraphSnapshotResponse();

        await waitFor(() => {
            expect(result.current.graphSnapshot).toEqual(expected);
            expect(result.current.stages).toEqual(
                expected.snapshot_resource?.stages ?? null
            );
        });
    });

    it.each([
        {
            responseType: "",
            expectedStatus: jobStatus.Done,
            description: "DONE"
        },
        {
            responseType: testResponseType.QUEUED,
            expectedStatus: jobStatus.Queued,
            description: "QUEUED"
        },
        {
            responseType: testResponseType.RUNNING,
            expectedStatus: jobStatus.Running,
            description: "RUNNING"
        },
        {
            responseType: testResponseType.FAILED,
            expectedStatus: jobStatus.Failed,
            description: "FAILED"
        }
    ])(
        "Should call setStatus with $description when INGEST status is $description",
        async ({ responseType, expectedStatus }) => {
            setTestGraphSnapshotResponseType(responseType);
            renderGraphSnapshot({ statusFn: setStatus });

            await waitFor(() =>
                expect(setStatus).toHaveBeenCalledWith(expectedStatus)
            );
        }
    );

    it("Should call handleError with correct title when request fails", async () => {
        setTestGraphSnapshotResponseType(testResponseType.ERROR);
        renderGraphSnapshot({ errorFn: handleError });

        await waitFor(() =>
            expect(handleError).toHaveBeenCalledWith(
                expect.anything(),
                "Unable to fetch graph snapshots"
            )
        );
    });

    it("Should call setWarnings when INGEST stage has warnings", async () => {
        setTestGraphSnapshotResponseType(testResponseType.FAILED);
        renderGraphSnapshot({ warningsFn: setWarnings });

        await waitFor(() =>
            expect(setWarnings).toHaveBeenCalledWith(
                expect.arrayContaining([
                    expect.objectContaining({
                        description: expect.any(String),
                        is_error: false,
                        is_dismissible: true
                    })
                ])
            )
        );
    });
});

describe("useGraphEdges", () => {
    const renderGraphEdges = (
        id: string,
        graphSnapshot: Parameters<typeof useGraphEdges>[1]
    ) =>
        renderHook(() => useGraphEdges(id, graphSnapshot), {
            wrapper: createWrapper(queryClient)
        });

    it.each([
        {
            id: SNAPSHOT_ID,
            graphSnapshot: null,
            description: "graphSnapshot is null"
        },
        { id: "", graphSnapshot: null, description: "id is empty" }
    ])(
        "Should return undefined graphEdges and no error when $description",
        ({ id, graphSnapshot }) => {
            const { result } = renderGraphEdges(id, graphSnapshot);
            expect(result.current.graphEdges).toBeUndefined();
            expect(result.current.isGraphEdgeStatsError).toBe(false);
        }
    );
});

describe("useGraphSnapshotStats", () => {
    const renderGraphSnapshotStats = (id: string) =>
        renderHook(() => useGraphSnapshotStats(id), {
            wrapper: createWrapper(queryClient)
        });

    it.each([
        {
            id: SNAPSHOT_ID,
            expectedTotalRows: 1067291,
            expectedTotalSizeBytes: 982056960,
            description: "id is provided"
        },
        {
            id: "",
            expectedTotalRows: null,
            expectedTotalSizeBytes: null,
            description: "id is empty"
        }
    ])(
        "Should return totalRows and totalSizeBytes when $description",
        async ({ id, expectedTotalRows, expectedTotalSizeBytes }) => {
            const { result } = renderGraphSnapshotStats(id);

            await waitFor(() => {
                expect(result.current.totalRows).toBe(expectedTotalRows);
                expect(result.current.totalSizeBytes).toBe(
                    expectedTotalSizeBytes
                );
            });
        }
    );
});

describe("useSnapshotColumnStats", () => {
    const renderSnapshotColumnStats = (snapshotId: string | null) =>
        renderHook(() => useSnapshotColumnStats(snapshotId), {
            wrapper: createWrapper(queryClient)
        });

    it("Should return null columnStats when snapshotId is null", () => {
        const { result } = renderSnapshotColumnStats(null);
        expect(result.current.columnStats).toBeNull();
        expect(result.current.fetchingSamples).toBe(false);
        expect(result.current.isError).toBe(false);
    });

    it("Should set nonEmpty to first non-null sample row", async () => {
        const { result } = renderSnapshotColumnStats(FINANCE_TRANS_ID);

        await waitFor(() => {
            expect(result.current.columnStats?.columns[0].nonEmpty).toBe(
                "3147245"
            );
        });
    });
});
