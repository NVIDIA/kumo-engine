// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { setColStatsResponseType } from "pages/Tables/TablesList/__tests__/MockFunction/Colstats.test.mock";

import {
    useGetSnapshotsColumnStatsInfo,
    useGetSnapshotsTableInfo
} from "../tables";

import "tests/mockAxios";

const FINANCE_TRANS_ID = "table-bc150313117da0324bfe31ca4ff499e5@1664307208";

let queryClient: QueryClient;

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    setColStatsResponseType();
});

afterEach(() => {
    queryClient.clear();
});

describe("useGetSnapshotsTableInfo", () => {
    const handleError = vi.fn();

    const renderSnapshotsTableInfo = (tableId: string | null, enabled = true) =>
        renderHook(
            () => useGetSnapshotsTableInfo(tableId, handleError, enabled),
            { wrapper: createWrapper(queryClient) }
        );

    it("Should return null snapshotTableInfo when query is disabled", () => {
        const { result } = renderSnapshotsTableInfo(FINANCE_TRANS_ID, false);
        expect(result.current.snapshotTableInfo).toBeNull();
        expect(result.current.loading).toBe(false);
    });

    it("Should call handleError when request fails", async () => {
        renderSnapshotsTableInfo(FINANCE_TRANS_ID);
        await waitFor(() => expect(handleError).toHaveBeenCalled());
    });
});

describe("useGetSnapshotsColumnStatsInfo", () => {
    const handleError = vi.fn();
    const isMounted = { current: true };

    const renderSnapshotsColumnStatsInfo = (
        id: string | null,
        enabled = true
    ) =>
        renderHook(
            () =>
                useGetSnapshotsColumnStatsInfo(
                    id,
                    handleError,
                    isMounted,
                    enabled
                ),
            { wrapper: createWrapper(queryClient) }
        );

    it.each([
        { id: null, enabled: false, description: "id is null and disabled" },
        {
            id: FINANCE_TRANS_ID,
            enabled: false,
            description: "enabled is false"
        }
    ])(
        "Should return null columnStats and not loading when $description",
        ({ id, enabled }) => {
            const { result } = renderSnapshotsColumnStatsInfo(id, enabled);
            expect(result.current.columnStats).toBeNull();
            expect(result.current.loading).toBe(false);
            expect(result.current.isError).toBe(false);
        }
    );

    it("Should return columnStats when data loads", async () => {
        const { result } = renderSnapshotsColumnStatsInfo(FINANCE_TRANS_ID);

        await waitFor(() => {
            expect(result.current.columnStats).not.toBeNull();
        });
    });

    it("Should call handleError with correct title when request fails", async () => {
        setColStatsResponseType(testResponseType.ERROR);
        renderSnapshotsColumnStatsInfo(FINANCE_TRANS_ID);

        await waitFor(() =>
            expect(handleError).toHaveBeenCalledWith(
                expect.anything(),
                "Failed to get the column statistics!"
            )
        );
    });
});
