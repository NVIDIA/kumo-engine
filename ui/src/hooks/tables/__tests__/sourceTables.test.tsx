// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { getProjectsSourceTablesByConnectorId } from "gen/v0/api/connectors/connectors";
import { createQueryClient, createWrapper } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { FlowType } from "types/tables";

import { ReactQueryKeys } from "utils/reactQueryKeys";

import { useSourceTables, UseSourceTablesProps } from "../sourceTables";

vi.mock("gen/v0/api/connectors/connectors", () => ({
    getProjectsSourceTablesByConnectorId: vi.fn()
}));

const PROJECT_ID = "project_00000000000000000000000000000001";
const CONNECTOR_ID = "connector_00000000000000000000000000000001";

vi.mock("hooks/project/projects", () => ({
    useProjectId: () => PROJECT_ID
}));

let queryClient: QueryClient;
const handleError = vi.fn();
const setLoading = vi.fn();

const defaultProps: UseSourceTablesProps = {
    selectedConnector: { value: CONNECTOR_ID, label: "Warehouse" },
    flowType: FlowType.CREATE,
    handleError,
    setLoading,
    connectorId: CONNECTOR_ID,
    sourceTableName: "",
    resetTableSelection: vi.fn()
};

const renderSourceTables = (overrides: Partial<UseSourceTablesProps> = {}) =>
    renderHook(() => useSourceTables({ ...defaultProps, ...overrides }), {
        wrapper: createWrapper(queryClient)
    });

beforeEach(() => {
    queryClient = createQueryClient();
    vi.clearAllMocks();
    vi.mocked(getProjectsSourceTablesByConnectorId).mockResolvedValue({
        data: { table_names: ["customer", "transaction"] }
    } as Awaited<ReturnType<typeof getProjectsSourceTablesByConnectorId>>);
});

describe("useSourceTables", () => {
    it("keeps the existing table selected while editing", () => {
        const { result } = renderSourceTables({
            flowType: FlowType.EDIT,
            sourceTableName: "existing_table"
        });

        expect(result.current.sourceTables).toEqual([
            { value: "existing_table", label: "existing_table" }
        ]);
        expect(getProjectsSourceTablesByConnectorId).not.toHaveBeenCalled();
    });

    it("loads tables through the exact project and connector route", async () => {
        const { result } = renderSourceTables();

        await waitFor(() =>
            expect(result.current.sourceTables).toHaveLength(2)
        );
        expect(getProjectsSourceTablesByConnectorId).toHaveBeenCalledWith(
            PROJECT_ID,
            CONNECTOR_ID,
            undefined,
            expect.any(AbortSignal)
        );
        expect(result.current.sourceTables).toEqual([
            {
                value: "customer",
                label: "customer",
                valid: true,
                connectorId: CONNECTOR_ID,
                fileType: "PARQUET"
            },
            {
                value: "transaction",
                label: "transaction",
                valid: true,
                connectorId: CONNECTOR_ID,
                fileType: "PARQUET"
            }
        ]);
    });

    it("uses only the cache entry for the active project and connector", async () => {
        queryClient.setQueryData(
            [ReactQueryKeys.SourceTable, "project_other", CONNECTOR_ID],
            { table_names: ["wrong_project"] }
        );
        queryClient.setQueryData(
            [ReactQueryKeys.SourceTable, PROJECT_ID, CONNECTOR_ID],
            { table_names: ["scoped_table"] }
        );

        const { result } = renderSourceTables();

        await waitFor(() =>
            expect(result.current.sourceTables[0]?.value).toBe("scoped_table")
        );
        expect(getProjectsSourceTablesByConnectorId).not.toHaveBeenCalled();
    });

    it("does not combine the project upload connector with global format connectors", async () => {
        vi.mocked(getProjectsSourceTablesByConnectorId).mockResolvedValue({
            data: { table_names: ["customers.csv", "events.parquet"] }
        } as Awaited<ReturnType<typeof getProjectsSourceTablesByConnectorId>>);

        const { result } = renderSourceTables({
            selectedConnector: {
                value: CONNECTOR_ID,
                label: "Project uploads"
            }
        });

        await waitFor(() =>
            expect(
                result.current.sourceTables.map((item) => item.value)
            ).toEqual(["customers.csv", "events.parquet"])
        );
        expect(getProjectsSourceTablesByConnectorId).toHaveBeenCalledTimes(1);
    });
});
