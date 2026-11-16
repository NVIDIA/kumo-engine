// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useUserStore } from "stores/userStore";
import { mockNavigateCall } from "tests/mockRouter";
import { createQueryClient } from "tests/testUtil";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { TableListResponse } from "types/tables";

import { storageKeys } from "utils/Constants";

import TablesList from "../TablesList";

import { setCustomTableListRows } from "./MockFunction/TableList.test.mock";

// Turn the Custom SQL surface on so the "Kind" filter renders. Mirrors the
// pattern used by TableOverviewCustomSql.test.tsx.
vi.mock("utils/Constants", async (importOriginal) => {
    const actual = await importOriginal<typeof import("utils/Constants")>();
    return { ...actual, ENABLE_CUSTOM_SQL_TABLES: true };
});

const ts = (day: number) => ({
    created_at: `2024-01-0${day}T00:00:00Z`,
    updated_at: `2024-01-0${day}T00:00:00Z`
});

// Mixed list rows exercising both classification paths:
//  - SQL via the explicit wire `table_kind` (backend that stamps the list), and
//  - SQL via the `custom_sql_` synthetic source name (backend that does not).
const MIXED_ROWS = [
    {
        id: "NA.NA-raw1",
        name: "raw_customers",
        timestamps: ts(1),
        connector_id: "c1",
        source_table_name: "customers"
    },
    {
        id: "NA.NA-raw2",
        name: "raw_orders",
        timestamps: ts(2),
        connector_id: "c1",
        source_table_name: "orders"
    },
    {
        id: "NA.NA-sql1",
        name: "sql_via_kind",
        timestamps: ts(3),
        connector_id: "c1",
        source_table_name: "custom_sql_aaa",
        table_kind: "sql"
    },
    {
        id: "NA.NA-sql2",
        name: "sql_via_name",
        timestamps: ts(4),
        connector_id: "c1",
        source_table_name: "custom_sql_bbb"
    }
] as unknown as TableListResponse[];

describe("TablesList Kind filter", () => {
    useUserStore.getState().setUserDetails({ isLoggedIn: true });

    const renderPage = async () => {
        await act(async () => {
            render(
                <MemoryRouter>
                    <QueryClientProvider client={createQueryClient()}>
                        <TablesList />
                    </QueryClientProvider>
                </MemoryRouter>
            );
        });
    };

    const visibleNames = () =>
        screen
            .getAllByTestId(/^entity-name-/)
            .map((el) => el.textContent?.trim() ?? "")
            .sort();

    const headerCount = () =>
        screen
            .getByRole("heading", { name: /tables/i })
            .parentElement?.textContent?.replace(/[^0-9]/g, "");

    const spinnerPresent = () =>
        screen
            .getByRole("heading", { name: /tables/i })
            .querySelector(".ps-2 .spinner") != null;

    const openKindFilterAndSelect = async (optionLabel: string) => {
        await userEvent.click(screen.getByTestId("testid-table-kind-filter"));
        const option = await screen.findByText(optionLabel);
        await userEvent.click(option);
    };

    beforeEach(() => {
        storeToLocalStorage(storageKeys.selectedProject, "project-tests");
        window.location.href = window.location.href + "tables/";
        mockNavigateCall();
        setCustomTableListRows(MIXED_ROWS);
        vi.clearAllMocks();
    });

    afterEach(() => {
        setCustomTableListRows(null);
        vi.clearAllMocks();
    });

    it("shows all tables and a settled numeric count on load (no stuck spinner)", async () => {
        await renderPage();

        await waitFor(() =>
            expect(screen.getAllByTestId(/^entity-name-/)).toHaveLength(4)
        );
        expect(headerCount()).toBe("4");
        expect(spinnerPresent()).toBe(false);
    });

    it("partitions to Custom SQL rows (via table_kind AND the custom_sql_ fallback)", async () => {
        await renderPage();
        await waitFor(() =>
            expect(screen.getAllByTestId(/^entity-name-/)).toHaveLength(4)
        );

        await openKindFilterAndSelect("Custom SQL");

        await waitFor(() =>
            expect(visibleNames()).toEqual(["sql_via_kind", "sql_via_name"])
        );
        expect(headerCount()).toBe("2");
        expect(spinnerPresent()).toBe(false);
    });

    it("partitions to Raw rows only", async () => {
        await renderPage();
        await waitFor(() =>
            expect(screen.getAllByTestId(/^entity-name-/)).toHaveLength(4)
        );

        await openKindFilterAndSelect("Raw");

        await waitFor(() =>
            expect(visibleNames()).toEqual(["raw_customers", "raw_orders"])
        );
        expect(headerCount()).toBe("2");
        expect(spinnerPresent()).toBe(false);
    });
});
