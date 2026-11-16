// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState } from "react";
import { MemoryRouter } from "react-router";

import { storeToLocalStorage } from "@kumo-ai/kumo-ui-catalog";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
    act,
    fireEvent,
    render,
    screen,
    waitFor
} from "@testing-library/react";
import { KumoapiTableTableDefinition } from "gen/v1/types";
import { mockAxiosInstance } from "tests/mockAxios";
import { testResponseType } from "tests/testResponseType";
import { createQueryClient } from "tests/testUtil";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { storageKeys } from "utils/Constants";

import CustomSqlTableSource from "../CustomSqlTableSource";

import { setColumnResponseType } from "./MockFunction/ColumnResponse.test.mock";
import { setSampleRowsResponseType } from "./MockFunction/SampleRowsResponse.test.mock";
import { setValidateSqlTableResponseType } from "./MockFunction/ValidateSqlTable.test.mock";

// The discovery debounce; edits settle after this window before a warehouse
// describe fires (see SQL_DISCOVERY_DEBOUNCE_MS in the component).
const DEBOUNCE_MS = 500;
const DEFAULT_SQL = "SELECT * FROM customers";

const defaultDefinition = {
    cols: [],
    source_table: {
        s3_path: "",
        connector_id: "connector-1",
        source_table_name: ""
    }
} as unknown as KumoapiTableTableDefinition;

let queryClient: QueryClient;

const columnRequests = () =>
    mockAxiosInstance.get.mock.calls.filter((call) =>
        String(call[0]).endsWith("/columns")
    );

const sampleRequests = () =>
    mockAxiosInstance.get.mock.calls.filter((call) =>
        String(call[0]).endsWith("/sample")
    );

/**
 * Stateful harness: mirrors how `ConnectTables` lifts the SQL draft and
 * validation flag, so clicking Validate actually flips `validated` and renders
 * the validate-returned preview.
 */
const Harness = ({
    connectorId = "connector-1",
    initialSql = DEFAULT_SQL,
    initialDefinition = defaultDefinition,
    argsRef
}: {
    connectorId?: string;
    initialSql?: string;
    initialDefinition?: KumoapiTableTableDefinition;
    argsRef?: { current: KumoapiTableTableDefinition | null };
}) => {
    const [sqlText, setSqlText] = useState(initialSql);
    const [validated, setValidated] = useState(false);
    const [tableArgs, setTableArgs] =
        useState<KumoapiTableTableDefinition>(initialDefinition);
    if (argsRef) {
        argsRef.current = tableArgs;
    }
    return (
        <CustomSqlTableSource
            connectorId={connectorId}
            tableDefinition={tableArgs}
            setTableArgs={setTableArgs}
            sqlText={sqlText}
            setSqlText={setSqlText}
            validated={validated}
            setValidated={setValidated}
        />
    );
};

const renderHarness = async (props?: {
    connectorId?: string;
    initialSql?: string;
    initialDefinition?: KumoapiTableTableDefinition;
    argsRef?: { current: KumoapiTableTableDefinition | null };
}) => {
    await act(async () => {
        render(
            <MemoryRouter>
                <QueryClientProvider client={queryClient}>
                    <Harness {...props} />
                </QueryClientProvider>
            </MemoryRouter>
        );
    });
};

// `fireEvent.click` (not `userEvent`) keeps these fast: the discovered-columns
// DataTable makes a full pointer-event simulation expensive to flush per click.
const clickValidate = async () => {
    await act(async () => {
        fireEvent.click(screen.getByRole("button", { name: "Validate" }));
    });
};

beforeEach(() => {
    localStorage.clear();
    // Discovery/validation are project-scoped; seed the selected project so the
    // hooks resolve a `projectId` (mirrors ConnectTable.test).
    storeToLocalStorage(storageKeys.selectedProject, "project-tests");
    queryClient = createQueryClient();
    setColumnResponseType(testResponseType.ALL);
    setSampleRowsResponseType(testResponseType.ALL);
    setValidateSqlTableResponseType(testResponseType.ALL);
});

describe("CustomSqlTableSource", () => {
    it("Should render the editor and Validate action", async () => {
        await renderHarness();
        expect(screen.getByTestId("pql-editor")).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: "Validate" })
        ).toBeInTheDocument();
    });

    it("Should disable Validate until a SQL query is entered", async () => {
        await renderHarness({ initialSql: "" });
        expect(screen.getByRole("button", { name: "Validate" })).toBeDisabled();
    });

    it("Should disable Validate when no connector is selected", async () => {
        await renderHarness({ connectorId: "" });
        expect(screen.getByRole("button", { name: "Validate" })).toBeDisabled();
    });

    it("Should discover columns then validate to confirm referenced tables", async () => {
        await renderHarness();

        // Discovery populates the column list first.
        expect(
            await screen.findByText("Discovered columns")
        ).toBeInTheDocument();
        await waitFor(() =>
            expect(
                screen.getByRole("button", { name: "Validate" })
            ).toBeEnabled()
        );

        await clickValidate();

        // Validation confirms the contract: referenced tables + a confirmed
        // schema. "orders" is absent from the query text, so it uniquely
        // identifies the rendered badge.
        expect(
            await screen.findByText("Referenced tables")
        ).toBeInTheDocument();
        expect(screen.getByText("orders")).toBeInTheDocument();
        expect(await screen.findByText("Columns")).toBeInTheDocument();

        // Any schema/role edit invalidates the prior result until the user
        // validates the changed definition again.
        fireEvent.click(screen.getAllByRole("checkbox")[2]);
        expect(
            await screen.findByText("Discovered columns")
        ).toBeInTheDocument();
    });

    it("Should post a Custom SQL source_table carrying the SELECT (no server-owned fields)", async () => {
        const argsRef: { current: KumoapiTableTableDefinition | null } = {
            current: null
        };
        await renderHarness({ argsRef });
        await screen.findByText("Discovered columns");
        await waitFor(() =>
            expect(
                screen.getByRole("button", { name: "Validate" })
            ).toBeEnabled()
        );

        await clickValidate();

        // The request must drive the SAME server contract as the create path: a
        // Custom SQL `source_table` carrying the SELECT (so the backend
        // recognizes `is_sql` and derives the server-owned `connector_type`),
        // not a physical source with a loose top-level `query` field.
        const validateCall = mockAxiosInstance.post.mock.calls
            .filter((call) => call[0] === "v1/tables/resolve_sql")
            .at(-1);
        expect(validateCall).toBeDefined();
        const body = validateCall?.[1] as {
            column_hints: { name: string }[];
            source_table: {
                data_source_type: string;
                sql: string;
                connector_id: string;
            };
        };
        expect(body.source_table.data_source_type).toBe("CUSTOM_SQL");
        expect(body.source_table.sql).toBe(DEFAULT_SQL);
        expect(body.source_table.connector_id).toBe("connector-1");
        expect(body.column_hints).toEqual(
            expect.arrayContaining([
                expect.objectContaining({ name: "StockCode" })
            ])
        );
        expect(body).not.toHaveProperty("cols");
        // `connector_type` and `referenced_tables` are server-owned: the client
        // must not send them on the validate path either.
        expect(body.source_table).not.toHaveProperty("connector_type");
        expect(body.source_table).not.toHaveProperty("referenced_tables");
        await waitFor(() => {
            expect(
                argsRef.current?.cols.find(
                    (column) => column.name === "InvoiceDate"
                )?.timestamp_format
            ).toBe("yyyy-MM-dd");
            expect(
                argsRef.current?.cols.find(
                    (column) => column.name === "Description"
                )?.multicategorical_delimiter
            ).toBe("|");
        });
    });

    it("Should validate only selected output columns", async () => {
        await renderHarness();
        await screen.findByText("Discovered columns");
        const checkboxes = screen.getAllByRole("checkbox");
        expect(checkboxes.length).toBeGreaterThan(2);
        // The first checkbox selects all rows; the second data-row checkbox is
        // StockCode in the discovery fixture.
        fireEvent.click(checkboxes[2]);

        await clickValidate();
        const body = mockAxiosInstance.post.mock.calls
            .filter((call) => call[0] === "v1/tables/resolve_sql")
            .at(-1)?.[1] as { column_hints: { name: string }[] };
        expect(body.column_hints.map((column) => column.name)).not.toContain(
            "StockCode"
        );
        expect(body.column_hints.map((column) => column.name)).toContain(
            "InvoiceNo"
        );
    });

    it("Should not fire a discovery call per keystroke (debounced)", async () => {
        vi.useFakeTimers();
        try {
            const { rerender } = render(
                <QueryClientProvider client={queryClient}>
                    <CustomSqlTableSource
                        connectorId="connector-1"
                        tableDefinition={defaultDefinition}
                        setTableArgs={vi.fn()}
                        sqlText=""
                        setSqlText={vi.fn()}
                        validated={false}
                        setValidated={vi.fn()}
                    />
                </QueryClientProvider>
            );

            const rerenderWith = (sql: string) =>
                rerender(
                    <QueryClientProvider client={queryClient}>
                        <CustomSqlTableSource
                            connectorId="connector-1"
                            tableDefinition={defaultDefinition}
                            setTableArgs={vi.fn()}
                            sqlText={sql}
                            setSqlText={vi.fn()}
                            validated={false}
                            setValidated={vi.fn()}
                        />
                    </QueryClientProvider>
                );

            // Simulate rapid typing.
            rerenderWith("SELECT 1");
            rerenderWith("SELECT 12");
            rerenderWith("SELECT 123 FROM customers");

            // Nothing fetched yet: the debounce window has not elapsed.
            expect(columnRequests().length).toBe(0);
            // Validate stays locked while discovery is stale for the edit.
            expect(
                screen.getByRole("button", { name: "Validate" })
            ).toBeDisabled();

            await act(async () => {
                await vi.advanceTimersByTimeAsync(DEBOUNCE_MS);
            });

            // Exactly one discovery fired for the settled query.
            expect(columnRequests().length).toBe(1);
        } finally {
            vi.useRealTimers();
        }
    });

    it("Should render the validate-returned preview without a separate sample fetch (Q2)", async () => {
        await renderHarness();
        await screen.findByText("Discovered columns");
        await waitFor(() =>
            expect(
                screen.getByRole("button", { name: "Validate" })
            ).toBeEnabled()
        );
        await clickValidate();
        await screen.findByText("Referenced tables");

        // Validation carries the preview, so no separate sample endpoint is hit
        // and there is no longer a "Load sample" button.
        expect(sampleRequests().length).toBe(0);
        expect(
            screen.queryByRole("button", { name: "Load sample" })
        ).not.toBeInTheDocument();
        // The preview column appears once the validate-returned rows land.
        expect((await screen.findAllByText("Preview")).length).toBeGreaterThan(
            0
        );
    });

    it("Should show a note when the preview is cost-gated (Q2)", async () => {
        setValidateSqlTableResponseType(testResponseType.PARTIAL);
        await renderHarness();
        await screen.findByText("Discovered columns");
        await waitFor(() =>
            expect(
                screen.getByRole("button", { name: "Validate" })
            ).toBeEnabled()
        );

        await clickValidate();

        // Schema is still confirmed, but the skipped preview is explained and no
        // rows (or preview column) render.
        expect(
            await screen.findByText("Preview skipped — query too large.")
        ).toBeInTheDocument();
        expect(sampleRequests().length).toBe(0);
        expect(await screen.findByText("Columns")).toBeInTheDocument();
        expect(screen.queryByText("Preview")).not.toBeInTheDocument();
    });

    it("Should block validation when the query returns no columns", async () => {
        setValidateSqlTableResponseType(testResponseType.EMPTY);
        await renderHarness();
        await screen.findByText("Discovered columns");
        await waitFor(() =>
            expect(
                screen.getByRole("button", { name: "Validate" })
            ).toBeEnabled()
        );

        await clickValidate();

        expect(
            await screen.findByText(/query returned no columns/i)
        ).toBeInTheDocument();
        // Not validated: referenced tables and the preview column stay absent.
        expect(screen.queryByText("Referenced tables")).not.toBeInTheDocument();
        expect(screen.queryByText("Preview")).not.toBeInTheDocument();
    });

    it("Should seed an edited SQL table: open validated and preserve the primary key", async () => {
        // Mirrors the edit surface: a non-empty draft plus persisted columns
        // whose primary key must survive the initial describe of the same query.
        const argsRef: { current: KumoapiTableTableDefinition | null } = {
            current: null
        };
        const editDefinition = {
            cols: [
                { name: "StockCode", dtype: "string", stype: "ID" },
                { name: "Quantity", dtype: "int64", stype: "numerical" }
            ],
            pkey: "StockCode",
            time_col: "",
            end_time_col: "",
            source_table: {
                connector_id: "connector-1",
                sql: DEFAULT_SQL,
                table: "abc",
                data_source_type: "CUSTOM_SQL"
            }
        } as unknown as KumoapiTableTableDefinition;

        await renderHarness({ initialDefinition: editDefinition, argsRef });

        // The persisted definition was already valid, so the seed opens on the
        // confirmed "Columns" list (not "Discovered columns") — Save is enabled
        // without forcing a redundant re-validate.
        expect(await screen.findByText("Columns")).toBeInTheDocument();
        expect(
            screen.queryByText("Discovered columns")
        ).not.toBeInTheDocument();

        // The persisted primary key is carried onto the freshly described schema
        // rather than being dropped by discovery.
        await waitFor(() => expect(argsRef.current?.pkey).toBe("StockCode"));
    });

    it("Should surface the scoped-connector rejection gracefully", async () => {
        setValidateSqlTableResponseType(testResponseType.ERROR);
        await renderHarness();
        await screen.findByText("Discovered columns");
        await waitFor(() =>
            expect(
                screen.getByRole("button", { name: "Validate" })
            ).toBeEnabled()
        );

        await clickValidate();

        expect(
            await screen.findByText(
                "Custom SQL isn't available on this connector"
            )
        ).toBeInTheDocument();
        // The editor still renders — the rejection did not crash the surface.
        expect(screen.getByTestId("pql-editor")).toBeInTheDocument();
    });
});
