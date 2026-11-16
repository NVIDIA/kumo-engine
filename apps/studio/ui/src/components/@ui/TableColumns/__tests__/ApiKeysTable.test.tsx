// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { APIKeySummary } from "gen/v0/types";
import { getColumn } from "tests/testHelpers";

import { formatToShortLocalDateString } from "utils/timeConverter";

import { AdminApiKeysTableColumns, ApiKeysTableColumns } from "../ApiKeysTable";

type ApiKeysColumnDef = ReturnType<typeof ApiKeysTableColumns>[number];
type AdminApiKeysColumnDef = ReturnType<
    typeof AdminApiKeysTableColumns
>[number];

const createMockRow = (
    overrides?: Partial<APIKeySummary>
): Row<APIKeySummary> =>
    ({
        original: {
            id: "key-001",
            name: "My API Key",
            type: "user",
            owner_user_id: "user-001",
            owner_display_name: "John Doe",
            created_time: "2026-01-15T10:00:00.000Z",
            value_preview: "kmo_abc...xyz",
            ...overrides
        }
    }) as Row<APIKeySummary>;

const renderCell = (
    column: ApiKeysColumnDef | AdminApiKeysColumnDef,
    row: Row<APIKeySummary>,
    getValue?: () => unknown
) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return column.cell({
        row,
        getValue: getValue ?? (() => undefined)
    } as CellContext<APIKeySummary, unknown>);
};

describe("ApiKeysTableColumns", () => {
    const onDeleteClick = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it.each([
        { index: 0, id: "name", header: "Name" },
        { index: 1, id: "created", header: "Created" },
        { index: 2, id: "value", header: "Value" },
        { index: 3, id: "delete", header: "" }
    ])(
        "Should have id '$id' and header '$header' at index $index",
        ({ index, id, header }) => {
            const columns = ApiKeysTableColumns(onDeleteClick);
            const column = getColumn(columns, index);

            expect(column.id).toBe(id);
            expect(column.header).toBe(header);
        }
    );

    it("Should call onDeleteClick when delete cell is clicked", async () => {
        const columns = ApiKeysTableColumns(onDeleteClick);
        expect(columns).toHaveLength(4);
        const deleteColumn = getColumn(columns, 3);
        const mockRow = createMockRow();

        render(renderCell(deleteColumn, mockRow) as React.ReactElement);

        await userEvent.click(screen.getByRole("button", { name: /delete/i }));

        expect(onDeleteClick).toHaveBeenCalledTimes(1);
    });
});

describe("AdminApiKeysTableColumns", () => {
    const onDeleteClick = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it.each([
        { index: 0, id: "name", header: "Name" },
        { index: 1, id: "owner_display_name", header: "User" },
        { index: 2, id: "created_time", header: "Created" },
        { index: 3, id: "value_preview", header: "Value" },
        { index: 4, id: "delete", header: "" }
    ])(
        "Should have id '$id' and header '$header' at index $index",
        ({ index, id, header }) => {
            const columns = AdminApiKeysTableColumns(onDeleteClick);
            const column = getColumn(columns, index);

            expect(column.id).toBe(id);
            expect(column.header).toBe(header);
        }
    );

    it("Should render formatted created_time in the created column cell", () => {
        const columns = AdminApiKeysTableColumns(onDeleteClick);
        const createdColumn = getColumn(columns, 2);
        const mockRow = createMockRow();

        if (typeof createdColumn.cell !== "function") throw new Error();
        const cell = createdColumn.cell({
            row: mockRow,
            getValue: () => mockRow.original.created_time
        } as CellContext<APIKeySummary, unknown>) as React.ReactElement;

        expect(cell.props.children).toBe(
            formatToShortLocalDateString(mockRow.original.created_time)
        );
    });

    // Regression for kumo-ai/kumo#29906: the Created cell rendered the UTC
    // wall time with no timezone label, so it read as a wrong local time
    // (7 hours in the future for a Pacific viewer). It must render the
    // viewer-local wall time with a zone label.
    it("Should render created_time as a zone-labeled local string", () => {
        const columns = AdminApiKeysTableColumns(onDeleteClick);
        const createdColumn = getColumn(columns, 2);
        const created_time = "2026-07-10T05:45:54.363073Z";
        const mockRow = createMockRow({ created_time });

        if (typeof createdColumn.cell !== "function") throw new Error();
        const cell = createdColumn.cell({
            row: mockRow,
            getValue: () => mockRow.original.created_time
        } as CellContext<APIKeySummary, unknown>) as React.ReactElement;

        const expected = formatToShortLocalDateString(created_time);
        // e.g. "PST", "UTC", or "GMT+9" — some zone label must be present.
        expect(expected).toMatch(/ [A-Z]{2,5}([+-]\d{1,2}(:\d{2})?)?$/);
        expect(cell.props.children).toBe(expected);
        expect(cell.props.children).toBe(
            formatToShortLocalDateString(created_time, {
                timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
            })
        );
    });

    it("Should call onDeleteClick with the row ID when delete cell is clicked", async () => {
        const columns = AdminApiKeysTableColumns(onDeleteClick);
        expect(columns).toHaveLength(5);
        const deleteColumn = getColumn(columns, 4);
        const mockRow = createMockRow({ id: "key-999" });

        render(renderCell(deleteColumn, mockRow) as React.ReactElement);

        await userEvent.click(screen.getByRole("button", { name: /delete/i }));

        expect(onDeleteClick).toHaveBeenCalledWith("key-999");
    });
});
