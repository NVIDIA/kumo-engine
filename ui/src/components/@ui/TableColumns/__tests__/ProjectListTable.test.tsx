// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ProjectDBEntry } from "gen/v0/types";
import { getColumn } from "tests/testHelpers";

import { ProjectListTableColumns } from "../ProjectListTable";

type ProjectListColumnDef = ReturnType<typeof ProjectListTableColumns>[number];

const createMockRow = (
    overrides?: Partial<ProjectDBEntry>
): Row<ProjectDBEntry> =>
    ({
        original: {
            id: "proj-001",
            group_id: "group-001",
            display_name: "My Project",
            created_by: "user-001",
            created_by_name: "John Doe",
            is_private: false,
            timestamps: {
                created_at: "2026-01-15T10:00:00.000Z",
                updated_at: "2026-02-01T12:00:00.000Z"
            },
            ...overrides
        }
    }) as unknown as Row<ProjectDBEntry>;

const renderCell = (
    column: ProjectListColumnDef,
    row: Row<ProjectDBEntry>,
    getValue?: () => unknown
) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    return column.cell({
        row,
        getValue: getValue ?? (() => undefined)
    } as CellContext<ProjectDBEntry, unknown>);
};

describe("ProjectListTableColumns", () => {
    it.each(["my-projects", "all"] as const)(
        "Should return 4 columns for %s tab",
        (activeTab) => {
            expect(ProjectListTableColumns(activeTab)).toHaveLength(4);
        }
    );

    it.each([
        { index: 0, id: "selected", header: "" },
        { index: 1, id: "display_name", header: "Name" },
        { index: 2, id: "access", header: "Access" },
        { index: 3, id: "updated_at", header: "Last Updated" }
    ])(
        "Should have id '$id' and header '$header' at index $index for my-projects tab",
        ({ index, id, header }) => {
            const columns = ProjectListTableColumns("my-projects");
            const column = getColumn(columns, index);

            expect(column.id).toBe(id);
            expect(column.header).toBe(header);
        }
    );

    it.each([
        { index: 0, id: "selected", header: "" },
        { index: 1, id: "display_name", header: "Name" },
        { index: 2, id: "created_by", header: "Owner" },
        { index: 3, id: "updated_at", header: "Last Updated" }
    ])(
        "Should have id '$id' and header '$header' at index $index for all tab",
        ({ index, id, header }) => {
            const columns = ProjectListTableColumns("all");
            const column = getColumn(columns, index);

            expect(column.id).toBe(id);
            expect(column.header).toBe(header);
        }
    );

    describe("selected column cell", () => {
        it.each([
            {
                desc: "matching row id",
                selectedProjectId: "proj-001" as string | null,
                rowId: "proj-001",
                expectCheck: true
            },
            {
                desc: "non-matching row id",
                selectedProjectId: "proj-999" as string | null,
                rowId: "proj-001",
                expectCheck: false
            }
        ])(
            "Should render check icon=$expectCheck when $desc",
            ({ selectedProjectId, rowId, expectCheck }) => {
                const columns = ProjectListTableColumns(
                    "my-projects",
                    selectedProjectId
                );
                const selectedColumn = getColumn(columns, 0);
                const mockRow = createMockRow({ id: rowId });

                render(<>{renderCell(selectedColumn, mockRow)}</>);

                if (expectCheck) {
                    expect(
                        screen.getByLabelText("Selected")
                    ).toBeInTheDocument();
                } else {
                    expect(
                        screen.queryByLabelText("Selected")
                    ).not.toBeInTheDocument();
                }
            }
        );
    });

    describe("display_name column cell", () => {
        const getDisplayNameColumn = () =>
            getColumn(ProjectListTableColumns("my-projects"), 1);

        it("Should render short name directly in a span", () => {
            const shortName = "Short Name";

            render(
                renderCell(
                    getDisplayNameColumn(),
                    createMockRow(),
                    () => shortName
                ) as React.ReactElement
            );

            expect(screen.getByText(shortName)).toBeInTheDocument();
        });

        it("Should render truncated name and show full name in tooltip on hover for long names", async () => {
            const longName =
                "A Very Long Project Name That Exceeds Twenty Characters";

            render(
                renderCell(
                    getDisplayNameColumn(),
                    createMockRow(),
                    () => longName
                ) as React.ReactElement
            );

            expect(
                screen.getByText(longName.substring(0, 20) + "...")
            ).toBeInTheDocument();

            const trigger = document.querySelector(
                '[data-slot="tooltip-trigger"]'
            );
            await userEvent.hover(trigger!);
            await waitFor(() => {
                expect(screen.getAllByText(longName).length).toBeGreaterThan(0);
            });
        });
    });

    describe("access column cell (my-projects tab)", () => {
        it.each([
            { is_private: true, expected: "Private" },
            { is_private: false, expected: "Shared" }
        ])(
            "Should render '$expected' when is_private is $is_private",
            ({ is_private, expected }) => {
                const columns = ProjectListTableColumns("my-projects");
                const accessColumn = getColumn(columns, 2);

                expect(
                    renderCell(accessColumn, createMockRow({ is_private }))
                ).toBe(expected);
            }
        );
    });
});
