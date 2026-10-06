// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { GenericJob } from "gen/v0/types";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PredictiveQueryInfo } from "types/predictiveQueries";

import { TrainingJobSelectListColumns } from "../TrainingJobSelectListColumns";

vi.mock("components/CodeEditor/PQCodeEditor", () => ({
    default: () => <div>PQCodeEditor</div>
}));

const mockJobWithPQInfo: GenericJob & PredictiveQueryInfo = {
    job_id: "job-123",
    metadata: {
        start_time: new Date("2024-01-01T00:00:00Z"),
        end_time: new Date("2024-01-01T01:00:00Z"),
        tags: {
            tag1: "production",
            tag2: "experiment"
        }
    },
    queryYaml: "PREDICT label\nFROM table\nWHERE condition = true",
    source: "api",
    taskType: "BINARY_CLASSIFICATION",
    execution: {
        start_time: new Date("2024-01-01T00:00:00Z")
    },
    user: "test-user@example.com"
} as unknown as GenericJob & PredictiveQueryInfo;

type ColumnDefinition = ReturnType<typeof TrainingJobSelectListColumns>[number];

const renderCell = (
    column: ColumnDefinition,
    job: Partial<GenericJob & PredictiveQueryInfo>
) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    const mockRow: Row<GenericJob & PredictiveQueryInfo> = {
        original: job as GenericJob & PredictiveQueryInfo
    } as Row<GenericJob & PredictiveQueryInfo>;
    return column.cell({
        row: mockRow,
        getValue: () => mockRow.original.user
    } as CellContext<GenericJob & PredictiveQueryInfo, unknown>);
};

describe("TrainingJobSelectListColumns", () => {
    let columns: ReturnType<typeof TrainingJobSelectListColumns>;

    beforeEach(() => {
        vi.clearAllMocks();
        columns = TrainingJobSelectListColumns();
    });

    const getColumn = (id: string) => columns.find((col) => col.id === id)!;

    it("Should return 4 columns with correct ids", () => {
        expect(columns).toBeDefined();
        expect(columns.length).toBe(4);

        const expectedIds = [
            "job_id",
            "user",
            "metadata.start_time",
            "metadata.tags"
        ];

        expectedIds.forEach((id, index) => {
            expect(columns[index].id).toBe(id);
        });
    });

    describe("Column Configuration", () => {
        it.each([
            {
                columnId: "job_id",
                header: "Job ID",
                accessorKey: "job_id",
                size: undefined,
                sticky: { type: "always", left: true, zIndex: 20 }
            },
            {
                columnId: "user",
                header: "User",
                accessorKey: "user",
                size: undefined,
                sticky: undefined
            },
            {
                columnId: "metadata.start_time",
                header: "Started",
                accessorKey: "metadata.start_time",
                size: undefined,
                sticky: undefined
            },
            {
                columnId: "metadata.tags",
                header: "Tags",
                accessorKey: "metadata.tags",
                size: undefined,
                sticky: undefined
            }
        ])(
            "Should have correct configuration for $columnId column",
            ({ columnId, header, accessorKey, size, sticky }) => {
                const column = getColumn(columnId);

                expect(column.header).toBe(header);
                expect("accessorKey" in column && column.accessorKey).toBe(
                    accessorKey
                );
                if (size !== undefined) {
                    expect(column.size).toBe(size);
                }
                if (sticky !== undefined) {
                    expect(
                        column.meta &&
                            "sticky" in column.meta &&
                            column.meta.sticky
                    ).toEqual(sticky);
                }
            }
        );
    });

    describe("Job ID Column Rendering", () => {
        it("Should render job ID with HoverCard", () => {
            const column = getColumn("job_id");
            const result = renderCell(column, mockJobWithPQInfo);

            expect(result.props.id).toBe("job-123-tooltip");
            expect(result.props.openDelay).toBe(100);
            expect(result.props.closeDelay).toBe(0);

            const hoverCardTrigger = result.props.children[0];
            const triggerDiv = hoverCardTrigger.props.children;
            expect(triggerDiv.props.id).toBe("job-123");
        });

        it("Should render empty fragment when job_id is missing", () => {
            const column = getColumn("job_id");
            const result = renderCell(column, {});

            expect(result).toEqual(<></>);
        });
    });

    describe("User Column Rendering", () => {
        it("Should render user email when present", () => {
            const column = getColumn("user");
            const result = renderCell(column, mockJobWithPQInfo);

            expect(result.type).toBe("span");
            expect(result.props.className).includes("text-title-secondary");
            expect(result.props.children).toBe("test-user@example.com");
        });

        it("Should render 'Admin user' when user is missing", () => {
            const column = getColumn("user");
            const jobWithoutUser = {
                job_id: "job-456",
                metadata: {
                    start_time: new Date("2024-01-01T00:00:00Z"),
                    tags: {}
                } as unknown as GenericJob["metadata"]
            };
            const result = renderCell(column, jobWithoutUser);

            expect(result.type).toBe("span");
            expect(result.props.className).toBe("text-title-secondary");
            expect(result.props.children).toBe("Admin user");
        });
    });

    describe("Started Column Rendering", () => {
        it("Should render time difference with tooltip", () => {
            const column = getColumn("metadata.start_time");
            const result = renderCell(column, mockJobWithPQInfo);

            expect(result.props.id).toBe("updated-job-123");
            expect(result.props.delayDuration).toBe(100);
            expect(result.props.skipDelayDuration).toBe(0);

            const tooltipTrigger = result.props.children[0];
            const span = tooltipTrigger.props.children;

            expect(span.props.className).toBe("text-title-secondary");
            expect(span.props.children).toBeDefined();
        });

        it("Should render empty fragment when start_time is missing", () => {
            const column = getColumn("metadata.start_time");
            const jobWithoutStartTime = {
                job_id: mockJobWithPQInfo.job_id,
                metadata: {
                    tags: {}
                } as unknown as GenericJob["metadata"]
            };
            const result = renderCell(column, jobWithoutStartTime);

            expect(result).toEqual(<></>);
        });
    });

    describe("Tags Column Rendering", () => {
        it("Should render multiple tags using Tag component", () => {
            const column = getColumn("metadata.tags");
            const result = renderCell(column, mockJobWithPQInfo);

            expect(result.type).toBe("div");
            expect(result.props.className).toBe("flex gap-2");

            const tags = result.props.children.filter(Boolean);
            expect(tags.length).toBe(2);
            expect(tags[0].key).toBe("tag1-production-0");
            expect(tags[1].key).toBe("tag2-experiment-1");

            tags.forEach((tag: React.ReactElement) => {
                expect(tag.props.tagKey).toBeDefined();
                expect(tag.props.value).toBeDefined();
            });
        });

        it("Should filter out tags starting with pquery_id", () => {
            const column = getColumn("metadata.tags");
            const jobWithPqueryTag = {
                job_id: "job-123",
                metadata: {
                    tags: {
                        pquery_id_123: "some-value",
                        validTag: "valid-value"
                    }
                } as unknown as GenericJob["metadata"]
            };
            const result = renderCell(column, jobWithPqueryTag);

            const tags = result.props.children.filter(Boolean);
            expect(tags.length).toBe(1);
            expect(tags[0].props.tagKey).toBe("validTag");
        });

        it("Should transform trainingjob- tags", () => {
            const column = getColumn("metadata.tags");
            const jobWithTrainingJobTag = {
                job_id: "job-123",
                metadata: {
                    tags: {
                        someTag: "trainingjob-abc123def456"
                    }
                } as unknown as GenericJob["metadata"]
            };
            const result = renderCell(column, jobWithTrainingJobTag);

            const tags = result.props.children.filter(Boolean);
            expect(tags.length).toBe(1);
            expect(tags[0].props.tagKey).toBe("TRAININGJOBID");
            // getIdHash returns first 5 chars of last segment after splitting by "-"
            expect(tags[0].props.value).toBe("abc12");
        });

        it.each([
            {
                data: {
                    job_id: "job-1",
                    metadata: {} as GenericJob["metadata"]
                },
                description: "missing tags"
            },
            { data: {}, description: "missing metadata" }
        ])("Should render empty div when $description", ({ data }) => {
            const column = getColumn("metadata.tags");
            const result = renderCell(column, data);

            expect(result.type).toBe("div");
            expect(result.props.children).toEqual([]);
        });
    });
});
