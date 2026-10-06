// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { CellContext, Row } from "@kumo-ai/kumo-ui-catalog";
import { GenericJob, JobType } from "gen/v0/types";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TaskType } from "types/predictiveQueries";

import { jobStatus } from "utils/Constants";

import { JobListColumns } from "../JobListColumns";

vi.mock("components/CodeEditor/PQCodeEditor", () => ({
    default: () => <div>PQCodeEditor</div>
}));

const mockQueryYamlResponse: { [key: string]: string } = {
    "job-1": "SELECT * FROM table"
};
const mockTaskType: { [key: string]: TaskType } = {
    "job-1": TaskType.BINARY_CLASSIFICATION
};
const mockActiveJobsResponse: { [job_id: string]: GenericJob | jobStatus } = {
    "job-1": {
        metadata: {
            status: jobStatus.Done
        }
    } as GenericJob
};

const mockJob: GenericJob = {
    job_id: "job-1",
    metadata: {
        job_type: JobType.TRAINING_JOB,
        status: jobStatus.Done,
        start_time: new Date("2024-01-01T00:00:00Z"),
        end_time: new Date("2024-01-01T01:00:00Z"),
        tags: {
            tag1: "value1",
            tag2: "value2"
        }
    },
    test_metric: {
        name: "accuracy",
        val: 0.95
    },
    source: "api"
} as unknown as GenericJob;

type ColumnDefinition = ReturnType<typeof JobListColumns>[number] & {
    accessorKey?: string;
    meta?: { sticky?: unknown };
};

const renderCell = (column: ColumnDefinition, job: Partial<GenericJob>) => {
    if (typeof column?.cell !== "function") {
        throw new Error("Column cell is not a function");
    }
    const mockRow: Row<GenericJob> = {
        original: job as GenericJob
    } as Row<GenericJob>;
    return column.cell({ row: mockRow } as CellContext<GenericJob, unknown>);
};

describe("JobListColumns", () => {
    let columns: ColumnDefinition[];

    beforeEach(() => {
        vi.clearAllMocks();
        columns = JobListColumns(
            mockQueryYamlResponse,
            mockTaskType,
            mockActiveJobsResponse
        );
    });

    const getColumn = (id: string) => columns.find((col) => col.id === id)!;

    it("Should return 7 columns with correct ids", () => {
        expect(columns).toBeDefined();
        expect(columns.length).toBe(9);

        const expectedIds = [
            "job_id",
            "metadata.status",
            "task_type",
            "source",
            "test_metric",
            "metadata.start_time",
            "user",
            "duration",
            "metadata.tags"
        ];

        expectedIds.forEach((id, index) => {
            expect(columns[index].id).toBe(id);
        });
    });

    describe("Job ID Column", () => {
        it("Should have correct accessorKey and size", () => {
            const column = getColumn("job_id");

            expect(column.accessorKey).toBe("job_id");
            expect(column.size).toBe(150);
        });

        it("Should have sticky behavior configured as always", () => {
            const column = getColumn("job_id");

            expect(column.meta?.sticky).toEqual({
                type: "always",
                left: true,
                zIndex: 20
            });
        });

        it("Should render job ID with HoverCard", () => {
            const column = getColumn("job_id");
            const result = renderCell(column, mockJob);

            expect(result.props.id).toBe("job-1-tooltip");
            expect(result.props.openDelay).toBe(100);
            expect(result.props.closeDelay).toBe(0);

            const hoverCardTrigger = result.props.children[0];
            const triggerDiv = hoverCardTrigger.props.children;
            expect(triggerDiv.props.id).toBe("job-1");
        });

        it("Should render empty fragment when job_id is missing", () => {
            const column = getColumn("job_id");
            const result = renderCell(column, {});

            expect(result).toEqual(<></>);
        });
    });

    describe("Test Metric Column", () => {
        it("Should have correct configuration", () => {
            const column = getColumn("test_metric");

            expect(column.accessorKey).toBe("test_metric");
            expect(column.sortDescFirst).toBe(false);
        });

        it("Should render test metric with tooltip and format to 2 decimals", () => {
            const column = getColumn("test_metric");
            const jobWithMetric = {
                ...mockJob,
                test_metric: {
                    name: "precision",
                    val: 0.123456
                }
            };

            const result = renderCell(column, jobWithMetric);

            expect(result.props.id).toBe("precision-tooltip");
            const tooltipTrigger = result.props.children[0];
            const span = tooltipTrigger.props.children;
            expect(span.props.children).toBe("0.12");
        });

        it("Should render emdash when test_metric is missing", () => {
            const column = getColumn("test_metric");
            const result = renderCell(column, { job_id: "job-1" });

            expect(result.props.type).toBe("emdash");
        });
    });

    describe("Start Time Column", () => {
        it("Should have correct configuration", () => {
            const column = getColumn("metadata.start_time");

            expect(column.header).toBe("Started");
            expect(column.accessorKey).toBe("metadata.start_time");
        });
    });

    describe("Status Column", () => {
        it("Should have correct configuration", () => {
            const column = getColumn("metadata.status");

            expect(column.header).toBe("Status");
            expect(column.accessorKey).toBe("metadata.status");
        });

        it("Should render status badge with correct size", () => {
            const column = getColumn("metadata.status");
            const result = renderCell(column, mockJob);

            expect(result.props.size).toBe("md");
        });

        it("Should handle string status in activeJobsResponse", () => {
            const columnsWithStringStatus = JobListColumns(
                mockQueryYamlResponse,
                mockTaskType,
                {
                    "job-1": jobStatus.Running
                }
            );
            const column = columnsWithStringStatus.find(
                (col) => col.id === "metadata.status"
            )!;
            const result = renderCell(column, mockJob);

            expect(result).toBeDefined();
        });
    });

    describe("Duration Column", () => {
        it("Should have correct configuration", () => {
            const column = getColumn("duration");

            expect(column.header).toBe("Duration");
            expect(column.accessorKey).toBe("duration");
            expect(column.size).toBe(100);
            expect(column.sortDescFirst).toBe(false);
        });

        it("Should render duration when both times are present", () => {
            const column = getColumn("duration");
            const result = renderCell(column, mockJob);

            expect(result).toBeDefined();
        });

        it.each<GenericJob["metadata"]["status"]>([
            jobStatus.Queued,
            jobStatus.Running,
            jobStatus.Untrained
        ])("Should render emdash before execution starts (%s)", (status) => {
            const column = getColumn("duration");
            const result = renderCell(column, {
                ...mockJob,
                metadata: { ...mockJob.metadata, status }
            });

            expect(result.props.type).toBe("emdash");
        });

        it.each([
            {
                description: "end time is missing",
                job: {
                    job_id: "job-1",
                    metadata: {
                        start_time: new Date("2024-01-01T00:00:00Z")
                    } as unknown as GenericJob["metadata"]
                }
            },
            {
                description: "start time is missing",
                job: {
                    job_id: "job-1",
                    metadata: {
                        end_time: new Date("2024-01-01T01:00:00Z")
                    } as unknown as GenericJob["metadata"]
                }
            },
            {
                description: "job_id is missing",
                job: {
                    metadata: {
                        start_time: new Date("2024-01-01T00:00:00Z"),
                        end_time: new Date("2024-01-01T01:00:00Z")
                    } as unknown as GenericJob["metadata"]
                }
            }
        ])("Should render emdash when $description", ({ job }) => {
            const column = getColumn("duration");
            const result = renderCell(column, job);

            expect(result.props.type).toBe("emdash");
        });
    });

    describe("Tags Column", () => {
        it("Should have correct configuration", () => {
            const column = getColumn("metadata.tags");

            expect(column.header).toBe("Tags");
            expect(column.accessorKey).toBe("metadata.tags");
            expect(column.sortDescFirst).toBe(false);
        });

        it("Should render multiple tags using Tag component", () => {
            const column = getColumn("metadata.tags");
            const result = renderCell(column, mockJob);

            expect(result.type).toBe("div");
            expect(result.props.className).toBe("flex gap-2");

            const tags = result.props.children.filter(Boolean);
            expect(tags.length).toBe(2);
            expect(tags[0].key).toBe("tag1-value1-0");
            expect(tags[1].key).toBe("tag2-value2-1");

            tags.forEach((tag: React.ReactElement) => {
                expect(tag.props.tagKey).toBeDefined();
                expect(tag.props.value).toBeDefined();
            });
        });

        it("Should filter out tags starting with pquery_id", () => {
            const column = getColumn("metadata.tags");
            const jobWithPqueryTag = {
                job_id: "job-1",
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
                job_id: "job-1",
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

        it("Should render empty div when tags are missing", () => {
            const column = getColumn("metadata.tags");
            const result = renderCell(column, {
                job_id: "job-1",
                metadata: {} as unknown as GenericJob["metadata"]
            });

            expect(result.type).toBe("div");
            expect(result.props.children).toEqual([]);
        });
    });

    describe("Empty Fragment Rendering", () => {
        it.each([
            {
                columnId: "job_id",
                description: "job_id column when job_id is missing",
                job: {}
            },
            {
                columnId: "metadata.start_time",
                description: "start_time column when start_time is missing",
                job: { job_id: "job-1" }
            },
            {
                columnId: "metadata.status",
                description: "status column when status is missing",
                job: { job_id: "unknown-job" }
            }
        ])(
            "Should render empty fragment for $description",
            ({ columnId, job }) => {
                const column = getColumn(columnId);
                const result = renderCell(column, job);

                expect(result).toEqual(<></>);
            }
        );
    });
});
